//! Tile-streaming resource owner: catalog + slot pool + SSBO + worker.
//!
//! Built once per program (or per autotune session; shared across multiple
//! Clipmaps). Owns the worker thread; dies cleanly on `deinit`.
//!
//! Per-frame, the caller invokes `recordStream(cmd_buf, frame)` before
//! recording compute dispatches that read the SSBO. That:
//!   1. Releases staging slots whose previous use is past `MAX_FRAMES_IN_FLIGHT`
//!      old (GPU is done reading them).
//!   2. Drains any completed loads from the streamer (non-blocking).
//!   3. For each completion, claims a free data slot, appends a directory
//!      entry, and records `vkCmdCopyBuffer` (staging->slot) +
//!      `vkCmdUpdateBuffer` (header+directory shadow flush).
//!   4. Inserts a single transfer->shader_read barrier covering both writes.
//!
//! `drainAll` is the synchronous warm-up path used by autotune to avoid
//! measuring tile-load latency in render benchmarks.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const renderer_mod = @import("../render/renderer.zig");
const debug = @import("../render/debug.zig");
const buffer = @import("../render/buffer.zig");
const coords = @import("coords.zig");
const tile_loader = @import("tile_loader.zig");
const tile_streamer = @import("tile_streamer.zig");
const tile_ssbo = @import("tile_ssbo.zig");
const tile_policy = @import("tile_policy.zig");

const MAX_FRAMES_IN_FLIGHT = renderer_mod.MAX_FRAMES_IN_FLIGHT;
const STAGING_SLOTS = tile_streamer.STAGING_SLOTS;

pub const CameraView = tile_policy.CameraView;

pub const TileSystem = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: renderer_mod.GpuContext,

    catalog: tile_loader.TileCatalog,
    tile_set: tile_loader.TileSet,
    shell: tile_ssbo.SSBOShell,
    streamer: *tile_streamer.TileStreamer,

    /// Spatial-grid dir lookup. One u32 per 1deg x 1deg world cell. Stores
    /// `dir_idx + 1` (0 = empty cell), so the compute shader can do a
    /// single indexed read instead of scanning the directory.
    grid_buf: buffer.BufferWithMemory,
    /// (cell_idx, value) pairs queued since the last `recordStream` flush.
    /// Each entry produces one `vkCmdUpdateBuffer` for 4 bytes. Capacity is
    /// sized so a full-frame churn (every resident tile + its swap-mate)
    /// never spills.
    pending_grid_updates: std.ArrayList(GridUpdate),

    staging: buffer.BufferWithMemory,
    staging_size: u64,
    staging_map: [*]u8,

    pending: std.ArrayList(PendingRelease),
    /// Dedup catalog indices already enqueued; streamer queues aren't
    /// searchable by catalog_idx.
    loading: std.DynamicBitSetUnmanaged,
    /// Maintained alongside `loading` to skip O(catalog_bits/64) popcount on
    /// per-frame budget checks.
    loading_count: u32 = 0,
    desired_mask: std.DynamicBitSetUnmanaged,

    policy_scratch: []tile_policy.Item,
    policy_config: tile_policy.Config,
    max_uploads_per_frame: u32,
    /// While true, `recordStream` lifts the per-frame upload cap to the staging
    /// ceiling so a fresh or just-rebuilt clipmap ring repopulates over a few
    /// presenting frames instead of the slow capped churn. Self-terminates once
    /// the streamer has caught up. Set via `beginFastFill` (startup + render-
    /// distance rebuild).
    fast_fill: bool = false,
    tile_pool_size: u32,
    /// Max tiles the policy may mark "wanted"; `< tile_pool_size` reserves
    /// slack for in-flight churn (quarantine + pending completions).
    desired_cap: u32,
    tick_buf: []u32,

    pub const PendingRelease = struct {
        frame: u64,
        idx: u32,
        kind: enum { staging, slot },
    };

    pub const GridUpdate = struct { cell_idx: u32, value: u32 };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        ctx: renderer_mod.GpuContext,
        weights_dir: []const u8,
        max_uploads_per_frame: u32,
        config_max_tiles: ?u32,
        ring_size: u32,
        num_levels: u32,
        base_spacing: f32,
    ) !*TileSystem {
        var catalog = try tile_loader.scanCatalog(io, weights_dir, allocator);
        errdefer catalog.deinit();

        const slot_stride: u32 = catalog.slotStride();

        const min_headroom: u32 = MAX_FRAMES_IN_FLIGHT * max_uploads_per_frame;
        const per_tile_bytes: u64 = @as(u64, tile_ssbo.SSBO_DIR_STRIDE) * 4 + @as(u64, slot_stride);
        const vram_budget: u64 = ctx.vram_mb * 1024 * 1024 / 2;
        const vram_cap: u32 = if (vram_budget > @as(u64, tile_ssbo.SSBO_HDR_SIZE) * 4)
            @intCast(@min(tile_loader.MAX_TILES, (vram_budget - @as(u64, tile_ssbo.SSBO_HDR_SIZE) * 4) / per_tile_bytes))
        else
            tile_loader.MIN_TILE_POOL;
        const catalog_cap: u32 = catalog.count();

        // Pool capacity: the SSBO + slot pool + tick buffers are sized ONCE here for
        // the most tiles the system could ever hold (the whole catalog, or as many
        // as fit in VRAM). That is exactly what the largest reachable render
        // distance would request (its `tilesForRenderDistance` saturates well above
        // this ceiling), so growing the ring at runtime never needs a realloc: only
        // the desired (visible) tile count changes (see `setRenderDistance`). An
        // explicit `--max-tiles` overrides the auto ceiling.
        const tile_pool_size: u32 = if (config_max_tiles) |explicit|
            @max(tile_loader.MIN_TILE_POOL, @min(explicit, @min(tile_loader.MAX_TILES, catalog_cap)))
        else
            @max(tile_loader.MIN_TILE_POOL, @min(vram_cap, catalog_cap));

        // Desired (visible) set for the INITIAL render distance, bounded by the
        // fixed pool. `setRenderDistance` recomputes this on a runtime change so a
        // larger ring requests the newly in-range tiles.
        const render_cap = tile_loader.tilesForRenderDistance(ring_size, num_levels, base_spacing);
        const desired_cap: u32 = sizeDesiredCap(render_cap, tile_pool_size, catalog_cap, min_headroom);

        std.log.info("Tile pool: {d} slots capacity (vram_cap={d}, catalog={d}{s}); desired_cap={d} for render={d}", .{
            tile_pool_size, vram_cap, catalog_cap,
            if (config_max_tiles != null) @as([]const u8, ", cli override") else "",
            desired_cap, render_cap,
        });

        const staging_size: u64 = @as(u64, slot_stride) * STAGING_SLOTS;

        const staging = try buffer.createBuffer(
            ctx.vkd,
            ctx.device,
            ctx.mem_props,
            staging_size,
            .{ .transfer_src_bit = true },
            .{ .host_visible_bit = true, .host_coherent_bit = true },
        );
        errdefer {
            ctx.vkd.destroyBuffer(ctx.device, staging.buffer, null);
            ctx.vkd.freeMemory(ctx.device, staging.memory, null);
        }

        const staging_ptr: [*]u8 = @ptrCast(
            try ctx.vkd.mapMemory(ctx.device, staging.memory, 0, staging_size, .{}) orelse
                return error.MapFailed,
        );
        errdefer ctx.vkd.unmapMemory(ctx.device, staging.memory);

        const shell = try tile_ssbo.createSSBOShell(
            allocator,
            ctx.vkd,
            ctx.device,
            ctx.mem_props,
            &catalog,
            tile_pool_size,
            ctx.queue,
            ctx.cmd_pool,
        );
        errdefer {
            ctx.vkd.destroyBuffer(ctx.device, shell.buf.buffer, null);
            ctx.vkd.freeMemory(ctx.device, shell.buf.memory, null);
        }

        const grid_size_bytes: u64 = @as(u64, coords.GRID_CELL_COUNT) * @sizeOf(u32);
        const grid_buf = try buffer.createBuffer(
            ctx.vkd,
            ctx.device,
            ctx.mem_props,
            grid_size_bytes,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true },
            .{ .device_local_bit = true },
        );
        errdefer {
            ctx.vkd.destroyBuffer(ctx.device, grid_buf.buffer, null);
            ctx.vkd.freeMemory(ctx.device, grid_buf.memory, null);
        }
        // Zero the GPU buffer so cells default to "empty". cmdFillBuffer is
        // free here; runs once at init, no perf concern.
        const grid_init_cmd = try buffer.beginOneShot(ctx.vkd, ctx.device, ctx.cmd_pool);
        ctx.vkd.cmdFillBuffer(grid_init_cmd, grid_buf.buffer, 0, grid_size_bytes, 0);
        try buffer.endOneShot(ctx.vkd, ctx.device, ctx.cmd_pool, ctx.queue, grid_init_cmd);

        // Post-dedup, each unique cell takes one entry: at most pool_size from
        // evictions + the per-frame completion drain from `recordStream`. The
        // completion term must be the real drain ceiling `drainCap` can return,
        // which under fast-fill is STAGING_SLOTS (not max_uploads_per_frame): a
        // moving camera during the post-rebuild fast-fill window can evict a batch
        // AND drain a full STAGING_SLOTS of completions in one frame, and an
        // under-sized reservation would trip appendAssumeCapacity.
        var pending_grid_updates: std.ArrayList(GridUpdate) = .empty;
        try pending_grid_updates.ensureTotalCapacity(allocator, @as(usize, tile_pool_size) + @max(max_uploads_per_frame, STAGING_SLOTS));
        errdefer pending_grid_updates.deinit(allocator);

        var pending: std.ArrayList(PendingRelease) = .empty;
        try pending.ensureTotalCapacity(allocator, STAGING_SLOTS * 2 + tile_pool_size);
        errdefer pending.deinit(allocator);

        var loading = try std.DynamicBitSetUnmanaged.initEmpty(allocator, catalog.count());
        errdefer loading.deinit(allocator);

        var desired_mask = try std.DynamicBitSetUnmanaged.initEmpty(allocator, catalog.count());
        errdefer desired_mask.deinit(allocator);

        const policy_scratch = try allocator.alloc(tile_policy.Item, catalog.count());
        errdefer allocator.free(policy_scratch);

        const tick_buf = try allocator.alloc(u32, @as(usize, tile_pool_size) * 3);
        errdefer allocator.free(tick_buf);

        const self = try allocator.create(TileSystem);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .io = io,
            .ctx = ctx,
            .catalog = catalog,
            .tile_set = undefined,
            .shell = shell,
            .streamer = undefined,
            .grid_buf = grid_buf,
            .pending_grid_updates = pending_grid_updates,
            .staging = staging,
            .staging_size = staging_size,
            .staging_map = staging_ptr,
            .pending = pending,
            .loading = loading,
            .loading_count = 0,
            .desired_mask = desired_mask,
            .policy_scratch = policy_scratch,
            .policy_config = .{
                .max_anchor_offset_arcsec = 0.5 * tile_loader.renderRadiusArcsec(ring_size, num_levels, base_spacing),
            },
            .max_uploads_per_frame = max_uploads_per_frame,
            .fast_fill = false,
            .tile_pool_size = tile_pool_size,
            .desired_cap = desired_cap,
            .tick_buf = tick_buf,
        };
        // catalog now lives at a stable address inside `self`; wire pointers.
        self.tile_set = try tile_loader.TileSet.init(allocator, &self.catalog, tile_pool_size);
        errdefer self.tile_set.deinit(allocator);
        self.streamer = try tile_streamer.TileStreamer.init(
            allocator,
            &self.catalog,
            io,
            staging_ptr[0..staging_size],
            slot_stride,
        );
        return self;
    }

    pub fn deinit(self: *TileSystem) void {
        const ctx = self.ctx;
        self.streamer.deinit();
        ctx.vkd.unmapMemory(ctx.device, self.staging.memory);
        ctx.vkd.destroyBuffer(ctx.device, self.staging.buffer, null);
        ctx.vkd.freeMemory(ctx.device, self.staging.memory, null);
        ctx.vkd.destroyBuffer(ctx.device, self.shell.buf.buffer, null);
        ctx.vkd.freeMemory(ctx.device, self.shell.buf.memory, null);
        self.allocator.free(self.shell.shadow);
        ctx.vkd.destroyBuffer(ctx.device, self.grid_buf.buffer, null);
        ctx.vkd.freeMemory(ctx.device, self.grid_buf.memory, null);
        self.pending_grid_updates.deinit(self.allocator);
        self.allocator.free(self.policy_scratch);
        self.allocator.free(self.tick_buf);
        self.desired_mask.deinit(self.allocator);
        self.loading.deinit(self.allocator);
        self.tile_set.deinit(self.allocator);
        self.pending.deinit(self.allocator);
        self.catalog.deinit();
        self.allocator.destroy(self);
    }

    pub fn weightsBuffer(self: *const TileSystem) vk.Buffer {
        return self.shell.buf.buffer;
    }

    pub fn weightsSize(self: *const TileSystem) u64 {
        return self.shell.size;
    }

    pub fn dirGridBuffer(self: *const TileSystem) vk.Buffer {
        return self.grid_buf.buffer;
    }

    /// Queue or overwrite a dir-grid cell write; dedup prevents the same
    /// cell from being written twice in one cmd buffer (WAW hazard).
    fn setGridCell(self: *TileSystem, cell_idx: u32, value: u32) void {
        std.debug.assert(cell_idx < coords.GRID_CELL_COUNT);
        appendGridUpdateDedup(&self.pending_grid_updates, cell_idx, value);
    }

    /// Emit one `vkCmdUpdateBuffer` per queued cell (4 bytes each). Cells
    /// scatter across 253 KB so a contiguous-range flush would write the
    /// whole grid every frame; far worse than a few hundred 4-byte updates.
    /// Returns true if any cell was uploaded.
    fn flushGridCells(self: *TileSystem, vkd: anytype, cmd_buf: vk.CommandBuffer) bool {
        if (self.pending_grid_updates.items.len == 0) return false;
        // cmdUpdateBuffer copies from `pData` at record time, so pointing
        // into the per-iteration value is safe.
        for (self.pending_grid_updates.items) |update| {
            const offset: vk.DeviceSize = @as(u64, update.cell_idx) * @sizeOf(u32);
            const bytes: [*]const u8 = @ptrCast(&update.value);
            vkd.cmdUpdateBuffer(cmd_buf, self.grid_buf.buffer, offset, @sizeOf(u32), bytes);
        }
        self.pending_grid_updates.clearRetainingCapacity();
        return true;
    }

    pub fn tileCount(self: *const TileSystem) u32 {
        return self.shell.tileCount();
    }

    pub fn catalogCount(self: *const TileSystem) u32 {
        return self.catalog.count();
    }

    /// Enter fast-fill: the next few `recordStream` frames drain at the staging
    /// ceiling instead of `max_uploads_per_frame`, so a fresh or rebuilt clipmap
    /// ring fills quickly while the render loop keeps presenting. Self-clears
    /// once the streamer has no loads in flight.
    pub fn beginFastFill(self: *TileSystem) void {
        self.fast_fill = true;
    }

    /// Match the desired (visible) tile set + prefetch-anchor cap to a new render
    /// distance, WITHOUT reallocating: the pool/SSBO are sized for the max render
    /// distance at init, so a runtime increase only needs the policy to want more
    /// tiles. The caller must then re-run `tickPolicy` + drain to load the
    /// now-in-range tiles (a decrease just lets the extras evict). Without this a
    /// render-distance increase grows the ring but the policy still wants only the
    /// old, smaller set, so the larger ring renders with empty outer terrain.
    pub fn setRenderDistance(self: *TileSystem, ring_size: u32, num_levels: u32, base_spacing: f32) void {
        const render_cap = tile_loader.tilesForRenderDistance(ring_size, num_levels, base_spacing);
        const min_headroom: u32 = MAX_FRAMES_IN_FLIGHT * self.max_uploads_per_frame;
        self.desired_cap = sizeDesiredCap(render_cap, self.tile_pool_size, self.catalog.count(), min_headroom);
        self.policy_config.max_anchor_offset_arcsec = 0.5 * tile_loader.renderRadiusArcsec(ring_size, num_levels, base_spacing);
    }

    /// Call once per frame.
    pub fn tickPolicy(self: *TileSystem, view: CameraView, frame: u64) void {
        if (self.catalog.count() == 0) return;
        // F4 freeze pins residency by default: with the camera frozen the
        // policy would still chase the live viewer pos and churn tiles
        // mid-inspection. Shift+F4 (streaming_override) lets streaming
        // continue against the live camera if the user wants it.
        if (debug.streamingFrozen()) return;

        const ps = self.tile_pool_size;
        const desired_buf = self.tick_buf[0..self.desired_cap];
        const evict_buf = self.tick_buf[ps .. ps * 2];
        const load_buf = self.tick_buf[ps * 2 .. ps * 3];

        const desired = tile_policy.rankDesired(
            &self.catalog,
            &self.tile_set,
            view,
            self.policy_config,
            self.policy_scratch,
            desired_buf,
        );

        self.desired_mask.unsetAll();
        for (desired) |cat| self.desired_mask.set(cat);

        var evict_count: u32 = 0;
        var dir: u32 = 0;
        while (dir < self.tile_set.tile_count) : (dir += 1) {
            if (!self.desired_mask.isSet(self.tile_set.resident_indices[dir])) {
                evict_buf[evict_count] = dir;
                evict_count += 1;
            }
        }
        // Descending order: swap-removes from the tail don't invalidate
        // earlier dir indices in this batch.
        std.mem.sort(u32, evict_buf[0..evict_count], {}, std.sort.desc(u32));
        for (evict_buf[0..evict_count]) |di| self.evict(di, frame);

        // Budget = free slots minus in-flight loads, so every load has a slot
        // waiting when it applies. Tiles past the budget defer to next tick;
        // `desired` is priority-sorted so the farthest tiles drop first.
        const free_count = self.tile_set.freeCount();
        const request_budget: u32 = if (free_count > self.loading_count) free_count - self.loading_count else 0;

        var load_count: u32 = 0;
        for (desired) |cat| {
            if (load_count >= request_budget) break;
            if (self.tile_set.isResident(cat) or self.loading.isSet(cat)) continue;
            load_buf[load_count] = cat;
            self.loading.set(cat);
            self.loading_count += 1;
            load_count += 1;
        }
        if (load_count > 0) self.streamer.requestLoadMany(load_buf[0..load_count]);
    }

    /// Returns true when the SSBO contents changed this frame; caller must
    /// invalidate any caches that store evaluations against the SSBO. A
    /// dirty directory shadow from `evict` (no new copies) still counts.
    pub fn recordStream(self: *TileSystem, cmd_buf: vk.CommandBuffer, frame: u64) bool {
        self.releasePending(frame);

        var batch: [STAGING_SLOTS]tile_streamer.LoadComplete = undefined;
        const cap = drainCap(self.fast_fill, self.max_uploads_per_frame);
        const n = self.streamer.tryDrain(batch[0..cap]);
        // Fast-fill self-terminates only once the streamer is fully caught up:
        // nothing drained this frame (`n == 0`) AND nothing still loading. The
        // `n == 0` guard keeps the lifted cap on while completions are still
        // arriving (inFlight counts requested-not-yet-completed loads, not
        // completed-but-undrained ones), so trailing tiles also fill fast, not
        // just the first batch. It also gates the inFlight() lock to idle frames.
        if (self.fast_fill and n == 0 and self.streamer.inFlight() == 0) self.fast_fill = false;

        var copies: [STAGING_SLOTS]vk.BufferCopy = undefined;
        var copy_count: u32 = 0;

        for (batch[0..n]) |c| {
            if (self.applyCompletion(c)) |region| {
                copies[copy_count] = region;
                copy_count += 1;
                self.pending.appendAssumeCapacity(.{ .frame = frame, .idx = c.staging_slot, .kind = .staging });
            }
        }

        const vkd = self.ctx.vkd;
        if (copy_count > 0) {
            self.shell.setTileCount(self.tile_set.tile_count);
            vkd.cmdCopyBuffer(cmd_buf, self.staging.buffer, self.shell.buf.buffer, copies[0..copy_count]);
        }
        const dir_flushed = self.shell.recordHeaderDirUpdate(vkd, cmd_buf);
        const grid_flushed = self.flushGridCells(vkd, cmd_buf);

        if (copy_count == 0 and !dir_flushed and !grid_flushed) return false;

        // dst includes transfer_write to chain WAW between consecutive frames'
        // cmdUpdateBuffer calls on the SSBO header+dir region.
        const barrier = [1]vk.MemoryBarrier{.{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true, .transfer_write_bit = true },
        }};
        vkd.cmdPipelineBarrier(
            cmd_buf,
            .{ .transfer_bit = true },
            .{ .compute_shader_bit = true, .transfer_bit = true },
            .{},
            &barrier,
            null,
            null,
        );
        return true;
    }

    /// Unbind dir entry and quarantine its data slot for `MAX_FRAMES_IN_FLIGHT`,
    /// so any in-flight CB that captured the pre-evict directory finishes
    /// reading the slot bytes before another tile can claim them.
    pub fn evict(self: *TileSystem, dir_idx: u32, frame: u64) void {
        // Capture the evicted descriptor's cell BEFORE evict swaps state.
        const evicted_desc = self.tile_set.descriptor(dir_idx);
        const evicted_cell = coords.worldOriginToGridCell(evicted_desc.origin_x, evicted_desc.origin_z);

        const result = self.tile_set.evict(dir_idx);
        self.setGridCell(evicted_cell, 0);

        if (result.swapped_dir_idx) |sdi| {
            // The tile previously at `last` now occupies `sdi`. Update its
            // grid cell to point to the new dir index; if we skipped this,
            // findTile would return the old (freed) dir for that tile.
            const moved_desc = self.tile_set.descriptor(sdi);
            const moved_slot = self.tile_set.slotForDir(sdi);
            self.shell.setDirEntry(sdi, moved_desc, moved_slot);
            const moved_cell = coords.worldOriginToGridCell(moved_desc.origin_x, moved_desc.origin_z);
            self.setGridCell(moved_cell, sdi + 1);
        }
        self.shell.setTileCount(self.tile_set.tile_count);
        self.pending.appendAssumeCapacity(.{ .frame = frame, .idx = result.freed_slot, .kind = .slot });
    }

    /// On success returns the BufferCopy region; on drop (load failed, no
    /// longer desired, or slot pool full) releases the staging slot inline
    /// and returns null. Successful staging-slot release is deferred to
    /// caller (frame quarantine in `recordStream`, sync drain in `drainAll`).
    fn applyCompletion(self: *TileSystem, c: tile_streamer.LoadComplete) ?vk.BufferCopy {
        std.debug.assert(self.loading.isSet(c.catalog_idx));
        self.loading.unset(c.catalog_idx);
        self.loading_count -= 1;
        if (c.weights_size == 0) {
            self.streamer.releaseStagingSlot(c.staging_slot);
            return null;
        }
        // Drop tiles no longer desired (camera moved past them while loading);
        // otherwise they consume slots and churn out on the next tick.
        if (!self.desired_mask.isSet(c.catalog_idx)) {
            self.streamer.releaseStagingSlot(c.staging_slot);
            return null;
        }
        const slot_idx = self.tile_set.claimSlot() orelse {
            std.log.warn("tile_system: slot pool full, dropping catalog idx={d}", .{c.catalog_idx});
            self.streamer.releaseStagingSlot(c.staging_slot);
            return null;
        };
        const dir_idx = self.tile_set.appendDir(c.catalog_idx, slot_idx);
        const desc = &self.catalog.descriptors[c.catalog_idx];
        self.shell.setDirEntry(dir_idx, desc, slot_idx);
        self.setGridCell(coords.worldOriginToGridCell(desc.origin_x, desc.origin_z), dir_idx + 1);
        return .{
            .src_offset = self.streamer.stagingByteOffset(c.staging_slot),
            .dst_offset = self.shell.slotByteOffset(slot_idx),
            .size = c.weights_size,
        };
    }

    /// Entries are appended in monotonic `frame` order from `recordStream` and
    /// `evict`, so once we see a non-stale entry the rest are also non-stale.
    /// Drain the contiguous stale prefix and shift the tail down once.
    fn releasePending(self: *TileSystem, current_frame: u64) void {
        var k: usize = 0;
        while (k < self.pending.items.len) : (k += 1) {
            const p = self.pending.items[k];
            if (current_frame < p.frame + MAX_FRAMES_IN_FLIGHT) break;
            switch (p.kind) {
                .staging => self.streamer.releaseStagingSlot(p.idx),
                .slot => self.tile_set.releaseSlot(p.idx),
            }
        }
        if (k > 0) self.pending.replaceRangeAssumeCapacity(0, k, &.{});
    }

    /// Requests still queued or being loaded by the streamer worker (not yet
    /// completed). Reaches 0 once the worker has drained its request queue, so an
    /// interactive startup loop can present a loading screen `while (inFlight() > 0)`.
    pub fn inFlight(self: *TileSystem) u32 {
        return self.streamer.inFlight();
    }

    /// Synchronous warm-up: drain every queued load via one-shot command
    /// buffers, releasing staging immediately after each batch. Used by
    /// autotune to avoid measuring tile-load latency in render benchmarks.
    pub fn drainAll(self: *TileSystem) !void {
        while (try self.drainSome()) {}
    }

    /// One batch of `drainAll`: block until at least one completion lands (or
    /// none remain), apply that batch via a one-shot command buffer, and return
    /// whether loads are still in flight afterward. Lets an interactive caller
    /// interleave work (event pumps, a loading frame) between batches:
    /// `while (try drainSome()) {}` is exactly `drainAll`.
    pub fn drainSome(self: *TileSystem) !bool {
        if (self.streamer.inFlight() == 0) return false;
        var batch: [STAGING_SLOTS]tile_streamer.LoadComplete = undefined;
        const n = try self.streamer.waitAndDrain(&batch);
        // n == 0 only when no completions remain and nothing is in flight.
        if (n == 0) return self.streamer.inFlight() > 0;

        var copies: [STAGING_SLOTS]vk.BufferCopy = undefined;
        // Track which staging slots applyCompletion succeeded on; failures
        // (load error, slot pool full) release inline. Releasing the same
        // slot twice trips releaseStagingSlot's double-free assert.
        var success_slots: [STAGING_SLOTS]u32 = undefined;
        var copy_count: u32 = 0;
        for (batch[0..n]) |c| {
            if (self.applyCompletion(c)) |region| {
                copies[copy_count] = region;
                success_slots[copy_count] = c.staging_slot;
                copy_count += 1;
            }
        }

        if (copy_count > 0) {
            self.shell.setTileCount(self.tile_set.tile_count);

            const ctx = self.ctx;
            const cmd_buf = try buffer.beginOneShot(ctx.vkd, ctx.device, ctx.cmd_pool);
            ctx.vkd.cmdCopyBuffer(cmd_buf, self.staging.buffer, self.shell.buf.buffer, copies[0..copy_count]);
            _ = self.shell.recordHeaderDirUpdate(ctx.vkd, cmd_buf);
            _ = self.flushGridCells(ctx.vkd, cmd_buf);
            try buffer.endOneShot(ctx.vkd, ctx.device, ctx.cmd_pool, ctx.queue, cmd_buf);

            for (success_slots[0..copy_count]) |s| self.streamer.releaseStagingSlot(s);
        }
        return self.streamer.inFlight() > 0;
    }
};

/// Per-frame completion drain cap. Steady state honors the configured
/// `max_uploads_per_frame` (clamped to the staging ceiling); fast-fill lifts it
/// to the full ceiling so a fresh/rebuilt ring repopulates in a few frames.
fn drainCap(fast_fill: bool, max_uploads_per_frame: u32) u32 {
    return if (fast_fill) STAGING_SLOTS else @min(STAGING_SLOTS, max_uploads_per_frame);
}

/// Desired (visible) tile count for a render distance, bounded by the fixed pool
/// `capacity`. The pool reserves churn slack (the MAX_FRAMES_IN_FLIGHT quarantine
/// of evicted slots under motion); the slack is zero once the working set covers
/// the whole catalog (no eviction can ever happen). Shared by `init` (the initial
/// render distance) and `setRenderDistance` (a runtime change), both bounded by
/// the same fixed `capacity` so a change never needs a realloc.
fn sizeDesiredCap(render_cap: u32, capacity: u32, catalog_cap: u32, min_headroom: u32) u32 {
    // Working set this render distance wants: the visible count plus sizing
    // headroom (max(MFIF*max_uploads, 15%)), clamped to the fixed pool capacity.
    const sizing_headroom: u32 = @max(min_headroom, render_cap * 3 / 20);
    const working = @max(tile_loader.MIN_TILE_POOL, @min(render_cap +| sizing_headroom, capacity));
    const churn: u32 = if (working >= catalog_cap)
        0
    else
        @max(min_headroom, @min(render_cap, working) * 3 / 20);
    // Fallback (only with extreme --max-tiles overrides): one slack slot keeps
    // streaming functional but degraded.
    return if (working > churn) working - churn else working - 1;
}

/// Append `value` keyed by `cell_idx`, overwriting any existing entry for the
/// same cell. Caller must have reserved capacity for at least one new entry.
fn appendGridUpdateDedup(updates: *std.ArrayList(TileSystem.GridUpdate), cell_idx: u32, value: u32) void {
    for (updates.items) |*update| {
        if (update.cell_idx == cell_idx) {
            update.value = value;
            return;
        }
    }
    updates.appendAssumeCapacity(.{ .cell_idx = cell_idx, .value = value });
}

// ---- Tests ----

const testing = std.testing;

test "drainCap: fast-fill lifts to the staging ceiling, steady state honors the cap" {
    try testing.expectEqual(STAGING_SLOTS, drainCap(true, 4)); // fast-fill -> ceiling
    try testing.expectEqual(STAGING_SLOTS, drainCap(true, 1)); // fast-fill ignores the small cap
    try testing.expectEqual(@as(u32, 4), drainCap(false, 4)); // steady state -> configured cap
    try testing.expectEqual(@as(u32, 1), drainCap(false, 1));
    try testing.expectEqual(STAGING_SLOTS, drainCap(false, 999)); // cap clamped to ceiling
}

test "sizeDesiredCap: tracks render distance, monotonic, bounded by capacity" {
    const cap: u32 = 37; // catalog-bound pool (e.g. the PNW set)
    const cat: u32 = 37;
    const mh: u32 = 8;
    // A small render distance desires only the visible set + slack, well below the
    // (max-sized) pool: this is the launch-small case that left the bug latent.
    const small = sizeDesiredCap(4, cap, cat, mh);
    try testing.expect(small < cap);
    // A large render distance saturates to the whole catalog (pool >= catalog ->
    // no churn slack), so a runtime increase reaches every tile.
    const large = sizeDesiredCap(100, cap, cat, mh);
    try testing.expectEqual(@as(u32, 37), large);
    // Growing the render distance never shrinks the desired set.
    try testing.expect(large >= small);
}

test "sizeDesiredCap: reserves churn slack below capacity, stays >= 1" {
    // Capacity below catalog (e.g. a small --max-tiles): the working set is capped
    // at the pool and still reserves slack, so desired stays strictly under it.
    const d = sizeDesiredCap(1000, 20, 500, 8);
    try testing.expect(d >= 1 and d < 20);
}

test "appendGridUpdateDedup: distinct cells append" {
    var updates: std.ArrayList(TileSystem.GridUpdate) = .empty;
    defer updates.deinit(testing.allocator);
    try updates.ensureTotalCapacity(testing.allocator, 4);

    appendGridUpdateDedup(&updates, 10, 1);
    appendGridUpdateDedup(&updates, 20, 2);
    appendGridUpdateDedup(&updates, 30, 3);

    try testing.expectEqual(@as(usize, 3), updates.items.len);
    try testing.expectEqual(@as(u32, 1), updates.items[0].value);
    try testing.expectEqual(@as(u32, 2), updates.items[1].value);
    try testing.expectEqual(@as(u32, 3), updates.items[2].value);
}

test "appendGridUpdateDedup: repeat cell overwrites in place" {
    var updates: std.ArrayList(TileSystem.GridUpdate) = .empty;
    defer updates.deinit(testing.allocator);
    try updates.ensureTotalCapacity(testing.allocator, 4);

    appendGridUpdateDedup(&updates, 10, 1);
    appendGridUpdateDedup(&updates, 20, 2);
    appendGridUpdateDedup(&updates, 10, 99); // overwrites cell 10's value
    appendGridUpdateDedup(&updates, 20, 42); // overwrites cell 20's value
    appendGridUpdateDedup(&updates, 10, 7); // overwrites cell 10 again

    try testing.expectEqual(@as(usize, 2), updates.items.len);
    try testing.expectEqual(@as(u32, 10), updates.items[0].cell_idx);
    try testing.expectEqual(@as(u32, 7), updates.items[0].value);
    try testing.expectEqual(@as(u32, 20), updates.items[1].cell_idx);
    try testing.expectEqual(@as(u32, 42), updates.items[1].value);
}

test "appendGridUpdateDedup: cascading-evict swap pattern produces one write per cell" {
    // Models the bug repro: evict_buf={9, 7, 5} on tile_count=10.
    // iter 1 (evict 9): no swap; queue [C9=0].
    // iter 2 (evict 7): T8 moves dir 8->7; queue [C9=0, C7=0, C8=8].
    // iter 3 (evict 5): T8 (now at dir 7) moves dir 7->5; queue C8 again.
    // Pre-fix, C8 appeared twice and tripped the validator. Post-fix, one entry.
    var updates: std.ArrayList(TileSystem.GridUpdate) = .empty;
    defer updates.deinit(testing.allocator);
    try updates.ensureTotalCapacity(testing.allocator, 8);

    // iter 1
    appendGridUpdateDedup(&updates, 9, 0);
    // iter 2
    appendGridUpdateDedup(&updates, 7, 0);
    appendGridUpdateDedup(&updates, 8, 8); // T8 at dir 7
    // iter 3
    appendGridUpdateDedup(&updates, 5, 0);
    appendGridUpdateDedup(&updates, 8, 6); // T8 now at dir 5; should overwrite, not append

    try testing.expectEqual(@as(usize, 4), updates.items.len);
    // C8 entry exists exactly once with the latest value.
    var c8_count: u32 = 0;
    var c8_value: u32 = 0;
    for (updates.items) |u| {
        if (u.cell_idx == 8) {
            c8_count += 1;
            c8_value = u.value;
        }
    }
    try testing.expectEqual(@as(u32, 1), c8_count);
    try testing.expectEqual(@as(u32, 6), c8_value);
}
