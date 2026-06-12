//! Async tile-weight streaming. Worker mmaps weights into externally-owned
//! staging slots; caller copies staging->SSBO and calls `releaseStagingSlot`
//! after the GPU has consumed each slot. No Vulkan dependencies; the
//! staging buffer is owned and mapped by the caller.

const std = @import("std");
const tile_loader = @import("tile_loader.zig");

const Io = std.Io;

pub const STAGING_SLOTS: u32 = 8;

pub const LoadRequest = struct {
    catalog_idx: u32,
};

/// `weights_size == 0` means load failed; release the slot, skip the copy.
pub const LoadComplete = struct {
    catalog_idx: u32,
    staging_slot: u32,
    weights_size: u32,
};

pub const TileStreamer = struct {
    allocator: std.mem.Allocator,
    catalog: *const tile_loader.TileCatalog,
    io: Io,

    /// Externally-owned staging memory. Layout: slot N occupies
    /// `staging[N * slot_stride .. N * slot_stride + slot_stride]`.
    staging: []u8,
    slot_stride: u32,

    mu: Io.Mutex = .init,
    request_cv: Io.Condition = .init,
    completion_cv: Io.Condition = .init,
    free_cv: Io.Condition = .init,

    requests: std.ArrayList(LoadRequest),
    completions: std.ArrayList(LoadComplete),
    free_slots: FreeSet,
    in_flight: u32 = 0,
    /// Atomic mirror of `completions.items.len` for `tryDrain`'s lock-free
    /// fast path. Stored under `mu`; loaded with `.acquire`.
    completion_count: std.atomic.Value(u32) = .init(0),

    worker_future: Io.Future(Io.Cancelable!void),

    const FreeSet = std.bit_set.IntegerBitSet(STAGING_SLOTS);

    pub fn init(
        allocator: std.mem.Allocator,
        catalog: *const tile_loader.TileCatalog,
        io: Io,
        staging: []u8,
        slot_stride: u32,
    ) !*TileStreamer {
        std.debug.assert(staging.len >= @as(usize, STAGING_SLOTS) * slot_stride);

        const self = try allocator.create(TileStreamer);
        errdefer allocator.destroy(self);

        var free_slots = FreeSet.initEmpty();
        for (0..STAGING_SLOTS) |i| free_slots.set(i);

        var requests: std.ArrayList(LoadRequest) = .empty;
        try requests.ensureTotalCapacity(allocator, tile_loader.MAX_TILES);
        errdefer requests.deinit(allocator);

        var completions: std.ArrayList(LoadComplete) = .empty;
        try completions.ensureTotalCapacity(allocator, tile_loader.MAX_TILES);
        errdefer completions.deinit(allocator);

        self.* = .{
            .allocator = allocator,
            .catalog = catalog,
            .io = io,
            .staging = staging,
            .slot_stride = slot_stride,
            .requests = requests,
            .completions = completions,
            .free_slots = free_slots,
            .worker_future = undefined,
        };

        self.worker_future = try io.concurrent(workerLoop, .{self});
        return self;
    }

    pub fn deinit(self: *TileStreamer) void {
        self.worker_future.cancel(self.io) catch {};
        self.requests.deinit(self.allocator);
        self.completions.deinit(self.allocator);
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    pub fn requestLoad(self: *TileStreamer, catalog_idx: u32) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.requests.appendAssumeCapacity(.{ .catalog_idx = catalog_idx });
        self.in_flight += 1;
        self.request_cv.signal(self.io);
    }

    /// Single lock acquire for the whole slice.
    pub fn requestLoadMany(self: *TileStreamer, catalog_idxs: []const u32) void {
        if (catalog_idxs.len == 0) return;
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        for (catalog_idxs) |idx| {
            self.requests.appendAssumeCapacity(.{ .catalog_idx = idx });
        }
        self.in_flight += @intCast(catalog_idxs.len);
        self.request_cv.signal(self.io);
    }

    /// Returns 0 only when nothing is in flight and nothing is queued.
    pub fn waitAndDrain(self: *TileStreamer, out: []LoadComplete) Io.Cancelable!usize {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.completions.items.len == 0 and self.in_flight > 0) {
            try self.completion_cv.wait(self.io, &self.mu);
        }
        const n = @min(out.len, self.completions.items.len);
        for (0..n) |i| out[i] = self.completions.orderedRemove(0);
        self.completion_count.store(@intCast(self.completions.items.len), .release);
        return n;
    }

    /// Non-blocking drain. Lock-free fast path when no completions have
    /// landed since the last call.
    pub fn tryDrain(self: *TileStreamer, out: []LoadComplete) usize {
        if (self.completion_count.load(.acquire) == 0) return 0;
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const n = @min(out.len, self.completions.items.len);
        for (0..n) |i| out[i] = self.completions.orderedRemove(0);
        self.completion_count.store(@intCast(self.completions.items.len), .release);
        return n;
    }

    /// Number of requests still in flight (queued or being processed).
    pub fn inFlight(self: *TileStreamer) u32 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return self.in_flight;
    }

    /// Caller must have observed GPU consumption of `staging_slot` first.
    pub fn releaseStagingSlot(self: *TileStreamer, staging_slot: u32) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        std.debug.assert(!self.free_slots.isSet(staging_slot));
        self.free_slots.set(staging_slot);
        self.free_cv.signal(self.io);
    }

    pub fn stagingByteOffset(self: *const TileStreamer, staging_slot: u32) u64 {
        return @as(u64, staging_slot) * self.slot_stride;
    }

    // ---- Worker task ----

    // TODO(cold-start SSD throughput): one worker drains the queue serially and
    // loadWeightsInto mmaps + page-faults each weights.bin in, so storage queue
    // depth is effectively 1. A PCIe 5.0 NVMe needs ~QD8-16 to reach its
    // sequential ceiling (measured QD1 ~1.8 GB/s vs QD16 ~14 GB/s), so cold start
    // uses ~1/6 of the drive. Warm runs hit the page cache, so this only costs on
    // a cold cache; but it dominates first-load latency and hurts most on the RPi
    // target (SD card / eMMC: same serialization tax, far less bandwidth). To
    // actually saturate the drive:
    //   1. Spawn N workers (io.concurrent xN) draining `requests` so N reads are
    //      in flight at once. STAGING_SLOTS (8) already sizes the ring for this;
    //      raise it if N grows.
    //   2. Replace mmap+memcpy with one large pread/readv into the staging slot
    //      (or fadvise WILLNEED): mmap faults pages in piecemeal with per-fault
    //      overhead; one big read lets the kernel pipeline the whole ~771 KB.
    //   3. Or submit the reads through the Io runtime's async file API so a
    //      single thread keeps the queue full (io_uring-style).
    //   4. Structural win: pack tiles into one catalog blob / a few shards so
    //      cold start is one sequential stream instead of 1222 open()+mmap
    //      cycles, also amortizing metadata lookups against cold dentry/inode.
    fn workerLoop(self: *TileStreamer) Io.Cancelable!void {
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        while (true) {
            const req = try self.popRequestBlocking();
            const staging_slot = try self.acquireStagingSlotBlocking();
            const bytes = try self.loadIntoStaging(req.catalog_idx, staging_slot, &path_buf);
            self.pushCompletion(.{
                .catalog_idx = req.catalog_idx,
                .staging_slot = staging_slot,
                .weights_size = bytes, // 0 means load failed
            });
        }
    }

    fn popRequestBlocking(self: *TileStreamer) Io.Cancelable!LoadRequest {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (self.requests.items.len == 0) {
            try self.request_cv.wait(self.io, &self.mu);
        }
        return self.requests.orderedRemove(0);
    }

    fn acquireStagingSlotBlocking(self: *TileStreamer) Io.Cancelable!u32 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        while (true) {
            if (self.free_slots.findFirstSet()) |s| {
                self.free_slots.unset(s);
                return @intCast(s);
            }
            try self.free_cv.wait(self.io, &self.mu);
        }
    }

    /// Returns 0 on a non-cancel error (already logged); Canceled propagates.
    fn loadIntoStaging(
        self: *TileStreamer,
        catalog_idx: u32,
        staging_slot: u32,
        path_buf: []u8,
    ) Io.Cancelable!u32 {
        const off = staging_slot * self.slot_stride;
        const dst = self.staging[off .. off + self.slot_stride];
        return tile_loader.loadWeightsInto(self.io, self.catalog, catalog_idx, path_buf, dst) catch |err| switch (err) {
            error.Canceled => return error.Canceled,
            else => {
                std.log.warn(
                    "tile_streamer: failed to load catalog idx {d}: {}",
                    .{ catalog_idx, err },
                );
                return 0;
            },
        };
    }

    fn pushCompletion(self: *TileStreamer, c: LoadComplete) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.completions.appendAssumeCapacity(c);
        std.debug.assert(self.in_flight > 0);
        self.in_flight -= 1;
        self.completion_count.store(@intCast(self.completions.items.len), .release);
        self.completion_cv.signal(self.io);
    }
};

// ---- Tests ----

const testing = std.testing;

fn writeFixtureMeta(io: Io, dir: Io.Dir, name: []const u8, weights_bytes: u32) !void {
    var sub = try dir.createDirPathOpen(io, name, .{});
    defer sub.close(io);

    // Reverse the formula in tile_loader.parseTileMeta:
    //   weights_bytes = (grid_uints + 2*features + mlp_floats) * 4
    // Pick features=4, mlp_hidden_dim=4, num_outputs=1 -> fixed_uints = 29.
    const features: u32 = 4;
    const mlp_hidden_dim: u32 = 4;
    const num_outputs: u32 = 1;
    const mlp_floats = mlp_hidden_dim * features + mlp_hidden_dim + mlp_hidden_dim * num_outputs + num_outputs;
    const fixed_uints = 2 * features + mlp_floats; // 8 + 21 = 29
    std.debug.assert(weights_bytes % 4 == 0);
    std.debug.assert(weights_bytes / 4 >= fixed_uints);
    const grid_uints = weights_bytes / 4 - fixed_uints;

    var json_buf: [512]u8 = undefined;
    const json = try std.fmt.bufPrint(&json_buf,
        \\{{
        \\  "resolution": 64,
        \\  "resolution_w": 64,
        \\  "features": {d},
        \\  "mlp_hidden_dim": {d},
        \\  "num_outputs": {d},
        \\  "grid_uints": {d},
        \\  "elev_min": 0.0,
        \\  "elev_max": 1000.0,
        \\  "grad_scale": 1.0
        \\}}
    , .{ features, mlp_hidden_dim, num_outputs, grid_uints });

    var meta_file = try sub.createFile(io, "meta.json", .{});
    defer meta_file.close(io);
    try meta_file.writeStreamingAll(io, json);
}

fn writeFixtureWeights(io: Io, dir: Io.Dir, name: []const u8, fill_byte: u8, weights_bytes: u32) !void {
    var sub = try dir.openDir(io, name, .{});
    defer sub.close(io);
    var weights_file = try sub.createFile(io, "weights.bin", .{});
    defer weights_file.close(io);
    const buf = try testing.allocator.alloc(u8, weights_bytes);
    defer testing.allocator.free(buf);
    @memset(buf, fill_byte);
    try weights_file.writeStreamingAll(io, buf);
}

test "TileStreamer: load synthetic catalog into staging" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const tile_names = [_][]const u8{ "n47w122", "n47w123", "n48w122" };
    for (tile_names, 0..) |name, i| {
        try writeFixtureMeta(io, tmp.dir, name, 256);
        try writeFixtureWeights(io, tmp.dir, name, @intCast(0xA0 + i), 256);
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp.dir.realPath(io, &path_buf);
    const tmp_path = path_buf[0..tmp_path_len];

    var catalog = try tile_loader.scanCatalog(io, tmp_path, testing.allocator);
    defer catalog.deinit();
    try testing.expectEqual(@as(u32, 3), catalog.count());

    const slot_stride: u32 = 256;
    const staging = try testing.allocator.alloc(u8, STAGING_SLOTS * slot_stride);
    defer testing.allocator.free(staging);
    @memset(staging, 0);

    const streamer = try TileStreamer.init(testing.allocator, &catalog, io, staging, slot_stride);
    defer streamer.deinit();

    for (0..catalog.count()) |i| {
        streamer.requestLoad(@intCast(i));
    }

    var drained: u32 = 0;
    var batch: [STAGING_SLOTS]LoadComplete = undefined;
    while (true) {
        const n = streamer.waitAndDrain(&batch) catch break;
        if (n == 0) break;
        for (batch[0..n]) |c| {
            try testing.expectEqual(@as(u32, 256), c.weights_size);
            // Each staging slot's bytes match the fill byte for that
            // catalog idx (catalog order matches request order here).
            const expected: u8 = @intCast(0xA0 + c.catalog_idx);
            const off = streamer.stagingByteOffset(c.staging_slot);
            try testing.expectEqual(expected, staging[off]);
            try testing.expectEqual(expected, staging[off + 255]);
            streamer.releaseStagingSlot(c.staging_slot);
            drained += 1;
        }
        if (drained == catalog.count()) break;
    }
    try testing.expectEqual(@as(u32, 3), drained);
}

test "TileStreamer: deinit cancels worker with no pending work" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    try writeFixtureMeta(io, tmp.dir, "n47w122", 256);
    try writeFixtureWeights(io, tmp.dir, "n47w122", 0xCC, 256);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp.dir.realPath(io, &path_buf);
    const tmp_path = path_buf[0..tmp_path_len];

    var catalog = try tile_loader.scanCatalog(io, tmp_path, testing.allocator);
    defer catalog.deinit();

    const slot_stride: u32 = 256;
    const staging = try testing.allocator.alloc(u8, STAGING_SLOTS * slot_stride);
    defer testing.allocator.free(staging);

    const streamer = try TileStreamer.init(testing.allocator, &catalog, io, staging, slot_stride);
    streamer.deinit(); // no requests; worker waiting on request_cv, cancel breaks it
}

test "TileStreamer: more requests than staging slots blocks then drains" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // STAGING_SLOTS+2 tiles forces the worker to block on free_cv at least
    // once. Names sort lexicographically so iteration order is predictable.
    const tile_count: u32 = STAGING_SLOTS + 2;
    var name_storage: [tile_count][7]u8 = undefined;
    for (&name_storage, 0..) |*name, i| {
        _ = std.fmt.bufPrint(name, "n{d:0>2}w122", .{i}) catch unreachable;
        try writeFixtureMeta(io, tmp.dir, name, 256);
        try writeFixtureWeights(io, tmp.dir, name, @intCast(0x10 + i), 256);
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp.dir.realPath(io, &path_buf);
    const tmp_path = path_buf[0..tmp_path_len];

    var catalog = try tile_loader.scanCatalog(io, tmp_path, testing.allocator);
    defer catalog.deinit();
    try testing.expectEqual(tile_count, catalog.count());

    const slot_stride: u32 = 256;
    const staging = try testing.allocator.alloc(u8, STAGING_SLOTS * slot_stride);
    defer testing.allocator.free(staging);
    @memset(staging, 0);

    const streamer = try TileStreamer.init(testing.allocator, &catalog, io, staging, slot_stride);
    defer streamer.deinit();

    for (0..catalog.count()) |i| streamer.requestLoad(@intCast(i));

    var seen = std.bit_set.IntegerBitSet(tile_count).initEmpty();
    var batch: [STAGING_SLOTS]LoadComplete = undefined;
    while (seen.count() < tile_count) {
        const n = try streamer.waitAndDrain(&batch);
        try testing.expect(n > 0);
        for (batch[0..n]) |c| {
            try testing.expectEqual(@as(u32, 256), c.weights_size);
            try testing.expect(!seen.isSet(c.catalog_idx));
            seen.set(c.catalog_idx);
            streamer.releaseStagingSlot(c.staging_slot);
        }
    }
}

test "TileStreamer: cancel mid-flight with pending requests" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // Enough tiles that not all can fit in staging; deinit() must unblock
    // the worker even if it's parked on free_cv waiting for a slot.
    const tile_count: u32 = STAGING_SLOTS * 2;
    var name_storage: [tile_count][7]u8 = undefined;
    for (&name_storage, 0..) |*name, i| {
        _ = std.fmt.bufPrint(name, "n{d:0>2}w122", .{i}) catch unreachable;
        try writeFixtureMeta(io, tmp.dir, name, 256);
        try writeFixtureWeights(io, tmp.dir, name, 0x42, 256);
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp.dir.realPath(io, &path_buf);
    const tmp_path = path_buf[0..tmp_path_len];

    var catalog = try tile_loader.scanCatalog(io, tmp_path, testing.allocator);
    defer catalog.deinit();

    const slot_stride: u32 = 256;
    const staging = try testing.allocator.alloc(u8, STAGING_SLOTS * slot_stride);
    defer testing.allocator.free(staging);

    const streamer = try TileStreamer.init(testing.allocator, &catalog, io, staging, slot_stride);
    for (0..catalog.count()) |i| streamer.requestLoad(@intCast(i));
    // Don't drain; worker fills STAGING_SLOTS and parks on free_cv. deinit
    // cancels and joins the worker; nothing should hang.
    streamer.deinit();
}
