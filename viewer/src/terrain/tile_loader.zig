//! Pure I/O tile loading; no Vulkan dependencies.
//!
//! `scanCatalog` walks a directory of tiles and parses `meta.json` for each
//! into a `TileCatalog` (no weights mmap). `TileSet` is a slot pool over the
//! catalog. `loadWeightsInto` mmaps + memcpys a tile's weights.bin into a
//! caller-provided destination buffer.

const std = @import("std");
const coords = @import("coords.zig");
const cull = @import("clipmap_cull.zig");

const Io = std.Io;
const Dir = Io.Dir;
const File = Io.File;

/// Compile-time upper bound on resident tiles. Fixed arrays (SlotMask, directory
/// shadow) are sized to this; the runtime tile_pool_size (the whole catalog or the
/// VRAM ceiling, capped by --max-tiles) controls the actual GPU allocation.
/// 8192 covers North America with headroom (~360 KB struct overhead).
/// The pool is sized for the max render distance at init, so a runtime
/// ring_size/num_levels change only grows the desired (visible) set
/// (`TileSystem.setRenderDistance`) and never reallocates.
pub const MAX_TILES: u32 = 8192;

pub const MIN_TILE_POOL: u32 = 4;

/// Outermost-level snap-safe visible radius in arcsec; the rendered set.
pub fn renderRadiusArcsec(ring_size: u32, num_levels: u32, base_spacing: f32) f32 {
    const g_outer = base_spacing * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(num_levels - 1)));
    return @sqrt(cull.rSafeSq(ring_size, g_outer));
}

/// Estimate max visible tiles for a given clipmap config (bounding-box count).
pub fn tilesForRenderDistance(ring_size: u32, num_levels: u32, base_spacing: f32) u32 {
    const r_arcsec = renderRadiusArcsec(ring_size, num_levels, base_spacing);
    const side = @ceil(2.0 * r_arcsec / coords.TILE_ARCSEC) + 1.0;
    const raw: u32 = @intFromFloat(@max(1.0, side * side));
    return @max(MIN_TILE_POOL, @min(MAX_TILES, raw));
}

const TILE_NAME_LEN: usize = 7;

/// Per-tile metadata. Lives in the catalog; resident slots reference these by
/// index.
pub const TileDescriptor = struct {
    /// Tile name as on disk (e.g. "n47w122"). Fixed length, not null-terminated.
    name: [TILE_NAME_LEN]u8,
    origin_x: f32,
    origin_z: f32,
    resolution_h: u32,
    resolution_w: u32,
    grid_uints: u32,
    elev_min: f32,
    elev_max: f32,
    weights_bytes: u32,
    grad_scale: f32 = 0.0,
    /// `cos(latitude)` at the tile's center, precomputed so policy ranking
    /// uses the tile's own latitude (not the anchor's) when scaling lon-arcsec
    /// distances. Defaults to 1.0 for test fixtures near the equator.
    cos_lat: f32 = 1.0,
};

/// All tiles discovered on disk plus the shared MLP/feature parameters they
/// were trained against. Built once at startup; treated as immutable
/// thereafter. The streamer reads from a catalog reference; nobody mutates it.
pub const TileCatalog = struct {
    allocator: std.mem.Allocator,
    /// Owned slice. Entries are in directory-iteration order; callers should
    /// not assume any spatial sort.
    descriptors: []TileDescriptor,
    /// Root directory the catalog was scanned from. Owned. In single-tile
    /// mode this *is* the tile directory; in multi-tile mode each tile lives
    /// in `dir_path/<name>/`.
    dir_path: []u8,
    /// True if `dir_path` itself contained a `meta.json` (single-tile mode).
    /// In that case the sole descriptor's `name` is the basename of dir_path.
    single_tile: bool,
    shared_features: u32,
    shared_mlp_hidden_dim: u32,
    shared_num_outputs: u32,
    /// Packed bitset: one bit per 1x1 degree cell. Set if the catalog
    /// contains a tile covering that cell. ~8 KB, fits in L1 cache.
    catalog_grid: [catalog_grid_len]u64,

    const catalog_grid_len = (coords.GRID_CELL_COUNT + 63) / 64;

    pub fn hasCatalogTile(self: *const TileCatalog, cell_idx: u32) bool {
        return (self.catalog_grid[cell_idx / 64] >> @intCast(cell_idx % 64)) & 1 != 0;
    }

    pub fn deinit(self: *TileCatalog) void {
        self.allocator.free(self.descriptors);
        self.allocator.free(self.dir_path);
    }

    pub fn count(self: *const TileCatalog) u32 {
        return @intCast(self.descriptors.len);
    }

    /// Worst-case `weights_bytes` across all descriptors. The fixed-stride
    /// SSBO slot pool and the streamer staging ring both use this so any
    /// catalog tile fits any slot.
    pub fn slotStride(self: *const TileCatalog) u32 {
        var s: u32 = 0;
        for (self.descriptors) |desc| s = @max(s, desc.weights_bytes);
        return s;
    }

    /// Build the path to a tile's weights.bin into `out`. Returns the slice
    /// of `out` that was written.
    pub fn weightsPath(self: *const TileCatalog, idx: u32, out: []u8) ![]u8 {
        if (self.single_tile) {
            return std.fmt.bufPrint(out, "{s}/weights.bin", .{self.dir_path});
        }
        const name = &self.descriptors[idx].name;
        return std.fmt.bufPrint(out, "{s}/{s}/weights.bin", .{ self.dir_path, name });
    }
};

/// Slot pool over the catalog. Two parallel arrays index by *directory index*
/// (the SSBO directory's compact prefix `0..tile_count`):
///   `resident_indices[d]`: which catalog entry occupies directory slot `d`.
///   `slot_indices[d]`    : which data-region slot holds that entry's bytes.
///
/// Directory indices and data-region slots are decoupled. A tile loaded from
/// the catalog gets a free data slot at load time; the directory entry stores
/// `data_offset = data_base + slot_idx * SLOT_STRIDE`. Eviction is swap-remove
/// of a directory entry; the freed data slot returns to `free_slots` and can
/// be reused for the next load.
pub const TileSet = struct {
    catalog: *const TileCatalog,
    resident_indices: [MAX_TILES]u32 = undefined,
    slot_indices: [MAX_TILES]u32 = undefined,
    free_slots: SlotMask = SlotMask.initFull(),
    /// Maintained alongside `free_slots` to avoid O(pool_bits/64) popcount on
    /// hot-path budget checks; queried via `freeCount`.
    free_count: u32 = 0,
    tile_count: u32 = 0,
    pool_size: u32 = MAX_TILES,
    /// O(1) `isResident`. Sized to `catalog.count()` because `rankDesired`
    /// queries it once per catalog tile per frame.
    resident_mask: std.DynamicBitSetUnmanaged,

    pub const SlotMask = std.bit_set.IntegerBitSet(MAX_TILES);

    pub fn init(allocator: std.mem.Allocator, catalog: *const TileCatalog, tile_pool_size: u32) !TileSet {
        std.debug.assert(tile_pool_size >= MIN_TILE_POOL and tile_pool_size <= MAX_TILES);
        var free = SlotMask.initEmpty();
        for (0..tile_pool_size) |i| free.set(i);
        return .{
            .catalog = catalog,
            .pool_size = tile_pool_size,
            .free_slots = free,
            .free_count = tile_pool_size,
            .resident_mask = try std.DynamicBitSetUnmanaged.initEmpty(allocator, catalog.count()),
        };
    }

    pub fn deinit(self: *TileSet, allocator: std.mem.Allocator) void {
        self.resident_mask.deinit(allocator);
    }

    /// Catalog descriptor at directory index `dir_idx`.
    pub fn descriptor(self: *const TileSet, dir_idx: u32) *const TileDescriptor {
        return &self.catalog.descriptors[self.resident_indices[dir_idx]];
    }

    /// Data-region slot holding the bytes for directory index `dir_idx`.
    pub fn slotForDir(self: *const TileSet, dir_idx: u32) u32 {
        return self.slot_indices[dir_idx];
    }

    /// Reserve a free data-region slot. Returns null when the pool is full.
    pub fn claimSlot(self: *TileSet) ?u32 {
        const s = self.free_slots.findFirstSet() orelse return null;
        self.free_slots.unset(s);
        self.free_count -= 1;
        return @intCast(s);
    }

    pub fn freeCount(self: *const TileSet) u32 {
        return self.free_count;
    }

    /// Return a data-region slot to the free pool. Caller is responsible for
    /// frame-quarantine: the slot must not be referenced by any in-flight
    /// command buffer.
    pub fn releaseSlot(self: *TileSet, slot_idx: u32) void {
        std.debug.assert(slot_idx < self.pool_size);
        std.debug.assert(!self.free_slots.isSet(slot_idx));
        self.free_slots.set(slot_idx);
        self.free_count += 1;
    }

    /// Append a directory entry binding `catalog_idx` to data-region slot
    /// `slot_idx`. Returns the new directory index. Caller must have already
    /// claimed `slot_idx`.
    pub fn appendDir(self: *TileSet, catalog_idx: u32, slot_idx: u32) u32 {
        std.debug.assert(self.tile_count < self.pool_size);
        std.debug.assert(!self.resident_mask.isSet(catalog_idx));
        const dir_idx = self.tile_count;
        self.resident_indices[dir_idx] = catalog_idx;
        self.slot_indices[dir_idx] = slot_idx;
        self.tile_count += 1;
        self.resident_mask.set(catalog_idx);
        return dir_idx;
    }

    pub fn isResident(self: *const TileSet, catalog_idx: u32) bool {
        return self.resident_mask.isSet(catalog_idx);
    }

    /// `swapped_dir_idx` is the dir index whose SSBO shadow the caller must
    /// rewrite (null if `dir_idx` was the last entry). Caller must quarantine
    /// `freed_slot` for `MAX_FRAMES_IN_FLIGHT` before `releaseSlot`.
    pub const EvictResult = struct {
        freed_slot: u32,
        swapped_dir_idx: ?u32,
    };


    pub fn evict(self: *TileSet, dir_idx: u32) EvictResult {
        std.debug.assert(dir_idx < self.tile_count);
        self.resident_mask.unset(self.resident_indices[dir_idx]);
        const freed_slot = self.slot_indices[dir_idx];
        const last = self.tile_count - 1;
        const swapped: ?u32 = if (dir_idx == last) null else blk: {
            self.resident_indices[dir_idx] = self.resident_indices[last];
            self.slot_indices[dir_idx] = self.slot_indices[last];
            break :blk dir_idx;
        };
        self.tile_count = last;
        return .{ .freed_slot = freed_slot, .swapped_dir_idx = swapped };
    }
};

/// Scan a directory for tiles. Single-tile mode if `dir_path/meta.json`
/// exists; otherwise iterate `dir_path/*/meta.json`. Does not mmap weights.
///
/// On success, the returned catalog owns its allocations; caller must
/// `deinit` it.
// TODO: replace per-tile mmap of <1 KB meta.json with a stack-buffer read;
// mmap setup costs (TLB, VMA, page-fault) dominate for tiny files. Now relevant
// (catalogs are 1000+ tiles), though this is the smaller cold-start cost: the
// weight-load path is the dominant one, see the worker TODO in tile_streamer.zig.
pub fn scanCatalog(io: Io, dir_path: []const u8, allocator: std.mem.Allocator) !TileCatalog {
    const cwd = Dir.cwd();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    var descriptors: std.ArrayList(TileDescriptor) = .empty;
    errdefer descriptors.deinit(allocator);

    var single_tile = false;
    var shared_features: u32 = 0;
    var shared_mlp_hidden_dim: u32 = 0;
    var shared_num_outputs: u32 = 1;

    const single_meta_path = std.fmt.bufPrint(&path_buf, "{s}/meta.json", .{dir_path}) catch return error.PathTooLong;
    if (mmapFile(io, cwd, single_meta_path)) |single_meta| {
        var meta_mm = single_meta;
        defer destroyMmap(io, &meta_mm);

        const tile_name = std.fs.path.basename(dir_path);
        if (tile_name.len != TILE_NAME_LEN) {
            std.log.err("Single-tile dir basename '{s}' is not {d} chars", .{ tile_name, TILE_NAME_LEN });
            return error.WeightsLoadFailed;
        }

        const parsed = try parseTileMeta(meta_mm.memory, tile_name, allocator);
        var desc = parsed.meta;
        @memcpy(&desc.name, tile_name);

        try descriptors.append(allocator, desc);
        single_tile = true;
        shared_features = parsed.features;
        shared_mlp_hidden_dim = parsed.mlp_hidden_dim;
        shared_num_outputs = parsed.num_outputs;

        std.log.debug("Catalog: single tile {s} (plane {d}x{d}x{d}, outputs={d})", .{
            tile_name,
            parsed.meta.resolution_h,
            parsed.meta.resolution_w,
            parsed.features,
            parsed.num_outputs,
        });
    } else |_| {
        var root_dir = cwd.openDir(io, dir_path, .{ .iterate = true }) catch |err| {
            std.log.err("Failed to open directory '{s}': {}", .{ dir_path, err });
            return error.WeightsLoadFailed;
        };
        defer root_dir.close(io);

        var iter = root_dir.iterate();
        while (iter.next(io) catch |err| {
            std.log.err("Directory iteration error: {}", .{err});
            return error.WeightsLoadFailed;
        }) |entry| {
            if (entry.kind != .directory) continue;
            if (entry.name.len != TILE_NAME_LEN) continue;
            _ = coords.tileToWorldRuntime(entry.name) catch continue;

            const meta_sub = std.fmt.bufPrint(&path_buf, "{s}/meta.json", .{entry.name}) catch continue;
            var meta_mm = mmapFile(io, root_dir, meta_sub) catch continue;
            defer destroyMmap(io, &meta_mm);

            const parsed = parseTileMeta(meta_mm.memory, entry.name, allocator) catch continue;

            if (descriptors.items.len == 0) {
                shared_features = parsed.features;
                shared_mlp_hidden_dim = parsed.mlp_hidden_dim;
                shared_num_outputs = parsed.num_outputs;
            } else if (parsed.features != shared_features or
                parsed.mlp_hidden_dim != shared_mlp_hidden_dim or
                parsed.num_outputs != shared_num_outputs)
            {
                std.log.err("Tile '{s}' has features={d} mlp_hidden_dim={d} num_outputs={d}, expected {d}/{d}/{d}; skipping", .{
                    entry.name,      parsed.features,         parsed.mlp_hidden_dim,         parsed.num_outputs,
                    shared_features, shared_mlp_hidden_dim, shared_num_outputs,
                });
                continue;
            }

            var desc = parsed.meta;
            @memcpy(&desc.name, entry.name);
            try descriptors.append(allocator, desc);
        }

        if (descriptors.items.len == 0) {
            std.log.err("No valid tile directories found in '{s}'", .{dir_path});
            return error.WeightsLoadFailed;
        }

        std.log.debug("Catalog: {d} tiles in '{s}' (features={d}, mlp_hidden_dim={d}, outputs={d})", .{
            descriptors.items.len, dir_path, shared_features, shared_mlp_hidden_dim, shared_num_outputs,
        });
    }

    const owned_dir = try allocator.dupe(u8, dir_path);
    errdefer allocator.free(owned_dir);

    var catalog_grid: [TileCatalog.catalog_grid_len]u64 = @splat(0);
    for (descriptors.items) |desc| {
        const cell = coords.worldOriginToGridCell(desc.origin_x, desc.origin_z);
        catalog_grid[cell / 64] |= @as(u64, 1) << @intCast(cell % 64);
    }

    return .{
        .allocator = allocator,
        .descriptors = try descriptors.toOwnedSlice(allocator),
        .dir_path = owned_dir,
        .single_tile = single_tile,
        .shared_features = shared_features,
        .shared_mlp_hidden_dim = shared_mlp_hidden_dim,
        .shared_num_outputs = shared_num_outputs,
        .catalog_grid = catalog_grid,
    };
}

/// Mmap weights.bin for `idx`, validate size against the descriptor, copy
/// into `dst[0..weights_bytes]`, and unmap. `dst` must be at least
/// `descriptor(idx).weights_bytes` long.
pub fn loadWeightsInto(
    io: Io,
    catalog: *const TileCatalog,
    idx: u32,
    path_buf: []u8,
    dst: []u8,
) !u32 {
    const desc = &catalog.descriptors[idx];
    std.debug.assert(dst.len >= desc.weights_bytes);

    const path = try catalog.weightsPath(idx, path_buf);
    var mm = mmapFile(io, Dir.cwd(), path) catch |err| switch (err) {
        error.Canceled => return err,
        else => {
            std.log.warn("Failed to mmap '{s}': {}", .{ path, err });
            return err;
        },
    };
    defer destroyMmap(io, &mm);

    if (mm.memory.len != desc.weights_bytes) {
        std.log.warn("'{s}': expected {d} bytes, got {d}", .{ path, desc.weights_bytes, mm.memory.len });
        return error.WeightsLoadFailed;
    }

    @memcpy(dst[0..desc.weights_bytes], mm.memory);
    return desc.weights_bytes;
}

// ---- Internal helpers ----

/// Open `<dir>/<path>` read-only and mmap its full contents. The returned
/// MemoryMap owns the underlying File handle (in `mm.file`); free it with
/// `destroyMmap`.
fn mmapFile(io: Io, dir: Dir, path: []const u8) !File.MemoryMap {
    const file = try dir.openFile(io, path, .{ .mode = .read_only });
    errdefer file.close(io);

    const len_u64 = try file.length(io);
    if (len_u64 == 0) return error.EmptyFile;
    const len: usize = std.math.cast(usize, len_u64) orelse return error.FileTooLarge;

    return File.MemoryMap.create(io, file, .{
        .len = len,
        .protection = .{ .read = true, .write = false },
    });
}

/// `MemoryMap.destroy` does not close the source file (verified in
/// std/Io/Threaded.zig:fileMemoryMapDestroy) and clobbers `mm.*` to undefined,
/// so the file handle has to be copied out before destroy is called.
fn destroyMmap(io: Io, mm: *File.MemoryMap) void {
    const file = mm.file;
    mm.destroy(io);
    file.close(io);
}

fn getJsonInt(root: std.json.ObjectMap, key: []const u8) ?u32 {
    const val = root.get(key) orelse return null;
    return switch (val) {
        .integer => |i| std.math.cast(u32, i),
        else => null,
    };
}

fn getJsonFloat(root: std.json.ObjectMap, key: []const u8) ?f32 {
    const val = root.get(key) orelse return null;
    return switch (val) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

const ParsedMeta = struct {
    meta: TileDescriptor,
    features: u32,
    mlp_hidden_dim: u32,
    num_outputs: u32,
};

fn parseTileMeta(meta_data: []const u8, tile_name: []const u8, allocator: std.mem.Allocator) !ParsedMeta {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, meta_data, .{}) catch |err| {
        std.log.err("Failed to parse meta.json for '{s}': {}", .{ tile_name, err });
        return error.WeightsLoadFailed;
    };
    defer parsed.deinit();
    const root = parsed.value.object;

    const resolution_h: u32 = getJsonInt(root, "resolution") orelse {
        std.log.err("{s}/meta.json: missing 'resolution' field", .{tile_name});
        return error.WeightsLoadFailed;
    };
    const resolution_w: u32 = getJsonInt(root, "resolution_w") orelse resolution_h;
    const features: u32 = getJsonInt(root, "features") orelse {
        std.log.err("{s}/meta.json: missing 'features' field", .{tile_name});
        return error.WeightsLoadFailed;
    };
    const mlp_hidden_dim: u32 = getJsonInt(root, "mlp_hidden_dim") orelse {
        std.log.err("{s}/meta.json: missing 'mlp_hidden_dim' field", .{tile_name});
        return error.WeightsLoadFailed;
    };
    const grid_uints: u32 = getJsonInt(root, "grid_uints") orelse {
        if (root.get("grid_floats") != null) {
            std.log.err("{s}/meta.json: found 'grid_floats' but not 'grid_uints' -- weights.bin is in float32 format. Re-export with export_weights_plane.py.", .{tile_name});
        } else {
            std.log.err("{s}/meta.json: missing 'grid_uints' field", .{tile_name});
        }
        return error.WeightsLoadFailed;
    };
    const num_outputs: u32 = getJsonInt(root, "num_outputs") orelse 1;
    if (features % 4 != 0) {
        std.log.err("{s}/meta.json: features ({d}) must be divisible by 4 for uint8 packing", .{ tile_name, features });
        return error.WeightsLoadFailed;
    }
    const elev_min: f32 = getJsonFloat(root, "elev_min") orelse {
        std.log.err("{s}/meta.json: missing 'elev_min' field", .{tile_name});
        return error.WeightsLoadFailed;
    };
    const elev_max: f32 = getJsonFloat(root, "elev_max") orelse {
        std.log.err("{s}/meta.json: missing 'elev_max' field", .{tile_name});
        return error.WeightsLoadFailed;
    };
    const grad_scale: f32 = getJsonFloat(root, "grad_scale") orelse 0.0;
    if (grad_scale == 0.0) {
        std.log.warn("{s}: no grad_scale in meta.json, normals will be flat", .{tile_name});
    }

    const mlp_floats = mlp_hidden_dim * features + mlp_hidden_dim + mlp_hidden_dim * num_outputs + num_outputs;
    const weights_bytes: u32 = (grid_uints + 2 * features + mlp_floats) * 4;

    const origin = coords.tileToWorldRuntime(tile_name) catch {
        std.log.err("Cannot parse tile name '{s}' for origin computation", .{tile_name});
        return error.WeightsLoadFailed;
    };

    const tile_center_z = origin.z + coords.TILE_ARCSEC * 0.5;

    return .{
        .meta = .{
            .name = undefined, // filled in by caller
            .origin_x = origin.x,
            .origin_z = origin.z,
            .resolution_h = resolution_h,
            .resolution_w = resolution_w,
            .grid_uints = grid_uints,
            .elev_min = elev_min,
            .elev_max = elev_max,
            .weights_bytes = weights_bytes,
            .grad_scale = grad_scale,
            .cos_lat = coords.cosLatFromZ(tile_center_z),
        },
        .features = features,
        .mlp_hidden_dim = mlp_hidden_dim,
        .num_outputs = num_outputs,
    };
}

// ---- Tests ----

const testing = std.testing;

fn fakeDesc(name: *const [TILE_NAME_LEN]u8, weights_bytes: u32) TileDescriptor {
    return .{
        .name = name.*,
        .origin_x = 0,
        .origin_z = 0,
        .resolution_h = 256,
        .resolution_w = 256,
        .grid_uints = 0,
        .elev_min = 0,
        .elev_max = 0,
        .weights_bytes = weights_bytes,
    };
}

fn fakeCatalog(allocator: std.mem.Allocator, descs: []TileDescriptor) TileCatalog {
    return .{
        .allocator = allocator,
        .descriptors = descs,
        .dir_path = &.{},
        .single_tile = false,
        .shared_features = 4,
        .shared_mlp_hidden_dim = 4,
        .shared_num_outputs = 1,
        .catalog_grid = @splat(0),
    };
}

test "TileCatalog.slotStride returns max weights_bytes across descriptors" {
    var descs = [_]TileDescriptor{
        fakeDesc("n47w122", 789504),
        fakeDesc("n80w122", 157184),
        fakeDesc("n45w120", 600000),
    };
    const catalog = fakeCatalog(testing.allocator, &descs);
    try testing.expectEqual(@as(u32, 789504), catalog.slotStride());
}

test "TileSet: claimSlot pulls distinct slots; releaseSlot returns them" {
    var descs = [_]TileDescriptor{fakeDesc("n47w122", 256)};
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    var seen = std.bit_set.IntegerBitSet(MAX_TILES).initEmpty();
    for (0..MAX_TILES) |_| {
        const s = ts.claimSlot() orelse return error.PoolExhaustedEarly;
        try testing.expect(!seen.isSet(s));
        seen.set(s);
    }
    try testing.expectEqual(@as(?u32, null), ts.claimSlot());

    ts.releaseSlot(42);
    try testing.expectEqual(@as(?u32, 42), ts.claimSlot());
}

test "TileSet: appendDir wires resident_indices and slot_indices in lockstep" {
    var descs = [_]TileDescriptor{ fakeDesc("n47w122", 256), fakeDesc("n48w122", 256) };
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    const slot_a = ts.claimSlot().?;
    const slot_b = ts.claimSlot().?;
    const dir_a = ts.appendDir(0, slot_a);
    const dir_b = ts.appendDir(1, slot_b);
    try testing.expectEqual(@as(u32, 0), dir_a);
    try testing.expectEqual(@as(u32, 1), dir_b);
    try testing.expectEqual(@as(u32, 2), ts.tile_count);
    try testing.expectEqual(slot_a, ts.slotForDir(dir_a));
    try testing.expectEqual(slot_b, ts.slotForDir(dir_b));
    try testing.expectEqual(@as(u32, 0), ts.resident_indices[dir_a]);
    try testing.expectEqual(@as(u32, 1), ts.resident_indices[dir_b]);
    try testing.expect(ts.isResident(0));
    try testing.expect(ts.isResident(1));
}

test "TileSet.evict: middle entry swap-remove" {
    var descs = [_]TileDescriptor{ fakeDesc("n47w122", 256), fakeDesc("n48w122", 256), fakeDesc("n49w122", 256) };
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    const slot_a = ts.claimSlot().?;
    const slot_b = ts.claimSlot().?;
    const slot_c = ts.claimSlot().?;
    _ = ts.appendDir(0, slot_a);
    _ = ts.appendDir(1, slot_b);
    _ = ts.appendDir(2, slot_c);

    const result = ts.evict(1);
    try testing.expectEqual(slot_b, result.freed_slot);
    try testing.expectEqual(@as(?u32, 1), result.swapped_dir_idx);
    try testing.expectEqual(@as(u32, 2), ts.tile_count);
    try testing.expectEqual(@as(u32, 0), ts.resident_indices[0]);
    try testing.expectEqual(@as(u32, 2), ts.resident_indices[1]);
    try testing.expectEqual(slot_a, ts.slot_indices[0]);
    try testing.expectEqual(slot_c, ts.slot_indices[1]);
    try testing.expect(!ts.free_slots.isSet(slot_b));
    try testing.expect(!ts.isResident(1)); // evicted catalog
    try testing.expect(ts.isResident(2)); // swapped, still resident
}

test "TileSet.evict: last entry no swap" {
    var descs = [_]TileDescriptor{ fakeDesc("n47w122", 256), fakeDesc("n48w122", 256) };
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    const slot_a = ts.claimSlot().?;
    const slot_b = ts.claimSlot().?;
    _ = ts.appendDir(0, slot_a);
    _ = ts.appendDir(1, slot_b);

    const result = ts.evict(1);
    try testing.expectEqual(slot_b, result.freed_slot);
    try testing.expectEqual(@as(?u32, null), result.swapped_dir_idx);
    try testing.expectEqual(@as(u32, 1), ts.tile_count);
    try testing.expectEqual(@as(u32, 0), ts.resident_indices[0]);
    try testing.expectEqual(slot_a, ts.slot_indices[0]);
}

test "TileSet.evict: only entry leaves empty set" {
    var descs = [_]TileDescriptor{fakeDesc("n47w122", 256)};
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    const slot_a = ts.claimSlot().?;
    _ = ts.appendDir(0, slot_a);

    const result = ts.evict(0);
    try testing.expectEqual(slot_a, result.freed_slot);
    try testing.expectEqual(@as(?u32, null), result.swapped_dir_idx);
    try testing.expectEqual(@as(u32, 0), ts.tile_count);
}

test "TileSet.evict + releaseSlot + reclaim: same slot returns" {
    var descs: [MAX_TILES]TileDescriptor = undefined;
    for (&descs) |*d| d.* = fakeDesc("n47w122", 256);
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    var i: u32 = 0;
    while (i < MAX_TILES) : (i += 1) {
        const s = ts.claimSlot().?;
        _ = ts.appendDir(i, s);
    }
    try testing.expectEqual(@as(?u32, null), ts.claimSlot());

    const result = ts.evict(7);
    try testing.expectEqual(@as(?u32, null), ts.claimSlot());
    ts.releaseSlot(result.freed_slot);
    try testing.expectEqual(@as(?u32, result.freed_slot), ts.claimSlot());
}

test "TileCatalog.slotStride single descriptor returns its size" {
    var descs = [_]TileDescriptor{fakeDesc("n47w122", 789504)};
    const catalog = fakeCatalog(testing.allocator, &descs);
    try testing.expectEqual(@as(u32, 789504), catalog.slotStride());
}

test "TileSet: invariants hold across randomized claim/append/evict cycles" {
    // Fuzz alternating loads and evicts, asserting the cross-field invariants
    // after every op. Designed to catch the class of bug where a refactor
    // drops one of: the resident_mask flip, the free_slots toggle, the
    // tile_count bump, or the swap-remove writes.
    const CATALOG_N: u32 = 64;
    var descs: [CATALOG_N]TileDescriptor = undefined;
    for (&descs) |*d| d.* = fakeDesc("n47w122", 256);
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, MAX_TILES);
    defer ts.deinit(testing.allocator);

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rng = prng.random();

    var resident_set = std.bit_set.IntegerBitSet(CATALOG_N).initEmpty();

    for (0..2000) |_| {
        const op_load = if (ts.tile_count == 0)
            true
        else if (ts.tile_count == CATALOG_N)
            false
        else
            rng.boolean();

        if (op_load) {
            var cat: u32 = rng.uintLessThan(u32, CATALOG_N);
            while (resident_set.isSet(cat)) cat = (cat + 1) % CATALOG_N;
            const slot = ts.claimSlot().?;
            _ = ts.appendDir(cat, slot);
            resident_set.set(cat);
        } else {
            const dir = rng.uintLessThan(u32, ts.tile_count);
            const evicted_cat = ts.resident_indices[dir];
            const result = ts.evict(dir);
            ts.releaseSlot(result.freed_slot);
            resident_set.unset(evicted_cat);
        }

        // (1) free_slots + tile_count == pool_size
        try testing.expectEqual(ts.pool_size, @as(u32, @intCast(ts.free_slots.count())) + ts.tile_count);
        // (1b) maintained free_count stays in sync with the bitset
        try testing.expectEqual(@as(u32, @intCast(ts.free_slots.count())), ts.free_count);
        // (2) resident_mask population matches tile_count
        try testing.expectEqual(@as(usize, ts.tile_count), ts.resident_mask.count());
        // (3) no duplicate catalog idx in resident_indices[0..tile_count]
        var seen = std.bit_set.IntegerBitSet(CATALOG_N).initEmpty();
        for (ts.resident_indices[0..ts.tile_count]) |c| {
            try testing.expect(!seen.isSet(c));
            seen.set(c);
        }
        // (4) resident_mask agrees with the externally-tracked resident set
        var c: u32 = 0;
        while (c < CATALOG_N) : (c += 1) {
            try testing.expectEqual(resident_set.isSet(c), ts.resident_mask.isSet(c));
        }
    }
}

test "tilesForRenderDistance: default config yields small count" {
    const n = tilesForRenderDistance(255, 7, 1.0);
    try testing.expect(n >= 25 and n <= 50);
}

test "tilesForRenderDistance: large ring yields large count" {
    const n = tilesForRenderDistance(2001, 9, 1.0);
    try testing.expect(n > 200);
}

test "tilesForRenderDistance: tiny ring clamps to MIN_TILE_POOL" {
    const n = tilesForRenderDistance(63, 1, 1.0);
    try testing.expectEqual(MIN_TILE_POOL, n);
}

test "TileSet: small pool exhausts at pool_size" {
    var descs: [8]TileDescriptor = undefined;
    for (&descs) |*d| d.* = fakeDesc("n47w122", 256);
    const catalog = fakeCatalog(testing.allocator, &descs);
    var ts = try TileSet.init(testing.allocator, &catalog, 4);
    defer ts.deinit(testing.allocator);

    for (0..4) |_| try testing.expect(ts.claimSlot() != null);
    try testing.expectEqual(@as(?u32, null), ts.claimSlot());
}
