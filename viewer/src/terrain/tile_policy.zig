//! Tile residency policy. Pure scoring + ranking; no allocator, no Vulkan.
//!
//! `rankDesired` scores every catalog tile by cosine-scaled distance^2 from a
//! velocity-projected anchor (`camera_pos + velocity * lookahead_seconds`)
//! and returns the top-N. Resident tiles get a small score discount
//! (`hysteresis`) so they don't churn at the resident-set boundary.
//!
//! Mach 500 design constraint: at ~5500 arcsec/s and the default 2 s lookahead,
//! the anchor lands ~3 tiles ahead of the camera, prefetching the path of travel.

const std = @import("std");
const tile_loader = @import("tile_loader.zig");
const coords = @import("coords.zig");

/// Camera state passed into `rankDesired` and `TileSystem.tickPolicy`. f64
/// position (arcsec, world coords) + f32 velocity/front (small scale, f32 fine).
pub const CameraView = struct {
    pos_xz: [2]f64,
    velocity_xz: [2]f32,
    front: [3]f32,
};

pub const Config = struct {
    lookahead_seconds: f32 = 2.0,
    /// Below this speed (arcsec/s) the anchor falls back to `front_xz *
    /// stationary_lookahead`. ~1 arcsec/s ~= 30 m/s; above keyboard jitter,
    /// below interactive flight.
    velocity_threshold: f32 = 1.0,
    /// 0 = disabled. Non-zero = offset along normalized `front_xz` when slow
    /// so prefetch follows the look direction at rest.
    stationary_lookahead: f32 = 0.0,
    /// Score multiplier for already-resident tiles. Score is distance^2, so
    /// the effective distance discount is `sqrt(hysteresis)`: default 0.90
    /// means residents win against tiles up to ~5% closer.
    hysteresis: f32 = 0.90,
    /// Cap on anchor displacement from camera (arcsec). 0 = no cap. Without
    /// this, high-speed `velocity * lookahead_seconds` can push the anchor
    /// past the visible bbox so backward tiles fall out of the top-N. Set to
    /// ~half the visible bbox radius for comfortable margin with prefetch.
    max_anchor_offset_arcsec: f32 = 0.0,
};

pub const Item = struct {
    catalog_idx: u32,
    score: f32,
};

/// `scratch` must be at least `catalog.count()`; caller allocates once.
/// Returns the prefix of `out` written (= min(out.len, catalog.count())).
pub fn rankDesired(
    catalog: *const tile_loader.TileCatalog,
    tile_set: *const tile_loader.TileSet,
    view: CameraView,
    config: Config,
    scratch: []Item,
    out: []u32,
) []u32 {
    const total = catalog.count();
    std.debug.assert(scratch.len >= total);
    const n = @min(out.len, total);
    if (n == 0) return out[0..0];

    const anchor = computeAnchor(view, config);
    const half_tile: f32 = coords.TILE_ARCSEC * 0.5;
    const world_x: f64 = @as(f64, coords.WORLD_X_ARCSEC);

    for (0..total) |i| {
        const desc = &catalog.descriptors[i];
        const cx = desc.origin_x + half_tile;
        const cz = desc.origin_z + half_tile;
        // Wrap dx so a tile 1 deg east across the antimeridian scores as 1 deg,
        // not 359 deg.
        const wrapped_dx = coords.wrapToNearestF64(@as(f64, cx) - anchor[0], world_x);
        const dx: f32 = @as(f32, @floatCast(wrapped_dx)) * desc.cos_lat;
        const dz = cz - @as(f32, @floatCast(anchor[1]));
        var s = dx * dx + dz * dz;
        if (tile_set.isResident(@intCast(i))) s *= config.hysteresis;
        scratch[i] = .{ .catalog_idx = @intCast(i), .score = s };
    }

    std.mem.sort(Item, scratch[0..total], {}, struct {
        fn lt(_: void, a: Item, b: Item) bool {
            return a.score < b.score;
        }
    }.lt);

    for (0..n) |i| out[i] = scratch[i].catalog_idx;
    return out[0..n];
}

pub fn computeAnchor(view: CameraView, cfg: Config) [2]f64 {
    const speed_sq = view.velocity_xz[0] * view.velocity_xz[0] + view.velocity_xz[1] * view.velocity_xz[1];
    var dx: f32 = 0;
    var dz: f32 = 0;
    if (speed_sq >= cfg.velocity_threshold * cfg.velocity_threshold) {
        dx = view.velocity_xz[0] * cfg.lookahead_seconds;
        dz = view.velocity_xz[1] * cfg.lookahead_seconds;
    } else if (cfg.stationary_lookahead > 0) {
        const fx = view.front[0];
        const fz = view.front[2];
        const flat_sq = fx * fx + fz * fz;
        if (flat_sq > 1e-6) {
            const inv = 1.0 / @sqrt(flat_sq);
            dx = fx * inv * cfg.stationary_lookahead;
            dz = fz * inv * cfg.stationary_lookahead;
        }
    }
    if (cfg.max_anchor_offset_arcsec > 0) {
        const mag_sq = dx * dx + dz * dz;
        const cap = cfg.max_anchor_offset_arcsec;
        if (mag_sq > cap * cap) {
            const scale = cap / @sqrt(mag_sq);
            dx *= scale;
            dz *= scale;
        }
    }
    return .{ view.pos_xz[0] + @as(f64, dx), view.pos_xz[1] + @as(f64, dz) };
}

// ---- Tests ----

const testing = std.testing;

fn fakeDescAt(name: *const [7]u8, origin_x: f32, origin_z: f32) tile_loader.TileDescriptor {
    return .{
        .name = name.*,
        .origin_x = origin_x,
        .origin_z = origin_z,
        .resolution_h = 256,
        .resolution_w = 256,
        .grid_uints = 0,
        .elev_min = 0,
        .elev_max = 0,
        .weights_bytes = 256,
        .cos_lat = coords.cosLatFromZ(origin_z + coords.TILE_ARCSEC * 0.5),
    };
}

fn testCatalog(descs: []tile_loader.TileDescriptor) tile_loader.TileCatalog {
    return .{
        .allocator = testing.allocator,
        .descriptors = descs,
        .dir_path = &.{},
        .single_tile = false,
        .shared_features = 4,
        .shared_mlp_hidden_dim = 4,
        .shared_num_outputs = 1,
        .catalog_grid = @splat(0),
    };
}

test "rankDesired: nearest tile to camera ranks first" {
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("a000000", 10000, 10000),
        fakeDescAt("b000000", 0, 0),
        fakeDescAt("c000000", 5000, 5000),
    };
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    var scratch: [3]Item = undefined;
    var out: [3]u32 = undefined;

    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
        .{},
        &scratch,
        &out,
    );

    try testing.expectEqual(@as(usize, 3), result.len);
    try testing.expectEqual(@as(u32, 1), result[0]);
    try testing.expectEqual(@as(u32, 2), result[1]);
    try testing.expectEqual(@as(u32, 0), result[2]);
}

test "rankDesired: velocity bias shifts ranking toward direction of travel" {
    // Origins -8600/5000 with z origin -1800 -> centers at (-6800, 0) / (+6800, 0):
    // half_tile (1800) shifts both axes, so we offset z origin too to keep z=0.
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("east000", 5000, -1800),
        fakeDescAt("west000", -8600, -1800),
    };
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    var scratch: [2]Item = undefined;
    var out: [2]u32 = undefined;

    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 1000, 0 }, .front = .{ 0, 0, -1 } },
        .{},
        &scratch,
        &out,
    );
    try testing.expectEqual(@as(u32, 0), result[0]);
    try testing.expectEqual(@as(u32, 1), result[1]);
}

test "rankDesired: hysteresis keeps borderline resident tile" {
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("a000000", 5000, -1800),
        fakeDescAt("b000000", -8600, -1800),
    };
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    const slot = tile_set.claimSlot().?;
    _ = tile_set.appendDir(0, slot);

    var scratch: [2]Item = undefined;
    var out: [2]u32 = undefined;

    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
        .{ .hysteresis = 0.5 },
        &scratch,
        &out,
    );
    try testing.expectEqual(@as(u32, 0), result[0]);
    try testing.expectEqual(@as(u32, 1), result[1]);
}

test "computeAnchor: stationary uses front when velocity below threshold" {
    const cfg: Config = .{ .stationary_lookahead = 1000, .velocity_threshold = 1.0 };
    const a = computeAnchor(
        .{ .pos_xz = .{ 100, 200 }, .velocity_xz = .{ 0.1, 0 }, .front = .{ 1, 0, 0 } },
        cfg,
    );
    try testing.expectApproxEqAbs(@as(f64, 1100), a[0], 0.01);
    try testing.expectApproxEqAbs(@as(f64, 200), a[1], 0.01);
}

test "computeAnchor: velocity dominates when above threshold" {
    const cfg: Config = .{ .lookahead_seconds = 2.0, .velocity_threshold = 1.0, .stationary_lookahead = 999 };
    const a = computeAnchor(
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 100, 0 }, .front = .{ 0, 0, 1 } },
        cfg,
    );
    try testing.expectApproxEqAbs(@as(f64, 200), a[0], 0.01);
    try testing.expectApproxEqAbs(@as(f64, 0), a[1], 0.01);
}

test "computeAnchor: max_anchor_offset clamps fast-velocity displacement" {
    // velocity 10000 * 2s = 20000 arcsec raw; cap at 5000 should clamp.
    const cfg: Config = .{ .lookahead_seconds = 2.0, .max_anchor_offset_arcsec = 5000 };
    const a = computeAnchor(
        .{ .pos_xz = .{ 1000, -1000 }, .velocity_xz = .{ 10000, 0 }, .front = .{ 0, 0, -1 } },
        cfg,
    );
    // anchor = pos + clamped_velocity_dir * cap = (1000 + 5000, -1000) = (6000, -1000).
    try testing.expectApproxEqAbs(@as(f64, 6000), a[0], 0.01);
    try testing.expectApproxEqAbs(@as(f64, -1000), a[1], 0.01);
}

test "computeAnchor: max_anchor_offset is a no-op when raw displacement is below cap" {
    const cfg: Config = .{ .lookahead_seconds = 2.0, .max_anchor_offset_arcsec = 5000 };
    const a = computeAnchor(
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 100, 0 }, .front = .{ 0, 0, -1 } },
        cfg,
    );
    // raw displacement = 200, well below 5000 cap.
    try testing.expectApproxEqAbs(@as(f64, 200), a[0], 0.01);
    try testing.expectApproxEqAbs(@as(f64, 0), a[1], 0.01);
}

test "computeAnchor: max_anchor_offset preserves direction under diagonal velocity" {
    const cfg: Config = .{ .lookahead_seconds = 1.0, .max_anchor_offset_arcsec = 10 };
    const a = computeAnchor(
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 100, 100 }, .front = .{ 0, 0, -1 } },
        cfg,
    );
    // raw = (100, 100), magnitude = sqrt(20000) ~= 141.4; cap to 10.
    // direction (1/sqrt(2), 1/sqrt(2)) * 10 = (7.071, 7.071).
    try testing.expectApproxEqAbs(@as(f64, 7.071), a[0], 0.01);
    try testing.expectApproxEqAbs(@as(f64, 7.071), a[1], 0.01);
}

test "rankDesired: tiles use their own cos_lat, not the anchor's" {
    // Two tiles at identical positions but with deliberately different cos_lat.
    // Per-tile cos_lat means smaller-cos_lat tile has smaller scaled dx and ranks
    // closer; old anchor-shared cos_lat would tie them (and ordering would be
    // arbitrary across the unstable sort).
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("a000000", 5000, -1800),
        fakeDescAt("b000000", 5000, -1800),
    };
    descs[1].cos_lat = 0.1;
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    var scratch: [2]Item = undefined;
    var out: [2]u32 = undefined;
    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
        .{},
        &scratch,
        &out,
    );
    try testing.expectEqual(@as(u32, 1), result[0]);
    try testing.expectEqual(@as(u32, 0), result[1]);
}

test "rankDesired: antimeridian-adjacent tile ranks closer than 9 deg away" {
    // Camera at lon=+179.5. Tile A at lon=-180 is 1 deg east through the wrap;
    // tile B at lon=+170 is 9 deg west. Naïve dx would rank A as 359 deg away.
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("an18000", -180.0 * 3600.0, -3600.0),
        fakeDescAt("bp17000", 170.0 * 3600.0, -3600.0),
    };
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    var scratch: [2]Item = undefined;
    var out: [2]u32 = undefined;
    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 179.5 * 3600.0, -1800.0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
        .{},
        &scratch,
        &out,
    );
    try testing.expectEqual(@as(u32, 0), result[0]);
    try testing.expectEqual(@as(u32, 1), result[1]);
}

test "rankDesired: multi-lap camera position still ranks wrap-adjacent first" {
    // Camera at lon=+540 (1.5 laps east) — wrap must handle camera magnitudes
    // outside [-180, 180).
    var descs = [_]tile_loader.TileDescriptor{
        fakeDescAt("an18000", -180.0 * 3600.0, -3600.0),
        fakeDescAt("bp17000", 170.0 * 3600.0, -3600.0),
    };
    const catalog = testCatalog(&descs);
    var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
    defer tile_set.deinit(testing.allocator);

    var scratch: [2]Item = undefined;
    var out: [2]u32 = undefined;
    const result = rankDesired(
        &catalog,
        &tile_set,
        .{ .pos_xz = .{ 540.5 * 3600.0, -1800.0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
        .{},
        &scratch,
        &out,
    );
    try testing.expectEqual(@as(u32, 0), result[0]);
    try testing.expectEqual(@as(u32, 1), result[1]);
}

test "rankDesired: hysteresis band matches sqrt(hysteresis) discount" {
    // hysteresis=0.90 means effective distance discount sqrt(0.90) ~= 0.949.
    // Resident at center (1000, 0); competitor at center (950, 0) is 5% closer
    // so resident still wins (0.95^2 = 0.9025 > 0.90). At (940, 0), competitor
    // is 6% closer so it wins (0.94^2 = 0.8836 < 0.90). Pins both the band size
    // and the squared-score semantics; a future "linear discount" refactor
    // would silently double the band.
    const cases = [_]struct { competitor_origin_x: f32, expected_first: u32 }{
        .{ .competitor_origin_x = -850, .expected_first = 0 }, // 5% closer -> resident
        .{ .competitor_origin_x = -860, .expected_first = 1 }, // 6% closer -> competitor
    };
    for (cases) |case| {
        var descs = [_]tile_loader.TileDescriptor{
            fakeDescAt("rsdt000", -800, -1800),
            fakeDescAt("comp000", case.competitor_origin_x, -1800),
        };
        const catalog = testCatalog(&descs);
        var tile_set = try tile_loader.TileSet.init(testing.allocator, &catalog, tile_loader.MAX_TILES);
        defer tile_set.deinit(testing.allocator);

        const slot = tile_set.claimSlot().?;
        _ = tile_set.appendDir(0, slot);

        var scratch: [2]Item = undefined;
        var out: [2]u32 = undefined;
        const result = rankDesired(
            &catalog,
            &tile_set,
            .{ .pos_xz = .{ 0, 0 }, .velocity_xz = .{ 0, 0 }, .front = .{ 0, 0, -1 } },
            .{},
            &scratch,
            &out,
        );
        try testing.expectEqual(case.expected_first, result[0]);
    }
}
