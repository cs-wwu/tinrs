//! Coordinate system utilities for arcsecond-based world space.
//!
//! Convention (matching srg-synvis):
//!   - 1 VK unit = 1 arcsecond in XZ plane
//!   - Y axis = elevation in arcseconds (meters * HEIGHT_SCALE)
//!   - X = longitude * 3600, Z = latitude * 3600 (with sign conventions)
//!   - Tile = 3600 x 3600 arcseconds (1 deg x 1 deg)

const std = @import("std");

/// Meters to arcseconds conversion factor.
/// 1 nautical mile = 1852m = 1 arcminute = 60 arcseconds.
pub const HEIGHT_SCALE: f32 = 60.0 / 1852.0;

/// Arcseconds per degree (tile side length).
pub const TILE_ARCSEC: f32 = 3600.0;

/// Spatial grid for O(1) `findTile`: one cell per 1deg x 1deg world tile, globe-wide.
pub const GRID_LON_CELLS: u32 = 360;
pub const GRID_LAT_CELLS: u32 = 180;
pub const GRID_CELL_COUNT: u32 = GRID_LON_CELLS * GRID_LAT_CELLS;

/// Earth circumference in arcseconds (modulus for antimeridian wrap).
pub const WORLD_X_ARCSEC: f32 = @as(f32, @floatFromInt(GRID_LON_CELLS)) * TILE_ARCSEC;

/// Shift `x` to its nearest equivalent in `[-period/2, +period/2)`. Used to
/// pick the shortest signed delta across the antimeridian (e.g. tile 1 deg
/// east when the camera is at lon +179.5).
pub fn wrapToNearestF64(x: f64, period: f64) f64 {
    return x - period * @floor(x / period + 0.5);
}

/// Map a tile origin (NW corner in arcsec, as stored on `TileDescriptor`) to
/// a flat dir-grid cell index. Mirror in the compute shader's `findTile`.
pub fn worldOriginToGridCell(origin_x: f32, origin_z: f32) u32 {
    const cell_x: i32 = @intFromFloat(@floor(origin_x / TILE_ARCSEC));
    const cell_z: i32 = @intFromFloat(@floor(origin_z / TILE_ARCSEC));
    const lon_idx: i32 = cell_x + 180;
    const lat_idx: i32 = 89 - cell_z;
    std.debug.assert(lon_idx >= 0 and lon_idx < @as(i32, @intCast(GRID_LON_CELLS)));
    std.debug.assert(lat_idx >= 0 and lat_idx < @as(i32, @intCast(GRID_LAT_CELLS)));
    return @as(u32, @intCast(lat_idx)) * GRID_LON_CELLS + @as(u32, @intCast(lon_idx));
}

pub fn metersToArcsec(m: f32) f32 {
    return m * HEIGHT_SCALE;
}

pub fn arcsecToMeters(a: f32) f32 {
    return a / HEIGHT_SCALE;
}

/// Mean Earth radius in arcseconds: 6,371,000m * (60/1852).
pub const EARTH_RADIUS_ARCSEC: f32 = 206404.0;

/// Mount Everest (8849m) in arcseconds: conservative ceiling for any terrain
/// peak that could stick up above the geometric horizon.
pub const MAX_TERRAIN_HEIGHT_ARCSEC: f32 = 8849.0 * HEIGHT_SCALE;

/// Horizon distance (arcsec) of the tallest possible peak (Everest).
const D_PEAK_ARCSEC: f32 = @sqrt(2.0 * EARTH_RADIUS_ARCSEC * MAX_TERRAIN_HEIGHT_ARCSEC);

/// Maximum distance (arcsec) at which terrain can be visible from altitude
/// `cam_alt_arcsec`, accounting for Earth curvature and the tallest possible
/// peak poking over the horizon. 10% margin prevents hard pop-in at the edge.
pub fn horizonDistArcsec(cam_alt_arcsec: f32) f32 {
    const h_cam = @max(cam_alt_arcsec, 0.0);
    const d_cam = @sqrt(2.0 * EARTH_RADIUS_ARCSEC * h_cam);
    return (d_cam + D_PEAK_ARCSEC) * 1.1;
}

const COS_LAT_FLOOR: f32 = 0.01;

/// `cos(latitude)`, floored to avoid div-by-zero near the poles. `z` is the
/// world Z coordinate in arcseconds (Z = -(lat+1)*3600 by `tileToWorld`'s
/// convention, so latitude is recovered as `-z/3600`).
pub fn cosLatFromZ(z_arcsec: f32) f32 {
    const lat_rad = -z_arcsec / TILE_ARCSEC * (std.math.pi / 180.0);
    return @max(@cos(lat_rad), COS_LAT_FLOOR);
}

/// `1 / cos(latitude)` in f64. Used by input/motion paths that need to scale
/// X (longitude) displacement to keep visual movement matched to look direction.
pub fn invCosLatFromZD(z_arcsec: f64) f64 {
    const lat_rad: f64 = -z_arcsec / @as(f64, TILE_ARCSEC) * (std.math.pi / 180.0);
    return 1.0 / @max(@cos(lat_rad), @as(f64, COS_LAT_FLOOR));
}

/// Sun direction at local solar noon, equinox approximation. Solar elevation
/// equals (90 deg - |lat|), tilted toward the equator: south in N hemisphere,
/// north in S hemisphere. +Z is south in our world, so latitude maps to the
/// Z component directly. At the equator the sun is overhead, so slope shading
/// degenerates there (only steep faces show contrast).
pub fn sunDirAtNoon(z_arcsec: f32) [3]f32 {
    const lat_rad = -z_arcsec / TILE_ARCSEC * (std.math.pi / 180.0);
    return .{ 0.0, @cos(lat_rad), @sin(lat_rad) };
}

pub const WorldPos = struct { x: f32, z: f32 };

/// Convert tile name (e.g., "n47w122") to global arcsecond origin.
/// Returns the top-left (northwest) corner of the tile.
/// Convention: X = longitude in arcseconds, Z = -(latitude+1) in arcseconds.
/// (Z is negated so north is +Z in Vulkan's coordinate system.)
pub fn tileToWorld(comptime name: []const u8) WorldPos {
    comptime {
        if (name.len != 7) @compileError("tile name must be 7 chars (e.g. n47w122)");

        const lat_sign: f32 = if (name[0] == 'n') 1.0 else -1.0;
        const lat = lat_sign * @as(f32, @floatFromInt(parseDigits(name[1..3])));

        const lon_sign: f32 = if (name[3] == 'e') 1.0 else -1.0;
        const lon = lon_sign * @as(f32, @floatFromInt(parseDigits(name[4..7])));

        return .{
            .x = lon * TILE_ARCSEC,
            .z = -(lat + 1.0) * TILE_ARCSEC,
        };
    }
}

fn parseDigits(comptime s: []const u8) u32 {
    comptime {
        var result: u32 = 0;
        for (s) |c| {
            if (c < '0' or c > '9') @compileError("invalid digit in tile name");
            result = result * 10 + (c - '0');
        }
        return result;
    }
}

/// Runtime version of tileToWorld; same math, parses at runtime.
pub fn tileToWorldRuntime(name: []const u8) !WorldPos {
    if (name.len != 7) return error.InvalidTileName;

    const lat_sign: f32 = switch (name[0]) {
        'n', 'N' => 1.0,
        's', 'S' => -1.0,
        else => return error.InvalidTileName,
    };
    const lat_digits = parseDigitsRuntime(name[1..3]) orelse return error.InvalidTileName;
    const lat = lat_sign * @as(f32, @floatFromInt(lat_digits));

    const lon_sign: f32 = switch (name[3]) {
        'e', 'E' => 1.0,
        'w', 'W' => -1.0,
        else => return error.InvalidTileName,
    };
    const lon_digits = parseDigitsRuntime(name[4..7]) orelse return error.InvalidTileName;
    const lon = lon_sign * @as(f32, @floatFromInt(lon_digits));

    return .{
        .x = lon * TILE_ARCSEC,
        .z = -(lat + 1.0) * TILE_ARCSEC,
    };
}

fn parseDigitsRuntime(s: []const u8) ?u32 {
    var result: u32 = 0;
    for (s) |c| {
        if (c < '0' or c > '9') return null;
        result = result * 10 + (c - '0');
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "coords: horizonDistArcsec at sea level is Everest-only term" {
    const d = horizonDistArcsec(0);
    // sqrt(2 * 206404 * 286.7) * 1.1 ~= 11959
    const d_peak = @sqrt(2.0 * EARTH_RADIUS_ARCSEC * MAX_TERRAIN_HEIGHT_ARCSEC) * 1.1;
    try testing.expectApproxEqAbs(d_peak, d, 1.0);
}

test "coords: horizonDistArcsec at 10km altitude" {
    const h_cam = metersToArcsec(10000);
    const d = horizonDistArcsec(h_cam);
    // d_cam = sqrt(2 * 206404 * 324) ~= 11570
    // d_peak ~= 10872
    // total * 1.1 ~= 24686
    try testing.expect(d > 20000 and d < 30000);
}

test "coords: horizonDistArcsec increases with altitude" {
    const d_low = horizonDistArcsec(metersToArcsec(100));
    const d_high = horizonDistArcsec(metersToArcsec(10000));
    try testing.expect(d_high > d_low);
}

test "coords: HEIGHT_SCALE matches srg-synvis" {
    // 60 arcseconds / 1852 meters = 0.03239...
    try testing.expectApproxEqAbs(@as(f32, 0.03239), HEIGHT_SCALE, 0.0001);
}

test "coords: metersToArcsec roundtrip" {
    const meters: f32 = 1000.0;
    const arcsec = metersToArcsec(meters);
    const back = arcsecToMeters(arcsec);
    try testing.expectApproxEqAbs(meters, back, 0.01);
}

test "coords: tileToWorld n47w122" {
    const pos = comptime tileToWorld("n47w122");
    // lon = -122 * 3600 = -439200
    try testing.expectApproxEqAbs(@as(f32, -439200.0), pos.x, 0.1);
    // lat = -(47+1) * 3600 = -172800
    try testing.expectApproxEqAbs(@as(f32, -172800.0), pos.z, 0.1);
}

test "coords: tileToWorld n00e000" {
    const pos = comptime tileToWorld("n00e000");
    try testing.expectApproxEqAbs(@as(f32, 0.0), pos.x, 0.1);
    try testing.expectApproxEqAbs(@as(f32, -3600.0), pos.z, 0.1);
}

test "coords: tileToWorld s01e001" {
    const pos = comptime tileToWorld("s01e001");
    try testing.expectApproxEqAbs(@as(f32, 3600.0), pos.x, 0.1);
    // lat = -(-1+1) * 3600 = 0
    try testing.expectApproxEqAbs(@as(f32, 0.0), pos.z, 0.1);
}

test "coords: tileToWorldRuntime matches comptime" {
    const rt = try tileToWorldRuntime("n47w122");
    const ct = comptime tileToWorld("n47w122");
    try testing.expectApproxEqAbs(ct.x, rt.x, 0.1);
    try testing.expectApproxEqAbs(ct.z, rt.z, 0.1);
}

test "coords: tileToWorldRuntime case insensitive" {
    const pos = try tileToWorldRuntime("N47W122");
    try testing.expectApproxEqAbs(@as(f32, -439200.0), pos.x, 0.1);
    try testing.expectApproxEqAbs(@as(f32, -172800.0), pos.z, 0.1);
}

test "coords: tileToWorldRuntime rejects invalid" {
    try testing.expectError(error.InvalidTileName, tileToWorldRuntime("bad"));
    try testing.expectError(error.InvalidTileName, tileToWorldRuntime("x47w122"));
    try testing.expectError(error.InvalidTileName, tileToWorldRuntime("n47x122"));
}

test "coords: worldOriginToGridCell n47w122" {
    const pos = comptime tileToWorld("n47w122");
    const cell = worldOriginToGridCell(pos.x, pos.z);
    // lon=-122 -> lon_idx=58, lat=47 -> lat_idx=137 -> 137*360 + 58 = 49378
    try testing.expectEqual(@as(u32, 137 * 360 + 58), cell);
}

test "coords: worldOriginToGridCell corners" {
    // s89 covers lat -89 -> -88: lat_idx = 89 - 88 = 1. lon -180 -> lon_idx = 0.
    const sw = comptime tileToWorld("s89w180");
    try testing.expectEqual(@as(u32, 1 * 360 + 0), worldOriginToGridCell(sw.x, sw.z));
    // n89 covers lat 89 -> 90: lat_idx = 89 - (-90) = 179. lon 179 -> lon_idx = 359.
    const ne = comptime tileToWorld("n89e179");
    try testing.expectEqual(@as(u32, 179 * 360 + 359), worldOriginToGridCell(ne.x, ne.z));
}

test "coords: wrapToNearestF64 picks shortest signed delta" {
    const W: f64 = @as(f64, WORLD_X_ARCSEC);
    try testing.expectApproxEqAbs(@as(f64, 0), wrapToNearestF64(0, W), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 3600), wrapToNearestF64(-W + 3600, W), 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -3600), wrapToNearestF64(W - 3600, W), 1e-6);
    // Multi-lap input still wraps to within [-W/2, +W/2).
    try testing.expectApproxEqAbs(@as(f64, 3600), wrapToNearestF64(5 * W + 3600, W), 1e-6);
}
