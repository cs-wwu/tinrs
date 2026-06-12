//! Per-chunk cull predicates for the geometry clipmap. Pure-Zig so it
//! unit-tests without Vulkan deps.
//!
//! Two-stage radial cull splits work between CPU and GPU:
//!
//! 1. Slop-margin chunk cull (this module): drop chunks whose AABB is
//!    definitely outside the kept annulus; keep everything else. The bound
//!    is conservative; chunks whose center is within `r +/- chunk_half_diag`
//!    are kept. ~6 ops/chunk.
//!
//! 2. Per-vertex clip-distance (vertex shader): each kept chunk's vertices
//!    write `gl_ClipDistance[0/1]` so triangles spanning the boundary get
//!    rasterizer-clipped at the exact circle.
//!
//! Adjacent level pairs reference a shared boundary radius computed from
//! one g (the parent's), so L's outer-clip and L+1's inner-clip describe
//! the same arcsec circle, giving gap-free transitions.
//!
//! At the outermost level there's no L+1, so no boundary to tile against.
//! At the finest visible level (level == min_level), `r_inner_sq = 0`: no
//! parent, no hole.

const std = @import("std");
const math = @import("math");

/// Pick chunks-per-side targeting ~32 cells per chunk. Small rings shrink
/// the chunk count to amortize per-instance setup; medium and large rings
/// stay at the cap so cull granularity isn't lost (empirically, ring 255
/// regresses ~6% if target is raised to 64). Even values keep the rounded
/// ring_size odd (chunks centered on a vertex). Capped at
/// `max_chunks_per_side` so static buffer sizing stays bounded.
pub fn pickChunksPerSide(requested_ring_size: u32, max_chunks_per_side: u32) u32 {
    const target_chunk_cells: u32 = 32;
    const cells = if (requested_ring_size > 1) requested_ring_size - 1 else 1;
    const raw = (cells + target_chunk_cells - 1) / target_chunk_cells;
    const even = if (raw & 1 != 0) raw + 1 else raw;
    return @max(2, @min(max_chunks_per_side, even));
}

/// Level-relative AABB of one chunk in arcseconds.
pub const ChunkAabb = struct {
    x_min: f32,
    x_max: f32,
    z_min: f32,
    z_max: f32,
};

/// Level-relative AABB of chunk `(cx, cy)`. Mirrors the vertex shader's
/// `(col - ring_size/2.0) * grid_spacing` mapping so cull and draw agree.
pub fn chunkAabb(
    ring_size: u32,
    chunk_cells: u32,
    cx: u32,
    cy: u32,
    grid_spacing: f32,
) ChunkAabb {
    const half: f32 = @as(f32, @floatFromInt(ring_size)) / 2.0;
    const col_min: f32 = @floatFromInt(cx * chunk_cells);
    const col_max: f32 = @floatFromInt((cx + 1) * chunk_cells);
    const row_min: f32 = @floatFromInt(cy * chunk_cells);
    const row_max: f32 = @floatFromInt((cy + 1) * chunk_cells);
    return .{
        .x_min = (col_min - half) * grid_spacing,
        .x_max = (col_max - half) * grid_spacing,
        .z_min = (row_min - half) * grid_spacing,
        .z_max = (row_max - half) * grid_spacing,
    };
}

/// Squared "snap-safe" radius: largest circle in arcsec that fits inside a
/// level's grid for every snap origin position.
///
///     r_safe = (ring_size/2 - 3) * grid_spacing
///
/// The `-3` covers:
/// - -1 grid asymmetry: max camera-relative vertex position is
///   `(half - 1) * g` (vertex count is odd).
/// - -2 floor-snap offset: snap interval `2g` with `floor` puts the level
///   origin in `(-2g, 0]` from the camera (one-sided, not centered).
pub fn rSafeSq(ring_size: u32, grid_spacing: f32) f32 {
    const half: f32 = @as(f32, @floatFromInt(ring_size)) / 2.0;
    const r: f32 = (half - 3.0) * grid_spacing;
    return r * r;
}

/// Chunk's half-diagonal length in arcsec, used as slop margin for
/// `chunkVisibleRadial`.
pub fn chunkHalfDiag(chunk_cells: u32, grid_spacing: f32) f32 {
    const half: f32 = @as(f32, @floatFromInt(chunk_cells)) * grid_spacing * 0.5;
    return half * std.math.sqrt2;
}

/// Slop-margin radial cull. Chunk visible iff its center *might* fall in
/// the annulus `[r_inner, r_outer]` in arcsec. The squared radii passed in
/// are pre-expanded by the chunk's arcsec half-diagonal (outer outward,
/// inner inward) for conservative over-keep. The vertex shader's per-vertex
/// `gl_ClipDistance` handles the exact-circle boundary at draw time.
///
/// Pass `r_inner_keep_sq = 0` for level == min_level (no hole).
pub fn chunkVisibleRadial(
    chunk: ChunkAabb,
    level_origin: [2]f32, // arcsec
    cam_xz: [2]f32, // arcsec
    r_outer_keep_sq: f32,
    r_inner_keep_sq: f32,
) bool {
    const dx_origin: f32 = level_origin[0] - cam_xz[0];
    const dz_origin: f32 = level_origin[1] - cam_xz[1];

    const cx_arcsec: f32 = (chunk.x_min + chunk.x_max) * 0.5;
    const cz_arcsec: f32 = (chunk.z_min + chunk.z_max) * 0.5;

    const cx: f32 = dx_origin + cx_arcsec;
    const cz: f32 = dz_origin + cz_arcsec;
    const center_sq: f32 = cx * cx + cz * cz;

    if (center_sq > r_outer_keep_sq) return false;
    if (r_inner_keep_sq > 0 and center_sq < r_inner_keep_sq) return false;
    return true;
}

const coords = @import("coords.zig");
const TileCatalog = @import("tile_loader.zig").TileCatalog;

/// True if the chunk's world footprint lies entirely outside the tile catalog.
/// Longitude wraps (chunks may straddle the antimeridian); latitude does not
/// (poles are singular).
pub fn chunkIsOcean(
    chunk: ChunkAabb,
    snapped_pos: [2]f32,
    catalog: *const TileCatalog,
) bool {
    const lon_min = @as(i32, @intFromFloat(@floor((snapped_pos[0] + chunk.x_min) / coords.TILE_ARCSEC)));
    const lon_max = @as(i32, @intFromFloat(@floor((snapped_pos[0] + chunk.x_max) / coords.TILE_ARCSEC)));
    const lat_min = @as(i32, @intFromFloat(@floor((snapped_pos[1] + chunk.z_min) / coords.TILE_ARCSEC)));
    const lat_max = @as(i32, @intFromFloat(@floor((snapped_pos[1] + chunk.z_max) / coords.TILE_ARCSEC)));

    const lon_cells_i32: i32 = @intCast(coords.GRID_LON_CELLS);
    const lat_cells_i32: i32 = @intCast(coords.GRID_LAT_CELLS);
    var lat = lat_min;
    while (lat <= lat_max) : (lat += 1) {
        const lat_idx = 89 - lat;
        if (lat_idx < 0 or lat_idx >= lat_cells_i32) continue;
        var lon = lon_min;
        while (lon <= lon_max) : (lon += 1) {
            // `@mod` with the const 360 divisor lowers to a branchless
            // magic-number multiply, not idiv. A branch-style wrap measured
            // slightly slower (branch prediction overhead) in the dGPU/ring-63
            // microbench, so keep this form.
            const lon_idx: i32 = @mod(lon + 180, lon_cells_i32);
            const cell: u32 = @intCast(lat_idx * lon_cells_i32 + lon_idx);
            if (catalog.hasCatalogTile(cell)) return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
// 8 chunks/side, ring 257 = 32 cells/chunk; canonical default.
const RS: u32 = 257;
const CC: u32 = 32;

test "chunkAabb tiles the level square exactly" {
    // Spacing 1 means corner chunk (0,0) reaches `-ring/2 * spacing`; far chunk
    // (7,7) reaches `(ring/2 - 1) * spacing`. With ring 257 / 8 chunks of 32:
    //  half = 128.5
    //  cx=0 col range [0, 32]      -> x range [-128.5, -96.5]
    //  cx=7 col range [224, 256]   -> x range [95.5, 127.5]
    const a = chunkAabb(RS, CC, 0, 0, 1.0);
    try testing.expectApproxEqAbs(@as(f32, -128.5), a.x_min, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -96.5), a.x_max, 1e-4);
    const b = chunkAabb(RS, CC, 7, 7, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 95.5), b.x_min, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 127.5), b.x_max, 1e-4);
}

test "rSafeSq is (half - 3) * spacing squared" {
    // ring=257 means half = 128.5, half - 3 = 125.5, r^2 = 15750.25
    const r2 = rSafeSq(RS, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 125.5 * 125.5), r2, 1e-4);
}

test "rSafeSq scales with grid spacing" {
    // Doubling g doubles r, quadruples r^2.
    const r2_a = rSafeSq(RS, 1.0);
    const r2_b = rSafeSq(RS, 2.0);
    try testing.expectApproxEqAbs(r2_a * 4.0, r2_b, 1e-3);
}

test "rSafeSq fits inside worst-case floor-snap-shifted grid extent" {
    // ring=257 max +X vertex pos = 127.5 * g; floor-snap offset is in
    // (-2g, 0] (asymmetric, one-sided), so worst-case cam-rel max extent
    // = 127.5 - 2 = 125.5 * g. rSafeSq must equal exactly that (largest
    // circle that always fits).
    const half: f32 = @as(f32, @floatFromInt(RS)) / 2.0;
    const max_pos: f32 = (half - 1.0) * 1.0; // 127.5
    const snap_offset: f32 = 2.0; // = 2g (floor snap at interval 2g)
    const worst_extent: f32 = max_pos - snap_offset; // 125.5
    const r = @sqrt(rSafeSq(RS, 1.0));
    try testing.expectApproxEqAbs(worst_extent, r, 1e-4);
}

// Build slop-adjusted bounds the way CullState does. Pass `r_inner = 0`
// for no inner cull.
fn slopBounds(r_outer: f32, r_inner: f32, half_diag: f32) struct { r_outer_keep_sq: f32, r_inner_keep_sq: f32 } {
    const r_outer_keep = r_outer + half_diag;
    const r_inner_keep = if (r_inner > half_diag) r_inner - half_diag else 0.0;
    return .{
        .r_outer_keep_sq = r_outer_keep * r_outer_keep,
        .r_inner_keep_sq = r_inner_keep * r_inner_keep,
    };
}

test "chunkHalfDiag is sqrt(2) * half-side in arcsec" {
    const hd = chunkHalfDiag(CC, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 16.0 * std.math.sqrt2), hd, 1e-4);

    // Doubles with grid spacing.
    const hd2 = chunkHalfDiag(CC, 2.0);
    try testing.expectApproxEqAbs(@as(f32, 32.0 * std.math.sqrt2), hd2, 1e-4);
}

test "cull keeps chunks under the camera (no inner hole)" {
    const a = chunkAabb(RS, CC, 4, 4, 1.0);
    const hd = chunkHalfDiag(CC, 1.0);
    const b = slopBounds(@sqrt(rSafeSq(RS, 1.0)), 0, hd);
    try testing.expect(chunkVisibleRadial(a, .{ 0, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

test "outer corner cull still drops corners" {
    // Corner chunk (0, 0) center (-112.5, -112.5), dist^2 = 25312.5 in arcsec.
    // r_outer = 125.5 (= half - 3, ring=257). r_outer_keep ~= 125.5 + 16*sqrt(2)
    // ~= 148.13, sq ~= 21943. 25312 > 21943 means DROPPED.
    const a = chunkAabb(RS, CC, 0, 0, 1.0);
    const hd = chunkHalfDiag(CC, 1.0);
    const b = slopBounds(@sqrt(rSafeSq(RS, 1.0)), 0, hd);
    try testing.expect(!chunkVisibleRadial(a, .{ 0, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

test "min_level: r_inner_keep_sq=0 disables hole entirely" {
    const a = chunkAabb(RS, CC, 4, 4, 1.0);
    const hd = chunkHalfDiag(CC, 1.0);
    const b = slopBounds(@sqrt(rSafeSq(RS, 1.0)), 0, hd);
    try testing.expect(chunkVisibleRadial(a, .{ 0, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

test "shared boundary scalar matches between L and L+1 in arcsec" {
    // L's outer and (L+1)'s inner both compute `(half - 3) * g_L` from the
    // same g, so the boundary is one arcsec circle (latitude-independent).
    const g_l: f32 = 1.0;
    const r_outer_l: f32 = @sqrt(rSafeSq(RS, g_l));
    const r_inner_lp1: f32 = @sqrt(rSafeSq(RS, g_l));
    try testing.expectEqual(r_outer_l, r_inner_lp1);
}

test "far-outside chunks still dropped by slop cull" {
    const r_outer = @sqrt(rSafeSq(RS, 1.0));
    const hd = chunkHalfDiag(CC, 1.0);
    const b = slopBounds(r_outer, 0, hd);

    // Chunk far-east at level origin (200, 0): center (215.5, 15.5),
    // dist^2 = 46680 >> outer keep^2 ~= 21943 means DROPPED.
    const a = chunkAabb(RS, CC, 7, 4, 1.0);
    try testing.expect(!chunkVisibleRadial(a, .{ 200, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

test "deep-interior chunks dropped by slop cull (covered by finer level)" {
    const r_inner = @sqrt(rSafeSq(RS, 1.0)); // 125.5
    const big_outer: f32 = 1e5;
    const hd = chunkHalfDiag(CC, 2.0);
    const b = slopBounds(big_outer, r_inner, hd);

    // L+1 chunk (4, 4) at spacing 2.0: center (31, 31), dist^2 = 1922.
    // r_inner_keep ~= 125.5 - 32*sqrt(2) ~= 80.25, sq ~= 6440. 1922 < 6440 means DROPPED.
    const a = chunkAabb(RS, CC, 4, 4, 2.0);
    try testing.expect(!chunkVisibleRadial(a, .{ 0, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

test "level-origin offset shifts cull outcome with slop" {
    const r_outer = @sqrt(rSafeSq(RS, 1.0));
    const hd = chunkHalfDiag(CC, 1.0);
    const b = slopBounds(r_outer, 0, hd);

    // Chunk (7, 4) at level origin (200, 0): center (311.5, 15.5), dist^2
    // = 97273 >> outer keep^2 means DROPPED.
    const far = chunkAabb(RS, CC, 7, 4, 1.0);
    try testing.expect(!chunkVisibleRadial(far, .{ 200, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));

    // Chunk (0, 4) at the same level origin: center (87.5, 15.5),
    // dist^2 = 7896 < outer keep^2 means KEPT.
    const near = chunkAabb(RS, CC, 0, 4, 1.0);
    try testing.expect(chunkVisibleRadial(near, .{ 200, 0 }, .{ 0, 0 }, b.r_outer_keep_sq, b.r_inner_keep_sq));
}

// ---------------------------------------------------------------------------
// Frustum cull (XZ-plane wedge)
// ---------------------------------------------------------------------------
//
// `frustumWedge` projects the camera frustum into the XZ data plane and
// returns two bounding rays plus a forward bisector. `chunkVisibleFrustum`
// runs a 3-axis SAT against any chunk AABB:
//   1. cw_bound:  chunk fully clockwise of cw_bound  means outside-CW
//   2. ccw_bound: chunk fully counter-CW of ccw_bound means outside-CCW
//   3. forward:   chunk fully in the back half-plane means outside-behind
//
// All three tests reduce to one signed scalar per chunk corner, so the per-
// chunk cost is ~24 mul/add + 12 compares. Y-extent (terrain elevation)
// drops out; projection to XZ is exact for the visibility question we ask.
// Conservative: never culls a visible chunk; may keep some that lie just
// outside the wedge.
//
// `degenerate` is returned when the camera looks too close to vertical (or
// when the projected wedge would exceed 180 deg). Caller should treat
// degenerate as "keep everything"; over-draw at near-vertical is bounded
// because only the finest level renders at those tilts.

/// Bound vectors of an XZ wedge. `cw_bound` is the corner ray with smallest
/// cross(forward_xz, ray); `ccw_bound` is the largest. `forward_xz` is the
/// camera's forward projected to XZ; bisector of the wedge and SAT axis
/// for "behind camera".
///
/// Magnitudes are not normalized (only signs of cross/dot matter for SAT)
/// and may be further skewed by `cos_lat` baked in at construction.
///
/// 2D cross convention used throughout: cross((ax, az), (bx, bz)) = ax*bz - az*bx.
pub const WedgeBounds = struct {
    cw_bound: [2]f32,
    ccw_bound: [2]f32,
    forward_xz: [2]f32,
};

/// XZ-plane wedge bounding the camera's view frustum in data space.
pub const Wedge = union(enum) {
    /// Camera near-vertical or wedge would wrap > 180 deg; caller keeps all.
    degenerate,
    bounded: WedgeBounds,
};

/// Build the wedge from a rotation-only view matrix and a perspective
/// projection. Recovers the camera basis from the view matrix's columns:
///   right.i   =  view[i][0]
///   up.i      =  view[i][1]
///   forward.i = -view[i][2]
/// FOV half-extents from the projection (Vulkan Y-flipped):
///   half_h =  1/proj[0][0]
///   half_v = -1/proj[1][1]
///
/// `cos_lat` matches the latitude scaling the vertex shader applies to
/// world.x (`clipmap_terrain.vert: x *= cos_lat`). Baked into the bound
/// vectors here so the per-chunk SAT can test raw arcsec corners directly:
/// pre-scaling `cw_bound[1]`, `ccw_bound[1]`, and `forward_xz[0]` by
/// `cos_lat` makes `cross(bound, raw_corner) == cross(bound_unscaled,
/// scaled_corner)` (signs match, magnitudes irrelevant for SAT). Pass 1.0
/// to disable the latitude scaling.
pub fn frustumWedge(view: math.Mat4, proj: math.Mat4, cos_lat: f32) Wedge {
    const right: [3]f32 = .{ view[0][0], view[1][0], view[2][0] };
    const up: [3]f32 = .{ view[0][1], view[1][1], view[2][1] };
    const forward: [3]f32 = .{ -view[0][2], -view[1][2], -view[2][2] };
    const half_h: f32 = 1.0 / proj[0][0];
    const half_v: f32 = -1.0 / proj[1][1];

    const eps_sq: f32 = 1e-8;

    const fxz_len_sq = forward[0] * forward[0] + forward[2] * forward[2];
    if (fxz_len_sq < eps_sq) return .degenerate;
    const inv_fxz = 1.0 / @sqrt(fxz_len_sq);
    const fwd_xz: [2]f32 = .{ forward[0] * inv_fxz, forward[2] * inv_fxz };

    const offsets = [4][2]f32{
        .{ -1, -1 }, .{ 1, -1 }, .{ -1, 1 }, .{ 1, 1 },
    };

    var min_sin: f32 = std.math.inf(f32);
    var max_sin: f32 = -std.math.inf(f32);
    var cw_bound: [2]f32 = .{ 0, 0 };
    var ccw_bound: [2]f32 = .{ 0, 0 };

    for (offsets) |o| {
        const sx = o[0] * half_h;
        const sy = o[1] * half_v;
        const r_xz: [2]f32 = .{
            forward[0] + sx * right[0] + sy * up[0],
            forward[2] + sx * right[2] + sy * up[2],
        };
        const len_sq = r_xz[0] * r_xz[0] + r_xz[1] * r_xz[1];
        if (len_sq < eps_sq) return .degenerate;
        const inv_len = 1.0 / @sqrt(len_sq);
        const r_n: [2]f32 = .{ r_xz[0] * inv_len, r_xz[1] * inv_len };

        // Reject if a corner ray's XZ direction has non-positive dot with the
        // bisector; wedge would wrap > 180 deg. Picking extrema via cross alone
        // is unsafe in that case (the "extreme" ray could be the wrong side).
        if (fwd_xz[0] * r_n[0] + fwd_xz[1] * r_n[1] <= 0) return .degenerate;

        const sin_a = math.cross2(fwd_xz, r_n);
        if (sin_a < min_sin) {
            min_sin = sin_a;
            cw_bound = r_n;
        }
        if (sin_a > max_sin) {
            max_sin = sin_a;
            ccw_bound = r_n;
        }
    }

    return .{ .bounded = .{
        .cw_bound = .{ cw_bound[0], cw_bound[1] * cos_lat },
        .ccw_bound = .{ ccw_bound[0], ccw_bound[1] * cos_lat },
        .forward_xz = .{ fwd_xz[0] * cos_lat, fwd_xz[1] },
    } };
}

/// Per-chunk frustum cull. Conservative SAT: chunk is culled iff all 4
/// corners share one of the three "outside" half-planes (CW, CCW, behind).
///
/// `dx_origin = level_origin[0] - cam_xz[0]` and `dz_origin = level_origin[1]
/// - cam_xz[1]` shift the level-relative AABB to camera-relative XZ.
///
/// Corners are tested as raw arcsec; the latitude scaling lives in the
/// wedge bounds (see `frustumWedge`'s `cos_lat`).
pub fn chunkVisibleFrustum(
    chunk: ChunkAabb,
    dx_origin: f32,
    dz_origin: f32,
    wedge: Wedge,
) bool {
    const b = switch (wedge) {
        .degenerate => return true,
        .bounded => |bb| bb,
    };

    const corners: [4][2]f32 = .{
        .{ dx_origin + chunk.x_min, dz_origin + chunk.z_min },
        .{ dx_origin + chunk.x_max, dz_origin + chunk.z_min },
        .{ dx_origin + chunk.x_min, dz_origin + chunk.z_max },
        .{ dx_origin + chunk.x_max, dz_origin + chunk.z_max },
    };

    var all_cw: bool = true;
    var all_ccw: bool = true;
    var all_behind: bool = true;
    for (corners) |c| {
        // outside-CW iff cross(cw_bound, c) < 0
        if (math.cross2(b.cw_bound, c) >= 0) all_cw = false;
        // outside-CCW iff cross(ccw_bound, c) > 0
        if (math.cross2(b.ccw_bound, c) <= 0) all_ccw = false;
        // outside-behind iff dot(forward_xz, c) <= 0
        if (b.forward_xz[0] * c[0] + b.forward_xz[1] * c[1] > 0) all_behind = false;
        if (!all_cw and !all_ccw and !all_behind) return true;
    }
    return !(all_cw or all_ccw or all_behind);
}

// ---------------------------------------------------------------------------
// Frustum cull tests
// ---------------------------------------------------------------------------

/// Build a column-major rotation-only view matrix from forward/up/right
/// vectors. Mirrors how `math.lookAt` packs basis vectors so test code can
/// assemble views without going through the full lookAt pipeline.
fn testView(right: [3]f32, up: [3]f32, forward: [3]f32) math.Mat4 {
    return .{
        .{ right[0], up[0], -forward[0], 0 },
        .{ right[1], up[1], -forward[1], 0 },
        .{ right[2], up[2], -forward[2], 0 },
        .{ 0, 0, 0, 1 },
    };
}

/// Synthetic projection matrix that produces given XZ-plane half-tans.
/// `proj[0][0] = 1/half_h`, `proj[1][1] = -1/half_v`; the only entries
/// `frustumWedge` reads.
fn testProj(half_h: f32, half_v: f32) math.Mat4 {
    return .{
        .{ 1.0 / half_h, 0, 0, 0 },
        .{ 0, -1.0 / half_v, 0, 0 },
        .{ 0, 0, 0, -1 },
        .{ 0, 0, 0, 0 },
    };
}

test "frustumWedge: identity orientation, 90 deg square FoV" {
    // Forward = -Z, half_h = half_v = 1 means corner rays at 45 deg from bisector.
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const proj = testProj(1.0, 1.0);
    const w = frustumWedge(view, proj, 1.0);
    try testing.expect(w == .bounded);
    const b = w.bounded;
    // Bisector points in -Z (XZ plane).
    try testing.expectApproxEqAbs(@as(f32, 0), b.forward_xz[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1), b.forward_xz[1], 1e-6);
    // Each bound at sin(45 deg) ~= 0.707 from the bisector.
    const sin_cw = math.cross2(b.forward_xz, b.cw_bound);
    const sin_ccw = math.cross2(b.forward_xz, b.ccw_bound);
    try testing.expectApproxEqAbs(@as(f32, -std.math.sqrt1_2), sin_cw, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, std.math.sqrt1_2), sin_ccw, 1e-5);
}

test "frustumWedge: yaw 90 deg rotates wedge to +X" {
    // Forward = +X, right = +Z, up = +Y.
    const view = testView(.{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 });
    const proj = testProj(1.0, 1.0);
    const w = frustumWedge(view, proj, 1.0);
    try testing.expect(w == .bounded);
    const b = w.bounded;
    try testing.expectApproxEqAbs(@as(f32, 1), b.forward_xz[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), b.forward_xz[1], 1e-6);
}

test "frustumWedge: pitched 30 deg down picks wider bottom corners" {
    // forward = (0, -0.5, -0.866), up = (0, 0.866, -0.5), 60 deg FoV (half-tan
    // 0.577); bottom corners reach 45 deg azimuth, top only 26.6 deg. Bounds
    // must come from the bottom corners (sin = +/-0.707), not top (+/-0.447).
    const f: [3]f32 = .{ 0, -0.5, -0.8660254 };
    const u: [3]f32 = .{ 0, 0.8660254, -0.5 };
    const r: [3]f32 = .{ 1, 0, 0 };
    const view = testView(r, u, f);
    const half_t = 0.57735026; // tan(30 deg)
    const proj = testProj(half_t, half_t);
    const w = frustumWedge(view, proj, 1.0);
    try testing.expect(w == .bounded);
    const b = w.bounded;
    const sin_cw = math.cross2(b.forward_xz, b.cw_bound);
    const sin_ccw = math.cross2(b.forward_xz, b.ccw_bound);
    try testing.expectApproxEqAbs(@as(f32, -std.math.sqrt1_2), sin_cw, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, std.math.sqrt1_2), sin_ccw, 1e-3);
}

test "frustumWedge: near-straight-down camera returns degenerate" {
    // Forward almost -Y. |forward.xz| < eps after squaring.
    const f: [3]f32 = .{ 0, -0.99999, -0.001 };
    const u: [3]f32 = .{ 0, 0.001, -0.99999 };
    const view = testView(.{ 1, 0, 0 }, u, f);
    const proj = testProj(1.0, 1.0);
    try testing.expect(frustumWedge(view, proj, 1.0) == .degenerate);
}

test "frustumWedge: 60 deg pitch + 90 deg FoV trips degenerate (bottom wraps past horizon)" {
    // forward.xz still well-defined, but bottom corner rays' XZ projection
    // points back behind the bisector means wrap > 180 deg fallback.
    const f: [3]f32 = .{ 0, -0.8660254, -0.5 };
    const u: [3]f32 = .{ 0, 0.5, -0.8660254 };
    const view = testView(.{ 1, 0, 0 }, u, f);
    const proj = testProj(1.0, 1.0); // 90 deg FoV
    try testing.expect(frustumWedge(view, proj, 1.0) == .degenerate);
}

// Helper: pretend chunk centered at (cx, cz) world-XZ with half-extent `h`,
// camera at origin. Skips the chunkAabb + level_origin gymnastics for tests.
fn pseudoChunk(cx: f32, cz: f32, h: f32) ChunkAabb {
    return .{
        .x_min = cx - h,
        .x_max = cx + h,
        .z_min = cz - h,
        .z_max = cz + h,
    };
}

test "chunkVisibleFrustum: chunk in front of forward-look camera is kept" {
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 1.0);
    try testing.expect(chunkVisibleFrustum(pseudoChunk(0, -50, 5), 0, 0, w));
}

test "chunkVisibleFrustum: chunk directly behind camera is culled" {
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 1.0);
    try testing.expect(!chunkVisibleFrustum(pseudoChunk(0, 50, 5), 0, 0, w));
}

test "chunkVisibleFrustum: chunk far to the side is culled (narrow FoV)" {
    // 30 deg FoV means half-angle 15 deg. Chunk at 90 deg azimuth from forward.
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const half_t = 0.26794919; // tan(15 deg)
    const w = frustumWedge(view, testProj(half_t, half_t), 1.0);
    try testing.expect(!chunkVisibleFrustum(pseudoChunk(50, 0, 5), 0, 0, w));
}

test "chunkVisibleFrustum: same side chunk is kept at very wide FoV" {
    // 160 deg FoV means half-angle 80 deg. Chunk at 70 deg azimuth (still inside).
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const half_t = 5.6712818; // tan(80 deg)
    const w = frustumWedge(view, testProj(half_t, half_t), 1.0);
    // 70 deg azimuth from -Z bisector: position = (sin 70 deg, cos 70 deg) * 50, but
    // forward is -Z so "in front" is -Z; corner at (sin70 * 50, -cos70 * 50)
    try testing.expect(chunkVisibleFrustum(pseudoChunk(46.98, -17.10, 5), 0, 0, w));
}

test "chunkVisibleFrustum: chunk straddling the bisector is kept" {
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 1.0);
    // Chunk (-5, -50) to (5, -40); squarely in front, straddling X=0.
    try testing.expect(chunkVisibleFrustum(pseudoChunk(0, -45, 5), 0, 0, w));
}

test "chunkVisibleFrustum: chunk containing camera origin is kept" {
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(0.5, 0.5), 1.0);
    // 5x5 chunk centered on (0, 0). Camera origin inside.
    try testing.expect(chunkVisibleFrustum(pseudoChunk(0, 0, 5), 0, 0, w));
}

test "chunkVisibleFrustum: degenerate wedge keeps everything" {
    const w: Wedge = .degenerate;
    try testing.expect(chunkVisibleFrustum(pseudoChunk(50, 0, 5), 0, 0, w));
    try testing.expect(chunkVisibleFrustum(pseudoChunk(0, 50, 5), 0, 0, w)); // behind
}

test "chunkVisibleFrustum: behind-camera chunk straddling X-axis is culled" {
    // Specifically the case the third SAT axis (forward bisector) catches:
    // corners at (+/-a, +b) span both sides of the bisector but all are behind.
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 1.0);
    // Chunk centered at (0, 50) with half-extent 30 means corners (+/-30, 20..80);
    // straddles X axis, fully behind camera (Z > 0).
    try testing.expect(!chunkVisibleFrustum(pseudoChunk(0, 50, 30), 0, 0, w));
}

test "chunkVisibleFrustum: level-origin offset moves chunk into/out of cull" {
    const view = testView(.{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, -1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 1.0);
    const chunk = pseudoChunk(0, 0, 5); // local-extent +/-5 at origin

    // Level origin (0, -50): chunk world-XZ at (0, -50); in front, kept.
    try testing.expect(chunkVisibleFrustum(chunk, 0, -50, w));
    // Level origin (0, +50): chunk world-XZ at (0, +50); behind, culled.
    try testing.expect(!chunkVisibleFrustum(chunk, 0, 50, w));
}

test "chunkVisibleFrustum: cos_lat fix keeps wide-arcsec chunks looking N/S" {
    // Lat 45 deg (cos_lat ~= 0.707), camera looking +Z (south), 90 deg FoV. Chunk
    // at arcsec X=+/-70 / Z=70 is at 45 deg in arcsec but only 35 deg in scaled
    // world (where the rendered frustum lives). Within visible frustum.
    //
    // Baking cos_lat into the wedge bounds (rather than scaling chunk
    // corners) makes the cull match the rendered frustum without per-chunk
    // multiplies on the hot path.
    const view = testView(.{ -1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 0.7071068);
    try testing.expect(chunkVisibleFrustum(pseudoChunk(70, 70, 5), 0, 0, w));
    try testing.expect(chunkVisibleFrustum(pseudoChunk(-70, 70, 5), 0, 0, w));
}

test "chunkVisibleFrustum: cos_lat fix culls narrow-arcsec chunks looking E/W" {
    // Same lat, camera looking +X (east), 90 deg FoV. Chunk at arcsec X=70 /
    // Z=+/-60 is at 40.6 deg in arcsec (inside a naive 45 deg wedge), but in scaled
    // world its X compresses to 49.5 -> angle 50.5 deg -> past the 45 deg bound.
    const view = testView(.{ 0, 0, 1 }, .{ 0, 1, 0 }, .{ 1, 0, 0 });
    const w = frustumWedge(view, testProj(1.0, 1.0), 0.7071068);
    try testing.expect(!chunkVisibleFrustum(pseudoChunk(70, 60, 5), 0, 0, w));
    try testing.expect(!chunkVisibleFrustum(pseudoChunk(70, -60, 5), 0, 0, w));
}

test "pickChunksPerSide: scales with ring_size, capped, even" {
    const max: u32 = 8;
    try testing.expectEqual(@as(u32, 2), pickChunksPerSide(63, max));
    try testing.expectEqual(@as(u32, 4), pickChunksPerSide(127, max));
    // ring 255 stays at 8x8; coarser chunks lose cull granularity here.
    try testing.expectEqual(@as(u32, 8), pickChunksPerSide(255, max));
    try testing.expectEqual(@as(u32, 8), pickChunksPerSide(501, max));
    try testing.expectEqual(@as(u32, 8), pickChunksPerSide(1001, max));
    try testing.expectEqual(@as(u32, 8), pickChunksPerSide(2047, max));
}

test "pickChunksPerSide: minimum 2, never 0 or 1" {
    const max: u32 = 8;
    try testing.expectEqual(@as(u32, 2), pickChunksPerSide(1, max));
    try testing.expectEqual(@as(u32, 2), pickChunksPerSide(33, max));
}

// ---------------------------------------------------------------------------
// chunkIsOcean
// ---------------------------------------------------------------------------

fn testCatalog(cells: []const u32) TileCatalog {
    var cat = TileCatalog{
        .allocator = undefined,
        .descriptors = &.{},
        .dir_path = &.{},
        .single_tile = false,
        .shared_features = 0,
        .shared_mlp_hidden_dim = 0,
        .shared_num_outputs = 0,
        .catalog_grid = @splat(0),
    };
    for (cells) |c| {
        cat.catalog_grid[c / 64] |= @as(u64, 1) << @intCast(c % 64);
    }
    return cat;
}

test "chunkIsOcean: empty catalog means ocean everywhere" {
    const cat = testCatalog(&.{});
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 3600, .z_min = 0, .z_max = 3600 };
    try testing.expect(chunkIsOcean(aabb, .{ 0, 0 }, &cat));
}

test "chunkIsOcean: chunk over a cataloged tile is not ocean" {
    // Tile at lon=-122, lat=47: lon_idx = -122 + 180 = 58, lat_idx = 89 - 47 = 42
    // cell = 42 * 360 + 58 = 15178
    const cat = testCatalog(&.{15178});
    // snapped_pos places chunk at lon=-122, lat=47 (in arcseconds)
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 3599, .z_min = 0, .z_max = 3599 };
    try testing.expect(!chunkIsOcean(aabb, .{ -122.0 * 3600.0, 47.0 * 3600.0 }, &cat));
}

test "chunkIsOcean: chunk spanning two tiles, one cataloged" {
    // Chunk straddles lon=-122 and lon=-121. Only -122 has a tile.
    // lon=-122 -> cell 15178 (lat_idx=42, lon_idx=58)
    const cat = testCatalog(&.{15178});
    // Chunk spans from lon=-122.5 to lon=-120.5 (7200 arcsec wide centered on -121.5)
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 7200, .z_min = 0, .z_max = 3599 };
    try testing.expect(!chunkIsOcean(aabb, .{ -122.5 * 3600.0, 47.0 * 3600.0 }, &cat));
}

test "chunkIsOcean: chunk beyond south pole is ocean (out-of-range cells skipped)" {
    const cat = testCatalog(&.{0});
    // lat = -91 -> lat_idx = 89 - (-91) = 180 >= GRID_LAT_CELLS(180), skipped
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 3599, .z_min = 0, .z_max = 3599 };
    try testing.expect(chunkIsOcean(aabb, .{ 0, -91.0 * 3600.0 }, &cat));
}

test "chunkIsOcean: chunk at antimeridian with no catalog tile" {
    const cat = testCatalog(&.{});
    // Chunk near lon=179 (arcsec = 179 * 3600 = 644400)
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 7200, .z_min = 0, .z_max = 3600 };
    try testing.expect(chunkIsOcean(aabb, .{ 179.0 * 3600.0, 0 }, &cat));
}

test "chunkIsOcean: chunk straddling antimeridian finds wrapped tile" {
    // Tile at lon=-180, lat=0: lon_idx=0, lat_idx=89 -> cell = 89*360 = 32040.
    const cat = testCatalog(&.{32040});
    // Chunk at lon=+179.5..+180.5, lat=0..1. The +180..+180.5 half should
    // wrap to lon=-180..-179.5 and hit the cataloged tile.
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 3600, .z_min = 0, .z_max = 3600 };
    try testing.expect(!chunkIsOcean(aabb, .{ 179.5 * 3600.0, 0 }, &cat));
}

test "chunkIsOcean: multi-lap east finds wrapped tile" {
    // Same wrapped lookup, but camera is many laps east. @mod must handle
    // arbitrary integer ranges, not just single-step wrap.
    const cat = testCatalog(&.{32040}); // lon=-180, lat=0
    const aabb = ChunkAabb{ .x_min = 0, .x_max = 3599, .z_min = 0, .z_max = 3599 };
    // 540 deg east of origin: lon_min = floor(540) = 540 -> @mod(540+180, 360) = 0.
    try testing.expect(!chunkIsOcean(aabb, .{ 540.0 * 3600.0, 0 }, &cat));
}
