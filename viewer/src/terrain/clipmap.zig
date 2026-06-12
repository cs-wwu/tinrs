//! Geometry clipmap terrain system.
//!
//! Owns both compute (strip updates) and graphics (terrain rendering) pipelines.
//! Ring buffers store vec4(z_fine, z_delta, normal_x, normal_z) per vertex.
//! The compute shader evaluates terrain for dirty strips on camera scroll.
//! The vertex shader reads cached data from the ring buffer via toroidal addressing.
//!
//! Tile weights are bound from a borrowed `?*TileSystem` (or an internal empty
//! SSBO when none is supplied; procedural-terrain fallback). Streaming is
//! delegated to `TileSystem.recordStream`, called at the start of `recordUpdate`.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const math = @import("math");
const renderer_mod = @import("../render/renderer.zig");
const debug = @import("../render/debug.zig");
const coords = @import("coords.zig");
const tile_ssbo = @import("tile_ssbo.zig");
const tile_system_mod = @import("tile_system.zig");
const Camera = @import("../app/camera.zig").Camera;

const setup = @import("clipmap_setup.zig");
const cull = @import("clipmap_cull.zig");

// Maximum number of LOD levels (actual count is runtime-configurable via --num-levels).
pub const MAX_LEVELS: u32 = 12;

/// Compile-time upper bound on chunks per side; sizes the per-frame
/// DrawEntries SSBO. Runtime `chunks_per_side` (per-Clipmap, picked by
/// `cull.pickChunksPerSide`) is at most this; small rings shrink the count
/// to amortize per-instance setup, large rings cap here for buffer budget.
pub const MAX_CHUNKS_PER_SIDE: u32 = 8;

/// Must match clipmap_update.comp push_constant layout exactly.
///
/// `scroll_offset` is `uvec2`: persistent CPU state may drift negative, but
/// every push pre-mods into `[0, ring_size)` so the shader can use plain
/// unsigned modulo (GLSL's `%` on negatives is implementation-defined).
///
/// `chunk_origin` and `chunk_vertex_dim` drive the chunked dispatch path
/// (`update_mode == 0`). Strip modes (1/2/3) ignore them.
// std430 alignment: every vec2/uvec2 field MUST land on an 8-byte boundary.
// chunk_vertex_dim (u32) is intentionally last so the trailing vec2/uvec2 fields
// stay 8-aligned without GLSL inserting padding that Zig wouldn't reproduce.
pub const ComputePushConstants = extern struct {
    ring_size: u32,
    update_mode: u32,
    strip_row: i32,
    strip_col: i32,
    scroll_offset: [2]u32,
    /// Camera-relative level center in arcsec (snapped_pos - cam_anchor).
    level_origin: [2]f32,
    grid_spacing: f32,
    level_base_offset: u32,
    chunk_origin: [2]u32 = .{ 0, 0 },
    /// Camera position rounded to integer arcsec. Exactly representable in
    /// f32 (|x| < 2^24). Shader adds this back when reconstructing absolute
    /// world position for tile lookup and cos(lat); tile origins (also exact
    /// integer arcsec) become small via `tile_origin - cam_anchor`, so the
    /// per-thread `local = world_rel - tile_origin_rel` avoids catastrophic
    /// cancellation that f32 absolute coords cause at high longitude.
    cam_anchor: [2]f32 = .{ 0, 0 },
    chunk_vertex_dim: u32 = 0,

    comptime {
        if (@sizeOf(ComputePushConstants) != 60) @compileError("ComputePushConstants size changed - check GLSL std430 alignment");
    }
};

/// Per-chunk draw record. `r_inner = 0` disables the inner clip (min_level,
/// no hole). `cull_state` is a CULL_* bitmask, non-zero only when the
/// `by_cull_state` overlay keeps culled chunks drawn for tinting.
///
/// std430 layout; must mirror the GLSL `DrawEntry` in clipmap_terrain.vert
/// and terrain.frag.
pub const DrawEntry = extern struct {
    level_origin: [2]f32, // camera-relative (x, z)
    grid_spacing: f32,
    r_inner: f32,
    r_outer: f32,
    level_base_offset: u32,
    level_idx: u32,
    cull_state: u32 = 0,
    scroll_offset: [2]u32, // pre-mod'd to [0, ring_size)
    chunk_origin: [2]u32,

    comptime {
        if (@sizeOf(DrawEntry) != 48) @compileError("DrawEntry size changed - check std430 alignment");
    }
};

/// Upper bound on per-frame visible chunks; sizes the per-frame SSBO.
pub const MAX_DRAWS: u32 = MAX_LEVELS * MAX_CHUNKS_PER_SIDE * MAX_CHUNKS_PER_SIDE;

/// UBO for view/proj matrices + scene parameters (shared between terrain + sky shaders).
/// Per-frame double-buffered to avoid writing data the GPU is still reading.
/// Layout matches GLSL std140 (vec4 fields before scalars, 16-byte aligned total).
pub const SceneUBO = extern struct {
    view: [4][4]f32, //                  offset 0    (64)  rotation-only; sky reconstructs rays from it
    proj: [4][4]f32, //                  offset 64   (64)  sky reads fov/aspect from [0][0] / [1][1]
    sun_dir: [4]f32, //                  offset 128  (16)
    fog_max_dist: f32, //                offset 144
    no_effects: u32, //                  offset 148
    transfer_function: u32, //           offset 152
    /// 0 = off, 1 = by_level, 2 = by_chunk, 3 = by_cull_state. Sky shader ignores.
    debug_overlay: u32 = 0, //           offset 156
    cam_elev: f32 = 0, //                offset 160  (camera Y in arcsec)
    cam_z: f32 = 0, //                   offset 164  (camera global Z in arcsec)
    aircraft_msl_m: f32 = 0, //          offset 168  (aircraft altitude MSL, meters; TAWS clearance)
    taws: u32 = 0, //                    offset 172  (0 = off, 1 = hazard overlay)
    proj_view: [4][4]f32, //             offset 176  (64)  proj * view, premultiplied on CPU (terrain vert)
    /// Constant palette LUT: written into both mapped UBOs once at init (see
    /// `init`); recordDraw copies only the dynamic prefix before this field.
    hypso_lut: [HYPSO_LUT_LEN][4]f32 = undefined, // offset 240 (std140 vec4 array, stride 16)

    comptime {
        if (@sizeOf(SceneUBO) != 240 + HYPSO_LUT_LEN * 16) @compileError("SceneUBO size changed - check GLSL std140 alignment");
    }
};

/// Photographic terrain palette tuned to read like aerial photography of
/// temperate terrain: dark conifer forest 0-1500m, treeline transition
/// 1500-2200m, bare gray-brown alpine rock 2200-4500m, light gray rising to
/// snow above. Saturated colors are deliberately avoided. Real terrain from
/// altitude is dark and desaturated; the lit/shadow falloff (driven by surface
/// normal) carries most of the visual structure, not the base hue.
///
/// Piecewise-linear knots; every knot elevation is a multiple of 100m so the
/// 100m-spaced LUT below reproduces the curve exactly. The shader samples the
/// LUT (2 UBO loads + 1 mix) instead of walking the branch chain per fragment.
const HYPSO_KNOTS = [_]struct { elev_m: f32, color: [3]f32 }{
    .{ .elev_m = -100.0, .color = .{ 0.06, 0.14, 0.07 } }, // below sea level
    .{ .elev_m = 0.0, .color = .{ 0.09, 0.20, 0.09 } },
    .{ .elev_m = 200.0, .color = .{ 0.12, 0.23, 0.10 } },
    .{ .elev_m = 800.0, .color = .{ 0.17, 0.27, 0.12 } },
    .{ .elev_m = 1500.0, .color = .{ 0.26, 0.29, 0.17 } },
    .{ .elev_m = 2200.0, .color = .{ 0.40, 0.34, 0.24 } },
    .{ .elev_m = 3000.0, .color = .{ 0.50, 0.44, 0.35 } },
    .{ .elev_m = 4500.0, .color = .{ 0.60, 0.55, 0.47 } },
    .{ .elev_m = 6000.0, .color = .{ 0.74, 0.71, 0.67 } },
    .{ .elev_m = 7500.0, .color = .{ 0.88, 0.88, 0.88 } },
    .{ .elev_m = 8500.0, .color = .{ 0.96, 0.97, 1.00 } }, // snow; holds above
};

/// LUT domain: [-100, 8500] m at 100m spacing. Entry i = color at -100 + i*100.
/// Must match HYPSO_LUT_LEN / the mapping in terrain.frag's hypsometric().
/// (The original branch chain held one flat color for ALL elev < 0; the LUT
/// ramps over [-100, 0] instead. Sub-sea land is vanishingly rare and mostly
/// water-masked, so the difference is invisible.)
pub const HYPSO_LUT_LEN = 87;

/// Piecewise-linear evaluation of HYPSO_KNOTS; clamps outside the knot range.
pub fn hypsoColor(elev_m: f32) [3]f32 {
    if (elev_m <= HYPSO_KNOTS[0].elev_m) return HYPSO_KNOTS[0].color;
    for (1..HYPSO_KNOTS.len) |i| {
        const k0 = HYPSO_KNOTS[i - 1];
        const k1 = HYPSO_KNOTS[i];
        if (elev_m < k1.elev_m) {
            const t = (elev_m - k0.elev_m) / (k1.elev_m - k0.elev_m);
            return math.lerpVec3(k0.color, k1.color, t);
        }
    }
    return HYPSO_KNOTS[HYPSO_KNOTS.len - 1].color;
}

pub const HYPSO_LUT: [HYPSO_LUT_LEN][4]f32 = blk: {
    @setEvalBranchQuota(2_000); // 87 entries x ~11-knot scan slightly exceeds the 1000 default
    var lut: [HYPSO_LUT_LEN][4]f32 = undefined;
    for (&lut, 0..) |*entry, i| {
        const c = hypsoColor(-100.0 + @as(f32, @floatFromInt(i)) * 100.0);
        entry.* = .{ c[0], c[1], c[2], 1.0 };
    }
    break :blk lut;
};

pub const SceneParams = struct {
    view: math.Mat4,
    proj: math.Mat4,
    sun_dir: math.Vec3,
    cam_pos: [3]f64,
    fog_max_dist: f32,
    no_effects: bool,
    transfer_function: u32,
    aircraft_msl_m: f32,
    taws: bool,
};

/// Per-frame inputs that don't derive from the camera. Passed as a struct so
/// call sites name what they're setting instead of trailing positional args.
pub const SceneOverrides = struct {
    fog_max_dist: f32,
    no_effects: bool,
    transfer_function: u32,
    /// Aircraft altitude in arcsec (pose.position[1]); converted to meters for
    /// the TAWS clearance test. This is the aircraft's altitude, not the
    /// camera's (they differ in chase/cockpit-offset views).
    aircraft_alt_arcsec: f32 = 0,
    /// TAWS hazard-coloring overlay enabled (terrain recolored by clearance).
    taws: bool = false,
};

/// Build SceneParams from a camera + per-frame overrides. Centralizes
/// view/proj/sun-direction derivation so main and benchmark loops stay in sync.
pub fn buildSceneParams(camera: *const Camera, aspect: f32, opts: SceneOverrides) SceneParams {
    return .{
        .view = camera.viewMatrixRotOnly(),
        .proj = camera.projMatrix(aspect),
        .sun_dir = coords.sunDirAtNoon(@floatCast(camera.pose.position[2])),
        .cam_pos = camera.pose.position,
        .fog_max_dist = opts.fog_max_dist,
        .no_effects = opts.no_effects,
        .transfer_function = opts.transfer_function,
        .aircraft_msl_m = coords.arcsecToMeters(opts.aircraft_alt_arcsec),
        .taws = opts.taws,
    };
}

pub const ClipmapLevel = struct {
    scroll_offset: [2]i32 = .{ 0, 0 },
    snapped_pos: [2]f32 = .{ 0.0, 0.0 },
    grid_spacing: f32,
    initialized: bool = false,
};

pub const Clipmap = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,

    // Compute pipeline (strip updates)
    compute_pipeline: vk.Pipeline,
    compute_pipeline_layout: vk.PipelineLayout,

    // Graphics pipeline (terrain rendering)
    /// Default (cheap) pipelines: no `frag_instance` varying, no DrawEntries
    /// SSBO read in the fragment stage. Bound when no debug overlay is active.
    graphics_pipeline: vk.Pipeline,
    /// Polygon-mode=line variant. Built unconditionally so F2 (render-mode
    /// cycle) doesn't have to allocate a pipeline mid-frame.
    wireframe_pipeline: vk.Pipeline,
    /// Debug variants: same pipelines plus the varying + frag SSBO read so
    /// the by_level / by_chunk / by_cull_state overlays can sample DrawEntries.
    /// `recordDraw` swaps to these when `debug.state.color_overlay != .off`.
    graphics_pipeline_debug: vk.Pipeline,
    wireframe_pipeline_debug: vk.Pipeline,
    graphics_pipeline_layout: vk.PipelineLayout,

    // Shared descriptor (ring buffer + weights + UBO). desc_layout is BORROWED:
    // owned by Scene so it survives clipmap rebuilds (Sky's pipeline layout is
    // built from it); deinit does not destroy it. pool + sets are owned here.
    desc_layout: vk.DescriptorSetLayout,
    desc_pool: vk.DescriptorPool,
    desc_sets: [2]vk.DescriptorSet, // per-frame (double-buffered for UBO)

    // Per-frame UBO buffers (view + proj matrices, persistently mapped)
    ubo_bufs: [2]renderer_mod.BufferWithMemory,
    ubo_maps: [2][*]u8,

    // Per-frame persistently-mapped DrawEntry SSBOs.
    draw_entry_bufs: [2]renderer_mod.BufferWithMemory,
    draw_entry_maps: [2][*]u8,

    // Ring buffer (all levels packed contiguously)
    ring_buf: renderer_mod.BufferWithMemory,
    ring_buf_size: vk.DeviceSize,

    // Tile-streaming hookup. Borrowed; null = procedural terrain mode.
    tile_system: ?*tile_system_mod.TileSystem,
    /// Owned 16-byte fallback SSBO. Bound at `binding = 1` only when
    /// `tile_system == null` so the shader sees `tile_count = 0` and falls
    /// through to its procedural path.
    procedural_ssbo: ?renderer_mod.BufferWithMemory,
    /// Owned full-size zero dir grid for procedural mode (binding 3).
    procedural_dir_grid: ?renderer_mod.BufferWithMemory,

    // Single shared chunk index buffer. Every chunk in every level draws the
    // same `chunk_cells^2 * 6` indices; chunk position is a push constant.
    chunk_index_buf: renderer_mod.BufferWithMemory,
    chunk_index_count: u32,

    // Per-level state
    levels: [MAX_LEVELS]ClipmapLevel,
    base_spacing: f32,
    ring_size: u32,
    num_levels: u32,
    entries_per_level: u32,
    /// Cells per chunk side. `chunk_vertex_dim = chunk_cells + 1`.
    chunk_cells: u32,
    chunk_vertex_dim: u32,
    /// Picked by `cull.pickChunksPerSide` from `requested_ring_size` at init.
    chunks_per_side: u32,

    r_outer_precomp: [MAX_LEVELS]f32,
    r_inner_precomp: [MAX_LEVELS]f32,
    r_outer_keep_sq_precomp: [MAX_LEVELS]f32,
    r_inner_keep_sq_precomp: [MAX_LEVELS]f32,

    // Stats
    last_evals: u32 = 0,
    last_strips: u32 = 0,
    last_refills: u32 = 0,

    /// Chunk-aligned ring layout for a requested ring dimension.
    pub const RingLayout = struct {
        chunks_per_side: u32,
        chunk_cells: u32,
        chunk_vertex_dim: u32,
        /// Always odd: `chunk_cells * chunks_per_side + 1`.
        ring_size: u32,
    };

    /// Resolve a requested ring dimension to the chunk-aligned layout that
    /// `init` will actually use (ring_size rounded up so chunks tile exactly).
    /// Exposed so the settings UI can show the achievable ring_size as the user
    /// drags, before a rebuild commits it. `requested` must be >= 2 (the clipmap
    /// range floor is 63; below 2 the `- 1` underflows).
    pub fn ringLayout(requested: u32) RingLayout {
        const chunks_per_side: u32 = cull.pickChunksPerSide(requested, MAX_CHUNKS_PER_SIDE);
        const chunk_cells: u32 = (requested - 1 + chunks_per_side - 1) / chunks_per_side;
        return .{
            .chunks_per_side = chunks_per_side,
            .chunk_cells = chunk_cells,
            .chunk_vertex_dim = chunk_cells + 1,
            .ring_size = chunk_cells * chunks_per_side + 1,
        };
    }

    /// Create the shared descriptor set layout (binding definitions for the
    /// clipmap compute/graphics + sky pipelines). Owned by the caller (Scene),
    /// passed into `Clipmap.init` and `Sky.init`, and kept alive across clipmap
    /// rebuilds so Sky's pipeline layout stays valid. Size-independent.
    pub fn createDescLayout(ctx: renderer_mod.GpuContext) !vk.DescriptorSetLayout {
        return setup.createDescLayout(ctx.vkd, ctx.device);
    }

    /// Build a Clipmap. `tile_system` is borrowed (caller owns it); pass null
    /// for procedural terrain mode (Clipmap allocates a tiny empty SSBO for the
    /// descriptor binding). `desc_layout` is borrowed (see `createDescLayout`).
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: renderer_mod.GpuContext,
        render_pass: vk.RenderPass,
        samples: vk.SampleCountFlags,
        desc_layout: vk.DescriptorSetLayout,
        base_spacing: f32,
        requested_ring_size: u32,
        num_levels: u32,
        tile_system: ?*tile_system_mod.TileSystem,
    ) !Clipmap {
        const vkd = ctx.vkd;
        const device = ctx.device;
        const mem_props = ctx.mem_props;
        const queue = ctx.queue;
        const cmd_pool = ctx.cmd_pool;

        // Pick chunks_per_side first, then derive chunk_cells; ring_size is
        // rounded up to `chunk_cells * chunks_per_side + 1` so chunks tile
        // exactly. chunks_per_side is even, so the result is always odd
        // (symmetric on a center vertex). Shared with `ringLayout` so the
        // settings UI previews the same value a rebuild will land on.
        const layout = ringLayout(requested_ring_size);
        const chunks_per_side: u32 = layout.chunks_per_side;
        const chunk_cells: u32 = layout.chunk_cells;
        const chunk_vertex_dim: u32 = layout.chunk_vertex_dim;
        const ring_size: u32 = layout.ring_size;
        if (ring_size != requested_ring_size) {
            std.log.info("Clipmap: rounded ring_size {d} -> {d} for chunk alignment ({d} cells/chunk x {d} chunks)", .{
                requested_ring_size, ring_size, chunk_cells, chunks_per_side,
            });
        }
        const entries_per_level: u32 = ring_size * ring_size;

        const total_entries: u64 = @as(u64, num_levels) * @as(u64, entries_per_level);
        const ring_buf_size: vk.DeviceSize = total_entries * 4 * @sizeOf(f32); // vec4 per entry

        const ring_buf = try renderer_mod.createBuffer(
            &vkd,
            device,
            mem_props,
            ring_buf_size,
            .{ .storage_buffer_bit = true },
            .{ .device_local_bit = true },
        );
        errdefer {
            vkd.destroyBuffer(device, ring_buf.buffer, null);
            vkd.freeMemory(device, ring_buf.memory, null);
        }

        // One shared `chunk_cells^2 * 6` index buffer; every chunk in every
        // level draws it. Per-chunk position rides in the push constant.
        const chunk_idx = try setup.createChunkIndexBuffer(allocator, &vkd, device, mem_props, queue, cmd_pool, chunk_cells);
        errdefer {
            vkd.destroyBuffer(device, chunk_idx.buf.buffer, null);
            vkd.freeMemory(device, chunk_idx.buf.memory, null);
        }

        // Streaming mode borrows from tile_system; procedural mode owns a
        // tiny empty SSBO sized for the 4-uint header (tile_count=0) plus a
        // zero-filled dir grid (so binding 3 has a valid handle).
        var procedural_ssbo: ?renderer_mod.BufferWithMemory = null;
        errdefer if (procedural_ssbo) |buf| {
            vkd.destroyBuffer(device, buf.buffer, null);
            vkd.freeMemory(device, buf.memory, null);
        };
        var procedural_dir_grid: ?renderer_mod.BufferWithMemory = null;
        errdefer if (procedural_dir_grid) |buf| {
            vkd.destroyBuffer(device, buf.buffer, null);
            vkd.freeMemory(device, buf.memory, null);
        };

        const weights_buf_handle: vk.Buffer = blk: {
            if (tile_system) |ts| break :blk ts.weightsBuffer();
            procedural_ssbo = try tile_ssbo.createEmptySSBO(&vkd, device, mem_props);
            break :blk procedural_ssbo.?.buffer;
        };
        const dir_grid_buf_handle: vk.Buffer = blk: {
            if (tile_system) |ts| break :blk ts.dirGridBuffer();
            procedural_dir_grid = try tile_ssbo.createEmptyDirGrid(&vkd, device, mem_props);
            break :blk procedural_dir_grid.?.buffer;
        };

        const desc = try setup.createDescriptors(vkd, device, mem_props, desc_layout, ring_buf, ring_buf_size, weights_buf_handle, dir_grid_buf_handle);
        // The hypsometric LUT is constant: write it into both mapped UBOs once
        // here so recordDraw only copies the dynamic prefix each frame.
        const lut_bytes = std.mem.sliceAsBytes(HYPSO_LUT[0..]);
        for (desc.ubo_maps) |map| {
            @memcpy(map[@offsetOf(SceneUBO, "hypso_lut")..][0..lut_bytes.len], lut_bytes);
        }
        errdefer {
            for (0..2) |i| {
                vkd.unmapMemory(device, desc.ubo_bufs[i].memory);
                vkd.destroyBuffer(device, desc.ubo_bufs[i].buffer, null);
                vkd.freeMemory(device, desc.ubo_bufs[i].memory, null);
            }
            vkd.destroyDescriptorPool(device, desc.desc_pool, null);
        }

        const pipes = try setup.createPipelines(vkd, device, render_pass, desc_layout, samples, ring_size, chunk_vertex_dim);
        errdefer {
            vkd.destroyPipeline(device, pipes.debug.wireframe, null);
            vkd.destroyPipeline(device, pipes.debug.filled, null);
            vkd.destroyPipeline(device, pipes.default.wireframe, null);
            vkd.destroyPipeline(device, pipes.default.filled, null);
            vkd.destroyPipelineLayout(device, pipes.graphics_pipeline_layout, null);
            vkd.destroyPipeline(device, pipes.compute_pipeline, null);
            vkd.destroyPipelineLayout(device, pipes.compute_pipeline_layout, null);
        }

        var levels: [MAX_LEVELS]ClipmapLevel = undefined;
        for (0..num_levels) |i| {
            const level_spacing = base_spacing * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(i)));
            levels[i] = .{ .grid_spacing = level_spacing };
        }

        const radii = precomputeCullRadii(num_levels, base_spacing, ring_size, chunk_cells);
        const r_outer_precomp = radii.r_outer;
        const r_inner_precomp = radii.r_inner;
        const r_outer_keep_sq_precomp = radii.r_outer_keep_sq;
        const r_inner_keep_sq_precomp = radii.r_inner_keep_sq;

        if (tile_system) |ts| {
            std.log.debug("Clipmap ready: {d} levels, {d}x{d} ring, {d}x{d} chunks/level, streaming {d} catalog tiles", .{
                num_levels, ring_size, ring_size, chunks_per_side, chunks_per_side, ts.catalogCount(),
            });
        } else {
            std.log.debug("Clipmap ready: {d} levels, {d}x{d} ring, {d}x{d} chunks/level, {d:.1} KB, base spacing {d:.6} (procedural)", .{
                num_levels, ring_size, ring_size, chunks_per_side, chunks_per_side,
                @as(f32, @floatFromInt(ring_buf_size)) / 1024.0, base_spacing,
            });
        }

        return .{
            .vkd = vkd,
            .device = device,
            .compute_pipeline = pipes.compute_pipeline,
            .compute_pipeline_layout = pipes.compute_pipeline_layout,
            .graphics_pipeline = pipes.default.filled,
            .wireframe_pipeline = pipes.default.wireframe,
            .graphics_pipeline_debug = pipes.debug.filled,
            .wireframe_pipeline_debug = pipes.debug.wireframe,
            .graphics_pipeline_layout = pipes.graphics_pipeline_layout,
            .desc_layout = desc_layout,
            .desc_pool = desc.desc_pool,
            .desc_sets = desc.desc_sets,
            .ubo_bufs = desc.ubo_bufs,
            .ubo_maps = desc.ubo_maps,
            .draw_entry_bufs = desc.draw_entry_bufs,
            .draw_entry_maps = desc.draw_entry_maps,
            .ring_buf = ring_buf,
            .ring_buf_size = ring_buf_size,
            .tile_system = tile_system,
            .procedural_ssbo = procedural_ssbo,
            .procedural_dir_grid = procedural_dir_grid,
            .chunk_index_buf = chunk_idx.buf,
            .chunk_index_count = chunk_idx.count,
            .levels = levels,
            .base_spacing = base_spacing,
            .ring_size = ring_size,
            .num_levels = num_levels,
            .entries_per_level = entries_per_level,
            .chunk_cells = chunk_cells,
            .chunk_vertex_dim = chunk_vertex_dim,
            .chunks_per_side = chunks_per_side,
            .r_outer_precomp = r_outer_precomp,
            .r_inner_precomp = r_inner_precomp,
            .r_outer_keep_sq_precomp = r_outer_keep_sq_precomp,
            .r_inner_keep_sq_precomp = r_inner_keep_sq_precomp,
        };
    }

    pub fn deinit(self: *Clipmap) void {
        const vkd = self.vkd;
        const device = self.device;
        vkd.destroyPipeline(device, self.compute_pipeline, null);
        vkd.destroyPipelineLayout(device, self.compute_pipeline_layout, null);
        vkd.destroyPipeline(device, self.wireframe_pipeline_debug, null);
        vkd.destroyPipeline(device, self.graphics_pipeline_debug, null);
        vkd.destroyPipeline(device, self.wireframe_pipeline, null);
        vkd.destroyPipeline(device, self.graphics_pipeline, null);
        vkd.destroyPipelineLayout(device, self.graphics_pipeline_layout, null);
        for (0..2) |i| {
            renderer_mod.destroyMappedBuffer(vkd, device, self.ubo_bufs[i]);
            renderer_mod.destroyMappedBuffer(vkd, device, self.draw_entry_bufs[i]);
        }
        vkd.destroyDescriptorPool(device, self.desc_pool, null);
        vkd.destroyBuffer(device, self.ring_buf.buffer, null);
        vkd.freeMemory(device, self.ring_buf.memory, null);
        vkd.destroyBuffer(device, self.chunk_index_buf.buffer, null);
        vkd.freeMemory(device, self.chunk_index_buf.memory, null);
        if (self.procedural_ssbo) |buf| {
            vkd.destroyBuffer(device, buf.buffer, null);
            vkd.freeMemory(device, buf.memory, null);
        }
        if (self.procedural_dir_grid) |buf| {
            vkd.destroyBuffer(device, buf.buffer, null);
            vkd.freeMemory(device, buf.memory, null);
        }
    }

    pub fn recreateGraphicsPipeline(self: *Clipmap, render_pass: vk.RenderPass, samples: vk.SampleCountFlags) !void {
        // Build new set first so if any pipeline create fails the old set
        // stays valid for the caller to retry.
        const set = try setup.recreateGraphicsPipeline(
            self.vkd, self.device, render_pass, self.graphics_pipeline_layout, samples,
            self.ring_size, self.chunk_vertex_dim,
        );
        self.vkd.destroyPipeline(self.device, self.wireframe_pipeline_debug, null);
        self.vkd.destroyPipeline(self.device, self.graphics_pipeline_debug, null);
        self.vkd.destroyPipeline(self.device, self.wireframe_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        self.graphics_pipeline = set.default.filled;
        self.wireframe_pipeline = set.default.wireframe;
        self.graphics_pipeline_debug = set.debug.filled;
        self.wireframe_pipeline_debug = set.debug.wireframe;
    }

    /// Fog max-distance for the current camera latitude. Clip is in arcsec;
    /// fog is computed in ground space against `length(view_pos)`. We scale
    /// r_arcsec by the camera's cos_lat: exact along X, slightly over-fogs
    /// off-axis (fog saturates before the clip cuts). Acceptable trade-off;
    /// alternative is plumbing arcsec_dist through varyings.
    pub fn currentFogMaxDist(self: *const Clipmap, cam_z: f32) f32 {
        const cos_lat = coords.cosLatFromZ(cam_z);
        const g_outer = self.levels[self.num_levels - 1].grid_spacing;
        const r_arcsec = @sqrt(cull.rSafeSq(self.ring_size, g_outer));
        return r_arcsec * cos_lat;
    }

    /// Total GPU buffer memory allocated (ring + chunk index + procedural
    /// buffers). Streaming SSBO is owned by TileSystem; caller queries separately.
    pub fn vramUsageMB(self: *const Clipmap) f32 {
        const ring: u64 = self.ring_buf_size;
        const chunk_idx: u64 = @as(u64, self.chunk_index_count) * @sizeOf(u32);
        const proc: u64 = if (self.procedural_ssbo) |_| @as(u64, tile_ssbo.SSBO_HDR_SIZE) * @sizeOf(u32) else 0;
        const proc_dir: u64 = if (self.procedural_dir_grid) |_| @as(u64, coords.GRID_CELL_COUNT) * @sizeOf(u32) else 0;
        return @as(f32, @floatFromInt(ring + chunk_idx + proc + proc_dir)) / (1024.0 * 1024.0);
    }

    /// Compute the finest LOD level worth rendering at the current altitude.
    /// Levels below this have subpixel grid cells and can be skipped entirely.
    pub fn minVisibleLevel(self: *const Clipmap, cam_altitude: f32, fov: f32, screen_height: f32) u32 {
        const altitude = @max(cam_altitude, 0.1);
        const pixel_size = altitude * 2.0 * @tan(fov / 2.0) / @max(screen_height, 1.0);
        if (pixel_size <= self.base_spacing) return 0;
        return @min(
            @as(u32, @intFromFloat(@log2(pixel_size / self.base_spacing))),
            self.num_levels - 1,
        );
    }

    /// Coarsest level worth computing/drawing: first level whose inner radius
    /// is still inside the camera's geometric horizon. Levels beyond this
    /// are entirely past the horizon and can be skipped.
    pub fn maxVisibleLevel(self: *const Clipmap, cam_altitude: f32, cam_z: f32) u32 {
        const d_horizon = coords.horizonDistArcsec(cam_altitude);
        // Ring boundaries are arcsec circles, but the shader's curvature drop
        // uses cos(lat)-scaled x. E/W ring segments have ground distance
        // r * cos(lat), so they cross the horizon later than N/S segments.
        // Cull only when even the closest (E/W) part exceeds the horizon.
        const cos_lat = coords.cosLatFromZ(cam_z);
        var max_level: u32 = self.num_levels - 1;
        while (max_level > 0) : (max_level -= 1) {
            if (self.r_inner_precomp[max_level] * cos_lat <= d_horizon) break;
        }
        return max_level;
    }

    /// Record compute commands to update dirty strips. Call BEFORE beginRenderPass.
    /// `total_frame` is a monotonic frame counter used by the tile system for
    /// staging-slot quarantine; pass main.zig's `total_frames`.
    pub fn recordUpdate(
        self: *Clipmap,
        cmd_buf: vk.CommandBuffer,
        camera_pos: math.Vec3d,
        frame_index: u32,
        total_frame: u64,
        fov: f32,
        screen_height: f32,
    ) void {
        // Ring buffer caches per-cell terrain evaluated against the SSBO at
        // write time; any SSBO mutation invalidates every cell's cached
        // value. Force a full refill via the `!initialized` path below.
        // TODO: this full refill of every level on ANY tile upload is the cause
        // of the framerate drop while tiles stream (one tile landing = whole
        // ring re-evaluated through the MLP that frame). A freshly loaded tile
        // only covers one lat/lon footprint, so only ring cells overlapping that
        // footprint need re-eval. Make this incremental (tile-footprint ->
        // affected cells per level) to smooth both startup and in-flight
        // streaming.
        const ssbo_changed = if (self.tile_system) |ts| ts.recordStream(cmd_buf, total_frame) else false;
        if (ssbo_changed) {
            for (self.levels[0..self.num_levels]) |*level| level.initialized = false;
        }

        // F4 freeze: substitute the captured camera position so scrolling
        // and LOD selection both lock. recordDraw still uses the live
        // camera for the view matrix (lets us fly out and inspect).
        const ref_pos: math.Vec3d = if (debug.state.freeze)
            debug.state.frozen_cam_pos
        else
            camera_pos;
        const cull_pos: math.Vec3 = .{
            @floatCast(ref_pos[0]),
            @floatCast(ref_pos[1]),
            @floatCast(ref_pos[2]),
        };

        // cam_anchor is the shader's reference origin for camera-relative
        // math: round to integer arcsec (exact in f32) so `tile_origin -
        // cam_anchor` and `snapped_pos - cam_anchor` are both exact integer
        // subtractions, eliminating the catastrophic cancellation that an
        // f32 absolute world coord would cause inside the compute shader at
        // large longitudes.
        const cam_anchor: [2]f32 = .{
            @floatFromInt(@as(i32, @intFromFloat(@round(ref_pos[0])))),
            @floatFromInt(@as(i32, @intFromFloat(@round(ref_pos[2])))),
        };
        const cull_fov: f32 = if (debug.state.freeze) debug.state.frozen_fov else fov;

        const min_level = self.minVisibleLevel(cull_pos[1], cull_fov, screen_height);
        const max_level = self.maxVisibleLevel(cull_pos[1], cull_pos[2]);
        const vkd = self.vkd;
        const rs_i32: i32 = @intCast(self.ring_size);
        // Pipeline + descriptor binds are deferred until the first dispatch:
        // on a stationary camera with nothing streaming, no level needs work
        // and the whole compute section records zero commands.
        var compute_bound = false;

        var total_evals: u32 = 0;
        self.last_strips = 0;
        self.last_refills = 0;

        // Update coarse-to-fine, skipping levels outside [min_level, max_level]
        var level_idx: i32 = @intCast(self.num_levels - 1);
        while (level_idx >= 0) : (level_idx -= 1) {
            const li: usize = @intCast(level_idx);
            if (li < min_level or li > max_level) continue;
            const level = &self.levels[li];
            const base_offset: u32 = @as(u32, @intCast(li)) * self.entries_per_level;

            const snapped_x = snapLevelOrigin(cull_pos[0], level.grid_spacing);
            const snapped_z = snapLevelOrigin(cull_pos[2], level.grid_spacing);

            // scroll_offset + level_origin + cam_anchor are populated by
            // updatePushConstants before each dispatch; the zero placeholders
            // here are just to satisfy the struct literal.
            var pc = ComputePushConstants{
                .ring_size = self.ring_size,
                .update_mode = 0,
                .strip_row = 0,
                .strip_col = 0,
                .scroll_offset = .{ 0, 0 },
                .level_origin = .{ 0, 0 },
                .grid_spacing = level.grid_spacing,
                .level_base_offset = base_offset,
                .chunk_origin = .{ 0, 0 },
                .chunk_vertex_dim = self.chunk_vertex_dim,
                .cam_anchor = .{ 0, 0 },
            };

            if (!level.initialized) {
                level.snapped_pos = .{ snapped_x, snapped_z };
                level.scroll_offset = .{ 0, 0 };
                level.initialized = true;

                self.bindComputeOnce(cmd_buf, frame_index, &compute_bound);
                updatePushConstants(&pc, level, rs_i32, cam_anchor);
                self.dispatchChunkFill(cmd_buf, &pc);
                total_evals += self.entries_per_level;
                continue;
            }

            const full_dx: i32 = computeCellDelta(snapped_x, level.snapped_pos[0], level.grid_spacing);
            const full_dz: i32 = computeCellDelta(snapped_z, level.snapped_pos[1], level.grid_spacing);

            if (full_dx == 0 and full_dz == 0) continue;

            const abs_dx: u32 = @intCast(@abs(full_dx));
            const abs_dz: u32 = @intCast(@abs(full_dz));

            // Large jump (teleport): full refill with immediate state update.
            if (shouldRefillLevel(abs_dx, abs_dz, self.ring_size)) {
                level.snapped_pos = .{ snapped_x, snapped_z };
                level.scroll_offset[0] += full_dx;
                level.scroll_offset[1] += full_dz;

                self.bindComputeOnce(cmd_buf, frame_index, &compute_bound);
                updatePushConstants(&pc, level, rs_i32, cam_anchor);
                self.dispatchChunkFill(cmd_buf, &pc);
                total_evals += self.entries_per_level;
                self.last_refills += 1;
                continue;
            }

            // Incremental strip updates: advance one cell at a time so each
            // dispatch writes to the correct toroidal position.
            // Compute->compute barriers between strip dispatches are required:
            // consecutive column strips write to adjacent ring buffer entries
            // (16 bytes apart = same GPU cache line), and concurrent dispatches
            // can stomp each other's cache lines, losing writes.
            const sx = stripSignAndCount(full_dx);
            const sz = stripSignAndCount(full_dz);
            var need_strip_barrier = false;

            // Reached only when full_dx or full_dz is nonzero, so at least one
            // strip dispatch follows.
            self.bindComputeOnce(cmd_buf, frame_index, &compute_bound);
            self.dispatchStrips(cmd_buf, level, &pc, 0, sx.sign, sx.count, rs_i32, cam_anchor, &need_strip_barrier, &total_evals);
            self.dispatchStrips(cmd_buf, level, &pc, 1, sz.sign, sz.count, rs_i32, cam_anchor, &need_strip_barrier, &total_evals);

            // Snap to computed value to avoid floating-point drift from accumulation
            level.snapped_pos = .{ snapped_x, snapped_z };
        }

        // Barrier: compute writes -> vertex reads
        if (total_evals > 0) {
            const barrier = [1]vk.MemoryBarrier{.{
                .src_access_mask = .{ .shader_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
            }};
            vkd.cmdPipelineBarrier(
                cmd_buf,
                .{ .compute_shader_bit = true },
                .{ .vertex_shader_bit = true },
                .{},
                &barrier,
                null,
                null,
            );
        }

        self.last_evals = total_evals;
    }

    /// Bind the compute pipeline + descriptor set on the first dispatch of the
    /// frame; no-op afterwards. Keeps idle frames free of compute commands.
    fn bindComputeOnce(self: *const Clipmap, cmd_buf: vk.CommandBuffer, frame_index: u32, bound: *bool) void {
        if (bound.*) return;
        bound.* = true;
        const vkd = self.vkd;
        vkd.cmdBindPipeline(cmd_buf, .compute, self.compute_pipeline);
        const compute_desc = [1]vk.DescriptorSet{self.desc_sets[frame_index]};
        vkd.cmdBindDescriptorSets(cmd_buf, .compute, self.compute_pipeline_layout, 0, &compute_desc, null);
    }

    /// Issue one compute dispatch per chunk in this level. Compute always
    /// writes every cell so the ring buffer stays fully defined regardless of
    /// what the per-frame graphics cull decides to draw; culling at compute
    /// time would leave stale cells when the camera moves within a snap
    /// interval.
    ///
    /// No inter-chunk barrier; chunks write to disjoint cell ranges. The
    /// terminating compute->vertex barrier in `recordUpdate` covers the batch.
    fn dispatchChunkFill(
        self: *const Clipmap,
        cmd_buf: vk.CommandBuffer,
        pc: *ComputePushConstants,
    ) void {
        const vkd = self.vkd;
        const dispatch_count: u32 = (self.chunk_vertex_dim * self.chunk_vertex_dim + 255) / 256;
        var cy: u32 = 0;
        while (cy < self.chunks_per_side) : (cy += 1) {
            var cx: u32 = 0;
            while (cx < self.chunks_per_side) : (cx += 1) {
                pc.chunk_origin = .{ cx * self.chunk_cells, cy * self.chunk_cells };
                vkd.cmdPushConstants(cmd_buf, self.compute_pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(ComputePushConstants), @ptrCast(pc));
                vkd.cmdDispatch(cmd_buf, dispatch_count, 1, 1);
            }
        }
    }

    fn dispatchStrips(
        self: *Clipmap,
        cmd_buf: vk.CommandBuffer,
        level: *ClipmapLevel,
        pc: *ComputePushConstants,
        axis: u32,
        sign: i32,
        count: u32,
        rs_i32: i32,
        cam_anchor: [2]f32,
        need_strip_barrier: *bool,
        total_evals: *u32,
    ) void {
        const vkd = self.vkd;
        for (0..count) |_| {
            if (need_strip_barrier.*) computeBarrier(vkd, cmd_buf);
            level.scroll_offset[axis] += sign;
            level.snapped_pos[axis] += @as(f32, @floatFromInt(sign)) * level.grid_spacing;

            if (axis == 0) {
                pc.update_mode = 2;
                pc.strip_col = toroidalStripIndex(sign, self.ring_size);
            } else {
                pc.update_mode = 1;
                pc.strip_row = toroidalStripIndex(sign, self.ring_size);
                pc.strip_col = 0;
            }
            updatePushConstants(pc, level, rs_i32, cam_anchor);
            vkd.cmdPushConstants(cmd_buf, self.compute_pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(ComputePushConstants), @ptrCast(pc));
            vkd.cmdDispatch(cmd_buf, (self.ring_size + 255) / 256, 1, 1);
            total_evals.* += self.ring_size;
            self.last_strips += 1;
            need_strip_barrier.* = true;
        }
    }

    /// Record graphics draw commands. Call INSIDE the render pass. Finest to
    /// coarsest order is encoded by the order entries are packed into the
    /// SSBO; the depth test culls coarser overlap.
    pub fn recordDraw(
        self: *const Clipmap,
        cmd_buf: vk.CommandBuffer,
        params: SceneParams,
        frame_index: u32,
        fov: f32,
        screen_height: f32,
    ) void {
        // Freeze pins LOD selection and chunk cull to the captured camera so
        // moving the live camera doesn't pop chunks in/out. The view matrix
        // stays live (params.cam_pos); that's how fly-out inspection works.
        const cull_pos: [3]f32 = if (debug.state.freeze) .{
            @floatCast(debug.state.frozen_cam_pos[0]),
            @floatCast(debug.state.frozen_cam_pos[1]),
            @floatCast(debug.state.frozen_cam_pos[2]),
        } else .{
            @floatCast(params.cam_pos[0]),
            @floatCast(params.cam_pos[1]),
            @floatCast(params.cam_pos[2]),
        };
        const cull_fov: f32 = if (debug.state.freeze) debug.state.frozen_fov else fov;
        const min_level = self.minVisibleLevel(cull_pos[1], cull_fov, screen_height);
        const max_level = self.maxVisibleLevel(cull_pos[1], cull_pos[2]);

        const vkd = self.vkd;
        const sun = math.normalize(params.sun_dir);
        const ubo = SceneUBO{
            .view = params.view,
            .proj = params.proj,
            .proj_view = math.matMul(params.proj, params.view),
            .sun_dir = .{ sun[0], sun[1], sun[2], 0.0 },
            .fog_max_dist = params.fog_max_dist,
            .no_effects = if (params.no_effects) 1 else 0,
            .transfer_function = params.transfer_function,
            .debug_overlay = @intFromEnum(debug.state.color_overlay),
            .cam_elev = @floatCast(params.cam_pos[1]),
            .cam_z = @floatCast(params.cam_pos[2]),
            .aircraft_msl_m = params.aircraft_msl_m,
            .taws = @intFromBool(params.taws),
        };
        // hypso_lut (constant tail) was written at init; copy only the dynamic prefix.
        const dyn_bytes = @offsetOf(SceneUBO, "hypso_lut");
        const ubo_bytes: *const [@sizeOf(SceneUBO)]u8 = @ptrCast(&ubo);
        @memcpy(self.ubo_maps[frame_index][0..dyn_bytes], ubo_bytes[0..dyn_bytes]);

        // Host-coherent: writes visible to the GPU at submit time.
        const entries_ptr: [*]DrawEntry = @ptrCast(@alignCast(self.draw_entry_maps[frame_index]));
        const num_draws = self.buildDrawEntries(params, cull_pos, min_level, max_level, entries_ptr);
        if (num_draws == 0) return;

        const gfx_desc = [1]vk.DescriptorSet{self.desc_sets[frame_index]};
        vkd.cmdBindDescriptorSets(cmd_buf, .graphics, self.graphics_pipeline_layout, 0, &gfx_desc, null);
        vkd.cmdBindIndexBuffer(cmd_buf, self.chunk_index_buf.buffer, 0, .uint32);

        // wireframe_overlay does 2 passes (filled, then wireframe on top).
        const passes: u32 = if (debug.state.render_mode == .wireframe_overlay) 2 else 1;
        const use_debug = debug.state.color_overlay != .off;
        var pass: u32 = 0;
        while (pass < passes) : (pass += 1) {
            const use_wireframe = switch (debug.state.render_mode) {
                .normal => false,
                .wireframe => true,
                .wireframe_overlay => pass == 1,
            };
            const pipe = if (use_wireframe)
                (if (use_debug) self.wireframe_pipeline_debug else self.wireframe_pipeline)
            else
                (if (use_debug) self.graphics_pipeline_debug else self.graphics_pipeline);
            vkd.cmdBindPipeline(cmd_buf, .graphics, pipe);
            vkd.cmdDrawIndexed(cmd_buf, self.chunk_index_count, num_draws, 0, 0, 0);
        }
    }

    /// `by_cull_state` overlay forces every chunk to draw with a tint bit
    /// instead of being skipped.
    fn buildDrawEntries(
        self: *const Clipmap,
        params: SceneParams,
        cull_pos: [3]f32,
        min_level: u32,
        max_level: u32,
        entries: [*]DrawEntry,
    ) u32 {
        const rs_i32: i32 = @intCast(self.ring_size);

        // Frustum cull: when freeze is on, build the wedge from the captured
        // view/proj so flying out doesn't change the cull set.
        const cull_view = if (debug.state.freeze) debug.state.frozen_view else params.view;
        const cull_proj = if (debug.state.freeze) debug.state.frozen_proj else params.proj;
        const cull_st = CullState.init(self, cull_pos, cull_view, cull_proj, min_level);

        const draw_all = debug.state.color_overlay == .by_cull_state;
        const catalog = if (self.tile_system) |ts| &ts.catalog else null;

        var count: u32 = 0;
        for (min_level..max_level + 1) |li| {
            const level = &self.levels[li];
            const base_offset: u32 = @as(u32, @intCast(li)) * self.entries_per_level;

            // Camera-relative origins computed in f64 to avoid catastrophic
            // cancellation, then cast to f32 (results within clipmap extent).
            const origin_rel_x: f32 = @floatCast(@as(f64, level.snapped_pos[0]) - params.cam_pos[0]);
            const origin_rel_z: f32 = @floatCast(@as(f64, level.snapped_pos[1]) - params.cam_pos[2]);
            const scroll_x: u32 = @intCast(@mod(level.scroll_offset[0], rs_i32));
            const scroll_y: u32 = @intCast(@mod(level.scroll_offset[1], rs_i32));
            const r_inner = cull_st.r_inner[li];
            const r_outer = cull_st.r_outer[li];

            var cy: u32 = 0;
            while (cy < self.chunks_per_side) : (cy += 1) {
                var cx: u32 = 0;
                while (cx < self.chunks_per_side) : (cx += 1) {
                    const aabb = cull.chunkAabb(self.ring_size, self.chunk_cells, cx, cy, level.grid_spacing);
                    var cs = chunkCullState(self, @intCast(li), aabb, &cull_st, draw_all);
                    if (cs != 0 and !draw_all) continue;

                    if (catalog) |cat| {
                        if (cull.chunkIsOcean(aabb, level.snapped_pos, cat))
                            cs |= CULL_OCEAN;
                    }

                    entries[count] = .{
                        .level_origin = .{ origin_rel_x, origin_rel_z },
                        .grid_spacing = level.grid_spacing,
                        .r_inner = r_inner,
                        .r_outer = r_outer,
                        .level_base_offset = base_offset,
                        .level_idx = @intCast(li),
                        .cull_state = cs,
                        .scroll_offset = .{ scroll_x, scroll_y },
                        .chunk_origin = .{ cx * self.chunk_cells, cy * self.chunk_cells },
                    };
                    count += 1;
                }
            }
        }
        return count;
    }
};

/// Per-frame, per-level cull state. Each level's visible boundary is a
/// circle in arcsec at radius `(half - 3) * g_L`. Comparing in arcsec
/// (not cos_lat-scaled ground) makes adjacent levels reference the same
/// world curve regardless of per-vertex cos_lat drift across the level.
///
/// Two flavors of bounds per level:
/// - `r_*_keep_sq`: slop-adjusted, squared, for the per-chunk cull predicate.
/// - `r_inner` / `r_outer`: true arcsec radii, pushed to the vertex shader
///   for `gl_ClipDistance`. The finest visible level has `r_inner = 0`.
///
/// `wedge` is the camera-frustum XZ projection. Same wedge for every level
/// (the cull is angular, not radial); per-chunk SAT is in `chunkCullState`.
/// The shader's `x *= cos_lat` is baked into the wedge so per-chunk testing
/// can use raw arcsec corners.
const CullState = struct {
    cam_xz: [2]f32,
    r_outer_keep_sq: *const [MAX_LEVELS]f32,
    r_inner_keep_sq: [MAX_LEVELS]f32,
    r_outer: *const [MAX_LEVELS]f32,
    r_inner: [MAX_LEVELS]f32,
    /// Per-level `snapped_pos - cam_xz` so the per-chunk frustum test
    /// doesn't recompute (level - cam) for every chunk in the level.
    dx_origin: [MAX_LEVELS]f32,
    dz_origin: [MAX_LEVELS]f32,
    wedge: cull.Wedge,

    fn init(clip: *const Clipmap, cam_pos: [3]f32, view: math.Mat4, proj: math.Mat4, min_level: u32) CullState {
        var st: CullState = .{
            .cam_xz = .{ cam_pos[0], cam_pos[2] },
            .r_outer_keep_sq = &clip.r_outer_keep_sq_precomp,
            .r_inner_keep_sq = clip.r_inner_keep_sq_precomp,
            .r_outer = &clip.r_outer_precomp,
            .r_inner = clip.r_inner_precomp,
            .dx_origin = undefined,
            .dz_origin = undefined,
            .wedge = cull.frustumWedge(view, proj, coords.cosLatFromZ(cam_pos[2])),
        };

        const num: usize = clip.num_levels;
        for (0..num) |i| {
            if (i <= min_level) {
                st.r_inner[i] = 0;
                st.r_inner_keep_sq[i] = 0;
            }
            const lvl = &clip.levels[i];
            st.dx_origin[i] = lvl.snapped_pos[0] - cam_pos[0];
            st.dz_origin[i] = lvl.snapped_pos[1] - cam_pos[2];
        }
        return st;
    }
};

/// Bitmask of cull reasons. 0 = visible. Used by the `by_cull_state` debug
/// overlay to tint chunks; non-zero values cause the draw to skip in the
/// hot path unless the overlay is on (then we draw all and let the shader
/// dim culled chunks).
const CULL_RADIAL: u32 = 1;
const CULL_FRUSTUM: u32 = 2;
const CULL_OCEAN: u32 = 4;

/// Per-chunk visibility evaluation. Returns 0 for visible, or a bitmask of
/// the reasons the chunk would be culled. With `all_reasons=false` short-
/// circuits on the first failed test (frustum work skipped when radial
/// already culls). The overlay path passes true to get all bits set.
fn chunkCullState(
    self: *const Clipmap,
    level: u32,
    aabb: cull.ChunkAabb,
    st: *const CullState,
    all_reasons: bool,
) u32 {
    const lvl = &self.levels[level];
    var s: u32 = 0;
    if (!cull.chunkVisibleRadial(
        aabb,
        lvl.snapped_pos,
        st.cam_xz,
        st.r_outer_keep_sq[level],
        st.r_inner_keep_sq[level],
    )) {
        s |= CULL_RADIAL;
        if (!all_reasons) return s;
    }

    if (!cull.chunkVisibleFrustum(aabb, st.dx_origin[level], st.dz_origin[level], st.wedge)) s |= CULL_FRUSTUM;
    return s;
}

const PrecomputedRadii = struct {
    r_outer: [MAX_LEVELS]f32,
    r_inner: [MAX_LEVELS]f32,
    r_outer_keep_sq: [MAX_LEVELS]f32,
    r_inner_keep_sq: [MAX_LEVELS]f32,
};

fn precomputeCullRadii(num_levels: u32, base_spacing: f32, ring_size: u32, chunk_cells: u32) PrecomputedRadii {
    var out: PrecomputedRadii = .{
        .r_outer = undefined,
        .r_inner = undefined,
        .r_outer_keep_sq = undefined,
        .r_inner_keep_sq = undefined,
    };
    for (0..num_levels) |i| {
        const g_self = base_spacing * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(i)));
        const r_outer_arcsec = @sqrt(cull.rSafeSq(ring_size, g_self));
        out.r_outer[i] = r_outer_arcsec;

        const g_prev = base_spacing * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(if (i == 0) 0 else i - 1)));
        const r_inner_arcsec: f32 = if (i == 0) 0.0 else @sqrt(cull.rSafeSq(ring_size, g_prev));
        out.r_inner[i] = r_inner_arcsec;

        const half_diag = cull.chunkHalfDiag(chunk_cells, g_self);
        const r_outer_keep = r_outer_arcsec + half_diag;
        out.r_outer_keep_sq[i] = r_outer_keep * r_outer_keep;
        if (r_inner_arcsec > half_diag) {
            const r_inner_keep = r_inner_arcsec - half_diag;
            out.r_inner_keep_sq[i] = r_inner_keep * r_inner_keep;
        } else {
            out.r_inner_keep_sq[i] = 0;
        }
    }
    return out;
}

fn snapLevelOrigin(pos: f32, spacing: f32) f32 {
    const snap_interval = spacing * 2.0;
    return @floor(pos / snap_interval) * snap_interval;
}

fn computeCellDelta(snapped_new: f32, snapped_old: f32, spacing: f32) i32 {
    return @intFromFloat(@round((snapped_new - snapped_old) / spacing));
}

const MAX_STRIP_CELLS: u32 = 8;

fn shouldRefillLevel(abs_dx: u32, abs_dz: u32, ring_size: u32) bool {
    return abs_dx >= ring_size / 2 or abs_dz >= ring_size / 2 or
        abs_dx > MAX_STRIP_CELLS or abs_dz > MAX_STRIP_CELLS;
}

fn stripSignAndCount(delta: i32) struct { sign: i32, count: u32 } {
    const sign: i32 = if (delta > 0) 1 else if (delta < 0) -1 else 0;
    return .{ .sign = sign, .count = @intCast(@abs(delta)) };
}

fn toroidalStripIndex(sign: i32, ring_size: u32) i32 {
    return if (sign > 0) @as(i32, @intCast(ring_size)) - 1 else 0;
}

fn updatePushConstants(
    pc: *ComputePushConstants,
    level: *const ClipmapLevel,
    rs_i32: i32,
    cam_anchor: [2]f32,
) void {
    pc.scroll_offset = .{
        @intCast(@mod(level.scroll_offset[0], rs_i32)),
        @intCast(@mod(level.scroll_offset[1], rs_i32)),
    };
    // snapped_pos is a multiple of (2 * grid_spacing) and cam_anchor is an
    // integer arcsec; both are well below 2^24 so the f32 subtraction is
    // exact. The shader reconstructs absolute world coords as
    // (level_origin + offset) + cam_anchor where needed.
    pc.level_origin = .{
        level.snapped_pos[0] - cam_anchor[0],
        level.snapped_pos[1] - cam_anchor[1],
    };
    pc.cam_anchor = cam_anchor;
}

/// Compute->compute memory barrier for strip dispatch serialization.
fn computeBarrier(vkd: vk.DeviceWrapper, cmd_buf: vk.CommandBuffer) void {
    const barrier = [1]vk.MemoryBarrier{.{
        .src_access_mask = .{ .shader_write_bit = true },
        .dst_access_mask = .{ .shader_write_bit = true },
    }};
    vkd.cmdPipelineBarrier(
        cmd_buf,
        .{ .compute_shader_bit = true },
        .{ .compute_shader_bit = true },
        .{},
        &barrier,
        null,
        null,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const RS: u32 = 257;
const CC: u32 = 32;

test "scroll_offset @mod wraps negative values into [0, ring_size)" {
    const rs_i32: i32 = @intCast(RS);
    try testing.expectEqual(@as(i32, 252), @mod(@as(i32, -5), rs_i32));
    try testing.expectEqual(@as(i32, 247), @mod(@as(i32, -10), rs_i32));
    try testing.expectEqual(@as(i32, 0), @mod(-rs_i32, rs_i32));
    try testing.expectEqual(@as(i32, 0), @mod(@as(i32, 0), rs_i32));
    try testing.expectEqual(@as(i32, 1), @mod(@as(i32, -(rs_i32 - 1)), rs_i32));
    try testing.expectEqual(@as(i32, 256), @mod(@as(i32, -1), rs_i32));
}

test "ringLayout rounds to chunk-aligned odd ring sizes" {
    try testing.expectEqual(@as(u32, 1001), Clipmap.ringLayout(1001).ring_size);
    try testing.expectEqual(@as(u32, 505), Clipmap.ringLayout(501).ring_size);
    try testing.expectEqual(@as(u32, 257), Clipmap.ringLayout(255).ring_size);
    try testing.expectEqual(@as(u32, 63), Clipmap.ringLayout(63).ring_size);
    // Chunk tiling is exact and the ring stays odd for any in-range request.
    const L = Clipmap.ringLayout(900);
    try testing.expectEqual(L.ring_size, L.chunk_cells * L.chunks_per_side + 1);
    try testing.expectEqual(@as(u32, 1), L.ring_size % 2);
    try testing.expectEqual(L.chunk_cells + 1, L.chunk_vertex_dim);
}

test "snapLevelOrigin floors to 2x-spacing intervals" {
    try testing.expectApproxEqAbs(@as(f32, 105.0), snapLevelOrigin(105.7, 10.5), 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 42.0), snapLevelOrigin(42.0, 10.5), 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -2.0), snapLevelOrigin(-1.0, 1.0), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), snapLevelOrigin(0.7, 0.25), 1e-5);
}

test "computeCellDelta rounds position change to integer cells" {
    try testing.expectEqual(@as(i32, 3), computeCellDelta(31.5, 0.0, 10.5));
    try testing.expectEqual(@as(i32, 0), computeCellDelta(4.5, 0.0, 10.0));
    try testing.expectEqual(@as(i32, 0), computeCellDelta(0.0, 4.999, 10.0));
    try testing.expectEqual(@as(i32, -3), computeCellDelta(0.0, 31.5, 10.5));
    try testing.expectEqual(@as(i32, 200), computeCellDelta(2000.0, 0.0, 10.0));
    try testing.expectEqual(@as(i32, 1), computeCellDelta(5.0, 0.0, 10.0));
}

test "shouldRefillLevel triggers on large jump or > MAX_STRIP_CELLS" {
    try testing.expect(shouldRefillLevel(129, 0, RS));
    try testing.expect(shouldRefillLevel(0, 200, RS));
    try testing.expect(shouldRefillLevel(128, 0, RS));
    try testing.expect(shouldRefillLevel(9, 0, RS));
    try testing.expect(shouldRefillLevel(0, 9, RS));
    try testing.expect(!shouldRefillLevel(8, 8, RS));
    try testing.expect(!shouldRefillLevel(7, 7, RS));
    try testing.expect(!shouldRefillLevel(0, 0, RS));
}

test "stripSignAndCount splits delta into sign and magnitude" {
    const a = stripSignAndCount(5);
    try testing.expectEqual(@as(i32, 1), a.sign);
    try testing.expectEqual(@as(u32, 5), a.count);

    const b = stripSignAndCount(-3);
    try testing.expectEqual(@as(i32, -1), b.sign);
    try testing.expectEqual(@as(u32, 3), b.count);

    const c = stripSignAndCount(0);
    try testing.expectEqual(@as(i32, 0), c.sign);
    try testing.expectEqual(@as(u32, 0), c.count);
}

test "toroidalStripIndex selects boundary row/col by sign" {
    try testing.expectEqual(@as(i32, 256), toroidalStripIndex(1, RS));
    try testing.expectEqual(@as(i32, 0), toroidalStripIndex(-1, RS));
    // sign=0 unreachable in production (count=0 skips loop); returns 0.
    try testing.expectEqual(@as(i32, 0), toroidalStripIndex(0, RS));
}

test "precomputeCullRadii matches closed-form rSafeSq / chunkHalfDiag" {
    const num_levels: u32 = 4;
    const base_spacing: f32 = 1.0;
    const r = precomputeCullRadii(num_levels, base_spacing, RS, CC);

    const r_outer_0 = @sqrt(cull.rSafeSq(RS, 1.0));
    try testing.expectApproxEqAbs(r_outer_0, r.r_outer[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), r.r_inner[0], 1e-6);

    const r_outer_2 = @sqrt(cull.rSafeSq(RS, 4.0));
    const r_inner_2 = @sqrt(cull.rSafeSq(RS, 2.0));
    try testing.expectApproxEqAbs(r_outer_2, r.r_outer[2], 1e-4);
    try testing.expectApproxEqAbs(r_inner_2, r.r_inner[2], 1e-4);

    const hd_2 = cull.chunkHalfDiag(CC, 4.0);
    const expected_outer_keep = (r_outer_2 + hd_2) * (r_outer_2 + hd_2);
    try testing.expectApproxEqAbs(expected_outer_keep, r.r_outer_keep_sq[2], 1.0);

    const expected_inner_keep = blk: {
        if (r_inner_2 > hd_2) {
            const diff = r_inner_2 - hd_2;
            break :blk diff * diff;
        }
        break :blk @as(f32, 0);
    };
    try testing.expectApproxEqAbs(expected_inner_keep, r.r_inner_keep_sq[2], 1.0);
}

test "CullState.init zeroes r_inner through min_level inclusive" {
    // Synthetic Clipmap: CullState.init reads only num_levels, r_*_precomp,
    // and levels[*].snapped_pos; Vulkan handles stay undefined.
    var clip: Clipmap = undefined;
    clip.num_levels = 5;
    var levels: [MAX_LEVELS]ClipmapLevel = undefined;
    for (0..clip.num_levels) |i| {
        const g = 1.0 * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(i)));
        levels[i] = .{ .grid_spacing = g, .snapped_pos = .{ 0, 0 } };
    }
    clip.levels = levels;
    const radii = precomputeCullRadii(clip.num_levels, 1.0, RS, CC);
    clip.r_outer_precomp = radii.r_outer;
    clip.r_inner_precomp = radii.r_inner;
    clip.r_outer_keep_sq_precomp = radii.r_outer_keep_sq;
    clip.r_inner_keep_sq_precomp = radii.r_inner_keep_sq;

    const view: math.Mat4 = .{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    };
    const proj: math.Mat4 = .{
        .{ 1, 0, 0, 0 },
        .{ 0, -1, 0, 0 },
        .{ 0, 0, 0, -1 },
        .{ 0, 0, 0, 0 },
    };

    const min_level: u32 = 2;
    const st = CullState.init(&clip, .{ 0, 100, 0 }, view, proj, min_level);

    try testing.expectEqual(@as(f32, 0), st.r_inner[0]);
    try testing.expectEqual(@as(f32, 0), st.r_inner[1]);
    try testing.expectEqual(@as(f32, 0), st.r_inner[2]);
    try testing.expectEqual(@as(f32, 0), st.r_inner_keep_sq[0]);
    try testing.expectEqual(@as(f32, 0), st.r_inner_keep_sq[1]);
    try testing.expectEqual(@as(f32, 0), st.r_inner_keep_sq[2]);
    try testing.expectEqual(clip.r_inner_precomp[3], st.r_inner[3]);
    try testing.expectEqual(clip.r_inner_keep_sq_precomp[3], st.r_inner_keep_sq[3]);
}

test "hypsoColor reproduces palette knots; LUT covers domain exactly" {
    // Every knot must round-trip exactly (knots sit on the 100m LUT grid).
    for (HYPSO_KNOTS) |k| {
        const c = hypsoColor(k.elev_m);
        try testing.expectEqual(k.color[0], c[0]);
        try testing.expectEqual(k.color[1], c[1]);
        try testing.expectEqual(k.color[2], c[2]);
    }
    // Clamping above the top knot and below the bottom knot.
    try testing.expectEqual(hypsoColor(8500.0), hypsoColor(20000.0));
    try testing.expectEqual(hypsoColor(-100.0), hypsoColor(-5000.0));
    // Midpoint interpolation: 1850m is halfway through the 1500..2200 band.
    const mid = hypsoColor(1850.0);
    try testing.expectApproxEqAbs(@as(f32, (0.26 + 0.40) / 2.0), mid[0], 1e-6);
    // LUT entry i corresponds to elevation -100 + i*100.
    try testing.expectEqual(@as(usize, HYPSO_LUT_LEN), HYPSO_LUT.len);
    const e23 = HYPSO_LUT[23]; // 2200m
    try testing.expectEqual(@as(f32, 0.40), e23[0]);
    try testing.expectEqual(@as(f32, 0.34), e23[1]);
    try testing.expectEqual(@as(f32, 0.24), e23[2]);
}
