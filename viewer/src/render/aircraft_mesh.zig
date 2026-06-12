//! Placeholder aircraft mesh: a colored flat swept delta wing drawn from a
//! pre-built `DrawParams`. The render module is intentionally decoupled from
//! `app/aircraft.zig` and `app/camera.zig`; callers (today: `scene.zig`)
//! extract the primitives and pass them in. Future phases swap the
//! delta wing for a real plane model.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const renderer_mod = @import("renderer.zig");
const pipeline = @import("pipeline.zig");
const math = @import("math");

const aircraft_vert_spv align(@alignOf(u32)) = @embedFile("aircraft_vert").*;
const aircraft_frag_spv align(@alignOf(u32)) = @embedFile("aircraft_frag").*;

/// Mesh "unit" radius in arcseconds. Vertex coordinates are unit-length;
/// this scales them. Equivalent to ~15 m of physical wingspan via the world's
/// HEIGHT_SCALE = 60/1852 (defined in terrain/coords.zig). Inlined here to
/// avoid a cross-tree import; keep in sync if HEIGHT_SCALE changes.
const MESH_RADIUS_ARCSEC: f64 = 15.0 * 60.0 / 1852.0;

/// Primitive inputs for `AircraftMesh.draw`. Caller (scene) builds these from
/// whatever app-layer state it owns; this module never sees Aircraft or Camera.
pub const DrawParams = struct {
    /// Rotation-only view matrix (eye at origin) in f64.
    view_rot: math.Mat4d,
    /// Projection matrix in f64.
    proj: math.Mat4d,
    /// Aircraft world position minus camera world position, in arcsec (f64).
    /// f64 to avoid catastrophic cancellation at large world coordinates.
    pos_rel_cam: math.Vec3d,
    /// Aircraft attitude.
    orientation: math.Quat,
    /// cos(latitude) for horizontal-axis scaling so the mesh aligns with the
    /// terrain shader's cos(lat)-corrected vertex pipeline.
    cos_lat: f64,
};

const PushConstants = extern struct {
    mvp: [4][4]f32,
};

pub const AircraftMesh = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,

    pub fn init(
        ctx: renderer_mod.GpuContext,
        render_pass: vk.RenderPass,
        samples: vk.SampleCountFlags,
    ) !AircraftMesh {
        const vkd = ctx.vkd;
        const device = ctx.device;
        const push_range = vk.PushConstantRange{
            .stage_flags = .{ .vertex_bit = true },
            .offset = 0,
            .size = @sizeOf(PushConstants),
        };
        const pipeline_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 0,
            .p_set_layouts = undefined,
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, pipeline_layout, null);

        const pipe = try createPipeline(&vkd, device, pipeline_layout, render_pass, samples);

        return .{
            .vkd = vkd,
            .device = device,
            .pipeline = pipe,
            .pipeline_layout = pipeline_layout,
        };
    }

    pub fn recreatePipeline(self: *AircraftMesh, render_pass: vk.RenderPass, samples: vk.SampleCountFlags) !void {
        self.vkd.destroyPipeline(self.device, self.pipeline, null);
        self.pipeline = try createPipeline(&self.vkd, self.device, self.pipeline_layout, render_pass, samples);
    }

    pub fn deinit(self: *AircraftMesh) void {
        self.vkd.destroyPipeline(self.device, self.pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
    }

    pub fn draw(self: *const AircraftMesh, cmd: vk.CommandBuffer, params: DrawParams) void {
        const pc = PushConstants{ .mvp = composeMvp(params) };
        self.vkd.cmdBindPipeline(cmd, .graphics, self.pipeline);
        self.vkd.cmdPushConstants(
            cmd,
            self.pipeline_layout,
            .{ .vertex_bit = true },
            0,
            @sizeOf(PushConstants),
            @ptrCast(&pc),
        );
        self.vkd.cmdDraw(cmd, 12, 1, 0, 0);
    }
};

/// Compose proj * view_rot * model in f64, truncate to f32. The mesh is a rigid
/// body: it is scaled uniformly, and cos(lat) is applied ONLY to the world-X
/// position (the translation column), matching the terrain shader's `x *=
/// cos_lat`. The old code also compressed the mesh geometry by cos(lat) (a
/// scale_h on world X), which squished the model heading- and latitude-
/// dependently: wings when facing N/S, fuselage when facing E/W.
///
/// Why uniform scale is correct (and not the terrain's per-vertex `x *=
/// cos_lat`): that shader multiply maps longitude-arcsec to the SAME physical
/// scale as latitude-arcsec (~30.9 m per unit) before the shared proj*view, so
/// the space proj*view consumes is physically isotropic. MESH_RADIUS_ARCSEC is
/// the mesh size in latitude-arcsec, i.e. already in that isotropic unit, so a
/// uniform-scaled mesh and a terrain feature of equal ground size render at
/// equal extent. (Compressing the geometry too, as the old code did, makes a
/// rigid plane render too small east-west: the squish that was reported.)
fn composeMvp(p: DrawParams) [4][4]f32 {
    return math.mat4dToMat4(math.matMulD(p.proj, math.matMulD(p.view_rot, meshModel(p))));
}

/// Model matrix: uniform-scaled rotation, with cos(lat) on the world-X position
/// only. Split out so the rigid-body invariant (geometry never distorts with
/// latitude or heading) is unit-testable.
fn meshModel(p: DrawParams) [4][4]f64 {
    const s: f64 = MESH_RADIUS_ARCSEC;
    const rot3 = math.quatToMat3D(p.orientation);
    return .{
        .{ rot3[0][0] * s, rot3[0][1] * s, rot3[0][2] * s, 0 },
        .{ rot3[1][0] * s, rot3[1][1] * s, rot3[1][2] * s, 0 },
        .{ rot3[2][0] * s, rot3[2][1] * s, rot3[2][2] * s, 0 },
        .{ p.pos_rel_cam[0] * p.cos_lat, p.pos_rel_cam[1], p.pos_rel_cam[2], 1 },
    };
}

fn createPipeline(
    vkd: *const vk.DeviceWrapper,
    device: vk.Device,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    samples: vk.SampleCountFlags,
) !vk.Pipeline {
    return pipeline.createPipeline(vkd, device, .{
        .vert_spv = &aircraft_vert_spv,
        .frag_spv = &aircraft_frag_spv,
        .layout = layout,
        .render_pass = render_pass,
        .samples = samples,
        // No back-face culling: placeholder mesh, easier to debug winding-free.
        .cull_mode = .{},
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "composeMvp: zero pos_rel_cam yields zero translation column" {
    const params = DrawParams{
        .view_rot = math.lookAtD(.{ 0, 0, 0 }, .{ 0, 0, -1 }, .{ 0, 1, 0 }),
        .proj = math.perspectiveD(std.math.pi / 4.0, 1.0, 0.01, 1000.0),
        .pos_rel_cam = .{ 0, 0, 0 },
        .orientation = math.quat_identity,
        .cos_lat = 1.0,
    };
    const mvp = composeMvp(params);
    try testing.expectApproxEqAbs(@as(f32, 0), mvp[3][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), mvp[3][1], 1e-6);
}

test "meshModel: rigid geometry (uniform scale) at high latitude and any heading" {
    // cos(lat) compresses only the world-X POSITION, never the mesh geometry.
    // Old bug: scale_h = scale_v*cos_lat squished a basis axis, so a 90-degree
    // yaw at high latitude changed the plane's shape. Each 3x3 column (a rotated
    // basis vector scaled by the mesh radius) must keep length MESH_RADIUS_ARCSEC.
    const p = DrawParams{
        .view_rot = math.lookAtD(.{ 0, 0, 0 }, .{ 0, 0, -1 }, .{ 0, 1, 0 }),
        .proj = math.perspectiveD(std.math.pi / 4.0, 1.0, 0.01, 1000.0),
        .pos_rel_cam = .{ 0, 0, 0 },
        .orientation = math.quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0),
        .cos_lat = 0.4,
    };
    const m = meshModel(p);
    inline for (0..3) |col| {
        const len = @sqrt(m[0][col] * m[0][col] + m[1][col] * m[1][col] + m[2][col] * m[2][col]);
        // Tolerance clears f32 quat orthonormality noise (orientation is [4]f32)
        // while still catching the old bug, which scaled a column to
        // MESH_RADIUS_ARCSEC * cos_lat (0.194 vs 0.486 here).
        try testing.expectApproxEqAbs(@as(f64, MESH_RADIUS_ARCSEC), len, 1e-5);
    }
}
