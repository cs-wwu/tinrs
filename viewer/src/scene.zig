//! Scene aggregate: terrain + sky + aircraft mesh + HUD subsystems.
//!
//! Owns the tile system, clipmap, sky, placeholder aircraft mesh, and the UI
//! backend as a single unit so main.zig stays a thin composition root. The HUD
//! geometry itself is built in hud.zig; Scene gathers the inputs and records it.

const std = @import("std");
const vkt = @import("vk_types.zig");
const vk = vkt.vk;
const math = @import("math");
const renderer_mod = @import("render/renderer.zig");
const Renderer = renderer_mod.Renderer;
const clipmap_mod = @import("terrain/clipmap.zig");
const Clipmap = clipmap_mod.Clipmap;
const Probe = @import("terrain/probe.zig").Probe;
const tile_system_mod = @import("terrain/tile_system.zig");
const Sky = @import("render/sky.zig").Sky;
const aircraft_mesh_mod = @import("render/aircraft_mesh.zig");
const AircraftMesh = aircraft_mesh_mod.AircraftMesh;
const UiBackend = @import("render/ui_backend.zig").UiBackend;
const ui = @import("ui");
const Aircraft = @import("app/aircraft.zig").Aircraft;
const Camera = @import("app/camera.zig").Camera;
const Pose = @import("app/pose.zig").Pose;
const FrameStats = @import("bench/frame_stats.zig");
const coords = @import("terrain/coords.zig");
const Config = @import("config/options.zig").Config;
const hud = @import("hud.zig");
const debug = @import("render/debug.zig");

pub const Scene = struct {
    tile_system: ?*tile_system_mod.TileSystem,
    /// Device handles, stored so deinit can free the Scene-owned desc_layout
    /// without reaching into clipmap (matches Sky/Probe, which store their own).
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    /// Shared terrain descriptor set layout. Owned by Scene (not Clipmap) so it
    /// survives a clipmap rebuild; Clipmap + Sky borrow it. See Clipmap.createDescLayout.
    desc_layout: vk.DescriptorSetLayout,
    clipmap: Clipmap,
    /// GPU terrain probe (AGL sampler). Null on procedural-only runs (no DB).
    probe: ?Probe,
    sky: Sky,
    aircraft_mesh: AircraftMesh,
    ui_backend: UiBackend,
    /// Per-frame UI geometry, cleared and refilled each draw. Lives on Scene so
    /// the allocation-free DrawList persists without a per-frame arena.
    draw_list: ui.DrawList,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        gpu_ctx: renderer_mod.GpuContext,
        render_pass: vk.RenderPass,
        samples: vk.SampleCountFlags,
        config: *const Config,
    ) !Scene {
        // ---- Tile system (streaming SSBO + worker) ----
        // Built once; outlives Clipmap so MSAA toggles / pipeline recreations
        // don't reload tiles. Null on procedural-only runs.
        var tile_system: ?*tile_system_mod.TileSystem = null;
        errdefer if (tile_system) |ts| ts.deinit();
        if (config.model) |dir| {
            if (tile_system_mod.TileSystem.init(
                allocator, io, gpu_ctx, dir,
                config.max_tile_uploads_per_frame,
                config.max_tiles,
                config.ring_size, config.num_levels, config.base_spacing,
            )) |ts| {
                tile_system = ts;
            } else |err| {
                std.log.warn("Failed to init tile system at '{s}': {} - falling back to procedural terrain", .{ dir, err });
            }
        }

        // ---- Shared descriptor layout ----
        // Owned by Scene, borrowed by Clipmap + Sky, so a future clipmap rebuild
        // can't invalidate the layout Sky's pipeline layout is built from.
        const desc_layout = try Clipmap.createDescLayout(gpu_ctx);
        errdefer gpu_ctx.vkd.destroyDescriptorSetLayout(gpu_ctx.device, desc_layout, null);

        // ---- Clipmap ----
        var clipmap = try Clipmap.init(
            allocator,
            gpu_ctx,
            render_pass,
            samples,
            desc_layout,
            config.base_spacing,
            config.ring_size,
            config.num_levels,
            tile_system,
        );
        errdefer clipmap.deinit();

        // ---- Terrain probe (GPU AGL sampler; only with a real terrain DB) ----
        var probe: ?Probe = null;
        errdefer if (probe) |*p| p.deinit();
        if (tile_system) |ts| probe = try Probe.init(gpu_ctx, ts);

        // ---- Sky ----
        var sky = try Sky.init(gpu_ctx, render_pass, desc_layout, samples);
        errdefer sky.deinit();

        // ---- Aircraft placeholder mesh ----
        var aircraft_mesh = try AircraftMesh.init(gpu_ctx, render_pass, samples);
        errdefer aircraft_mesh.deinit();

        // ---- UI (shapes + text + numeric) ----
        // The numeric pipeline reads AGL from the probe's per-frame buffers; hand
        // them over (null on procedural runs, where AGL isn't shown).
        var agl_bufs: ?[2]vk.Buffer = null;
        if (probe) |*p| agl_bufs = .{ p.outBuffer(0), p.outBuffer(1) };
        const ui_backend = try UiBackend.init(gpu_ctx, render_pass, samples, agl_bufs);

        return .{
            .tile_system = tile_system,
            .vkd = gpu_ctx.vkd,
            .device = gpu_ctx.device,
            .desc_layout = desc_layout,
            .clipmap = clipmap,
            .probe = probe,
            .sky = sky,
            .aircraft_mesh = aircraft_mesh,
            .ui_backend = ui_backend,
            .draw_list = .{},
        };
    }

    pub fn deinit(self: *Scene) void {
        // Drain the GPU before tearing down any Scene-owned resource: on exit the
        // last presented frame may still be reading the clipmap ring SSBO / probe
        // AGL buffers, and Scene.deinit runs before renderer.deinit's idle (defer
        // LIFO). Matches rebuildClipmap's deviceWaitIdle; don't depend on a
        // subsystem deinit (ui_backend) happening to idle first.
        self.vkd.deviceWaitIdle(self.device) catch |err| {
            std.log.warn("deviceWaitIdle before Scene.deinit failed: {}", .{err});
        };
        self.ui_backend.deinit();
        self.aircraft_mesh.deinit();
        self.sky.deinit();
        if (self.probe) |*p| p.deinit();
        self.clipmap.deinit();
        // Scene owns desc_layout (Clipmap borrows it); free it after every
        // pipeline/set built from it is gone.
        self.vkd.destroyDescriptorSetLayout(self.device, self.desc_layout, null);
        if (self.tile_system) |ts| ts.deinit();
    }

    /// Rebuild every subsystem's graphics pipeline against the renderer's current render pass.
    /// Called when the renderer signals render_pass_dirty; the old render pass is already destroyed
    /// so a partial failure here is unrecoverable; caller should exit on error.
    pub fn rebuildPipelines(self: *Scene, renderer: *const Renderer) !void {
        self.clipmap.recreateGraphicsPipeline(renderer.render_pass, renderer.samples) catch |err| {
            std.log.err("Fatal: failed to recreate clipmap pipeline: {}", .{err});
            return err;
        };
        self.sky.recreatePipeline(renderer.render_pass, renderer.samples) catch |err| {
            std.log.err("Fatal: failed to recreate sky pipeline: {}", .{err});
            return err;
        };
        self.aircraft_mesh.recreatePipeline(renderer.render_pass, renderer.samples) catch |err| {
            std.log.err("Fatal: failed to recreate aircraft pipeline: {}", .{err});
            return err;
        };
        self.ui_backend.recreatePipelines(renderer.render_pass, renderer.samples) catch |err| {
            std.log.err("Fatal: failed to recreate UI pipelines: {}", .{err});
            return err;
        };
    }

    /// Tear down and rebuild the clipmap with new ring/level sizes (render
    /// distance). The ring buffer, descriptor pool/sets, and pipelines are all
    /// sized from `ring_size`/`num_levels` at `Clipmap.init`, so a change is a
    /// full rebuild rather than an in-place resize. The Scene-owned `desc_layout`
    /// is reused (Sky's pipeline layout is built from it and must survive), and
    /// the borrowed `tile_system` + `probe` are untouched (the probe binds the
    /// tile SSBOs, not the clipmap). Call BETWEEN frames (not mid-render-pass):
    /// it does a `deviceWaitIdle` so the old buffers are safe to free.
    ///
    /// On failure the existing clipmap is left intact (the new one is built
    /// before the old is destroyed), so the caller can log and keep running.
    ///
    /// TODO: the tile pool was sized for the startup render distance; for
    /// catalogs larger than the pool, growing distance here wants more tiles
    /// than fit. Harmless for PNW (whole catalog fits the pool); a CONUS-scale
    /// catalog needs a tile-pool resize alongside this.
    pub fn rebuildClipmap(
        self: *Scene,
        allocator: std.mem.Allocator,
        gpu_ctx: renderer_mod.GpuContext,
        render_pass: vk.RenderPass,
        samples: vk.SampleCountFlags,
        base_spacing: f32,
        ring_size: u32,
        num_levels: u32,
    ) !void {
        // The one unavoidable hard hitch: the old ring/SSBOs can't be freed
        // until the GPU stops reading them. The expensive MLP re-eval is soft
        // (fast-fill below), but the realloc must wait for idle.
        gpu_ctx.vkd.deviceWaitIdle(gpu_ctx.device) catch |err| {
            std.log.warn("deviceWaitIdle before clipmap rebuild failed: {}", .{err});
        };
        const new_clipmap = try Clipmap.init(
            allocator,
            gpu_ctx,
            render_pass,
            samples,
            self.desc_layout,
            base_spacing,
            ring_size,
            num_levels,
            self.tile_system,
        );
        self.clipmap.deinit();
        self.clipmap = new_clipmap;
        // Soft-fill the new (empty) ring: lift the per-frame upload cap so any
        // newly-wanted tiles land over a few presenting frames. The whole-ring
        // MLP re-eval happens automatically on the next recordUpdate (every
        // level starts uninitialized).
        if (self.tile_system) |ts| ts.beginFastFill();
    }

    pub fn drawAircraft(
        self: *Scene,
        cmd: vk.CommandBuffer,
        camera: *const Camera,
        aircraft_pose: Pose,
        aspect: f32,
    ) void {
        if (camera.mode == .cockpit) return;

        const params = aircraft_mesh_mod.DrawParams{
            .view_rot = math.lookAtD(.{ 0, 0, 0 }, camera.pose.front(), camera.pose.up()),
            .proj = math.perspectiveD(camera.fov, aspect, camera.near, camera.far),
            .pos_rel_cam = .{
                aircraft_pose.position[0] - camera.pose.position[0],
                aircraft_pose.position[1] - camera.pose.position[1],
                aircraft_pose.position[2] - camera.pose.position[2],
            },
            .orientation = aircraft_pose.orientation,
            // cos_lat must match the terrain shader's correction at the aircraft's
            // latitude; in free mode the camera can be far from the aircraft.
            .cos_lat = @as(f64, coords.cosLatFromZ(@floatCast(aircraft_pose.position[2]))),
        };
        self.aircraft_mesh.draw(cmd, params);
    }

    /// Build the always-on HUD into the per-frame DrawList. ALWAYS clears the list
    /// first (so the interactive UI can append to it even when the HUD is hidden),
    /// then emits HUD geometry only when visible. The caller appends UI widgets
    /// after this and finishes with `recordOverlay` (one upload + draw per frame:
    /// the backend re-uploads the whole list, so HUD and UI must share it).
    pub fn buildHud(
        self: *Scene,
        hud_visible: bool,
        camera: *const Camera,
        aircraft: *const Aircraft,
        // The aircraft boresight from the *interpolated* render pose (what the
        // scene was drawn from this frame), so the conformal core stays locked to
        // the terrain instead of jittering on the raw 120Hz physics tick.
        ac_render_front: math.Vec3,
        frame_stats: *const FrameStats,
        extent: vk.Extent2D,
        imperial: bool,
    ) void {
        const dl = &self.draw_list;
        dl.clear();
        if (!hud_visible) return;

        // VRAM / tile residency live on Scene (clipmap + tile system); gather them
        // here and hand the HUD primitives so hud.zig stays free of those types.
        const ssbo_mb: f32 = if (self.tile_system) |ts| @as(f32, @floatFromInt(ts.weightsSize())) / (1024.0 * 1024.0) else 0.0;
        const vram_mb = self.clipmap.vramUsageMB() + ssbo_mb;
        const tiles_resident: u32 = if (self.tile_system) |ts| ts.tileCount() else 0;

        var svs = svsInputs(aircraft);
        // AGL is only valid with a terrain DB; the value itself is GPU-resident
        // (decoded in numeric.vert), this just gates whether the slots are drawn.
        svs.agl_available = self.probe != null;
        svs.imperial = imperial;
        const inputs = hud.Inputs{
            .svs = svs,
            .dev = devInputs(camera, aircraft, frame_stats, vram_mb, tiles_resident),
            .att = attitudeInputs(camera, ac_render_front),
            .show_dev = debug.state.show_block,
        };
        hud.draw(dl, @floatFromInt(extent.width), @floatFromInt(extent.height), inputs);
    }

    /// Record the overlay DrawList (HUD + any UI the caller appended) in a single
    /// pass. `transfer_function` drives the UI backend's color-space encoding (the
    /// HUD/UI colors are SDR-referred; the shaders encode them for the swapchain).
    pub fn recordOverlay(
        self: *Scene,
        cmd: vk.CommandBuffer,
        frame_index: u32,
        extent: vk.Extent2D,
        transfer_function: u32,
        imperial: bool,
    ) void {
        // The GPU-resident AGL digits (numeric.vert) scale by this; the CPU readouts
        // convert in hud.zig. Both share hud.M_TO_FT so they can't drift.
        const numeric_scale: f32 = if (imperial) hud.M_TO_FT else 1.0;
        self.ui_backend.record(cmd, &self.draw_list, frame_index, extent, transfer_function, numeric_scale);
    }
};

const MACH1_KMH: f32 = 1234.8;

/// Groundspeed in km/h from a pose's velocity (arcsec/s).
fn speedKmh(velocity: math.Vec3) f32 {
    const v = velocity;
    return coords.arcsecToMeters(@sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])) * 3.6;
}

/// Gather the conformal attitude-core inputs. The symbology content is
/// aircraft-referenced (`ac_front`, the interpolated render-pose boresight) but
/// projected through the *camera* basis, so it overlays the rendered scene and
/// slides to track the nose under free-look. Both the camera basis and `ac_front`
/// come from the same interpolated frame the terrain was drawn from, so the core
/// does not jitter against the smoothly-moving terrain.
fn attitudeInputs(camera: *const Camera, ac_front: math.Vec3) hud.Attitude {
    const cam = camera.pose;
    return .{
        .cam_right = cam.right(),
        .cam_up = cam.up(),
        .cam_front = cam.front(),
        .fov_y_rad = camera.fov,
        .ac_front = ac_front,
    };
}

/// Gather the aircraft-sourced SVS readout values (always metric here; the HUD
/// converts to the display unit at draw time per `Svs.imperial`). The HUD reports
/// the aircraft's sensor state, so everything here reads `aircraft.pose`.
fn svsInputs(aircraft: *const Aircraft) hud.Svs {
    const pose = aircraft.pose;
    const ll = pose.latLonDeg();
    const kmh = speedKmh(pose.velocity);
    return .{
        .heading_rad = math.headingFromQuat(pose.orientation),
        .speed_kmh = kmh,
        .mach = kmh / MACH1_KMH,
        .alt_m = coords.arcsecToMeters(@floatCast(pose.position[1])),
        // Pose velocity is EMA-smoothed (see Pose), so the vertical component is a
        // GPS-altitude-rate VSI. TODO: complementary-filter with IMU/accel later.
        .vs_mps = coords.arcsecToMeters(pose.velocity[1]),
        .lat = ll[0],
        .lon = ll[1],
        // Set by the caller (buildHud) from probe presence; the AGL value is GPU-side.
        .agl_available = false,
        // Set by the caller (buildHud) from the units setting.
        .imperial = false,
    };
}

/// Gather the dev/debug overlay values: camera-viewpoint position/speed (it
/// answers "where is my viewpoint") plus diagnostics. Toggled via F1 (default off).
fn devInputs(
    camera: *const Camera,
    aircraft: *const Aircraft,
    frame_stats: *const FrameStats,
    vram_mb: f32,
    tiles_resident: u32,
) hud.Dev {
    return .{
        .fps = frame_stats.avg_fps,
        .show_sim = camera.mode == .cockpit or camera.mode == .chase,
        .throttle = aircraft.throttle,
        .auto_level = aircraft.auto_level,
        .vram_mb = vram_mb,
        .tiles_resident = tiles_resident,
        .render_label = if (debug.state.render_mode != .normal) debug.state.render_mode.label() else null,
        .overlay_label = if (debug.state.color_overlay != .off) debug.state.color_overlay.label() else null,
        .freeze_stream = if (debug.state.freeze) (if (debug.state.streaming_override) "live" else "pinned") else null,
    };
}
