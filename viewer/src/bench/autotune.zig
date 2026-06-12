//! Auto-tuning for clipmap parameters.
//!
//! Searches ring_size candidates bottom-up to find the largest that keeps
//! worst-case (P99) frame time under the display refresh budget. Loads tiles
//! once and reuses the weights SSBO across iterations.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const c = vkt.c;
const vk = vkt.vk;
const renderer_mod = @import("../render/renderer.zig");
const Renderer = renderer_mod.Renderer;
const Camera = @import("../app/camera.zig").Camera;
const clipmap_mod = @import("../terrain/clipmap.zig");
const Clipmap = clipmap_mod.Clipmap;
const tile_system_mod = @import("../terrain/tile_system.zig");
const Sky = @import("../render/sky.zig").Sky;
const Bench = @import("runner.zig");
const Profile = @import("profile.zig");
const Convergence = @import("convergence.zig");
const config_mod = @import("../config/options.zig");
const Config = config_mod.Config;
const display = @import("../render/display.zig");

pub const Result = struct {
    ring_size: u32,
    num_levels: u32,
    target_fps: u32,
    static_p99_ms: f64,
    moving_p99_ms: f64,
};

const coarse_candidates = [_]u32{ 63, 127, 255, 511, 1023 };

fn makeOdd(val: u32) u32 {
    return if (val % 2 == 0) val | 1 else val;
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    renderer: *Renderer,
    config: *const Config,
    window: ?*c.SDL_Window,
) !Result {
    const wall_clock: std.Io.Clock = .awake;

    const target_fps: u32 = config.target_fps orelse detectRefreshRate(window) orelse 60;
    const raw_budget_ms: f64 = 1000.0 / @as(f64, @floatFromInt(target_fps));
    const budget_ms: f64 = raw_budget_ms * 0.75;

    std.log.info("Autotune: target {d} fps ({d:.2}ms budget, {d:.2}ms with 75% margin)", .{ target_fps, raw_budget_ms, budget_ms });

    const gpu_ctx = renderer.gpuCtx();

    // Build tile system once (or none for procedural mode), shared across
    // candidate iterations. drainAll synchronously fills every tile so
    // candidate measurements aren't polluted by streaming latency.
    var tile_system: ?*tile_system_mod.TileSystem = null;
    defer if (tile_system) |ts| ts.deinit();
    if (config.model) |dir| {
        if (tile_system_mod.TileSystem.init(
            allocator, io, gpu_ctx, dir,
            config.max_tile_uploads_per_frame,
            config.max_tiles,
            config.ring_size, config.num_levels, config.base_spacing,
        )) |ts| {
            tile_system = ts;
            const start = config.cameraStartPos();
            ts.tickPolicy(.{
                .pos_xz = .{ start[0], start[2] },
                .velocity_xz = .{ 0, 0 },
                .front = .{ 0, 0, -1 },
            }, 0);
            ts.drainAll() catch |err| {
                std.log.warn("Autotune: drainAll failed: {} - continuing with partial residency", .{err});
            };
        } else |err| {
            std.log.warn("Autotune: tile system init failed: {} - falling back to procedural", .{err});
        }
    }

    var best: Result = .{
        .ring_size = coarse_candidates[0],
        .num_levels = config.num_levels,
        .target_fps = target_fps,
        .static_p99_ms = 0,
        .moving_p99_ms = 0,
    };
    var found_viable = false;
    var fail_size: u32 = 2047; // upper bound that failed
    var idx: usize = 1; // start at 127

    // --- Phase 1: Coarse search ---
    while (idx < coarse_candidates.len) {
        const ring_size = coarse_candidates[idx];

        const measure_result = try testCandidate(
            allocator, io, renderer, config, wall_clock,
            tile_system, ring_size, config.num_levels, budget_ms,
        );
        if (measure_result.aborted) return error.UserAborted;

        const worst_p99 = @max(measure_result.static_p99_ms, measure_result.moving_p99_ms);
        if (worst_p99 <= budget_ms) {
            best = .{ .ring_size = ring_size, .num_levels = config.num_levels, .target_fps = target_fps, .static_p99_ms = measure_result.static_p99_ms, .moving_p99_ms = measure_result.moving_p99_ms };
            found_viable = true;
            if (idx == 1 and worst_p99 < budget_ms * 0.7) {
                idx = 3;
                continue;
            }
            idx += 1;
        } else {
            fail_size = ring_size;
            break;
        }
    }

    // If starting candidate (127) failed, try 63
    if (!found_viable) {
        const measure_result = try testCandidate(
            allocator, io, renderer, config, wall_clock,
            tile_system, 63, config.num_levels, budget_ms,
        );
        if (measure_result.aborted) return error.UserAborted;
        best = .{ .ring_size = 63, .num_levels = config.num_levels, .target_fps = target_fps, .static_p99_ms = measure_result.static_p99_ms, .moving_p99_ms = measure_result.moving_p99_ms };

        const worst_p99 = @max(measure_result.static_p99_ms, measure_result.moving_p99_ms);
        if (worst_p99 > budget_ms) {
            std.log.warn("Autotune: even ring_size=63 exceeds budget ({d:.2}ms > {d:.2}ms)", .{ worst_p99, budget_ms });
        }
        std.log.info("Autotune: selected ring_size={d} num_levels={d}", .{ best.ring_size, best.num_levels });
        return best;
    }

    // --- Phase 2: Binary search refinement ---
    // Bisect between best.ring_size (passes) and fail_size (fails or untested upper bound)
    var lo = best.ring_size;
    var hi = fail_size;
    // Only refine if there's meaningful gap (at least 32 apart)
    while (hi - lo > 32) {
        const mid = makeOdd((lo + hi) / 2);
        if (mid == lo or mid == hi) break;

        const measure_result = try testCandidate(
            allocator, io, renderer, config, wall_clock,
            tile_system, mid, config.num_levels, budget_ms,
        );
        if (measure_result.aborted) return error.UserAborted;

        const worst_p99 = @max(measure_result.static_p99_ms, measure_result.moving_p99_ms);
        if (worst_p99 <= budget_ms) {
            lo = mid;
            best = .{ .ring_size = mid, .num_levels = config.num_levels, .target_fps = target_fps, .static_p99_ms = measure_result.static_p99_ms, .moving_p99_ms = measure_result.moving_p99_ms };
        } else {
            hi = mid;
        }
    }

    std.log.info("Autotune: selected ring_size={d} num_levels={d}", .{ best.ring_size, best.num_levels });
    return best;
}

fn testCandidate(
    allocator: std.mem.Allocator,
    io: std.Io,
    renderer: *Renderer,
    config: *const Config,
    wall_clock: std.Io.Clock,
    tile_system: ?*tile_system_mod.TileSystem,
    ring_size: u32,
    num_levels: u32,
    budget_ms: f64,
) !MeasureResult {
    std.log.info("Autotune: testing ring_size={d}...", .{ring_size});
    const result = measureCandidate(
        allocator, io, renderer, config, wall_clock,
        tile_system, ring_size, num_levels,
    ) catch |err| {
        std.log.warn("Autotune: ring_size={d} failed: {}", .{ ring_size, err });
        return err;
    };
    if (!result.aborted) {
        std.log.info("Autotune: ring_size={d} -> P99 static={d:.2}ms moving={d:.2}ms (budget={d:.2}ms)", .{
            ring_size, result.static_p99_ms, result.moving_p99_ms, budget_ms,
        });
    }
    return result;
}

const MeasureResult = struct {
    static_p99_ms: f64,
    moving_p99_ms: f64,
    aborted: bool = false,
};

const CANDIDATE_TIMEOUT_NS: u64 = 60_000_000_000; // 60s hard timeout per candidate

fn measureCandidate(
    allocator: std.mem.Allocator,
    io: std.Io,
    renderer: *Renderer,
    config: *const Config,
    wall_clock: std.Io.Clock,
    tile_system: ?*tile_system_mod.TileSystem,
    ring_size: u32,
    num_levels: u32,
) !MeasureResult {
    std.log.debug("autotune: init clipmap ring_size={d} num_levels={d}", .{ ring_size, num_levels });
    const gpu_ctx = renderer.gpuCtx();
    // Layout created per candidate (registered first -> destroyed last, after
    // clipmap + sky teardown). Borrowed by both, mirroring Scene's ownership.
    const desc_layout = try Clipmap.createDescLayout(gpu_ctx);
    defer gpu_ctx.vkd.destroyDescriptorSetLayout(gpu_ctx.device, desc_layout, null);
    var clipmap = try Clipmap.init(
        allocator, gpu_ctx, renderer.render_pass, renderer.samples, desc_layout,
        config.base_spacing, ring_size, num_levels, tile_system,
    );
    defer clipmap.deinit();

    var sky = try Sky.init(gpu_ctx, renderer.render_pass, desc_layout, renderer.samples);
    defer sky.deinit();

    var profile = try Profile.init(allocator, 0, .{
        .measure_duration_ns = 3_000_000_000,
        .convergence_opts = .{ .max_time_ns = 15_000_000_000, .max_spread_pct = 0.10 },
    });
    defer profile.deinit();

    // Registered last, runs first on scope exit (LIFO). Drains the GPU
    // before any pipeline teardown above so vkDestroyPipeline doesn't fire
    // VUID-vkDestroyPipeline-pipeline-00765 against in-flight command buffers.
    defer renderer.vkd.deviceWaitIdle(renderer.device) catch {};

    var camera = Camera.init(config.cameraStartPos(), config.fov, config.cameraStartDir());
    camera.near = config.near;
    camera.far = config.far;

    var last_ts = wall_clock.now(io);
    const measure_start = last_ts;
    var prev_phase = profile.phase;
    var frame_count: u64 = 0;

    std.log.debug("autotune: entering frame loop for ring_size={d}", .{ring_size});

    while (true) {
        // SDL installs SIGINT/SIGTERM handlers that convert those signals into
        // SDL_EVENT_QUIT. Drain the queue whenever SDL is up, including the
        // headless+autotune case (no window but SDL_INIT_VIDEO is on), so
        // Ctrl+C exits cleanly instead of being swallowed.
        if (c.SDL_WasInit(c.SDL_INIT_VIDEO) != 0) {
            var event: c.SDL_Event = undefined;
            while (c.SDL_PollEvent(&event)) {
                if (event.type == c.SDL_EVENT_QUIT)
                    return .{ .static_p99_ms = 0, .moving_p99_ms = 0, .aborted = true };
            }
        }

        const now_ts = wall_clock.now(io);
        const dt_ns = Bench.deltaNs(last_ts, now_ts);
        const dt: f32 = @floatCast(@as(f64, @floatFromInt(dt_ns)) / 1e9);
        last_ts = now_ts;

        // Hard timeout per candidate
        const elapsed_ns = Bench.deltaNs(measure_start, now_ts);
        if (elapsed_ns >= CANDIDATE_TIMEOUT_NS) {
            std.log.warn("autotune: ring_size={d} timed out after {d:.1}s (phase={s}, {d} frames)", .{
                ring_size,
                @as(f64, @floatFromInt(elapsed_ns)) / 1e9,
                @tagName(profile.phase),
                frame_count,
            });
            break;
        }

        if (profile.shouldFly()) camera.autopilot(dt);

        const ctx = renderer.beginFrame() catch break orelse continue;
        const record_t0 = wall_clock.now(io);

        const h: f32 = @floatFromInt(renderer.swapchain.extent.height);

        clipmap.recordUpdate(ctx.cmd_buf, camera.pose.position, renderer.current_frame, frame_count, camera.fov, h);
        renderer.beginRenderPass(ctx.cmd_buf, ctx.image_index);
        const aspect: f32 = if (h > 0) @as(f32, @floatFromInt(renderer.swapchain.extent.width)) / h else 1.0;
        clipmap.recordDraw(ctx.cmd_buf, clipmap_mod.buildSceneParams(&camera, aspect, .{
            .fog_max_dist = clipmap.currentFogMaxDist(@floatCast(camera.pose.position[2])),
            .no_effects = false,
            .transfer_function = renderer.transfer_function,
        }), renderer.current_frame, camera.fov, h);
        sky.draw(ctx.cmd_buf, clipmap.desc_sets[renderer.current_frame]);
        renderer.vkd.cmdEndRenderPass(ctx.cmd_buf);

        const record_ns = Bench.deltaNs(record_t0, wall_clock.now(io));
        renderer.endFrame(ctx) catch break;

        var sample = renderer.lastSample();
        sample.scrolling = (clipmap.last_strips + clipmap.last_refills) > 0;
        sample.cpu_record_ns = record_ns;
        frame_count += 1;

        const phase_result = profile.tick(dt_ns, sample);

        // Log phase transitions
        if (profile.phase != prev_phase) {
            std.log.debug("autotune: ring_size={d} phase {s} -> {s} (frame {d}, {d:.1}s elapsed)", .{
                ring_size,
                @tagName(prev_phase),
                @tagName(profile.phase),
                frame_count,
                @as(f64, @floatFromInt(elapsed_ns)) / 1e9,
            });
            prev_phase = profile.phase;
        }

        if (phase_result == .done) break;
    }

    const results = profile.results();
    std.log.debug("autotune: ring_size={d} done - {d} frames, static_p99={d:.2}ms moving_p99={d:.2}ms", .{
        ring_size, frame_count, results.static_p99_ms, results.moving_p99_ms,
    });
    return .{
        .static_p99_ms = results.static_p99_ms,
        .moving_p99_ms = results.moving_p99_ms,
    };
}

fn detectRefreshRate(window: ?*c.SDL_Window) ?u32 {
    const d = if (window) |w| display.forWindow(w) else display.best();
    const rate: i32 = @intFromFloat((d orelse return null).refresh_hz);
    return if (rate > 0) @intCast(rate) else null;
}
