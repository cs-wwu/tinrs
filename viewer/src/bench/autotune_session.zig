const std = @import("std");
const vkt = @import("../vk_types.zig");
const c = vkt.c;
const renderer_mod = @import("../render/renderer.zig");
const Renderer = renderer_mod.Renderer;
const autotune_mod = @import("autotune.zig");
const settings_mod = @import("../config/settings.zig");
const config_mod = @import("../config/options.zig");
const Config = config_mod.Config;

/// When `--autotune` is set, run the tuner and persist the result, then return
/// true so the caller exits. Otherwise a no-op returning false: a normal launch
/// reads the saved render distance from `settings.zon` via the usual
/// `Settings.applyToConfig` path (main.zig), so there is nothing to read here.
pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    config_dir: ?[]const u8,
    renderer: *Renderer,
    config: *Config,
    window: ?*c.SDL_Window,
) !bool {
    if (!config.autotune) return false;

    const result = autotune_mod.run(allocator, io, renderer, config, window) catch |err| {
        std.log.err("Autotune failed: {}", .{err});
        return err;
    };

    // Persist the tuned render distance into settings.zon (the single store now;
    // the old per-GPU viewer.zon is gone). Load-modify-save so other prefs are
    // preserved: in particular autotune forces vsync off in `config`, which a full
    // captureFromConfig would wrongly persist. Per-GPU keying is intentionally
    // dropped (a GPU swap just keeps the last tuned value; re-run --autotune).
    if (config_dir) |dir| {
        var s: settings_mod.Settings = settings_mod.Settings.load(allocator, io, dir) orelse .{};
        s.rendering.ring_size = result.ring_size;
        s.rendering.num_levels = result.num_levels;
        s.save(allocator, io, dir) catch |err| {
            std.log.warn("Failed to write settings: {}", .{err});
        };
    }

    const worst = @max(result.static_p99_ms, result.moving_p99_ms);
    std.log.info("Autotune complete: ring_size={d} (P99 {d:.2}ms, budget {d:.2}ms)", .{
        result.ring_size, worst, 1000.0 / @as(f64, @floatFromInt(result.target_fps)),
    });
    return true;
}
