//! SDL3 window + display setup.
//!
//! Encapsulates SDL_Init / window creation / display placement so main.zig
//! can stay focused on subsystem composition. Caller owns the returned
//! window handle and is responsible for SDL_DestroyWindow + SDL_Quit.

const std = @import("std");
const vkt = @import("vk_types.zig");
const c = vkt.c;
const display = @import("render/display.zig");
const sysinfo = @import("config/sysinfo.zig");
const Config = @import("config/options.zig").Config;

pub const WindowSetup = struct {
    window: ?*c.SDL_Window,
    width: c_int,
    height: c_int,
    /// True when SDL_INIT_VIDEO was attempted; caller must SDL_Quit if so.
    sdl_video: bool,
};

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    environ_map: *std.process.Environ.Map,
    config: *const Config,
) !WindowSetup {
    // ---- SDL3 ----
    // Headless mode normally skips SDL entirely, but autotune still wants the
    // best monitor's refresh rate + resolution to set its frame budget, so
    // init SDL_INIT_VIDEO whenever autotune is on, headless or not.
    var win_w: c_int = @intCast(config.width);
    var win_h: c_int = @intCast(config.height);
    var window: ?*c.SDL_Window = null;
    const sdl_video = !config.headless or config.autotune;

    if (sdl_video) {
        // Gamepad subsystem only for interactive (windowed) runs; headless
        // autotune needs video for refresh detection but no controller.
        // SDL_INIT_GAMEPAD implies JOYSTICK + EVENTS.
        const init_flags = if (config.headless) c.SDL_INIT_VIDEO else c.SDL_INIT_VIDEO | c.SDL_INIT_GAMEPAD;
        if (!c.SDL_Init(init_flags)) {
            if (!config.headless) {
                std.log.err("Failed to initialize SDL: {s}", .{c.SDL_GetError()});
                return error.SdlInitFailed;
            }
            std.log.warn("Autotune (headless): SDL init failed - refresh detection unavailable: {s}", .{c.SDL_GetError()});
        }
    }

    if (config.debug) sysinfo.logDetected(allocator, io, environ_map, .{
        .cores = @intCast(@max(c.SDL_GetNumLogicalCPUCores(), 0)),
        .ram_mb = @intCast(@max(c.SDL_GetSystemRAM(), 0)),
        .has_avx512f = c.SDL_HasAVX512F(),
        .has_avx2 = c.SDL_HasAVX2(),
        .has_avx = c.SDL_HasAVX(),
        .has_sse42 = c.SDL_HasSSE42(),
        .has_neon = c.SDL_HasNEON(),
        .video_driver = if (c.SDL_GetCurrentVideoDriver()) |drv| std.mem.span(drv) else null,
    });

    if (!config.headless) {
        // Never start fullscreen: fullscreen-at-create-time pins the window to
        // the default display under Wayland/Mutter and ignores subsequent
        // SetWindowPosition calls. display.place() handles the position-then-fullscreen
        // sequence correctly.
        const flags: c.SDL_WindowFlags = c.SDL_WINDOW_VULKAN | c.SDL_WINDOW_RESIZABLE;
        window = c.SDL_CreateWindow("tinrs viewer", win_w, win_h, flags) orelse {
            std.log.err("Failed to create SDL window: {s}", .{c.SDL_GetError()});
            return error.WindowCreationFailed;
        };

        // Pick the target display: autotune wants the highest-refresh one;
        // --monitor N wants index N; otherwise leave the window where SDL put it.
        // Renderer.init below locks the surface format from the window's current
        // display, so placement has to happen first.
        // TODO(monitor-sentinel): config.monitor == 0 doubles as "no preference" and
        // "Monitor 1", so a menu-selected Monitor 1 isn't placed at startup when SDL's
        // default display isn't index 0. Make config.monitor an ?u32 (null = no
        // preference) to disambiguate. Fullscreen still lands on the default display.
        const target: ?display.Display = if (config.autotune)
            display.best()
        else if (config.monitor != 0) blk: {
            const d = display.byIndex(config.monitor);
            if (d == null) std.log.warn("Monitor {d} not available, falling back to default display", .{config.monitor});
            break :blk d;
        } else null;
        display.place(window.?, target, config.fullscreen);

        display.logAll(c.SDL_GetDisplayForWindow(window.?));
    } else {
        if (config.autotune) {
            if (display.best()) |b| {
                win_w = b.w;
                win_h = b.h;
                display.logAll(b.id);
            }
        }
        std.log.debug("Headless mode: VK_EXT_headless_surface, {d}x{d}, no input", .{ win_w, win_h });
    }

    return .{
        .window = window,
        .width = win_w,
        .height = win_h,
        .sdl_video = sdl_video,
    };
}

pub fn deinit(setup: WindowSetup) void {
    if (setup.window) |w| c.SDL_DestroyWindow(w);
    if (setup.sdl_video) c.SDL_Quit();
}
