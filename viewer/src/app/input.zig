const std = @import("std");
const vkt = @import("../vk_types.zig");
const c = vkt.c;
const renderer_mod = @import("../render/renderer.zig");
const debug = @import("../render/debug.zig");
const Renderer = renderer_mod.Renderer;
const camera_mod = @import("camera.zig");
const Camera = camera_mod.Camera;
const CameraMode = camera_mod.CameraMode;
const Aircraft = @import("aircraft.zig").Aircraft;
const coords = @import("../terrain/coords.zig");
const Config = @import("../config/options.zig");
const Clipmap = @import("../terrain/clipmap.zig").Clipmap;
const ui = @import("ui");

pub const Context = struct {
    hud_visible: *bool,
    camera: *Camera,
    aircraft: *Aircraft,
    config: *Config.Config,
    renderer: *Renderer,
    clipmap: *const Clipmap,
    samples_off: vkt.vk.SampleCountFlags,
    samples_full: vkt.vk.SampleCountFlags,
};

/// Window-management keys that must work regardless of menu/game state: the menu
/// must NOT swallow these (see main's tiered key dispatch). Returns true if the
/// event was a window key (so the caller stops routing it onward). F11 / Alt+Enter
/// toggle fullscreen; a held key is consumed but only acts on the non-repeat edge.
pub fn handleWindowKey(event: c.SDL_Event, window: ?*c.SDL_Window) bool {
    const is_fs_toggle = event.key.scancode == c.SDL_SCANCODE_F11 or
        (event.key.scancode == c.SDL_SCANCODE_RETURN and (event.key.mod & c.SDL_KMOD_ALT) != 0);
    if (!is_fs_toggle) return false;
    if (!event.key.repeat) {
        const is_fs = (c.SDL_GetWindowFlags(window) & c.SDL_WINDOW_FULLSCREEN) != 0;
        _ = c.SDL_SetWindowFullscreen(window, !is_fs);
    }
    return true; // consume even on repeat so a held key never leaks into UI nav
}

/// Gameplay/scene one-shot keys, dispatched only while no menu is open (the menu
/// captures the keyboard, see main). Window keys (`handleWindowKey`) and the
/// menu-toggle key are handled in earlier tiers and never reach here.
pub fn handleKeyDown(event: c.SDL_Event, ctx: *Context) void {
    if (event.key.scancode == c.SDL_SCANCODE_H and !event.key.repeat) {
        ctx.hud_visible.* = !ctx.hud_visible.*;
    } else if (event.key.scancode == c.SDL_SCANCODE_P and !event.key.repeat) {
        dumpDiagnostics(ctx);
    } else if (event.key.scancode == c.SDL_SCANCODE_R and !event.key.repeat) {
        ctx.camera.resetOrientation();
    } else if (event.key.scancode == c.SDL_SCANCODE_L and !event.key.repeat) {
        ctx.aircraft.auto_level = !ctx.aircraft.auto_level;
        std.log.info("Auto-level: {s}", .{if (ctx.aircraft.auto_level) "ON" else "OFF"});
    } else if (event.key.scancode == c.SDL_SCANCODE_M and !event.key.repeat) {
        cycleCameraMode(ctx);
    } else if (event.key.scancode == c.SDL_SCANCODE_V and !event.key.repeat) {
        ctx.config.no_effects = !ctx.config.no_effects;
        const target = if (ctx.config.no_effects) ctx.samples_off else ctx.samples_full;
        ctx.renderer.setSampleCount(target) catch |err| {
            std.log.err("setSampleCount failed: {}", .{err});
            ctx.config.no_effects = !ctx.config.no_effects;
            return;
        };
        std.log.info("Effects {s} (MSAA {d}x)", .{ if (ctx.config.no_effects) "OFF" else "ON", renderer_mod.sampleCountToInt(ctx.renderer.samples) });
    } else if (event.key.scancode == c.SDL_SCANCODE_O and !event.key.repeat) {
        const ll = ctx.camera.pose.latLonDeg();
        var url_buf: [128:0]u8 = undefined;
        const url = std.fmt.bufPrintZ(&url_buf, "https://www.google.com/maps/@{d:.6},{d:.6},14z", .{ ll[0], ll[1] }) catch return;
        _ = c.SDL_OpenURL(url.ptr);
    } else if (event.key.scancode == c.SDL_SCANCODE_F1 and !event.key.repeat) {
        debug.state.show_block = !debug.state.show_block;
    } else if (event.key.scancode == c.SDL_SCANCODE_F2 and !event.key.repeat) {
        debug.state.render_mode = debug.state.render_mode.next();
        std.log.info("Render mode: {s}", .{debug.state.render_mode.label()});
    } else if (event.key.scancode == c.SDL_SCANCODE_F3 and !event.key.repeat) {
        debug.state.color_overlay = debug.state.color_overlay.next();
        std.log.info("Color overlay: {s}", .{debug.state.color_overlay.label()});
    } else if (event.key.scancode == c.SDL_SCANCODE_F4 and !event.key.repeat) {
        const shift = (event.key.mod & c.SDL_KMOD_SHIFT) != 0;
        if (shift) {
            debug.state.streaming_override = !debug.state.streaming_override;
            std.log.info("Freeze streaming override: {s} (Shift+F4)", .{
                if (debug.state.streaming_override) "ON (streaming continues)" else "OFF (streaming pinned with freeze)",
            });
        } else {
            debug.state.freeze = !debug.state.freeze;
            if (debug.state.freeze) {
                debug.state.frozen_cam_pos = ctx.camera.pose.position;
                debug.state.frozen_fov = ctx.camera.fov;
                const fw: f32 = @floatFromInt(ctx.renderer.swapchain.extent.width);
                const fh: f32 = @floatFromInt(ctx.renderer.swapchain.extent.height);
                const fa: f32 = if (fh > 0) fw / fh else 1.0;
                debug.state.frozen_view = ctx.camera.viewMatrixRotOnly();
                debug.state.frozen_proj = ctx.camera.projMatrix(fa);
            }
            std.log.info("Freeze: {s}", .{if (debug.state.freeze) "ON" else "OFF"});
        }
    }
}

/// Cycle free -> cockpit -> chase -> free. Shared by the M key and the gamepad
/// North button so both stay in sync.
fn cycleCameraMode(ctx: *Context) void {
    ctx.camera.mode = switch (ctx.camera.mode) {
        .free => .cockpit,
        .cockpit => .chase,
        .chase => .free,
    };
    // Drop any free-look glance offset so a new mode starts looking forward.
    ctx.camera.look_yaw = 0;
    ctx.camera.look_pitch = 0;
    ctx.camera.recentering = false;
    if (ctx.camera.mode == .chase)
        ctx.camera.syncChaseAngles(ctx.aircraft.pose.orientation);
    std.log.info("Camera mode: {s}", .{@tagName(ctx.camera.mode)});
}

/// Discrete gamepad buttons while NO menu is open. Flight axes are polled
/// separately in gamepad.zig; this handles one-shot actions, mirroring
/// handleKeyDown. Start (menu toggle) and the in-menu nav buttons are handled in
/// earlier tiers of main's dispatch and never reach here. North (Y on Xbox)
/// cycles camera mode as an alternate to M; R3 (right-stick click) recenters
/// the cockpit free-look glance.
pub fn handleGamepadButton(event: c.SDL_Event, ctx: *Context) void {
    if (event.gbutton.button == c.SDL_GAMEPAD_BUTTON_NORTH) {
        cycleCameraMode(ctx);
    } else if (event.gbutton.button == c.SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
        // R3: recenter the free-look glance offset (springs the view back to the nose).
        ctx.camera.requestRecenter();
    }
}

/// Translate a key-down into a neutral UI focus/nav edge while the menu captures
/// input. Repeat is filtered for discrete moves (Tab/Enter/Esc/Up/Down) but
/// allowed for Left/Right so a held arrow keeps nudging a focused slider. Esc maps
/// to `cancel` (the menu's back/close path); the top-level open is handled in main.
pub fn uiKeyEdge(in: *ui.InputState, event: c.SDL_Event) void {
    const repeat = event.key.repeat;
    switch (event.key.scancode) {
        c.SDL_SCANCODE_TAB => if (!repeat) {
            in.tab = true;
        },
        c.SDL_SCANCODE_RETURN, c.SDL_SCANCODE_KP_ENTER, c.SDL_SCANCODE_SPACE => if (!repeat) {
            in.activate = true;
        },
        c.SDL_SCANCODE_ESCAPE => if (!repeat) {
            in.cancel = true;
        },
        c.SDL_SCANCODE_UP => if (!repeat) {
            in.nav_up = true;
        },
        c.SDL_SCANCODE_DOWN => if (!repeat) {
            in.nav_down = true;
        },
        c.SDL_SCANCODE_LEFT => in.nav_left = true,
        c.SDL_SCANCODE_RIGHT => in.nav_right = true,
        c.SDL_SCANCODE_LEFTBRACKET => if (!repeat) {
            in.tab_prev = true; // previous category tab
        },
        c.SDL_SCANCODE_RIGHTBRACKET => if (!repeat) {
            in.tab_next = true; // next category tab
        },
        else => {},
    }
}

/// Translate a gamepad button into a neutral UI focus/nav edge while the menu
/// captures input: d-pad drives focus nav, South (A) activates, East (B) cancels
/// (back/close). Start (menu toggle) is handled in main's dispatch, not here.
/// Gamepad buttons post one DOWN with no auto-repeat, so a held d-pad does not
/// keep nudging a slider the way a held arrow key does.
// TODO: stick/repeat-driven slider scrubbing (hold to keep stepping nav_left/right);
// needs per-frame polling of the held d-pad/stick rather than discrete button events.
pub fn uiGamepadEdge(in: *ui.InputState, button: u8) void {
    if (button == c.SDL_GAMEPAD_BUTTON_DPAD_UP) {
        in.nav_up = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_DPAD_DOWN) {
        in.nav_down = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_DPAD_LEFT) {
        in.nav_left = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_DPAD_RIGHT) {
        in.nav_right = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_SOUTH) {
        in.activate = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_EAST) {
        in.cancel = true;
    } else if (button == c.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER) {
        in.tab_prev = true; // previous category tab
    } else if (button == c.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER) {
        in.tab_next = true; // next category tab
    }
}

fn dumpDiagnostics(ctx: *Context) void {
    const cam = ctx.camera;
    const clip = ctx.clipmap;
    const cam_alt_arcsec: f32 = @floatCast(cam.pose.position[1]);
    const cam_z: f32 = @floatCast(cam.pose.position[2]);
    const alt_m = coords.arcsecToMeters(cam_alt_arcsec);
    const cos_lat = coords.cosLatFromZ(cam_z);
    const ll = cam.pose.latLonDeg();
    const f = cam.pose.front();
    const d_horizon = coords.horizonDistArcsec(cam_alt_arcsec);
    const d_horizon_km = coords.arcsecToMeters(d_horizon) / 1000.0;
    const fh: f32 = @floatFromInt(ctx.renderer.swapchain.extent.height);
    const min_level = clip.minVisibleLevel(cam_alt_arcsec, cam.fov, fh);
    const max_level = clip.maxVisibleLevel(cam_alt_arcsec, cam_z);

    std.log.info("=== Diagnostics (P) ===", .{});
    std.log.info("Camera: {d:.4}{c} {d:.4}{c}  alt {d:.0}m  cos(lat) {d:.4}", .{
        @abs(ll[0]), @as(u8, if (ll[0] >= 0) 'N' else 'S'),
        @abs(ll[1]), @as(u8, if (ll[1] >= 0) 'E' else 'W'),
        alt_m, cos_lat,
    });
    std.log.info("  front: ({d:.3}, {d:.3}, {d:.3})  speed: {d:.1} arcsec/s ({d:.0} km/h)", .{
        f[0], f[1], f[2], cam.speed, coords.arcsecToMeters(cam.speed) * 3.6,
    });
    std.log.info("Horizon: {d:.0} arcsec ({d:.1} km)  cull threshold: {d:.0} arcsec", .{
        d_horizon, d_horizon_km, d_horizon / cos_lat,
    });
    std.log.info("Clipmap: {d} levels, visible [{d}, {d}]  ring {d}  spacing {d:.2}", .{
        clip.num_levels, min_level, max_level, clip.ring_size, clip.base_spacing,
    });
    std.log.info("  L#   grid_sp   r_inner(as)  r_outer(as)  r_inner(km)  r_outer(km)  r_in*cos  status", .{});
    for (0..clip.num_levels) |i| {
        const g = clip.base_spacing * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(i)));
        const r_in = clip.r_inner_precomp[i];
        const r_out = clip.r_outer_precomp[i];
        const r_in_km = coords.arcsecToMeters(r_in) / 1000.0;
        const r_out_km = coords.arcsecToMeters(r_out) / 1000.0;
        const status: []const u8 = if (i < min_level)
            "SKIP (subpixel)"
        else if (i > max_level)
            "SKIP (horizon)"
        else
            "ACTIVE";
        std.log.info("  L{d}   {d:>7.1}   {d:>11.0}  {d:>11.0}  {d:>11.1}  {d:>11.1}  {d:>8.0}  {s}", .{
            i, g, r_in, r_out, r_in_km, r_out_km, r_in * cos_lat, status,
        });
    }
    std.log.info("=======================", .{});
}

// ---------------------------------------------------------------------------
// Tests: the pure key/button -> InputState edge mappings. The SDL-calling
// handlers (handleKeyDown / handleWindowKey / handleGamepadButton) mutate live
// app state and are exercised at runtime, not here.
// ---------------------------------------------------------------------------

const testing = std.testing;

fn keyEvent(scancode: c_uint, repeat: bool) c.SDL_Event {
    var ev: c.SDL_Event = std.mem.zeroes(c.SDL_Event);
    ev.key.scancode = scancode;
    ev.key.repeat = repeat;
    return ev;
}

test "uiKeyEdge: discrete keys fire on the non-repeat edge only" {
    var in: ui.InputState = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_TAB, false));
    try testing.expect(in.tab);

    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_TAB, true)); // held: suppressed
    try testing.expect(!in.tab);

    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_RETURN, false));
    try testing.expect(in.activate);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_SPACE, false));
    try testing.expect(in.activate);

    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_ESCAPE, false));
    try testing.expect(in.cancel);
}

test "uiKeyEdge: up/down are discrete nav, left/right repeat for slider scrub" {
    var in: ui.InputState = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_UP, false));
    try testing.expect(in.nav_up);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_DOWN, false));
    try testing.expect(in.nav_down);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_DOWN, true)); // discrete: repeat suppressed
    try testing.expect(!in.nav_down);

    // Left/right keep firing while held so a focused slider keeps stepping.
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_LEFT, true));
    try testing.expect(in.nav_left);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_RIGHT, true));
    try testing.expect(in.nav_right);
}

test "uiKeyEdge: brackets switch category tabs (discrete)" {
    var in: ui.InputState = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_LEFTBRACKET, false));
    try testing.expect(in.tab_prev);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_RIGHTBRACKET, false));
    try testing.expect(in.tab_next);
    in = .{};
    uiKeyEdge(&in, keyEvent(c.SDL_SCANCODE_RIGHTBRACKET, true)); // held: suppressed
    try testing.expect(!in.tab_next);
}

test "uiGamepadEdge: shoulders switch category tabs" {
    var in: ui.InputState = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    try testing.expect(in.tab_prev);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    try testing.expect(in.tab_next);
}

test "uiGamepadEdge: d-pad navigates, South activates, East cancels" {
    var in: ui.InputState = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_DPAD_UP);
    try testing.expect(in.nav_up);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    try testing.expect(in.nav_down);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_DPAD_LEFT);
    try testing.expect(in.nav_left);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
    try testing.expect(in.nav_right);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_SOUTH);
    try testing.expect(in.activate);
    in = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_EAST);
    try testing.expect(in.cancel);
}

test "uiGamepadEdge: Start is not a nav edge here (menu toggle lives in main)" {
    var in: ui.InputState = .{};
    uiGamepadEdge(&in, c.SDL_GAMEPAD_BUTTON_START);
    try testing.expect(!in.activate and !in.cancel and !in.tab);
    try testing.expect(!in.nav_up and !in.nav_down and !in.nav_left and !in.nav_right);
}
