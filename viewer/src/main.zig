const std = @import("std");
const vkt = @import("vk_types.zig");
const c = vkt.c;
const renderer_mod = @import("render/renderer.zig");
const swapchain_mod = @import("render/swapchain.zig");
const Renderer = renderer_mod.Renderer;
const clipmap_mod = @import("terrain/clipmap.zig");
const tile_system_mod = @import("terrain/tile_system.zig");
const tile_loader = @import("terrain/tile_loader.zig");
const coords = @import("terrain/coords.zig");
const Config = @import("config/options.zig");
const settings_mod = @import("config/settings.zig");
const autotune_session = @import("bench/autotune_session.zig");
const config_file = @import("config/config_file.zig");
const display = @import("render/display.zig");
const input = @import("app/input.zig");
const camera_mod = @import("app/camera.zig");
const controls_mod = @import("app/controls.zig");
const gamepad = @import("app/gamepad.zig");
const pointer_look_mod = @import("app/pointer_look.zig");
const Pose = @import("app/pose.zig").Pose;
const sim = @import("app/sim.zig");
const Bench = @import("bench/runner.zig");
const Scene = @import("scene.zig").Scene;
const Session = @import("session.zig").Session;
const window_mod = @import("window.zig");
const ui = @import("ui");

/// Which screen the pause menu is showing. A single panel switches content on
/// this rather than nesting panels (the immediate-mode core does not nest panels
/// yet; see context.zig). `back` (Esc/B) pops a sub-page to `root`, then closes.
const MenuPage = enum { root, settings };

/// Category tab within the Settings page. Display (vsync/msaa) applies on change
/// via the renderer; Graphics (effects/fog) is live; Input is live tuning. The
/// Debug tab (render-mode cycles) drips in later. Order must match
/// `settings_tab_labels`. The active tab is driven by the `tabBar` widget (a usize
/// index), so this needs no cycle helpers.
const SettingsTab = enum { display, graphics, input };

const settings_tab_labels = [_][]const u8{ "Display", "Graphics", "Input" };

/// App state the pause menu reads/writes at the call site (immediate mode). The
/// menu mutates through these pointers: Resume clears `open`, Settings switches
/// `page`, Exit sets `should_exit`. `config`/`tuning` back the Settings content.
/// MSAA cycle options, built once at startup and filtered to GPU support. `labels`
/// and `counts` are index-aligned (`counts[i]` is the sample count for `labels[i]`).
/// The cycle's active index is NOT stored: it is derived fresh each frame from the
/// renderer's applied count, so it can never desync (e.g. from the V-key shortcut).
const MsaaOptions = struct {
    labels: []const []const u8,
    counts: []const u32,
};

const window_mode_labels = [_][]const u8{ "Windowed", "Borderless" };
// Index-aligned with Config.Units (metric=0, imperial=1).
const units_labels = [_][]const u8{ "Metric", "Imperial" };

/// A requested window placement (monitor + borderless-fullscreen), set by the menu
/// and applied between frames via `display.place`. Deferred (not applied at the call
/// site) so the SDL window change + its swapchain recreate land outside the render
/// pass, the same timing `pending_msaa` uses.
const PendingPlace = struct { monitor: u32, borderless: bool };

/// A requested runtime clipmap rebuild (render distance). `ring_size` =
/// per-ring detail (re-rounded to chunk alignment by `Clipmap.init`),
/// `num_levels` = horizon reach. Set by the Graphics-tab Apply button (and the
/// auto-revert), applied between frames via `Scene.rebuildClipmap` (the rebuild
/// can't run mid-render-pass). Mirrors the `pending_msaa`/`pending_place` model.
const RingRebuild = struct { ring_size: u32, num_levels: u32 };

/// Auto-revert window after a render-distance Apply: if a too-aggressive setting
/// tanks fps so badly the menu is unusable, it reverts to `prev` once `start +
/// REVERT_NS` passes without the user clicking Keep. The OS-display-change pattern.
const RevertGuard = struct { prev: RingRebuild, start: std.Io.Timestamp };
const REVERT_NS: u64 = 12 * std.time.ns_per_s;

/// Render-distance controls passed to the Settings/Graphics tab. Apply-gated: the
/// sliders edit the staged values; Apply commits a rebuild; while a just-applied
/// change awaits confirmation `revert_seconds` is non-null and a Keep button +
/// countdown replace the sliders. The menu only flips `apply`/`keep`; the main
/// loop owns the staged values, the pending rebuild, and the revert timer.
const RenderDist = struct {
    staged_ring: *i32,
    staged_levels: *i32,
    live_ring: u32,
    live_levels: u32,
    apply: *bool,
    keep: *bool,
    revert_seconds: ?u32,
};

const MenuCtx = struct {
    open: *bool,
    page: *MenuPage,
    tab: *SettingsTab,
    should_exit: *bool,
    config: *Config.Config,
    tuning: *gamepad.Tuning,
    renderer: *Renderer,
    msaa: MsaaOptions,
    /// Desired MSAA sample count, set when the cycle changes. Applied by the main
    /// loop BETWEEN frames (not here): the menu runs inside the render pass, and
    /// `setSampleCount` destroys the render pass + framebuffers the in-flight command
    /// buffer is recording into, which corrupts it. `null` when nothing is pending.
    pending_msaa: *?u32,
    window: ?*c.SDL_Window,
    /// Monitor labels ("Monitor N"), one per connected display (built once).
    monitor_labels: []const []const u8,
    /// Whether monitor selection can work here: multi-monitor and not Wayland (whose
    /// compositors ignore client window positioning). Greys the Monitor cycle out.
    monitor_selectable: bool,
    /// Desired window placement, set when the Mode/Monitor cycle changes; applied
    /// between frames (see `PendingPlace`). `null` when nothing is pending.
    pending_place: *?PendingPlace,
    /// Render-distance controls (Graphics tab); apply-gated rebuild + auto-revert.
    rdist: RenderDist,
};

/// Build the pause menu for this frame. One panel; content switches on the page.
/// Returns nothing: actions mutate app state through `MenuCtx` pointers. Focus is
/// cleared on every page/tab switch so a stale `focused_id` from the old view
/// (whose widget ids no longer exist) cannot linger.
fn buildMenu(u: *ui.Ui, m: MenuCtx) void {
    const title = switch (m.page.*) {
        .root => "Menu",
        .settings => "Settings",
    };
    const width: f32 = switch (m.page.*) {
        .root => 200,
        .settings => 300, // wider page for the tab bar + content list
    };
    if (!u.beginPanel(@src(), title, .{ .anchor = .center, .off_x = -width / 2, .off_y = -90, .width = width })) return;
    defer u.endPanel();

    switch (m.page.*) {
        .root => {
            if (ui.button(u, @src(), "Resume")) m.open.* = false;
            if (ui.button(u, @src(), "Settings")) {
                m.page.* = .settings;
                m.tab.* = .display; // open Settings on the first tab
                u.clearFocus();
            }
            if (ui.button(u, @src(), "Info")) {
                _ = c.SDL_OpenURL("https://github.com/cs-wwu/tinrs");
            }
            if (ui.button(u, @src(), "Exit")) m.should_exit.* = true;
        },
        .settings => buildSettings(u, m),
    }
}

/// Settings page: a top tab bar over a vertical list of controls bound to live
/// state. The tab bar is one focus stop, so Up/Down navigates `[tabBar, controls,
/// Back]`, Left/Right switch category while on the bar, and Down drops into the
/// content below. Every control applies immediately (read each frame by its
/// subsystem) and persists on exit via the settings.zon capture. Entering the page
/// auto-focuses the bar (see main loop), so Left/Right work without a Down first.
fn buildSettings(u: *ui.Ui, m: MenuCtx) void {
    var idx: usize = @intFromEnum(m.tab.*);
    // Accelerator: shoulder buttons / `[` `]` cycle the tab from anywhere on the page.
    if (u.input.tab_next) idx = (idx + 1) % settings_tab_labels.len;
    if (u.input.tab_prev) idx = (idx + settings_tab_labels.len - 1) % settings_tab_labels.len;
    // The bar itself: Left/Right while focused, or a click. Focus stays on the bar.
    _ = ui.tabBar(u, @src(), &idx, &settings_tab_labels);
    m.tab.* = @enumFromInt(idx);

    switch (m.tab.*) {
        .display => {
            // VSync applies on change: only the swapchain present mode swaps, no
            // pipeline rebuild. Keep `config.vsync` in sync so it persists on exit.
            var vsync = m.config.vsync;
            if (ui.toggle(u, @src(), "VSync", &vsync)) {
                m.config.vsync = vsync;
                m.renderer.setVsync(vsync);
            }
            // MSAA cycle over GPU-supported counts. The active index is derived fresh
            // from the actually-applied count, so the cycle always shows the truth
            // (including after the V-key shortcut) and never desyncs. A change only
            // records the desired count; the loop applies it between frames (the
            // rebuild can't run mid-render-pass; see `pending_msaa`).
            var msaa_idx = renderer_mod.indexOfCount(renderer_mod.sampleCountToInt(m.renderer.samples), m.msaa.counts);
            if (ui.cycle(u, @src(), "MSAA", &msaa_idx, m.msaa.labels)) {
                m.pending_msaa.* = m.msaa.counts[msaa_idx];
            }
            // HDR: greyed out unless the hardware + current display support it. The
            // toggle shows the user preference; applying it re-runs surface-format
            // detection via the display-changed path (deferred recreate, like VSync).
            u.beginDisabled(!m.renderer.hdrAvailable());
            // Read config (kept in lockstep with renderer.hdr_enabled) for the same
            // source-of-truth as VSync above; write both on change.
            var hdr = !m.config.no_hdr;
            if (ui.toggle(u, @src(), "HDR", &hdr)) {
                m.renderer.setHdr(hdr);
                m.config.no_hdr = !hdr;
            }
            u.endDisabled();

            // Window mode + monitor. The current mode is read from the live SDL flag
            // (so it tracks the F11 toggle, which doesn't touch config), and the
            // current monitor from the window. A change records a deferred placement;
            // the loop applies it via display.place between frames.
            const is_fs = m.window != null and (c.SDL_GetWindowFlags(m.window) & c.SDL_WINDOW_FULLSCREEN) != 0;
            const cur_mon = display.indexForWindow(m.window);
            var mode_idx: usize = if (is_fs) 1 else 0;
            if (ui.cycle(u, @src(), "Mode", &mode_idx, &window_mode_labels)) {
                m.pending_place.* = .{ .monitor = @intCast(cur_mon), .borderless = mode_idx == 1 };
            }
            // Monitor: greyed on Wayland / single-monitor (where it can't work).
            u.beginDisabled(!m.monitor_selectable);
            var mon_idx = cur_mon;
            if (m.monitor_labels.len > 1 and ui.cycle(u, @src(), "Monitor", &mon_idx, m.monitor_labels)) {
                m.pending_place.* = .{ .monitor = @intCast(mon_idx), .borderless = is_fs };
            }
            u.endDisabled();

            // HUD units: live (the HUD reads config.units each frame; the GPU AGL
            // scale and CPU readouts both follow). Persists to settings.zon on exit.
            var units_idx: usize = @intFromEnum(m.config.units);
            if (ui.cycle(u, @src(), "Units", &units_idx, &units_labels)) {
                m.config.units = @enumFromInt(units_idx);
            }
        },
        .graphics => {
            // `Config` stores the negative sense; present + bind the positive one.
            // Effects here is lighting only and independent of MSAA (which is its
            // own Display setting). The V key drops both together as a benchmark
            // shortcut; the menu keeps them separate.
            var effects = !m.config.no_effects;
            if (ui.toggle(u, @src(), "Effects", &effects)) m.config.no_effects = !effects;
            // Fog is part of the effects set: with Effects off, terrain.frag takes
            // the simple-lighting early-out and never calls applyFog, so the Fog
            // toggle is a no-op. Grey it out (like HDR-on-SDR) to show it's subsumed.
            u.beginDisabled(m.config.no_effects);
            var fog = !m.config.no_fog;
            if (ui.toggle(u, @src(), "Fog", &fog)) m.config.no_fog = !fog;
            u.endDisabled();
            // TAWS hazard overlay: live (terrain.frag reads it via the SceneUBO
            // each frame), persists to settings.zon on exit.
            _ = ui.toggle(u, @src(), "TAWS", &m.config.taws);

            // Render distance: apply-gated (a rebuild is too expensive for
            // apply-on-change). The sliders stage values; Apply commits one rebuild.
            // While a just-applied change awaits confirmation the sliders give way
            // to a Keep prompt + auto-revert countdown, so a setting that tanks fps
            // on weak hardware can't strand you (see RevertGuard).
            const rd = m.rdist;
            if (rd.revert_seconds) |secs| {
                var buf: [48]u8 = undefined;
                ui.labelRow(u, std.fmt.bufPrint(&buf, "Keep? Reverting in {d}s", .{secs}) catch "Keep?");
                if (ui.button(u, @src(), "Keep")) rd.keep.* = true;
            } else {
                // Sane floors above the CLI hard limits ([63,2047] / [1,MAX_LEVELS]):
                // below ~255 ring / 3 levels the world is too sparse to be useful.
                // ring is odd (step 2); Clipmap.init re-rounds it to chunk alignment,
                // so compare the *achievable* ring against the live value.
                _ = ui.sliderInt(u, @src(), "Detail (ring)", rd.staged_ring, 255, 2047, 2);
                _ = ui.sliderInt(u, @src(), "View distance", rd.staged_levels, 3, @intCast(clipmap_mod.MAX_LEVELS), 1);
                const target_ring = clipmap_mod.Clipmap.ringLayout(@intCast(rd.staged_ring.*)).ring_size;

                // Ground render radius these slider values produce, so the two
                // abstract counts read as a real distance. The world is parameterized
                // in arcsec, so arcsecToMeters yields the N/S radius, which is constant
                // with latitude and equals the E/W radius only at the equator; E/W
                // coverage is this * cos(lat) and shrinks toward the poles, hence the
                // "equator" qualifier. Reflects the staged (achievable) values live,
                // before Apply. Imperial uses NM to match the kt/ft aviation set.
                const r_m = coords.arcsecToMeters(tile_loader.renderRadiusArcsec(
                    target_ring,
                    @intCast(rd.staged_levels.*),
                    m.config.base_spacing,
                ));
                var range_buf: [48]u8 = undefined;
                const range_row = switch (m.config.units) {
                    .metric => std.fmt.bufPrint(&range_buf, "Range ~{d:.0} km (equator)", .{r_m / 1000.0}),
                    .imperial => std.fmt.bufPrint(&range_buf, "Range ~{d:.0} NM (equator)", .{r_m / 1852.0}),
                } catch "Range";
                ui.labelRow(u, range_row);

                const differs = target_ring != rd.live_ring or @as(u32, @intCast(rd.staged_levels.*)) != rd.live_levels;
                if (differs and ui.button(u, @src(), "Apply")) rd.apply.* = true;
            }
        },
        .input => {
            _ = ui.sliderF32(u, @src(), "Stick deadzone", &m.tuning.deadzone, 0, 0.4);
            _ = ui.toggle(u, @src(), "Invert pitch", &m.tuning.invert_pitch);
            _ = ui.sliderF32(u, @src(), "Trigger DZ", &m.tuning.trigger_deadzone, 0, 0.2);
            _ = ui.sliderF32(u, @src(), "Look speed", &m.tuning.look_sensitivity, 0.5, 6.0);
            _ = ui.sliderF32(u, @src(), "Drag sens", &m.config.look_drag_sensitivity, 0.1, 3.0);
        },
    }

    if (ui.button(u, @src(), "Back")) {
        m.page.* = .root;
        u.clearFocus();
    }
}

/// On-screen pause control for pointer / touch, top-right corner. A rounded,
/// mostly-opaque button with a "||" glyph (two bars). Returns true the frame it is
/// clicked/tapped. Pointer-only (not a focus target): keyboard/gamepad open the
/// menu with Esc/Start. Touch reaches it through SDL's touch->mouse synthesis, so
/// a single tap target needs no finger-event wiring. Shown only while the menu is
/// closed (open uses Resume/Back).
fn buildPauseButton(u: *ui.Ui) bool {
    const s = u.screen;
    // Keep a finger-sized hit target but draw a smaller button inside it (invisible
    // padding), so it reads as a subtle corner control rather than a big block.
    const hit = s.px(44);
    const vis = s.px(32);
    const margin = s.px(12);
    const hit_rect = ui.Rect{ .x = s.w - margin - hit, .y = margin, .w = hit, .h = hit };
    const wid = u.id(@src(), 0);
    const resp = u.behavior(wid, hit_rect);

    const pad = (hit - vis) * 0.5;
    const r = ui.Rect{ .x = hit_rect.x + pad, .y = hit_rect.y + pad, .w = vis, .h = vis };

    // Deliberately understated: a translucent fill with an alpha-only hover/press
    // ramp (no hue jump) and a faint thin border, separate from the solid modal
    // panel chrome so it stays in the background until reached for.
    const bg_a: f32 = if (resp.held) 0.7 else if (u.isHot(wid)) 0.55 else 0.4;
    u.dl.roundedRect(r.x, r.y, r.w, r.h, s.px(u.theme.corner), ui.Color.rgba(0.06, 0.08, 0.07, bg_a));
    u.dl.roundedRectOutline(r.x, r.y, r.w, r.h, s.px(u.theme.corner), s.px(1), ui.Color.rgba(0.45, 0.95, 0.55, 0.3));

    // "||" pause glyph: two vertical bars centered in the button.
    const bar_w = vis * 0.16;
    const bar_h = vis * 0.40;
    const gap = vis * 0.16;
    const bar_y = r.y + (r.h - bar_h) * 0.5;
    const cx = r.x + r.w * 0.5;
    const bar_color = ui.Color.rgba(0.65, 1.0, 0.72, 0.85);
    u.dl.roundedRect(cx - gap * 0.5 - bar_w, bar_y, bar_w, bar_h, s.px(1), bar_color);
    u.dl.roundedRect(cx + gap * 0.5, bar_y, bar_w, bar_h, s.px(1), bar_color);
    return resp.activated;
}

/// Apply the free-look glance for an "attached" camera mode (cockpit, and the
/// upcoming sensor mode): the camera follows `base` (the aircraft pose) for
/// position + velocity, while the right stick and a pointer drag (hold-RMB mouse
/// or one-finger touch) offset ONLY the orientation, held when idle and sprung
/// back on recenter. Shared so sensor mode reuses the exact glance behavior
/// instead of copying this glue. `drag_dx`/`drag_dy` are this frame's accumulated
/// pointer motion in PIXELS (mouse relative px + touch normalized * screen px;
/// 0 when not engaged). Because the camera orientation carries the glance, the
/// rendered view (and the tile prefetch that follows camera.front()) tracks where
/// you look.
fn applyAttachedGlance(
    camera: *camera_mod.Camera,
    base: Pose,
    pad: *const gamepad.Gamepad,
    drag_active: bool,
    drag_dx: f32,
    drag_dy: f32,
    drag_sensitivity: f32,
    dt: f32,
) void {
    var dyaw: f32 = 0;
    var dpitch: f32 = 0;
    var looking = false;
    if (pad.readLook()) |lk| {
        if (lk[0] != 0 or lk[1] != 0) {
            dyaw = lk[0] * pad.tuning.look_sensitivity * dt;
            dpitch = lk[1] * pad.tuning.look_sensitivity * dt;
            looking = true;
        }
    }
    if (drag_active) {
        // Drag deltas are angular already (rad/px * setting), not rate-based, so a
        // fixed drag glances a fixed amount regardless of dt. +dx right, +dy down.
        const dsens = camera_mod.Camera.LOOK_DRAG_BASE * drag_sensitivity;
        dyaw += drag_dx * dsens;
        dpitch += -drag_dy * dsens;
        looking = true;
    }
    camera.applyLook(dyaw, dpitch, looking, dt);
    camera.pose = base;
    camera.pose.orientation = camera_mod.lookOffsetOrientation(base.orientation, camera.look_yaw, camera.look_pitch);
}

/// Advance the aircraft one frame and return the pose the camera follows / the
/// model renders at, for an attached camera mode (cockpit, chase).
///   `.sim`    - the fixed-timestep ticker integrates the flight model; the
///               returned pose is interpolated between the last two physics states.
///   `.sensor` - the aircraft holds a FROZEN pose: the sim tick is skipped (no
///               flight input) and the frozen pose is returned unchanged. The
///               ticker anchor is kept fresh so a future sim re-entry won't lerp
///               from a stale snapshot. This is the SensorInput swap-in seam.
fn advanceAircraft(session: *Session, source: Config.FlightSource, dt: f32, controls: controls_mod.Controls) Pose {
    switch (source) {
        .sim => {
            const alpha = session.ticker.step(&session.aircraft, dt, controls);
            return Pose.lerp(session.ticker.prev_pose, session.aircraft.pose, alpha);
        },
        .sensor => {
            // TODO: SensorInput (GPS + IMU + compass fusion) writes
            // session.aircraft.pose / airspeed here; until then it is a frozen hold.
            session.ticker.reset(session.aircraft.pose);
            return session.aircraft.pose;
        },
    }
}

/// Runtime log gate, flipped on by `--debug` so std.log.debug calls fire.
/// Kept atomic so subsystems can read it without coordination, though in
/// practice only main.zig writes it (once, at startup).
var verbose_logging: std.atomic.Value(bool) = .init(false);

fn viewerLog(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime fmt: []const u8,
    args: anytype,
) void {
    if (level == .debug and !verbose_logging.load(.monotonic)) return;
    std.log.defaultLog(level, scope, fmt, args);
}

/// Compile-time max log level is .debug so log.debug calls aren't dead code;
/// viewerLog drops them at runtime unless --debug bumps the gate.
pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = viewerLog,
};

/// Silent-probe window before the startup loading screen reveals: a load that
/// finishes inside this never presents a loading frame (no flash on fast disks /
/// small catalogs); one that crosses it gets the screen. Generous enough that the
/// 9070 XT (~33ms) never shows it, low enough that a slow disk / RPi / large pool
/// reveals it promptly. Tune against the actual Pi if the threshold feels wrong.
const LOAD_PROBE_NS: u64 = 100 * std.time.ns_per_ms;

/// Minimal event pump for the startup load (no menu / gameplay routing). The
/// loading screen is a modal pre-game state with no menu, so the universal cancel
/// inputs all bail out of it (set `should_exit`; the main loop is then skipped):
/// window close, Esc, or gamepad Start / B. Gamepad add/remove is forwarded so a
/// pad opens and its buttons reach us during loading (the RPi target flies on a
/// pad, where there is no window X to click). Resize / display changes notify the
/// renderer so a mid-load swapchain recreate is handled by the next `beginFrame`.
fn pumpLoadingEvents(window: ?*c.SDL_Window, should_exit: *bool, renderer: *Renderer, pad: *gamepad.Gamepad) void {
    if (window == null) return;
    // TODO: touch-only users (no keyboard / gamepad) can't cancel the load yet.
    // Add an on-screen cancel button to the loading screen; touch reaches it via
    // SDL's touch->mouse synthesis, the same path buildPauseButton relies on.
    var event: c.SDL_Event = undefined;
    while (c.SDL_PollEvent(&event)) {
        switch (event.type) {
            c.SDL_EVENT_QUIT => should_exit.* = true,
            c.SDL_EVENT_KEY_DOWN => {
                if (event.key.scancode == c.SDL_SCANCODE_ESCAPE) should_exit.* = true;
            },
            c.SDL_EVENT_GAMEPAD_ADDED, c.SDL_EVENT_GAMEPAD_REMOVED => pad.handleEvent(event),
            c.SDL_EVENT_GAMEPAD_BUTTON_DOWN => {
                if (event.gbutton.button == c.SDL_GAMEPAD_BUTTON_START or event.gbutton.button == c.SDL_GAMEPAD_BUTTON_EAST) {
                    should_exit.* = true;
                }
            },
            c.SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED => renderer.notifyResized(),
            c.SDL_EVENT_WINDOW_DISPLAY_CHANGED, c.SDL_EVENT_WINDOW_HDR_STATE_CHANGED => renderer.notifyDisplayChanged(),
            else => {},
        }
    }
}

/// Build the startup loading screen into `dl`: a full-screen opaque backdrop (so
/// no stale swapchain contents show through), a centered title, a progress bar,
/// and a percent + resident/total readout. `target` is the monotonic max of
/// resident+inFlight, so the bar fills to 100% as the final loads land.
fn drawLoadingScreen(dl: *ui.DrawList, screen: ui.Screen, resident: u32, target: u32) void {
    dl.clear();
    dl.rect(0, 0, screen.w, screen.h, ui.Color.rgba(0.04, 0.05, 0.07, 1.0));

    const frac: f32 = if (target > 0)
        @min(1.0, @as(f32, @floatFromInt(resident)) / @as(f32, @floatFromInt(target)))
    else
        0.0;

    const cx = screen.w * 0.5;
    const cy = screen.h * 0.5;

    const title = "Loading terrain...";
    const title_scale = screen.textScale(2.0);
    const title_w = ui.DrawList.textWidth(title.len, title_scale);
    _ = dl.text(cx - title_w * 0.5, cy - screen.px(48), title_scale, ui.Color.white, title);

    // Progress bar: track then fill, both flat shapes (the glyph stream draws on top).
    const bar_w = screen.px(360);
    const bar_h = screen.px(16);
    const bar_x = cx - bar_w * 0.5;
    const bar_y = cy - bar_h * 0.5;
    dl.rect(bar_x, bar_y, bar_w, bar_h, ui.Color.rgba(0.16, 0.18, 0.22, 1.0));
    dl.rect(bar_x, bar_y, bar_w * frac, bar_h, ui.Color.rgba(0.30, 0.62, 0.92, 1.0));

    var buf: [48]u8 = undefined;
    const pct: u32 = @intFromFloat(frac * 100.0 + 0.5);
    const status = std.fmt.bufPrint(&buf, "{d}%   {d} / {d}", .{ pct, resident, target }) catch "";
    const status_scale = screen.textScale(1.0);
    const status_w = ui.DrawList.textWidth(status.len, status_scale);
    _ = dl.text(cx - status_w * 0.5, bar_y + bar_h + screen.px(14), status_scale, ui.Color.rgba(0.78, 0.84, 0.92, 1.0), status);
}

/// Interactive tile load, used at startup AND on a runtime render-distance
/// increase (the blocking `drainAll` is kept for benchmark / headless runs at both
/// call sites). Phase 1 loads silently in batches, pumping events between each but
/// presenting nothing, so a fast load never flashes a screen. If it crosses
/// `LOAD_PROBE_NS`, phase 2 presents a loading screen per frame: pump events, a
/// non-blocking `recordStream` upload, draw, present, until the streamer drains.
/// It does NOT re-run `tickPolicy` (the caller's tick already requested every
/// in-range tile that fits the pool; the camera is static so no eviction frees
/// slots for more) nor `clipmap.recordUpdate` (the whole-ring refill happens on the
/// next real frame). `total_frames` advances per loading frame so `recordStream`'s
/// staging-slot quarantine stays monotonic.
fn runInteractiveLoad(
    ts: *tile_system_mod.TileSystem,
    renderer: *Renderer,
    scene: *Scene,
    window: ?*c.SDL_Window,
    io: std.Io,
    total_frames: *u64,
    should_exit: *bool,
    pad: *gamepad.Gamepad,
) void {
    const wall_clock: std.Io.Clock = .awake;
    // Drain at the staging ceiling during the screen so loads visibly progress
    // fast and a single final frame can flush every straggler (see `idle` below).
    ts.beginFastFill();

    // Phase 1: silent probe. Drain batches, pumping events between each, no present.
    // The loop's only non-return exit is the timeout break -> fall through to phase 2.
    const probe_t0 = wall_clock.now(io);
    while (true) {
        const more = ts.drainSome() catch |err| {
            std.log.warn("startup tile drainSome failed: {}", .{err});
            return;
        };
        pumpLoadingEvents(window, should_exit, renderer, pad);
        if (should_exit.*) return;
        if (!more) return; // all tiles resident inside the probe window: no screen
        if (Bench.deltaNs(probe_t0, wall_clock.now(io)) >= LOAD_PROBE_NS) break;
    }

    // Phase 2: present a loading screen until the streamer drains.
    var target: u32 = 0;
    while (!should_exit.*) {
        pumpLoadingEvents(window, should_exit, renderer, pad);
        if (should_exit.*) return;
        // A resize / display change during loading destroyed the render pass.
        if (renderer.consumeRenderPassDirty()) scene.rebuildPipelines(renderer) catch return;

        // inFlight()==0 means the worker has pushed every completion and will push
        // no more, so the completion queue is stable: this frame's recordStream
        // (fast-fill cap == staging slots) drains all remaining stragglers in one
        // pass. Present that frame (bar at its true final value) and stop, leaving
        // every loadable tile resident before frame 1.
        const idle = ts.inFlight() == 0;

        const ctx = renderer.beginFrame() catch |err| {
            std.log.err("loading beginFrame failed: {}", .{err});
            return;
        } orelse {
            if (idle) break; // worker done but can't present (e.g. minimized)
            // Can't present yet (minimized / swapchain recreating): throttle so the
            // loading loop doesn't busy-spin the CPU at 100% while waiting for the
            // window to come back. Events are still pumped each iteration above.
            c.SDL_Delay(4);
            continue;
        };
        _ = ts.recordStream(ctx.cmd_buf, total_frames.*);
        target = @max(target, ts.tileCount() + ts.inFlight());

        const w: f32 = @floatFromInt(renderer.swapchain.extent.width);
        const h: f32 = @floatFromInt(renderer.swapchain.extent.height);
        const screen = ui.Screen.fromExtent(w, h, ui.REF_H, ui.DEFAULT_USER_SCALE);
        drawLoadingScreen(&scene.draw_list, screen, ts.tileCount(), target);

        renderer.beginRenderPass(ctx.cmd_buf, ctx.image_index);
        // The loading screen draws no numeric readouts, so the AGL unit scale is moot.
        scene.recordOverlay(ctx.cmd_buf, renderer.current_frame, renderer.swapchain.extent, renderer.transfer_function, false);
        renderer.vkd.cmdEndRenderPass(ctx.cmd_buf);
        renderer.endFrame(ctx) catch |err| {
            std.log.err("loading endFrame failed: {}", .{err});
            return;
        };
        total_frames.* += 1;
        if (idle) break; // worker done and the last stragglers drained this frame
    }
}

pub fn main(init: std.process.Init) !void {
    // ---- Config ----
    var config = Config.parseArgs(init.minimal.args, init.gpa, init.io) catch return;
    defer config.deinit();
    verbose_logging.store(config.debug, .monotonic);
    if (config.debug) Config.dump(config);

    // ---- Allocator ----
    // Init.gpa has leak checking in debug builds already.
    const allocator = init.gpa;
    const io = init.io;
    const wall_clock: std.Io.Clock = .awake;
    const program_start = wall_clock.now(io);

    // ---- Persisted settings (interactive runs only) ----
    // Load + apply BEFORE the renderer/gamepad read `config`, so saved prefs take
    // effect. Gated to interactive runs: benchmark/headless/profile/autotune never
    // read or write the file, keeping their measurements CLI-reproducible.
    // applyToConfig honors CLI-explicit flags (precedence defaults < file < CLI).
    const config_dir = config_file.configDirAlloc(allocator, init.environ_map) catch null;
    defer if (config_dir) |d| allocator.free(d);
    const persist_settings = !config.headless and !config.benchmark and !config.profile and !config.autotune;
    var startup_settings: settings_mod.Settings = .{};
    if (persist_settings) if (config_dir) |dir| {
        startup_settings = settings_mod.Settings.load(allocator, io, dir) orelse .{};
        startup_settings.applyToConfig(&config);
    };

    // ---- SDL3 + window ----
    const win_setup = try window_mod.init(allocator, io, init.environ_map, &config);
    defer window_mod.deinit(win_setup);
    const window = win_setup.window;

    // ---- Renderer (Vulkan infrastructure) ----
    const surface_mode: renderer_mod.SurfaceMode = if (window) |w|
        .{ .windowed = w }
    else
        .{ .headless = .{ .width = @intCast(@max(win_setup.width, 1)), .height = @intCast(@max(win_setup.height, 1)) } };
    var renderer = try Renderer.init(allocator, io, surface_mode, .{
        .validate = config.validate,
        .vsync = config.vsync,
        .bench_enabled = config.benchmark,
        .gpu_override = config.gpu,
        .enable_hdr = !config.no_hdr,
        .msaa_request = config.msaa,
        .verbose = config.debug,
    });
    defer renderer.deinit();

    // V-key A/B toggle: remember the configured MSAA so we can restore it after toggling off.
    const samples_full = renderer.samples;
    const samples_off = vkt.vk.SampleCountFlags{ .@"1_bit" = true };

    // ---- Autotune (runs before clipmap init, may update config.ring_size) ----
    if (try autotune_session.run(allocator, io, config_dir, &renderer, &config, window)) return;

    // ---- Scene (tile system + clipmap + sky + debug text) ----
    var scene = try Scene.init(allocator, io, renderer.gpuCtx(), renderer.render_pass, renderer.samples, &config);
    defer scene.deinit();

    // ---- Session (camera + frame stats + bench) ----
    const init_time_ns = Bench.deltaNs(program_start, wall_clock.now(io));
    var session = try Session.init(allocator, io, &config, init_time_ns);
    defer session.deinit();

    var last_ts = wall_clock.now(io);
    var total_frames: u64 = 0;
    // Default: shown windowed, hidden headless. --hud / --no-hud override either
    // way (e.g. --hud forces it on under --headless --profile to measure cost).
    var hud_visible: bool = config.hud_override orelse !config.headless;

    // Interactive UI: immediate-mode context (persistent focus/state across
    // frames). The context is a main-loop local; it only needs &scene.draw_list
    // each frame. Esc / gamepad Start toggle `menu_open` (see buildMenu).
    var ui_ctx = ui.Ui.init(allocator);
    defer ui_ctx.deinit();
    var menu_open: bool = false;
    var menu_page: MenuPage = .root; // reset to root on every open
    var settings_tab: SettingsTab = .display; // reset when Settings is opened
    var on_settings_prev: bool = false; // edge-detect entering the Settings page (auto-focus the tab bar)
    var pointer_look: pointer_look_mod.PointerLook = .{}; // hold-RMB + touch cockpit free-look seam
    var pending_msaa: ?u32 = null; // menu-requested MSAA, applied between frames (see MenuCtx)
    var pending_place: ?PendingPlace = null; // menu-requested fullscreen/monitor, applied between frames
    var pending_rebuild: ?RingRebuild = null; // menu Apply / auto-revert, applied between frames
    // Render-distance staging (Graphics tab). Start at the live clipmap sizes; the
    // sliders edit these, Apply commits, and every rebuild resyncs them to the
    // clipmap's actual (chunk-aligned) values. revert_guard arms the auto-revert.
    var staged_ring: i32 = @intCast(scene.clipmap.ring_size);
    var staged_levels: i32 = @intCast(scene.clipmap.num_levels);
    var revert_guard: ?RevertGuard = null;
    var apply_render_dist = false;
    var keep_render_dist = false;

    // MSAA cycle options, filtered to GPU support once (the query re-reads device
    // limits, so it is not per-frame). Labels parallel counts; the cycle's active
    // index is derived fresh each frame from `renderer.samples` in buildSettings,
    // so no index is kept here.
    const msaa_candidates = [_]u32{ 1, 2, 4, 8 };
    const msaa_candidate_labels = [_][]const u8{ "Off", "2x", "4x", "8x" };
    var msaa_counts_buf: [msaa_candidates.len]u32 = undefined;
    var msaa_labels_buf: [msaa_candidates.len][]const u8 = undefined;
    var msaa_n: usize = 0;
    {
        const mask = renderer.supportedSampleCounts();
        for (msaa_candidates, msaa_candidate_labels) |cnt, lbl| {
            if (renderer_mod.countSupported(mask, cnt)) {
                msaa_counts_buf[msaa_n] = cnt;
                msaa_labels_buf[msaa_n] = lbl;
                msaa_n += 1;
            }
        }
    }
    const msaa_options = MsaaOptions{ .labels = msaa_labels_buf[0..msaa_n], .counts = msaa_counts_buf[0..msaa_n] };

    // Monitor options, built once. Index-based labels avoid SDL display-name
    // lifetime issues. Monitor select works on X11/Windows; Wayland compositors
    // ignore client window positioning, and a lone monitor has nothing to choose,
    // so the cycle is greyed out in those cases.
    // TODO(hotplug): labels + monitor_selectable are frozen at startup; rebuild them
    // on SDL_EVENT_DISPLAY_ADDED/REMOVED so the list tracks connect/disconnect.
    const MAX_MONITORS = 8;
    var mon_label_storage: [MAX_MONITORS][16]u8 = undefined;
    var mon_labels_buf: [MAX_MONITORS][]const u8 = undefined;
    const mon_count = @min(display.count(), MAX_MONITORS);
    for (0..mon_count) |i| {
        mon_labels_buf[i] = std.fmt.bufPrint(&mon_label_storage[i], "Monitor {d}", .{i + 1}) catch "Monitor";
    }
    const monitor_labels = mon_labels_buf[0..mon_count];
    const is_wayland = blk: {
        const drv = c.SDL_GetCurrentVideoDriver();
        break :blk drv != null and std.mem.eql(u8, std.mem.span(drv), "wayland");
    };
    const monitor_selectable = mon_count > 1 and !is_wayland;

    // Single gamepad (first connected) flying via the Controls seam. Opened on
    // SDL_EVENT_GAMEPAD_ADDED, which SDL also posts for already-connected pads at
    // init, so this covers startup and hot-plug. No-op when none is connected.
    var pad = gamepad.Gamepad.init();
    defer pad.deinit();
    // Saved input prefs -> live gamepad tuning (the cross-tree copy settings.zig
    // leaves to the caller). No-op when nothing was loaded (defaults match). The
    // deadzones are clamped against a hand-edited settings.zon for the same reason
    // as the sensitivities below: a value >= 1 deadens the stick entirely and a
    // negative one makes a centered stick read as a phantom deflection.
    pad.tuning.deadzone = std.math.clamp(startup_settings.input.deadzone, 0.0, 0.9);
    pad.tuning.invert_pitch = startup_settings.input.invert_pitch;
    pad.tuning.trigger_deadzone = std.math.clamp(startup_settings.input.trigger_deadzone, 0.0, 0.9);
    // Both look sensitivities are clamped against a hand-edited settings.zon so a
    // negative / huge value can't invert or saturate the glance (neither has a CLI
    // flag). Mouse sensitivity also lives on Config (the menu binds it there).
    pad.tuning.look_sensitivity = std.math.clamp(startup_settings.input.look_sensitivity, 0.1, 12.0);
    config.look_drag_sensitivity = std.math.clamp(startup_settings.input.look_drag_sensitivity, 0.05, 5.0);
    // Persist current prefs on exit (interactive only). Captured at scope exit so
    // runtime changes (V/H keys, settings menu) are saved too.
    defer if (persist_settings) if (config_dir) |dir| {
        // Reconcile fullscreen from the live SDL flag: F11 / Alt+Enter toggle it
        // directly (handleWindowKey) without touching config, so capture the truth.
        if (window) |win| config.fullscreen = (c.SDL_GetWindowFlags(win) & c.SDL_WINDOW_FULLSCREEN) != 0;
        var out: settings_mod.Settings = .{};
        out.captureFromConfig(&config);
        out.input = .{
            .deadzone = pad.tuning.deadzone,
            .invert_pitch = pad.tuning.invert_pitch,
            .trigger_deadzone = pad.tuning.trigger_deadzone,
            .look_sensitivity = pad.tuning.look_sensitivity,
            .look_drag_sensitivity = config.look_drag_sensitivity,
        };
        out.save(allocator, io, dir) catch |err| std.log.warn("settings save failed: {}", .{err});
    };

    // Synchronously load every tile the camera-start view needs BEFORE the first
    // frame, so frame 1 renders fully populated (one whole-ring refill, then
    // steady fps) instead of the slow capped churn over many frames. The soft
    // per-frame fill can't beat this: tile throughput is frame-rate-bound (the
    // streamer's staging slots recycle on a per-frame quarantine), so at capped
    // fps the ring only trickles in. drainAll releases staging inline between
    // one-shot submits, so it is NOT frame-gated. Profile/autotune already relied
    // on this for clean measurement; now every run with a DB does it.
    // TODO: startup tile-load churn (incremental refill) is still open.
    // `should_exit` is declared here (not at the main loop) so a quit during the
    // interactive load below can skip the main loop entirely.
    var should_exit = false;
    var tile_load_ns: u64 = 0;
    var tile_load_count: u32 = 0;
    if (scene.tile_system) |ts| {
        const start = config.cameraStartPos();
        ts.tickPolicy(.{
            .pos_xz = .{ start[0], start[2] },
            .velocity_xz = .{ 0, 0 },
            .front = .{ 0, 0, -1 },
        }, 0);
        const drain_t0 = wall_clock.now(io);
        if (window != null and !config.benchmark) {
            // Interactive: load silently for a short probe, then present a loading
            // screen if it is still going (see runInteractiveLoad). A quit during
            // the load sets should_exit so the main loop is skipped.
            runInteractiveLoad(ts, &renderer, &scene, window, io, &total_frames, &should_exit, &pad);
        } else {
            // Benchmark / headless / autotune: blocking load for deterministic
            // timing (captures tile_load_ns; no window to present a screen to).
            ts.drainAll() catch |err| std.log.warn("startup tile drainAll failed: {}", .{err});
        }
        tile_load_ns = Bench.deltaNs(drain_t0, wall_clock.now(io));
        tile_load_count = ts.tileCount();
    }

    if (config.profile) {
        try session.attachProfile(allocator, .{
            .tile_load_ns = tile_load_ns,
            .tile_count = tile_load_count,
        });
    }

    // Under --debug the verbose enumerations above already cover GPU/Display/Clipmap;
    // the consolidated info lines would just duplicate them.
    if (!config.debug) {
        var extra_buf: [96]u8 = undefined;
        const display_extra = std.fmt.bufPrint(&extra_buf, ", {s}, MSAA {d}x, vsync {s}", .{
            swapchain_mod.SurfaceFormatChoice.label(renderer.transfer_function),
            renderer_mod.sampleCountToInt(renderer.samples),
            if (config.vsync) "on" else "off",
        }) catch "";
        display.logActive(window, display_extra);

        var detail_buf: [64]u8 = undefined;
        const tile_detail = if (scene.tile_system) |ts|
            std.fmt.bufPrint(&detail_buf, "streaming {d} tiles, {d:.1} MB VRAM", .{
                ts.catalogCount(),
                scene.clipmap.vramUsageMB() + @as(f32, @floatFromInt(ts.weightsSize())) / (1024.0 * 1024.0),
            }) catch "?"
        else
            "procedural terrain";
        const init_ms = @as(f64, @floatFromInt(init_time_ns)) / 1e6;
        std.log.info("Clipmap: {d} levels, {d}x{d} ring, {s}, init {d:.0}ms", .{
            config.num_levels, scene.clipmap.ring_size, scene.clipmap.ring_size, tile_detail, init_ms,
        });
    }
    if (scene.tile_system != null) {
        std.log.info("Ready: camera at {s}", .{config.tile});
    } else {
        std.log.info("Ready: procedural terrain", .{});
    }

    // Rebase the dt clock past the (potentially multi-second) startup load: the
    // loading screen is a non-interactive pause, so frame 1 must not integrate it
    // into the flight model and jolt the aircraft forward on launch. Mirrors the
    // render-distance load rebase in the pending_rebuild block.
    last_ts = wall_clock.now(io);

    while (!should_exit) {
        // Neutral UI input snapshot for this frame: key/nav edges accumulate
        // during the poll (only while the menu captures input); pointer level is
        // sampled after. Reset each frame so edges are one-frame pulses.
        var ui_input: ui.InputState = .{};
        pointer_look.beginFrame(); // clear this frame's accumulated pointer-look motion
        if (window) |_| {
            // Stable pointers into long-lived state; build once per frame and
            // reuse across this frame's events (handlers mutate through the
            // pointers, so the struct itself never changes).
            var ctx = input.Context{
                .hud_visible = &hud_visible,
                .camera = &session.camera,
                .aircraft = &session.aircraft,
                .config = &config,
                .renderer = &renderer,
                .clipmap = &scene.clipmap,
                .samples_off = samples_off,
                .samples_full = samples_full,
            };
            var event: c.SDL_Event = undefined;
            while (c.SDL_PollEvent(&event)) {
                switch (event.type) {
                    c.SDL_EVENT_QUIT => {
                        should_exit = true;
                    },
                    // Tiered key routing: (1) window keys (fullscreen) always win,
                    // regardless of menu state; (2) Esc opens the menu when closed;
                    // (3) while the menu is open it captures the keyboard (keys
                    // become UI focus/nav edges, Esc backs out/closes); (4) otherwise
                    // the gameplay one-shot handler runs.
                    c.SDL_EVENT_KEY_DOWN => {
                        if (input.handleWindowKey(event, window)) {
                            // consumed (fullscreen toggle)
                        } else if (event.key.scancode == c.SDL_SCANCODE_ESCAPE and !event.key.repeat and !menu_open) {
                            menu_open = true;
                            menu_page = .root; // every open starts at the root page
                            ui_ctx.clearFocus(); // fresh focus on open
                        } else if (menu_open) {
                            input.uiKeyEdge(&ui_input, event);
                        } else {
                            input.handleKeyDown(event, &ctx);
                        }
                    },
                    c.SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED => {
                        renderer.notifyResized();
                    },
                    c.SDL_EVENT_WINDOW_DISPLAY_CHANGED, c.SDL_EVENT_WINDOW_HDR_STATE_CHANGED => {
                        renderer.notifyDisplayChanged();
                    },
                    c.SDL_EVENT_GAMEPAD_ADDED, c.SDL_EVENT_GAMEPAD_REMOVED => {
                        pad.handleEvent(event);
                    },
                    // Gamepad mirrors the key tiers: Start toggles the menu; while
                    // open, buttons become UI nav edges (d-pad nav, A activate, B
                    // back/close); otherwise the gameplay button handler runs.
                    c.SDL_EVENT_GAMEPAD_BUTTON_DOWN => {
                        if (event.gbutton.button == c.SDL_GAMEPAD_BUTTON_START) {
                            menu_open = !menu_open;
                            if (menu_open) menu_page = .root; // open starts at root
                            ui_ctx.clearFocus();
                        } else if (menu_open) {
                            input.uiGamepadEdge(&ui_input, event.gbutton.button);
                        } else {
                            input.handleGamepadButton(event, &ctx);
                        }
                    },
                    else => {},
                }
                // Pointer free-look: accumulate mouse motion + drive the finger
                // lifecycle (non-pointer events are ignored). engage_ok gates
                // starting a glance to a glance-capable mode with the menu closed.
                pointer_look.handleEvent(event, !menu_open and session.camera.mode == .cockpit);
            }

            // Pointer level for the UI, in device px (window logical px ==
            // swapchain px while high-DPI density is unset). Only meaningful when
            // the menu is open; sampling unconditionally is harmless.
            var mx: f32 = 0;
            var my: f32 = 0;
            const btns = c.SDL_GetMouseState(&mx, &my);
            ui_input.pointer_x = mx;
            ui_input.pointer_y = my;
            ui_input.pointer_down = (btns & (@as(u32, 1) << @intCast(c.SDL_BUTTON_LEFT - 1))) != 0;
            const ks: ?[*]const bool = c.SDL_GetKeyboardState(null);
            if (ks) |k| ui_input.shift = k[c.SDL_SCANCODE_LSHIFT] or k[c.SDL_SCANCODE_RSHIFT];

            // Pointer free-look engage/disengage (hold-RMB relative mouse mode +
            // touch), gated on focus + menu-closed + a glance-capable mode. Losing
            // focus or opening the menu auto-disengages (the Esc / menu path is the
            // escape hatch so the cursor can't get stuck locked); a disengage springs
            // the glance back. See app/pointer_look.zig.
            const rmb_down = (btns & (@as(u32, 1) << @intCast(c.SDL_BUTTON_RIGHT - 1))) != 0;
            const focused = if (window) |win| (c.SDL_GetWindowFlags(win) & c.SDL_WINDOW_INPUT_FOCUS) != 0 else false;
            pointer_look.update(window, focused and !menu_open and session.camera.mode == .cockpit, rmb_down);
            if (pointer_look.takeRecenter()) session.camera.requestRecenter();
        }

        // Apply a menu-requested MSAA change here, between frames: setSampleCount
        // tears down the render pass + framebuffers, which is unsafe mid-recording
        // (the menu sets this from inside the render pass). Same safe timing the V
        // key uses. On success it flags render_pass_dirty, handled just below; on
        // failure the renderer keeps its prior count and the cycle re-derives.
        if (pending_msaa) |n| {
            pending_msaa = null;
            if (renderer.setMsaa(n)) {
                config.msaa = n;
            } else |err| {
                std.log.err("setMsaa({d}) failed: {}", .{ n, err });
            }
        }

        // Apply a menu-requested window placement here, between frames: the SDL
        // fullscreen/monitor change triggers a swapchain recreate that must not run
        // mid-render-pass. Persist the resulting state for next launch.
        if (pending_place) |p| {
            pending_place = null;
            if (window) |win| {
                display.place(win, display.byIndex(p.monitor), p.borderless);
                config.monitor = p.monitor;
                config.fullscreen = p.borderless;
            }
        }

        // The renderer destroyed the old render pass (V key / menu MSAA change or
        // surface format change); pipelines bound to it would crash on next draw.
        if (renderer.consumeRenderPassDirty()) {
            scene.rebuildPipelines(&renderer) catch break;
        }

        // Apply a render-distance change here, between frames: rebuildClipmap does
        // a deviceWaitIdle + full clipmap teardown/rebuild, unsafe mid-render-pass.
        // Built against the now-settled render pass + sample count. Set by the
        // Graphics-tab Apply button or the auto-revert (see below).
        if (pending_rebuild) |rb| {
            pending_rebuild = null;
            scene.rebuildClipmap(allocator, renderer.gpuCtx(), renderer.render_pass, renderer.samples, config.base_spacing, rb.ring_size, rb.num_levels) catch |err| {
                std.log.err("clipmap rebuild ({d} levels) failed: {}", .{ rb.num_levels, err });
            };
            // Resync the menu's staged values to the clipmap's actual (chunk-aligned)
            // sizes so the sliders show the truth after a rebuild. `config` is NOT
            // updated here: only a confirmed (Keep) change persists; an un-kept Apply
            // reverts, and its transient value must not reach settings.zon (below).
            staged_ring = @intCast(scene.clipmap.ring_size);
            staged_levels = @intCast(scene.clipmap.num_levels);
            // Match the tile system's desired (visible) set to the new render
            // distance and load any now-in-range tiles BEFORE the rebuilt (empty)
            // ring re-evaluates this frame. Without this an INCREASE grows the ring
            // but the policy still wants only the old, smaller set, so the larger
            // ring renders with empty outer terrain. Same flush path as cold start:
            // tickPolicy (anchored at the current camera pos, velocity 0, so the
            // just-rebuilt ring is filled rather than prefetched ahead) followed by
            // the interactive loading screen, which keeps the window responsive
            // (pumps events + presents) so the compositor doesn't flag a hang on a
            // slow load. A bare drainAll here froze the event loop -> "not
            // responding". A decrease loads nothing (the extras simply evict).
            if (scene.tile_system) |ts| {
                ts.setRenderDistance(scene.clipmap.ring_size, scene.clipmap.num_levels, config.base_spacing);
                ts.tickPolicy(.{
                    .pos_xz = .{ session.camera.pose.position[0], session.camera.pose.position[2] },
                    .velocity_xz = .{ 0, 0 },
                    .front = session.camera.pose.front(),
                }, total_frames);
                if (window != null and !config.benchmark) {
                    runInteractiveLoad(ts, &renderer, &scene, window, io, &total_frames, &should_exit, &pad);
                    if (should_exit) break;
                } else {
                    ts.drainAll() catch |err| std.log.warn("render-distance tile drainAll failed: {}", .{err});
                }
            }
            // The rebuild (deviceWaitIdle) + load is a non-interactive pause, so
            // neither the flight model (dt) nor the auto-revert Keep countdown
            // should count it. Rebase both past the load: without rebasing
            // revert_guard.start, a load longer than REVERT_NS (slow disk / RPi /
            // big bump) would fire the revert on the very first frame back, before
            // the user can click Keep. Next frame resumes normal dt.
            const after_load = wall_clock.now(io);
            last_ts = after_load;
            if (revert_guard) |*g| g.start = after_load;
        }

        const now_ts = wall_clock.now(io);

        // Render-distance auto-revert: a just-applied change reverts itself unless
        // the user confirms within REVERT_NS. Ticks in wall-clock time even with
        // the menu closed, so a setting that craters fps can't strand you. The
        // remaining seconds drive the Keep prompt (revert_seconds, below).
        var revert_seconds: ?u32 = null;
        if (revert_guard) |g| {
            const elapsed = Bench.deltaNs(g.start, now_ts);
            if (elapsed >= REVERT_NS) {
                pending_rebuild = g.prev; // reverts next frame's apply block
                revert_guard = null;
            } else {
                revert_seconds = @intCast((REVERT_NS - elapsed + std.time.ns_per_s - 1) / std.time.ns_per_s);
            }
        }
        const dt_ns = Bench.deltaNs(last_ts, now_ts);
        const dt: f32 = @floatCast(@as(f64, @floatFromInt(dt_ns)) / 1e9);
        last_ts = now_ts;

        var aircraft_render_pose = session.aircraft.pose;
        if (session.profile) |*p| {
            if (p.shouldFly()) session.camera.autopilot(dt);
        } else if (config.benchmark_fly) {
            session.camera.autopilot(dt);
        } else if (!menu_open) {
            // Menu open => camera/flight input suspended (the panel captures input).
            const keys = if (window != null) c.SDL_GetKeyboardState(null) else null;
            switch (session.camera.mode) {
                .free => {
                    session.ticker.reset(session.aircraft.pose);
                    camera_mod.applyFreeFlyInput(&session.camera.pose, dt, keys, session.camera.speed);
                    if (pad.readFreeLook()) |fl|
                        camera_mod.applyFreeFlyGamepad(&session.camera.pose, dt, fl.move_fwd, fl.move_right, fl.move_up, fl.look_yaw, fl.look_pitch, fl.boost, &session.camera.speed);
                },
                .cockpit => {
                    const interp = advanceAircraft(&session, config.flight_source, dt, controls_mod.merge(controls_mod.fromKeyboard(keys), pad.readControls()));
                    aircraft_render_pose = interp;
                    // Free-look glance offsets the view from the nose (right stick /
                    // hold-RMB / touch drag), holds when idle, recenters on R3 /
                    // release / lift. The pointer seam merges mouse + touch into one
                    // px drag delta (touch scaled by screen size). See pointer_look.zig.
                    const d = pointer_look.drag(
                        @floatFromInt(renderer.swapchain.extent.width),
                        @floatFromInt(renderer.swapchain.extent.height),
                    );
                    applyAttachedGlance(&session.camera, interp, &pad, d.active, d.dx, d.dy, config.look_drag_sensitivity, dt);
                    session.camera.updateFlightFov(session.aircraft.airspeed, dt);
                },
                .chase => {
                    const interp = advanceAircraft(&session, config.flight_source, dt, controls_mod.merge(controls_mod.fromKeyboard(keys), pad.readControls()));
                    aircraft_render_pose = interp;
                    camera_mod.updateChaseCamera(&session.camera, interp, dt);
                    session.camera.updateFlightFov(session.aircraft.airspeed, dt);
                },
            }
            session.camera.updateOptics(dt, keys);
        }

        // -- Begin frame (cmd buffer, NO render pass yet) --
        const ctx = renderer.beginFrame() catch |err| {
            std.log.err("beginFrame failed: {}", .{err});
            break;
        } orelse {
            if (session.bench) |*b| b.recordSkip();
            continue;
        };
        const record_t0 = std.Io.Clock.awake.now(io);

        // -- Compute: update dirty strips (before render pass) --
        const w: f32 = @floatFromInt(renderer.swapchain.extent.width);
        const h: f32 = @floatFromInt(renderer.swapchain.extent.height);
        const aspect = if (h > 0) w / h else 1.0;

        if (scene.tile_system) |ts| ts.tickPolicy(.{
            .pos_xz = .{ session.camera.pose.position[0], session.camera.pose.position[2] },
            .velocity_xz = .{ session.camera.pose.velocity[0], session.camera.pose.velocity[2] },
            .front = session.camera.pose.front(),
        }, total_frames);

        scene.clipmap.recordUpdate(ctx.cmd_buf, session.camera.pose.position, renderer.current_frame, total_frames, session.camera.fov, h);
        if (scene.probe) |*p| p.recordDispatch(ctx.cmd_buf, renderer.current_frame, aircraft_render_pose.position);

        const scene_params = clipmap_mod.buildSceneParams(&session.camera, aspect, .{
            .fog_max_dist = if (config.no_fog) 1e9 else scene.clipmap.currentFogMaxDist(@floatCast(session.camera.pose.position[2])),
            .no_effects = config.no_effects,
            .transfer_function = renderer.transfer_function,
            // TAWS colors terrain by the aircraft's clearance, so use the
            // aircraft pose's altitude (== probe input), not the camera's.
            .aircraft_alt_arcsec = @floatCast(aircraft_render_pose.position[1]),
            .taws = config.taws,
        });

        total_frames += 1;
        session.frame_stats.push(@as(f64, @floatFromInt(dt_ns)) / 1e9);

        renderer.writeGraphicsTimestamps(ctx.cmd_buf);

        renderer.beginRenderPass(ctx.cmd_buf, ctx.image_index);
        scene.clipmap.recordDraw(ctx.cmd_buf, scene_params, renderer.current_frame, session.camera.fov, h);
        if (!config.no_sky) scene.sky.draw(ctx.cmd_buf, scene.clipmap.desc_sets[renderer.current_frame]);
        scene.drawAircraft(ctx.cmd_buf, &session.camera, aircraft_render_pose, aspect);

        // Overlay pass: HUD + interactive UI share ONE DrawList, recorded once
        // (the backend re-uploads the whole list per frame). The HUD is suppressed
        // while the menu is open: the menu's background lives on the shape stream,
        // which renders before HUD strokes/text, so it cannot occlude them; hiding
        // the HUD avoids that bleed (a settings menu replaces the scene anyway).
        const imperial = config.units == .imperial;
        scene.buildHud(hud_visible and !menu_open, &session.camera, &session.aircraft, aircraft_render_pose.front(), &session.frame_stats, renderer.swapchain.extent, imperial);
        const ui_screen = ui.Screen.fromExtent(w, h, ui.REF_H, ui.DEFAULT_USER_SCALE);
        ui_ctx.beginFrame(ui_input, ui_screen, &scene.draw_list);
        // On the frame we first show the Settings page, focus its first widget (the
        // tab bar) so a controller's Left/Right act immediately. Edge-detected from
        // the page state, which already reflects last frame's navigation.
        const on_settings = menu_open and menu_page == .settings;
        if (on_settings and !on_settings_prev) ui_ctx.focusFirst();
        on_settings_prev = on_settings;
        if (menu_open) {
            buildMenu(&ui_ctx, .{
                .open = &menu_open,
                .page = &menu_page,
                .tab = &settings_tab,
                .should_exit = &should_exit,
                .config = &config,
                .tuning = &pad.tuning,
                .renderer = &renderer,
                .msaa = msaa_options,
                .pending_msaa = &pending_msaa,
                .window = window,
                .monitor_labels = monitor_labels,
                .monitor_selectable = monitor_selectable,
                .pending_place = &pending_place,
                .rdist = .{
                    .staged_ring = &staged_ring,
                    .staged_levels = &staged_levels,
                    .live_ring = scene.clipmap.ring_size,
                    .live_levels = scene.clipmap.num_levels,
                    .apply = &apply_render_dist,
                    .keep = &keep_render_dist,
                    .revert_seconds = revert_seconds,
                },
            });
        } else if (window != null and !config.benchmark) {
            // On-screen pause for pointer/touch (no Esc/Start on a touchscreen).
            if (buildPauseButton(&ui_ctx)) {
                menu_open = true;
                menu_page = .root;
                ui_ctx.clearFocus();
            }
        }
        ui_ctx.endFrame();
        // Esc / B backs out one level: a sub-page returns to root, root closes the
        // menu. consumeClose fires on the single cancel press (see endFrame).
        if (ui_ctx.consumeClose()) {
            if (menu_page != .root) {
                menu_page = .root;
                ui_ctx.clearFocus();
            } else {
                menu_open = false;
            }
        }

        // Render-distance Apply/Keep, set by the menu this frame. Apply arms the
        // auto-revert (snapshotting the current sizes) and queues the rebuild for
        // next frame's apply block; Keep confirms (disarms the revert). Both queue
        // a between-frames rebuild rather than rebuilding mid-render-pass.
        if (apply_render_dist) {
            apply_render_dist = false;
            revert_guard = .{
                .prev = .{ .ring_size = scene.clipmap.ring_size, .num_levels = scene.clipmap.num_levels },
                .start = now_ts,
            };
            pending_rebuild = .{ .ring_size = @intCast(staged_ring), .num_levels = @intCast(staged_levels) };
        }
        if (keep_render_dist) {
            keep_render_dist = false;
            revert_guard = null;
            // Confirmed: fold the live sizes into config so the exit-save persists
            // them to settings.zon (interactive runs). Only Keep writes config.
            config.ring_size = scene.clipmap.ring_size;
            config.num_levels = scene.clipmap.num_levels;
        }

        scene.recordOverlay(ctx.cmd_buf, renderer.current_frame, renderer.swapchain.extent, renderer.transfer_function, imperial);
        renderer.vkd.cmdEndRenderPass(ctx.cmd_buf);

        const record_ns = Bench.deltaNs(record_t0, wall_clock.now(io));
        renderer.endFrame(ctx) catch |err| {
            std.log.err("endFrame failed: {}", .{err});
            break;
        };

        var sample = renderer.lastSample();
        sample.scrolling = (scene.clipmap.last_strips + scene.clipmap.last_refills) > 0;
        sample.cpu_record_ns = record_ns;

        if (session.profile) |*p| {
            if (p.tick(dt_ns, sample) == .done) should_exit = true;
        } else if (session.bench) |*b| {
            b.recordFrame(sample);
            if (config.benchmark_frames != 0 and total_frames >= config.benchmark_frames) {
                should_exit = true;
            }
        }
    }

    if (session.profile) |*p| p.report();
    if (session.bench) |*b| b.report();

    std.log.info("Exiting.", .{});
}
