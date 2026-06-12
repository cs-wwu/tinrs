//! SDL display enumeration and Wayland-correct window placement.
//!
//! On Wayland (Mutter, KWin) and some X11 compositors, `SDL_SetWindowPosition`
//! is silently ignored on a window that's already fullscreen; the compositor
//! owns geometry while a window is fullscreen. To put a window on a specific
//! display in fullscreen, you must position it first, then enter fullscreen.
//! `place()` does this in the right order regardless of starting state.
//!
//! Requires `SDL_INIT_VIDEO`.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const c = vkt.c;

pub const Display = struct {
    id: c.SDL_DisplayID,
    w: i32,
    h: i32,
    refresh_hz: f32,
    hdr_enabled: bool,
    is_primary: bool,
};

fn fromMode(id: c.SDL_DisplayID, mode: *const c.SDL_DisplayMode) Display {
    // SDL_GetDisplayProperties returns 0 on failure; SDL_GetBooleanProperty
    // accepts that and falls back to the default. SDL_GetPrimaryDisplay is a
    // cheap getter; fine to call per display.
    const props = c.SDL_GetDisplayProperties(id);
    return .{
        .id = id,
        .w = mode.*.w,
        .h = mode.*.h,
        .refresh_hz = mode.*.refresh_rate,
        .hdr_enabled = c.SDL_GetBooleanProperty(props, c.SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, false),
        .is_primary = id == c.SDL_GetPrimaryDisplay(),
    };
}

/// Display at the given user-facing index (0-based). Null if out of range or
/// if SDL can't enumerate displays.
pub fn byIndex(idx: u32) ?Display {
    var n: c_int = 0;
    const ids_ptr = c.SDL_GetDisplays(&n) orelse {
        std.log.warn("SDL_GetDisplays failed: {s}", .{c.SDL_GetError()});
        return null;
    };
    defer c.SDL_free(ids_ptr);
    if (n <= 0) return null;
    if (idx >= @as(u32, @intCast(n))) return null;
    const id = ids_ptr[idx];
    const mode = c.SDL_GetDesktopDisplayMode(id) orelse {
        std.log.warn("SDL_GetDesktopDisplayMode({d}) failed: {s}", .{ id, c.SDL_GetError() });
        return null;
    };
    return fromMode(id, mode);
}

/// Number of connected displays (0 if SDL can't enumerate). Used to size the
/// settings-menu monitor list.
pub fn count() usize {
    var n: c_int = 0;
    const ids = c.SDL_GetDisplays(&n) orelse return 0;
    defer c.SDL_free(ids);
    return if (n > 0) @intCast(n) else 0;
}

/// 0-based index (in `SDL_GetDisplays` order, matching `byIndex`) of the display
/// the window is currently on. 0 if headless or undeterminable.
pub fn indexForWindow(window: ?*c.SDL_Window) usize {
    const win = window orelse return 0;
    const cur = c.SDL_GetDisplayForWindow(win);
    if (cur == 0) return 0;
    var n: c_int = 0;
    const ids = c.SDL_GetDisplays(&n) orelse return 0;
    defer c.SDL_free(ids);
    if (n <= 0) return 0;
    for (0..@intCast(n)) |i| if (ids[i] == cur) return i;
    return 0;
}

/// Highest-refresh display; ties broken by larger area. Used by autotune to
/// pick the display whose refresh rate sets the frame budget.
pub fn best() ?Display {
    var n: c_int = 0;
    const ids_ptr = c.SDL_GetDisplays(&n) orelse {
        std.log.warn("SDL_GetDisplays failed: {s}", .{c.SDL_GetError()});
        return null;
    };
    defer c.SDL_free(ids_ptr);
    if (n <= 0) return null;
    const ids: []const c.SDL_DisplayID = ids_ptr[0..@intCast(n)];

    var winner: ?Display = null;
    for (ids) |id| {
        const mode = c.SDL_GetDesktopDisplayMode(id) orelse {
            std.log.warn("SDL_GetDesktopDisplayMode({d}) failed: {s}", .{ id, c.SDL_GetError() });
            continue;
        };
        const d = fromMode(id, mode);
        if (winner) |w| {
            const better_rate = d.refresh_hz > w.refresh_hz;
            const same_rate = d.refresh_hz == w.refresh_hz;
            const better_area = (@as(i64, d.w) * @as(i64, d.h)) > (@as(i64, w.w) * @as(i64, w.h));
            if (better_rate or (same_rate and better_area)) winner = d;
        } else {
            winner = d;
        }
    }
    return winner;
}

/// Display the given window is currently on. Null if SDL can't determine it.
pub fn forWindow(window: *c.SDL_Window) ?Display {
    const id = c.SDL_GetDisplayForWindow(window);
    if (id == 0) {
        std.log.warn("SDL_GetDisplayForWindow failed: {s}", .{c.SDL_GetError()});
        return null;
    }
    const mode = c.SDL_GetDesktopDisplayMode(id) orelse {
        std.log.warn("SDL_GetDesktopDisplayMode({d}) failed: {s}", .{ id, c.SDL_GetError() });
        return null;
    };
    return fromMode(id, mode);
}

/// Place a window. If `target` is non-null, position the window centered on
/// that display (un-fullscreening first if necessary, since SetWindowPosition
/// is a no-op on fullscreen windows under Wayland/Mutter). If `fullscreen`,
/// then enter borderless desktop fullscreen.
pub fn place(window: *c.SDL_Window, target: ?Display, fullscreen: bool) void {
    if (target) |t| {
        _ = c.SDL_SetWindowFullscreen(window, false);
        const pos = c.SDL_WINDOWPOS_CENTERED_DISPLAY(t.id);
        _ = c.SDL_SetWindowPosition(window, @intCast(pos), @intCast(pos));
        _ = c.SDL_SyncWindow(window);
    }
    if (fullscreen) {
        _ = c.SDL_SetWindowFullscreenMode(window, null); // null = borderless desktop fullscreen
        _ = c.SDL_SetWindowFullscreen(window, true);
        _ = c.SDL_SyncWindow(window);
    } else if (target == null) {
        // Honor a windowed request even with no reposition target: with a target the
        // block above already un-fullscreened, but a bare place(win, null, false)
        // must still exit fullscreen (else a "Windowed" menu pick is a silent no-op).
        _ = c.SDL_SetWindowFullscreen(window, false);
        _ = c.SDL_SyncWindow(window);
    }
}

/// Log every display at debug level. Selection marker `*` goes on the entry
/// whose SDL_DisplayID matches `highlight_id` (pass 0 for no marker).
/// Use `logActive` for the default-output one-line summary.
pub fn logAll(highlight_id: c.SDL_DisplayID) void {
    var n: c_int = 0;
    const ids_ptr = c.SDL_GetDisplays(&n) orelse {
        std.log.warn("SDL_GetDisplays failed: {s}", .{c.SDL_GetError()});
        return;
    };
    defer c.SDL_free(ids_ptr);
    if (n <= 0) return;
    const ids: []const c.SDL_DisplayID = ids_ptr[0..@intCast(n)];

    std.log.debug("Displays:", .{});
    for (ids, 0..) |id, idx| {
        const mode = c.SDL_GetDesktopDisplayMode(id) orelse {
            std.log.warn("SDL_GetDesktopDisplayMode({d}) failed: {s}", .{ id, c.SDL_GetError() });
            continue;
        };
        const d = fromMode(id, mode);
        const name_ptr = c.SDL_GetDisplayName(id);
        const name: []const u8 = if (name_ptr != null) std.mem.span(name_ptr) else "<unknown>";
        const hdr_tag: []const u8 = if (d.hdr_enabled) " HDR" else "";
        const primary_tag: []const u8 = if (d.is_primary) " primary" else "";
        const sel: u8 = if (id == highlight_id) '*' else ' ';
        std.log.debug("  [{d}]{c} {s} {d}x{d} @ {d}Hz{s}{s} (id={d})", .{
            idx,           sel,            name, d.w, d.h,
            @as(i32, @intFromFloat(d.refresh_hz)),
            hdr_tag,       primary_tag,    d.id,
        });
    }
}

/// One-line summary of the active display. Used for default-mode output.
/// `extra` is appended verbatim (e.g., ", HDR10/PQ, MSAA 4x, vsync on") so
/// callers can fold rendering details into the same line. The display's HDR
/// state is not surfaced here; callers append a colorspace label if relevant.
pub fn logActive(window: ?*c.SDL_Window, extra: []const u8) void {
    const d: ?Display = if (window) |w| forWindow(w) else null;
    if (d) |entry| {
        const name_ptr = c.SDL_GetDisplayName(entry.id);
        const name: []const u8 = if (name_ptr != null) std.mem.span(name_ptr) else "<unknown>";
        std.log.info("Display: {s} {d}x{d} @ {d}Hz{s}", .{
            name,                                         entry.w, entry.h,
            @as(i32, @intFromFloat(entry.refresh_hz)),    extra,
        });
    } else {
        std.log.info("Display: headless{s}", .{extra});
    }
}
