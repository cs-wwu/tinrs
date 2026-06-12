//! Pointer free-look producer: folds hold-RMB mouse drag and one-finger touch
//! drag into one "glance" delta, decoupled from the camera the way `controls`/
//! `gamepad` are decoupled from the aircraft. Main glues the output to the Camera
//! glance (applyAttachedGlance + requestRecenter), so cockpit and the upcoming
//! sensor mode reuse ONE pointer state machine instead of duplicating it.
//!
//! Per-frame lifecycle:
//!   beginFrame()                         // clear this frame's motion
//!   handleEvent(event, engage_ok) * N    // accumulate; finger down/up lifecycle
//!   update(window, enabled, rmb_down)    // level-based engage/disengage (relative mode)
//!   if (takeRecenter()) camera.requestRecenter()
//!   const d = drag(screen_w, screen_h)   // merged px delta + active flag
//!
//! SDL-facing (like gamepad.zig); the merge / latch / lifecycle logic is pure and
//! unit-tested. `enabled` / `engage_ok` are booleans main computes from camera
//! mode + focus + menu state, so this stays free of CameraMode and app types.

const std = @import("std");
const c = @import("../vk_types.zig").c;

pub const PointerLook = struct {
    /// Hold-RMB mouse drag (SDL relative mouse mode while engaged).
    mouse_active: bool = false,
    mouse_dx: f32 = 0, // accumulated relative px this frame
    mouse_dy: f32 = 0,
    /// One-finger touch drag.
    touch_active: bool = false,
    touch_finger: c.SDL_FingerID = 0,
    touch_dx: f32 = 0, // accumulated normalized-to-window delta this frame
    touch_dy: f32 = 0,
    /// Latched when a source disengages; the caller drains it to recenter.
    recenter: bool = false,

    pub const Drag = struct { active: bool, dx: f32, dy: f32 };

    /// Clear this frame's motion accumulators. Call before the event poll.
    pub fn beginFrame(self: *PointerLook) void {
        self.mouse_dx = 0;
        self.mouse_dy = 0;
        self.touch_dx = 0;
        self.touch_dy = 0;
    }

    /// Feed one polled SDL event; non-pointer events are ignored, so this can be
    /// called for every event. `engage_ok` gates STARTING / accumulating a glance
    /// (glance-capable mode + menu closed); a finger lift always disengages so a
    /// drag can never get stuck even if `engage_ok` flips while it is held.
    pub fn handleEvent(self: *PointerLook, event: c.SDL_Event, engage_ok: bool) void {
        switch (event.type) {
            c.SDL_EVENT_MOUSE_MOTION => if (self.mouse_active) {
                // Relative mouse mode reports motion in xrel/yrel.
                self.mouse_dx += event.motion.xrel;
                self.mouse_dy += event.motion.yrel;
            },
            c.SDL_EVENT_FINGER_DOWN => if (engage_ok and !self.touch_active) {
                // Track the first finger only; a second finger is ignored.
                self.touch_active = true;
                self.touch_finger = event.tfinger.fingerID;
            },
            c.SDL_EVENT_FINGER_MOTION => if (self.touch_active and event.tfinger.fingerID == self.touch_finger and engage_ok) {
                self.touch_dx += event.tfinger.dx;
                self.touch_dy += event.tfinger.dy;
            },
            c.SDL_EVENT_FINGER_UP, c.SDL_EVENT_FINGER_CANCELED => if (self.touch_active and event.tfinger.fingerID == self.touch_finger) {
                self.touch_active = false;
                self.touch_finger = 0;
                self.recenter = true; // gradual spring back on lift
            },
            else => {},
        }
    }

    /// Level-based engage/disengage, once per frame after the poll. `window`
    /// toggles SDL relative mouse mode for the hold-RMB path (null = headless, the
    /// SDL call is skipped). `enabled` = look-capable + focused + menu closed;
    /// `rmb_down` = right mouse button held. Disengaging a source latches recenter.
    pub fn update(self: *PointerLook, window: ?*c.SDL_Window, enabled: bool, rmb_down: bool) void {
        const want_mouse = enabled and rmb_down;
        if (want_mouse != self.mouse_active) {
            if (window) |win| _ = c.SDL_SetWindowRelativeMouseMode(win, want_mouse);
            self.mouse_active = want_mouse;
            if (!want_mouse) self.recenter = true; // snap back on release
        }
        // Touch normally lifts via handleEvent; this catches an in-progress drag
        // interrupted by focus loss / the menu opening / a mode change.
        if (self.touch_active and !enabled) {
            self.touch_active = false;
            self.touch_finger = 0;
            self.recenter = true;
        }
    }

    /// True once if a source disengaged since the last call; the caller springs
    /// the glance back to center (camera.requestRecenter).
    pub fn takeRecenter(self: *PointerLook) bool {
        const r = self.recenter;
        self.recenter = false;
        return r;
    }

    /// This frame's merged pointer drag in PIXELS, gated per source so a delta
    /// from a source that disengaged mid-frame can't leak into the other. Mouse is
    /// already relative px; touch is normalized to window size, scaled here.
    pub fn drag(self: *const PointerLook, screen_w: f32, screen_h: f32) Drag {
        var dx: f32 = 0;
        var dy: f32 = 0;
        var active = false;
        if (self.mouse_active) {
            dx += self.mouse_dx;
            dy += self.mouse_dy;
            active = true;
        }
        if (self.touch_active) {
            dx += self.touch_dx * screen_w;
            dy += self.touch_dy * screen_h;
            active = true;
        }
        return .{ .active = active, .dx = dx, .dy = dy };
    }
};

// ---------------------------------------------------------------------------
// Tests (pure lifecycle / merge / latch; the SDL relative-mode call in update is
// skipped with a null window and exercised at runtime).
// ---------------------------------------------------------------------------

const testing = std.testing;

fn fingerEvent(comptime ty: c_uint, id: c.SDL_FingerID, dx: f32, dy: f32) c.SDL_Event {
    var ev = std.mem.zeroes(c.SDL_Event);
    ev.type = ty;
    ev.tfinger.fingerID = id;
    ev.tfinger.dx = dx;
    ev.tfinger.dy = dy;
    return ev;
}

fn motionEvent(xrel: f32, yrel: f32) c.SDL_Event {
    var ev = std.mem.zeroes(c.SDL_Event);
    ev.type = c.SDL_EVENT_MOUSE_MOTION;
    ev.motion.xrel = xrel;
    ev.motion.yrel = yrel;
    return ev;
}

test "PointerLook: mouse motion accumulates only while active; beginFrame resets" {
    var pl: PointerLook = .{};
    pl.handleEvent(motionEvent(5, -3), true);
    try testing.expectEqual(@as(f32, 0), pl.mouse_dx); // not active yet
    pl.mouse_active = true;
    pl.handleEvent(motionEvent(5, -3), true);
    pl.handleEvent(motionEvent(2, 1), true);
    try testing.expectEqual(@as(f32, 7), pl.mouse_dx);
    try testing.expectEqual(@as(f32, -2), pl.mouse_dy);
    pl.beginFrame();
    try testing.expectEqual(@as(f32, 0), pl.mouse_dx);
    try testing.expectEqual(@as(f32, 0), pl.mouse_dy);
}

test "PointerLook: finger down/move/up lifecycle latches a recenter on lift" {
    var pl: PointerLook = .{};
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_DOWN, 7, 0, 0), true);
    try testing.expect(pl.touch_active);
    try testing.expectEqual(@as(c.SDL_FingerID, 7), pl.touch_finger);
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_MOTION, 7, 0.1, -0.2), true);
    try testing.expectApproxEqAbs(@as(f32, 0.1), pl.touch_dx, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.2), pl.touch_dy, 1e-6);
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_UP, 7, 0, 0), true);
    try testing.expect(!pl.touch_active);
    try testing.expect(pl.takeRecenter());
    try testing.expect(!pl.takeRecenter()); // latch drained
}

test "PointerLook: finger down blocked when engage_ok is false" {
    var pl: PointerLook = .{};
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_DOWN, 1, 0, 0), false);
    try testing.expect(!pl.touch_active);
}

test "PointerLook: second finger ignored, foreign finger's events ignored" {
    var pl: PointerLook = .{};
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_DOWN, 1, 0, 0), true);
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_DOWN, 2, 0, 0), true); // second finger
    try testing.expectEqual(@as(c.SDL_FingerID, 1), pl.touch_finger);
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_MOTION, 2, 0.5, 0.5), true); // foreign motion
    try testing.expectEqual(@as(f32, 0), pl.touch_dx);
    pl.handleEvent(fingerEvent(c.SDL_EVENT_FINGER_UP, 2, 0, 0), true); // foreign lift
    try testing.expect(pl.touch_active); // still tracking finger 1
}

test "PointerLook: update engages on RMB and recenters on release" {
    var pl: PointerLook = .{};
    pl.update(null, true, true); // enabled + rmb down
    try testing.expect(pl.mouse_active);
    try testing.expect(!pl.takeRecenter()); // engage does not request recenter
    pl.update(null, true, false); // rmb released
    try testing.expect(!pl.mouse_active);
    try testing.expect(pl.takeRecenter()); // release springs back
}

test "PointerLook: update disengages an active touch when disabled" {
    var pl: PointerLook = .{};
    pl.touch_active = true;
    pl.touch_finger = 3;
    pl.update(null, false, false); // e.g. focus lost / menu opened / mode changed
    try testing.expect(!pl.touch_active);
    try testing.expectEqual(@as(c.SDL_FingerID, 0), pl.touch_finger);
    try testing.expect(pl.takeRecenter());
}

test "PointerLook: drag merges per source and scales touch by screen size" {
    var pl: PointerLook = .{};
    // Neither active: no drag.
    try testing.expect(!pl.drag(1000, 800).active);

    pl.mouse_active = true;
    pl.mouse_dx = 4;
    pl.mouse_dy = -2;
    // touch_dx set but inactive: must NOT leak into the merged delta.
    pl.touch_dx = 0.5;
    const d_mouse = pl.drag(1000, 800);
    try testing.expect(d_mouse.active);
    try testing.expectEqual(@as(f32, 4), d_mouse.dx);
    try testing.expectEqual(@as(f32, -2), d_mouse.dy);

    pl.mouse_active = false;
    pl.touch_active = true;
    pl.touch_dx = 0.1; // normalized
    pl.touch_dy = 0.25;
    const d_touch = pl.drag(1000, 800);
    try testing.expectApproxEqAbs(@as(f32, 100), d_touch.dx, 1e-4); // 0.1 * 1000
    try testing.expectApproxEqAbs(@as(f32, 200), d_touch.dy, 1e-4); // 0.25 * 800
}
