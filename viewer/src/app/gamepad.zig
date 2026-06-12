//! Gamepad flight producer. Reads a single connected controller and maps it to
//! the device-neutral `Controls` seam, so `sim.applySimInput` never sees SDL.
//! Peer to `controls.zig` (keyboard producer): the SDL adapter is thin, the
//! axis mapping is a pure, unit-tested function.
//!
//! Default scheme: left stick = pitch/roll, triggers = yaw (LT left, RT right),
//! L1/R1 = throttle (R1 up, L1 down). The right stick is reserved for camera/
//! view control (wired in a later phase).
//!
//! Sign conventions match controls.zig and sim.zig: positive pitch = nose up,
//! positive roll = roll left, positive yaw = nose right. The left stick is
//! inverted on pitch by default (stick-up = nose down, flight-stick feel);
//! invert_pitch=false restores stick-up = nose up to match the keyboard.
//!
//! TODO: fully customizable controls (preset schemes + per-action rebinding for
//! both keyboard and gamepad) land with the settings menu. `mapAxes` is the
//! layer a data-driven binding table will plug into; the `Controls` output seam
//! stays fixed so nothing downstream changes.

const std = @import("std");
const c = @import("../vk_types.zig").c;
const Controls = @import("controls.zig").Controls;

/// Per-feel tunables. These are the first knobs the settings menu will expose;
/// hardcoded defaults until then.
pub const Tuning = struct {
    /// Stick travel (0..1) ignored around center before input registers.
    deadzone: f32 = 0.1,
    /// Flip pitch so stick-up = nose down (classic flight-stick feel). On by
    /// default: the left stick behaves like a joystick (push forward = nose
    /// down). Set false to restore stick-up = nose up (keyboard W/Up feel).
    invert_pitch: bool = true,
    /// Trigger travel (0..1) ignored before throttle responds. Throttle is a
    /// RATE the sim integrates over time, so without a dead band a resting or
    /// worn trigger (rest value a few hundred / 32767) would drift the throttle
    /// continuously toward 0 or 1 with hands off.
    trigger_deadzone: f32 = 0.05,
    /// Right-stick free-look angular rate at full deflection (rad/s). The view
    /// glance offset integrates at this rate; a settings-menu slider tunes it.
    look_sensitivity: f32 = 2.5,
};

/// Raw free-camera axes (left stick move, shoulders vertical, right stick look,
/// trigger speed scale), separate from the Controls flight mapping. readFreeLook
/// returns null when no pad is connected.
pub const FreeLook = struct {
    move_fwd: f32 = 0,
    move_right: f32 = 0,
    /// World-vertical move from the shoulders: R1 = up (+1), L1 = down (-1).
    /// Digital, matching the keyboard's full-speed Space/LShift. Free here
    /// because shoulders only drive throttle in flight mode (readControls).
    move_up: f32 = 0,
    look_yaw: f32 = 0,
    look_pitch: f32 = 0,
    boost: f32 = 0,
};

/// SDL axis range: sticks report roughly [-AXIS_MAX, AXIS_MAX], triggers
/// [0, AXIS_MAX]. Dividing by AXIS_MAX and clamping normalizes both.
const AXIS_MAX: f32 = 32767.0;

pub const Gamepad = struct {
    handle: ?*c.SDL_Gamepad = null,
    /// Joystick instance id of the open pad; matched against REMOVED events.
    id: c.SDL_JoystickID = 0,
    tuning: Tuning = .{},

    pub fn init() Gamepad {
        return .{};
    }

    pub fn deinit(self: *Gamepad) void {
        self.closePad();
    }

    /// Open on ADDED (if we hold none), close on REMOVED (if it's ours). SDL
    /// also posts ADDED for already-connected pads at init, so this single path
    /// covers both startup and hot-plug.
    // TODO: two-pad hot-swap gap. With two pads connected, the second's ADDED is
    // ignored (single-pad by design); but if the active pad is then unplugged,
    // closePad leaves handle=null and no fresh ADDED fires for the still-connected
    // second pad, so gamepad input goes dead until a physical replug. Fix: on
    // REMOVED, after closePad, re-scan SDL_GetGamepads and adopt the first still
    // connected pad. Niche (needs two controllers); deferred.
    pub fn handleEvent(self: *Gamepad, event: c.SDL_Event) void {
        switch (event.type) {
            c.SDL_EVENT_GAMEPAD_ADDED => {
                if (self.handle == null) self.openPad(event.gdevice.which);
            },
            c.SDL_EVENT_GAMEPAD_REMOVED => {
                if (self.handle != null and event.gdevice.which == self.id) self.closePad();
            },
            else => {},
        }
    }

    fn openPad(self: *Gamepad, instance_id: c.SDL_JoystickID) void {
        const h = c.SDL_OpenGamepad(instance_id) orelse {
            std.log.warn("Gamepad {d} failed to open: {s}", .{ instance_id, c.SDL_GetError() });
            return;
        };
        self.handle = h;
        self.id = instance_id;
        const name: [*c]const u8 = c.SDL_GetGamepadName(h) orelse "(unknown)";
        std.log.info("Gamepad connected: {s}", .{name});
    }

    fn closePad(self: *Gamepad) void {
        if (self.handle) |h| {
            c.SDL_CloseGamepad(h);
            std.log.info("Gamepad disconnected", .{});
        }
        self.handle = null;
        self.id = 0;
    }

    /// This frame's pilot intent from the pad, or zero if none connected.
    pub fn readControls(self: *const Gamepad) Controls {
        const pad = self.handle orelse return .{};
        const lx = stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFTX));
        const ly = stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFTY));
        const lt = triggerNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
        const rt = triggerNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
        const throttle_up = c.SDL_GetGamepadButton(pad, c.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
        const throttle_down = c.SDL_GetGamepadButton(pad, c.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
        return mapAxes(lx, ly, rt, lt, throttle_up, throttle_down, self.tuning);
    }

    /// Raw axes for the free debug camera (distinct from the Controls flight
    /// mapping): left stick moves, right stick looks, triggers scale fly speed.
    /// Deadzoned/normalized; null when no pad is connected.
    pub fn readFreeLook(self: *const Gamepad) ?FreeLook {
        const pad = self.handle orelse return null;
        const dz = self.tuning.deadzone;
        const tdz = self.tuning.trigger_deadzone;
        const up = c.SDL_GetGamepadButton(pad, c.SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
        const down = c.SDL_GetGamepadButton(pad, c.SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
        const rs = rightStickLook(pad, dz);
        return .{
            .move_right = deadzone(stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFTX)), dz),
            .move_fwd = -deadzone(stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFTY)), dz),
            .move_up = (if (up) @as(f32, 1) else 0) - (if (down) @as(f32, 1) else 0),
            .look_yaw = rs[0],
            .look_pitch = rs[1],
            // Deadzoned like the throttle so a resting trigger doesn't drift speed.
            .boost = deadzone(triggerNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)), tdz) - deadzone(triggerNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_LEFT_TRIGGER)), tdz),
        };
    }

    /// Right-stick look axes for cockpit/sensor free-look (the glance offset),
    /// deadzoned and normalized to [-1, 1]: [0] = yaw (+right), [1] = pitch
    /// (+up). null when no pad is connected. Distinct from readFreeLook (the
    /// whole free camera) and readControls (flight): this is just the glance
    /// stick, the rest of the pad keeps driving flight in cockpit view.
    pub fn readLook(self: *const Gamepad) ?[2]f32 {
        const pad = self.handle orelse return null;
        return rightStickLook(pad, self.tuning.deadzone);
    }
};

/// Right-stick look axes (deadzoned, [-1, 1]): [0] = yaw (+right), [1] = pitch
/// (+up, SDL's -Y). Shared by readLook (cockpit/sensor glance) and readFreeLook
/// (free camera) so the right-stick sign + deadzone live in one place.
fn rightStickLook(pad: *c.SDL_Gamepad, dz: f32) [2]f32 {
    return .{
        deadzone(stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_RIGHTX)), dz),
        -deadzone(stickNorm(c.SDL_GetGamepadAxis(pad, c.SDL_GAMEPAD_AXIS_RIGHTY)), dz),
    };
}

fn stickNorm(raw: i16) f32 {
    return std.math.clamp(@as(f32, @floatFromInt(raw)) / AXIS_MAX, -1.0, 1.0);
}

fn triggerNorm(raw: i16) f32 {
    return std.math.clamp(@as(f32, @floatFromInt(raw)) / AXIS_MAX, 0.0, 1.0);
}

/// Center deadzone with edge rescale: output ramps from 0 at the deadzone
/// boundary to 1 at full deflection, so there's no jump as the stick leaves the
/// zone.
fn deadzone(v: f32, dz: f32) f32 {
    const a = @abs(v);
    if (a <= dz) return 0;
    const scaled = (a - dz) / (1.0 - dz);
    return if (v < 0) -scaled else scaled;
}

/// Pure mapping from normalized inputs to pilot intent. Sticks in [-1, 1],
/// triggers in [0, 1] drive yaw, the two shoulder buttons drive throttle. Kept
/// SDL-free so it is unit-testable and so a future rebinding table has one place
/// to target.
pub fn mapAxes(lx: f32, ly: f32, rt: f32, lt: f32, throttle_up: bool, throttle_down: bool, t: Tuning) Controls {
    const roll = -deadzone(lx, t.deadzone); // stick right (SDL +X) -> roll right (negative)
    var pitch = -deadzone(ly, t.deadzone); // stick up (SDL -Y) -> nose up
    if (t.invert_pitch) pitch = -pitch; // default on: stick up -> nose down (flight-stick feel)
    // Triggers yaw: RT spools nose right, LT nose left; deadzoned so a resting or
    // worn trigger does not drift yaw with hands off.
    const yaw = deadzone(rt, t.trigger_deadzone) - deadzone(lt, t.trigger_deadzone);
    // Shoulder buttons throttle: R1 spools up, L1 spools down.
    const throttle = (if (throttle_up) @as(f32, 1) else 0) - (if (throttle_down) @as(f32, 1) else 0);
    return .{
        .pitch = std.math.clamp(pitch, -1, 1),
        .roll = std.math.clamp(roll, -1, 1),
        .yaw = std.math.clamp(yaw, -1, 1),
        .throttle_delta = std.math.clamp(throttle, -1, 1),
    };
}

// ---------------------------------------------------------------------------
// Tests (pure mapAxes / deadzone; the SDL adapter is exercised at runtime)
// ---------------------------------------------------------------------------

const testing = std.testing;

// Force semantic analysis of the SDL-calling methods (openPad/handleEvent/
// readControls). Without this, the pure-function tests below don't reference
// them, so a type error in their bodies only surfaces in the exe build.
test {
    testing.refAllDecls(Gamepad);
}

test "deadzone: inside zone is zero, outside rescales from zero, full reaches 1" {
    try testing.expectEqual(@as(f32, 0), deadzone(0.1, 0.15));
    try testing.expectEqual(@as(f32, 0), deadzone(-0.15, 0.15));
    // Just outside the zone: small, not a jump to ~0.15.
    try testing.expect(deadzone(0.16, 0.15) > 0);
    try testing.expect(deadzone(0.16, 0.15) < 0.05);
    try testing.expectApproxEqAbs(@as(f32, 1), deadzone(1.0, 0.15), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -1), deadzone(-1.0, 0.15), 1e-6);
}

test "mapAxes: default inverts pitch (stick up = nose down), stick right is roll right" {
    const t: Tuning = .{ .deadzone = 0.1 }; // invert_pitch defaults on
    // SDL reports stick-up as negative Y; default flight-stick feel = nose down.
    const up = mapAxes(0, -1, 0, 0, false, false, t);
    try testing.expect(up.pitch < 0); // nose down
    // SDL reports stick-right as positive X; +roll is roll left, so this is negative.
    const right = mapAxes(1, 0, 0, 0, false, false, t);
    try testing.expect(right.roll < 0);
}

test "mapAxes: invert_pitch=false restores stick up = nose up" {
    const t: Tuning = .{ .deadzone = 0.1, .invert_pitch = false };
    const up = mapAxes(0, -1, 0, 0, false, false, t);
    try testing.expect(up.pitch > 0); // nose up
}

test "mapAxes: yaw from triggers, throttle from shoulders" {
    const t: Tuning = .{};
    try testing.expectEqual(@as(f32, 1), mapAxes(0, 0, 1, 0, false, false, t).yaw); // RT -> nose right
    try testing.expectEqual(@as(f32, -1), mapAxes(0, 0, 0, 1, false, false, t).yaw); // LT -> nose left
    try testing.expectEqual(@as(f32, 0), mapAxes(0, 0, 1, 1, false, false, t).yaw); // both cancel
    try testing.expectEqual(@as(f32, 1), mapAxes(0, 0, 0, 0, true, false, t).throttle_delta); // R1 -> spool up
    try testing.expectEqual(@as(f32, -1), mapAxes(0, 0, 0, 0, false, true, t).throttle_delta); // L1 -> spool down
}

test "mapAxes: dead center is all zero" {
    const z = mapAxes(0, 0, 0, 0, false, false, .{});
    try testing.expectEqual(@as(f32, 0), z.pitch);
    try testing.expectEqual(@as(f32, 0), z.roll);
    try testing.expectEqual(@as(f32, 0), z.yaw);
    try testing.expectEqual(@as(f32, 0), z.throttle_delta);
}
