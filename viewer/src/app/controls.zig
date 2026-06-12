//! Pilot intent for one tick: normalized stick deflections + throttle delta.
//! The seam where keyboard / gamepad / mouse-flight (future) all converge so
//! `sim.applySimInput` doesn't care which device the bits came from.
//!
//! Sign conventions match the body-frame rotations in sim.zig: positive pitch
//! = nose up, positive roll = roll left (right wing rises), positive yaw =
//! nose right.

const c = @import("../vk_types.zig").c;

pub const Controls = struct {
    /// Stick deflections, [-1, +1]. Multiplied by per-axis rate scalars in sim.
    pitch: f32 = 0,
    roll: f32 = 0,
    yaw: f32 = 0,
    /// Rate of throttle change, [-1, +1]. Multiplied by `THROTTLE_RATE` in sim
    /// to get per-second delta. Persistent throttle setting lives on Aircraft.
    throttle_delta: f32 = 0,
};

pub fn fromKeyboard(keys: ?[*]const bool) Controls {
    const k = keys orelse return .{};
    var ctrl: Controls = .{};
    if (k[c.SDL_SCANCODE_W]) ctrl.pitch += 1;
    if (k[c.SDL_SCANCODE_S]) ctrl.pitch -= 1;
    if (k[c.SDL_SCANCODE_UP]) ctrl.pitch += 1;
    if (k[c.SDL_SCANCODE_DOWN]) ctrl.pitch -= 1;
    if (k[c.SDL_SCANCODE_A]) ctrl.roll += 1;
    if (k[c.SDL_SCANCODE_D]) ctrl.roll -= 1;
    if (k[c.SDL_SCANCODE_Q]) ctrl.yaw -= 1;
    if (k[c.SDL_SCANCODE_E]) ctrl.yaw += 1;
    if (k[c.SDL_SCANCODE_LSHIFT]) ctrl.throttle_delta += 1;
    if (k[c.SDL_SCANCODE_LCTRL]) ctrl.throttle_delta -= 1;
    // Multiple bindings per axis (W + Up, etc.) can sum past 1; clamp so
    // control authority stays device-independent in sim.applySimInput.
    ctrl.pitch = std.math.clamp(ctrl.pitch, -1, 1);
    ctrl.roll = std.math.clamp(ctrl.roll, -1, 1);
    ctrl.yaw = std.math.clamp(ctrl.yaw, -1, 1);
    ctrl.throttle_delta = std.math.clamp(ctrl.throttle_delta, -1, 1);
    return ctrl;
}

/// Combine two control sources (e.g. keyboard + gamepad) by summing each axis
/// and clamping, so either device works alone and both compose without a mode
/// switch. Matches fromKeyboard's per-axis clamp.
pub fn merge(a: Controls, b: Controls) Controls {
    return .{
        .pitch = std.math.clamp(a.pitch + b.pitch, -1, 1),
        .roll = std.math.clamp(a.roll + b.roll, -1, 1),
        .yaw = std.math.clamp(a.yaw + b.yaw, -1, 1),
        .throttle_delta = std.math.clamp(a.throttle_delta + b.throttle_delta, -1, 1),
    };
}

const std = @import("std");
const testing = std.testing;

test "Controls: default is zero" {
    const ctrl: Controls = .{};
    try testing.expectEqual(@as(f32, 0), ctrl.pitch);
    try testing.expectEqual(@as(f32, 0), ctrl.roll);
    try testing.expectEqual(@as(f32, 0), ctrl.yaw);
    try testing.expectEqual(@as(f32, 0), ctrl.throttle_delta);
}

test "Controls: keyboard maps W/S pitch, A/D roll, Q/E yaw, Shift/Ctrl throttle" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_W] = true;
    keys[c.SDL_SCANCODE_A] = true;
    keys[c.SDL_SCANCODE_Q] = true;
    keys[c.SDL_SCANCODE_LSHIFT] = true;
    const ctrl = fromKeyboard(@as([*]const bool, &keys));
    try testing.expectEqual(@as(f32, 1), ctrl.pitch);
    try testing.expectEqual(@as(f32, 1), ctrl.roll);
    try testing.expectEqual(@as(f32, -1), ctrl.yaw);
    try testing.expectEqual(@as(f32, 1), ctrl.throttle_delta);
}

test "Controls: arrow keys work as secondary pitch" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_UP] = true;
    const ctrl = fromKeyboard(@as([*]const bool, &keys));
    try testing.expectEqual(@as(f32, 1), ctrl.pitch);
}

test "Controls: opposing keys cancel" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_W] = true;
    keys[c.SDL_SCANCODE_S] = true;
    keys[c.SDL_SCANCODE_LSHIFT] = true;
    keys[c.SDL_SCANCODE_LCTRL] = true;
    const ctrl = fromKeyboard(@as([*]const bool, &keys));
    try testing.expectEqual(@as(f32, 0), ctrl.pitch);
    try testing.expectEqual(@as(f32, 0), ctrl.throttle_delta);
}

test "Controls: doubled bindings on one axis clamp to 1" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_W] = true;
    keys[c.SDL_SCANCODE_UP] = true;
    const ctrl = fromKeyboard(@as([*]const bool, &keys));
    try testing.expectEqual(@as(f32, 1), ctrl.pitch);
}

test "Controls: fromKeyboard null returns zero" {
    const ctrl = fromKeyboard(null);
    try testing.expectEqual(@as(f32, 0), ctrl.pitch);
    try testing.expectEqual(@as(f32, 0), ctrl.throttle_delta);
}

test "Controls: merge sums axes and clamps to [-1, 1]" {
    const a: Controls = .{ .pitch = 0.8, .yaw = 1 };
    const b: Controls = .{ .pitch = 0.8, .roll = -0.5 };
    const m = merge(a, b);
    try testing.expectEqual(@as(f32, 1), m.pitch); // 1.6 clamped
    try testing.expectEqual(@as(f32, -0.5), m.roll);
    try testing.expectEqual(@as(f32, 1), m.yaw);
    try testing.expectEqual(@as(f32, 0), m.throttle_delta);
}
