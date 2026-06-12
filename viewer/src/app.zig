//! Umbrella for app-level state and UI files. Re-exports each file in
//! `app/` as a namespace.

pub const aircraft = @import("app/aircraft.zig");
pub const camera = @import("app/camera.zig");
pub const controls = @import("app/controls.zig");
pub const gamepad = @import("app/gamepad.zig");
pub const input = @import("app/input.zig");
pub const pointer_look = @import("app/pointer_look.zig");
pub const pose = @import("app/pose.zig");
pub const sim = @import("app/sim.zig");

test {
    _ = aircraft;
    _ = camera;
    _ = controls;
    _ = gamepad;
    _ = input;
    _ = pointer_look;
    _ = pose;
    _ = sim;
}
