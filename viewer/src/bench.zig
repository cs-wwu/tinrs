//! Umbrella for the bench subsystem. Re-exports each file in `bench/`
//! as a namespace so callers can do `@import("bench.zig").runner.Bench`.

pub const autotune = @import("bench/autotune.zig");
pub const autotune_session = @import("bench/autotune_session.zig");
pub const convergence = @import("bench/convergence.zig");
pub const frame_stats = @import("bench/frame_stats.zig");
pub const profile = @import("bench/profile.zig");
pub const runner = @import("bench/runner.zig");

test {
    _ = autotune;
    _ = autotune_session;
    _ = convergence;
    _ = frame_stats;
    _ = profile;
    _ = runner;
}
