//! Position + velocity + orientation.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Vec3d = math.Vec3d;
const Quat = math.Quat;
const coords = @import("../terrain/coords.zig");

/// Default EMA blend for `Pose.updateVelocityEma`. ~10-frame window @60Hz.
pub const VELOCITY_ALPHA: f32 = 0.1;

pub const Pose = struct {
    /// World position in arcseconds. f64 to avoid ULP deadzone at large
    /// coordinates (CONUS-scale and beyond).
    position: Vec3d,
    /// Velocity in arcsec/s.
    velocity: Vec3 = .{ 0, 0, 0 },
    orientation: Quat,

    /// Identity-orientation basis (looking down -Z, +X right, +Y up).
    const START_FRONT: Vec3 = .{ 0, 0, -1 };
    const START_RIGHT: Vec3 = .{ 1, 0, 0 };
    const START_UP: Vec3 = .{ 0, 1, 0 };

    /// Initialize at world position (arcseconds). initial_dir is an optional
    /// look direction; null means identity orientation.
    pub fn init(position: Vec3d, initial_dir: ?[3]f32) Pose {
        const orientation = if (initial_dir) |dir|
            math.quatFromDir(START_FRONT, dir, math.world_up)
        else
            math.quat_identity;
        return .{ .position = position, .orientation = orientation };
    }

    /// Position truncated to f32 for GPU handoff (view matrix, push constants).
    pub fn positionF32(self: Pose) Vec3 {
        return .{
            @floatCast(self.position[0]),
            @floatCast(self.position[1]),
            @floatCast(self.position[2]),
        };
    }

    pub fn latLonDeg(self: Pose) [2]f64 {
        const lon_deg = coords.wrapToNearestF64(self.position[0] / 3600.0, 360.0);
        return .{ -self.position[2] / 3600.0, lon_deg };
    }

    pub fn front(self: Pose) Vec3 {
        return math.quatRotateVec3(self.orientation, START_FRONT);
    }

    pub fn right(self: Pose) Vec3 {
        return math.quatRotateVec3(self.orientation, START_RIGHT);
    }

    pub fn up(self: Pose) Vec3 {
        return math.quatRotateVec3(self.orientation, START_UP);
    }

    /// Translate `pos` by `dir * scale`, with X scaled by `inv_cos_lat` so
    /// visual movement matches look direction (the vertex shader applies
    /// cos(lat) to X). Used by every pose-driving input that walks the world.
    pub fn translateLonCorrected(pos: *Vec3d, dir: Vec3, scale: f64, inv_cos_lat: f64) void {
        pos[0] += @as(f64, dir[0]) * scale * inv_cos_lat;
        pos[1] += @as(f64, dir[1]) * scale;
        pos[2] += @as(f64, dir[2]) * scale;
    }

    /// Render-side smoothing for fixed-timestep physics.
    pub fn lerp(a: Pose, b: Pose, t: f32) Pose {
        return .{
            .position = math.lerpVec3d(a.position, b.position, @floatCast(t)),
            .velocity = math.lerpVec3(a.velocity, b.velocity, t),
            .orientation = math.quatNlerp(a.orientation, b.orientation, t),
        };
    }

    /// EMA-smoothed velocity update from a position delta. Shared by every
    /// input source that drives a pose. `alpha` is dt-independent blend;
    /// smaller = more smoothing.
    pub fn updateVelocityEma(self: *Pose, prev: Vec3d, dt: f32, alpha: f32) void {
        if (dt <= 0) return;
        const inv_dt: f64 = 1.0 / @as(f64, dt);
        const raw: Vec3 = .{
            @floatCast((self.position[0] - prev[0]) * inv_dt),
            @floatCast((self.position[1] - prev[1]) * inv_dt),
            @floatCast((self.position[2] - prev[2]) * inv_dt),
        };
        self.velocity = math.lerpVec3(self.velocity, raw, alpha);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const eps: f32 = 1e-4;

const test_pos: Vec3d = .{ 1800, coords.metersToArcsec(1200), 1800 };

fn testPose() Pose {
    return Pose.init(test_pos, null);
}

test "Pose: init defaults to identity orientation" {
    const p = testPose();
    try testing.expectEqual(@as(f32, 0), p.orientation[0]);
    try testing.expectEqual(@as(f32, 0), p.orientation[1]);
    try testing.expectEqual(@as(f32, 0), p.orientation[2]);
    try testing.expectEqual(@as(f32, 1), p.orientation[3]);
    try testing.expectEqual(@as(f32, 0), p.velocity[0]);
    try testing.expectEqual(@as(f32, 0), p.velocity[1]);
    try testing.expectEqual(@as(f32, 0), p.velocity[2]);
}

test "Pose: init stores f64 position exactly" {
    const p = testPose();
    try testing.expectEqual(test_pos[0], p.position[0]);
    try testing.expectEqual(test_pos[1], p.position[1]);
    try testing.expectEqual(test_pos[2], p.position[2]);
}

test "Pose: basis vectors are normalized" {
    const p = testPose();
    const front_len = @sqrt(math.dot(p.front(), p.front()));
    const right_len = @sqrt(math.dot(p.right(), p.right()));
    const up_len = @sqrt(math.dot(p.up(), p.up()));
    try testing.expectApproxEqAbs(@as(f32, 1.0), front_len, eps);
    try testing.expectApproxEqAbs(@as(f32, 1.0), right_len, eps);
    try testing.expectApproxEqAbs(@as(f32, 1.0), up_len, eps);
}

test "Pose: basis vectors are orthogonal" {
    const p = testPose();
    try testing.expectApproxEqAbs(@as(f32, 0), math.dot(p.front(), p.right()), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), math.dot(p.front(), p.up()), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), math.dot(p.right(), p.up()), eps);
}

test "Pose: identity orientation gives default basis" {
    var p = testPose();
    p.orientation = math.quat_identity;
    const f = p.front();
    const r = p.right();
    const u = p.up();
    try testing.expectApproxEqAbs(@as(f32, 0), f[0], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), f[1], eps);
    try testing.expectApproxEqAbs(@as(f32, -1), f[2], eps);
    try testing.expectApproxEqAbs(@as(f32, 1), r[0], eps);
    try testing.expectApproxEqAbs(@as(f32, 1), u[1], eps);
}

test "Pose: init with initial_dir rotates front to dir" {
    const p = Pose.init(test_pos, .{ 1, 0, 0 });
    const f = p.front();
    try testing.expectApproxEqAbs(@as(f32, 1), f[0], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), f[1], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), f[2], eps);
}

test "Pose: positionF32 truncates from f64" {
    var p = testPose();
    p.position = .{ 1.5, -2.25, 3.125 };
    const f = p.positionF32();
    try testing.expectEqual(@as(f32, 1.5), f[0]);
    try testing.expectEqual(@as(f32, -2.25), f[1]);
    try testing.expectEqual(@as(f32, 3.125), f[2]);
}

test "Pose: latLonDeg converts arcseconds to degrees" {
    var p = testPose();
    // 47N, 122W in arcseconds: lat -> z = -47*3600 (z is negated lat), lon -> x = -122*3600
    p.position = .{ -122.0 * 3600.0, 0, -47.0 * 3600.0 };
    const ll = p.latLonDeg();
    try testing.expectApproxEqAbs(@as(f64, 47.0), ll[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, -122.0), ll[1], 1e-9);
}

test "Pose: lerp t=0 and t=1 endpoints match" {
    var a = testPose();
    a.position = .{ 0, 0, 0 };
    a.velocity = .{ 1, 2, 3 };
    var b = testPose();
    b.position = .{ 100, 200, 300 };
    b.velocity = .{ 10, 20, 30 };
    b.orientation = math.quatFromAxisAngle(.{ 0, 1, 0 }, 1.0);
    const at_zero = Pose.lerp(a, b, 0.0);
    try testing.expectEqual(a.position[0], at_zero.position[0]);
    try testing.expectEqual(a.velocity[0], at_zero.velocity[0]);
    const at_one = Pose.lerp(a, b, 1.0);
    try testing.expectApproxEqAbs(b.position[0], at_one.position[0], 1e-6);
    try testing.expectApproxEqAbs(b.velocity[0], at_one.velocity[0], eps);
}

test "Pose: lerp midpoint averages position" {
    var a = testPose();
    a.position = .{ 0, 0, 0 };
    var b = testPose();
    b.position = .{ 100, 200, 300 };
    const mid = Pose.lerp(a, b, 0.5);
    try testing.expectApproxEqAbs(@as(f64, 50), mid.position[0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 100), mid.position[1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 150), mid.position[2], 1e-9);
}
