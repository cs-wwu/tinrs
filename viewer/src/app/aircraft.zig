//! Physical state of the flying object.

const Vec3d = @import("math").Vec3d;
const Pose = @import("pose.zig").Pose;

pub const Aircraft = struct {
    pose: Pose,
    /// Forward speed in arcsec/s.
    airspeed: f32 = 0,
    /// Commanded thrust [0, 1].
    throttle: f32 = 0,
    // TODO: pilot-control state (auto_level, future trim/assists) belongs on
    // a Controls / SimInput struct once the input abstraction lands; Aircraft
    // should be physical state only.
    auto_level: bool = true,
    smooth_pitch: f32 = 0,
    smooth_roll: f32 = 0,
    smooth_yaw: f32 = 0,

    pub fn init(position: Vec3d, initial_dir: ?[3]f32) Aircraft {
        return .{ .pose = Pose.init(position, initial_dir) };
    }
};
