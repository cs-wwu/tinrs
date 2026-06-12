//! Free-fly camera: pose + optics + viewing mode.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const c = vkt.c;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Vec3d = math.Vec3d;
const pose_mod = @import("pose.zig");
const Pose = pose_mod.Pose;
const coords = @import("../terrain/coords.zig");

pub const CameraMode = enum { free, cockpit, chase };

pub const Camera = struct {
    pose: Pose,
    /// Free-fly movement speed (arcsec/s). +/- adjusts.
    speed: f32,
    fov: f32,
    /// Near and far clip planes in arcseconds. With reverse-Z + D32_SFLOAT,
    /// depth precision stays uniform across the full range, so a very small
    /// near and very large far carry no precision penalty.
    near: f32 = DEFAULT_NEAR,
    far: f32 = DEFAULT_FAR,
    mode: CameraMode = .cockpit,
    chase_yaw: f32 = 0,
    chase_pitch: f32 = 0,
    /// Free-look glance offset from the aircraft nose (radians), applied in
    /// cockpit/sensor view via lookOffsetOrientation. Held while a look input is
    /// active, sprung back to 0 on release (mouse RMB up, gamepad R3, touch lift).
    look_yaw: f32 = 0,
    look_pitch: f32 = 0,
    /// True while springing the glance offset back to center.
    recentering: bool = false,

    const MOVE_SPEED: f32 = 30.0; // arcsec/s (~926 m/s)
    const LOOK_SPEED: f32 = 2.0; // rad/s for arrows
    const SPEED_RATE: f32 = 1.5; // exponential speed scale rate (e^(rate*dt))
    const MIN_SPEED: f32 = 3.0; // ~93 m/s
    const MAX_SPEED: f32 = 12000.0; // ~370 km/s
    const DEFAULT_FOV: f32 = std.math.pi / 4.0; // 45 deg
    const MIN_FOV: f32 = 10.0 * std.math.pi / 180.0;
    const MAX_FOV: f32 = 120.0 * std.math.pi / 180.0;
    const FOV_RATE: f32 = 30.0 * std.math.pi / 180.0; // rad/s, linear slide
    pub const FLIGHT_FOV_LOW: f32 = 70.0 * std.math.pi / 180.0;
    const FLIGHT_FOV_HIGH: f32 = 78.0 * std.math.pi / 180.0;
    const FLIGHT_FOV_SPEED_LOW: f32 = 2.0; // arcsec/s
    const FLIGHT_FOV_SPEED_HIGH: f32 = 6.75; // arcsec/s (cruise)
    const FOV_SMOOTH_RATE: f32 = 3.0;
    pub const DEFAULT_NEAR: f32 = 0.01; // ~0.3m
    pub const DEFAULT_FAR: f32 = 200000.0; // ~6175 km, fits CONUS-scale clipmaps
    const FLY_SPEED: f32 = 300.0; // autopilot arcsec/s (~Mach 10)

    // Free-look glance offset limits (radians) and spring-back rate. The
    // per-device look SENSITIVITY lives with each input source (gamepad
    // `Tuning.look_sensitivity`; mouse sensitivity in settings) so the menu can
    // tune them independently; the camera owns only the limits + recenter feel.
    const LOOK_YAW_LIMIT: f32 = 160.0 * std.math.pi / 180.0; // can't look through your own seat
    const LOOK_PITCH_LIMIT: f32 = 85.0 * std.math.pi / 180.0; // shy of straight up/down
    const LOOK_RECENTER_RATE: f32 = 8.0; // expBlend rate for spring-back
    /// Pointer free-look: radians of glance per drag pixel at sensitivity 1.0,
    /// shared by mouse (relative px) and touch (normalized * screen px). Pub so
    /// main (the pointer glue) scales raw deltas by this * the user setting.
    pub const LOOK_DRAG_BASE: f32 = 0.0025;

    /// Initialize at world position (arcseconds). fov_degrees: 0 -> default 45.
    /// initial_dir: optional look direction; null = identity orientation.
    pub fn init(position: Vec3d, fov_degrees: f32, initial_dir: ?[3]f32) Camera {
        const fov = if (fov_degrees > 0 and fov_degrees < 180)
            fov_degrees * (std.math.pi / 180.0)
        else
            DEFAULT_FOV;
        return .{
            .pose = Pose.init(position, initial_dir),
            .speed = MOVE_SPEED,
            .fov = fov,
        };
    }

    pub fn syncChaseAngles(self: *Camera, aircraft_orientation: math.Quat) void {
        const yp = math.yawPitchFromQuat(aircraft_orientation);
        self.chase_yaw = yp[0];
        self.chase_pitch = yp[1];
    }

    pub fn updateFlightFov(self: *Camera, airspeed: f32, dt: f32) void {
        const t = std.math.clamp(
            (airspeed - FLIGHT_FOV_SPEED_LOW) / (FLIGHT_FOV_SPEED_HIGH - FLIGHT_FOV_SPEED_LOW),
            0.0,
            1.0,
        );
        const target = FLIGHT_FOV_LOW + (FLIGHT_FOV_HIGH - FLIGHT_FOV_LOW) * t;
        self.fov += (target - self.fov) * math.expBlend(FOV_SMOOTH_RATE, dt);
    }

    pub fn resetOrientation(self: *Camera) void {
        self.pose.orientation = math.quat_identity;
    }

    /// Ask the free-look glance offset to spring back to center (mouse RMB
    /// release, gamepad R3, touch lift). The decay runs over the next frames in
    /// applyLook; a fresh look input cancels it.
    pub fn requestRecenter(self: *Camera) void {
        self.recentering = true;
    }

    /// Integrate this frame's free-look deltas (radians) onto the glance offset.
    /// `active` = a look input is present this frame: it holds the offset and
    /// cancels any pending recenter (so a centered stick simply holds). When
    /// inactive and recentering, springs both axes back toward 0.
    pub fn applyLook(self: *Camera, dyaw: f32, dpitch: f32, active: bool, dt: f32) void {
        if (active) {
            self.recentering = false;
            self.look_yaw = std.math.clamp(self.look_yaw + dyaw, -LOOK_YAW_LIMIT, LOOK_YAW_LIMIT);
            self.look_pitch = std.math.clamp(self.look_pitch + dpitch, -LOOK_PITCH_LIMIT, LOOK_PITCH_LIMIT);
        } else if (self.recentering) {
            const blend = math.expBlend(LOOK_RECENTER_RATE, dt);
            self.look_yaw -= self.look_yaw * blend;
            self.look_pitch -= self.look_pitch * blend;
            if (@abs(self.look_yaw) < 1e-4 and @abs(self.look_pitch) < 1e-4) {
                self.look_yaw = 0;
                self.look_pitch = 0;
                self.recentering = false;
            }
        }
    }

    /// Fly horizontally at FLY_SPEED with the current heading projected onto
    /// the XZ plane. Used by --benchmark-fly and the profile flight phase.
    pub fn autopilot(self: *Camera, dt: f32) void {
        const prev = self.pose.position;
        const f = self.pose.front();
        var flat: Vec3 = .{ f[0], 0, f[2] };
        const len_sq = flat[0] * flat[0] + flat[2] * flat[2];
        if (len_sq < 1e-6) {
            flat = .{ 0, 0, -1 };
        } else {
            const inv_len = 1.0 / @sqrt(len_sq);
            flat = .{ flat[0] * inv_len, 0, flat[2] * inv_len };
        }
        const move: f64 = @as(f64, FLY_SPEED) * @as(f64, dt);
        const inv_cos_lat = coords.invCosLatFromZD(self.pose.position[2]);
        addScaledLonCorrected(&self.pose.position, flat, move, inv_cos_lat);
        self.pose.updateVelocityEma(prev, dt, pose_mod.VELOCITY_ALPHA);
    }

    /// Continuous input for camera-only keys: FOV ([/]) and movement speed (+/-).
    pub fn updateOptics(self: *Camera, dt: f32, keys: ?[*]const bool) void {
        const k = keys orelse return;

        if (k[c.SDL_SCANCODE_EQUALS])
            self.speed = @min(self.speed * @exp(SPEED_RATE * dt), MAX_SPEED);
        if (k[c.SDL_SCANCODE_MINUS])
            self.speed = @max(self.speed * @exp(-SPEED_RATE * dt), MIN_SPEED);
        if (k[c.SDL_SCANCODE_LEFTBRACKET])
            self.fov = @max(self.fov - FOV_RATE * dt, MIN_FOV);
        if (k[c.SDL_SCANCODE_RIGHTBRACKET])
            self.fov = @min(self.fov + FOV_RATE * dt, MAX_FOV);
    }

    pub fn viewMatrix(self: Camera) Mat4 {
        const pos = self.pose.positionF32();
        const f = self.pose.front();
        const u = self.pose.up();
        const target = math.add(pos, f);
        return math.lookAt(pos, target, u);
    }

    /// Full MVP in f64, truncated to f32 for push constants. Large world
    /// coordinates (~400K arcseconds) cause catastrophic cancellation in f32
    /// matrix multiply; f64 preserves precision.
    pub fn mvpMatrix(self: Camera, aspect: f32) Mat4 {
        const pos = self.pose.positionF32();
        const f = self.pose.front();
        const u = self.pose.up();
        const target = math.add(pos, f);
        const view = math.lookAtD(pos, target, u);
        const proj = math.perspectiveD(self.fov, aspect, self.near, self.far);
        return math.mat4dToMat4(math.matMulD(proj, view));
    }

    /// Rotation-only view matrix (eye at origin) for camera-relative rendering.
    /// Geometry is pre-shifted by (pos - camera) on the CPU in f64, so the
    /// view matrix only needs to rotate; no large translation values.
    pub fn viewMatrixRotOnly(self: Camera) Mat4 {
        const f = self.pose.front();
        const u = self.pose.up();
        return math.mat4dToMat4(math.lookAtD(.{ 0, 0, 0 }, f, u));
    }

    pub fn projMatrix(self: Camera, aspect: f32) Mat4 {
        return math.perspective(self.fov, aspect, self.near, self.far);
    }
};

/// Free-fly style input (WASD/arrows/Q/E + Space/LShift) against an arbitrary
/// pose. `speed` is arcsec/s in the local horizontal plane. keys=null decays
/// velocity toward zero (headless tick keeps the prefetch policy unbiased).
pub fn applyFreeFlyInput(pose: *Pose, dt: f32, keys: ?[*]const bool, speed: f32) void {
    const k = keys orelse {
        pose.updateVelocityEma(pose.position, dt, pose_mod.VELOCITY_ALPHA);
        return;
    };
    const prev = pose.position;
    const move: f64 = @as(f64, speed) * @as(f64, dt);
    const f = pose.front();
    const r = pose.right();
    const u = pose.up();

    // X (longitude) displacement scaled by 1/cos(lat) so visual movement
    // matches look direction (the vertex shader applies cos(lat) to X).
    const inv_cos_lat = coords.invCosLatFromZD(pose.position[2]);

    if (k[c.SDL_SCANCODE_W])
        addScaledLonCorrected(&pose.position, f, move, inv_cos_lat);
    if (k[c.SDL_SCANCODE_S])
        addScaledLonCorrected(&pose.position, f, -move, inv_cos_lat);
    if (k[c.SDL_SCANCODE_A])
        addScaledLonCorrected(&pose.position, r, -move, inv_cos_lat);
    if (k[c.SDL_SCANCODE_D])
        addScaledLonCorrected(&pose.position, r, move, inv_cos_lat);
    if (k[c.SDL_SCANCODE_SPACE])
        addScaled(&pose.position, u, move);
    if (k[c.SDL_SCANCODE_LSHIFT])
        addScaled(&pose.position, u, -move);

    var yaw_delta: f32 = 0;
    var pitch_delta: f32 = 0;

    if (k[c.SDL_SCANCODE_LEFT])
        yaw_delta -= Camera.LOOK_SPEED * dt;
    if (k[c.SDL_SCANCODE_RIGHT])
        yaw_delta += Camera.LOOK_SPEED * dt;
    if (k[c.SDL_SCANCODE_UP])
        pitch_delta += Camera.LOOK_SPEED * dt;
    if (k[c.SDL_SCANCODE_DOWN])
        pitch_delta -= Camera.LOOK_SPEED * dt;

    if (yaw_delta != 0 or pitch_delta != 0) {
        // Yaw around WORLD up, pitch around body right: keeps free-cam roll flat
        // (no roll control; free cam stays level, same for mnk + gamepad).
        const q_yaw = math.quatFromAxisAngle(math.world_up, -yaw_delta);
        const q_pitch = math.quatFromAxisAngle(r, pitch_delta);
        pose.orientation = math.quatNormalize(math.quatMul(math.quatMul(q_yaw, q_pitch), pose.orientation));
    }

    pose.updateVelocityEma(prev, dt, pose_mod.VELOCITY_ALPHA);
}

// Free-cam gamepad speed: triggers ratchet the (shared) camera.speed,
// multiplicatively at FREE_SPEED_ADJUST/s. The bounds MUST match the keyboard's
// [MIN_SPEED, MAX_SPEED] (updateOptics): the clamp runs every frame a pad is
// connected, so a narrower ceiling here would silently pin a keyboard-raised speed
// back down (e.g. fast world traversal becomes unreachable with a pad plugged in).
const FREE_SPEED_MIN: f32 = Camera.MIN_SPEED;
const FREE_SPEED_MAX: f32 = Camera.MAX_SPEED;
const FREE_SPEED_ADJUST: f32 = 2.0;

/// Gamepad free-fly, additive with applyFreeFlyInput: left stick moves
/// (forward/strafe), shoulders move vertical (`move_up`, R1 up / L1 down in
/// [-1,1], mirroring the keyboard's Space/LShift), right stick looks (yaw/pitch,
/// flat roll), and `boost` (RT minus LT, in [-1,1]) adjusts the persistent fly
/// speed like a throttle: hold RT to speed up, LT to slow down, clamped to
/// [FREE_SPEED_MIN, FREE_SPEED_MAX]; released holds. `speed` is in/out so the
/// ratchet persists and is shared with the keyboard path. Args are primitives so
/// camera stays decoupled from the gamepad module (main glues the two).
pub fn applyFreeFlyGamepad(
    pose: *Pose,
    dt: f32,
    move_fwd: f32,
    move_right: f32,
    move_up: f32,
    look_yaw: f32,
    look_pitch: f32,
    boost: f32,
    speed: *f32,
) void {
    const prev = pose.position;
    speed.* = std.math.clamp(speed.* * (1.0 + boost * FREE_SPEED_ADJUST * dt), FREE_SPEED_MIN, FREE_SPEED_MAX);
    const move: f64 = @as(f64, speed.*) * @as(f64, dt);
    const f = pose.front();
    const r = pose.right();
    const u = pose.up();
    const inv_cos_lat = coords.invCosLatFromZD(pose.position[2]);
    addScaledLonCorrected(&pose.position, f, move * @as(f64, move_fwd), inv_cos_lat);
    addScaledLonCorrected(&pose.position, r, move * @as(f64, move_right), inv_cos_lat);
    // Vertical along body-up, not lon-corrected: matches the keyboard Space/LShift path.
    addScaled(&pose.position, u, move * @as(f64, move_up));

    const yaw_delta = look_yaw * Camera.LOOK_SPEED * dt;
    const pitch_delta = look_pitch * Camera.LOOK_SPEED * dt;
    if (yaw_delta != 0 or pitch_delta != 0) {
        // Yaw around WORLD up keeps free-cam roll flat (matches the keyboard path).
        const q_yaw = math.quatFromAxisAngle(math.world_up, -yaw_delta);
        const q_pitch = math.quatFromAxisAngle(r, pitch_delta);
        pose.orientation = math.quatNormalize(math.quatMul(math.quatMul(q_yaw, q_pitch), pose.orientation));
    }
    pose.updateVelocityEma(prev, dt, pose_mod.VELOCITY_ALPHA);
}

/// Aircraft orientation with the free-look glance offset applied: yaw about the
/// body up, pitch about the body right, so the glance is relative to the cockpit
/// frame (not world axes) and stays correct as the aircraft banks. Pure: `base`
/// is the aircraft orientation, yaw/pitch in radians (yaw +right, pitch +up),
/// signs matching the keyboard/free-cam look. Returns base unchanged at zero.
// TODO: the yaw-then-pitch quaternion compose here repeats applyFreeFlyInput,
// applyFreeFlyGamepad, and (loosely) updateChaseCamera. They differ only in the
// yaw axis (world up vs body up), so a shared math.composeYawPitch(base, yaw_axis,
// right_axis, yaw, pitch) could fold all four. Deferred: refactoring the tested
// free-cam paths mid-feature risks a sign/order regression for little gain now.
pub fn lookOffsetOrientation(base: math.Quat, yaw: f32, pitch: f32) math.Quat {
    if (yaw == 0 and pitch == 0) return base;
    const base_up = math.quatRotateVec3(base, Vec3{ 0, 1, 0 });
    const base_right = math.quatRotateVec3(base, Vec3{ 1, 0, 0 });
    const q_yaw = math.quatFromAxisAngle(base_up, -yaw);
    const q_pitch = math.quatFromAxisAngle(base_right, pitch);
    return math.quatNormalize(math.quatMul(math.quatMul(q_yaw, q_pitch), base));
}

const CHASE_BEHIND: f32 = 6.0;
const CHASE_ABOVE: f32 = 1.0;
const CHASE_LAG: f32 = 4.0;

pub fn updateChaseCamera(camera: *Camera, aircraft_pose: Pose, dt: f32) void {
    const target_yp = math.yawPitchFromQuat(aircraft_pose.orientation);

    const yaw_err = math.wrapAngle(target_yp[0] - camera.chase_yaw);

    const blend = math.expBlend(CHASE_LAG, dt);
    camera.chase_yaw = math.wrapAngle(camera.chase_yaw + yaw_err * blend);
    camera.chase_pitch += (target_yp[1] - camera.chase_pitch) * blend;

    const q_yaw = math.quatFromAxisAngle(math.world_up, camera.chase_yaw);
    const chase_right = math.quatRotateVec3(q_yaw, Vec3{ 1, 0, 0 });
    const q_pitch = math.quatFromAxisAngle(chase_right, camera.chase_pitch);
    const chase_orient = math.quatMul(q_pitch, q_yaw);

    const chase_front = math.quatRotateVec3(chase_orient, Vec3{ 0, 0, -1 });
    const chase_up = math.quatRotateVec3(chase_orient, Vec3{ 0, 1, 0 });

    const inv_cos_lat = coords.invCosLatFromZD(aircraft_pose.position[2]);
    var cam_pos = aircraft_pose.position;
    Pose.translateLonCorrected(&cam_pos, chase_front, -@as(f64, CHASE_BEHIND), inv_cos_lat);
    Pose.translateLonCorrected(&cam_pos, chase_up, @as(f64, CHASE_ABOVE), inv_cos_lat);

    const prev = camera.pose.position;
    camera.pose.position = cam_pos;
    camera.pose.orientation = chase_orient;
    camera.pose.updateVelocityEma(prev, dt, pose_mod.VELOCITY_ALPHA);
}

fn addScaled(pos: *Vec3d, dir: Vec3, scale: f64) void {
    pos[0] += @as(f64, dir[0]) * scale;
    pos[1] += @as(f64, dir[1]) * scale;
    pos[2] += @as(f64, dir[2]) * scale;
}

const addScaledLonCorrected = Pose.translateLonCorrected;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const eps: f32 = 1e-4;

const test_pos: Vec3d = .{ 1800, coords.metersToArcsec(1200), 1800 };

fn testCamera() Camera {
    return Camera.init(test_pos, 0, null);
}

test "Camera: init defaults are valid" {
    const cam = testCamera();
    try testing.expect(cam.pose.position[1] > 0);
    try testing.expectApproxEqAbs(Camera.MOVE_SPEED, cam.speed, eps);
    try testing.expectApproxEqAbs(Camera.DEFAULT_FOV, cam.fov, eps);
    try testing.expectEqual(CameraMode.cockpit, cam.mode);
}

test "Camera: init with custom FOV" {
    const cam = Camera.init(test_pos, 60, null);
    try testing.expectApproxEqAbs(@as(f32, 60.0 * std.math.pi / 180.0), cam.fov, 0.001);
}

test "Camera: viewMatrix produces valid matrix" {
    const cam = testCamera();
    const view = cam.viewMatrix();
    try testing.expect(view[3][0] != 0 or view[3][1] != 0 or view[3][2] != 0);
}

test "Camera: mvpMatrix produces valid matrix" {
    const cam = testCamera();
    const m = cam.mvpMatrix(16.0 / 9.0);
    try testing.expect(m[3][3] != 1.0);
}

test "Camera: projMatrix uses self.near/self.far (reverse-Z)" {
    var cam = testCamera();
    cam.near = 0.5;
    cam.far = 1000.0;
    const m = cam.projMatrix(1.0);

    const z_at_near = m[2][2] * -cam.near + m[3][2];
    const w_at_near = m[2][3] * -cam.near + m[3][3];
    try testing.expectApproxEqAbs(@as(f32, 1.0), z_at_near / w_at_near, 1e-4);

    const z_at_far = m[2][2] * -cam.far + m[3][2];
    const w_at_far = m[2][3] * -cam.far + m[3][3];
    try testing.expectApproxEqAbs(@as(f32, 0.0), z_at_far / w_at_far, 1e-4);
}

test "Camera: default near/far match documented constants" {
    const cam = testCamera();
    try testing.expectEqual(Camera.DEFAULT_NEAR, cam.near);
    try testing.expectEqual(Camera.DEFAULT_FAR, cam.far);
}

test "Camera: autopilot moves in XZ plane" {
    var cam = testCamera();
    const y_before = cam.pose.position[1];
    cam.autopilot(0.1);
    try testing.expect(cam.pose.position[0] != test_pos[0] or cam.pose.position[2] != test_pos[2]);
    try testing.expectApproxEqAbs(y_before, cam.pose.position[1], 1e-9);
}

test "Camera: autopilot preserves altitude over many frames" {
    var cam = testCamera();
    const y_before = cam.pose.position[1];
    for (0..1000) |_| cam.autopilot(1.0 / 60.0);
    try testing.expectApproxEqAbs(y_before, cam.pose.position[1], 1e-9);
}

test "Camera: autopilot with pitched-down camera moves horizontally" {
    var cam = Camera.init(test_pos, 0, .{ 0, -0.4, -0.9 });
    const f = cam.pose.front();
    try testing.expect(f[1] < 0);
    const y_before = cam.pose.position[1];
    cam.autopilot(0.5);
    try testing.expectApproxEqAbs(y_before, cam.pose.position[1], 1e-9);
    try testing.expect(cam.pose.position[2] < test_pos[2]);
}

test "Camera: autopilot applies cos(lat) correction" {
    var cam_eq = Camera.init(.{ 1800, coords.metersToArcsec(1200), 0 }, 0, .{ 1, 0, 0 });
    var cam_hi = Camera.init(.{ 1800, coords.metersToArcsec(1200), -47 * 3600 }, 0, .{ 1, 0, 0 });
    cam_eq.autopilot(1.0);
    cam_hi.autopilot(1.0);
    const dx_eq = cam_eq.pose.position[0] - 1800.0;
    const dx_hi = cam_hi.pose.position[0] - 1800.0;
    try testing.expect(dx_hi > dx_eq * 1.3);
}

test "Camera: speed clamping via Camera bounds" {
    var cam = testCamera();
    cam.speed = 0.1;
    cam.speed = @max(cam.speed, Camera.MIN_SPEED);
    try testing.expectApproxEqAbs(Camera.MIN_SPEED, cam.speed, eps);

    cam.speed = 99999.0;
    cam.speed = @min(cam.speed, Camera.MAX_SPEED);
    try testing.expectApproxEqAbs(Camera.MAX_SPEED, cam.speed, eps);
}

test "Camera: FOV keys slide linearly and clamp" {
    var keys = [_]bool{false} ** 512;
    var cam = testCamera();
    const start = cam.fov;

    keys[c.SDL_SCANCODE_RIGHTBRACKET] = true;
    cam.updateOptics(0.1, &keys);
    try testing.expectApproxEqAbs(start + Camera.FOV_RATE * 0.1, cam.fov, eps);

    for (0..1000) |_| cam.updateOptics(1.0 / 60.0, &keys);
    try testing.expectApproxEqAbs(Camera.MAX_FOV, cam.fov, eps);
    keys[c.SDL_SCANCODE_RIGHTBRACKET] = false;

    keys[c.SDL_SCANCODE_LEFTBRACKET] = true;
    for (0..1000) |_| cam.updateOptics(1.0 / 60.0, &keys);
    try testing.expectApproxEqAbs(Camera.MIN_FOV, cam.fov, eps);
}

test "Camera: autopilot velocity converges to FLY_SPEED" {
    var cam = testCamera();
    for (0..100) |_| cam.autopilot(1.0 / 60.0);
    const speed = @sqrt(cam.pose.velocity[0] * cam.pose.velocity[0] + cam.pose.velocity[2] * cam.pose.velocity[2]);
    try testing.expectApproxEqAbs(@as(f32, 300.0), speed, 1.0);
}

// ---- applyFreeFlyInput (free function) ----

test "applyFreeFlyInput: null keys decays velocity to ~0" {
    var pose = Pose.init(test_pos, null);
    pose.velocity = .{ 100, 0, -100 };
    for (0..200) |_| applyFreeFlyInput(&pose, 1.0 / 60.0, null, Camera.MOVE_SPEED);
    try testing.expectApproxEqAbs(@as(f32, 0), pose.velocity[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0), pose.velocity[2], 0.001);
}

test "applyFreeFlyInput: W key moves pose forward" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_W] = true;
    var pose = Pose.init(test_pos, null);
    const z_before = pose.position[2];
    applyFreeFlyInput(&pose, 0.1, &keys, Camera.MOVE_SPEED);
    // identity orientation: front = -Z, so W should decrease z
    try testing.expect(pose.position[2] < z_before);
}

test "applyFreeFlyGamepad: forward stick moves; triggers ratchet speed" {
    var pose = Pose.init(test_pos, null); // identity: front = -Z
    var spd: f32 = Camera.MOVE_SPEED;
    applyFreeFlyGamepad(&pose, 0.1, 1.0, 0, 0, 0, 0, 0, &spd); // forward, no trigger
    try testing.expect(pose.position[2] < test_pos[2]); // moved -Z (forward)
    try testing.expectApproxEqAbs(Camera.MOVE_SPEED, spd, 1e-3); // boost 0 holds speed
    // RT (boost +1) speeds up; LT (boost -1) slows down.
    var up: f32 = Camera.MOVE_SPEED;
    applyFreeFlyGamepad(&pose, 0.1, 0, 0, 0, 0, 0, 1.0, &up);
    try testing.expect(up > Camera.MOVE_SPEED);
    var down: f32 = Camera.MOVE_SPEED;
    applyFreeFlyGamepad(&pose, 0.1, 0, 0, 0, 0, 0, -1.0, &down);
    try testing.expect(down < Camera.MOVE_SPEED);
}

test "applyFreeFlyGamepad: shoulders move vertical (R1 up, L1 down)" {
    var spd: f32 = Camera.MOVE_SPEED;
    var rise = Pose.init(test_pos, null); // identity: up = +Y
    applyFreeFlyGamepad(&rise, 0.1, 0, 0, 1.0, 0, 0, 0, &spd); // R1: move_up = +1
    try testing.expect(rise.position[1] > test_pos[1]);
    spd = Camera.MOVE_SPEED;
    var fall = Pose.init(test_pos, null);
    applyFreeFlyGamepad(&fall, 0.1, 0, 0, -1.0, 0, 0, 0, &spd); // L1: move_up = -1
    try testing.expect(fall.position[1] < test_pos[1]);
}

test "applyFreeFlyInput: yaw rotation changes orientation" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_RIGHT] = true;
    var pose = Pose.init(test_pos, null);
    const front_before = pose.front();
    applyFreeFlyInput(&pose, 0.5, &keys, Camera.MOVE_SPEED);
    const front_after = pose.front();
    try testing.expect(front_after[0] != front_before[0] or front_after[2] != front_before[2]);
}

test "applyFreeFlyInput: same input produces same result on independent poses" {
    var keys = [_]bool{false} ** 512;
    keys[c.SDL_SCANCODE_W] = true;
    var a = Pose.init(test_pos, null);
    var b = Pose.init(test_pos, null);
    applyFreeFlyInput(&a, 0.1, &keys, Camera.MOVE_SPEED);
    applyFreeFlyInput(&b, 0.1, &keys, Camera.MOVE_SPEED);
    try testing.expectEqual(a.position[2], b.position[2]);
    try testing.expect(a.position[2] < test_pos[2]);
}

// ---- free-look glance offset ----

test "lookOffsetOrientation: zero offset returns base unchanged" {
    const base = math.quat_identity;
    try testing.expectEqual(base, lookOffsetOrientation(base, 0, 0));
}

test "lookOffsetOrientation: +yaw looks right, +pitch looks up" {
    const base = math.quat_identity; // front = -Z, right = +X, up = +Y
    const fy = math.quatRotateVec3(lookOffsetOrientation(base, 0.5, 0), Vec3{ 0, 0, -1 });
    try testing.expect(fy[0] > 0); // front gains +X: glancing right
    const fp = math.quatRotateVec3(lookOffsetOrientation(base, 0, 0.5), Vec3{ 0, 0, -1 });
    try testing.expect(fp[1] > 0); // front gains +Y: glancing up
}

test "applyLook: active input accumulates and clamps to limits" {
    var cam = testCamera();
    cam.applyLook(0.1, -0.1, true, 1.0 / 60.0);
    try testing.expect(cam.look_yaw > 0 and cam.look_pitch < 0);
    // Drive far past the limits: the offset saturates, never exceeds.
    for (0..1000) |_| cam.applyLook(1.0, 1.0, true, 1.0 / 60.0);
    try testing.expectApproxEqAbs(Camera.LOOK_YAW_LIMIT, cam.look_yaw, 1e-4);
    try testing.expectApproxEqAbs(Camera.LOOK_PITCH_LIMIT, cam.look_pitch, 1e-4);
}

test "applyLook: inactive holds offset until recenter, then springs to 0" {
    var cam = testCamera();
    cam.look_yaw = 0.5;
    cam.look_pitch = -0.3;
    // No recenter requested: a released (centered) stick holds the offset.
    cam.applyLook(0, 0, false, 1.0 / 60.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), cam.look_yaw, 1e-6);
    // Recenter requested: springs both back to exactly 0 and clears the flag.
    cam.requestRecenter();
    for (0..600) |_| cam.applyLook(0, 0, false, 1.0 / 60.0);
    try testing.expectEqual(@as(f32, 0), cam.look_yaw);
    try testing.expectEqual(@as(f32, 0), cam.look_pitch);
    try testing.expect(!cam.recentering);
}

test "applyLook: fresh input cancels an in-progress recenter" {
    var cam = testCamera();
    cam.look_yaw = 0.5;
    cam.requestRecenter();
    cam.applyLook(0.05, 0, true, 1.0 / 60.0); // grab the stick mid-recenter
    try testing.expect(!cam.recentering);
    try testing.expect(cam.look_yaw > 0.5);
}
