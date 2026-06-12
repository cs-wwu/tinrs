//! Kinematic arcade flight model. Not physical: no airfoil, no lift, no
//! control surfaces. Energy is exchanged between altitude and airspeed via
//! gravity acting along the body-forward axis (`g * sin(pitch)`); drag is
//! quadratic. Throttle is the command, airspeed is the integrated state.

const std = @import("std");
const math = @import("math");
const pose_mod = @import("pose.zig");
const Pose = pose_mod.Pose;
const Aircraft = @import("aircraft.zig").Aircraft;
const Controls = @import("controls.zig").Controls;
const coords = @import("../terrain/coords.zig");

const PITCH_RATE: f32 = 1.5; // rad/s
const ROLL_RATE: f32 = 2.5; // rad/s
const YAW_RATE: f32 = 0.6; // rad/s

/// Throttle command slew rate. 0 -> 1 in ~2.9s of held Shift.
const THROTTLE_RATE: f32 = 0.35;

/// Game gravity in arcsec/s^2. Inflated over real earth gravity so that
/// dive-acceleration / climb-deceleration is felt at our compressed time and
/// distance scales.
const GRAVITY: f32 = 1.5;

/// Max thrust at full throttle in arcsec/s^2. Kept below GRAVITY so a vertical
/// climb at full throttle bleeds airspeed (eventually stalling to zero), which
/// is the consequence the player feels for pointing the nose straight up.
const MAX_THRUST: f32 = 1.4;
const CRUISE_AIRSPEED: f32 = 6.75;
/// k * v^2 = thrust at cruise -> k = T_max / v_cruise^2. Drag is the only
/// terminal-velocity limiter; no hard cap on airspeed. Resulting terminals:
///   level cruise (full throttle):  sqrt(T/k)     = 6.75 arcsec/s ~= 750 km/h
///   free-fall  (zero throttle):    sqrt(g/k)     = 6.99 arcsec/s ~= 776 km/h
///   powered dive (full throttle):  sqrt((T+g)/k) = 9.70 arcsec/s ~= 1077 km/h
/// Free-fall terminal exceeds cruise because T < g (required so vertical
/// climbs bleed).
const K_DRAG: f32 = MAX_THRUST / (CRUISE_AIRSPEED * CRUISE_AIRSPEED);

/// Counter to "banked plane's nose pulls down": auto-pitch-up proportional
/// to `sin^2(bank)`. `right.y == sin(bank)` for level pitch.
const K_BANK_PITCH: f32 = 0.4;

/// Auto-roll-leveling strength. Applied only when no roll stick input.
const K_AUTO_LEVEL: f32 = 0.5;

/// Bank-to-turn yaw coupling. Not coordinated (real value is `g*tan(bank)/V`).
const BTT_COEFF: f32 = 0.05;

/// Control surface smoothing. Higher = snappier. ~10 gives ~100ms ramp at 120Hz.
const CONTROL_LAG: f32 = 10.0;

/// Speed at which control surfaces reach full authority.
const AUTHORITY_FULL_SPEED: f32 = 6.0;
/// Minimum authority fraction at zero airspeed.
const AUTHORITY_MIN: f32 = 0.2;

/// Airbrake drag multiplier (applied when throttle-down held at zero throttle).
const K_AIRBRAKE: f32 = K_DRAG * 4.0;

/// Stall onset speed (arcsec/s). Below this, stall effects begin.
const STALL_SPEED_ONSET: f32 = 2.5;
/// Full stall speed (arcsec/s). Below this, fully stalled.
const STALL_SPEED_FULL: f32 = 1.0;
/// Nose-down torque at full stall (rad/s).
const STALL_DROP: f32 = 0.5;
/// Authority reduction at full stall (0-1).
const STALL_AUTHORITY_LOSS: f32 = 0.7;

/// Fixed physics tick. 120 Hz: snappy enough that input latency stays below
/// human perception (~10 ms); cheap enough we have plenty of CPU headroom.
pub const FIXED_DT: f32 = 1.0 / 120.0;
/// Cap accumulator to prevent the death-spiral after a long stall (debugger
/// breakpoint, GC pause, alt-tab). Drop time rather than try to "catch up".
const MAX_ACCUMULATOR: f32 = 0.25;

/// Drives `applySimInput` at fixed dt regardless of render frame rate, and
/// retains the pre-tick pose snapshot so the render side can interpolate
/// between the last two physics states.
pub const Ticker = struct {
    accumulator: f32 = 0,
    prev_pose: Pose,

    pub fn init(aircraft_pose: Pose) Ticker {
        return .{ .prev_pose = aircraft_pose };
    }

    /// Returns interpolation alpha in [0, 1) for render-side pose smoothing.
    /// `controls` are sampled at frame time and re-applied unchanged for
    /// every catch-up tick this frame; sub-tick input transitions don't exist
    /// at any plausible frame rate (smallest human keypress >> FIXED_DT).
    pub fn step(self: *Ticker, aircraft: *Aircraft, frame_dt: f32, controls: Controls) f32 {
        self.accumulator = @min(self.accumulator + frame_dt, MAX_ACCUMULATOR);
        while (self.accumulator >= FIXED_DT) {
            self.prev_pose = aircraft.pose;
            applySimInput(aircraft, FIXED_DT, controls);
            self.accumulator -= FIXED_DT;
        }
        return self.accumulator / FIXED_DT;
    }

    /// Drop accumulated time and snap the lerp anchor forward. Call when
    /// pausing (free mode) so re-entry doesn't lerp from a stale snapshot.
    pub fn reset(self: *Ticker, aircraft_pose: Pose) void {
        self.accumulator = 0;
        self.prev_pose = aircraft_pose;
    }
};

pub fn applySimInput(aircraft: *Aircraft, dt: f32, controls: Controls) void {
    const prev = aircraft.pose.position;

    aircraft.throttle = std.math.clamp(
        aircraft.throttle + controls.throttle_delta * THROTTLE_RATE * dt,
        0.0,
        1.0,
    );

    const smooth_blend = math.expBlend(CONTROL_LAG, dt);
    aircraft.smooth_pitch += (controls.pitch - aircraft.smooth_pitch) * smooth_blend;
    aircraft.smooth_roll += (controls.roll - aircraft.smooth_roll) * smooth_blend;
    aircraft.smooth_yaw += (controls.yaw - aircraft.smooth_yaw) * smooth_blend;

    const authority = std.math.clamp(aircraft.airspeed / AUTHORITY_FULL_SPEED, AUTHORITY_MIN, 1.0);

    var pitch_delta = aircraft.smooth_pitch * PITCH_RATE * authority * dt;
    var roll_delta = aircraft.smooth_roll * ROLL_RATE * authority * dt;
    var yaw_delta = aircraft.smooth_yaw * YAW_RATE * authority * dt;

    // TODO: collapse front/right/up into a single `pose.basis()` so this
    // block (and scene.zig:134) doesn't pay 3x quatRotateVec3 per frame.
    const f = aircraft.pose.front();
    const r = aircraft.pose.right();
    const u = aircraft.pose.up();

    pitch_delta += K_BANK_PITCH * r[1] * r[1] * dt;
    if (controls.roll == 0 and aircraft.auto_level)
        roll_delta -= K_AUTO_LEVEL * r[1] * dt;
    yaw_delta -= BTT_COEFF * r[1] * aircraft.airspeed * dt;

    if (aircraft.airspeed < STALL_SPEED_ONSET) {
        const stall_t = std.math.clamp(
            (STALL_SPEED_ONSET - aircraft.airspeed) / (STALL_SPEED_ONSET - STALL_SPEED_FULL),
            0.0,
            1.0,
        );
        pitch_delta *= (1.0 - STALL_AUTHORITY_LOSS * stall_t);
        pitch_delta -= STALL_DROP * stall_t * dt;
    }

    if (pitch_delta != 0 or roll_delta != 0 or yaw_delta != 0) {
        const q_pitch = math.quatFromAxisAngle(r, pitch_delta);
        const q_roll = math.quatFromAxisAngle(f, -roll_delta);
        const q_yaw = math.quatFromAxisAngle(u, -yaw_delta);
        const rotation = math.quatMul(q_yaw, math.quatMul(q_pitch, q_roll));
        aircraft.pose.orientation = math.quatNormalize(math.quatMul(rotation, aircraft.pose.orientation));
    }

    const f_post = aircraft.pose.front();
    const braking = controls.throttle_delta < 0 and aircraft.throttle <= 0;
    integrateAirspeed(aircraft, f_post, dt, braking);
    movePosition(aircraft, f_post, dt);
    aircraft.pose.updateVelocityEma(prev, dt, pose_mod.VELOCITY_ALPHA);
}

fn integrateAirspeed(aircraft: *Aircraft, front: math.Vec3, dt: f32, braking: bool) void {
    const thrust = MAX_THRUST * aircraft.throttle;
    const k = if (braking) K_DRAG + K_AIRBRAKE else K_DRAG;
    const drag = k * aircraft.airspeed * aircraft.airspeed;
    const gravity_loss = GRAVITY * front[1];
    aircraft.airspeed = @max(aircraft.airspeed + (thrust - drag - gravity_loss) * dt, 0.0);
}

fn movePosition(aircraft: *Aircraft, front: math.Vec3, dt: f32) void {
    const inv_cos_lat = coords.invCosLatFromZD(aircraft.pose.position[2]);
    const move: f64 = @as(f64, aircraft.airspeed) * @as(f64, dt);
    Pose.translateLonCorrected(&aircraft.pose.position, front, move, inv_cos_lat);
}

/// Throttle that holds `airspeed` in steady level flight (thrust == drag, no net
/// acceleration). Inverts `integrateAirspeed`'s level-flight equilibrium
/// `MAX_THRUST * throttle == K_DRAG * v^2`, clamped to [0, 1] (airspeeds above
/// the full-throttle cruise terminal can't be held, so they pin at full throttle
/// and then bleed). Used to trim the spawn so the aircraft cruises on launch.
pub fn trimThrottle(airspeed: f32) f32 {
    return std.math.clamp(K_DRAG * airspeed * airspeed / MAX_THRUST, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const Vec3d = math.Vec3d;
const test_pos: Vec3d = .{ 1800, coords.metersToArcsec(1200), 1800 };

fn testAircraft() Aircraft {
    return Aircraft.init(test_pos, null);
}

const zero: Controls = .{};

test "sim: throttle_delta +1 ramps throttle, -1 retracts and clamps at zero" {
    var ac = testAircraft();
    applySimInput(&ac, 0.5, .{ .throttle_delta = 1 });
    try testing.expect(ac.throttle > 0);
    try testing.expect(ac.throttle <= 1.0);

    for (0..100) |_| applySimInput(&ac, 0.1, .{ .throttle_delta = -1 });
    try testing.expectEqual(@as(f32, 0), ac.throttle);
}

test "sim: full throttle level flight converges to cruise airspeed" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.throttle = 1.0;
    for (0..1200) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expectApproxEqAbs(CRUISE_AIRSPEED, ac.airspeed, 0.1);
}

test "sim: trimThrottle holds its airspeed in level flight" {
    // Level flight (testAircraft spawns level): thrust == drag, so airspeed holds.
    // Cover more than the production default (5.0) so trim drift at other speeds
    // is caught; both are above STALL_SPEED_ONSET so no stall torque interferes.
    for ([_]f32{ 3.0, 5.0 }) |v| {
        var ac = testAircraft();
        ac.airspeed = v;
        ac.throttle = trimThrottle(v);
        for (0..600) |_| applySimInput(&ac, 1.0 / 60.0, zero);
        try testing.expectApproxEqAbs(v, ac.airspeed, 0.05);
    }
}

test "sim: trimThrottle clamps to [0,1] and is zero at zero airspeed" {
    try testing.expectEqual(@as(f32, 0), trimThrottle(0));
    try testing.expectApproxEqAbs(@as(f32, 1.0), trimThrottle(CRUISE_AIRSPEED), 1e-5);
    try testing.expectEqual(@as(f32, 1.0), trimThrottle(CRUISE_AIRSPEED * 2)); // above terminal -> pinned
}

test "sim: pitch down accelerates airspeed (gravity)" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.throttle = 0.3;
    for (0..600) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    const cruise_speed = ac.airspeed;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 1, 0, 0 }, -0.52);
    for (0..600) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.airspeed > cruise_speed);
}

test "sim: pitch up decelerates airspeed (gravity)" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.throttle = 0.5;
    for (0..600) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    const cruise_speed = ac.airspeed;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 1, 0, 0 }, 0.52);
    for (0..120) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.airspeed < cruise_speed);
}

test "sim: zero throttle level flight decays airspeed via drag" {
    var ac = testAircraft();
    ac.airspeed = 8.0;
    // Quadratic decay 1 / (1 + k*v0*t): at t=10s with k=0.02, v ~= 8/2.6 ~= 3.
    for (0..600) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.airspeed < 4.0);
    try testing.expect(ac.airspeed > 0.0);
}

test "sim: nonzero airspeed moves aircraft along front axis" {
    var ac = testAircraft();
    ac.airspeed = 5.0;
    const z_before = ac.pose.position[2];
    applySimInput(&ac, 0.1, zero);
    try testing.expect(ac.pose.position[2] < z_before);
}

test "sim: pitch input pitches nose up" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    const front_y_before = ac.pose.front()[1];
    applySimInput(&ac, 0.5, .{ .pitch = 1 });
    try testing.expect(ac.pose.front()[1] > front_y_before);
}

test "sim: roll input rolls left (right vector y becomes positive)" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.auto_level = false;
    applySimInput(&ac, 0.5, .{ .roll = 1 });
    try testing.expect(ac.pose.right()[1] > 0);
}

test "sim: yaw input rotates nose left" {
    var ac = testAircraft();
    ac.airspeed = 8.0;
    for (0..60) |_| applySimInput(&ac, 1.0 / 60.0, .{ .yaw = -1 });
    try testing.expect(ac.pose.front()[0] < -0.1);
}

test "sim: auto-level rolls wings toward level when stick released" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 0, 0, -1 }, -0.5);
    const right_y_before = ac.pose.right()[1];
    try testing.expect(right_y_before > 0.4);
    for (0..120) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.right()[1] < right_y_before * 0.5);
}

test "sim: auto-level does nothing when disabled" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.auto_level = false;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 0, 0, -1 }, -0.5);
    const right_y_before = ac.pose.right()[1];
    for (0..120) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expectApproxEqAbs(right_y_before, ac.pose.right()[1], 0.02);
}

test "sim: roll input overrides auto-level" {
    var ac = testAircraft();
    ac.airspeed = 5.0;
    for (0..30) |_| applySimInput(&ac, 1.0 / 60.0, .{ .roll = 1 });
    try testing.expect(ac.pose.right()[1] > 0.5);
}

test "sim: BTT induces yaw when banked and throttled" {
    var ac = testAircraft();
    ac.auto_level = false;
    ac.airspeed = 5.0;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 0, 0, -1 }, -0.5);
    try testing.expect(ac.pose.right()[1] > 0);
    const front_x_before = ac.pose.front()[0];
    for (0..30) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.front()[0] < front_x_before);
}

test "sim: powered dive faster than unpowered" {
    var powered = testAircraft();
    powered.airspeed = STALL_SPEED_ONSET;
    powered.throttle = 1.0;
    powered.pose.orientation = math.quatFromAxisAngle(.{ 1, 0, 0 }, -0.35);
    var unpowered = testAircraft();
    unpowered.airspeed = STALL_SPEED_ONSET;
    unpowered.throttle = 0;
    unpowered.pose.orientation = powered.pose.orientation;
    for (0..3600) |_| {
        applySimInput(&powered, 1.0 / 60.0, zero);
        applySimInput(&unpowered, 1.0 / 60.0, zero);
    }
    try testing.expect(powered.airspeed > unpowered.airspeed);
    try testing.expect(unpowered.airspeed > 3.0);
}

test "sim: steep climb bleeds airspeed" {
    var ac = testAircraft();
    ac.throttle = 1.0;
    ac.airspeed = 10.0;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 1, 0, 0 }, 0.35);
    const start_speed = ac.airspeed;
    for (0..120) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.airspeed < start_speed);
}

test "sim: bank-pitch-up corrects 'nose drops in turn'" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_ONSET;
    ac.auto_level = false;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 0, 0, -1 }, -0.5);
    const front_y_before = ac.pose.front()[1];
    for (0..60) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.front()[1] > front_y_before);
}

// ---- Stall ----

test "sim: zero airspeed = full stall, nose drops" {
    var ac = testAircraft();
    ac.airspeed = 0;
    const front_y_before = ac.pose.front()[1];
    for (0..60) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.front()[1] < front_y_before);
}

test "sim: high speed steep pitch has no stall" {
    var ac = testAircraft();
    ac.airspeed = 8.0;
    ac.throttle = 1.0;
    ac.pose.orientation = math.quatFromAxisAngle(.{ 1, 0, 0 }, 1.2);
    const front_y_before = ac.pose.front()[1];
    applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.front()[1] >= front_y_before - 0.01);
}

test "sim: low speed level flight stalls and nose drops" {
    var ac = testAircraft();
    ac.airspeed = STALL_SPEED_FULL * 0.5;
    ac.throttle = 0;
    const front_y_before = ac.pose.front()[1];
    for (0..60) |_| applySimInput(&ac, 1.0 / 60.0, zero);
    try testing.expect(ac.pose.front()[1] < front_y_before);
}

// ---- Ticker ----

test "sim: Ticker small frame_dt accumulates without ticking" {
    var ac = testAircraft();
    var t = Ticker.init(ac.pose);
    const alpha = t.step(&ac, FIXED_DT * 0.5, zero);
    try testing.expectApproxEqAbs(FIXED_DT * 0.5, t.accumulator, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.5), alpha, 1e-4);
}

test "sim: Ticker frame_dt = exactly N ticks runs N times, alpha returns to 0" {
    var ac = testAircraft();
    ac.airspeed = 5.0;
    var t = Ticker.init(ac.pose);
    const alpha = t.step(&ac, FIXED_DT * 3.0, zero);
    try testing.expectApproxEqAbs(@as(f32, 0), t.accumulator, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), alpha, 1e-4);
    try testing.expect(ac.pose.position[2] != t.prev_pose.position[2]);
}

test "sim: Ticker large frame_dt clamps at MAX_ACCUMULATOR" {
    var ac = testAircraft();
    var t = Ticker.init(ac.pose);
    _ = t.step(&ac, 10.0, zero);
    try testing.expect(t.accumulator < FIXED_DT);
    try testing.expect(t.accumulator >= 0);
}

test "sim: Ticker prev_pose holds pre-tick state of the latest tick" {
    var ac = testAircraft();
    ac.airspeed = 5.0;
    var t = Ticker.init(ac.pose);
    const start_z = ac.pose.position[2];
    _ = t.step(&ac, FIXED_DT * 2.5, zero);
    try testing.expect(t.prev_pose.position[2] < start_z);
    try testing.expect(ac.pose.position[2] < t.prev_pose.position[2]);
}

test "sim: Ticker reset clears accumulator and snaps prev_pose forward" {
    var ac = testAircraft();
    ac.airspeed = 5.0;
    var t = Ticker.init(ac.pose);
    _ = t.step(&ac, FIXED_DT * 2.5, zero);
    try testing.expect(t.accumulator > 0);
    try testing.expect(t.prev_pose.position[2] != ac.pose.position[2]);
    t.reset(ac.pose);
    try testing.expectEqual(@as(f32, 0), t.accumulator);
    try testing.expectEqual(ac.pose.position[2], t.prev_pose.position[2]);
}
