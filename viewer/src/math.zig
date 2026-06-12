//! Minimal linear algebra for camera and projection math.
//!
//! Column-major Mat4 layout: mat[col][row]. Matches Vulkan/GLSL expectations
//! when cast to push constant bytes; no transpose needed.

const std = @import("std");

pub const Vec3 = [3]f32;
/// Column-major 4x4 matrix. Access as mat[col][row].
pub const Mat4 = [4][4]f32;

pub const identity: Mat4 = .{
    .{ 1, 0, 0, 0 },
    .{ 0, 1, 0, 0 },
    .{ 0, 0, 1, 0 },
    .{ 0, 0, 0, 1 },
};

/// World-fixed up axis (Y+).
pub const world_up: Vec3 = .{ 0, 1, 0 };

pub fn sub(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

pub fn dot(a: Vec3, b: Vec3) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

pub fn cross(a: Vec3, b: Vec3) Vec3 {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

/// 2D cross product (the z-component of the 3D cross with y=0).
/// Sign: positive when `b` is CCW of `a` in standard 2D (X right, Y up).
/// Used for SAT half-plane tests where only the sign matters.
pub fn cross2(a: [2]f32, b: [2]f32) f32 {
    return a[0] * b[1] - a[1] * b[0];
}

pub fn normalize(v: Vec3) Vec3 {
    const len = @sqrt(dot(v, v));
    if (len == 0) return .{ 0, 0, 0 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

/// Column-major matrix multiply: matMul(A, B) = mathematical A * B.
/// For MVP: matMul(projection, matMul(view, model)).
pub fn matMul(a: Mat4, b: Mat4) Mat4 {
    var result: Mat4 = undefined;
    for (0..4) |col| {
        for (0..4) |row| {
            var sum: f32 = 0;
            for (0..4) |k| {
                sum += a[k][row] * b[col][k];
            }
            result[col][row] = sum;
        }
    }
    return result;
}

/// Right-handed perspective projection with reverse-Z depth.
/// Vulkan clip space: Y down, Z in [0, 1] with **near plane at Z=1, far plane at Z=0**.
/// Combined with a float depth buffer (D32_SFLOAT), this yields near-uniform depth
/// precision across the full near/far range, critical for our 0.01 to 100k+ arcsec ratio.
/// Pipelines must use `compare_op = greater_or_equal` and clear depth to 0.0.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const half_tan = @tan(fov_y / 2.0);
    const z_range = far - near;
    return .{
        .{ 1.0 / (aspect * half_tan), 0, 0, 0 },
        .{ 0, -1.0 / half_tan, 0, 0 },
        .{ 0, 0, near / z_range, -1 },
        .{ 0, 0, (far * near) / z_range, 0 },
    };
}

/// Right-handed look-at view matrix.
pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
    const f = normalize(sub(target, eye));
    const s = normalize(cross(f, up));
    const u = cross(s, f);
    return .{
        .{ s[0], u[0], -f[0], 0 },
        .{ s[1], u[1], -f[1], 0 },
        .{ s[2], u[2], -f[2], 0 },
        .{ -dot(s, eye), -dot(u, eye), dot(f, eye), 1 },
    };
}

// ---------------------------------------------------------------------------
// f64 matrix math, for computing MVP on CPU with full precision.
// Large world coordinates (~400K arcseconds) cause catastrophic cancellation
// in f32 matrix multiply. Computing in f64 and truncating the final MVP to
// f32 preserves full precision in the matrix entries (same approach as
// srg-synvis via Go's default float64 arithmetic).
// ---------------------------------------------------------------------------

pub const Vec3d = [3]f64;
pub const Mat4d = [4][4]f64;

fn toF64(v: Vec3) Vec3d {
    return .{ v[0], v[1], v[2] };
}

fn dotD(a: Vec3d, b: Vec3d) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn subD(a: Vec3d, b: Vec3d) Vec3d {
    return .{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
}

fn crossD(a: Vec3d, b: Vec3d) Vec3d {
    return .{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

fn normalizeD(v: Vec3d) Vec3d {
    const len = @sqrt(dotD(v, v));
    if (len == 0) return .{ 0, 0, 0 };
    return .{ v[0] / len, v[1] / len, v[2] / len };
}

pub fn lookAtD(eye: Vec3, target: Vec3, up: Vec3) Mat4d {
    const eye_d = toF64(eye);
    const f = normalizeD(subD(toF64(target), eye_d));
    const s = normalizeD(crossD(f, toF64(up)));
    const u = crossD(s, f);
    return .{
        .{ s[0], u[0], -f[0], 0 },
        .{ s[1], u[1], -f[1], 0 },
        .{ s[2], u[2], -f[2], 0 },
        .{ -dotD(s, eye_d), -dotD(u, eye_d), dotD(f, eye_d), 1 },
    };
}

/// Reverse-Z perspective in f64 (see `perspective` doc). Used for MVP composition
/// at large world coordinates where f32 cancellation would erase precision.
pub fn perspectiveD(fov_y: f32, aspect: f32, near: f32, far: f32) Mat4d {
    const half_tan: f64 = @tan(@as(f64, fov_y) / 2.0);
    const a: f64 = aspect;
    const n: f64 = near;
    const f: f64 = far;
    const z_range = f - n;
    return .{
        .{ 1.0 / (a * half_tan), 0, 0, 0 },
        .{ 0, -1.0 / half_tan, 0, 0 },
        .{ 0, 0, n / z_range, -1 },
        .{ 0, 0, (f * n) / z_range, 0 },
    };
}

pub fn matMulD(a: Mat4d, b: Mat4d) Mat4d {
    var result: Mat4d = undefined;
    for (0..4) |col| {
        for (0..4) |row| {
            var s: f64 = 0;
            for (0..4) |k| {
                s += a[k][row] * b[col][k];
            }
            result[col][row] = s;
        }
    }
    return result;
}

pub fn mat4dToMat4(m: Mat4d) Mat4 {
    var result: Mat4 = undefined;
    for (0..4) |col| {
        for (0..4) |row| {
            result[col][row] = @floatCast(m[col][row]);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Vec3 arithmetic
// ---------------------------------------------------------------------------

pub fn add(a: Vec3, b: Vec3) Vec3 {
    return .{ a[0] + b[0], a[1] + b[1], a[2] + b[2] };
}

pub fn scale(v: Vec3, s: f32) Vec3 {
    return .{ v[0] * s, v[1] * s, v[2] * s };
}

// ---------------------------------------------------------------------------
// Quaternions: (x, y, z, w) layout, Hamilton product convention
// ---------------------------------------------------------------------------

/// Quaternion: (x, y, z, w) where w is the scalar part.
pub const Quat = [4]f32;

pub const quat_identity: Quat = .{ 0, 0, 0, 1 };

/// Create a unit quaternion representing a rotation of `angle` radians around `axis`.
/// `axis` must be normalized.
pub fn quatFromAxisAngle(axis: Vec3, angle: f32) Quat {
    const half = angle * 0.5;
    const s = @sin(half);
    return .{ axis[0] * s, axis[1] * s, axis[2] * s, @cos(half) };
}

/// Hamilton product: apply rotation b first, then a.
pub fn quatMul(a: Quat, b: Quat) Quat {
    return .{
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    };
}

/// Rotate vector v by unit quaternion q: q * v * q_conjugate.
pub fn quatRotateVec3(q: Quat, v: Vec3) Vec3 {
    // Optimized formula: v' = v + 2w(u x v) + 2(u x (u x v))
    // where q = (u, w), u = (x, y, z)
    const u = Vec3{ q[0], q[1], q[2] };
    const w = q[3];
    const uv = cross(u, v);
    const uuv = cross(u, uv);
    return .{
        v[0] + 2.0 * (w * uv[0] + uuv[0]),
        v[1] + 2.0 * (w * uv[1] + uuv[1]),
        v[2] + 2.0 * (w * uv[2] + uuv[2]),
    };
}

pub fn expBlend(rate: f32, dt: f32) f32 {
    return 1.0 - @exp(-rate * dt);
}

pub fn wrapAngle(x: f32) f32 {
    const tau = 2.0 * std.math.pi;
    return x - tau * @floor(x / tau + 0.5);
}

pub fn lerpF32(a: f32, b: f32, t: f32) f32 {
    return (1.0 - t) * a + t * b;
}

pub fn lerpVec3(a: Vec3, b: Vec3, t: f32) Vec3 {
    const inv = 1.0 - t;
    return .{ inv * a[0] + t * b[0], inv * a[1] + t * b[1], inv * a[2] + t * b[2] };
}

pub fn lerpVec3d(a: Vec3d, b: Vec3d, t: f64) Vec3d {
    const inv = 1.0 - t;
    return .{ inv * a[0] + t * b[0], inv * a[1] + t * b[1], inv * a[2] + t * b[2] };
}

/// Normalize quaternion to unit length. Call periodically to prevent drift.
pub fn quatNormalize(q: Quat) Quat {
    const len = @sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (len == 0) return quat_identity;
    return .{ q[0] / len, q[1] / len, q[2] / len, q[3] / len };
}

/// Normalized lerp between unit quaternions. Cheap (no acos/sin like true
/// slerp) and visually indistinguishable from slerp for the small angular
/// deltas typical of fixed-timestep tick-to-tick render interpolation. Picks
/// the shorter arc by negating `b` when `dot(a,b) < 0`.
pub fn quatNlerp(a: Quat, b: Quat, t: f32) Quat {
    const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const b_signed: Quat = if (d < 0) .{ -b[0], -b[1], -b[2], -b[3] } else b;
    return quatNormalize(.{
        lerpF32(a[0], b_signed[0], t),
        lerpF32(a[1], b_signed[1], t),
        lerpF32(a[2], b_signed[2], t),
        lerpF32(a[3], b_signed[3], t),
    });
}

/// Quaternion that rotates `from` to `to`, both expected unit-length. Used to
/// orient cameras/aircraft from a "look at" direction. `fallback_axis` is used
/// when `from` and `to` are antiparallel (the rotation axis is ambiguous).
pub fn quatFromDir(from: Vec3, to: Vec3, fallback_axis: Vec3) Quat {
    const d = normalize(to);
    const dp = dot(from, d);
    if (dp > 0.9999) return quat_identity;
    if (dp < -0.9999) return quatFromAxisAngle(fallback_axis, std.math.pi);
    const axis = normalize(cross(from, d));
    return quatFromAxisAngle(axis, std.math.acos(std.math.clamp(dp, -1.0, 1.0)));
}

/// Extract yaw (around world Y) and pitch (nose up/down) from a unit
/// quaternion, discarding roll. Uses the rotated front vector (-Z basis).
pub fn yawPitchFromQuat(q: Quat) [2]f32 {
    const f = quatRotateVec3(q, Vec3{ 0, 0, -1 });
    const yaw = std.math.atan2(-f[0], -f[2]);
    const pitch = std.math.asin(std.math.clamp(f[1], -1.0, 1.0));
    return .{ yaw, pitch };
}

/// Compass heading in radians from an orientation quaternion: 0 = North,
/// pi/2 = East, pi = South, 3*pi/2 = West, normalized to [0, 2*pi). This is the
/// negation of `yawPitchFromQuat`'s yaw (which is measured counter-clockwise), so
/// HUD/compass consumers get the sign convention from one place instead of each
/// re-deriving it.
pub fn headingFromQuat(q: Quat) f32 {
    return @mod(-yawPitchFromQuat(q)[0], std.math.tau);
}

/// Column-major 3x3 rotation matrix in f64 from a unit quaternion. The 3
/// columns are the rotated standard basis vectors (X, Y, Z).
pub fn quatToMat3D(q: Quat) [3][3]f64 {
    const x: f64 = q[0];
    const y: f64 = q[1];
    const z: f64 = q[2];
    const w: f64 = q[3];
    const xx = x * x;
    const yy = y * y;
    const zz = z * z;
    const xy = x * y;
    const xz = x * z;
    const yz = y * z;
    const wx = w * x;
    const wy = w * y;
    const wz = w * z;
    return .{
        .{ 1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy) },
        .{ 2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx) },
        .{ 2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy) },
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
const testing = std.testing;
const eps: f32 = 1e-5;

fn expectVec3Approx(expected: Vec3, actual: Vec3) !void {
    try testing.expectApproxEqAbs(expected[0], actual[0], eps);
    try testing.expectApproxEqAbs(expected[1], actual[1], eps);
    try testing.expectApproxEqAbs(expected[2], actual[2], eps);
}

fn expectMat4Approx(expected: Mat4, actual: Mat4) !void {
    for (0..4) |i| {
        for (0..4) |j| {
            try testing.expectApproxEqAbs(expected[i][j], actual[i][j], eps);
        }
    }
}

// ---- sub ----

test "sub: basic subtraction" {
    const result = sub(.{ 5, 3, 1 }, .{ 2, 1, 4 });
    try expectVec3Approx(.{ 3, 2, -3 }, result);
}

test "sub: subtracting zero vector" {
    const result = sub(.{ 7, -2, 3 }, .{ 0, 0, 0 });
    try expectVec3Approx(.{ 7, -2, 3 }, result);
}

test "sub: negative values" {
    const result = sub(.{ -1, -2, -3 }, .{ -4, -5, -6 });
    try expectVec3Approx(.{ 3, 3, 3 }, result);
}

test "sub: self subtraction gives zero" {
    const v: Vec3 = .{ 42, -17, 3.14 };
    const result = sub(v, v);
    try expectVec3Approx(.{ 0, 0, 0 }, result);
}

// ---- dot ----

test "dot: orthogonal vectors give zero" {
    try testing.expectApproxEqAbs(@as(f32, 0), dot(.{ 1, 0, 0 }, .{ 0, 1, 0 }), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), dot(.{ 1, 0, 0 }, .{ 0, 0, 1 }), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), dot(.{ 0, 1, 0 }, .{ 0, 0, 1 }), eps);
}

test "dot: parallel vectors" {
    // dot(v, v) = |v|^2
    try testing.expectApproxEqAbs(@as(f32, 14), dot(.{ 1, 2, 3 }, .{ 1, 2, 3 }), eps);
}

test "dot: anti-parallel vectors" {
    try testing.expectApproxEqAbs(@as(f32, -14), dot(.{ 1, 2, 3 }, .{ -1, -2, -3 }), eps);
}

test "dot: known values" {
    // (2,3,4) . (5,6,7) = 10 + 18 + 28 = 56
    try testing.expectApproxEqAbs(@as(f32, 56), dot(.{ 2, 3, 4 }, .{ 5, 6, 7 }), eps);
}

// ---- cross ----

test "cross: i x j = k" {
    const result = cross(.{ 1, 0, 0 }, .{ 0, 1, 0 });
    try expectVec3Approx(.{ 0, 0, 1 }, result);
}

test "cross: j x k = i" {
    const result = cross(.{ 0, 1, 0 }, .{ 0, 0, 1 });
    try expectVec3Approx(.{ 1, 0, 0 }, result);
}

test "cross: k x i = j" {
    const result = cross(.{ 0, 0, 1 }, .{ 1, 0, 0 });
    try expectVec3Approx(.{ 0, 1, 0 }, result);
}

test "cross: anti-commutativity a x b = -(b x a)" {
    const a: Vec3 = .{ 2, 3, 4 };
    const b: Vec3 = .{ 5, 6, 7 };
    const ab = cross(a, b);
    const ba = cross(b, a);
    try expectVec3Approx(.{ -ba[0], -ba[1], -ba[2] }, ab);
}

test "cross: self cross product is zero" {
    const v: Vec3 = .{ 3, -1, 7 };
    const result = cross(v, v);
    try expectVec3Approx(.{ 0, 0, 0 }, result);
}

// ---- normalize ----

test "normalize: unit vector stays unit" {
    const result = normalize(.{ 1, 0, 0 });
    try expectVec3Approx(.{ 1, 0, 0 }, result);
}

test "normalize: known vector (3,4,0)" {
    const result = normalize(.{ 3, 4, 0 });
    try expectVec3Approx(.{ 0.6, 0.8, 0 }, result);
}

test "normalize: negative components" {
    const result = normalize(.{ 0, -3, 4 });
    try expectVec3Approx(.{ 0, -0.6, 0.8 }, result);
}

test "normalize: zero vector returns zero" {
    const result = normalize(.{ 0, 0, 0 });
    try expectVec3Approx(.{ 0, 0, 0 }, result);
}

test "normalize: result has unit length" {
    const result = normalize(.{ 7, -2, 5 });
    const len = @sqrt(dot(result, result));
    try testing.expectApproxEqAbs(@as(f32, 1.0), len, eps);
}

// ---- matMul ----

test "matMul: identity * M = M" {
    const m: Mat4 = .{
        .{ 1, 2, 3, 4 },
        .{ 5, 6, 7, 8 },
        .{ 9, 10, 11, 12 },
        .{ 13, 14, 15, 16 },
    };
    const result = matMul(identity, m);
    try expectMat4Approx(m, result);
}

test "matMul: M * identity = M" {
    const m: Mat4 = .{
        .{ 1, 2, 3, 4 },
        .{ 5, 6, 7, 8 },
        .{ 9, 10, 11, 12 },
        .{ 13, 14, 15, 16 },
    };
    const result = matMul(m, identity);
    try expectMat4Approx(m, result);
}

test "matMul: known column-major multiplication" {
    // Column-major: mat[col][row]
    // A = | 1  0 |  (upper-left 2x2, rest identity)
    //     | 2  1 |
    const a: Mat4 = .{
        .{ 1, 2, 0, 0 }, // col 0
        .{ 0, 1, 0, 0 }, // col 1
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    };
    // B = | 1  3 |
    //     | 0  1 |
    const b: Mat4 = .{
        .{ 1, 0, 0, 0 }, // col 0
        .{ 3, 1, 0, 0 }, // col 1
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    };
    // A * B = | 1*1+0*0  1*3+0*1 | = | 1  3 |
    //         | 2*1+1*0  2*3+1*1 |   | 2  7 |
    const expected: Mat4 = .{
        .{ 1, 2, 0, 0 }, // col 0
        .{ 3, 7, 0, 0 }, // col 1
        .{ 0, 0, 1, 0 },
        .{ 0, 0, 0, 1 },
    };
    const result = matMul(a, b);
    try expectMat4Approx(expected, result);
}

test "matMul: associativity (A*B)*C = A*(B*C)" {
    const a: Mat4 = .{
        .{ 1, 2, 3, 4 },
        .{ 5, 6, 7, 8 },
        .{ 9, 10, 11, 12 },
        .{ 13, 14, 15, 16 },
    };
    const b: Mat4 = .{
        .{ 2, 0, 0, 1 },
        .{ 0, 3, 0, 0 },
        .{ 0, 0, 4, 0 },
        .{ 1, 0, 0, 5 },
    };
    const c: Mat4 = .{
        .{ 1, 0, 1, 0 },
        .{ 0, 1, 0, 1 },
        .{ 1, 0, 1, 0 },
        .{ 0, 1, 0, 1 },
    };
    const ab_c = matMul(matMul(a, b), c);
    const a_bc = matMul(a, matMul(b, c));
    // Use slightly relaxed tolerance for accumulated float error
    for (0..4) |i| {
        for (0..4) |j| {
            try testing.expectApproxEqAbs(ab_c[i][j], a_bc[i][j], 1e-3);
        }
    }
}

// ---- perspective ----

test "perspective: aspect ratio appears in [0][0]" {
    const fov = std.math.pi / 4.0; // 45 degrees
    const aspect: f32 = 2.0;
    const m = perspective(fov, aspect, 0.1, 100.0);
    const half_tan = @tan(fov / 2.0);
    try testing.expectApproxEqAbs(1.0 / (aspect * half_tan), m[0][0], eps);
}

test "perspective: Y is flipped (negative [1][1])" {
    const m = perspective(std.math.pi / 4.0, 1.0, 0.1, 100.0);
    try testing.expect(m[1][1] < 0);
}

test "perspective: near plane maps to Z=1 (reverse-Z)" {
    const near: f32 = 0.1;
    const far: f32 = 100.0;
    const m = perspective(std.math.pi / 4.0, 1.0, near, far);

    // Transform point at near plane: (0, 0, -near, 1)
    // Column-major storage: m[col][row]
    const z_in = -near;
    const z_clip = m[2][2] * z_in + m[3][2];
    const w_clip = m[2][3] * z_in + m[3][3];
    try testing.expectApproxEqAbs(@as(f32, 1), z_clip / w_clip, eps);
}

test "perspective: far plane maps to Z=0 (reverse-Z)" {
    const near: f32 = 0.1;
    const far: f32 = 100.0;
    const m = perspective(std.math.pi / 4.0, 1.0, near, far);

    // Transform point at far plane: (0, 0, -far, 1)
    const z_in = -far;
    const z_clip = m[2][2] * z_in + m[3][2];
    const w_clip = m[2][3] * z_in + m[3][3];
    try testing.expectApproxEqAbs(@as(f32, 0), z_clip / w_clip, eps);
}

test "perspective: W component is -z (perspective divide)" {
    // The W row should be [0, 0, -1, 0] in the mathematical matrix.
    // In our column-major storage: m[col][3] gives row 3 of the matrix.
    const m = perspective(std.math.pi / 3.0, 1.5, 1.0, 500.0);
    try testing.expectApproxEqAbs(@as(f32, 0), m[0][3], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), m[1][3], eps);
    try testing.expectApproxEqAbs(@as(f32, -1), m[2][3], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), m[3][3], eps);
}

// ---- lookAt ----

test "lookAt: eye at origin looking down -Z" {
    // Standard OpenGL/Vulkan default: camera at origin, looking toward -Z, up = +Y
    const eye: Vec3 = .{ 0, 0, 0 };
    const target: Vec3 = .{ 0, 0, -1 };
    const up: Vec3 = .{ 0, 1, 0 };
    const m = lookAt(eye, target, up);

    // f = normalize((0,0,-1) - (0,0,0)) = (0,0,-1)
    // s = normalize(cross(f, up)) = normalize(cross((0,0,-1),(0,1,0))) = normalize((1,0,0)) = (1,0,0)
    // u = cross(s, f) = cross((1,0,0),(0,0,-1)) = (0,1,0)
    //
    // Column 0 (m[0]): [s.x, u.x, -f.x, 0] = [1, 0, 0, 0]
    // Column 1 (m[1]): [s.y, u.y, -f.y, 0] = [0, 1, 0, 0]
    // Column 2 (m[2]): [s.z, u.z, -f.z, 0] = [0, 0, 1, 0]
    // Column 3 (m[3]): [-dot(s,eye), -dot(u,eye), dot(f,eye), 1] = [0, 0, 0, 1]
    try expectMat4Approx(identity, m);
}

test "lookAt: eye offset along Z axis" {
    const eye: Vec3 = .{ 0, 0, 5 };
    const target: Vec3 = .{ 0, 0, 0 };
    const up: Vec3 = .{ 0, 1, 0 };
    const m = lookAt(eye, target, up);

    // f = normalize((0,0,-5)) = (0,0,-1)
    // s = normalize(cross((0,0,-1),(0,1,0))) = (1,0,0)
    // u = cross((1,0,0),(0,0,-1)) = (0,1,0)
    //
    // Column 0: [1, 0, 0, 0]
    // Column 1: [0, 1, 0, 0]
    // Column 2: [0, 0, 1, 0]
    // Column 3: [-dot(s,eye), -dot(u,eye), dot(f,eye), 1] = [0, 0, -5, 1]
    const expected: Mat4 = .{
        .{ 1, 0, 0, 0 },
        .{ 0, 1, 0, 0 },
        .{ 0, 0, 1, 0 },
        .{ 0, 0, -5, 1 },
    };
    try expectMat4Approx(expected, m);
}

test "lookAt: looking along +X axis" {
    const eye: Vec3 = .{ 0, 0, 0 };
    const target: Vec3 = .{ 1, 0, 0 };
    const up: Vec3 = .{ 0, 1, 0 };
    const m = lookAt(eye, target, up);

    // f = (1,0,0)
    // s = normalize(cross((1,0,0),(0,1,0))) = (0,0,1)
    // u = cross((0,0,1),(1,0,0)) = (0,1,0)
    //
    // Column 0 (m[0]): [s.x, u.x, -f.x, 0] = [0, 0, -1, 0]
    // Column 1 (m[1]): [s.y, u.y, -f.y, 0] = [0, 1, 0, 0]
    // Column 2 (m[2]): [s.z, u.z, -f.z, 0] = [1, 0, 0, 0]
    // Column 3 (m[3]): [0, 0, 0, 1]
    const expected: Mat4 = .{
        .{ 0, 0, -1, 0 },
        .{ 0, 1, 0, 0 },
        .{ 1, 0, 0, 0 },
        .{ 0, 0, 0, 1 },
    };
    try expectMat4Approx(expected, m);
}

test "lookAt: view matrix is orthonormal (rotation part)" {
    const eye: Vec3 = .{ 3, 4, 5 };
    const target: Vec3 = .{ 0, 0, 0 };
    const up: Vec3 = .{ 0, 1, 0 };
    const m = lookAt(eye, target, up);

    // The 3x3 upper-left of the rotation (columns 0-2, rows 0-2) should be orthonormal.
    // In our column-major storage, column i = m[i], rows 0-2.
    // Column vectors should be unit length and mutually orthogonal.
    const col0: Vec3 = .{ m[0][0], m[0][1], m[0][2] };
    const col1: Vec3 = .{ m[1][0], m[1][1], m[1][2] };
    const col2: Vec3 = .{ m[2][0], m[2][1], m[2][2] };

    // Unit length
    try testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(dot(col0, col0)), eps);
    try testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(dot(col1, col1)), eps);
    try testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(dot(col2, col2)), eps);

    // Mutually orthogonal
    try testing.expectApproxEqAbs(@as(f32, 0), dot(col0, col1), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), dot(col0, col2), eps);
    try testing.expectApproxEqAbs(@as(f32, 0), dot(col1, col2), eps);
}

// ---- add ----

test "add: basic addition" {
    const result = add(.{ 1, 2, 3 }, .{ 4, 5, 6 });
    try expectVec3Approx(.{ 5, 7, 9 }, result);
}

test "add: adding zero vector" {
    const result = add(.{ 7, -2, 3 }, .{ 0, 0, 0 });
    try expectVec3Approx(.{ 7, -2, 3 }, result);
}

// ---- scale ----

test "scale: by 2" {
    const result = scale(.{ 1, -2, 3 }, 2.0);
    try expectVec3Approx(.{ 2, -4, 6 }, result);
}

test "scale: by zero" {
    const result = scale(.{ 7, -2, 3 }, 0);
    try expectVec3Approx(.{ 0, 0, 0 }, result);
}

test "scale: by negative" {
    const result = scale(.{ 1, 2, 3 }, -1);
    try expectVec3Approx(.{ -1, -2, -3 }, result);
}

// ---- quaternion ----

fn expectQuatApprox(expected: Quat, actual: Quat) !void {
    try testing.expectApproxEqAbs(expected[0], actual[0], eps);
    try testing.expectApproxEqAbs(expected[1], actual[1], eps);
    try testing.expectApproxEqAbs(expected[2], actual[2], eps);
    try testing.expectApproxEqAbs(expected[3], actual[3], eps);
}

test "quatFromAxisAngle: zero angle gives identity" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, 0);
    try expectQuatApprox(quat_identity, q);
}

test "quatFromAxisAngle: 90 deg around Y" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0);
    const s = @sin(std.math.pi / 4.0);
    const c_val = @cos(std.math.pi / 4.0);
    try expectQuatApprox(.{ 0, s, 0, c_val }, q);
}

test "quatMul: identity * q = q" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, 1.0);
    const result = quatMul(quat_identity, q);
    try expectQuatApprox(q, result);
}

test "quatMul: q * identity = q" {
    const q = quatFromAxisAngle(.{ 1, 0, 0 }, 0.5);
    const result = quatMul(q, quat_identity);
    try expectQuatApprox(q, result);
}

test "quatRotateVec3: 90 deg Y rotation maps (0,0,-1) to (-1,0,0)" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0);
    const result = quatRotateVec3(q, .{ 0, 0, -1 });
    try expectVec3Approx(.{ -1, 0, 0 }, result);
}

test "quatRotateVec3: 90 deg X rotation maps (0,1,0) to (0,0,1)" {
    const q = quatFromAxisAngle(.{ 1, 0, 0 }, std.math.pi / 2.0);
    const result = quatRotateVec3(q, .{ 0, 1, 0 });
    try expectVec3Approx(.{ 0, 0, 1 }, result);
}

test "quatRotateVec3: identity rotation leaves vector unchanged" {
    const v: Vec3 = .{ 3, -7, 2 };
    const result = quatRotateVec3(quat_identity, v);
    try expectVec3Approx(v, result);
}

test "quatRotateVec3: 360 deg rotation returns to original" {
    const q = quatFromAxisAngle(.{ 0, 0, 1 }, 2.0 * std.math.pi);
    const v: Vec3 = .{ 1, 2, 3 };
    const result = quatRotateVec3(q, v);
    try expectVec3Approx(v, result);
}

test "quatNormalize: non-unit becomes unit" {
    const q: Quat = .{ 2, 0, 0, 0 };
    const result = quatNormalize(q);
    const len = @sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2] + result[3] * result[3]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), len, eps);
    try expectQuatApprox(.{ 1, 0, 0, 0 }, result);
}

test "quatNlerp: t=0 returns a, t=1 returns b" {
    const a = quatFromAxisAngle(.{ 0, 1, 0 }, 0.0);
    const b = quatFromAxisAngle(.{ 0, 1, 0 }, 1.0);
    try expectQuatApprox(a, quatNlerp(a, b, 0.0));
    try expectQuatApprox(b, quatNlerp(a, b, 1.0));
}

test "quatNlerp: midpoint rotates a vector to roughly half-angle" {
    const a = quat_identity;
    const b = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0);
    const half = quatNlerp(a, b, 0.5);
    const v = quatRotateVec3(half, .{ 0, 0, -1 });
    // 45 deg yaw of (0,0,-1) -> (-sin45, 0, -cos45)
    try testing.expectApproxEqAbs(@as(f32, -std.math.sqrt1_2), v[0], 0.01);
    try testing.expectApproxEqAbs(@as(f32, -std.math.sqrt1_2), v[2], 0.01);
}

test "quatNlerp: picks shorter arc when dot is negative" {
    const a = quat_identity;
    // Negate every component: same rotation, opposite hemisphere.
    const b: Quat = .{ -a[0], -a[1], -a[2], -a[3] };
    const result = quatNlerp(a, b, 0.5);
    try expectQuatApprox(quat_identity, result);
}

test "quatFromDir: identical direction is identity" {
    const q = quatFromDir(.{ 0, 0, -1 }, .{ 0, 0, -1 }, .{ 0, 1, 0 });
    try expectQuatApprox(quat_identity, q);
}

test "quatFromDir: 180 deg uses fallback axis" {
    // antiparallel: from (0,0,-1) to (0,0,1), fallback axis Y
    const q = quatFromDir(.{ 0, 0, -1 }, .{ 0, 0, 1 }, .{ 0, 1, 0 });
    const rotated = quatRotateVec3(q, .{ 0, 0, -1 });
    try expectVec3Approx(.{ 0, 0, 1 }, rotated);
}

test "quatFromDir: 90 deg rotation maps front" {
    const q = quatFromDir(.{ 0, 0, -1 }, .{ 1, 0, 0 }, .{ 0, 1, 0 });
    const rotated = quatRotateVec3(q, .{ 0, 0, -1 });
    try expectVec3Approx(.{ 1, 0, 0 }, rotated);
}

test "quatToMat3D: identity quaternion yields identity matrix" {
    const m = quatToMat3D(quat_identity);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m[0][0], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m[1][1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m[2][2], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[0][1], 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[1][0], 1e-9);
}

test "quatToMat3D: 180 deg Y flips X and Z" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi);
    const m = quatToMat3D(q);
    try testing.expectApproxEqAbs(@as(f64, -1.0), m[0][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m[1][1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -1.0), m[2][2], 1e-6);
}

test "quatToMat3D: 90 deg X rotation" {
    // 90 deg X: column 1 (Y axis) -> +Z, column 2 (Z axis) -> -Y
    const q = quatFromAxisAngle(.{ 1, 0, 0 }, std.math.pi / 2.0);
    const m = quatToMat3D(q);
    // m[1] is the rotated Y basis (column 1 of column-major matrix)
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[1][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[1][1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 1.0), m[1][2], 1e-6);
    // m[2] is the rotated Z basis
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[2][0], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, -1.0), m[2][1], 1e-6);
    try testing.expectApproxEqAbs(@as(f64, 0.0), m[2][2], 1e-6);
}

// ---- yawPitchFromQuat ----

test "yawPitchFromQuat: identity gives yaw=0 pitch=0" {
    const yp = yawPitchFromQuat(quat_identity);
    try testing.expectApproxEqAbs(@as(f32, 0), yp[0], eps);
    try testing.expectApproxEqAbs(@as(f32, 0), yp[1], eps);
}

test "yawPitchFromQuat: 90 deg yaw right" {
    const q = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0);
    const yp = yawPitchFromQuat(q);
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), yp[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), yp[1], 1e-4);
}

test "yawPitchFromQuat: pitch up 30 deg" {
    const q = quatFromAxisAngle(.{ 1, 0, 0 }, std.math.pi / 6.0);
    const yp = yawPitchFromQuat(q);
    try testing.expectApproxEqAbs(@as(f32, 0), yp[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 6.0), yp[1], 1e-4);
}

test "yawPitchFromQuat: roll around front axis does not change result" {
    const q_yaw = quatFromAxisAngle(.{ 0, 1, 0 }, 0.5);
    const yawed_right = quatRotateVec3(q_yaw, Vec3{ 1, 0, 0 });
    const q_pitch = quatFromAxisAngle(yawed_right, 0.3);
    const yaw_pitch = quatMul(q_pitch, q_yaw);
    const front = quatRotateVec3(yaw_pitch, Vec3{ 0, 0, -1 });
    const q_roll = quatFromAxisAngle(front, 0.8);
    const full = quatMul(q_roll, yaw_pitch);
    const yp_no_roll = yawPitchFromQuat(yaw_pitch);
    const yp_with_roll = yawPitchFromQuat(full);
    try testing.expectApproxEqAbs(yp_no_roll[0], yp_with_roll[0], 1e-4);
    try testing.expectApproxEqAbs(yp_no_roll[1], yp_with_roll[1], 1e-4);
}

test "headingFromQuat: each cardinal direction reads itself (0=N, 90=E, 180=S, 270=W)" {
    // Front basis is -Z = North; build orientations facing each cardinal.
    const front: Vec3 = .{ 0, 0, -1 };
    const cases = .{
        .{ Vec3{ 0, 0, -1 }, @as(f32, 0) }, // North
        .{ Vec3{ 1, 0, 0 }, @as(f32, std.math.pi / 2.0) }, // East
        .{ Vec3{ 0, 0, 1 }, @as(f32, std.math.pi) }, // South
        .{ Vec3{ -1, 0, 0 }, @as(f32, 3.0 * std.math.pi / 2.0) }, // West
    };
    inline for (cases) |case| {
        const q = quatFromDir(front, case[0], world_up);
        try testing.expectApproxEqAbs(case[1], headingFromQuat(q), 1e-4);
    }
}

test "quatRotateVec3: composed yaw then pitch" {
    // Yaw 90 deg right around Y, then pitch 90 deg up around new right axis
    const q_yaw = quatFromAxisAngle(.{ 0, 1, 0 }, std.math.pi / 2.0);
    // After yaw, front was (0,0,-1) -> (-1,0,0), right was (1,0,0) -> (0,0,-1)... wait
    // Actually: right = quatRotateVec3(q_yaw, (1,0,0))
    const new_right = quatRotateVec3(q_yaw, .{ 1, 0, 0 });
    const q_pitch = quatFromAxisAngle(new_right, std.math.pi / 2.0);
    const combined = quatMul(q_pitch, q_yaw);

    // Start looking at (0,0,-1). After yaw 90 deg right: looking at (-1,0,0).
    // After pitch 90 deg up around new right axis: looking at (0,1,0).
    const result = quatRotateVec3(combined, .{ 0, 0, -1 });
    try expectVec3Approx(.{ 0, 1, 0 }, result);
}
