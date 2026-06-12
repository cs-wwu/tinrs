//! Frame time ring buffer with percentile statistics.
//!
//! Tracks the last CAPACITY frame times and periodically computes
//! average FPS, P99, and P0.1 (99.9th percentile) frame times.
//! Display values are cached and updated at a fixed interval
//! so the overlay is readable (not flickering every frame).

const std = @import("std");

const CAPACITY = 600; // ~10s at 60fps
const UPDATE_INTERVAL: f64 = 0.5; // seconds between display updates
const MIN_SAMPLES = 10; // don't compute stats with fewer samples

times: [CAPACITY]f64 = [_]f64{0} ** CAPACITY,
head: usize = 0,
count: usize = 0,
time_since_update: f64 = 0,

// Cached display values (updated at UPDATE_INTERVAL)
avg_fps: f64 = 0,
p99_ms: f64 = 0,
p01_ms: f64 = 0, // worst 0.1%

const Self = @This();

pub fn push(self: *Self, dt: f64) void {
    self.times[self.head] = dt;
    self.head = (self.head + 1) % CAPACITY;
    if (self.count < CAPACITY) self.count += 1;
    self.time_since_update += dt;

    if (self.time_since_update >= UPDATE_INTERVAL and self.count >= MIN_SAMPLES) {
        self.time_since_update = 0;
        self.compute();
    }
}

fn compute(self: *Self) void {
    // Copy and sort the valid portion
    var sorted: [CAPACITY]f64 = undefined;
    @memcpy(sorted[0..self.count], self.times[0..self.count]);
    std.mem.sort(f64, sorted[0..self.count], {}, std.sort.asc(f64));

    // Average
    var sum: f64 = 0;
    for (sorted[0..self.count]) |t| sum += t;
    const avg_dt = sum / @as(f64, @floatFromInt(self.count));
    self.avg_fps = if (avg_dt > 0) 1.0 / avg_dt else 0;

    // P99 = 99th percentile frame time (1% worst)
    const p99_idx = @min(self.count - 1, self.count * 99 / 100);
    self.p99_ms = sorted[p99_idx] * 1000.0;

    // P0.1 = 99.9th percentile (0.1% worst)
    const p01_idx = @min(self.count - 1, self.count * 999 / 1000);
    self.p01_ms = sorted[p01_idx] * 1000.0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;
const eps: f64 = 0.01;

test "FrameStats: initial values are zero" {
    const fs = Self{};
    try testing.expectApproxEqAbs(@as(f64, 0), fs.avg_fps, eps);
    try testing.expectApproxEqAbs(@as(f64, 0), fs.p99_ms, eps);
    try testing.expectApproxEqAbs(@as(f64, 0), fs.p01_ms, eps);
}

test "FrameStats: stats computed after interval" {
    var fs = Self{};
    // Push 60 frames of 16.67ms (60fps); total 1.0s > UPDATE_INTERVAL
    for (0..60) |_| fs.push(1.0 / 60.0);
    try testing.expect(fs.avg_fps > 59.0 and fs.avg_fps < 61.0);
    try testing.expect(fs.p99_ms > 16.0 and fs.p99_ms < 17.5);
}

test "FrameStats: not computed before interval" {
    var fs = Self{};
    // Push 5 frames of 16.67ms; total 83ms < UPDATE_INTERVAL
    for (0..5) |_| fs.push(1.0 / 60.0);
    try testing.expectApproxEqAbs(@as(f64, 0), fs.avg_fps, eps);
}

test "FrameStats: spike shows in P99" {
    var fs = Self{};
    // 197 good frames + 3 spikes (>1% are spikes, so P99 catches them)
    for (0..197) |_| fs.push(0.001); // 1ms
    for (0..3) |_| fs.push(0.050); // 50ms spikes
    // Force compute
    fs.time_since_update = UPDATE_INTERVAL;
    fs.push(0.001);
    // P99 should capture the spike region
    try testing.expect(fs.p99_ms > 10.0);
}

test "FrameStats: ring buffer wraps correctly" {
    var fs = Self{};
    // Fill past capacity with 10ms frames
    for (0..CAPACITY + 100) |_| {
        fs.push(0.01);
    }
    try testing.expectEqual(CAPACITY, fs.count);
    try testing.expect(fs.avg_fps > 99.0 and fs.avg_fps < 101.0);
}
