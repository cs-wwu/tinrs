//! Full-run benchmarking: frame-time percentiles + per-phase running stats.
//!
//! Companion to frame_stats.zig (HUD ring buffer). Bench stores every frame
//! time for end-of-run percentile reports and accumulates running stats for
//! per-phase timings (CPU acquire/wait/present, GPU compute, GPU graphics).
//!
//! Frames are split into a warmup window (first N) and steady-state (rest).
//! Compute is further split between scrolling (compute dispatched) and
//! stationary (no dispatch) frames in steady-state, useful because compute
//! only runs on camera scroll.

const std = @import("std");

const DEFAULT_CAPACITY: usize = 1 << 20; // ~1M frames, ~8 MB; floor for open-ended runs
const clock: std.Io.Clock = .awake;

/// Per-frame phase samples gathered from the renderer.
pub const PhaseSample = struct {
    cpu_wait_ns: u64 = 0, // waitForFences (blocking on prior frame's GPU work)
    cpu_acquire_ns: u64 = 0, // acquireNextImageKHR
    cpu_submit_ns: u64 = 0, // queueSubmit
    cpu_present_ns: u64 = 0, // queuePresentKHR
    cpu_record_ns: u64 = 0, // gap between beginFrame return and endFrame call (filled by main)
    gpu_compute_ns: u64 = 0, // 0 if timestamps unsupported or no dispatch
    gpu_graphics_ns: u64 = 0,
    scrolling: bool = false, // compute dispatched any work this frame
    fence_pre_signaled: bool = false, // fence was already signaled when we got to waitForFences
};

/// Welford-style running aggregator: O(1) update, no per-sample storage.
pub const PhaseAccum = struct {
    count: u64 = 0,
    min_ns: u64 = std.math.maxInt(u64),
    max_ns: u64 = 0,
    mean_ns: f64 = 0,
    m2: f64 = 0, // sum of squared deviations from mean

    pub fn add(self: *PhaseAccum, ns: u64) void {
        self.count += 1;
        if (ns < self.min_ns) self.min_ns = ns;
        if (ns > self.max_ns) self.max_ns = ns;
        const x: f64 = @floatFromInt(ns);
        const delta = x - self.mean_ns;
        self.mean_ns += delta / @as(f64, @floatFromInt(self.count));
        const delta2 = x - self.mean_ns;
        self.m2 += delta * delta2;
    }

    pub fn meanMs(self: PhaseAccum) f64 {
        return self.mean_ns / 1e6;
    }
    pub fn minMs(self: PhaseAccum) f64 {
        return if (self.count == 0) 0 else nsToMs(self.min_ns);
    }
    pub fn maxMs(self: PhaseAccum) f64 {
        return nsToMs(self.max_ns);
    }
    pub fn stddevMs(self: PhaseAccum) f64 {
        if (self.count < 2) return 0;
        const variance = self.m2 / @as(f64, @floatFromInt(self.count));
        return @sqrt(variance) / 1e6;
    }
};

pub const PhaseSet = struct {
    cpu_wait: PhaseAccum = .{},
    cpu_acquire: PhaseAccum = .{},
    cpu_submit: PhaseAccum = .{},
    cpu_present: PhaseAccum = .{},
    cpu_record: PhaseAccum = .{},
    gpu_compute: PhaseAccum = .{},
    gpu_graphics: PhaseAccum = .{},
    fence_pre_signaled_count: u64 = 0,
    fence_total_count: u64 = 0,

    pub fn accumulate(self: *PhaseSet, sample: PhaseSample) void {
        self.cpu_wait.add(sample.cpu_wait_ns);
        self.cpu_acquire.add(sample.cpu_acquire_ns);
        self.cpu_submit.add(sample.cpu_submit_ns);
        self.cpu_present.add(sample.cpu_present_ns);
        self.cpu_record.add(sample.cpu_record_ns);
        self.fence_total_count += 1;
        if (sample.fence_pre_signaled) self.fence_pre_signaled_count += 1;
        if (sample.gpu_compute_ns > 0 or sample.gpu_graphics_ns > 0) {
            self.gpu_compute.add(sample.gpu_compute_ns);
            self.gpu_graphics.add(sample.gpu_graphics_ns);
        }
    }
};

frame_times_ns: std.ArrayList(u64),
prev_frame_ts: std.Io.Timestamp,
skipped_frames: u64 = 0,
dropped_frames: u64 = 0,
warmup_frames: u64,
warmup: PhaseSet = .{},
steady: PhaseSet = .{},
steady_compute_scrolling: PhaseAccum = .{},
steady_compute_stationary: PhaseAccum = .{},
allocator: std.mem.Allocator,
io: std.Io,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, io: std.Io, expected_frames: u64, warmup_frames: u64) !Self {
    const cap = if (expected_frames == 0) DEFAULT_CAPACITY else expected_frames;
    var list: std.ArrayList(u64) = .empty;
    try list.ensureTotalCapacityPrecise(allocator, cap);
    const now = clock.now(io);
    return .{
        .frame_times_ns = list,
        .prev_frame_ts = now,
        .warmup_frames = warmup_frames,
        .allocator = allocator,
        .io = io,
    };
}

pub fn deinit(self: *Self) void {
    self.frame_times_ns.deinit(self.allocator);
}

pub fn recordFrame(self: *Self, sample: PhaseSample) void {
    const now = clock.now(self.io);
    const dt = deltaNs(self.prev_frame_ts, now);
    self.prev_frame_ts = now;

    // Cap at preallocated capacity: growth during the hot path would stall the
    // measurement itself, polluting the tail of the distribution.
    const idx = self.frame_times_ns.items.len;
    if (idx < self.frame_times_ns.capacity) {
        self.frame_times_ns.appendAssumeCapacity(dt);
    } else {
        self.dropped_frames += 1;
    }

    const set = if (idx < self.warmup_frames) &self.warmup else &self.steady;
    set.accumulate(sample);
    if (idx >= self.warmup_frames) {
        const compute_set = if (sample.scrolling) &self.steady_compute_scrolling else &self.steady_compute_stationary;
        if (sample.gpu_compute_ns > 0 or sample.scrolling) compute_set.add(sample.gpu_compute_ns);
    }
}

/// Swapchain-recreate path: advance the reference timestamp so the next
/// recorded frame measures only the next frame's duration.
pub fn recordSkip(self: *Self) void {
    self.prev_frame_ts = clock.now(self.io);
    self.skipped_frames += 1;
}

pub const Stats = struct {
    frames: u64,
    runtime_s: f64,
    min_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    p99_ms: f64,
    p999_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    stddev_ms: f64,
};

pub const FullStats = struct {
    warmup: ?Stats,
    steady: ?Stats,
    skipped: u64,
    dropped: u64,
};

fn computeStats(self: *const Self) !FullStats {
    const total = self.frame_times_ns.items.len;
    const warmup_n = @min(self.warmup_frames, total);
    const steady_n = total - warmup_n;

    var warmup_stats: ?Stats = null;
    var steady_stats: ?Stats = null;

    if (warmup_n > 0) {
        const sorted = try self.allocator.alloc(u64, warmup_n);
        defer self.allocator.free(sorted);
        @memcpy(sorted, self.frame_times_ns.items[0..warmup_n]);
        std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
        warmup_stats = statsFromSorted(sorted, sumNs(sorted));
    }

    if (steady_n > 0) {
        const sorted = try self.allocator.alloc(u64, steady_n);
        defer self.allocator.free(sorted);
        @memcpy(sorted, self.frame_times_ns.items[warmup_n..]);
        std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
        steady_stats = statsFromSorted(sorted, sumNs(sorted));
    }

    return .{
        .warmup = warmup_stats,
        .steady = steady_stats,
        .skipped = self.skipped_frames,
        .dropped = self.dropped_frames,
    };
}

pub fn report(self: *const Self) void {
    const full = self.computeStats() catch |err| {
        std.log.err("bench: failed to compute stats: {}", .{err});
        return;
    };

    if (full.warmup == null and full.steady == null) {
        std.log.info("=== Benchmark ===  (no frames recorded)", .{});
        return;
    }

    if (full.warmup) |s| {
        std.log.info("=== Benchmark warmup (frames 0..{d}) ===", .{s.frames});
        printFrameLines(s);
        printPhaseLines(self.warmup);
    }

    if (full.steady) |s| {
        const start = if (full.warmup) |w| w.frames else 0;
        std.log.info("=== Benchmark steady (frames {d}..{d}) ===", .{ start, start + s.frames });
        printFrameLines(s);
        printPhaseLines(self.steady);
        printSteadyComputeSplit(self);
    }

    if (full.skipped > 0) std.log.info("Skipped (swapchain recreate): {d}", .{full.skipped});
    if (full.dropped > 0) std.log.info("Dropped (capacity exceeded): {d}", .{full.dropped});
}

pub fn printFrameLines(s: Stats) void {
    const fps = if (s.runtime_s > 0) @as(f64, @floatFromInt(s.frames)) / s.runtime_s else 0;
    std.log.info("Frames: {d}  Runtime: {d:.2}s  Throughput: {d:.0} fps", .{ s.frames, s.runtime_s, fps });
    std.log.info("Frame time (ms):  min {d:.4}  P50 {d:.4}  P90 {d:.4}  P99 {d:.4}  P99.9 {d:.4}  max {d:.4}", .{
        s.min_ms, s.p50_ms, s.p90_ms, s.p99_ms, s.p999_ms, s.max_ms,
    });
    std.log.info("                  mean {d:.4}  stddev {d:.4}", .{ s.mean_ms, s.stddev_ms });
}

pub fn printPhaseLines(set: PhaseSet) void {
    printPhaseLine("CPU wait    ", set.cpu_wait);
    printPhaseLine("CPU acquire ", set.cpu_acquire);
    printPhaseLine("CPU submit  ", set.cpu_submit);
    printPhaseLine("CPU present ", set.cpu_present);
    printPhaseLine("CPU record  ", set.cpu_record);
    if (set.gpu_compute.count > 0) printPhaseLine("GPU compute ", set.gpu_compute);
    if (set.gpu_graphics.count > 0) printPhaseLine("GPU graphics", set.gpu_graphics);
    if (set.fence_total_count > 0) {
        const pct = 100.0 * @as(f64, @floatFromInt(set.fence_pre_signaled_count)) / @as(f64, @floatFromInt(set.fence_total_count));
        std.log.info("  Fence pre-signaled: {d}/{d} ({d:.1}%)", .{
            set.fence_pre_signaled_count, set.fence_total_count, pct,
        });
    }
}

pub fn printPhaseLine(label: []const u8, p: PhaseAccum) void {
    if (p.count == 0) return;
    std.log.info("  {s} (n={d:>6}):  mean {d:.4}  min {d:.4}  max {d:.4}  stddev {d:.4} ms", .{
        label, p.count, p.meanMs(), p.minMs(), p.maxMs(), p.stddevMs(),
    });
}

fn printSteadyComputeSplit(self: *const Self) void {
    const scrolling = self.steady_compute_scrolling;
    const stationary = self.steady_compute_stationary;
    if (scrolling.count == 0 and stationary.count == 0) return;
    if (scrolling.count > 0) printPhaseLine("GPU compute (scrolling) ", scrolling);
    if (stationary.count > 0) printPhaseLine("GPU compute (stationary)", stationary);
}

pub fn statsFromSorted(sorted: []const u64, sum: u128) Stats {
    const n = sorted.len;
    const mean_ns: f64 = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(n));
    const runtime_s = @as(f64, @floatFromInt(sum)) / 1e9;

    var sq_sum: f64 = 0;
    for (sorted) |t| {
        const d = @as(f64, @floatFromInt(t)) - mean_ns;
        sq_sum += d * d;
    }
    const var_ns = sq_sum / @as(f64, @floatFromInt(n));

    return .{
        .frames = n,
        .runtime_s = runtime_s,
        .min_ms = nsToMs(sorted[0]),
        .p50_ms = nsToMs(percentile(sorted, 50.0)),
        .p90_ms = nsToMs(percentile(sorted, 90.0)),
        .p99_ms = nsToMs(percentile(sorted, 99.0)),
        .p999_ms = nsToMs(percentile(sorted, 99.9)),
        .max_ms = nsToMs(sorted[n - 1]),
        .mean_ms = mean_ns / 1e6,
        .stddev_ms = @sqrt(var_ns) / 1e6,
    };
}

pub fn sumNs(slice: []const u64) u128 {
    var s: u128 = 0;
    for (slice) |t| s += t;
    return s;
}

pub fn deltaNs(from: std.Io.Timestamp, to: std.Io.Timestamp) u64 {
    const ns = from.durationTo(to).toNanoseconds();
    return if (ns < 0) 0 else @intCast(ns);
}

pub fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Nearest-rank percentile over an already-sorted slice.
pub fn percentile(sorted: []const u64, pct: f64) u64 {
    std.debug.assert(sorted.len > 0);
    const n = sorted.len;
    const raw = pct / 100.0 * @as(f64, @floatFromInt(n));
    var rank: usize = @intFromFloat(@ceil(raw));
    if (rank < 1) rank = 1;
    if (rank > n) rank = n;
    return sorted[rank - 1];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "percentile: monotone on uniform distribution" {
    var buf: [100]u64 = undefined;
    for (0..100) |i| buf[i] = @as(u64, i + 1) * 1_000_000; // 1..100 ms, already sorted
    const s = statsFromSorted(&buf, sumNs(&buf));
    try testing.expect(s.min_ms <= s.p50_ms);
    try testing.expect(s.p50_ms <= s.p90_ms);
    try testing.expect(s.p90_ms <= s.p99_ms);
    try testing.expect(s.p99_ms <= s.p999_ms);
    try testing.expect(s.p999_ms <= s.max_ms);
}

test "percentile: spike caught by P99" {
    var buf: [1000]u64 = undefined;
    // 989 fast frames + 11 spikes; >1% spikes, so P99 must catch them.
    for (0..989) |i| buf[i] = 1_000_000;
    for (989..1000) |i| buf[i] = 100_000_000;
    const s = statsFromSorted(&buf, sumNs(&buf));
    try testing.expectApproxEqAbs(@as(f64, 1.0), s.p50_ms, 0.01);
    try testing.expect(s.p99_ms > 50.0);
    try testing.expectApproxEqAbs(@as(f64, 100.0), s.max_ms, 0.01);
}

test "deltaNs: clamps negative to zero" {
    const a: std.Io.Timestamp = .{ .nanoseconds = 100 };
    const b: std.Io.Timestamp = .{ .nanoseconds = 50 };
    try testing.expectEqual(@as(u64, 0), deltaNs(a, b));
    try testing.expectEqual(@as(u64, 50), deltaNs(b, a));
}

test "PhaseAccum: matches direct mean/stddev" {
    var p: PhaseAccum = .{};
    const vals = [_]u64{ 1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000 };
    for (vals) |v| p.add(v);

    try testing.expectEqual(@as(u64, 5), p.count);
    try testing.expectEqual(@as(u64, 1_000_000), p.min_ns);
    try testing.expectEqual(@as(u64, 5_000_000), p.max_ns);
    try testing.expectApproxEqAbs(@as(f64, 3.0), p.meanMs(), 1e-9);
    // Population stddev of {1,2,3,4,5} = sqrt(2) ms ~= 1.4142
    try testing.expectApproxEqAbs(@sqrt(@as(f64, 2.0)), p.stddevMs(), 1e-6);
}
