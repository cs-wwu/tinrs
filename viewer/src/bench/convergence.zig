//! Rolling-window convergence detector for frame times.
//!
//! Tracks a sliding window of frame times and uses median + interquartile range
//! (IQR) as the stability metric: "converged" means IQR/median has stayed below
//! a threshold for K consecutive full windows. Median+IQR is robust to outliers
//! AND to bimodal/trimodal distributions (e.g. compositor pacing producing
//! pairs of fast/slow frames at vsync boundaries), much more so than
//! mean+stddev, which inflate any time frames split across modes.
//! Used by the profile mode to auto-detect warmup completion and measurement
//! readiness.

const std = @import("std");

pub const State = enum { filling, waiting, converged, capped };

const Self = @This();

pub const Options = struct {
    window_size: usize = 120,
    spread_threshold: f64 = 0.05, // IQR / median ratio: relative spread of the middle 50% vs the typical frame
    max_spread_ns: u64 = 250_000, // 0.25ms; absolute IQR floor for fast GPUs where the ratio is noisy
    max_spread_pct: f64 = 0.0, // alternate absolute floor as a fraction of median; effective floor = max(max_spread_ns, median * pct)
    consecutive_required: u32 = 3,
    max_frames: u64 = 30_000,
    max_time_ns: u64 = 60_000_000_000, // 60s wall-clock cap
};

window: []u64,
head: usize = 0,
count: usize = 0,
total_frames: u64 = 0,
total_time_ns: u64 = 0,
max_frames: u64,
max_time_ns: u64,

spread_threshold: f64,
max_spread_ns: u64,
max_spread_pct: f64,
consecutive_required: u32,
consecutive_passed: u32 = 0,

state: State = .filling,

pub fn init(allocator: std.mem.Allocator, opts: Options) !Self {
    const buf = try allocator.alloc(u64, opts.window_size);
    @memset(buf, 0);
    return .{
        .window = buf,
        .max_frames = opts.max_frames,
        .max_time_ns = opts.max_time_ns,
        .spread_threshold = opts.spread_threshold,
        .max_spread_ns = opts.max_spread_ns,
        .max_spread_pct = opts.max_spread_pct,
        .consecutive_required = opts.consecutive_required,
    };
}

pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
    allocator.free(self.window);
}

pub fn push(self: *Self, dt_ns: u64) State {
    self.window[self.head] = dt_ns;
    self.head = (self.head + 1) % self.window.len;
    if (self.count < self.window.len) self.count += 1;
    self.total_frames += 1;
    self.total_time_ns += dt_ns;

    if (self.state == .converged or self.state == .capped) return self.state;

    if (self.total_frames >= self.max_frames or self.total_time_ns >= self.max_time_ns) {
        self.state = .capped;
        return .capped;
    }

    if (self.count < self.window.len) {
        self.state = .filling;
        return .filling;
    }

    // Window full: check stability at window boundaries
    if (self.head != 0) {
        self.state = .waiting;
        return .waiting;
    }

    // Sort in-place for percentile-based stats. Safe: push() only writes at `head`
    // and never reads existing values, so reordering the bag of samples is fine.
    std.mem.sort(u64, self.window[0..self.count], {}, std.sort.asc(u64));
    const data = self.window[0..self.count];
    const median = percentileFloor(data, 0.50);
    const iqr = percentileFloor(data, 0.75) -| percentileFloor(data, 0.25);
    const median_f: f64 = @floatFromInt(median);
    const iqr_f: f64 = @floatFromInt(iqr);
    const spread = if (median == 0) 0.0 else iqr_f / median_f;
    const scaled_thresh: u64 = @intFromFloat(median_f * self.max_spread_pct);
    const effective_thresh = @max(self.max_spread_ns, scaled_thresh);
    if (spread < self.spread_threshold or iqr < effective_thresh) {
        self.consecutive_passed += 1;
        if (self.consecutive_passed >= self.consecutive_required) {
            self.state = .converged;
            return .converged;
        }
    } else {
        self.consecutive_passed = 0;
    }

    self.state = .waiting;
    return .waiting;
}

pub fn reset(self: *Self) void {
    self.head = 0;
    self.count = 0;
    self.total_frames = 0;
    self.total_time_ns = 0;
    self.consecutive_passed = 0;
    self.state = .filling;
}

pub const Stats = struct {
    median_ns: f64,
    iqr_ns: f64,
    spread: f64,
    frames: u64,
};

/// Diagnostic-only: sorts a copy on the stack so the caller's view of `self`
/// stays unmodified. Window size is bounded by stats_buf_max.
fn stats(self: *const Self) Stats {
    if (self.count == 0) return .{ .median_ns = 0, .iqr_ns = 0, .spread = 0, .frames = 0 };
    var buf: [stats_buf_max]u64 = undefined;
    std.debug.assert(self.count <= buf.len);
    @memcpy(buf[0..self.count], self.window[0..self.count]);
    std.mem.sort(u64, buf[0..self.count], {}, std.sort.asc(u64));
    const data = buf[0..self.count];
    const median = percentileFloor(data, 0.50);
    const iqr = percentileFloor(data, 0.75) -| percentileFloor(data, 0.25);
    const median_f: f64 = @floatFromInt(median);
    const iqr_f: f64 = @floatFromInt(iqr);
    return .{
        .median_ns = median_f,
        .iqr_ns = iqr_f,
        .spread = if (median == 0) 0.0 else iqr_f / median_f,
        .frames = self.total_frames,
    };
}

const stats_buf_max: usize = 256;

/// Nearest-rank percentile on a sorted slice. p in [0, 1].
/// Index = floor(p * (len - 1)). Discrete (no interpolation) so multimodal
/// distributions don't get a synthetic value between modes; distinct from
/// `Bench.percentile` which uses ceil-rounding for tail-percentile reporting.
fn percentileFloor(sorted: []const u64, p: f64) u64 {
    if (sorted.len == 0) return 0;
    const n: f64 = @floatFromInt(sorted.len);
    const idx_f = p * (n - 1);
    const idx: usize = @intFromFloat(@floor(idx_f));
    return sorted[@min(idx, sorted.len - 1)];
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "stable input converges" {
    var det = try Self.init(testing.allocator, .{ .window_size = 10, .consecutive_required = 2 });
    defer det.deinit(testing.allocator);
    // 30 frames of identical values: IQR = 0 every window, passes immediately
    for (0..30) |_| _ = det.push(1_000_000);
    try testing.expectEqual(State.converged, det.state);
}

test "noisy input does not converge" {
    var det = try Self.init(testing.allocator, .{
        .window_size = 10,
        .spread_threshold = 0.01,
        .consecutive_required = 2,
        .max_frames = 100,
    });
    defer det.deinit(testing.allocator);
    var prng = std.Random.DefaultPrng.init(42);
    var state: State = .filling;
    for (0..100) |_| {
        // Wide uniform [500us, 2ms]: IQR/median ~ 0.6, way above 1%
        state = det.push(prng.random().intRangeAtMost(u64, 500_000, 2_000_000));
    }
    try testing.expect(state != .converged);
}

test "spike then stability converges" {
    var det = try Self.init(testing.allocator, .{ .window_size = 10, .consecutive_required = 2 });
    defer det.deinit(testing.allocator);
    // Spiky warmup: alternating 1ms / 5ms, IQR = 4ms
    for (0..10) |i| _ = det.push(if (i % 2 == 0) 1_000_000 else 5_000_000);
    try testing.expect(det.state != .converged);
    // Settle: 30 stable frames overwrite the window
    for (0..30) |_| _ = det.push(1_000_000);
    try testing.expectEqual(State.converged, det.state);
}

test "reset clears state" {
    var det = try Self.init(testing.allocator, .{ .window_size = 10, .consecutive_required = 2 });
    defer det.deinit(testing.allocator);
    for (0..30) |_| _ = det.push(1_000_000);
    try testing.expectEqual(State.converged, det.state);
    det.reset();
    try testing.expectEqual(State.filling, det.state);
    try testing.expectEqual(@as(u64, 0), det.total_frames);
}

test "max_frames cap triggers" {
    var det = try Self.init(testing.allocator, .{
        .window_size = 10,
        .spread_threshold = 0.001, // very tight; won't converge
        .max_spread_ns = 1, // very tight absolute floor too
        .max_frames = 50,
    });
    defer det.deinit(testing.allocator);
    var prng = std.Random.DefaultPrng.init(99);
    var state: State = .filling;
    for (0..50) |_| {
        state = det.push(1_000_000 + prng.random().intRangeAtMost(u64, 0, 100_000));
    }
    try testing.expectEqual(State.capped, state);
}

test "stats returns valid values" {
    var det = try Self.init(testing.allocator, .{ .window_size = 5 });
    defer det.deinit(testing.allocator);
    for (0..5) |_| _ = det.push(2_000_000);
    const s = det.stats();
    try testing.expectApproxEqAbs(@as(f64, 2_000_000), s.median_ns, 1);
    try testing.expectApproxEqAbs(@as(f64, 0), s.iqr_ns, 1);
    try testing.expectApproxEqAbs(@as(f64, 0), s.spread, 1e-9);
    try testing.expectEqual(@as(u64, 5), s.frames);
}

test "max_spread_pct scales threshold with frame time" {
    var det = try Self.init(testing.allocator, .{
        .window_size = 10,
        .spread_threshold = 0.001,
        .max_spread_ns = 1,
        .max_spread_pct = 0.10,
        .consecutive_required = 2,
        .max_frames = 200,
    });
    defer det.deinit(testing.allocator);
    // ~6ms frames with ~600us uniform jitter; relative spread fails the 0.1%
    // threshold and absolute IQR fails 1ns, but IQR is well under 10% of median.
    var prng = std.Random.DefaultPrng.init(77);
    var state: State = .filling;
    for (0..200) |_| {
        const jitter = prng.random().intRangeAtMost(u64, 0, 600_000);
        state = det.push(5_700_000 + jitter);
        if (state == .converged) break;
    }
    try testing.expectEqual(State.converged, state);
}

test "outlier spikes do not prevent convergence" {
    // IQR is naturally robust; single big outlier per window stays outside the
    // P25..P75 range and doesn't leak into the spread.
    var det = try Self.init(testing.allocator, .{
        .window_size = 20,
        .spread_threshold = 0.03,
        .max_spread_ns = 100_000,
        .consecutive_required = 2,
        .max_frames = 500,
    });
    defer det.deinit(testing.allocator);
    var state: State = .filling;
    for (0..500) |i| {
        // 5% outliers (50ms) in a stream of 5ms frames
        const val: u64 = if (i % 20 == 0) 50_000_000 else 5_000_000;
        state = det.push(val);
        if (state == .converged) break;
    }
    try testing.expectEqual(State.converged, state);
}

test "compositor-pacing trimodal converges (1440p / Mutter case)" {
    // Reproduces the pattern observed at 1440p with Mutter pacing:
    //   ~22.5% fast (frame stolen by next), ~55% middle, ~22.5% slow (paid back).
    // Mean+stddev would see CV ~0.65 here and never converge; IQR/median lands
    // at 0 because the middle mode (>50%) covers both P25 and P75.
    //
    // Pattern is deterministic 9/22/9 over a 40-frame cycle so each 120-frame
    // window has exactly 27/66/27, cleanly inside the "middle dominates" regime.
    var det = try Self.init(testing.allocator, .{
        .window_size = 120,
        .spread_threshold = 0.05,
        .max_spread_ns = 250_000,
        .consecutive_required = 3,
        .max_frames = 1_000,
    });
    defer det.deinit(testing.allocator);
    var state: State = .filling;
    for (0..1_000) |i| {
        const phase = i % 40;
        const val: u64 = if (phase < 9) 130_000 else if (phase < 31) 4_140_000 else 8_270_000;
        state = det.push(val);
        if (state == .converged) break;
    }
    try testing.expectEqual(State.converged, state);
}

test "evenly bimodal does not converge" {
    // 50/50 split: neither mode dominates, so P25 lands in one mode and P75
    // in the other. IQR is huge, correctly NOT a steady state.
    var det = try Self.init(testing.allocator, .{
        .window_size = 20,
        .spread_threshold = 0.05,
        .max_spread_ns = 100_000,
        .consecutive_required = 2,
        .max_frames = 200,
    });
    defer det.deinit(testing.allocator);
    var state: State = .filling;
    for (0..200) |i| {
        const val: u64 = if (i % 2 == 0) 1_000_000 else 5_000_000;
        state = det.push(val);
    }
    try testing.expectEqual(State.capped, state);
}

test "wall-clock time cap triggers" {
    var det = try Self.init(testing.allocator, .{
        .window_size = 10,
        .spread_threshold = 0.001,
        .max_spread_ns = 1,
        .max_frames = 100_000,
        .max_time_ns = 500_000_000, // 0.5s
    });
    defer det.deinit(testing.allocator);
    var state: State = .filling;
    // 30 frames at 20ms each = 600ms, exceeds 500ms cap
    for (0..30) |_| {
        state = det.push(20_000_000);
        if (state == .capped) break;
    }
    try testing.expectEqual(State.capped, state);
    try testing.expect(det.total_time_ns >= 500_000_000);
}
