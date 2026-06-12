//! Self-calibrating GPU profiler.
//!
//! Runs a multi-phase benchmark that auto-detects warmup convergence,
//! measures static rendering, then measures rendering under camera movement.
//! No manual frame counts needed; convergence detection handles phase transitions.
//!
//! Phases: cold_start -> warmup -> static_measure -> moving_settle -> moving_measure -> done
//!
//! Tile streaming runs ahead of profile.tick(); the caller is expected to
//! synchronously drain pending tile loads before constructing Profile and
//! pass the elapsed time via `Options.tile_load_ns` (see autotune.zig and
//! main.zig for the pattern). Without that, async streaming completions
//! invalidate the clipmap mid-measurement and pollute the static phase.
//! The cold_start phase is now just a single-frame transition; actual
//! steady-state stabilization (GPU clocks, caches) happens in warmup.
//!
//! Usage: `--profile` flag (implies --benchmark for GPU timestamp collection).

const std = @import("std");
const Bench = @import("runner.zig");
const Convergence = @import("convergence.zig");

pub const Phase = enum {
    cold_start,
    warmup,
    static_measure,
    moving_settle,
    moving_measure,
    done,
};

const Measurement = struct {
    frame_times: std.ArrayList(u64),
    phases: Bench.PhaseSet = .{},
    scrolling_compute: Bench.PhaseAccum = .{},
    stationary_compute: Bench.PhaseAccum = .{},

    fn init(allocator: std.mem.Allocator, capacity: u64) !Measurement {
        var list: std.ArrayList(u64) = .empty;
        try list.ensureTotalCapacityPrecise(allocator, capacity);
        return .{ .frame_times = list };
    }

    fn deinit(self: *Measurement, allocator: std.mem.Allocator) void {
        self.frame_times.deinit(allocator);
    }

    fn record(self: *Measurement, dt_ns: u64, sample: Bench.PhaseSample) void {
        if (self.frame_times.items.len < self.frame_times.capacity) {
            self.frame_times.appendAssumeCapacity(dt_ns);
        }
        self.phases.accumulate(sample);
        const compute_set = if (sample.scrolling) &self.scrolling_compute else &self.stationary_compute;
        if (sample.gpu_compute_ns > 0 or sample.scrolling) compute_set.add(sample.gpu_compute_ns);
    }

    fn report(self: *const Measurement, allocator: std.mem.Allocator, label: []const u8, show_compute_split: bool) void {
        const n = self.frame_times.items.len;
        if (n == 0) {
            std.log.info("=== Profile: {s} ===  (no frames)", .{label});
            return;
        }

        const sorted = allocator.alloc(u64, n) catch {
            std.log.err("profile: alloc failed for {s} stats", .{label});
            return;
        };
        defer allocator.free(sorted);
        @memcpy(sorted, self.frame_times.items);
        std.mem.sort(u64, sorted, {}, std.sort.asc(u64));

        const s = Bench.statsFromSorted(sorted, Bench.sumNs(sorted));
        std.log.info("=== Profile: {s} ===", .{label});
        Bench.printFrameLines(s);
        Bench.printPhaseLines(self.phases);
        if (show_compute_split) {
            if (self.scrolling_compute.count > 0)
                Bench.printPhaseLine("GPU compute (scrolling) ", self.scrolling_compute);
            if (self.stationary_compute.count > 0)
                Bench.printPhaseLine("GPU compute (stationary)", self.stationary_compute);
        }
    }
};

pub const Options = struct {
    measure_duration_ns: u64 = 3_000_000_000, // 3 seconds per measurement phase
    convergence_opts: Convergence.Options = .{},
    /// Wall-clock time spent synchronously draining the tile streamer before
    /// profiling started. Reported under "Cold Start"; 0 in procedural mode.
    tile_load_ns: u64 = 0,
    tile_count: u32 = 0,
};

phase: Phase = .cold_start,
allocator: std.mem.Allocator,
convergence: Convergence,
measure_duration_ns: u64,
measure_elapsed_ns: u64 = 0,

init_time_ns: u64,
tile_load_ns: u64,
tile_count: u32,
first_frame_ns: u64 = 0,

warmup_frames: u64 = 0,
warmup_time_ns: u64 = 0,
settle_frames: u64 = 0,
settle_time_ns: u64 = 0,

static_data: Measurement,
moving_data: Measurement,

const Self = @This();

pub fn init(allocator: std.mem.Allocator, init_time_ns: u64, opts: Options) !Self {
    const capacity = @min(60_000, @max(256, opts.measure_duration_ns / 100_000));
    return .{
        .allocator = allocator,
        .convergence = try Convergence.init(allocator, opts.convergence_opts),
        .measure_duration_ns = opts.measure_duration_ns,
        .init_time_ns = init_time_ns,
        .tile_load_ns = opts.tile_load_ns,
        .tile_count = opts.tile_count,
        .static_data = try Measurement.init(allocator, capacity),
        .moving_data = try Measurement.init(allocator, capacity),
    };
}

pub fn deinit(self: *Self) void {
    self.convergence.deinit(self.allocator);
    self.static_data.deinit(self.allocator);
    self.moving_data.deinit(self.allocator);
}

pub fn shouldFly(self: *const Self) bool {
    return self.phase == .moving_settle or self.phase == .moving_measure;
}

pub const Results = struct {
    static_p99_ms: f64,
    moving_p99_ms: f64,
};

pub fn results(self: *const Self) Results {
    return .{
        .static_p99_ms = measurementP99(&self.static_data, self.allocator),
        .moving_p99_ms = measurementP99(&self.moving_data, self.allocator),
    };
}

fn measurementP99(m: *const Measurement, allocator: std.mem.Allocator) f64 {
    const n = m.frame_times.items.len;
    if (n == 0) return 0;
    const sorted = allocator.alloc(u64, n) catch return 0;
    defer allocator.free(sorted);
    @memcpy(sorted, m.frame_times.items);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    return Bench.nsToMs(Bench.percentile(sorted, 99.0));
}

pub fn tick(self: *Self, dt_ns: u64, sample: Bench.PhaseSample) Phase {
    switch (self.phase) {
        .cold_start => {
            self.first_frame_ns = dt_ns;
            self.phase = .warmup;
            self.convergence.reset();
        },
        .warmup => {
            self.warmup_frames += 1;
            self.warmup_time_ns += dt_ns;
            const state = self.convergence.push(dt_ns);
            if (state == .converged or state == .capped) {
                if (state == .capped)
                    std.log.warn("profile: warmup did not converge - capped at {d} frames", .{self.warmup_frames});
                self.phase = .static_measure;
            }
        },
        .static_measure => {
            self.static_data.record(dt_ns, sample);
            self.measure_elapsed_ns += dt_ns;
            if (self.measure_elapsed_ns >= self.measure_duration_ns) {
                self.measure_elapsed_ns = 0;
                self.convergence.reset();
                self.phase = .moving_settle;
            }
        },
        .moving_settle => {
            self.settle_frames += 1;
            self.settle_time_ns += dt_ns;
            const state = self.convergence.push(dt_ns);
            if (state == .converged or state == .capped) {
                if (state == .capped)
                    std.log.warn("profile: moving settle did not converge - capped at {d} frames", .{self.settle_frames});
                self.phase = .moving_measure;
            }
        },
        .moving_measure => {
            self.moving_data.record(dt_ns, sample);
            self.measure_elapsed_ns += dt_ns;
            if (self.measure_elapsed_ns >= self.measure_duration_ns) {
                self.phase = .done;
            }
        },
        .done => {},
    }
    return self.phase;
}

pub fn report(self: *const Self) void {
    std.log.info("=== Profile: Cold Start ===", .{});
    if (self.tile_load_ns > 0) {
        std.log.info("  Init: {d:.1}ms  Tile load: {d:.1}ms ({d} tiles)  First frame: {d:.4}ms", .{
            @as(f64, @floatFromInt(self.init_time_ns)) / 1e6,
            @as(f64, @floatFromInt(self.tile_load_ns)) / 1e6,
            self.tile_count,
            @as(f64, @floatFromInt(self.first_frame_ns)) / 1e6,
        });
    } else {
        std.log.info("  Init: {d:.1}ms  First frame: {d:.4}ms", .{
            @as(f64, @floatFromInt(self.init_time_ns)) / 1e6,
            @as(f64, @floatFromInt(self.first_frame_ns)) / 1e6,
        });
    }
    std.log.info("=== Profile: Warmup ===", .{});
    std.log.info("  Converged after {d} frames ({d:.2}s)", .{
        self.warmup_frames,
        @as(f64, @floatFromInt(self.warmup_time_ns)) / 1e9,
    });
    self.static_data.report(self.allocator, "Static Rendering", false);
    if (self.settle_frames > 0) {
        std.log.info("=== Profile: Moving Settle ===", .{});
        std.log.info("  Converged after {d} frames ({d:.2}s)", .{
            self.settle_frames,
            @as(f64, @floatFromInt(self.settle_time_ns)) / 1e9,
        });
    }
    self.moving_data.report(self.allocator, "Moving (autopilot)", true);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn makeSample(scrolling: bool) Bench.PhaseSample {
    return .{
        .cpu_wait_ns = 50_000,
        .cpu_acquire_ns = 1_000,
        .cpu_submit_ns = 5_000,
        .cpu_present_ns = 5_000,
        .cpu_record_ns = 8_000,
        .gpu_compute_ns = if (scrolling) 3_000 else 0,
        .gpu_graphics_ns = 60_000,
        .scrolling = scrolling,
    };
}

test "phase transitions in order" {
    // Use short durations for testing: 500ms measurement windows
    var p = try Self.init(testing.allocator, 100_000, .{
        .measure_duration_ns = 500_000_000,
    });
    defer p.deinit();

    try testing.expectEqual(Phase.cold_start, p.phase);

    // Cold start -> warmup (1 frame)
    _ = p.tick(1_000_000, makeSample(false));
    try testing.expectEqual(Phase.warmup, p.phase);

    // Warmup -> static_measure (feed stable frames until convergence)
    // Convergence: window=120, consecutive=3, so need 120*4 = 480 frames
    for (0..480) |_| _ = p.tick(1_000_000, makeSample(false));
    try testing.expectEqual(Phase.static_measure, p.phase);

    // Static measure -> moving_settle (time-based: 500 frames at 1ms = 0.5s)
    for (0..500) |_| _ = p.tick(1_000_000, makeSample(false));
    try testing.expectEqual(Phase.moving_settle, p.phase);

    // Moving settle -> moving_measure (converge again)
    for (0..480) |_| _ = p.tick(1_000_000, makeSample(true));
    try testing.expectEqual(Phase.moving_measure, p.phase);

    // Moving measure -> done (time-based: 500 frames at 1ms = 0.5s)
    for (0..500) |_| _ = p.tick(1_000_000, makeSample(true));
    try testing.expectEqual(Phase.done, p.phase);
}

test "shouldFly matches phase" {
    var p = try Self.init(testing.allocator, 0, .{});
    defer p.deinit();

    try testing.expect(!p.shouldFly()); // cold_start
    _ = p.tick(1_000_000, makeSample(false));
    try testing.expect(!p.shouldFly()); // warmup
}

test "cold start records init and first frame time" {
    var p = try Self.init(testing.allocator, 42_000_000, .{});
    defer p.deinit();

    _ = p.tick(500_000, makeSample(false));
    try testing.expectEqual(@as(u64, 42_000_000), p.init_time_ns);
    try testing.expectEqual(@as(u64, 500_000), p.first_frame_ns);
}

test "measurement records frame data" {
    var m = try Measurement.init(testing.allocator, 10);
    defer m.deinit(testing.allocator);

    m.record(1_000_000, makeSample(false));
    m.record(1_100_000, makeSample(true));

    try testing.expectEqual(@as(usize, 2), m.frame_times.items.len);
    try testing.expectEqual(@as(u64, 2), m.phases.cpu_wait.count);
    try testing.expectEqual(@as(u64, 1), m.scrolling_compute.count);
    try testing.expectEqual(@as(u64, 0), m.stationary_compute.count);
}

test "results returns P99 in milliseconds" {
    var p = try Self.init(testing.allocator, 0, .{
        .measure_duration_ns = 200_000_000, // 200ms
    });
    defer p.deinit();

    // Drive through to static_measure
    _ = p.tick(1_000_000, makeSample(false)); // cold_start -> warmup
    for (0..480) |_| _ = p.tick(1_000_000, makeSample(false)); // warmup -> static

    // Record 100 frames at 10ms, 1 outlier at 20ms
    for (0..100) |_| _ = p.tick(10_000_000, makeSample(false));
    _ = p.tick(20_000_000, makeSample(false));

    const r = p.results();
    // P99 of 101 frames: rank = ceil(0.99*101) = 100, so the 100th sorted value
    // 100 frames at 10ms + 1 at 20ms -> P99 should be 10ms or 20ms depending on rank
    try testing.expect(r.static_p99_ms >= 10.0);
    try testing.expect(r.static_p99_ms <= 20.0);
}
