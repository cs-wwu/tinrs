//! Session aggregate: per-run state for camera, aircraft, frame stats, and benchmarks.
//!
//! Profile is attached after the tile streamer has been drained so its
//! tile_load metrics are real.

const std = @import("std");
const Aircraft = @import("app/aircraft.zig").Aircraft;
const Camera = @import("app/camera.zig").Camera;
const sim = @import("app/sim.zig");
const FrameStats = @import("bench/frame_stats.zig");
const Bench = @import("bench/runner.zig");
const Profile = @import("bench/profile.zig");
const Config = @import("config/options.zig").Config;

pub const Session = struct {
    camera: Camera,
    aircraft: Aircraft,
    ticker: sim.Ticker,
    frame_stats: FrameStats,
    bench: ?Bench,
    profile: ?Profile,
    init_time_ns: u64,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        config: *const Config,
        init_time_ns: u64,
    ) !Session {
        const start_pos = config.cameraStartPos();
        const start_dir = config.cameraStartDir();
        var camera = Camera.init(start_pos, config.fov, start_dir);
        camera.near = config.near;
        camera.far = config.far;
        if (camera.mode == .cockpit or camera.mode == .chase)
            camera.fov = Camera.FLIGHT_FOV_LOW;
        var aircraft = Aircraft.init(start_pos, start_dir);
        // Spawn flying (sim only): trim the throttle to hold the start airspeed
        // (thrust == drag in level flight) so the plane cruises on launch instead
        // of stalling. pose.velocity stays {0} and converges to airspeed*front over
        // the EMA window, so HUD groundspeed ramps in over the first ~10 frames on
        // launch. In sensor mode the aircraft is a frozen hold (airspeed/throttle
        // stay 0) until SensorInput writes real state, so we don't spawn it flying.
        if (config.flight_source == .sim) {
            aircraft.airspeed = config.start_airspeed;
            aircraft.throttle = sim.trimThrottle(config.start_airspeed);
        }

        const bench: ?Bench = if (config.benchmark and !config.profile)
            try Bench.init(allocator, io, config.benchmark_frames, config.benchmark_warmup)
        else
            null;

        return .{
            .camera = camera,
            .aircraft = aircraft,
            .ticker = sim.Ticker.init(aircraft.pose),
            .frame_stats = .{},
            .bench = bench,
            .profile = null,
            .init_time_ns = init_time_ns,
        };
    }

    /// Attach a Profile collector. Called after the tile streamer has been
    /// synchronously drained so the reported tile_load metrics are accurate.
    pub fn attachProfile(
        self: *Session,
        allocator: std.mem.Allocator,
        opts: Profile.Options,
    ) !void {
        self.profile = try Profile.init(allocator, self.init_time_ns, opts);
    }

    pub fn deinit(self: *Session) void {
        if (self.profile) |*p| p.deinit();
        if (self.bench) |*b| b.deinit();
    }
};
