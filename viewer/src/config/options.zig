const std = @import("std");
const clap = @import("clap");
const coords = @import("../terrain/coords.zig");
const Camera = @import("../app/camera.zig").Camera;
const build_options = @import("build_options");

/// Vulkan validation layers + extras. All-false means no layers loaded.
/// `core` is the standard `VK_LAYER_KHRONOS_validation`; `sync` and `bp` are
/// extra features enabled via `VkValidationFeaturesEXT` in the instance pNext.
pub const ValidateLayers = struct {
    core: bool = false,
    sync: bool = false,
    bp: bool = false,

    pub fn any(self: ValidateLayers) bool {
        return self.core or self.sync or self.bp;
    }
};

/// HUD display units. `metric` = m / km/h / m/s; `imperial` = the aviation set
/// (ft / kt / ft/min). Drives both the CPU readouts (hud.zig) and the GPU-resident
/// AGL scale (numeric.vert).
pub const Units = enum { metric, imperial };

/// Flight state source, orthogonal to `CameraMode`. `sim` = the kinematic arcade
/// flight model drives the aircraft (WASD / gamepad flight). `sensor` = the
/// aircraft holds a frozen pose (the `SensorInput` seam: GPS / IMU / compass
/// fusion writes it later); the sim tick is skipped so there is no flight input,
/// but free-look stays live for look-around. Set by the `--sensor` CLI flag for
/// the current run ONLY: it is deliberately NOT persisted to settings.zon, so a
/// one-off can't strand you with a frozen aircraft and no way back. Production
/// will flip it on from the HUD once real sensors exist; until then there is no
/// runtime toggle.
pub const FlightSource = enum { sim, sensor };

pub const Config = struct {
    // Terrain & model
    tile: []const u8 = "n47w122",
    model: ?[]const u8 = build_options.default_model_path,
    tile_origin: coords.WorldPos = .{ .x = 0, .z = 0 }, // overwritten by parseArgs from tile name

    // Window
    width: u32 = 1280,
    height: u32 = 720,
    fullscreen: bool = false,
    monitor: u32 = 0,
    fullscreen_explicit: bool = false,
    monitor_explicit: bool = false,
    gpu: ?u32 = null,

    // Camera
    start_pos: [3]f32 = .{ 46.764308, -122.21734, 750.0 },
    start_heading: f32 = 90.0, // degrees: 0=N, 90=E, 180=S, 270=W
    start_pitch: f32 = 0.0, // degrees: + up, - down
    fov: f32 = 45.0, // degrees
    near: f32 = Camera.DEFAULT_NEAR, // arcsec
    far: f32 = Camera.DEFAULT_FAR, // arcsec

    // HUD
    units: Units = .metric,
    units_explicit: bool = false, // true if --units was passed on CLI

    // Flight
    /// Spawn airspeed in arcsec/s. The default cruises above stall onset (2.5)
    /// and below the full-throttle terminal (6.75) so the aircraft is flying on
    /// launch instead of stalling; the spawn throttle is trimmed to hold it
    /// (sim.trimThrottle). 0 = spawn stalled (the old behavior).
    start_airspeed: f32 = 5.0,

    /// Flight state source (see `FlightSource`). Default `.sim`; the `--sensor`
    /// CLI flag selects `.sensor` for the current run. Not persisted to
    /// settings.zon, not a runtime toggle.
    flight_source: FlightSource = .sim,

    // Input
    /// Pointer free-look (mouse + touch drag) sensitivity multiplier (Settings/
    /// Input tab; persisted to settings.zon). Scales `Camera.LOOK_DRAG_BASE`
    /// rad/px. No CLI flag.
    look_drag_sensitivity: f32 = 1.0,

    // Clipmap LOD
    ring_size: u32 = 1001,
    num_levels: u32 = 6,
    base_spacing: f32 = 1.0,
    ring_size_explicit: bool = false, // true if --ring-size was passed on CLI
    num_levels_explicit: bool = false, // true if --num-levels was passed on CLI

    // Runtime modes
    vsync: bool = true,
    procedural: bool = false,
    benchmark: bool = false,
    benchmark_frames: u64 = 0, // 0 = run until window closes; >0 = exit after N frames
    benchmark_warmup: u64 = 100, // frames excluded from steady-state stats (lets GPU spin up)
    benchmark_fly: bool = false, // autopilot camera movement during benchmark (exercises INR compute)
    profile: bool = false, // self-calibrating multi-phase benchmark (implies benchmark)
    autotune: bool = false, // auto-detect best ring_size for this GPU
    target_fps: ?u32 = null, // target framerate for autotune (null = monitor refresh rate)
    headless: bool = false, // skip SDL + window; render to VK_EXT_headless_surface for clean benching
    debug: bool = false, // verbose runtime logs (enumerations, full config dump, log.debug fires)
    validate: ValidateLayers = .{},
    no_effects: bool = false, // use simple lighting (benchmark-stable, no fog/ambient/slopes)
    no_hdr: bool = false,
    no_sky: bool = false,
    no_fog: bool = false,
    /// TAWS terrain-hazard overlay: recolor terrain red/yellow by clearance
    /// below the aircraft (Garmin SVT-style). Toggled in the Settings/Graphics
    /// tab, persisted to settings.zon.
    taws: bool = false,

    /// HUD visibility override. null = default (shown windowed, hidden headless).
    /// `--hud` forces it on (e.g. to measure HUD cost under --headless --profile);
    /// `--no-hud` starts hidden. Either way the `H` key still toggles at runtime.
    hud_override: ?bool = null,
    msaa: u32 = 4, // MSAA sample count: 1 (off), 2, 4, or 8. Clamped to GPU support at runtime.
    msaa_explicit: bool = false, // true if --msaa was passed on CLI

    // "Explicit" flags: true when the CLI set the field, so a saved settings file
    // does not override it (precedence defaults < settings.zon < CLI). See
    // settings.zig applyToConfig. msaa_explicit above serves the same role.
    vsync_explicit: bool = false,
    no_effects_explicit: bool = false,
    no_fog_explicit: bool = false,
    no_hdr_explicit: bool = false,
    fov_explicit: bool = false,
    taws_explicit: bool = false,

    /// Cap on tile-load uploads (cmdCopyBuffer regions) per frame. Defaults
    /// to 4 on desktop; lower (1) on Pi-class GPUs where each tile copy is
    /// proportionally more expensive against frame budget.
    max_tile_uploads_per_frame: u32 = 4,

    /// Cap on resident tile pool size. null = auto-derive from render distance
    /// and VRAM; explicit value overrides the auto calculation.
    max_tiles: ?u32 = null,

    // Allocator for owned strings (tile, model). Null when using defaults without parseArgs.
    allocator: ?std.mem.Allocator = null,

    pub fn deinit(self: Config) void {
        const alloc = self.allocator orelse return;
        alloc.free(self.tile);
        if (self.model) |m| alloc.free(m);
    }

    /// Compute the camera starting position in arcseconds from lat/lon/alt.
    pub fn cameraStartPos(self: Config) [3]f64 {
        return .{
            @as(f64, self.start_pos[1]) * coords.TILE_ARCSEC, // lon -> X arcseconds
            @as(f64, self.start_pos[2]) * coords.HEIGHT_SCALE, // alt meters -> Y arcseconds
            @as(f64, -self.start_pos[0]) * coords.TILE_ARCSEC, // lat -> Z arcseconds (negated)
        };
    }

    /// Convert (heading, pitch) in degrees to a unit direction vector.
    /// Heading 0=-Z (north), 90=+X (east); pitch + tilts +Y (up). All in degrees.
    pub fn cameraStartDir(self: Config) [3]f32 {
        const h = self.start_heading * (std.math.pi / 180.0);
        const p = self.start_pitch * (std.math.pi / 180.0);
        const cp = @cos(p);
        return .{ @sin(h) * cp, @sin(p), -@cos(h) * cp };
    }
};

const params = clap.parseParamsComptime(std.fmt.comptimePrint(
    \\-h, --help             Display this help and exit.
    \\-m, --model <str>      Weights directory path (default: {s}).
    \\-t, --tile <str>       Tile name, e.g. n47w122 (default: n47w122).
    \\    --width <u32>      Window width (default: 1280).
    \\    --height <u32>     Window height (default: 720).
    \\-f, --fullscreen       Fullscreen on selected monitor.
    \\-M, --monitor <u32>    Monitor index for fullscreen (default: 0).
    \\-g, --gpu <u32>        Override GPU selection by candidate index (default: highest score).
    \\    --pos <str>        Start position "lat,lon,alt_m" (default: tile center).
    \\    --heading <f32>    Compass heading in degrees: 0=N, 90=E, 180=S, 270=W (default: 90).
    \\    --pitch <f32>      Pitch in degrees: + up, - down (default: 0).
    \\    --airspeed <f32>   Spawn airspeed in arcsec/s (default: 5.0, ~555 km/h; 0 = stalled).
    \\    --sensor           Non-simulated flight: freeze the aircraft pose for sensor swap-in (skips the sim flight model); free-look stays live.
    \\    --units <str>      HUD units: metric (m, km/h) or imperial (ft, kt) (default: metric).
    \\    --fov <f32>        Field of view in degrees (default: 45).
    \\    --near <f32>       Near clip plane in arcseconds (default: 0.01, ~0.3m).
    \\    --far <f32>        Far clip plane in arcseconds (default: 200000, ~6175km).
    \\-u, --no-vsync         Disable vsync (uncap framerate).
    \\-b, --benchmark        Enable benchmark stat tracking.
    \\    --benchmark-frames <u64>  Exit after N frames (implies -b; default: 0 = run until window close).
    \\    --benchmark-warmup <u64>  Frames excluded from steady-state stats (default: 100).
    \\    --benchmark-fly         Autopilot camera during benchmark; exercises INR compute (implies -b).
    \\    --profile               Self-calibrating benchmark: auto-detect warmup, measure static + moving (implies -b).
    \\    --autotune              Auto-detect best ring_size for this GPU and save to config file.
    \\    --target-fps <u32>      Target framerate for autotune (default: monitor refresh rate or 60).
    \\-H, --headless          Run without a window; pure Vulkan benchmark (implies -b).
    \\-d, --debug            Verbose runtime logging; full enumerations, config dump, log.debug fires.
    \\-V, --validate         Vulkan validation layers (core only).
    \\    --validate-all     Vulkan validation layers + sync validation + best-practices.
    \\    --ring-size <u32>   Clipmap ring grid dimension per level (default: 1001, must be odd, 63-2047).
    \\    --num-levels <u32>  Number of clipmap LOD levels (default: 6, range 1-12).
    \\    --base-spacing <f32> Finest level grid spacing in arcseconds (default: 1.0, range 0.1-10.0).
    \\-p, --procedural       Use procedural terrain (skip model loading).
    \\    --no-effects       Use simple lighting (benchmark-stable, no fog/ambient/slopes). Implies --msaa 1 unless --msaa is set.
    \\    --no-hdr           Disable HDR auto-detection (force sRGB output).
    \\    --no-sky           Disable sky rendering.
    \\    --no-fog           Disable distance fog.
    \\    --taws             Enable the TAWS terrain-hazard color overlay (red/yellow by clearance).
    \\    --hud              Force the HUD on (e.g. measure its cost under --headless --profile).
    \\    --no-hud           Start with the HUD hidden (toggle at runtime with H).
    \\    --msaa <u32>       MSAA sample count: 1 (off), 2, 4, 8 (default: 4). Clamped to GPU support.
    \\    --max-tile-uploads-per-frame <u32>  Cap on tile uploads per frame (default: 4 desktop / 1 Pi).
    \\    --max-tiles <u32>                   Cap on resident tile pool (default: auto from render distance + VRAM).
    \\
, .{build_options.default_model_path}));

pub fn parseArgs(process_args: std.process.Args, allocator: std.mem.Allocator, io: std.Io) !Config {
    @setEvalBranchQuota(4000); // clap's comptime parser scales with param count
    var diag: clap.Diagnostic = .{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, process_args, .{
        .allocator = allocator,
        .diagnostic = &diag,
    }) catch |err| {
        diag.reportToFile(io, .stderr(), err) catch {};
        return error.InvalidArgs;
    };
    defer res.deinit();

    if (res.args.help != 0) {
        clap.helpToFile(io, .stderr(), clap.Help, &params, .{}) catch {};
        std.process.exit(0);
    }

    var config = Config{};

    // Always dupe string args so Config owns them (freed via config.deinit()).
    config.tile = try allocator.dupe(u8, res.args.tile orelse config.tile);
    errdefer allocator.free(config.tile);
    if (res.args.procedural == 0) {
        const model_src: []const u8 = res.args.model orelse config.model orelse "";
        config.model = try allocator.dupe(u8, model_src);
    } else {
        config.procedural = true;
        config.model = null;
    }
    errdefer if (config.model) |m| allocator.free(m);
    config.allocator = allocator;

    if (res.args.width) |w| config.width = w;
    if (res.args.height) |h| config.height = h;
    if (res.args.monitor) |m| {
        config.monitor = m;
        config.monitor_explicit = true;
    }
    if (res.args.gpu) |g| config.gpu = g;
    if (res.args.fov) |f| {
        config.fov = f;
        config.fov_explicit = true;
    }
    if (res.args.near) |n| config.near = n;
    if (res.args.far) |f| config.far = f;

    if (res.args.fullscreen != 0) {
        config.fullscreen = true;
        config.fullscreen_explicit = true;
    }
    if (res.args.@"no-vsync" != 0) {
        config.vsync = false;
        config.vsync_explicit = true;
    }
    if (res.args.benchmark != 0) config.benchmark = true;
    if (res.args.@"benchmark-frames") |n| {
        config.benchmark_frames = n;
        config.benchmark = true;
    }
    if (res.args.@"benchmark-warmup") |n| config.benchmark_warmup = n;
    if (res.args.@"benchmark-fly" != 0) {
        config.benchmark_fly = true;
        config.benchmark = true;
    }
    if (res.args.profile != 0) {
        config.profile = true;
        config.benchmark = true;
    }
    if (res.args.autotune != 0) {
        config.autotune = true;
        config.benchmark = true;
        config.vsync = false;
    }
    if (res.args.@"target-fps") |fps| config.target_fps = fps;
    if (res.args.headless != 0) {
        config.headless = true;
        config.benchmark = true; // headless only makes sense for benchmarking
    }
    if (res.args.debug != 0) config.debug = true;
    if (res.args.validate != 0) config.validate.core = true;
    if (res.args.@"validate-all" != 0) config.validate = .{ .core = true, .sync = true, .bp = true };
    if (res.args.@"no-effects" != 0) {
        config.no_effects = true;
        config.no_effects_explicit = true;
    }
    if (res.args.@"no-hdr" != 0) {
        config.no_hdr = true;
        config.no_hdr_explicit = true;
    }
    if (res.args.@"no-sky" != 0) config.no_sky = true;
    if (res.args.@"no-fog" != 0) {
        config.no_fog = true;
        config.no_fog_explicit = true;
    }
    if (res.args.taws != 0) {
        config.taws = true;
        config.taws_explicit = true;
    }
    if (res.args.hud != 0) config.hud_override = true;
    if (res.args.@"no-hud" != 0) config.hud_override = false;
    if (res.args.msaa) |s| {
        if (s != 1 and s != 2 and s != 4 and s != 8) {
            std.log.err("--msaa must be 1, 2, 4, or 8 (got {d})", .{s});
            return error.InvalidArgs;
        }
        config.msaa = s;
        config.msaa_explicit = true;
    }
    // Mirror the V-key A/B toggle: --no-effects without an explicit --msaa
    // also drops MSAA to 1x. Effects + MSAA are coupled in the interactive UX,
    // and MSAA dominates frame cost; leaving it on undercuts the "minimal pipeline" intent.
    if (config.no_effects and !config.msaa_explicit) config.msaa = 1;
    if (res.args.@"ring-size") |rs| {
        config.ring_size = rs;
        config.ring_size_explicit = true;
    }
    if (res.args.@"num-levels") |nl| {
        config.num_levels = nl;
        config.num_levels_explicit = true;
    }
    if (res.args.@"base-spacing") |bs| config.base_spacing = bs;
    if (res.args.@"max-tile-uploads-per-frame") |n| config.max_tile_uploads_per_frame = n;
    if (res.args.@"max-tiles") |n| {
        if (n < 4 or n > 8192) {
            std.log.err("--max-tiles must be in [4, 8192] (got {d})", .{n});
            return error.InvalidArgs;
        }
        config.max_tiles = n;
    }

    // Parse --pos "lat,lon,alt"
    if (res.args.pos) |pos_str| {
        config.start_pos = parseVec3(pos_str) orelse {
            std.log.err("Invalid --pos format, expected \"lat,lon,alt\" (e.g. \"48.5,-122.5,2000\")", .{});
            return error.InvalidArgs;
        };
    }

    if (res.args.heading) |h| config.start_heading = h;
    if (res.args.pitch) |p| {
        if (p < -90.0 or p > 90.0) {
            std.log.err("Invalid --pitch {d}, must be in [-90, 90]", .{p});
            return error.InvalidArgs;
        }
        config.start_pitch = p;
    }
    if (res.args.airspeed) |a| {
        // Reject negatives and non-finite (parseFloat accepts "nan"/"inf"); a NaN
        // airspeed would propagate into the pose and corrupt position/camera.
        if (a < 0 or !std.math.isFinite(a)) {
            std.log.err("Invalid --airspeed {d}, must be a finite value >= 0", .{a});
            return error.InvalidArgs;
        }
        config.start_airspeed = a;
    }
    if (res.args.sensor != 0) config.flight_source = .sensor;
    if (res.args.units) |u_str| {
        if (std.mem.eql(u8, u_str, "metric")) {
            config.units = .metric;
        } else if (std.mem.eql(u8, u_str, "imperial")) {
            config.units = .imperial;
        } else {
            std.log.err("Invalid --units \"{s}\", expected metric or imperial", .{u_str});
            return error.InvalidArgs;
        }
        config.units_explicit = true;
    }

    // Validate
    if (config.fov <= 0 or config.fov >= 180) {
        std.log.err("Invalid --fov {d}, must be in (0, 180)", .{config.fov});
        return error.InvalidArgs;
    }
    if (config.near <= 0) {
        std.log.err("Invalid --near {d}, must be > 0", .{config.near});
        return error.InvalidArgs;
    }
    if (config.far <= config.near) {
        std.log.err("Invalid --far {d}, must be > --near {d}", .{ config.far, config.near });
        return error.InvalidArgs;
    }

    if (config.ring_size < 63 or config.ring_size > 2047) {
        std.log.err("Invalid --ring-size {d}, must be in [63, 2047]", .{config.ring_size});
        return error.InvalidArgs;
    }
    if (config.ring_size % 2 == 0) {
        std.log.err("Invalid --ring-size {d}, must be odd", .{config.ring_size});
        return error.InvalidArgs;
    }
    if (config.num_levels < 1 or config.num_levels > 12) {
        std.log.err("Invalid --num-levels {d}, must be in [1, 12]", .{config.num_levels});
        return error.InvalidArgs;
    }
    if (config.base_spacing < 0.1 or config.base_spacing > 10.0) {
        std.log.err("Invalid --base-spacing {d:.2}, must be in [0.1, 10.0]", .{config.base_spacing});
        return error.InvalidArgs;
    }

    // Resolve tile origin
    config.tile_origin = coords.tileToWorldRuntime(config.tile) catch {
        std.log.err("Invalid --tile \"{s}\", expected format like \"n47w122\"", .{config.tile});
        return error.InvalidArgs;
    };

    return config;
}

/// Pretty-print every field of Config. Used by main.zig under --debug so users
/// see every effective setting, not a hand-picked subset that drifts.
pub fn dump(c: Config) void {
    std.log.debug("Config:", .{});
    std.log.debug("  tile={s} model={s} tile_origin=({d},{d})", .{
        c.tile, c.model orelse "(procedural)", c.tile_origin.x, c.tile_origin.z,
    });
    std.log.debug("  window: {d}x{d} fullscreen={} monitor={d} gpu={?d}", .{
        c.width, c.height, c.fullscreen, c.monitor, c.gpu,
    });
    std.log.debug("  camera: pos={f} heading={d:.1} pitch={d:.1} fov={d:.1} (deg)", .{
        fmtVec3(c.start_pos), c.start_heading, c.start_pitch, c.fov,
    });
    std.log.debug("  flight: source={s} start_airspeed={d:.2} arcsec/s", .{ @tagName(c.flight_source), c.start_airspeed });
    std.log.debug("  clipmap: ring_size={d}{s} num_levels={d} base_spacing={d:.3}", .{
        c.ring_size, if (c.ring_size_explicit) " (explicit)" else "", c.num_levels, c.base_spacing,
    });
    std.log.debug("  modes: vsync={} procedural={} bench={} bench_frames={d} bench_warmup={d} bench_fly={} profile={} autotune={} target_fps={?d} headless={}", .{
        c.vsync, c.procedural, c.benchmark, c.benchmark_frames, c.benchmark_warmup, c.benchmark_fly,
        c.profile, c.autotune, c.target_fps, c.headless,
    });
    std.log.debug("  streaming: max_tile_uploads={d} max_tiles={?d}", .{
        c.max_tile_uploads_per_frame, c.max_tiles,
    });
    std.log.debug("  rendering: msaa={d}{s} no_effects={} no_hdr={} no_sky={} no_fog={} taws={} hud_override={?} units={s}", .{
        c.msaa, if (c.msaa_explicit) " (explicit)" else "", c.no_effects, c.no_hdr, c.no_sky, c.no_fog, c.taws, c.hud_override, @tagName(c.units),
    });
    std.log.debug("  validate: core={} sync={} bp={}", .{ c.validate.core, c.validate.sync, c.validate.bp });
}

const Vec3Fmt = struct {
    v: [3]f32,
    pub fn format(self: Vec3Fmt, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("({d:.3},{d:.3},{d:.3})", .{ self.v[0], self.v[1], self.v[2] });
    }
};

fn fmtVec3(v: [3]f32) Vec3Fmt {
    return .{ .v = v };
}

fn parseVec3(s: []const u8) ?[3]f32 {
    var it = std.mem.splitScalar(u8, s, ',');
    const x = std.fmt.parseFloat(f32, it.next() orelse return null) catch return null;
    const y = std.fmt.parseFloat(f32, it.next() orelse return null) catch return null;
    const z = std.fmt.parseFloat(f32, it.next() orelse return null) catch return null;
    if (it.next() != null) return null; // too many components
    return .{ x, y, z };
}
