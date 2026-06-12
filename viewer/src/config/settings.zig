//! Persisted user preferences. This is the SERIALIZATION model, not the runtime
//! source of truth: subsystems keep reading their live fields (`Config.vsync`,
//! `gamepad.Tuning.deadzone`, ...). At startup the app loads a `Settings` from ZON
//! and copies it into those live fields (`applyToConfig` + caller-side tuning
//! copy); on change/exit it captures the live fields back (`captureFromConfig`)
//! and writes ZON.
//!
//! Import boundary: this file maps onto `Config` (same `config/` tree, so no extra
//! test drag) but deliberately does NOT import app types. The `gamepad.Tuning`
//! (and mouse-sensitivity) mapping is a few trivial field copies left to the
//! caller (`main.zig`), so a cross-tree import (which would duplicate gamepad's
//! tests here) is avoided.
//!
//! Field defaults MUST mirror the live defaults (`Config` / `gamepad.Tuning`) so a
//! fresh file (or a missing field) reproduces current behavior. Forward-compat:
//! parse ignores unknown fields and fills missing ones from defaults, so adding a
//! field later still loads an older file (and dropping one still loads a newer file).

const std = @import("std");
const Config = @import("options.zig").Config;
const Units = @import("options.zig").Units;
const Dir = std.Io.Dir;

const file_name = "settings.zon";
const max_file_size: std.Io.Limit = .limited(64 * 1024);

pub const Settings = struct {
    display: Display = .{},
    rendering: Rendering = .{},
    input: Input = .{},

    /// Mirrors `Config.vsync` / `Config.msaa`. `hdr` is the positive sense of
    /// `Config.no_hdr` (a file reads better as "HDR: on"); the app glue inverts it.
    pub const Display = struct {
        vsync: bool = true,
        msaa: u32 = 4, // 1 / 2 / 4 / 8; clamped to GPU support when applied
        hdr: bool = true, // use HDR when the hardware + display allow it
        fullscreen: bool = false, // borderless desktop fullscreen vs windowed
        monitor: u32 = 0, // display index; out-of-range falls back at apply time
        units: Units = .metric, // HUD units (mirrors Config.units)
    };

    /// `effects`/`fog` are the positive sense of `Config.no_effects`/`no_fog`
    /// (a settings file reads better as "Effects: on" than "no_effects: false");
    /// the app glue inverts on apply/capture. `fov` mirrors `Config.fov` (degrees).
    /// Also carries the clipmap render distance (the Graphics-tab sliders),
    /// replacing the old per-GPU `viewer.zon` autotune store: `ring_size` =
    /// per-ring detail (re-rounded to chunk alignment at load), `num_levels` =
    /// horizon reach. Single global value (no per-GPU keying); `--autotune` writes
    /// these, a normal launch reads them, the menu's Apply+Keep persists them.
    pub const Rendering = struct {
        effects: bool = true,
        fog: bool = true,
        taws: bool = false, // TAWS hazard overlay (mirrors Config.taws)
        fov: f32 = 45.0,
        ring_size: u32 = 1001, // mirrors Config.ring_size
        num_levels: u32 = 6, //   mirrors Config.num_levels
    };

    /// Input prefs, all copied caller-side in main (this file does not import the
    /// app types). `deadzone`/`invert_pitch`/`trigger_deadzone`/`look_sensitivity`
    /// mirror `gamepad.Tuning`; `look_drag_sensitivity` mirrors
    /// `Config.look_drag_sensitivity`.
    pub const Input = struct {
        deadzone: f32 = 0.1,
        invert_pitch: bool = true,
        trigger_deadzone: f32 = 0.05,
        look_sensitivity: f32 = 2.5, // mirrors gamepad.Tuning.look_sensitivity
        look_drag_sensitivity: f32 = 1.0, // mirrors Config.look_drag_sensitivity
    };

    /// Write `self` as ZON to `w`. Every field is emitted (a complete,
    /// hand-editable file) in standard Zig whitespace style.
    pub fn serialize(self: Settings, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try std.zon.stringify.serialize(self, .{}, w);
    }

    /// Parse ZON `source` (must be null-terminated) into a `Settings`. Unknown
    /// fields are ignored and missing fields take their defaults (forward-compat).
    /// `Settings` is pointer-free, so the result owns nothing and needs no free;
    /// `gpa` is only for the parser's transient allocations.
    pub fn parse(gpa: std.mem.Allocator, source: [:0]const u8) error{ OutOfMemory, ParseZon }!Settings {
        return std.zon.parse.fromSlice(Settings, gpa, source, null, .{ .ignore_unknown_fields = true });
    }

    /// Copy saved values into `config`, EXCEPT fields the CLI set explicitly: a
    /// flag on the command line wins over the file (precedence defaults < file <
    /// CLI). `effects`/`fog` invert to `no_effects`/`no_fog`. The caller copies the
    /// `input` category into `gamepad.Tuning` separately (see module doc).
    pub fn applyToConfig(self: Settings, config: *Config) void {
        if (!config.vsync_explicit) config.vsync = self.display.vsync;
        // Validate like parseArgs does for --msaa: a hand-edited out-of-range value
        // would otherwise reach `unreachable` in the renderer. fov is guarded by
        // Camera.init, so it needs no clamp here.
        if (!config.msaa_explicit) config.msaa = switch (self.display.msaa) {
            1, 2, 4, 8 => self.display.msaa,
            else => 4,
        };
        if (!config.no_hdr_explicit) config.no_hdr = !self.display.hdr;
        if (!config.fullscreen_explicit) config.fullscreen = self.display.fullscreen;
        if (!config.monitor_explicit) config.monitor = self.display.monitor;
        if (!config.units_explicit) config.units = self.display.units;
        if (!config.no_effects_explicit) config.no_effects = !self.rendering.effects;
        if (!config.no_fog_explicit) config.no_fog = !self.rendering.fog;
        if (!config.taws_explicit) config.taws = self.rendering.taws;
        if (!config.fov_explicit) config.fov = self.rendering.fov;
        // Clamp render distance like parseArgs validates --ring-size/--num-levels:
        // a hand-edited out-of-range value would otherwise reach Clipmap.init and
        // allocate a degenerate (or huge) ring. ring_size is re-rounded to chunk
        // alignment by Clipmap.init, so no oddness fix is needed here.
        if (!config.ring_size_explicit) config.ring_size = std.math.clamp(self.rendering.ring_size, 63, 2047);
        if (!config.num_levels_explicit) config.num_levels = std.math.clamp(self.rendering.num_levels, 1, 12);
    }

    /// Snapshot the live `config` fields into this `Settings` for saving. Inverts
    /// `no_effects`/`no_fog` back to `effects`/`fog`. Leaves `input` untouched (the
    /// caller fills it from `gamepad.Tuning`).
    pub fn captureFromConfig(self: *Settings, config: *const Config) void {
        self.display.vsync = config.vsync;
        self.display.msaa = config.msaa;
        self.display.hdr = !config.no_hdr;
        self.display.fullscreen = config.fullscreen;
        self.display.monitor = config.monitor;
        self.display.units = config.units;
        self.rendering.effects = !config.no_effects;
        self.rendering.fog = !config.no_fog;
        self.rendering.taws = config.taws;
        self.rendering.fov = config.fov;
        self.rendering.ring_size = config.ring_size;
        self.rendering.num_levels = config.num_levels;
    }

    /// Load `<config_dir>/settings.zon`, or null if it is missing or unparseable
    /// (the caller falls back to defaults). The result is pointer-free, so the
    /// parse arena is freed before returning; `gpa` backs only transient work.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, config_dir: []const u8) ?Settings {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const aa = arena.allocator();
        const path = std.fs.path.join(aa, &.{ config_dir, file_name }) catch return null;
        const content = Dir.cwd().readFileAllocOptions(io, path, aa, max_file_size, .of(u8), 0) catch |err| {
            // A missing file is the normal first-run case; anything else (perms,
            // too large) is worth a line before we silently fall back to defaults.
            if (err != error.FileNotFound) std.log.warn("Failed to read {s}: {} - using default settings", .{ path, err });
            return null;
        };
        return parse(aa, content) catch |err| {
            std.log.warn("Failed to parse {s}: {} - using default settings", .{ path, err });
            return null;
        };
    }

    /// Write this `Settings` to `<config_dir>/settings.zon`, creating the dir if
    /// needed. The full struct is emitted so the file is complete and editable.
    pub fn save(self: Settings, gpa: std.mem.Allocator, io: std.Io, config_dir: []const u8) !void {
        Dir.cwd().createDirPath(io, config_dir) catch {};
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const aa = arena.allocator();
        const path = try std.fs.path.join(aa, &.{ config_dir, file_name });
        var aw: std.Io.Writer.Allocating = .init(gpa);
        defer aw.deinit();
        try self.serialize(&aw.writer);
        try Dir.cwd().writeFile(io, .{ .sub_path = path, .data = aw.written() });
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

/// Serialize to a heap buffer, returning a null-terminated copy the caller frees.
fn toZonZ(gpa: std.mem.Allocator, s: Settings) ![:0]u8 {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try s.serialize(&aw.writer);
    return gpa.dupeZ(u8, aw.written());
}

test "settings: ZON round-trips every category" {
    const a = testing.allocator;
    var s = Settings{};
    s.display.vsync = false;
    s.display.msaa = 8;
    s.display.hdr = false;
    s.display.fullscreen = true;
    s.display.monitor = 2;
    s.display.units = .imperial;
    s.rendering.effects = false;
    s.rendering.taws = true;
    s.rendering.fov = 60.0;
    s.rendering.ring_size = 505;
    s.rendering.num_levels = 8;
    s.input.deadzone = 0.2;
    s.input.invert_pitch = false;
    s.input.look_sensitivity = 4.0;
    s.input.look_drag_sensitivity = 2.0;

    const text = try toZonZ(a, s);
    defer a.free(text);
    const back = try Settings.parse(a, text);

    try testing.expectEqual(false, back.display.vsync);
    try testing.expectEqual(@as(u32, 8), back.display.msaa);
    try testing.expectEqual(false, back.display.hdr);
    try testing.expectEqual(true, back.display.fullscreen);
    try testing.expectEqual(@as(u32, 2), back.display.monitor);
    try testing.expectEqual(Units.imperial, back.display.units);
    try testing.expectEqual(false, back.rendering.effects);
    try testing.expectEqual(true, back.rendering.taws);
    try testing.expectEqual(true, back.rendering.fog); // untouched default survives
    try testing.expectApproxEqAbs(@as(f32, 60.0), back.rendering.fov, 1e-4);
    try testing.expectEqual(@as(u32, 505), back.rendering.ring_size);
    try testing.expectEqual(@as(u32, 8), back.rendering.num_levels);
    try testing.expectApproxEqAbs(@as(f32, 0.2), back.input.deadzone, 1e-4);
    try testing.expectEqual(false, back.input.invert_pitch);
    try testing.expectApproxEqAbs(@as(f32, 0.05), back.input.trigger_deadzone, 1e-4); // default
    try testing.expectApproxEqAbs(@as(f32, 4.0), back.input.look_sensitivity, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.0), back.input.look_drag_sensitivity, 1e-4);
}

test "settings: parse fills missing fields and ignores unknown (forward compat)" {
    const a = testing.allocator;
    // A file from an older build (no `msaa`, no `rendering`) AND a newer build
    // (an unknown `future` field) must both load without error.
    const text: [:0]const u8 = ".{ .display = .{ .vsync = false }, .future = 123 }";
    const s = try Settings.parse(a, text);
    try testing.expectEqual(false, s.display.vsync); // present field read
    try testing.expectEqual(@as(u32, 4), s.display.msaa); // missing field -> default
    try testing.expectApproxEqAbs(@as(f32, 45.0), s.rendering.fov, 1e-4); // missing category -> default
}

test "settings: applyToConfig fills non-explicit fields, CLI-explicit wins" {
    var config = Config{}; // defaults: vsync true, msaa 4, no_effects false, no_fog false, fov 45
    config.vsync_explicit = true; // pretend `--no-vsync` was passed
    config.vsync = false;
    config.no_hdr_explicit = true; // pretend `--no-hdr` was passed
    config.no_hdr = true;
    config.fullscreen_explicit = true; // pretend `--fullscreen` was passed
    config.fullscreen = false;
    config.taws_explicit = true; // pretend `--taws` was passed
    config.taws = true;

    var s = Settings{};
    s.display.vsync = true; // file says vsync on...
    s.display.msaa = 8; // ...msaa 8 (not explicit -> applies)
    s.display.hdr = true; // file says HDR on, but --no-hdr is explicit -> must not apply
    s.display.fullscreen = true; // file says fullscreen, but explicit -> must not apply
    s.display.monitor = 3; // not explicit -> applies
    s.rendering.effects = false; // effects off -> no_effects true
    s.rendering.taws = false; // file says TAWS off, but --taws is explicit -> must not apply
    s.rendering.fov = 70.0;
    s.applyToConfig(&config);

    try testing.expectEqual(false, config.vsync); // CLI flag wins over file
    try testing.expectEqual(@as(u32, 8), config.msaa); // file applied
    try testing.expectEqual(true, config.no_hdr); // CLI --no-hdr wins over file hdr=true
    try testing.expectEqual(false, config.fullscreen); // CLI flag wins over file
    try testing.expectEqual(@as(u32, 3), config.monitor); // file applied (not explicit)
    try testing.expectEqual(true, config.no_effects); // inverted from effects=false
    try testing.expectEqual(true, config.taws); // CLI --taws wins over file taws=false
    try testing.expectApproxEqAbs(@as(f32, 70.0), config.fov, 1e-4);
}

test "settings: applyToConfig rejects an out-of-range msaa (hand-edited file)" {
    var config = Config{};
    var s = Settings{};
    s.display.msaa = 3; // not 1/2/4/8 -> would hit `unreachable` in the renderer
    s.applyToConfig(&config);
    try testing.expectEqual(@as(u32, 4), config.msaa); // clamped to a safe default
    s.display.msaa = 8; // a valid value still applies
    s.applyToConfig(&config);
    try testing.expectEqual(@as(u32, 8), config.msaa);
}

test "settings: applyToConfig clamps hand-edited render distance, CLI-explicit wins" {
    var config = Config{};
    var s = Settings{};
    s.rendering.ring_size = 999_999; // out of [63, 2047]
    s.rendering.num_levels = 99; // out of [1, 12]
    s.applyToConfig(&config);
    try testing.expectEqual(@as(u32, 2047), config.ring_size);
    try testing.expectEqual(@as(u32, 12), config.num_levels);

    // A CLI --ring-size / --num-levels wins over the file.
    var c2 = Config{};
    c2.ring_size_explicit = true;
    c2.ring_size = 777;
    c2.num_levels_explicit = true;
    c2.num_levels = 5;
    var s2 = Settings{};
    s2.rendering.ring_size = 1001;
    s2.rendering.num_levels = 8;
    s2.applyToConfig(&c2);
    try testing.expectEqual(@as(u32, 777), c2.ring_size);
    try testing.expectEqual(@as(u32, 5), c2.num_levels);
}

test "settings: captureFromConfig inverts effects/fog back" {
    var config = Config{};
    config.no_effects = true;
    config.no_fog = true;
    config.msaa = 2;
    config.fov = 55.0;
    var s = Settings{};
    s.captureFromConfig(&config);
    try testing.expectEqual(false, s.rendering.effects);
    try testing.expectEqual(false, s.rendering.fog);
    try testing.expectEqual(@as(u32, 2), s.display.msaa);
    try testing.expectApproxEqAbs(@as(f32, 55.0), s.rendering.fov, 1e-4);
}

test "settings: capture then apply is a round-trip through Config" {
    var src = Config{};
    src.vsync = false;
    src.msaa = 8;
    src.no_effects = true;
    src.no_fog = true;
    src.taws = true;
    src.fov = 90.0;

    var s = Settings{};
    s.captureFromConfig(&src);

    var dst = Config{}; // fresh defaults, nothing explicit
    s.applyToConfig(&dst);
    try testing.expectEqual(false, dst.vsync);
    try testing.expectEqual(@as(u32, 8), dst.msaa);
    try testing.expectEqual(true, dst.no_effects);
    try testing.expectEqual(true, dst.no_fog);
    try testing.expectEqual(true, dst.taws);
    try testing.expectApproxEqAbs(@as(f32, 90.0), dst.fov, 1e-4);
}

test "settings: defaults mirror live defaults (fresh file == current behavior)" {
    const s = Settings{};
    try testing.expectEqual(true, s.display.vsync);
    try testing.expectEqual(@as(u32, 4), s.display.msaa);
    try testing.expectEqual(true, s.display.hdr);
    try testing.expectEqual(false, s.display.fullscreen);
    try testing.expectEqual(@as(u32, 0), s.display.monitor);
    try testing.expectEqual(Units.metric, s.display.units);
    try testing.expectEqual(true, s.rendering.effects);
    try testing.expectEqual(true, s.rendering.fog);
    try testing.expectEqual((Config{}).taws, s.rendering.taws);
    try testing.expectApproxEqAbs(@as(f32, 45.0), s.rendering.fov, 1e-4);
    // Render-distance defaults must mirror Config's (a fresh file == current behavior).
    const c = Config{};
    try testing.expectEqual(c.ring_size, s.rendering.ring_size);
    try testing.expectEqual(c.num_levels, s.rendering.num_levels);
    try testing.expectApproxEqAbs(@as(f32, 0.1), s.input.deadzone, 1e-4);
    try testing.expectEqual(true, s.input.invert_pitch);
    try testing.expectApproxEqAbs(@as(f32, 0.05), s.input.trigger_deadzone, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.5), s.input.look_sensitivity, 1e-4); // mirrors gamepad.Tuning default
    try testing.expectApproxEqAbs((Config{}).look_drag_sensitivity, s.input.look_drag_sensitivity, 1e-4);
}
