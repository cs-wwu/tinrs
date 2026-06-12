const std = @import("std");

fn compileShader(b: *std.Build, path: []const u8, out: []const u8) std.Build.LazyPath {
    return compileShaderWithDefines(b, path, out, &.{});
}

fn compileShaderWithDefines(
    b: *std.Build,
    path: []const u8,
    out: []const u8,
    defines: []const []const u8,
) std.Build.LazyPath {
    const cmd = b.addSystemCommand(&.{ "glslc", "--target-env=vulkan1.2" });
    for (defines) |d| cmd.addArg(b.fmt("-D{s}", .{d}));
    // Include search path for `#include "hdr.glsl"` (shared HDR transfer helpers).
    cmd.addPrefixedDirectoryArg("-I", b.path("shaders/common"));
    cmd.addArg("-o");
    const spv = cmd.addOutputFileArg(out);
    cmd.addFileArg(b.path(path));
    return spv;
}

const README_LINUX =
    \\tinrs-viewer: terrain INR renderer
    \\
    \\Requirements (system, not bundled):
    \\  - Linux x86_64, CPU with x86_64-v3 baseline (Intel Haswell 2013+ /
    \\    AMD Excavator 2015+, AVX2 + BMI2 + FMA, no AVX-512 required).
    \\  - glibc 2.31+ (Debian 11+, Ubuntu 20.04+, Fedora 32+, RPi OS Bullseye+).
    \\  - Vulkan loader + driver: libvulkan.so.1 plus an ICD (Mesa RADV / amdvlk /
    \\    NVIDIA / Intel ANV). Any distro with a working GPU has these.
    \\  - Wayland or X11 (auto-detected at runtime via dlopen).
    \\
    \\No SDL install required: SDL3 is statically linked into the binary.
    \\
    \\Run:
    \\  ./tinrs-viewer --model assets/planes --tile n47w122
    \\
    \\If --model is omitted it defaults to ./assets/planes (next to the binary).
    \\Use -p / --procedural to skip weight loading entirely (smoke test).
    \\Use --headless --profile for a no-window benchmark.
    \\Use --help for the full flag list.
    \\
;

const README_WINDOWS =
    \\tinrs-viewer: terrain INR renderer (Windows x86_64)
    \\
    \\Requirements:
    \\  - Windows 10 or newer, x86_64.
    \\  - GPU driver providing vulkan-1.dll plus a Vulkan ICD (any recent
    \\    NVIDIA / AMD / Intel driver from the last few years includes both).
    \\
    \\No SDL install required: SDL3 is statically linked into the binary.
    \\
    \\Run:
    \\  tinrs-viewer.exe --model assets\planes --tile n47w122
    \\
    \\If --model is omitted it defaults to .\assets\planes (next to the binary).
    \\Use -p / --procedural to skip weight loading entirely (smoke test).
    \\Use --headless --profile for a no-window benchmark.
    \\Use --help for the full flag list.
    \\
;

const README_MACOS =
    \\tinrs-viewer: terrain INR renderer (macOS)
    \\
    \\Requirements:
    \\  - macOS 11 (Big Sur) or newer. Apple Silicon or Intel.
    \\  - LunarG Vulkan SDK installed (provides libvulkan.1.dylib + MoltenVK
    \\    ICD). Download: https://vulkan.lunarg.com/sdk/home#mac
    \\    Run the SDK installer once; it sets VK_ICD_FILENAMES so the loader
    \\    finds MoltenVK without per-shell config.
    \\
    \\No SDL install required: SDL3 is statically linked into the binary.
    \\
    \\Gatekeeper: this binary is not codesigned. First-run will be blocked;
    \\either right-click the binary in Finder and choose Open, or run
    \\  xattr -d com.apple.quarantine ./tinrs-viewer
    \\once after extracting the tarball.
    \\
    \\Run:
    \\  ./tinrs-viewer --model assets/planes --tile n47w122
    \\
    \\If --model is omitted it defaults to ./assets/planes (next to the binary).
    \\Use -p / --procedural to skip weight loading entirely (smoke test).
    \\Use --headless --profile for a no-window benchmark.
    \\Use --help for the full flag list.
    \\
;

/// External package modules resolved once and reused across every artifact.
const ExtDeps = struct {
    vulkan: *std.Build.Module,
    vulkan_headers: *std.Build.Dependency,
    clap: *std.Build.Module,
    math: *std.Build.Module,
    ui: *std.Build.Module,
};

/// SPIR-V outputs of every shader the source code @imports via anonymous module.
const Shaders = struct {
    terrain_frag: std.Build.LazyPath,
    terrain_frag_debug: std.Build.LazyPath,
    clipmap_update: std.Build.LazyPath,
    probe_eval: std.Build.LazyPath,
    clipmap_vert: std.Build.LazyPath,
    clipmap_vert_debug: std.Build.LazyPath,
    sky_vert: std.Build.LazyPath,
    sky_frag: std.Build.LazyPath,
    aircraft_vert: std.Build.LazyPath,
    aircraft_frag: std.Build.LazyPath,
    text_vert: std.Build.LazyPath,
    text_frag: std.Build.LazyPath,
    numeric_vert: std.Build.LazyPath,
    gauge_vert: std.Build.LazyPath,
    gauge_frag: std.Build.LazyPath,
    ui_shape_vert: std.Build.LazyPath,
    ui_shape_frag: std.Build.LazyPath,
    ui_sdf_vert: std.Build.LazyPath,
    ui_sdf_frag: std.Build.LazyPath,
};

fn addShaderImports(mod: *std.Build.Module, shaders: Shaders) void {
    mod.addAnonymousImport("terrain_frag", .{ .root_source_file = shaders.terrain_frag });
    mod.addAnonymousImport("terrain_frag_debug", .{ .root_source_file = shaders.terrain_frag_debug });
    mod.addAnonymousImport("clipmap_update_shader", .{ .root_source_file = shaders.clipmap_update });
    mod.addAnonymousImport("probe_eval_shader", .{ .root_source_file = shaders.probe_eval });
    mod.addAnonymousImport("clipmap_vert", .{ .root_source_file = shaders.clipmap_vert });
    mod.addAnonymousImport("clipmap_vert_debug", .{ .root_source_file = shaders.clipmap_vert_debug });
    mod.addAnonymousImport("sky_vert", .{ .root_source_file = shaders.sky_vert });
    mod.addAnonymousImport("sky_frag", .{ .root_source_file = shaders.sky_frag });
    mod.addAnonymousImport("aircraft_vert", .{ .root_source_file = shaders.aircraft_vert });
    mod.addAnonymousImport("aircraft_frag", .{ .root_source_file = shaders.aircraft_frag });
    mod.addAnonymousImport("text_vert", .{ .root_source_file = shaders.text_vert });
    mod.addAnonymousImport("text_frag", .{ .root_source_file = shaders.text_frag });
    mod.addAnonymousImport("numeric_vert", .{ .root_source_file = shaders.numeric_vert });
    mod.addAnonymousImport("gauge_vert", .{ .root_source_file = shaders.gauge_vert });
    mod.addAnonymousImport("gauge_frag", .{ .root_source_file = shaders.gauge_frag });
    mod.addAnonymousImport("ui_shape_vert", .{ .root_source_file = shaders.ui_shape_vert });
    mod.addAnonymousImport("ui_shape_frag", .{ .root_source_file = shaders.ui_shape_frag });
    mod.addAnonymousImport("ui_sdf_vert", .{ .root_source_file = shaders.ui_sdf_vert });
    mod.addAnonymousImport("ui_sdf_frag", .{ .root_source_file = shaders.ui_sdf_frag });
}

/// Source files reach SDL3/Vulkan via vk_types.zig's `@cImport`. The module
/// containing that cImport (i.e. any module whose tree reaches vk_types.zig)
/// needs SDL3 linkage and libc.
const SdlSource = union(enum) {
    /// pkg-config "sdl3" (native dev builds).
    system,
    /// SDL3 Zig package compiled for this artifact's target (cross-compile / dist).
    static: *std.Build.Dependency,
};

const ModuleConfig = struct {
    root_source_file: std.Build.LazyPath,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: ?bool = null,
};

/// Build a module fully wired with: external deps, shader anon imports, SDL3
/// linkage, libc, build_options, and Vulkan-Headers (cross-compile path).
/// Used for both the viewer exe and every test target so any reachable source
/// file resolves the same import surface regardless of which root it's under.
fn createSourceModule(
    b: *std.Build,
    deps: ExtDeps,
    shaders: Shaders,
    sdl: SdlSource,
    build_options: *std.Build.Step.Options,
    cfg: ModuleConfig,
) *std.Build.Module {
    const mod = b.createModule(.{
        .root_source_file = cfg.root_source_file,
        .target = cfg.target,
        .optimize = cfg.optimize,
        .strip = cfg.strip,
        .imports = &.{
            .{ .name = "vulkan", .module = deps.vulkan },
            .{ .name = "clap", .module = deps.clap },
            .{ .name = "math", .module = deps.math },
            .{ .name = "ui", .module = deps.ui },
        },
    });
    mod.addOptions("build_options", build_options);
    addShaderImports(mod, shaders);
    switch (sdl) {
        .system => mod.linkSystemLibrary("sdl3", .{}),
        .static => |sdl_dep| {
            mod.linkLibrary(sdl_dep.artifact("SDL3"));
            // Cross-compile bypasses pkg-config; SDL_vulkan.h's transitive
            // header lookup needs Vulkan-Headers on the include path explicitly.
            mod.addSystemIncludePath(deps.vulkan_headers.path("include"));
        },
    }
    mod.link_libc = true;
    return mod;
}

const ViewerOpts = struct {
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    sdl: SdlSource,
    /// Comptime default for --model. Differs between dev (CWD-relative) and dist (exe-relative).
    default_model_path: []const u8,
    /// Drop DWARF + symbols. Dev keeps them for panics and `perf`; dist strips
    /// (~22 MB -> ~5 MB). Anyone debugging deeply should build from source.
    strip: bool = false,
};

fn buildViewerExe(
    b: *std.Build,
    deps: ExtDeps,
    shaders: Shaders,
    opts: ViewerOpts,
) *std.Build.Step.Compile {
    const build_options = b.addOptions();
    build_options.addOption([]const u8, "default_model_path", opts.default_model_path);

    const root_module = createSourceModule(b, deps, shaders, opts.sdl, build_options, .{
        .root_source_file = b.path("src/main.zig"),
        .target = opts.target,
        .optimize = opts.optimize,
        .strip = opts.strip,
    });

    return b.addExecutable(.{
        .name = "tinrs-viewer",
        .root_module = root_module,
    });
}

/// Resolve the dist target. If the user passed -Dtarget, honor it; otherwise
/// default to a portable Linux x86_64-v3 + glibc 2.31 baseline that runs on
/// Debian 11+, Ubuntu 20.04+, RPi OS Bullseye+, and any modern Intel/AMD chip.
fn resolveDistTarget(b: *std.Build, user_target: std.Build.ResolvedTarget) std.Build.ResolvedTarget {
    if (!user_target.query.isNative()) return user_target;
    return b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
        .abi = .gnu,
        .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
        .glibc_version = .{ .major = 2, .minor = 31, .patch = 0 },
    });
}

fn addDistStep(
    b: *std.Build,
    deps: ExtDeps,
    shaders: Shaders,
    user_target: std.Build.ResolvedTarget,
    user_optimize: std.builtin.OptimizeMode,
    with_assets: bool,
) void {
    const dist_target = resolveDistTarget(b, user_target);
    const target_info = dist_target.result;
    const is_windows = target_info.os.tag == .windows;
    const exe_suffix: []const u8 = if (is_windows) ".exe" else "";

    // ReleaseSafe: workload is GPU-bound so safety check overhead is noise,
    // and a panic beats silent UB for shipped bug reports. Honor -Doptimize.
    const dist_optimize: std.builtin.OptimizeMode = if (b.user_input_options.contains("optimize"))
        user_optimize
    else
        .ReleaseSafe;

    // SDL3 instantiated per-target so its glibc/CPU baseline matches the exe.
    const sdl_dep = b.dependency("sdl", .{ .target = dist_target, .optimize = dist_optimize });

    const dist_exe = buildViewerExe(b, deps, shaders, .{
        .target = dist_target,
        .optimize = dist_optimize,
        .sdl = .{ .static = sdl_dep },
        // Exe-relative so the tester runs `./tinrs-viewer` from the unpacked
        // tarball without passing --model.
        .default_model_path = "./assets/planes",
        .strip = true,
    });

    // Target tag in dir name so concurrent cross-builds don't clobber.
    const target_tag = b.fmt("{s}-{s}", .{ @tagName(target_info.cpu.arch), @tagName(target_info.os.tag) });
    const stage_dir = b.fmt("dist/tinrs-viewer-{s}", .{target_tag});

    const install_bin = b.addInstallFileWithDir(
        dist_exe.getEmittedBin(),
        .prefix,
        b.fmt("{s}/tinrs-viewer{s}", .{ stage_dir, exe_suffix }),
    );
    install_bin.step.dependOn(&dist_exe.step);

    const readme_text = switch (target_info.os.tag) {
        .windows => README_WINDOWS,
        .macos => README_MACOS,
        else => README_LINUX,
    };
    const readme_wf = b.addWriteFiles();
    const readme_path = readme_wf.add("README.txt", readme_text);
    const install_readme = b.addInstallFileWithDir(
        readme_path,
        .prefix,
        b.fmt("{s}/README.txt", .{stage_dir}),
    );

    // tar.gz: Windows 10+ has built-in tar support, and a single archive
    // format keeps the Linux build host's deps minimal.
    //
    // BUG (not yet fixed): the staged dir is read via `-C <install
    // path>` as a raw string, which is NOT a tracked input, so this step's cache
    // key is only the constant args. It therefore cache-hits forever after the
    // first run and `zig build dist` SHIPS A STALE TARBALL on any exe rebuild or
    // --with-assets change (the staged dir updates, the tarball does not). Until
    // fixed, regenerate by hand after a dist build:
    //   tar czf zig-out/dist/tinrs-viewer-<tag>.tar.gz -C zig-out/dist tinrs-viewer-<tag>
    // TODO: stage exe/README/assets into a b.addWriteFiles() and tar that
    // LazyPath dir via addDirectoryArg so the cache tracks real content; or, if
    // caching is not worth it, set `tar_cmd.has_side_effects = true` to always
    // re-tar (verify that is allowed alongside addOutputFileArg in this Zig).
    const tar_cmd = b.addSystemCommand(&.{ "tar", "czf" });
    const tarball_lp = tar_cmd.addOutputFileArg(b.fmt("tinrs-viewer-{s}.tar.gz", .{target_tag}));
    tar_cmd.addArgs(&.{
        "-C",                                       b.getInstallPath(.prefix, "dist"),
        b.fmt("tinrs-viewer-{s}", .{target_tag}),
    });
    tar_cmd.step.dependOn(&install_bin.step);
    tar_cmd.step.dependOn(&install_readme.step);

    // Bundle assets/planes/ (~29 MB): opt-in via -Dwith-assets=true.
    if (with_assets) {
        const install_assets = b.addInstallDirectory(.{
            .source_dir = b.path("../assets/planes"),
            .install_dir = .prefix,
            .install_subdir = b.fmt("{s}/assets/planes", .{stage_dir}),
            // reference.bin is a training/eval artifact; viewer only needs meta.json + weights.bin.
            .include_extensions = &.{ ".json", ".bin" },
            .exclude_extensions = &.{"reference.bin"},
        });
        tar_cmd.step.dependOn(&install_assets.step);
    }

    const install_tar = b.addInstallFileWithDir(tarball_lp, .prefix, b.fmt("dist/tinrs-viewer-{s}.tar.gz", .{target_tag}));
    const dist_step = b.step("dist", "Build a redistributable tarball at zig-out/dist/tinrs-viewer-<arch>-<os>.tar.gz");
    dist_step.dependOn(&install_tar.step);
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Vulkan bindings from registry
    const vulkan_headers = b.dependency("vulkan_headers", .{});
    const registry = vulkan_headers.path("registry/vk.xml");
    const vulkan = b.dependency("vulkan_zig", .{ .registry = registry }).module("vulkan-zig");

    // CLI argument parsing
    const clap = b.dependency("clap", .{}).module("clap");

    // Math is a leaf module (no project deps) exposed externally as `@import("math")`.
    const math_mod = b.createModule(.{
        .root_source_file = b.path("src/math.zig"),
        .target = target,
        .optimize = optimize,
    });

    // UI core: a pure module (std only) exposed as `@import("ui")`. Knows nothing
    // about Vulkan/SDL/app types; the render/ui_backend.zig consumes its DrawList.
    // Kept leaf so it's liftable into another project and GPU-free testable.
    const ui_mod = b.createModule(.{
        .root_source_file = b.path("src/ui/ui.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Compile shaders to SPIR-V. Two variants for terrain shaders: default omits
    // the frag_instance varying and the DrawEntries SSBO read in the fragment
    // stage; debug adds them back so the by_level / by_chunk / by_cull_state
    // overlays can light up.
    const shaders: Shaders = .{
        .terrain_frag = compileShader(b, "shaders/terrain/terrain.frag", "terrain.frag.spv"),
        .terrain_frag_debug = compileShaderWithDefines(b, "shaders/terrain/terrain.frag", "terrain.frag.debug.spv", &.{"DEBUG_OVERLAY"}),
        .clipmap_update = compileShader(b, "shaders/terrain/clipmap_update.comp", "clipmap_update.spv"),
        .probe_eval = compileShader(b, "shaders/terrain/probe_eval.comp", "probe_eval.spv"),
        .clipmap_vert = compileShader(b, "shaders/terrain/clipmap_terrain.vert", "clipmap_terrain.vert.spv"),
        .clipmap_vert_debug = compileShaderWithDefines(b, "shaders/terrain/clipmap_terrain.vert", "clipmap_terrain.vert.debug.spv", &.{"DEBUG_OVERLAY"}),
        .sky_vert = compileShader(b, "shaders/sky/sky.vert", "sky.vert.spv"),
        .sky_frag = compileShader(b, "shaders/sky/sky.frag", "sky.frag.spv"),
        .aircraft_vert = compileShader(b, "shaders/aircraft/aircraft.vert", "aircraft.vert.spv"),
        .aircraft_frag = compileShader(b, "shaders/aircraft/aircraft.frag", "aircraft.frag.spv"),
        .text_vert = compileShader(b, "shaders/hud/text.vert", "text.vert.spv"),
        .text_frag = compileShader(b, "shaders/hud/text.frag", "text.frag.spv"),
        .numeric_vert = compileShader(b, "shaders/hud/numeric.vert", "numeric.vert.spv"),
        .gauge_vert = compileShader(b, "shaders/hud/gauge.vert", "gauge.vert.spv"),
        .gauge_frag = compileShader(b, "shaders/hud/gauge.frag", "gauge.frag.spv"),
        .ui_shape_vert = compileShader(b, "shaders/ui/shape.vert", "ui_shape.vert.spv"),
        .ui_shape_frag = compileShader(b, "shaders/ui/shape.frag", "ui_shape.frag.spv"),
        .ui_sdf_vert = compileShader(b, "shaders/ui/sdf.vert", "ui_sdf.vert.spv"),
        .ui_sdf_frag = compileShader(b, "shaders/ui/sdf.frag", "ui_sdf.frag.spv"),
    };

    const deps: ExtDeps = .{
        .vulkan = vulkan,
        .vulkan_headers = vulkan_headers,
        .clap = clap,
        .math = math_mod,
        .ui = ui_mod,
    };

    // ---- Viewer (dev build) ----
    // Native build: link system SDL3 via pkg-config for fast iteration.
    // Cross-compile (-Dtarget=...): route through the SDL3 Zig package as a
    // compile-check without needing the target's SDL3 headers on the host.
    // For shipped tarballs use `zig build dist -Dtarget=...`.
    const dev_opts: ViewerOpts = .{
        .target = target,
        .optimize = optimize,
        .default_model_path = "../assets/planes",
        .sdl = if (target.query.isNative())
            .system
        else
            .{ .static = b.dependency("sdl", .{ .target = target, .optimize = optimize }) },
    };
    const viewer_exe = buildViewerExe(b, deps, shaders, dev_opts);
    b.installArtifact(viewer_exe);

    const run_cmd = b.addRunArtifact(viewer_exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the terrain viewer").dependOn(&run_cmd.step);

    // ---- Distribution tarball ----
    // `zig build dist` defaults to a portable Linux x86_64-v3 + glibc 2.31
    // binary. Override with -Dtarget for cross-compile:
    //   zig build dist -Dtarget=x86_64-windows-gnu     (Windows .exe)
    //   zig build dist -Dtarget=aarch64-linux-gnu      (RPi 5 / arm64 Linux)
    //   zig build dist -Dtarget=x86_64-linux-gnu       (current default)
    // Tarball goes to zig-out/dist/tinrs-viewer-<arch>-<os>.tar.gz so multiple
    // targets can coexist without clobbering.
    const with_assets = b.option(bool, "with-assets", "Bundle ../assets/planes into the dist tarball (~29 MB)") orelse false;
    addDistStep(b, deps, shaders, target, optimize, with_assets);

    // ---- Tests ----
    // Three test binaries: the two leaf MODULES (math, ui) plus ONE file-tree
    // binary rooted at src/tests.zig, which `_ = @import`s every subsystem
    // umbrella. A single compilation pulls in the whole file-import tree, so Zig
    // runs each test exactly once. (Per-umbrella addTest roots instead recompile
    // shared files like terrain/coords.zig into several binaries and run their
    // tests once per binary; see src/tests.zig and Ghostty's single-test-binary
    // pattern.) New files are picked up by adding them to their subdir umbrella;
    // a new top-level subsystem gets one `_ = @import` line in src/tests.zig.
    const test_step = b.step("test", "Run unit tests");
    const test_options = b.addOptions();
    test_options.addOption([]const u8, "default_model_path", "../assets/planes");
    // Focused dev runs (replaces the old per-subsystem test binaries):
    //   zig build test -Dtest-filter=coords
    const test_filters = b.option(
        []const []const u8,
        "test-filter",
        "Only run tests whose name matches one of the given filters",
    ) orelse &.{};

    // math and ui are leaf modules (created above) reached elsewhere via
    // @import("math") / @import("ui") MODULE imports, so their tests only run
    // when the module itself is the addTest root. Each gets its own pure
    // (Vulkan/SDL-free) test binary.
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = math_mod, .filters = test_filters })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = ui_mod, .filters = test_filters })).step);

    // The file-import tree: one binary, Vulkan/SDL-linked like the viewer exe.
    const tests_mod = createSourceModule(b, deps, shaders, .system, test_options, .{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = tests_mod, .filters = test_filters })).step);
}
