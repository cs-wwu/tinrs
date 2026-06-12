//! Best-effort host system info: CPU brand, core count, RAM, OS, DE/compositor.
//! Used by --debug startup logs. Detection is per-OS / per-arch and degrades
//! gracefully; any field that can't be probed becomes "unknown".

const std = @import("std");
const builtin = @import("builtin");

/// Caller-provided probes: values main.zig can read from SDL/OS without
/// dragging the SDL3 + Vulkan @cImport graph into this module. Keeps the file
/// pure Zig so it joins the no-deps test bucket.
pub const Probes = struct {
    cores: u32,
    ram_mb: u32,
    has_avx512f: bool,
    has_avx2: bool,
    has_avx: bool,
    has_sse42: bool,
    has_neon: bool,
    /// Result of SDL_GetCurrentVideoDriver; null if SDL_INIT_VIDEO is off.
    video_driver: ?[]const u8,
};

pub const SystemInfo = struct {
    os_line: []const u8,
    cpu_line: []const u8,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *SystemInfo) void {
        self.arena.deinit();
    }
};

const Ctx = struct {
    a: std.mem.Allocator,
    io: std.Io,
    env_map: *const std.process.Environ.Map,
    probes: Probes,
};

pub fn logDetected(
    gpa: std.mem.Allocator,
    io: std.Io,
    env_map: *const std.process.Environ.Map,
    probes: Probes,
) void {
    var si = detect(gpa, io, env_map, probes) catch |err| {
        std.log.debug("sysinfo unavailable: {}", .{err});
        return;
    };
    defer si.deinit();
    std.log.debug("System: {s}", .{si.os_line});
    std.log.debug("CPU: {s}", .{si.cpu_line});
}

pub fn detect(
    gpa: std.mem.Allocator,
    io: std.Io,
    env_map: *const std.process.Environ.Map,
    probes: Probes,
) !SystemInfo {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const ctx: Ctx = .{
        .a = arena.allocator(),
        .io = io,
        .env_map = env_map,
        .probes = probes,
    };

    return .{
        .os_line = try buildOsLine(ctx),
        .cpu_line = try buildCpuLine(ctx),
        .arena = arena,
    };
}

fn buildOsLine(ctx: Ctx) ![]const u8 {
    const os_str = try osDescription(ctx);
    const driver = try videoDriverLabel(ctx);
    const de = desktopEnv(ctx);
    if (de) |d| {
        return std.fmt.allocPrint(ctx.a, "{s} | {s} | {s}", .{ os_str, driver, d });
    }
    return std.fmt.allocPrint(ctx.a, "{s} | {s}", .{ os_str, driver });
}

fn osDescription(ctx: Ctx) ![]const u8 {
    return switch (builtin.os.tag) {
        .linux => linuxOsDescription(ctx),
        .windows => windowsOsDescription(ctx.a),
        .macos => macosOsDescription(ctx.a),
        else => "unknown OS",
    };
}

fn linuxOsDescription(ctx: Ctx) ![]const u8 {
    const uts = std.posix.uname();
    const release = std.mem.sliceTo(&uts.release, 0);

    var buf: [4096]u8 = undefined;
    const content = std.Io.Dir.cwd().readFile(ctx.io, "/etc/os-release", &buf) catch "";
    const distro = parsePrettyName(content);

    if (distro.len > 0) {
        return std.fmt.allocPrint(ctx.a, "Linux {s} ({s})", .{ release, distro });
    }
    return std.fmt.allocPrint(ctx.a, "Linux {s}", .{release});
}

fn macosOsDescription(a: std.mem.Allocator) ![]const u8 {
    if (builtin.os.tag != .macos) return "macOS";
    var buf: [32]u8 = undefined;
    if (darwinSysctlString("kern.osproductversion", &buf)) |ver| {
        return std.fmt.allocPrint(a, "macOS {s}", .{ver});
    }
    return "macOS";
}

/// sysctlbyname wrapper for null-terminated string keys. macOS-only; the
/// early return keeps the extern reference dead on other targets, so Linux
/// and Windows builds don't try to resolve the symbol at link time.
fn darwinSysctlString(name: [*:0]const u8, buf: []u8) ?[]const u8 {
    if (builtin.os.tag != .macos) return null;
    const sysctl = struct {
        extern "c" fn sysctlbyname(
            name: [*:0]const u8,
            oldp: ?*anyopaque,
            oldlenp: *usize,
            newp: ?*anyopaque,
            newlen: usize,
        ) c_int;
    };
    var len: usize = buf.len;
    if (sysctl.sysctlbyname(name, buf.ptr, &len, null, 0) != 0) return null;
    if (len == 0) return null;
    return std.mem.sliceTo(buf[0..len], 0);
}

fn windowsOsDescription(a: std.mem.Allocator) ![]const u8 {
    if (builtin.os.tag != .windows) return "Windows";

    // RtlGetVersion returns the actual OS version (vs GetVersionEx, which lies
    // for backward-compat with apps that lack a manifest declaring Windows-10
    // support). Declared inline to avoid relying on stdlib's windows surface,
    // which has churned across recent Zig releases.
    const OSVERSIONINFOW = extern struct {
        dwOSVersionInfoSize: u32,
        dwMajorVersion: u32,
        dwMinorVersion: u32,
        dwBuildNumber: u32,
        dwPlatformId: u32,
        szCSDVersion: [128]u16,
    };
    const ntdll = struct {
        extern "ntdll" fn RtlGetVersion(*OSVERSIONINFOW) callconv(.winapi) i32;
    };

    var info: OSVERSIONINFOW = std.mem.zeroes(OSVERSIONINFOW);
    info.dwOSVersionInfoSize = @sizeOf(OSVERSIONINFOW);
    if (ntdll.RtlGetVersion(&info) == 0) {
        // Microsoft kept the "10" major version for Windows 11; the only way to
        // tell them apart is the build number (22000+ => Windows 11).
        const product: []const u8 = if (info.dwMajorVersion >= 10 and info.dwBuildNumber >= 22000) "11" else "10";
        return std.fmt.allocPrint(a, "Windows {s} (build {d})", .{ product, info.dwBuildNumber });
    }
    return "Windows";
}

fn parsePrettyName(content: []const u8) []const u8 {
    var iter = std.mem.splitScalar(u8, content, '\n');
    while (iter.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        const prefix = "PRETTY_NAME=";
        if (!std.mem.startsWith(u8, trimmed, prefix)) continue;
        var val = trimmed[prefix.len..];
        if (val.len >= 2 and val[0] == '"' and val[val.len - 1] == '"') {
            val = val[1 .. val.len - 1];
        }
        return val;
    }
    return "";
}

fn videoDriverLabel(ctx: Ctx) ![]const u8 {
    if (ctx.probes.video_driver) |drv| if (drv.len > 0) return try ctx.a.dupe(u8, drv);
    // SDL not initialized (e.g. headless without autotune). Fall back to the
    // session-type env var on Linux, otherwise return a placeholder.
    if (builtin.os.tag == .linux) {
        if (ctx.env_map.get("XDG_SESSION_TYPE")) |s| {
            if (s.len > 0) return try ctx.a.dupe(u8, s);
        }
    }
    return "?";
}

fn desktopEnv(ctx: Ctx) ?[]const u8 {
    if (builtin.os.tag != .linux) return null;
    const v = ctx.env_map.get("XDG_CURRENT_DESKTOP") orelse return null;
    if (v.len == 0) return null;
    return v;
}

fn buildCpuLine(ctx: Ctx) ![]const u8 {
    const brand = try cpuBrand(ctx);
    const ram_gb = @as(f64, @floatFromInt(ctx.probes.ram_mb)) / 1024.0;
    const isa = isaFeatures(ctx);

    if (isa.len > 0) {
        return std.fmt.allocPrint(ctx.a, "{s} | {d} cores | {d:.0} GB RAM | {s}", .{ brand, ctx.probes.cores, ram_gb, isa });
    }
    return std.fmt.allocPrint(ctx.a, "{s} | {d} cores | {d:.0} GB RAM", .{ brand, ctx.probes.cores, ram_gb });
}

fn cpuBrand(ctx: Ctx) ![]const u8 {
    if (builtin.os.tag == .macos) return darwinCpuBrand(ctx.a);
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => x86CpuBrand(ctx.a),
        .aarch64 => aarch64CpuBrand(ctx),
        else => "unknown CPU",
    };
}

/// `machdep.cpu.brand_string` returns "Apple M3 Pro" on Apple Silicon and the
/// Intel brand string on Intel Macs; single path covers both.
fn darwinCpuBrand(a: std.mem.Allocator) ![]const u8 {
    if (builtin.os.tag != .macos) return "Mac";
    var buf: [128]u8 = undefined;
    if (darwinSysctlString("machdep.cpu.brand_string", &buf)) |brand| {
        return try a.dupe(u8, brand);
    }
    return if (builtin.cpu.arch == .aarch64) "Apple Silicon" else "Mac";
}

fn x86CpuBrand(a: std.mem.Allocator) ![]const u8 {
    // 0x80000000.eax reports the highest extended leaf. Brand string lives in
    // 0x80000002-04 (16 ASCII bytes per leaf packed across EAX/EBX/ECX/EDX,
    // little-endian). Any x86 CPU from the past ~25 years supports these.
    const max_ext = cpuid(0x80000000, 0)[0];
    if (max_ext < 0x80000004) return "unknown CPU";

    var raw: [48]u8 = undefined;
    inline for (0..3) |i| {
        const regs = cpuid(0x80000002 + @as(u32, @intCast(i)), 0);
        std.mem.writeInt(u32, raw[i * 16 + 0 ..][0..4], regs[0], .little);
        std.mem.writeInt(u32, raw[i * 16 + 4 ..][0..4], regs[1], .little);
        std.mem.writeInt(u32, raw[i * 16 + 8 ..][0..4], regs[2], .little);
        std.mem.writeInt(u32, raw[i * 16 + 12 ..][0..4], regs[3], .little);
    }
    return try a.dupe(u8, trimBrand(&raw));
}

fn cpuid(leaf: u32, sub: u32) [4]u32 {
    var eax: u32 = undefined;
    var ebx: u32 = undefined;
    var ecx: u32 = undefined;
    var edx: u32 = undefined;
    asm volatile ("cpuid"
        : [_] "={eax}" (eax),
          [_] "={ebx}" (ebx),
          [_] "={ecx}" (ecx),
          [_] "={edx}" (edx),
        : [_] "{eax}" (leaf),
          [_] "{ecx}" (sub),
    );
    return .{ eax, ebx, ecx, edx };
}

/// Intel CPUs left-pad the brand string with spaces; AMD typically right-pads.
fn trimBrand(raw: []const u8) []const u8 {
    const nul_end = std.mem.indexOfScalar(u8, raw, 0) orelse raw.len;
    return std.mem.trim(u8, raw[0..nul_end], " \t");
}

fn aarch64CpuBrand(ctx: Ctx) ![]const u8 {
    if (builtin.os.tag != .linux) return "ARM";

    // Device-tree model is the friendliest source on RPi-class boards;
    // contains "Raspberry Pi 5 Model B Rev 1.0" verbatim.
    var dt_buf: [256]u8 = undefined;
    if (std.Io.Dir.cwd().readFile(ctx.io, "/sys/firmware/devicetree/base/model", &dt_buf)) |bytes| {
        const trimmed = std.mem.trim(u8, std.mem.sliceTo(bytes, 0), " \t\r\n");
        if (trimmed.len > 0) return try ctx.a.dupe(u8, trimmed);
    } else |_| {}

    var pbuf: [4096]u8 = undefined;
    const content = std.Io.Dir.cwd().readFile(ctx.io, "/proc/cpuinfo", &pbuf) catch return "ARM";
    const model = parseCpuinfoModel(content);
    if (model.len > 0) return try ctx.a.dupe(u8, model);
    return "ARM";
}

/// Prefers `Model:` (e.g. "Raspberry Pi 5 Model B Rev 1.0") over `Hardware:`
/// (e.g. "BCM2712"); `Hardware` typically lists first but the SoC code is
/// less useful than the board name.
fn parseCpuinfoModel(content: []const u8) []const u8 {
    var hardware: []const u8 = "";
    var iter = std.mem.splitScalar(u8, content, '\n');
    while (iter.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = std.mem.trim(u8, line[0..colon], " \t");
        const val = std.mem.trim(u8, line[colon + 1 ..], " \t\r");
        if (std.mem.eql(u8, key, "Model")) return val;
        if (std.mem.eql(u8, key, "Hardware") and hardware.len == 0) hardware = val;
    }
    return hardware;
}

/// Highest supported SIMD tier only; AVX-512 includes AVX2 includes AVX includes SSE4.2 strictly,
/// so listing every level just clutters the line.
fn isaFeatures(ctx: Ctx) []const u8 {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => if (ctx.probes.has_avx512f) "AVX-512" //
            else if (ctx.probes.has_avx2) "AVX2" //
            else if (ctx.probes.has_avx) "AVX" //
            else if (ctx.probes.has_sse42) "SSE4.2" //
            else "",
        .aarch64, .arm => if (ctx.probes.has_neon) "NEON" else "",
        else => "",
    };
}

test "parsePrettyName extracts quoted distro name" {
    const content =
        \\NAME="Fedora Linux"
        \\VERSION="44 (Workstation Edition)"
        \\PRETTY_NAME="Fedora Linux 44 (Workstation Edition)"
        \\ID=fedora
        \\
    ;
    try std.testing.expectEqualStrings("Fedora Linux 44 (Workstation Edition)", parsePrettyName(content));
}

test "parsePrettyName handles unquoted value" {
    try std.testing.expectEqualStrings("Arch Linux", parsePrettyName("PRETTY_NAME=Arch Linux\n"));
}

test "parsePrettyName returns empty when missing" {
    try std.testing.expectEqualStrings("", parsePrettyName("NAME=foo\n"));
}

test "parsePrettyName skips comments and blanks" {
    const content =
        \\# comment
        \\
        \\ID=foo
        \\PRETTY_NAME="Bar 1.0"
        \\
    ;
    try std.testing.expectEqualStrings("Bar 1.0", parsePrettyName(content));
}

test "trimBrand strips trailing nulls" {
    var buf = [_]u8{0} ** 48;
    @memcpy(buf[0..18], "AMD Ryzen 7 9800X3");
    try std.testing.expectEqualStrings("AMD Ryzen 7 9800X3", trimBrand(&buf));
}

test "trimBrand strips Intel-style leading spaces" {
    var buf = [_]u8{0} ** 48;
    @memcpy(buf[0..14], "  Intel Xeon  ");
    try std.testing.expectEqualStrings("Intel Xeon", trimBrand(&buf));
}

test "parseCpuinfoModel finds Model line" {
    const content = "processor\t: 0\nmodel name\t: Cortex-A76\n\nHardware\t: BCM2712\nModel\t\t: Raspberry Pi 5 Model B Rev 1.0\n";
    try std.testing.expectEqualStrings("Raspberry Pi 5 Model B Rev 1.0", parseCpuinfoModel(content));
}

test "parseCpuinfoModel falls back to Hardware" {
    try std.testing.expectEqualStrings("BCM2712", parseCpuinfoModel("Hardware\t: BCM2712\n"));
}

test "parseCpuinfoModel returns empty when neither field present" {
    try std.testing.expectEqualStrings("", parseCpuinfoModel("processor\t: 0\nmodel name\t: foo\n"));
}
