//! One-time Vulkan setup: instance, surface, physical device, logical device, queues,
//! command pool/buffers, timestamp query pool. Owns immutable per-process Vulkan state
//! that the Renderer composes for its frame-cycling work.

const std = @import("std");
const builtin = @import("builtin");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const c = vkt.c;
const swapchain_mod = @import("swapchain.zig");
const msaa_mod = @import("msaa.zig");
const ValidateLayers = @import("../config/options.zig").ValidateLayers;

pub const MAX_FRAMES_IN_FLIGHT = 2;

// Vulkan loader is opened at runtime so the binary has zero link-time
// dependency on libvulkan, portable across distros and OSes. std.DynLib in
// Zig 0.16 is `@compileError("unsupported platform")` on Windows, so we route
// through a per-OS shim: LoadLibraryA/GetProcAddress on Windows, std.DynLib
// elsewhere.
const VkLoader = if (builtin.os.tag == .windows) struct {
    const windows = std.os.windows;
    extern "kernel32" fn LoadLibraryA(lpLibFileName: [*:0]const u8) callconv(.winapi) ?windows.HMODULE;
    extern "kernel32" fn GetProcAddress(hModule: windows.HMODULE, lpProcName: [*:0]const u8) callconv(.winapi) ?windows.FARPROC;

    handle: windows.HMODULE,

    fn open(name: [:0]const u8) !VkLoader {
        const h = LoadLibraryA(name.ptr) orelse return error.FileNotFound;
        return .{ .handle = h };
    }

    fn lookup(self: *VkLoader, comptime T: type, name: [:0]const u8) ?T {
        const sym = GetProcAddress(self.handle, name.ptr) orelse return null;
        return @ptrCast(@alignCast(sym));
    }
} else struct {
    inner: std.DynLib,

    fn open(name: [:0]const u8) !VkLoader {
        return .{ .inner = try std.DynLib.open(name) };
    }

    fn lookup(self: *VkLoader, comptime T: type, name: [:0]const u8) ?T {
        return self.inner.lookup(T, name);
    }
};

var vk_loader_lib: VkLoader = undefined;
var vkGetInstanceProcAddr: vk.PfnGetInstanceProcAddr = undefined;

fn loadVulkanLoader() !void {
    const lib_name = switch (builtin.os.tag) {
        .windows => "vulkan-1.dll",
        .macos, .ios, .tvos, .watchos => "libvulkan.1.dylib",
        else => "libvulkan.so.1",
    };
    vk_loader_lib = VkLoader.open(lib_name) catch |err| {
        std.log.err("Failed to load Vulkan loader '{s}': {s}", .{ lib_name, @errorName(err) });
        return error.VulkanLoaderNotFound;
    };
    vkGetInstanceProcAddr = vk_loader_lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse {
        std.log.err("Vulkan loader '{s}' is missing vkGetInstanceProcAddr symbol", .{lib_name});
        return error.VulkanLoaderInvalid;
    };
}

const DeviceTypeInfo = struct { weight: u64, name: []const u8 };

fn deviceTypeInfo(t: vk.PhysicalDeviceType) DeviceTypeInfo {
    return switch (t) {
        .discrete_gpu => .{ .weight = 4, .name = "discrete" },
        .integrated_gpu => .{ .weight = 3, .name = "integrated" },
        .virtual_gpu => .{ .weight = 2, .name = "virtual" },
        .cpu => .{ .weight = 1, .name = "cpu" },
        else => .{ .weight = 0, .name = "other" },
    };
}

fn hasExtension(props: []const vk.ExtensionProperties, name: []const u8) bool {
    for (props) |prop| {
        const prop_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&prop.extension_name)), 0);
        if (std.mem.eql(u8, prop_name, name)) return true;
    }
    return false;
}

/// How the renderer obtains its surface. Windowed presents to an SDL window;
/// headless creates a VK_EXT_headless_surface at the given extent (no display).
pub const SurfaceMode = union(enum) {
    windowed: *c.SDL_Window,
    headless: vk.Extent2D,
};

/// Borrowed Vulkan handles needed by subsystems (clipmap, sky, text) to
/// allocate buffers and submit one-shot transfer commands. All fields point
/// at Renderer-owned resources; do not destroy them here.
/// Mutable per-pass state (render_pass, samples) is passed separately because
/// the V-key MSAA toggle rebuilds those at runtime.
pub const GpuContext = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    queue: vk.Queue,
    cmd_pool: vk.CommandPool,
    vram_mb: u64,
};

pub const InitOptions = struct {
    /// Vulkan validation layers. core = standard layer; sync/bp are extras
    /// added via VkValidationFeaturesEXT in the instance pNext chain.
    validate: ValidateLayers = .{},
    vsync: bool = true,
    /// Allocate timestamp query pool for GPU phase timing.
    bench_enabled: bool = false,
    /// Index into the suitable-GPU candidate list (use --debug to enumerate).
    gpu_override: ?u32 = null,
    enable_hdr: bool = false,
    /// 1 = MSAA off; pass 2/4/8 for multisampling (clamped to GPU support).
    msaa_request: u32 = 1,
    /// When true, suppresses the consolidated single-line GPU summary;
    /// caller is rendering the verbose enumeration which already shows the choice.
    verbose: bool = false,
};

pub const VulkanContext = struct {
    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,

    instance: vk.Instance,
    surface: vk.SurfaceKHR,
    pdev: vk.PhysicalDevice,
    device: vk.Device,
    device_name: [256]u8,

    graphics_family: u32,
    present_family: u32,
    graphics_queue: vk.Queue,
    present_queue: vk.Queue,

    mem_props: vk.PhysicalDeviceMemoryProperties,
    vram_mb: u64,
    cmd_pool: vk.CommandPool,
    cmd_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer,

    /// MSAA sample count chosen at init (clamped to GPU support).
    samples: vk.SampleCountFlags,
    /// Initial surface format chosen at init.
    surface_format: swapchain_mod.SurfaceFormatChoice,
    /// Initial framebuffer extent (window size or headless extent).
    initial_extent: vk.Extent2D,

    hdr_capable: bool, // pure capability: the VK_EXT_swapchain_colorspace extension is present

    bench_enabled: bool,
    /// 8 slots = MAX_FRAMES_IN_FLIGHT x 4 timestamps (compute_start, compute_end, gfx_start, gfx_end).
    /// .null_handle when bench is disabled or timestamps unsupported on the graphics queue.
    query_pool: vk.QueryPool,
    timestamp_period_ns: f32,
    timestamp_valid_bits: u32,

    window: ?*c.SDL_Window, // null = headless mode (VK_EXT_headless_surface, no resize)

    allocator: std.mem.Allocator,
    io: std.Io,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, surface_mode: SurfaceMode, opts: InitOptions) !VulkanContext {
        const window: ?*c.SDL_Window = switch (surface_mode) {
            .windowed => |w| w,
            .headless => null,
        };

        try loadVulkanLoader();

        // ---- Vulkan instance ----
        const MAX_INSTANCE_EXTS = 8;
        var ext_buf: [MAX_INSTANCE_EXTS][*:0]const u8 = undefined;
        var ext_count: usize = 0;
        switch (surface_mode) {
            .windowed => {
                var sdl_ext_count: u32 = 0;
                const sdl_extensions = c.SDL_Vulkan_GetInstanceExtensions(&sdl_ext_count);
                if (sdl_extensions == null) {
                    std.log.err("SDL failed to provide required Vulkan extensions: {s}", .{c.SDL_GetError()});
                    return error.NoVulkanExtensions;
                }
                if (sdl_ext_count > MAX_INSTANCE_EXTS) return error.TooManyInstanceExtensions;
                for (0..sdl_ext_count) |i| {
                    ext_buf[ext_count] = sdl_extensions[i];
                    ext_count += 1;
                }
            },
            .headless => {
                ext_buf[0] = "VK_KHR_surface";
                ext_count = 1;
            },
        }

        const vkb = vk.BaseWrapper.load(vkGetInstanceProcAddr);

        const want_layer_settings = opts.validate.sync or opts.validate.bp;
        var has_colorspace_ext = false;
        var has_layer_settings_ext = false;
        var has_portability_ext = false;
        var has_headless_ext = false;
        const ext_props = vkb.enumerateInstanceExtensionPropertiesAlloc(null, allocator) catch null;
        if (ext_props) |props| {
            defer allocator.free(props);
            has_colorspace_ext = hasExtension(props, "VK_EXT_swapchain_colorspace");
            has_layer_settings_ext = hasExtension(props, "VK_EXT_layer_settings");
            has_portability_ext = hasExtension(props, "VK_KHR_portability_enumeration");
            has_headless_ext = hasExtension(props, "VK_EXT_headless_surface");
        }
        // Enable the colorspace extension whenever the loader exposes it, regardless
        // of the initial HDR request: the settings menu toggles HDR at runtime, which
        // needs the extension present even if we launched in SDR (e.g. via --no-hdr).
        // Enabling an unused extension is harmless. Bounds-guard like the adds below.
        if (has_colorspace_ext) {
            if (ext_count >= MAX_INSTANCE_EXTS) return error.TooManyInstanceExtensions;
            ext_buf[ext_count] = "VK_EXT_swapchain_colorspace";
            ext_count += 1;
        }
        // Headless mode needs VK_EXT_headless_surface. Probe rather than blindly
        // request: some loaders (older NVIDIA on Linux, some MoltenVK builds)
        // crash or fail instance creation if asked for it without support.
        if (std.meta.activeTag(surface_mode) == .headless) {
            if (!has_headless_ext) {
                std.log.err("Headless mode requires VK_EXT_headless_surface, but the Vulkan loader does not expose it. Update the Vulkan SDK (MoltenVK 1.2.5+ on macOS).", .{});
                return error.HeadlessSurfaceUnsupported;
            }
            if (ext_count >= MAX_INSTANCE_EXTS) return error.TooManyInstanceExtensions;
            ext_buf[ext_count] = "VK_EXT_headless_surface";
            ext_count += 1;
        }
        // MoltenVK enumerates zero physical devices unless the portability
        // enumeration extension is enabled and the matching create-info flag
        // is set. Harmless on Linux/Windows when the extension isn't exposed.
        if (has_portability_ext) {
            if (ext_count >= MAX_INSTANCE_EXTS) return error.TooManyInstanceExtensions;
            ext_buf[ext_count] = "VK_KHR_portability_enumeration";
            ext_count += 1;
        }

        const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        const want_layer = opts.validate.any();

        // The Khronos validation layer's default `report_flags` is `error` only,
        // so sync hazards (warn severity) and best-practices messages (perf) get
        // dropped unless we override. Use VK_EXT_layer_settings to mirror the
        // settings-file flow we know works.
        const layer_name: [*:0]const u8 = "VK_LAYER_KHRONOS_validation";
        const true_val: vk.Bool32 = .true;
        const dup_limit: u32 = 5;
        const report_flag_strs = [_][*:0]const u8{ "error", "warn", "perf" };
        var settings_buf: [4]vk.LayerSettingEXT = undefined;
        var settings_count: u32 = 0;
        if (want_layer_settings and has_layer_settings_ext) {
            if (opts.validate.sync) {
                settings_buf[settings_count] = .{
                    .p_layer_name = layer_name,
                    .p_setting_name = "validate_sync",
                    .type = .bool32_ext,
                    .value_count = 1,
                    .p_values = &true_val,
                };
                settings_count += 1;
            }
            if (opts.validate.bp) {
                settings_buf[settings_count] = .{
                    .p_layer_name = layer_name,
                    .p_setting_name = "validate_best_practices",
                    .type = .bool32_ext,
                    .value_count = 1,
                    .p_values = &true_val,
                };
                settings_count += 1;
            }
            settings_buf[settings_count] = .{
                .p_layer_name = layer_name,
                .p_setting_name = "report_flags",
                .type = .string_ext,
                .value_count = report_flag_strs.len,
                .p_values = &report_flag_strs,
            };
            settings_count += 1;
            settings_buf[settings_count] = .{
                .p_layer_name = layer_name,
                .p_setting_name = "duplicate_message_limit",
                .type = .uint32_ext,
                .value_count = 1,
                .p_values = &dup_limit,
            };
            settings_count += 1;

            if (ext_count >= MAX_INSTANCE_EXTS) return error.TooManyInstanceExtensions;
            ext_buf[ext_count] = "VK_EXT_layer_settings";
            ext_count += 1;
        } else if (want_layer_settings) {
            std.log.warn("VK_EXT_layer_settings unavailable; sync/bp validation extras ignored", .{});
        }
        const layer_settings: vk.LayerSettingsCreateInfoEXT = .{
            .setting_count = settings_count,
            .p_settings = &settings_buf,
        };

        const instance = try vkb.createInstance(&.{
            .p_next = if (settings_count > 0) &layer_settings else null,
            .flags = .{ .enumerate_portability_bit_khr = has_portability_ext },
            .p_application_info = &.{
                .p_application_name = "tinrs-viewer",
                .application_version = 0,
                .engine_version = 0,
                .api_version = vk.API_VERSION_1_2.toU32(),
            },
            .enabled_layer_count = if (want_layer) validation_layers.len else 0,
            .pp_enabled_layer_names = &validation_layers,
            .enabled_extension_count = @intCast(ext_count),
            .pp_enabled_extension_names = &ext_buf,
        }, null);

        const vki = vk.InstanceWrapper.load(instance, vkGetInstanceProcAddr);
        errdefer vki.destroyInstance(instance, null);

        // ---- Surface ----
        var surface: vk.SurfaceKHR = undefined;
        switch (surface_mode) {
            .windowed => |w| {
                var vk_surface: c.VkSurfaceKHR = undefined;
                if (!c.SDL_Vulkan_CreateSurface(w, @ptrFromInt(@intFromEnum(instance)), null, &vk_surface)) {
                    std.log.err("Failed to create Vulkan surface: {s}", .{c.SDL_GetError()});
                    return error.SurfaceCreationFailed;
                }
                surface = @enumFromInt(@intFromPtr(vk_surface));
            },
            .headless => surface = try vki.createHeadlessSurfaceEXT(instance, &.{}, null),
        }
        errdefer vki.destroySurfaceKHR(instance, surface, null);

        // ---- Physical device + queue families ----
        const pdevs = try vki.enumeratePhysicalDevicesAlloc(instance, allocator);
        defer allocator.free(pdevs);

        if (pdevs.len == 0) {
            std.log.err("No Vulkan physical devices found", .{});
            return error.NoPhysicalDevice;
        }

        const Candidate = struct {
            pdev: vk.PhysicalDevice,
            props: vk.PhysicalDeviceProperties,
            mem_props: vk.PhysicalDeviceMemoryProperties,
            graphics_family: u32,
            present_family: u32,
            type_str: []const u8,
            vram_mb: u64,
            score: u64,
        };

        var candidates: std.ArrayList(Candidate) = .empty;
        defer candidates.deinit(allocator);

        for (pdevs) |pd| {
            const queue_families = try vki.getPhysicalDeviceQueueFamilyPropertiesAlloc(pd, allocator);
            defer allocator.free(queue_families);

            var gfx: ?u32 = null;
            var prs: ?u32 = null;
            var cmp: ?u32 = null;

            for (queue_families, 0..) |qf, i| {
                const idx: u32 = @intCast(i);
                if (qf.queue_flags.graphics_bit) gfx = idx;
                if (qf.queue_flags.compute_bit) cmp = idx;
                if (vki.getPhysicalDeviceSurfaceSupportKHR(pd, idx, surface)) |supported| {
                    if (supported == .true) prs = idx;
                } else |_| {}
            }

            if (gfx == null or prs == null or cmp == null) continue;

            const pd_props = vki.getPhysicalDeviceProperties(pd);
            const pd_mem = vki.getPhysicalDeviceMemoryProperties(pd);

            var pd_vram_bytes: u64 = 0;
            for (0..pd_mem.memory_heap_count) |i| {
                if (pd_mem.memory_heaps[i].flags.device_local_bit)
                    pd_vram_bytes += pd_mem.memory_heaps[i].size;
            }
            const vram_mb = pd_vram_bytes / (1024 * 1024);

            const ti = deviceTypeInfo(pd_props.device_type);

            try candidates.append(allocator, .{
                .pdev = pd,
                .props = pd_props,
                .mem_props = pd_mem,
                .graphics_family = gfx.?,
                .present_family = prs.?,
                .type_str = ti.name,
                .vram_mb = vram_mb,
                .score = ti.weight * 10_000_000 + vram_mb,
            });
        }

        if (candidates.items.len == 0) {
            std.log.err("No suitable GPU found (need graphics + present + compute)", .{});
            return error.NoSuitableDevice;
        }

        const chosen_idx: usize = if (opts.gpu_override) |n| blk: {
            if (n >= candidates.items.len) {
                std.log.err("--gpu {d} out of range; only {d} suitable candidate(s) found", .{ n, candidates.items.len });
                return error.InvalidGpuIndex;
            }
            break :blk n;
        } else blk: {
            var best: usize = 0;
            for (candidates.items[1..], 1..) |cand, i| {
                if (cand.score > candidates.items[best].score) best = i;
            }
            break :blk best;
        };

        const chosen = candidates.items[chosen_idx];
        const pdev = chosen.pdev;
        const graphics_family = chosen.graphics_family;
        const present_family = chosen.present_family;
        const props = chosen.props;
        const mem_props = chosen.mem_props;

        // ---- Device info ----
        std.log.debug("GPUs:", .{});
        for (candidates.items, 0..) |cand, i| {
            const sel: u8 = if (i == chosen_idx) '*' else ' ';
            std.log.debug("  [{d}]{c} {s} ({s}, {d} MB) score={d}", .{
                i,
                sel,
                @as([*:0]const u8, @ptrCast(&cand.props.device_name)),
                cand.type_str,
                cand.vram_mb,
                cand.score,
            });
        }
        std.log.debug("       API {d}.{d}.{d}, push constants {d}B, compute workgroup {d}, memory types {d}", .{
            props.api_version >> 22,
            (props.api_version >> 12) & 0x3FF,
            props.api_version & 0xFFF,
            props.limits.max_push_constants_size,
            props.limits.max_compute_work_group_invocations,
            mem_props.memory_type_count,
        });
        if (!opts.verbose) {
            std.log.info("GPU: {s} ({s}, {d} MB, API {d}.{d}.{d})", .{
                @as([*:0]const u8, @ptrCast(&props.device_name)),
                chosen.type_str,
                chosen.vram_mb,
                props.api_version >> 22,
                (props.api_version >> 12) & 0x3FF,
                props.api_version & 0xFFF,
            });
        }

        // ---- Logical device ----
        var unique_families: [3]u32 = undefined;
        var unique_count: u32 = 0;
        for ([_]u32{ graphics_family, present_family }) |fam| {
            var found = false;
            for (unique_families[0..unique_count]) |existing| {
                if (existing == fam) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique_families[unique_count] = fam;
                unique_count += 1;
            }
        }

        const queue_priority = [_]f32{1.0};
        var queue_create_infos: [3]vk.DeviceQueueCreateInfo = undefined;
        for (0..unique_count) |i| {
            queue_create_infos[i] = .{
                .queue_family_index = unique_families[i],
                .queue_count = 1,
                .p_queue_priorities = &queue_priority,
            };
        }

        // Spec requires VK_KHR_portability_subset to be enabled if the device
        // exposes it (MoltenVK does). Probe and append.
        var dev_ext_buf: [2][*:0]const u8 = .{ "VK_KHR_swapchain", undefined };
        var dev_ext_count: u32 = 1;
        const dev_ext_props = vki.enumerateDeviceExtensionPropertiesAlloc(pdev, null, allocator) catch null;
        if (dev_ext_props) |props_d| {
            defer allocator.free(props_d);
            if (hasExtension(props_d, "VK_KHR_portability_subset")) {
                dev_ext_buf[dev_ext_count] = "VK_KHR_portability_subset";
                dev_ext_count += 1;
            }
        }
        const device_extensions = dev_ext_buf[0..dev_ext_count];

        // Wireframe debug mode (F2) needs polygon_mode=line. Clipmap radial
        // boundary uses gl_ClipDistance for per-vertex circular clipping.
        // Both are Vulkan 1.0 core features universally supported on desktop
        // GPUs and on RPi VC7; we don't gate on the feature query.
        const enabled_features = vk.PhysicalDeviceFeatures{
            .fill_mode_non_solid = .true,
            .shader_clip_distance = .true,
        };

        var queried_features2: vk.PhysicalDeviceFeatures2 = .{ .features = .{} };
        vki.getPhysicalDeviceFeatures2(pdev, &queried_features2);
        if (queried_features2.features.fill_mode_non_solid != .true) {
            std.log.err("Required Vulkan feature missing: fill_mode_non_solid", .{});
            return error.RequiredVulkanFeatureMissing;
        }
        if (queried_features2.features.shader_clip_distance != .true) {
            std.log.err("Required Vulkan feature missing: shader_clip_distance", .{});
            return error.RequiredVulkanFeatureMissing;
        }

        const device = try vki.createDevice(pdev, &.{
            .queue_create_info_count = unique_count,
            .p_queue_create_infos = &queue_create_infos,
            .enabled_extension_count = @intCast(device_extensions.len),
            .pp_enabled_extension_names = device_extensions.ptr,
            .p_enabled_features = &enabled_features,
        }, null);

        const vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);
        errdefer vkd.destroyDevice(device, null);

        const graphics_queue = vkd.getDeviceQueue(device, graphics_family, 0);
        const present_queue = vkd.getDeviceQueue(device, present_family, 0);

        // ---- Surface format ----
        // `hdr_capable` is now the pure hardware/loader capability (the colorspace
        // extension), independent of the request, so the runtime HDR toggle can read
        // it for gating. The initial format still honors `opts.enable_hdr`.
        const hdr_capable = has_colorspace_ext;
        const display_hdr = hdr_capable and opts.enable_hdr and (window == null or
            c.SDL_GetBooleanProperty(c.SDL_GetWindowProperties(window), c.SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false));
        const surface_fmt = try swapchain_mod.chooseSurfaceFormat(&vki, pdev, surface, allocator, display_hdr);

        // ---- MSAA sample count (clamped to GPU support for color+depth) ----
        const samples = msaa_mod.pickSampleCount(props.limits, opts.msaa_request);
        if (msaa_mod.sampleCountToInt(samples) != opts.msaa_request) {
            std.log.warn("MSAA: requested {d}x, GPU clamped to {d}x", .{ opts.msaa_request, msaa_mod.sampleCountToInt(samples) });
        }

        const initial_extent = switch (surface_mode) {
            .windowed => |w| swapchain_mod.framebufferExtent(w),
            .headless => |e| e,
        };

        // ---- Command pool + buffers ----
        const cmd_pool = try vkd.createCommandPool(device, &.{
            .queue_family_index = graphics_family,
            .flags = .{ .reset_command_buffer_bit = true },
        }, null);
        errdefer vkd.destroyCommandPool(device, cmd_pool, null);

        var cmd_buffers: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer = undefined;
        try vkd.allocateCommandBuffers(device, &.{
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = MAX_FRAMES_IN_FLIGHT,
        }, &cmd_buffers);

        // Re-fetch chosen queue's timestampValidBits; 0 means timestamps unsupported on this queue
        // (e.g., RPi v3dv graphics queue). We then run without GPU timestamps but still time CPU-side.
        const qf_props = try vki.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, allocator);
        defer allocator.free(qf_props);
        const timestamp_valid_bits = qf_props[graphics_family].timestamp_valid_bits;

        var query_pool: vk.QueryPool = .null_handle;
        if (opts.bench_enabled and timestamp_valid_bits > 0 and props.limits.timestamp_period > 0) {
            query_pool = try vkd.createQueryPool(device, &.{
                .query_type = .timestamp,
                .query_count = MAX_FRAMES_IN_FLIGHT * 4,
            }, null);
        } else if (opts.bench_enabled) {
            std.log.info("  Timestamps unavailable on graphics queue (valid_bits={d}, period={d}); GPU phase timing disabled", .{
                timestamp_valid_bits, props.limits.timestamp_period,
            });
        }
        errdefer if (query_pool != .null_handle) vkd.destroyQueryPool(device, query_pool, null);

        return .{
            .vki = vki,
            .vkd = vkd,
            .instance = instance,
            .surface = surface,
            .pdev = pdev,
            .device = device,
            .device_name = props.device_name,
            .graphics_family = graphics_family,
            .present_family = present_family,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .mem_props = mem_props,
            .vram_mb = chosen.vram_mb,
            .cmd_pool = cmd_pool,
            .cmd_buffers = cmd_buffers,
            .samples = samples,
            .surface_format = surface_fmt,
            .initial_extent = initial_extent,
            .hdr_capable = hdr_capable,
            .bench_enabled = opts.bench_enabled,
            .query_pool = query_pool,
            .timestamp_period_ns = props.limits.timestamp_period,
            .timestamp_valid_bits = timestamp_valid_bits,
            .window = window,
            .allocator = allocator,
            .io = io,
        };
    }

    pub fn deinit(self: *VulkanContext) void {
        if (self.query_pool != .null_handle) self.vkd.destroyQueryPool(self.device, self.query_pool, null);
        self.vkd.destroyCommandPool(self.device, self.cmd_pool, null);
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroySurfaceKHR(self.instance, self.surface, null);
        self.vki.destroyInstance(self.instance, null);
    }

    pub fn gpuCtx(self: *const VulkanContext) GpuContext {
        return .{
            .vkd = self.vkd,
            .device = self.device,
            .mem_props = self.mem_props,
            .queue = self.graphics_queue,
            .cmd_pool = self.cmd_pool,
            .vram_mb = self.vram_mb,
        };
    }
};
