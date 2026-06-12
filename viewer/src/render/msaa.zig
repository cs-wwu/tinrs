//! MSAA color resources + sample-count helpers.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const buffer = @import("buffer.zig");

/// Multisampled color image used as the render target when MSAA is enabled.
/// Resolved to the swapchain image at end-of-pass via pResolveAttachments.
/// Only allocated when samples > 1; zero-handles otherwise.
pub const MSAAColorResources = struct {
    image: vk.Image,
    view: vk.ImageView,
    memory: vk.DeviceMemory,
};

/// Pick the highest supported sample count <= requested.
/// Returns 1_bit if requested == 1 or GPU supports nothing higher.
/// Limited by the intersection of color + depth framebuffer support.
pub fn pickSampleCount(limits: vk.PhysicalDeviceLimits, requested: u32) vk.SampleCountFlags {
    if (requested <= 1) return .{ .@"1_bit" = true };
    const supported: u32 = @bitCast(supportedMask(limits));
    var n: u32 = requested;
    while (n > 1) : (n /= 2) {
        const bit: u32 = n;
        if ((supported & bit) != 0) return intToSampleCount(n);
    }
    return .{ .@"1_bit" = true };
}

/// Sample counts usable as a render target with both color and depth: the
/// intersection of the two framebuffer limits. The single owner of that rule;
/// `pickSampleCount` (init clamp) and `Renderer.supportedSampleCounts` (menu
/// options) both go through it so the offered set can never exceed what init picks.
pub fn supportedMask(limits: vk.PhysicalDeviceLimits) vk.SampleCountFlags {
    const color: u32 = @bitCast(limits.framebuffer_color_sample_counts);
    const depth: u32 = @bitCast(limits.framebuffer_depth_sample_counts);
    return @bitCast(color & depth);
}

pub fn intToSampleCount(n: u32) vk.SampleCountFlags {
    return switch (n) {
        1 => .{ .@"1_bit" = true },
        2 => .{ .@"2_bit" = true },
        4 => .{ .@"4_bit" = true },
        8 => .{ .@"8_bit" = true },
        16 => .{ .@"16_bit" = true },
        32 => .{ .@"32_bit" = true },
        64 => .{ .@"64_bit" = true },
        else => unreachable, // pickSampleCount only ever calls us with a power-of-2 sample count
    };
}

pub fn sampleCountToInt(s: vk.SampleCountFlags) u32 {
    if (s.@"64_bit") return 64;
    if (s.@"32_bit") return 32;
    if (s.@"16_bit") return 16;
    if (s.@"8_bit") return 8;
    if (s.@"4_bit") return 4;
    if (s.@"2_bit") return 2;
    return 1;
}

/// Whether a power-of-2 sample count `n` is present in the support `mask` (the
/// color-and-depth intersection from `supportedSampleCounts`). 1x is always
/// available (Vulkan guarantees `SAMPLE_COUNT_1_BIT`). Drives the settings-menu
/// MSAA cycle, which only offers supported counts.
pub fn countSupported(mask: vk.SampleCountFlags, n: u32) bool {
    if (n <= 1) return true;
    return (@as(u32, @bitCast(mask)) & n) != 0;
}

/// Index of `count` within `counts` (a list of offered sample counts), or 0 if
/// absent. Seeds and recovers the MSAA cycle's active index from the renderer's
/// actually-applied sample count.
pub fn indexOfCount(count: u32, counts: []const u32) usize {
    for (counts, 0..) |c, i| if (c == count) return i;
    return 0;
}

pub fn create(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    extent: vk.Extent2D,
    format: vk.Format,
    samples: vk.SampleCountFlags,
) !MSAAColorResources {
    if (samples.@"1_bit") return std.mem.zeroes(MSAAColorResources);

    const image = try vkd.createImage(device, &.{
        .image_type = .@"2d",
        .format = format,
        .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = samples,
        .tiling = .optimal,
        // TRANSIENT lets tile-based GPUs (RPi VideoCore VII) keep MSAA on-chip,
        // never backing it with real memory; the resolve consumes it inside the pass.
        .usage = .{ .color_attachment_bit = true, .transient_attachment_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer vkd.destroyImage(device, image, null);

    const mem_reqs = vkd.getImageMemoryRequirements(device, image);
    const mem_idx = buffer.findTransientMemory(mem_props, mem_reqs.memory_type_bits) orelse return error.NoSuitableMemory;

    const memory = try vkd.allocateMemory(device, &.{
        .allocation_size = mem_reqs.size,
        .memory_type_index = mem_idx,
    }, null);
    errdefer vkd.freeMemory(device, memory, null);

    try vkd.bindImageMemory(device, image, memory, 0);

    const view = try vkd.createImageView(device, &.{
        .image = image,
        .view_type = .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    }, null);

    return .{ .image = image, .view = view, .memory = memory };
}

pub fn destroy(vkd: anytype, device: vk.Device, msaa: MSAAColorResources) void {
    if (msaa.image == .null_handle) return;
    vkd.destroyImageView(device, msaa.view, null);
    vkd.destroyImage(device, msaa.image, null);
    vkd.freeMemory(device, msaa.memory, null);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "countSupported: 1x always available; others gated by the mask" {
    // Mask advertising 1/2/4 but not 8 (4_bit|2_bit|1_bit).
    const mask = vk.SampleCountFlags{ .@"1_bit" = true, .@"2_bit" = true, .@"4_bit" = true };
    try testing.expect(countSupported(mask, 1));
    try testing.expect(countSupported(mask, 2));
    try testing.expect(countSupported(mask, 4));
    try testing.expect(!countSupported(mask, 8));
    // 1x is available even against an (impossible) empty mask.
    try testing.expect(countSupported(.{}, 1));
    try testing.expect(!countSupported(.{}, 4));
}

test "indexOfCount: finds the count, falls back to 0 when absent" {
    const counts = [_]u32{ 1, 2, 4 };
    try testing.expectEqual(@as(usize, 0), indexOfCount(1, &counts));
    try testing.expectEqual(@as(usize, 2), indexOfCount(4, &counts));
    try testing.expectEqual(@as(usize, 0), indexOfCount(8, &counts)); // not offered -> 0
    try testing.expectEqual(@as(usize, 0), indexOfCount(4, &.{})); // empty -> 0
}

test "sampleCountToInt round-trips intToSampleCount" {
    for ([_]u32{ 1, 2, 4, 8, 16, 32, 64 }) |n| {
        try testing.expectEqual(n, sampleCountToInt(intToSampleCount(n)));
    }
}
