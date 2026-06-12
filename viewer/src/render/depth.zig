//! Depth buffer resources.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const buffer = @import("buffer.zig");

pub const DEPTH_FORMAT = vk.Format.d32_sfloat;

pub const DepthResources = struct {
    image: vk.Image,
    view: vk.ImageView,
    memory: vk.DeviceMemory,
};

pub fn create(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    extent: vk.Extent2D,
    samples: vk.SampleCountFlags,
) !DepthResources {
    const image = try vkd.createImage(device, &.{
        .image_type = .@"2d",
        .format = DEPTH_FORMAT,
        .extent = .{ .width = extent.width, .height = extent.height, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = samples,
        .tiling = .optimal,
        .usage = .{ .depth_stencil_attachment_bit = true, .transient_attachment_bit = true },
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
        .format = DEPTH_FORMAT,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{
            .aspect_mask = .{ .depth_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    }, null);

    return .{ .image = image, .view = view, .memory = memory };
}

pub fn destroy(vkd: anytype, device: vk.Device, depth: DepthResources) void {
    vkd.destroyImageView(device, depth.view, null);
    vkd.destroyImage(device, depth.image, null);
    vkd.freeMemory(device, depth.memory, null);
}
