//! Vulkan buffer + memory helpers shared by every subsystem.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;

pub const BufferWithMemory = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
};

pub fn createBuffer(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    property_flags: vk.MemoryPropertyFlags,
) !BufferWithMemory {
    const buffer = try vkd.createBuffer(device, &.{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    }, null);
    errdefer vkd.destroyBuffer(device, buffer, null);

    const mem_reqs = vkd.getBufferMemoryRequirements(device, buffer);
    const mem_idx = findMemoryType(mem_props, mem_reqs.memory_type_bits, property_flags) orelse
        return error.NoSuitableMemory;

    const memory = try vkd.allocateMemory(device, &.{
        .allocation_size = mem_reqs.size,
        .memory_type_index = mem_idx,
    }, null);
    errdefer vkd.freeMemory(device, memory, null);

    try vkd.bindBufferMemory(device, buffer, memory, 0);
    return .{ .buffer = buffer, .memory = memory };
}

pub const MappedBuffer = struct {
    buf: BufferWithMemory,
    map: [*]u8,
};

/// Host-visible, host-coherent buffer with a persistent memory map. Pair
/// with `destroyMappedBuffer` on teardown.
pub fn createMappedBuffer(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
) !MappedBuffer {
    const buf = try createBuffer(
        vkd,
        device,
        mem_props,
        size,
        usage,
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    errdefer {
        vkd.destroyBuffer(device, buf.buffer, null);
        vkd.freeMemory(device, buf.memory, null);
    }
    const map: [*]u8 = @ptrCast(try vkd.mapMemory(device, buf.memory, 0, size, .{}) orelse return error.MapFailed);
    return .{ .buf = buf, .map = map };
}

pub fn destroyMappedBuffer(vkd: anytype, device: vk.Device, b: BufferWithMemory) void {
    vkd.unmapMemory(device, b.memory);
    vkd.destroyBuffer(device, b.buffer, null);
    vkd.freeMemory(device, b.memory, null);
}

/// Memory type for transient framebuffer attachments (depth, MSAA color).
/// Prefers LAZILY_ALLOCATED so tile-based GPUs (RPi VC7) keep the attachment
/// in tile RAM and never back it with real device memory; falls back to
/// DEVICE_LOCAL on desktop GPUs where lazy allocation isn't supported.
pub fn findTransientMemory(mem_props: vk.PhysicalDeviceMemoryProperties, type_filter: u32) ?u32 {
    return findMemoryType(mem_props, type_filter, .{ .lazily_allocated_bit = true }) orelse
        findMemoryType(mem_props, type_filter, .{ .device_local_bit = true });
}

pub fn findMemoryType(
    mem_props: vk.PhysicalDeviceMemoryProperties,
    type_filter: u32,
    property_flags: vk.MemoryPropertyFlags,
) ?u32 {
    for (0..mem_props.memory_type_count) |i| {
        const idx: u5 = @intCast(i);
        if (type_filter & (@as(u32, 1) << idx) != 0) {
            const flags = mem_props.memory_types[i].property_flags;
            const required: u32 = @bitCast(property_flags);
            const available: u32 = @bitCast(flags);
            if (required & available == required) {
                return @intCast(i);
            }
        }
    }
    return null;
}

/// Allocate and begin a one-shot command buffer for transfer operations.
pub fn beginOneShot(
    vkd: anytype,
    device: vk.Device,
    cmd_pool: vk.CommandPool,
) !vk.CommandBuffer {
    var cmd_buf: [1]vk.CommandBuffer = undefined;
    try vkd.allocateCommandBuffers(device, &.{
        .command_pool = cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, &cmd_buf);
    errdefer vkd.freeCommandBuffers(device, cmd_pool, &cmd_buf);

    try vkd.beginCommandBuffer(cmd_buf[0], &.{
        .flags = .{ .one_time_submit_bit = true },
    });

    return cmd_buf[0];
}

/// End, submit, wait, and free a one-shot command buffer.
pub fn endOneShot(
    vkd: anytype,
    device: vk.Device,
    cmd_pool: vk.CommandPool,
    queue: vk.Queue,
    cmd_buf: vk.CommandBuffer,
) !void {
    const bufs = [1]vk.CommandBuffer{cmd_buf};
    defer vkd.freeCommandBuffers(device, cmd_pool, &bufs);

    try vkd.endCommandBuffer(cmd_buf);

    const submits = [1]vk.SubmitInfo{.{
        .command_buffer_count = 1,
        .p_command_buffers = &bufs,
    }};
    try vkd.queueSubmit(queue, &submits, .null_handle);

    try vkd.queueWaitIdle(queue);
}

/// Create a DEVICE_LOCAL buffer and upload data via a staging buffer.
/// The staging buffer is created, used, and destroyed within this call.
/// `usage` should include the buffer's purpose bits (e.g., storage_buffer_bit,
/// index_buffer_bit). TRANSFER_DST is added automatically.
pub fn uploadBuffer(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    queue: vk.Queue,
    cmd_pool: vk.CommandPool,
    data: []const u8,
    usage: vk.BufferUsageFlags,
) !BufferWithMemory {
    const size: vk.DeviceSize = data.len;

    const staging = try createBuffer(
        vkd,
        device,
        mem_props,
        size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    defer {
        vkd.destroyBuffer(device, staging.buffer, null);
        vkd.freeMemory(device, staging.memory, null);
    }

    const ptr: [*]u8 = @ptrCast(
        try vkd.mapMemory(device, staging.memory, 0, size, .{}) orelse return error.MapFailed,
    );
    @memcpy(ptr[0..data.len], data);
    vkd.unmapMemory(device, staging.memory);

    var dst_usage = usage;
    dst_usage.transfer_dst_bit = true;
    const dst = try createBuffer(
        vkd,
        device,
        mem_props,
        size,
        dst_usage,
        .{ .device_local_bit = true },
    );
    errdefer {
        vkd.destroyBuffer(device, dst.buffer, null);
        vkd.freeMemory(device, dst.memory, null);
    }

    const cmd_buf = try beginOneShot(vkd, device, cmd_pool);
    const regions = [1]vk.BufferCopy{.{ .src_offset = 0, .dst_offset = 0, .size = size }};
    vkd.cmdCopyBuffer(cmd_buf, staging.buffer, dst.buffer, &regions);
    try endOneShot(vkd, device, cmd_pool, queue, cmd_buf);

    return dst;
}
