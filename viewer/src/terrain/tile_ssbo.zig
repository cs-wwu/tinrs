//! GPU-side tile SSBO: layout, allocation, and incremental updates.
//!
//! Layout: `[Header: 5 uint32s] [Directory: tile_pool_size x SSBO_DIR_STRIDE uint32s] [Slots: tile_pool_size x SLOT_STRIDE bytes]`
//!
//! Slots are fixed-stride (`SLOT_STRIDE = catalog.slotStride() = max(weights_bytes)`)
//! so any catalog tile fits any slot. `SSBOShell` holds a CPU shadow of the
//! header+directory plus a dirty-range marker. `setTileCount`/`setDirEntry`
//! update the shadow; `recordHeaderDirUpdate` flushes the dirty range to the
//! GPU via `vkCmdUpdateBuffer` inside the caller's command buffer. Slot bytes
//! are uploaded separately by `tile_streamer` via staged `vkCmdCopyBuffer`.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const buffer = @import("../render/buffer.zig");
const coords = @import("coords.zig");
const tile_loader = @import("tile_loader.zig");

/// Header size in uint32s: [tile_count, features, mlp_hidden_dim, num_outputs, has_catalog].
/// `has_catalog` lets the shader distinguish "catalog loaded but no tiles streamed yet"
/// (ocean fallback) from "no catalog at all, --procedural" (procedural fallback).
pub const SSBO_HDR_SIZE: u32 = 5;
/// Directory entry size in uint32s per tile.
pub const SSBO_DIR_STRIDE: u32 = 9;
pub const SSBOShell = struct {
    buf: buffer.BufferWithMemory,
    size: u64,
    data_base_bytes: u64,
    slot_stride_bytes: u32,
    tile_pool_size: u32,

    /// CPU shadow of the GPU header+directory, sized to tile_pool_size.
    shadow: []u32,

    /// Inclusive lo, exclusive hi, in shadow uint indices. Empty range
    /// (`lo == hi`) means clean.
    dirty_lo: u32,
    dirty_hi: u32,

    pub fn slotByteOffset(self: *const SSBOShell, slot_idx: u32) u64 {
        return self.data_base_bytes + @as(u64, slot_idx) * @as(u64, self.slot_stride_bytes);
    }

    pub fn tileCount(self: *const SSBOShell) u32 {
        return self.shadow[0];
    }

    fn markDirty(self: *SSBOShell, lo: u32, hi: u32) void {
        if (self.dirty_lo == self.dirty_hi) {
            self.dirty_lo = lo;
            self.dirty_hi = hi;
        } else {
            self.dirty_lo = @min(self.dirty_lo, lo);
            self.dirty_hi = @max(self.dirty_hi, hi);
        }
    }

    pub fn setTileCount(self: *SSBOShell, n: u32) void {
        self.shadow[0] = n;
        self.markDirty(0, 1);
    }

    /// Write directory entry `dir_idx` referencing data slot `slot_idx`. The
    /// caller is responsible for the staged `vkCmdCopyBuffer` that fills the
    /// slot bytes; this only updates the directory metadata + offset pointer.
    pub fn setDirEntry(
        self: *SSBOShell,
        dir_idx: u32,
        desc: *const tile_loader.TileDescriptor,
        slot_idx: u32,
    ) void {
        std.debug.assert(dir_idx < self.tile_pool_size);
        const base = SSBO_HDR_SIZE + dir_idx * SSBO_DIR_STRIDE;
        const slot_uint_offset: u32 = @intCast(
            self.data_base_bytes / 4 + @as(u64, slot_idx) * (@as(u64, self.slot_stride_bytes) / 4),
        );
        self.shadow[base + 0] = @bitCast(desc.origin_x);
        self.shadow[base + 1] = @bitCast(desc.origin_z);
        self.shadow[base + 2] = desc.resolution_h;
        self.shadow[base + 3] = desc.resolution_w;
        self.shadow[base + 4] = desc.grid_uints;
        self.shadow[base + 5] = @bitCast(desc.elev_min);
        self.shadow[base + 6] = @bitCast(desc.elev_max);
        self.shadow[base + 7] = slot_uint_offset;
        self.shadow[base + 8] = @bitCast(desc.grad_scale);
        self.markDirty(base, base + SSBO_DIR_STRIDE);
    }

    /// Flush the dirty header+directory range via `vkCmdUpdateBuffer`. Returns
    /// true when an update was emitted. Caller is responsible for the
    /// surrounding `transfer->shader_read` barrier.
    pub fn recordHeaderDirUpdate(self: *SSBOShell, vkd: anytype, cmd_buf: vk.CommandBuffer) bool {
        if (self.dirty_lo == self.dirty_hi) return false;
        const max_chunk: u32 = 65536 / 4; // vkCmdUpdateBuffer limit in uints
        var pos = self.dirty_lo;
        while (pos < self.dirty_hi) {
            const remaining = self.dirty_hi - pos;
            const chunk = @min(remaining, max_chunk);
            const offset_bytes: vk.DeviceSize = @as(u64, pos) * 4;
            const chunk_bytes: vk.DeviceSize = @as(u64, chunk) * 4;
            const ptr: [*]const u32 = @ptrCast(&self.shadow[pos]);
            const bytes: [*]const u8 = @ptrCast(ptr);
            vkd.cmdUpdateBuffer(cmd_buf, self.buf.buffer, offset_bytes, chunk_bytes, bytes);
            pos += chunk;
        }
        self.dirty_lo = 0;
        self.dirty_hi = 0;
        return true;
    }
};

/// Allocate the SSBO with capacity for `tile_pool_size` slots and write a
/// header of `[tile_count=0, features, mlp_hidden_dim, num_outputs]`.
/// Directory and slot bytes are uninitialized; the shader never reads beyond
/// `tile_count`, so directory entries past 0 are unobservable until written.
/// TODO: variable slot stride per tile for mixed-resolution catalogs.
pub fn createSSBOShell(
    allocator: std.mem.Allocator,
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    catalog: *const tile_loader.TileCatalog,
    tile_pool_size: u32,
    queue: vk.Queue,
    cmd_pool: vk.CommandPool,
) !SSBOShell {
    std.debug.assert(tile_pool_size >= tile_loader.MIN_TILE_POOL and tile_pool_size <= tile_loader.MAX_TILES);
    const slot_stride_bytes: u32 = catalog.slotStride();
    std.debug.assert(slot_stride_bytes % 4 == 0);

    const dir_uints: u32 = SSBO_HDR_SIZE + tile_pool_size * SSBO_DIR_STRIDE;
    const data_base_bytes: u64 = @as(u64, dir_uints) * 4;
    const total_bytes: u64 = data_base_bytes + @as(u64, tile_pool_size) * @as(u64, slot_stride_bytes);

    const shadow = try allocator.alloc(u32, dir_uints);
    errdefer allocator.free(shadow);
    @memset(shadow, 0);

    var shell: SSBOShell = .{
        .buf = undefined,
        .size = total_bytes,
        .data_base_bytes = data_base_bytes,
        .slot_stride_bytes = slot_stride_bytes,
        .tile_pool_size = tile_pool_size,
        .shadow = shadow,
        .dirty_lo = 0,
        .dirty_hi = 0,
    };
    shell.shadow[0] = 0; // tile_count
    shell.shadow[1] = catalog.shared_features;
    shell.shadow[2] = catalog.shared_mlp_hidden_dim;
    shell.shadow[3] = catalog.shared_num_outputs;
    shell.shadow[4] = 1; // has_catalog

    shell.buf = try buffer.createBuffer(
        vkd,
        device,
        mem_props,
        total_bytes,
        .{ .storage_buffer_bit = true, .transfer_dst_bit = true },
        .{ .device_local_bit = true },
    );
    errdefer {
        vkd.destroyBuffer(device, shell.buf.buffer, null);
        vkd.freeMemory(device, shell.buf.memory, null);
    }

    // One-shot upload of the 16-byte header. Directory entries beyond
    // tile_count=0 are never read by the shader, so leaving them as
    // uninitialized device memory is safe; setDirEntry+recordHeaderDirUpdate
    // populates them as tiles arrive.
    const cmd_buf = try buffer.beginOneShot(vkd, device, cmd_pool);
    const hdr_bytes: [*]const u8 = @ptrCast(&shell.shadow[0]);
    vkd.cmdUpdateBuffer(cmd_buf, shell.buf.buffer, 0, SSBO_HDR_SIZE * 4, hdr_bytes);
    try buffer.endOneShot(vkd, device, cmd_pool, queue, cmd_buf);

    std.log.debug("SSBO shell: 0/{d} slots x {d:.0} KB = {d:.1} MB ({d} catalog tiles)", .{
        tile_pool_size,
        @as(f32, @floatFromInt(slot_stride_bytes)) / 1024.0,
        @as(f32, @floatFromInt(total_bytes)) / (1024.0 * 1024.0),
        catalog.count(),
    });

    return shell;
}

/// Minimal SSBO sized for the header alone, all zeros (`tile_count = 0`,
/// `has_catalog = 0`); `--procedural` fallback path. Shader sees no catalog
/// and renders procedural terrain.
pub fn createEmptySSBO(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
) !buffer.BufferWithMemory {
    const size: u64 = SSBO_HDR_SIZE * 4;
    const empty = try buffer.createBuffer(
        vkd,
        device,
        mem_props,
        size,
        .{ .storage_buffer_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    const ptr: [*]u32 = @ptrCast(@alignCast(try vkd.mapMemory(device, empty.memory, 0, size, .{}) orelse return error.MapFailed));
    for (0..SSBO_HDR_SIZE) |i| ptr[i] = 0;
    vkd.unmapMemory(device, empty.memory);
    return empty;
}

/// Full-size (~253 KB), all-zero dir grid for `--procedural` mode. The
/// compute shader's `findTile` short-circuits on `tile_count == 0` before
/// reading, but binding a real-sized buffer keeps any out-of-bounds path
/// from tripping driver UB.
pub fn createEmptyDirGrid(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
) !buffer.BufferWithMemory {
    const size: u64 = @as(u64, coords.GRID_CELL_COUNT) * @sizeOf(u32);
    const empty = try buffer.createBuffer(
        vkd,
        device,
        mem_props,
        size,
        .{ .storage_buffer_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    const ptr: [*]u32 = @ptrCast(@alignCast(try vkd.mapMemory(device, empty.memory, 0, size, .{}) orelse return error.MapFailed));
    @memset(ptr[0..coords.GRID_CELL_COUNT], 0);
    vkd.unmapMemory(device, empty.memory);
    return empty;
}
