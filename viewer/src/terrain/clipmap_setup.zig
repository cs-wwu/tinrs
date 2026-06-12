//! One-time Vulkan setup for the clipmap system: index buffer, descriptor
//! sets, and pipeline creation. Called only during Clipmap.init.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const buffer = @import("../render/buffer.zig");
const pipeline = @import("../render/pipeline.zig");

const comp_spv align(@alignOf(u32)) = @embedFile("clipmap_update_shader").*;
const clipmap_vert_spv align(@alignOf(u32)) = @embedFile("clipmap_vert").*;
const terrain_frag_spv align(@alignOf(u32)) = @embedFile("terrain_frag").*;
const clipmap_vert_debug_spv align(@alignOf(u32)) = @embedFile("clipmap_vert_debug").*;
const terrain_frag_debug_spv align(@alignOf(u32)) = @embedFile("terrain_frag_debug").*;

const clipmap = @import("clipmap.zig");
const ComputePushConstants = clipmap.ComputePushConstants;
const SceneUBO = clipmap.SceneUBO;
const DrawEntry = clipmap.DrawEntry;
const MAX_DRAWS = clipmap.MAX_DRAWS;

pub const IndexBufferResult = struct {
    buf: buffer.BufferWithMemory,
    count: u32,
};

/// Build the shared per-chunk index buffer. Vertices are addressed in
/// chunk-local coordinates with stride `chunk_vertex_dim = chunk_cells + 1`.
/// Every chunk in every level binds this buffer; level-relative origin rides
/// in push constants.
pub fn createChunkIndexBuffer(
    allocator: std.mem.Allocator,
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    queue: vk.Queue,
    cmd_pool: vk.CommandPool,
    chunk_cells: u32,
) !IndexBufferResult {
    const chunk_vertex_dim: u32 = chunk_cells + 1;
    const index_count: u32 = chunk_cells * chunk_cells * 6;

    const indices = try allocator.alloc(u32, index_count);
    defer allocator.free(indices);

    var idx: usize = 0;
    for (0..chunk_cells) |row| {
        for (0..chunk_cells) |col| {
            const tl: u32 = @intCast(row * chunk_vertex_dim + col);
            const tr: u32 = tl + 1;
            const bl: u32 = tl + chunk_vertex_dim;
            const br: u32 = bl + 1;
            indices[idx] = tl;
            indices[idx + 1] = bl;
            indices[idx + 2] = br;
            indices[idx + 3] = tl;
            indices[idx + 4] = br;
            indices[idx + 5] = tr;
            idx += 6;
        }
    }
    std.debug.assert(idx == index_count);

    const buf = try buffer.uploadBuffer(
        vkd,
        device,
        mem_props,
        queue,
        cmd_pool,
        std.mem.sliceAsBytes(indices),
        .{ .index_buffer_bit = true },
    );
    return .{ .buf = buf, .count = index_count };
}

pub const DescriptorResult = struct {
    desc_pool: vk.DescriptorPool,
    desc_sets: [2]vk.DescriptorSet,
    ubo_bufs: [2]buffer.BufferWithMemory,
    ubo_maps: [2][*]u8,
    draw_entry_bufs: [2]buffer.BufferWithMemory,
    draw_entry_maps: [2][*]u8,
};

/// Create the descriptor set layout shared by the clipmap compute + graphics
/// pipelines and the sky pipeline. Size-independent (binding definitions only),
/// so the owner (Scene) creates it once and it outlives Clipmap rebuilds; the
/// size-dependent pool + sets live in `createDescriptors`.
pub fn createDescLayout(vkd: anytype, device: vk.Device) !vk.DescriptorSetLayout {
    // binding 0 = ring buffer, 1 = weights, 2 = UBO, 3 = dir grid, 4 = draw entries.
    const bindings = [5]vk.DescriptorSetLayoutBinding{
        .{
            .binding = 0,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .compute_bit = true, .vertex_bit = true },
        },
        .{
            .binding = 1,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .compute_bit = true },
        },
        .{
            .binding = 2,
            .descriptor_type = .uniform_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
        },
        .{
            .binding = 3,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .compute_bit = true },
        },
        .{
            .binding = 4,
            .descriptor_type = .storage_buffer,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
        },
    };
    return vkd.createDescriptorSetLayout(device, &.{
        .binding_count = bindings.len,
        .p_bindings = &bindings,
    }, null);
}

/// Create the descriptor pool, UBOs, allocate sets, and write descriptors
/// against a caller-owned `desc_layout` (see `createDescLayout`). `weights_buf`
/// is bound at binding 1; pass the streaming SSBO with tiles, or a tiny empty
/// SSBO for the procedural fallback.
pub fn createDescriptors(
    vkd: anytype,
    device: vk.Device,
    mem_props: vk.PhysicalDeviceMemoryProperties,
    desc_layout: vk.DescriptorSetLayout,
    ring_buf: buffer.BufferWithMemory,
    ring_buf_size: vk.DeviceSize,
    weights_buf: vk.Buffer,
    dir_grid_buf: vk.Buffer,
) !DescriptorResult {

    // 4 storage bindings * 2 sets = 8 storage descriptors; 1 UBO * 2 = 2.
    const pool_sizes = [2]vk.DescriptorPoolSize{
        .{ .type = .storage_buffer, .descriptor_count = 8 },
        .{ .type = .uniform_buffer, .descriptor_count = 2 },
    };
    const desc_pool = try vkd.createDescriptorPool(device, &.{
        .max_sets = 2,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
    }, null);
    errdefer vkd.destroyDescriptorPool(device, desc_pool, null);

    const ubo_size: vk.DeviceSize = @sizeOf(SceneUBO);
    var ubo_bufs: [2]buffer.BufferWithMemory = undefined;
    var ubo_maps: [2][*]u8 = undefined;
    var ubo_count: usize = 0;
    errdefer for (0..ubo_count) |i| buffer.destroyMappedBuffer(&vkd, device, ubo_bufs[i]);
    for (0..2) |i| {
        const m = try buffer.createMappedBuffer(&vkd, device, mem_props, ubo_size, .{ .uniform_buffer_bit = true });
        ubo_bufs[i] = m.buf;
        ubo_maps[i] = m.map;
        ubo_count += 1;
    }

    const draw_entry_size: vk.DeviceSize = @as(u64, MAX_DRAWS) * @sizeOf(DrawEntry);
    var draw_entry_bufs: [2]buffer.BufferWithMemory = undefined;
    var draw_entry_maps: [2][*]u8 = undefined;
    var draw_entry_count: usize = 0;
    errdefer for (0..draw_entry_count) |i| buffer.destroyMappedBuffer(&vkd, device, draw_entry_bufs[i]);
    for (0..2) |i| {
        const m = try buffer.createMappedBuffer(&vkd, device, mem_props, draw_entry_size, .{ .storage_buffer_bit = true });
        draw_entry_bufs[i] = m.buf;
        draw_entry_maps[i] = m.map;
        draw_entry_count += 1;
    }

    const set_layouts = [2]vk.DescriptorSetLayout{ desc_layout, desc_layout };
    var desc_sets: [2]vk.DescriptorSet = undefined;
    try vkd.allocateDescriptorSets(device, &.{
        .descriptor_pool = desc_pool,
        .descriptor_set_count = 2,
        .p_set_layouts = &set_layouts,
    }, &desc_sets);

    const ring_buf_info = vk.DescriptorBufferInfo{
        .buffer = ring_buf.buffer,
        .offset = 0,
        .range = ring_buf_size,
    };
    const binding1_info = vk.DescriptorBufferInfo{
        .buffer = weights_buf,
        .offset = 0,
        .range = vk.WHOLE_SIZE,
    };
    const grid_info = vk.DescriptorBufferInfo{
        .buffer = dir_grid_buf,
        .offset = 0,
        .range = vk.WHOLE_SIZE,
    };

    for (0..2) |i| {
        const ubo_info = vk.DescriptorBufferInfo{
            .buffer = ubo_bufs[i].buffer,
            .offset = 0,
            .range = ubo_size,
        };
        const draw_entry_info = vk.DescriptorBufferInfo{
            .buffer = draw_entry_bufs[i].buffer,
            .offset = 0,
            .range = draw_entry_size,
        };
        const writes = [5]vk.WriteDescriptorSet{
            .{
                .dst_set = desc_sets[i],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&ring_buf_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = desc_sets[i],
                .dst_binding = 1,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&binding1_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = desc_sets[i],
                .dst_binding = 2,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .uniform_buffer,
                .p_buffer_info = @ptrCast(&ubo_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = desc_sets[i],
                .dst_binding = 3,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&grid_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
            .{
                .dst_set = desc_sets[i],
                .dst_binding = 4,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&draw_entry_info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            },
        };
        vkd.updateDescriptorSets(device, &writes, null);
    }

    return .{
        .desc_pool = desc_pool,
        .desc_sets = desc_sets,
        .ubo_bufs = ubo_bufs,
        .ubo_maps = ubo_maps,
        .draw_entry_bufs = draw_entry_bufs,
        .draw_entry_maps = draw_entry_maps,
    };
}

pub const PipelineResult = struct {
    compute_pipeline: vk.Pipeline,
    compute_pipeline_layout: vk.PipelineLayout,
    graphics_pipeline_layout: vk.PipelineLayout,
    default: GraphicsPipelinePair,
    debug: GraphicsPipelinePair,
};

pub const GraphicsPipelinePair = struct {
    filled: vk.Pipeline,
    wireframe: vk.Pipeline,
};

pub const GraphicsPipelineSet = struct {
    default: GraphicsPipelinePair,
    debug: GraphicsPipelinePair,
};

/// Mirrors `layout(constant_id = N)` in clipmap_terrain.vert / terrain.frag.
/// Pipelines must be rebuilt when these change (fixed after Clipmap.init today).
const TerrainSpec = extern struct {
    ring_size: u32,
    chunk_vertex_dim: u32,
};

const terrain_spec_entries = [_]vk.SpecializationMapEntry{
    .{ .constant_id = 0, .offset = @offsetOf(TerrainSpec, "ring_size"), .size = @sizeOf(u32) },
    .{ .constant_id = 1, .offset = @offsetOf(TerrainSpec, "chunk_vertex_dim"), .size = @sizeOf(u32) },
};

/// Wireframe uses `polygon_mode=line` and disables back-face cull so back
/// edges stay visible. Wireframe MUST write depth, otherwise sky (drawing
/// after with reverse-Z greater_or_equal) sees the cleared 0.0 and overpaints
/// the lines.
fn createGraphicsPair(
    vkd: anytype,
    device: vk.Device,
    render_pass: vk.RenderPass,
    layout: vk.PipelineLayout,
    samples: vk.SampleCountFlags,
    spec: pipeline.SpecInfo,
    vert_spv: []const u8,
    frag_spv: []const u8,
) !GraphicsPipelinePair {
    const filled = try pipeline.createPipeline(&vkd, device, .{
        .vert_spv = vert_spv,
        .frag_spv = frag_spv,
        .layout = layout,
        .render_pass = render_pass,
        .samples = samples,
        .vert_spec = spec,
        .frag_spec = spec,
    });
    errdefer vkd.destroyPipeline(device, filled, null);

    const wireframe = try pipeline.createPipeline(&vkd, device, .{
        .vert_spv = vert_spv,
        .frag_spv = frag_spv,
        .layout = layout,
        .render_pass = render_pass,
        .samples = samples,
        .polygon_mode = .line,
        .cull_mode = .{},
        .vert_spec = spec,
        .frag_spec = spec,
    });

    return .{ .filled = filled, .wireframe = wireframe };
}

/// Build both the default (cheap) and debug (with frag_instance varying +
/// DrawEntries SSBO read in frag) variants.
fn createGraphicsSet(
    vkd: anytype,
    device: vk.Device,
    render_pass: vk.RenderPass,
    layout: vk.PipelineLayout,
    samples: vk.SampleCountFlags,
    ring_size: u32,
    chunk_vertex_dim: u32,
) !GraphicsPipelineSet {
    const spec_blob = TerrainSpec{
        .ring_size = ring_size,
        .chunk_vertex_dim = chunk_vertex_dim,
    };
    const spec: pipeline.SpecInfo = .{
        .entries = &terrain_spec_entries,
        .data = std.mem.asBytes(&spec_blob),
    };

    const default_pair = try createGraphicsPair(vkd, device, render_pass, layout, samples, spec, &clipmap_vert_spv, &terrain_frag_spv);
    errdefer {
        vkd.destroyPipeline(device, default_pair.wireframe, null);
        vkd.destroyPipeline(device, default_pair.filled, null);
    }

    const debug_pair = try createGraphicsPair(vkd, device, render_pass, layout, samples, spec, &clipmap_vert_debug_spv, &terrain_frag_debug_spv);

    return .{ .default = default_pair, .debug = debug_pair };
}

/// Rebuild all four pipelines (e.g. after MSAA toggle).
pub fn recreateGraphicsPipeline(
    vkd: anytype,
    device: vk.Device,
    render_pass: vk.RenderPass,
    graphics_pipeline_layout: vk.PipelineLayout,
    samples: vk.SampleCountFlags,
    ring_size: u32,
    chunk_vertex_dim: u32,
) !GraphicsPipelineSet {
    return createGraphicsSet(vkd, device, render_pass, graphics_pipeline_layout, samples, ring_size, chunk_vertex_dim);
}

pub fn createPipelines(
    vkd: anytype,
    device: vk.Device,
    render_pass: vk.RenderPass,
    desc_layout: vk.DescriptorSetLayout,
    samples: vk.SampleCountFlags,
    ring_size: u32,
    chunk_vertex_dim: u32,
) !PipelineResult {
    const compute_push_range = vk.PushConstantRange{
        .stage_flags = .{ .compute_bit = true },
        .offset = 0,
        .size = @sizeOf(ComputePushConstants),
    };
    const compute_pipeline_layout = try vkd.createPipelineLayout(device, &.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&desc_layout),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&compute_push_range),
    }, null);
    errdefer vkd.destroyPipelineLayout(device, compute_pipeline_layout, null);

    const comp_module = try vkd.createShaderModule(device, &.{
        .code_size = comp_spv.len,
        .p_code = @ptrCast(&comp_spv),
    }, null);
    defer vkd.destroyShaderModule(device, comp_module, null);

    var compute_pipelines: [1]vk.Pipeline = undefined;
    const compute_create_infos = [1]vk.ComputePipelineCreateInfo{.{
        .stage = .{
            .stage = .{ .compute_bit = true },
            .module = comp_module,
            .p_name = "main",
        },
        .layout = compute_pipeline_layout,
        .base_pipeline_index = -1,
    }};
    _ = try vkd.createComputePipelines(device, .null_handle, &compute_create_infos, null, &compute_pipelines);
    const compute_pipeline = compute_pipelines[0];
    errdefer vkd.destroyPipeline(device, compute_pipeline, null);

    const graphics_pipeline_layout = try vkd.createPipelineLayout(device, &.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&desc_layout),
    }, null);
    errdefer vkd.destroyPipelineLayout(device, graphics_pipeline_layout, null);

    const set = try createGraphicsSet(vkd, device, render_pass, graphics_pipeline_layout, samples, ring_size, chunk_vertex_dim);

    return .{
        .compute_pipeline = compute_pipeline,
        .compute_pipeline_layout = compute_pipeline_layout,
        .graphics_pipeline_layout = graphics_pipeline_layout,
        .default = set.default,
        .debug = set.debug,
    };
}

