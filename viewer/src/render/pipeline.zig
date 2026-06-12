//! Unified graphics-pipeline factory.
//!
//! All three viewer subsystems (terrain, sky, text) used to maintain
//! near-identical 80-line pipeline create functions that differed only in
//! vertex input, cull, front face, depth, and blend. `createPipeline` takes a
//! `PipelineOpts` struct whose defaults match the terrain pipeline; sky and
//! text override the few fields they care about.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;

pub const BlendMode = enum {
    /// Opaque: blend_enable = false. Output replaces destination.
    none,
    /// Standard alpha blend: src.a + (1 - src.a).
    alpha,
};

/// Field defaults match the opaque terrain pipeline (back-cull CCW, reverse-Z
/// depth test+write greater_or_equal, no blend, no vertex input). Sky and text
/// override only the few fields they need.
pub const PipelineOpts = struct {
    vert_spv: []const u8,
    frag_spv: []const u8,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    samples: vk.SampleCountFlags,

    vertex_bindings: []const vk.VertexInputBindingDescription = &.{},
    vertex_attrs: []const vk.VertexInputAttributeDescription = &.{},

    cull_mode: vk.CullModeFlags = .{ .back_bit = true },
    front_face: vk.FrontFace = .counter_clockwise,

    depth_test: bool = true,
    depth_write: bool = true,
    depth_compare: vk.CompareOp = .greater_or_equal,

    polygon_mode: vk.PolygonMode = .fill,

    blend: BlendMode = .none,

    /// Slices must outlive the createPipeline call; Vulkan reads through
    /// the pointers during pipeline build.
    vert_spec: SpecInfo = .{},
    frag_spec: SpecInfo = .{},
};

pub const SpecInfo = struct {
    entries: []const vk.SpecializationMapEntry = &.{},
    data: []const u8 = &.{},
};

fn makeSpecInfo(s: SpecInfo) ?vk.SpecializationInfo {
    if (s.entries.len == 0) return null;
    return .{
        .map_entry_count = @intCast(s.entries.len),
        .p_map_entries = s.entries.ptr,
        .data_size = s.data.len,
        .p_data = s.data.ptr,
    };
}

pub fn createPipeline(
    vkd: *const vk.DeviceWrapper,
    device: vk.Device,
    opts: PipelineOpts,
) !vk.Pipeline {
    const vert_module = try vkd.createShaderModule(device, &.{
        .code_size = opts.vert_spv.len,
        .p_code = @ptrCast(@alignCast(opts.vert_spv.ptr)),
    }, null);
    defer vkd.destroyShaderModule(device, vert_module, null);

    const frag_module = try vkd.createShaderModule(device, &.{
        .code_size = opts.frag_spv.len,
        .p_code = @ptrCast(@alignCast(opts.frag_spv.ptr)),
    }, null);
    defer vkd.destroyShaderModule(device, frag_module, null);

    const vert_spec = makeSpecInfo(opts.vert_spec);
    const frag_spec = makeSpecInfo(opts.frag_spec);

    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .stage = .{ .vertex_bit = true },
            .module = vert_module,
            .p_name = "main",
            .p_specialization_info = if (vert_spec) |*s| s else null,
        },
        .{
            .stage = .{ .fragment_bit = true },
            .module = frag_module,
            .p_name = "main",
            .p_specialization_info = if (frag_spec) |*s| s else null,
        },
    };

    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
    const FALSE = @as(vk.Bool32, .false);
    const TRUE = @as(vk.Bool32, .true);

    const blend_attachment: vk.PipelineColorBlendAttachmentState = switch (opts.blend) {
        .none => .{
            .blend_enable = FALSE,
            .src_color_blend_factor = .one,
            .dst_color_blend_factor = .zero,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        },
        .alpha => .{
            .blend_enable = TRUE,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one_minus_src_alpha,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
        },
    };

    const multisample_state = vk.PipelineMultisampleStateCreateInfo{
        .rasterization_samples = opts.samples,
        .sample_shading_enable = FALSE,
        .min_sample_shading = 1.0,
        .alpha_to_coverage_enable = FALSE,
        .alpha_to_one_enable = FALSE,
    };

    const vertex_input = vk.PipelineVertexInputStateCreateInfo{
        .vertex_binding_description_count = @intCast(opts.vertex_bindings.len),
        .p_vertex_binding_descriptions = if (opts.vertex_bindings.len == 0) null else opts.vertex_bindings.ptr,
        .vertex_attribute_description_count = @intCast(opts.vertex_attrs.len),
        .p_vertex_attribute_descriptions = if (opts.vertex_attrs.len == 0) null else opts.vertex_attrs.ptr,
    };

    var pipelines: [1]vk.Pipeline = undefined;
    const create_infos = [1]vk.GraphicsPipelineCreateInfo{.{
        .stage_count = shader_stages.len,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &vertex_input,
        .p_input_assembly_state = &.{ .topology = .triangle_list, .primitive_restart_enable = FALSE },
        .p_viewport_state = &.{ .viewport_count = 1, .scissor_count = 1 },
        .p_rasterization_state = &.{
            .depth_clamp_enable = FALSE,
            .rasterizer_discard_enable = FALSE,
            .polygon_mode = opts.polygon_mode,
            .cull_mode = opts.cull_mode,
            .front_face = opts.front_face,
            .depth_bias_enable = FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1.0,
        },
        .p_multisample_state = &multisample_state,
        .p_depth_stencil_state = &.{
            .depth_test_enable = if (opts.depth_test) TRUE else FALSE,
            .depth_write_enable = if (opts.depth_write) TRUE else FALSE,
            .depth_compare_op = opts.depth_compare,
            .depth_bounds_test_enable = FALSE,
            .stencil_test_enable = FALSE,
            .front = std.mem.zeroes(vk.StencilOpState),
            .back = std.mem.zeroes(vk.StencilOpState),
            .min_depth_bounds = 0,
            .max_depth_bounds = 1,
        },
        .p_color_blend_state = &.{
            .logic_op_enable = FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&blend_attachment),
            .blend_constants = .{ 0, 0, 0, 0 },
        },
        .p_dynamic_state = &.{
            .dynamic_state_count = dynamic_states.len,
            .p_dynamic_states = &dynamic_states,
        },
        .layout = opts.layout,
        .render_pass = opts.render_pass,
        .subpass = 0,
        .base_pipeline_index = -1,
    }};
    _ = try vkd.createGraphicsPipelines(device, .null_handle, &create_infos, null, &pipelines);
    return pipelines[0];
}
