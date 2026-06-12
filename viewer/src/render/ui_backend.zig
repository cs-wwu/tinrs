//! Vulkan backend for the `ui` core. Owns the three screen-space pipelines (flat
//! shapes + SDF strokes/curves + text) and the per-frame mapped vertex buffers,
//! and renders a `ui.DrawList` in three draw passes. This is the only UI file that
//! touches Vulkan; the core (`ui` module) stays GPU-free and liftable. Port to
//! another renderer by rewriting this file against the same `DrawList`.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const Renderer = @import("renderer.zig");
const pipeline = @import("pipeline.zig");
const ui = @import("ui");

const shape_vert_spv align(@alignOf(u32)) = @embedFile("ui_shape_vert").*;
const shape_frag_spv align(@alignOf(u32)) = @embedFile("ui_shape_frag").*;
const sdf_vert_spv align(@alignOf(u32)) = @embedFile("ui_sdf_vert").*;
const sdf_frag_spv align(@alignOf(u32)) = @embedFile("ui_sdf_frag").*;
const text_vert_spv align(@alignOf(u32)) = @embedFile("text_vert").*;
const text_frag_spv align(@alignOf(u32)) = @embedFile("text_frag").*;
const numeric_vert_spv align(@alignOf(u32)) = @embedFile("numeric_vert").*;
const gauge_vert_spv align(@alignOf(u32)) = @embedFile("gauge_vert").*;
const gauge_frag_spv align(@alignOf(u32)) = @embedFile("gauge_frag").*;

const FRAMES_IN_FLIGHT = Renderer.MAX_FRAMES_IN_FLIGHT;

const SHAPE_VB_SIZE: vk.DeviceSize = ui.MAX_SHAPE_VERTS * @sizeOf(ui.ShapeVertex);
const SDF_VB_SIZE: vk.DeviceSize = ui.MAX_SDF_VERTS * @sizeOf(ui.SdfVertex);
const GLYPH_VB_SIZE: vk.DeviceSize = ui.MAX_GLYPH_VERTS * @sizeOf(ui.GlyphVertex);
const NUMERIC_VB_SIZE: vk.DeviceSize = ui.MAX_NUMERIC_VERTS * @sizeOf(ui.NumericVertex);
const GAUGE_VB_SIZE: vk.DeviceSize = ui.MAX_GAUGE_VERTS * @sizeOf(ui.GaugeVertex);
const POINT_BUF_SIZE: vk.DeviceSize = ui.MAX_PATH_POINTS * @sizeOf([2]f32);

/// Push constants for all UI pipelines: screen_size (vertex stage, pixel->NDC)
/// plus the swapchain transfer function + HUD paper-white (fragment stage, color
/// encoding).
const PushConstants = extern struct {
    screen_w: f32,
    screen_h: f32,
    transfer_function: u32,
    ui_paper_white: f32,
    // Unit scale for the GPU-decoded numeric readout (1.0 = m, 3.28084 = ft).
    // Read only by numeric.vert; the other pipelines' shaders declare a shorter
    // prefix of this block and ignore the trailing bytes, so the wider range is
    // harmless (it stays well under the 128-byte guaranteed push-constant limit).
    numeric_scale: f32,
};

/// HUD paper-white target in nits. The UI's SDR-referred colors (1.0 = full) get
/// encoded to this brightness for the swapchain's transfer function, so the HUD
/// reads at a deliberate level in HDR rather than the ~1460 nits the un-encoded
/// path hit by accident. Tune on the live HDR panel; SDR is display-capped.
/// TODO: wire to a settings knob (panel-dependent, like USER_PREF's ui_scale).
const UI_PAPER_WHITE_NITS: f32 = 300.0;

pub const UiBackend = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,

    shape_pipeline: vk.Pipeline,
    shape_layout: vk.PipelineLayout,

    sdf_pipeline: vk.Pipeline,
    sdf_layout: vk.PipelineLayout,
    // Per-frame path-point SSBO for polyline (kind 3) SDF elements: one descriptor
    // set per frame, indexed by frame_index (mirrors the clipmap's per-frame sets).
    point_desc_layout: vk.DescriptorSetLayout,
    point_desc_pool: vk.DescriptorPool,
    point_desc_sets: [FRAMES_IN_FLIGHT]vk.DescriptorSet,

    text_pipeline: vk.Pipeline,
    text_layout: vk.PipelineLayout,
    font_desc_layout: vk.DescriptorSetLayout,
    font_desc_pool: vk.DescriptorPool,
    font_desc_set: vk.DescriptorSet,
    font_buf: Renderer.BufferWithMemory,

    shape_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,
    sdf_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,
    point_bufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,
    glyph_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,

    // Numeric pipeline: GPU-decoded digit slots (numeric.vert + reused text.frag).
    // Its per-frame descriptor set binds the font (binding 0, fragment) plus a
    // per-frame GPU value buffer (binding 1, vertex), so the set is per-frame.
    numeric_pipeline: vk.Pipeline,
    numeric_layout: vk.PipelineLayout,
    numeric_desc_layout: vk.DescriptorSetLayout,
    numeric_desc_pool: vk.DescriptorPool,
    numeric_desc_sets: [FRAMES_IN_FLIGHT]vk.DescriptorSet,
    numeric_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,

    // Gauge pipeline (AGL fill bar): reuses numeric_layout + numeric_desc_sets
    // (only binding 1 = the probe buffer is needed), so just its own pipeline + vbufs.
    gauge_pipeline: vk.Pipeline,
    gauge_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer,

    pub fn init(
        ctx: Renderer.GpuContext,
        render_pass: vk.RenderPass,
        samples: vk.SampleCountFlags,
        // Per-frame GPU buffers holding the numeric readout values (the terrain
        // probe's ProbeOut). Null on procedural runs with no terrain DB; the
        // numeric pipeline then binds the font buffer as a never-drawn placeholder.
        agl_bufs: ?[FRAMES_IN_FLIGHT]vk.Buffer,
    ) !UiBackend {
        const vkd = ctx.vkd;
        const device = ctx.device;

        // ---- Per-frame mapped vertex buffers (shapes + glyphs) ----
        // Separate counters so a glyph-buffer failure still frees the shape
        // buffer created earlier in the same iteration.
        var shape_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var sdf_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var point_bufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var glyph_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var numeric_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var gauge_vbufs: [FRAMES_IN_FLIGHT]Renderer.MappedBuffer = undefined;
        var shapes_made: u32 = 0;
        var sdfs_made: u32 = 0;
        var points_made: u32 = 0;
        var glyphs_made: u32 = 0;
        var numerics_made: u32 = 0;
        var gauges_made: u32 = 0;
        errdefer {
            for (0..shapes_made) |i| Renderer.destroyMappedBuffer(&vkd, device, shape_vbufs[i].buf);
            for (0..sdfs_made) |i| Renderer.destroyMappedBuffer(&vkd, device, sdf_vbufs[i].buf);
            for (0..points_made) |i| Renderer.destroyMappedBuffer(&vkd, device, point_bufs[i].buf);
            for (0..glyphs_made) |i| Renderer.destroyMappedBuffer(&vkd, device, glyph_vbufs[i].buf);
            for (0..numerics_made) |i| Renderer.destroyMappedBuffer(&vkd, device, numeric_vbufs[i].buf);
            for (0..gauges_made) |i| Renderer.destroyMappedBuffer(&vkd, device, gauge_vbufs[i].buf);
        }
        for (0..FRAMES_IN_FLIGHT) |i| {
            shape_vbufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, SHAPE_VB_SIZE, .{ .vertex_buffer_bit = true });
            shapes_made += 1;
            sdf_vbufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, SDF_VB_SIZE, .{ .vertex_buffer_bit = true });
            sdfs_made += 1;
            point_bufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, POINT_BUF_SIZE, .{ .storage_buffer_bit = true });
            points_made += 1;
            glyph_vbufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, GLYPH_VB_SIZE, .{ .vertex_buffer_bit = true });
            glyphs_made += 1;
            numeric_vbufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, NUMERIC_VB_SIZE, .{ .vertex_buffer_bit = true });
            numerics_made += 1;
            gauge_vbufs[i] = try Renderer.createMappedBuffer(&vkd, device, ctx.mem_props, GAUGE_VB_SIZE, .{ .vertex_buffer_bit = true });
            gauges_made += 1;
        }

        // Vertex stage reads screen_size; fragment stage reads transfer_function
        // + ui_paper_white for color-space encoding. All pipelines share it.
        const push_range = vk.PushConstantRange{
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
            .offset = 0,
            .size = @sizeOf(PushConstants),
        };

        // ---- Path-point SSBO descriptor: one set per frame, each bound to that
        // frame's mapped points buffer (written once here, selected by frame_index
        // at record time). The SDF pipeline's fragment stage reads it for polylines.
        const point_desc_layout = try vkd.createDescriptorSetLayout(device, &.{
            .binding_count = 1,
            .p_bindings = @ptrCast(&vk.DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .fragment_bit = true },
            }),
        }, null);
        errdefer vkd.destroyDescriptorSetLayout(device, point_desc_layout, null);

        const point_desc_pool = try vkd.createDescriptorPool(device, &.{
            .max_sets = FRAMES_IN_FLIGHT,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&vk.DescriptorPoolSize{ .type = .storage_buffer, .descriptor_count = FRAMES_IN_FLIGHT }),
        }, null);
        errdefer vkd.destroyDescriptorPool(device, point_desc_pool, null);

        var point_desc_sets: [FRAMES_IN_FLIGHT]vk.DescriptorSet = undefined;
        const point_set_layouts = [_]vk.DescriptorSetLayout{point_desc_layout} ** FRAMES_IN_FLIGHT;
        try vkd.allocateDescriptorSets(device, &.{
            .descriptor_pool = point_desc_pool,
            .descriptor_set_count = FRAMES_IN_FLIGHT,
            .p_set_layouts = &point_set_layouts,
        }, &point_desc_sets);
        for (0..FRAMES_IN_FLIGHT) |i| {
            const info = vk.DescriptorBufferInfo{ .buffer = point_bufs[i].buf.buffer, .offset = 0, .range = POINT_BUF_SIZE };
            vkd.updateDescriptorSets(device, &.{.{
                .dst_set = point_desc_sets[i],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = .storage_buffer,
                .p_buffer_info = @ptrCast(&info),
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            }}, null);
        }

        // ---- Shape pipeline (push constant only, no descriptors) ----
        const shape_layout = try vkd.createPipelineLayout(device, &.{
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, shape_layout, null);

        const shape_pipeline = try createPipeline(&vkd, device, shape_layout, render_pass, samples, &shape_vert_spv, &shape_frag_spv, &shape_bindings, &shape_attrs);
        errdefer vkd.destroyPipeline(device, shape_pipeline, null);

        // ---- SDF pipeline (push constant + the path-point SSBO set for polylines) ----
        const sdf_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&point_desc_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, sdf_layout, null);

        const sdf_pipeline = try createPipeline(&vkd, device, sdf_layout, render_pass, samples, &sdf_vert_spv, &sdf_frag_spv, &sdf_bindings, &sdf_attrs);
        errdefer vkd.destroyPipeline(device, sdf_pipeline, null);

        // ---- Font storage buffer (staged upload to device-local) ----
        const font_buf = try Renderer.uploadBuffer(&vkd, device, ctx.mem_props, ctx.queue, ctx.cmd_pool, &ui.font_8x8, .{ .storage_buffer_bit = true });
        errdefer {
            vkd.destroyBuffer(device, font_buf.buffer, null);
            vkd.freeMemory(device, font_buf.memory, null);
        }

        // ---- Font descriptor (storage buffer -> fragment stage) ----
        const font_desc_layout = try vkd.createDescriptorSetLayout(device, &.{
            .binding_count = 1,
            .p_bindings = @ptrCast(&vk.DescriptorSetLayoutBinding{
                .binding = 0,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .fragment_bit = true },
            }),
        }, null);
        errdefer vkd.destroyDescriptorSetLayout(device, font_desc_layout, null);

        const font_desc_pool = try vkd.createDescriptorPool(device, &.{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&vk.DescriptorPoolSize{ .type = .storage_buffer, .descriptor_count = 1 }),
        }, null);
        errdefer vkd.destroyDescriptorPool(device, font_desc_pool, null);

        var font_desc_set: vk.DescriptorSet = undefined;
        try vkd.allocateDescriptorSets(device, &.{
            .descriptor_pool = font_desc_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&font_desc_layout),
        }, @ptrCast(&font_desc_set));

        const buf_info = vk.DescriptorBufferInfo{ .buffer = font_buf.buffer, .offset = 0, .range = ui.font_8x8.len };
        vkd.updateDescriptorSets(device, &.{.{
            .dst_set = font_desc_set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_buffer_info = @ptrCast(&buf_info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        }}, null);

        // ---- Text pipeline (font descriptor + push constant) ----
        const text_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&font_desc_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, text_layout, null);

        const text_pipeline = try createPipeline(&vkd, device, text_layout, render_pass, samples, &text_vert_spv, &text_frag_spv, &glyph_bindings, &glyph_attrs);
        errdefer vkd.destroyPipeline(device, text_pipeline, null);

        // ---- Numeric pipeline: font at binding 0 (fragment, reused by text.frag)
        // + a per-frame GPU value buffer at binding 1 (vertex). The gauge pipeline
        // reuses this exact layout + these sets, so binding 1 is read by BOTH
        // numeric.vert and gauge.vert: keep them in sync if this layout changes.
        const numeric_set_bindings = [2]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .fragment_bit = true } },
            .{ .binding = 1, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .vertex_bit = true } },
        };
        const numeric_desc_layout = try vkd.createDescriptorSetLayout(device, &.{
            .binding_count = numeric_set_bindings.len,
            .p_bindings = &numeric_set_bindings,
        }, null);
        errdefer vkd.destroyDescriptorSetLayout(device, numeric_desc_layout, null);

        const numeric_desc_pool = try vkd.createDescriptorPool(device, &.{
            .max_sets = FRAMES_IN_FLIGHT,
            .pool_size_count = 1,
            .p_pool_sizes = @ptrCast(&vk.DescriptorPoolSize{ .type = .storage_buffer, .descriptor_count = 2 * FRAMES_IN_FLIGHT }),
        }, null);
        errdefer vkd.destroyDescriptorPool(device, numeric_desc_pool, null);

        var numeric_desc_sets: [FRAMES_IN_FLIGHT]vk.DescriptorSet = undefined;
        const numeric_set_layouts = [_]vk.DescriptorSetLayout{numeric_desc_layout} ** FRAMES_IN_FLIGHT;
        try vkd.allocateDescriptorSets(device, &.{
            .descriptor_pool = numeric_desc_pool,
            .descriptor_set_count = FRAMES_IN_FLIGHT,
            .p_set_layouts = &numeric_set_layouts,
        }, &numeric_desc_sets);
        for (0..FRAMES_IN_FLIGHT) |i| {
            const font_info = vk.DescriptorBufferInfo{ .buffer = font_buf.buffer, .offset = 0, .range = ui.font_8x8.len };
            // No probe (procedural run): bind the font buffer as a harmless,
            // never-drawn placeholder (the HUD emits no digit slots without a DB).
            const val_buf = if (agl_bufs) |b| b[i] else font_buf.buffer;
            const val_info = vk.DescriptorBufferInfo{ .buffer = val_buf, .offset = 0, .range = vk.WHOLE_SIZE };
            const writes = [2]vk.WriteDescriptorSet{
                .{ .dst_set = numeric_desc_sets[i], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&font_info), .p_image_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = numeric_desc_sets[i], .dst_binding = 1, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&val_info), .p_image_info = undefined, .p_texel_buffer_view = undefined },
            };
            vkd.updateDescriptorSets(device, &writes, null);
        }

        const numeric_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&numeric_desc_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, numeric_layout, null);

        const numeric_pipeline = try createPipeline(&vkd, device, numeric_layout, render_pass, samples, &numeric_vert_spv, &text_frag_spv, &numeric_bindings, &numeric_attrs);
        errdefer vkd.destroyPipeline(device, numeric_pipeline, null);

        // Gauge pipeline reuses numeric_layout (same set: binding 1 = probe value).
        const gauge_pipeline = try createPipeline(&vkd, device, numeric_layout, render_pass, samples, &gauge_vert_spv, &gauge_frag_spv, &gauge_bindings, &gauge_attrs);
        errdefer vkd.destroyPipeline(device, gauge_pipeline, null);

        return .{
            .vkd = vkd,
            .device = device,
            .shape_pipeline = shape_pipeline,
            .shape_layout = shape_layout,
            .sdf_pipeline = sdf_pipeline,
            .sdf_layout = sdf_layout,
            .point_desc_layout = point_desc_layout,
            .point_desc_pool = point_desc_pool,
            .point_desc_sets = point_desc_sets,
            .text_pipeline = text_pipeline,
            .text_layout = text_layout,
            .font_desc_layout = font_desc_layout,
            .font_desc_pool = font_desc_pool,
            .font_desc_set = font_desc_set,
            .font_buf = font_buf,
            .shape_vbufs = shape_vbufs,
            .sdf_vbufs = sdf_vbufs,
            .point_bufs = point_bufs,
            .glyph_vbufs = glyph_vbufs,
            .numeric_pipeline = numeric_pipeline,
            .numeric_layout = numeric_layout,
            .numeric_desc_layout = numeric_desc_layout,
            .numeric_desc_pool = numeric_desc_pool,
            .numeric_desc_sets = numeric_desc_sets,
            .numeric_vbufs = numeric_vbufs,
            .gauge_pipeline = gauge_pipeline,
            .gauge_vbufs = gauge_vbufs,
        };
    }

    pub fn recreatePipelines(self: *UiBackend, render_pass: vk.RenderPass, samples: vk.SampleCountFlags) !void {
        // Build the replacements before destroying the old ones so a failure
        // leaves the existing pipelines intact (no dangling handle to double-free).
        const shape = try createPipeline(&self.vkd, self.device, self.shape_layout, render_pass, samples, &shape_vert_spv, &shape_frag_spv, &shape_bindings, &shape_attrs);
        const sdf = try createPipeline(&self.vkd, self.device, self.sdf_layout, render_pass, samples, &sdf_vert_spv, &sdf_frag_spv, &sdf_bindings, &sdf_attrs);
        const text = try createPipeline(&self.vkd, self.device, self.text_layout, render_pass, samples, &text_vert_spv, &text_frag_spv, &glyph_bindings, &glyph_attrs);
        const numeric = try createPipeline(&self.vkd, self.device, self.numeric_layout, render_pass, samples, &numeric_vert_spv, &text_frag_spv, &numeric_bindings, &numeric_attrs);
        const gauge = try createPipeline(&self.vkd, self.device, self.numeric_layout, render_pass, samples, &gauge_vert_spv, &gauge_frag_spv, &gauge_bindings, &gauge_attrs);
        self.vkd.destroyPipeline(self.device, self.shape_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.sdf_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.text_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.numeric_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.gauge_pipeline, null);
        self.shape_pipeline = shape;
        self.sdf_pipeline = sdf;
        self.text_pipeline = text;
        self.numeric_pipeline = numeric;
        self.gauge_pipeline = gauge;
    }

    pub fn deinit(self: *UiBackend) void {
        self.vkd.deviceWaitIdle(self.device) catch {};
        self.vkd.destroyPipeline(self.device, self.shape_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.shape_layout, null);
        self.vkd.destroyPipeline(self.device, self.sdf_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.sdf_layout, null);
        self.vkd.destroyPipeline(self.device, self.text_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.text_layout, null);
        self.vkd.destroyPipeline(self.device, self.numeric_pipeline, null);
        self.vkd.destroyPipeline(self.device, self.gauge_pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.numeric_layout, null);
        self.vkd.destroyDescriptorPool(self.device, self.font_desc_pool, null);
        self.vkd.destroyDescriptorSetLayout(self.device, self.font_desc_layout, null);
        self.vkd.destroyDescriptorPool(self.device, self.point_desc_pool, null);
        self.vkd.destroyDescriptorSetLayout(self.device, self.point_desc_layout, null);
        self.vkd.destroyDescriptorPool(self.device, self.numeric_desc_pool, null);
        self.vkd.destroyDescriptorSetLayout(self.device, self.numeric_desc_layout, null);
        // TODO: this destroyBuffer+freeMemory pair on a BufferWithMemory is
        // open-coded here and in buffer.zig (uploadBuffer/createBuffer errdefers).
        // Add a buffer.destroyBufferWithMemory helper mirroring destroyMappedBuffer
        // and call it from all sites.
        self.vkd.destroyBuffer(self.device, self.font_buf.buffer, null);
        self.vkd.freeMemory(self.device, self.font_buf.memory, null);
        for (0..FRAMES_IN_FLIGHT) |i| {
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.shape_vbufs[i].buf);
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.sdf_vbufs[i].buf);
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.point_bufs[i].buf);
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.glyph_vbufs[i].buf);
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.numeric_vbufs[i].buf);
            Renderer.destroyMappedBuffer(&self.vkd, self.device, self.gauge_vbufs[i].buf);
        }
    }

    /// Upload `dl`'s geometry into this frame's buffers and record the draws:
    /// flat shapes first (behind), then SDF strokes/curves, then glyphs (on top),
    /// so text always sits over panels and strokes over fills. Each stream is
    /// sub-divided by the draw list's clip groups:
    /// within each pipeline pass we loop the groups, setting a scissor per group
    /// and drawing only that group's vertex range. `frame_index` must match the
    /// renderer's current_frame to avoid CPU/GPU races.
    pub fn record(self: *const UiBackend, cmd: vk.CommandBuffer, dl: *const ui.DrawList, frame_index: u32, extent: vk.Extent2D, transfer_function: u32, numeric_scale: f32) void {
        const push = PushConstants{
            .screen_w = @floatFromInt(extent.width),
            .screen_h = @floatFromInt(extent.height),
            .transfer_function = transfer_function,
            .ui_paper_white = UI_PAPER_WHITE_NITS,
            .numeric_scale = numeric_scale,
        };
        const offsets = [_]vk.DeviceSize{0};
        const full = vk.Rect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = extent };

        var ranges: [ui.MAX_CLIP_GROUPS]ui.ResolvedClip = undefined;
        const group_count = dl.resolveClipRanges(&ranges);

        // Resolve each clip group's scissor once; both pipeline passes index this.
        var scissors: [ui.MAX_CLIP_GROUPS]vk.Rect2D = undefined;
        for (ranges[0..group_count], 0..) |g, i| {
            scissors[i] = if (g.clip) |c| clampScissor(c, extent) else full;
        }

        if (dl.shape_count > 0) {
            const bytes = std.mem.sliceAsBytes(dl.shape_verts[0..dl.shape_count]);
            @memcpy(self.shape_vbufs[frame_index].map[0..bytes.len], bytes);

            self.vkd.cmdBindPipeline(cmd, .graphics, self.shape_pipeline);
            const vbufs = [1]vk.Buffer{self.shape_vbufs[frame_index].buf.buffer};
            self.vkd.cmdBindVertexBuffers(cmd, 0, &vbufs, &offsets);
            self.vkd.cmdPushConstants(cmd, self.shape_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&push));
            for (ranges[0..group_count], 0..) |g, i| {
                if (g.shape_count == 0) continue;
                const sc = [1]vk.Rect2D{scissors[i]};
                self.vkd.cmdSetScissor(cmd, 0, &sc);
                self.vkd.cmdDraw(cmd, g.shape_count, 1, g.shape_first, 0);
            }
        }

        if (dl.sdf_count > 0) {
            const bytes = std.mem.sliceAsBytes(dl.sdf_verts[0..dl.sdf_count]);
            @memcpy(self.sdf_vbufs[frame_index].map[0..bytes.len], bytes);
            // Upload this frame's polyline path points for the SDF frag's SSBO read.
            if (dl.path_count > 0) {
                const pbytes = std.mem.sliceAsBytes(dl.path_points[0..dl.path_count]);
                @memcpy(self.point_bufs[frame_index].map[0..pbytes.len], pbytes);
            }

            self.vkd.cmdBindPipeline(cmd, .graphics, self.sdf_pipeline);
            // Bind the points set unconditionally: the shader declares the SSBO even
            // for non-polyline kinds, which simply never index it.
            const point_sets = [1]vk.DescriptorSet{self.point_desc_sets[frame_index]};
            self.vkd.cmdBindDescriptorSets(cmd, .graphics, self.sdf_layout, 0, &point_sets, null);
            const vbufs = [1]vk.Buffer{self.sdf_vbufs[frame_index].buf.buffer};
            self.vkd.cmdBindVertexBuffers(cmd, 0, &vbufs, &offsets);
            self.vkd.cmdPushConstants(cmd, self.sdf_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&push));
            for (ranges[0..group_count], 0..) |g, i| {
                if (g.sdf_count == 0) continue;
                const sc = [1]vk.Rect2D{scissors[i]};
                self.vkd.cmdSetScissor(cmd, 0, &sc);
                self.vkd.cmdDraw(cmd, g.sdf_count, 1, g.sdf_first, 0);
            }
        }

        if (dl.glyph_count > 0) {
            const bytes = std.mem.sliceAsBytes(dl.glyph_verts[0..dl.glyph_count]);
            @memcpy(self.glyph_vbufs[frame_index].map[0..bytes.len], bytes);

            self.vkd.cmdBindPipeline(cmd, .graphics, self.text_pipeline);
            const desc_sets = [1]vk.DescriptorSet{self.font_desc_set};
            self.vkd.cmdBindDescriptorSets(cmd, .graphics, self.text_layout, 0, &desc_sets, null);
            const vbufs = [1]vk.Buffer{self.glyph_vbufs[frame_index].buf.buffer};
            self.vkd.cmdBindVertexBuffers(cmd, 0, &vbufs, &offsets);
            self.vkd.cmdPushConstants(cmd, self.text_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&push));
            for (ranges[0..group_count], 0..) |g, i| {
                if (g.glyph_count == 0) continue;
                const sc = [1]vk.Rect2D{scissors[i]};
                self.vkd.cmdSetScissor(cmd, 0, &sc);
                self.vkd.cmdDraw(cmd, g.glyph_count, 1, g.glyph_first, 0);
            }
        }

        // Numeric digit slots: GPU-decoded, full-screen (not clip-tracked), drawn
        // last so they sit over the glyph readouts. binding 1 = this frame's value
        // buffer, bound via the per-frame descriptor set written at init.
        if (dl.numeric_count > 0) {
            const bytes = std.mem.sliceAsBytes(dl.numeric_verts[0..dl.numeric_count]);
            @memcpy(self.numeric_vbufs[frame_index].map[0..bytes.len], bytes);

            self.vkd.cmdBindPipeline(cmd, .graphics, self.numeric_pipeline);
            const dsets = [1]vk.DescriptorSet{self.numeric_desc_sets[frame_index]};
            self.vkd.cmdBindDescriptorSets(cmd, .graphics, self.numeric_layout, 0, &dsets, null);
            const vbufs = [1]vk.Buffer{self.numeric_vbufs[frame_index].buf.buffer};
            self.vkd.cmdBindVertexBuffers(cmd, 0, &vbufs, &offsets);
            self.vkd.cmdPushConstants(cmd, self.numeric_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&push));
            const sc = [1]vk.Rect2D{full};
            self.vkd.cmdSetScissor(cmd, 0, &sc);
            self.vkd.cmdDraw(cmd, dl.numeric_count, 1, 0, 0);
        }

        // Gauge fills: GPU-driven, full-screen (not clip-tracked). Reuse the
        // numeric descriptor set (binding 1 = probe value); gauge.frag is solid.
        if (dl.gauge_count > 0) {
            const bytes = std.mem.sliceAsBytes(dl.gauge_verts[0..dl.gauge_count]);
            @memcpy(self.gauge_vbufs[frame_index].map[0..bytes.len], bytes);

            self.vkd.cmdBindPipeline(cmd, .graphics, self.gauge_pipeline);
            const dsets = [1]vk.DescriptorSet{self.numeric_desc_sets[frame_index]};
            self.vkd.cmdBindDescriptorSets(cmd, .graphics, self.numeric_layout, 0, &dsets, null);
            const vbufs = [1]vk.Buffer{self.gauge_vbufs[frame_index].buf.buffer};
            self.vkd.cmdBindVertexBuffers(cmd, 0, &vbufs, &offsets);
            self.vkd.cmdPushConstants(cmd, self.numeric_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(PushConstants), @ptrCast(&push));
            const sc = [1]vk.Rect2D{full};
            self.vkd.cmdSetScissor(cmd, 0, &sc);
            self.vkd.cmdDraw(cmd, dl.gauge_count, 1, 0, 0);
        }

        // Restore the full-framebuffer scissor so anything recorded after the UI
        // (today nothing; future-proofing) inherits a sane rect.
        const restore = [1]vk.Rect2D{full};
        self.vkd.cmdSetScissor(cmd, 0, &restore);
    }
};

/// Convert a `ui.ClipRect` (which may extend past the framebuffer) into a Vulkan
/// scissor clamped to `extent`. Clamping is mandatory: Vulkan requires the scissor
/// to lie within the framebuffer, and a center-anchored HUD element easily yields
/// a negative x or an overshooting width. A fully off-screen rect collapses to
/// zero area (the caller skips zero-count draws anyway).
fn clampScissor(clip: ui.ClipRect, extent: vk.Extent2D) vk.Rect2D {
    const fb_w: i64 = extent.width;
    const fb_h: i64 = extent.height;
    const x0 = std.math.clamp(@as(i64, clip.x), 0, fb_w);
    const y0 = std.math.clamp(@as(i64, clip.y), 0, fb_h);
    const x1 = std.math.clamp(@as(i64, clip.x) + @as(i64, clip.w), 0, fb_w);
    const y1 = std.math.clamp(@as(i64, clip.y) + @as(i64, clip.h), 0, fb_h);
    return .{
        .offset = .{ .x = @intCast(x0), .y = @intCast(y0) },
        .extent = .{ .width = @intCast(@max(0, x1 - x0)), .height = @intCast(@max(0, y1 - y0)) },
    };
}

// =============================================================================
// Pipeline creation
// =============================================================================

const shape_bindings = [_]vk.VertexInputBindingDescription{.{ .binding = 0, .stride = @sizeOf(ui.ShapeVertex), .input_rate = .vertex }};
const shape_attrs = [_]vk.VertexInputAttributeDescription{
    .{ .binding = 0, .location = 0, .format = .r32g32_sfloat, .offset = 0 },
    .{ .binding = 0, .location = 1, .format = .r32g32b32a32_sfloat, .offset = 8 },
};

// Mirrors ui.SdfVertex: pos | ab (capsule A/B or box center+half-extent) |
// radius,border | kind | color.
const sdf_bindings = [_]vk.VertexInputBindingDescription{.{ .binding = 0, .stride = @sizeOf(ui.SdfVertex), .input_rate = .vertex }};
const sdf_attrs = [_]vk.VertexInputAttributeDescription{
    .{ .binding = 0, .location = 0, .format = .r32g32_sfloat, .offset = 0 },
    .{ .binding = 0, .location = 1, .format = .r32g32b32a32_sfloat, .offset = 8 },
    .{ .binding = 0, .location = 2, .format = .r32g32_sfloat, .offset = 24 },
    .{ .binding = 0, .location = 3, .format = .r32_sfloat, .offset = 32 },
    .{ .binding = 0, .location = 4, .format = .r32g32b32a32_sfloat, .offset = 36 },
};

const glyph_bindings = [_]vk.VertexInputBindingDescription{.{ .binding = 0, .stride = @sizeOf(ui.GlyphVertex), .input_rate = .vertex }};
const glyph_attrs = [_]vk.VertexInputAttributeDescription{
    .{ .binding = 0, .location = 0, .format = .r32g32_sfloat, .offset = 0 },
    .{ .binding = 0, .location = 1, .format = .r32g32_sfloat, .offset = 8 },
    .{ .binding = 0, .location = 2, .format = .r32g32b32a32_sfloat, .offset = 16 },
    .{ .binding = 0, .location = 3, .format = .r32_uint, .offset = 32 },
};

// Mirrors ui.NumericVertex: pos | uv | color | place (u32). Same layout as the
// glyph stream, but location 3 is the decimal place, not a char index.
const numeric_bindings = [_]vk.VertexInputBindingDescription{.{ .binding = 0, .stride = @sizeOf(ui.NumericVertex), .input_rate = .vertex }};
const numeric_attrs = [_]vk.VertexInputAttributeDescription{
    .{ .binding = 0, .location = 0, .format = .r32g32_sfloat, .offset = 0 },
    .{ .binding = 0, .location = 1, .format = .r32g32_sfloat, .offset = 8 },
    .{ .binding = 0, .location = 2, .format = .r32g32b32a32_sfloat, .offset = 16 },
    .{ .binding = 0, .location = 3, .format = .r32_uint, .offset = 32 },
};

// Mirrors ui.GaugeVertex: (x, y_bottom) | (y_top, fill_weight) | color.
const gauge_bindings = [_]vk.VertexInputBindingDescription{.{ .binding = 0, .stride = @sizeOf(ui.GaugeVertex), .input_rate = .vertex }};
const gauge_attrs = [_]vk.VertexInputAttributeDescription{
    .{ .binding = 0, .location = 0, .format = .r32g32_sfloat, .offset = 0 },
    .{ .binding = 0, .location = 1, .format = .r32g32_sfloat, .offset = 8 },
    .{ .binding = 0, .location = 2, .format = .r32g32b32a32_sfloat, .offset = 16 },
};

/// Both UI pipelines share the same screen-space render state (no cull, no
/// depth, alpha blend); they differ only in shaders and vertex layout.
fn createPipeline(
    vkd: *const vk.DeviceWrapper,
    device: vk.Device,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    samples: vk.SampleCountFlags,
    vert_spv: []const u8,
    frag_spv: []const u8,
    bindings: []const vk.VertexInputBindingDescription,
    attrs: []const vk.VertexInputAttributeDescription,
) !vk.Pipeline {
    return pipeline.createPipeline(vkd, device, .{
        .vert_spv = vert_spv,
        .frag_spv = frag_spv,
        .layout = layout,
        .render_pass = render_pass,
        .samples = samples,
        .vertex_bindings = bindings,
        .vertex_attrs = attrs,
        .cull_mode = .{},
        .depth_test = false,
        .depth_write = false,
        // Blends in the swapchain's encoded (PQ / scaled-scRGB) space, not linear,
        // since the frag shaders encode before this fixed-function blend.
        // TODO: HDR UI compositing in linear (composite into a linear HDR target).
        .blend = .alpha,
    });
}
