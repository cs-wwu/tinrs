const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const renderer_mod = @import("renderer.zig");
const pipeline = @import("pipeline.zig");

const sky_vert_spv align(@alignOf(u32)) = @embedFile("sky_vert").*;
const sky_frag_spv align(@alignOf(u32)) = @embedFile("sky_frag").*;

pub const Sky = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,

    pub fn init(
        ctx: renderer_mod.GpuContext,
        render_pass: vk.RenderPass,
        desc_layout: vk.DescriptorSetLayout,
        samples: vk.SampleCountFlags,
    ) !Sky {
        const vkd = ctx.vkd;
        const device = ctx.device;
        const pipeline_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&desc_layout),
            .push_constant_range_count = 0,
            .p_push_constant_ranges = undefined,
        }, null);
        errdefer vkd.destroyPipelineLayout(device, pipeline_layout, null);

        const sky_pipeline = try createSkyPipeline(&vkd, device, pipeline_layout, render_pass, samples);

        return .{
            .vkd = vkd,
            .device = device,
            .pipeline = sky_pipeline,
            .pipeline_layout = pipeline_layout,
        };
    }

    pub fn recreatePipeline(self: *Sky, render_pass: vk.RenderPass, samples: vk.SampleCountFlags) !void {
        self.vkd.destroyPipeline(self.device, self.pipeline, null);
        self.pipeline = try createSkyPipeline(&self.vkd, self.device, self.pipeline_layout, render_pass, samples);
    }

    pub fn deinit(self: *Sky) void {
        self.vkd.destroyPipeline(self.device, self.pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
    }

    pub fn draw(self: *const Sky, cmd: vk.CommandBuffer, desc_set: vk.DescriptorSet) void {
        self.vkd.cmdBindPipeline(cmd, .graphics, self.pipeline);
        const sets = [1]vk.DescriptorSet{desc_set};
        self.vkd.cmdBindDescriptorSets(cmd, .graphics, self.pipeline_layout, 0, &sets, null);
        self.vkd.cmdDraw(cmd, 3, 1, 0, 0);
    }
};

fn createSkyPipeline(
    vkd: *const vk.DeviceWrapper,
    device: vk.Device,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    samples: vk.SampleCountFlags,
) !vk.Pipeline {
    return pipeline.createPipeline(vkd, device, .{
        .vert_spv = &sky_vert_spv,
        .frag_spv = &sky_frag_spv,
        .layout = layout,
        .render_pass = render_pass,
        .samples = samples,
        .cull_mode = .{},
        .depth_write = false,
        .depth_compare = .greater_or_equal,
    });
}
