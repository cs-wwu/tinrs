//! Terrain probe: a tiny compute pass that evaluates the terrain INR at a single
//! world point (the aircraft, straight down) and writes the height plus a
//! display-ready AGL to a small GPU buffer. Shares the exact INR eval the clipmap
//! uses (shaders/common/terrain_inr.glsl via shaders/terrain/probe_eval.comp), so
//! the probed height can't disagree with the rendered terrain.
//!
//! The result stays GPU-resident: a later HUD numeric pipeline reads `agl_m` and
//! decodes its digits in a vertex shader, so the value never round-trips to the
//! CPU. `elevation_m` is the general terrain-height primitive (reusable later for
//! FPM ground-intersection / glide footprint / TAWS).
//!
//! Output buffers are per-frame (x2 frames-in-flight) so the two in-flight frames
//! never alias. Bound against the SAME Weights/DirGrid SSBOs the clipmap uses:
//! those are allocated once by the TileSystem and never reallocated on stream-in
//! (tiles are copied into the pre-sized shell), so writing the descriptors once
//! at init is safe.
//!
//! Imports are kept to math (module) + terrain siblings + render helpers already
//! in the terrain compilation; it does NOT import app/ (recordDispatch takes a
//! plain math.Vec3d), so no app test blocks leak into the terrain test binary.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const math = @import("math");
const buffer = @import("../render/buffer.zig");
const renderer_mod = @import("../render/renderer.zig");
const tile_system_mod = @import("tile_system.zig");
const coords = @import("coords.zig");

const probe_spv align(@alignOf(u32)) = @embedFile("probe_eval_shader").*;

/// CPU mirror of the std430 `ProbeOut` block in shaders/terrain/probe_eval.comp.
/// The CPU never reads the contents (the value stays GPU-resident); this exists
/// so the buffer size and field offsets have one typed source of truth instead of
/// a magic number. Chunk B's HUD reads `agl_m` at `@offsetOf(ProbeOut, "agl_m")`.
pub const ProbeOut = extern struct {
    elevation_m: f32,
    water_logit: f32,
    agl_m: i32,
    tile_found: i32,
};
const PROBE_OUT_SIZE: vk.DeviceSize = @sizeOf(ProbeOut);

const FRAMES: usize = 2;

/// Push constants for probe_eval.comp; must match its `PC` block byte layout
/// (vec2 at offset 0, float at offset 8 = 12 bytes).
pub const ProbePush = extern struct {
    aircraft_world: [2]f32,
    observer_msl_m: f32,
};

pub const Probe = struct {
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    desc_layout: vk.DescriptorSetLayout,
    desc_pool: vk.DescriptorPool,
    desc_sets: [FRAMES]vk.DescriptorSet,
    out_bufs: [FRAMES]buffer.BufferWithMemory,
    pipeline_layout: vk.PipelineLayout,
    pipeline: vk.Pipeline,

    pub fn init(ctx: renderer_mod.GpuContext, tile_system: *tile_system_mod.TileSystem) !Probe {
        const vkd = ctx.vkd;
        const device = ctx.device;
        const mem_props = ctx.mem_props;

        // Per-frame device-local output buffers (compute writes, vertex reads).
        var out_bufs: [FRAMES]buffer.BufferWithMemory = undefined;
        var out_count: usize = 0;
        errdefer for (0..out_count) |i| {
            vkd.destroyBuffer(device, out_bufs[i].buffer, null);
            vkd.freeMemory(device, out_bufs[i].memory, null);
        };
        for (0..FRAMES) |i| {
            out_bufs[i] = try buffer.createBuffer(
                &vkd,
                device,
                mem_props,
                PROBE_OUT_SIZE,
                .{ .storage_buffer_bit = true },
                .{ .device_local_bit = true },
            );
            out_count += 1;
        }

        // Descriptor layout: 0 = Weights, 1 = DirGrid, 2 = ProbeOut (all compute).
        const bindings = [3]vk.DescriptorSetLayoutBinding{
            .{ .binding = 0, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true } },
            .{ .binding = 1, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true } },
            .{ .binding = 2, .descriptor_type = .storage_buffer, .descriptor_count = 1, .stage_flags = .{ .compute_bit = true } },
        };
        const desc_layout = try vkd.createDescriptorSetLayout(device, &.{
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);
        errdefer vkd.destroyDescriptorSetLayout(device, desc_layout, null);

        // 3 storage bindings * 2 sets = 6 storage descriptors.
        const pool_sizes = [1]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = 6 },
        };
        const desc_pool = try vkd.createDescriptorPool(device, &.{
            .max_sets = FRAMES,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null);
        errdefer vkd.destroyDescriptorPool(device, desc_pool, null);

        const set_layouts = [FRAMES]vk.DescriptorSetLayout{ desc_layout, desc_layout };
        var desc_sets: [FRAMES]vk.DescriptorSet = undefined;
        try vkd.allocateDescriptorSets(device, &.{
            .descriptor_pool = desc_pool,
            .descriptor_set_count = FRAMES,
            .p_set_layouts = &set_layouts,
        }, &desc_sets);

        // Weights + DirGrid handles are stable for the TileSystem's lifetime, so
        // these writes are done once.
        const weights_info = vk.DescriptorBufferInfo{ .buffer = tile_system.weightsBuffer(), .offset = 0, .range = vk.WHOLE_SIZE };
        const grid_info = vk.DescriptorBufferInfo{ .buffer = tile_system.dirGridBuffer(), .offset = 0, .range = vk.WHOLE_SIZE };
        for (0..FRAMES) |i| {
            const out_info = vk.DescriptorBufferInfo{ .buffer = out_bufs[i].buffer, .offset = 0, .range = PROBE_OUT_SIZE };
            const writes = [3]vk.WriteDescriptorSet{
                .{ .dst_set = desc_sets[i], .dst_binding = 0, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&weights_info), .p_image_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = desc_sets[i], .dst_binding = 1, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&grid_info), .p_image_info = undefined, .p_texel_buffer_view = undefined },
                .{ .dst_set = desc_sets[i], .dst_binding = 2, .dst_array_element = 0, .descriptor_count = 1, .descriptor_type = .storage_buffer, .p_buffer_info = @ptrCast(&out_info), .p_image_info = undefined, .p_texel_buffer_view = undefined },
            };
            vkd.updateDescriptorSets(device, &writes, null);
        }

        // Compute pipeline.
        const push_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(ProbePush),
        };
        const pipeline_layout = try vkd.createPipelineLayout(device, &.{
            .set_layout_count = 1,
            .p_set_layouts = @ptrCast(&desc_layout),
            .push_constant_range_count = 1,
            .p_push_constant_ranges = @ptrCast(&push_range),
        }, null);
        errdefer vkd.destroyPipelineLayout(device, pipeline_layout, null);

        const module = try vkd.createShaderModule(device, &.{
            .code_size = probe_spv.len,
            .p_code = @ptrCast(&probe_spv),
        }, null);
        defer vkd.destroyShaderModule(device, module, null);

        var pipelines: [1]vk.Pipeline = undefined;
        const create_infos = [1]vk.ComputePipelineCreateInfo{.{
            .stage = .{
                .stage = .{ .compute_bit = true },
                .module = module,
                .p_name = "main",
            },
            .layout = pipeline_layout,
            .base_pipeline_index = -1,
        }};
        _ = try vkd.createComputePipelines(device, .null_handle, &create_infos, null, &pipelines);

        return .{
            .vkd = vkd,
            .device = device,
            .desc_layout = desc_layout,
            .desc_pool = desc_pool,
            .desc_sets = desc_sets,
            .out_bufs = out_bufs,
            .pipeline_layout = pipeline_layout,
            .pipeline = pipelines[0],
        };
    }

    pub fn deinit(self: *Probe) void {
        self.vkd.destroyPipeline(self.device, self.pipeline, null);
        self.vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        self.vkd.destroyDescriptorPool(self.device, self.desc_pool, null);
        self.vkd.destroyDescriptorSetLayout(self.device, self.desc_layout, null);
        for (self.out_bufs) |b| {
            self.vkd.destroyBuffer(self.device, b.buffer, null);
            self.vkd.freeMemory(self.device, b.memory, null);
        }
    }

    /// The GPU buffer holding `frame_index`'s probe result; chunk B's HUD numeric
    /// pipeline binds it for vertex-stage reads. `agl_m` lives at byte offset 8.
    pub fn outBuffer(self: *const Probe, frame_index: u32) vk.Buffer {
        return self.out_bufs[frame_index].buffer;
    }

    /// Record the probe dispatch + a compute->vertex/fragment barrier. Call once
    /// per frame, BEFORE the render pass. `position` is the aircraft world pos in
    /// arcsec (x = lon*3600, y = altitude_arcsec, z = -lat*3600).
    pub fn recordDispatch(self: *Probe, cmd_buf: vk.CommandBuffer, frame_index: u32, position: math.Vec3d) void {
        const push = ProbePush{
            .aircraft_world = .{ @floatCast(position[0]), @floatCast(position[2]) },
            .observer_msl_m = coords.arcsecToMeters(@floatCast(position[1])),
        };
        self.vkd.cmdBindPipeline(cmd_buf, .compute, self.pipeline);
        const sets = [1]vk.DescriptorSet{self.desc_sets[frame_index]};
        self.vkd.cmdBindDescriptorSets(cmd_buf, .compute, self.pipeline_layout, 0, &sets, null);
        self.vkd.cmdPushConstants(cmd_buf, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(ProbePush), @ptrCast(&push));
        self.vkd.cmdDispatch(cmd_buf, 1, 1, 1);

        // Probe write -> HUD read. The numeric pipeline (chunk B) reads `agl_m`
        // in the vertex stage; fragment is included for value-driven color later.
        const barrier = [1]vk.MemoryBarrier{.{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        }};
        self.vkd.cmdPipelineBarrier(
            cmd_buf,
            .{ .compute_shader_bit = true },
            .{ .vertex_shader_bit = true, .fragment_shader_bit = true },
            .{},
            &barrier,
            null,
            null,
        );
    }
};
