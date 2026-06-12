//! Vulkan renderer: owns per-frame GPU state on top of a VulkanContext.
//!
//! VulkanContext (built first) holds the one-time instance/device/queues/cmd_pool
//! setup. Renderer adds swapchain, render pass, depth buffer, MSAA color, and
//! frame-sync primitives, and provides the beginFrame/endFrame API.
//! Handles swapchain recreation on resize.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const c = vkt.c;
const Bench = @import("../bench/runner.zig");
const buffer_mod = @import("buffer.zig");
const depth_mod = @import("depth.zig");
const msaa_mod = @import("msaa.zig");
const swapchain_mod = @import("swapchain.zig");
const vulkan_context_mod = @import("vulkan_context.zig");

const ts_clock: std.Io.Clock = .awake;

pub const MAX_FRAMES_IN_FLIGHT = vulkan_context_mod.MAX_FRAMES_IN_FLIGHT;

// Re-exports: stable API for subsystems. New code may import the submodule
// directly (e.g. `@import("buffer.zig")`) instead of going through here.
pub const DEPTH_FORMAT = depth_mod.DEPTH_FORMAT;
pub const BufferWithMemory = buffer_mod.BufferWithMemory;
pub const MappedBuffer = buffer_mod.MappedBuffer;
pub const DepthResources = depth_mod.DepthResources;
pub const MSAAColorResources = msaa_mod.MSAAColorResources;
pub const Swapchain = swapchain_mod.Swapchain;
pub const createBuffer = buffer_mod.createBuffer;
pub const createMappedBuffer = buffer_mod.createMappedBuffer;
pub const destroyMappedBuffer = buffer_mod.destroyMappedBuffer;
pub const uploadBuffer = buffer_mod.uploadBuffer;
pub const findMemoryType = buffer_mod.findMemoryType;
pub const beginOneShot = buffer_mod.beginOneShot;
pub const endOneShot = buffer_mod.endOneShot;
pub const pickSampleCount = msaa_mod.pickSampleCount;
pub const sampleCountToInt = msaa_mod.sampleCountToInt;
pub const countSupported = msaa_mod.countSupported;
pub const indexOfCount = msaa_mod.indexOfCount;

pub const VulkanContext = vulkan_context_mod.VulkanContext;
pub const GpuContext = vulkan_context_mod.GpuContext;
pub const SurfaceMode = vulkan_context_mod.SurfaceMode;

// File-private aliases so the Renderer body code can stay terse.
const framebufferExtent = swapchain_mod.framebufferExtent;
const createDepthResources = depth_mod.create;
const destroyDepthResources = depth_mod.destroy;
const createMSAAColorResources = msaa_mod.create;
const destroyMSAAColorResources = msaa_mod.destroy;
const chooseSurfaceFormat = swapchain_mod.chooseSurfaceFormat;
const cleanupSwapchainHelper = swapchain_mod.cleanup;
const destroySwapchainSubresources = swapchain_mod.destroySubresources;
const createSwapchainFramebuffersInto = swapchain_mod.createFramebuffers;
const createRenderPass = swapchain_mod.createRenderPass;

/// Context returned by beginFrame for the caller to record commands.
pub const FrameContext = struct {
    cmd_buf: vk.CommandBuffer,
    image_index: u32,
};

/// Vulkan renderer: per-frame GPU state on top of VulkanContext.
///
/// Create with `init`, destroy with `deinit`.
/// Each frame: beginFrame -> record commands -> endFrame.
pub const Renderer = struct {
    ctx: VulkanContext,

    // Aliases of ctx fields kept on Renderer for the small set of external
    // callers that reach them via `renderer.<field>`. Value copies of handles
    // and wrappers; lifetimes are owned by ctx. Everything else goes through
    // `self.ctx.<field>`.
    vkd: vk.DeviceWrapper,
    device: vk.Device,
    device_name: [256]u8,
    window: ?*c.SDL_Window, // null = headless mode (VK_EXT_headless_surface, no resize)
    io: std.Io,

    render_pass: vk.RenderPass,
    swapchain: Swapchain,
    depth: DepthResources,
    msaa_color: MSAAColorResources, // zero-handles when samples == 1_bit
    samples: vk.SampleCountFlags,

    image_available: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore,
    /// Per-swapchain-image (not per-frame): can't reuse a semaphore while a prior present still tracks it.
    render_finished: []vk.Semaphore,
    in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence,
    current_frame: u32,

    framebuffer_resized: bool,
    display_changed: bool,
    render_pass_dirty: bool,
    swapchain_generation: u32,
    allocator: std.mem.Allocator,

    vsync: bool,
    hdr_enabled: bool, // user preference: use HDR when the hardware + display allow it
    transfer_function: u32, // 0=sRGB, 1=PQ(HDR10), 2=scRGB linear

    last_sample: Bench.PhaseSample = .{},
    query_ready: bool = false,

    pub const InitOptions = vulkan_context_mod.InitOptions;

    /// Create the Vulkan infrastructure: instance, device, swapchain, render pass, sync.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, surface_mode: SurfaceMode, opts: InitOptions) !Renderer {
        var ctx = try VulkanContext.init(allocator, io, surface_mode, opts);
        errdefer ctx.deinit();

        const samples = ctx.samples;
        const surface_fmt = ctx.surface_format;
        const initial_extent = ctx.initial_extent;
        const color_format = surface_fmt.format;

        // ---- Render pass (color + depth, with optional resolve target for MSAA) ----
        const render_pass = try createRenderPass(&ctx.vkd, ctx.device, color_format, samples);
        errdefer ctx.vkd.destroyRenderPass(ctx.device, render_pass, null);

        // ---- Depth + MSAA color resources ----
        const depth = try createDepthResources(&ctx.vkd, ctx.device, ctx.mem_props, initial_extent, samples);
        errdefer destroyDepthResources(&ctx.vkd, ctx.device, depth);

        const msaa_color = try createMSAAColorResources(&ctx.vkd, ctx.device, ctx.mem_props, initial_extent, color_format, samples);
        errdefer destroyMSAAColorResources(&ctx.vkd, ctx.device, msaa_color);

        // ---- Swapchain ----
        const chosen_surface_format = vk.SurfaceFormatKHR{ .format = surface_fmt.format, .color_space = surface_fmt.color_space };
        var swapchain = try swapchain_mod.create(&ctx.vki, &ctx.vkd, .{
            .pdev = ctx.pdev,
            .device = ctx.device,
            .surface = ctx.surface,
            .render_pass = render_pass,
            .graphics_family = ctx.graphics_family,
            .present_family = ctx.present_family,
            .window = ctx.window,
            .fallback_extent = initial_extent,
            .old_swapchain = .null_handle,
            .depth_view = depth.view,
            .msaa_color_view = msaa_color.view,
            .samples = samples,
            .allocator = allocator,
            .vsync = opts.vsync,
            .surface_format = chosen_surface_format,
        });
        errdefer cleanupSwapchainHelper(&ctx.vkd, ctx.device, &swapchain, allocator);

        // ---- Sync objects ----
        var image_available: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
        var in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = undefined;

        var sync_init_count: u32 = 0;
        errdefer for (0..sync_init_count) |i| {
            ctx.vkd.destroySemaphore(ctx.device, image_available[i], null);
            ctx.vkd.destroyFence(ctx.device, in_flight_fences[i], null);
        };

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            image_available[i] = try ctx.vkd.createSemaphore(ctx.device, &.{}, null);
            in_flight_fences[i] = try ctx.vkd.createFence(ctx.device, &.{ .flags = .{ .signaled_bit = true } }, null);
            sync_init_count += 1;
        }

        const render_finished = try createSemaphoreSlice(ctx.vkd, ctx.device, swapchain.images.len, allocator);
        errdefer destroySemaphoreSlice(ctx.vkd, ctx.device, render_finished, allocator);

        return .{
            .ctx = ctx,
            .vkd = ctx.vkd,
            .device = ctx.device,
            .device_name = ctx.device_name,
            .window = ctx.window,
            .io = io,
            .render_pass = render_pass,
            .swapchain = swapchain,
            .depth = depth,
            .msaa_color = msaa_color,
            .samples = samples,
            .image_available = image_available,
            .render_finished = render_finished,
            .in_flight_fences = in_flight_fences,
            .current_frame = 0,
            .framebuffer_resized = false,
            .display_changed = false,
            .render_pass_dirty = false,
            .swapchain_generation = 0,
            .allocator = allocator,
            .vsync = opts.vsync,
            .hdr_enabled = opts.enable_hdr,
            .transfer_function = surface_fmt.transfer_function,
        };
    }

    pub fn deinit(self: *Renderer) void {
        self.vkd.deviceWaitIdle(self.device) catch |err| {
            std.log.warn("deviceWaitIdle failed in deinit: {}", .{err});
        };

        for (0..MAX_FRAMES_IN_FLIGHT) |i| {
            self.vkd.destroySemaphore(self.device, self.image_available[i], null);
            self.vkd.destroyFence(self.device, self.in_flight_fences[i], null);
        }
        destroySemaphoreSlice(self.vkd, self.device, self.render_finished, self.allocator);

        cleanupSwapchainHelper(&self.vkd, self.device, &self.swapchain, self.allocator);
        destroyMSAAColorResources(&self.vkd, self.device, self.msaa_color);
        destroyDepthResources(&self.vkd, self.device, self.depth);
        self.vkd.destroyRenderPass(self.device, self.render_pass, null);
        self.ctx.deinit();
    }

    pub fn notifyResized(self: *Renderer) void {
        self.framebuffer_resized = true;
    }

    /// Sets framebuffer_resized too so the next endFrame triggers a recreate.
    pub fn notifyDisplayChanged(self: *Renderer) void {
        self.display_changed = true;
        self.framebuffer_resized = true;
    }

    /// True (and cleared) if the render pass was rebuilt; caller must recreate dependent pipelines.
    pub fn consumeRenderPassDirty(self: *Renderer) bool {
        const dirty = self.render_pass_dirty;
        self.render_pass_dirty = false;
        return dirty;
    }

    /// Borrowed view of immutable Vulkan handles for subsystem init.
    pub fn gpuCtx(self: *const Renderer) GpuContext {
        return .{
            .vkd = self.vkd,
            .device = self.device,
            .mem_props = self.ctx.mem_props,
            .queue = self.ctx.graphics_queue,
            .cmd_pool = self.ctx.cmd_pool,
            .vram_mb = self.ctx.vram_mb,
        };
    }

    // =========================================================================
    // Frame API
    // =========================================================================

    /// Begin a frame: wait for fence, acquire image, begin command buffer.
    /// Returns null if swapchain needed recreation (skip this frame).
    /// After this, the caller can record compute dispatches, then call
    /// beginRenderPass, record draw commands, and finally endFrame.
    pub fn beginFrame(self: *Renderer) !?FrameContext {
        const frame = self.current_frame;

        self.last_sample = .{};

        // Non-blocking fence status check; distinguishes syscall overhead from real GPU stall.
        if (self.ctx.bench_enabled) {
            const status = self.vkd.getFenceStatus(self.device, self.in_flight_fences[frame]) catch .not_ready;
            self.last_sample.fence_pre_signaled = (status == .success);
        }

        const fences = [1]vk.Fence{self.in_flight_fences[frame]};
        const wait_t0 = if (self.ctx.bench_enabled) ts_clock.now(self.io) else undefined;
        _ = try self.vkd.waitForFences(self.device, &fences, @as(vk.Bool32, .true), std.math.maxInt(u64));
        if (self.ctx.bench_enabled) self.last_sample.cpu_wait_ns = Bench.deltaNs(wait_t0, ts_clock.now(self.io));

        if (self.ctx.query_pool != .null_handle and self.query_ready)
            self.readbackTimestamps(frame);

        const acquire_t0 = if (self.ctx.bench_enabled) ts_clock.now(self.io) else undefined;
        const acquire_result = self.vkd.acquireNextImageKHR(self.device, self.swapchain.handle, std.math.maxInt(u64), self.image_available[frame], .null_handle);
        if (self.ctx.bench_enabled) self.last_sample.cpu_acquire_ns = Bench.deltaNs(acquire_t0, ts_clock.now(self.io));

        const image_index = blk: {
            if (acquire_result) |result| {
                break :blk result.image_index;
            } else |err| {
                if (err == error.OutOfDateKHR) {
                    try self.recreateSwapchain();
                    return null;
                }
                return err;
            }
        };

        try self.vkd.resetFences(self.device, &fences);

        const cmd = self.ctx.cmd_buffers[frame];
        try self.vkd.resetCommandBuffer(cmd, .{});
        try self.vkd.beginCommandBuffer(cmd, &.{});

        // ---- Reset this frame's query slots and write timestamp 0 (compute_start) ----
        if (self.ctx.query_pool != .null_handle) {
            self.vkd.cmdResetQueryPool(cmd, self.ctx.query_pool, frame * 4, 4);
            self.vkd.cmdWriteTimestamp(cmd, .{ .top_of_pipe_bit = true }, self.ctx.query_pool, frame * 4 + 0);
        }

        return .{ .cmd_buf = cmd, .image_index = image_index };
    }

    fn readbackTimestamps(self: *Renderer, frame: u32) void {
        // [value0, avail0, value1, avail1, value2, avail2, value3, avail3]
        var buf: [8]u64 = undefined;
        const result = self.vkd.getQueryPoolResults(
            self.device,
            self.ctx.query_pool,
            frame * 4,
            4,
            @sizeOf(@TypeOf(buf)),
            &buf,
            @sizeOf(u64) * 2,
            .{ .@"64_bit" = true, .with_availability_bit = true },
        ) catch {
            return;
        };
        if (result == .not_ready) return;
        // All four must be available to compute deltas.
        if (buf[1] == 0 or buf[3] == 0 or buf[5] == 0 or buf[7] == 0) return;

        const compute_ticks = buf[2] -% buf[0];
        const graphics_ticks = buf[6] -% buf[4];
        const period: f64 = self.ctx.timestamp_period_ns;
        self.last_sample.gpu_compute_ns = @intFromFloat(@as(f64, @floatFromInt(compute_ticks)) * period);
        self.last_sample.gpu_graphics_ns = @intFromFloat(@as(f64, @floatFromInt(graphics_ticks)) * period);
    }

    pub fn lastSample(self: *const Renderer) Bench.PhaseSample {
        return self.last_sample;
    }

    /// Write GPU timestamps marking the compute -> graphics boundary.
    /// Call after compute dispatches, before any render pass.
    pub fn writeGraphicsTimestamps(self: *Renderer, cmd: vk.CommandBuffer) void {
        if (self.ctx.query_pool != .null_handle) {
            const frame = self.current_frame;
            self.vkd.cmdWriteTimestamp(cmd, .{ .compute_shader_bit = true }, self.ctx.query_pool, frame * 4 + 1);
            self.vkd.cmdWriteTimestamp(cmd, .{ .top_of_pipe_bit = true }, self.ctx.query_pool, frame * 4 + 2);
        }
    }

    /// Begin the render pass. Call after any pre-render compute dispatches.
    pub fn beginRenderPass(self: *Renderer, cmd: vk.CommandBuffer, image_index: u32) void {
        const clear_values = [2]vk.ClearValue{
            .{ .color = .{ .float_32 = .{ 0.01, 0.01, 0.05, 1.0 } } },
            // Reverse-Z: clear to 0.0 (= far plane).
            .{ .depth_stencil = .{ .depth = 0.0, .stencil = 0 } },
        };
        self.vkd.cmdBeginRenderPass(cmd, &.{
            .render_pass = self.render_pass,
            .framebuffer = self.swapchain.framebuffers[image_index],
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapchain.extent },
            .clear_value_count = clear_values.len,
            .p_clear_values = &clear_values,
        }, .@"inline");

        const viewports = [1]vk.Viewport{.{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swapchain.extent.width),
            .height = @floatFromInt(self.swapchain.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        }};
        self.vkd.cmdSetViewport(cmd, 0, &viewports);
        const scissors = [1]vk.Rect2D{.{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain.extent,
        }};
        self.vkd.cmdSetScissor(cmd, 0, &scissors);
    }

    /// End a frame: submit command buffer and present.
    /// The caller must end any active render pass before calling this.
    pub fn endFrame(self: *Renderer, ctx: FrameContext) !void {
        const frame = self.current_frame;
        const cmd = ctx.cmd_buf;

        // Stamp end-of-graphics (after all render passes).
        if (self.ctx.query_pool != .null_handle) {
            self.vkd.cmdWriteTimestamp(cmd, .{ .color_attachment_output_bit = true }, self.ctx.query_pool, frame * 4 + 3);
        }

        try self.vkd.endCommandBuffer(cmd);

        const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
        const submits = [1]vk.SubmitInfo{.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&self.image_available[frame]),
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&self.render_finished[ctx.image_index]),
        }};
        const submit_t0 = if (self.ctx.bench_enabled) ts_clock.now(self.io) else undefined;
        try self.vkd.queueSubmit(self.ctx.graphics_queue, &submits, self.in_flight_fences[frame]);
        if (self.ctx.bench_enabled) self.last_sample.cpu_submit_ns = Bench.deltaNs(submit_t0, ts_clock.now(self.io));

        const present_t0 = if (self.ctx.bench_enabled) ts_clock.now(self.io) else undefined;
        const present_result = self.vkd.queuePresentKHR(self.ctx.present_queue, &.{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&self.render_finished[ctx.image_index]),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&self.swapchain.handle),
            .p_image_indices = @ptrCast(&ctx.image_index),
        });
        if (self.ctx.bench_enabled) self.last_sample.cpu_present_ns = Bench.deltaNs(present_t0, ts_clock.now(self.io));

        if (present_result) |_| {} else |err| {
            if (err == error.OutOfDateKHR) {
                try self.recreateSwapchain();
                if (frame == MAX_FRAMES_IN_FLIGHT - 1) self.query_ready = true;
                self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
                return;
            }
            return err;
        }

        if (self.framebuffer_resized) {
            self.framebuffer_resized = false;
            try self.recreateSwapchain();
        }

        if (frame == MAX_FRAMES_IN_FLIGHT - 1) self.query_ready = true;
        self.current_frame = (frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    // =========================================================================
    // Swapchain management
    // =========================================================================

    const FormatChange = struct {
        surface_format: vk.SurfaceFormatKHR,
        render_pass: vk.RenderPass,
        transfer_function: u32,
    };

    /// Returns null when the format is unchanged, when re-querying fails, or when the
    /// new render pass can't be built; caller keeps using the current state in all cases.
    fn redetectSurfaceFormat(self: *Renderer, win: *c.SDL_Window) ?FormatChange {
        // HDR is used only when the hardware is capable, the user preference is on,
        // AND the display currently reports HDR. The preference (`hdr_enabled`) lets
        // the settings menu force SDR on an HDR display; this same path runs for the
        // OS-driven `SDL_EVENT_WINDOW_HDR_STATE_CHANGED`.
        const display_hdr = self.ctx.hdr_capable and self.hdr_enabled and
            c.SDL_GetBooleanProperty(c.SDL_GetWindowProperties(win), c.SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
        const new_fmt = chooseSurfaceFormat(&self.ctx.vki, self.ctx.pdev, self.ctx.surface, self.allocator, display_hdr) catch |err| {
            std.log.warn("Surface format re-detection failed: {} - keeping current format", .{err});
            return null;
        };
        const cur = self.swapchain.surface_format;
        if (new_fmt.format == cur.format and new_fmt.color_space == cur.color_space) return null;
        const rp = createRenderPass(&self.vkd, self.device, new_fmt.format, self.samples) catch |err| {
            std.log.warn("Render pass rebuild for new surface format failed: {} - keeping current format", .{err});
            return null;
        };
        return .{
            .surface_format = .{ .format = new_fmt.format, .color_space = new_fmt.color_space },
            .render_pass = rp,
            .transfer_function = new_fmt.transfer_function,
        };
    }

    fn recreateSwapchain(self: *Renderer) !void {
        // Headless: surfaces don't resize, never reach this path.
        const win = self.window orelse return error.HeadlessNoResize;

        // Wait for the framebuffer to have non-zero area (e.g. coming back from a minimize).
        var new_extent = framebufferExtent(win);
        while (new_extent.width <= 1 and new_extent.height <= 1) {
            _ = c.SDL_WaitEvent(null);
            new_extent = framebufferExtent(win);
        }

        // Built before deviceWaitIdle so a failure here leaves the live state untouched.
        var target_format = self.swapchain.surface_format;
        var target_transfer_function = self.transfer_function;
        var new_render_pass: ?vk.RenderPass = null;
        if (self.display_changed) {
            self.display_changed = false;
            if (self.redetectSurfaceFormat(win)) |change| {
                target_format = change.surface_format;
                target_transfer_function = change.transfer_function;
                new_render_pass = change.render_pass;
            }
        }
        // Live render pass isn't replaced until the commit block below; if we error before
        // that, the new one is orphaned. Cleared by the assignment when commit happens.
        errdefer if (new_render_pass) |rp| self.vkd.destroyRenderPass(self.device, rp, null);

        try self.vkd.deviceWaitIdle(self.device);

        const old_handle = self.swapchain.handle;
        destroySwapchainSubresources(&self.vkd, self.device, &self.swapchain, self.allocator);

        destroyDepthResources(&self.vkd, self.device, self.depth);
        self.depth = std.mem.zeroes(DepthResources);
        self.depth = try createDepthResources(&self.vkd, self.device, self.ctx.mem_props, new_extent, self.samples);

        destroyMSAAColorResources(&self.vkd, self.device, self.msaa_color);
        self.msaa_color = std.mem.zeroes(MSAAColorResources);
        self.msaa_color = try createMSAAColorResources(&self.vkd, self.device, self.ctx.mem_props, new_extent, target_format.format, self.samples);

        if (new_render_pass) |rp| {
            self.vkd.destroyRenderPass(self.device, self.render_pass, null);
            self.render_pass = rp;
            self.transfer_function = target_transfer_function;
            self.render_pass_dirty = true;
            new_render_pass = null; // ownership moved to self.render_pass; disarm the errdefer
            std.log.info("Surface format changed (transfer_function={d}); render pass rebuilt", .{target_transfer_function});
        }

        self.swapchain = try swapchain_mod.create(&self.ctx.vki, &self.vkd, .{
            .pdev = self.ctx.pdev,
            .device = self.device,
            .surface = self.ctx.surface,
            .render_pass = self.render_pass,
            .graphics_family = self.ctx.graphics_family,
            .present_family = self.ctx.present_family,
            .window = self.window,
            .fallback_extent = new_extent,
            .old_swapchain = old_handle,
            .depth_view = self.depth.view,
            .msaa_color_view = self.msaa_color.view,
            .samples = self.samples,
            .allocator = self.allocator,
            .vsync = self.vsync,
            .surface_format = target_format,
        });

        self.vkd.destroySwapchainKHR(self.device, old_handle, null);

        destroySemaphoreSlice(self.vkd, self.device, self.render_finished, self.allocator);
        self.render_finished = try createSemaphoreSlice(self.vkd, self.device, self.swapchain.images.len, self.allocator);

        self.swapchain_generation +%= 1;
        std.log.debug("Swapchain recreated: {d}x{d}", .{ self.swapchain.extent.width, self.swapchain.extent.height });
    }

    /// Switch MSAA sample count at runtime. Recreates render pass, depth, MSAA color image,
    /// and swapchain framebuffers, then flags render_pass_dirty so the caller's
    /// consumeRenderPassDirty path rebuilds dependent pipelines (which still reference the
    /// destroyed old render pass and baked-in sample count).
    /// On failure the renderer is left in its prior state (build-new-first).
    pub fn setSampleCount(self: *Renderer, new_samples: vk.SampleCountFlags) !void {
        if (@as(u32, @bitCast(new_samples)) == @as(u32, @bitCast(self.samples))) return;

        try self.vkd.deviceWaitIdle(self.device);

        // Build everything new first; any failure here unwinds via errdefer
        // and leaves the existing self.* state untouched.
        const new_render_pass = try createRenderPass(&self.vkd, self.device, self.swapchain.format(), new_samples);
        errdefer self.vkd.destroyRenderPass(self.device, new_render_pass, null);

        const new_depth = try createDepthResources(&self.vkd, self.device, self.ctx.mem_props, self.swapchain.extent, new_samples);
        errdefer destroyDepthResources(&self.vkd, self.device, new_depth);

        const new_msaa_color = try createMSAAColorResources(&self.vkd, self.device, self.ctx.mem_props, self.swapchain.extent, self.swapchain.format(), new_samples);
        errdefer destroyMSAAColorResources(&self.vkd, self.device, new_msaa_color);

        const new_framebuffers = try self.allocator.alloc(vk.Framebuffer, self.swapchain.image_views.len);
        errdefer self.allocator.free(new_framebuffers);
        try createSwapchainFramebuffersInto(
            &self.vkd, self.device, new_render_pass,
            self.swapchain.image_views, new_depth.view, new_msaa_color.view,
            new_samples, self.swapchain.extent, new_framebuffers,
        );
        errdefer for (new_framebuffers) |fb| self.vkd.destroyFramebuffer(self.device, fb, null);

        // Commit: tear down old, swap in new.
        for (self.swapchain.framebuffers) |fb| self.vkd.destroyFramebuffer(self.device, fb, null);
        self.allocator.free(self.swapchain.framebuffers);
        self.vkd.destroyRenderPass(self.device, self.render_pass, null);
        destroyMSAAColorResources(&self.vkd, self.device, self.msaa_color);
        destroyDepthResources(&self.vkd, self.device, self.depth);

        self.samples = new_samples;
        self.render_pass = new_render_pass;
        self.depth = new_depth;
        self.msaa_color = new_msaa_color;
        self.swapchain.framebuffers = new_framebuffers;
        self.render_pass_dirty = true;
    }

    /// Set MSAA from a plain power-of-2 sample count (settings menu convenience).
    /// See `setSampleCount` for the rebuild + failure semantics.
    pub fn setMsaa(self: *Renderer, n: u32) !void {
        return self.setSampleCount(msaa_mod.intToSampleCount(n));
    }

    /// Toggle vsync at runtime. Only the swapchain present mode changes (FIFO vs
    /// immediate/mailbox), so this reuses the resize -> `recreateSwapchain` path
    /// (which re-reads `self.vsync`) instead of rebuilding the render pass or
    /// pipelines. No-op when unchanged or headless (headless never recreates the
    /// swapchain).
    pub fn setVsync(self: *Renderer, enabled: bool) void {
        if (self.vsync == enabled or self.window == null) return;
        self.vsync = enabled;
        self.notifyResized();
    }

    /// Set the HDR preference at runtime. Reuses the display-changed path: the next
    /// swapchain recreate re-runs surface-format detection (which reads `hdr_enabled`)
    /// and rebuilds the render pass only if the format actually changes, exactly as
    /// the OS-driven `SDL_EVENT_WINDOW_HDR_STATE_CHANGED` does. No-op when unchanged
    /// or headless. The menu gates this behind `hdrAvailable`.
    pub fn setHdr(self: *Renderer, enabled: bool) void {
        if (self.hdr_enabled == enabled or self.window == null) return;
        self.hdr_enabled = enabled;
        self.notifyDisplayChanged();
    }

    /// Whether HDR can actually be enabled here: the loader exposes the colorspace
    /// extension AND the current display reports HDR. False on an SDR monitor (where
    /// toggling HDR would do nothing), so the menu greys the toggle out.
    pub fn hdrAvailable(self: *const Renderer) bool {
        if (!self.ctx.hdr_capable) return false;
        const win = self.window orelse return false;
        return c.SDL_GetBooleanProperty(c.SDL_GetWindowProperties(win), c.SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
    }

    /// Sample counts supported for both color and depth framebuffers, as a
    /// `SampleCountFlags` bitmask. The settings-menu MSAA cycle offers only these
    /// (via `countSupported`). Re-queries the device; call once, not per frame.
    pub fn supportedSampleCounts(self: *const Renderer) vk.SampleCountFlags {
        return msaa_mod.supportedMask(self.ctx.vki.getPhysicalDeviceProperties(self.ctx.pdev).limits);
    }
};

fn createSemaphoreSlice(
    vkd: anytype,
    device: vk.Device,
    count: usize,
    allocator: std.mem.Allocator,
) ![]vk.Semaphore {
    const sems = try allocator.alloc(vk.Semaphore, count);
    errdefer allocator.free(sems);
    var created: usize = 0;
    errdefer for (sems[0..created]) |s| vkd.destroySemaphore(device, s, null);
    while (created < count) : (created += 1) {
        sems[created] = try vkd.createSemaphore(device, &.{}, null);
    }
    return sems;
}

fn destroySemaphoreSlice(
    vkd: anytype,
    device: vk.Device,
    sems: []vk.Semaphore,
    allocator: std.mem.Allocator,
) void {
    for (sems) |s| vkd.destroySemaphore(device, s, null);
    allocator.free(sems);
}
