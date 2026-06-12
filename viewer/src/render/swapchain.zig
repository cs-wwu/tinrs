//! Swapchain creation, render pass, framebuffer assembly, and surface format selection.

const std = @import("std");
const vkt = @import("../vk_types.zig");
const vk = vkt.vk;
const c = vkt.c;
const depth_mod = @import("depth.zig");

pub const Swapchain = struct {
    handle: vk.SwapchainKHR,
    images: []vk.Image,
    image_views: []vk.ImageView,
    surface_format: vk.SurfaceFormatKHR,
    extent: vk.Extent2D,
    framebuffers: []vk.Framebuffer,

    pub fn format(self: Swapchain) vk.Format {
        return self.surface_format.format;
    }
};

pub const SurfaceFormatChoice = struct {
    format: vk.Format,
    color_space: vk.ColorSpaceKHR,
    transfer_function: u32, // 0=sRGB, 1=PQ(HDR10), 2=scRGB linear

    pub fn label(tf: u32) []const u8 {
        return switch (tf) {
            1 => "HDR10/PQ",
            2 => "scRGB",
            else => "sRGB",
        };
    }
};

pub fn framebufferExtent(window: *c.SDL_Window) vk.Extent2D {
    var w: c_int = 0;
    var h: c_int = 0;
    _ = c.SDL_GetWindowSizeInPixels(window, &w, &h);
    return .{
        .width = @intCast(@max(w, 1)),
        .height = @intCast(@max(h, 1)),
    };
}

pub fn chooseSurfaceFormat(
    vki: anytype,
    pdev: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
    allocator: std.mem.Allocator,
    enable_hdr: bool,
) !SurfaceFormatChoice {
    const formats = try vki.getPhysicalDeviceSurfaceFormatsAllocKHR(pdev, surface, allocator);
    defer allocator.free(formats);

    if (enable_hdr) {
        // Prefer scRGB (linear float, widest gamut, easiest; no transfer function needed)
        for (formats) |fmt| {
            if (fmt.format == .r16g16b16a16_sfloat and fmt.color_space == .extended_srgb_linear_ext) {
                std.log.debug("HDR: scRGB (R16G16B16A16_SFLOAT, extended sRGB linear)", .{});
                return .{ .format = fmt.format, .color_space = fmt.color_space, .transfer_function = 2 };
            }
        }
        // Fall back to HDR10 (PQ curve, 10-bit)
        for (formats) |fmt| {
            if (fmt.format == .a2b10g10r10_unorm_pack32 and fmt.color_space == .hdr10_st2084_ext) {
                std.log.debug("HDR: HDR10 (A2B10G10R10_UNORM, PQ/ST2084)", .{});
                return .{ .format = fmt.format, .color_space = fmt.color_space, .transfer_function = 1 };
            }
        }
        std.log.debug("HDR: no HDR surface format available, falling back to sRGB", .{});
    }

    for (formats) |fmt| {
        if (fmt.format == .b8g8r8a8_srgb and fmt.color_space == .srgb_nonlinear_khr) {
            return .{ .format = fmt.format, .color_space = fmt.color_space, .transfer_function = 0 };
        }
    }
    return .{ .format = formats[0].format, .color_space = formats[0].color_space, .transfer_function = 0 };
}

pub const CreateOpts = struct {
    pdev: vk.PhysicalDevice,
    device: vk.Device,
    surface: vk.SurfaceKHR,
    render_pass: vk.RenderPass,
    graphics_family: u32,
    present_family: u32,
    window: ?*c.SDL_Window,
    fallback_extent: vk.Extent2D,
    old_swapchain: vk.SwapchainKHR,
    depth_view: vk.ImageView,
    msaa_color_view: vk.ImageView, // .null_handle when samples == 1
    samples: vk.SampleCountFlags,
    allocator: std.mem.Allocator,
    vsync: bool,
    surface_format: vk.SurfaceFormatKHR,
};

pub fn create(vki: anytype, vkd: anytype, opts: CreateOpts) !Swapchain {
    const caps = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(opts.pdev, opts.surface);

    var extent: vk.Extent2D = undefined;
    if (caps.current_extent.width != std.math.maxInt(u32)) {
        extent = caps.current_extent;
    } else {
        const hint = if (opts.window) |win| framebufferExtent(win) else opts.fallback_extent;
        extent = .{
            .width = std.math.clamp(hint.width, caps.min_image_extent.width, caps.max_image_extent.width),
            .height = std.math.clamp(hint.height, caps.min_image_extent.height, caps.max_image_extent.height),
        };
    }

    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count > 0 and image_count > caps.max_image_count) {
        image_count = caps.max_image_count;
    }

    var present_mode: vk.PresentModeKHR = .fifo_khr;
    if (!opts.vsync) {
        const present_modes = try vki.getPhysicalDeviceSurfacePresentModesAllocKHR(opts.pdev, opts.surface, opts.allocator);
        defer opts.allocator.free(present_modes);
        for (present_modes) |pm| {
            if (pm == .immediate_khr) {
                present_mode = .immediate_khr;
                break;
            }
            if (pm == .mailbox_khr) present_mode = .mailbox_khr;
        }
    }

    const same_family = opts.graphics_family == opts.present_family;
    const family_indices = [_]u32{ opts.graphics_family, opts.present_family };

    const sc = try vkd.createSwapchainKHR(opts.device, &.{
        .surface = opts.surface,
        .min_image_count = image_count,
        .image_format = opts.surface_format.format,
        .image_color_space = opts.surface_format.color_space,
        .image_extent = extent,
        .image_array_layers = 1,
        .image_usage = .{ .color_attachment_bit = true },
        .image_sharing_mode = if (same_family) .exclusive else .concurrent,
        .queue_family_index_count = if (same_family) 0 else 2,
        .p_queue_family_indices = if (same_family) undefined else &family_indices,
        .pre_transform = caps.current_transform,
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = present_mode,
        .clipped = @as(vk.Bool32, .true),
        .old_swapchain = opts.old_swapchain,
    }, null);

    const images = try vkd.getSwapchainImagesAllocKHR(opts.device, sc, opts.allocator);
    errdefer opts.allocator.free(images);

    const image_views = try opts.allocator.alloc(vk.ImageView, images.len);
    errdefer opts.allocator.free(image_views);
    for (images, 0..) |img, i| {
        image_views[i] = try vkd.createImageView(opts.device, &.{
            .image = img,
            .view_type = .@"2d",
            .format = opts.surface_format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
    }

    const framebuffers = try opts.allocator.alloc(vk.Framebuffer, images.len);
    errdefer opts.allocator.free(framebuffers);
    try createFramebuffers(vkd, opts.device, opts.render_pass, image_views, opts.depth_view, opts.msaa_color_view, opts.samples, extent, framebuffers);

    return .{
        .handle = sc,
        .images = images,
        .image_views = image_views,
        .surface_format = opts.surface_format,
        .extent = extent,
        .framebuffers = framebuffers,
    };
}

pub fn destroySubresources(vkd: anytype, device: vk.Device, swapchain: *Swapchain, allocator: std.mem.Allocator) void {
    for (swapchain.framebuffers) |fb| vkd.destroyFramebuffer(device, fb, null);
    for (swapchain.image_views) |iv| vkd.destroyImageView(device, iv, null);
    allocator.free(swapchain.framebuffers);
    allocator.free(swapchain.image_views);
    allocator.free(swapchain.images);
    // Reset to empty so an errored recreateSwapchain doesn't leave deinit
    // double-freeing dangling slice pointers.
    swapchain.framebuffers = &.{};
    swapchain.image_views = &.{};
    swapchain.images = &.{};
}

pub fn cleanup(vkd: anytype, device: vk.Device, swapchain: *Swapchain, allocator: std.mem.Allocator) void {
    destroySubresources(vkd, device, swapchain, allocator);
    vkd.destroySwapchainKHR(device, swapchain.handle, null);
}

/// Build per-image framebuffers into `out`. Attachment order must match `createRenderPass`:
///   no MSAA: [swapchain_color, depth]
///   MSAA:    [msaa_color, depth, swapchain_resolve]
/// `msaa_color_view` may be `.null_handle` when samples == 1.
/// On error, framebuffers created so far are destroyed before returning.
pub fn createFramebuffers(
    vkd: anytype,
    device: vk.Device,
    render_pass: vk.RenderPass,
    image_views: []const vk.ImageView,
    depth_view: vk.ImageView,
    msaa_color_view: vk.ImageView,
    samples: vk.SampleCountFlags,
    extent: vk.Extent2D,
    out: []vk.Framebuffer,
) !void {
    std.debug.assert(out.len == image_views.len);
    const msaa = !samples.@"1_bit";
    var created: usize = 0;
    errdefer for (out[0..created]) |fb| vkd.destroyFramebuffer(device, fb, null);
    for (image_views, 0..) |iv, i| {
        const attachments_no_msaa = [2]vk.ImageView{ iv, depth_view };
        const attachments_msaa = [3]vk.ImageView{ msaa_color_view, depth_view, iv };
        out[i] = try vkd.createFramebuffer(device, &.{
            .render_pass = render_pass,
            .attachment_count = if (msaa) 3 else 2,
            .p_attachments = if (msaa) &attachments_msaa else &attachments_no_msaa,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        }, null);
        created = i + 1;
    }
}

pub fn createRenderPass(vkd: anytype, device: vk.Device, format: vk.Format, samples: vk.SampleCountFlags) !vk.RenderPass {
    const msaa = !samples.@"1_bit";

    // Color attachment 0: render target (multisampled when MSAA on).
    // With MSAA: storeOp=DONT_CARE since the resolve target receives the data.
    // Without MSAA: this IS the swapchain image, so store + present_src layout.
    const color_attachment = vk.AttachmentDescription{
        .format = format,
        .samples = samples,
        .load_op = .clear,
        .store_op = if (msaa) .dont_care else .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = if (msaa) .color_attachment_optimal else .present_src_khr,
    };
    const depth_attachment = vk.AttachmentDescription{
        .format = depth_mod.DEPTH_FORMAT,
        .samples = samples,
        .load_op = .clear,
        .store_op = .dont_care,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };
    // Resolve attachment 2: single-sampled swapchain image (only when MSAA on).
    const resolve_attachment = vk.AttachmentDescription{
        .format = format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .dont_care,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const all_attachments = [3]vk.AttachmentDescription{ color_attachment, depth_attachment, resolve_attachment };
    const attachment_count: u32 = if (msaa) 3 else 2;

    const color_ref = vk.AttachmentReference{ .attachment = 0, .layout = .color_attachment_optimal };
    const depth_ref = vk.AttachmentReference{ .attachment = 1, .layout = .depth_stencil_attachment_optimal };
    const resolve_ref = vk.AttachmentReference{ .attachment = 2, .layout = .color_attachment_optimal };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_ref),
        .p_resolve_attachments = if (msaa) @ptrCast(&resolve_ref) else null,
        .p_depth_stencil_attachment = @ptrCast(&depth_ref),
    };

    // Depth (and the MSAA color image) are shared across framebuffers, so the
    // prior frame's end-of-pass layout transition WAW-conflicts with this
    // frame's begin transition unless we declare both writes in src.
    const dependency = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
            .late_fragment_tests_bit = true,
        },
        .src_access_mask = .{
            .color_attachment_write_bit = true,
            .depth_stencil_attachment_write_bit = true,
        },
        .dst_stage_mask = .{
            .color_attachment_output_bit = true,
            .early_fragment_tests_bit = true,
            .late_fragment_tests_bit = true,
        },
        .dst_access_mask = .{ .color_attachment_write_bit = true, .depth_stencil_attachment_write_bit = true },
    };

    return vkd.createRenderPass(device, &.{
        .attachment_count = attachment_count,
        .p_attachments = &all_attachments,
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    }, null);
}
