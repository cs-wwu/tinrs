//! Umbrella for the render subsystem. Re-exports each file in `render/`
//! as a namespace so callers can do `@import("render.zig").buffer.Buffer`.
//!
//! The `test {}` block at the bottom pulls every submodule's tests into any
//! `addTest` that roots on this file.

pub const aircraft_mesh = @import("render/aircraft_mesh.zig");
pub const buffer = @import("render/buffer.zig");
pub const debug = @import("render/debug.zig");
pub const depth = @import("render/depth.zig");
pub const display = @import("render/display.zig");
pub const msaa = @import("render/msaa.zig");
pub const pipeline = @import("render/pipeline.zig");
pub const renderer = @import("render/renderer.zig");
pub const sky = @import("render/sky.zig");
pub const swapchain = @import("render/swapchain.zig");
pub const ui_backend = @import("render/ui_backend.zig");
pub const vulkan_context = @import("render/vulkan_context.zig");

test {
    _ = aircraft_mesh;
    _ = buffer;
    _ = debug;
    _ = depth;
    _ = display;
    _ = msaa;
    _ = pipeline;
    _ = renderer;
    _ = sky;
    _ = swapchain;
    _ = ui_backend;
    _ = vulkan_context;
}
