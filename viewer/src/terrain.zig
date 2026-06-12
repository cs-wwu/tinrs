//! Umbrella for the terrain subsystem. Re-exports each file in `terrain/`
//! as a namespace so callers can do `@import("terrain.zig").clipmap.Clipmap`.

pub const clipmap = @import("terrain/clipmap.zig");
pub const clipmap_cull = @import("terrain/clipmap_cull.zig");
pub const clipmap_setup = @import("terrain/clipmap_setup.zig");
pub const coords = @import("terrain/coords.zig");
pub const tile_loader = @import("terrain/tile_loader.zig");
pub const tile_policy = @import("terrain/tile_policy.zig");
pub const tile_ssbo = @import("terrain/tile_ssbo.zig");
pub const tile_streamer = @import("terrain/tile_streamer.zig");
pub const tile_system = @import("terrain/tile_system.zig");
pub const probe = @import("terrain/probe.zig");

test {
    _ = clipmap;
    _ = clipmap_cull;
    _ = clipmap_setup;
    _ = coords;
    _ = tile_loader;
    _ = tile_policy;
    _ = tile_ssbo;
    _ = tile_streamer;
    _ = tile_system;
    _ = probe;
}
