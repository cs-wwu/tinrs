//! Render-debug toolchain. Always available; no CLI flag, no parameter
//! threading. Sites that need a toggle read `state.<field>` directly; sites
//! that don't, never see the module.

const std = @import("std");
const math = @import("math");

pub const RenderMode = enum(u32) {
    normal,
    wireframe,
    wireframe_overlay,

    pub fn next(self: RenderMode) RenderMode {
        return switch (self) {
            .normal => .wireframe,
            .wireframe => .wireframe_overlay,
            .wireframe_overlay => .normal,
        };
    }

    pub fn label(self: RenderMode) []const u8 {
        return switch (self) {
            .normal => "normal",
            .wireframe => "wireframe",
            .wireframe_overlay => "wireframe+shaded",
        };
    }
};

/// Shader-side overlay. The integer value is written to `SceneUBO.debug_overlay`
/// and read by `terrain.frag`; keep enum order in sync with the shader switch.
pub const ColorOverlay = enum(u32) {
    off,
    by_level,
    by_chunk,
    /// Tints chunks by per-chunk cull bits (radial, frustum). When this is
    /// active the cull predicate becomes a tag rather than a skip; culled
    /// chunks are still drawn, dimmed, so freeze + fly-out shows the cull set.
    by_cull_state,

    pub fn next(self: ColorOverlay) ColorOverlay {
        return switch (self) {
            .off => .by_level,
            .by_level => .by_chunk,
            .by_chunk => .by_cull_state,
            .by_cull_state => .off,
        };
    }

    pub fn label(self: ColorOverlay) []const u8 {
        return switch (self) {
            .off => "off",
            .by_level => "by_level",
            .by_chunk => "by_chunk",
            .by_cull_state => "by_cull_state",
        };
    }
};

pub const DebugState = struct {
    render_mode: RenderMode = .normal,
    color_overlay: ColorOverlay = .off,
    /// F1: show/hide the top-left dev/debug readout block (FPS, VRAM, active
    /// toggles). Default off so a fresh launch shows only the SVS HUD; press F1
    /// to reveal it. Session-only, like its sibling toggles below.
    show_block: bool = false,
    /// F4: when on, recordUpdate/recordDraw substitute frozen_cam_pos
    /// for camera-derived inputs (scroll, LOD selection).
    freeze: bool = false,
    /// Shift+F4: when freeze is on, this lets tile streaming continue against
    /// the live camera. Default off, so freeze pins residency too, and
    /// fly-out-and-inspect doesn't change which tiles are loaded.
    streaming_override: bool = false,
    frozen_cam_pos: [3]f64 = .{ 0, 0, 0 },
    frozen_fov: f32 = 0,
    /// Captured rotation-only view + projection at freeze time. Used by the
    /// frustum cull so flying out doesn't change which chunks are culled.
    frozen_view: math.Mat4 = math.identity,
    frozen_proj: math.Mat4 = math.identity,
};

/// Module-level state. Read directly: `debug.state.freeze`, etc.
/// Writes happen in main.zig's key handler.
pub var state: DebugState = .{};

/// True when tile streaming should pause. Helper so callers (tickPolicy)
/// don't have to spell out the "freeze AND NOT override" combination.
pub fn streamingFrozen() bool {
    return state.freeze and !state.streaming_override;
}

test "RenderMode cycles" {
    var m: RenderMode = .normal;
    m = m.next();
    try std.testing.expectEqual(RenderMode.wireframe, m);
    m = m.next();
    try std.testing.expectEqual(RenderMode.wireframe_overlay, m);
    m = m.next();
    try std.testing.expectEqual(RenderMode.normal, m);
}

test "ColorOverlay cycles" {
    var o: ColorOverlay = .off;
    o = o.next();
    try std.testing.expectEqual(ColorOverlay.by_level, o);
    o = o.next();
    try std.testing.expectEqual(ColorOverlay.by_chunk, o);
    o = o.next();
    try std.testing.expectEqual(ColorOverlay.by_cull_state, o);
    o = o.next();
    try std.testing.expectEqual(ColorOverlay.off, o);
}

test "streamingFrozen reflects freeze + override" {
    const orig = state;
    defer state = orig;

    state = .{};
    try std.testing.expect(!streamingFrozen());

    state.freeze = true;
    try std.testing.expect(streamingFrozen());

    state.streaming_override = true;
    try std.testing.expect(!streamingFrozen());
}
