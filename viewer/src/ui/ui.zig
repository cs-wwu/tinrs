//! Reusable immediate-mode UI core. Pure: depends only on `std`, knows nothing
//! about Vulkan, SDL, or this project's types. It produces a `DrawList` (CPU-side
//! geometry) that a separate backend uploads and renders. This is the seam that
//! keeps the UI liftable into another project: swap the backend, keep this core.
//!
//! Phase 1 surface: the `DrawList` painter (rects, gradients, lines, text) plus
//! the bitmap font. Layout, widgets, input, and focus land in later phases and
//! will live alongside this file under the same `ui` module.

const std = @import("std");

// Interactive layer (immediate-mode context + widgets), sibling files under the
// same `ui` module. The painter below (DrawList/Screen/font) stays the pure
// foundation; these build on it. Re-exported so call sites read `ui.Ui`,
// `ui.toggle`, etc. Module-import isolation keeps their tests in the `ui` target.
pub const context = @import("context.zig");
pub const widgets = @import("widgets.zig");
pub const Ui = context.Ui;
pub const InputState = context.InputState;
pub const PointerKind = context.PointerKind;
pub const Rect = context.Rect;
pub const button = widgets.button;
pub const tabBar = widgets.tabBar;
pub const toggle = widgets.toggle;
pub const sliderF32 = widgets.sliderF32;
pub const sliderInt = widgets.sliderInt;
pub const cycle = widgets.cycle;
pub const labelRow = widgets.labelRow;

test {
    _ = context;
    _ = widgets;
}

/// Reference height + default user scale for fraction-of-screen UI. The HUD and
/// the interactive menus both build their `Screen` from these, so they lay out at
/// a consistent logical-px size across resolutions. (User scale will later come
/// from a settings ui_scale knob; keep one source so HUD and menu can't desync.)
pub const REF_H: f32 = 720.0;
pub const DEFAULT_USER_SCALE: f32 = 1.5;

/// RGBA in [0, 1]. Alpha enables the translucent panels the HUD draws over terrain.
pub const Color = struct {
    r: f32,
    g: f32,
    b: f32,
    a: f32 = 1.0,

    pub fn rgba(r: f32, g: f32, b: f32, a: f32) Color {
        return .{ .r = r, .g = g, .b = b, .a = a };
    }

    /// Same RGB, new alpha. Lets a single base color (e.g. the HUD's mil-green)
    /// be reused at location-dependent opacity without redefining the channels.
    pub fn withAlpha(self: Color, a: f32) Color {
        return .{ .r = self.r, .g = self.g, .b = self.b, .a = a };
    }

    pub const white: Color = .{ .r = 1, .g = 1, .b = 1, .a = 1 };
    pub const black: Color = .{ .r = 0, .g = 0, .b = 0, .a = 1 };
};

/// One vertex of a solid/gradient shape: pixel position + per-vertex color.
/// Per-vertex color is what makes gradients free (interpolated by the rasterizer).
pub const ShapeVertex = extern struct {
    x: f32,
    y: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

/// One vertex of a glyph quad: pixel position, cell UV, color, and the ASCII code
/// the fragment shader looks up in the font bitmap.
pub const GlyphVertex = extern struct {
    x: f32,
    y: f32,
    u: f32,
    v: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    char_idx: u32,
};

/// One vertex of a GPU-decoded numeric digit slot: same shape as GlyphVertex but
/// the trailing u32 is the decimal `place` (0 = ones, 1 = tens, ...) instead of a
/// char index. numeric.vert reads the value from a GPU buffer and computes the
/// glyph for this place, so the number is decoded on the GPU and never hits the CPU.
pub const NumericVertex = extern struct {
    x: f32,
    y: f32,
    u: f32,
    v: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    place: u32,
};

/// One vertex of a GPU-driven gauge fill (the AGL bar). The CPU emits a quad over
/// the bar's full extent; `fill_weight` (0 = bottom, 1 = top) plus the bar bounds
/// let gauge.vert place the top at the GPU value's fill level. `y_bottom`/`y_top`
/// are the same across all 4 verts; only `x` and `fill_weight` differ.
pub const GaugeVertex = extern struct {
    x: f32,
    y_bottom: f32,
    y_top: f32,
    fill_weight: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

/// One vertex of an SDF shape quad. `x,y` is the bounding-quad corner (screen px);
/// the remaining params are constant across the element's 6 verts (the backend
/// declares them `flat`) and define the shape the fragment shader distance-tests
/// against `gl_FragCoord`. Four kinds share the fields:
///   - capsule (`ax,ay`/`bx,by` = segment A/B, `radius` = half-thickness; a==b is a
///     circle): round-capped, covers circle/ring.
///   - box (`ax,ay` = center, `bx,by` = half-extent, `radius` = corner radius):
///     axis-aligned, covers roundedRect.
///   - segment (`ax,ay`/`bx,by` = endpoints A/B, `radius` = half-thickness): an
///     oriented box with butt caps, covers line.
///   - polyline (`ax` = point offset, `ay` = point count into the `path_points`
///     pool, `radius` = half-thickness): connected stroke, min-distance over the
///     path. Note `ax,ay` are an (offset, count) here, not endpoints.
/// `border` > 0 turns a filled shape into an outline of that half-width.
/// `kind`: 0 = capsule, 1 = box, 2 = segment, 3 = polyline (mirrored in sdf.frag).
pub const SdfVertex = extern struct {
    x: f32,
    y: f32,
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    radius: f32,
    border: f32,
    kind: f32,
    r: f32,
    g: f32,
    b: f32,
    a: f32,
};

/// Font cell is 8x8 px at scale 1. `text()` multiplies by its `scale` arg.
pub const GLYPH_PX: f32 = 8.0;

/// Fixed capacities keep the DrawList allocation-free on the per-frame path.
/// 4096 shape verts ~= 682 rects; 8192 glyph verts = 1365 chars. Overflowing
/// appends are dropped (see `roomFor`); bump these if a screen needs more.
pub const MAX_SHAPE_VERTS = 4096;
pub const MAX_GLYPH_VERTS = 8192;
pub const MAX_SDF_VERTS = 4096;
/// GPU numeric digit slots (a few HUD readouts x slots x 6 verts). AGL = 5 x 6 = 30.
pub const MAX_NUMERIC_VERTS = 256;
/// GPU gauge fills: a few bars x 6 verts each. AGL bar = 6.
pub const MAX_GAUGE_VERTS = 64;
/// Polyline path-point pool (see `DrawList.path_points`). ~110 points/frame today
/// (waterline + ladder ticks); sized generously, drop-on-overflow like the rest.
pub const MAX_PATH_POINTS = 1024;

/// Max distinct scissor regions per frame. The heading ribbon uses 3 (full ->
/// clipped -> full). On overflow, clip changes are dropped and geometry falls
/// into the last accepted group; bump if a screen needs more.
pub const MAX_CLIP_GROUPS = 16;

/// Screen anchor points. Top-left origin, Y growing downward (matching the
/// shaders' pixel->NDC mapping).
pub const Anchor = enum {
    top_left,
    top_center,
    top_right,
    center_left,
    center,
    center_right,
    bottom_left,
    bottom_center,
    bottom_right,
};

/// Logical-pixel scaling + anchor math for screen-space UI, built each frame from
/// the framebuffer size. For the HUD, `scale` is fraction-of-screen: the caller
/// passes a reference height so the UI holds a constant fraction of the viewport
/// across resolutions. Pure: the core never reads the framebuffer; the caller
/// supplies width/height.
pub const Screen = struct {
    w: f32,
    h: f32,
    scale: f32,

    /// `scale = (h_px / ref_h) * user_pref`. With ref_h=720 a 720p screen is 1.0
    /// and 1440p is 2.0, so the HUD keeps a constant fraction of the viewport.
    pub fn fromExtent(w_px: f32, h_px: f32, ref_h: f32, user_pref: f32) Screen {
        return .{ .w = w_px, .h = h_px, .scale = (h_px / ref_h) * user_pref };
    }

    /// Logical pixels -> device pixels.
    pub fn px(self: Screen, logical: f32) f32 {
        return logical * self.scale;
    }

    /// Device-space text scale for `DrawList.text` from a logical glyph scale.
    /// Arithmetically the same as `px`, but named for intent: the argument is a
    /// scale multiplier, not a pixel length.
    pub fn textScale(self: Screen, logical_scale: f32) f32 {
        return logical_scale * self.scale;
    }

    /// Device-pixel position for an anchor plus a logical-pixel offset. Offsets
    /// are raw deltas: +x right, +y down.
    pub fn anchor(self: Screen, a: Anchor, off_x: f32, off_y: f32) [2]f32 {
        const bx: f32 = switch (a) {
            .top_left, .center_left, .bottom_left => 0,
            .top_center, .center, .bottom_center => self.w * 0.5,
            .top_right, .center_right, .bottom_right => self.w,
        };
        const by: f32 = switch (a) {
            .top_left, .top_center, .top_right => 0,
            .center_left, .center, .center_right => self.h * 0.5,
            .bottom_left, .bottom_center, .bottom_right => self.h,
        };
        return .{ bx + off_x * self.scale, by + off_y * self.scale };
    }
};

/// Scissor rectangle in device pixels (Vulkan-free ints). May extend past the
/// framebuffer; the backend converts to a Vulkan scissor and clamps it.
pub const ClipRect = struct {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
};

/// Marks where a scissor region begins in the vertex streams. `clip == null` is
/// the full framebuffer. A group spans [this.start, next.start) (the last group
/// ends at the live vertex count); resolve via `DrawList.resolveClipRanges`.
pub const ClipGroup = struct {
    clip: ?ClipRect,
    shape_start: u32,
    glyph_start: u32,
    sdf_start: u32,
};

/// A clip group with its vertex ranges resolved, ready for the backend to draw.
pub const ResolvedClip = struct {
    clip: ?ClipRect,
    shape_first: u32,
    shape_count: u32,
    glyph_first: u32,
    glyph_count: u32,
    sdf_first: u32,
    sdf_count: u32,
};

/// CPU-side geometry for one frame. Components append; the backend reads each
/// stream (`shape_verts`/`sdf_verts`/`glyph_verts`) and issues one draw per stream.
/// Streams render back-to-front in that order: flat shapes (fills), then SDF
/// (strokes/curves), then glyphs (text on top), so panel backgrounds sit under
/// their text regardless of append order. The order is fixed by the pipeline
/// passes, so a flat shape cannot sit above an SDF element within one clip group.
pub const DrawList = struct {
    shape_verts: [MAX_SHAPE_VERTS]ShapeVertex = undefined,
    shape_count: u32 = 0,
    glyph_verts: [MAX_GLYPH_VERTS]GlyphVertex = undefined,
    glyph_count: u32 = 0,
    sdf_verts: [MAX_SDF_VERTS]SdfVertex = undefined,
    sdf_count: u32 = 0,
    // GPU-decoded numeric digit slots (frame-global, not clip-tracked: numeric
    // readouts are tier-1 / full-screen). Drawn after glyphs, so they sit on top.
    numeric_verts: [MAX_NUMERIC_VERTS]NumericVertex = undefined,
    numeric_count: u32 = 0,
    // GPU-driven gauge fills (e.g. the AGL bar): one quad spanning the bar; the
    // vertex shader shrinks the top to the GPU value's fill fraction. Frame-global.
    gauge_verts: [MAX_GAUGE_VERTS]GaugeVertex = undefined,
    gauge_count: u32 = 0,
    // Path points for polyline (`kind` 3) SDF elements: a flat, frame-global pool
    // the backend uploads to an SSBO. Each polyline's quad carries an (offset,
    // count) into this pool; the fragment shader takes the min distance over the
    // path, so connected strokes have single coverage (no translucent double-blend
    // at joints) and filled joins. Frame-global, so no clip-group tracking: the
    // referencing quad lives in the clip-tracked sdf stream.
    path_points: [MAX_PATH_POINTS][2]f32 = undefined,
    path_count: u32 = 0,

    /// Ordered scissor-region boundaries. There is always at least one group
    /// (full framebuffer); `pushClip`/`popClip` append boundaries.
    clip_groups: [MAX_CLIP_GROUPS]ClipGroup = [_]ClipGroup{.{ .clip = null, .shape_start = 0, .glyph_start = 0, .sdf_start = 0 }} ** MAX_CLIP_GROUPS,
    clip_count: u32 = 1,

    pub fn clear(self: *DrawList) void {
        self.shape_count = 0;
        self.glyph_count = 0;
        self.sdf_count = 0;
        self.numeric_count = 0;
        self.gauge_count = 0;
        self.path_count = 0;
        self.clip_count = 1;
        self.clip_groups[0] = .{ .clip = null, .shape_start = 0, .glyph_start = 0, .sdf_start = 0 };
    }

    /// Begin a scissor region: geometry appended until the matching `popClip` is
    /// clipped to `rect`. Dropped silently on overflow (geometry then falls into
    /// the last accepted group).
    pub fn pushClip(self: *DrawList, clip_rect: ClipRect) void {
        if (self.clip_count >= MAX_CLIP_GROUPS) return;
        self.clip_groups[self.clip_count] = .{ .clip = clip_rect, .shape_start = self.shape_count, .glyph_start = self.glyph_count, .sdf_start = self.sdf_count };
        self.clip_count += 1;
    }

    /// End the current scissor region, returning to the full framebuffer.
    pub fn popClip(self: *DrawList) void {
        if (self.clip_count >= MAX_CLIP_GROUPS) return;
        self.clip_groups[self.clip_count] = .{ .clip = null, .shape_start = self.shape_count, .glyph_start = self.glyph_count, .sdf_start = self.sdf_count };
        self.clip_count += 1;
    }

    /// Resolve clip boundaries into per-group vertex ranges (each group's end is
    /// the next group's start, or the live count for the last). Writes `clip_count`
    /// entries to `out` and returns that count.
    pub fn resolveClipRanges(self: *const DrawList, out: *[MAX_CLIP_GROUPS]ResolvedClip) u32 {
        var i: u32 = 0;
        while (i < self.clip_count) : (i += 1) {
            const g = self.clip_groups[i];
            const shape_end = if (i + 1 < self.clip_count) self.clip_groups[i + 1].shape_start else self.shape_count;
            const glyph_end = if (i + 1 < self.clip_count) self.clip_groups[i + 1].glyph_start else self.glyph_count;
            const sdf_end = if (i + 1 < self.clip_count) self.clip_groups[i + 1].sdf_start else self.sdf_count;
            out[i] = .{
                .clip = g.clip,
                .shape_first = g.shape_start,
                .shape_count = shape_end - g.shape_start,
                .glyph_first = g.glyph_start,
                .glyph_count = glyph_end - g.glyph_start,
                .sdf_first = g.sdf_start,
                .sdf_count = sdf_end - g.sdf_start,
            };
        }
        return self.clip_count;
    }

    // TODO: track an accumulated content bounding box as verts are appended so
    // callers (e.g. the HUD panel in scene.zig) can size backgrounds from real
    // geometry instead of re-deriving extents from layout constants by hand.
    // Natural to add once panel/layout helpers land in a later UI phase.

    /// Pixel width a string occupies at the given scale.
    pub fn textWidth(len: usize, scale: f32) f32 {
        return @as(f32, @floatFromInt(len)) * GLYPH_PX * scale;
    }

    fn pushShapeVert(self: *DrawList, x: f32, y: f32, c: Color) void {
        self.shape_verts[self.shape_count] = .{ .x = x, .y = y, .r = c.r, .g = c.g, .b = c.b, .a = c.a };
        self.shape_count += 1;
    }

    /// Solid axis-aligned rectangle (two triangles).
    pub fn rect(self: *DrawList, x: f32, y: f32, w: f32, h: f32, c: Color) void {
        self.rectGradient(x, y, w, h, c, c);
    }

    /// Vertical-gradient rectangle: `top` color along the top edge, `bottom`
    /// along the bottom. The rasterizer interpolates between them.
    pub fn rectGradient(self: *DrawList, x: f32, y: f32, w: f32, h: f32, top: Color, bottom: Color) void {
        if (self.shape_count + 6 > MAX_SHAPE_VERTS) return;
        const x1 = x + w;
        const y1 = y + h;
        // TL-BL-BR, TL-BR-TR
        self.pushShapeVert(x, y, top);
        self.pushShapeVert(x, y1, bottom);
        self.pushShapeVert(x1, y1, bottom);
        self.pushShapeVert(x, y, top);
        self.pushShapeVert(x1, y1, bottom);
        self.pushShapeVert(x1, y, top);
    }

    // SDF shape kinds, mirrored in sdf.frag:
    //   capsule: round caps (circle, ring)
    //   box:      axis-aligned rounded box (roundedRect)
    //   segment:  oriented box with butt caps (line)
    //   polyline: connected path, min-distance over its segments (single coverage)
    const SDF_CAPSULE: f32 = 0;
    const SDF_BOX: f32 = 1;
    const SDF_SEGMENT: f32 = 2;
    const SDF_POLYLINE: f32 = 3;
    // Device-px AA padding around every SDF bounding quad, so the smoothstep edge
    // is never clipped by the quad it lives in.
    const SDF_AA_MARGIN: f32 = 1.5;

    /// Thick line from (x0,y0) to (x1,y1), drawn on the SDF stream as an
    /// anti-aliased segment: an oriented box with flat (butt) caps, matching the
    /// old quad-based line(). Same signature, so ticks, ladders, and the attitude
    /// horizon get crisp edges with no call-site change and no end overhang.
    pub fn line(self: *DrawList, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, c: Color) void {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const len = @sqrt(dx * dx + dy * dy);
        if (len <= 0.0) return;
        const ext = thickness * 0.5 + SDF_AA_MARGIN;
        // Oriented bounding quad: extended past each endpoint and offset across the
        // segment by `ext`, so the AA band around the flat caps and edges is covered.
        const ux = dx / len;
        const uy = dy / len;
        const ex = ux * ext;
        const ey = uy * ext;
        const ox = -uy * ext;
        const oy = ux * ext;
        const x0e = x0 - ex;
        const y0e = y0 - ey;
        const x1e = x1 + ex;
        const y1e = y1 + ey;
        const corners = [4][2]f32{
            .{ x0e + ox, y0e + oy },
            .{ x0e - ox, y0e - oy },
            .{ x1e - ox, y1e - oy },
            .{ x1e + ox, y1e + oy },
        };
        self.pushSdfQuad(corners, sdfVert(x0, y0, x1, y1, thickness * 0.5, 0, SDF_SEGMENT, c));
    }

    /// Filled anti-aliased circle (a capsule with a zero-length segment).
    pub fn circle(self: *DrawList, cx: f32, cy: f32, radius: f32, c: Color) void {
        self.sdfDisc(cx, cy, radius, 0, c);
    }

    /// Circle outline: a stroke of `thickness` centered on the circle of `radius`
    /// (spans [radius - thickness/2, radius + thickness/2]). The FPM ring.
    pub fn ring(self: *DrawList, cx: f32, cy: f32, radius: f32, thickness: f32, c: Color) void {
        self.sdfDisc(cx, cy, radius, thickness * 0.5, c);
    }

    fn sdfDisc(self: *DrawList, cx: f32, cy: f32, radius: f32, border: f32, c: Color) void {
        const ext = radius + border + SDF_AA_MARGIN;
        self.pushSdfQuad(axisAlignedQuad(cx, cy, ext, ext), sdfVert(cx, cy, cx, cy, radius, border, SDF_CAPSULE, c));
    }

    /// Filled rounded rectangle (corner radius `corner_radius`); corner_radius 0 is
    /// a crisp AA'd rect. For panels/menu chrome.
    pub fn roundedRect(self: *DrawList, x: f32, y: f32, w: f32, h: f32, corner_radius: f32, c: Color) void {
        self.sdfBox(x, y, w, h, corner_radius, 0, c);
    }

    /// Rounded-rectangle outline of `thickness`, centered on the rect edge.
    pub fn roundedRectOutline(self: *DrawList, x: f32, y: f32, w: f32, h: f32, corner_radius: f32, thickness: f32, c: Color) void {
        self.sdfBox(x, y, w, h, corner_radius, thickness * 0.5, c);
    }

    fn sdfBox(self: *DrawList, x: f32, y: f32, w: f32, h: f32, corner_radius: f32, border: f32, c: Color) void {
        // The rounded-box SDF needs the corner radius to fit inside the box; a
        // larger radius makes (half-extent - radius) negative and distorts the
        // shape, so clamp to the largest radius the box can hold (a pill/stadium).
        const r = @min(corner_radius, @min(w, h) * 0.5);
        const m = border + SDF_AA_MARGIN;
        const cx = x + w * 0.5;
        const cy = y + h * 0.5;
        // ax,ay = center; bx,by = half-extent; radius = corner radius.
        self.pushSdfQuad(axisAlignedQuad(cx, cy, w * 0.5 + m, h * 0.5 + m), sdfVert(cx, cy, w * 0.5, h * 0.5, r, border, SDF_BOX, c));
    }

    /// Connected anti-aliased stroke through `pts` (>= 2 points), drawn as ONE SDF
    /// element: the fragment shader takes the min distance over the path's segments,
    /// so joints have single coverage (no translucent double-blend) and fill (no
    /// butt-corner gaps), with round joins/ends. The points go in a frame-global
    /// pool the backend uploads to an SSBO; this quad carries the (offset, count)
    /// into it. Dropped silently if it would overflow the point pool or sdf stream.
    pub fn polyline(self: *DrawList, pts: []const [2]f32, thickness: f32, c: Color) void {
        if (pts.len < 2) return;
        if (@as(usize, self.path_count) + pts.len > MAX_PATH_POINTS) return;
        if (self.sdf_count + 6 > MAX_SDF_VERTS) return;
        const offset = self.path_count;
        var min_x = pts[0][0];
        var min_y = pts[0][1];
        var max_x = pts[0][0];
        var max_y = pts[0][1];
        for (pts) |p| {
            self.path_points[self.path_count] = p;
            self.path_count += 1;
            min_x = @min(min_x, p[0]);
            min_y = @min(min_y, p[1]);
            max_x = @max(max_x, p[0]);
            max_y = @max(max_y, p[1]);
        }
        const ext = thickness * 0.5 + SDF_AA_MARGIN;
        const cx = (min_x + max_x) * 0.5;
        const cy = (min_y + max_y) * 0.5;
        const quad = axisAlignedQuad(cx, cy, (max_x - min_x) * 0.5 + ext, (max_y - min_y) * 0.5 + ext);
        // ax,ay = (point offset, count) into path_points; radius = half-thickness.
        self.pushSdfQuad(quad, sdfVert(@floatFromInt(offset), @floatFromInt(pts.len), 0, 0, thickness * 0.5, 0, SDF_POLYLINE, c));
    }

    /// Solid triangle (three shape verts). Used for the HUD lubber marker and,
    /// later, the waterline / flight-path symbols.
    pub fn tri(self: *DrawList, x0: f32, y0: f32, x1: f32, y1: f32, x2: f32, y2: f32, c: Color) void {
        if (self.shape_count + 3 > MAX_SHAPE_VERTS) return;
        self.pushShapeVert(x0, y0, c);
        self.pushShapeVert(x1, y1, c);
        self.pushShapeVert(x2, y2, c);
    }

    /// Lay out a left-aligned ASCII string starting at (x, y) (top-left of the
    /// first cell). Non-ASCII bytes render as '?'. Returns the pixel width of the
    /// glyphs actually emitted: if the buffer fills mid-string the return shrinks
    /// to match, so callers sizing a backing panel never overshoot.
    pub fn text(self: *DrawList, x: f32, y: f32, scale: f32, c: Color, bytes: []const u8) f32 {
        const cell = GLYPH_PX * scale;
        var drawn: usize = 0;
        for (bytes, 0..) |ch, i| {
            if (self.glyph_count + 6 > MAX_GLYPH_VERTS) break;
            const cx = x + @as(f32, @floatFromInt(i)) * cell;
            const char_idx: u32 = if (ch >= 128) '?' else ch;
            self.pushGlyphQuad(cx, y, cell, c, char_idx);
            drawn += 1;
        }
        return textWidth(drawn, scale);
    }

    /// Right-aligned ASCII string: the right edge of the last cell lands at
    /// `x_right`, text growing leftward. Returns the emitted pixel width (same as
    /// `text`). For numeric readouts pinned to a right margin (the altitude column).
    pub fn textRight(self: *DrawList, x_right: f32, y: f32, scale: f32, c: Color, bytes: []const u8) f32 {
        const w = textWidth(bytes.len, scale);
        return self.text(x_right - w, y, scale, c, bytes);
    }

    fn pushGlyphQuad(self: *DrawList, x: f32, y: f32, cell: f32, c: Color, char_idx: u32) void {
        const x1 = x + cell;
        const y1 = y + cell;
        const verts = [6]GlyphVertex{
            glyphVert(x, y, 0, 0, c, char_idx),
            glyphVert(x, y1, 0, 1, c, char_idx),
            glyphVert(x1, y1, 1, 1, c, char_idx),
            glyphVert(x, y, 0, 0, c, char_idx),
            glyphVert(x1, y1, 1, 1, c, char_idx),
            glyphVert(x1, y, 1, 0, c, char_idx),
        };
        for (verts) |v| {
            self.glyph_verts[self.glyph_count] = v;
            self.glyph_count += 1;
        }
    }

    /// Emit the two triangles of an SDF element's bounding quad from its four
    /// corners (any consistent order), all carrying the same shape params from
    /// `template` (its `x`/`y` are overwritten per corner). The pipeline does not
    /// cull and the fragment shader does the distance test, so only the bounding
    /// region matters, not the winding.
    fn pushSdfQuad(self: *DrawList, corners: [4][2]f32, template: SdfVertex) void {
        if (self.sdf_count + 6 > MAX_SDF_VERTS) return;
        self.writeSdfQuad(self.sdf_count, corners, template);
        self.sdf_count += 6;
    }

    /// Write the 6 verts of an SDF quad at an existing slot (no append). Backs both
    /// `pushSdfQuad` and the reserve/commit pair for deferred-size content.
    fn writeSdfQuad(self: *DrawList, slot: u32, corners: [4][2]f32, template: SdfVertex) void {
        const order = [6]usize{ 0, 1, 2, 0, 2, 3 };
        for (order, 0..) |idx, i| {
            var v = template;
            v.x = corners[idx][0];
            v.y = corners[idx][1];
            self.sdf_verts[slot + @as(u32, @intCast(i))] = v;
        }
    }

    /// Reserve a rounded-rect (filled) SDF slot NOW so it renders BEHIND later SDF
    /// content, to be positioned by `commitRoundedRect` once its size is known.
    /// Used for a panel background, which sits behind its widgets but whose extent
    /// is content-driven. Returns the slot's first vertex index, or null if full.
    /// The placeholder is a zero-area transparent quad; the caller must commit it.
    pub fn reserveRoundedRect(self: *DrawList) ?u32 {
        if (self.sdf_count + 6 > MAX_SDF_VERTS) return null;
        const slot = self.sdf_count;
        self.sdf_count += 6;
        self.writeSdfQuad(slot, .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } }, sdfVert(0, 0, 0, 0, 0, 0, SDF_BOX, Color.rgba(0, 0, 0, 0)));
        return slot;
    }

    /// Fill a slot from `reserveRoundedRect` with a rounded filled rect (matches
    /// `roundedRect`, but writes in place rather than appending).
    pub fn commitRoundedRect(self: *DrawList, slot: u32, x: f32, y: f32, w: f32, h: f32, corner_radius: f32, c: Color) void {
        const r = @min(corner_radius, @min(w, h) * 0.5);
        const m = SDF_AA_MARGIN; // border = 0 (filled)
        const cx = x + w * 0.5;
        const cy = y + h * 0.5;
        self.writeSdfQuad(slot, axisAlignedQuad(cx, cy, w * 0.5 + m, h * 0.5 + m), sdfVert(cx, cy, w * 0.5, h * 0.5, r, 0, SDF_BOX, c));
    }

    /// Emit a right-aligned field of GPU-decoded digit slots: the rightmost cell
    /// ends at `x_right`, growing leftward, `slots` cells wide. Each slot carries
    /// its decimal place (0 = ones); numeric.vert reads the value from a GPU buffer
    /// and picks the glyph per place. The value is NOT known here (it lives on the
    /// GPU); this only lays out the slots. Leading slots above the value's
    /// magnitude render blank (the shader emits the space glyph).
    pub fn numberField(self: *DrawList, x_right: f32, y: f32, scale: f32, c: Color, slots: u32) void {
        const cell = GLYPH_PX * scale;
        var place: u32 = 0;
        while (place < slots) : (place += 1) {
            if (self.numeric_count + 6 > MAX_NUMERIC_VERTS) break;
            // place 0 (ones) is the rightmost cell; place grows leftward.
            const x = x_right - @as(f32, @floatFromInt(place + 1)) * cell;
            self.pushNumericQuad(x, y, cell, c, place);
        }
    }

    fn pushNumericQuad(self: *DrawList, x: f32, y: f32, cell: f32, c: Color, place: u32) void {
        const x1 = x + cell;
        const y1 = y + cell;
        const verts = [6]NumericVertex{
            numericVert(x, y, 0, 0, c, place),
            numericVert(x, y1, 0, 1, c, place),
            numericVert(x1, y1, 1, 1, c, place),
            numericVert(x, y, 0, 0, c, place),
            numericVert(x1, y1, 1, 1, c, place),
            numericVert(x1, y, 1, 0, c, place),
        };
        for (verts) |v| {
            self.numeric_verts[self.numeric_count] = v;
            self.numeric_count += 1;
        }
    }

    /// Emit one quad spanning a vertical gauge bar's full extent (`y_bottom` >
    /// `y_top` in screen px, since Y grows downward). The fill height is
    /// GPU-driven: gauge.vert reads the value and shrinks the top toward `y_top`
    /// by the fill fraction. The caller draws the static track separately.
    pub fn gaugeBar(self: *DrawList, x0: f32, x1: f32, y_bottom: f32, y_top: f32, c: Color) void {
        if (self.gauge_count + 6 > MAX_GAUGE_VERTS) return;
        const verts = [6]GaugeVertex{
            gaugeVert(x0, y_bottom, y_top, 0, c),
            gaugeVert(x0, y_bottom, y_top, 1, c),
            gaugeVert(x1, y_bottom, y_top, 1, c),
            gaugeVert(x0, y_bottom, y_top, 0, c),
            gaugeVert(x1, y_bottom, y_top, 1, c),
            gaugeVert(x1, y_bottom, y_top, 0, c),
        };
        for (verts) |v| {
            self.gauge_verts[self.gauge_count] = v;
            self.gauge_count += 1;
        }
    }
};

fn glyphVert(x: f32, y: f32, u: f32, v: f32, c: Color, char_idx: u32) GlyphVertex {
    return .{ .x = x, .y = y, .u = u, .v = v, .r = c.r, .g = c.g, .b = c.b, .a = c.a, .char_idx = char_idx };
}

fn numericVert(x: f32, y: f32, u: f32, v: f32, c: Color, place: u32) NumericVertex {
    return .{ .x = x, .y = y, .u = u, .v = v, .r = c.r, .g = c.g, .b = c.b, .a = c.a, .place = place };
}

fn gaugeVert(x: f32, y_bottom: f32, y_top: f32, fill_weight: f32, c: Color) GaugeVertex {
    return .{ .x = x, .y_bottom = y_bottom, .y_top = y_top, .fill_weight = fill_weight, .r = c.r, .g = c.g, .b = c.b, .a = c.a };
}

/// Build an SDF vertex template (shape params); `x`/`y` are filled per corner by
/// `pushSdfQuad`. See `SdfVertex` for the capsule/box/segment/polyline field meanings.
fn sdfVert(ax: f32, ay: f32, bx: f32, by: f32, radius: f32, border: f32, kind: f32, c: Color) SdfVertex {
    return .{ .x = 0, .y = 0, .ax = ax, .ay = ay, .bx = bx, .by = by, .radius = radius, .border = border, .kind = kind, .r = c.r, .g = c.g, .b = c.b, .a = c.a };
}

/// Axis-aligned bounding quad centered at (cx, cy) with half-width/height (hw, hh),
/// corners CCW from top-left, for `pushSdfQuad`.
fn axisAlignedQuad(cx: f32, cy: f32, hw: f32, hh: f32) [4][2]f32 {
    return .{
        .{ cx - hw, cy - hh },
        .{ cx - hw, cy + hh },
        .{ cx + hw, cy + hh },
        .{ cx + hw, cy - hh },
    };
}

// =============================================================================
// IBM PC BIOS 8x8 font: 128 ASCII characters, 8 bytes each, MSB = leftmost pixel.
// Source: susam/pcface (extracted from IBM PC BIOS ROM, public domain).
//
// The bytes live in the core (no GPU dependency) so the font contract is owned
// in one place alongside GLYPH_PX and the char-index mapping in `text()`. A
// backend uploads this to wherever its glyph shader samples; the only
// backend-specific part is the upload, not the data.
// =============================================================================

pub const font_8x8 = [1024]u8{
    // 0x00-0x07
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // NUL
    0x7e, 0x81, 0xa5, 0x81, 0xbd, 0x99, 0x81, 0x7e, // SOH
    0x7e, 0xff, 0xdb, 0xff, 0xc3, 0xe7, 0xff, 0x7e, // STX
    0x6c, 0xfe, 0xfe, 0xfe, 0x7c, 0x38, 0x10, 0x00, // ETX
    0x10, 0x38, 0x7c, 0xfe, 0x7c, 0x38, 0x10, 0x00, // EOT
    0x38, 0x7c, 0x38, 0xfe, 0xfe, 0x7c, 0x38, 0x7c, // ENQ
    0x10, 0x10, 0x38, 0x7c, 0xfe, 0x7c, 0x38, 0x7c, // ACK
    0x00, 0x00, 0x18, 0x3c, 0x3c, 0x18, 0x00, 0x00, // BEL
    // 0x08-0x0F
    0xff, 0xff, 0xe7, 0xc3, 0xc3, 0xe7, 0xff, 0xff, // BS
    0x00, 0x3c, 0x66, 0x42, 0x42, 0x66, 0x3c, 0x00, // HT
    0xff, 0xc3, 0x99, 0xbd, 0xbd, 0x99, 0xc3, 0xff, // LF
    0x0f, 0x07, 0x0f, 0x7d, 0xcc, 0xcc, 0xcc, 0x78, // VT
    0x3c, 0x66, 0x66, 0x66, 0x3c, 0x18, 0x7e, 0x18, // FF
    0x3f, 0x33, 0x3f, 0x30, 0x30, 0x70, 0xf0, 0xe0, // CR
    0x7f, 0x63, 0x7f, 0x63, 0x63, 0x67, 0xe6, 0xc0, // SO
    0x99, 0x5a, 0x3c, 0xe7, 0xe7, 0x3c, 0x5a, 0x99, // SI
    // 0x10-0x17
    0x80, 0xe0, 0xf8, 0xfe, 0xf8, 0xe0, 0x80, 0x00, // DLE
    0x02, 0x0e, 0x3e, 0xfe, 0x3e, 0x0e, 0x02, 0x00, // DC1
    0x18, 0x3c, 0x7e, 0x18, 0x18, 0x7e, 0x3c, 0x18, // DC2
    0x66, 0x66, 0x66, 0x66, 0x66, 0x00, 0x66, 0x00, // DC3
    0x7f, 0xdb, 0xdb, 0x7b, 0x1b, 0x1b, 0x1b, 0x00, // DC4
    0x3e, 0x63, 0x38, 0x6c, 0x6c, 0x38, 0xcc, 0x78, // NAK
    0x00, 0x00, 0x00, 0x00, 0x7e, 0x7e, 0x7e, 0x00, // SYN
    0x18, 0x3c, 0x7e, 0x18, 0x7e, 0x3c, 0x18, 0xff, // ETB
    // 0x18-0x1F
    0x18, 0x3c, 0x7e, 0x18, 0x18, 0x18, 0x18, 0x00, // CAN
    0x18, 0x18, 0x18, 0x18, 0x7e, 0x3c, 0x18, 0x00, // EM
    0x00, 0x18, 0x0c, 0xfe, 0x0c, 0x18, 0x00, 0x00, // SUB
    0x00, 0x30, 0x60, 0xfe, 0x60, 0x30, 0x00, 0x00, // ESC
    0x00, 0x00, 0xc0, 0xc0, 0xc0, 0xfe, 0x00, 0x00, // FS
    0x00, 0x24, 0x66, 0xff, 0x66, 0x24, 0x00, 0x00, // GS
    0x00, 0x18, 0x3c, 0x7e, 0xff, 0xff, 0x00, 0x00, // RS
    0x00, 0xff, 0xff, 0x7e, 0x3c, 0x18, 0x00, 0x00, // US
    // 0x20-0x27  (space through ')
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // space
    0x30, 0x78, 0x78, 0x30, 0x30, 0x00, 0x30, 0x00, // !
    0x6c, 0x6c, 0x6c, 0x00, 0x00, 0x00, 0x00, 0x00, // "
    0x6c, 0x6c, 0xfe, 0x6c, 0xfe, 0x6c, 0x6c, 0x00, // #
    0x30, 0x7c, 0xc0, 0x78, 0x0c, 0xf8, 0x30, 0x00, // $
    0x00, 0xc6, 0xcc, 0x18, 0x30, 0x66, 0xc6, 0x00, // %
    0x38, 0x6c, 0x38, 0x76, 0xdc, 0xcc, 0x76, 0x00, // &
    0x60, 0x60, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, // '
    // 0x28-0x2F  ( through /
    0x18, 0x30, 0x60, 0x60, 0x60, 0x30, 0x18, 0x00, // (
    0x60, 0x30, 0x18, 0x18, 0x18, 0x30, 0x60, 0x00, // )
    0x00, 0x66, 0x3c, 0xff, 0x3c, 0x66, 0x00, 0x00, // *
    0x00, 0x30, 0x30, 0xfc, 0x30, 0x30, 0x00, 0x00, // +
    0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30, 0x60, // ,
    0x00, 0x00, 0x00, 0xfc, 0x00, 0x00, 0x00, 0x00, // -
    0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x30, 0x00, // .
    0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0, 0x80, 0x00, // /
    // 0x30-0x37  0 through 7
    0x7c, 0xc6, 0xce, 0xde, 0xf6, 0xe6, 0x7c, 0x00, // 0
    0x30, 0x70, 0x30, 0x30, 0x30, 0x30, 0xfc, 0x00, // 1
    0x78, 0xcc, 0x0c, 0x38, 0x60, 0xcc, 0xfc, 0x00, // 2
    0x78, 0xcc, 0x0c, 0x38, 0x0c, 0xcc, 0x78, 0x00, // 3
    0x1c, 0x3c, 0x6c, 0xcc, 0xfe, 0x0c, 0x1e, 0x00, // 4
    0xfc, 0xc0, 0xf8, 0x0c, 0x0c, 0xcc, 0x78, 0x00, // 5
    0x38, 0x60, 0xc0, 0xf8, 0xcc, 0xcc, 0x78, 0x00, // 6
    0xfc, 0xcc, 0x0c, 0x18, 0x30, 0x30, 0x30, 0x00, // 7
    // 0x38-0x3F  8 through ?
    0x78, 0xcc, 0xcc, 0x78, 0xcc, 0xcc, 0x78, 0x00, // 8
    0x78, 0xcc, 0xcc, 0x7c, 0x0c, 0x18, 0x70, 0x00, // 9
    0x00, 0x30, 0x30, 0x00, 0x00, 0x30, 0x30, 0x00, // :
    0x00, 0x30, 0x30, 0x00, 0x00, 0x30, 0x30, 0x60, // ;
    0x18, 0x30, 0x60, 0xc0, 0x60, 0x30, 0x18, 0x00, // <
    0x00, 0x00, 0xfc, 0x00, 0x00, 0xfc, 0x00, 0x00, // =
    0x60, 0x30, 0x18, 0x0c, 0x18, 0x30, 0x60, 0x00, // >
    0x78, 0xcc, 0x0c, 0x18, 0x30, 0x00, 0x30, 0x00, // ?
    // 0x40-0x47  @ through G
    0x7c, 0xc6, 0xde, 0xde, 0xde, 0xc0, 0x78, 0x00, // @
    0x30, 0x78, 0xcc, 0xcc, 0xfc, 0xcc, 0xcc, 0x00, // A
    0xfc, 0x66, 0x66, 0x7c, 0x66, 0x66, 0xfc, 0x00, // B
    0x3c, 0x66, 0xc0, 0xc0, 0xc0, 0x66, 0x3c, 0x00, // C
    0xf8, 0x6c, 0x66, 0x66, 0x66, 0x6c, 0xf8, 0x00, // D
    0xfe, 0x62, 0x68, 0x78, 0x68, 0x62, 0xfe, 0x00, // E
    0xfe, 0x62, 0x68, 0x78, 0x68, 0x60, 0xf0, 0x00, // F
    0x3c, 0x66, 0xc0, 0xc0, 0xce, 0x66, 0x3e, 0x00, // G
    // 0x48-0x4F  H through O
    0xcc, 0xcc, 0xcc, 0xfc, 0xcc, 0xcc, 0xcc, 0x00, // H
    0x78, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00, // I
    0x1e, 0x0c, 0x0c, 0x0c, 0xcc, 0xcc, 0x78, 0x00, // J
    0xe6, 0x66, 0x6c, 0x78, 0x6c, 0x66, 0xe6, 0x00, // K
    0xf0, 0x60, 0x60, 0x60, 0x62, 0x66, 0xfe, 0x00, // L
    0xc6, 0xee, 0xfe, 0xfe, 0xd6, 0xc6, 0xc6, 0x00, // M
    0xc6, 0xe6, 0xf6, 0xde, 0xce, 0xc6, 0xc6, 0x00, // N
    0x38, 0x6c, 0xc6, 0xc6, 0xc6, 0x6c, 0x38, 0x00, // O
    // 0x50-0x57  P through W
    0xfc, 0x66, 0x66, 0x7c, 0x60, 0x60, 0xf0, 0x00, // P
    0x78, 0xcc, 0xcc, 0xcc, 0xdc, 0x78, 0x1c, 0x00, // Q
    0xfc, 0x66, 0x66, 0x7c, 0x6c, 0x66, 0xe6, 0x00, // R
    0x78, 0xcc, 0xe0, 0x70, 0x1c, 0xcc, 0x78, 0x00, // S
    0xfc, 0xb4, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00, // T
    0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xfc, 0x00, // U
    0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0x78, 0x30, 0x00, // V
    0xc6, 0xc6, 0xc6, 0xd6, 0xfe, 0xee, 0xc6, 0x00, // W
    // 0x58-0x5F  X through _
    0xc6, 0xc6, 0x6c, 0x38, 0x38, 0x6c, 0xc6, 0x00, // X
    0xcc, 0xcc, 0xcc, 0x78, 0x30, 0x30, 0x78, 0x00, // Y
    0xfe, 0xc6, 0x8c, 0x18, 0x32, 0x66, 0xfe, 0x00, // Z
    0x78, 0x60, 0x60, 0x60, 0x60, 0x60, 0x78, 0x00, // [
    0xc0, 0x60, 0x30, 0x18, 0x0c, 0x06, 0x02, 0x00, // backslash
    0x78, 0x18, 0x18, 0x18, 0x18, 0x18, 0x78, 0x00, // ]
    0x10, 0x38, 0x6c, 0xc6, 0x00, 0x00, 0x00, 0x00, // ^
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, // _
    // 0x60-0x67  ` through g
    0x30, 0x30, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, // `
    0x00, 0x00, 0x78, 0x0c, 0x7c, 0xcc, 0x76, 0x00, // a
    0xe0, 0x60, 0x60, 0x7c, 0x66, 0x66, 0xdc, 0x00, // b
    0x00, 0x00, 0x78, 0xcc, 0xc0, 0xcc, 0x78, 0x00, // c
    0x1c, 0x0c, 0x0c, 0x7c, 0xcc, 0xcc, 0x76, 0x00, // d
    0x00, 0x00, 0x78, 0xcc, 0xfc, 0xc0, 0x78, 0x00, // e
    0x38, 0x6c, 0x60, 0xf0, 0x60, 0x60, 0xf0, 0x00, // f
    0x00, 0x00, 0x76, 0xcc, 0xcc, 0x7c, 0x0c, 0xf8, // g
    // 0x68-0x6F  h through o
    0xe0, 0x60, 0x6c, 0x76, 0x66, 0x66, 0xe6, 0x00, // h
    0x30, 0x00, 0x70, 0x30, 0x30, 0x30, 0x78, 0x00, // i
    0x0c, 0x00, 0x0c, 0x0c, 0x0c, 0xcc, 0xcc, 0x78, // j
    0xe0, 0x60, 0x66, 0x6c, 0x78, 0x6c, 0xe6, 0x00, // k
    0x70, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00, // l
    0x00, 0x00, 0xcc, 0xfe, 0xfe, 0xd6, 0xc6, 0x00, // m
    0x00, 0x00, 0xf8, 0xcc, 0xcc, 0xcc, 0xcc, 0x00, // n
    0x00, 0x00, 0x78, 0xcc, 0xcc, 0xcc, 0x78, 0x00, // o
    // 0x70-0x77  p through w
    0x00, 0x00, 0xdc, 0x66, 0x66, 0x7c, 0x60, 0xf0, // p
    0x00, 0x00, 0x76, 0xcc, 0xcc, 0x7c, 0x0c, 0x1e, // q
    0x00, 0x00, 0xdc, 0x76, 0x66, 0x60, 0xf0, 0x00, // r
    0x00, 0x00, 0x7c, 0xc0, 0x78, 0x0c, 0xf8, 0x00, // s
    0x10, 0x30, 0x7c, 0x30, 0x30, 0x34, 0x18, 0x00, // t
    0x00, 0x00, 0xcc, 0xcc, 0xcc, 0xcc, 0x76, 0x00, // u
    0x00, 0x00, 0xcc, 0xcc, 0xcc, 0x78, 0x30, 0x00, // v
    0x00, 0x00, 0xc6, 0xd6, 0xfe, 0xfe, 0x6c, 0x00, // w
    // 0x78-0x7F  x through DEL
    0x00, 0x00, 0xc6, 0x6c, 0x38, 0x6c, 0xc6, 0x00, // x
    0x00, 0x00, 0xcc, 0xcc, 0xcc, 0x7c, 0x0c, 0xf8, // y
    0x00, 0x00, 0xfc, 0x98, 0x30, 0x64, 0xfc, 0x00, // z
    0x1c, 0x30, 0x30, 0xe0, 0x30, 0x30, 0x1c, 0x00, // {
    0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00, // |
    0xe0, 0x30, 0x30, 0x1c, 0x30, 0x30, 0xe0, 0x00, // }
    0x76, 0xdc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // ~
    0x00, 0x10, 0x38, 0x6c, 0xc6, 0xc6, 0xfe, 0x00, // DEL
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "DrawList: rect emits 6 verts at the right corners" {
    var dl: DrawList = .{};
    dl.rect(10, 20, 100, 30, Color.white);
    try testing.expectEqual(@as(u32, 6), dl.shape_count);
    try testing.expectEqual(@as(f32, 10), dl.shape_verts[0].x);
    try testing.expectEqual(@as(f32, 20), dl.shape_verts[0].y);
    // Third vertex is bottom-right (x+w, y+h).
    try testing.expectEqual(@as(f32, 110), dl.shape_verts[2].x);
    try testing.expectEqual(@as(f32, 50), dl.shape_verts[2].y);
}

test "DrawList: reserveRoundedRect holds a slot before later SDF; commit fills it in place" {
    var dl: DrawList = .{};
    const slot = dl.reserveRoundedRect().?;
    try testing.expectEqual(@as(u32, 0), slot);
    try testing.expectEqual(@as(u32, 6), dl.sdf_count); // 6 verts reserved up front

    // A later SDF element lands AFTER the reserved slot, so the reserved fill stays
    // behind it in append order (the whole point: panel bg behind its widgets).
    dl.roundedRect(0, 0, 10, 10, 2, Color.white);
    try testing.expectEqual(@as(u32, 12), dl.sdf_count);

    // Commit positions the reserved slot without moving it.
    dl.commitRoundedRect(slot, 5, 6, 100, 40, 4, Color.rgba(0.1, 0.2, 0.3, 0.9));
    const v = dl.sdf_verts[0]; // first vert of the reserved (still-first) quad
    try testing.expectApproxEqAbs(@as(f32, 55), v.ax, 1e-4); // box center x = 5 + 100/2
    try testing.expectApproxEqAbs(@as(f32, 26), v.ay, 1e-4); // box center y = 6 + 40/2
    try testing.expectApproxEqAbs(@as(f32, 4), v.radius, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), v.border, 1e-4); // filled, not an outline
    try testing.expectApproxEqAbs(@as(f32, 0.9), v.a, 1e-4);
}

test "DrawList: clear resets all streams" {
    var dl: DrawList = .{};
    dl.rect(0, 0, 1, 1, Color.white);
    dl.line(0, 0, 10, 0, 2, Color.white);
    _ = dl.text(0, 0, 1, Color.white, "hi");
    try testing.expect(dl.shape_count > 0);
    try testing.expect(dl.sdf_count > 0);
    try testing.expect(dl.glyph_count > 0);
    dl.clear();
    try testing.expectEqual(@as(u32, 0), dl.shape_count);
    try testing.expectEqual(@as(u32, 0), dl.sdf_count);
    try testing.expectEqual(@as(u32, 0), dl.glyph_count);
}

test "DrawList: gradient puts top color on top edge, bottom on bottom edge" {
    var dl: DrawList = .{};
    const top = Color.rgba(1, 0, 0, 1);
    const bot = Color.rgba(0, 0, 1, 1);
    dl.rectGradient(0, 0, 10, 10, top, bot);
    // Vertex 0 is top-left -> top color; vertex 1 is bottom-left -> bottom color.
    try testing.expectEqual(@as(f32, 1), dl.shape_verts[0].r);
    try testing.expectEqual(@as(f32, 0), dl.shape_verts[0].b);
    try testing.expectEqual(@as(f32, 0), dl.shape_verts[1].r);
    try testing.expectEqual(@as(f32, 1), dl.shape_verts[1].b);
}

test "DrawList: text emits 6 verts per char and advances width" {
    var dl: DrawList = .{};
    const w = dl.text(5, 5, 2, Color.white, "abc");
    try testing.expectEqual(@as(u32, 18), dl.glyph_count);
    try testing.expectEqual(@as(f32, 3 * 8 * 2), w);
    try testing.expectEqual(@as(f32, 5), dl.glyph_verts[0].x);
    // Second char starts one cell (8*scale) to the right.
    try testing.expectEqual(@as(f32, 5 + 16), dl.glyph_verts[6].x);
}

test "Color.withAlpha: keeps rgb, replaces alpha" {
    const c = Color.rgba(0.2, 0.4, 0.6, 1.0).withAlpha(0.3);
    try testing.expectEqual(@as(f32, 0.2), c.r);
    try testing.expectEqual(@as(f32, 0.4), c.g);
    try testing.expectEqual(@as(f32, 0.6), c.b);
    try testing.expectEqual(@as(f32, 0.3), c.a);
}

test "DrawList: textRight ends the last glyph at x_right" {
    var dl: DrawList = .{};
    const scale: f32 = 2; // cell = 16 px, "ab" width = 32
    const w = dl.textRight(100, 5, scale, Color.white, "ab");
    try testing.expectEqual(@as(f32, 32), w);
    // First glyph's left edge = x_right - width.
    try testing.expectEqual(@as(f32, 68), dl.glyph_verts[0].x);
    // Second glyph starts one cell right (84); its right edge is x_right (100).
    try testing.expectEqual(@as(f32, 84), dl.glyph_verts[6].x);
}

test "DrawList: non-ascii byte maps to '?'" {
    var dl: DrawList = .{};
    _ = dl.text(0, 0, 1, Color.white, &.{0xFF});
    try testing.expectEqual(@as(u32, '?'), dl.glyph_verts[0].char_idx);
}

test "DrawList: line goes to the SDF stream as a segment, not the shape stream" {
    var dl: DrawList = .{};
    dl.line(0, 10, 100, 10, 4, Color.white);
    try testing.expectEqual(@as(u32, 0), dl.shape_count);
    try testing.expectEqual(@as(u32, 6), dl.sdf_count);
    const v = dl.sdf_verts[0];
    // Segment (butt-capped) params: endpoints A=(0,10), B=(100,10), radius =
    // thickness/2, filled. kind 2 = segment (not the round-capped capsule).
    try testing.expectEqual(@as(f32, 2), v.kind);
    try testing.expectEqual(@as(f32, 0), v.ax);
    try testing.expectEqual(@as(f32, 10), v.ay);
    try testing.expectEqual(@as(f32, 100), v.bx);
    try testing.expectEqual(@as(f32, 2), v.radius);
    try testing.expectEqual(@as(f32, 0), v.border);
    // Bounding quad pads by radius + AA margin (3.5) across the (vertical) normal:
    // every corner's y stays within [6.5, 13.5].
    for (dl.sdf_verts[0..6]) |c| {
        try testing.expect(c.y >= 6.5 - 1e-4 and c.y <= 13.5 + 1e-4);
    }
}

test "DrawList: circle is a filled zero-length capsule" {
    var dl: DrawList = .{};
    dl.circle(50, 60, 10, Color.white);
    try testing.expectEqual(@as(u32, 0), dl.shape_count);
    try testing.expectEqual(@as(u32, 6), dl.sdf_count);
    const v = dl.sdf_verts[0];
    try testing.expectEqual(@as(f32, 0), v.kind); // capsule
    try testing.expectEqual(v.ax, v.bx); // a == b
    try testing.expectEqual(v.ay, v.by);
    try testing.expectEqual(@as(f32, 10), v.radius);
    try testing.expectEqual(@as(f32, 0), v.border); // filled
    // Bounding quad covers radius + AA margin around the center.
    for (dl.sdf_verts[0..6]) |c| {
        try testing.expect(@abs(c.x - 50) <= 10 + 1.5 + 1e-4);
        try testing.expect(@abs(c.y - 60) <= 10 + 1.5 + 1e-4);
    }
}

test "DrawList: ring sets a nonzero border half-width" {
    var dl: DrawList = .{};
    dl.ring(0, 0, 20, 4, Color.white); // thickness 4 -> border 2
    try testing.expectEqual(@as(f32, 20), dl.sdf_verts[0].radius);
    try testing.expectEqual(@as(f32, 2), dl.sdf_verts[0].border);
}

test "DrawList: roundedRect is a box with center + half-extent" {
    var dl: DrawList = .{};
    dl.roundedRect(10, 20, 100, 40, 8, Color.white);
    try testing.expectEqual(@as(u32, 6), dl.sdf_count);
    const v = dl.sdf_verts[0];
    try testing.expectEqual(@as(f32, 1), v.kind); // box
    try testing.expectEqual(@as(f32, 60), v.ax); // center x = 10 + 100/2
    try testing.expectEqual(@as(f32, 40), v.ay); // center y = 20 + 40/2
    try testing.expectEqual(@as(f32, 50), v.bx); // half-extent x
    try testing.expectEqual(@as(f32, 20), v.by); // half-extent y
    try testing.expectEqual(@as(f32, 8), v.radius); // corner radius
}

test "DrawList: roundedRect clamps corner radius to fit the box" {
    var dl: DrawList = .{};
    // corner_radius 30 exceeds min(w,h)/2 = 20; the rounded-box SDF would distort
    // (half-extent - radius going negative) without the clamp.
    dl.roundedRect(0, 0, 100, 40, 30, Color.white);
    try testing.expectEqual(@as(f32, 20), dl.sdf_verts[0].radius);
    // A radius that already fits is left untouched.
    var dl2: DrawList = .{};
    dl2.roundedRect(0, 0, 100, 40, 8, Color.white);
    try testing.expectEqual(@as(f32, 8), dl2.sdf_verts[0].radius);
}

test "DrawList: resolveClipRanges partitions the SDF stream too" {
    var dl: DrawList = .{};
    dl.clear();
    dl.line(0, 0, 10, 0, 2, Color.white); // group 0: sdf [0,6)
    dl.pushClip(.{ .x = 0, .y = 0, .w = 100, .h = 20 });
    dl.circle(5, 5, 3, Color.white); // group 1: sdf [6,12)
    dl.popClip();
    var ranges: [MAX_CLIP_GROUPS]ResolvedClip = undefined;
    const n = dl.resolveClipRanges(&ranges);
    try testing.expectEqual(@as(u32, 3), n);
    try testing.expectEqual(@as(u32, 0), ranges[0].sdf_first);
    try testing.expectEqual(@as(u32, 6), ranges[0].sdf_count);
    try testing.expectEqual(@as(u32, 6), ranges[1].sdf_first);
    try testing.expectEqual(@as(u32, 6), ranges[1].sdf_count);
    try testing.expectEqual(@as(u32, 12), dl.sdf_count);
}

test "DrawList: SDF overflow appends are dropped, not out-of-bounds" {
    var dl: DrawList = .{};
    for (0..MAX_SDF_VERTS) |_| dl.circle(0, 0, 1, Color.white);
    try testing.expect(dl.sdf_count <= MAX_SDF_VERTS);
}

test "DrawList: polyline appends points and emits one SDF quad referencing them" {
    var dl: DrawList = .{};
    const pts = [_][2]f32{ .{ 0, 0 }, .{ 10, 0 }, .{ 10, 10 } };
    dl.polyline(&pts, 2, Color.white);
    try testing.expectEqual(@as(u32, 3), dl.path_count);
    try testing.expectEqual(@as(u32, 6), dl.sdf_count); // one quad
    const v = dl.sdf_verts[0];
    try testing.expectEqual(@as(f32, 3), v.kind); // polyline
    try testing.expectEqual(@as(f32, 0), v.ax); // point offset
    try testing.expectEqual(@as(f32, 3), v.ay); // point count
    try testing.expectEqual(@as(f32, 1), v.radius); // thickness/2
    try testing.expectEqual(@as(f32, 10), dl.path_points[1][0]); // points copied verbatim
    // Bounding quad covers the bbox + margin (ext = 1 + 1.5 = 2.5).
    for (dl.sdf_verts[0..6]) |q| {
        try testing.expect(q.x >= -2.5 - 1e-4 and q.x <= 12.5 + 1e-4);
        try testing.expect(q.y >= -2.5 - 1e-4 and q.y <= 12.5 + 1e-4);
    }
}

test "DrawList: polyline offset chains across calls; <2 points is a no-op" {
    var dl: DrawList = .{};
    const a = [_][2]f32{ .{ 0, 0 }, .{ 1, 1 } };
    dl.polyline(&a, 1, Color.white);
    dl.polyline(&a, 1, Color.white);
    try testing.expectEqual(@as(u32, 4), dl.path_count);
    try testing.expectEqual(@as(f32, 2), dl.sdf_verts[6].ax); // second quad's offset = 2
    const before = dl.sdf_count;
    const one = [_][2]f32{.{ 0, 0 }};
    dl.polyline(&one, 1, Color.white); // < 2 points -> dropped
    try testing.expectEqual(before, dl.sdf_count);
    try testing.expectEqual(@as(u32, 4), dl.path_count);
}

test "DrawList: polyline point-pool overflow is dropped, and clear resets it" {
    var dl: DrawList = .{};
    const seg = [_][2]f32{ .{ 0, 0 }, .{ 1, 1 } };
    for (0..MAX_PATH_POINTS) |_| dl.polyline(&seg, 1, Color.white);
    try testing.expect(dl.path_count <= MAX_PATH_POINTS);
    dl.clear();
    try testing.expectEqual(@as(u32, 0), dl.path_count);
}

test "DrawList: overflow appends are dropped, not out-of-bounds" {
    var dl: DrawList = .{};
    // Way more rects than capacity; must clamp silently.
    for (0..MAX_SHAPE_VERTS) |_| dl.rect(0, 0, 1, 1, Color.white);
    try testing.expect(dl.shape_count <= MAX_SHAPE_VERTS);
}

test "Screen.fromExtent: fraction-of-screen scale keys off height" {
    try testing.expectEqual(@as(f32, 1.0), Screen.fromExtent(1280, 720, 720, 1.0).scale);
    try testing.expectEqual(@as(f32, 2.0), Screen.fromExtent(2560, 1440, 720, 1.0).scale);
    try testing.expectEqual(@as(f32, 1.5), Screen.fromExtent(1920, 1080, 720, 1.0).scale);
    // user_pref multiplies on top of the resolution factor.
    try testing.expectEqual(@as(f32, 2.0), Screen.fromExtent(1280, 720, 720, 2.0).scale);
}

test "Screen.px / textScale: logical -> device" {
    const s = Screen.fromExtent(2560, 1440, 720, 1.0); // scale 2
    try testing.expectEqual(@as(f32, 20), s.px(10));
    try testing.expectEqual(@as(f32, 2.5), s.textScale(1.25)); // glyph scale 1.25 -> device 2.5
}

test "Screen.anchor: base points and scaled offset" {
    const s = Screen.fromExtent(1000, 800, 800, 1.0); // scale 1
    const tc = s.anchor(.top_center, 0, 0);
    try testing.expectEqual(@as(f32, 500), tc[0]);
    try testing.expectEqual(@as(f32, 0), tc[1]);
    const cr = s.anchor(.center_right, 0, 0);
    try testing.expectEqual(@as(f32, 1000), cr[0]);
    try testing.expectEqual(@as(f32, 400), cr[1]);
    const bc = s.anchor(.bottom_center, 0, 0);
    try testing.expectEqual(@as(f32, 500), bc[0]);
    try testing.expectEqual(@as(f32, 800), bc[1]);
    // Offset is scaled by `scale` (2 here).
    const s2 = Screen.fromExtent(1000, 1600, 800, 1.0); // scale 2
    const off = s2.anchor(.top_center, 0, 10);
    try testing.expectEqual(@as(f32, 500), off[0]);
    try testing.expectEqual(@as(f32, 20), off[1]);
}

test "DrawList: clear seeds exactly one full-screen clip group" {
    var dl: DrawList = .{};
    dl.clear();
    try testing.expectEqual(@as(u32, 1), dl.clip_count);
    try testing.expect(dl.clip_groups[0].clip == null);
    try testing.expectEqual(@as(u32, 0), dl.clip_groups[0].shape_start);
}

test "DrawList: pushClip/popClip record boundary starts at the live counts" {
    var dl: DrawList = .{};
    dl.clear();
    dl.rect(0, 0, 1, 1, Color.white); // 6 shape verts in group 0
    dl.pushClip(.{ .x = 0, .y = 0, .w = 100, .h = 20 });
    _ = dl.text(0, 0, 1, Color.white, "ab"); // 12 glyph verts in the clipped group
    dl.popClip();
    try testing.expectEqual(@as(u32, 3), dl.clip_count);
    try testing.expectEqual(@as(u32, 6), dl.clip_groups[1].shape_start);
    try testing.expectEqual(@as(u32, 0), dl.clip_groups[1].glyph_start);
    try testing.expectEqual(@as(u32, 6), dl.clip_groups[2].shape_start);
    try testing.expectEqual(@as(u32, 12), dl.clip_groups[2].glyph_start);
}

test "DrawList: resolveClipRanges partitions both streams in order" {
    var dl: DrawList = .{};
    dl.clear();
    dl.rect(0, 0, 1, 1, Color.white); // group 0: shapes [0,6)
    dl.pushClip(.{ .x = 0, .y = 0, .w = 100, .h = 20 });
    _ = dl.text(0, 0, 1, Color.white, "ab"); // group 1: glyphs [0,12)
    dl.popClip();
    dl.tri(0, 0, 1, 0, 0, 1, Color.white); // group 2: shapes [6,9)

    var ranges: [MAX_CLIP_GROUPS]ResolvedClip = undefined;
    const n = dl.resolveClipRanges(&ranges);
    try testing.expectEqual(@as(u32, 3), n);
    // Group 0: full screen, the rect.
    try testing.expect(ranges[0].clip == null);
    try testing.expectEqual(@as(u32, 0), ranges[0].shape_first);
    try testing.expectEqual(@as(u32, 6), ranges[0].shape_count);
    try testing.expectEqual(@as(u32, 0), ranges[0].glyph_count);
    // Group 1: clipped, the text (shapes-only count is zero here).
    try testing.expect(ranges[1].clip != null);
    try testing.expectEqual(@as(u32, 0), ranges[1].shape_count);
    try testing.expectEqual(@as(u32, 0), ranges[1].glyph_first);
    try testing.expectEqual(@as(u32, 12), ranges[1].glyph_count);
    // Group 2: full again, the triangle.
    try testing.expect(ranges[2].clip == null);
    try testing.expectEqual(@as(u32, 6), ranges[2].shape_first);
    try testing.expectEqual(@as(u32, 3), ranges[2].shape_count);
    // Ranges cover the whole streams.
    try testing.expectEqual(@as(u32, 9), dl.shape_count);
    try testing.expectEqual(@as(u32, 12), dl.glyph_count);
}

test "DrawList: tri emits three shape verts" {
    var dl: DrawList = .{};
    dl.tri(0, 0, 10, 0, 5, 10, Color.white);
    try testing.expectEqual(@as(u32, 3), dl.shape_count);
    try testing.expectEqual(@as(f32, 5), dl.shape_verts[2].x);
    try testing.expectEqual(@as(f32, 10), dl.shape_verts[2].y);
}

test "DrawList: clip-group overflow is dropped, not out-of-bounds" {
    var dl: DrawList = .{};
    dl.clear();
    for (0..MAX_CLIP_GROUPS + 5) |_| dl.pushClip(.{ .x = 0, .y = 0, .w = 1, .h = 1 });
    try testing.expect(dl.clip_count <= MAX_CLIP_GROUPS);
}
