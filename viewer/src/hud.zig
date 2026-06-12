//! HUD overlay: builds the per-frame `ui.DrawList` for the synthetic-vision HUD.
//! Pure-ish DrawList builder: it takes primitive `Inputs` (gathered by the
//! caller, scene.zig) and emits CPU geometry that `render/ui_backend.zig`
//! uploads. It imports only the `ui` and `math` modules, no app types, so it
//! drags no cross-tree tests and the seam to app state lives at the call site.
//!
//! All SVS readouts are aircraft-sourced: the HUD reports the aircraft's sensor
//! state (GPS/IMU), never the camera viewpoint. In free-cam the aircraft can be
//! frozen while the camera flies around; the HUD then reads frozen, as intended.
//! The dev/debug block is the exception: it deliberately reports the camera pose
//! (a viewpoint/dev readout) and is toggled via F1 (default off).

const std = @import("std");
const ui = @import("ui");
const math = @import("math");

/// Single mil-green accent for all HUD symbology; opacity varies per location
/// (full readouts, translucent ribbon), the RGB never does. Colors are
/// SDR-referred (1.0 = full); the UI backend's shaders encode them to the
/// swapchain transfer function at a chosen paper-white (`UI_PAPER_WHITE_NITS` in
/// render/ui_backend.zig), so they read consistently across sRGB / scRGB / PQ.
const GREEN = ui.Color.rgba(0.25, 0.85, 0.20, 1.0);
const RIBBON_ALPHA: f32 = 0.85; // ribbon symbology sits under the readouts
const RIBBON_COLOR = GREEN.withAlpha(RIBBON_ALPHA);
const READOUT_COLOR = GREEN;

/// Metric -> imperial conversions for the readouts. `M_TO_FT` is also the AGL
/// numeric scale pushed into numeric.vert (see scene.recordOverlay), so the
/// GPU-resident AGL digits and the CPU altitude readout share one factor.
pub const M_TO_FT: f32 = 3.280840;
const KMH_TO_KTS: f32 = 0.5399568; // 1 / 1.852
const MPS_TO_FTMIN: f32 = 196.8504; // m/s -> ft/min (aviation VSI unit)

/// Everything the HUD draws, as primitives gathered by the caller (scene.zig).
/// Keeping app types out of this struct is what lets hud.zig import only the
/// `ui` and `math` modules.
pub const Inputs = struct {
    svs: Svs,
    dev: Dev,
    att: Attitude,
    /// F1 toggle; default off (dev diagnostics, not SVS state).
    show_dev: bool,
};

/// SVS readouts + ribbon, aircraft-sourced. Metric units.
pub const Svs = struct {
    heading_rad: f32,
    speed_kmh: f32,
    mach: f32,
    alt_m: f32,
    vs_mps: f32,
    lat: f64,
    lon: f64,
    /// Whether a terrain DB / probe is present, so the GPU AGL readout is valid.
    /// The AGL value itself lives in a GPU buffer (decoded in numeric.vert), not
    /// here; this only gates whether the digit slots are emitted.
    agl_available: bool,
    /// Imperial units: altitude/AGL in ft, speed in kt, VSI in ft/min. The raw
    /// fields above stay metric; the readouts convert at draw time. The AGL digits
    /// are GPU-resident, so their unit scale is applied in numeric.vert, not here.
    imperial: bool,
};

/// Conformal attitude core inputs, all world-space primitives (gathered by
/// scene.zig). The symbology *content* is aircraft-referenced (`ac_front`, world
/// elevation bars) but it is drawn by projecting those directions through the
/// *camera* basis, so it stays conformal to the rendered terrain and slides to
/// track the aircraft nose under free-look (camera diverging from the aircraft).
/// In normal cockpit view the camera basis equals the aircraft's, so the waterline
/// lands at screen center and the ladder is centered.
pub const Attitude = struct {
    cam_right: math.Vec3, // camera basis, world space (unit)
    cam_up: math.Vec3,
    cam_front: math.Vec3,
    fov_y_rad: f32, // vertical FOV, matches the scene projection
    ac_front: math.Vec3, // aircraft boresight, world space (unit)
};

/// Dev/debug overlay: frame rate, VRAM/tile residency, sim state, and active
/// toggles. Toggled via F1 (default off). The `?` label fields are null when
/// the toggle sits at its default, so the line is omitted.
pub const Dev = struct {
    fps: f64,
    show_sim: bool,
    throttle: f32,
    auto_level: bool,
    vram_mb: f32,
    tiles_resident: u32,
    render_label: ?[]const u8,
    overlay_label: ?[]const u8,
    freeze_stream: ?[]const u8,
};

/// Build the whole HUD into `dl`. Caller supplies framebuffer dims (for the
/// per-frame `Screen`) and the gathered inputs.
pub fn draw(dl: *ui.DrawList, extent_w: f32, extent_h: f32, in: Inputs) void {
    // One scale for the whole HUD: fraction-of-screen (see ui.Screen). Every
    // component multiplies its logical-pixel layout by screen.scale so the HUD
    // grows together with resolution.
    const screen = ui.Screen.fromExtent(extent_w, extent_h, REF_H, USER_PREF);

    // Conformal core first: its dimmed strokes sit beneath the ribbon, and its
    // labels beneath the full-alpha readouts (text always draws over the SDF
    // stream, and within each stream draw order is append order).
    drawAttitude(dl, screen, in.att);
    if (in.show_dev) drawDebugBlock(dl, screen, in.dev);
    drawReadouts(dl, screen, in.svs);
    drawHeadingRibbon(dl, screen, in.svs.heading_rad);
}

// =============================================================================
// SVS numeric readouts (aircraft-sourced, metric, mil-green)
// =============================================================================

// Layout in logical pixels (multiplied by Screen.scale at draw time). Tunables
// for the visual fly-through: column inset, glyph sizes, line gap.
const READOUT_INSET: f32 = 24.0; // column inset from the screen edge
const READOUT_SUB_SCALE: f32 = 1.0; // unit / secondary-line glyph scale
const READOUT_VALUE_SCALE: f32 = 1.75; // primary value glyph scale
const READOUT_GAP: f32 = 4.0; // gap between stacked readout lines
const READOUT_CORNER_OFF: f32 = 16.0; // GPS coords inset from the bottom-left corner
const GAUGE_BAR_W: f32 = 7.0; // AGL gauge bar width (logical px)
const GAUGE_BAR_GAP: f32 = 8.0; // gap from the altitude column to the gauge bar
const GAUGE_TRACK_COLOR = ui.Color{ .r = 0.30, .g = 0.45, .b = 0.30, .a = 0.30 }; // dim track

/// Speed column (left of center), altitude/VSI column (right of center), and the
/// GPS coordinate readout (bottom-left corner). The unit sits a small-font space
/// off the value; the secondary line (Mach / VS) sits below.
fn drawReadouts(dl: *ui.DrawList, screen: ui.Screen, svs: Svs) void {
    const sub_scale = screen.textScale(READOUT_SUB_SCALE);
    const value_scale = screen.textScale(READOUT_VALUE_SCALE);
    const sub_cell = screen.px(ui.GLYPH_PX * READOUT_SUB_SCALE);
    const value_cell = screen.px(ui.GLYPH_PX * READOUT_VALUE_SCALE);
    const gap = screen.px(READOUT_GAP);

    // Two rows per column, centered on the screen mid-line: the value (with its
    // unit set small, on the value's baseline) on top, a secondary line below.
    const cy = screen.h * 0.5;
    const value_y = cy - (value_cell + gap + sub_cell) * 0.5;
    const sub_y = value_y + value_cell + gap;
    const unit_y = value_y + (value_cell - sub_cell); // small unit on the value's baseline

    var buf: [32]u8 = undefined;

    // ---- Speed column: left of center, left-aligned. "<spd> <unit>" + Mach ----
    const lx = screen.anchor(.center_left, READOUT_INSET, 0)[0];
    const speed = if (svs.imperial) svs.speed_kmh * KMH_TO_KTS else svs.speed_kmh;
    const speed_unit: []const u8 = if (svs.imperial) "kt" else "km/h";
    const spd = fmt(&buf, "{d:.0}", .{speed});
    const spd_w = ui.DrawList.textWidth(spd.len, value_scale);
    _ = dl.text(lx, value_y, value_scale, READOUT_COLOR, spd);
    _ = dl.text(lx + spd_w + gap, unit_y, sub_scale, READOUT_COLOR, speed_unit);
    _ = dl.text(lx, sub_y, sub_scale, READOUT_COLOR, fmt(&buf, "M {d:.2}", .{svs.mach}));

    // ---- Altitude column: right of center, right-aligned. "<alt> <unit>" + VS ----
    const rx = screen.anchor(.center_right, -READOUT_INSET, 0)[0];
    const alt = if (svs.imperial) svs.alt_m * M_TO_FT else svs.alt_m;
    const alt_unit: []const u8 = if (svs.imperial) "ft" else "m";
    const unit_w = ui.DrawList.textWidth(alt_unit.len, sub_scale);
    _ = dl.textRight(rx, unit_y, sub_scale, READOUT_COLOR, alt_unit);
    _ = dl.textRight(rx - unit_w - gap, value_y, value_scale, READOUT_COLOR, fmt(&buf, "{d:.0}", .{alt}));
    _ = dl.textRight(rx, sub_y, sub_scale, READOUT_COLOR, formatVs(&buf, svs.vs_mps, svs.imperial));

    // ---- AGL line below VS: GPU-decoded digit slots fed by the terrain probe.
    // The value lives in a GPU buffer (numeric.vert decodes it); here we only lay
    // out the slots and gate on DB presence. The field is fixed-width (the CPU
    // can't size it to the digit count, since the value is GPU-resident), so a
    // short "R" label sits flush against it to keep the readout tight.
    if (svs.agl_available) {
        const agl_y = sub_y + sub_cell + gap;
        const agl_slots: u32 = 5; // up to 99999; covers AGL in feet (m*3.28 > 9999)
        const field_w = ui.DrawList.textWidth(agl_slots, sub_scale);
        dl.numberField(rx, agl_y, sub_scale, READOUT_COLOR, agl_slots);
        const label_w = ui.DrawList.textWidth(1, sub_scale); // "R" (radio/AGL)
        _ = dl.text(rx - field_w - label_w, agl_y, sub_scale, READOUT_COLOR, "R");

        // AGL fill bar in the right margin (right of the readouts), spanning the
        // altitude column block: a dim static track + the GPU-driven fill.
        const bar_w = screen.px(GAUGE_BAR_W);
        const bar_x0 = rx + screen.px(GAUGE_BAR_GAP);
        const bar_top = value_y;
        const bar_bottom = agl_y + sub_cell;
        dl.rect(bar_x0, bar_top, bar_w, bar_bottom - bar_top, GAUGE_TRACK_COLOR);
        dl.gaugeBar(bar_x0, bar_x0 + bar_w, bar_bottom, bar_top, READOUT_COLOR);
    }

    // ---- GPS coordinates: bottom-left corner, left-aligned ----
    const lat_ch: u8 = if (svs.lat >= 0) 'N' else 'S';
    const lon_ch: u8 = if (svs.lon >= 0) 'E' else 'W';
    const coord_txt = fmt(&buf, "{d:.4}{c} {d:.4}{c}", .{ @abs(svs.lat), lat_ch, @abs(svs.lon), lon_ch });
    const corner = screen.anchor(.bottom_left, READOUT_CORNER_OFF, -(READOUT_CORNER_OFF + ui.GLYPH_PX * READOUT_SUB_SCALE));
    _ = dl.text(corner[0], corner[1], sub_scale, READOUT_COLOR, coord_txt);
}

/// Format the vertical-speed readout, signed. Metric: m/s, one decimal within
/// +/-10 m/s and integer beyond. Imperial: ft/min, integer (the conventional
/// aviation VSI unit and precision). Round to the displayed precision first, then
/// take the sign from the rounded value, so a near-zero or negative-zero rate
/// prints "+0.0 m/s" rather than "-0.0 m/s" or the garbled "+-0" (the sign string
/// and `{d}`'s own sign would otherwise disagree on signed zero).
fn formatVs(buf: []u8, vs_mps: f32, imperial: bool) []const u8 {
    if (imperial) {
        const r = @round(vs_mps * MPS_TO_FTMIN);
        const sign: []const u8 = if (r < 0) "-" else "+";
        return fmt(buf, "{s}{d:.0} ft/min", .{ sign, @abs(r) });
    }
    const one_dp = @abs(vs_mps) < 10;
    const q: f32 = if (one_dp) 10 else 1; // quantize to the displayed precision
    const r = @round(vs_mps * q) / q;
    const sign: []const u8 = if (r < 0) "-" else "+";
    const mag = @abs(r);
    return if (one_dp)
        fmt(buf, "{s}{d:.1} m/s", .{ sign, mag })
    else
        fmt(buf, "{s}{d:.0} m/s", .{ sign, mag });
}

/// bufPrint wrapper: on overflow draw nothing rather than an uninitialized
/// buffer. The returned slice is consumed by the immediately-following
/// `text`/`textRight` call before `buf` is reused, so one buffer is safe to share.
fn fmt(buf: []u8, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, f, args) catch buf[0..0];
}

// =============================================================================
// Conformal attitude core: horizon + pitch ladder + waterline
// =============================================================================
//
// Every symbol is a world-space direction projected through the camera basis (see
// `Attitude` + `project`), so the group overlays the rendered terrain and slides
// to track the aircraft nose under free-look. The ladder/horizon are clipped to a
// center box and dimmed; the readouts sit on top at full alpha.

const ATT_DIM: f32 = 0.6; // attitude-core opacity (readouts sit at full alpha over it)
const ATT_COLOR = GREEN.withAlpha(ATT_DIM);
const ATT_THICK: f32 = 2.0; // stroke thickness (logical px)
const ATT_BOX_W_FRAC: f32 = 0.5; // center clip box as a fraction of the screen
const ATT_BOX_H_FRAC: f32 = 0.5;
const LADDER_STEP_DEG: f32 = 10.0; // pitch between labeled bars
const LADDER_MAX_DEG: f32 = 80.0; // highest/lowest bar
const LADDER_HALF_AZ_DEG: f32 = 9.0; // half angular width of a ladder bar
const HORIZON_HALF_AZ_DEG: f32 = 20.0; // wider half-width for the horizon bar
const LADDER_GAP: f32 = 26.0; // center split gap (logical px)
const LADDER_TICK: f32 = 8.0; // tick length toward the horizon (logical px)
const LADDER_LABEL_SCALE: f32 = 1.0; // glyph scale for the degree labels
const LADDER_LABEL_OFF: f32 = 6.0; // label gap outside the bar end (logical px)
const WATERLINE_HALF: f32 = 12.0; // half width of the waterline W (logical px)
const WATERLINE_DROP: f32 = 4.0; // depth of the W's vees (logical px)

/// Project a world-space unit direction to a device-pixel screen point through the
/// camera basis + vertical FOV, matching the scene's perspective projection.
/// Returns null when the direction is at or behind the camera plane. Deliberately
/// a 3-dot basis projection, not the scene's 4x4 MVP: a direction needs no
/// clip-space depth, so do not "unify" this with `math.perspective`/`lookAt`.
fn project(dir: math.Vec3, att: Attitude, screen: ui.Screen) ?[2]f32 {
    const depth = math.dot(dir, att.cam_front);
    if (depth <= 1e-4) return null;
    // Vertical focal length in pixels; equal on both axes (square pixels), so the
    // one value drives x and y and ties the symbology scale to the rendered FOV.
    const f_px = (screen.h * 0.5) / @tan(att.fov_y_rad * 0.5);
    const sx = screen.w * 0.5 + f_px * math.dot(dir, att.cam_right) / depth;
    const sy = screen.h * 0.5 - f_px * math.dot(dir, att.cam_up) / depth; // +up -> -y
    return .{ sx, sy };
}

/// World direction at elevation `elev` (rad) above horizontal, in the vertical
/// plane through `fwd_h` (the aircraft's horizontal forward). world_up = +Y.
fn elevDir(fwd_h: math.Vec3, elev: f32) math.Vec3 {
    const ce = @cos(elev);
    return .{ fwd_h[0] * ce, @sin(elev), fwd_h[2] * ce };
}

/// Rotate a vector about world-up (+Y) by `a` radians (sweeps bar endpoints in
/// azimuth around the bar center).
fn rotY(v: math.Vec3, a: f32) math.Vec3 {
    const ca = @cos(a);
    const sa = @sin(a);
    return .{ v[0] * ca + v[2] * sa, v[1], -v[0] * sa + v[2] * ca };
}

/// Unit screen vector toward decreasing elevation (toward the horizon), for tick
/// orientation under any bank. Taken from the projected elevation gradient so it
/// rolls with the view. Falls back to screen-down if a sample is culled.
fn ladderDown(fwd_h: math.Vec3, att: Attitude, screen: ui.Screen) [2]f32 {
    const hi = project(elevDir(fwd_h, 0.05), att, screen);
    const lo = project(elevDir(fwd_h, -0.05), att, screen);
    if (hi == null or lo == null) return .{ 0, 1 };
    const dx = lo.?[0] - hi.?[0];
    const dy = lo.?[1] - hi.?[1];
    const len = @sqrt(dx * dx + dy * dy);
    if (len < 1e-4) return .{ 0, 1 };
    return .{ dx / len, dy / len };
}

/// One pitch bar at elevation `elev`: a split horizontal segment (center gap),
/// with inner ticks toward the horizon and upright degree labels (non-horizon).
fn drawBar(dl: *ui.DrawList, fwd_h: math.Vec3, elev: f32, half_az: f32, att: Attitude, screen: ui.Screen, thick: f32, down: [2]f32, is_horizon: bool) void {
    const c = elevDir(fwd_h, elev);
    // TODO: free-look only. If exactly one endpoint is behind the camera the whole
    // bar drops instead of clipping at the near plane, so the wide horizon bar can
    // pop out when the (free-cam) view diverges >~70deg from the aircraft nose.
    // Never triggers in cockpit/chase (camera ~ aircraft); fix with near-plane
    // segment clipping alongside the deferred free-look work.
    const pl = project(rotY(c, half_az), att, screen) orelse return;
    const pr = project(rotY(c, -half_az), att, screen) orelse return;

    var dx = pr[0] - pl[0];
    var dy = pr[1] - pl[1];
    const len = @sqrt(dx * dx + dy * dy);
    if (len < 1e-3) return;
    dx /= len;
    dy /= len;
    const mx = (pl[0] + pr[0]) * 0.5;
    const my = (pl[1] + pr[1]) * 0.5;
    const hg = screen.px(LADDER_GAP) * 0.5;
    const glx = mx - dx * hg;
    const gly = my - dy * hg;
    const grx = mx + dx * hg;
    const gry = my + dy * hg;
    if (is_horizon) {
        // No ticks: two disjoint segments across the center gap, so no joints.
        dl.line(pl[0], pl[1], glx, gly, thick, ATT_COLOR);
        dl.line(grx, gry, pr[0], pr[1], thick, ATT_COLOR);
        return;
    }

    // Each half-bar + its inner tick is one L-polyline, so the tick/bar joint has
    // single coverage (no double-blend) and a filled corner. Ticks point toward the
    // horizon: down for bars above it, up for bars below.
    const tick = screen.px(LADDER_TICK);
    const sgn: f32 = if (elev >= 0) 1 else -1;
    const tx = down[0] * tick * sgn;
    const ty = down[1] * tick * sgn;
    const left = [_][2]f32{ .{ pl[0], pl[1] }, .{ glx, gly }, .{ glx + tx, gly + ty } };
    const right = [_][2]f32{ .{ pr[0], pr[1] }, .{ grx, gry }, .{ grx + tx, gry + ty } };
    dl.polyline(&left, thick, ATT_COLOR);
    dl.polyline(&right, thick, ATT_COLOR);

    // Upright degree labels just outside each end. Glyphs can't rotate, so they
    // read upright even when the bar is banked (standard HUD behavior).
    const label_scale = screen.textScale(LADDER_LABEL_SCALE);
    const off = screen.px(LADDER_LABEL_OFF);
    const half_cell = screen.px(ui.GLYPH_PX * LADDER_LABEL_SCALE) * 0.5;
    var buf: [4]u8 = undefined;
    const deg_i: i32 = @intFromFloat(@round(std.math.radiansToDegrees(elev)));
    const txt = fmt(&buf, "{d}", .{deg_i});
    _ = dl.textRight(pl[0] - off, pl[1] - half_cell, label_scale, ATT_COLOR, txt);
    _ = dl.text(pr[0] + off, pr[1] - half_cell, label_scale, ATT_COLOR, txt);
}

/// Waterline / boresight marker: a `-\/\/-` W, drawn upright (body-fixed) at the
/// projected nose direction. Screen center in cockpit; slides under free-look.
fn drawWaterline(dl: *ui.DrawList, cx: f32, cy: f32, screen: ui.Screen, thick: f32) void {
    const half = screen.px(WATERLINE_HALF);
    const drop = screen.px(WATERLINE_DROP);
    const lo = cy + drop;
    // -\/\/- as one connected stroke (dash, two vees, dash): single coverage at the
    // joints (no double-blend) and filled corners.
    const pts = [_][2]f32{
        .{ cx - half, cy },
        .{ cx - half * 0.6, cy },
        .{ cx - half * 0.3, lo },
        .{ cx, cy },
        .{ cx + half * 0.3, lo },
        .{ cx + half * 0.6, cy },
        .{ cx + half, cy },
    };
    dl.polyline(&pts, thick, ATT_COLOR);
}

/// Conformal attitude core: horizon + pitch ladder (clipped to a center box,
/// dimmed) + waterline. See the section header.
fn drawAttitude(dl: *ui.DrawList, screen: ui.Screen, att: Attitude) void {
    const box_w = screen.w * ATT_BOX_W_FRAC;
    const box_h = screen.h * ATT_BOX_H_FRAC;
    dl.pushClip(.{
        .x = @intFromFloat((screen.w - box_w) * 0.5),
        .y = @intFromFloat((screen.h - box_h) * 0.5),
        .w = @intFromFloat(box_w),
        .h = @intFromFloat(box_h),
    });

    const thick = screen.px(ATT_THICK);

    // Aircraft horizontal forward = the ladder's azimuth reference. Degenerates
    // when the nose points near-straight up/down; skip the ladder/horizon then.
    // TODO: a vertical-flight ladder mode (or zenith/nadir cue) for that case.
    const f = att.ac_front;
    const horiz = @sqrt(f[0] * f[0] + f[2] * f[2]);
    if (horiz > 1e-3) {
        const fwd_h = math.Vec3{ f[0] / horiz, 0, f[2] / horiz };
        const down = ladderDown(fwd_h, att, screen);

        drawBar(dl, fwd_h, 0, std.math.degreesToRadians(HORIZON_HALF_AZ_DEG), att, screen, thick, down, true);

        const half_az = std.math.degreesToRadians(LADDER_HALF_AZ_DEG);
        var deg: f32 = LADDER_STEP_DEG;
        while (deg <= LADDER_MAX_DEG + 0.5) : (deg += LADDER_STEP_DEG) {
            drawBar(dl, fwd_h, std.math.degreesToRadians(deg), half_az, att, screen, thick, down, false);
            drawBar(dl, fwd_h, std.math.degreesToRadians(-deg), half_az, att, screen, thick, down, false);
        }
    }

    if (project(att.ac_front, att, screen)) |c| {
        drawWaterline(dl, c[0], c[1], screen, thick);
    }

    dl.popClip();
}

// =============================================================================
// Dev / debug text block (toggled via F1, default off)
// =============================================================================

// Logical pixels (multiplied by Screen.scale at draw time). Reference height
// 720 -> scale 1.0 at 720p. REF_H / USER_PREF come from the ui module so the HUD
// and the interactive menus share one logical-px scale and can't desync.
const REF_H = ui.REF_H;
const USER_PREF = ui.DEFAULT_USER_SCALE; // TODO: wire to a settings ui_scale knob
const HUD_SCALE: f32 = 1.0; // debug glyph scale at the reference height (8 px cell at 720p)
const HUD_CHAR: f32 = ui.GLYPH_PX * HUD_SCALE; // 8 px cell (logical)
const HUD_LINE_H: f32 = HUD_CHAR + 4.0; // 12 px line pitch (logical)
const HUD_PAD: f32 = 12.0; // text origin from the top-left (logical)
const HUD_INSET: f32 = 6.0; // panel padding around the text block (logical)
const HUD_TEXT: ui.Color = ui.Color.white;

/// Top-left dev overlay: FPS / frame times, camera-viewpoint position + speed,
/// VRAM / tile residency, sim state, and active debug toggles. Reports the
/// camera pose on purpose (it answers "where is my viewpoint"), unlike the
/// aircraft-sourced SVS readouts. Toggled via F1.
fn drawDebugBlock(dl: *ui.DrawList, screen: ui.Screen, dev: Dev) void {
    var line: u32 = 0;
    var max_w: f32 = 0;

    // Position/altitude/speed live in the SVS readouts now; the debug block keeps
    // only dev diagnostics, kept narrow so it clears the heading ribbon.
    hudLine(dl, screen, &line, &max_w, "FPS: {d:.0}", .{dev.fps});
    hudLine(dl, screen, &line, &max_w, "VRAM: {d:.1} MB  Tiles: {d}", .{ dev.vram_mb, dev.tiles_resident });
    if (dev.show_sim) {
        const level_state: []const u8 = if (dev.auto_level) "ON" else "OFF";
        hudLine(dl, screen, &line, &max_w, "Throttle: {d:.0}%  Level: {s} (L)", .{ dev.throttle * 100, level_state });
    }

    // Surface only non-default debug-toggle state. With everything off the HUD
    // looks identical to before, which keeps the overlay quiet during normal use.
    if (dev.render_label) |lbl| {
        hudLine(dl, screen, &line, &max_w, "F2 render: {s}", .{lbl});
    }
    if (dev.overlay_label) |lbl| {
        hudLine(dl, screen, &line, &max_w, "F3 overlay: {s}", .{lbl});
    }
    if (dev.freeze_stream) |lbl| {
        hudLine(dl, screen, &line, &max_w, "F4 freeze: ON  streaming: {s}", .{lbl});
    }

    // Translucent gradient panel behind the readouts. Appended last but drawn
    // first (shapes render behind glyphs), so it backs the text either way.
    if (line > 0) {
        const h = screen.px(@as(f32, @floatFromInt(line - 1)) * HUD_LINE_H + HUD_CHAR + 2 * HUD_INSET);
        dl.rectGradient(
            screen.px(HUD_PAD - HUD_INSET),
            screen.px(HUD_PAD - HUD_INSET),
            max_w + screen.px(2 * HUD_INSET),
            h,
            ui.Color.rgba(0.05, 0.06, 0.09, 0.62),
            ui.Color.rgba(0.01, 0.02, 0.03, 0.40),
        );
    }
}

/// Format one debug line into the draw list at the next row, scaled by `screen`,
/// tracking the widest line (device px) so the backing panel can be sized to fit.
fn hudLine(dl: *ui.DrawList, screen: ui.Screen, line: *u32, max_w: *f32, comptime f: []const u8, args: anytype) void {
    var buf: [128]u8 = undefined;
    const txt = fmt(&buf, f, args);
    const scale = screen.textScale(HUD_SCALE);
    const y = screen.px(HUD_PAD + @as(f32, @floatFromInt(line.*)) * HUD_LINE_H);
    const w = dl.text(screen.px(HUD_PAD), y, scale, HUD_TEXT, txt);
    if (w > max_w.*) max_w.* = w;
    line.* += 1;
}

// =============================================================================
// Heading ribbon (top-center scrolling tape)
// =============================================================================

// Logical pixels (multiplied by Screen.scale at draw time). Tunables: tweak
// during the visual fly-through.
const RIBBON_W_FRAC: f32 = 0.4; // ribbon width as a fraction of screen width...
const RIBBON_MAX_W: f32 = 440.0; // ...clamped to this logical width on wide aspects
const RIBBON_H: f32 = 28.0; // ribbon height
const RIBBON_TOP: f32 = 12.0; // gap from the top screen edge
const RIBBON_FOV_DEG: f32 = 60.0; // degrees of heading visible across the ribbon
const RIBBON_TICK_MINOR: f32 = 5.0; // minor (5 deg) tick length, from the top edge down
const RIBBON_TICK_MAJOR: f32 = 9.0; // major (30 deg) tick length
const RIBBON_LABEL_Y: f32 = 11.0; // label baseline offset from the ribbon top
const RIBBON_LABEL_SCALE: f32 = 1.25; // glyph scale multiplier (times Screen.scale)

/// Draw the top-center scrolling heading ribbon. `heading` is the aircraft
/// compass heading in radians (0 = N, 90 = E, 180 = S, 270 = W). Ticks sit at
/// fixed compass headings; the tape scrolls past a fixed center lubber.
fn drawHeadingRibbon(dl: *ui.DrawList, screen: ui.Screen, heading: f32) void {
    const ribbon_w = @min(screen.w * RIBBON_W_FRAC, screen.px(RIBBON_MAX_W));
    const ribbon_h = screen.px(RIBBON_H);
    const center = screen.anchor(.top_center, 0, RIBBON_TOP);
    const cx = center[0];
    const top = center[1];
    const left = cx - ribbon_w * 0.5;
    const bottom = top + ribbon_h;

    // Scrolling ticks + labels are clipped to the ribbon window so edge content
    // crops instead of spilling. Floor the top-left / ceil the size so the window
    // never under-covers a tick (the backend clamps to the framebuffer).
    dl.pushClip(.{
        .x = @intFromFloat(@floor(left)),
        .y = @intFromFloat(@floor(top)),
        .w = @intFromFloat(@ceil(ribbon_w)),
        .h = @intFromFloat(@ceil(ribbon_h)),
    });

    const px_per_deg = ribbon_w / RIBBON_FOV_DEG;
    const half = RIBBON_FOV_DEG * 0.5 + 5.0; // a touch past the edge so edge ticks clip cleanly
    const thick_minor = @max(1.0, screen.px(1.0));
    const thick_major = @max(1.0, screen.px(1.5));
    const minor_y1 = top + screen.px(RIBBON_TICK_MINOR);
    const major_y1 = top + screen.px(RIBBON_TICK_MAJOR);
    const label_y = top + screen.px(RIBBON_LABEL_Y);
    const label_scale = screen.textScale(RIBBON_LABEL_SCALE);
    const heading_deg = std.math.radiansToDegrees(math.wrapAngle(heading));

    // Iterate fixed compass headings on a 5-deg grid across the visible window.
    // Stepping absolute headings (not raw 0..360) keeps the 0/360 seam correct.
    var t: f32 = @ceil((heading_deg - half) / 5.0) * 5.0;
    const hi = heading_deg + half;
    while (t <= hi) : (t += 5.0) {
        const x = ribbonTickX(heading, std.math.degreesToRadians(t), cx, px_per_deg);
        const hdg_norm = @mod(t, 360.0); // [0, 360)
        const hdg: u32 = @intFromFloat(@round(hdg_norm));
        if (@mod(hdg, 30) == 0) {
            dl.line(x, top, x, major_y1, thick_major, RIBBON_COLOR);
            var buf: [2]u8 = undefined;
            const label = headingLabel(hdg % 360, &buf);
            const lw = ui.DrawList.textWidth(label.len, label_scale);
            _ = dl.text(x - lw * 0.5, label_y, label_scale, RIBBON_COLOR, label);
        } else {
            dl.line(x, top, x, minor_y1, thick_minor, RIBBON_COLOR);
        }
    }
    dl.popClip();

    // Fixed center lubber: a small triangle at the ribbon's bottom edge pointing
    // up. Unclipped (not part of the scrolling content) and positioned clear of
    // the labels, which render on top of it (glyphs always draw over shapes).
    const tw = screen.px(5.0);
    const th = screen.px(6.0);
    dl.tri(cx - tw, bottom, cx + tw, bottom, cx, bottom - th, RIBBON_COLOR);
}

/// Device-x of a heading tick on the ribbon. `heading` (the ribbon's center) and
/// `tick` are compass radians; the signed angular delta (via `wrapAngle`, so the
/// 0/360 seam is handled) maps to pixels at `px_per_deg`. Pure: unit-tested
/// without Vulkan.
fn ribbonTickX(heading: f32, tick: f32, center_x: f32, px_per_deg: f32) f32 {
    const delta_deg = std.math.radiansToDegrees(math.wrapAngle(tick - heading));
    return center_x + delta_deg * px_per_deg;
}

/// Compass label for a normalized heading: cardinal letters at N/E/S/W, else the
/// tens-of-degrees (30 -> "3", 120 -> "12"), the standard directional-gyro form.
fn headingLabel(hdg: u32, buf: *[2]u8) []const u8 {
    return switch (hdg) {
        0 => "N",
        90 => "E",
        180 => "S",
        270 => "W",
        else => std.fmt.bufPrint(buf, "{d}", .{hdg / 10}) catch "?",
    };
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

test "formatVs: signed m/s, one decimal within +/-10, integer beyond, no neg-zero" {
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("+3.2 m/s", formatVs(&buf, 3.2, false));
    try testing.expectEqualStrings("-8.0 m/s", formatVs(&buf, -8.0, false));
    try testing.expectEqualStrings("+12 m/s", formatVs(&buf, 12.4, false)); // |vs| >= 10 -> integer
    try testing.expectEqualStrings("-25 m/s", formatVs(&buf, -25.0, false));
    try testing.expectEqualStrings("+0.0 m/s", formatVs(&buf, 0.0, false));
    // Negative zero must not leak through as "+-0", and a sub-decimal descent
    // that rounds to zero shows "+0.0", not "-0.0".
    try testing.expectEqualStrings("+0.0 m/s", formatVs(&buf, -0.0, false));
    try testing.expectEqualStrings("+0.0 m/s", formatVs(&buf, -0.04, false));
    try testing.expectEqualStrings("-0.3 m/s", formatVs(&buf, -0.3, false));
}

test "formatVs: imperial is ft/min, integer, signed, no neg-zero" {
    var buf: [16]u8 = undefined;
    try testing.expectEqualStrings("+591 ft/min", formatVs(&buf, 3.0, true)); // 3 m/s * 196.85
    try testing.expectEqualStrings("-1969 ft/min", formatVs(&buf, -10.0, true));
    try testing.expectEqualStrings("+0 ft/min", formatVs(&buf, -0.0, true));
}

test "ribbonTickX: heading-aligned tick sits at the center" {
    try testing.expectApproxEqAbs(@as(f32, 500), ribbonTickX(1.0, 1.0, 500, 4), 1e-3);
}

test "ribbonTickX: +10 degrees is 10*px_per_deg right of center" {
    const x = ribbonTickX(0.0, std.math.degreesToRadians(10.0), 500, 4);
    try testing.expectApproxEqAbs(@as(f32, 540), x, 1e-3);
}

test "ribbonTickX: 350deg at heading 5deg wraps left, not far right" {
    // delta = wrap(350 - 5) = -15 deg -> 15*4 = 60 px left of center.
    const x = ribbonTickX(std.math.degreesToRadians(5.0), std.math.degreesToRadians(350.0), 500, 4);
    try testing.expectApproxEqAbs(@as(f32, 440), x, 1e-3);
}

test "headingLabel: cardinals and tens-of-degrees" {
    var buf: [2]u8 = undefined;
    try testing.expectEqualStrings("N", headingLabel(0, &buf));
    try testing.expectEqualStrings("E", headingLabel(90, &buf));
    try testing.expectEqualStrings("3", headingLabel(30, &buf));
    try testing.expectEqualStrings("12", headingLabel(120, &buf));
    try testing.expectEqualStrings("33", headingLabel(330, &buf));
}

// Camera + aircraft both looking straight down -Z (identity / cockpit boresight).
fn attForward() Attitude {
    return .{
        .cam_right = .{ 1, 0, 0 },
        .cam_up = .{ 0, 1, 0 },
        .cam_front = .{ 0, 0, -1 },
        .fov_y_rad = std.math.degreesToRadians(@as(f32, 60)),
        .ac_front = .{ 0, 0, -1 },
    };
}

test "project: boresight maps to screen center" {
    const att = attForward();
    const screen = ui.Screen.fromExtent(1920, 1080, REF_H, 1.0);
    const p = project(att.cam_front, att, screen).?;
    try testing.expectApproxEqAbs(@as(f32, 960), p[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 540), p[1], 1e-3);
}

test "project: rightward lands right of center, upward above center" {
    const att = attForward();
    const screen = ui.Screen.fromExtent(1920, 1080, REF_H, 1.0);
    const a = std.math.degreesToRadians(@as(f32, 10));
    const right = project(math.Vec3{ @sin(a), 0, -@cos(a) }, att, screen).?;
    try testing.expect(right[0] > 960);
    try testing.expectApproxEqAbs(@as(f32, 540), right[1], 1e-3);
    const up = project(math.Vec3{ 0, @sin(a), -@cos(a) }, att, screen).?;
    try testing.expect(up[1] < 540); // +y is screen-down
    try testing.expectApproxEqAbs(@as(f32, 960), up[0], 1e-3);
}

test "project: a direction behind the camera is culled" {
    const att = attForward();
    const screen = ui.Screen.fromExtent(1920, 1080, REF_H, 1.0);
    try testing.expect(project(math.Vec3{ 0, 0, 1 }, att, screen) == null);
}

test "elevDir: 0 elevation is horizontal forward, 90 is straight up" {
    const fwd_h = math.Vec3{ 0, 0, -1 };
    const flat = elevDir(fwd_h, 0);
    try testing.expectApproxEqAbs(@as(f32, 0), flat[1], 1e-6);
    const up = elevDir(fwd_h, std.math.degreesToRadians(@as(f32, 90)));
    try testing.expectApproxEqAbs(@as(f32, 1), up[1], 1e-6);
}

test "drawAttitude: emits strokes inside a pushed center-box clip group" {
    var dl: ui.DrawList = .{};
    const att = attForward();
    const screen = ui.Screen.fromExtent(1920, 1080, REF_H, 1.0);
    drawAttitude(&dl, screen, att);
    // Horizon + ladder bars + waterline all land on the SDF stream.
    try testing.expect(dl.sdf_count > 0);
    // A center-box clip was pushed and popped (1 initial + push + pop).
    try testing.expectEqual(@as(u32, 3), dl.clip_count);
}
