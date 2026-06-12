//! Interactive widgets for the immediate-mode UI. Pure: each widget reserves a
//! layout row, runs the shared `Ui.behavior` activation state machine (so pointer
//! and focus-nav activate symmetrically), mutates the caller's bound value, and
//! emits geometry into the `Ui`'s `DrawList`. Call as `ui.button(&ctx, @src(), ...)`.
//!
//! Widget chrome uses the SDF stream (`roundedRect`/`circle`) and text uses the
//! glyph stream; both render above the panel's shape-stream background regardless
//! of append order (see `context.endPanel`). Keep widget fills off the shape
//! stream so that layering holds.

const std = @import("std");
const draw = @import("ui.zig");
const ctx = @import("context.zig");
const Ui = ctx.Ui;
const Rect = ctx.Rect;
const SrcLoc = ctx.SrcLoc;
const Color = draw.Color;

/// Full-width button. Returns true on the frame it activates (click released
/// inside, or Enter/Space while focused).
pub fn button(ui: *Ui, src: SrcLoc, label: []const u8) bool {
    if (!ui.panel_open) return false;
    const wid = ui.id(src, 0);
    const row = ui.rowRect();
    const w = beginWidget(ui, wid, row);
    const dis = w.dis;
    const resp = w.resp;

    fillRow(ui, row, if (dis) ui.theme.widget_disabled else stateColor(ui, wid, resp));
    focusRing(ui, wid, row);
    textCentered(ui, row, label, textColor(ui));
    return resp.activated;
}

/// Non-interactive left-aligned text row. Not a focus stop; for static labels,
/// prompts, and readouts inside a panel. Dims inside a `beginDisabled` scope.
pub fn labelRow(ui: *Ui, text: []const u8) void {
    if (!ui.panel_open) return;
    const s = ui.screen;
    const row = ui.rowRect();
    const scale = s.textScale(ui.theme.font_scale);
    const ty = row.y + (row.h - glyphCell(ui)) * 0.5;
    _ = ui.dl.text(row.x + s.px(ui.theme.label_pad), ty, scale, textColor(ui), text);
}

/// Horizontal segmented control / tab bar: ONE focus stop showing all `labels`
/// across a single row with `active.*` highlighted. While focused, Left/Right
/// cycle the active index (wrapping); a click selects the cell under the pointer.
/// Because it is one focus stop, Up/Down focus-nav treats the whole bar as a
/// single step, so it sits cleanly above a vertical content list (Down drops into
/// content, Left/Right switches category). Returns true the frame `active.*`
/// changes. Reused for category tabs and small enum settings.
pub fn tabBar(ui: *Ui, src: SrcLoc, active: *usize, labels: []const []const u8) bool {
    if (!ui.panel_open or labels.len == 0) return false;
    if (active.* >= labels.len) active.* = labels.len - 1; // guard a caller-supplied index
    const wid = ui.id(src, 0);
    const row = ui.rowRect();
    const w = beginWidget(ui, wid, row);
    const dis = w.dis;
    const resp = w.resp;

    const before = active.*;
    if (!dis) {
        // Pointer: a click (release inside the bar) selects the cell under the cursor.
        // Gate on `ptr_released` so a focus+activate-key press (which also sets
        // `resp.activated`, with the mouse resting anywhere) cannot hijack the
        // selection to whatever cell the cursor happens to sit over. Enter on the bar
        // is a no-op; Left/Right is the keyboard interaction.
        if (resp.activated and ui.ptr_released) {
            for (0..labels.len) |i| {
                if (cellRect(ui, row, i, labels.len).contains(ui.input.pointer_x, ui.input.pointer_y)) {
                    active.* = i;
                    break;
                }
            }
        }
        // Keyboard / d-pad: Left/Right cycle while the bar holds focus.
        if (ui.isFocused(wid)) {
            if (ui.input.nav_right) active.* = (active.* + 1) % labels.len;
            if (ui.input.nav_left) active.* = (active.* + labels.len - 1) % labels.len;
        }
    }

    for (0..labels.len) |i| {
        const cell = cellRect(ui, row, i, labels.len);
        const is_active = (i == active.*);
        const fill = if (dis) ui.theme.widget_disabled else if (is_active) ui.theme.widget_active else ui.theme.widget_bg;
        const tcol = if (dis) ui.theme.text_disabled else if (is_active) ui.theme.title else ui.theme.text;
        fillRow(ui, cell, fill);
        textCentered(ui, cell, labels[i], tcol);
    }
    focusRing(ui, wid, row);
    return active.* != before;
}

/// The i-th of `n` equal-width cells spanning `row`, separated by a small gap.
fn cellRect(ui: *Ui, row: Rect, i: usize, n: usize) Rect {
    const gap = ui.screen.px(4);
    const fi: f32 = @floatFromInt(i);
    const fn_: f32 = @floatFromInt(n);
    const cell_w = (row.w - gap * (fn_ - 1)) / fn_;
    return .{ .x = row.x + (cell_w + gap) * fi, .y = row.y, .w = cell_w, .h = row.h };
}

/// Boolean toggle (pill + knob) bound to `value`. The whole row is clickable.
/// Returns true on the frame it flips.
pub fn toggle(ui: *Ui, src: SrcLoc, label: []const u8, value: *bool) bool {
    if (!ui.panel_open) return false;
    const wid = ui.id(src, 0);
    const row = ui.rowRect();
    const w = beginWidget(ui, wid, row);
    const dis = w.dis;
    const resp = w.resp;
    if (resp.activated) value.* = !value.*;

    textLeft(ui, row, label, textColor(ui));
    focusRing(ui, wid, row);

    // Compact ~2:1 stadium pill on the right edge, vertically centered. Sized for a
    // 2-state switch rather than the slider's control column, so the row carries no
    // dead space. Right edge insets by `label_pad` to line up with other row values.
    const s = ui.screen;
    const pill_h = row.h * 0.5;
    const pill_w = pill_h * 2;
    const c = Rect{
        .x = row.x + row.w - pill_w - s.px(ui.theme.label_pad),
        .y = row.y + (row.h - pill_h) * 0.5,
        .w = pill_w,
        .h = pill_h,
    };
    // The knob still reflects `value.*` when disabled (state stays readable), just dimmed.
    const track = if (dis) ui.theme.widget_disabled else if (value.*) ui.theme.accent else stateColor(ui, wid, resp);
    ui.dl.roundedRect(c.x, c.y, c.w, c.h, c.h * 0.5, track); // stadium pill
    const knob_r = c.h * 0.5 - s.px(2);
    const cx = if (value.*) c.x + c.w - c.h * 0.5 else c.x + c.h * 0.5;
    ui.dl.circle(cx, c.y + c.h * 0.5, knob_r, if (dis) ui.theme.text_disabled else ui.theme.knob);
    return resp.activated;
}

/// Float slider bound to `value`, clamped to [min, max]. Pointer drag maps the
/// pointer's x absolutely onto the track (click-to-position); when focused,
/// left/right nudge by a step. Returns true on the frame the value changes.
pub fn sliderF32(ui: *Ui, src: SrcLoc, label: []const u8, value: *f32, min: f32, max: f32) bool {
    if (!ui.panel_open) return false;
    const wid = ui.id(src, 0);
    const s = ui.screen;
    // Two lines: label + value on top, a full-width track below. The wide track is
    // far easier to drag than the old 64px control-column slot, and the value sits
    // in its own right-aligned slot instead of overlapping the moving knob.
    const row = ui.rowRectH(draw.GLYPH_PX * ui.theme.font_scale + 20);

    const scale = s.textScale(ui.theme.font_scale);
    const cell = glyphCell(ui);
    const text_y = row.y + s.px(4);
    // Track occupies the lower band of the row, inset by `label_pad` on each side so
    // its ends line up with the label (left) and value (right) above it.
    const pad = s.px(ui.theme.label_pad);
    const track_top = text_y + cell + s.px(4);
    const track = Rect{ .x = row.x + pad, .y = track_top, .w = row.w - 2 * pad, .h = row.y + row.h - track_top };
    const w = beginWidget(ui, wid, track);
    const dis = w.dis;
    const resp = w.resp;

    var changed = false;
    if (!dis) {
        if (resp.held and ui.active_via_pointer) {
            // Pointer drag maps the cursor x absolutely onto the track. Takes
            // precedence over the focus nudge so a stray arrow during a drag can't
            // add a step on top of the pointer-set value (press also set focus here).
            const nv = trackXToValue(ui.input.pointer_x, track, min, max);
            if (nv != value.*) {
                value.* = nv;
                changed = true;
            }
        } else if (ui.isFocused(wid)) {
            const step = (max - min) / 20.0;
            if (ui.input.nav_right) {
                value.* = std.math.clamp(value.* + step, min, max);
                changed = true;
            }
            if (ui.input.nav_left) {
                value.* = std.math.clamp(value.* - step, min, max);
                changed = true;
            }
        }
    }

    // Top line: label left, value right-aligned and brighter so it reads at a glance.
    _ = ui.dl.text(row.x + s.px(ui.theme.label_pad), text_y, scale, textColor(ui), label);
    // TODO(text-input): make the value a click-to-edit field for direct float entry
    // once the text_input widget lands (Phase 9); this right-aligned slot is its home.
    var buf: [24]u8 = undefined;
    const txt = std.fmt.bufPrint(&buf, "{d:.2}", .{value.*}) catch "";
    _ = ui.dl.textRight(row.x + row.w - s.px(ui.theme.label_pad), text_y, scale, if (dis) ui.theme.text_disabled else ui.theme.title, txt);

    // Track: thin bar with an accent fill up to the knob.
    const bar_h = s.px(4);
    const by = track.y + (track.h - bar_h) * 0.5;
    ui.dl.roundedRect(track.x, by, track.w, bar_h, bar_h * 0.5, ui.theme.widget_bg);
    const kx = valueToTrackX(value.*, track, min, max);
    if (kx > track.x) ui.dl.roundedRect(track.x, by, kx - track.x, bar_h, bar_h * 0.5, if (dis) ui.theme.widget_disabled else ui.theme.accent);
    ui.dl.circle(kx, track.y + track.h * 0.5, track.h * 0.4, if (dis) ui.theme.text_disabled else ui.theme.knob);

    focusRing(ui, wid, row);
    return changed;
}

/// Integer slider bound to `value.*`, snapping to `step`-multiples offset from
/// `min` (so e.g. odd-only ring sizes or unit level counts land on legal stops).
/// Same two-line layout + drag/nav behavior as `sliderF32`; the value is shown as
/// a plain integer. Returns true the frame `value.*` changes. `step` must be >= 1.
/// (Shares the row layout shape with sliderF32; kept separate rather than
/// generalized over a numeric type to keep each widget's value mapping legible.)
pub fn sliderInt(ui: *Ui, src: SrcLoc, label: []const u8, value: *i32, min: i32, max: i32, step: i32) bool {
    if (!ui.panel_open) return false;
    const wid = ui.id(src, 0);
    const s = ui.screen;
    const row = ui.rowRectH(draw.GLYPH_PX * ui.theme.font_scale + 20);

    const scale = s.textScale(ui.theme.font_scale);
    const cell = glyphCell(ui);
    const text_y = row.y + s.px(4);
    const pad = s.px(ui.theme.label_pad);
    const track_top = text_y + cell + s.px(4);
    const track = Rect{ .x = row.x + pad, .y = track_top, .w = row.w - 2 * pad, .h = row.y + row.h - track_top };
    const w = beginWidget(ui, wid, track);
    const dis = w.dis;
    const resp = w.resp;

    const fmin: f32 = @floatFromInt(min);
    const fmax: f32 = @floatFromInt(max);

    var changed = false;
    if (!dis) {
        if (resp.held and ui.active_via_pointer) {
            const nv = snapToStep(trackXToValue(ui.input.pointer_x, track, fmin, fmax), min, max, step);
            if (nv != value.*) {
                value.* = nv;
                changed = true;
            }
        } else if (ui.isFocused(wid)) {
            const ns = navStep(min, max, step);
            if (ui.input.nav_right) {
                value.* = std.math.clamp(value.* + ns, min, max);
                changed = true;
            }
            if (ui.input.nav_left) {
                value.* = std.math.clamp(value.* - ns, min, max);
                changed = true;
            }
        }
    }

    _ = ui.dl.text(row.x + s.px(ui.theme.label_pad), text_y, scale, textColor(ui), label);
    var buf: [24]u8 = undefined;
    const txt = std.fmt.bufPrint(&buf, "{d}", .{value.*}) catch "";
    _ = ui.dl.textRight(row.x + row.w - s.px(ui.theme.label_pad), text_y, scale, if (dis) ui.theme.text_disabled else ui.theme.title, txt);

    const bar_h = s.px(4);
    const by = track.y + (track.h - bar_h) * 0.5;
    ui.dl.roundedRect(track.x, by, track.w, bar_h, bar_h * 0.5, ui.theme.widget_bg);
    const kx = valueToTrackX(@floatFromInt(value.*), track, fmin, fmax);
    if (kx > track.x) ui.dl.roundedRect(track.x, by, kx - track.x, bar_h, bar_h * 0.5, if (dis) ui.theme.widget_disabled else ui.theme.accent);
    ui.dl.circle(kx, track.y + track.h * 0.5, track.h * 0.4, if (dis) ui.theme.text_disabled else ui.theme.knob);

    focusRing(ui, wid, row);
    return changed;
}

/// Value selector bound to `active.*`, an index into `options`. Renders `< value >`
/// over a discrete position indicator (one mark per option, the active one
/// highlighted). One focus stop: Left/Right step (CLAMPED, no wrap, the ends are
/// visible in the indicator); the left half of the control decrements on click,
/// the right half increments. `options` is read fresh each call, so the set may
/// change between frames (e.g. GPU-dependent MSAA levels) and `active.*` is clamped
/// back into range; the caller owns any semantic remap when the set changes.
/// Returns true the frame the index changes. For row-style enum settings.
pub fn cycle(ui: *Ui, src: SrcLoc, label: []const u8, active: *usize, options: []const []const u8) bool {
    if (!ui.panel_open or options.len == 0) return false;
    if (active.* >= options.len) active.* = options.len - 1; // option set may have shrunk
    const wid = ui.id(src, 0);
    const s = ui.screen;
    const scale = s.textScale(ui.theme.font_scale);
    // Two lines (value + position indicator below) with comfortable vertical room.
    const row = ui.rowRectH(draw.GLYPH_PX * ui.theme.font_scale + 24);
    const w = beginWidget(ui, wid, row);
    const dis = w.dis;
    const resp = w.resp;

    // Control region on the right (the label sits left of it), sized to fit the widest
    // option plus the < > arrows so long values like "Borderless" don't overflow;
    // clamped to 3/4 of the row so the label keeps room.
    var widest_val: f32 = 0;
    for (options) |opt| widest_val = @max(widest_val, draw.DrawList.textWidth(opt.len, scale));
    const arrow_w = draw.DrawList.textWidth(1, scale);
    const ctrl_w = @min(@max(s.px(ui.theme.control_w + 16), widest_val + 4 * arrow_w + s.px(12)), row.w * 0.75);
    const ctrl = Rect{ .x = row.x + row.w - ctrl_w, .y = row.y, .w = ctrl_w, .h = row.h };

    const before = active.*;
    if (!dis) {
        // Pointer: left half decrements, right half increments (both clamped).
        if (resp.activated and ui.ptr_released and ctrl.contains(ui.input.pointer_x, ui.input.pointer_y)) {
            if (ui.input.pointer_x < ctrl.x + ctrl.w * 0.5) {
                if (active.* > 0) active.* -= 1;
            } else if (active.* + 1 < options.len) {
                active.* += 1;
            }
        }
        // Keyboard / d-pad: Left/Right step while focused (clamped).
        if (ui.isFocused(wid)) {
            if (ui.input.nav_left and active.* > 0) active.* -= 1;
            if (ui.input.nav_right and active.* + 1 < options.len) active.* += 1;
        }
    }

    const cell = glyphCell(ui);
    const val_y = ctrl.y + s.px(6);

    // Label + value share the top line; arrows dim at the ends (clamp cue), and the
    // whole control dims inside a disabled scope.
    const arrow_dim = if (dis) ui.theme.widget_disabled else ui.theme.widget_bg;
    const arrow_lit = if (dis) ui.theme.text_disabled else ui.theme.text;
    _ = ui.dl.text(row.x + s.px(ui.theme.label_pad), val_y, scale, textColor(ui), label);
    const at_min = active.* == 0;
    const at_max = active.* + 1 >= options.len;
    _ = ui.dl.text(ctrl.x + s.px(2), val_y, scale, if (at_min) arrow_dim else arrow_lit, "<");
    _ = ui.dl.textRight(ctrl.x + ctrl.w - s.px(2), val_y, scale, if (at_max) arrow_dim else arrow_lit, ">");
    const v = options[active.*];
    _ = ui.dl.text(ctrl.x + (ctrl.w - draw.DrawList.textWidth(v.len, scale)) * 0.5, val_y, scale, if (dis) ui.theme.text_disabled else ui.theme.title, v);

    // Position indicator: one mark per option below the value, active highlighted.
    const n: f32 = @floatFromInt(options.len);
    const mark_strip = ctrl.w - s.px(8);
    const step = mark_strip / n;
    const mark_w = @min(step * 0.5, s.px(10));
    const base_h = s.px(2.5);
    const mark_y = val_y + cell + s.px(6);
    const mark_on = if (dis) ui.theme.text_disabled else ui.theme.accent;
    const mark_off = if (dis) ui.theme.widget_disabled else ui.theme.widget_bg;
    for (0..options.len) |i| {
        const is_active = (i == active.*);
        const mh = if (is_active) base_h * 1.8 else base_h;
        const mx = ctrl.x + s.px(4) + step * (@as(f32, @floatFromInt(i)) + 0.5) - mark_w * 0.5;
        ui.dl.roundedRect(mx, mark_y + (base_h - mh) * 0.5, mark_w, mh, s.px(1), if (is_active) mark_on else mark_off);
    }

    focusRing(ui, wid, ctrl);
    return active.* != before;
}

// ---- Pure value <-> track mapping (unit-testable without a Ui) ---------------

/// Map a pointer x within `track` to a value in [min, max], clamped to the track.
pub fn trackXToValue(px: f32, track: Rect, min: f32, max: f32) f32 {
    if (track.w <= 0) return min;
    const t = std.math.clamp((px - track.x) / track.w, 0, 1);
    return min + t * (max - min);
}

/// Map a value to its knob x on `track`, clamped to the track extent.
pub fn valueToTrackX(value: f32, track: Rect, min: f32, max: f32) f32 {
    const t = if (max == min) 0 else std.math.clamp((value - min) / (max - min), 0, 1);
    return track.x + t * track.w;
}

/// Snap a continuous value to the nearest `step`-multiple offset from `min`,
/// clamped to [min, max]. `step` is floored to 1. Used by `sliderInt` so the
/// knob lands only on legal integer stops.
pub fn snapToStep(value: f32, min: i32, max: i32, step: i32) i32 {
    const stp: f32 = @floatFromInt(@max(step, 1));
    const lo: f32 = @floatFromInt(min);
    const snapped: i32 = @intFromFloat(@round(lo + @round((value - lo) / stp) * stp));
    return std.math.clamp(snapped, min, max);
}

/// Per-arrow-press step for an integer slider: ~1/20 of the range, rounded to a
/// `step` multiple, but at least one `step` (so small ranges still move).
fn navStep(min: i32, max: i32, step: i32) i32 {
    const stp = @max(step, 1);
    const coarse = @divTrunc(max - min, 20);
    return @max(stp, @divTrunc(coarse, stp) * stp);
}

// ---- Drawing helpers --------------------------------------------------------

fn stateColor(ui: *Ui, wid: u64, resp: ctx.Response) Color {
    if (resp.held) return ui.theme.widget_active;
    if (ui.isHot(wid)) return ui.theme.widget_hot;
    return ui.theme.widget_bg;
}

fn fillRow(ui: *Ui, rect: Rect, color: Color) void {
    ui.dl.roundedRect(rect.x, rect.y, rect.w, rect.h, ui.screen.px(ui.theme.corner), color);
}

/// Row label/value text color: dimmed inside a `beginDisabled` scope.
fn textColor(ui: *Ui) Color {
    return if (ui.isDisabled()) ui.theme.text_disabled else ui.theme.text;
}

/// Per-widget disabled-aware setup, used by every interactive widget. Live: registers
/// the focus stop and runs the activation state machine on `rect`. Disabled: skips
/// both and releases any focus/active this widget still owns (so a widget disabled
/// mid-interaction neither strands keyboard focus nor leaves a pointer-drag lock that
/// suppresses sibling hover), returning an inert Response. Routing every widget
/// through this keeps the disabled contract from being half-applied.
fn beginWidget(ui: *Ui, wid: u64, rect: Rect) struct { dis: bool, resp: ctx.Response } {
    if (ui.isDisabled()) {
        if (ui.focused_id == wid) ui.focused_id = 0;
        if (ui.active_id == wid) {
            ui.active_id = 0;
            ui.active_via_pointer = false;
        }
        return .{ .dis = true, .resp = .{} };
    }
    ui.registerFocusable(wid);
    return .{ .dis = false, .resp = ui.behavior(wid, rect) };
}

fn focusRing(ui: *Ui, wid: u64, rect: Rect) void {
    if (ui.isDisabled() or !ui.isFocused(wid)) return;
    ui.dl.roundedRectOutline(rect.x, rect.y, rect.w, rect.h, ui.screen.px(ui.theme.corner), ui.screen.px(ui.theme.border), ui.theme.focus_ring);
}

fn glyphCell(ui: *Ui) f32 {
    return ui.screen.px(draw.GLYPH_PX * ui.theme.font_scale);
}

fn textLeft(ui: *Ui, row: Rect, s: []const u8, color: Color) void {
    const y = row.y + (row.h - glyphCell(ui)) * 0.5;
    _ = ui.dl.text(row.x + ui.screen.px(ui.theme.label_pad), y, ui.screen.textScale(ui.theme.font_scale), color, s);
}

fn textCentered(ui: *Ui, rect: Rect, s: []const u8, color: Color) void {
    const w = draw.DrawList.textWidth(s.len, ui.screen.textScale(ui.theme.font_scale));
    const x = rect.x + (rect.w - w) * 0.5;
    const y = rect.y + (rect.h - glyphCell(ui)) * 0.5;
    _ = ui.dl.text(x, y, ui.screen.textScale(ui.theme.font_scale), color, s);
}

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;
const Screen = draw.Screen;
const DrawList = draw.DrawList;
const InputState = ctx.InputState;

const Harness = struct {
    ui: Ui,
    dl: *DrawList,

    fn init() !Harness {
        const dl = try testing.allocator.create(DrawList);
        dl.* = .{};
        return .{ .ui = Ui.init(testing.allocator), .dl = dl };
    }
    fn deinit(h: *Harness) void {
        h.ui.deinit();
        testing.allocator.destroy(h.dl);
    }
    fn frame(h: *Harness, in: InputState) void {
        h.ui.beginFrame(in, Screen.fromExtent(800, 600, 600, 1.0), h.dl);
    }
};

test "trackXToValue: endpoints, midpoint, clamp" {
    const t = Rect{ .x = 100, .y = 0, .w = 200, .h = 10 };
    try testing.expectEqual(@as(f32, 0), trackXToValue(100, t, 0, 1));
    try testing.expectEqual(@as(f32, 1), trackXToValue(300, t, 0, 1));
    try testing.expectEqual(@as(f32, 0.5), trackXToValue(200, t, 0, 1));
    try testing.expectEqual(@as(f32, 0), trackXToValue(0, t, 0, 1)); // left of track clamps
    try testing.expectEqual(@as(f32, 1), trackXToValue(9999, t, 0, 1)); // right clamps
}

test "valueToTrackX: round-trips with trackXToValue" {
    const t = Rect{ .x = 50, .y = 0, .w = 120, .h = 10 };
    const min: f32 = -3;
    const max: f32 = 7;
    for ([_]f32{ -3, -1, 0, 2.5, 7 }) |v| {
        const x = valueToTrackX(v, t, min, max);
        try testing.expectApproxEqAbs(v, trackXToValue(x, t, min, max), 1e-4);
    }
}

test "snapToStep: snaps to step offsets from min and clamps" {
    // Unit steps (num_levels-style 1..12).
    try testing.expectEqual(@as(i32, 6), snapToStep(6.4, 1, 12, 1));
    try testing.expectEqual(@as(i32, 7), snapToStep(6.6, 1, 12, 1));
    try testing.expectEqual(@as(i32, 12), snapToStep(99.0, 1, 12, 1)); // clamp high
    try testing.expectEqual(@as(i32, 1), snapToStep(-5.0, 1, 12, 1)); // clamp low
    // Step 2 from an odd min stays odd (ring_size-style 63..2047).
    try testing.expectEqual(@as(i32, 501), snapToStep(500.0, 63, 2047, 2));
    try testing.expectEqual(@as(i32, 2047), snapToStep(2047.0, 63, 2047, 2));
    try testing.expectEqual(@as(i32, 6), snapToStep(4.0, 1, 12, 5)); // stops at {1,6,11}; 4 -> 6
}

test "navStep: ~1/20 of range rounded to a step multiple, min one step" {
    try testing.expectEqual(@as(i32, 1), navStep(1, 12, 1)); // tiny range -> one step
    try testing.expectEqual(@as(i32, 98), navStep(63, 2047, 2)); // 1984/20=99 -> 98 (even multiple of 2)
    try testing.expectEqual(@as(i32, 5), navStep(0, 100, 5)); // 100/20=5
}

fn runSliderIntFrame(h: *Harness, v: *i32, min: i32, max: i32, step: i32, in: InputState) bool {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "S", test_panel);
    const changed = sliderInt(&h.ui, @src(), "x", v, min, max, step);
    h.ui.endPanel();
    h.ui.endFrame();
    return changed;
}

test "sliderInt: arrow nudge only when focused, snapped to step" {
    var h = try Harness.init();
    defer h.deinit();
    var v: i32 = 6;
    // Not focused: ignored.
    _ = runSliderIntFrame(&h, &v, 1, 12, 1, .{ .nav_right = true });
    try testing.expectEqual(@as(i32, 6), v);
    // Focus, then nudge by navStep (=1 for this range).
    _ = runSliderIntFrame(&h, &v, 1, 12, 1, .{ .tab = true });
    _ = runSliderIntFrame(&h, &v, 1, 12, 1, .{ .nav_right = true });
    try testing.expectEqual(@as(i32, 7), v);
    _ = runSliderIntFrame(&h, &v, 1, 12, 1, .{ .nav_left = true });
    try testing.expectEqual(@as(i32, 6), v);
}

test "toggle: pointer click flips the bound bool (and back)" {
    var h = try Harness.init();
    defer h.deinit();
    var on = false;

    // press inside, then release inside -> one activation
    _ = runToggleFrame(&h, &on, rowPointer(true));
    _ = runToggleFrame(&h, &on, rowPointer(false));
    try testing.expect(on);
    _ = runToggleFrame(&h, &on, rowPointer(true));
    _ = runToggleFrame(&h, &on, rowPointer(false));
    try testing.expect(!on);
}

test "toggle: key activate flips identically to a click" {
    var h = try Harness.init();
    defer h.deinit();
    var on = false;

    // Frame 1 registers the toggle and Tab focuses it (same call site across
    // frames via the helper, so the id is stable). Frame 2 activates by key.
    _ = runToggleFrame(&h, &on, .{ .tab = true });
    const changed = runToggleFrame(&h, &on, .{ .activate = true });
    try testing.expect(changed);
    try testing.expect(on);
}

test "toggle: press without release does not flip" {
    var h = try Harness.init();
    defer h.deinit();
    var on = false;
    _ = runToggleFrame(&h, &on, rowPointer(true)); // press down inside, never released
    try testing.expect(!on);
}

test "sliderF32: arrow nudge only when focused" {
    var h = try Harness.init();
    defer h.deinit();
    var v: f32 = 0.5;

    // Not focused: nav_right ignored.
    _ = runSliderFrame(&h, &v, .{ .nav_right = true });
    try testing.expectEqual(@as(f32, 0.5), v);

    // Focus via Tab, then nudge by a step (= 1/20 over [0,1]).
    _ = runSliderFrame(&h, &v, .{ .tab = true });
    _ = runSliderFrame(&h, &v, .{ .nav_right = true });
    try testing.expectApproxEqAbs(@as(f32, 0.55), v, 1e-4);
    _ = runSliderFrame(&h, &v, .{ .nav_left = true });
    try testing.expectApproxEqAbs(@as(f32, 0.5), v, 1e-4);
}

test "sliderF32: pointer drag suppresses a same-frame arrow nudge" {
    var h = try Harness.init();
    defer h.deinit();
    var v: f32 = 0.5;
    // First slider row (top_left panel, scale 1): track inset by label_pad to x[17,203]
    // in the lower band y[44,56]. Press at the track's left edge -> value maps to 0; a
    // nav_right in the same frame must NOT add a step on top (drag takes precedence
    // over the focus nudge).
    h.frame(.{ .pointer_x = 17, .pointer_y = 50, .pointer_down = true, .nav_right = true });
    _ = h.ui.beginPanel(@src(), "S", test_panel);
    _ = sliderF32(&h.ui, @src(), "x", &v, 0, 1);
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expectEqual(@as(f32, 0), v); // pointer-mapped 0, not 0 + step
}

// --- test helpers ---
// Panels anchor top_left at (0,0) so widget rows land at known device px. With
// scale 1 + the default theme: title row at y=10 (cell 10, gap 6), so the first
// widget row spans x[10,210], y[26,52]. (60,40) is comfortably inside it.

const test_panel = Ui.PanelOpts{ .anchor = .top_left };

fn rowPointer(down: bool) InputState {
    return .{ .pointer_x = 60, .pointer_y = 40, .pointer_down = down };
}

fn runToggleFrame(h: *Harness, on: *bool, in: InputState) bool {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    const changed = toggle(&h.ui, @src(), "x", on);
    h.ui.endPanel();
    h.ui.endFrame();
    return changed;
}

fn runSliderFrame(h: *Harness, v: *f32, in: InputState) bool {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "S", test_panel);
    const changed = sliderF32(&h.ui, @src(), "x", v, 0, 1);
    h.ui.endPanel();
    h.ui.endFrame();
    return changed;
}

const tab_labels = [_][]const u8{ "A", "B", "C" };

fn runTabBarFrame(h: *Harness, active: *usize, in: InputState) bool {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    const changed = tabBar(&h.ui, @src(), active, &tab_labels);
    h.ui.endPanel();
    h.ui.endFrame();
    return changed;
}

test "tabBar: Left/Right cycle the active index when focused (wrapping)" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    _ = runTabBarFrame(&h, &active, .{ .tab = true }); // focus the bar
    _ = runTabBarFrame(&h, &active, .{ .nav_right = true });
    try testing.expectEqual(@as(usize, 1), active);
    _ = runTabBarFrame(&h, &active, .{ .nav_right = true }); // 1 -> 2
    _ = runTabBarFrame(&h, &active, .{ .nav_right = true }); // 2 -> 0 (wrap)
    try testing.expectEqual(@as(usize, 0), active);
    _ = runTabBarFrame(&h, &active, .{ .nav_left = true }); // 0 -> 2 (wrap)
    try testing.expectEqual(@as(usize, 2), active);
}

test "tabBar: Left/Right ignored when the bar is not focused" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    _ = runTabBarFrame(&h, &active, .{ .nav_right = true });
    try testing.expectEqual(@as(usize, 0), active);
}

test "tabBar: focus + Enter does not hijack selection to the cell under the mouse" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    _ = runTabBarFrame(&h, &active, .{ .tab = true }); // focus the bar
    // Enter pressed with the mouse resting over the last cell (no click): no change.
    _ = runTabBarFrame(&h, &active, .{ .activate = true, .pointer_x = 180, .pointer_y = 40 });
    try testing.expectEqual(@as(usize, 0), active);
}

test "tabBar: a click selects the cell under the pointer" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    // First row spans x[10,210]; with 3 cells a click at x~180 lands in the last.
    _ = runTabBarFrame(&h, &active, .{ .pointer_x = 180, .pointer_y = 40, .pointer_down = true });
    _ = runTabBarFrame(&h, &active, .{ .pointer_x = 180, .pointer_y = 40, .pointer_down = false });
    try testing.expectEqual(@as(usize, 2), active);
}

test "tabBar: registers exactly one focus stop (not one per cell)" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    h.frame(.{});
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    _ = tabBar(&h.ui, @src(), &active, &tab_labels);
    try testing.expectEqual(@as(usize, 1), h.ui.focus_order.items.len);
    h.ui.endPanel();
    h.ui.endFrame();
}

const cycle_opts = [_][]const u8{ "1x", "2x", "4x", "8x" };

fn runCycleFrame(h: *Harness, active: *usize, in: InputState) bool {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "C", test_panel);
    const changed = cycle(&h.ui, @src(), "MSAA", active, &cycle_opts);
    h.ui.endPanel();
    h.ui.endFrame();
    return changed;
}

test "cycle: Left/Right step and clamp (no wrap) when focused" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    _ = runCycleFrame(&h, &active, .{ .tab = true }); // focus
    _ = runCycleFrame(&h, &active, .{ .nav_right = true });
    try testing.expectEqual(@as(usize, 1), active);
    _ = runCycleFrame(&h, &active, .{ .nav_left = true });
    try testing.expectEqual(@as(usize, 0), active);
    _ = runCycleFrame(&h, &active, .{ .nav_left = true }); // clamp at the bottom
    try testing.expectEqual(@as(usize, 0), active);
    for (0..5) |_| _ = runCycleFrame(&h, &active, .{ .nav_right = true });
    try testing.expectEqual(@as(usize, 3), active); // clamp at the top (4 options)
}

test "cycle: ignores Left/Right when not focused" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 1;
    _ = runCycleFrame(&h, &active, .{ .nav_right = true });
    try testing.expectEqual(@as(usize, 1), active);
}

test "cycle: clamps a stale index when the option set shrinks" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 3; // was on 8x
    h.frame(.{});
    _ = h.ui.beginPanel(@src(), "C", test_panel);
    _ = cycle(&h.ui, @src(), "MSAA", &active, &[_][]const u8{ "1x", "4x" }); // GPU now supports two
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expectEqual(@as(usize, 1), active); // clamped to the last valid index
}

test "cycle: click left half decrements, right half increments" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 2;
    // Control occupies the right of the row; x=200 is its right half -> increment.
    _ = runCycleFrame(&h, &active, .{ .pointer_x = 200, .pointer_y = 40, .pointer_down = true });
    _ = runCycleFrame(&h, &active, .{ .pointer_x = 200, .pointer_y = 40, .pointer_down = false });
    try testing.expectEqual(@as(usize, 3), active);
    // x=140 is the left half -> decrement.
    _ = runCycleFrame(&h, &active, .{ .pointer_x = 140, .pointer_y = 40, .pointer_down = true });
    _ = runCycleFrame(&h, &active, .{ .pointer_x = 140, .pointer_y = 40, .pointer_down = false });
    try testing.expectEqual(@as(usize, 2), active);
}

test "cycle: registers one focus stop" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    h.frame(.{});
    _ = h.ui.beginPanel(@src(), "C", test_panel);
    _ = cycle(&h.ui, @src(), "MSAA", &active, &cycle_opts);
    try testing.expectEqual(@as(usize, 1), h.ui.focus_order.items.len);
    h.ui.endPanel();
    h.ui.endFrame();
}

test "disabled: toggle is not a focus stop and does not flip on click" {
    var h = try Harness.init();
    defer h.deinit();
    var on = false;
    // Press inside then release inside (a normal activation) while disabled.
    h.frame(rowPointer(true));
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    h.ui.beginDisabled(true);
    _ = toggle(&h.ui, @src(), "x", &on);
    h.ui.endDisabled();
    try testing.expectEqual(@as(usize, 0), h.ui.focus_order.items.len); // not focusable
    h.ui.endPanel();
    h.ui.endFrame();

    h.frame(rowPointer(false));
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    h.ui.beginDisabled(true);
    _ = toggle(&h.ui, @src(), "x", &on);
    h.ui.endDisabled();
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expect(!on); // never flipped
}

test "disabled: cycle ignores a click and is not a focus stop" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 1;
    // x=200 is the right half -> would increment if enabled.
    h.frame(.{ .pointer_x = 200, .pointer_y = 40, .pointer_down = true });
    _ = h.ui.beginPanel(@src(), "C", test_panel);
    h.ui.beginDisabled(true);
    _ = cycle(&h.ui, @src(), "MSAA", &active, &cycle_opts);
    h.ui.endDisabled();
    try testing.expectEqual(@as(usize, 0), h.ui.focus_order.items.len);
    h.ui.endPanel();
    h.ui.endFrame();

    h.frame(.{ .pointer_x = 200, .pointer_y = 40, .pointer_down = false });
    _ = h.ui.beginPanel(@src(), "C", test_panel);
    h.ui.beginDisabled(true);
    _ = cycle(&h.ui, @src(), "MSAA", &active, &cycle_opts);
    h.ui.endDisabled();
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expectEqual(@as(usize, 1), active); // unchanged
}

// Fixed `@src()` call site so the toggle id is stable across the two frames below.
fn disabledToggleFrame(h: *Harness, on: *bool, in: InputState, disabled: bool) void {
    h.frame(in);
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    h.ui.beginDisabled(disabled);
    _ = toggle(&h.ui, @src(), "x", on);
    h.ui.endDisabled();
    h.ui.endPanel();
    h.ui.endFrame();
}

test "disabled: a widget disabled while focused releases focus" {
    var h = try Harness.init();
    defer h.deinit();
    var on = false;
    disabledToggleFrame(&h, &on, .{ .tab = true }, false); // enabled: Tab focuses it
    try testing.expect(h.ui.focused_id != 0);
    disabledToggleFrame(&h, &on, .{}, true); // now disabled -> focus released this frame
    try testing.expectEqual(@as(u64, 0), h.ui.focused_id);
}

test "disabled: slider ignores a drag and is not a focus stop" {
    var h = try Harness.init();
    defer h.deinit();
    var v: f32 = 0.5;
    h.frame(.{ .pointer_x = 17, .pointer_y = 50, .pointer_down = true });
    _ = h.ui.beginPanel(@src(), "S", test_panel);
    h.ui.beginDisabled(true);
    _ = sliderF32(&h.ui, @src(), "x", &v, 0, 1);
    h.ui.endDisabled();
    try testing.expectEqual(@as(usize, 0), h.ui.focus_order.items.len);
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expectEqual(@as(f32, 0.5), v); // drag ignored
}

test "disabled: tabBar ignores a click and is not a focus stop" {
    var h = try Harness.init();
    defer h.deinit();
    var active: usize = 0;
    h.frame(.{ .pointer_x = 180, .pointer_y = 40, .pointer_down = true });
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    h.ui.beginDisabled(true);
    _ = tabBar(&h.ui, @src(), &active, &tab_labels);
    h.ui.endDisabled();
    try testing.expectEqual(@as(usize, 0), h.ui.focus_order.items.len);
    h.ui.endPanel();
    h.ui.endFrame();

    h.frame(.{ .pointer_x = 180, .pointer_y = 40, .pointer_down = false });
    _ = h.ui.beginPanel(@src(), "T", test_panel);
    h.ui.beginDisabled(true);
    _ = tabBar(&h.ui, @src(), &active, &tab_labels);
    h.ui.endDisabled();
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expectEqual(@as(usize, 0), active); // unchanged
}
