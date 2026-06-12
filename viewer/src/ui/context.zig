//! Immediate-mode UI context: the interactive layer over the pure DrawList
//! painter (`ui.zig`). Pure: depends only on `std` and the painter; knows nothing
//! about Vulkan, SDL, or app types. It holds per-frame layout + a latched input
//! snapshot + the cross-frame focus/active state, and emits geometry into a
//! caller-owned `DrawList` that the backend renders.
//!
//! Input arrives as a NEUTRAL `InputState` the app fills from its window system
//! (no SDL types here). Pointer (mouse/touch) and focus-nav (keyboard/gamepad)
//! are co-equal: a widget activates identically from a pointer release inside it
//! or from a focused activate-key edge (see `behavior`).
//!
//! State model: a per-frame `ArenaAllocator` (reset each `beginFrame`, backs the
//! frame's focusable list) plus a persistent `AutoHashMap` keyed by widget id for
//! cross-frame scratch (drag/scroll/text-input). The current widget set
//! (button/toggle/slider) does not yet touch the map; it is the keepable
//! foundation later widgets build on.

const std = @import("std");
const draw = @import("ui.zig");
const Color = draw.Color;
const Screen = draw.Screen;
const DrawList = draw.DrawList;

pub const SrcLoc = std.builtin.SourceLocation;

pub const PointerKind = enum { mouse, touch };

/// Neutral per-frame input snapshot the app fills from its window system. The
/// pointer is in DEVICE pixels (the same space as `DrawList`/`Screen`, so hit
/// tests need no conversion). The app supplies the pointer *level*
/// (`pointer_down`) and one-frame key/nav *edges*; the context derives the
/// pointer press/release edges itself (one owner for that logic).
pub const InputState = struct {
    pointer_x: f32 = 0,
    pointer_y: f32 = 0,
    pointer_kind: PointerKind = .mouse,
    pointer_down: bool = false,
    scroll: f32 = 0,

    // Focus / nav edges (one-frame pulses). Fill from key-DOWN events or gamepad
    // buttons; the same fields accept either source (d-pad nav is purely an app
    // population detail, no widget change).
    activate: bool = false, // Enter / Space / gamepad A
    cancel: bool = false, //   Esc / gamepad B
    tab: bool = false,
    shift: bool = false, //    level modifier: shift+tab moves focus backward
    nav_up: bool = false, //   focus prev (also: arrow up)
    nav_down: bool = false, // focus next (also: arrow down)
    nav_left: bool = false, // decrement focused slider
    nav_right: bool = false, // increment focused slider

    // Tab-bar / category switch (distinct from `tab` focus-advance): shoulder
    // buttons on a gamepad, `[` / `]` on a keyboard. The app reads these at the
    // call site to cycle a category enum; the widget layer does not consume them.
    tab_prev: bool = false,
    tab_next: bool = false,
};

/// Cross-frame per-widget scratch. Empty for the current widget set; reserved for
/// drag-with-grab-offset, scroll position, and text-input cursor/selection.
pub const WidgetState = struct {
    drag_origin: f32 = 0,
    scroll_offset: f32 = 0,
};

/// Axis-aligned rect in device px. Half-open on the high edge so adjacent rows
/// don't both claim a boundary pixel.
pub const Rect = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    pub fn contains(r: Rect, px: f32, py: f32) bool {
        return px >= r.x and px < r.x + r.w and py >= r.y and py < r.y + r.h;
    }
};

/// What a widget's `behavior` call resolved this frame.
pub const Response = struct {
    /// Fired this frame (pointer click released inside, or focused activate key).
    activated: bool = false,
    /// This widget currently owns `active_id` (mid-press / mid-drag).
    held: bool = false,
};

/// Colors + logical-pixel metrics. Aviation-flat: dark translucent panel, thin
/// mil-green strokes, monospace text. Metrics are LOGICAL px (multiplied by
/// `Screen.scale` at emit time).
pub const Theme = struct {
    panel_bg: Color = Color.rgba(0.05, 0.07, 0.06, 0.92),
    panel_border: Color = Color.rgba(0.45, 0.95, 0.55, 1.0), // opaque: a faint border clashed with the fill
    title: Color = Color.rgba(0.65, 1.0, 0.72, 1.0),
    text: Color = Color.rgba(0.82, 0.96, 0.84, 1.0),
    widget_bg: Color = Color.rgba(0.15, 0.19, 0.16, 1.0),
    widget_hot: Color = Color.rgba(0.23, 0.29, 0.24, 1.0),
    widget_active: Color = Color.rgba(0.30, 0.40, 0.32, 1.0),
    accent: Color = Color.rgba(0.45, 0.95, 0.55, 1.0),
    focus_ring: Color = Color.rgba(1.0, 0.85, 0.40, 1.0),
    knob: Color = Color.rgba(0.86, 0.96, 0.87, 1.0),
    text_disabled: Color = Color.rgba(0.42, 0.49, 0.43, 1.0), // dimmed label/value on a greyed-out row
    widget_disabled: Color = Color.rgba(0.11, 0.13, 0.11, 1.0), // dimmed control fill when disabled

    pad: f32 = 10, //       panel inner padding
    // TODO(touch): rows are sized for mouse/keyboard; bump to a >=44 logical-px
    // hit target (visual smaller, invisible padding) when touch events are wired.
    row_h: f32 = 26, //     interactive row height
    row_gap: f32 = 6, //    vertical gap between rows
    corner: f32 = 4, //     widget corner radius
    border: f32 = 1.5, //   stroke half-... actually full thickness for outlines
    font_scale: f32 = 1.25, // glyph scale (x GLYPH_PX) in logical px
    label_pad: f32 = 7, //  text inset inside a widget
    control_w: f32 = 64, // width of the control portion (toggle/slider track) on a row
};

const MAX_ID_STACK = 8;

pub const Ui = struct {
    // ---- Persistent (live for the Ui's lifetime) ----
    gpa: std.mem.Allocator,
    persist: std.AutoHashMap(u64, WidgetState),
    arena: std.heap.ArenaAllocator,
    theme: Theme = .{},

    /// Pointer is over this widget THIS frame (recomputed each frame). Mouse only;
    /// touch sets it only while pressing (no hover-before-contact).
    hot_id: u64 = 0,
    /// Pressed / being dragged; held across frames until release.
    active_id: u64 = 0,
    /// Keyboard / d-pad focus; persists across frames.
    focused_id: u64 = 0,
    /// True when `active_id` was claimed by a pointer press (vs a key).
    active_via_pointer: bool = false,

    prev_pointer_down: bool = false,
    ptr_pressed: bool = false, // derived edge for this frame
    ptr_released: bool = false,
    close_signal: bool = false, // Esc with nothing focused; app consumes via consumeClose

    // ---- Per-frame (reset each beginFrame) ----
    input: InputState = .{},
    screen: Screen = .{ .w = 0, .h = 0, .scale = 1 },
    dl: *DrawList = undefined,
    any_panel_open: bool = false,

    id_stack: [MAX_ID_STACK]u64 = [_]u64{0} ** MAX_ID_STACK,
    id_depth: u32 = 0,

    // TODO(nesting): `panel_open`/`panel` are single values, not a stack, and
    // `pushSeed` silently no-ops at MAX_ID_STACK while `popSeed` always pops, so
    // nested panels (or >=8 deep id scopes) would corrupt panel/seed state. Only
    // one flat panel is built today; make these a stack before nesting panels.
    panel_open: bool = false,
    panel: Panel = undefined,

    /// Ordered focusable ids registered THIS frame, in the arena. `endFrame`
    /// resolves a pending focus move against the complete list.
    focus_order: std.ArrayList(u64) = .empty,

    /// One-shot request (set via `focusFirst`) to focus the first focusable in the
    /// coming `endFrame`. Used when entering a sub-page so a controller's Left/Right
    /// acts on the tab bar immediately instead of needing a Down press first.
    pending_focus_first: bool = false,

    /// `beginDisabled`/`endDisabled` nesting. `disabled_depth` counts open scopes;
    /// `first_disabled` is the depth at which the OUTERMOST still-open disabled scope
    /// began (null = none). Disabled == `first_disabled != null`. Two counters instead
    /// of a fixed stack so arbitrary nesting can't overflow a cap and desync the state.
    disabled_depth: u32 = 0,
    first_disabled: ?u32 = null,

    const Panel = struct {
        outer_x: f32,
        outer_w: f32,
        x: f32, // content left
        w: f32, // content width
        y: f32, // panel top
        cursor_y: f32, // next row top
        pad: f32,
        radius: f32,
        bg_slot: ?u32, // reserved SDF slot for the (rounded) background, filled in endPanel
    };

    pub fn init(gpa: std.mem.Allocator) Ui {
        return .{
            .gpa = gpa,
            .persist = std.AutoHashMap(u64, WidgetState).init(gpa),
            .arena = std.heap.ArenaAllocator.init(gpa),
        };
    }

    pub fn deinit(ui: *Ui) void {
        ui.persist.deinit();
        ui.arena.deinit();
    }

    // ---- Id hashing ----------------------------------------------------------

    fn currentSeed(ui: *const Ui) u64 {
        return ui.id_stack[ui.id_depth];
    }

    fn pushSeed(ui: *Ui, seed: u64) void {
        if (ui.id_depth + 1 < MAX_ID_STACK) {
            ui.id_depth += 1;
            ui.id_stack[ui.id_depth] = seed;
        }
    }

    fn popSeed(ui: *Ui) void {
        if (ui.id_depth > 0) ui.id_depth -= 1;
    }

    /// Hash a call site (`@src()`) plus a caller `extra` (loop disambiguator) into
    /// a widget id, mixed with the current scope seed (so the same source line in
    /// two panels does not collide). Id 0 is reserved as "none".
    pub fn id(ui: *const Ui, src: SrcLoc, extra: u64) u64 {
        return mixId(ui.currentSeed(), src, extra);
    }

    fn mixId(seed: u64, src: SrcLoc, extra: u64) u64 {
        var h = std.hash.Wyhash.init(seed);
        // Hash the source FIELDS explicitly: std.hash.autoHash @compileErrors on
        // SourceLocation's slice fields. column distinguishes two widgets on one line.
        h.update(src.file);
        h.update(std.mem.asBytes(&src.line));
        h.update(std.mem.asBytes(&src.column));
        h.update(std.mem.asBytes(&extra));
        const v = h.final();
        return if (v == 0) 1 else v;
    }

    // ---- Frame lifecycle -----------------------------------------------------

    pub fn beginFrame(ui: *Ui, input: InputState, screen: Screen, dl: *DrawList) void {
        _ = ui.arena.reset(.retain_capacity);
        ui.input = input;
        ui.screen = screen;
        ui.dl = dl;
        ui.hot_id = 0;
        ui.id_depth = 0;
        ui.any_panel_open = false;
        ui.panel_open = false;
        ui.pending_focus_first = false;
        ui.disabled_depth = 0;
        ui.first_disabled = null;
        ui.focus_order = .empty;

        ui.ptr_pressed = input.pointer_down and !ui.prev_pointer_down;
        ui.ptr_released = !input.pointer_down and ui.prev_pointer_down;
        // Stale-active guard: the button has been up for a full frame (no release
        // edge here) yet a pointer-active widget lingers, so its widget stopped
        // being drawn mid-press (e.g. menu closed while held). Drop it; this never
        // fires on the genuine release frame (prev_pointer_down is true then).
        if (!input.pointer_down and !ui.prev_pointer_down and ui.active_via_pointer) {
            ui.active_id = 0;
            ui.active_via_pointer = false;
        }
        ui.prev_pointer_down = input.pointer_down;
    }

    pub fn endFrame(ui: *Ui) void {
        // Cancel (Esc / B) backs out in a single press: signal the app and drop
        // focus the same frame. The app decides what "back" means (pop a sub-page,
        // then close). This is modal-menu semantics; there is no "unfocus first"
        // step, which on a controller (always something focused) would force two
        // presses to leave a screen.
        if (ui.input.cancel) {
            ui.close_signal = true;
            ui.focused_id = 0;
        }

        // Focus move, resolved here where `focus_order` is complete (so wrap and
        // "from none -> an end" are correct). Tab/down = forward, shift-tab/up = back.
        var move: i32 = 0;
        if (ui.input.tab) move = if (ui.input.shift) -1 else 1;
        if (ui.input.nav_down) move = 1;
        if (ui.input.nav_up) move = -1;

        // A cancel (Esc) this frame owns the input: don't also run a focus move
        // (which would re-select an item the same frame). Two keys can land in one
        // frame's snapshot.
        const list = ui.focus_order.items;
        if (move != 0 and !ui.input.cancel and list.len > 0) {
            const n: i32 = @intCast(list.len);
            const next: i32 = if (std.mem.indexOfScalar(u64, list, ui.focused_id)) |ci|
                @mod(@as(i32, @intCast(ci)) + move, n)
            else if (move > 0) 0 else n - 1;
            ui.focused_id = list[@intCast(next)];
        }

        // One-shot focus-first (entering a sub-page): land focus on the first
        // focusable so Left/Right work immediately. A cancel this frame wins.
        if (ui.pending_focus_first and !ui.input.cancel and list.len > 0) {
            ui.focused_id = list[0];
        }
        ui.pending_focus_first = false;
    }

    /// Request that the coming `endFrame` focus the first focusable registered this
    /// frame. Call after `beginFrame`, before the widgets are built.
    pub fn focusFirst(ui: *Ui) void {
        ui.pending_focus_first = true;
    }

    /// Drop keyboard focus and any in-progress pointer interaction. Called when the
    /// owning menu is toggled, so focus state never leaks across open/close cycles
    /// (a stale `focused_id` would otherwise make the first Esc on reopen merely
    /// unfocus instead of closing).
    pub fn clearFocus(ui: *Ui) void {
        ui.focused_id = 0;
        ui.active_id = 0;
        ui.active_via_pointer = false;
    }

    /// Open a disabled scope: until the matching `endDisabled`, widgets inside render
    /// greyed and ignore focus + pointer input (they still occupy their row and show
    /// their bound value, just inert). `cond=false` is a live no-op scope, so call
    /// sites read cleanly as `beginDisabled(!supported)`. Nestable to any depth; a
    /// child stays disabled while any ancestor scope is disabled.
    pub fn beginDisabled(ui: *Ui, cond: bool) void {
        if (cond and ui.first_disabled == null) ui.first_disabled = ui.disabled_depth;
        ui.disabled_depth += 1;
    }

    /// Close the most recent `beginDisabled` scope. An unbalanced extra call is a
    /// no-op (cannot underflow).
    pub fn endDisabled(ui: *Ui) void {
        if (ui.disabled_depth == 0) return;
        ui.disabled_depth -= 1;
        // Cleared once we pop back to (or above) the scope that first disabled.
        if (ui.first_disabled) |fd| {
            if (ui.disabled_depth <= fd) ui.first_disabled = null;
        }
    }

    /// True when the current scope (or any ancestor) is disabled.
    pub fn isDisabled(ui: *const Ui) bool {
        return ui.first_disabled != null;
    }

    /// True if a panel was opened this frame. The app's menu-open flag is the
    /// authoritative gate for suspending camera/flight input (it is known before
    /// the UI is built, unlike this); `capturesInput` mirrors it for assertions /
    /// tests and future multi-panel logic.
    pub fn capturesInput(ui: *const Ui) bool {
        return ui.any_panel_open;
    }

    /// Read-and-clear the Esc-close request raised in `endFrame`.
    pub fn consumeClose(ui: *Ui) bool {
        const c = ui.close_signal;
        ui.close_signal = false;
        return c;
    }

    // ---- Panels + layout -----------------------------------------------------

    pub const PanelOpts = struct {
        anchor: draw.Anchor = .center,
        /// Logical-px offset of the panel's top-left from the anchor point.
        off_x: f32 = 0,
        off_y: f32 = 0,
        width: f32 = 220, // logical px
        open: bool = true,
    };

    /// Begin a panel. Returns false (and draws nothing) when closed; the caller
    /// gates its widgets with `if (ui.beginPanel(...)) { ... ui.endPanel(); }`.
    /// The background is a rounded fill sized to its content: its SDF slot is
    /// reserved here (so it stays BEHIND the widgets in append order) and filled in
    /// `endPanel` once the height is known. The rounded fill matches the rounded
    /// border (a sharp shape-stream rect would clash with it at the corners).
    pub fn beginPanel(ui: *Ui, src: SrcLoc, title: []const u8, opts: PanelOpts) bool {
        if (!opts.open) return false;
        ui.any_panel_open = true;
        ui.panel_open = true;
        ui.pushSeed(mixId(ui.currentSeed(), src, 0));

        const s = ui.screen;
        const pad = s.px(ui.theme.pad);
        const w_px = s.px(opts.width);
        const o = s.anchor(opts.anchor, opts.off_x, opts.off_y);
        ui.panel = .{
            .outer_x = o[0],
            .outer_w = w_px,
            .x = o[0] + pad,
            .w = w_px - 2 * pad,
            .y = o[1],
            .cursor_y = o[1] + pad,
            .pad = pad,
            .radius = s.px(ui.theme.corner),
            // Reserve the background quad first so it renders behind every widget.
            .bg_slot = ui.dl.reserveRoundedRect(),
        };

        // Title row (glyph stream -> always above the panel background).
        const cell = s.px(draw.GLYPH_PX * ui.theme.font_scale);
        _ = ui.dl.text(ui.panel.x, ui.panel.cursor_y, s.textScale(ui.theme.font_scale), ui.theme.title, title);
        ui.panel.cursor_y += cell + s.px(ui.theme.row_gap);
        return true;
    }

    pub fn endPanel(ui: *Ui) void {
        if (!ui.panel_open) return;
        const p = ui.panel;
        const height = (p.cursor_y - p.y) + p.pad - ui.screen.px(ui.theme.row_gap);
        // Fill the reserved background slot (rounded, behind the widgets) now that
        // the height is known. Both the fill and the outline use `p.radius`, so the
        // corners match.
        if (p.bg_slot) |slot| ui.dl.commitRoundedRect(slot, p.outer_x, p.y, p.outer_w, height, p.radius, ui.theme.panel_bg);
        // Frame outline on the SDF stream, emitted last -> a clean border on top.
        ui.dl.roundedRectOutline(p.outer_x, p.y, p.outer_w, height, p.radius, ui.screen.px(ui.theme.border), ui.theme.panel_border);
        ui.popSeed();
        ui.panel_open = false;
    }

    /// Reserve the next full-width interactive row; advances the layout cursor.
    pub fn rowRect(ui: *Ui) Rect {
        return ui.rowRectH(ui.theme.row_h);
    }

    /// Reserve a full-width row of a custom logical height; for controls that need
    /// more than one line (e.g. a value-cycle with a position indicator below it).
    pub fn rowRectH(ui: *Ui, logical_h: f32) Rect {
        const s = ui.screen;
        const h = s.px(logical_h);
        const r = Rect{ .x = ui.panel.x, .y = ui.panel.cursor_y, .w = ui.panel.w, .h = h };
        ui.panel.cursor_y += h + s.px(ui.theme.row_gap);
        return r;
    }

    /// Append a widget id to this frame's focusable list (Tab/nav order).
    pub fn registerFocusable(ui: *Ui, wid: u64) void {
        ui.focus_order.append(ui.arena.allocator(), wid) catch {};
    }

    // ---- The activation state machine (symmetric pointer + focus) ------------

    /// Resolve hover/press/release/activate for a widget occupying `rect`. Pointer
    /// release-inside (while we own active) and focused+activate are the SAME
    /// activation. Touch never hovers before contact.
    pub fn behavior(ui: *Ui, wid: u64, rect: Rect) Response {
        var r = Response{};
        const inside = rect.contains(ui.input.pointer_x, ui.input.pointer_y);

        // Mouse hover, before the press block so pressing a widget doesn't lock
        // out its own hover this frame. last-drawn-wins; a drag on another widget
        // (active_id != 0) suppresses hover.
        if (ui.input.pointer_kind == .mouse and inside and ui.active_id == 0) ui.hot_id = wid;

        if (ui.ptr_pressed and inside and ui.active_id == 0) {
            ui.active_id = wid;
            ui.active_via_pointer = true;
            ui.focused_id = wid; // pointer press moves focus here too
        }

        if (ui.ptr_released and ui.active_id == wid and ui.active_via_pointer) {
            if (inside) r.activated = true; // release inside = click; drag-off cancels
            ui.active_id = 0;
            ui.active_via_pointer = false;
        }

        // Touch has no cursor-before-press: "hot" exists only while the finger is
        // on this widget. Evaluated after the press block so the contact frame counts.
        if (ui.input.pointer_kind == .touch and ui.active_id == wid) ui.hot_id = wid;

        // Focus-key activate. Gated on NOT being pointer-held so a pointer press
        // and an activate-key edge in the same frame can't both fire (the pointer
        // path activates on release-inside instead).
        if (ui.focused_id == wid and ui.input.activate and !ui.active_via_pointer) r.activated = true;

        r.held = (ui.active_id == wid);
        return r;
    }

    pub fn isHot(ui: *const Ui, wid: u64) bool {
        return ui.hot_id == wid;
    }
    pub fn isFocused(ui: *const Ui, wid: u64) bool {
        return ui.focused_id == wid;
    }
};

// =============================================================================
// Tests
// =============================================================================

const testing = std.testing;

/// A test Ui with a scale-1 Screen (device px == logical px) and a heap DrawList.
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
    fn frame(h: *Harness, input: InputState) void {
        h.ui.beginFrame(input, Screen.fromExtent(800, 600, 600, 1.0), h.dl);
    }
};

fn srcA() SrcLoc {
    return @src();
}
fn srcB() SrcLoc {
    return @src();
}

test "id: stable across frames, never zero" {
    const a1 = Ui.mixId(0, srcA(), 0);
    const a2 = Ui.mixId(0, srcA(), 0);
    try testing.expectEqual(a1, a2);
    try testing.expect(a1 != 0);
}

test "id: differs by call site and by extra (loop disambiguation)" {
    try testing.expect(Ui.mixId(0, srcA(), 0) != Ui.mixId(0, srcB(), 0));
    try testing.expect(Ui.mixId(0, srcA(), 0) != Ui.mixId(0, srcA(), 1));
    // n distinct loop ids
    var seen: [4]u64 = undefined;
    for (0..4) |i| seen[i] = Ui.mixId(0, srcA(), i);
    for (0..4) |i| for (i + 1..4) |j| try testing.expect(seen[i] != seen[j]);
}

test "id: panel scope seed avoids same-line collisions across panels" {
    const s = srcA();
    try testing.expect(Ui.mixId(111, s, 0) != Ui.mixId(222, s, 0));
}

test "hit-test: mouse over rect sets hot, outside clears" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 42;
    const rect = Rect{ .x = 10, .y = 10, .w = 100, .h = 20 };

    h.frame(.{ .pointer_x = 50, .pointer_y = 15 });
    _ = h.ui.behavior(wid, rect);
    try testing.expectEqual(wid, h.ui.hot_id);

    h.frame(.{ .pointer_x = 500, .pointer_y = 15 });
    _ = h.ui.behavior(wid, rect);
    try testing.expectEqual(@as(u64, 0), h.ui.hot_id);
}

test "hit-test: touch does not hover before press, hovers only while pressed" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 7;
    const rect = Rect{ .x = 0, .y = 0, .w = 50, .h = 50 };

    h.frame(.{ .pointer_kind = .touch, .pointer_x = 25, .pointer_y = 25, .pointer_down = false });
    _ = h.ui.behavior(wid, rect);
    try testing.expectEqual(@as(u64, 0), h.ui.hot_id);

    h.frame(.{ .pointer_kind = .touch, .pointer_x = 25, .pointer_y = 25, .pointer_down = true });
    _ = h.ui.behavior(wid, rect);
    try testing.expectEqual(wid, h.ui.hot_id);
}

test "hit-test: last-drawn-wins on overlap" {
    var h = try Harness.init();
    defer h.deinit();
    const rect = Rect{ .x = 0, .y = 0, .w = 100, .h = 100 };
    h.frame(.{ .pointer_x = 50, .pointer_y = 50 });
    _ = h.ui.behavior(1, rect);
    _ = h.ui.behavior(2, rect);
    try testing.expectEqual(@as(u64, 2), h.ui.hot_id);
}

test "behavior: pointer press then release inside fires once" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 9;
    const rect = Rect{ .x = 0, .y = 0, .w = 40, .h = 20 };

    h.frame(.{ .pointer_x = 10, .pointer_y = 10, .pointer_down = true }); // press edge
    var r = h.ui.behavior(wid, rect);
    try testing.expect(!r.activated);
    try testing.expect(r.held);

    h.frame(.{ .pointer_x = 10, .pointer_y = 10, .pointer_down = false }); // release edge
    r = h.ui.behavior(wid, rect);
    try testing.expect(r.activated);
}

test "behavior: release outside does not fire, clears active" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 9;
    const rect = Rect{ .x = 0, .y = 0, .w = 40, .h = 20 };

    h.frame(.{ .pointer_x = 10, .pointer_y = 10, .pointer_down = true });
    _ = h.ui.behavior(wid, rect);
    h.frame(.{ .pointer_x = 500, .pointer_y = 500, .pointer_down = false }); // released off-widget
    const r = h.ui.behavior(wid, rect);
    try testing.expect(!r.activated);
    try testing.expectEqual(@as(u64, 0), h.ui.active_id);
}

test "behavior: focused + activate key fires with no pointer" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 5;
    const rect = Rect{ .x = 0, .y = 0, .w = 40, .h = 20 };
    h.ui.focused_id = wid;
    h.frame(.{ .pointer_x = 999, .pointer_y = 999, .activate = true });
    const r = h.ui.behavior(wid, rect);
    try testing.expect(r.activated);
}

test "focus: Tab cycles forward and wraps; Shift-Tab backward and wraps" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 100, 200, 300 };

    // From none, Tab picks the first.
    runFocusFrame(&h, &ids, .{ .tab = true });
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);
    runFocusFrame(&h, &ids, .{ .tab = true });
    try testing.expectEqual(@as(u64, 200), h.ui.focused_id);
    runFocusFrame(&h, &ids, .{ .tab = true });
    try testing.expectEqual(@as(u64, 300), h.ui.focused_id);
    runFocusFrame(&h, &ids, .{ .tab = true }); // wrap
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);

    // Shift-Tab backward, wraps to last.
    runFocusFrame(&h, &ids, .{ .tab = true, .shift = true });
    try testing.expectEqual(@as(u64, 300), h.ui.focused_id);
}

test "focus: arrows behave as Tab; nav_up == shift-tab" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 100, 200, 300 };
    runFocusFrame(&h, &ids, .{ .nav_down = true });
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);
    runFocusFrame(&h, &ids, .{ .nav_down = true });
    try testing.expectEqual(@as(u64, 200), h.ui.focused_id);
    runFocusFrame(&h, &ids, .{ .nav_up = true });
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);
}

fn runFocusFrame(h: *Harness, ids: []const u64, input: InputState) void {
    h.frame(input);
    for (ids) |i| h.ui.registerFocusable(i);
    h.ui.endFrame();
}

test "focus: arena resets focus_order each frame (no cross-frame growth)" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 1, 2 };
    runFocusFrame(&h, &ids, .{});
    try testing.expectEqual(@as(usize, 2), h.ui.focus_order.items.len);
    runFocusFrame(&h, &ids, .{});
    try testing.expectEqual(@as(usize, 2), h.ui.focus_order.items.len); // reset, not 4
}

test "Esc backs out in one press: signals close and drops focus" {
    var h = try Harness.init();
    defer h.deinit();
    h.ui.focused_id = 123; // something focused (e.g. controller on a widget)
    h.frame(.{ .cancel = true });
    h.ui.endFrame();
    try testing.expectEqual(@as(u64, 0), h.ui.focused_id); // focus dropped
    try testing.expect(h.ui.consumeClose()); // and close signalled the SAME press
    try testing.expect(!h.ui.consumeClose()); // cleared on read
}

test "Esc the same frame as a focus-move clears focus without re-selecting" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 100, 200, 300 };
    h.ui.focused_id = 200;
    h.frame(.{ .cancel = true, .tab = true }); // two keys in one frame's snapshot
    for (ids) |i| h.ui.registerFocusable(i);
    h.ui.endFrame();
    // Cancel wins: focus cleared, the move is suppressed (not re-selected to list[0]).
    try testing.expectEqual(@as(u64, 0), h.ui.focused_id);
    try testing.expect(h.ui.consumeClose()); // cancel always signals back-out now
}

test "focusFirst lands focus on the first focusable in endFrame" {
    var h = try Harness.init();
    defer h.deinit();
    const ids = [_]u64{ 100, 200, 300 };
    h.frame(.{});
    h.ui.focusFirst();
    for (ids) |i| h.ui.registerFocusable(i);
    h.ui.endFrame();
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);
    // One-shot: next frame with no request leaves focus alone.
    h.frame(.{});
    for (ids) |i| h.ui.registerFocusable(i);
    h.ui.endFrame();
    try testing.expectEqual(@as(u64, 100), h.ui.focused_id);
}

test "clearFocus drops focus and in-progress pointer interaction" {
    var h = try Harness.init();
    defer h.deinit();
    h.ui.focused_id = 5;
    h.ui.active_id = 5;
    h.ui.active_via_pointer = true;
    h.ui.clearFocus();
    try testing.expectEqual(@as(u64, 0), h.ui.focused_id);
    try testing.expectEqual(@as(u64, 0), h.ui.active_id);
    try testing.expect(!h.ui.active_via_pointer);
}

test "behavior: pointer-held widget does not also key-activate the same frame" {
    var h = try Harness.init();
    defer h.deinit();
    const wid: u64 = 9;
    const rect = Rect{ .x = 0, .y = 0, .w = 40, .h = 20 };
    h.ui.focused_id = wid;
    // press inside (claims active_via_pointer) AND an activate edge in one frame.
    h.frame(.{ .pointer_x = 10, .pointer_y = 10, .pointer_down = true, .activate = true });
    const r = h.ui.behavior(wid, rect);
    try testing.expect(!r.activated); // pointer path fires on release, not now
}

test "pointer edge derivation from the down level" {
    var h = try Harness.init();
    defer h.deinit();
    h.frame(.{ .pointer_down = false });
    try testing.expect(!h.ui.ptr_pressed and !h.ui.ptr_released);
    h.frame(.{ .pointer_down = true });
    try testing.expect(h.ui.ptr_pressed and !h.ui.ptr_released);
    h.frame(.{ .pointer_down = true }); // held: no new edge
    try testing.expect(!h.ui.ptr_pressed and !h.ui.ptr_released);
    h.frame(.{ .pointer_down = false });
    try testing.expect(!h.ui.ptr_pressed and h.ui.ptr_released);
}

test "panel: capturesInput false without a panel, true with one open" {
    var h = try Harness.init();
    defer h.deinit();
    h.frame(.{});
    h.ui.endFrame();
    try testing.expect(!h.ui.capturesInput());

    h.frame(.{});
    try testing.expect(h.ui.beginPanel(@src(), "Demo", .{}));
    h.ui.endPanel();
    h.ui.endFrame();
    try testing.expect(h.ui.capturesInput());
}

test "panel: closed gates content (returns false, emits nothing)" {
    var h = try Harness.init();
    defer h.deinit();
    h.frame(.{});
    const opened = h.ui.beginPanel(@src(), "Demo", .{ .open = false });
    try testing.expect(!opened);
    h.ui.endFrame();
    try testing.expectEqual(@as(u32, 0), h.dl.glyph_count); // no title drawn
    try testing.expect(!h.ui.capturesInput());
}

test "layout: rowRect advances the cursor by row_h + gap" {
    var h = try Harness.init();
    defer h.deinit();
    h.frame(.{});
    _ = h.ui.beginPanel(@src(), "Demo", .{});
    const r0 = h.ui.rowRect();
    const r1 = h.ui.rowRect();
    const s = h.ui.screen;
    try testing.expectEqual(r0.y + s.px(h.ui.theme.row_h) + s.px(h.ui.theme.row_gap), r1.y);
    h.ui.endPanel();
    h.ui.endFrame();
}

test "disabled: stack nests and ORs with the parent; balanced pop restores" {
    var ui = Ui.init(testing.allocator);
    defer ui.deinit();
    try testing.expect(!ui.isDisabled());
    ui.beginDisabled(false); // a live (no-op) scope
    try testing.expect(!ui.isDisabled());
    ui.beginDisabled(true); // now disabled
    try testing.expect(ui.isDisabled());
    ui.beginDisabled(false); // child of a disabled scope stays disabled
    try testing.expect(ui.isDisabled());
    ui.endDisabled();
    try testing.expect(ui.isDisabled());
    ui.endDisabled();
    try testing.expect(!ui.isDisabled()); // back to the live scope
    ui.endDisabled();
    try testing.expect(!ui.isDisabled());
    ui.endDisabled(); // extra pop saturates, no underflow
    try testing.expect(!ui.isDisabled());
}

test "disabled: deep nesting past the old cap stays balanced (no desync)" {
    var ui = Ui.init(testing.allocator);
    defer ui.deinit();
    // One outer disabled scope, then many nested live scopes (>4, the old cap).
    ui.beginDisabled(true);
    for (0..10) |_| ui.beginDisabled(false);
    try testing.expect(ui.isDisabled());
    // Unwind the inner live scopes: still disabled because the outer scope is open.
    for (0..10) |_| {
        ui.endDisabled();
        try testing.expect(ui.isDisabled());
    }
    ui.endDisabled(); // close the outer disabled scope
    try testing.expect(!ui.isDisabled());
}

test "disabled: beginFrame resets the stack" {
    var h = try Harness.init();
    defer h.deinit();
    h.frame(.{});
    h.ui.beginDisabled(true);
    try testing.expect(h.ui.isDisabled());
    h.frame(.{}); // next frame must clear a leaked disabled scope
    try testing.expect(!h.ui.isDisabled());
    h.ui.endFrame();
}
