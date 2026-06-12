#version 450

// GPU-resident numeric readout: decodes one digit per vertex-quad from a value
// that lives in a GPU buffer (the terrain probe's AGL), so the number is never
// read back to the CPU. Pairs with text.frag UNCHANGED: it emits the same
// (frag_uv, frag_color, frag_char_idx) varyings text.frag expects, just with
// char_idx computed here instead of supplied as a vertex attribute.
//
// The CPU emits a fixed layout of digit-slot quads (one per decimal place) at
// fixed screen positions, tagged with `in_place`; this shader reads the value,
// extracts the digit for each place, right-aligns with leading blanks, and maps
// the digit to the font glyph.

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // fragment-stage; declared so the block matches text.frag
    float ui_paper_white;   // fragment-stage
    float numeric_scale;    // unit scale for the decoded value (1.0 = m, 3.28084 = ft)
};

layout(location = 0) in vec2 in_pos;     // pixel coordinates
layout(location = 1) in vec2 in_uv;      // [0,1] within the character cell
layout(location = 2) in vec4 in_color;   // glyph color (RGBA)
layout(location = 3) in uint in_place;   // decimal place: 0 = ones, 1 = tens, ...

// TODO: generalize for reuse (currently AGL-specific). Make this a drop-in for
// any GPU-resident number: read `values[in_value_slot]` from a generic
// `hud_values: i32[]` SSBO (producers write display ints to slots: AGL=0,
// FPM-touchdown=1, glide=2, TAWS=3, ...) plus a per-quad value_slot attribute.
// `agl_m` would then move out of ProbeOut into hud_values, leaving ProbeOut as
// pure terrain primitives for GPU consumers (glide/TAWS).
//
// Probe output (std430), bound at the vertex stage. Mirrors `ProbeOut` in
// shaders/terrain/probe_eval.comp; we read only `agl_m`.
layout(std430, set = 0, binding = 1) readonly buffer ProbeVals {
    float elevation_m;
    float water_logit;
    int agl_m;
    int tile_found;
};

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
layout(location = 2) flat out uint frag_char_idx;

void main() {
    // AGL is clamped >= 0 in the probe; guard anyway. Scale to the display unit
    // (meters * numeric_scale) and round to the nearest whole displayed unit.
    int value = int(round(float(max(agl_m, 0)) * numeric_scale));

    // 10^place (place is small: a few digits).
    uint p10 = 1u;
    for (uint i = 0u; i < in_place; i++) p10 *= 10u;

    uint digit = (uint(value) / p10) % 10u;

    // Right-align: blank a leading slot whose place is above the value's
    // magnitude, but always render the ones place (so 0 shows as "0"). The space
    // glyph (0x20) is all-zero in the font, so text.frag draws nothing for it.
    uint char_idx;
    if (in_place > 0u && uint(value) < p10) {
        char_idx = 0x20u;
    } else {
        char_idx = 0x30u + digit; // '0'..'9' are contiguous in the bitmap font
    }

    float x = in_pos.x / screen_size.x * 2.0 - 1.0;
    float y = in_pos.y / screen_size.y * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    frag_uv = in_uv;
    frag_color = in_color;
    frag_char_idx = char_idx;
}
