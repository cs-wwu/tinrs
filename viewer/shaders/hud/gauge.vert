#version 450

// GPU-driven AGL fill bar. The fill height encodes the terrain probe's AGL, read
// from the same GPU buffer the numeric readout uses (binding 1), so it never hits
// the CPU. The CPU emits one quad spanning the bar's FULL extent; this shader
// shrinks the top down to the fill fraction. Pairs with gauge.frag (solid).
//
// Reuses the numeric pipeline's descriptor set + pipeline layout (only binding 1
// is needed). TODO: generalize alongside numeric.vert onto a shared hud_values[]
// SSBO so any GPU value can drive a digit field or a bar.

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // fragment-stage; declared so the block matches gauge.frag
    float ui_paper_white;   // fragment-stage
};

layout(location = 0) in vec2 in_pos;     // (x, y_bottom) screen px
layout(location = 1) in vec2 in_params;  // (y_top_full, fill_weight: 0 = bottom, 1 = top)
layout(location = 2) in vec4 in_color;   // normal (in-band) fill color

layout(std430, set = 0, binding = 1) readonly buffer ProbeVals {
    float elevation_m;
    float water_logit;
    int agl_m;
    int tile_found;
};

layout(location = 0) out vec4 frag_color;

// AGL fill band + warning threshold (meters). Tune here; later a settings knob.
// Keep BAND_MAX aligned with the design intent (low-altitude regime). The fill is
// linear; for a log scale (more travel when low) swap the `fraction` line for
// clamp(log(agl + 1.0) / log(BAND_MAX + 1.0), 0.0, 1.0).
const float BAND_MAX = 500.0;
const float WARN_AGL = 150.0;
const vec3 WARN_COLOR = vec3(0.90, 0.25, 0.20);

void main() {
    float agl = float(max(agl_m, 0));
    float fraction = clamp(agl / BAND_MAX, 0.0, 1.0);

    float x = in_pos.x;
    float y_bottom = in_pos.y;
    float y_top_full = in_params.x;
    float fill_weight = in_params.y;

    // Filled top rides between bottom and full-top by the fraction; bottom corners
    // (weight 0) stay put, top corners (weight 1) follow the fill level.
    float filled_top = mix(y_bottom, y_top_full, fraction);
    float y = mix(y_bottom, filled_top, fill_weight);

    float nx = x / screen_size.x * 2.0 - 1.0;
    float ny = y / screen_size.y * 2.0 - 1.0;
    gl_Position = vec4(nx, ny, 0.0, 1.0);

    frag_color = (agl < WARN_AGL) ? vec4(WARN_COLOR, in_color.a) : in_color;
}
