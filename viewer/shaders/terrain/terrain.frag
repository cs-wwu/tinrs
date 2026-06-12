#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

layout(location = 0) in float frag_elev_m;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in float frag_isWater;
layout(location = 3) in vec3 frag_view_pos;
#ifdef DEBUG_OVERLAY
layout(location = 4) flat in uint frag_instance;
#endif

#include "scene_ubo.glsl"

layout(constant_id = 0) const uint RING_SIZE = 255;
layout(constant_id = 1) const uint CHUNK_VERTEX_DIM = 32;

#ifdef DEBUG_OVERLAY
// Must mirror clipmap.zig DrawEntry exactly (std430, 48 bytes).
struct DrawEntry {
    vec2 level_origin;
    float grid_spacing;
    float r_inner;
    float r_outer;
    uint level_base_offset;
    uint level_idx;
    uint cull_state;
    uvec2 scroll_offset;
    uvec2 chunk_origin;
};
layout(std430, set = 0, binding = 4) readonly buffer DrawEntries { DrawEntry entries[]; };
#endif

layout(location = 0) out vec4 out_color;

// 8-color qualitative palette for the by_level debug overlay. Order picked
// so adjacent LODs (likely abutting on screen) get unrelated hues.
const vec3 LEVEL_PALETTE[8] = vec3[8](
    vec3(0.85, 0.20, 0.20),  // red
    vec3(0.20, 0.55, 0.85),  // blue
    vec3(0.85, 0.70, 0.15),  // yellow
    vec3(0.30, 0.75, 0.30),  // green
    vec3(0.70, 0.30, 0.80),  // purple
    vec3(0.95, 0.55, 0.20),  // orange
    vec3(0.30, 0.80, 0.80),  // cyan
    vec3(0.85, 0.45, 0.65)   // pink
);

vec3 applyTransferFunction(vec3 color) {
    if (transfer_function == 1u) return linearToPQ(color);
    return color;
}

// SDR-only chroma boost. The desaturated palette is tuned for HDR display,
// where high dynamic range (bright highlights vs. dark terrain) carries
// perceived saturation. Crammed into 100-nit SDR the same colors crowd the
// middle of the gamut and read as washed; ~20% saturation restores the pop.
vec3 sdrChroma(vec3 color) {
    float lum = dot(color, vec3(0.299, 0.587, 0.114));
    return mix(vec3(lum), color, 1.20);
}

vec3 applyFog(vec3 color, vec3 view_pos) {
    float dist = length(view_pos);
    float fog_amount = smoothstep(fog_max_dist * 0.90, fog_max_dist, dist);
    return mix(color, vec3(0.6, 0.7, 0.85), fog_amount);
}

// Photographic terrain palette (see clipmap.zig HYPSO_KNOTS for the colors and
// rationale). Branchless LUT walk: the old 10-branch mix chain cost real ALU on
// fragment-bound GPUs (RPi V3D); this is 2 UBO loads + 1 mix.
vec3 hypsometric(float elev_m) {
    float t = clamp((elev_m + 100.0) * 0.01, 0.0, float(HYPSO_LUT_LEN - 1u));
    uint i = min(uint(t), HYPSO_LUT_LEN - 2u);
    return mix(hypso_lut[i].rgb, hypso_lut[i + 1u].rgb, t - float(i));
}

// TAWS terrain-hazard overlay (Garmin SVT-style, forward-view convention):
// TINT the terrain toward a hazard hue by vertical clearance below the aircraft
// rather than replacing it, so the natural hypsometric color + 3D shape still
// read through the wash (no clearance-green, so green keeps meaning "lowland").
// The hue ramps red (solid at/below TAWS_RED_M) -> yellow (out to TAWS_YELLOW_M),
// giving a continuous severity gradient; the tint then fades out over
// TAWS_EDGE_M past the outer boundary so safe terrain is untouched. Clearance is
// negative for terrain above the aircraft, which pins to solid red. Transitions
// are smoothstepped in-shader on purpose: MSAA anti-aliases geometry edges, not
// a mid-triangle shading contour, so the band edge would shimmer without this.
// Thresholds are tuned for low-and-slow GA/helo flight.
// TODO: expose the TAWS thresholds + tint strength as settings sliders when the menu grows them.
const float TAWS_RED_M    =  30.0; // ~100 ft: solid red at/below this clearance
const float TAWS_YELLOW_M = 150.0; // ~500 ft: outer edge of the caution zone
const float TAWS_EDGE_M   =  20.0; // soft fade width past the outer (yellow) boundary
const float TAWS_TINT     =   0.7; // hazard opacity over the natural terrain (0..1)
const vec3  TAWS_RED      = vec3(0.85, 0.10, 0.10);
const vec3  TAWS_YELLOW   = vec3(0.90, 0.75, 0.10);

// Blend a clearance-driven hazard hue over the natural terrain color. `shade`
// (lighting * slope darkening) is applied to the hue too, so tinted terrain
// keeps its 3D shape instead of reading as a flat fill.
vec3 applyTaws(vec3 natural, float elev_m, float shade) {
    float clearance = aircraft_msl_m - elev_m;
    float sev = smoothstep(TAWS_RED_M, TAWS_YELLOW_M, clearance); // 0 = red, 1 = yellow
    vec3 hue = mix(TAWS_RED, TAWS_YELLOW, sev);
    float amt = (1.0 - smoothstep(TAWS_YELLOW_M, TAWS_YELLOW_M + TAWS_EDGE_M, clearance)) * TAWS_TINT;
    return mix(natural, hue * shade, amt);
}

void main() {
#ifdef DEBUG_OVERLAY
    // Uniform branch: driver resolves once per draw, no per-fragment cost.
    if (debug_overlay != 0u) {
        DrawEntry e = entries[frag_instance];
        vec3 base;
        if (debug_overlay == 3u) {
            // by_cull_state: bit 1 = radial, bit 2 = frustum, bit 4 = ocean.
            // Ocean checked first since it's the most distinctive category.
            if ((e.cull_state & 4u) != 0u) {
                base = vec3(0.20, 0.40, 0.85); // blue (ocean)
            } else if (e.cull_state == 0u) {
                base = vec3(0.18, 0.18, 0.20); // dim gray (kept)
            } else if (e.cull_state == 1u) {
                base = vec3(0.85, 0.55, 0.20); // orange (radial only)
            } else if (e.cull_state == 2u) {
                base = vec3(0.85, 0.20, 0.20); // red (frustum only)
            } else {
                base = vec3(0.85, 0.30, 0.85); // magenta (both)
            }
        } else {
            uint palette_idx;
            if (debug_overlay == 1u) {
                palette_idx = e.level_idx & 7u;
            } else {
                // by_chunk: hash chunk origin + level into a 3-bit slot. The
                // 3 / 5 / 7 multipliers are odd primes so neighbors rarely collide.
                uint cells = CHUNK_VERTEX_DIM - 1u;
                uint cx = e.chunk_origin.x / cells;
                uint cy = e.chunk_origin.y / cells;
                palette_idx = (cx * 3u + cy * 5u + e.level_idx * 7u) & 7u;
            }
            base = LEVEL_PALETTE[palette_idx];
        }
        vec3 n_dbg = normalize(frag_normal);
        vec3 sun_dbg = sun_dir.xyz;
        // Half-Lambert so the overlay reads with terrain shape, not flat.
        float lit = 0.4 + 0.6 * (max(dot(n_dbg, sun_dbg), 0.0) * 0.5 + 0.5);
        out_color = vec4(applyTransferFunction(base * lit), 1.0);
        return;
    }
#endif

    vec3 n = normalize(frag_normal);
    vec3 sun_d = sun_dir.xyz;
    bool hdr = (transfer_function != 0u);

    // Anti-alias water mask across ~1 screen pixel using fwidth (a
    // fixed-range smoothstep would bleed across many pixels at close range).
    float w = fwidth(frag_isWater) * 0.5;
    float water_mask = smoothstep(0.5 - w, 0.5 + w, frag_isWater);

    // no_effects: simple lighting (benchmark-stable, 1 branch + 1 mix per
    // pixel) so benchmarks measure the rendering ceiling, not shading.
    if (no_effects != 0u) {
        float diffuse = max(dot(n, sun_d), 0.0);
        vec3 land_low  = vec3(0.15, 0.30, 0.13);
        vec3 land_mid  = vec3(0.45, 0.40, 0.30);
        vec3 land_high = vec3(0.85, 0.85, 0.85);
        float shade = 0.25 + 0.75 * diffuse;
        vec3 land = (frag_elev_m < 2500.0)
            ? mix(land_low, land_mid, frag_elev_m / 2500.0)
            : mix(land_mid, land_high, clamp((frag_elev_m - 2500.0) / 2500.0, 0.0, 1.0));
        land *= shade;
        // TAWS is a hazard overlay, not cosmetic shading, so it stays independent
        // of the Effects toggle (this no_effects path). Uniform branch: free when
        // off, which is always the case in benchmarks (--no-effects, TAWS off).
        if (taws != 0u) land = applyTaws(land, frag_elev_m, shade);
        vec3 water = vec3(0.04, 0.10, 0.20) * (0.4 + 0.6 * diffuse);
        vec3 color = mix(land, water, water_mask);
        out_color = vec4(applyTransferFunction(color), 1.0);
        return;
    }

    // Half-Lambert wrap: extends diffuse slightly past terminator.
    float NdotL = dot(n, sun_d);
    float diffuse = NdotL * 0.5 + 0.5;
    diffuse *= diffuse; // square for softer falloff

    // Hemisphere ambient: sky above, warm ground bounce below.
    float sky_amount = n.y * 0.5 + 0.5;
    float ambient = mix(0.06, 0.25, sky_amount);
    float lighting = ambient + (1.0 - ambient) * diffuse;

    // Slope darkening: steep terrain (small ny) gets darker.
    float steepness = 1.0 - n.y;
    float slope_darken = 1.0 - steepness * 0.35;

    // Hazard overlay (uniform branch: resolved once per draw, no per-fragment
    // cost) recolors land only; water cells keep the water shade after the mix.
    float shade = lighting * slope_darken;
    vec3 land = hypsometric(frag_elev_m) * shade;
    if (taws != 0u) land = applyTaws(land, frag_elev_m, shade);

    vec3 color = land;
    // Water shading (Fresnel + HDR glint) only where the mask touches this
    // fragment. Water is spatially coherent on screen, so most land-only quads
    // skip the whole block (normalize + pow + mixes). Branching here is legal
    // because water_mask's fwidth() was computed above, in uniform control
    // flow; no derivatives are taken inside this non-uniform branch.
    if (water_mask > 0.0) {
        // Dark base with sky reflection at glancing angles. Shade the water
        // SURFACE as flat (water_n = up), not with the terrain gradient normal
        // `n`. At a steep shoreline that gradient normal is tilted (the AA
        // land/water normal blend, plus INR gradients leaking onto water
        // cells), which spiked both the Fresnel sky-mix (visible in SDR) and
        // the sun glint (HDR) into a bright rim. Our water model is flat, so
        // its reflectance must depend on view/sun geometry over a level
        // surface, not on the adjacent hillside slope.
        vec3 water = vec3(0.04, 0.10, 0.20) * lighting;
        vec3 water_n = vec3(0.0, 1.0, 0.0);
        vec3 view_dir = normalize(-frag_view_pos);
        float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(view_dir, water_n), 0.0), 5.0);
        water = mix(water, vec3(0.35, 0.45, 0.65), fresnel * 0.25);
        // HDR sun glint via Blinn-Phong half-vector specular. SDR omitted: the
        // highlight clips to white and reads as washout there.
        if (hdr) {
            vec3 H = normalize(sun_d + view_dir);
            float spec = pow(max(dot(water_n, H), 0.0), 80.0);
            water += vec3(1.0, 0.95, 0.85) * spec * fresnel * 3.0;
        }
        color = mix(land, water, water_mask);
    }

    color = applyFog(color, frag_view_pos);
    if (!hdr) color = sdrChroma(color);

    out_color = vec4(applyTransferFunction(color), 1.0);
}
