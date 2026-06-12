#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

layout(location = 0) in vec2 frag_ndc;

#include "scene_ubo.glsl"

layout(location = 0) out vec4 out_color;

vec3 applyTransferFunction(vec3 color) {
    if (transfer_function == 1u) return linearToPQ(color);
    return color;
}

void main() {
    if (no_effects != 0u) {
        out_color = vec4(applyTransferFunction(vec3(0.01, 0.01, 0.05)), 1.0);
        return;
    }

    // Reconstruct world-space view ray from NDC.
    // proj[1][1] = -1/tan(fov/2) (negative for Vulkan Y-flip).
    float tan_half_fov = -1.0 / proj[1][1];
    float aspect = -proj[1][1] / proj[0][0];

    vec3 ray_view = normalize(vec3(
        frag_ndc.x * aspect * tan_half_fov,
        -frag_ndc.y * tan_half_fov,
        -1.0
    ));

    // View matrix is rotation-only; its transpose is its inverse.
    mat3 inv_rot = transpose(mat3(view));
    vec3 ray = normalize(inv_rot * ray_view);

    vec3 sun_d = normalize(sun_dir.xyz);
    float up = ray.y;

    // Rayleigh-like gradient: deep blue zenith to pale horizon
    vec3 zenith = vec3(0.15, 0.3, 0.65);
    vec3 horizon = vec3(0.6, 0.7, 0.85);
    float horizon_blend = 1.0 - pow(max(up, 0.0), 0.6);
    vec3 sky = mix(zenith, horizon, horizon_blend);

    // Below horizon: hold at horizon color (matches fog edge fade exactly)
    if (up < 0.0) {
        sky = horizon;
    }

    // Sun disc (~0.5 degree diameter, matching the real sun's angular size).
    // HDR pushes peak past 1.0 linear so it reads as an intense point source
    // (~5000 nits at 25.0 linear); SDR clamps to white.
    float sun_cos = dot(ray, sun_d);
    float sun_intensity = (transfer_function != 0u) ? 25.0 : 5.0;
    sky += vec3(1.0, 0.95, 0.85) * smoothstep(0.99995, 0.99999, sun_cos) * sun_intensity;

    // Mie glow: wide soft + tight bright halo
    sky += vec3(1.0, 0.85, 0.6) * pow(max(sun_cos, 0.0), 4.0) * 0.15;
    sky += vec3(1.0, 0.9, 0.7) * pow(max(sun_cos, 0.0), 32.0) * 0.3;

    out_color = vec4(applyTransferFunction(sky), 1.0);
}
