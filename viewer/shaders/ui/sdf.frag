#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

layout(location = 0) flat in vec4 ab;
layout(location = 1) flat in vec2 radius_border;
layout(location = 2) flat in float kind;
layout(location = 3) flat in vec4 frag_color;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;       // vertex-stage; declared so the block matches
    uint transfer_function; // 0 = sRGB (hardware), 1 = PQ/HDR10, 2 = scRGB linear
    float ui_paper_white;   // UI paper-white target, nits
};

// Connected-stroke path points (screen px), uploaded per frame. A polyline element
// (kind 3) carries an (offset, count) into this in `ab.xy`; the fragment takes the
// min distance over the path, giving joints single coverage (no double-blend).
// std430: tight 8-byte vec2 array stride, matching the CPU-side [2]f32 upload.
// Without it the block uses the implementation-defined `shared` layout, which may
// pad vec2 to a 16-byte stride (std140-style) and mismatch on some drivers (RPi).
layout(std430, set = 0, binding = 0) readonly buffer PathPoints {
    vec2 path_points[];
};

// Unsigned distance from p to the segment a->b.
float segDist(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    // gl_FragCoord is window-space pixels, top-left origin in Vulkan, matching the
    // pixel coords the shape params were built in. No extra position varying needed.
    vec2 p = gl_FragCoord.xy;
    float radius = radius_border.x;
    float border = radius_border.y;

    float d;
    if (kind < 0.5) {
        // Capsule (round caps): distance to segment A->B, minus radius. a == b
        // degenerates to a point distance, i.e. a circle of `radius`.
        d = segDist(p, ab.xy, ab.zw) - radius;
    } else if (kind < 1.5) {
        // Rounded box: ab.xy = center, ab.zw = half-extent, radius = corner radius
        // (clamped <= min(half-extent) on the CPU so the field stays valid).
        vec2 q = abs(p - ab.xy) - (ab.zw - radius);
        d = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    } else if (kind < 2.5) {
        // Segment (butt caps): an oriented box from A to B with half-thickness
        // `radius`, matching the old quad-based line(). The capsule covers circles
        // (a == b), so here len > 0.
        vec2 a = ab.xy;
        vec2 b = ab.zw;
        vec2 ba = b - a;
        float len = max(length(ba), 1e-6);
        vec2 u = ba / len;
        vec2 nrm = vec2(-u.y, u.x);
        vec2 pm = p - 0.5 * (a + b);
        vec2 q = vec2(abs(dot(pm, u)) - 0.5 * len, abs(dot(pm, nrm)) - radius);
        d = min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0)));
    } else {
        // Polyline: min distance over the path's segments (ab.x = point offset,
        // ab.y = count into path_points), minus radius. Single coverage at the
        // joints, so connected translucent strokes do not double-blend.
        int off = int(ab.x);
        int n = int(ab.y);
        d = 1e9;
        for (int i = 0; i + 1 < n; i++) {
            d = min(d, segDist(p, path_points[off + i], path_points[off + i + 1]));
        }
        d -= radius;
    }

    // border > 0 turns the filled field into a stroke of half-width `border`.
    if (border > 0.0) d = abs(d) - border;

    // Constant ~1px AA in window-space pixels: d is already a Euclidean distance
    // field (|grad d| ~= 1), so the AA band is a fixed pixel width and no screen-space
    // derivative is needed. Avoids VideoCore VII's unreliable fwidth() on the 2x2
    // quads that straddle the bounding quad's triangle edges, which left a faint
    // box+diagonal around SDF strokes on the RPi 5. Assumes screen_size matches the
    // framebuffer pixel dims (it does; the vertex stage maps px coords straight to NDC).
    const float aa = 1.0;
    float cov = clamp(0.5 - d / aa, 0.0, 1.0);
    if (cov <= 0.0) discard;

    out_color = vec4(encodeUi(frag_color.rgb, transfer_function, ui_paper_white), frag_color.a * cov);
}
