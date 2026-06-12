#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // fragment-stage; declared here so the block matches
    float ui_paper_white;   // fragment-stage
};

layout(location = 0) in vec2 in_pos;           // bounding-quad corner, pixel coords
layout(location = 1) in vec4 in_ab;            // capsule A/B, or box center + half-extent
layout(location = 2) in vec2 in_radius_border; // x = radius, y = outline half-width (0 = filled)
layout(location = 3) in float in_kind;         // 0 = capsule, 1 = rounded box
layout(location = 4) in vec4 in_color;

// Shape params are constant per element: flat so they are not interpolated. The
// fragment shader distance-tests gl_FragCoord against them.
layout(location = 0) flat out vec4 ab;
layout(location = 1) flat out vec2 radius_border;
layout(location = 2) flat out float kind;
layout(location = 3) flat out vec4 frag_color;

void main() {
    // Pixel coords to NDC: (0,0) top-left -> (-1,-1), (w,h) bottom-right -> (1,1).
    float x = in_pos.x / screen_size.x * 2.0 - 1.0;
    float y = in_pos.y / screen_size.y * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);

    ab = in_ab;
    radius_border = in_radius_border;
    kind = in_kind;
    frag_color = in_color;
}
