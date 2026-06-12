#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

// Solid fill for the AGL gauge bar. Color (in-band vs warning red) is decided in
// gauge.vert; here we just encode it for the swapchain's transfer function, like
// the other UI fragment shaders.

layout(location = 0) in vec4 frag_color;
layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function;
    float ui_paper_white;
};

void main() {
    out_color = vec4(encodeUi(frag_color.rgb, transfer_function, ui_paper_white), frag_color.a);
}
