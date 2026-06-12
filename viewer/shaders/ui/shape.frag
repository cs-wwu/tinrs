#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

layout(location = 0) in vec4 frag_color;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // 0 = sRGB (hardware), 1 = PQ/HDR10, 2 = scRGB linear
    float ui_paper_white;   // HUD paper-white target, nits
};

void main() {
    out_color = vec4(encodeUi(frag_color.rgb, transfer_function, ui_paper_white), frag_color.a);
}
