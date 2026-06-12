#version 450

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // fragment-stage; declared here so the block matches
    float ui_paper_white;   // fragment-stage
};

layout(location = 0) in vec2 in_pos;       // pixel coordinates
layout(location = 1) in vec2 in_uv;        // [0,1] within character cell
layout(location = 2) in vec4 in_color;     // glyph color (RGBA)
layout(location = 3) in uint in_char_idx;  // ASCII code

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;
layout(location = 2) flat out uint frag_char_idx;

void main() {
    // Pixel coords to NDC: (0,0) top-left -> (-1,-1), (w,h) bottom-right -> (1,1)
    float x = in_pos.x / screen_size.x * 2.0 - 1.0;
    float y = in_pos.y / screen_size.y * 2.0 - 1.0;

    gl_Position = vec4(x, y, 0.0, 1.0);
    frag_uv = in_uv;
    frag_color = in_color;
    frag_char_idx = in_char_idx;
}
