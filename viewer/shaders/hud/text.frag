#version 450
#extension GL_GOOGLE_include_directive : require
#include "hdr.glsl"

// Font bitmap: 128 chars * 8 bytes = 1024 bytes, packed as 256 uints
layout(set = 0, binding = 0) readonly buffer FontData {
    uint font_bitmap[];
};

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;
layout(location = 2) flat in uint frag_char_idx;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    uint transfer_function; // 0 = sRGB (hardware), 1 = PQ/HDR10, 2 = scRGB linear
    float ui_paper_white;   // HUD paper-white target, nits
};

void main() {
    // Pixel position within the 8x8 character cell
    int px = clamp(int(frag_uv.x * 8.0), 0, 7);
    int py = clamp(int(frag_uv.y * 8.0), 0, 7);

    // Each character is 8 bytes (1 byte per row).
    // font_bitmap[] is packed as uint32 (4 rows per uint, little-endian).
    uint byte_offset = frag_char_idx * 8u + uint(py);
    uint word_idx = byte_offset / 4u;
    uint byte_in_word = byte_offset % 4u;
    uint row_bits = (font_bitmap[word_idx] >> (byte_in_word * 8u)) & 0xFFu;

    // Check if pixel is set (MSB = leftmost pixel)
    bool lit = ((row_bits >> uint(7 - px)) & 1u) != 0u;

    // Lit pixels take the glyph's vertex color; unlit pixels are transparent.
    // Backgrounds come from the shapes pass (panels), not per-glyph fill.
    if (!lit) discard;
    out_color = vec4(encodeUi(frag_color.rgb, transfer_function, ui_paper_white), frag_color.a);
}
