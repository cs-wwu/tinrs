#version 450

layout(location = 0) out vec2 frag_ndc;

void main() {
    // Fullscreen triangle from vertex index (no vertex buffer).
    // Vertex 0: (-1, -1), Vertex 1: (3, -1), Vertex 2: (-1, 3)
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;
    frag_ndc = pos;
    gl_Position = vec4(pos, 0.0, 1.0); // reverse-Z: depth = z/w = 0.0 (far plane)
}
