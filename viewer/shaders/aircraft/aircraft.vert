#version 450

// Placeholder aircraft mesh: a tetrahedron at the aircraft pose, drawn for
// Phase B verification that the camera and aircraft are decoupled. Future
// phases swap this for a real plane model.

layout(push_constant) uniform PC {
    // Camera-relative MVP. CPU computes proj * view_rot * translate(ac - cam) *
    // rotate(ac_orient) * scale(mesh_scale) in f64 and truncates to f32.
    mat4 mvp;
} pc;

const vec3 VERTS[12] = vec3[12](
    // Flat delta wing: wide wingspan (1.4), thin profile (0.25 tall),
    // swept tail fin. Nose extends forward (-1.2) for length.
    // Face 0 (top-right): nose, tail-fin, right-wing
    vec3(0.0, 0.0, -1.2), vec3(0.0, 0.25, 0.8), vec3(1.4, 0.0, 1.0),
    // Face 1 (top-left): nose, left-wing, tail-fin
    vec3(0.0, 0.0, -1.2), vec3(-1.4, 0.0, 1.0), vec3(0.0, 0.25, 0.8),
    // Face 2 (belly): nose, right-wing, left-wing
    vec3(0.0, 0.0, -1.2), vec3(1.4, 0.0, 1.0), vec3(-1.4, 0.0, 1.0),
    // Face 3 (back): right-wing, tail-fin, left-wing
    vec3(1.4, 0.0, 1.0), vec3(0.0, 0.25, 0.8), vec3(-1.4, 0.0, 1.0)
);

const vec3 FACE_COLORS[4] = vec3[4](
    vec3(1.0, 0.25, 0.25),  // right: red
    vec3(0.25, 1.0, 0.25),  // left: green
    vec3(0.25, 0.4, 1.0),   // belly: blue
    vec3(1.0, 1.0, 0.3)     // back: yellow
);

layout(location = 0) out vec3 v_color;

void main() {
    vec3 pos = VERTS[gl_VertexIndex];
    v_color = FACE_COLORS[gl_VertexIndex / 3];
    gl_Position = pc.mvp * vec4(pos, 1.0);
}
