// Scene-wide UBO shared by the terrain and sky pipelines (set 0, binding 2 of
// the clipmap descriptor layout). Single source of truth for the block layout:
// every shader includes this file instead of redeclaring a prefix, so member
// offsets cannot drift between shaders. Must mirror clipmap.zig's SceneUBO
// (std140; the Zig side asserts the total size at comptime).

// Hypsometric palette LUT: [-100, 8500] m at 100m spacing. Table contents and
// palette rationale live in clipmap.zig (HYPSO_KNOTS / HYPSO_LUT).
const uint HYPSO_LUT_LEN = 87u;

layout(set = 0, binding = 2) uniform SceneUBO {
    mat4 view;            // rotation-only; sky reconstructs view rays from it
    mat4 proj;            // sky reads fov/aspect from [0][0] / [1][1]
    vec4 sun_dir;
    float fog_max_dist;
    uint no_effects;
    uint transfer_function; // 0 = sRGB (hardware), 1 = PQ, 2 = scRGB linear
    uint debug_overlay;   // 0=off, 1=by_level, 2=by_chunk, 3=by_cull_state (matches debug.ColorOverlay enum)
    float cam_elev;       // camera Y in arcsec
    float cam_z;          // camera global Z in arcsec
    float aircraft_msl_m; // aircraft altitude MSL (meters), for TAWS clearance
    uint taws;            // 0 = off, 1 = hazard overlay (red/yellow by clearance)
    mat4 proj_view;       // proj * view, premultiplied on CPU (terrain vert)
    vec4 hypso_lut[HYPSO_LUT_LEN]; // constant; written once at init, not per frame
};
