#version 450
#extension GL_GOOGLE_include_directive : require

// Clipmap terrain vertex shader.
//
// Reads cached elevation from a toroidal ring buffer (filled by
// clipmap_update.comp). Each entry is vec4(z_fine, z_delta, nx, nz).
//
// Camera-relative rendering: level_origin is pre-shifted by (origin - camera)
// on the CPU in f64; view matrix is rotation-only. All values in the shader
// are small, avoiding f32 catastrophic cancellation.

layout(std430, set = 0, binding = 0) readonly buffer RingBuffer { vec4 data[]; };

#include "scene_ubo.glsl"

// Specialization constants: let the compiler strength-reduce the modulo /
// divide ops below (CHUNK_VERTEX_DIM is power-of-two in current configs).
layout(constant_id = 0) const uint RING_SIZE = 255;
layout(constant_id = 1) const uint CHUNK_VERTEX_DIM = 32;

// Must mirror clipmap.zig DrawEntry exactly (std430, 48 bytes).
struct DrawEntry {
    vec2 level_origin;       // camera-relative (x, z)
    float grid_spacing;
    float r_inner;
    float r_outer;
    uint level_base_offset;
    uint level_idx;
    uint cull_state;         // bitmask: 1=radial, 2=frustum, 4=ocean
    uvec2 scroll_offset;     // pre-mod'd to [0, ring_size)
    uvec2 chunk_origin;      // chunk's level-relative (col, row)
};
layout(std430, set = 0, binding = 4) readonly buffer DrawEntries { DrawEntry entries[]; };

// Per-vertex radial clip distances make the rendered boundary the exact
// arcsec circle at r_inner / r_outer. Triangles spanning the boundary are
// rasterizer-clipped (no chunk-sized gaps from grid-vs-circle mismatch).
out float gl_ClipDistance[2];

layout(location = 0) out float frag_elev_m;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out float frag_isWater;
layout(location = 3) out vec3 frag_view_pos;
#ifdef DEBUG_OVERLAY
layout(location = 4) flat out uint frag_instance;
#endif

const uint CULL_OCEAN = 4u;

void main() {
    DrawEntry e = entries[gl_InstanceIndex];
#ifdef DEBUG_OVERLAY
    frag_instance = uint(gl_InstanceIndex);
#endif

    uint local_col = gl_VertexIndex % CHUNK_VERTEX_DIM;
    uint local_row = gl_VertexIndex / CHUNK_VERTEX_DIM;
    uint col = local_col + e.chunk_origin.x;
    uint row = local_row + e.chunk_origin.y;

    float half_extent = float(RING_SIZE) / 2.0;
    float x = e.level_origin.x + (float(col) - half_extent) * e.grid_spacing;
    float z = e.level_origin.y + (float(row) - half_extent) * e.grid_spacing;

    // Radial clip in arcsec, before cos(lat) scaling.
    // TODO: gate the inner clip on r_inner > 0. At min_level r_inner is 0
    // (always positive distance, never trims), but the GPU still computes
    // sqrt + writes + interpolates + per-primitive plane test. A separate
    // pipeline variant with gl_ClipDistance[1] (outer only) for min_level
    // would skip that work. Picks up a few % at ring < 256 no-effects on
    // discrete GPUs. Pair with a specialization constant on array size.
    // Vulkan keeps fragments where gl_ClipDistance >= 0.
    float arcsec_dist = length(vec2(x, z));
    // Inner skirt: vertices inside r_inner render but get pulled downward.
    // From the viewer's perspective the skirt faces toward the camera,
    // covering inter-level cracks. The finer level in front wins depth.
    float skirt_width = e.grid_spacing * 3.0;
    float undershoot = max(e.r_inner - arcsec_dist, 0.0);
    gl_ClipDistance[0] = arcsec_dist - (e.r_inner - skirt_width);
    gl_ClipDistance[1] = e.r_outer - arcsec_dist;

    // 1 arcsec of longitude = cos(lat) * 1 arcsec of latitude in ground.
    const float PI = 3.14159265358979;
    float global_z = z + cam_z;
    float lat_rad = -global_z / 3600.0 * (PI / 180.0);
    float cos_lat = max(cos(lat_rad), 0.01);
    x *= cos_lat;

    // R_EARTH = 6,371,000m * (60/1852) = 206,404 arcseconds.
    const float R_EARTH = 206404.0;

    // Ocean chunks: skip ring buffer read, flat y=0 water surface.
    if ((e.cull_state & CULL_OCEAN) != 0u) {
        float y = -cam_elev - (x * x + z * z) / (2.0 * R_EARTH);
        y -= undershoot * 2.0;
        frag_normal = vec3(0.0, 1.0, 0.0);
        frag_isWater = 1.0;
        frag_elev_m = 0.0;
        frag_view_pos = vec3(x, y, z);
        gl_Position = proj_view * vec4(x, y, z, 1.0);
        return;
    }

    // Terrain path: ring buffer SSBO read + geomorphing + normal decode.
    uint buf_row = (row + e.scroll_offset.y) % RING_SIZE;
    uint buf_col = (col + e.scroll_offset.x) % RING_SIZE;
    vec4 d = data[e.level_base_offset + buf_row * RING_SIZE + buf_col];

    // TODO: geomorphing. The blend was `d.x + alpha * d.y` with alpha ramping
    // over the outer ~10% of the ring, but clipmap_update.comp never writes
    // z_delta (d.y is always 0.0; see its TODO), so the alpha math was dead
    // per-vertex work. Restore the blend when compute fills z_delta.
    float z_fine = d.x;
    float y = z_fine - cam_elev;
    y -= (x * x + z * z) / (2.0 * R_EARTH);
    y -= undershoot * 2.0;

    // Water flag encoded as a +4.0 offset on normal_z in the ring buffer (see
    // clipmap_update.comp). normal.z is in [-1,1]: land stays in [-1,1], water
    // lands in [3,5], so a 2.0 threshold separates them for the full range. (A
    // +2.0 offset / 1.5 threshold aliased: water with normal.z < -0.5 fell below
    // 1.5, misdecoded as land with nz>1 -> ny=0 garbage normal -> a bright
    // terrain diamond at steep shorelines.)
    float raw_nz = d.w;
    float isWater = step(2.0, raw_nz);
    float nz = raw_nz - isWater * 4.0;
    float ny = sqrt(max(1.0 - d.z * d.z - nz * nz, 0.0));
    frag_normal = vec3(d.z, ny, nz);
    frag_isWater = isWater;

    frag_view_pos = vec3(x, y, z);
    gl_Position = proj_view * vec4(x, y, z, 1.0);

    const float HEIGHT_SCALE = 60.0 / 1852.0;
    frag_elev_m = z_fine / HEIGHT_SCALE;
}
