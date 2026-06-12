// Shared terrain INR evaluation: spatial tile lookup + feature-plane bilinear
// interpolation + forward MLP. Included by clipmap_update.comp (ring fill) and
// probe_eval.comp (point probe for AGL, and later FPM-touchdown / glide / TAWS).
// Resolved via `glslc -I shaders/common`; the including shader needs
// `#extension GL_GOOGLE_include_directive : require`.
//
// LOGIC-ONLY include (same style as hdr.glsl): it references the SSBO arrays
// `weights_raw` (the multi-tile Weights buffer) and `dir_grid` (the spatial dir
// grid) by name, so each including shader must declare those two buffers BEFORE
// the #include. Binding numbers / sets may differ per shader; only the instance
// array names must match. `cam_anchor` is passed as a parameter (not read from a
// push constant) so the include stays decoupled from each shader's PC layout.
//
// Multi-tile Weights buffer layout (the `weights_raw` SSBO):
//   Header (5 uint32s): tile_count, features, mlp_hidden_dim, num_outputs, has_catalog
//   Directory (tile_count x 9 uint32s each):
//     origin_x(f32), origin_z(f32), resolution_h, resolution_w,
//     grid_uints, elev_min(f32), elev_max(f32), data_offset, grad_scale(f32)
//   Weights data: concatenated per-tile [grid | dequant | MLP] blocks

// Spatial dir grid dimensions. Mirrors `coords.GRID_*`.
const uint GRID_LON_CELLS = 360u;
const uint GRID_LAT_CELLS = 180u;

// Must match coords.zig: HEIGHT_SCALE, TILE_ARCSEC, WORLD_X_ARCSEC.
const float HEIGHT_SCALE = 60.0 / 1852.0;  // meters -> arcseconds
const float TILE_ARCSEC = 3600.0;
const float WORLD_X_ARCSEC = 360.0 * TILE_ARCSEC;

#define HDR_SIZE 5    // header size (uint32s) in the Weights buffer
#define DIR_STRIDE 9  // per-tile directory stride (uint32s)

uint getTileCount()    { return weights_raw[0]; }
uint getFeatures()     { return weights_raw[1]; }
uint getMlpHiddenDim() { return weights_raw[2]; }
uint getNumOutputs()   { return weights_raw[3]; }
uint getHasCatalog()   { return weights_raw[4]; }

// Find which tile contains world_pos. Returns tile index or -1 if none.
// Cell formula mirrors `coords.worldOriginToGridCell`. Keep them in sync.
//
// Longitude wraps (antimeridian-continuous); latitude does not (pole is
// singular). Wrap is done by folding world_pos.x into [-WORLD_X/2, +WORLD_X/2)
// upfront via `mod()` (well-defined for negatives, unlike GLSL int `%`; see
// `bug_nvidia_wrapping`). Handles arbitrary camera magnitude in one step.
int findTile(vec2 world_pos) {
    if (getTileCount() == 0u) return -1;
    float wrapped_x = mod(world_pos.x + WORLD_X_ARCSEC * 0.5, WORLD_X_ARCSEC) - WORLD_X_ARCSEC * 0.5;
    int cell_x = int(floor(wrapped_x / TILE_ARCSEC));
    int cell_z = int(floor(world_pos.y / TILE_ARCSEC));
    int lon_idx = cell_x + 180;
    int lat_idx = 89 - cell_z;
    if (lon_idx < 0 || lon_idx >= int(GRID_LON_CELLS) ||
        lat_idx < 0 || lat_idx >= int(GRID_LAT_CELLS)) return -1;
    uint v = dir_grid[uint(lat_idx) * GRID_LON_CELLS + uint(lon_idx)];
    if (v == 0u) return -1;
    return int(v - 1u);
}

// Feature plane INR evaluation: bilinear grid interpolation + forward MLP.
// Returns elevation in meters; writes gradients (m/arcsec) and water logit.
// MLP outputs: [elevation, water_logit, dx, dy].
//
// `world_pos_rel` is camera-relative arcsec (small). `cam_anchor` is the camera
// world pos rounded to integer arcsec (the caller's precision anchor). Tile
// origins live in the directory as absolute integer arcsec; we shift them to the
// same camera-relative frame before subtracting so the final `local` is the
// difference of two small numbers, avoiding the catastrophic cancellation that an
// f32 absolute subtract would have at high longitudes.
float evalPlaneINRDirect(vec2 world_pos_rel, vec2 cam_anchor, int tile_idx, out float grad_x, out float grad_z, out float water_logit) {
    uint dir_base = HDR_SIZE + uint(tile_idx) * DIR_STRIDE;

    float tile_origin_x = uintBitsToFloat(weights_raw[dir_base + 0]);
    float tile_origin_z = uintBitsToFloat(weights_raw[dir_base + 1]);
    uint resolution_h   = weights_raw[dir_base + 2];
    uint resolution_w   = weights_raw[dir_base + 3];
    uint grid_uints     = weights_raw[dir_base + 4];
    float elev_min      = uintBitsToFloat(weights_raw[dir_base + 5]);
    float elev_max      = uintBitsToFloat(weights_raw[dir_base + 6]);
    uint data_offset    = weights_raw[dir_base + 7];

    uint plane_features = getFeatures();
    uint mlp_hidden_dim = getMlpHiddenDim();

    // tile_origin is always a multiple of TILE_ARCSEC (3600) and cam_anchor is
    // an integer arcsec; both are well below 2^24 so this subtract is exact.
    vec2 tile_origin_rel = vec2(tile_origin_x - cam_anchor.x, tile_origin_z - cam_anchor.y);
    // Wrap dx to nearest equivalent so a wrap-matched tile sits within one
    // tile of `world_pos_rel`, not WORLD_X away.
    tile_origin_rel.x -= WORLD_X_ARCSEC * floor(tile_origin_rel.x / WORLD_X_ARCSEC + 0.5);
    vec2 local = world_pos_rel - tile_origin_rel;
    vec2 t_uv = clamp(local / TILE_ARCSEC, 0.0, 1.0);
    vec2 uv = t_uv * vec2(float(resolution_w - 1), float(resolution_h - 1));

    int x0 = int(floor(uv.x));
    int y0 = int(floor(uv.y));
    int x1 = min(x0 + 1, int(resolution_w) - 1);
    int y1 = min(y0 + 1, int(resolution_h) - 1);
    float fx = uv.x - float(x0);
    float fy = uv.y - float(y0);

    uint packs_per_cell = plane_features / 4;
    uint i00 = data_offset + (uint(y0) * resolution_w + uint(x0)) * packs_per_cell;
    uint i10 = data_offset + (uint(y0) * resolution_w + uint(x1)) * packs_per_cell;
    uint i01 = data_offset + (uint(y1) * resolution_w + uint(x0)) * packs_per_cell;
    uint i11 = data_offset + (uint(y1) * resolution_w + uint(x1)) * packs_per_cell;

    float w00 = (1.0 - fx) * (1.0 - fy);
    float w10 = fx * (1.0 - fy);
    float w01 = (1.0 - fx) * fy;
    float w11 = fx * fy;

    uint dq_scale_base  = data_offset + grid_uints;
    uint dq_offset_base = dq_scale_base + plane_features;
    uint mlp_base       = data_offset + grid_uints + 2 * plane_features;

    uint num_outputs = getNumOutputs();
    uint w0_offset = mlp_base;
    uint b0_offset = w0_offset + mlp_hidden_dim * plane_features;
    uint w1_offset = b0_offset + mlp_hidden_dim;
    uint b1_offset = w1_offset + mlp_hidden_dim * num_outputs;

    float elev_val  = uintBitsToFloat(weights_raw[b1_offset]);
    float water_val = uintBitsToFloat(weights_raw[b1_offset + 1]);
    float dx_val    = uintBitsToFloat(weights_raw[b1_offset + 2]);
    float dy_val    = uintBitsToFloat(weights_raw[b1_offset + 3]);

    // Dequantized features are loop-invariant w.r.t. the hidden-unit loop;
    // compute once and reuse across all hidden units.
    const uint MAX_PACKS = 16;
    vec4 feat_dq[MAX_PACKS];
    for (uint g = 0; g < packs_per_cell; g++) {
        vec4 f00 = unpackUnorm4x8(weights_raw[i00 + g]);
        vec4 f10 = unpackUnorm4x8(weights_raw[i10 + g]);
        vec4 f01 = unpackUnorm4x8(weights_raw[i01 + g]);
        vec4 f11 = unpackUnorm4x8(weights_raw[i11 + g]);
        vec4 feat_unorm = w00 * f00 + w10 * f10 + w01 * f01 + w11 * f11;

        uint f_base = g * 4;
        vec4 scale = vec4(
            uintBitsToFloat(weights_raw[dq_scale_base + f_base + 0]),
            uintBitsToFloat(weights_raw[dq_scale_base + f_base + 1]),
            uintBitsToFloat(weights_raw[dq_scale_base + f_base + 2]),
            uintBitsToFloat(weights_raw[dq_scale_base + f_base + 3])
        );
        vec4 offset = vec4(
            uintBitsToFloat(weights_raw[dq_offset_base + f_base + 0]),
            uintBitsToFloat(weights_raw[dq_offset_base + f_base + 1]),
            uintBitsToFloat(weights_raw[dq_offset_base + f_base + 2]),
            uintBitsToFloat(weights_raw[dq_offset_base + f_base + 3])
        );
        feat_dq[g] = feat_unorm * scale + offset;
    }

    for (uint h = 0; h < mlp_hidden_dim; h++) {
        float hidden = uintBitsToFloat(weights_raw[b0_offset + h]);
        uint w0_row = w0_offset + h * plane_features;

        for (uint g = 0; g < packs_per_cell; g++) {
            uint f_base = g * 4;
            vec4 w0v = vec4(
                uintBitsToFloat(weights_raw[w0_row + f_base + 0]),
                uintBitsToFloat(weights_raw[w0_row + f_base + 1]),
                uintBitsToFloat(weights_raw[w0_row + f_base + 2]),
                uintBitsToFloat(weights_raw[w0_row + f_base + 3])
            );
            hidden += dot(w0v, feat_dq[g]);
        }

        hidden = max(hidden, 0.0);
        uint w1_h = w1_offset + h * num_outputs;
        elev_val  += uintBitsToFloat(weights_raw[w1_h])     * hidden;
        water_val += uintBitsToFloat(weights_raw[w1_h + 1]) * hidden;
        dx_val    += uintBitsToFloat(weights_raw[w1_h + 2]) * hidden;
        dy_val    += uintBitsToFloat(weights_raw[w1_h + 3]) * hidden;
    }

    water_logit = water_val;

    // MLP outputs normalized gradients (divided by grad_scale during training).
    // Recover meters/arcsecond. DEM pixel = 1 arcsec for GLO-30.
    float trained_grad_scale = uintBitsToFloat(weights_raw[dir_base + 8]);
    float elev_half_range = (elev_max - elev_min) * 0.5;
    grad_x = dx_val * trained_grad_scale * elev_half_range;
    grad_z = dy_val * trained_grad_scale * elev_half_range;

    return (elev_val + 1.0) * 0.5 * (elev_max - elev_min) + elev_min;
}
