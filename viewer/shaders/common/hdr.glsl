// Shared HDR transfer-function helpers, included by every shader that writes the
// swapchain directly. Resolved via `glslc -I shaders/common`; the including shader
// needs `#extension GL_GOOGLE_include_directive : require`. Keeping the PQ curve
// and the UI reference-white scale in one place means a transfer-function change
// stays consistent across sky / terrain / UI instead of being hand-synced.

// Linear -> PQ (ST.2084). Input is scene-referred linear where 1.0 = SDR diffuse
// reference white, scaled to BT.2408 reference (203 nits). Linear values above 1.0
// (specular highlights, sun, fresnel) extend up to ~49 before the PQ ceiling
// (10,000 nits), giving real HDR headroom.
vec3 linearToPQ(vec3 c) {
    c = clamp(c * 0.0203, vec3(0.0), vec3(1.0));
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 Ym1 = pow(c, vec3(m1));
    return pow((c1 + c2 * Ym1) / (1.0 + c3 * Ym1), vec3(m2));
}

// Encode an SDR-referred UI color (1.0 = full) to the swapchain's transfer function
// at `paper_white` nits, so the UI reads at a deliberate brightness in HDR. PQ:
// scale into the 203-nit linear reference, then PQ-encode. scRGB: linear where
// 1.0 = 80 nits. sRGB: passthrough (the _srgb format hardware-encodes).
// `tf`: 0 = sRGB (hardware), 1 = PQ/HDR10, 2 = scRGB linear.
vec3 encodeUi(vec3 c, uint tf, float paper_white) {
    if (tf == 1u) return linearToPQ(c * (paper_white / 203.0));
    if (tf == 2u) return c * (paper_white / 80.0);
    return c;
}
