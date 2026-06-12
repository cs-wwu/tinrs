"""
Export feature plane MLP weights to raw binary for Vulkan compute shader.

Grid features are quantized to uint8 (per-channel min/max) for 4x compression.
MLP weights remain float32 (they're tiny).

Outputs per tile (to assets/planes/<tile>/):
    weights.bin   - [grid_uint8 (packed as uint32), dequant_scale (F floats),
                     dequant_offset (F floats), mlp_W0, mlp_b0, mlp_W1, mlp_b1]
    meta.json     - model metadata (resolution, features, mlp dims, elev_min/max)
    reference.bin - [num_points: u32, coords: N*2 f32, elevations: N f32]

Usage:
    # Single model (auto-detects tile name, exports to assets/planes/n47w122/)
    python export_weights_plane.py --model output/n47w122_plane_256x8_300k.pt

    # Batch export all .pt files in a directory
    python export_weights_plane.py --batch output/

    # Override output location
    python export_weights_plane.py --model output/n47w122_plane_256x8_300k.pt --output-dir /tmp/test
"""

import argparse
import json
import re
import struct
from pathlib import Path

import numpy as np
import torch
from models import FeaturePlaneMLP

# Project root: training/ -> ..
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLANES_DIR = PROJECT_ROOT / "assets" / "planes"

TILE_NAME_RE = re.compile(r"^([nsNS]\d{2}[ewEW]\d{3})")


def parse_tile_name(filename: str) -> str | None:
    """Extract tile name (e.g. 'n47w122') from a checkpoint filename."""
    m = TILE_NAME_RE.match(filename)
    return m.group(1).lower() if m else None


def export_one(model_path: Path, output_dir: Path, num_ref: int, device: str) -> bool:
    """Export a single .pt checkpoint. Returns True on success."""
    tile_name = parse_tile_name(model_path.stem)
    if tile_name is None:
        print(f"  SKIP: cannot parse tile name from '{model_path.name}'")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    sd = checkpoint["state_dict"]

    resolution = checkpoint["resolution"]
    resolution_w = checkpoint.get("resolution_w", resolution)
    features = checkpoint["features"]
    mlp_hidden_dim = checkpoint["mlp_hidden_dim"]
    elev_min = checkpoint["elev_min_m"]
    elev_max = checkpoint["elev_max_m"]
    elev_range = elev_max - elev_min

    print(f"  Plane {resolution}x{resolution_w}x{features}, MLP 1x{mlp_hidden_dim}")
    print(f"  Elevation: [{elev_min:.1f}, {elev_max:.1f}]m (range {elev_range:.1f}m)")
    if elev_range < 1e-6:
        print(f"  SKIP: constant elevation (ocean tile)")
        return False
    # Reconstruct model for reference evaluation
    model = FeaturePlaneMLP(
        resolution=resolution,
        features=features,
        mlp_hidden_dim=mlp_hidden_dim,
        resolution_w=resolution_w,
    )
    model.load_state_dict(sd)
    model.eval()

    # -- Export weights as flat binary (INT8 grid + float32 MLP) --
    assert features % 4 == 0, f"Features ({features}) must be divisible by 4 for uint32 packing"

    # Grid: NCHW (1, F, H, W) -> HWC (H, W, F) for coalesced GPU access
    grid = sd["grid"].squeeze(0)  # (F, H, W)
    grid_hwc = grid.permute(1, 2, 0).contiguous().numpy().astype(np.float32)  # (H, W, F)
    assert grid_hwc.shape == (resolution, resolution_w, features)

    # Per-channel quantization to uint8
    grid_flat = grid_hwc.reshape(-1, features)
    ch_min = grid_flat.min(axis=0)
    ch_max = grid_flat.max(axis=0)
    ch_range = ch_max - ch_min
    ch_range[ch_range == 0] = 1.0

    grid_norm = (grid_flat - ch_min) / ch_range
    grid_u8 = np.clip(np.round(grid_norm * 255.0), 0, 255).astype(np.uint8)
    grid_u8 = grid_u8.reshape(resolution, resolution_w, features)

    # Pack 4 uint8 into each uint32 (little-endian, matches GLSL unpackUnorm4x8)
    grid_packed = grid_u8.view(np.uint32)
    assert grid_packed.shape == (resolution, resolution_w, features // 4)
    grid_uints = grid_packed.size

    dequant_scale = ch_range.astype(np.float32)
    dequant_offset = ch_min.astype(np.float32)

    # Quantization error stats
    grid_dequant = grid_u8.astype(np.float32) / 255.0 * ch_range + ch_min
    quant_err = np.abs(grid_flat - grid_dequant.reshape(-1, features))
    print(f"  Grid: {grid_uints} uint32s ({grid_uints * 4} bytes, was {grid_hwc.size * 4} bytes)")
    print(f"  Quantization error: mean={quant_err.mean():.6f}, max={quant_err.max():.6f}")

    # MLP weights: mlp.0 (Linear hidden), mlp.2 (Linear output)
    mlp_parts = []
    mlp_floats = 0
    for li, label in [(0, "Hidden"), (2, "Output")]:
        w = sd[f"mlp.{li}.weight"].numpy().astype(np.float32)
        b = sd[f"mlp.{li}.bias"].numpy().astype(np.float32)
        mlp_parts.append(w.ravel())
        mlp_parts.append(b.ravel())
        mlp_floats += w.size + b.size
    print(f"  MLP: {mlp_floats} floats ({mlp_floats * 4} bytes)")

    mlp_weights = np.concatenate(mlp_parts)

    # Write binary
    weights_path = output_dir / "weights.bin"
    with open(weights_path, "wb") as f:
        grid_packed.ravel().tofile(f)
        dequant_scale.tofile(f)
        dequant_offset.tofile(f)
        mlp_weights.tofile(f)

    total_bytes = weights_path.stat().st_size
    old_bytes = (grid_hwc.size + mlp_floats) * 4
    print(f"  weights.bin: {total_bytes:,} bytes ({old_bytes/total_bytes:.1f}x smaller)")

    # Write metadata
    meta = {
        "arch": "plane",
        "tile": tile_name,
        "resolution": resolution,
        "resolution_w": resolution_w,
        "features": features,
        "mlp_hidden_dim": mlp_hidden_dim,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "elev_range": elev_range,
        "quantized": True,
        "grid_uints": int(grid_uints),
        "mlp_floats": mlp_floats,
        "total_params": model.param_count(),
        "source": model_path.name,
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Generate reference coords + expected elevations
    side = int(np.ceil(np.sqrt(num_ref)))
    n = side * side
    coords_1d = np.linspace(-1, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    with torch.no_grad():
        coords_t = torch.from_numpy(coords)
        pred_norm = model(coords_t).squeeze(-1).numpy()

    pred_meters = (pred_norm + 1.0) / 2.0 * elev_range + elev_min

    ref_path = output_dir / "reference.bin"
    with open(ref_path, "wb") as f:
        f.write(struct.pack("I", n))
        coords.tofile(f)
        pred_meters.astype(np.float32).tofile(f)

    print(f"  -> {output_dir}/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export feature plane MLP weights for compute shader"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Path to a single .pt checkpoint")
    group.add_argument("--batch", help="Directory of .pt files to export (all plane models)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: assets/planes/<tile>/)"
    )
    parser.add_argument(
        "--num-ref", type=int, default=1024,
        help="Number of reference points for GPU verification",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Collect .pt files
    if args.model:
        pt_files = [Path(args.model)]
    else:
        batch_dir = Path(args.batch)
        pt_files = sorted(batch_dir.glob("*_plane_*.pt"))
        if not pt_files:
            print(f"No plane .pt files found in {batch_dir}")
            return
        print(f"Found {len(pt_files)} plane checkpoints in {batch_dir}/\n")

    exported = 0
    skipped = 0
    for pt in pt_files:
        tile_name = parse_tile_name(pt.stem)
        if tile_name is None:
            print(f"[SKIP] {pt.name}: cannot parse tile name")
            skipped += 1
            continue

        if args.output_dir:
            out = Path(args.output_dir) / tile_name if args.batch else Path(args.output_dir)
        else:
            out = DEFAULT_PLANES_DIR / tile_name

        print(f"[{tile_name}] {pt.name}")
        if export_one(pt, out, args.num_ref, args.device):
            exported += 1
        else:
            skipped += 1
        print()

    if len(pt_files) > 1:
        print(f"Done: {exported} exported, {skipped} skipped")


if __name__ == "__main__":
    main()
