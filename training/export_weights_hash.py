"""
Export hash-encoded MLP weights to raw binary for Vulkan compute shader.

Outputs:
    weights.bin   - [hash_tables (all levels flat), mlp_W0, mlp_b0, ..., mlp_W_final, mlp_b_final]
    meta.json     - model metadata (num_levels, hashmap_size, growth_factor, mlp dims, elev_min/max)
    reference.bin - [num_points: u32, coords: N*2 f32, elevations: N f32]

Usage:
    python export_weights_hash.py --model output/n47w122_hash_L16_H14_40k.pt
    python export_weights_hash.py --model output/n47w122_hash_L16_H14_40k.pt --output-dir ../compute
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch
from models import HashMLP


def main():
    parser = argparse.ArgumentParser(
        description="Export hash MLP weights for compute shader"
    )
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--output-dir", default=None, help="Output directory (default: same as model)"
    )
    parser.add_argument(
        "--num-ref",
        type=int,
        default=1024,
        help="Number of reference points for GPU verification",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_path = Path(args.model)
    output_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    sd = checkpoint["state_dict"]

    num_levels = checkpoint["num_levels"]
    features_per_level = checkpoint["features_per_level"]
    log2_hashmap_size = checkpoint["log2_hashmap_size"]
    hashmap_size = 2**log2_hashmap_size
    mlp_hidden_dim = checkpoint["mlp_hidden_dim"]
    mlp_num_layers = checkpoint["mlp_num_layers"]
    elev_min = checkpoint["elev_min_m"]
    elev_max = checkpoint["elev_max_m"]
    elev_range = elev_max - elev_min

    # Compute growth factor and resolutions (matching HashEncoding.__init__)
    base_resolution = 16
    finest_resolution = 3601
    if num_levels > 1:
        growth_factor = math.exp(
            math.log(finest_resolution / base_resolution) / (num_levels - 1)
        )
    else:
        growth_factor = 1.0

    resolutions = [
        int(base_resolution * (growth_factor**level)) for level in range(num_levels)
    ]

    print(
        f"Model: Hash L{num_levels} H{log2_hashmap_size}, MLP {mlp_num_layers}x{mlp_hidden_dim}"
    )
    print(
        f"  Hash: {num_levels} levels, {hashmap_size} entries/level, {features_per_level} features/level"
    )
    print(f"  Growth factor: {growth_factor:.6f}")
    print(f"  Resolutions: {resolutions}")
    print(f"  Elevation: [{elev_min:.1f}, {elev_max:.1f}]m (range {elev_range:.1f}m)")

    # Reconstruct model for reference evaluation
    model = HashMLP(
        num_levels=num_levels,
        features_per_level=features_per_level,
        log2_hashmap_size=log2_hashmap_size,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_num_layers=mlp_num_layers,
    )
    model.load_state_dict(sd)
    model.eval()

    # -- Export weights as flat binary --
    # Layout: [hash_tables (all levels concatenated), mlp_weights]
    weight_data = []
    total_floats = 0

    # Hash tables: level 0, level 1, ..., level N-1
    # Each table: [hashmap_size, features_per_level] row-major
    for level in range(num_levels):
        table = sd[f"encoding.hash_tables.{level}"].numpy().astype(np.float32)
        assert table.shape == (hashmap_size, features_per_level), (
            f"Table {level} shape {table.shape} != ({hashmap_size}, {features_per_level})"
        )
        weight_data.append(table.ravel())
        total_floats += table.size
    hash_table_floats = total_floats
    print(f"  Hash tables: {hash_table_floats} floats ({hash_table_floats * 4} bytes)")

    # MLP weights: Linear layers only (skip ReLU)
    # Sequential indices: 0 (Linear), 1 (ReLU), 2 (Linear), 3 (ReLU), ..., 2*N (Linear output)
    mlp_linear_indices = list(range(0, 2 * mlp_num_layers + 1, 2))
    mlp_floats = 0
    for i, li in enumerate(mlp_linear_indices):
        w = sd[f"mlp.{li}.weight"].numpy().astype(np.float32)
        b = sd[f"mlp.{li}.bias"].numpy().astype(np.float32)
        weight_data.append(w.ravel())
        weight_data.append(b.ravel())
        total_floats += w.size + b.size
        mlp_floats += w.size + b.size
        label = "Output" if i == len(mlp_linear_indices) - 1 else f"Hidden {i}"
        print(f"  MLP {label}: W{list(w.shape)} ({w.size}), b[{b.size}]")
    print(f"  MLP total: {mlp_floats} floats ({mlp_floats * 4} bytes)")

    all_weights = np.concatenate(weight_data)
    assert all_weights.shape[0] == total_floats
    print(f"  Combined: {total_floats} floats = {total_floats * 4} bytes")

    weights_path = output_dir / "weights.bin"
    all_weights.tofile(str(weights_path))
    print(f"  Saved: {weights_path} ({weights_path.stat().st_size} bytes)")

    # -- Export metadata --
    meta = {
        "arch": "hash",
        "num_levels": num_levels,
        "features_per_level": features_per_level,
        "log2_hashmap_size": log2_hashmap_size,
        "hashmap_size": hashmap_size,
        "mlp_hidden_dim": mlp_hidden_dim,
        "mlp_num_layers": mlp_num_layers,
        "base_resolution": base_resolution,
        "growth_factor": growth_factor,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "elev_range": elev_range,
        "hash_table_floats": hash_table_floats,
        "mlp_floats": mlp_floats,
        "total_floats": total_floats,
        "total_params": model.param_count(),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # -- Generate reference coords + expected elevations --
    n = args.num_ref
    side = int(np.ceil(np.sqrt(n)))
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

    print(
        f"  Saved: {ref_path} ({n} reference points, {ref_path.stat().st_size} bytes)"
    )
    print(f"\n  Sample predictions (first 5):")
    for i in range(min(5, n)):
        print(
            f"    ({coords[i, 0]:+.4f}, {coords[i, 1]:+.4f}) -> {pred_meters[i]:.2f}m"
        )


if __name__ == "__main__":
    main()
