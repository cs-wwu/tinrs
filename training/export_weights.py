"""
Export SIREN weights to raw binary for Vulkan compute shader consumption.

Outputs:
    {name}.bin       - flat float32 binary: [W0, b0, W1, b1, ..., W_final, b_final]
    {name}.json      - metadata (num_layers, hidden_dim, omega, elev_min/max/range)
    {name}_ref.bin   - reference: [coords (N*2 floats), elevations (N floats)]

Usage:
    python export_weights.py --model output/n47w122_siren_9x144_w60_gc_1.0_wu_1000.pt
    python export_weights.py --model output/n47w122_siren_9x144_w60_gc_1.0_wu_1000.pt --num-ref 1000
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
from models import SIREN


def main():
    parser = argparse.ArgumentParser(
        description="Export SIREN weights for compute shader"
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
    name = model_path.stem

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=args.device, weights_only=False)
    sd = checkpoint["state_dict"]

    # Infer architecture from state dict
    layer_keys = [
        k for k in sd if k.startswith("net.") and k.endswith(".linear.weight")
    ]
    num_layers = len(layer_keys)
    hidden_dim = sd["net.0.linear.weight"].shape[0]
    omega = checkpoint.get("omega", 60.0)
    elev_min = checkpoint.get("elev_min_m", 0.0)
    elev_max = checkpoint.get("elev_max_m", 1.0)
    elev_range = elev_max - elev_min

    print(f"Model: SIREN {num_layers}x{hidden_dim}, omega={omega}")
    print(f"Elevation: [{elev_min:.1f}, {elev_max:.1f}]m (range {elev_range:.1f}m)")

    # Reconstruct model for reference evaluation
    model = SIREN(hidden_dim=hidden_dim, num_layers=num_layers, omega=omega)
    model.load_state_dict(sd)
    model.eval()

    # -- Export weights as flat binary --
    # Layout: [W0, b0, W1, b1, ..., W_final, b_final]
    # PyTorch stores weights as [out_features, in_features] (row-major)
    weight_data = []
    total_floats = 0

    # Hidden layers
    for i in range(num_layers):
        w = sd[f"net.{i}.linear.weight"].numpy().astype(np.float32)
        b = sd[f"net.{i}.linear.bias"].numpy().astype(np.float32)
        weight_data.append(w.ravel())
        weight_data.append(b.ravel())
        total_floats += w.size + b.size
        print(f"  Layer {i}: W{list(w.shape)} ({w.size}), b[{b.size}]")

    # Final linear layer
    w_final = sd["final.weight"].numpy().astype(np.float32)
    b_final = sd["final.bias"].numpy().astype(np.float32)
    weight_data.append(w_final.ravel())
    weight_data.append(b_final.ravel())
    total_floats += w_final.size + b_final.size
    print(f"  Final:   W{list(w_final.shape)} ({w_final.size}), b[{b_final.size}]")

    all_weights = np.concatenate(weight_data)
    assert all_weights.shape[0] == total_floats
    print(f"  Total: {total_floats} floats = {total_floats * 4} bytes")

    weights_path = output_dir / f"{name}.bin"
    all_weights.tofile(str(weights_path))
    print(f"  Saved: {weights_path} ({weights_path.stat().st_size} bytes)")

    # -- Export metadata --
    meta = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "omega": omega,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "elev_range": elev_range,
        "total_floats": total_floats,
        "total_params": model.param_count(),
    }
    meta_path = output_dir / f"{name}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved: {meta_path}")

    # -- Generate reference coords + expected elevations --
    n = args.num_ref
    # Evenly spaced grid
    side = int(np.ceil(np.sqrt(n)))
    n = side * side  # round up to square
    coords_1d = np.linspace(-1, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N, 2)

    with torch.no_grad():
        coords_t = torch.from_numpy(coords)
        pred_norm = model(coords_t).squeeze(-1).numpy()

    # Denormalize to meters
    pred_meters = (pred_norm + 1.0) / 2.0 * elev_range + elev_min

    # Write reference: [coords_flat (N*2), elevations (N)]
    ref_path = output_dir / f"{name}_ref.bin"
    with open(ref_path, "wb") as f:
        # Header: num_points as uint32
        f.write(struct.pack("I", n))
        # Coords: N*2 float32
        coords.tofile(f)
        # Expected elevations: N float32
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
