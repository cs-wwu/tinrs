"""
Analyze where max errors occur in a trained INR model.

Key question: Are the worst errors at cliff edges (acceptable Gibbs-like
ringing from sinusoidal representation) or systematic regional biases
(unacceptable: whole hills shifted up/down)?

Usage:
    python analyze_errors.py --model output/n47w122_siren_9x144_w60_gc_1.0_wu_1000.pt \
                             --tile n47w122
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from models import SIREN
from train import load_hgt_png, make_coordinate_grid


def load_model(model_path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    sd = checkpoint["state_dict"]

    # Infer architecture from state dict keys
    layer_keys = [
        k for k in sd if k.startswith("net.") and k.endswith(".linear.weight")
    ]
    num_layers = len(layer_keys)
    hidden_dim = sd["net.0.linear.weight"].shape[0]
    omega = checkpoint.get("omega", 60.0)

    model = SIREN(hidden_dim=hidden_dim, num_layers=num_layers, omega=omega)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def compute_predictions(model, size, device):
    coords = make_coordinate_grid(size)
    coords_t = torch.from_numpy(coords).to(device)
    chunk = 512 * 1024
    preds = []
    with torch.no_grad():
        for i in range(0, coords_t.shape[0], chunk):
            preds.append(model(coords_t[i : i + chunk]))
    return torch.cat(preds, dim=0).squeeze(-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Analyze INR error distribution")
    parser.add_argument("--model", required=True, help="Path to .pt model file")
    parser.add_argument("--tile", required=True, help="Tile name (e.g. n47w122)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--maps-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = Path(__file__).parent
    maps_dir = (
        Path(args.maps_dir)
        if args.maps_dir
        else script_dir / ".." / ".." / "assets" / "maps"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir / ".." / ".." / "output"
    )

    # Load data and model
    tile_path = maps_dir / f"{args.tile}.hgt.png"
    print(f"Loading tile: {tile_path}")
    elevation = load_hgt_png(str(tile_path))
    h, w = elevation.shape
    elev_min, elev_max = float(elevation.min()), float(elevation.max())
    elev_range = elev_max - elev_min

    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.device)

    # Inference
    print("Running inference...")
    pred_norm = compute_predictions(model, h, args.device)
    pred_meters = (pred_norm + 1.0) / 2.0 * elev_range + elev_min
    pred_map = pred_meters.reshape(h, w)
    error_map = pred_map - elevation
    abs_error = np.abs(error_map)

    rmse = math.sqrt(np.mean(error_map**2))
    print(f"RMSE: {rmse:.2f}m, Max error: {abs_error.max():.2f}m")

    # -- Slope analysis --
    # Slope in meters per pixel (approximate, ~30m per pixel at equator)
    grad_y, grad_x = np.gradient(elevation)
    slope = np.sqrt(grad_x**2 + grad_y**2)  # m/pixel

    # -- 1. Error vs slope correlation --
    flat = slope.ravel()
    err_flat = abs_error.ravel()

    # Bin by slope
    slope_bins = [0, 5, 15, 30, 60, 100, 200, float("inf")]
    bin_labels = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200", "200+"]

    print(f"\n{'=' * 70}")
    print("ERROR BY TERRAIN STEEPNESS (slope = m elevation change per pixel)")
    print(f"{'=' * 70}")
    print(
        f"{'Slope (m/px)':<15} {'Pixels':>10} {'% Area':>8} "
        f"{'Mean Err':>10} {'P95 Err':>10} {'P99 Err':>10} {'Max Err':>10}"
    )
    print("-" * 75)

    for i in range(len(bin_labels)):
        lo, hi = slope_bins[i], slope_bins[i + 1]
        mask = (flat >= lo) & (flat < hi)
        count = mask.sum()
        if count == 0:
            continue
        errs = err_flat[mask]
        pct = 100.0 * count / len(flat)
        print(
            f"{bin_labels[i]:<15} {count:>10,} {pct:>7.1f}% "
            f"{errs.mean():>10.2f} {np.percentile(errs, 95):>10.2f} "
            f"{np.percentile(errs, 99):>10.2f} {errs.max():>10.2f}"
        )

    # -- 2. Top error locations --
    print(f"\n{'=' * 70}")
    print("TOP 20 ERROR LOCATIONS")
    print(f"{'=' * 70}")
    print(
        f"{'Rank':<6} {'Row':>5} {'Col':>5} {'Error(m)':>10} {'Slope':>8} "
        f"{'Elev(m)':>9} {'Pred(m)':>9} {'Context'}"
    )
    print("-" * 75)

    # Get top error indices
    flat_idx = np.argsort(err_flat)[::-1][:20]
    for rank, idx in enumerate(flat_idx):
        r, c = divmod(idx, w)
        err = error_map.ravel()[idx]
        slp = flat[idx]
        elev = elevation[r, c]
        pred = pred_map[r, c]

        # Characterize: is this near a cliff edge?
        r0, r1 = max(0, r - 2), min(h, r + 3)
        c0, c1 = max(0, c - 2), min(w, c + 3)
        local_slope = slope[r0:r1, c0:c1]
        local_elev = elevation[r0:r1, c0:c1]
        local_range = local_elev.max() - local_elev.min()

        if local_slope.max() > 100:
            ctx = f"CLIFF (local relief {local_range:.0f}m in 5px)"
        elif local_slope.max() > 30:
            ctx = f"STEEP (local relief {local_range:.0f}m in 5px)"
        elif local_slope.max() > 10:
            ctx = f"moderate slope (relief {local_range:.0f}m)"
        else:
            ctx = f"FLAT (relief {local_range:.0f}m) *** CONCERNING ***"

        print(
            f"{rank + 1:<6} {r:>5} {c:>5} {err:>+10.2f} {slp:>8.1f} "
            f"{elev:>9.1f} {pred:>9.1f} {ctx}"
        )

    # -- 3. Regional bias check --
    # Downsample to 36x36 blocks (~100x100 pixels each) and check mean error
    print(f"\n{'=' * 70}")
    print("REGIONAL BIAS CHECK (36x36 blocks, ~100px each)")
    print(f"{'=' * 70}")

    block = h // 36
    biases = []
    for br in range(36):
        for bc in range(36):
            r0, r1 = br * block, (br + 1) * block
            c0, c1 = bc * block, (bc + 1) * block
            block_err = error_map[r0:r1, c0:c1]
            biases.append(
                (
                    br,
                    bc,
                    block_err.mean(),
                    block_err.std(),
                    np.abs(block_err).max(),
                    elevation[r0:r1, c0:c1].mean(),
                )
            )

    biases.sort(key=lambda x: abs(x[2]), reverse=True)

    print(
        f"{'Block':>8} {'Mean Bias':>10} {'Std':>8} {'Max Abs':>8} "
        f"{'Avg Elev':>9} {'Verdict'}"
    )
    print("-" * 60)

    concerning = 0
    for br, bc, mean_b, std_b, max_b, avg_elev in biases[:15]:
        if abs(mean_b) > 10:
            verdict = "*** BIASED ***"
            concerning += 1
        elif abs(mean_b) > 5:
            verdict = "slight bias"
        else:
            verdict = "ok"
        print(
            f"({br:>2},{bc:>2}) {mean_b:>+10.2f} {std_b:>8.2f} {max_b:>8.2f} "
            f"{avg_elev:>9.1f} {verdict}"
        )

    total_blocks = 36 * 36
    biased_blocks = sum(1 for _, _, mb, _, _, _ in biases if abs(mb) > 10)
    print(
        f"\nBlocks with >10m mean bias: {biased_blocks}/{total_blocks} "
        f"({100 * biased_blocks / total_blocks:.1f}%)"
    )

    mild_bias = sum(1 for _, _, mb, _, _, _ in biases if abs(mb) > 5)
    print(
        f"Blocks with >5m mean bias: {mild_bias}/{total_blocks} "
        f"({100 * mild_bias / total_blocks:.1f}%)"
    )

    # -- 4. Save error magnitude image --
    from PIL import Image

    # Error magnitude: white=0, red=high
    err_vis = np.clip(abs_error / 50.0, 0, 1)  # 50m = full red
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = (255 * err_vis).astype(np.uint8)
    rgb[:, :, 1] = (255 * (1 - err_vis)).astype(np.uint8)
    rgb[:, :, 2] = (255 * (1 - err_vis)).astype(np.uint8)

    out_path = output_dir / f"{args.tile}_error_magnitude.png"
    Image.fromarray(rgb, "RGB").save(str(out_path))
    print(f"\nError magnitude map saved: {out_path}")

    # Slope map for reference
    slope_vis = np.clip(slope / 100.0, 0, 1)
    slope_rgb = (255 * slope_vis).astype(np.uint8)
    out_path2 = output_dir / f"{args.tile}_slope.png"
    Image.fromarray(slope_rgb, "L").save(str(out_path2))
    print(f"Slope map saved: {out_path2}")


if __name__ == "__main__":
    main()
