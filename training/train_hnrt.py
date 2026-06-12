"""
HNRT (Hierarchical Neural Residual Terrain) experiment.

Tests whether a tiny MLP can learn the bilinear interpolation residual at
each of 6 grid resolution levels (8x8 through 256x256), replacing the
explicit PTC/wavelet residual coding with a learned network.

Usage:
    uv run python training/train_hnrt.py --tile n47w122
    uv run python training/train_hnrt.py --tile n47w122 --steps 10000 --width 16 --layers 3
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import helpers from train.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from models import ResidualMLP
from train import load_hgt_png, make_coordinate_grid


# Grid sizes: 8, 16, 32, 64, 128, 256
LEVELS = [8 * (2**k) for k in range(6)]  # [8, 16, 32, 64, 128, 256]

# Tile arc-second spacing: 3601 points over 1 degree = 1/3600 degree = ~30.9m
# Spacing in meters at grid resolution n: (3601/n) * 30.9m
_ARC_SEC_METERS = 30.9


def downsample_elevation(elevation: np.ndarray, grid_n: int) -> np.ndarray:
    """Downsample (3601, 3601) elevation to (grid_n, grid_n) via PIL LANCZOS."""
    from PIL import Image

    # PIL expects uint8 or similar; use float32 -> encode as 32-bit
    # Simpler: use torch average pooling which is more representative
    t = torch.from_numpy(elevation).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    downsampled = F.adaptive_avg_pool2d(t, (grid_n, grid_n))
    return downsampled.squeeze().numpy()


def bilinear_interp(grid: np.ndarray, coords: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation of a (grid_n, grid_n) heightmap at (N, 2) coords in [-1, 1].

    Uses torch.nn.functional.grid_sample to mirror GPU texture sampler behavior.

    Args:
        grid: (grid_n, grid_n) numpy array of elevation values
        coords: (N, 2) tensor of (x, y) coordinates in [-1, 1]

    Returns:
        (N,) tensor of interpolated elevation values
    """
    grid_t = torch.from_numpy(grid).float()
    # grid_sample expects (N, C, H, W) input and (N, H_out, W_out, 2) grid
    grid_4d = grid_t.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_n, grid_n)

    # Reshape coords to (1, N, 1, 2) for grid_sample
    sample_grid = coords.unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)

    result = F.grid_sample(
        grid_4d, sample_grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    # result: (1, 1, N, 1) -> (N,)
    return result.squeeze()


def train_residual(
    coords_t: torch.Tensor,
    residual_norm: torch.Tensor,
    model: ResidualMLP,
    steps: int,
    device: str,
    batch_size: int = 256 * 1024,
) -> ResidualMLP:
    """Train a ResidualMLP on normalized residuals. Returns trained model."""
    model = model.to(device)
    coords_t = coords_t.to(device)
    targets_t = residual_norm.to(device).unsqueeze(-1)
    n_points = coords_t.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def lr_lambda(step):
        t = step / max(steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    model.train()
    for step in range(steps):
        bs = min(batch_size, n_points)
        idx = torch.randint(0, n_points, (bs,), device=device)
        pred = model(coords_t[idx])
        loss = nn.functional.mse_loss(pred, targets_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    return model


def eval_combined_rmse(
    model: ResidualMLP,
    coords_t: torch.Tensor,
    base_elev: np.ndarray,
    true_elev: np.ndarray,
    resid_min: float,
    resid_max: float,
    device: str,
) -> float:
    """Evaluate RMSE of bilinear base + network residual correction."""
    resid_range = resid_max - resid_min
    model = model.to(device)
    coords_t = coords_t.to(device)

    chunk = 512 * 1024
    n_points = coords_t.shape[0]
    preds = []

    with torch.no_grad():
        for i in range(0, n_points, chunk):
            end = min(i + chunk, n_points)
            pred_norm = model(coords_t[i:end]).squeeze(-1)
            # Denormalize residual from [-1, 1] back to meters
            pred_resid = (pred_norm.cpu().numpy() + 1.0) / 2.0 * resid_range + resid_min
            preds.append(pred_resid)

    pred_resid_all = np.concatenate(preds)
    combined = base_elev + pred_resid_all
    error = combined - true_elev
    return float(math.sqrt(np.mean(error**2)))


def run_level(
    k: int,
    grid_n: int,
    elevation: np.ndarray,
    coords: np.ndarray,
    args,
    output_dir: Path,
) -> dict:
    """Run the full HNRT experiment for one resolution level."""
    spacing_m = (3601.0 / grid_n) * _ARC_SEC_METERS

    print(f"\n  Level L{k}: {grid_n}x{grid_n} grid (~{spacing_m:.0f}m spacing)")

    # 1. Downsample
    downsampled = downsample_elevation(elevation, grid_n)

    # 2. Bilinear interpolation at all 3601x3601 points
    coords_t = torch.from_numpy(coords)
    base_elev = bilinear_interp(downsampled, coords_t).numpy()
    true_elev = elevation.ravel()

    residual = true_elev - base_elev  # meters

    # 3. Bilinear-only RMSE
    bilinear_rmse = float(math.sqrt(np.mean(residual**2)))
    resid_abs_max = float(np.max(np.abs(residual)))
    print(f"    Bilinear-only RMSE: {bilinear_rmse:.2f}m  |  Max residual: {resid_abs_max:.1f}m")

    # Normalize residual to [-1, 1] for training
    resid_min = float(residual.min())
    resid_max = float(residual.max())
    resid_range = resid_max - resid_min

    if resid_range < 1e-6:
        resid_norm = np.zeros_like(residual)
    else:
        resid_norm = 2.0 * (residual - resid_min) / resid_range - 1.0

    resid_norm_t = torch.from_numpy(resid_norm.astype(np.float32))

    results_level = {
        "level": k,
        "grid_n": grid_n,
        "spacing_m": spacing_m,
        "bilinear_rmse_m": bilinear_rmse,
        "residual_abs_max_m": resid_abs_max,
    }

    # 4 & 5. Train ReLU and SIREN residual networks
    for activation in ("relu", "siren"):
        model = ResidualMLP(
            hidden_dim=args.width,
            num_layers=args.layers,
            activation=activation,
            omega=args.omega,
        )
        n_params = model.param_count()
        size_bytes = n_params  # INT8: 1 byte per param

        print(
            f"    Training {activation.upper()} ({n_params} params, {size_bytes} B INT8)...",
            end="",
            flush=True,
        )
        t0 = time.time()
        model = train_residual(
            coords_t, resid_norm_t, model, args.steps, args.device
        )
        elapsed = time.time() - t0

        combined_rmse = eval_combined_rmse(
            model, coords_t, base_elev, true_elev, resid_min, resid_max, args.device
        )
        print(f" {elapsed:.1f}s -> Combined RMSE: {combined_rmse:.2f}m")

        results_level[f"{activation}_rmse_m"] = combined_rmse
        results_level[f"net_params"] = n_params
        results_level[f"net_size_bytes"] = size_bytes

        # Save model
        model_path = output_dir / f"{args.tile}_hnrt_L{k}_{activation}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "level": k,
                "grid_n": grid_n,
                "activation": activation,
                "hidden_dim": args.width,
                "num_layers": args.layers,
                "omega": args.omega,
                "resid_min": resid_min,
                "resid_max": resid_max,
                "bilinear_rmse_m": bilinear_rmse,
                "combined_rmse_m": combined_rmse,
            },
            model_path,
        )

    return results_level


def print_summary_table(results: list[dict], args) -> None:
    n_params = results[0]["net_params"]
    size_bytes = results[0]["net_size_bytes"]

    header = f"{'Level':<6} {'Grid':<9} {'Spacing':<9} {'Bilinear RMSE':>14} {'+ ReLU':>10} {'+ SIREN':>10} {'Net size':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        grid_str = f"{r['grid_n']}x{r['grid_n']}"
        spacing_str = f"~{r['spacing_m']:.0f}m"
        bilinear_str = f"{r['bilinear_rmse_m']:.2f}m"
        relu_str = f"{r['relu_rmse_m']:.2f}m"
        siren_str = f"{r['siren_rmse_m']:.2f}m"
        size_str = f"{r['net_size_bytes']} B"
        print(
            f"L{r['level']:<5} {grid_str:<9} {spacing_str:<9} {bilinear_str:>14} {relu_str:>10} {siren_str:>10} {size_str:>10}"
        )

    print(f"\nNetwork: {args.width}x{args.layers} ({n_params} params, {size_bytes} B INT8)")
    print("Note: Net size is per level; total HNRT = grid storage + 6x this network")


def main():
    parser = argparse.ArgumentParser(
        description="HNRT residual experiment: bilinear grid + learned residual MLP"
    )
    parser.add_argument("--tile", type=str, default="n47w122")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--omega", type=float, default=60.0, help="SIREN omega")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    script_dir = Path(__file__).parent
    maps_dir = (
        Path(args.maps_dir)
        if args.maps_dir
        else script_dir.parent / "assets" / "maps"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else script_dir.parent / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_path = maps_dir / f"{args.tile}.hgt.png"
    if not tile_path.exists():
        print(f"Tile not found: {tile_path}")
        sys.exit(1)

    print(f"HNRT Residual Experiment")
    print(f"  Tile:    {args.tile} ({tile_path.stat().st_size / 1024 / 1024:.1f} MB PNG)")
    print(f"  Device:  {args.device}")
    print(f"  Network: {args.width}x{args.layers} ResidualMLP, {args.steps} steps/level")
    print(f"  Output:  {output_dir}")

    elevation = load_hgt_png(str(tile_path))
    h, w = elevation.shape
    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    print(f"  Elevation: [{elev_min:.0f}m, {elev_max:.0f}m], range {elev_max - elev_min:.0f}m")

    coords = make_coordinate_grid(h)  # (3601*3601, 2)

    all_results = []
    for k, grid_n in enumerate(LEVELS):
        result = run_level(k, grid_n, elevation, coords, args, output_dir)
        all_results.append(result)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print_summary_table(all_results, args)

    # Compare against hash MLP reference
    print(f"\nReference: Hash H15 L16 = 5.23m RMSE, 1030 KB INT8")
    print(f"           Hash H14 L16 = 8.32m RMSE, 518 KB INT8")
    print(f"           SIREN 5x256  = 5.08m RMSE, 258 KB INT8")


if __name__ == "__main__":
    main()
