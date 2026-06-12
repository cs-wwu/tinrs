"""
Standalone feature plane INR trainer + exporter for terrain data.

Extracted from the unified trainer (train.py) so the plane architecture can
evolve independently (water body INR, custom losses, etc.) without touching
the shared multi-architecture interface.

Examples:
    python train_plane.py --tile n47w122
    python train_plane.py --tile n47w122 --diff --no-export
    python train_plane.py --tile-dir glo30
    python train_plane.py --tile n47w122 --features 12 --resolution 256
"""

import argparse
import json
import math
import re
import struct
import time
import urllib.error
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from download_glo30 import ensure_tile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAPS_DIR = PROJECT_ROOT / "assets" / "maps"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_PLANES_DIR = PROJECT_ROOT / "assets" / "planes"
MIN_MODEL_BYTES = 1024  # sanity floor for the cleanup-after-save check


# -- Data Loading --------------------------------------------------------------


def load_geotiff(path: str) -> np.ndarray:
    """Load GeoTIFF (GLO-30) -> (rows, cols) float32 elevation in meters."""
    from PIL import Image

    img = Image.open(path)
    assert img.mode == "F", f"Expected float32 GeoTIFF (mode 'F'), got {img.mode}"
    data = np.array(img, dtype=np.float32)
    data[data < -1000] = 0.0
    return data


def load_wbm(wbm_path: Path):
    """Load .wbm file -> (rows, cols, classes) or None if not found.

    WBM files are zlib-compressed uint8 with a 4-byte header (rows: u16, cols: u16).
    Values: 0=land, 1=ocean, 2=lake, 3=river. Returns raw classes (not collapsed).
    """
    if not wbm_path.exists():
        return None
    data = wbm_path.read_bytes()
    if len(data) < 5:
        return None
    rows, cols = struct.unpack("<HH", data[:4])
    pixels = np.frombuffer(zlib.decompress(data[4:]), dtype=np.uint8)
    if pixels.size != rows * cols:
        return None
    return rows, cols, pixels.reshape(rows, cols)


def make_coordinate_grid(height: int, width: int = 0) -> np.ndarray:
    """Create (height*width, 2) grid of coordinates in [-1, 1]."""
    if width == 0:
        width = height
    coords_x = np.linspace(-1, 1, width, dtype=np.float32)
    coords_y = np.linspace(-1, 1, height, dtype=np.float32)
    xx, yy = np.meshgrid(coords_x, coords_y, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=-1)


def compute_plane_resolution(
    base_res: int, tile_h: int, tile_w: int
) -> tuple[int, int]:
    """Compute (res_h, res_w) for rectangular feature plane.

    base_res is the latitude resolution. Longitude resolution scales
    proportionally to tile aspect ratio, rounded to nearest multiple of 4
    (for uint32 packing in export).
    """
    aspect = tile_w / tile_h
    res_w = max(4, round(base_res * aspect / 4) * 4)
    return base_res, res_w


def find_glo30_tile(glo30_dir: Path, short_name: str):
    """Map short name like 'n48w123' to GLO-30 GeoTIFF path, or None.

    Path-only resolver. Does not check existence or fetch. Use ensure_tile()
    from download_glo30 when you need the file on disk.
    """
    m = re.match(r"^([ns])(\d+)([ew])(\d+)$", short_name.lower())
    if not m:
        return None
    ns = m.group(1).upper()
    lat = int(m.group(2))
    ew = m.group(3).upper()
    lon = int(m.group(4))
    filename = f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
    return glo30_dir / filename


def find_wbm_path(maps_dir: Path, tile_name: str) -> Path:
    """Map tile name to WBM file path."""
    ns = tile_name[0].upper()
    lat = int(tile_name[1:3])
    ew = tile_name[3].upper()
    lon = int(tile_name[4:7])
    filename = f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_WBM.wbm"
    return maps_dir / "glo30" / "wbm" / filename


def extract_tile_name(path: Path) -> str:
    """Extract short tile name from any supported file path."""
    name = path.stem
    if name.endswith(".hgt"):
        return name.replace(".hgt", "")
    m = re.search(r"_([NS])(\d+)_\d+_([EW])(\d+)_\d+_DEM", name)
    if m:
        ns = m.group(1).lower()
        lat = int(m.group(2))
        ew = m.group(3).lower()
        lon = int(m.group(4))
        return f"{ns}{lat:02d}{ew}{lon:03d}"
    return name


# -- Model ---------------------------------------------------------------------

OUT_ELEV = 0
OUT_WATER = 1
OUT_DX = 2
OUT_DY = 3


class FeaturePlaneMLP(nn.Module):
    """Dense 2D feature grid + tiny MLP decoder for terrain.

    Outputs 2 channels (elev, water) or 4 channels (elev, water, dx, dy).
    """

    def __init__(self, resolution=256, features=4, mlp_hidden_dim=32, resolution_w=0,
                 learn_grad=False):
        super().__init__()
        self.resolution = resolution
        self.resolution_w = resolution_w if resolution_w > 0 else resolution
        self.features = features
        self.mlp_hidden_dim = mlp_hidden_dim
        self.learn_grad = learn_grad
        self.num_outputs = 4 if learn_grad else 2

        self.grid = nn.Parameter(
            torch.empty(1, features, self.resolution, self.resolution_w).uniform_(
                -1e-4, 1e-4
            )
        )

        self.mlp = nn.Sequential(
            nn.Linear(features, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, self.num_outputs),
        )

    def forward(self, coords):
        """coords: (N, 2) in [-1, 1]. Returns: (N, num_outputs)."""
        sample_grid = coords.unsqueeze(0).unsqueeze(0)
        sampled = torch.nn.functional.grid_sample(
            self.grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        features = sampled[0, :, 0, :].T
        return self.mlp(features)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def grid_size_kb(self):
        return self.grid.numel() / 1024

    def mlp_size_kb(self):
        return sum(p.numel() for p in self.mlp.parameters()) / 1024


# -- Training ------------------------------------------------------------------


def config_label(args) -> str:
    """Short string for filenames."""
    parts = ["plane"]
    res_w = args.resolution_w or args.resolution
    if res_w != args.resolution:
        parts.append(f"{args.resolution}x{res_w}x{args.features}")
    else:
        parts.append(f"{args.resolution}x{args.features}")
    if args.mlp_width != 32:
        parts.append(f"mlp{args.mlp_width}")
    if args.steps != 450000:
        if args.steps >= 1000:
            parts.append(f"{args.steps // 1000}k")
        else:
            parts.append(f"{args.steps}s")
    if args.l4_weight > 0:
        parts.append(f"l4_{args.l4_weight}")
    if args.slope_weight > 0:
        parts.append(f"sw_{args.slope_weight}")
    if args.grad_clip > 0:
        parts.append(f"gc_{args.grad_clip}")
    if args.warmup > 0:
        parts.append(f"wu_{args.warmup}")
    if args.learn_grad:
        parts.append("grad")
    return "_".join(parts)


def train(model, elevation, wbm, args):
    """Train feature plane model. Returns (model, metrics, pred_map, error_map)."""
    h, w = elevation.shape
    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    elev_range = elev_max - elev_min

    if elev_range < 1e-6:
        elev_norm = np.zeros_like(elevation)
    else:
        elev_norm = 2.0 * (elevation - elev_min) / elev_range - 1.0

    coords = make_coordinate_grid(h, w)
    targets = elev_norm.ravel()
    coords_t = torch.from_numpy(coords).to(args.device)
    elev_targets_t = torch.from_numpy(targets).to(args.device)
    n_points = coords_t.shape[0]

    # Gradient targets: finite differences of elev_norm, normalized to [-1, 1]
    if args.learn_grad:
        grad_y_pix, grad_x_pix = np.gradient(elev_norm)
        grad_scale = float(max(np.abs(grad_x_pix).max(), np.abs(grad_y_pix).max(), 1e-8))
        dx_targets = (grad_x_pix / grad_scale).ravel().astype(np.float32)
        dy_targets = (grad_y_pix / grad_scale).ravel().astype(np.float32)
        dx_targets_t = torch.from_numpy(dx_targets).to(args.device)
        dy_targets_t = torch.from_numpy(dy_targets).to(args.device)
        print(f"  Gradient targets: dx range [{dx_targets.min():.2f}, {dx_targets.max():.2f}], "
              f"dy range [{dy_targets.min():.2f}, {dy_targets.max():.2f}] "
              f"(grad_scale={grad_scale:.6f})")

    # Water targets from WBM (same grid as elevation)
    # WBM classes: 0=land, 1=ocean, 2=lake, 3=river
    if wbm is not None:
        wbm_rows, wbm_cols, wbm_classes = wbm
        assert wbm_rows == h and wbm_cols == w, (
            f"WBM dimensions ({wbm_rows}x{wbm_cols}) don't match elevation ({h}x{w})"
        )
        water_mask = (wbm_classes > 0).astype(np.uint8)
    else:
        wbm_rows, wbm_cols, wbm_classes, water_mask = 0, 0, None, None

    has_water = water_mask is not None and args.water_lambda > 0
    if has_water:
        water_targets_t = torch.from_numpy(water_mask.ravel().astype(np.float32)).to(args.device)
        wbm_classes_flat = wbm_classes.ravel()
        # Per-pixel BCE weights by water class, normalized to mean 1.0
        weight_map = np.ones(wbm_classes_flat.size, dtype=np.float32)
        weight_map[wbm_classes_flat == 1] = args.w_ocean
        weight_map[wbm_classes_flat == 2] = args.w_lake
        weight_map[wbm_classes_flat == 3] = args.w_river
        # Boundary-distance weighting: land pixels near water get higher weight
        k = args.boundary_k
        if k > 0 and water_mask.any():
            # Iterative dilation to compute approximate distance bands (no scipy needed)
            boundary_boost = np.ones_like(water_mask, dtype=np.float32)
            expanded = water_mask.astype(bool)
            for d in range(1, k + 1):
                prev = expanded
                expanded = np.zeros_like(expanded)
                expanded[1:, :] |= prev[:-1, :]
                expanded[:-1, :] |= prev[1:, :]
                expanded[:, 1:] |= prev[:, :-1]
                expanded[:, :-1] |= prev[:, 1:]
                expanded |= prev
                ring = expanded & ~prev
                weight_at_d = 1.0 + max(0, k - d) / k * (args.boundary_weight - 1.0)
                boundary_boost[ring] = weight_at_d
            weight_map *= boundary_boost.ravel()
        weight_map /= weight_map.mean()
        water_weights_t = torch.from_numpy(weight_map).to(args.device)
        # Class labels on GPU for per-class IoU
        river_mask_t = torch.from_numpy((wbm_classes_flat == 3).astype(np.float32)).to(args.device)
    else:
        water_targets_t = None
        water_weights_t = None
        river_mask_t = None

    # Water oversampling: ensure enough water pixels per batch on sparse tiles
    water_oversample_n = 0
    if has_water and args.water_oversample > 0:
        water_frac = float(water_mask.sum()) / water_mask.size
        oversample_alpha = max(0.0, args.water_oversample - water_frac)
        if oversample_alpha > 0:
            water_indices = torch.from_numpy(
                np.where(water_mask.ravel() > 0)[0].astype(np.int64)
            ).to(args.device)
            water_oversample_n = min(
                int(oversample_alpha * args.batch_size),
                args.batch_size - 1,
            )
            print(
                f"  Water oversampling: {water_frac*100:.2f}% water -> "
                f"~{args.water_oversample*100:.0f}% target, "
                f"{water_oversample_n:,} extra water px/batch"
            )

    # Slope-weighted sampling
    if args.slope_weight > 0:
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_x**2 + grad_y**2).ravel()
        uniform = np.ones_like(slope)
        blend = (
            (1 - args.slope_weight) * uniform / uniform.sum()
            + args.slope_weight * slope / slope.sum()
        )
        sample_weights = torch.from_numpy(blend.astype(np.float32)).to(args.device)
        print(f"  Slope-weighted sampling: weight={args.slope_weight}")
    else:
        sample_weights = None

    model = model.to(args.device)
    total_p = model.param_count()
    train_p = model.trainable_param_count()
    if args.compile:
        compile_mode = "reduce-overhead" if args.device == "cuda" else "default"
        model = torch.compile(model, mode=compile_mode)

    print(f"  Model: plane ({total_p:,} params, {train_p:,} trainable)")
    print(f"  Size: {train_p / 1024:.1f} KB INT8")
    print(
        f"  Plane: {model.resolution}x{model.resolution_w} grid, "
        f"{model.features} features/cell"
    )
    print(f"  Grid: {model.grid_size_kb():.1f} KB INT8")
    print(f"  MLP: {model.features} -> {model.mlp_hidden_dim} -> {model.num_outputs}")
    print(
        f"  Data: {h}x{w} = {n_points:,} points, "
        f"range [{elev_min:.0f}m, {elev_max:.0f}m] ({elev_range:.0f}m)"
    )
    if wbm_classes is not None:
        total_px = wbm_classes.size
        n_ocean = int(np.sum(wbm_classes == 1))
        n_lake = int(np.sum(wbm_classes == 2))
        n_river = int(np.sum(wbm_classes == 3))
        water_pct = (n_ocean + n_lake + n_river) / total_px * 100
        status = f"lambda={args.water_lambda}" if has_water else "not training, lambda=0"
        print(f"  WBM: {wbm_rows}x{wbm_cols}, {water_pct:.1f}% water ({status})")
        if has_water:
            print(
                f"  WBM classes: ocean={n_ocean/total_px*100:.1f}% (w={args.w_ocean}), "
                f"lake={n_lake/total_px*100:.1f}% (w={args.w_lake}), "
                f"river={n_river/total_px*100:.1f}% (w={args.w_river})"
            )
    else:
        print("  WBM: not available")

    # Optimizer: grid gets 10x LR (dense gradients from grid_sample)
    use_fused = args.device == "cuda"

    extras = ""
    if args.grad_clip > 0:
        extras += f", grad_clip={args.grad_clip}"
    if args.warmup > 0:
        extras += f", warmup={args.warmup}"
    if args.amp:
        extras += ", AMP"
    if args.compile:
        extras += ", compiled"
    if use_fused:
        extras += ", fused_adam"
    print(f"  Training: {args.steps} steps, batch {args.batch_size:,}, lr {args.lr}{extras}")
    optimizer = torch.optim.Adam(
        [
            {"params": [model.grid], "lr": args.lr * 10},
            {"params": model.mlp.parameters(), "lr": args.lr},
        ],
        fused=use_fused,
    )
    scaler = torch.amp.GradScaler(args.device) if args.amp else None

    warmup = args.warmup
    total = args.steps

    if args.scheduler == "cosine":
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return 1e-3 + (1.0 - 1e-3) * step / warmup
            t = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == "step":
        step_size = total // 4
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return 1e-3 + (1.0 - 1e-3) * step / warmup
            return 0.5 ** ((step - warmup) // step_size)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        if warmup > 0:
            def lr_lambda(step):
                if step < warmup:
                    return 1e-3 + (1.0 - 1e-3) * step / warmup
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None

    if warmup > 0:
        print(f"  LR warmup: {warmup} steps")

    start_time = time.time()
    model.train()
    log_interval = max(args.steps // 10, 500)
    best_rmse = float("inf")
    best_step = 0
    patience_counter = 0
    stopped_step = args.steps

    for step in range(args.steps):
        bs = min(args.batch_size, n_points)
        if water_oversample_n > 0:
            n_general = bs - water_oversample_n
            if sample_weights is not None:
                idx_general = torch.multinomial(sample_weights, n_general, replacement=True)
            else:
                idx_general = torch.randint(0, n_points, (n_general,), device=args.device)
            idx_water = water_indices[
                torch.randint(0, len(water_indices), (water_oversample_n,), device=args.device)
            ]
            idx = torch.cat([idx_general, idx_water])
        elif sample_weights is not None:
            idx = torch.multinomial(sample_weights, bs, replacement=True)
        else:
            idx = torch.randint(0, n_points, (bs,), device=args.device)
        batch_coords = coords_t[idx]
        batch_elev = elev_targets_t[idx]
        if has_water:
            batch_water = water_targets_t[idx]
            batch_w_weight = water_weights_t[idx]

        with torch.amp.autocast(args.device, enabled=args.amp):
            pred = model(batch_coords)
            elev_pred = pred[:, OUT_ELEV]
            mse_val = nn.functional.mse_loss(elev_pred, batch_elev)
            loss = mse_val

            if args.l4_weight > 0:
                loss = loss + args.l4_weight * torch.mean(
                    (elev_pred - batch_elev) ** 4
                )

            if has_water:
                water_pred = pred[:, OUT_WATER]
                bce = nn.functional.binary_cross_entropy_with_logits(
                    water_pred, batch_water, reduction="none"
                )
                loss = loss + args.water_lambda * (bce * batch_w_weight).mean()

            if args.learn_grad:
                dx_pred = pred[:, OUT_DX]
                dy_pred = pred[:, OUT_DY]
                batch_dx = dx_targets_t[idx]
                batch_dy = dy_targets_t[idx]
                grad_loss = (nn.functional.mse_loss(dx_pred, batch_dx)
                             + nn.functional.mse_loss(dy_pred, batch_dy))
                loss = loss + args.grad_lambda * grad_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (step + 1) % log_interval == 0 or step == 0 or step == args.steps - 1:
            rmse_m = math.sqrt(mse_val.item() * (elev_range / 2.0) ** 2)
            extra = ""
            if has_water:
                with torch.no_grad():
                    w_pred = (water_pred > 0).float()
                    tp = (w_pred * batch_water).sum().item()
                    fp = (w_pred * (1 - batch_water)).sum().item()
                    fn = ((1 - w_pred) * batch_water).sum().item()
                    batch_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
                    # River recall: fraction of river pixels predicted as water
                    batch_river = river_mask_t[idx]
                    r_tp = (w_pred * batch_river).sum().item()
                    r_total = batch_river.sum().item()
                    r_recall = r_tp / r_total if r_total > 0 else 1.0
                extra = f", iou={batch_iou:.3f}, river={r_recall:.3f}"
            if args.learn_grad:
                extra += f", grad={grad_loss.item():.6f}"
            print(f"    Step {step + 1:>5}/{args.steps}: loss={loss.item():.6f}, ~RMSE={rmse_m:.2f}m{extra}")

            if rmse_m < best_rmse:
                best_rmse = rmse_m
                best_step = step + 1
                patience_counter = 0
            elif step > 0 and args.patience > 0:
                patience_counter += 1
                if patience_counter >= args.patience:
                    stopped_step = step + 1
                    print(f"    Early stopping: no improvement for {args.patience} checks")
                    break

    train_time = time.time() - start_time

    # -- Evaluation --
    model.eval()
    chunk = 512 * 1024
    with torch.no_grad():
        preds = []
        for i in range(0, n_points, chunk):
            end = min(i + chunk, n_points)
            preds.append(model(coords_t[i:end]))
        pred_all = torch.cat(preds, dim=0)  # (N, 2)

    # Elevation metrics (output 0)
    elev_pred_norm = pred_all[:, OUT_ELEV].cpu().numpy()
    pred_meters = (elev_pred_norm + 1.0) / 2.0 * elev_range + elev_min
    pred_map = pred_meters.reshape(h, w)
    error_map = pred_map - elevation

    abs_error = np.abs(error_map)
    mse_eval = np.mean(error_map**2)
    rmse = math.sqrt(mse_eval)
    mae = float(np.mean(abs_error))
    max_err = float(np.max(abs_error))
    p99_err = float(np.percentile(abs_error, 99))
    psnr = (
        20.0 * math.log10(elev_range / math.sqrt(mse_eval))
        if mse_eval > 0
        else float("inf")
    )

    # Water metrics (output 1), per-class recall
    water_metrics = {}
    water_pred_map = None
    if water_mask is not None:
        water_logits = pred_all[:, OUT_WATER].cpu().numpy()
        water_pred_map = (water_logits > 0).astype(np.uint8).reshape(h, w)
        tp = float(np.sum((water_pred_map == 1) & (water_mask == 1)))
        fp = float(np.sum((water_pred_map == 1) & (water_mask == 0)))
        fn = float(np.sum((water_pred_map == 0) & (water_mask == 1)))
        tn = float(np.sum((water_pred_map == 0) & (water_mask == 0)))
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        water_metrics = {
            "water_accuracy": accuracy,
            "water_iou": iou,
            "water_precision": precision,
            "water_recall": recall,
            "water_tp": int(tp),
            "water_fp": int(fp),
            "water_fn": int(fn),
        }
        # Per-class recall
        for cls, name in [(1, "ocean"), (2, "lake"), (3, "river")]:
            cls_mask = (wbm_classes == cls)
            cls_total = float(np.sum(cls_mask))
            if cls_total > 0:
                cls_tp = float(np.sum((water_pred_map == 1) & cls_mask))
                water_metrics[f"{name}_recall"] = cls_tp / cls_total
                water_metrics[f"{name}_pixels"] = int(cls_total)

    # Gradient metrics (outputs 2, 3)
    grad_metrics = {}
    if args.learn_grad:
        dx_pred = pred_all[:, OUT_DX].cpu().numpy()
        dy_pred = pred_all[:, OUT_DY].cpu().numpy()
        # Normal angle via dot product: n=(-dx, 1, -dy), avoid allocating Nx3 arrays
        pred_mag = np.sqrt(1.0 + dx_pred**2 + dy_pred**2)
        gt_mag = np.sqrt(1.0 + dx_targets**2 + dy_targets**2)
        dot = 1.0 + dx_pred * dx_targets + dy_pred * dy_targets
        cos_angle = np.clip(dot / (pred_mag * gt_mag), -1, 1)
        angle_err = np.degrees(np.arccos(cos_angle))
        grad_metrics = {
            "dx_mae": float(np.mean(np.abs(dx_pred - dx_targets))),
            "dy_mae": float(np.mean(np.abs(dy_pred - dy_targets))),
            "normal_angle_mean_deg": float(np.mean(angle_err)),
            "normal_angle_p99_deg": float(np.percentile(angle_err, 99)),
            "normal_angle_max_deg": float(np.max(angle_err)),
        }

    metrics = {
        "architecture": "plane",
        "total_params": total_p,
        "trainable_params": train_p,
        "size_int8_kb": train_p / 1024,
        "raw_size_bytes": n_points * 2,
        "elev_min_m": elev_min,
        "elev_max_m": elev_max,
        "elev_range_m": elev_range,
        "rmse_m": rmse,
        "mae_m": mae,
        "max_error_m": max_err,
        "p99_error_m": p99_err,
        "psnr_db": psnr,
        "train_steps": args.steps,
        "stopped_step": stopped_step,
        "best_batch_rmse": best_rmse if best_rmse < float("inf") else None,
        "best_step": best_step,
        "train_time_s": train_time,
        "resolution": args.resolution,
        "resolution_w": model.resolution_w,
        "features": model.features,
        "mlp_hidden_dim": model.mlp_hidden_dim,
        "num_outputs": model.num_outputs,
        "water_lambda": args.water_lambda,
        "grad_lambda": args.grad_lambda,
        "grad_scale": grad_scale if args.learn_grad else None,
        "grid_size_kb": model.grid_size_kb(),
        "mlp_size_kb": model.mlp_size_kb(),
        **water_metrics,
        **grad_metrics,
    }

    print(f"\n  Results:")
    print(f"    RMSE:        {rmse:.2f} m")
    print(f"    MAE:         {mae:.2f} m")
    print(f"    Max Error:   {max_err:.2f} m (P99: {p99_err:.2f} m)")
    print(f"    PSNR:        {psnr:.1f} dB")
    print(f"    Train time:  {train_time:.1f}s")
    print(f"    Grid:        {model.grid_size_kb():.1f} KB, MLP: {model.mlp_size_kb():.1f} KB")
    if water_metrics:
        print(
            f"    Water:       IoU={water_metrics['water_iou']:.3f}, "
            f"prec={water_metrics['water_precision']:.3f}, "
            f"recall={water_metrics['water_recall']:.3f}"
        )
        cls_parts = []
        for name in ["ocean", "lake", "river"]:
            if f"{name}_recall" in water_metrics:
                cls_parts.append(f"{name}={water_metrics[f'{name}_recall']:.3f}")
        if cls_parts:
            print(f"    Recall:      {', '.join(cls_parts)}")
    if grad_metrics:
        print(
            f"    Normals:     mean={grad_metrics['normal_angle_mean_deg']:.2f} deg, "
            f"P99={grad_metrics['normal_angle_p99_deg']:.2f} deg, "
            f"max={grad_metrics['normal_angle_max_deg']:.2f} deg"
        )
    if stopped_step < args.steps:
        print(f"    Stopped:     step {stopped_step}/{args.steps} (patience {args.patience})")
    else:
        print(f"    Completed:   {args.steps} steps")
    print(f"    Best ~RMSE:  {best_rmse:.2f}m at step {best_step}")

    return model, metrics, pred_map, error_map, water_pred_map


# -- Diff Maps -----------------------------------------------------------------


def save_diff_map(error_map, path, vmax=20.0):
    """Save diverging red/blue elevation difference map as PNG."""
    from PIL import Image

    h, w = error_map.shape
    normalized = np.clip(error_map / vmax * 127 + 128, 0, 255).astype(np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.clip(
        (normalized.astype(np.int16) - 128) * 2 + 128, 0, 255
    ).astype(np.uint8)
    rgb[:, :, 2] = np.clip(
        (128 - normalized.astype(np.int16)) * 2 + 128, 0, 255
    ).astype(np.uint8)
    rgb[:, :, 1] = (
        (255 - np.abs(normalized.astype(np.int16) - 128) * 2)
        .clip(0, 255)
        .astype(np.uint8)
    )
    Image.fromarray(rgb, "RGB").save(path)


def save_water_diff_map(water_pred, water_gt, path):
    """Save water classification diff: TP=green, FP=red, FN=blue, TN=dark gray."""
    from PIL import Image

    h, w = water_pred.shape
    rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # TN = dark gray
    gt = water_gt > 0
    tp = (water_pred == 1) & gt
    fp = (water_pred == 1) & ~gt
    fn = (water_pred == 0) & gt
    rgb[tp] = [0, 200, 0]    # green
    rgb[fp] = [220, 0, 0]    # red
    rgb[fn] = [0, 0, 220]    # blue
    Image.fromarray(rgb, "RGB").save(path)


# -- Export --------------------------------------------------------------------


def export(model, metrics, output_dir: Path, num_ref: int = 1024):
    """Export trained model to binary format for Vulkan compute shader.

    Produces weights.bin, meta.json, and reference.bin in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = metrics["resolution"]
    resolution_w = metrics["resolution_w"]
    features = metrics["features"]
    mlp_hidden_dim = metrics["mlp_hidden_dim"]
    elev_min = metrics["elev_min_m"]
    elev_max = metrics["elev_max_m"]
    elev_range = metrics["elev_range_m"]

    sd = model.state_dict()

    assert features % 4 == 0, f"Features ({features}) must be divisible by 4 for uint32 packing"

    # Grid: NCHW (1, F, H, W) -> HWC (H, W, F) for coalesced GPU access
    grid = sd["grid"].squeeze(0)
    grid_hwc = grid.permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32)
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

    # Pack 4 uint8 into each uint32 (matches GLSL unpackUnorm4x8)
    grid_packed = grid_u8.view(np.uint32)
    assert grid_packed.shape == (resolution, resolution_w, features // 4)
    grid_uints = grid_packed.size

    dequant_scale = ch_range.astype(np.float32)
    dequant_offset = ch_min.astype(np.float32)

    # Quantization error stats
    grid_dequant = grid_u8.astype(np.float32) / 255.0 * ch_range + ch_min
    quant_err = np.abs(grid_flat - grid_dequant.reshape(-1, features))
    print(f"  Grid: {grid_uints} uint32s ({grid_uints * 4} bytes)")
    print(f"  Quantization error: mean={quant_err.mean():.6f}, max={quant_err.max():.6f}")

    # MLP weights: W0 (hidden, features), b0 (hidden),
    # W1 transposed to (hidden, num_outputs) for contiguous GPU access, b1 (num_outputs)
    w0 = sd["mlp.0.weight"].cpu().numpy().astype(np.float32)
    b0 = sd["mlp.0.bias"].cpu().numpy().astype(np.float32)
    w1 = sd["mlp.2.weight"].cpu().numpy().astype(np.float32).T  # (out, hid) -> (hid, out)
    b1 = sd["mlp.2.bias"].cpu().numpy().astype(np.float32)
    mlp_parts = [w0.ravel(), b0.ravel(), w1.ravel(), b1.ravel()]
    mlp_floats = sum(p.size for p in mlp_parts)
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
    print(f"  weights.bin: {total_bytes:,} bytes")

    # Write metadata
    tile_name = metrics.get("tile", "unknown")
    num_outputs = metrics.get("num_outputs", model.num_outputs)
    meta = {
        "arch": "plane",
        "tile": tile_name,
        "resolution": resolution,
        "resolution_w": resolution_w,
        "features": features,
        "mlp_hidden_dim": mlp_hidden_dim,
        "num_outputs": num_outputs,
        "elev_min": elev_min,
        "elev_max": elev_max,
        "elev_range": elev_range,
        "grad_scale": metrics.get("grad_scale"),
        "quantized": True,
        "grid_uints": int(grid_uints),
        "mlp_floats": mlp_floats,
        "total_params": model.param_count(),
        "source": metrics.get("source", "train_plane.py"),
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Reference coords + expected elevations for GPU verification
    side = int(np.ceil(np.sqrt(num_ref)))
    n = side * side
    coords_1d = np.linspace(-1, 1, side, dtype=np.float32)
    xx, yy = np.meshgrid(coords_1d, coords_1d, indexing="xy")
    ref_coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    model_device = next(model.parameters()).device
    with torch.no_grad():
        ref_out = model(torch.from_numpy(ref_coords).to(model_device))
        pred_norm = ref_out[:, OUT_ELEV].cpu().numpy()
    pred_meters = (pred_norm + 1.0) / 2.0 * elev_range + elev_min

    ref_path = output_dir / "reference.bin"
    with open(ref_path, "wb") as f:
        f.write(struct.pack("I", n))
        ref_coords.tofile(f)
        pred_meters.astype(np.float32).tofile(f)

    print(f"  -> {output_dir}/")


# -- Save / Load ---------------------------------------------------------------


def save_model(model, metrics, path):
    save_dict = {"state_dict": model.state_dict()}
    for k, v in metrics.items():
        save_dict[k] = v
    torch.save(save_dict, path)


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Feature plane INR terrain trainer + exporter",
    )

    # Tile selection
    tile_group = parser.add_mutually_exclusive_group(required=True)
    tile_group.add_argument("--tile", type=str, help="Single tile name (e.g., n47w122)")
    tile_group.add_argument(
        "--tile-dir", type=str,
        help="Directory of GeoTIFF tiles to batch train (e.g., glo30)",
    )

    # Model architecture
    parser.add_argument("--resolution", type=int, default=256, help="Grid height (lat rows)")
    parser.add_argument("--resolution-w", type=int, default=0, help="Grid width (0 = auto from tile aspect)")
    parser.add_argument("--features", type=int, default=12, help="Features per grid cell")
    parser.add_argument("--mlp-width", type=int, default=48, help="MLP hidden dimension")

    # Training
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=256 * 1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--warmup", type=int, default=0, help="Linear LR warmup steps")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm (0=disabled)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (0=disabled)")
    parser.add_argument("--l4-weight", type=float, default=0.0, help="L4 loss penalty weight")
    parser.add_argument("--slope-weight", type=float, default=0.0, help="Slope-weighted sampling blend")
    parser.add_argument("--water-lambda", type=float, default=0.01, help="Water loss weight (0=disable water training)")
    parser.add_argument("--water-oversample", type=float, default=0.0, help="Target water fraction per batch (0=disable)")
    parser.add_argument("--boundary-k", type=int, default=7, help="Boundary band width in pixels (0=disable)")
    parser.add_argument("--boundary-weight", type=float, default=3.0, help="Max BCE weight at water boundary")
    parser.add_argument("--w-ocean", type=float, default=0.1, help="Per-pixel weight for ocean class")
    parser.add_argument("--w-lake", type=float, default=1.0, help="Per-pixel weight for lake class")
    parser.add_argument("--w-river", type=float, default=5.0, help="Per-pixel weight for river class")
    parser.add_argument("--grad-lambda", type=float, default=0.01,
                        help="Weight for gradient (dx/dy) MSE loss (>0 enables 4-output model, 0 for 2-output)")
    parser.add_argument("--amp", action="store_true", help="Automatic mixed precision (FP16)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (reduce-overhead on CUDA, default on ROCm)")

    # Output
    parser.add_argument("--diff", action="store_true", help="Save elevation diff map PNG")
    parser.add_argument("--no-export", action="store_true", help="Skip binary export for viewer")
    parser.add_argument("--cleanup-tile", action="store_true",
                        help="Delete the .tif after model saves (sweep mode; keeps WBM)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if assets/planes/<tile>/meta.json already matches current "
                             "{resolution, features, mlp_width, num_outputs}. Single-tile mode only.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    args.learn_grad = args.grad_lambda > 0

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

    maps_dir = Path(args.maps_dir) if args.maps_dir else DEFAULT_MAPS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip-existing short-circuit (before download). Single-tile mode only;
    # batch mode already has its own per-tile skip-if-exists later in the loop.
    if args.skip_existing and args.tile is not None:
        short = args.tile.lower()
        meta_path = DEFAULT_PLANES_DIR / short / "meta.json"
        expected_outputs = 4 if args.learn_grad else 2
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    existing = json.load(f)
                if (existing.get("resolution") == args.resolution
                        and existing.get("features") == args.features
                        and existing.get("mlp_hidden_dim") == args.mlp_width
                        and existing.get("num_outputs") == expected_outputs):
                    print(f"Skipping {short}: matching meta.json at {meta_path}")
                    return
                print(f"  Existing meta.json for {short} differs from current config; retraining")
            except (OSError, json.JSONDecodeError) as e:
                print(f"  Could not read existing meta.json (will retrain): {e}")

    # -- Tile discovery --
    batch_mode = False
    if args.tile_dir:
        tile_dir = Path(args.tile_dir)
        if not tile_dir.is_absolute():
            tile_dir = maps_dir / tile_dir
        tile_paths = sorted(tile_dir.glob("*_DEM.tif"))
        if not tile_paths:
            print(f"No *_DEM.tif tiles found in {tile_dir}")
            return
        batch_mode = True
    else:
        try:
            tile_path, _ = ensure_tile(args.tile, maps_dir / "glo30")
        except ValueError as e:
            print(f"Bad tile name: {e}")
            return
        except urllib.error.HTTPError as e:
            print(f"Tile not in S3: {args.tile} (HTTP {e.code})")
            return
        tile_paths = [tile_path]

    tile_names = [extract_tile_name(p) for p in tile_paths]
    print(f"Tiles: {tile_names}")
    print(f"Output: {output_dir}")
    print()

    all_results = []
    orig_res_w = args.resolution_w

    for tile_path in tile_paths:
        tile_name = extract_tile_name(tile_path)

        elevation = load_geotiff(str(tile_path))
        h, w = elevation.shape

        if elevation.max() - elevation.min() < 1e-6:
            print(f"Skipping {tile_name} (constant elevation: ocean)")
            continue

        # Load WBM
        wbm_path = find_wbm_path(maps_dir, tile_name)
        wbm = load_wbm(wbm_path)
        if wbm is None:
            print(f"  Warning: no WBM for {tile_name}")

        # Auto-size for rectangular tiles
        if orig_res_w == 0 and h != w:
            _, res_w = compute_plane_resolution(args.resolution, h, w)
            args.resolution_w = res_w
        else:
            args.resolution_w = orig_res_w

        label = config_label(args)

        # Skip if exists (batch mode)
        model_path = output_dir / f"{tile_name}_{label}.pt"
        if batch_mode and model_path.exists():
            print(f"Skipping {tile_name} (exists: {model_path.name})")
            continue

        print(f"{'=' * 60}")
        print(f"TILE: {tile_name} ({h}x{w})")
        print(f"{'=' * 60}")

        model = FeaturePlaneMLP(
            resolution=args.resolution,
            features=args.features,
            mlp_hidden_dim=args.mlp_width,
            resolution_w=args.resolution_w,
            learn_grad=args.learn_grad,
        )

        model, metrics, pred_map, error_map, water_pred_map = train(model, elevation, wbm, args)
        metrics["tile"] = tile_name

        save_model(model, metrics, str(model_path))
        saved_bytes = model_path.stat().st_size
        print(f"    Saved: {model_path.name} ({saved_bytes / 1024:.1f} KB)")

        # Sweep cleanup: delete the .tif once the model is durably on disk.
        # WBM stays: KB-sized, useful for any re-runs.
        if args.cleanup_tile and not batch_mode and saved_bytes > MIN_MODEL_BYTES:
            try:
                tile_path.unlink()
                print(f"    Cleaned: {tile_path.name}")
            except OSError as e:
                print(f"    Cleanup failed (non-fatal): {e}")

        if args.diff:
            diff_path = output_dir / f"{tile_name}_{label}_elev_diff.png"
            save_diff_map(error_map, str(diff_path))
            print(f"    Diff:  {diff_path.name}")
            if water_pred_map is not None and wbm is not None:
                water_diff_path = output_dir / f"{tile_name}_{label}_water_diff.png"
                save_water_diff_map(water_pred_map, wbm[2], str(water_diff_path))
                print(f"    Diff:  {water_diff_path.name}")

        if not args.no_export:
            planes_dir = DEFAULT_PLANES_DIR / tile_name
            print(f"  Exporting to {planes_dir}/")
            export(model, metrics, planes_dir)

        all_results.append(metrics)

    # Summary
    if all_results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        for r in all_results:
            water_str = f" | IoU={r['water_iou']:.3f}" if "water_iou" in r else ""
            print(
                f"  {r['tile']}: RMSE={r['rmse_m']:.2f}m | "
                f"MaxErr={r['max_error_m']:.2f}m (P99: {r['p99_error_m']:.2f}m) | "
                f"PSNR={r['psnr_db']:.1f}dB | {r['size_int8_kb']:.1f}KB INT8 | "
                f"{r['train_time_s']:.1f}s{water_str}"
            )

        if len(all_results) > 1:
            args.resolution_w = orig_res_w
            results_path = output_dir / f"{config_label(args)}_results.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
