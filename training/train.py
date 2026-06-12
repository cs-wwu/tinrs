"""
Unified INR trainer for terrain data.

Examples:
    python train.py --arch siren --tile n48w122 --layers 5 --width 256 --omega 60
    python train.py --arch bacon --tile n48w122 --layers 5 --width 256 --max-freq 128
    python train.py --arch hash  --tile n48w122 --hash-log2 14
    python train.py --arch wire  --tile n48w122 --layers 5 --width 256 --omega 20 --sigma 5
    python train.py --arch siren --tile n48w122 --layers 5 --width 256 --omega 60 --amp
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from models import BACON, SIREN, FeaturePlaneMLP, GaborNet, HashMLP, MultiResGridMLP

# -- Data Loading --------------------------------------------------------------


def load_hgt_png(path: str) -> np.ndarray:
    """Load .hgt.png (Gray16 PNG) -> (3601, 3601) float32 elevation in meters."""
    from PIL import Image

    img = Image.open(path)
    assert img.mode == "I;16", f"Expected 16-bit grayscale, got {img.mode}"
    assert img.size == (3601, 3601), f"Expected 3601x3601, got {img.size}"
    data = np.array(img, dtype=np.uint16)
    return data.astype(np.float32) - 32768.0


def load_geotiff(path: str) -> np.ndarray:
    """Load GeoTIFF (GLO-30) -> (rows, cols) float32 elevation in meters.

    GLO-30 tiles are Cloud Optimized GeoTIFF, float32, EGM2008 vertical datum.
    Dimensions vary by latitude: 3600 rows, 3600/2400/1800/1200/720 cols.
    """
    from PIL import Image

    img = Image.open(path)
    assert img.mode == "F", f"Expected float32 GeoTIFF (mode 'F'), got {img.mode}"
    data = np.array(img, dtype=np.float32)
    # GLO-30 nodata is -32767.0 (ocean cells); clamp to 0m
    data[data < -1000] = 0.0
    return data


def make_coordinate_grid(height: int, width: int = 0) -> np.ndarray:
    """Create (height*width, 2) grid of coordinates in [-1, 1].

    Both axes span [-1, 1] regardless of aspect ratio. For rectangular tiles
    (e.g., 3600x1800 at 65N), the longitude axis is sampled at fewer points.
    """
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
    """Map short name like 'n48w123' to GLO-30 GeoTIFF path, or None."""
    import re

    m = re.match(r"^([ns])(\d+)([ew])(\d+)$", short_name.lower())
    if not m:
        return None
    ns = m.group(1).upper()
    lat = int(m.group(2))
    ew = m.group(3).upper()
    lon = int(m.group(4))
    filename = f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
    path = glo30_dir / filename
    return path if path.exists() else None


def extract_tile_name(path: Path) -> str:
    """Extract short tile name from any supported file path.

    'n47w122.hgt.png' -> 'n47w122'
    'Copernicus_DSM_COG_10_N48_00_W123_00_DEM.tif' -> 'n48w123'
    """
    import re

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


# -- Utilities -----------------------------------------------------------------


def estimate_hash_compressed_kb(encoding):
    """Estimate compressed size of INT8-quantized hash features via entropy."""
    total_bits = 0.0
    with torch.no_grad():
        for table in encoding.hash_tables:
            flat = table.reshape(-1).float()
            vmin, vmax = flat.min(), flat.max()
            if vmax - vmin < 1e-8:
                continue
            q = ((flat - vmin) / (vmax - vmin) * 255).round().clamp(0, 255)
            hist = torch.histc(q, bins=256, min=0, max=255)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            entropy_bits = -(probs * torch.log2(probs)).sum().item()
            total_bits += flat.numel() * entropy_bits
    return total_bits / 8.0 / 1024


# -- Model Construction --------------------------------------------------------


def build_model(args):
    if args.arch == "siren":
        return SIREN(hidden_dim=args.width, num_layers=args.layers, omega=args.omega)
    elif args.arch == "bacon":
        return BACON(
            hidden_dim=args.width, num_layers=args.layers, max_freq=args.max_freq
        )
    elif args.arch == "hash":
        latent_dim = args.latent_dim if args.latent_dim > 0 else None
        return HashMLP(
            num_levels=args.hash_levels,
            features_per_level=args.hash_features,
            log2_hashmap_size=args.hash_log2,
            base_resolution=args.hash_base_res,
            finest_resolution=args.hash_finest_res,
            mlp_hidden_dim=args.mlp_width,
            mlp_num_layers=args.mlp_layers,
            latent_dim=latent_dim,
        )
    elif args.arch == "wire":
        return GaborNet(
            hidden_dim=args.width,
            num_layers=args.layers,
            omega=args.omega,
            sigma=args.sigma,
        )
    elif args.arch == "plane":
        return FeaturePlaneMLP(
            resolution=args.plane_resolution,
            features=args.plane_features,
            mlp_hidden_dim=args.plane_mlp_width,
            resolution_w=getattr(args, "plane_resolution_w", 0),
        )
    elif args.arch == "mgrid":
        return MultiResGridMLP(
            num_levels=args.mgrid_levels,
            features_per_level=args.mgrid_features,
            base_resolution=args.mgrid_base_res,
            finest_resolution=args.mgrid_finest_res,
            mlp_hidden_dim=args.mgrid_mlp_width,
            mlp_num_layers=args.mgrid_mlp_layers,
        )


def build_optimizer(model, args):
    if args.arch == "hash":
        if model.encoding.latent_dim is not None:
            # SHACIRA: separate LRs for latents, decoder, entropy model, MLP
            return torch.optim.Adam(
                [
                    {"params": model.encoding.hash_tables.parameters(), "lr": 0.02},
                    {"params": model.encoding.decoder.parameters(), "lr": 0.01},
                    {"params": model.encoding.bit_estimator.parameters(), "lr": 0.001},
                    {"params": model.mlp.parameters(), "lr": args.lr},
                ]
            )
        # Standard: hash tables need higher LR (sparse gradient updates)
        return torch.optim.Adam(
            [
                {"params": model.encoding.parameters(), "lr": args.lr * 100},
                {"params": model.mlp.parameters(), "lr": args.lr},
            ]
        )
    elif args.arch == "bacon":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.arch == "plane":
        # Grid gets 10x LR boost (less than hash's 100x because grid has dense
        # gradients: every point updates 4 cells, no collision noise)
        return torch.optim.Adam(
            [
                {"params": [model.grid], "lr": args.lr * 10},
                {"params": model.mlp.parameters(), "lr": args.lr},
            ]
        )
    elif args.arch == "mgrid":
        # Same 10x grid LR boost as feature plane (dense gradients from grid_sample)
        return torch.optim.Adam(
            [
                {"params": model.encoding.parameters(), "lr": args.lr * 10},
                {"params": model.mlp.parameters(), "lr": args.lr},
            ]
        )
    else:
        return torch.optim.Adam(model.parameters(), lr=args.lr)


def config_label(args):
    """Short string for filenames. Includes all params that affect results."""
    parts = [args.arch]

    if args.arch in ("siren", "bacon", "wire"):
        parts.append(f"{args.layers}x{args.width}")
    if args.arch in ("siren", "wire"):
        parts.append(f"w{int(args.omega)}")
    if args.arch == "wire":
        parts.append(f"s{args.sigma}")
    if args.arch == "bacon":
        parts.append(f"f{int(args.max_freq)}")
    if args.arch == "hash":
        parts.append(f"L{args.hash_levels}_H{args.hash_log2}")
        if args.latent_dim > 0:
            parts.append(f"lat{args.latent_dim}")
            if args.entropy_lambda != 1e-4:
                parts.append(f"ent{args.entropy_lambda}")
        if args.hash_reg_lambda > 0:
            parts.append(f"reg{args.hash_reg_lambda}")
    if args.arch == "plane":
        res_w = getattr(args, "plane_resolution_w", 0) or args.plane_resolution
        if res_w != args.plane_resolution:
            parts.append(f"{args.plane_resolution}x{res_w}x{args.plane_features}")
        else:
            parts.append(f"{args.plane_resolution}x{args.plane_features}")
        if args.plane_mlp_width != 32:
            parts.append(f"mlp{args.plane_mlp_width}")
    if args.arch == "mgrid":
        parts.append(
            f"{args.mgrid_levels}L_{args.mgrid_base_res}-{args.mgrid_finest_res}_F{args.mgrid_features}"
        )
        if args.mgrid_mlp_width != 32:
            parts.append(f"mlp{args.mgrid_mlp_width}")
        if args.mgrid_mlp_layers != 1:
            parts.append(f"d{args.mgrid_mlp_layers}")

    if args.steps != 20000:
        parts.append(f"{args.steps // 1000}k")
    if args.l4_weight > 0:
        parts.append(f"l4_{args.l4_weight}")
    if args.slope_weight > 0:
        parts.append(f"sw_{args.slope_weight}")
    if args.grad_clip > 0:
        parts.append(f"gc_{args.grad_clip}")
    if args.warmup > 0:
        parts.append(f"wu_{args.warmup}")

    return "_".join(parts)


# -- Training ------------------------------------------------------------------


def train(model, elevation, args, png_size_bytes=0):
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
    targets_t = torch.from_numpy(targets).to(args.device).unsqueeze(-1)
    n_points = coords_t.shape[0]

    # Gradient-weighted sampling: oversample steep terrain
    if args.slope_weight > 0:
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_x**2 + grad_y**2).ravel()
        # Blend uniform + slope-weighted: (1-w)*uniform + w*slope
        uniform = np.ones_like(slope)
        blend = (
            1 - args.slope_weight
        ) * uniform / uniform.sum() + args.slope_weight * slope / slope.sum()
        sample_weights = torch.from_numpy(blend.astype(np.float32)).to(args.device)
        print(f"  Slope-weighted sampling: weight={args.slope_weight}")
    else:
        sample_weights = None

    model = model.to(args.device)
    total_p = model.param_count()
    train_p = model.trainable_param_count()
    frozen_p = total_p - train_p

    # -- Print model info --
    print(
        f"  Model: {args.arch.upper()} ({total_p:,} params, "
        f"{train_p:,} trainable, {frozen_p:,} frozen)"
    )
    print(
        f"  Size: {train_p / 1024:.1f} KB INT8 trainable, "
        f"{total_p / 1024:.1f} KB INT8 total"
    )
    if args.arch == "hash":
        ht_entries = (
            model.encoding.hashmap_size
            * model.encoding.features_per_level
            * model.encoding.num_levels
        )
        print(
            f"  Hash: {model.num_levels} levels, 2^{model.log2_hashmap_size} entries, "
            f"{model.features_per_level}F/level, {ht_entries / 1024:.1f} KB hash (INT8)"
        )
        print(f"  MLP: {model.mlp_num_layers}x{model.mlp_hidden_dim}")
        print(
            f"  Resolutions: {model.encoding.resolutions[0]} -> {model.encoding.resolutions[-1]}"
        )
        if model.encoding.latent_dim is not None:
            print(
                f"  SHACIRA: latent_dim={model.encoding.latent_dim}, "
                f"lambda={args.entropy_lambda}"
            )
    elif args.arch == "bacon":
        freq_info = model.get_freq_info()
        print(f"  Max freq: {model.max_freq}, per-layer: {model.freq_per_layer:.1f}")
        print(
            f"  Layer freqs: "
            + ", ".join(f"L{i}: cum={cf:.1f}" for i, (_, cf) in enumerate(freq_info))
        )
    elif args.arch == "plane":
        print(
            f"  Plane: {model.resolution}x{model.resolution_w} grid, "
            f"{model.features} features/cell"
        )
        print(f"  Grid: {model.grid_size_kb():.1f} KB INT8")
        print(f"  MLP: {model.features} -> {model.mlp_hidden_dim} -> 1")
    elif args.arch == "mgrid":
        enc = model.encoding
        print(
            f"  Multi-res grid: {enc.num_levels} levels, "
            f"{enc.base_resolution} -> {enc.finest_resolution} "
            f"(growth {enc.growth_factor:.2f}x)"
        )
        print(
            f"  Features/level: {enc.features_per_level}, "
            f"MLP input: {enc.output_dim}"
        )
        print(f"  Grid: {model.grid_size_kb():.1f} KB INT8")
        print(
            f"  MLP: {enc.output_dim} -> {model.mlp_hidden_dim} -> 1 "
            f"({model.mlp_size_kb():.1f} KB)"
        )
        for i, res in enumerate(enc.resolutions):
            level_kb = res * res * enc.features_per_level / 1024
            print(f"    L{i}: {res}x{res} ({level_kb:.1f} KB)")
    if args.arch == "hash" and args.hash_reg_lambda > 0:
        print(f"  Feature L1 reg: lambda={args.hash_reg_lambda}")
    print(
        f"  Data: {h}x{w} = {n_points:,} points, "
        f"range [{elev_min:.0f}m, {elev_max:.0f}m] ({elev_range:.0f}m)"
    )
    extras = ""
    if args.grad_clip > 0:
        extras += f", grad_clip={args.grad_clip}"
    if args.warmup > 0:
        extras += f", warmup={args.warmup}"
    if args.amp:
        extras += ", AMP"
    print(
        f"  Training: {args.steps} steps, batch {args.batch_size:,}, lr {args.lr}{extras}"
    )

    optimizer = build_optimizer(model, args)
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

    use_multiscale = args.arch == "bacon" and not args.no_multiscale_loss
    use_shacira = (
        args.arch == "hash" and model.encoding.latent_dim is not None
    )
    use_hash_reg = (
        args.arch == "hash"
        and args.hash_reg_lambda > 0
        and model.encoding.latent_dim is None
    )

    start_time = time.time()
    model.train()
    entropy_loss_val = 0.0
    hash_reg_val = 0.0
    sga_temp = 0.0
    anneal_end = int(args.steps * args.sga_anneal_fraction) if use_shacira else 0

    log_interval = max(args.steps // 10, 500)
    best_rmse = float("inf")
    best_step = 0
    patience_counter = 0
    stopped_step = args.steps

    for step in range(args.steps):
        # SHACIRA: anneal SGA temperature
        if use_shacira:
            if step < anneal_end:
                t = step / max(anneal_end, 1)
                sga_temp = args.sga_temp_init + (
                    args.sga_temp_final - args.sga_temp_init
                ) * t
            else:
                sga_temp = 0.0  # hard STE for final portion
            model.encoding._sga_temperature = sga_temp

        bs = min(args.batch_size, n_points)
        if sample_weights is not None:
            idx = torch.multinomial(sample_weights, bs, replacement=True)
        else:
            idx = torch.randint(0, n_points, (bs,), device=args.device)
        batch_coords = coords_t[idx]
        batch_targets = targets_t[idx]

        with torch.amp.autocast(args.device, enabled=args.amp):
            if use_multiscale:
                layer_outputs = model(batch_coords, return_all_layers=True)
                loss = sum(
                    nn.functional.mse_loss(lo, batch_targets) for lo in layer_outputs
                )
                final_pred = layer_outputs[-1]
            else:
                pred = model(batch_coords)
                loss = nn.functional.mse_loss(pred, batch_targets)
                final_pred = pred

            # L4 penalty: heavily penalizes large individual errors
            if args.l4_weight > 0:
                loss = loss + args.l4_weight * torch.mean((final_pred - batch_targets) ** 4)

            # SHACIRA: entropy regularization
            if use_shacira and args.entropy_lambda > 0:
                ent_loss = model.encoding.compute_entropy_loss()
                loss = loss + args.entropy_lambda * ent_loss
                entropy_loss_val = ent_loss.item()

            # Hash feature L1 regularization (CAwa-NeRF style compressibility)
            if use_hash_reg:
                l1 = sum(t.abs().mean() for t in model.encoding.hash_tables)
                loss = loss + args.hash_reg_lambda * l1
                hash_reg_val = l1.item()

        optimizer.zero_grad()
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
            with torch.no_grad():
                final_mse = nn.functional.mse_loss(final_pred, batch_targets).item()
            rmse_m = math.sqrt(final_mse * (elev_range / 2.0) ** 2)
            extra = ""
            if use_shacira:
                extra = f", ent={entropy_loss_val:.3f}bpp, T={sga_temp:.3f}"
            if use_hash_reg:
                extra += f", l1={hash_reg_val:.5f}"
            print(
                f"    Step {step + 1:>5}/{args.steps}: loss={loss.item():.6f}, ~RMSE={rmse_m:.2f}m{extra}"
            )

            # Track best batch RMSE and early stopping
            if rmse_m < best_rmse:
                best_rmse = rmse_m
                best_step = step + 1
                patience_counter = 0
            elif step > 0 and args.patience > 0:
                patience_counter += 1
                if patience_counter >= args.patience:
                    stopped_step = step + 1
                    print(
                        f"    Early stopping: no improvement for {args.patience} checks"
                    )
                    break

    train_time = time.time() - start_time

    # -- Evaluation --
    model.eval()
    if use_shacira:
        model.encoding._sga_temperature = 0.0  # hard rounding for eval
    chunk = 512 * 1024

    with torch.no_grad():
        if use_multiscale:
            num_layers = model.num_layers
            layer_preds = [[] for _ in range(num_layers)]
            for i in range(0, n_points, chunk):
                end = min(i + chunk, n_points)
                los = model(coords_t[i:end], return_all_layers=True)
                for l in range(num_layers):
                    layer_preds[l].append(los[l])
            layer_pred_all = [torch.cat(lp, dim=0) for lp in layer_preds]
            pred_all = layer_pred_all[-1]
        else:
            preds = []
            for i in range(0, n_points, chunk):
                end = min(i + chunk, n_points)
                preds.append(model(coords_t[i:end]))
            pred_all = torch.cat(preds, dim=0)

    pred_meters = (
        pred_all.squeeze(-1).cpu().numpy() + 1.0
    ) / 2.0 * elev_range + elev_min
    pred_map = pred_meters.reshape(h, w)
    error_map = pred_map - elevation

    rmse = math.sqrt(np.mean(error_map**2))
    mae = float(np.mean(np.abs(error_map)))
    max_err = float(np.max(np.abs(error_map)))
    p99_err = float(np.percentile(np.abs(error_map), 99))
    mse_val = np.mean(error_map**2)
    psnr = (
        20.0 * math.log10(elev_range / math.sqrt(mse_val))
        if mse_val > 0
        else float("inf")
    )

    metrics = {
        "architecture": args.arch,
        "total_params": total_p,
        "trainable_params": train_p,
        "frozen_params": frozen_p,
        "size_int8_kb": train_p / 1024,
        "raw_size_bytes": n_points * 2,
        "png_size_bytes": png_size_bytes,
        "compression_vs_raw": n_points * 2 / train_p,
        "compression_vs_png": png_size_bytes / train_p if png_size_bytes else None,
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
    }

    # Architecture-specific metadata
    if args.arch == "siren":
        metrics.update(hidden_dim=args.width, num_layers=args.layers, omega=args.omega)
    elif args.arch == "wire":
        metrics.update(
            hidden_dim=args.width,
            num_layers=args.layers,
            omega=args.omega,
            sigma=args.sigma,
        )
    elif args.arch == "hash":
        metrics.update(
            num_levels=args.hash_levels,
            features_per_level=args.hash_features,
            log2_hashmap_size=args.hash_log2,
            mlp_hidden_dim=args.mlp_width,
            mlp_num_layers=args.mlp_layers,
        )
        if model.encoding.latent_dim is not None:
            comp_kb = model.compressed_size_kb()
            metrics.update(
                latent_dim=args.latent_dim,
                entropy_lambda=args.entropy_lambda,
                compressed_size_kb=comp_kb,
            )
        else:
            comp_kb = estimate_hash_compressed_kb(model.encoding)
            metrics["hash_compressed_kb"] = comp_kb
            if args.hash_reg_lambda > 0:
                metrics["hash_reg_lambda"] = args.hash_reg_lambda
    elif args.arch == "bacon":
        metrics.update(
            hidden_dim=args.width, num_layers=args.layers, max_freq=args.max_freq
        )
        freq_info = model.get_freq_info()
        layer_metrics = []
        for l in range(model.num_layers):
            lp = layer_pred_all[l].squeeze(-1).cpu().numpy()
            lm = (lp + 1.0) / 2.0 * elev_range + elev_min
            le = lm.reshape(h, w) - elevation
            l_rmse = math.sqrt(np.mean(le**2))
            l_mae = float(np.mean(np.abs(le)))
            l_max = float(np.max(np.abs(le)))
            l_mse = np.mean(le**2)
            l_psnr = (
                20.0 * math.log10(elev_range / math.sqrt(l_mse))
                if l_mse > 0
                else float("inf")
            )
            _, cf = freq_info[l]
            layer_metrics.append(
                {
                    "layer": l,
                    "cumulative_freq": cf,
                    "rmse_m": l_rmse,
                    "mae_m": l_mae,
                    "max_error_m": l_max,
                    "psnr_db": l_psnr,
                }
            )
        metrics["layer_metrics"] = layer_metrics
    elif args.arch == "plane":
        metrics.update(
            resolution=args.plane_resolution,
            resolution_w=model.resolution_w,
            features=args.plane_features,
            mlp_hidden_dim=args.plane_mlp_width,
            grid_size_kb=model.grid_size_kb(),
            mlp_size_kb=model.mlp_size_kb(),
        )
    elif args.arch == "mgrid":
        metrics.update(
            num_levels=args.mgrid_levels,
            features_per_level=args.mgrid_features,
            base_resolution=args.mgrid_base_res,
            finest_resolution=args.mgrid_finest_res,
            mlp_hidden_dim=args.mgrid_mlp_width,
            mlp_num_layers=args.mgrid_mlp_layers,
            grid_size_kb=model.grid_size_kb(),
            mlp_size_kb=model.mlp_size_kb(),
            resolutions=model.encoding.resolutions,
        )

    # -- Print results --
    print(f"\n  Results:")
    print(f"    RMSE:        {rmse:.2f} m")
    print(f"    MAE:         {mae:.2f} m")
    print(f"    Max Error:   {max_err:.2f} m (P99: {p99_err:.2f} m)")
    print(f"    PSNR:        {psnr:.1f} dB")
    print(f"    Train time:  {train_time:.1f}s")
    comp_str = f"{metrics['compression_vs_raw']:.1f}:1 vs raw"
    if metrics["compression_vs_png"]:
        comp_str += f", {metrics['compression_vs_png']:.1f}:1 vs PNG"
    print(f"    Compression: {comp_str}")
    if use_shacira:
        comp_kb = metrics["compressed_size_kb"]
        print(
            f"    SHACIRA:     {comp_kb:.1f} KB compressed "
            f"(vs {train_p / 1024:.1f} KB INT8, "
            f"{train_p / 1024 / comp_kb:.1f}x reduction)"
        )
    if args.arch == "hash" and model.encoding.latent_dim is None:
        comp_kb = metrics.get("hash_compressed_kb")
        if comp_kb:
            int8_kb = sum(t.numel() for t in model.encoding.hash_tables) / 1024
            print(
                f"    Compressed:  {comp_kb:.1f} KB entropy-est "
                f"(from {int8_kb:.1f} KB INT8, {int8_kb / comp_kb:.1f}x)"
            )
    if args.arch == "plane":
        print(
            f"    Grid:        {model.grid_size_kb():.1f} KB, "
            f"MLP: {model.mlp_size_kb():.1f} KB"
        )
    if args.arch == "mgrid":
        print(
            f"    Grid:        {model.grid_size_kb():.1f} KB, "
            f"MLP: {model.mlp_size_kb():.1f} KB"
        )
    if stopped_step < args.steps:
        print(
            f"    Stopped:     step {stopped_step}/{args.steps} (patience {args.patience})"
        )
    else:
        print(f"    Completed:   {args.steps} steps")
    print(f"    Best ~RMSE:  {best_rmse:.2f}m at step {best_step}")

    if args.arch == "bacon" and use_multiscale:
        print(f"\n  Per-layer LOD quality:")
        for lm in layer_metrics:
            lod = (
                "distant"
                if lm["layer"] == 0
                else "close-up"
                if lm["layer"] == model.num_layers - 1
                else "mid-range"
            )
            print(
                f"    L{lm['layer']}: freq={lm['cumulative_freq']:.1f}, "
                f"RMSE={lm['rmse_m']:.2f}m, MaxErr={lm['max_error_m']:.2f}m ({lod})"
            )

    return model, metrics, pred_map, error_map


# -- Saving --------------------------------------------------------------------


def save_difference_map(error_map, path, vmax=20.0):
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


def save_model(model, metrics, path):
    save_dict = {"state_dict": model.state_dict()}
    for k, v in metrics.items():
        if k != "layer_metrics":
            save_dict[k] = v
    torch.save(save_dict, path)


# -- CLI -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Unified INR terrain trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture-specific args:
  siren: --layers, --width, --omega
  bacon: --layers, --width, --max-freq, --no-multiscale-loss
  hash:  --hash-levels, --hash-features, --hash-log2, --mlp-width, --mlp-layers
         SHACIRA: --latent-dim 1 --entropy-lambda 1e-4
         Compressibility: --hash-reg-lambda 1e-5
  wire:  --layers, --width, --omega, --sigma
  plane: --plane-resolution, --plane-features, --plane-mlp-width
  mgrid: --mgrid-levels, --mgrid-features, --mgrid-base-res, --mgrid-finest-res
        """,
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["siren", "bacon", "hash", "wire", "plane", "mgrid"],
    )
    parser.add_argument("--tile", type=str, default=None)
    parser.add_argument(
        "--tile-dir",
        type=str,
        default=None,
        help="Directory of GeoTIFF tiles to batch train (e.g., glo30)",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256 * 1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler", default="cosine", choices=["cosine", "step", "none"]
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.0,
        help="Max gradient norm (0=disabled, try 1.0 for deep nets)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Linear LR warmup steps (try 500-2000 for deep nets)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience in log intervals (0=disabled, default 3)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision (FP16) training",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--maps-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    # Shared MLP args (siren/bacon/wire)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Frequency param (default: 60 siren, 20 wire)",
    )
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="Gabor envelope width (wire)"
    )

    # Loss / sampling
    parser.add_argument(
        "--l4-weight",
        type=float,
        default=0.0,
        help="Weight for L4 loss penalty on large errors (try 0.1)",
    )
    parser.add_argument(
        "--slope-weight",
        type=float,
        default=0.0,
        help="Blend weight for slope-based sampling 0=uniform 1=full slope (try 0.5)",
    )

    # BACON
    parser.add_argument("--max-freq", type=float, default=64.0)
    parser.add_argument("--no-multiscale-loss", action="store_true")

    # Hash
    parser.add_argument("--hash-levels", type=int, default=16)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument(
        "--hash-log2",
        type=int,
        default=14,
        help="log2 hash table size per level (14=16K entries, 512KB total INT8)",
    )
    parser.add_argument("--hash-base-res", type=int, default=16)
    parser.add_argument("--hash-finest-res", type=int, default=3601)
    parser.add_argument("--mlp-width", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=2)

    # SHACIRA compression (hash only)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=0,
        help="SHACIRA latent dim per hash entry (0=off, 1=recommended)",
    )
    parser.add_argument(
        "--entropy-lambda",
        type=float,
        default=1e-4,
        help="Entropy regularization weight (SHACIRA)",
    )
    parser.add_argument(
        "--sga-temp-init",
        type=float,
        default=1.0,
        help="Initial SGA temperature (SHACIRA)",
    )
    parser.add_argument(
        "--sga-temp-final",
        type=float,
        default=0.0,
        help="Final SGA temperature, 0=hard STE (SHACIRA)",
    )
    parser.add_argument(
        "--sga-anneal-fraction",
        type=float,
        default=0.9,
        help="Fraction of training for SGA annealing (SHACIRA)",
    )

    # Feature plane
    parser.add_argument(
        "--plane-resolution",
        type=int,
        default=256,
        help="Feature plane grid resolution (default 256)",
    )
    parser.add_argument(
        "--plane-features",
        type=int,
        default=4,
        help="Features per grid cell (default 4)",
    )
    parser.add_argument(
        "--plane-mlp-width",
        type=int,
        default=32,
        help="Plane decoder MLP hidden dimension (default 32)",
    )
    parser.add_argument(
        "--plane-resolution-w",
        type=int,
        default=0,
        help="Feature plane lon resolution (0 = auto from tile aspect ratio)",
    )

    # Multi-resolution grid
    parser.add_argument(
        "--mgrid-levels",
        type=int,
        default=5,
        help="Number of grid resolution levels (default 5)",
    )
    parser.add_argument(
        "--mgrid-features",
        type=int,
        default=4,
        help="Features per grid cell per level (default 4)",
    )
    parser.add_argument(
        "--mgrid-base-res",
        type=int,
        default=8,
        help="Coarsest grid resolution (default 8)",
    )
    parser.add_argument(
        "--mgrid-finest-res",
        type=int,
        default=128,
        help="Finest grid resolution (default 128)",
    )
    parser.add_argument(
        "--mgrid-mlp-width",
        type=int,
        default=32,
        help="Multi-res grid MLP hidden dimension (default 32)",
    )
    parser.add_argument(
        "--mgrid-mlp-layers",
        type=int,
        default=1,
        help="Multi-res grid MLP hidden layers (default 1)",
    )

    # Hash feature regularization (CAwa-NeRF style)
    parser.add_argument(
        "--hash-reg-lambda",
        type=float,
        default=0.0,
        help="L1 regularization on hash features for compressibility (try 1e-5)",
    )

    args = parser.parse_args()

    # Architecture-specific defaults
    if args.omega is None:
        args.omega = 60.0 if args.arch == "siren" else 20.0

    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Using device: {args.device}")

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
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Tile discovery --
    batch_mode = False
    if args.tile_dir:
        # Batch mode: discover all GeoTIFF tiles in directory
        tile_dir = Path(args.tile_dir)
        if not tile_dir.is_absolute():
            tile_dir = maps_dir / tile_dir
        tile_paths = sorted(tile_dir.glob("*_DEM.tif"))
        if not tile_paths:
            print(f"No *_DEM.tif tiles found in {tile_dir}")
            return
        batch_mode = True
    elif args.tile:
        # Single tile: try .hgt.png first, then glo30/
        tile_path = maps_dir / f"{args.tile}.hgt.png"
        if not tile_path.exists():
            tile_path = find_glo30_tile(maps_dir / "glo30", args.tile)
        if tile_path is None or not tile_path.exists():
            print(f"Tile not found: {args.tile}")
            return
        tile_paths = [tile_path]
    else:
        # Legacy: discover .hgt.png tiles
        tile_paths = sorted(maps_dir.glob("*.hgt.png"))

    if not tile_paths:
        print(f"No tiles found in {maps_dir}")
        return

    tile_names = [extract_tile_name(p) for p in tile_paths]
    print(f"Architecture: {args.arch}")
    print(f"Tiles: {tile_names}")
    print(f"Output: {output_dir}")
    print()

    all_results = []

    # Save original plane_resolution_w for per-tile auto-sizing
    orig_plane_res_w = args.plane_resolution_w

    for tile_path in tile_paths:
        tile_name = extract_tile_name(tile_path)
        file_size = tile_path.stat().st_size

        # Load elevation data (format-specific)
        if tile_path.suffix == ".tif":
            elevation = load_geotiff(str(tile_path))
        else:
            elevation = load_hgt_png(str(tile_path))

        h, w = elevation.shape

        # Skip all-ocean tiles (constant elevation, nothing to learn)
        if elevation.max() - elevation.min() < 1e-6:
            print(f"Skipping {tile_name} (constant elevation {elevation.max():.1f}m: ocean)")
            continue

        # Auto-size plane for rectangular tiles (reset per tile)
        if args.arch == "plane" and orig_plane_res_w == 0 and h != w:
            _, res_w = compute_plane_resolution(args.plane_resolution, h, w)
            args.plane_resolution_w = res_w
        else:
            args.plane_resolution_w = orig_plane_res_w

        # Compute label after auto-sizing (label depends on resolution_w)
        label = config_label(args)

        # Skip if model already exists (batch mode only)
        model_path = output_dir / f"{tile_name}_{label}.pt"
        if batch_mode and model_path.exists():
            print(f"Skipping {tile_name} (exists: {model_path.name})")
            continue

        print(f"{'=' * 60}")
        print(
            f"TILE: {tile_name} ({h}x{w}, "
            f"{file_size / 1024 / 1024:.1f} MB)"
        )
        print(f"{'=' * 60}")

        model = build_model(args)
        model, metrics, pred_map, error_map = train(
            model, elevation, args, png_size_bytes=file_size
        )

        metrics["tile"] = tile_name
        all_results.append(metrics)

        save_model(model, metrics, str(model_path))
        print(f"    Saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

        diff_path = output_dir / f"{tile_name}_{label}_diff.png"
        save_difference_map(error_map, str(diff_path))

    # -- Summary --
    if all_results:
        print(f"\n{'=' * 80}")
        print("SUMMARY")
        print(f"{'=' * 80}")
        for r in all_results:
            print(
                f"  {r['tile']}: {r['architecture']} | "
                f"RMSE={r['rmse_m']:.2f}m | MaxErr={r['max_error_m']:.2f}m (P99: {r['p99_error_m']:.2f}m) | "
                f"PSNR={r['psnr_db']:.1f}dB | {r['size_int8_kb']:.1f}KB INT8 | "
                f"{r['train_time_s']:.1f}s"
            )

        # Use base config label for batch results (per-tile labels may vary)
        args.plane_resolution_w = orig_plane_res_w
        results_path = output_dir / f"{config_label(args)}_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
