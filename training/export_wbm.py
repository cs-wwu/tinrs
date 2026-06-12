"""
Export water body masks (WBM) to 1-bit packed binary for Vulkan compute shader.

Reads GLO-30 .wbm files (zlib-compressed uint8), collapses to binary (water/land),
packs 32 pixels per uint32 (LSB-first, row-major).

Output per tile (to assets/planes/<tile>/):
    wbm.bin - [rows: u32] [cols: u32] [packed uint32 data]

Usage:
    # Batch: export all tiles that have both a planes dir and a WBM file
    python export_wbm.py --wbm-dir assets/maps/glo30/wbm --planes-dir assets/planes

    # Single tile
    python export_wbm.py --wbm-dir assets/maps/glo30/wbm --planes-dir assets/planes --tile n48w123

TODO: Replace this stopgap with INR-based water. Options:
  A) Add water as second output to existing feature plane INR (preferred).
  B) Train separate tiny water INR per tile.
"""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WBM_DIR = PROJECT_ROOT / "assets" / "maps" / "glo30" / "wbm"
DEFAULT_PLANES_DIR = PROJECT_ROOT / "assets" / "planes"


def tile_to_wbm_filename(tile_name: str) -> str:
    """Convert 'n48w123' to Copernicus WBM filename."""
    ns = tile_name[0].upper()
    lat = int(tile_name[1:3])
    ew = tile_name[3].upper()
    lon = int(tile_name[4:7])
    return f"Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_WBM.wbm"


def export_one(wbm_path: Path, output_dir: Path) -> bool:
    """Export a single .wbm file to 1-bit packed wbm.bin. Returns True on success."""
    data = wbm_path.read_bytes()
    if len(data) < 5:
        print(f"  SKIP: file too small ({len(data)} bytes)")
        return False

    rows, cols = struct.unpack("<HH", data[:4])
    print(f"  WBM: {rows}x{cols}, {len(data)} bytes compressed")

    # Decompress zlib payload
    pixels = np.frombuffer(zlib.decompress(data[4:]), dtype=np.uint8)
    expected = rows * cols
    if pixels.size != expected:
        print(f"  SKIP: decompressed {pixels.size} pixels, expected {expected}")
        return False
    pixels = pixels.reshape(rows, cols)

    # Collapse to binary: 0=land, 1=water (any non-zero)
    water = (pixels > 0).astype(np.uint8)
    water_pct = water.sum() / water.size * 100
    print(f"  Water coverage: {water_pct:.1f}%")

    # Pack 32 pixels per uint32, LSB-first, row-major
    stride = (cols + 31) // 32  # uint32s per row
    packed = np.zeros((rows, stride), dtype=np.uint32)
    for bit in range(cols):
        word_idx = bit // 32
        bit_idx = bit % 32
        packed[:, word_idx] |= water[:, bit].astype(np.uint32) << bit_idx

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "wbm.bin"
    with open(out_path, "wb") as f:
        f.write(struct.pack("<II", rows, cols))
        packed.tofile(f)

    total_bytes = out_path.stat().st_size
    print(f"  wbm.bin: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export WBM to 1-bit packed binary for compute shader"
    )
    parser.add_argument(
        "--wbm-dir", type=Path, default=DEFAULT_WBM_DIR,
        help="Directory containing .wbm files",
    )
    parser.add_argument(
        "--planes-dir", type=Path, default=DEFAULT_PLANES_DIR,
        help="Root directory of plane tile exports (writes wbm.bin into each tile subdir)",
    )
    parser.add_argument(
        "--tile", type=str, default=None,
        help="Single tile name (e.g., n48w123). Omit for batch mode.",
    )
    args = parser.parse_args()

    if args.tile:
        tile_names = [args.tile.lower()]
    else:
        # Batch: find all tile subdirs in planes-dir
        tile_names = sorted(
            d.name for d in args.planes_dir.iterdir()
            if d.is_dir() and len(d.name) == 7 and (d / "meta.json").exists()
        )
        if not tile_names:
            print(f"No tile directories found in {args.planes_dir}")
            return
        print(f"Found {len(tile_names)} tile directories in {args.planes_dir}/\n")

    exported = 0
    skipped = 0
    missing = 0
    for tile in tile_names:
        wbm_filename = tile_to_wbm_filename(tile)
        wbm_path = args.wbm_dir / wbm_filename
        out_dir = args.planes_dir / tile

        if not wbm_path.exists():
            print(f"[{tile}] SKIP: no WBM file ({wbm_filename})")
            missing += 1
            continue

        print(f"[{tile}] {wbm_filename}")
        if export_one(wbm_path, out_dir):
            exported += 1
        else:
            skipped += 1
        print()

    if len(tile_names) > 1:
        print(f"Done: {exported} exported, {skipped} skipped, {missing} missing WBM")


if __name__ == "__main__":
    main()
