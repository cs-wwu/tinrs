"""
Download Copernicus GLO-30 DEM tiles from the public AWS S3 bucket.

Tiles are Cloud Optimized GeoTIFFs at 1 arcsecond resolution (3600 rows,
variable columns by latitude). No authentication required.

CLI usage:
    python download_glo30.py --region iceland
    python download_glo30.py --bbox 63,-25,67,-13
    python download_glo30.py --list --region washington   # dry run

Library usage (train_plane.py, gen_tiles.py):
    from download_glo30 import ensure_tile, fetch_tile_list, tiles_for_bbox
    dem_path, wbm_path = ensure_tile("n47w122", glo30_dir)
"""

import argparse
import io
import math
import os
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def _atomic_write_bytes(dest: Path, data: bytes) -> None:
    """Write bytes via .part + os.replace so concurrent writers can't tear the file."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)

S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
TILE_LIST_URL = f"{S3_BASE}/tileList.txt"

REGIONS = {
    "washington": (46, -125, 50, -120),
    "iceland": (63, -25, 67, -13),
    "svalbard": (76, 10, 81, 34),
    "pole": (85, -180, 90, 180),
}

SCRIPT_DIR = Path(__file__).parent
CACHE_PATH = SCRIPT_DIR / ".tile_list_cache.txt"
DEFAULT_OUTPUT = SCRIPT_DIR.parent / "assets" / "maps" / "glo30"


def tile_name(lat: int, lon: int) -> str:
    """Build the GLO-30 tile directory/file name for a 1x1 degree cell.

    lat/lon are the SW corner (bottom-left) of the cell.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def tiles_for_bbox(lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> list[str]:
    """Return tile names covering a bounding box."""
    # Each tile covers [lat, lat+1) x [lon, lon+1)
    lat_start = math.floor(lat_min)
    lat_end = math.ceil(lat_max) - 1  # inclusive upper bound
    lon_start = math.floor(lon_min)
    lon_end = math.ceil(lon_max) - 1

    # TODO: handle antimeridian crossing (lon_min > lon_max)
    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append(tile_name(lat, lon))
    return tiles


def fetch_tile_list(refresh: bool = False) -> set[str]:
    """Fetch the S3 tile list, caching locally."""
    if CACHE_PATH.exists() and not refresh:
        return set(CACHE_PATH.read_text().splitlines())

    print(f"Fetching tile list from {TILE_LIST_URL} ...", file=sys.stderr)
    req = urllib.request.Request(TILE_LIST_URL)
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")

    # tileList.txt contains paths like "Copernicus_DSM_COG_10_N64_00_W022_00_DEM/"
    names = set()
    for line in data.splitlines():
        line = line.strip().rstrip("/")
        if line:
            names.add(line)

    CACHE_PATH.write_text("\n".join(sorted(names)) + "\n")
    print(f"  Cached {len(names)} tiles at {CACHE_PATH}", file=sys.stderr)
    return names


def download_file(url: str, dest: Path) -> tuple[bool, int]:
    """Download a file if not already present (or size differs). Returns (downloaded, size_bytes)."""
    if dest.exists():
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
            if dest.stat().st_size == remote_size and remote_size > 0:
                return False, dest.stat().st_size
        except urllib.error.URLError:
            pass  # fall through to download

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        data = resp.read()

    _atomic_write_bytes(dest, data)
    return True, len(data)


def convert_wbm_to_zlib(tif_data: bytes, dest: Path) -> int:
    """Convert a WBM GeoTIFF to a compact zlib-compressed format.

    Output format: [rows: u16] [cols: u16] [zlib(uint8 pixels row-major)]
    Values: 0=land, 1=ocean, 2=lake, 3=river.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(tif_data))
    data = img.tobytes()
    rows, cols = img.size[1], img.size[0]

    header = struct.pack("<HH", rows, cols)
    compressed = zlib.compress(data, 9)
    payload = header + compressed

    _atomic_write_bytes(dest, payload)
    return len(payload)


def download_wbm(name: str, output_dir: Path) -> tuple[Path | None, int]:
    """Download and convert WBM for a tile. Returns (path, size_bytes) or (None, 0)."""
    wbm_s3_name = name.replace("_DEM", "_WBM")
    wbm_tif = f"{wbm_s3_name}.tif"
    wbm_url = f"{S3_BASE}/{name}/AUXFILES/{wbm_tif}"
    wbm_dest = output_dir / "wbm" / f"{wbm_s3_name}.wbm"
    wbm_dest.parent.mkdir(exist_ok=True)

    if wbm_dest.exists():
        return wbm_dest, wbm_dest.stat().st_size

    try:
        req = urllib.request.Request(wbm_url)
        with urllib.request.urlopen(req) as resp:
            tif_data = resp.read()
        size = convert_wbm_to_zlib(tif_data, wbm_dest)
        return wbm_dest, size
    except urllib.error.HTTPError:
        return None, 0


def download_tile(name: str, output_dir: Path, wbm: bool = True) -> tuple[bool, int, int]:
    """Download a DEM tile and optionally its WBM.

    Idempotent: skips redownload when local size matches remote Content-Length.
    Returns (downloaded, dem_bytes, wbm_bytes).
    """
    wbm_size = 0
    if wbm:
        _, wbm_size = download_wbm(name, output_dir)

    dem_filename = f"{name}.tif"
    dem_url = f"{S3_BASE}/{name}/{dem_filename}"
    dem_downloaded, dem_size = download_file(dem_url, output_dir / dem_filename)

    return dem_downloaded, dem_size, wbm_size


_SHORT_NAME_RE = re.compile(r"^([ns])(\d{1,2})([ew])(\d{1,3})$")


def parse_short_name(short: str) -> tuple[int, int]:
    """Parse 'n47w122' -> (lat=47, lon=-122) signed degrees of the SW corner."""
    m = _SHORT_NAME_RE.match(short.lower())
    if not m:
        raise ValueError(f"Invalid tile short name: {short!r} (expected e.g. 'n47w122')")
    lat = int(m.group(2))
    lon = int(m.group(4))
    if m.group(1) == "s":
        lat = -lat
    if m.group(3) == "w":
        lon = -lon
    return lat, lon


def s3_name_to_short(s3_name: str) -> str:
    """Convert 'Copernicus_DSM_COG_10_N47_00_W122_00_DEM' -> 'n47w122'."""
    parts = s3_name.split("_")
    ns = parts[4][0].lower()
    lat = int(parts[4][1:])
    ew = parts[6][0].lower()
    lon = int(parts[6][1:])
    return f"{ns}{lat:02d}{ew}{lon:03d}"


def ensure_tile(short_name: str, glo30_dir: Path, *, wbm: bool = True) -> tuple[Path, Path | None]:
    """Download tile if missing, return (dem_path, wbm_path or None).

    Idempotent: existing files with matching remote Content-Length are reused
    without redownload. Suitable for parallel sweep jobs sharing a filesystem;
    each tile maps to a unique job in the common case so write contention is rare.

    Raises ValueError for malformed names. Lets HTTPError propagate so callers
    can distinguish "not in S3" from successful runs.
    """
    lat, lon = parse_short_name(short_name)
    s3_name = tile_name(lat, lon)
    glo30_dir.mkdir(parents=True, exist_ok=True)

    download_tile(s3_name, glo30_dir, wbm=wbm)

    dem_path = glo30_dir / f"{s3_name}.tif"
    wbm_path: Path | None = glo30_dir / "wbm" / f"{s3_name.replace('_DEM', '_WBM')}.wbm"
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM did not land at {dem_path} after download")
    if wbm_path is not None and not wbm_path.exists():
        wbm_path = None
    return dem_path, wbm_path


def main():
    parser = argparse.ArgumentParser(
        description="Download Copernicus GLO-30 DEM tiles from AWS S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""Named regions:
  {"".join(f"  {name:12s} ({b[0]},{b[1]}) to ({b[2]},{b[3]})" + chr(10) for name, b in REGIONS.items())}
Examples:
  %(prog)s --region iceland              Download all Iceland tiles
  %(prog)s --bbox 63,-25,67,-13          Same thing, explicit bbox
  %(prog)s --list --region washington    Show tiles without downloading
  %(prog)s --refresh                     Re-fetch tile list from S3""",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bbox", help="lat_min,lon_min,lat_max,lon_max")
    group.add_argument("--region", choices=REGIONS.keys(), help="Named region preset")
    parser.add_argument("--list", action="store_true", help="List tiles and exit (dry run)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--no-wbm", action="store_true", help="Skip water body mask downloads")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch tile list from S3")
    args = parser.parse_args()

    # Parse bbox
    if args.region:
        lat_min, lon_min, lat_max, lon_max = REGIONS[args.region]
    else:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            parser.error("--bbox requires exactly 4 comma-separated values: lat_min,lon_min,lat_max,lon_max")
        lat_min, lon_min, lat_max, lon_max = parts

    # Compute candidate tiles
    candidates = tiles_for_bbox(lat_min, lon_min, lat_max, lon_max)
    print(f"Bounding box: ({lat_min}, {lon_min}) to ({lat_max}, {lon_max})")
    print(f"Candidate tiles: {len(candidates)}")

    # Filter against S3 tile list (tiles above ~84 deg N don't exist at all)
    available = fetch_tile_list(refresh=args.refresh)
    tiles = [t for t in candidates if t in available]
    missing = [t for t in candidates if t not in available]

    print(f"Available: {len(tiles)}, missing from S3: {len(missing)}")

    if args.list:
        print(f"\nTiles to download:")
        for t in sorted(tiles):
            print(f"  {t}.tif")
        if missing:
            print(f"\nNot available (ocean/no data):")
            for t in sorted(missing):
                print(f"  {t}")
        return

    if not tiles:
        print("No tiles to download.")
        return

    # Download
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    downloaded = 0
    skipped = 0
    total_wbm_bytes = 0

    for i, name in enumerate(sorted(tiles), 1):
        prefix = f"[{i}/{len(tiles)}]"
        try:
            was_downloaded, dem_size, wbm_size = download_tile(
                name, args.output_dir, wbm=not args.no_wbm,
            )
            total_wbm_bytes += wbm_size
            total_bytes += dem_size
            dem_mb = dem_size / (1024 * 1024)
            wbm_kb = wbm_size / 1024
            wbm_str = f" + WBM {wbm_kb:.0f} KB" if wbm_size else ""
            if was_downloaded:
                downloaded += 1
                print(f"  {prefix} {name}.tif  ({dem_mb:.1f} MB{wbm_str})")
            else:
                skipped += 1
                print(f"  {prefix} {name}.tif  (exists, {dem_mb:.1f} MB{wbm_str})")
        except urllib.error.HTTPError as e:
            print(f"  {prefix} {name}.tif  FAILED: HTTP {e.code}", file=sys.stderr)
        except urllib.error.URLError as e:
            print(f"  {prefix} {name}.tif  FAILED: {e.reason}", file=sys.stderr)

    total_mb = total_bytes / (1024 * 1024)
    wbm_total_kb = total_wbm_bytes / 1024
    wbm_str = f", WBM: {wbm_total_kb:.0f} KB" if total_wbm_bytes else ""
    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, DEM: {total_mb:.1f} MB{wbm_str} in {args.output_dir}")


if __name__ == "__main__":
    main()
