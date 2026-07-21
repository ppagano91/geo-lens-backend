"""Generate tiny local Sentinel-2-like sample GeoTIFFs for development.

Creates (relative to repo root):

    data/sample/scenes/test_scene/B04.tif  # Red
    data/sample/scenes/test_scene/B08.tif  # NIR

These files are synthetic fixtures for validating Fase 7A path resolution and
metadata endpoints. They have no scientific value.

Usage (from backend/):

    .venv\\Scripts\\activate
    python scripts/create_sample_rasters.py

Idempotent: existing files are overwritten with a clear notice.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


def _ensure_rasterio_proj_data() -> None:
    """Prefer rasterio's bundled PROJ data over a conflicting system PROJ_LIB.

    On Windows, PostgreSQL/PostGIS often sets PROJ_LIB to an older proj.db that
    breaks rasterio CRS lookups (EPSG). Same fix as app.raster.readers.
    """
    bundled = Path(rasterio.__file__).resolve().parent / "proj_data"
    if not (bundled / "proj.db").is_file():
        return
    bundled_str = str(bundled)
    current = os.environ.get("PROJ_LIB") or os.environ.get("PROJ_DATA")
    if current:
        current_path = Path(current)
        if current_path.resolve() == bundled.resolve():
            return
        if (current_path / "proj.db").is_file() and "rasterio" in current_path.as_posix():
            return
    os.environ["PROJ_LIB"] = bundled_str
    os.environ["PROJ_DATA"] = bundled_str


_ensure_rasterio_proj_data()

# Repo layout: backend/scripts/this_file.py → repo root is parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "sample" / "scenes" / "test_scene"

# Small enough to keep as a local fixture; large enough to exercise readers.
WIDTH = 50
HEIGHT = 50
DTYPE = "uint16"
NODATA = 0
CRS = "EPSG:4326"

# Upper-left near CABA (lon, lat); ~0.001° pixels (~100 m).
ORIGIN_X = -58.45
ORIGIN_Y = -34.55
PIXEL_SIZE = 0.001


def _write_band(path: Path, data: np.ndarray, *, label: str) -> None:
    if data.shape != (HEIGHT, WIDTH):
        raise ValueError(f"{label}: expected shape ({HEIGHT}, {WIDTH}), got {data.shape}")
    if data.dtype != np.uint16:
        raise ValueError(f"{label}: expected uint16, got {data.dtype}")

    existed = path.exists()
    transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE, PIXEL_SIZE)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype=DTYPE,
        crs=CRS,
        transform=transform,
        nodata=NODATA,
        compress="lzw",
    ) as dataset:
        dataset.write(data, 1)
        dataset.update_tags(
            DESCRIPTION=label,
            NOTE="Synthetic GeoChange Analyzer sample — not real Sentinel-2 data",
        )

    action = "Overwrote" if existed else "Created"
    size_kb = path.stat().st_size / 1024
    print(f"  {action}: {path} ({size_kb:.1f} KiB)")


def _build_b04() -> np.ndarray:
    """Red band: moderate reflectance; corner cell is nodata."""
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    cols = np.arange(WIDTH, dtype=np.float32)[None, :]
    # ~800–2200 DN-like values (uint16)
    data = (800 + cols * 20 + rows * 8).astype(np.uint16)
    data[0, 0] = NODATA
    return data


def _build_b08() -> np.ndarray:
    """NIR band: higher than Red so NDVI later is positive in most cells."""
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    cols = np.arange(WIDTH, dtype=np.float32)[None, :]
    # ~2500–5500 DN-like values; stronger "vegetation" toward center-right
    center = np.exp(-((rows - HEIGHT / 2) ** 2 + (cols - WIDTH * 0.65) ** 2) / (2 * 12**2))
    data = (2500 + cols * 25 + rows * 10 + center * 1500).astype(np.uint16)
    data[0, 0] = NODATA
    return data


def main() -> int:
    print("GeoChange Analyzer — sample rasters (Fase 7A.1)")
    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = {
        "B04.tif": (_build_b04(), "Sentinel-2-like Red (B04) — synthetic"),
        "B08.tif": (_build_b08(), "Sentinel-2-like NIR (B08) — synthetic"),
    }

    existing = [name for name in targets if (OUTPUT_DIR / name).exists()]
    if existing:
        print(
            "Notice: the following files already exist and will be overwritten: "
            + ", ".join(existing)
        )

    print("Writing GeoTIFFs…")
    for filename, (array, label) in targets.items():
        _write_band(OUTPUT_DIR / filename, array, label=label)

    print()
    print("Done. Relative asset_path values for POST /api/v1/scenes:")
    print("  sample/scenes/test_scene/B04.tif")
    print("  sample/scenes/test_scene/B08.tif")
    print()
    print("Absolute paths:")
    for filename in targets:
        print(f"  {(OUTPUT_DIR / filename).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
