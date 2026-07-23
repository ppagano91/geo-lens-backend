"""Generate tiny local Sentinel-2-like sample GeoTIFFs for development.

Creates (relative to repo root):

    data/sample/scenes/test_scene/B02.tif  # Blue
    data/sample/scenes/test_scene/B03.tif  # Green
    data/sample/scenes/test_scene/B04.tif  # Red
    data/sample/scenes/test_scene/B08.tif  # NIR
    data/sample/scenes/test_scene/B11.tif  # SWIR1
    data/sample/scenes/test_scene/B12.tif  # SWIR2

These files are synthetic fixtures for validating path resolution, metadata
endpoints and spectral-index smoke tests. They have no scientific value.

Bands share identical grid, CRS, transform, nodata and dtype so they are
perfectly co-registered.

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

BAND_FILES: dict[str, str] = {
    "B02.tif": "Sentinel-2-like Blue (B02) — synthetic",
    "B03.tif": "Sentinel-2-like Green (B03) — synthetic",
    "B04.tif": "Sentinel-2-like Red (B04) — synthetic",
    "B08.tif": "Sentinel-2-like NIR (B08) — synthetic",
    "B11.tif": "Sentinel-2-like SWIR1 (B11) — synthetic",
    "B12.tif": "Sentinel-2-like SWIR2 (B12) — synthetic",
}


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


def _soft_blob(rows: np.ndarray, cols: np.ndarray, cy: float, cx: float, sigma: float) -> np.ndarray:
    """Soft radial weight in [0, 1] centred at (cy, cx)."""
    return np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2.0 * sigma**2))


def _build_scene_bands() -> dict[str, np.ndarray]:
    """Build co-registered synthetic bands with simple land-cover patterns.

    Patterns (not scientific, only coherent for technical tests):
    - gentle west→east / north→south gradients as base reflectance
    - vegetation blob (center-right): high NIR, lower Red → positive NDVI
    - water blob (bottom-left): higher Blue/Green, depressed NIR/SWIR
    - dry/burned blob (top-left): elevated Red/SWIR, depressed NIR
    - corner (0,0) is nodata on every band
    """
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    cols = np.arange(WIDTH, dtype=np.float32)[None, :]

    # Soft zone weights ≈ [0, 1]
    vegetation = _soft_blob(rows, cols, cy=HEIGHT * 0.48, cx=WIDTH * 0.68, sigma=11.0)
    water = _soft_blob(rows, cols, cy=HEIGHT * 0.78, cx=WIDTH * 0.22, sigma=8.5)
    dry = _soft_blob(rows, cols, cy=HEIGHT * 0.22, cx=WIDTH * 0.28, sigma=9.0)

    # Mild spatial gradients shared by all bands
    col_grad = cols / max(WIDTH - 1, 1)
    row_grad = rows / max(HEIGHT - 1, 1)

    # Base DN-like levels (uint16 reflectance-ish). Order of magnitude only.
    # Blue low/mid; Green a bit above Blue; Red mid; NIR higher than Red;
    # SWIR mid/high.
    b02 = 450.0 + col_grad * 350.0 + row_grad * 120.0
    b03 = 550.0 + col_grad * 400.0 + row_grad * 140.0
    b04 = 900.0 + col_grad * 550.0 + row_grad * 180.0
    b08 = 2200.0 + col_grad * 900.0 + row_grad * 250.0
    b11 = 1800.0 + col_grad * 700.0 + row_grad * 220.0
    b12 = 1500.0 + col_grad * 650.0 + row_grad * 200.0

    # Vegetation: boost NIR / Green; damp Red and SWIR slightly
    b02 += vegetation * 80.0
    b03 += vegetation * 220.0
    b04 -= vegetation * 350.0
    b08 += vegetation * 2200.0
    b11 -= vegetation * 250.0
    b12 -= vegetation * 300.0

    # Water: raise visible Blue/Green; crush NIR and SWIR
    b02 += water * 700.0
    b03 += water * 550.0
    b04 += water * 150.0
    b08 -= water * 1600.0
    b11 -= water * 1200.0
    b12 -= water * 1100.0

    # Dry / burned: raise Red + SWIR; lower NIR (and a bit of Green)
    b02 += dry * 120.0
    b03 -= dry * 80.0
    b04 += dry * 900.0
    b08 -= dry * 900.0
    b11 += dry * 1400.0
    b12 += dry * 1600.0

    bands = {
        "B02.tif": b02,
        "B03.tif": b03,
        "B04.tif": b04,
        "B08.tif": b08,
        "B11.tif": b11,
        "B12.tif": b12,
    }

    # Keep valid DN in a safe uint16 range; reserve 0 for nodata.
    out: dict[str, np.ndarray] = {}
    for name, arr in bands.items():
        clipped = np.clip(arr, 1.0, 65000.0).astype(np.uint16)
        clipped[0, 0] = NODATA
        out[name] = clipped
    return out


def main() -> int:
    print("GeoChange Analyzer — sample rasters")
    print(f"Output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    arrays = _build_scene_bands()
    targets = {name: (arrays[name], label) for name, label in BAND_FILES.items()}

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
    for filename in BAND_FILES:
        print(f"  sample/scenes/test_scene/{filename}")
    print()
    print("Absolute paths:")
    for filename in BAND_FILES:
        print(f"  {(OUTPUT_DIR / filename).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
