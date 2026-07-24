"""Validate derived spectral-index GeoTIFFs under DATA_ROOT (Fase 7D.1).

Checks that compute-and-save outputs for ndvi / ndwi / nbr / ndmi are readable
GeoTIFF float32 rasters with the expected metadata and value range.

Uses app.core.config.settings.data_root (DATA_ROOT) — no hardcoded absolute paths.

Usage (from geo-lens-backend/, with venv active):

    python scripts/validate_derived_indices.py \\
        --scene-id 2f707fd8-c4f5-40da-92aa-6b2e7c0202c4

Exit code 0 if all indices pass; 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio

# Allow `python scripts/...` without installing the package.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _ensure_rasterio_proj_data() -> None:
    """Prefer rasterio's bundled PROJ data over a conflicting system PROJ_LIB."""
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

from app.core.config import settings  # noqa: E402
from app.raster.writers import DEFAULT_INDEX_NODATA  # noqa: E402

INDEX_KEYS = ("ndvi", "ndwi", "nbr", "ndmi")

EXPECTED_WIDTH = 50
EXPECTED_HEIGHT = 50
EXPECTED_CRS = "EPSG:4326"
EXPECTED_DTYPE = "float32"
EXPECTED_DRIVER = "GTiff"
EXPECTED_COUNT = 1
EXPECTED_NODATA = float(DEFAULT_INDEX_NODATA)
VALUE_MIN = -1.0
VALUE_MAX = 1.0


@dataclass
class CheckResult:
    index_key: str
    ok: bool
    messages: list[str]
    stats: dict[str, float | int] | None = None


def _derived_path(data_root: Path, scene_id: str, index_key: str) -> Path:
    return data_root / "derived" / "scenes" / scene_id / f"{index_key}.tif"


def _crs_matches(crs: object | None, expected: str) -> bool:
    if crs is None:
        return False
    try:
        return crs == rasterio.crs.CRS.from_string(expected)
    except Exception:
        return str(crs).upper() == expected.upper()


def validate_index(data_root: Path, scene_id: str, index_key: str) -> CheckResult:
    path = _derived_path(data_root, scene_id, index_key)
    messages: list[str] = []

    if not path.is_file():
        return CheckResult(
            index_key=index_key,
            ok=False,
            messages=[f"missing file: {path}"],
        )

    try:
        with rasterio.open(path) as ds:
            if ds.driver != EXPECTED_DRIVER:
                messages.append(f"driver={ds.driver!r} (expected {EXPECTED_DRIVER!r})")
            if ds.count != EXPECTED_COUNT:
                messages.append(f"count={ds.count} (expected {EXPECTED_COUNT})")
            if ds.width != EXPECTED_WIDTH:
                messages.append(f"width={ds.width} (expected {EXPECTED_WIDTH})")
            if ds.height != EXPECTED_HEIGHT:
                messages.append(f"height={ds.height} (expected {EXPECTED_HEIGHT})")
            if ds.dtypes[0] != EXPECTED_DTYPE:
                messages.append(f"dtype={ds.dtypes[0]!r} (expected {EXPECTED_DTYPE!r})")

            nodata = ds.nodata
            if nodata is None or float(nodata) != EXPECTED_NODATA:
                messages.append(f"nodata={nodata!r} (expected {EXPECTED_NODATA})")

            if not _crs_matches(ds.crs, EXPECTED_CRS):
                messages.append(f"crs={ds.crs!r} (expected {EXPECTED_CRS!r})")

            transform = ds.transform
            if transform is None or transform.is_identity:
                messages.append(f"transform invalid or identity: {transform}")

            data = ds.read(1)
    except Exception as exc:  # noqa: BLE001 — report any open/read failure
        return CheckResult(
            index_key=index_key,
            ok=False,
            messages=[f"cannot open/read with rasterio: {exc}"],
        )

    if data.dtype != np.float32:
        messages.append(f"array dtype={data.dtype} (expected float32)")

    valid_mask = data != np.float32(EXPECTED_NODATA)
    # Also treat NaN as invalid if present
    valid_mask &= ~np.isnan(data)
    valid = data[valid_mask]

    if valid.size == 0:
        messages.append("no valid pixels (all nodata/NaN)")
        stats = {
            "valid_pixels": 0,
            "nodata_pixels": int(data.size),
        }
        return CheckResult(index_key=index_key, ok=False, messages=messages, stats=stats)

    if not np.isfinite(valid).all():
        n_inf = int(np.isinf(valid).sum())
        messages.append(f"infinite values in valid pixels: {n_inf}")

    vmin = float(valid.min())
    vmax = float(valid.max())
    vmean = float(valid.mean())
    if vmin < VALUE_MIN or vmax > VALUE_MAX:
        messages.append(
            f"valid values out of [{VALUE_MIN}, {VALUE_MAX}]: min={vmin}, max={vmax}"
        )

    stats = {
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "valid_pixels": int(valid.size),
        "nodata_pixels": int(data.size - valid.size),
    }

    return CheckResult(
        index_key=index_key,
        ok=len(messages) == 0,
        messages=messages,
        stats=stats,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate derived index GeoTIFFs under DATA_ROOT (Fase 7D.1)."
    )
    parser.add_argument(
        "--scene-id",
        required=True,
        help="Scene UUID (folder under derived/scenes/)",
    )
    args = parser.parse_args(argv)

    data_root = settings.data_root_path
    scene_id = args.scene_id.strip()

    print(f"DATA_ROOT (settings.data_root) = {settings.data_root!r}")
    print(f"DATA_ROOT resolved            = {data_root}")
    print(f"scene_id                      = {scene_id}")
    print(f"expected layout               = derived/scenes/{{scene_id}}/{{index}}.tif")
    print()

    if not data_root.is_dir():
        print(f"FAIL: DATA_ROOT is not a directory: {data_root}")
        return 1

    results = [validate_index(data_root, scene_id, key) for key in INDEX_KEYS]
    all_ok = True

    for result in results:
        path = _derived_path(data_root, scene_id, result.index_key)
        rel = path.relative_to(data_root) if path.is_relative_to(data_root) else path
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.index_key}: {rel}")
        if result.stats:
            print(
                "       stats: "
                f"min={result.stats['min']:.6f} "
                f"max={result.stats['max']:.6f} "
                f"mean={result.stats['mean']:.6f} "
                f"valid={result.stats['valid_pixels']} "
                f"nodata={result.stats['nodata_pixels']}"
            )
        for msg in result.messages:
            print(f"       - {msg}")
        if not result.ok:
            all_ok = False

    print()
    if all_ok:
        print(f"PASS: all {len(INDEX_KEYS)} derived GeoTIFFs validated.")
        return 0

    failed = [r.index_key for r in results if not r.ok]
    print(f"FAIL: {len(failed)} index(es) failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
