"""Stage Landsat 8 cropped bands under DATA_ROOT with native SR_B* names.

Fase 8B.1 helper: copies QGIS crops from data/temp/band_stack/B{2..7}.tif
into sample/scenes/landsat8_lc08_225084/SR_B{2..7}.tif (no Sentinel rename).

Usage (from geo-lens-backend/, DATA_ROOT=../data):

    python scripts/stage_landsat8_sample_bands.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from app.core.config import settings

# Short export name (common from QGIS) → native Landsat 8 band_key filename.
BAND_COPIES: tuple[tuple[str, str], ...] = (
    ("B2.tif", "SR_B2.tif"),
    ("B3.tif", "SR_B3.tif"),
    ("B4.tif", "SR_B4.tif"),
    ("B5.tif", "SR_B5.tif"),
    ("B6.tif", "SR_B6.tif"),
    ("B7.tif", "SR_B7.tif"),
)

DEFAULT_SOURCE_REL = Path("temp/band_stack")
DEFAULT_DEST_REL = Path("sample/scenes/landsat8_lc08_225084")


def main() -> int:
    data_root = settings.data_root_path
    src_dir = data_root / DEFAULT_SOURCE_REL
    dest_dir = data_root / DEFAULT_DEST_REL
    dest_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    for src_name, dest_name in BAND_COPIES:
        src = src_dir / src_name
        if not src.is_file():
            missing.append(str(src))
            continue
        dest = dest_dir / dest_name
        shutil.copy2(src, dest)
        aux = Path(str(src) + ".aux.xml")
        if aux.is_file():
            shutil.copy2(aux, Path(str(dest) + ".aux.xml"))
        print(f"OK  {src_name} → {dest.relative_to(data_root)}")

    if missing:
        print("Missing source files:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        return 1

    print(f"Staged under {dest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
