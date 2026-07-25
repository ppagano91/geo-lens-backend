"""Validate derived spectral-index PNG previews under DATA_ROOT (Fase 7E).

Checks that preview PNGs for ndvi / ndwi / nbr / ndmi exist beside their
GeoTIFF sources, are readable RGBA images, and have expected dimensions.

Uses app.core.config.settings.data_root (DATA_ROOT) — no hardcoded absolute paths.

Usage (from geo-lens-backend/, with venv active):

    python scripts/validate_index_previews.py \\
        --scene-id 2f707fd8-c4f5-40da-92aa-6b2e7c0202c4

Optional: generate missing PNGs via the API first, or pass --generate to
create previews in-process from existing .tif files.

Exit code 0 if all indices pass; 1 if any check fails.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from PIL import Image

# Allow `python scripts/...` without installing the package.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _ensure_rasterio_proj_data() -> None:
    """Prefer rasterio's bundled PROJ data over a conflicting system PROJ_LIB."""
    import rasterio

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
from app.services.index_preview_service import IndexPreviewService  # noqa: E402

INDEX_KEYS = ("ndvi", "ndwi", "nbr", "ndmi")

EXPECTED_WIDTH = 50
EXPECTED_HEIGHT = 50


@dataclass
class CheckResult:
    index_key: str
    ok: bool
    messages: list[str]
    info: dict[str, int | str] | None = None


def _tif_path(data_root: Path, scene_id: str, index_key: str) -> Path:
    return data_root / "derived" / "scenes" / scene_id / f"{index_key}.tif"


def _png_path(data_root: Path, scene_id: str, index_key: str) -> Path:
    return data_root / "derived" / "scenes" / scene_id / f"{index_key}.png"


def validate_preview(data_root: Path, scene_id: str, index_key: str) -> CheckResult:
    tif = _tif_path(data_root, scene_id, index_key)
    png = _png_path(data_root, scene_id, index_key)
    messages: list[str] = []

    if not tif.is_file():
        messages.append(f"missing source GeoTIFF: {tif}")
    if not png.is_file():
        messages.append(f"missing preview PNG: {png}")
        return CheckResult(index_key=index_key, ok=False, messages=messages)

    try:
        with Image.open(png) as img:
            if img.format != "PNG":
                messages.append(f"format={img.format!r} (expected 'PNG')")
            if img.mode not in ("RGBA", "RGB"):
                messages.append(f"mode={img.mode!r} (expected RGBA or RGB)")
            width, height = img.size
            if width != EXPECTED_WIDTH:
                messages.append(f"width={width} (expected {EXPECTED_WIDTH})")
            if height != EXPECTED_HEIGHT:
                messages.append(f"height={height} (expected {EXPECTED_HEIGHT})")
            info = {
                "width": width,
                "height": height,
                "mode": img.mode,
                "bytes": png.stat().st_size,
            }
    except Exception as exc:  # noqa: BLE001 — report any open failure
        return CheckResult(
            index_key=index_key,
            ok=False,
            messages=[f"cannot open/read PNG: {exc}"],
        )

    return CheckResult(
        index_key=index_key,
        ok=len(messages) == 0,
        messages=messages,
        info=info,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate derived index PNG previews under DATA_ROOT (Fase 7E)."
    )
    parser.add_argument(
        "--scene-id",
        required=True,
        help="Scene UUID (folder under derived/scenes/)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Create missing PNGs from existing derived GeoTIFFs before validating",
    )
    args = parser.parse_args(argv)

    data_root = settings.data_root_path
    scene_id = args.scene_id.strip()

    print(f"DATA_ROOT (settings.data_root) = {settings.data_root!r}")
    print(f"DATA_ROOT resolved            = {data_root}")
    print(f"scene_id                      = {scene_id}")
    print(f"expected layout               = derived/scenes/{{scene_id}}/{{index}}.png")
    print()

    if not data_root.is_dir():
        print(f"FAIL: DATA_ROOT is not a directory: {data_root}")
        return 1

    if args.generate:
        service = IndexPreviewService(data_root)
        scene_uuid = UUID(scene_id)
        print("Generating previews from derived GeoTIFFs...")
        for key in INDEX_KEYS:
            try:
                result = service.create_preview(scene_uuid, key)
                print(f"  [gen] {key}: {result.output.asset_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [gen FAIL] {key}: {exc}")
        print()

    results = [validate_preview(data_root, scene_id, key) for key in INDEX_KEYS]
    all_ok = True

    for result in results:
        path = _png_path(data_root, scene_id, result.index_key)
        rel = path.relative_to(data_root) if path.is_relative_to(data_root) else path
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.index_key}: {rel}")
        if result.info:
            print(
                "       info: "
                f"size={result.info['width']}x{result.info['height']} "
                f"mode={result.info['mode']} "
                f"bytes={result.info['bytes']}"
            )
        for msg in result.messages:
            print(f"       - {msg}")
        if not result.ok:
            all_ok = False

    print()
    if all_ok:
        print(f"PASS: all {len(INDEX_KEYS)} index PNG previews validated.")
        return 0

    failed = [r.index_key for r in results if not r.ok]
    print(f"FAIL: {len(failed)} index(es) failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
