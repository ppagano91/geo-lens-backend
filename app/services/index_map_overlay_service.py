"""Build MapLibre image-overlay metadata from derived index GeoTIFF + PNG.

Fase 9E: read georef from the derived float32 GeoTIFF, transform corners to
EPSG:4326, and point ``image_url`` at the existing preview PNG. Does not
generate tiles, recompute indices, or create missing assets.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from rasterio.warp import transform as warp_transform

from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_metadata,
)
from app.schemas.index_compute import (
    IndexMapOverlayBounds,
    IndexMapOverlayResult,
)
from app.services.asset_storage_service import AssetStorageService
from app.services.index_preview_service import (
    DerivedGeotiffNotFoundError,
    PreviewPngNotFoundError,
)
from app.services.local_index_compute_service import (
    LOCAL_INDEX_REGISTRY,
    UnsupportedIndexError,
)

_WGS84_ALIASES = frozenset({"EPSG:4326", "OGC:CRS84"})


class IndexMapOverlayError(Exception):
    """Derived GeoTIFF cannot be positioned on the map (missing CRS/bounds)."""


class IndexMapOverlayService:
    """Resolve derived assets and emit MapLibre-compatible overlay metadata."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    def get_map_overlay(
        self, scene_id: UUID, index_key: str
    ) -> IndexMapOverlayResult:
        """Return overlay metadata for an existing derived index + preview PNG."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        tif_asset = self._storage.build_derived_asset_path(
            scene_id, spec.key, "tif"
        )
        png_asset = self._storage.build_derived_asset_path(
            scene_id, spec.key, "png"
        )

        if not self._storage.exists(tif_asset):
            raise DerivedGeotiffNotFoundError(scene_id, spec.key, tif_asset)
        if not self._storage.exists(png_asset):
            raise PreviewPngNotFoundError(scene_id, spec.key, png_asset)

        meta = read_raster_metadata(tif_asset, self.data_root)
        if meta.width is None or meta.height is None:
            raise IndexMapOverlayError(
                f"Derived GeoTIFF for scene {scene_id} index '{spec.key}' "
                "has no width/height"
            )
        if not meta.bounds:
            raise IndexMapOverlayError(
                f"Derived GeoTIFF for scene {scene_id} index '{spec.key}' "
                "has no bounds"
            )
        if not meta.crs:
            raise IndexMapOverlayError(
                f"Derived GeoTIFF for scene {scene_id} index '{spec.key}' "
                "has no CRS; cannot georeference overlay"
            )

        left = float(meta.bounds["left"])
        bottom = float(meta.bounds["bottom"])
        right = float(meta.bounds["right"])
        top = float(meta.bounds["top"])

        coordinates = _corners_to_wgs84(
            meta.crs,
            left=left,
            bottom=bottom,
            right=right,
            top=top,
        )

        image_url = (
            f"/api/v1/scenes/{scene_id}/indices/{spec.key}/preview.png"
        )

        return IndexMapOverlayResult(
            scene_id=scene_id,
            index_key=spec.key,
            image_url=image_url,
            width=int(meta.width),
            height=int(meta.height),
            crs_original=meta.crs,
            bounds_original=IndexMapOverlayBounds(
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            ),
            coordinates_wgs84=coordinates,
        )


def _corners_to_wgs84(
    crs: str,
    *,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> list[list[float]]:
    """MapLibre image source corners: top-left, top-right, bottom-right, bottom-left."""
    xs = [left, right, right, left]
    ys = [top, top, bottom, bottom]

    if crs not in _WGS84_ALIASES:
        try:
            xs, ys = warp_transform(crs, "EPSG:4326", xs, ys)
        except Exception as exc:  # noqa: BLE001 — surface as overlay error
            raise IndexMapOverlayError(
                f"Cannot reproject overlay corners from {crs} to EPSG:4326: {exc}"
            ) from exc

    return [
        [float(xs[0]), float(ys[0])],
        [float(xs[1]), float(ys[1])],
        [float(xs[2]), float(ys[2])],
        [float(xs[3]), float(ys[3])],
    ]


__all__ = [
    "IndexMapOverlayService",
    "IndexMapOverlayError",
    "UnsupportedIndexError",
    "DerivedGeotiffNotFoundError",
    "PreviewPngNotFoundError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
]
