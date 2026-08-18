"""Upload, catalog, and hillshade visualization for experimental DEMs (v0.1-P5).

Independent of satellite scenes. Does not generate tiles, COGs, or MapLibre
terrain. Overlay is a 2D hillshade PNG positioned with image-source corners.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID, uuid4

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dem_asset import DemAsset
from app.raster.hillshade import (
    DEFAULT_HILLSHADE_ALTITUDE,
    DEFAULT_HILLSHADE_AZIMUTH,
    compute_hillshade,
    hillshade_to_rgba,
)
from app.raster.preview import PreviewWriteError, write_preview_png
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
    read_raster_metadata,
)
from app.repositories.dem_asset_repository import DemAssetRepository
from app.schemas.dem import (
    DemAssetRead,
    DemBounds,
    DemHillshadeResult,
    DemMapOverlayResult,
)
from app.services.asset_storage_service import AssetStorageService
from app.services.index_map_overlay_service import (
    IndexMapOverlayError,
    corners_to_wgs84,
)

ALLOWED_DEM_EXTENSIONS = {".tif", ".tiff"}
DEM_GEOTIFF_FILENAME = "dem.tif"
HILLSHADE_PNG_FILENAME = "hillshade.png"


class DemError(Exception):
    """Domain error for DEM upload / hillshade (mapped to HTTP by the endpoint)."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DemNotFoundError(DemError):
    def __init__(self, dem_id: UUID | str) -> None:
        super().__init__(f"DEM {dem_id} not found", status_code=404)


class DemHillshadeNotFoundError(DemError):
    def __init__(self, dem_id: UUID | str) -> None:
        super().__init__(
            f"Hillshade PNG for DEM {dem_id} not found. Generate it first.",
            status_code=404,
        )


class DemService:
    def __init__(
        self,
        db: Session,
        data_root: Path | str | None = None,
    ) -> None:
        self.repository = DemAssetRepository(db)
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    def upload(
        self,
        *,
        content: bytes,
        filename: str,
        name: Optional[str] = None,
    ) -> DemAssetRead:
        """Validate a GeoTIFF DEM, store it under DATA_ROOT, and catalog it."""
        if not content:
            raise DemError("DEM file is empty")

        original_name = (filename or "").strip()
        if not original_name:
            raise DemError("DEM file is missing a filename")

        suffix = Path(original_name).suffix.lower()
        if suffix not in ALLOWED_DEM_EXTENSIONS:
            raise DemError("DEM must be a GeoTIFF (.tif / .tiff)")

        max_bytes = int(getattr(settings, "max_upload_file_bytes", 0) or 0)
        if max_bytes > 0 and len(content) > max_bytes:
            raise DemError(
                f"DEM file exceeds max upload size ({max_bytes} bytes)",
                status_code=413,
            )

        dem_id = uuid4()
        asset_path = self._storage.build_uploaded_dem_asset_path(
            dem_id, DEM_GEOTIFF_FILENAME
        )
        dest = self._storage.resolve_write_path(asset_path)
        dest.parent.mkdir(parents=True, exist_ok=False)

        try:
            dest.write_bytes(content)
            inspected = self._inspect_dem(asset_path)
            display_name = (name or "").strip() or Path(original_name).stem
            if not display_name:
                display_name = f"DEM {str(dem_id)[:8]}"

            meta = dict(inspected["metadata"])
            meta["original_filename"] = original_name

            asset = DemAsset(
                id=dem_id,
                name=display_name[:255],
                asset_path=asset_path,
                preview_path=None,
                crs=inspected["crs"],
                width=inspected["width"],
                height=inspected["height"],
                bounds=inspected["bounds"],
                min_elevation=inspected["min_elevation"],
                max_elevation=inspected["max_elevation"],
                metadata_=meta,
            )
            created = self.repository.create(asset)
            return self._to_read(created)
        except Exception:
            if dest.parent.exists():
                shutil.rmtree(dest.parent, ignore_errors=True)
            raise

    def list(self, *, limit: int = 50, offset: int = 0) -> list[DemAssetRead]:
        return [
            self._to_read(asset)
            for asset in self.repository.list(limit=limit, offset=offset)
        ]

    def get(self, dem_id: UUID) -> DemAssetRead:
        return self._to_read(self._require(dem_id))

    def generate_hillshade(
        self,
        dem_id: UUID,
        *,
        azimuth: float = DEFAULT_HILLSHADE_AZIMUTH,
        altitude: float = DEFAULT_HILLSHADE_ALTITUDE,
    ) -> DemHillshadeResult:
        asset = self._require(dem_id)
        if not self._storage.exists(asset.asset_path):
            raise DemError(
                f"DEM GeoTIFF missing on disk: {asset.asset_path}",
                status_code=404,
            )

        try:
            raster = read_raster_array(asset.asset_path, self.data_root)
        except RasterFileNotFoundError as exc:
            raise DemError(str(exc), status_code=404) from exc
        except (RasterReadError, RasterPathError) as exc:
            raise DemError(str(exc)) from exc

        x_cellsize = abs(float(raster.transform[0]))
        y_cellsize = abs(float(raster.transform[4]))
        try:
            shaded = compute_hillshade(
                raster.data,
                x_cellsize=x_cellsize,
                y_cellsize=y_cellsize,
                azimuth_deg=azimuth,
                altitude_deg=altitude,
            )
            rgba = hillshade_to_rgba(shaded)
        except ValueError as exc:
            raise DemError(str(exc)) from exc

        preview_path = self._storage.build_uploaded_dem_asset_path(
            dem_id, HILLSHADE_PNG_FILENAME
        )
        try:
            write_preview_png(preview_path, self.data_root, rgba)
        except PreviewWriteError as exc:
            raise DemError(str(exc), status_code=500) from exc

        meta = dict(asset.metadata_ or {})
        meta["hillshade"] = {
            "azimuth": float(azimuth),
            "altitude": float(altitude),
            "asset_path": preview_path,
            "nodata_transparent": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        asset.preview_path = preview_path
        asset.metadata_ = meta
        saved = self.repository.save(asset)

        return DemHillshadeResult(
            dem_id=saved.id,
            status="hillshade_created",
            preview_path=preview_path,
            width=saved.width,
            height=saved.height,
            azimuth=float(azimuth),
            altitude=float(altitude),
            nodata_transparent=True,
        )

    def resolve_hillshade_png(self, dem_id: UUID) -> Path:
        asset = self._require(dem_id)
        if not asset.preview_path:
            raise DemHillshadeNotFoundError(dem_id)
        if not self._storage.exists(asset.preview_path):
            raise DemHillshadeNotFoundError(dem_id)
        return self._storage.resolve_read_path(asset.preview_path)

    def get_map_overlay(self, dem_id: UUID) -> DemMapOverlayResult:
        asset = self._require(dem_id)
        if not asset.preview_path or not self._storage.exists(asset.preview_path):
            raise DemHillshadeNotFoundError(dem_id)
        if not self._storage.exists(asset.asset_path):
            raise DemError(
                f"DEM GeoTIFF missing on disk: {asset.asset_path}",
                status_code=404,
            )

        bounds = _bounds_from_json(asset.bounds)
        try:
            coordinates = corners_to_wgs84(
                asset.crs,
                left=bounds.left,
                bottom=bounds.bottom,
                right=bounds.right,
                top=bounds.top,
            )
        except IndexMapOverlayError as exc:
            raise DemError(str(exc)) from exc

        return DemMapOverlayResult(
            dem_id=asset.id,
            image_url=f"/api/v1/dems/{asset.id}/hillshade.png",
            width=asset.width,
            height=asset.height,
            crs_original=asset.crs,
            bounds_original=bounds,
            coordinates_wgs84=coordinates,
        )

    def _require(self, dem_id: UUID) -> DemAsset:
        asset = self.repository.get_by_id(dem_id)
        if asset is None:
            raise DemNotFoundError(dem_id)
        return asset

    def _inspect_dem(self, asset_path: str) -> dict[str, Any]:
        try:
            meta = read_raster_metadata(asset_path, self.data_root)
        except RasterFileNotFoundError as exc:
            raise DemError(str(exc), status_code=404) from exc
        except (RasterReadError, RasterPathError) as exc:
            raise DemError(f"Cannot read DEM GeoTIFF: {exc}") from exc

        if meta.count != 1:
            raise DemError(
                f"DEM must be a single-band raster; got {meta.count} bands"
            )
        if not meta.width or not meta.height or meta.width < 1 or meta.height < 1:
            raise DemError("DEM has invalid width/height")
        if not meta.crs:
            raise DemError("DEM has no CRS; cannot georeference relief")
        transform = meta.transform
        if (
            not transform
            or len(transform) < 6
            or transform[0] == 0
            or transform[4] == 0
        ):
            raise DemError("DEM has an invalid affine transform")
        if not meta.bounds:
            raise DemError("DEM has no bounds")

        try:
            raster = read_raster_array(asset_path, self.data_root)
        except RasterFileNotFoundError as exc:
            raise DemError(str(exc), status_code=404) from exc
        except (RasterReadError, RasterPathError) as exc:
            raise DemError(f"Cannot read DEM GeoTIFF: {exc}") from exc

        valid = raster.data[np.isfinite(raster.data)]
        if valid.size == 0:
            raise DemError("DEM has no valid elevation pixels")

        return {
            "crs": meta.crs,
            "width": int(meta.width),
            "height": int(meta.height),
            "bounds": {
                "left": float(meta.bounds["left"]),
                "bottom": float(meta.bounds["bottom"]),
                "right": float(meta.bounds["right"]),
                "top": float(meta.bounds["top"]),
            },
            "min_elevation": float(np.min(valid)),
            "max_elevation": float(np.max(valid)),
            "metadata": {
                "driver": meta.driver,
                "count": int(meta.count),
                "dtype": meta.dtype,
                "nodata": meta.nodata,
                "transform": [float(v) for v in transform[:6]],
                "resolution": (
                    [float(meta.resolution[0]), float(meta.resolution[1])]
                    if meta.resolution
                    else None
                ),
            },
        }

    @staticmethod
    def _to_read(asset: DemAsset) -> DemAssetRead:
        return DemAssetRead(
            id=asset.id,
            name=asset.name,
            asset_path=asset.asset_path,
            preview_path=asset.preview_path,
            crs=asset.crs,
            width=asset.width,
            height=asset.height,
            bounds=_bounds_from_json(asset.bounds),
            min_elevation=asset.min_elevation,
            max_elevation=asset.max_elevation,
            metadata=asset.metadata_,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
        )


def _bounds_from_json(raw: dict[str, Any] | DemBounds) -> DemBounds:
    if isinstance(raw, DemBounds):
        return raw
    return DemBounds(
        left=float(raw["left"]),
        bottom=float(raw["bottom"]),
        right=float(raw["right"]),
        top=float(raw["top"]),
    )


__all__ = [
    "ALLOWED_DEM_EXTENSIONS",
    "DEM_GEOTIFF_FILENAME",
    "DemError",
    "DemHillshadeNotFoundError",
    "DemNotFoundError",
    "DemService",
    "HILLSHADE_PNG_FILENAME",
]
