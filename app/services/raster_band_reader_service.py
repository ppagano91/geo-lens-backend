"""Service: read local GeoTIFF metadata/sample stats for a registered band."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.band import RasterBand
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_metadata,
    read_raster_sample,
)
from app.schemas.raster import (
    RasterBounds,
    RasterMetadataRead,
    RasterSampleStatsRead,
)


class BandNotFoundError(Exception):
    pass


class RasterBandReaderService:
    """Load a RasterBand row and open its asset_path with rasterio."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_metadata(self, band_id: UUID) -> RasterMetadataRead:
        band = self._get_band(band_id)
        meta = read_raster_metadata(band.asset_path, settings.data_root_path)
        bounds = (
            RasterBounds(**meta.bounds) if meta.bounds is not None else None
        )
        return RasterMetadataRead(
            band_id=band.id,
            asset_path=band.asset_path,
            resolved_path=meta.path,
            exists=meta.exists,
            driver=meta.driver,
            width=meta.width,
            height=meta.height,
            count=meta.count,
            dtypes=meta.dtypes,
            crs=meta.crs,
            bounds=bounds,
            nodata=meta.nodata,
            resolution=meta.resolution,
            transform=meta.transform,
            indexes=meta.indexes,
            is_readable=meta.is_readable,
        )

    def get_sample_stats(
        self,
        band_id: UUID,
        max_size: int = 256,
    ) -> RasterSampleStatsRead:
        band = self._get_band(band_id)
        sample = read_raster_sample(
            band.asset_path,
            settings.data_root_path,
            max_size=max_size,
        )
        return RasterSampleStatsRead(
            band_id=band.id,
            asset_path=band.asset_path,
            resolved_path=sample.path,
            sample_shape=sample.sample_shape,
            min=sample.sample_min,
            max=sample.sample_max,
            mean=sample.sample_mean,
            valid_count=sample.valid_count,
            sample_has_nan=sample.sample_has_nan,
        )

    def _get_band(self, band_id: UUID) -> RasterBand:
        stmt = select(RasterBand).where(RasterBand.id == band_id)
        band = self.db.scalars(stmt).one_or_none()
        if band is None:
            raise BandNotFoundError(str(band_id))
        return band


__all__ = [
    "BandNotFoundError",
    "RasterBandReaderService",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
]
