"""Endpoints for reading local raster band files (metadata / sample stats)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
)
from app.schemas.raster import RasterMetadataRead, RasterSampleStatsRead
from app.services.raster_band_reader_service import (
    BandNotFoundError,
    RasterBandReaderService,
)

router = APIRouter()


@router.get("/{band_id}/metadata", response_model=RasterMetadataRead)
def get_raster_band_metadata(
    band_id: UUID,
    db: Session = Depends(get_db),
) -> RasterMetadataRead:
    """Open the GeoTIFF registered for a band and return basic metadata."""
    service = RasterBandReaderService(db)
    try:
        return service.get_metadata(band_id)
    except BandNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster band {band_id} not found",
        ) from exc
    except RasterFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RasterPathError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RasterReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("/{band_id}/sample-stats", response_model=RasterSampleStatsRead)
def get_raster_band_sample_stats(
    band_id: UUID,
    max_size: int = Query(default=256, ge=1, le=1024),
    db: Session = Depends(get_db),
) -> RasterSampleStatsRead:
    """Return summary stats from a downsampled sample of band 1 (no full array)."""
    service = RasterBandReaderService(db)
    try:
        return service.get_sample_stats(band_id, max_size=max_size)
    except BandNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raster band {band_id} not found",
        ) from exc
    except RasterFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RasterPathError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RasterReadError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
