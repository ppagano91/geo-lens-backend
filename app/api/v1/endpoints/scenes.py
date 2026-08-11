from typing import NoReturn
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.band import BandRead
from app.schemas.derived_asset import DerivedAssetRead
from app.schemas.index_compute import (
    IndexAoiCropMapOverlayResult,
    IndexAoiCropRequest,
    IndexAoiCropResult,
    IndexComputeResult,
    IndexComputeSaveResult,
    IndexMapOverlayResult,
    IndexPreviewResult,
    NdviComputeResult,
)
from app.schemas.rgb_composite import (
    RgbCompositeAoiMapOverlayResult,
    RgbCompositeAoiPreviewRequest,
    RgbCompositeAoiPreviewResult,
    RgbCompositeMapOverlayResult,
    RgbCompositePreviewRequest,
    RgbCompositePreviewResult,
)
from app.schemas.scene import SceneCreate, SceneListItem, SceneRead
from app.services.index_aoi_crop_service import (
    IndexAoiCropConflictError,
    IndexAoiCropService,
    IndexAoiNoIntersectionError,
    IndexAoiReprojectionError,
)
from app.services.aoi_service import AoiNotFoundError
from app.services.derived_asset_service import DerivedAssetService
from app.services.index_map_overlay_service import (
    IndexMapOverlayError,
    IndexMapOverlayService,
)
from app.services.index_preview_service import (
    CroppedGeotiffNotFoundError,
    CroppedPreviewPngNotFoundError,
    DerivedGeotiffNotFoundError,
    IndexPreviewService,
    PreviewPngNotFoundError,
    PreviewWriteError,
)
from app.services.local_index_compute_service import (
    IncompatibleRasterBandsError,
    LocalIndexComputeService,
    MissingRequiredBandError,
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    RasterWriteError,
    UnsupportedIndexError,
)
from app.services.rgb_composite_service import (
    RgbAoiCompositePngNotFoundError,
    RgbAoiNoIntersectionError,
    RgbCompositeExistsError,
    RgbCompositePngNotFoundError,
    RgbCompositeService,
    UnsupportedRgbPresetError,
)
from app.services.scene_service import (
    BandKeyDuplicateError,
    GeometryValidationError,
    SceneNotFoundError,
    SceneService,
)

router = APIRouter()


def _raise_index_compute_http(exc: Exception, *, scene_id: UUID) -> NoReturn:
    """Map local index compute domain errors to HTTP responses."""
    if isinstance(exc, SceneNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc
    if isinstance(exc, AoiNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AOI {exc} not found" if str(exc) else "AOI not found",
        ) from exc
    if isinstance(exc, (UnsupportedIndexError, UnsupportedRgbPresetError)):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, MissingRequiredBandError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, IncompatibleRasterBandsError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, RasterFileNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(
        exc,
        (
            PreviewPngNotFoundError,
            DerivedGeotiffNotFoundError,
            CroppedGeotiffNotFoundError,
            CroppedPreviewPngNotFoundError,
            RgbCompositePngNotFoundError,
            RgbAoiCompositePngNotFoundError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (IndexAoiCropConflictError, RgbCompositeExistsError)):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    if isinstance(
        exc,
        (
            IndexAoiNoIntersectionError,
            RgbAoiNoIntersectionError,
            IndexAoiReprojectionError,
            GeometryValidationError,
            IndexMapOverlayError,
        ),
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if isinstance(exc, (RasterPathError, RasterReadError, RasterWriteError, PreviewWriteError)):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    raise exc


@router.post("", response_model=SceneRead, status_code=status.HTTP_201_CREATED)
def create_scene(payload: SceneCreate, db: Session = Depends(get_db)) -> SceneRead:
    service = SceneService(db)
    try:
        return service.create(payload)
    except GeometryValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except BandKeyDuplicateError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[SceneListItem])
def list_scenes(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    include_inactive: bool = Query(
        default=False,
        description="If true, include logically deactivated scenes",
    ),
    db: Session = Depends(get_db),
) -> list[SceneListItem]:
    service = SceneService(db)
    return service.list(
        limit=limit, offset=offset, include_inactive=include_inactive
    )


@router.get("/{scene_id}", response_model=SceneRead)
def get_scene(scene_id: UUID, db: Session = Depends(get_db)) -> SceneRead:
    service = SceneService(db)
    try:
        return service.get(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc


@router.get("/{scene_id}/bands", response_model=list[BandRead])
def list_scene_bands(scene_id: UUID, db: Session = Depends(get_db)) -> list[BandRead]:
    service = SceneService(db)
    try:
        return service.list_bands(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc


@router.get("/{scene_id}/derived-assets", response_model=list[DerivedAssetRead])
def list_scene_derived_assets(
    scene_id: UUID,
    asset_type: str | None = Query(
        default=None,
        description=(
            "Optional filter: index, index_aoi_crop, rgb_composite, rgb_composite_aoi"
        ),
    ),
    product_key: str | None = Query(
        default=None,
        description="Optional product key filter (e.g. ndvi, true_color)",
    ),
    aoi_id: UUID | None = Query(
        default=None,
        description="Optional AOI filter (exact match)",
    ),
    limit: int = Query(default=100, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    include_inactive: bool = Query(
        default=False,
        description="If true, include logically deactivated catalog rows",
    ),
    db: Session = Depends(get_db),
) -> list[DerivedAssetRead]:
    """List derived products registered for a scene (paths + metadata only)."""
    service = DerivedAssetService(db)
    try:
        return service.list_for_scene(
            scene_id,
            asset_type=asset_type,
            product_key=product_key,
            aoi_id=aoi_id,
            limit=limit,
            offset=offset,
            include_inactive=include_inactive,
        )
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc


@router.post(
    "/{scene_id}/indices/ndvi/compute",
    response_model=NdviComputeResult,
    include_in_schema=True,
)
def compute_scene_ndvi(
    scene_id: UUID,
    db: Session = Depends(get_db),
) -> NdviComputeResult:
    """Compatibility alias for NDVI local compute (Fase 7B)."""
    service = LocalIndexComputeService(db)
    try:
        return service.compute_ndvi(scene_id)
    except (
        SceneNotFoundError,
        UnsupportedIndexError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/indices/{index_key}/compute",
    response_model=IndexComputeResult,
)
def compute_scene_index(
    scene_id: UUID,
    index_key: str,
    db: Session = Depends(get_db),
) -> IndexComputeResult:
    """Compute a supported spectral index in-memory from local scene GeoTIFFs."""
    service = LocalIndexComputeService(db)
    try:
        return service.compute_index(scene_id, index_key)
    except (
        SceneNotFoundError,
        UnsupportedIndexError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/indices/{index_key}/compute-and-save",
    response_model=IndexComputeSaveResult,
)
def compute_and_save_scene_index(
    scene_id: UUID,
    index_key: str,
    db: Session = Depends(get_db),
) -> IndexComputeSaveResult:
    """Compute a spectral index and persist it as a derived float32 GeoTIFF."""
    service = LocalIndexComputeService(db)
    try:
        return service.compute_and_save_index(scene_id, index_key)
    except (
        SceneNotFoundError,
        UnsupportedIndexError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        RasterWriteError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/indices/{index_key}/preview",
    response_model=IndexPreviewResult,
)
def create_scene_index_preview(
    scene_id: UUID,
    index_key: str,
    db: Session = Depends(get_db),
) -> IndexPreviewResult:
    """Generate a PNG preview from an existing derived index GeoTIFF."""
    service = IndexPreviewService(db)
    try:
        return service.create_preview(scene_id, index_key)
    except (
        UnsupportedIndexError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        PreviewWriteError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.get(
    "/{scene_id}/indices/{index_key}/preview.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing preview PNG (inline)",
        },
        404: {"description": "Unsupported index or preview PNG missing"},
    },
)
def get_scene_index_preview_png(
    scene_id: UUID,
    index_key: str,
) -> FileResponse:
    """Serve an existing index preview PNG (does not regenerate)."""
    service = IndexPreviewService()
    try:
        path = service.resolve_preview_png(scene_id, index_key)
    except (
        UnsupportedIndexError,
        PreviewPngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{index_key.strip().lower()}.png",
        content_disposition_type="inline",
    )


@router.get(
    "/{scene_id}/indices/{index_key}/map-overlay",
    response_model=IndexMapOverlayResult,
)
def get_scene_index_map_overlay(
    scene_id: UUID,
    index_key: str,
) -> IndexMapOverlayResult:
    """Return MapLibre image-overlay metadata for a derived index PNG.

    Reads CRS/bounds from the existing GeoTIFF and points ``image_url`` at the
    existing preview PNG. Does not generate tiles or missing assets.
    """
    service = IndexMapOverlayService()
    try:
        return service.get_map_overlay(scene_id, index_key)
    except (
        UnsupportedIndexError,
        DerivedGeotiffNotFoundError,
        PreviewPngNotFoundError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        IndexMapOverlayError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.get(
    "/{scene_id}/indices/{index_key}/download.tif",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/tiff": {}},
            "description": "Existing derived index GeoTIFF (attachment)",
        },
        404: {"description": "Unsupported index or derived GeoTIFF missing"},
    },
)
def download_scene_index_geotiff(
    scene_id: UUID,
    index_key: str,
) -> FileResponse:
    """Download an existing derived index GeoTIFF (does not recompute)."""
    service = IndexPreviewService()
    normalized_key = index_key.strip().lower()
    try:
        path = service.resolve_derived_geotiff(scene_id, index_key)
    except (
        UnsupportedIndexError,
        DerivedGeotiffNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/tiff",
        filename=f"{scene_id}_{normalized_key}.tif",
        content_disposition_type="attachment",
    )


@router.get(
    "/{scene_id}/indices/{index_key}/download.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing index preview PNG (attachment)",
        },
        404: {"description": "Unsupported index or preview PNG missing"},
    },
)
def download_scene_index_png(
    scene_id: UUID,
    index_key: str,
) -> FileResponse:
    """Download an existing index preview PNG (does not regenerate)."""
    service = IndexPreviewService()
    normalized_key = index_key.strip().lower()
    try:
        path = service.resolve_preview_png(scene_id, index_key)
    except (
        UnsupportedIndexError,
        PreviewPngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{scene_id}_{normalized_key}.png",
        content_disposition_type="attachment",
    )


@router.post(
    "/{scene_id}/indices/{index_key}/crop-by-aoi",
    response_model=IndexAoiCropResult,
)
def crop_scene_index_by_aoi(
    scene_id: UUID,
    index_key: str,
    payload: IndexAoiCropRequest,
    db: Session = Depends(get_db),
) -> IndexAoiCropResult:
    """Crop an existing derived index GeoTIFF by a saved AOI (Fase 9F).

    Does not crop original bands or recalculate the index. Requires a prior
    ``compute-and-save`` for the full derived GeoTIFF.
    """
    service = IndexAoiCropService(db)
    try:
        return service.crop_by_aoi(
            scene_id,
            index_key,
            payload.aoi_id,
            overwrite=payload.overwrite,
            generate_preview=payload.generate_preview,
        )
    except (
        SceneNotFoundError,
        AoiNotFoundError,
        UnsupportedIndexError,
        DerivedGeotiffNotFoundError,
        IndexAoiCropConflictError,
        IndexAoiNoIntersectionError,
        IndexAoiReprojectionError,
        GeometryValidationError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        RasterWriteError,
        PreviewWriteError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.get(
    "/{scene_id}/indices/{index_key}/aois/{aoi_id}/download.tif",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/tiff": {}},
            "description": "Existing AOI-cropped index GeoTIFF (attachment)",
        },
        404: {"description": "Unsupported index or cropped GeoTIFF missing"},
    },
)
def download_scene_index_aoi_geotiff(
    scene_id: UUID,
    index_key: str,
    aoi_id: UUID,
) -> FileResponse:
    """Download an existing AOI-cropped derived index GeoTIFF."""
    service = IndexAoiCropService()
    normalized_key = index_key.strip().lower()
    try:
        path = service.resolve_cropped_geotiff(scene_id, index_key, aoi_id)
    except (
        UnsupportedIndexError,
        CroppedGeotiffNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/tiff",
        filename=f"{scene_id}_{normalized_key}_{aoi_id}.tif",
        content_disposition_type="attachment",
    )


@router.get(
    "/{scene_id}/indices/{index_key}/aois/{aoi_id}/download.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing AOI-cropped index preview PNG (attachment)",
        },
        404: {"description": "Unsupported index or cropped PNG missing"},
    },
)
def download_scene_index_aoi_png(
    scene_id: UUID,
    index_key: str,
    aoi_id: UUID,
) -> FileResponse:
    """Download an existing AOI-cropped index preview PNG."""
    service = IndexAoiCropService()
    normalized_key = index_key.strip().lower()
    try:
        path = service.resolve_cropped_png(scene_id, index_key, aoi_id)
    except (
        UnsupportedIndexError,
        CroppedPreviewPngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{scene_id}_{normalized_key}_{aoi_id}.png",
        content_disposition_type="attachment",
    )


@router.get(
    "/{scene_id}/indices/{index_key}/aois/{aoi_id}/map-overlay",
    response_model=IndexAoiCropMapOverlayResult,
)
def get_scene_index_aoi_map_overlay(
    scene_id: UUID,
    index_key: str,
    aoi_id: UUID,
) -> IndexAoiCropMapOverlayResult:
    """Return MapLibre image-overlay metadata for an AOI-cropped index PNG."""
    service = IndexAoiCropService()
    try:
        return service.get_map_overlay(scene_id, index_key, aoi_id)
    except (
        UnsupportedIndexError,
        CroppedGeotiffNotFoundError,
        CroppedPreviewPngNotFoundError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        IndexMapOverlayError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/rgb-composites/preview",
    response_model=RgbCompositePreviewResult,
)
def create_scene_rgb_composite_preview(
    scene_id: UUID,
    payload: RgbCompositePreviewRequest,
    db: Session = Depends(get_db),
) -> RgbCompositePreviewResult:
    """Generate an RGB composite PNG from scene bands (Fase 9H).

    Resolves spectral roles via sensor band maps, applies percentile stretch,
    and writes ``derived/scenes/{scene_id}/rgb/{preset}.png``.
    """
    service = RgbCompositeService(db)
    try:
        return service.create_preview(scene_id, payload)
    except (
        SceneNotFoundError,
        UnsupportedRgbPresetError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RgbCompositeExistsError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        PreviewWriteError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.get(
    "/{scene_id}/rgb-composites/{preset}/preview.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing RGB composite PNG (inline)",
        },
        404: {"description": "Unsupported preset or RGB PNG missing"},
    },
)
def get_scene_rgb_composite_preview_png(
    scene_id: UUID,
    preset: str,
) -> FileResponse:
    """Serve an existing RGB composite PNG (does not regenerate)."""
    service = RgbCompositeService()
    normalized = preset.strip().lower()
    try:
        path = service.resolve_preview_png(scene_id, preset)
    except (
        UnsupportedRgbPresetError,
        RgbCompositePngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{normalized}.png",
        content_disposition_type="inline",
    )


@router.get(
    "/{scene_id}/rgb-composites/{preset}/map-overlay",
    response_model=RgbCompositeMapOverlayResult,
)
def get_scene_rgb_composite_map_overlay(
    scene_id: UUID,
    preset: str,
    db: Session = Depends(get_db),
) -> RgbCompositeMapOverlayResult:
    """Return MapLibre image-overlay metadata for an RGB composite PNG.

    Georeferences from a source band of the preset; ``image_url`` points at the
    existing preview PNG. Does not generate tiles or missing assets.
    """
    service = RgbCompositeService(db)
    try:
        return service.get_map_overlay(scene_id, preset)
    except (
        SceneNotFoundError,
        UnsupportedRgbPresetError,
        MissingRequiredBandError,
        RgbCompositePngNotFoundError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        IndexMapOverlayError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.post(
    "/{scene_id}/rgb-composites/preview-by-aoi",
    response_model=RgbCompositeAoiPreviewResult,
)
def create_scene_rgb_composite_preview_by_aoi(
    scene_id: UUID,
    payload: RgbCompositeAoiPreviewRequest,
    db: Session = Depends(get_db),
) -> RgbCompositeAoiPreviewResult:
    """Crop source bands by AOI, then generate an RGB composite PNG (Fase 9H.1).

    Does not generate a full-scene RGB first. Writes
    ``derived/scenes/{scene_id}/aois/{aoi_id}/rgb/{preset}.png``.
    """
    service = RgbCompositeService(db)
    try:
        return service.create_preview_by_aoi(scene_id, payload)
    except (
        SceneNotFoundError,
        AoiNotFoundError,
        UnsupportedRgbPresetError,
        MissingRequiredBandError,
        IncompatibleRasterBandsError,
        RgbCompositeExistsError,
        RgbAoiNoIntersectionError,
        IndexAoiReprojectionError,
        GeometryValidationError,
        RasterFileNotFoundError,
        RasterPathError,
        RasterReadError,
        PreviewWriteError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.get(
    "/{scene_id}/rgb-composites/aois/{aoi_id}/{preset}/preview.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing AOI RGB composite PNG (inline)",
        },
        404: {"description": "Unsupported preset or AOI RGB PNG missing"},
    },
)
def get_scene_rgb_aoi_preview_png(
    scene_id: UUID,
    aoi_id: UUID,
    preset: str,
) -> FileResponse:
    """Serve an existing AOI-cropped RGB composite PNG (does not regenerate)."""
    service = RgbCompositeService()
    normalized = preset.strip().lower()
    try:
        path = service.resolve_aoi_preview_png(scene_id, aoi_id, preset)
    except (
        UnsupportedRgbPresetError,
        RgbAoiCompositePngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{normalized}.png",
        content_disposition_type="inline",
    )


@router.get(
    "/{scene_id}/rgb-composites/aois/{aoi_id}/{preset}/download.png",
    response_class=FileResponse,
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Existing AOI RGB composite PNG (attachment)",
        },
        404: {"description": "Unsupported preset or AOI RGB PNG missing"},
    },
)
def download_scene_rgb_aoi_png(
    scene_id: UUID,
    aoi_id: UUID,
    preset: str,
) -> FileResponse:
    """Download an existing AOI-cropped RGB composite PNG."""
    service = RgbCompositeService()
    normalized = preset.strip().lower()
    try:
        path = service.resolve_aoi_preview_png(scene_id, aoi_id, preset)
    except (
        UnsupportedRgbPresetError,
        RgbAoiCompositePngNotFoundError,
        RasterPathError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)

    return FileResponse(
        path,
        media_type="image/png",
        filename=f"{scene_id}_{aoi_id}_{normalized}.png",
        content_disposition_type="attachment",
    )


@router.get(
    "/{scene_id}/rgb-composites/aois/{aoi_id}/{preset}/map-overlay",
    response_model=RgbCompositeAoiMapOverlayResult,
)
def get_scene_rgb_aoi_map_overlay(
    scene_id: UUID,
    aoi_id: UUID,
    preset: str,
) -> RgbCompositeAoiMapOverlayResult:
    """Return MapLibre overlay metadata for an AOI-cropped RGB composite PNG."""
    service = RgbCompositeService()
    try:
        return service.get_aoi_map_overlay(scene_id, aoi_id, preset)
    except (
        UnsupportedRgbPresetError,
        RgbAoiCompositePngNotFoundError,
        RasterPathError,
        IndexMapOverlayError,
    ) as exc:
        _raise_index_compute_http(exc, scene_id=scene_id)


@router.delete("/{scene_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_scene(scene_id: UUID, db: Session = Depends(get_db)) -> None:
    service = SceneService(db)
    try:
        service.delete(scene_id)
    except SceneNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene {scene_id} not found",
        ) from exc
