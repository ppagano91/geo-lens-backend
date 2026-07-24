"""Generate PNG previews from derived spectral-index GeoTIFFs (Fase 7E).

Reads existing float32 GeoTIFFs under DATA_ROOT/derived/scenes/{scene_id}/;
does not recompute indices, touch the DB, or modify compute / compute-and-save.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.core.config import settings
from app.raster.preview import (
    PreviewWriteError,
    render_index_preview_rgba,
    write_preview_png,
)
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterPathError,
    RasterReadError,
    read_raster_array,
)
from app.raster.writers import DEFAULT_INDEX_NODATA
from app.schemas.index_compute import (
    IndexPreviewInputInfo,
    IndexPreviewOutputInfo,
    IndexPreviewResult,
)
from app.services.local_index_compute_service import (
    LOCAL_INDEX_REGISTRY,
    UnsupportedIndexError,
)


class IndexPreviewService:
    """Orchestrate derived GeoTIFF → RGBA PNG preview for a scene index."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else settings.data_root_path
        )

    def create_preview(self, scene_id: UUID, index_key: str) -> IndexPreviewResult:
        """Read a derived index GeoTIFF and write a colocated PNG preview."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        input_asset = self._derived_tif_path(scene_id, spec.key)
        output_asset = self._derived_png_path(scene_id, spec.key)

        raster = read_raster_array(input_asset, self.data_root)
        rgba = render_index_preview_rgba(
            raster.data,
            spec.key,
            nodata=DEFAULT_INDEX_NODATA,
        )
        resolved = write_preview_png(output_asset, self.data_root, rgba)

        return IndexPreviewResult(
            scene_id=scene_id,
            index=spec.display_name,
            status="preview_created",
            input=IndexPreviewInputInfo(asset_path=input_asset),
            output=IndexPreviewOutputInfo(
                asset_path=output_asset,
                resolved_path=str(resolved),
            ),
            width=raster.width,
            height=raster.height,
        )

    @staticmethod
    def _derived_tif_path(scene_id: UUID, index_key: str) -> str:
        return f"derived/scenes/{scene_id}/{index_key}.tif"

    @staticmethod
    def _derived_png_path(scene_id: UUID, index_key: str) -> str:
        return f"derived/scenes/{scene_id}/{index_key}.png"


__all__ = [
    "IndexPreviewService",
    "PreviewWriteError",
    "UnsupportedIndexError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "DEFAULT_INDEX_NODATA",
]
