"""Generate and serve PNG previews from derived spectral-index GeoTIFFs.

Fase 7E: create PNG from float32 GeoTIFF under DATA_ROOT/derived/scenes/.
Fase 7E.1: resolve an existing PNG for HTTP serving (no regeneration).
Fase 8C: resolve existing GeoTIFF / PNG for attachment download.

Does not recompute indices, touch the DB, or modify compute / compute-and-save.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

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
from app.services.asset_storage_service import AssetStorageService
from app.services.local_index_compute_service import (
    LOCAL_INDEX_REGISTRY,
    UnsupportedIndexError,
)


class PreviewPngNotFoundError(Exception):
    """Derived preview PNG is missing; POST .../preview must be run first."""

    def __init__(self, scene_id: UUID, index_key: str, asset_path: str) -> None:
        self.scene_id = scene_id
        self.index_key = index_key
        self.asset_path = asset_path
        super().__init__(
            f"Preview PNG not found for scene {scene_id} index '{index_key}' "
            f"at '{asset_path}'. Generate it first with "
            f"POST /api/v1/scenes/{scene_id}/indices/{index_key}/preview"
        )


class DerivedGeotiffNotFoundError(Exception):
    """Derived index GeoTIFF is missing; POST .../compute-and-save must run first."""

    def __init__(self, scene_id: UUID, index_key: str, asset_path: str) -> None:
        self.scene_id = scene_id
        self.index_key = index_key
        self.asset_path = asset_path
        super().__init__(
            f"Derived GeoTIFF not found for scene {scene_id} index '{index_key}' "
            f"at '{asset_path}'. Generate it first with "
            f"POST /api/v1/scenes/{scene_id}/indices/{index_key}/compute-and-save"
        )


class IndexPreviewService:
    """Orchestrate derived GeoTIFF → RGBA PNG preview for a scene index."""

    def __init__(self, data_root: Path | str | None = None) -> None:
        self._storage = AssetStorageService(data_root)

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    @data_root.setter
    def data_root(self, value: Path | str) -> None:
        self._storage = AssetStorageService(value)

    def create_preview(self, scene_id: UUID, index_key: str) -> IndexPreviewResult:
        """Read a derived index GeoTIFF and write a colocated PNG preview."""
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        input_asset = self._storage.build_derived_asset_path(
            scene_id, spec.key, "tif"
        )
        output_asset = self._storage.build_derived_asset_path(
            scene_id, spec.key, "png"
        )

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

    def resolve_preview_png(self, scene_id: UUID, index_key: str) -> Path:
        """Return the absolute path of an existing preview PNG (serve only).

        Does not generate or overwrite the file. Raises if the index is
        unsupported or the PNG is missing.
        """
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        asset_path = self._storage.build_derived_asset_path(
            scene_id, spec.key, "png"
        )
        if not self._storage.exists(asset_path):
            raise PreviewPngNotFoundError(scene_id, spec.key, asset_path)
        return self._storage.resolve_read_path(asset_path)

    def resolve_derived_geotiff(self, scene_id: UUID, index_key: str) -> Path:
        """Return the absolute path of an existing derived index GeoTIFF.

        Does not compute or overwrite. Raises if the index is unsupported
        or the GeoTIFF is missing.
        """
        normalized_key = index_key.strip().lower()
        spec = LOCAL_INDEX_REGISTRY.get(normalized_key)
        if spec is None:
            raise UnsupportedIndexError(normalized_key)

        asset_path = self._storage.build_derived_asset_path(
            scene_id, spec.key, "tif"
        )
        if not self._storage.exists(asset_path):
            raise DerivedGeotiffNotFoundError(scene_id, spec.key, asset_path)
        return self._storage.resolve_read_path(asset_path)


__all__ = [
    "IndexPreviewService",
    "DerivedGeotiffNotFoundError",
    "PreviewPngNotFoundError",
    "PreviewWriteError",
    "UnsupportedIndexError",
    "RasterFileNotFoundError",
    "RasterPathError",
    "RasterReadError",
    "DEFAULT_INDEX_NODATA",
]
