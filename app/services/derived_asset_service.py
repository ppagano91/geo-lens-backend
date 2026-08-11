"""Derived-asset catalog service (Fase 9I / 9J).

Registers metadata + relative path references for products written under
DATA_ROOT. Never stores GeoTIFF/PNG bytes in the database.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.derived_asset import RasterDerivedAsset
from app.repositories.derived_asset_repository import DerivedAssetRepository
from app.schemas.derived_asset import DerivedAssetExistsResult, DerivedAssetRead
from app.services.asset_storage_service import (
    AssetStorageError,
    AssetStorageService,
)
from app.services.scene_service import SceneNotFoundError, SceneService


class DerivedAssetNotFoundError(Exception):
    pass


class DerivedAssetConflictError(Exception):
    """Restore would collide with an already-active catalog row."""


class DerivedAssetService:
    def __init__(self, db: Session) -> None:
        self.db = db
        self.repository = DerivedAssetRepository(db)
        self.storage = AssetStorageService()

    def create_or_update_derived_asset(
        self,
        *,
        scene_id: UUID,
        asset_type: str,
        product_key: str,
        asset_path: str,
        aoi_id: UUID | None = None,
        preview_path: str | None = None,
        georef_path: str | None = None,
        crs: str | None = None,
        width: int | None = None,
        height: int | None = None,
        nodata: str | None = None,
        dtype: str | None = None,
        stats: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        update_preview_path: bool = False,
        update_georef_path: bool = False,
    ) -> DerivedAssetRead:
        """Insert or update a catalog row keyed by scene/type/product/aoi.

        Optional path fields (``preview_path``, ``georef_path``) are only
        overwritten when the corresponding ``update_*`` flag is True, so
        compute-and-save can refresh the GeoTIFF without clearing an existing
        PNG preview path.
        """
        key = (product_key or "").strip().lower()
        asset_type_norm = (asset_type or "").strip().lower()
        if not key:
            raise ValueError("product_key is empty")
        if not asset_type_norm:
            raise ValueError("asset_type is empty")

        existing = self.repository.find_by_scene_aoi_product(
            scene_id,
            asset_type_norm,
            key,
            aoi_id=aoi_id,
            include_inactive=True,
        )

        if existing is None:
            asset = RasterDerivedAsset(
                scene_id=scene_id,
                aoi_id=aoi_id,
                asset_type=asset_type_norm,
                product_key=key,
                asset_path=asset_path,
                preview_path=preview_path,
                georef_path=georef_path,
                crs=crs,
                width=width,
                height=height,
                nodata=nodata,
                dtype=dtype,
                stats=stats,
                metadata_=metadata,
                is_active=True,
                deleted_at=None,
            )
            created = self.repository.create(asset)
            return self._to_read(created)

        existing.asset_path = asset_path
        if update_preview_path:
            existing.preview_path = preview_path
        if update_georef_path:
            existing.georef_path = georef_path
        if crs is not None:
            existing.crs = crs
        if width is not None:
            existing.width = width
        if height is not None:
            existing.height = height
        if nodata is not None:
            existing.nodata = nodata
        if dtype is not None:
            existing.dtype = dtype
        if stats is not None:
            existing.stats = stats
        if metadata is not None:
            existing.metadata_ = metadata
        existing.is_active = True
        existing.deleted_at = None
        existing.aoi_id = aoi_id

        saved = self.repository.save(existing)
        return self._to_read(saved)

    def find_by_scene(
        self,
        scene_id: UUID,
        *,
        include_inactive: bool = False,
    ) -> list[DerivedAssetRead]:
        assets = self.repository.find_by_scene(
            scene_id, include_inactive=include_inactive
        )
        return [self._to_read(asset) for asset in assets]

    def find_by_scene_and_type(
        self,
        scene_id: UUID,
        asset_type: str,
        *,
        include_inactive: bool = False,
    ) -> list[DerivedAssetRead]:
        assets = self.repository.find_by_scene_and_type(
            scene_id,
            asset_type.strip().lower(),
            include_inactive=include_inactive,
        )
        return [self._to_read(asset) for asset in assets]

    def find_by_scene_aoi_product(
        self,
        scene_id: UUID,
        asset_type: str,
        product_key: str,
        *,
        aoi_id: UUID | None = None,
        include_inactive: bool = False,
    ) -> DerivedAssetRead | None:
        asset = self.repository.find_by_scene_aoi_product(
            scene_id,
            asset_type.strip().lower(),
            product_key.strip().lower(),
            aoi_id=aoi_id,
            include_inactive=include_inactive,
        )
        if asset is None:
            return None
        return self._to_read(asset)

    def list_derived_assets(
        self,
        *,
        scene_id: UUID | None = None,
        asset_type: str | None = None,
        product_key: str | None = None,
        aoi_id: UUID | None = None,
        limit: int = 50,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> list[DerivedAssetRead]:
        assets = self.repository.list(
            scene_id=scene_id,
            asset_type=asset_type.strip().lower() if asset_type else None,
            product_key=product_key.strip().lower() if product_key else None,
            aoi_id=aoi_id,
            limit=limit,
            offset=offset,
            include_inactive=include_inactive,
        )
        return [self._to_read(asset) for asset in assets]

    def list_for_scene(
        self,
        scene_id: UUID,
        *,
        asset_type: str | None = None,
        product_key: str | None = None,
        aoi_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> list[DerivedAssetRead]:
        """List derived assets for a scene; validates the scene exists."""
        SceneService(self.db).get(scene_id)
        return self.list_derived_assets(
            scene_id=scene_id,
            asset_type=asset_type,
            product_key=product_key,
            aoi_id=aoi_id,
            limit=limit,
            offset=offset,
            include_inactive=include_inactive,
        )

    def get(
        self,
        asset_id: UUID,
        *,
        include_inactive: bool = False,
    ) -> DerivedAssetRead:
        asset = self.repository.get_by_id(asset_id)
        if asset is None:
            raise DerivedAssetNotFoundError(str(asset_id))
        if not include_inactive and not asset.is_active:
            raise DerivedAssetNotFoundError(str(asset_id))
        return self._to_read(asset)

    def soft_delete(self, asset_id: UUID) -> None:
        """Logically deactivate a catalog row. Does not delete files."""
        asset = self.repository.get_by_id(asset_id)
        if asset is None:
            raise DerivedAssetNotFoundError(str(asset_id))
        self.repository.soft_delete(asset)

    def restore(self, asset_id: UUID) -> DerivedAssetRead:
        """Reactivate a soft-deleted catalog row. Does not move or create files."""
        asset = self.repository.get_by_id(asset_id)
        if asset is None:
            raise DerivedAssetNotFoundError(str(asset_id))
        if asset.is_active:
            return self._to_read(asset)

        conflict = self.repository.find_by_scene_aoi_product(
            asset.scene_id,
            asset.asset_type,
            asset.product_key,
            aoi_id=asset.aoi_id,
            include_inactive=False,
        )
        if conflict is not None and conflict.id != asset.id:
            raise DerivedAssetConflictError(
                f"An active derived asset already exists for "
                f"scene={asset.scene_id} type={asset.asset_type} "
                f"product={asset.product_key} aoi={asset.aoi_id}"
            )

        try:
            restored = self.repository.restore(asset)
        except IntegrityError as exc:
            self.db.rollback()
            raise DerivedAssetConflictError(
                f"Cannot restore derived asset {asset_id}: unique conflict"
            ) from exc
        return self._to_read(restored)

    def check_exists(self, asset_id: UUID) -> DerivedAssetExistsResult:
        """Verify whether catalog path references exist under DATA_ROOT."""
        asset = self.repository.get_by_id(asset_id)
        if asset is None:
            raise DerivedAssetNotFoundError(str(asset_id))

        missing: list[str] = []
        asset_exists = self._path_exists(asset.asset_path, missing)
        preview_exists = True
        if asset.preview_path:
            preview_exists = self._path_exists(asset.preview_path, missing)
        georef_exists = True
        if asset.georef_path:
            georef_exists = self._path_exists(asset.georef_path, missing)

        return DerivedAssetExistsResult(
            asset_id=asset.id,
            asset_exists=asset_exists,
            preview_exists=preview_exists,
            georef_exists=georef_exists,
            missing_paths=missing,
        )

    def _path_exists(self, relative_path: str, missing: list[str]) -> bool:
        try:
            if self.storage.exists(relative_path):
                return True
        except AssetStorageError:
            missing.append(relative_path)
            return False
        missing.append(relative_path)
        return False

    @staticmethod
    def _to_read(asset: RasterDerivedAsset) -> DerivedAssetRead:
        return DerivedAssetRead(
            id=asset.id,
            scene_id=asset.scene_id,
            aoi_id=asset.aoi_id,
            asset_type=asset.asset_type,
            product_key=asset.product_key,
            asset_path=asset.asset_path,
            preview_path=asset.preview_path,
            georef_path=asset.georef_path,
            crs=asset.crs,
            width=asset.width,
            height=asset.height,
            nodata=asset.nodata,
            dtype=asset.dtype,
            stats=asset.stats,
            metadata=asset.metadata_,
            is_active=asset.is_active,
            deleted_at=asset.deleted_at,
            created_at=asset.created_at,
            updated_at=asset.updated_at,
        )


__all__ = [
    "DerivedAssetService",
    "DerivedAssetNotFoundError",
    "DerivedAssetConflictError",
    "SceneNotFoundError",
]
