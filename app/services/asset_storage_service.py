"""Central resolution and validation of raster asset references (Fase 9C).

The database stores metadata and *references* (``asset_path``), never GeoTIFF
bytes. ``asset_path`` is a path relative to the active storage root (today:
``DATA_ROOT`` on the local filesystem).

``DATA_ROOT`` is the app's internal local/dev storage — not a user-facing
upload drop zone. Future backends (persistent volume, object storage / bucket,
UI upload, STAC remote assets) can replace the filesystem implementation
behind this service without changing the DB column ``raster_bands.asset_path``.

Future (not implemented): optional ``asset_uri`` / ``storage_uri`` in metadata
or a dedicated column for ``file://``, ``s3://``, ``https://`` STAC assets.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from app.core.config import settings

# Layout for derived index products (GeoTIFF + PNG previews).
DERIVED_SCENES_PREFIX = "derived/scenes"


class AssetStorageError(Exception):
    """Invalid or unsafe asset path / storage resolution failure."""


class AssetStorageService:
    """Resolve and validate relative ``asset_path`` values against storage root.

    Current backend: local filesystem under ``DATA_ROOT``.
    """

    def __init__(self, data_root: Path | str | None = None) -> None:
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else settings.data_root_path
        )

    def validate_relative_asset_path(self, asset_path: str) -> str:
        """Normalize and validate a relative asset path (no absolute, no ``..``).

        Returns a POSIX-style relative path string safe to store or resolve.
        """
        raw = (asset_path or "").strip().replace("\\", "/")
        if not raw:
            raise AssetStorageError("asset_path is empty")

        candidate = Path(raw)
        if candidate.is_absolute():
            raise AssetStorageError(
                "asset_path must be relative to DATA_ROOT "
                "(absolute paths are not allowed)"
            )

        parts = [p for p in candidate.parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise AssetStorageError(
                f"asset_path escapes DATA_ROOT: {asset_path}"
            )
        if not parts:
            raise AssetStorageError("asset_path is empty")

        return Path(*parts).as_posix()

    def resolve_read_path(self, asset_path: str) -> Path:
        """Resolve a relative ``asset_path`` to an absolute filesystem path for reading.

        Ensures the resolved path stays under ``DATA_ROOT``.
        """
        relative = self.validate_relative_asset_path(asset_path)
        resolved = (self.data_root / relative).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise AssetStorageError(
                f"asset_path escapes DATA_ROOT ({self.data_root}): {asset_path}"
            ) from exc
        return resolved

    def resolve_write_path(self, asset_path: str) -> Path:
        """Resolve a relative ``asset_path`` to an absolute path for writing.

        Same filesystem rules as :meth:`resolve_read_path` today. Kept as a
        separate method so a future object-storage backend can diverge
        (e.g. signed upload URL vs local mkdir).
        """
        return self.resolve_read_path(asset_path)

    def exists(self, asset_path: str) -> bool:
        """Return True if the resolved asset exists as a file under DATA_ROOT."""
        path = self.resolve_read_path(asset_path)
        return path.is_file()

    def build_derived_asset_path(
        self,
        scene_id: UUID | str,
        index_key: str,
        extension: str,
    ) -> str:
        """Build a relative path for a derived index product under DATA_ROOT.

        Convention: ``derived/scenes/{scene_id}/{index_key}.{extension}``
        """
        key = (index_key or "").strip().lower()
        if not key:
            raise AssetStorageError("index_key is empty")

        ext = (extension or "").strip().lstrip(".")
        if not ext:
            raise AssetStorageError("extension is empty")

        return f"{DERIVED_SCENES_PREFIX}/{scene_id}/{key}.{ext}"


__all__ = [
    "AssetStorageError",
    "AssetStorageService",
    "DERIVED_SCENES_PREFIX",
]
