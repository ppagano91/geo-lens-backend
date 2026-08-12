"""Local GeoTIFF scene ingest under DATA_ROOT (Fase 9A / 9D / 9K / 9L).

Registers ``raster_scenes`` + ``raster_bands`` from a folder of co-registered
bands. Supported sensors:

* Landsat 8 Collection 2 Level-2 Surface Reflectance (``SR_B2``…``SR_B7``)
* Sentinel-2 L2A / simplified local set at 10 m (``B02``, ``B03``, ``B04``, ``B08``)

Optional Sentinel-2 SWIR bands (``B11``, ``B12``, typically 20 m) are registered
when already co-registered with the 10 m grid, or after bilinear resampling /
alignment onto the reference 10 m grid (prefer ``B08``, else ``B04``). Aligned
assets are written under ``derived/scenes/{scene_id}/aligned/`` without moving
or deleting the originals.

Supports path-based ingest (9A) and UI upload (9D/9K/9L). No STAC, tiles, or AOI crop.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from rasterio.warp import transform_bounds
from sqlalchemy.orm import Session

from app.core.config import settings
from app.raster.mtl import MtlMetadata, find_mtl_file, parse_mtl_file
from app.raster.readers import (
    RasterFileNotFoundError,
    RasterMetadata,
    RasterPathError,
    RasterReadError,
    read_raster_metadata,
)
from app.raster.sensors import (
    LANDSAT_8_BAND_MAP,
    SENSOR_LANDSAT_8,
    SENSOR_SENTINEL_2,
    SENTINEL_2_BAND_MAP,
    detect_sensor,
    normalize_sensor_token,
    resolve_band_key,
)
from app.raster.sentinel_safe import is_safe_metadata_filename
from app.repositories.scene_repository import SceneRepository
from app.schemas.band import BandCreate
from app.schemas.ingest import (
    AvailableIndexInfo,
    IngestedBandInfo,
    IngestionWarning,
    LocalSceneIngestRequest,
    LocalSceneIngestResult,
)
from app.schemas.scene import SceneCreate
from app.services.asset_storage_service import AssetStorageError, AssetStorageService
from app.services.band_alignment_service import (
    BandAlignmentError,
    BandAlignmentService,
)
from app.services.local_index_compute_service import LOCAL_INDEX_REGISTRY
from app.services.radiometry_service import RadiometryService
from app.services.scene_service import SceneService

GEOTIFF_SUFFIXES = {".tif", ".tiff", ".TIF", ".TIFF"}
UPLOAD_GEOTIFF_SUFFIXES = {".tif", ".tiff"}
UPLOAD_MTL_SUFFIXES = {".txt"}
UPLOAD_SAFE_METADATA_SUFFIXES = {".xml", ".safe"}
UPLOAD_ALLOWED_SUFFIXES = (
    UPLOAD_GEOTIFF_SUFFIXES | UPLOAD_MTL_SUFFIXES | UPLOAD_SAFE_METADATA_SUFFIXES
)

SUPPORTED_INGEST_SENSORS: frozenset[str] = frozenset(
    {SENSOR_LANDSAT_8, SENSOR_SENTINEL_2}
)

LANDSAT_8_REQUIRED_BANDS: tuple[str, ...] = (
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
)

SENTINEL_2_REQUIRED_BANDS: tuple[str, ...] = (
    "B02",
    "B03",
    "B04",
    "B08",
)

# Native 20 m SWIR — aligned to 10 m when needed (Fase 9L).
SENTINEL_2_OPTIONAL_BANDS: tuple[str, ...] = (
    "B11",
    "B12",
)

# Preferred 10 m reference for SWIR alignment (fallback: SENSOR_REF_BAND).
SENTINEL_2_ALIGNMENT_REF_BAND = "B08"

LANDSAT_8_BAND_INFO: dict[str, tuple[str, str, int, str]] = {
    # band_key → (band_name, description, native_band_number, wavelength role)
    "SR_B2": ("Blue", "Landsat 8 OLI Surface Reflectance Blue (band 2)", 2, "blue"),
    "SR_B3": ("Green", "Landsat 8 OLI Surface Reflectance Green (band 3)", 3, "green"),
    "SR_B4": ("Red", "Landsat 8 OLI Surface Reflectance Red (band 4)", 4, "red"),
    "SR_B5": ("NIR", "Landsat 8 OLI Surface Reflectance NIR (band 5)", 5, "nir"),
    "SR_B6": ("SWIR1", "Landsat 8 OLI Surface Reflectance SWIR1 (band 6)", 6, "swir1"),
    "SR_B7": ("SWIR2", "Landsat 8 OLI Surface Reflectance SWIR2 (band 7)", 7, "swir2"),
}

SENTINEL_2_BAND_INFO: dict[str, tuple[str, str, int, str]] = {
    "B02": ("Blue", "Sentinel-2 MSI Blue (B02, 10 m)", 2, "blue"),
    "B03": ("Green", "Sentinel-2 MSI Green (B03, 10 m)", 3, "green"),
    "B04": ("Red", "Sentinel-2 MSI Red (B04, 10 m)", 4, "red"),
    "B08": ("NIR", "Sentinel-2 MSI NIR (B08, 10 m)", 8, "nir"),
    "B11": ("SWIR1", "Sentinel-2 MSI SWIR1 (B11, 20 m)", 11, "swir1"),
    "B12": ("SWIR2", "Sentinel-2 MSI SWIR2 (B12, 20 m)", 12, "swir2"),
}

SENSOR_PLATFORM_LABEL: dict[str, str] = {
    SENSOR_LANDSAT_8: "Landsat-8",
    SENSOR_SENTINEL_2: "Sentinel-2",
}

SENSOR_REF_BAND: dict[str, str] = {
    SENSOR_LANDSAT_8: "SR_B4",
    SENSOR_SENTINEL_2: "B04",
}

# Match native short names and full USGS / ESA product filenames.
_LANDSAT_SR_BAND_RE = re.compile(
    r"(?:^|[_-])SR_B([2-7])(?:[_.-]|$)",
    re.IGNORECASE,
)
_SENTINEL_BAND_RE = re.compile(
    r"(?:^|[_-])B(0[2348]|1[12])(?:[_.-]|$)",
    re.IGNORECASE,
)
_DATE_IN_NAME_RE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")


class LocalIngestError(Exception):
    """Domain error for local scene ingest (mapped to HTTP by the endpoint)."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class SceneAlreadyExistsError(LocalIngestError):
    def __init__(self, scene_path: str, scene_id: str) -> None:
        super().__init__(
            (
                f"Scene already ingested from '{scene_path}' "
                f"(scene_id={scene_id}). Pass overwrite=true to replace it."
            ),
            status_code=409,
        )
        self.scene_path = scene_path
        self.existing_scene_id = scene_id


@dataclass(frozen=True)
class _DiscoveredBand:
    band_key: str
    path: Path
    relative_asset_path: str
    # Optional alignment / resampling metadata (Fase 9L).
    alignment_meta: Optional[dict[str, Any]] = field(default=None)


@dataclass(frozen=True)
class _AlignedSwir:
    discovered: _DiscoveredBand
    meta: RasterMetadata


@dataclass(frozen=True)
class UploadedSceneFile:
    """In-memory upload payload before writing under DATA_ROOT."""

    filename: str
    content: bytes


class LocalSceneIngestService:
    def __init__(self, db: Session, *, data_root: Path | str | None = None) -> None:
        self.db = db
        self.repository = SceneRepository(db)
        self.scene_service = SceneService(db)
        self._storage = AssetStorageService(data_root)
        self._alignment = BandAlignmentService(data_root)
        self._radiometry = RadiometryService()

    @property
    def data_root(self) -> Path:
        return self._storage.data_root

    @data_root.setter
    def data_root(self, value: Path | str) -> None:
        self._storage = AssetStorageService(value)
        self._alignment = BandAlignmentService(value)

    def ingest(self, payload: LocalSceneIngestRequest) -> LocalSceneIngestResult:
        """Register a scene from a folder already present under DATA_ROOT (9A)."""
        return self.ingest_prepared_folder(
            scene_path=payload.scene_path,
            source=payload.source,
            name=payload.name,
            overwrite=payload.overwrite,
            ingest_method="local-scene",
            phase="9A",
            product_level=payload.product_level,
            source_product_id=payload.source_product_id,
        )

    def ingest_upload(
        self,
        *,
        files: Sequence[UploadedSceneFile],
        source: str,
        name: Optional[str] = None,
        overwrite: bool = False,
        product_level: Optional[str] = None,
        source_product_id: Optional[str] = None,
    ) -> LocalSceneIngestResult:
        """Save uploaded bands under storage and register the scene (9D / 9K / 9L / 9M.1)."""
        if not files:
            raise LocalIngestError("No files uploaded; attach GeoTIFF bands (.tif/.tiff)")

        source_sensor = normalize_sensor_token(source) if source else None
        if source_sensor not in SUPPORTED_INGEST_SENSORS:
            raise LocalIngestError(
                (
                    "Upload ingest supports source='landsat-8' or "
                    f"source='sentinel-2'; got '{source}'."
                )
            )

        prepared = self._prepare_upload_files(files)
        scene_slug = str(uuid4())
        relative_scene_path = self._storage.build_uploaded_scene_dir(scene_slug)
        scene_dir = self._storage.resolve_write_path(relative_scene_path)
        scene_dir.mkdir(parents=True, exist_ok=False)

        phase = "9L" if source_sensor == SENSOR_SENTINEL_2 else "9D"
        try:
            for filename, content in prepared:
                dest = scene_dir / filename
                dest.write_bytes(content)

            return self.ingest_prepared_folder(
                scene_path=relative_scene_path,
                source=source,
                name=name,
                overwrite=overwrite,
                ingest_method="upload-scene",
                phase=phase,
                product_level=product_level,
                source_product_id=source_product_id,
            )
        except Exception:
            # Drop orphaned upload folder when registration fails.
            if scene_dir.exists():
                shutil.rmtree(scene_dir, ignore_errors=True)
            raise

    def ingest_prepared_folder(
        self,
        *,
        scene_path: str,
        source: str,
        name: Optional[str] = None,
        overwrite: bool = False,
        ingest_method: str = "local-scene",
        phase: str = "9A",
        product_level: Optional[str] = None,
        source_product_id: Optional[str] = None,
    ) -> LocalSceneIngestResult:
        """Shared register path for an already-prepared scene folder under DATA_ROOT."""
        warnings: list[IngestionWarning] = []
        relative_scene_path = self._normalize_relative_path(scene_path)
        scene_dir = self._resolve_scene_dir(relative_scene_path)

        geotiffs = self._list_geotiffs(scene_dir)
        if not geotiffs:
            raise LocalIngestError(
                f"No GeoTIFF files found under scene_path '{relative_scene_path}'"
            )

        discovered = self._discover_bands(geotiffs, relative_scene_path, warnings)
        sensor = self._detect_ingest_sensor(
            source=source,
            mtl=None,
            discovered_keys=set(discovered),
            warnings=warnings,
        )

        if sensor not in SUPPORTED_INGEST_SENSORS:
            raise LocalIngestError(
                (
                    "Local ingest currently supports landsat-8 and sentinel-2; "
                    f"detected sensor '{sensor}'. Pass source='landsat-8' "
                    "(SR_B2…SR_B7) or source='sentinel-2' (B02/B03/B04/B08)."
                )
            )

        # Landsat MTL enrichment only; Sentinel-2 uses SAFE/other metadata later.
        mtl_meta: Optional[MtlMetadata] = None
        mtl_path: Optional[Path] = None
        # Pre-allocate scene id so aligned SWIR assets can live under
        # derived/scenes/{scene_id}/aligned/ before the DB insert (9L).
        scene_id: Optional[UUID] = None
        if sensor == SENSOR_LANDSAT_8:
            mtl_meta, mtl_path = self._load_mtl(scene_dir, warnings)
            effective_phase = phase
        else:
            # Sentinel-2 path/upload ingest with optional SWIR alignment is Fase 9L.
            effective_phase = "9L" if phase in ("9A", "9D", "9K", "9L", "9M", "9M.1") else phase
            scene_id = uuid4()

        # Resolve overwrite before writing aligned derived assets.
        existing = self.repository.find_by_ingest_scene_path(relative_scene_path)
        overwritten = False
        if existing is not None:
            if not overwrite:
                raise SceneAlreadyExistsError(
                    relative_scene_path, str(existing.id)
                )
            self.scene_service.delete(existing.id)
            overwritten = True
            warnings.append(
                IngestionWarning(
                    code="scene_overwritten",
                    title="Escena sobrescrita",
                    description=(
                        f"Se reemplazó la escena existente {existing.id} "
                        f"para la ruta '{relative_scene_path}'."
                    ),
                    severity="info",
                )
            )

        if sensor == SENSOR_LANDSAT_8:
            required = self._require_bands(
                discovered,
                LANDSAT_8_REQUIRED_BANDS,
                product_label="Landsat 8 Surface Reflectance",
            )
            band_meta = self._read_and_validate_bands(
                required, ref_key=SENSOR_REF_BAND[SENSOR_LANDSAT_8]
            )
            registered_order = list(LANDSAT_8_REQUIRED_BANDS)
            bands_for_create = required
        else:
            required = self._require_bands(
                discovered,
                SENTINEL_2_REQUIRED_BANDS,
                product_label="Sentinel-2 (10 m)",
            )
            band_meta = self._read_and_validate_bands(
                required, ref_key=SENSOR_REF_BAND[SENSOR_SENTINEL_2]
            )
            align_ref_key = (
                SENTINEL_2_ALIGNMENT_REF_BAND
                if SENTINEL_2_ALIGNMENT_REF_BAND in band_meta
                else SENSOR_REF_BAND[SENSOR_SENTINEL_2]
            )
            assert scene_id is not None
            optional_meta = self._resolve_optional_sentinel_swir(
                discovered,
                ref_key=align_ref_key,
                ref_meta=band_meta[align_ref_key],
                scene_id=scene_id,
                warnings=warnings,
            )
            band_meta.update(optional_meta)
            registered_order = list(SENTINEL_2_REQUIRED_BANDS) + [
                key for key in SENTINEL_2_OPTIONAL_BANDS if key in optional_meta
            ]
            bands_for_create = {
                key: discovered[key] for key in registered_order
            }

        ref_key = SENSOR_REF_BAND[sensor]
        acquisition_date = self._resolve_acquisition_date(
            mtl=mtl_meta,
            scene_dir=scene_dir,
            sample_band=bands_for_create[ref_key].path,
            warnings=warnings,
        )
        resolved_name = self._resolve_name(
            payload_name=name,
            mtl=mtl_meta,
            scene_dir=scene_dir,
            relative_scene_path=relative_scene_path,
        )
        footprint = self._footprint_from_raster(band_meta[ref_key])
        scene_metadata = self._build_scene_metadata(
            sensor=sensor,
            relative_scene_path=relative_scene_path,
            mtl=mtl_meta,
            mtl_path=mtl_path,
            band_meta=band_meta,
            ref_key=ref_key,
            ingest_method=ingest_method,
            phase=effective_phase,
        )

        safe_files: list[Path] = []
        if sensor == SENSOR_SENTINEL_2:
            safe_files = self._radiometry.discover_safe_metadata(scene_dir)

        radiometry = self._radiometry.detect_scene_radiometry(
            source=sensor,
            name=resolved_name,
            metadata=scene_metadata,
            band_keys=registered_order,
            product_id=(
                source_product_id
                or scene_metadata.get("product_id")
                or scene_metadata.get("landsat_product_id")
                or resolved_name
            ),
            product_level=product_level,
            source_product_id=source_product_id,
            scene_path=relative_scene_path,
            metadata_files=safe_files,
            prefer_stored=False,
        )
        scene_metadata = self._radiometry.merge_into_scene_metadata(
            scene_metadata, radiometry
        )
        self._append_radiometry_warnings(warnings, radiometry)

        create_payload = SceneCreate(
            id=scene_id,
            name=resolved_name,
            source=sensor,
            acquisition_date=acquisition_date,
            cloud_cover=(
                Decimal(str(mtl_meta.cloud_cover))
                if mtl_meta and mtl_meta.cloud_cover is not None
                else None
            ),
            footprint=footprint,
            metadata=scene_metadata,
            bands=[
                self._band_create(
                    sensor,
                    band_key,
                    bands_for_create[band_key],
                    band_meta[band_key],
                    radiometry=radiometry,
                )
                for band_key in registered_order
            ],
        )
        created = self.scene_service.create(create_payload)

        band_info = self._band_info_for_sensor(sensor)
        registered = [
            self._ingested_band_info(
                band_key=band_key,
                band_name=band_info[band_key][0],
                discovered=bands_for_create[band_key],
                meta=band_meta[band_key],
            )
            for band_key in registered_order
        ]

        return LocalSceneIngestResult(
            scene_id=created.id,
            name=created.name,
            source=created.source,
            sensor=sensor,
            acquisition_date=created.acquisition_date,
            scene_path=relative_scene_path,
            bands=registered,
            warnings=warnings,
            available_indices=self._available_indices(
                sensor, {b.band_key for b in registered}
            ),
            metadata=created.metadata,
            radiometry=radiometry.to_info(),
            metadata_files_detected=list(radiometry.metadata_files_detected),
            overwritten=overwritten,
        )

    def _prepare_upload_files(
        self, files: Sequence[UploadedSceneFile]
    ) -> list[tuple[str, bytes]]:
        prepared: list[tuple[str, bytes]] = []
        seen_names: set[str] = set()
        max_bytes = int(getattr(settings, "max_upload_file_bytes", 0) or 0)

        for item in files:
            safe_name = self._safe_upload_filename(item.filename)
            suffix = Path(safe_name).suffix.lower()
            if suffix not in UPLOAD_ALLOWED_SUFFIXES:
                raise LocalIngestError(
                    f"Invalid file extension for '{item.filename}'. "
                    f"Allowed: .tif, .tiff, .txt (MTL), .xml / .safe (Sentinel metadata)."
                )

            content = item.content
            if max_bytes > 0 and len(content) > max_bytes:
                raise LocalIngestError(
                    f"File '{safe_name}' exceeds max upload size "
                    f"({max_bytes} bytes)."
                )

            if safe_name.lower() in seen_names:
                raise LocalIngestError(
                    f"Duplicate uploaded filename '{safe_name}'."
                )
            seen_names.add(safe_name.lower())
            prepared.append((safe_name, content))

        has_geotiff = any(
            Path(name).suffix.lower() in UPLOAD_GEOTIFF_SUFFIXES for name, _ in prepared
        )
        if not has_geotiff:
            raise LocalIngestError(
                "Upload must include at least one GeoTIFF (.tif/.tiff) band file."
            )

        return prepared

    @staticmethod
    def _safe_upload_filename(filename: str) -> str:
        raw = (filename or "").strip()
        if not raw:
            raise LocalIngestError("Uploaded file is missing a filename.")
        # Reject path separators / traversal; keep basename only.
        name = Path(raw.replace("\\", "/")).name
        if not name or name in (".", ".."):
            raise LocalIngestError(f"Invalid uploaded filename: {filename!r}")
        if "/" in name or "\\" in name:
            raise LocalIngestError(f"Invalid uploaded filename: {filename!r}")
        return name

    def _normalize_relative_path(self, scene_path: str) -> str:
        try:
            return self._storage.validate_relative_asset_path(scene_path)
        except AssetStorageError as exc:
            # Surface scene_path wording for ingest API clarity.
            message = str(exc).replace("asset_path", "scene_path")
            raise LocalIngestError(message, status_code=422) from exc

    def _resolve_scene_dir(self, relative_scene_path: str) -> Path:
        try:
            resolved = self._storage.resolve_read_path(relative_scene_path)
        except (AssetStorageError, RasterPathError) as exc:
            raise LocalIngestError(str(exc), status_code=422) from exc

        if not resolved.exists():
            raise LocalIngestError(
                f"scene_path does not exist under DATA_ROOT: {relative_scene_path}",
                status_code=422,
            )
        if not resolved.is_dir():
            raise LocalIngestError(
                f"scene_path must be a directory: {relative_scene_path}",
                status_code=422,
            )
        return resolved

    def _list_geotiffs(self, scene_dir: Path) -> list[Path]:
        files = [
            p
            for p in sorted(scene_dir.iterdir())
            if p.is_file() and p.suffix in GEOTIFF_SUFFIXES
        ]
        if files:
            return files
        # Fall back to one level of nesting (full USGS product folders sometimes nest).
        nested: list[Path] = []
        for child in sorted(scene_dir.iterdir()):
            if child.is_dir():
                nested.extend(
                    p
                    for p in sorted(child.iterdir())
                    if p.is_file() and p.suffix in GEOTIFF_SUFFIXES
                )
        return nested

    def _load_mtl(
        self, scene_dir: Path, warnings: list[IngestionWarning]
    ) -> tuple[Optional[MtlMetadata], Optional[Path]]:
        mtl_path = find_mtl_file(scene_dir)
        if mtl_path is None:
            warnings.append(
                IngestionWarning(
                    code="no_mtl_file",
                    title="MTL no encontrado",
                    description=(
                        "No se encontró MTL.txt; se infieren la fecha de adquisición "
                        "y metadatos del producto desde GeoTIFF / nombre de carpeta."
                    ),
                    severity="warning",
                )
            )
            return None, None
        try:
            return parse_mtl_file(mtl_path), mtl_path
        except OSError as exc:
            warnings.append(
                IngestionWarning(
                    code="mtl_read_failed",
                    title="Error al leer MTL",
                    description=f"No se pudo leer el archivo MTL {mtl_path.name}: {exc}",
                    items=[mtl_path.name],
                    severity="warning",
                )
            )
            return None, mtl_path

    def _discover_bands(
        self,
        geotiffs: list[Path],
        relative_scene_path: str,
        warnings: list[IngestionWarning],
    ) -> dict[str, _DiscoveredBand]:
        discovered: dict[str, _DiscoveredBand] = {}
        unused: list[str] = []
        for path in geotiffs:
            band_key = self._band_key_from_filename(path.name)
            if band_key is None:
                unused.append(path.name)
                continue
            rel = f"{relative_scene_path}/{path.name}".replace("\\", "/")
            # Prefer paths relative to DATA_ROOT even if nested one level.
            try:
                rel = path.resolve().relative_to(self.data_root).as_posix()
            except ValueError:
                pass
            if band_key in discovered:
                warnings.append(
                    IngestionWarning(
                        code="duplicate_band_key",
                        title="Banda duplicada",
                        description=(
                            f"Clave de banda {band_key} duplicada: se mantiene "
                            f"{discovered[band_key].path.name}."
                        ),
                        items=[path.name],
                        severity="warning",
                    )
                )
                continue
            discovered[band_key] = _DiscoveredBand(
                band_key=band_key,
                path=path,
                relative_asset_path=rel,
            )
        if unused:
            warnings.append(
                IngestionWarning(
                    code="ignored_non_band_geotiffs",
                    title="Archivos GeoTIFF ignorados",
                    description=(
                        "Se encontraron archivos GeoTIFF que no corresponden "
                        "a bandas registrables."
                    ),
                    items=sorted(unused),
                    severity="warning",
                )
            )
        return discovered

    @staticmethod
    def _band_key_from_filename(filename: str) -> Optional[str]:
        stem_and_name = filename
        match = _LANDSAT_SR_BAND_RE.search(stem_and_name)
        if match:
            return f"SR_B{match.group(1)}"
        match = _SENTINEL_BAND_RE.search(stem_and_name)
        if match:
            return f"B{match.group(1)}"
        # Exact short names: SR_B4.tif / B04.tif
        stem = Path(filename).stem.upper()
        if re.fullmatch(r"SR_B[2-7]", stem):
            return stem
        if re.fullmatch(r"B(0[2348]|1[12])", stem):
            return stem
        return None

    def _detect_ingest_sensor(
        self,
        *,
        source: str,
        mtl: Optional[MtlMetadata],
        discovered_keys: set[str],
        warnings: list[IngestionWarning],
    ) -> str:
        meta: dict[str, Any] = {}
        if mtl:
            if mtl.spacecraft_id:
                meta["platform"] = mtl.spacecraft_id.replace("_", "-")
            if mtl.sensor_id:
                meta["sensor"] = mtl.sensor_id

        # Explicit source wins when it maps to a known sensor.
        source_sensor = normalize_sensor_token(source) if source else None
        if source_sensor:
            return source_sensor

        detected = detect_sensor(source=source, metadata=meta or None)
        if detected == SENSOR_LANDSAT_8:
            return detected

        landsat_keys = set(LANDSAT_8_BAND_MAP.values())
        sentinel_keys = set(SENTINEL_2_BAND_MAP.values())
        if discovered_keys & landsat_keys:
            warnings.append(
                IngestionWarning(
                    code="sensor_inferred_from_bands",
                    title="Sensor inferido",
                    description=(
                        "Sensor inferido como landsat-8 a partir de los "
                        "nombres de archivo SR_B*."
                    ),
                    severity="info",
                )
            )
            return SENSOR_LANDSAT_8
        if discovered_keys & sentinel_keys:
            warnings.append(
                IngestionWarning(
                    code="sensor_inferred_from_bands",
                    title="Sensor inferido",
                    description=(
                        "Sensor inferido como sentinel-2 a partir de los "
                        "nombres de archivo B0* / B1*."
                    ),
                    severity="info",
                )
            )
            return SENSOR_SENTINEL_2
        return detected

    @staticmethod
    def _require_bands(
        discovered: dict[str, _DiscoveredBand],
        required_keys: Sequence[str],
        *,
        product_label: str,
    ) -> dict[str, _DiscoveredBand]:
        missing = [key for key in required_keys if key not in discovered]
        if missing:
            raise LocalIngestError(
                f"Missing required {product_label} bands: "
                + ", ".join(missing)
                + f". Found: {', '.join(sorted(discovered)) or '(none)'}"
            )
        return {key: discovered[key] for key in required_keys}

    def _require_landsat_bands(
        self, discovered: dict[str, _DiscoveredBand]
    ) -> dict[str, _DiscoveredBand]:
        """Backward-compatible wrapper used by older tests/callers."""
        return self._require_bands(
            discovered,
            LANDSAT_8_REQUIRED_BANDS,
            product_label="Landsat 8 Surface Reflectance",
        )

    def _read_and_validate_bands(
        self,
        bands: dict[str, _DiscoveredBand],
        *,
        ref_key: str,
    ) -> dict[str, RasterMetadata]:
        metas: dict[str, RasterMetadata] = {}
        for band_key, discovered in bands.items():
            metas[band_key] = self._read_single_band_meta(band_key, discovered)

        if ref_key not in metas:
            raise LocalIngestError(
                f"Reference band {ref_key} is required for grid validation"
            )

        ref = metas[ref_key]
        for band_key, meta in metas.items():
            if band_key == ref_key:
                continue
            self._assert_aligned(ref_key, ref, band_key, meta)
        return metas

    def _read_single_band_meta(
        self, band_key: str, discovered: _DiscoveredBand
    ) -> RasterMetadata:
        try:
            meta = read_raster_metadata(
                discovered.relative_asset_path, self.data_root
            )
        except (RasterFileNotFoundError, RasterReadError, RasterPathError) as exc:
            raise LocalIngestError(
                f"Cannot read band {band_key} ({discovered.path.name}): {exc}"
            ) from exc

        if meta.count != 1:
            raise LocalIngestError(
                f"Band {band_key} must be a single-band GeoTIFF; "
                f"got count={meta.count} ({discovered.path.name})"
            )
        return meta

    @staticmethod
    def _assert_aligned(
        ref_key: str,
        ref: RasterMetadata,
        band_key: str,
        meta: RasterMetadata,
    ) -> None:
        if meta.crs != ref.crs:
            raise LocalIngestError(
                f"Band CRS mismatch: {ref_key} has {ref.crs}, "
                f"{band_key} has {meta.crs}"
            )
        if meta.width != ref.width or meta.height != ref.height:
            raise LocalIngestError(
                f"Band size mismatch: {ref_key} is {ref.width}x{ref.height}, "
                f"{band_key} is {meta.width}x{meta.height}"
            )
        if meta.transform != ref.transform:
            raise LocalIngestError(
                f"Band transform mismatch between {ref_key} and {band_key}"
            )

    def _resolve_optional_sentinel_swir(
        self,
        discovered: dict[str, _DiscoveredBand],
        *,
        ref_key: str,
        ref_meta: RasterMetadata,
        scene_id: UUID,
        warnings: list[IngestionWarning],
    ) -> dict[str, RasterMetadata]:
        """Accept B11/B12 when aligned, or resample 20 m → 10 m (Fase 9L).

        Already co-registered SWIR bands are registered as-is. Misaligned native
        20 m rasters are bilinear-resampled onto the reference 10 m grid
        (prefer B08) and registered with ``band_key`` B11/B12 pointing at the
        aligned asset under ``derived/scenes/{scene_id}/aligned/``.
        """
        accepted: dict[str, RasterMetadata] = {}
        resampled_items: list[str] = []
        detected_20m: list[str] = []

        if ref_meta.width is None or ref_meta.height is None:
            raise LocalIngestError(
                f"Reference band {ref_key} is missing width/height for SWIR alignment"
            )
        if not ref_meta.transform:
            raise LocalIngestError(
                f"Reference band {ref_key} is missing transform for SWIR alignment"
            )

        for band_key in SENTINEL_2_OPTIONAL_BANDS:
            if band_key not in discovered:
                continue
            discovered_band = discovered[band_key]
            try:
                meta = self._read_single_band_meta(band_key, discovered_band)
            except LocalIngestError as exc:
                raise LocalIngestError(
                    f"Cannot read optional Sentinel-2 band {band_key}: {exc.message}"
                ) from exc

            try:
                self._assert_aligned(ref_key, ref_meta, band_key, meta)
            except LocalIngestError:
                detected_20m.append(discovered_band.path.name)
                aligned = self._align_sentinel_swir_band(
                    band_key=band_key,
                    discovered_band=discovered_band,
                    ref_key=ref_key,
                    ref_meta=ref_meta,
                    scene_id=scene_id,
                )
                discovered[band_key] = aligned.discovered
                accepted[band_key] = aligned.meta
                resampled_items.append(aligned.discovered.relative_asset_path)
                continue

            accepted[band_key] = meta

        if detected_20m:
            warnings.append(
                IngestionWarning(
                    code="sentinel_swir_20m_detected",
                    title="B11/B12 a 20 m detectadas",
                    description=(
                        "Se detectaron bandas SWIR Sentinel-2 a resolución nativa "
                        "distinta de la grilla de 10 m. Se aplicará resampling "
                        "bilinear a la grilla de referencia."
                    ),
                    items=detected_20m,
                    severity="info",
                )
            )

        if resampled_items:
            warnings.append(
                IngestionWarning(
                    code="sentinel_swir_resampled",
                    title="Resampling SWIR 20 m → 10 m aplicado",
                    description=(
                        f"B11/B12 se alinearon a la grilla de 10 m usando "
                        f"{ref_key} como referencia (bilinear). Los archivos "
                        f"originales no se modifican ni se eliminan. NBR/NDMI y "
                        f"composiciones SWIR quedan habilitados cuando ambas "
                        f"bandas están disponibles."
                    ),
                    items=resampled_items,
                    severity="info",
                )
            )

        return accepted

    def _align_sentinel_swir_band(
        self,
        *,
        band_key: str,
        discovered_band: _DiscoveredBand,
        ref_key: str,
        ref_meta: RasterMetadata,
        scene_id: UUID,
    ) -> _AlignedSwir:
        """Resample one SWIR band onto the 10 m reference grid and update discovery."""
        dest_rel = self._storage.build_aligned_band_asset_path(
            scene_id, f"{band_key}_10m.tif"
        )
        try:
            result = self._alignment.align_to_reference(
                source_asset_path=discovered_band.relative_asset_path,
                destination_asset_path=dest_rel,
                reference_crs=ref_meta.crs,
                reference_transform=ref_meta.transform or [],
                reference_width=int(ref_meta.width or 0),
                reference_height=int(ref_meta.height or 0),
                original_band_key=band_key,
                aligned_band_key=band_key,
                reference_band=ref_key,
            )
        except BandAlignmentError as exc:
            raise LocalIngestError(
                f"Failed to resample Sentinel-2 {band_key} to 10 m grid "
                f"(reference {ref_key}): {exc.message}"
            ) from exc

        alignment_meta = result.as_metadata()
        alignment_meta["original_asset_path"] = discovered_band.relative_asset_path

        aligned_discovered = _DiscoveredBand(
            band_key=band_key,
            path=result.absolute_path,
            relative_asset_path=result.relative_asset_path,
            alignment_meta=alignment_meta,
        )
        try:
            aligned_meta = self._read_single_band_meta(band_key, aligned_discovered)
            self._assert_aligned(ref_key, ref_meta, band_key, aligned_meta)
        except LocalIngestError as exc:
            raise LocalIngestError(
                f"Aligned {band_key} failed grid validation against {ref_key}: "
                f"{exc.message}"
            ) from exc

        return _AlignedSwir(discovered=aligned_discovered, meta=aligned_meta)

    def _resolve_acquisition_date(
        self,
        *,
        mtl: Optional[MtlMetadata],
        scene_dir: Path,
        sample_band: Path,
        warnings: list[IngestionWarning],
    ) -> date:
        if mtl and mtl.date_acquired:
            return mtl.date_acquired

        for text in (scene_dir.name, sample_band.name):
            match = _DATE_IN_NAME_RE.search(text)
            if match:
                try:
                    parsed = datetime.strptime(match.group(1), "%Y%m%d").date()
                    warnings.append(
                        IngestionWarning(
                            code="acquisition_date_from_name",
                            title="Fecha de adquisición inferida",
                            description=(
                                f"Fecha de adquisición inferida del token de "
                                f"nombre {match.group(1)}."
                            ),
                            severity="info",
                        )
                    )
                    return parsed
                except ValueError:
                    continue

        mtime = datetime.fromtimestamp(sample_band.stat().st_mtime, tz=timezone.utc)
        warnings.append(
            IngestionWarning(
                code="acquisition_date_from_mtime",
                title="Fecha de adquisición por mtime",
                description=(
                    "Fecha de adquisición no disponible; se usa el mtime del "
                    f"archivo de banda ({mtime.date().isoformat()})."
                ),
                items=[sample_band.name],
                severity="warning",
            )
        )
        return mtime.date()

    @staticmethod
    def _resolve_name(
        *,
        payload_name: Optional[str],
        mtl: Optional[MtlMetadata],
        scene_dir: Path,
        relative_scene_path: str,
    ) -> str:
        if payload_name and payload_name.strip():
            return payload_name.strip()
        if mtl and mtl.landsat_product_id:
            return mtl.landsat_product_id
        return scene_dir.name or Path(relative_scene_path).name

    def _footprint_from_raster(self, meta: RasterMetadata) -> dict[str, Any]:
        if not meta.bounds:
            raise LocalIngestError("Reference band has no bounds; cannot build footprint")
        left = meta.bounds["left"]
        bottom = meta.bounds["bottom"]
        right = meta.bounds["right"]
        top = meta.bounds["top"]

        if meta.crs and meta.crs not in ("EPSG:4326", "OGC:CRS84"):
            try:
                left, bottom, right, top = transform_bounds(
                    meta.crs,
                    "EPSG:4326",
                    left,
                    bottom,
                    right,
                    top,
                    densify_pts=21,
                )
            except Exception as exc:  # noqa: BLE001 — surface as ingest error
                raise LocalIngestError(
                    f"Cannot reproject band bounds to EPSG:4326 from {meta.crs}: {exc}"
                ) from exc

        ring = [
            [left, bottom],
            [right, bottom],
            [right, top],
            [left, top],
            [left, bottom],
        ]
        return {"type": "Polygon", "coordinates": [ring]}

    def _build_scene_metadata(
        self,
        *,
        sensor: str,
        relative_scene_path: str,
        mtl: Optional[MtlMetadata],
        mtl_path: Optional[Path],
        band_meta: dict[str, RasterMetadata],
        ref_key: str,
        ingest_method: str = "local-scene",
        phase: str = "9A",
    ) -> dict[str, Any]:
        ref = band_meta[ref_key]
        platform = SENSOR_PLATFORM_LABEL.get(sensor, sensor)
        meta: dict[str, Any] = {
            "platform": platform,
            "sensor": sensor,
            "ingest_scene_path": relative_scene_path,
            "ingest": {
                "method": ingest_method,
                "phase": phase,
                "scene_path": relative_scene_path,
            },
            "crs": ref.crs,
            "width": ref.width,
            "height": ref.height,
            "transform": ref.transform,
            "bounds": ref.bounds,
            "dtype": ref.dtype,
            "nodata": ref.nodata,
        }
        if sensor == SENSOR_LANDSAT_8 and mtl:
            meta.update(mtl.as_dict())
            # Keep display platform consistent for detect_sensor / UI.
            meta["platform"] = "Landsat-8"
            meta["sensor"] = SENSOR_LANDSAT_8
            if mtl_path is not None:
                try:
                    meta["mtl_path"] = mtl_path.resolve().relative_to(
                        self.data_root
                    ).as_posix()
                except ValueError:
                    meta["mtl_path"] = mtl_path.name
        elif sensor == SENSOR_LANDSAT_8:
            folder = Path(relative_scene_path).name
            if folder.upper().startswith("LC08"):
                meta["product_id"] = folder
        else:
            folder = Path(relative_scene_path).name
            upper = folder.upper()
            if upper.startswith("S2") or "MSIL1C" in upper or "MSIL2A" in upper:
                meta["product_id"] = folder
        return meta

    @staticmethod
    def _band_info_for_sensor(sensor: str) -> dict[str, tuple[str, str, int, str]]:
        if sensor == SENSOR_LANDSAT_8:
            return LANDSAT_8_BAND_INFO
        if sensor == SENSOR_SENTINEL_2:
            return SENTINEL_2_BAND_INFO
        raise LocalIngestError(f"No band info table for sensor '{sensor}'")

    @staticmethod
    def _append_radiometry_warnings(
        warnings: list[IngestionWarning],
        radiometry,
    ) -> None:
        source = (radiometry.radiometry_source or "").lower()
        if radiometry.product_level == "unknown":
            warnings.append(
                IngestionWarning(
                    code="radiometry_unknown",
                    title="Radiometry could not be determined automatically",
                    description=(
                        radiometry.radiometry_warning
                        or (
                            "Indicate MSIL1C/MSIL2A manually or upload SAFE metadata."
                        )
                    ),
                    severity="warning",
                )
            )
            return
        if source == "manual_override":
            warnings.append(
                IngestionWarning(
                    code="radiometry_manual_override",
                    title="Radiometría indicada manualmente",
                    description=(
                        f"product_level={radiometry.product_level}; "
                        f"tipo={radiometry.radiometry_type}."
                    ),
                    severity="info",
                )
            )
        elif source == "manual_product_id":
            warnings.append(
                IngestionWarning(
                    code="radiometry_detected_from_product_id",
                    title="Radiometría detectada desde Product ID",
                    description=radiometry.source_product_id,
                    severity="info",
                )
            )
        elif source == "sentinel_metadata":
            warnings.append(
                IngestionWarning(
                    code="radiometry_detected_from_safe_metadata",
                    title="Radiometría detectada desde metadata SAFE",
                    description=(
                        ", ".join(radiometry.metadata_files_detected)
                        if radiometry.metadata_files_detected
                        else None
                    ),
                    items=list(radiometry.metadata_files_detected),
                    severity="info",
                )
            )

    @staticmethod
    def _ingested_band_info(
        *,
        band_key: str,
        band_name: str,
        discovered: _DiscoveredBand,
        meta: RasterMetadata,
    ) -> IngestedBandInfo:
        nodata: Optional[str] = None
        if meta.nodata is not None:
            if float(meta.nodata).is_integer():
                nodata = str(int(meta.nodata))
            else:
                nodata = str(meta.nodata)
        return IngestedBandInfo(
            band_key=band_key,
            band_name=band_name,
            asset_path=discovered.relative_asset_path,
            width=int(meta.width or 0),
            height=int(meta.height or 0),
            crs=meta.crs,
            dtype=meta.dtype,
            nodata=nodata,
            metadata=discovered.alignment_meta,
        )

    def _band_create(
        self,
        sensor: str,
        band_key: str,
        discovered: _DiscoveredBand,
        meta: RasterMetadata,
        *,
        radiometry=None,
    ) -> BandCreate:
        name, description, native_band, wavelength = self._band_info_for_sensor(
            sensor
        )[band_key]
        nodata: Optional[str] = None
        if meta.nodata is not None:
            if float(meta.nodata).is_integer():
                nodata = str(int(meta.nodata))
            else:
                nodata = str(meta.nodata)

        resolution: Optional[Decimal] = None
        if meta.resolution:
            resolution = Decimal(str(abs(meta.resolution[0])))

        platform = SENSOR_PLATFORM_LABEL.get(sensor, sensor)
        band_meta: dict[str, Any] = {
            "platform": platform,
            "wavelength": wavelength,
            "width": meta.width,
            "height": meta.height,
            "crs": meta.crs,
            "transform": meta.transform,
            "bounds": meta.bounds,
        }
        if sensor == SENSOR_LANDSAT_8:
            band_meta["oli_band"] = native_band
        else:
            band_meta["msi_band"] = native_band

        if discovered.alignment_meta:
            band_meta.update(discovered.alignment_meta)

        if radiometry is not None:
            band_meta.update(self._radiometry.band_radiometry_slice(radiometry))

        return BandCreate(
            band_key=band_key,
            band_name=name,
            description=description,
            resolution=resolution,
            asset_path=discovered.relative_asset_path,
            nodata=nodata,
            dtype=meta.dtype,
            metadata=band_meta,
        )

    @staticmethod
    def _available_indices(
        sensor: str, present_keys: set[str]
    ) -> list[AvailableIndexInfo]:
        results: list[AvailableIndexInfo] = []
        for key, spec in LOCAL_INDEX_REGISTRY.items():
            missing: list[str] = []
            for role in spec.formula_roles:
                try:
                    physical = resolve_band_key(sensor, role)
                except KeyError:
                    missing.append(role)
                    continue
                if physical not in present_keys:
                    missing.append(role)
            results.append(
                AvailableIndexInfo(
                    index_key=key,
                    display_name=spec.display_name,
                    compatible=not missing,
                    missing_roles=missing,
                )
            )
        return results


__all__ = [
    "LocalSceneIngestService",
    "LocalIngestError",
    "SceneAlreadyExistsError",
    "UploadedSceneFile",
    "LANDSAT_8_REQUIRED_BANDS",
    "SENTINEL_2_REQUIRED_BANDS",
    "SENTINEL_2_OPTIONAL_BANDS",
    "SENTINEL_2_ALIGNMENT_REF_BAND",
    "SUPPORTED_INGEST_SENSORS",
    "UPLOAD_ALLOWED_SUFFIXES",
]
