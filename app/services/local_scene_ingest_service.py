"""Local GeoTIFF scene ingest under DATA_ROOT (Fase 9A).

Registers ``raster_scenes`` + ``raster_bands`` from a folder of co-registered
bands. Initial support: Landsat 8 Collection 2 Level-2 Surface Reflectance
(``SR_B2``…``SR_B7``). No download, STAC, tiles, or AOI crop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

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
    resolve_asset_path,
)
from app.raster.sensors import (
    LANDSAT_8_BAND_MAP,
    SENSOR_LANDSAT_8,
    detect_sensor,
    normalize_sensor_token,
    resolve_band_key,
)
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
from app.services.local_index_compute_service import LOCAL_INDEX_REGISTRY
from app.services.scene_service import SceneService

GEOTIFF_SUFFIXES = {".tif", ".tiff", ".TIF", ".TIFF"}

LANDSAT_8_REQUIRED_BANDS: tuple[str, ...] = (
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
)

LANDSAT_8_BAND_INFO: dict[str, tuple[str, str, int, str]] = {
    # band_key → (band_name, description, oli_band, wavelength role)
    "SR_B2": ("Blue", "Landsat 8 OLI Surface Reflectance Blue (band 2)", 2, "blue"),
    "SR_B3": ("Green", "Landsat 8 OLI Surface Reflectance Green (band 3)", 3, "green"),
    "SR_B4": ("Red", "Landsat 8 OLI Surface Reflectance Red (band 4)", 4, "red"),
    "SR_B5": ("NIR", "Landsat 8 OLI Surface Reflectance NIR (band 5)", 5, "nir"),
    "SR_B6": ("SWIR1", "Landsat 8 OLI Surface Reflectance SWIR1 (band 6)", 6, "swir1"),
    "SR_B7": ("SWIR2", "Landsat 8 OLI Surface Reflectance SWIR2 (band 7)", 7, "swir2"),
}

# Match native short names and full USGS product filenames.
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


class LocalSceneIngestService:
    def __init__(self, db: Session, *, data_root: Path | str | None = None) -> None:
        self.db = db
        self.repository = SceneRepository(db)
        self.scene_service = SceneService(db)
        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else settings.data_root_path
        )

    def ingest(self, payload: LocalSceneIngestRequest) -> LocalSceneIngestResult:
        warnings: list[IngestionWarning] = []
        relative_scene_path = self._normalize_relative_path(payload.scene_path)
        scene_dir = self._resolve_scene_dir(relative_scene_path)

        geotiffs = self._list_geotiffs(scene_dir)
        if not geotiffs:
            raise LocalIngestError(
                f"No GeoTIFF files found under scene_path '{relative_scene_path}'"
            )

        mtl_meta, mtl_path = self._load_mtl(scene_dir, warnings)
        discovered = self._discover_bands(geotiffs, relative_scene_path, warnings)
        sensor = self._detect_ingest_sensor(
            source=payload.source,
            mtl=mtl_meta,
            discovered_keys=set(discovered),
            warnings=warnings,
        )

        if sensor != SENSOR_LANDSAT_8:
            raise LocalIngestError(
                (
                    f"Local ingest currently supports landsat-8 only; "
                    f"detected sensor '{sensor}'. Pass source='landsat-8' "
                    f"with SR_B2…SR_B7 GeoTIFFs."
                )
            )

        required = self._require_landsat_bands(discovered)
        band_meta = self._read_and_validate_bands(required)

        existing = self.repository.find_by_ingest_scene_path(relative_scene_path)
        overwritten = False
        if existing is not None:
            if not payload.overwrite:
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

        acquisition_date = self._resolve_acquisition_date(
            mtl=mtl_meta,
            scene_dir=scene_dir,
            sample_band=required["SR_B4"].path,
            warnings=warnings,
        )
        name = self._resolve_name(
            payload_name=payload.name,
            mtl=mtl_meta,
            scene_dir=scene_dir,
            relative_scene_path=relative_scene_path,
        )
        footprint = self._footprint_from_raster(band_meta["SR_B4"])
        scene_metadata = self._build_scene_metadata(
            sensor=sensor,
            relative_scene_path=relative_scene_path,
            mtl=mtl_meta,
            mtl_path=mtl_path,
            band_meta=band_meta,
        )

        create_payload = SceneCreate(
            name=name,
            source=SENSOR_LANDSAT_8,
            acquisition_date=acquisition_date,
            cloud_cover=(
                Decimal(str(mtl_meta.cloud_cover))
                if mtl_meta and mtl_meta.cloud_cover is not None
                else None
            ),
            footprint=footprint,
            metadata=scene_metadata,
            bands=[
                self._band_create(band_key, discovered_band, band_meta[band_key])
                for band_key, discovered_band in required.items()
            ],
        )
        created = self.scene_service.create(create_payload)

        registered = [
            IngestedBandInfo(
                band_key=band_key,
                band_name=LANDSAT_8_BAND_INFO[band_key][0],
                asset_path=required[band_key].relative_asset_path,
                width=int(band_meta[band_key].width or 0),
                height=int(band_meta[band_key].height or 0),
                crs=band_meta[band_key].crs,
                dtype=band_meta[band_key].dtype,
                nodata=(
                    str(int(band_meta[band_key].nodata))
                    if band_meta[band_key].nodata is not None
                    and float(band_meta[band_key].nodata).is_integer()
                    else (
                        str(band_meta[band_key].nodata)
                        if band_meta[band_key].nodata is not None
                        else None
                    )
                ),
            )
            for band_key in LANDSAT_8_REQUIRED_BANDS
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
            overwritten=overwritten,
        )

    def _normalize_relative_path(self, scene_path: str) -> str:
        raw = (scene_path or "").strip().replace("\\", "/")
        if not raw:
            raise LocalIngestError("scene_path is empty")
        if Path(raw).is_absolute():
            raise LocalIngestError(
                "scene_path must be relative to DATA_ROOT (absolute paths are not allowed)",
                status_code=422,
            )
        # Reject traversal segments before resolve.
        parts = [p for p in Path(raw).parts if p not in ("", ".")]
        if any(p == ".." for p in parts):
            raise LocalIngestError(
                f"scene_path escapes DATA_ROOT: {scene_path}",
                status_code=422,
            )
        return Path(*parts).as_posix() if parts else ""

    def _resolve_scene_dir(self, relative_scene_path: str) -> Path:
        try:
            resolved = resolve_asset_path(relative_scene_path, self.data_root)
        except RasterPathError as exc:
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
        if detected != SENSOR_LANDSAT_8:
            # Band-name heuristic.
            if discovered_keys & set(LANDSAT_8_BAND_MAP.values()):
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
        return detected

    def _require_landsat_bands(
        self, discovered: dict[str, _DiscoveredBand]
    ) -> dict[str, _DiscoveredBand]:
        missing = [key for key in LANDSAT_8_REQUIRED_BANDS if key not in discovered]
        if missing:
            raise LocalIngestError(
                "Missing required Landsat 8 Surface Reflectance bands: "
                + ", ".join(missing)
                + f". Found: {', '.join(sorted(discovered)) or '(none)'}"
            )
        return {key: discovered[key] for key in LANDSAT_8_REQUIRED_BANDS}

    def _read_and_validate_bands(
        self, bands: dict[str, _DiscoveredBand]
    ) -> dict[str, RasterMetadata]:
        metas: dict[str, RasterMetadata] = {}
        for band_key, discovered in bands.items():
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
            metas[band_key] = meta

        ref_key = "SR_B4"
        ref = metas[ref_key]
        for band_key, meta in metas.items():
            if band_key == ref_key:
                continue
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
        return metas

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
    ) -> dict[str, Any]:
        ref = band_meta["SR_B4"]
        meta: dict[str, Any] = {
            "platform": "Landsat-8",
            "sensor": sensor,
            "ingest_scene_path": relative_scene_path,
            "ingest": {
                "method": "local-scene",
                "phase": "9A",
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
        if mtl:
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
        else:
            # product_id hint from folder when no MTL
            folder = Path(relative_scene_path).name
            if folder.upper().startswith("LC08"):
                meta["product_id"] = folder
        return meta

    @staticmethod
    def _band_create(
        band_key: str,
        discovered: _DiscoveredBand,
        meta: RasterMetadata,
    ) -> BandCreate:
        name, description, oli_band, wavelength = LANDSAT_8_BAND_INFO[band_key]
        nodata: Optional[str] = None
        if meta.nodata is not None:
            if float(meta.nodata).is_integer():
                nodata = str(int(meta.nodata))
            else:
                nodata = str(meta.nodata)

        resolution: Optional[Decimal] = None
        if meta.resolution:
            resolution = Decimal(str(abs(meta.resolution[0])))

        return BandCreate(
            band_key=band_key,
            band_name=name,
            description=description,
            resolution=resolution,
            asset_path=discovered.relative_asset_path,
            nodata=nodata,
            dtype=meta.dtype,
            metadata={
                "platform": "Landsat-8",
                "oli_band": oli_band,
                "wavelength": wavelength,
                "width": meta.width,
                "height": meta.height,
                "crs": meta.crs,
                "transform": meta.transform,
                "bounds": meta.bounds,
            },
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
    "LANDSAT_8_REQUIRED_BANDS",
]
