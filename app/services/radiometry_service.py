"""Radiometric metadata detection and scaling (Fase 9M / 9M.1).

Detects product level / radiometry type for Landsat 8 and Sentinel-2, applies
Collection 2 / ESA scale+offset when known, and exposes compact metadata for
API responses and ``raster_* .metadata`` JSON.

Fase 9M.1 priority (ingest / explicit hints):

1. manual ``product_level`` override
2. ``source_product_id`` / product id containing MSIL1C / MSIL2A
3. SAFE auxiliary metadata files
4. path / folder segments
5. existing heuristics (scene name / bands)
6. unknown + warning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from app.raster.sensors import (
    SENSOR_LANDSAT_8,
    SENSOR_SENTINEL_2,
    SENSOR_SYNTHETIC_SENTINEL_2,
    detect_sensor,
    normalize_sensor_token,
)
from app.raster.sentinel_safe import (
    detect_level_from_path,
    detect_level_from_text,
    find_safe_metadata_files,
    sniff_safe_file,
    summarize_safe_hits,
)
from app.schemas.radiometry import RadiometryInfo

# Landsat Collection 2 Level-2 Surface Reflectance (OLI).
LANDSAT_L2_SCALE = 0.0000275
LANDSAT_L2_OFFSET = -0.2

# Sentinel-2 L1C / L2A reflectance scale (quantification value 10000).
SENTINEL_REFLECTANCE_SCALE = 0.0001
SENTINEL_REFLECTANCE_OFFSET = 0.0

UNKNOWN_RADIOMETRY_WARNING = (
    "Radiometry could not be determined automatically. "
    "Indicate MSIL1C/MSIL2A manually or upload SAFE metadata."
)

RADIOMETRY_TYPE_LABELS: dict[str, str] = {
    "dn": "DN",
    "toa_reflectance": "TOA Reflectance",
    "surface_reflectance": "Surface Reflectance",
    "synthetic": "Synthetic",
    "unknown": "DN / Unknown",
}


@dataclass(frozen=True)
class RadiometryMetadata:
    """Internal radiometry payload used across ingest / compute / RGB."""

    product_level: str = "unknown"
    radiometry_type: str = "unknown"
    scale_factor: Optional[float] = None
    offset: Optional[float] = None
    scale_applied: bool = False
    source_product_id: Optional[str] = None
    radiometry_source: str = "unknown"
    radiometry_warning: Optional[str] = None
    metadata_files_detected: tuple[str, ...] = ()

    def as_scene_metadata(self) -> dict[str, Any]:
        """Flat keys stored under ``raster_scenes.metadata``."""
        payload: dict[str, Any] = {
            "product_level": self.product_level,
            "radiometry_type": self.radiometry_type,
            "scale_factor": self.scale_factor,
            "offset": self.offset,
            "scale_applied": self.scale_applied,
            "source_product_id": self.source_product_id,
            "radiometry_source": self.radiometry_source,
            "radiometry_warning": self.radiometry_warning,
        }
        if self.metadata_files_detected:
            payload["metadata_files_detected"] = list(self.metadata_files_detected)
        return payload

    def as_nested_metadata(self) -> dict[str, Any]:
        """Nested ``{"radiometry": {...}}`` for derived-asset metadata."""
        return {"radiometry": self.as_response_dict()}

    def as_response_dict(self) -> dict[str, Any]:
        """API-facing block (``warning`` alias of ``radiometry_warning``)."""
        payload = {
            "product_level": self.product_level,
            "radiometry_type": self.radiometry_type,
            "scale_factor": self.scale_factor,
            "offset": self.offset,
            "scale_applied": self.scale_applied,
            "source_product_id": self.source_product_id,
            "radiometry_source": self.radiometry_source,
            "warning": self.radiometry_warning,
        }
        if self.metadata_files_detected:
            payload["metadata_files_detected"] = list(self.metadata_files_detected)
        return payload

    def to_info(self) -> RadiometryInfo:
        data = self.as_response_dict()
        data.pop("metadata_files_detected", None)
        return RadiometryInfo.model_validate(data)

    def with_files(self, files: Sequence[str]) -> RadiometryMetadata:
        if not files:
            return self
        return RadiometryMetadata(
            product_level=self.product_level,
            radiometry_type=self.radiometry_type,
            scale_factor=self.scale_factor,
            offset=self.offset,
            scale_applied=self.scale_applied,
            source_product_id=self.source_product_id,
            radiometry_source=self.radiometry_source,
            radiometry_warning=self.radiometry_warning,
            metadata_files_detected=tuple(files),
        )


class RadiometryService:
    """Detect radiometry, apply scale/offset, and build response metadata."""

    def detect_scene_radiometry(
        self,
        scene: Any = None,
        bands: Sequence[Any] | None = None,
        metadata_files: Sequence[Any] | None = None,
        *,
        source: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        band_keys: Sequence[str] | None = None,
        product_id: str | None = None,
        product_level: str | None = None,
        source_product_id: str | None = None,
        scene_path: str | Path | None = None,
        prefer_stored: bool = True,
    ) -> RadiometryMetadata:
        """Resolve radiometry for a scene from overrides, SAFE metadata, heuristics."""
        scene_meta = self._coerce_metadata(scene, metadata)
        if prefer_stored:
            stored = self._from_stored_metadata(scene_meta)
            if stored is not None and not product_level and not source_product_id:
                # Recompute paths still honor persisted ingest radiometry unless
                # the caller supplies an explicit override (ingest).
                return stored

        resolved_source = source
        if resolved_source is None and scene is not None:
            resolved_source = getattr(scene, "source", None)
        resolved_name = name
        if resolved_name is None and scene is not None:
            resolved_name = getattr(scene, "name", None)

        keys = list(band_keys or [])
        if not keys and bands is not None:
            keys = [
                str(getattr(b, "band_key", b) if not isinstance(b, str) else b)
                for b in bands
            ]

        explicit_product_id = _optional_str(source_product_id) or _optional_str(
            product_id
        )
        if explicit_product_id is None and scene_meta:
            explicit_product_id = (
                _optional_str(scene_meta.get("source_product_id"))
                or _optional_str(scene_meta.get("product_id"))
                or _optional_str(scene_meta.get("landsat_product_id"))
            )
        fallback_name_id = explicit_product_id or _optional_str(resolved_name)

        sensor = detect_sensor(source=resolved_source, metadata=scene_meta)
        token = normalize_sensor_token(sensor) or sensor

        safe_paths = self._normalize_metadata_files(metadata_files)
        safe_names: list[str] = []
        safe_level: Optional[str] = None
        safe_product_id: Optional[str] = None
        if safe_paths:
            hits = [sniff_safe_file(path) for path in safe_paths]
            safe_level, safe_product_id, safe_names = summarize_safe_hits(hits)

        path_level = (
            detect_level_from_path(scene_path)
            if scene_path is not None
            else None
        )
        if path_level is None and scene_meta:
            path_level = detect_level_from_path(
                str(
                    scene_meta.get("ingest_scene_path")
                    or scene_meta.get("product_id")
                    or ""
                )
            )

        # --- Priority 1: manual product_level override ---
        manual_level = _normalize_product_level(product_level)
        if manual_level is not None:
            return self._from_explicit_level(
                manual_level,
                sensor=token,
                product_id=explicit_product_id or safe_product_id or fallback_name_id,
                radiometry_source="manual_override",
                metadata_files=safe_names,
            )

        # --- Priority 2: source_product_id / explicit product id markers ---
        if explicit_product_id:
            from_pid = detect_level_from_text(explicit_product_id)
            if from_pid in {"sentinel_l1c", "sentinel_l2a"}:
                return self._sentinel_level(
                    from_pid,
                    product_id=explicit_product_id,
                    radiometry_source="manual_product_id",
                    metadata_files=safe_names,
                )
            if "L2SP" in explicit_product_id.upper() or "LC08_L2" in explicit_product_id.upper():
                return self._landsat_l2(
                    product_id=explicit_product_id,
                    radiometry_source="manual_product_id",
                )

        # --- Priority 3: SAFE metadata files ---
        if safe_level in {"sentinel_l1c", "sentinel_l2a"}:
            return self._sentinel_level(
                safe_level,
                product_id=explicit_product_id or safe_product_id or fallback_name_id,
                radiometry_source="sentinel_metadata",
                metadata_files=safe_names,
            )

        # --- Priority 4: path / folder segments ---
        if path_level in {"sentinel_l1c", "sentinel_l2a"}:
            return self._sentinel_level(
                path_level,
                product_id=explicit_product_id or safe_product_id or fallback_name_id,
                radiometry_source="sentinel_path",
                metadata_files=safe_names,
            )

        # --- Priority 5: legacy heuristics (name / bands / MTL) ---
        heuristic = self._detect_from_signals(
            sensor=token,
            product_id=fallback_name_id,
            band_keys=keys,
            scene_meta=scene_meta,
        )
        if safe_names:
            heuristic = heuristic.with_files(safe_names)
        if (
            heuristic.product_level != "unknown"
            and not heuristic.source_product_id
            and (explicit_product_id or safe_product_id)
        ):
            heuristic = RadiometryMetadata(
                product_level=heuristic.product_level,
                radiometry_type=heuristic.radiometry_type,
                scale_factor=heuristic.scale_factor,
                offset=heuristic.offset,
                scale_applied=heuristic.scale_applied,
                source_product_id=explicit_product_id or safe_product_id,
                radiometry_source=heuristic.radiometry_source,
                radiometry_warning=heuristic.radiometry_warning,
                metadata_files_detected=heuristic.metadata_files_detected,
            )
        return heuristic

    def apply_radiometric_scaling(
        self,
        array: np.ndarray,
        nodata: float | None,
        radiometry: RadiometryMetadata | Mapping[str, Any],
    ) -> np.ndarray:
        """Apply ``reflectance = DN * scale + offset`` when ``scale_applied``.

        Nodata / NaN pixels stay NaN and do not participate in scaling math.
        """
        meta = self._coerce_radiometry(radiometry)
        data = np.asarray(array, dtype=np.float32)
        if not meta.scale_applied:
            return data

        scale = float(meta.scale_factor) if meta.scale_factor is not None else 1.0
        offset = float(meta.offset) if meta.offset is not None else 0.0

        out = data.copy()
        valid = np.isfinite(out)
        if nodata is not None and not (
            isinstance(nodata, float) and np.isnan(nodata)
        ):
            valid &= out != np.float32(nodata)
            valid &= out != float(nodata)

        out[valid] = out[valid] * np.float32(scale) + np.float32(offset)
        out[~valid] = np.float32(np.nan)
        return out

    def build_radiometry_response_metadata(
        self,
        radiometry: RadiometryMetadata | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the compact API radiometry dict."""
        return self._coerce_radiometry(radiometry).as_response_dict()

    def merge_into_scene_metadata(
        self,
        scene_metadata: Mapping[str, Any] | None,
        radiometry: RadiometryMetadata,
    ) -> dict[str, Any]:
        """Copy scene metadata and stamp flat radiometry fields."""
        merged = dict(scene_metadata or {})
        merged.update(radiometry.as_scene_metadata())
        return merged

    def band_radiometry_slice(
        self,
        radiometry: RadiometryMetadata,
    ) -> dict[str, Any]:
        """Subset of radiometry fields stored on each band."""
        return {
            "product_level": radiometry.product_level,
            "radiometry_type": radiometry.radiometry_type,
            "scale_factor": radiometry.scale_factor,
            "offset": radiometry.offset,
            "scale_applied": radiometry.scale_applied,
            "radiometry_source": radiometry.radiometry_source,
        }

    def discover_safe_metadata(
        self,
        scene_dir: Path,
        *,
        extra_files: Sequence[Path] | None = None,
    ) -> list[Path]:
        """Find SAFE metadata near a scene folder, plus any explicit extras."""
        found = list(find_safe_metadata_files(scene_dir))
        if extra_files:
            for path in extra_files:
                if path not in found and path.is_file():
                    found.append(path)
        return found

    # --- builders ----------------------------------------------------------

    def _from_explicit_level(
        self,
        level: str,
        *,
        sensor: str,
        product_id: str | None,
        radiometry_source: str,
        metadata_files: Sequence[str] = (),
    ) -> RadiometryMetadata:
        if level == "unknown":
            return self._unknown(
                product_id=product_id,
                metadata_files=metadata_files,
            )
        if level in {"sentinel_l1c", "sentinel_l2a"}:
            return self._sentinel_level(
                level,
                product_id=product_id,
                radiometry_source=radiometry_source,
                metadata_files=metadata_files,
            )
        if level == "landsat_l2":
            return self._landsat_l2(
                product_id=product_id,
                radiometry_source=radiometry_source,
            )
        # Unexpected manual value: fall back carefully.
        if normalize_sensor_token(sensor) == SENSOR_LANDSAT_8:
            return self._landsat_l2(
                product_id=product_id,
                radiometry_source=radiometry_source,
            )
        return self._unknown(product_id=product_id, metadata_files=metadata_files)

    def _sentinel_level(
        self,
        level: str,
        *,
        product_id: str | None,
        radiometry_source: str,
        metadata_files: Sequence[str] = (),
    ) -> RadiometryMetadata:
        if level == "sentinel_l1c":
            return RadiometryMetadata(
                product_level="sentinel_l1c",
                radiometry_type="toa_reflectance",
                scale_factor=SENTINEL_REFLECTANCE_SCALE,
                offset=SENTINEL_REFLECTANCE_OFFSET,
                scale_applied=True,
                source_product_id=product_id,
                radiometry_source=radiometry_source,
                radiometry_warning=None,
                metadata_files_detected=tuple(metadata_files),
            )
        return RadiometryMetadata(
            product_level="sentinel_l2a",
            radiometry_type="surface_reflectance",
            scale_factor=SENTINEL_REFLECTANCE_SCALE,
            offset=SENTINEL_REFLECTANCE_OFFSET,
            scale_applied=True,
            source_product_id=product_id,
            radiometry_source=radiometry_source,
            radiometry_warning=None,
            metadata_files_detected=tuple(metadata_files),
        )

    def _landsat_l2(
        self,
        *,
        product_id: str | None,
        radiometry_source: str,
    ) -> RadiometryMetadata:
        return RadiometryMetadata(
            product_level="landsat_l2",
            radiometry_type="surface_reflectance",
            scale_factor=LANDSAT_L2_SCALE,
            offset=LANDSAT_L2_OFFSET,
            scale_applied=True,
            source_product_id=product_id,
            radiometry_source=radiometry_source,
            radiometry_warning=None,
        )

    # --- detection helpers -------------------------------------------------

    def _detect_from_signals(
        self,
        *,
        sensor: str,
        product_id: str | None,
        band_keys: Sequence[str],
        scene_meta: Mapping[str, Any] | None,
    ) -> RadiometryMetadata:
        token = normalize_sensor_token(sensor) or sensor
        upper_pid = (product_id or "").upper()
        keys_upper = {str(k).upper() for k in band_keys}
        has_sr_bands = any(k.startswith("SR_B") for k in keys_upper)

        if (
            token == SENSOR_SYNTHETIC_SENTINEL_2
            or token == "synthetic"
            or "SYNTHETIC" in upper_pid
            or (
                scene_meta
                and str(scene_meta.get("product_level", "")).lower() == "synthetic"
            )
        ):
            return RadiometryMetadata(
                product_level="synthetic",
                radiometry_type="synthetic",
                scale_factor=None,
                offset=None,
                scale_applied=False,
                source_product_id=product_id,
                radiometry_source="synthetic",
                radiometry_warning=None,
            )

        if token == SENSOR_LANDSAT_8 or has_sr_bands or "LC08" in upper_pid:
            return self._detect_landsat(
                product_id=product_id,
                upper_pid=upper_pid,
                has_sr_bands=has_sr_bands,
                scene_meta=scene_meta,
            )

        if (
            token == SENSOR_SENTINEL_2
            or upper_pid.startswith("S2")
            or "MSIL1C" in upper_pid
            or "MSIL2A" in upper_pid
            or any(k.startswith("B0") or k in {"B11", "B12"} for k in keys_upper)
        ):
            return self._detect_sentinel(
                product_id=product_id,
                upper_pid=upper_pid,
                scene_meta=scene_meta,
            )

        return self._unknown(product_id=product_id)

    def _detect_landsat(
        self,
        *,
        product_id: str | None,
        upper_pid: str,
        has_sr_bands: bool,
        scene_meta: Mapping[str, Any] | None,
    ) -> RadiometryMetadata:
        processing = ""
        if scene_meta:
            processing = str(
                scene_meta.get("processing_level")
                or scene_meta.get("collection")
                or ""
            ).upper()

        is_l2 = (
            "L2SP" in upper_pid
            or "L2SR" in upper_pid
            or "L2" in processing
            or has_sr_bands
        )
        is_l1 = (
            "L1TP" in upper_pid
            or "L1GT" in upper_pid
            or "L1GS" in upper_pid
            or "L1" in processing
        ) and not is_l2

        source = "landsat_mtl" if scene_meta and (
            scene_meta.get("mtl_path") or scene_meta.get("product_id")
        ) else "unknown"
        if "L2SP" in upper_pid or "LC08_L2" in upper_pid:
            source = "landsat_mtl"

        if is_l2:
            return self._landsat_l2(
                product_id=product_id,
                radiometry_source=source if source != "unknown" else "landsat_mtl",
            )

        if is_l1:
            return RadiometryMetadata(
                product_level="landsat_l1",
                radiometry_type="dn",
                scale_factor=None,
                offset=None,
                scale_applied=False,
                source_product_id=product_id,
                radiometry_source=source,
                radiometry_warning=(
                    "Landsat Level-1 detected; TOA reflectance scaling not "
                    "applied automatically"
                ),
            )

        return self._unknown(product_id=product_id)

    def _detect_sentinel(
        self,
        *,
        product_id: str | None,
        upper_pid: str,
        scene_meta: Mapping[str, Any] | None,
    ) -> RadiometryMetadata:
        meta_level = ""
        if scene_meta:
            meta_level = str(
                scene_meta.get("product_level")
                or scene_meta.get("processing_level")
                or ""
            ).upper()

        is_l1c = "MSIL1C" in upper_pid or "L1C" in meta_level
        is_l2a = "MSIL2A" in upper_pid or "L2A" in meta_level

        if is_l1c and not is_l2a:
            return self._sentinel_level(
                "sentinel_l1c",
                product_id=product_id,
                radiometry_source="sentinel_product_name",
            )

        if is_l2a:
            return self._sentinel_level(
                "sentinel_l2a",
                product_id=product_id,
                radiometry_source="sentinel_product_name",
            )

        return self._unknown(product_id=product_id)

    @staticmethod
    def _unknown(
        *,
        product_id: str | None = None,
        metadata_files: Sequence[str] = (),
    ) -> RadiometryMetadata:
        return RadiometryMetadata(
            product_level="unknown",
            radiometry_type="unknown",
            scale_factor=None,
            offset=None,
            scale_applied=False,
            source_product_id=product_id,
            radiometry_source="unknown",
            radiometry_warning=UNKNOWN_RADIOMETRY_WARNING,
            metadata_files_detected=tuple(metadata_files),
        )

    @staticmethod
    def _normalize_metadata_files(
        metadata_files: Sequence[Any] | None,
    ) -> list[Path]:
        if not metadata_files:
            return []
        paths: list[Path] = []
        for item in metadata_files:
            if item is None:
                continue
            if isinstance(item, Path):
                path = item
            else:
                path = Path(str(item))
            if path.is_file():
                paths.append(path)
        return paths

    @staticmethod
    def _coerce_metadata(
        scene: Any,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if metadata is not None:
            return dict(metadata)
        if scene is None:
            return None
        meta = getattr(scene, "metadata_", None)
        if meta is None:
            meta = getattr(scene, "metadata", None)
        return dict(meta) if isinstance(meta, Mapping) else None

    def _from_stored_metadata(
        self,
        scene_meta: Mapping[str, Any] | None,
    ) -> RadiometryMetadata | None:
        if not scene_meta:
            return None
        nested = scene_meta.get("radiometry")
        if isinstance(nested, Mapping) and nested.get("product_level"):
            return self._coerce_radiometry(nested)
        if scene_meta.get("product_level") and scene_meta.get("radiometry_type"):
            files = scene_meta.get("metadata_files_detected") or ()
            if not isinstance(files, (list, tuple)):
                files = ()
            return RadiometryMetadata(
                product_level=str(scene_meta.get("product_level")),
                radiometry_type=str(scene_meta.get("radiometry_type")),
                scale_factor=_optional_float(scene_meta.get("scale_factor")),
                offset=_optional_float(scene_meta.get("offset")),
                scale_applied=bool(scene_meta.get("scale_applied", False)),
                source_product_id=_optional_str(scene_meta.get("source_product_id"))
                or _optional_str(scene_meta.get("product_id")),
                radiometry_source=str(
                    scene_meta.get("radiometry_source") or "unknown"
                ),
                radiometry_warning=_optional_str(
                    scene_meta.get("radiometry_warning")
                    or scene_meta.get("warning")
                ),
                metadata_files_detected=tuple(str(x) for x in files),
            )
        return None

    @staticmethod
    def _coerce_radiometry(
        radiometry: RadiometryMetadata | Mapping[str, Any],
    ) -> RadiometryMetadata:
        if isinstance(radiometry, RadiometryMetadata):
            return radiometry
        warning = radiometry.get("warning")
        if warning is None:
            warning = radiometry.get("radiometry_warning")
        files = radiometry.get("metadata_files_detected") or ()
        if not isinstance(files, (list, tuple)):
            files = ()
        return RadiometryMetadata(
            product_level=str(radiometry.get("product_level") or "unknown"),
            radiometry_type=str(radiometry.get("radiometry_type") or "unknown"),
            scale_factor=_optional_float(radiometry.get("scale_factor")),
            offset=_optional_float(radiometry.get("offset")),
            scale_applied=bool(radiometry.get("scale_applied", False)),
            source_product_id=_optional_str(radiometry.get("source_product_id")),
            radiometry_source=str(radiometry.get("radiometry_source") or "unknown"),
            radiometry_warning=_optional_str(warning),
            metadata_files_detected=tuple(str(x) for x in files),
        )


def _normalize_product_level(value: str | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text in {"auto", "detect", "automatic", "detect_automatically"}:
        return None
    aliases = {
        "sentinel_l1c": "sentinel_l1c",
        "l1c": "sentinel_l1c",
        "msil1c": "sentinel_l1c",
        "sentinel_l2a": "sentinel_l2a",
        "l2a": "sentinel_l2a",
        "msil2a": "sentinel_l2a",
        "landsat_l2": "landsat_l2",
        "l2": "landsat_l2",
        "l2sp": "landsat_l2",
        "unknown": "unknown",
    }
    return aliases.get(text)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "RadiometryService",
    "RadiometryMetadata",
    "LANDSAT_L2_SCALE",
    "LANDSAT_L2_OFFSET",
    "SENTINEL_REFLECTANCE_SCALE",
    "SENTINEL_REFLECTANCE_OFFSET",
    "UNKNOWN_RADIOMETRY_WARNING",
    "RADIOMETRY_TYPE_LABELS",
]
