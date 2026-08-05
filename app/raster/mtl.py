"""Minimal Landsat Collection 2 MTL.txt parser (no external deps)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class MtlMetadata:
    """Subset of Landsat MTL fields useful for local scene ingest."""

    raw: dict[str, str] = field(default_factory=dict)
    spacecraft_id: Optional[str] = None
    sensor_id: Optional[str] = None
    date_acquired: Optional[date] = None
    cloud_cover: Optional[float] = None
    wrs_path: Optional[int] = None
    wrs_row: Optional[int] = None
    landsat_product_id: Optional[str] = None
    collection_number: Optional[str] = None
    processing_level: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        """Compact dict for JSONB metadata (omit empty values)."""
        out: dict[str, Any] = {}
        if self.spacecraft_id:
            out["spacecraft_id"] = self.spacecraft_id
        if self.sensor_id:
            out["sensor_id"] = self.sensor_id
        if self.date_acquired:
            out["date_acquired"] = self.date_acquired.isoformat()
        if self.cloud_cover is not None:
            out["cloud_cover"] = self.cloud_cover
        if self.wrs_path is not None:
            out["wrs_path"] = self.wrs_path
        if self.wrs_row is not None:
            out["wrs_row"] = self.wrs_row
        if self.landsat_product_id:
            out["product_id"] = self.landsat_product_id
        if self.collection_number:
            out["collection"] = self.collection_number
        if self.processing_level:
            out["processing_level"] = self.processing_level
        return out


_KV_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.+?)\s*$")


def parse_mtl_text(text: str) -> MtlMetadata:
    """Parse Landsat MTL key=value lines into structured metadata."""
    raw: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("GROUP") or stripped.startswith("END"):
            continue
        match = _KV_RE.match(stripped)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            value = value[1:-1]
        raw[key] = value

    return MtlMetadata(
        raw=raw,
        spacecraft_id=raw.get("SPACECRAFT_ID"),
        sensor_id=raw.get("SENSOR_ID"),
        date_acquired=_parse_date(raw.get("DATE_ACQUIRED")),
        cloud_cover=_parse_float(raw.get("CLOUD_COVER")),
        wrs_path=_parse_int(raw.get("WRS_PATH")),
        wrs_row=_parse_int(raw.get("WRS_ROW")),
        landsat_product_id=raw.get("LANDSAT_PRODUCT_ID") or raw.get("LANDSAT_SCENE_ID"),
        collection_number=raw.get("COLLECTION_NUMBER"),
        processing_level=raw.get("PROCESSING_LEVEL") or raw.get("DATA_TYPE"),
    )


def parse_mtl_file(path: Path) -> MtlMetadata:
    """Read and parse an MTL.txt file from disk."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return parse_mtl_text(text)


def find_mtl_file(scene_dir: Path) -> Optional[Path]:
    """Return the first ``*MTL.txt`` under ``scene_dir`` (non-recursive then recursive)."""
    direct = sorted(scene_dir.glob("*MTL.txt")) + sorted(scene_dir.glob("*_MTL.txt"))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    candidates: list[Path] = []
    for path in direct:
        resolved = path.resolve()
        if resolved not in seen and path.is_file():
            seen.add(resolved)
            candidates.append(path)
    if candidates:
        return candidates[0]

    for path in sorted(scene_dir.rglob("*MTL.txt")):
        if path.is_file():
            return path
    return None


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Compact YYYYMMDD
    if re.fullmatch(r"\d{8}", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date()
        except ValueError:
            return None
    return None


def _parse_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


__all__ = [
    "MtlMetadata",
    "parse_mtl_text",
    "parse_mtl_file",
    "find_mtl_file",
]
