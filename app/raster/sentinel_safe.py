"""Lightweight Sentinel-2 SAFE metadata sniffing (Fase 9M.1).

Looks for a small set of auxiliary files (``MTD_MSIL*.xml``, ``manifest.safe``,
``MTD_TL.xml``) without parsing the full SAFE tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

SAFE_METADATA_FILENAMES: tuple[str, ...] = (
    "MTD_MSIL1C.xml",
    "MTD_MSIL2A.xml",
    "manifest.safe",
    "MTD_TL.xml",
)

SAFE_METADATA_NAMES_UPPER = {name.upper() for name in SAFE_METADATA_FILENAMES}

# Max bytes to read when sniffing XML / manifest content.
_SAFE_SNIFF_BYTES = 256_000


@dataclass(frozen=True)
class SafeMetadataHit:
    """One SAFE auxiliary file that contributed to level detection."""

    path: Path
    relative_name: str
    product_level: Optional[str]  # sentinel_l1c | sentinel_l2a | None
    product_id: Optional[str] = None


def is_safe_metadata_filename(filename: str) -> bool:
    """True if basename matches a known SAFE metadata file."""
    name = Path(filename.replace("\\", "/")).name
    return name.upper() in SAFE_METADATA_NAMES_UPPER


def detect_level_from_text(text: str) -> Optional[str]:
    """Return ``sentinel_l1c`` / ``sentinel_l2a`` if markers appear in text."""
    upper = (text or "").upper()
    has_l1c = "MSIL1C" in upper or ">L1C<" in upper or "PRODUCT_TYPE>S2MSI1C" in upper
    has_l2a = "MSIL2A" in upper or ">L2A<" in upper or "PRODUCT_TYPE>S2MSI2A" in upper
    if has_l1c and not has_l2a:
        return "sentinel_l1c"
    if has_l2a and not has_l1c:
        return "sentinel_l2a"
    if has_l2a:
        # Prefer L2A when both appear (unusual).
        return "sentinel_l2a"
    if has_l1c:
        return "sentinel_l1c"
    return None


def detect_level_from_filename(filename: str) -> Optional[str]:
    upper = Path(filename.replace("\\", "/")).name.upper()
    if "MSIL1C" in upper:
        return "sentinel_l1c"
    if "MSIL2A" in upper:
        return "sentinel_l2a"
    return None


def extract_product_id_from_text(text: str) -> Optional[str]:
    """Best-effort product id from SAFE XML / path-like text."""
    import re

    # Typical ESA product id: S2A_MSIL1C_YYYYMMDDTHHMMSS_...
    match = re.search(
        r"(S2[AB]_MSIL(?:1C|2A)_[0-9T]+_N[0-9]+_R[0-9]+_T[0-9A-Z]+_[0-9T]+)",
        text or "",
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    # Shorter PRODUCT_URI / PRODUCT_ID tags.
    tag = re.search(
        r"<(?:PRODUCT_URI|PRODUCT_ID)>\s*([^<\s]+)\s*</(?:PRODUCT_URI|PRODUCT_ID)>",
        text or "",
        flags=re.IGNORECASE,
    )
    if tag:
        value = tag.group(1).strip()
        if "MSIL1C" in value.upper() or "MSIL2A" in value.upper():
            return value.rstrip("/")
    return None


def sniff_safe_file(path: Path) -> SafeMetadataHit:
    """Inspect one SAFE metadata file (name + limited content)."""
    relative_name = path.name
    level = detect_level_from_filename(relative_name)
    product_id: Optional[str] = None
    try:
        raw = path.read_bytes()[:_SAFE_SNIFF_BYTES]
        text = raw.decode("utf-8", errors="ignore")
    except OSError:
        text = ""
    if level is None and text:
        level = detect_level_from_text(text)
    if text:
        product_id = extract_product_id_from_text(text)
    if product_id is None:
        # Filename itself may be the product id folder context.
        product_id = None
    return SafeMetadataHit(
        path=path,
        relative_name=relative_name,
        product_level=level,
        product_id=product_id,
    )


def find_safe_metadata_files(
    scene_dir: Path,
    *,
    max_depth: int = 3,
    check_parent: bool = True,
) -> list[Path]:
    """Locate SAFE auxiliary files near ``scene_dir`` (bounded search)."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if resolved in seen or not resolved.is_file():
            return
        seen.add(resolved)
        found.append(resolved)

    if not scene_dir.exists() or not scene_dir.is_dir():
        return found

    # 1) Files directly in scene_dir.
    for name in SAFE_METADATA_FILENAMES:
        _add(scene_dir / name)
    for child in scene_dir.iterdir():
        if child.is_file() and is_safe_metadata_filename(child.name):
            _add(child)

    # 2) Bounded downward walk (SAFE-like nesting).
    _walk_down(scene_dir, depth=0, max_depth=max_depth, add=_add)

    # 3) If scene sits under GRANULE/…, peek at parents for MTD_MSIL*.xml.
    if check_parent:
        current = scene_dir
        for _ in range(3):
            parent = current.parent
            if parent == current:
                break
            for name in SAFE_METADATA_FILENAMES:
                _add(parent / name)
            # Also one level of siblings / top-level SAFE root.
            try:
                for child in parent.iterdir():
                    if child.is_file() and is_safe_metadata_filename(child.name):
                        _add(child)
            except OSError:
                pass
            current = parent

    return found


def _walk_down(
    root: Path,
    *,
    depth: int,
    max_depth: int,
    add,
) -> None:
    if depth >= max_depth:
        return
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        if child.is_file() and is_safe_metadata_filename(child.name):
            add(child)
        elif child.is_dir() and depth + 1 <= max_depth:
            # Skip huge unrelated trees by name heuristics when deep.
            name_upper = child.name.upper()
            if depth >= 1 and name_upper not in {
                "GRANULE",
                "DATASTRIP",
                "HTML",
                "REP_INFO",
                "QI_DATA",
                "IMG_DATA",
                "R10M",
                "R20M",
                "R60M",
            } and not name_upper.startswith(("L1C_", "L2A_", "S2A_", "S2B_", "T")):
                # Still allow one more level for typical tile folders.
                if depth >= 2:
                    continue
            _walk_down(child, depth=depth + 1, max_depth=max_depth, add=add)


def detect_level_from_path(path_like: str | Path) -> Optional[str]:
    """Inspect path segments for MSIL1C / MSIL2A."""
    text = str(path_like).replace("\\", "/")
    return detect_level_from_text(text)


def summarize_safe_hits(hits: Sequence[SafeMetadataHit]) -> tuple[
    Optional[str],
    Optional[str],
    list[str],
]:
    """Pick best level / product_id and list of detected filenames."""
    names = [hit.relative_name for hit in hits]
    level: Optional[str] = None
    product_id: Optional[str] = None
    for hit in hits:
        if hit.product_level and level is None:
            level = hit.product_level
        if hit.product_id and product_id is None:
            product_id = hit.product_id
        if level and product_id:
            break
    return level, product_id, names


__all__ = [
    "SAFE_METADATA_FILENAMES",
    "SafeMetadataHit",
    "is_safe_metadata_filename",
    "detect_level_from_text",
    "detect_level_from_filename",
    "extract_product_id_from_text",
    "sniff_safe_file",
    "find_safe_metadata_files",
    "detect_level_from_path",
    "summarize_safe_hits",
]
