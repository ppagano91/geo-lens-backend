"""PNG preview generation from derived float32 index GeoTIFFs (Fase 7E)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.raster.readers import resolve_asset_path
from app.raster.writers import DEFAULT_INDEX_NODATA

# Spectral indices are defined on [-1, 1]; stretch that range to the LUT.
INDEX_VALUE_MIN = -1.0
INDEX_VALUE_MAX = 1.0

# Palette stops: (position in [0, 1], RGB). Interpolated to a 256-entry LUT.
PaletteStops = tuple[tuple[float, tuple[int, int, int]], ...]

# NDVI: brown → yellow → green
NDVI_PALETTE: PaletteStops = (
    (0.0, (140, 81, 10)),
    (0.35, (191, 129, 45)),
    (0.5, (246, 232, 195)),
    (0.65, (199, 233, 192)),
    (1.0, (1, 102, 94)),
)

# NDWI: white → celeste → blue
NDWI_PALETTE: PaletteStops = (
    (0.0, (255, 255, 255)),
    (0.35, (224, 243, 248)),
    (0.65, (127, 205, 187)),
    (1.0, (44, 127, 184)),
)

# NBR: red → yellow → green (divergent / burn severity style)
NBR_PALETTE: PaletteStops = (
    (0.0, (165, 0, 38)),
    (0.25, (215, 48, 39)),
    (0.5, (254, 224, 139)),
    (0.75, (145, 207, 96)),
    (1.0, (26, 152, 80)),
)

# NDMI: brown → white → blue
NDMI_PALETTE: PaletteStops = (
    (0.0, (140, 81, 10)),
    (0.35, (216, 179, 101)),
    (0.5, (245, 245, 245)),
    (0.65, (153, 184, 208)),
    (1.0, (5, 48, 97)),
)

INDEX_PALETTES: dict[str, PaletteStops] = {
    "ndvi": NDVI_PALETTE,
    "ndwi": NDWI_PALETTE,
    "nbr": NBR_PALETTE,
    "ndmi": NDMI_PALETTE,
}


class PreviewWriteError(Exception):
    """PNG preview could not be written to disk."""


def build_palette_lut(stops: PaletteStops) -> np.ndarray:
    """Build a (256, 3) uint8 RGB lookup table from color stops."""
    if len(stops) < 2:
        raise ValueError("Palette requires at least two color stops")

    positions = np.array([s[0] for s in stops], dtype=np.float64)
    colors = np.array([s[1] for s in stops], dtype=np.float64)
    if positions[0] != 0.0 or positions[-1] != 1.0:
        raise ValueError("Palette stops must start at 0.0 and end at 1.0")
    if not np.all(np.diff(positions) > 0):
        raise ValueError("Palette stop positions must be strictly increasing")

    xs = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        lut[:, channel] = np.clip(
            np.interp(xs, positions, colors[:, channel]),
            0,
            255,
        ).astype(np.uint8)
    return lut


def valid_mask(
    data: np.ndarray,
    *,
    nodata: float = DEFAULT_INDEX_NODATA,
) -> np.ndarray:
    """Boolean mask of pixels that are finite and not the index nodata sentinel."""
    array = np.asarray(data)
    mask = np.isfinite(array)
    if nodata is not None:
        mask &= array != np.float32(nodata)
        mask &= array != float(nodata)
    return mask


def normalize_to_uint8(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    vmin: float = INDEX_VALUE_MIN,
    vmax: float = INDEX_VALUE_MAX,
) -> np.ndarray:
    """Map valid pixels from [vmin, vmax] to 0–255; invalid pixels stay 0."""
    if vmax <= vmin:
        raise ValueError(f"Invalid stretch range: vmin={vmin}, vmax={vmax}")

    out = np.zeros(data.shape, dtype=np.uint8)
    if not np.any(mask):
        return out

    scaled = (np.asarray(data, dtype=np.float64) - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    out[mask] = (scaled[mask] * 255.0 + 0.5).astype(np.uint8)
    return out


def apply_palette_rgba(
    indices: np.ndarray,
    mask: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:
    """Apply a 256-entry RGB LUT; nodata/invalid pixels are fully transparent."""
    height, width = indices.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(mask):
        return rgba

    rgb = lut[indices[mask]]
    rgba[mask, :3] = rgb
    rgba[mask, 3] = 255
    return rgba


def render_index_preview_rgba(
    data: np.ndarray,
    index_key: str,
    *,
    nodata: float = DEFAULT_INDEX_NODATA,
) -> np.ndarray:
    """Render a float32 index array to RGBA uint8 using the index palette."""
    key = index_key.strip().lower()
    palette = INDEX_PALETTES.get(key)
    if palette is None:
        raise KeyError(f"No preview palette for index '{key}'")

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array for preview; got shape {array.shape}")

    mask = valid_mask(array, nodata=nodata)
    stretched = normalize_to_uint8(array, mask)
    lut = build_palette_lut(palette)
    return apply_palette_rgba(stretched, mask, lut)


def write_preview_png(
    asset_path: str,
    data_root: Path | str,
    rgba: np.ndarray,
) -> Path:
    """Write an RGBA PNG under DATA_ROOT (parents created; overwrite allowed)."""
    path = resolve_asset_path(asset_path, data_root)
    array = np.asarray(rgba)
    if array.ndim != 3 or array.shape[2] != 4:
        raise PreviewWriteError(
            f"Expected HxWx4 RGBA array for PNG write; got shape {array.shape}"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array.astype(np.uint8), mode="RGBA").save(path, format="PNG")
    except (OSError, ValueError, TypeError) as exc:
        raise PreviewWriteError(f"Cannot write PNG preview: {path}") from exc

    return path


__all__ = [
    "INDEX_PALETTES",
    "INDEX_VALUE_MAX",
    "INDEX_VALUE_MIN",
    "NDMI_PALETTE",
    "NBR_PALETTE",
    "NDVI_PALETTE",
    "NDWI_PALETTE",
    "PreviewWriteError",
    "apply_palette_rgba",
    "build_palette_lut",
    "normalize_to_uint8",
    "render_index_preview_rgba",
    "valid_mask",
    "write_preview_png",
]
