"""Per-channel visual stretch for RGB composites (Fase 9H).

Independent of index palettes: maps each float channel to uint8 via percentile
stretch and combines into RGBA (nodata → transparent).
"""

from __future__ import annotations

import numpy as np

from app.raster.preview import normalize_to_uint8


def channel_valid_mask(
    data: np.ndarray,
    *,
    nodata: float | None = None,
) -> np.ndarray:
    """Boolean mask of finite pixels; optionally exclude raster nodata."""
    array = np.asarray(data)
    mask = np.isfinite(array)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        mask &= array != np.float32(nodata)
        mask &= array != float(nodata)
    return mask


def percentile_limits(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    p_min: float = 2.0,
    p_max: float = 98.0,
) -> tuple[float, float]:
    """Compute stretch vmin/vmax from valid pixels; fallback to [0, 1] if empty."""
    if p_max <= p_min:
        raise ValueError(f"Invalid percentile range: p_min={p_min}, p_max={p_max}")

    if not np.any(mask):
        return 0.0, 1.0

    valid = np.asarray(data, dtype=np.float64)[mask]
    vmin = float(np.percentile(valid, p_min))
    vmax = float(np.percentile(valid, p_max))
    if vmax <= vmin:
        # Flat or near-flat channel: expand slightly so normalize stays defined.
        pad = 1.0 if vmax == 0.0 else abs(vmax) * 0.01
        return vmin, vmax + pad
    return vmin, vmax


def stretch_channel_to_uint8(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    p_min: float = 2.0,
    p_max: float = 98.0,
) -> np.ndarray:
    """Percentile-stretch one channel to uint8 (invalid pixels → 0)."""
    vmin, vmax = percentile_limits(data, mask, p_min=p_min, p_max=p_max)
    return normalize_to_uint8(data, mask, vmin=vmin, vmax=vmax)


def render_rgb_rgba(
    red: np.ndarray,
    green: np.ndarray,
    blue: np.ndarray,
    *,
    red_nodata: float | None = None,
    green_nodata: float | None = None,
    blue_nodata: float | None = None,
    p_min: float = 2.0,
    p_max: float = 98.0,
) -> np.ndarray:
    """Build HxWx4 uint8 RGBA from three aligned float channels."""
    r = np.asarray(red)
    g = np.asarray(green)
    b = np.asarray(blue)
    if r.ndim != 2 or g.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"Expected 2D channels; got shapes {r.shape}, {g.shape}, {b.shape}"
        )
    if r.shape != g.shape or r.shape != b.shape:
        raise ValueError(
            f"Channel shapes must match; got {r.shape}, {g.shape}, {b.shape}"
        )

    r_mask = channel_valid_mask(r, nodata=red_nodata)
    g_mask = channel_valid_mask(g, nodata=green_nodata)
    b_mask = channel_valid_mask(b, nodata=blue_nodata)
    combined = r_mask & g_mask & b_mask

    height, width = r.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    if not np.any(combined):
        return rgba

    rgba[:, :, 0] = stretch_channel_to_uint8(r, combined, p_min=p_min, p_max=p_max)
    rgba[:, :, 1] = stretch_channel_to_uint8(g, combined, p_min=p_min, p_max=p_max)
    rgba[:, :, 2] = stretch_channel_to_uint8(b, combined, p_min=p_min, p_max=p_max)
    rgba[combined, 3] = 255
    return rgba


__all__ = [
    "channel_valid_mask",
    "percentile_limits",
    "stretch_channel_to_uint8",
    "render_rgb_rgba",
]
