"""Unit tests for BandAlignmentService (Fase 9L)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.services.band_alignment_service import (
    BandAlignmentError,
    BandAlignmentService,
)


def _write_uint16(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str = "EPSG:4326",
    nodata: float | int = 0,
) -> None:
    height, width = data.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint16",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(data, 1)


def test_align_to_reference_matches_grid(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    src_rel = "sample/scenes/s2/B11.tif"
    dst_rel = "derived/scenes/abc/aligned/B11_10m.tif"

    src_transform = from_origin(-58.45, -34.55, 0.002, 0.002)
    ref_transform = from_origin(-58.45, -34.55, 0.001, 0.001)
    _write_uint16(
        data_root / src_rel,
        np.full((2, 2), 50, dtype=np.uint16),
        transform=src_transform,
        nodata=0,
    )

    service = BandAlignmentService(data_root)
    result = service.align_to_reference(
        source_asset_path=src_rel,
        destination_asset_path=dst_rel,
        reference_crs="EPSG:4326",
        reference_transform=list(ref_transform)[:6],
        reference_width=4,
        reference_height=4,
        original_band_key="B11",
        aligned_band_key="B11",
        reference_band="B08",
    )

    assert result.relative_asset_path == dst_rel
    assert result.width == 4
    assert result.height == 4
    assert result.crs == "EPSG:4326"
    assert result.resampling_method == "bilinear"
    assert result.dtype == "uint16"
    assert (data_root / dst_rel).is_file()

    with rasterio.open(data_root / dst_rel) as dataset:
        assert dataset.width == 4
        assert dataset.height == 4
        assert dataset.crs.to_string() == "EPSG:4326"
        assert list(dataset.transform)[:6] == list(ref_transform)[:6]
        assert dataset.nodata == 0


def test_align_missing_source_raises(tmp_path: Path) -> None:
    service = BandAlignmentService(tmp_path / "data")
    with pytest.raises(BandAlignmentError, match="not found"):
        service.align_to_reference(
            source_asset_path="missing/B11.tif",
            destination_asset_path="derived/scenes/x/aligned/B11_10m.tif",
            reference_crs="EPSG:4326",
            reference_transform=[0.001, 0, -58.45, 0, -0.001, -34.55],
            reference_width=4,
            reference_height=4,
            original_band_key="B11",
            aligned_band_key="B11",
            reference_band="B08",
        )
