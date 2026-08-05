"""Unit tests for Landsat MTL.txt parsing."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from app.raster.mtl import find_mtl_file, parse_mtl_text


SAMPLE_MTL = """
GROUP = LANDSAT_METADATA_FILE
  GROUP = PRODUCT_CONTENTS
    ORIGIN = "Image courtesy of the U.S. Geological Survey"
    DIGITAL_OBJECT_IDENTIFIER = "https://doi.org/10.5066/P9OGBGM6"
    LANDSAT_PRODUCT_ID = "LC08_L2SP_225084_20260510_20260515_02_T1"
    PROCESSING_LEVEL = "L2SP"
    COLLECTION_NUMBER = 02
  END_GROUP = PRODUCT_CONTENTS
  GROUP = IMAGE_ATTRIBUTES
    SPACECRAFT_ID = "LANDSAT_8"
    SENSOR_ID = "OLI_TIRS"
    WRS_PATH = 225
    WRS_ROW = 84
    DATE_ACQUIRED = 2026-05-10
    CLOUD_COVER = 1.77
  END_GROUP = IMAGE_ATTRIBUTES
END_GROUP = LANDSAT_METADATA_FILE
END
"""


def test_parse_mtl_text_extracts_key_fields() -> None:
    meta = parse_mtl_text(SAMPLE_MTL)

    assert meta.spacecraft_id == "LANDSAT_8"
    assert meta.sensor_id == "OLI_TIRS"
    assert meta.date_acquired == date(2026, 5, 10)
    assert meta.cloud_cover == 1.77
    assert meta.wrs_path == 225
    assert meta.wrs_row == 84
    assert meta.landsat_product_id == "LC08_L2SP_225084_20260510_20260515_02_T1"
    assert meta.collection_number == "02"
    assert meta.processing_level == "L2SP"
    assert meta.as_dict()["product_id"] == meta.landsat_product_id


def test_find_mtl_file(tmp_path: Path) -> None:
    mtl = tmp_path / "LC08_L2SP_225084_20260510_20260515_02_T1_MTL.txt"
    mtl.write_text(SAMPLE_MTL, encoding="utf-8")

    found = find_mtl_file(tmp_path)

    assert found == mtl
