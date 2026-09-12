import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app.artifacts.raster import (
    MASK_NODATA,
    PROBABILITY_NODATA,
    write_mask_geotiff,
    write_probability_geotiff,
)


def test_write_probability_geotiff(tmp_path):
    probability = np.linspace(
        0.0,
        1.0,
        20,
        dtype=np.float32,
    ).reshape(4, 5)

    transform = from_origin(
        72.74,
        19.26,
        0.001,
        0.001,
    )

    output = tmp_path / "oil_probability.tif"

    write_probability_geotiff(
        probability_map=probability,
        output_path=output,
        crs="EPSG:4326",
        transform=transform,
    )

    with rasterio.open(output) as source:
        saved = source.read(1)

        assert source.count == 1
        assert source.dtypes[0] == "float32"
        assert source.crs.to_string() == "EPSG:4326"
        assert source.transform == transform
        assert source.nodata == PROBABILITY_NODATA
        assert source.descriptions[0] == "oil_probability"
        assert np.allclose(saved, probability)


def test_write_mask_geotiff(tmp_path):
    mask = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.uint8,
    )

    transform = from_origin(
        72.74,
        19.26,
        0.001,
        0.001,
    )

    output = tmp_path / "oil_mask.tif"

    write_mask_geotiff(
        mask=mask,
        output_path=output,
        crs="EPSG:4326",
        transform=transform,
    )

    with rasterio.open(output) as source:
        saved = source.read(1)

        assert source.dtypes[0] == "uint8"
        assert source.crs.to_string() == "EPSG:4326"
        assert source.transform == transform
        assert source.nodata == MASK_NODATA
        assert set(np.unique(saved)) == {0, 1}


def test_probability_writer_rejects_invalid_range(
    tmp_path,
):
    probability = np.array(
        [[0.2, 1.5]],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=r"within \[0, 1\]",
    ):
        write_probability_geotiff(
            probability_map=probability,
            output_path=tmp_path / "invalid.tif",
            crs="EPSG:4326",
            transform=from_origin(
                72.74,
                19.26,
                0.001,
                0.001,
            ),
        )