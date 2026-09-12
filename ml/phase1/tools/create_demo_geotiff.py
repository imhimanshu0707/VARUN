from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


WIDTH = 512
HEIGHT = 512

WEST = 72.74
SOUTH = 19.24
EAST = 72.76
NORTH = 19.26

OUTPUT_PATH = Path(
    "data/fixtures/images/synthetic_spill.tif"
)


def main() -> None:
    rng = np.random.default_rng(42)

    # Synthetic SAR-like backscatter in dB.
    vv = rng.normal(
        loc=-12.0,
        scale=2.0,
        size=(HEIGHT, WIDTH),
    ).astype(np.float32)

    vh = rng.normal(
        loc=-19.0,
        scale=2.5,
        size=(HEIGHT, WIDTH),
    ).astype(np.float32)

    rows, cols = np.ogrid[:HEIGHT, :WIDTH]

    # Irregular dark region: integration-test oil-slick pattern.
    main_slick = (
        ((cols - 270) / 105) ** 2
        + ((rows - 250) / 42) ** 2
    ) <= 1.0

    secondary_slick = (
        ((cols - 350) / 58) ** 2
        + ((rows - 285) / 24) ** 2
    ) <= 1.0

    connecting_band = (
        (cols >= 265)
        & (cols <= 355)
        & (rows >= 255)
        & (rows <= 285)
    )

    slick = (
        main_slick
        | secondary_slick
        | connecting_band
    )

    vv[slick] -= 7.0
    vh[slick] -= 5.5

    transform = from_bounds(
        WEST,
        SOUTH,
        EAST,
        NORTH,
        WIDTH,
        HEIGHT,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(
        OUTPUT_PATH,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=2,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(vv, 1)
        destination.write(vh, 2)

        destination.set_band_description(1, "VV_dB")
        destination.set_band_description(2, "VH_dB")

        destination.update_tags(
            scene_id="SCENE_DEMO_001",
            acquisition_time_utc="2026-09-02T12:00:00Z",
            data_origin="SYNTHETIC",
            purpose="PHASE1_INTEGRATION_TEST_ONLY",
        )

    print(f"Created: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()