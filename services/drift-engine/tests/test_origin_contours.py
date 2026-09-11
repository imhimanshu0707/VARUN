import numpy as np
import xarray as xr
from shapely.ops import unary_union

from app.origin.contours import generate_contours


def test_origin_contours_expand_with_density_level() -> None:
    latitudes = np.linspace(19.0, 19.5, 51)
    longitudes = np.linspace(72.5, 73.0, 51)

    y_grid, x_grid = np.meshgrid(
        latitudes,
        longitudes,
        indexing="ij",
    )

    density = np.exp(
        -(
            ((x_grid - 72.75) ** 2)
            + ((y_grid - 19.25) ** 2)
        )
        / 0.01
    )

    density = density / density.sum()

    dataset = xr.Dataset(
        {
            "density_normalized": (
                ("latitude", "longitude"),
                density,
            )
        },
        coords={
            "latitude": latitudes,
            "longitude": longitudes,
        },
    )

    contours = generate_contours(
        dataset,
        levels=[0.50, 0.75, 0.90],
        release_age_hours=24,
    )

    assert contours is not None
    assert set(contours["level"]) == {
        0.50,
        0.75,
        0.90,
    }

    contour_areas = {}

    for level in [0.50, 0.75, 0.90]:
        geometries = contours.loc[
            contours["level"] == level,
            "geometry",
        ].tolist()

        contour_areas[level] = unary_union(
            geometries
        ).area

    assert contour_areas[0.50] > 0
    assert contour_areas[0.50] < contour_areas[0.75]
    assert contour_areas[0.75] < contour_areas[0.90]

    assert set(contours["releaseAgeHours"]) == {24}
    assert contours.crs.to_string() == "EPSG:4326"