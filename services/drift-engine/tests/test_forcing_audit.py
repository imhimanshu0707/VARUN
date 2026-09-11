from pathlib import Path

import numpy as np
import xarray as xr

from app.forcing.audit import audit_forcing_file


def test_audit_accepts_opendrift_canonical_forcing_names(
    tmp_path: Path,
) -> None:
    forcing_path = tmp_path / "synthetic_forcing.nc"

    shape = (2, 2, 2)

    dataset = xr.Dataset(
        data_vars={
            "x_sea_water_velocity": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.20, dtype=np.float32),
                {"units": "m s-1"},
            ),
            "y_sea_water_velocity": (
                ("time", "latitude", "longitude"),
                np.full(shape, 0.05, dtype=np.float32),
                {"units": "m s-1"},
            ),
            "x_wind": (
                ("time", "latitude", "longitude"),
                np.full(shape, 5.0, dtype=np.float32),
                {"units": "m s-1"},
            ),
            "y_wind": (
                ("time", "latitude", "longitude"),
                np.full(shape, 1.0, dtype=np.float32),
                {"units": "m s-1"},
            ),
        },
        coords={
            "time": np.array(
                ["2026-09-01T00:00:00", "2026-09-01T01:00:00"],
                dtype="datetime64[ns]",
            ),
            "latitude": np.array([19.0, 19.1]),
            "longitude": np.array([72.5, 72.6]),
        },
    )

    dataset.to_netcdf(forcing_path)

    result = audit_forcing_file(forcing_path)

    assert result.passed is True
    assert result.errors == []
    assert set(result.current_variables_found) == {
        "x_sea_water_velocity",
        "y_sea_water_velocity",
    }
    assert set(result.wind_variables_found) == {
        "x_wind",
        "y_wind",
    }
    assert len(result.variables) == 4