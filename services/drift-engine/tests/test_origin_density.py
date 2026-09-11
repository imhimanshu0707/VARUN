from datetime import datetime, timedelta

import numpy as np

from app.hindcast.models import HindcastReleaseAge, HindcastResult
from app.origin.density import calculate_origin_density
from app.simulation.models import (
    SimulationConfig,
    SimulationDirection,
    SimulationResult,
)


OBSERVATION_TIME = datetime(2026, 9, 2, 3, 0, 0)


def make_simulation_result(
    simulation_id: str,
    release_age_hours: int,
    final_lats: list[float],
    final_lons: list[float],
) -> SimulationResult:
    particle_count = len(final_lats)
    release_time = OBSERVATION_TIME - timedelta(
        hours=release_age_hours
    )

    initial_lats = np.full(
        particle_count,
        19.11,
        dtype=float,
    )
    initial_lons = np.full(
        particle_count,
        72.70,
        dtype=float,
    )

    return SimulationResult(
        simulation_id=simulation_id,
        config=SimulationConfig(
            direction=SimulationDirection.BACKWARD,
            start_time=OBSERVATION_TIME,
            end_time=release_time,
            initial_lats=initial_lats.tolist(),
            initial_lons=initial_lons.tolist(),
        ),
        start_time=OBSERVATION_TIME,
        end_time=release_time,
        trajectory_times=[
            OBSERVATION_TIME,
            release_time,
        ],
        trajectory_lats=np.vstack(
            [
                initial_lats,
                np.asarray(final_lats),
            ]
        ),
        trajectory_lons=np.vstack(
            [
                initial_lons,
                np.asarray(final_lons),
            ]
        ),
    )


def test_origin_density_can_be_calculated_per_release_age() -> None:
    hindcast = HindcastResult(
        run_id="TEST-RUN",
        observation_time=OBSERVATION_TIME,
        release_ages=[
            HindcastReleaseAge(
                release_age_hours=1,
                simulation_result=make_simulation_result(
                    "AGE-1",
                    1,
                    [19.10, 19.105, 19.11],
                    [72.60, 72.605, 72.61],
                ),
            ),
            HindcastReleaseAge(
                release_age_hours=3,
                simulation_result=make_simulation_result(
                    "AGE-3",
                    3,
                    [19.20, 19.205, 19.21],
                    [72.80, 72.805, 72.81],
                ),
            ),
        ],
    )

    density_age_1 = calculate_origin_density(
        hindcast,
        grid_resolution_km=0.5,
        sigma_km=0.5,
        release_age_hours=1,
    )

    density_age_3 = calculate_origin_density(
        hindcast,
        grid_resolution_km=0.5,
        sigma_km=0.5,
        release_age_hours=3,
    )

    density_all = calculate_origin_density(
        hindcast,
        grid_resolution_km=0.5,
        sigma_km=0.5,
    )

    assert density_age_1 is not None
    assert density_age_3 is not None
    assert density_all is not None

    assert density_age_1.attrs["release_age_hours"] == 1
    assert density_age_3.attrs["release_age_hours"] == 3
    assert density_all.attrs["release_age_hours"] == "all"

    assert (
        float(density_age_1.longitude.max())
        < float(density_age_3.longitude.min())
    )


def test_origin_density_returns_none_for_unknown_release_age() -> None:
    hindcast = HindcastResult(
        run_id="TEST-RUN",
        observation_time=OBSERVATION_TIME,
        release_ages=[],
    )

    density = calculate_origin_density(
        hindcast,
        release_age_hours=24,
    )

    assert density is None