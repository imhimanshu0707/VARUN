from datetime import datetime
from pathlib import Path

from app.forcing.readers import create_combined_readers
from app.hindcast.runner import HindcastRunner
from app.seeding.models import SeedingConfig
from app.simulation.engine import DriftSimulationEngine


def main() -> None:
    spill_polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [72.600, 19.100],
                [72.620, 19.100],
                [72.620, 19.120],
                [72.600, 19.120],
                [72.600, 19.100],
            ]
        ],
    }

    readers = create_combined_readers(
        combined_file=Path("data/forcing/mock_combined.nc"),
    )

    seeding_config = SeedingConfig(
        particle_count=20,
        seed_buffer_km=0.0,
        random_seed=42,
    )

    runner = HindcastRunner(
        engine=DriftSimulationEngine(),
    )

    result = runner.run_hindcast(
        run_id="SYNTHETIC-POLYGON-HINDCAST",
        observation_time=datetime(2026, 9, 2, 3, 0, 0),
        spill_polygon_geojson=spill_polygon,
        release_ages_hours=[1, 3],
        readers=readers,
        seeding_config=seeding_config,
    )

    successful = [
        candidate
        for candidate in result.release_ages
        if candidate.simulation_result is not None
    ]

    print()
    print("=" * 60)
    print("POLYGON MULTI-AGE HINDCAST RESULT")
    print("=" * 60)
    print("Candidates requested:", len(result.release_ages))
    print("Candidates successful:", len(successful))

    for candidate in result.release_ages:
        simulation = candidate.simulation_result

        print()
        print("Release age:", candidate.release_age_hours, "hours")
        print("Successful:", simulation is not None)

        if simulation is not None:
            print("Particles:", simulation.particle_count)
            print("Time steps:", simulation.time_steps)
            print("First time:", simulation.trajectory_times[0])
            print("Last time:", simulation.trajectory_times[-1])

            assert simulation.particle_count == 20
            assert simulation.time_steps >= 2
            assert (
                simulation.trajectory_times[0]
                > simulation.trajectory_times[-1]
            )

    assert len(result.release_ages) == 2
    assert len(successful) == 2

    print()
    print("RESULT: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()