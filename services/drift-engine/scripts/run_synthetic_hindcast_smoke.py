from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from app.forcing.readers import create_combined_readers
from app.simulation.engine import DriftSimulationEngine
from app.simulation.models import SimulationConfig, SimulationDirection


def main() -> None:
    forcing_path = Path("data/forcing/mock_combined.nc")

    readers = create_combined_readers(
        combined_file=forcing_path,
    )

    observation_time = datetime(2026, 9, 2, 3, 0, 0)
    release_time = observation_time - timedelta(hours=3)

    observed_lats = [19.10, 19.105, 19.11, 19.115, 19.12]
    observed_lons = [72.60, 72.605, 72.61, 72.615, 72.62]

    config = SimulationConfig(
        direction=SimulationDirection.BACKWARD,
        start_time=observation_time,
        end_time=release_time,
        initial_lats=observed_lats,
        initial_lons=observed_lons,
        time_step_hours=0.5,
        output_interval_hours=1.0,
    )

    engine = DriftSimulationEngine()

    result = engine.run_simulation(
        config=config,
        readers=readers,
        simulation_id="SYNTHETIC-OPENOIL-HINDCAST-SMOKE",
    )

    origin_lats, origin_lons = result.get_final_positions()

    observed_lats_array = np.asarray(observed_lats)
    observed_lons_array = np.asarray(observed_lons)

    movement = np.sqrt(
        (origin_lats - observed_lats_array) ** 2
        + (origin_lons - observed_lons_array) ** 2
    )

    print()
    print("=" * 60)
    print("SYNTHETIC OPENOIL BACKWARD HINDCAST RESULT")
    print("=" * 60)
    print("Status:", result.status.value)
    print("Particles:", result.particle_count)
    print("Time steps:", result.time_steps)
    print("Observation time:", result.start_time)
    print("Estimated release time:", result.end_time)
    print("First trajectory time:", result.trajectory_times[0])
    print("Last trajectory time:", result.trajectory_times[-1])
    print("Estimated origin latitudes:", origin_lats)
    print("Estimated origin longitudes:", origin_lons)
    print("Mean backward movement (degrees):", float(np.nanmean(movement)))

    assert result.particle_count == 5
    assert result.time_steps >= 2
    assert result.trajectory_times[0] > result.trajectory_times[-1]
    assert np.isfinite(origin_lats).any()
    assert np.isfinite(origin_lons).any()
    assert float(np.nanmean(movement)) > 0.0

    print("RESULT: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()