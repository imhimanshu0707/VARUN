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

    start_time = datetime(2026, 9, 2, 0, 0, 0)
    end_time = start_time + timedelta(hours=3)

    initial_lats = [19.24, 19.245, 19.25, 19.255, 19.26]
    initial_lons = [72.74, 72.745, 72.75, 72.755, 72.76]

    config = SimulationConfig(
        direction=SimulationDirection.FORWARD,
        start_time=start_time,
        end_time=end_time,
        initial_lats=initial_lats,
        initial_lons=initial_lons,
        time_step_hours=0.5,
        output_interval_hours=1.0,
    )

    engine = DriftSimulationEngine()

    result = engine.run_simulation(
        config=config,
        readers=readers,
        simulation_id="SYNTHETIC-OPENOIL-SMOKE",
    )

    final_lats, final_lons = result.get_final_positions()

    initial_lats_array = np.asarray(initial_lats)
    initial_lons_array = np.asarray(initial_lons)

    movement = np.sqrt(
        (final_lats - initial_lats_array) ** 2
        + (final_lons - initial_lons_array) ** 2
    )

    print()
    print("=" * 60)
    print("REAL SYNTHETIC OPENOIL SMOKE RESULT")
    print("=" * 60)
    print("Status:", result.status.value)
    print("Particles:", result.particle_count)
    print("Time steps:", result.time_steps)
    print("Start time:", result.start_time)
    print("End time:", result.end_time)
    print("Final latitudes:", final_lats)
    print("Final longitudes:", final_lons)
    print("Mean movement (degrees):", float(np.nanmean(movement)))

    assert result.particle_count == 5
    assert result.time_steps >= 2
    assert np.isfinite(final_lats).any()
    assert np.isfinite(final_lons).any()
    assert float(np.nanmean(movement)) > 0.0

    print("RESULT: PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()