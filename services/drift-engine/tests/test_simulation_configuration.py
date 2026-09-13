from datetime import datetime, timezone

import pytest

from app.forcing.models import ForcingConfig
from app.simulation.engine import DriftSimulationEngine
from app.simulation.models import SimulationDirection
from app.simulation.runner import Phase2CompleteRunner


IRREVERSIBLE_BACKWARD_SETTINGS = (
    "processes:evaporation",
    "processes:emulsification",
    "processes:dispersion",
    "processes:biodegradation",
    "processes:update_oilfilm_thickness",
    "drift:vertical_mixing",
)


def test_backward_openoil_disables_irreversible_processes() -> None:
    engine = DriftSimulationEngine()

    model = engine._configure_openoil(
        readers=[],
        direction=SimulationDirection.BACKWARD,
    )

    for key in IRREVERSIBLE_BACKWARD_SETTINGS:
        assert model.get_config(key) is False


def test_forward_openoil_keeps_weathering_enabled() -> None:
    engine = DriftSimulationEngine()

    model = engine._configure_openoil(
        readers=[],
        direction=SimulationDirection.FORWARD,
    )

    assert model.get_config("processes:evaporation") is True
    assert model.get_config("processes:emulsification") is True
    assert model.get_config("processes:dispersion") is True
    assert model.get_config("drift:vertical_mixing") is True

def test_runner_rejects_unsafe_external_run_id() -> None:
    runner = Phase2CompleteRunner()

    with pytest.raises(
        ValueError,
        match="run_id must contain",
    ):
        runner.run(
            case_id="CASE_TEST_001",
            scene_id="SCENE_TEST_001",
            observation_time=datetime.now(
                timezone.utc
            ),
            spill_polygon_geojson={
                "type": "Polygon",
                "coordinates": [],
            },
            forcing_config=ForcingConfig(),
            run_id="../unsafe-run",
            particle_count=1,
        )