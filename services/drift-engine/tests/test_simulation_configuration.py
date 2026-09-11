from app.simulation.engine import DriftSimulationEngine
from app.simulation.models import SimulationDirection


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