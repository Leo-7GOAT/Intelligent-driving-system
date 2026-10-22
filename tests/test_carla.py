import os
import pytest
from navstack.carla_adapter import probe, CarlaPlant
from navstack.vehicle import Command


@pytest.mark.carla
def test_existing_carla_sync_plant():
    if os.environ.get("RUN_CARLA") != "1":
        pytest.skip(
            "Optional native-client adapter test; this project uses the separately tested cp37 bridge"
        )
    import importlib.util
    if importlib.util.find_spec("carla") is None:
        pytest.skip("Native client ABI differs; use RUN_CARLA_BRIDGE=1 for the real isolated bridge")
    available = probe()
    if available["status"] != "AVAILABLE":
        pytest.skip(str(available))
    with CarlaPlant() as plant:
        before = plant.state
        frame = plant.frame
        plant.step(Command(0.3, 0), 0.05)
        assert plant.state.time > before.time and plant.frame > frame
    # Adapter test only, not a trained PPO CARLA closed-loop test.
