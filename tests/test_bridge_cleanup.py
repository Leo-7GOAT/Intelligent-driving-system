import importlib.util
from pathlib import Path
import sys
import types
import pytest


def test_cleanup_attempts_all_owned_actors_despite_sensor_stop_failure(monkeypatch):
    monkeypatch.setitem(sys.modules, "carla", types.ModuleType("carla"))
    spec = importlib.util.spec_from_file_location("bridge_cleanup_test", Path(__file__).parents[1] / "scripts/carla_bridge37.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destroyed = []
    class Actor:
        is_alive = True
        def __init__(self, identity): self.id = identity
        def stop(self):
            if self.id == 2: raise RuntimeError("sensor stopped unexpectedly")
        def destroy(self): destroyed.append(self.id); return True
    bridge = module.Bridge.__new__(module.Bridge)
    bridge.owned = [Actor(1), Actor(2), Actor(3)]
    bridge.retired_ids = set()
    with pytest.raises(RuntimeError, match="sensor stopped unexpectedly"):
        bridge.clear()
    assert destroyed == [3, 2, 1]
    assert bridge.owned == []
