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


@pytest.mark.parametrize('presence',['absent','retiring','alive','rpc_failure'])
def test_false_destroy_requires_verified_world_absence(monkeypatch,presence):
    monkeypatch.setitem(sys.modules,'carla',types.ModuleType('carla'))
    spec=importlib.util.spec_from_file_location('bridge_false_destroy_test',Path(__file__).parents[1]/'scripts/carla_bridge37.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.time,'sleep',lambda seconds:None)
    class Actor:
        id=57;is_alive=True
        def destroy(self):return False
    class World:
        ticks=0
        def get_actor(self,identity):
            assert identity==57
            if presence=='rpc_failure':raise RuntimeError('lookup failed')
            return None if presence=='absent' or presence=='retiring' and self.ticks else Actor()
        def tick(self,timeout):self.ticks+=1
    bridge=module.Bridge.__new__(module.Bridge);bridge.owned=[Actor()];bridge.retired_ids=set();bridge.world=World()
    if presence in ('absent','retiring'):
        bridge.clear()
        assert not bridge.owned and bridge.retired_ids=={57} and bridge.world.ticks==1
    else:
        with pytest.raises(RuntimeError):bridge.clear()
        assert len(bridge.owned)==1 and not bridge.retired_ids
        assert bridge.world.ticks==(1 if presence=='alive' else 0)
