import importlib.util
from pathlib import Path
import sys
import types
import pytest


@pytest.mark.parametrize('snapshot_present',[True,False])
def test_unknown_actor_guard_records_identity_but_never_ticks_destroys_or_ignores_it(monkeypatch,snapshot_present):
    monkeypatch.setitem(sys.modules,'carla',types.ModuleType('carla'))
    spec=importlib.util.spec_from_file_location('bridge_foreign_test',Path(__file__).parents[1]/'scripts/carla_bridge37.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Actor:
        id=77;type_id='vehicle.foreign';is_alive=True
        def destroy(self):raise AssertionError('Must never destroy an unknown actor')
    class World:
        def get_actors(self):return [Actor()]
        def get_snapshot(self):return types.SimpleNamespace(frame=9,find=lambda identity:Actor() if snapshot_present else None)
        def tick(self,*args):raise AssertionError('Must never tick with an unknown actor')
    bridge=module.Bridge.__new__(module.Bridge);bridge.cal={'recommended_adapter':{}}
    bridge.world=World();bridge.owned=[];bridge.retired_ids=set()
    with pytest.raises(RuntimeError,match='Foreign actors appeared'):
        bridge.begin({'episode_id':'fixture','scenario':'static'})
    assert bridge.foreign_actor_diagnostic['actors']==[{'id':77,'type_id':'vehicle.foreign','is_alive':True,'present_in_snapshot':snapshot_present}]
    assert bridge.owned==[] and bridge.retired_ids==set()


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


@pytest.mark.parametrize('presence',['absent','retiring','stale_descriptor','alive','rpc_failure','wrong_frame'])
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
            return None if presence=='absent' or presence=='retiring' and self.ticks else Actor()
        def tick(self,timeout):self.ticks+=1;return self.ticks
        def get_snapshot(self):
            if presence=='rpc_failure':raise RuntimeError('snapshot failed')
            return types.SimpleNamespace(frame=self.ticks+(presence=='wrong_frame'),
                find=lambda identity:Actor() if presence in ('alive','wrong_frame') else None)
    bridge=module.Bridge.__new__(module.Bridge);bridge.owned=[Actor()];bridge.retired_ids=set();bridge.world=World()
    if presence in ('absent','retiring','stale_descriptor'):
        bridge.clear()
        assert not bridge.owned and bridge.retired_ids=={57} and bridge.world.ticks==1
    else:
        with pytest.raises(RuntimeError):bridge.clear()
        assert len(bridge.owned)==1 and not bridge.retired_ids
        assert bridge.world.ticks==1
