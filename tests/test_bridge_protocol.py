import pytest
from navstack.bridge_protocol import EpisodeProtocol,ProtocolError
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.vehicle import VehicleState,Command
from navstack.world import World
from navstack.shield import risk_brake


def test_duplicate_tick_stale_episode_and_changed_payload():
    p=EpisodeProtocol();ticks=[]
    def handler(r):
        if r["op"]=="step":ticks.append(1)
        return {"frame":len(ticks)}
    p.execute({"op":"begin","episode_id":"a","request_id":1},handler)
    request={"op":"step","episode_id":"a","request_id":2}
    assert p.execute(request,handler)==p.execute(request,handler)
    assert len(ticks)==1
    with pytest.raises(ProtocolError):p.execute({**request,"acceleration":1},handler)
    p.execute({"op":"begin","episode_id":"b","request_id":3},handler)
    with pytest.raises(ProtocolError):p.execute({"op":"step","episode_id":"a","request_id":4},handler)
    assert len(ticks)==1


def test_failed_rpc_is_cached_not_reticked():
    p=EpisodeProtocol();calls=[]
    def fail(r):calls.append(1);raise RuntimeError("after tick connection lost")
    r={"op":"begin","episode_id":"a","request_id":1}
    assert not p.execute(r,fail)["ok"]
    assert not p.execute(r,fail)["ok"] and len(calls)==1


def test_shared_shield_brakes_flat_without_planner_or_mpc(monkeypatch):
    env=NavigationEnv(EnvConfig(method="B1",scenario="empty",shared_shield=True))
    env.reset(seed=1)
    env.world.static=[(8,10,9,14)]
    env.plant._state=VehicleState(4,12,v=2)
    def forbidden(*args,**kwargs):raise AssertionError("Flat called a planner/controller")
    monkeypatch.setattr(env.planner,"plan",forbidden);monkeypatch.setattr(env.controller,"step",forbidden)
    _,_,_,_,info=env.step([1,0])
    assert info["trace"][0]["shared_shield_intervention"]
    assert info["trace"][0]["command"]["acceleration"]<0
