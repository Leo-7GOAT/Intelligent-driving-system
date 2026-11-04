import os
import pytest
from navstack.carla_bridge import BridgeClient,CarlaNavigationEnv,InfrastructureFailure
from navstack.config import EnvConfig,VehicleConfig,PlannerConfig
from pathlib import Path
import json,gzip


@pytest.mark.carla
def test_bridge_owner_crash_without_actors_can_reconnect():
    if os.environ.get("RUN_CARLA_BRIDGE") != "1":
        pytest.skip("Requires an exclusively owned, empty CARLA server")
    batch = Path(os.environ["CARLA_BATCH"])
    client = BridgeClient(batch)
    try:
        # Invalid read acknowledges constructor completion without spawning actors
        # or advancing physics. Only this test-owned subprocess is terminated.
        with pytest.raises(InfrastructureFailure):
            client.request("read")
        assert client.process.poll() is None
        owner = client.process.pid
        client.process.kill()
        client.process.wait(timeout=10)
    finally:
        client.close()
    replacement = BridgeClient(batch)
    try:
        result = replacement.request("begin", scenario="empty", condition=7301)
        assert result["imu_frame"] >= result["frame"]
        assert result["reset_ticks"] == 30
        assert any(p.read_text().strip() == str(owner) for p in batch.glob("carla_tick_owner.lock.stale.*"))
    finally:
        replacement.close()
    assert (batch / "carla_tick_owner.lock").read_text() == ""


@pytest.mark.carla
def test_real_bridge_frames_replay_and_episode_isolation():
    if os.environ.get("RUN_CARLA_BRIDGE")!="1":
        pytest.skip("Opt-in live bridge test; run only when the batch CARLA worker is idle")
    batch=Path(os.environ["CARLA_BATCH"])
    cal=json.loads((batch/"calibration/calibration.json").read_text(encoding="utf-8"))
    client=BridgeClient(batch)
    try:
        env=CarlaNavigationEnv(EnvConfig(method="B1",vehicle=VehicleConfig(**cal["vehicle"]),planner=PlannerConfig(max_seconds=None)),client)
        env.reset(options={"case":{"scenario":"empty","condition":99}})
        initial=env.state
        for _ in range(60):env.step([1,0])
        assert env.state.x-initial.x>1 and env.state.time-initial.time==pytest.approx(3,abs=1e-5)
        old_request=dict(client.last_request);frame=env.plant.frame
        assert client.exchange(old_request)["frame"]==frame
        assert client.request("read")["frame"]==frame
        env.reset(options={"case":{"scenario":"empty","condition":98}})
        new_frame=env.plant.frame
        with pytest.raises(InfrastructureFailure):client.exchange(old_request)
        assert client.request("read")["frame"]==new_frame
        env.plant.confirm_terminal()
        assert not env.plant.collision_history
    finally:client.close()
    with gzip.open(client.response_audit_path,"rt",encoding="utf-8") as stream:
        responses=[json.loads(line)["response"] for line in stream]
    assert any(r.get("frame")==frame and r.get("ok") for r in responses)
    assert any(not r.get("ok") and "bridge_diagnostic" in r for r in responses)
    assert len([r for r in responses if r.get("ok") and "vehicle_control" in r]) >= 60


@pytest.mark.carla
def test_repeated_real_sensor_startup_keeps_reset_cost_and_frame_health():
    if os.environ.get("RUN_CARLA_BRIDGE") != "1":
        pytest.skip("Opt-in repeated live sensor initialization")
    client = BridgeClient(Path(os.environ["CARLA_BATCH"]))
    try:
        for i in range(30):
            response = client.request("begin", scenario=("static", "curve", "dynamic")[i % 3], condition=i // 3)
            assert response["imu_frame"] >= response["frame"]
            assert response["reset_sensor_warmup_ticks"] == 5
            assert response["reset_ticks"] == (30 if i == 0 else 31)
        assert client.total_ticks == 929
    finally:
        client.close()
