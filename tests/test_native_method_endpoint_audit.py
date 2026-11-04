import gzip
import hashlib
import json

import pytest

from navstack.experiments import write_json
from scripts.audit_native_method_endpoint import audit


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def _response(episode, request, frame, x):
    return {
        "ok": True, "episode_id": episode, "request_id": request,
        "frame": frame, "imu_frame": frame,
        "state": {"x": x, "y": 0.0, "yaw": 0.0, "v": 1.0,
                  "steering": 0.0, "acceleration": 0.0, "time": 0.05},
        "command": {"acceleration": 0.0, "steering": 0.0},
        "collisions": [], "sensor_health": "frame_heartbeat_ok",
        "vehicle_control": {"throttle": 0.2, "brake": 0.0, "steer": 0.0},
        "speed_setpoint": 1.0, "road_surface_probe": None,
    }


def _tick(response):
    evidence = {key: response[key] for key in (
        "frame", "request_id", "episode_id", "imu_frame", "collisions",
        "sensor_health", "vehicle_control", "speed_setpoint", "road_surface_probe")}
    return {"state": response["state"], "command": response["command"],
            "backend_frame": response["frame"], "backend_evidence": evidence}


def _batch(tmp_path):
    batch = tmp_path / "batch"; method = batch / "B1"; checkpoint = method / "checkpoint_000000001"
    (batch / "infrastructure").mkdir(parents=True); checkpoint.mkdir(parents=True)
    write_json(batch / "protocol.json", {"methods": ["B1"], "recovery": {}})
    write_json(method / "config.json", {"env": {"method": "B1"}})
    train = _response("train", 2, 10, 1.0)
    evaluate = _response("evaluate", 4, 20, 2.0)
    responses = [
        {"response": {"ok": True, "episode_id": "train", "request_id": 1,
                      "reset_ticks": 2, "frame": 9}},
        {"response": train},
        {"response": {"ok": True, "episode_id": "evaluate", "request_id": 3,
                      "reset_ticks": 2, "frame": 19}},
        {"response": evaluate},
    ]
    _write_gzip(batch / "infrastructure/bridge_fixture_responses.jsonl.gz", responses)
    _write_gzip(method / "transitions_000.jsonl.gz", [{
        "episode_id": "train", "trace": [_tick(train)], "termination_reason": "deadline"}])
    terminal = {"physics_steps": 1, "actions": 1, "success": False,
                "termination_reason": "deadline"}
    _write_gzip(method / "evaluation_trace.jsonl.gz", [{
        "case_id": 0, "info": {"trace": [_tick(evaluate)],
                                  "termination_reason": "deadline",
                                  "episode_metrics": terminal}}])
    for name, content in (("model.zip", b"model"), ("normalization.pkl", b"norm"),
                          ("resume_state.pkl", b"rng")):
        (checkpoint / name).write_bytes(content)
    (method / "model.zip").write_bytes(b"model")
    (method / "normalization.pkl").write_bytes(b"norm")
    metadata = {"physics_steps": 1, "lineage_control_steps": 1,
                "charged_control_steps_lower_bound": 1, "optimizer_steps": 2,
                "model_sha256": _sha(checkpoint / "model.zip"),
                "normalization_sha256": _sha(checkpoint / "normalization.pkl"),
                "resume_state_sha256": _sha(checkpoint / "resume_state.pkl")}
    write_json(checkpoint / "metadata.json", metadata)
    write_json(method / "latest_checkpoint.json", {"folder": checkpoint.name})
    write_json(method / "summary.json", {
        "physics_steps": 1, "lineage_control_steps": 1,
        "charged_control_steps_lower_bound": 1,
        "inherited_controls_not_in_checkpoint": 0,
        "optimization_phases": 1, "optimizer_steps": 2, "world_ticks": 3})
    (method / "optimization.jsonl").write_text(json.dumps({
        "physics_steps": 1, "optimization_phases": 1, "optimizer_steps": 2}) + "\n",
        encoding="utf-8")
    write_json(method / "evaluation.json", {
        "episodes": [{"case_id": 0, **terminal}], "world_ticks": 3,
        "model_sha256": _sha(method / "model.zip"),
        "normalization_sha256": _sha(method / "normalization.pkl")})
    return batch


def test_method_audit_binds_every_control_to_bridge(tmp_path):
    result = audit(_batch(tmp_path), "B1")
    assert result["training_control_frames_verified"] == 1
    assert result["evaluation_control_frames_verified"] == 1
    assert result["total_applied_controls_verified"] == 2
    assert result["training_world_ticks_verified"] == 3


@pytest.mark.parametrize('fault',[None,'early','nonterminal','duplicate'])
def test_terminal_unused_auto_reset_is_charged_and_cannot_hide_other_resets(tmp_path,fault):
    batch=_batch(tmp_path);path=batch/'infrastructure/bridge_fixture_responses.jsonl.gz'
    with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
    idle={'response':{'ok':True,'episode_id':'idle','request_id':7,'reset_ticks':2,'frame':15}}
    rows.insert(0 if fault=='early' else 2,idle)
    if fault=='duplicate':rows.insert(3,{'response':{'ok':True,'episode_id':'idle2','request_id':8,'reset_ticks':2,'frame':16}})
    if fault=='nonterminal':
        trace=batch/'B1/transitions_000.jsonl.gz'
        with gzip.open(trace,'rt') as stream:action=json.loads(next(stream))
        action['termination_reason']=None;_write_gzip(trace,[action])
    _write_gzip(path,rows)
    summary=json.loads((batch/'B1/summary.json').read_text());summary['world_ticks']=5
    write_json(batch/'B1/summary.json',summary)
    if fault:
        with pytest.raises(ValueError,match='terminal training auto-reset'):audit(batch,'B1')
    else:
        result=audit(batch,'B1')
        assert result['training_world_ticks_verified']==5
        assert result['training_idle_reset_ticks_verified']==2
        assert result['training_idle_reset_episodes_verified']==['idle']


def test_method_audit_rejects_trace_state_not_measured_by_carla(tmp_path):
    batch = _batch(tmp_path)
    path = batch / "B1/transitions_000.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        row = json.loads(next(stream))
    row["trace"][0]["state"]["x"] = 99.0
    _write_gzip(path, [row])
    with pytest.raises(ValueError, match="measured CARLA state"):
        audit(batch, "B1")


def _evaluation_only(tmp_path):
    import shutil
    batch=_batch(tmp_path);folder=batch/'B1'
    origin=batch/'resume/B1';origin.parent.mkdir();shutil.copytree(folder/'checkpoint_000000001',origin)
    write_json(origin/'config.json',{'env':{'method':'B1'}})
    protocol={'methods':['B1'],'evaluation_only':True,'routes':[{}, {}, {}, {}],
        'recovery':{'B1':{'checkpoint':'resume/B1','lineage_controls':1,
            'consumed_controls_lower_bound':1,'retained_optimizer_steps':2}}}
    write_json(batch/'protocol.json',protocol)
    _write_gzip(folder/'transitions_000.jsonl.gz',[])
    responses=[];rows=[];episodes=[]
    for i in range(4):
        episode='evaluate'+str(i);response=_response(episode,2*i+2,20+i*10,float(i))
        responses.extend([{'response':{'ok':True,'episode_id':episode,'request_id':2*i+1,'reset_ticks':2,'frame':19+i*10}}, {'response':response}])
        terminal={'physics_steps':1,'actions':1,'success':False,'termination_reason':'deadline'}
        rows.append({'case_id':i,'info':{'trace':[_tick(response)],'termination_reason':'deadline','episode_metrics':terminal}})
        episodes.append({'case_id':i,**terminal})
    _write_gzip(batch/'infrastructure/bridge_fixture_responses.jsonl.gz',responses)
    _write_gzip(folder/'evaluation_trace.jsonl.gz',rows)
    data=json.loads((folder/'summary.json').read_text());data.update(physics_steps=0,actions=0,optimization_phases=0,optimizer_steps=0,world_ticks=0)
    write_json(folder/'summary.json',data);(folder/'optimization.jsonl').write_text('')
    cp=folder/'checkpoint_000000001';metadata=json.loads((cp/'metadata.json').read_text());metadata.update(physics_steps=0,optimizer_steps=0)
    write_json(cp/'metadata.json',metadata)
    evaluation=json.loads((folder/'evaluation.json').read_text());evaluation.update(episodes=episodes,world_ticks=12)
    write_json(folder/'evaluation.json',evaluation)
    return batch


def test_frozen_evaluation_only_audit_requires_zero_training_and_original_learning_bytes(tmp_path):
    batch=_evaluation_only(tmp_path);report=audit(batch,'B1')
    assert report['training_control_frames_verified']==0
    assert report['evaluation_control_frames_verified']==4
    assert report['optimizer_steps_total_lower_bound']==2 and report['segment_optimizer_steps']==0
    (batch/'B1/model.zip').write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='saved learning bytes'):audit(batch,'B1')


def test_frozen_evaluation_only_cannot_hide_training_or_drop_registered_cases(tmp_path):
    batch=_evaluation_only(tmp_path)
    (batch/'B1/optimization.jsonl').write_text('{"optimizer_steps":1}\n')
    with pytest.raises(ValueError,match='contains training'):audit(batch,'B1')
    (batch/'B1/optimization.jsonl').write_text('')
    e=json.loads((batch/'B1/evaluation.json').read_text());e['episodes'].pop()
    write_json(batch/'B1/evaluation.json',e)
    with pytest.raises(ValueError,match='all four'):audit(batch,'B1')
