"""Local execution and fabricated evidence checks; no CARLA result is claimed."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures('ample_training_disk')
import torch

from navstack.config import EnvConfig, PlannerConfig, VehicleConfig
from navstack.experiments import write_json
from navstack.metrics import timing_summary
from scripts.native_road_learning import training_config
from scripts.evaluate_native_policy_weights import (initial_policy, evaluate_arm,
    normalization_state, assert_normalization, prepare, ARMS)
from tests.test_native_rule_evaluation import LightweightFixture
import scripts.audit_native_policy_weights as auditing
import scripts.evaluate_native_policy_weights as staging


PROTOCOL = dict(road_candidate_policy=True, rollout_actions=32, ppo_epochs=2,
                seed=11, feedback_ablation=True, episode_seconds=45.)


def test_seeded_policy_has_no_environment_or_optimizer_steps():
    torch.set_num_threads(1)
    config = training_config(PROTOCOL,'H0',VehicleConfig())
    left,right = initial_policy(config,PROTOCOL),initial_policy(config,PROTOCOL)
    assert left.env is right.env is None
    assert left.num_timesteps == right.num_timesteps == left._n_updates == right._n_updates == 0
    assert all(torch.equal(v,right.policy.state_dict()[k]) for k,v in left.policy.state_dict().items())
    with pytest.raises(ValueError,match='recovery'):
        initial_policy(config,{**PROTOCOL,'recovery':{'H0':{}}})


def normalizer():
    moment = lambda:SimpleNamespace(mean=np.zeros(1),var=np.ones(1),count=1.)
    return SimpleNamespace(obs_rms=moment(),ret_rms=moment(),training=False,normalize_obs=lambda obs:obs)


def test_normalization_guard_covers_returns_count_and_training_mode():
    norm = normalizer();before = normalization_state(norm)
    norm.ret_rms.count += 1
    with pytest.raises(AssertionError):assert_normalization(norm,before)
    norm.ret_rms.count -= 1;norm.training=True
    with pytest.raises(ValueError,match='frozen'):assert_normalization(norm,before)


def test_preparation_refuses_live_parent_before_any_new_batch(tmp_path):
    with pytest.raises(ValueError,match='finish first'):prepare(tmp_path,'H')
    assert not list(tmp_path.iterdir())


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    monkeypatch.setattr(staging,'ROOT',tmp_path)
    parent=tmp_path/'runs/parent'
    for folder in ('source','H0','calibration','infrastructure'):(parent/folder).mkdir(parents=True)
    write_json(parent/'summary.json',dict(status='complete'))
    write_json(parent/'guard_final.json',dict(exit_code=0,reason=None))
    write_json(parent/'infrastructure/server_stop.json',dict(status='owned_empty_server_stopped'))
    write_json(parent/'calibration/calibration.json',{})
    source=parent/'source/fixture.py';source.write_text('pass\n')
    config=training_config(PROTOCOL,'H0',VehicleConfig())
    from navstack.experiments import signature
    write_json(parent/'H0/config.json',dict(env=config.to_dict(),signature=signature(config)))
    write_json(parent/'H0/endpoint_raw_audit.json',dict(fixture=True))
    for name in ('model.zip','normalization.pkl','summary.json','evaluation.json'):
        (parent/'H0'/name).write_bytes(b'fixture artifact; no model or CARLA claim')
    protocol={**PROTOCOL,'methods':['H0','H','B1'],'routes':[dict(geometry_id=str(i)) for i in range(4)],
        'source_hashes':{'fixture.py':staging.sha(source)},
        'calibration_sha256':staging.sha(parent/'calibration/calibration.json'),
        'carla_rpc_host':'127.0.0.1','carla_rpc_port':2100,'carla_rpc_port_span':[2100,2101,2102]}
    write_json(parent/'protocol.json',protocol)
    for name in staging.HARNESS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('pass\n')
    monkeypatch.setattr(staging,'resources_ok',lambda path:(True,None))
    monkeypatch.setattr(staging,'endpoint_audit',lambda *args:dict(fixture=True))
    batch=staging.prepare(parent,'H0')
    monkeypatch.setattr(staging,'ROOT',batch/'source')
    return batch


def test_preparation_binds_same_normalizer_both_arms_and_parent_source(prepared):
    protocol=json.loads((prepared/'protocol.json').read_text())
    staging.verify_inputs(prepared,protocol)
    assert protocol['arms']==list(ARMS)
    assert protocol['config']==json.loads((prepared/'parent_config.json').read_text())['env']
    assert protocol['normalization_sha256']==staging.sha(prepared/'parent_normalization.pkl')
    assert protocol['training_physics_steps']==protocol['optimizer_updates']==0
    assert not (prepared/'started.json').exists()


@pytest.mark.parametrize('changed,match',[('parent_model.zip','parent input'),
    ('parent_normalization.pkl','parent input'),('parent_summary.json','parent input'),
    ('source/fixture.py','frozen source'),('protocol.json','protocol changed')])
def test_frozen_input_mutation_is_rejected(prepared,changed,match):
    (prepared/changed).write_bytes(b'changed')
    protocol=json.loads((prepared/'protocol.json').read_text()) if changed!='protocol.json' else {}
    with pytest.raises(ValueError,match=match):staging.verify_inputs(prepared,protocol)


class WaitingPolicy:
    def __init__(self):
        self.policy = torch.nn.Linear(1,1);self._n_updates=0
    def predict(self,obs,deterministic):
        assert deterministic
        return np.array([0]),None


def test_actual_local_runner_retains_all_deadlines_and_physical_steps(tmp_path):
    env = LightweightFixture(EnvConfig(deadline=.5,planner=PlannerConfig(max_nodes=32,max_seconds=None)))
    output = tmp_path/'initial_weights';output.mkdir()
    try:
        results,controls = evaluate_arm(env,WaitingPolicy(),normalizer(),[{}, {}, {}, {}],output,lambda:None)
    finally:env.close()
    assert controls == 40 and len(results) == 4
    assert all(r['termination_reason']=='deadline' and not r['success'] for r in results)
    with gzip.open(output/'evaluation_trace.jsonl.gz','rt') as stream:rows=list(map(json.loads,stream))
    assert {r['case_id'] for r in rows} == set(range(4))
    assert sum(len(r['info']['trace']) for r in rows) == controls
    assert all(r['policy_kind']=='initial_weights' and r['policy_action']==0 for r in rows)


def fabricated_batch(batch,monkeypatch):
    """Synthetic one-control deadlines test raw evidence rejection only."""
    (batch/'infrastructure').mkdir();(batch/'calibration').mkdir()
    write_json(batch/'guard_final.json',dict(exit_code=0,reason=None))
    write_json(batch/'infrastructure/server_stop.json',dict(status='owned_empty_server_stopped'))
    write_json(batch/'parent_protocol.json',PROTOCOL)
    write_json(batch/'parent_summary.json',dict(actor_l2_change=0.))
    write_json(batch/'initialization.json',dict(initial_model_sha256='fixture',actor_l2_change=0.,optimizer_updates=0,new_controls=0))
    config = EnvConfig().to_dict()
    protocol = dict(config=config,routes=[dict(geometry_id=str(i)) for i in range(4)],
        training_physics_steps=0,optimizer_updates=0,scope='fabricated evidence only',
        source_model_sha256='fixture',normalization_sha256='fixture')
    write_json(batch/'protocol.json',protocol)
    model=WaitingPolicy()
    monkeypatch.setattr(auditing,'verify_inputs',lambda *args:None)
    monkeypatch.setattr(auditing,'initial_policy',lambda *args:model)
    monkeypatch.setattr(auditing,'load_ppo',lambda *args,**kw:model)
    monkeypatch.setattr(auditing,'actor_vector',lambda *args:torch.zeros(1))
    monkeypatch.setattr(auditing,'load_normalizer',lambda *args:normalizer())
    monkeypatch.setattr(auditing,'sha',lambda *args:'fixture')
    monkeypatch.setattr(auditing,'observation_env',lambda *args:SimpleNamespace(_observation=lambda:np.zeros(1),close=lambda:None))
    monkeypatch.setattr(auditing,'advance_observation',lambda *args:None)
    summary = dict(status='complete',arms=[],training_physics_steps=0,optimizer_updates=0)
    all_rows = {}
    for arm in ARMS:
        folder=batch/arm;folder.mkdir();rows=[];wrappers=[];results=[]
        for case in range(4):
            episode=arm+str(case)
            initial=dict(initial=dict(x=0.,yaw=0.),road_sha256='road')
            response=dict(episode_id=episode,request_id=2,frame=31,imu_frame=31,
                state=dict(x=0.,time=.05),command=dict(acceleration=0.,steering=0.),vehicle_control=dict(throttle=0.),collisions=[])
            wrappers.extend([dict(response=dict(episode_id=episode,request_id=1,reset_ticks=30)),dict(response=deepcopy(response))])
            terminal=dict(physics_steps=1,actions=1,success=False,termination_reason='deadline')
            tick=dict(state=deepcopy(response['state']),command=response['command'],backend_frame=31,
                backend_evidence={k:response[k] for k in ('episode_id','request_id','frame','imu_frame','collisions')})
            rows.append(dict(case_id=case,policy_kind=arm,policy_action=0,inference_seconds=.01,
                full_decision_seconds=.02,info=dict(initial_scenario=initial,trace=[tick],executed_substeps=1,
                    termination_reason='deadline',episode_metrics=terminal)))
            write_json(folder/f'scene_{case:04d}.json',dict(initial=initial,scene=dict(native_layout=dict(geometry_id=str(case)))))
            results.append(dict(case_id=case,**terminal,inference_times=timing_summary([.01],.5),
                full_decision_times=timing_summary([.02],.5)))
        def zipped(path,data):
            with gzip.open(path,'wt',encoding='utf-8') as stream:
                for row in data:stream.write(json.dumps(row)+'\n')
        zipped(folder/'evaluation_trace.jsonl.gz',rows)
        zipped(batch/f'infrastructure/bridge_{arm}_responses.jsonl.gz',wrappers)
        result=dict(arm=arm,episodes=results,evaluation_controls=4,reset_ticks=120,world_ticks=124,
            training_physics_steps=0,optimizer_updates=0)
        write_json(folder/'evaluation.json',result);summary['arms'].append(result);all_rows[arm]=rows
    write_json(batch/'summary.json',summary)
    return all_rows


def test_raw_audit_retains_all_eight_failed_cases_and_reproduces_actions(tmp_path,monkeypatch):
    fabricated_batch(tmp_path,monkeypatch)
    report=auditing.audit(tmp_path)
    assert report['total_applied_controls_verified']==8 and report['world_ticks_verified']==248
    assert len(report['paired_cases'])==4
    assert all(r['decisions_reproduced']==4 and r['termination_counts']=={'deadline':4} for r in report['arms'].values())


@pytest.mark.parametrize('corruption,match',[('state','measured CARLA state'),
    ('action','saved policy'),('missing','Missing'),('extra_reset','Unaccounted'),
    ('truncated','Truncated'),('training','must not claim')])
def test_raw_audit_rejects_incomplete_or_false_evidence(tmp_path,monkeypatch,corruption,match):
    rows=fabricated_batch(tmp_path,monkeypatch)['initial_weights']
    path=tmp_path/'initial_weights/evaluation_trace.jsonl.gz'
    if corruption=='state':rows[0]['info']['trace'][0]['state']['x']=99.
    elif corruption=='action':rows[0]['policy_action']=1
    elif corruption=='missing':rows.pop()
    elif corruption=='extra_reset':
        bridge=tmp_path/'infrastructure/bridge_initial_weights_responses.jsonl.gz'
        with gzip.open(bridge,'at') as stream:stream.write(json.dumps(dict(response=dict(episode_id='extra',request_id=1,reset_ticks=30)))+'\n')
    elif corruption=='training':
        summary=json.loads((tmp_path/'summary.json').read_text());summary['optimizer_updates']=1
        write_json(tmp_path/'summary.json',summary)
    if corruption in ('state','action','missing'):
        with gzip.open(path,'wt') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
    elif corruption=='truncated':path.write_bytes(path.read_bytes()[:-8])
    with pytest.raises(ValueError,match=match):auditing.audit(tmp_path)
