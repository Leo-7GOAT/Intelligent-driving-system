"""Reconstruct policy inputs at measured native-road evaluation starts, offline."""
import argparse
from collections import deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT=Path(os.environ.get('NAVSTACK_AUDIT_SOURCE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehicleState,Command
from navstack.world import World


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(batch,method,output):
    batch=Path(batch);folder=batch/method;output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous inference evidence')
    protocol=read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:
            raise ValueError('Inference must use the recorded observation/policy implementation')
    evaluation=read(folder/'evaluation.json')
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]:
        if sha(folder/name)!=evaluation[key]:raise ValueError('Evaluation model/normalizer mismatch')
    config=EnvConfig.from_dict(read(folder/'config.json')['env'])
    if not config.road_observation:raise ValueError('Measured road input required')
    scenes={}
    for file in folder.glob('scene_*.json'):
        scene=read(file)['scene'];scenes[scene['road_grid']['sha256']]=scene
    torch.set_num_threads(1)
    model=PPO.load(folder/'model.zip',device='cpu')
    with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    loaded_normalization_training=bool(norm.training)
    norm.training=False
    before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    results=[];seen=set()
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['case_id'] in seen:continue
            seen.add(row['case_id'])
            initial=row['info']['initial_scenario'];scene=scenes[initial['road_sha256']]
            if initial['dynamic'] or initial['static']:
                raise ValueError('This diagnostic is limited to the recorded empty native roads')
            env=NavigationEnv(config)
            env.world=World(scene['width'],scene['height'],road=RoadGrid(scene['road_grid']))
            env.goal=initial['goal'];env.last_behavior=np.zeros(4);env.execution_outcome=np.zeros(6,dtype=np.float32)
            env.navigation_route=scene['native_layout']['path'] if config.subgoal_mode=='route_frame_terminal' else None
            # No plant integration or tick is permitted: only measured initial state.
            env.plant=SimpleNamespace(state=VehicleState(**initial['initial']),previous_command=Command(),delay_queue=deque(),
                evidence=initial.get('actuator_evidence',{}))
            raw=env._observation()[None];obs=norm.normalize_obs(raw)
            assert raw.shape[1:]==model.observation_space.shape
            if protocol.get('road_candidate_policy'):
                np.testing.assert_array_equal(raw[:,605:4701],obs[:,605:4701])
            action=model.predict(obs,deterministic=True)[0]
            np.testing.assert_array_equal(action[0],row['policy_action'])
            np.testing.assert_array_equal(action,model.predict(obs,deterministic=True)[0])
            with torch.no_grad():distribution=model.policy.get_distribution(torch.as_tensor(obs)).distribution
            result={'case_id':row['case_id'],'action':np.asarray(action[0]).tolist(),
                    'recorded_action':row['policy_action'],'road_sha256':initial['road_sha256']}
            if isinstance(distribution,torch.distributions.Categorical):
                probs=distribution.probs.numpy()[0]
                result.update(max_probability=float(probs.max()),probability_range=float(np.ptp(probs)),
                              entropy=float(distribution.entropy()[0]))
            elif isinstance(distribution,list) and all(isinstance(d,torch.distributions.Categorical) for d in distribution):
                result.update(component_probabilities=[d.probs.numpy()[0].tolist() for d in distribution],
                    component_entropies=[float(d.entropy()[0]) for d in distribution])
            else:
                result.update(mean=distribution.mean.numpy()[0].tolist(),std=distribution.stddev.numpy()[0].tolist())
            results.append(result)
    assert seen=={e['case_id'] for e in evaluation['episodes']}
    np.testing.assert_array_equal(before[0],norm.obs_rms.mean)
    np.testing.assert_array_equal(before[1],norm.obs_rms.var)
    assert before[2]==norm.obs_rms.count
    result={'scope':'Offline inference at measured evaluation starts; zero new CARLA ticks, not whole-trajectory re-execution',
        'method':method,'model_sha256':evaluation['model_sha256'],
        'trace_sha256':sha(folder/'evaluation_trace.jsonl.gz'),
        'loaded_normalization_training':loaded_normalization_training,
        'inference_normalization_frozen':not norm.training,
        'audit_script_sha256':sha(Path(__file__).resolve()),
        'frozen_source_root':str(ROOT),
        'batch_protocol_sha256':sha(batch/'protocol.json'),
        'new_controls':0,'optimizer_updates':0,'results':results}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True)
    p.add_argument('--method',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.batch,a.method,a.output)
