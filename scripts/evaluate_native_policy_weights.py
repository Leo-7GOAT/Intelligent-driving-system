"""Actual native-road initial/trained weight contrast with a common normalizer.

Prepare only after the complete parent lifecycle closes. Execute the resulting
frozen script with the existing owned-server launcher and process guard. This
is a known-road development contrast, not a fresh-normalizer untrained system
or independent confirmation experiment.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.carla_bridge import BridgeClient, CarlaNavigationEnv
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import actor_vector, jsonable, signature, write_json
from navstack.independent_clip_ppo import load_ppo
from navstack.metrics import timing_summary
from navstack.research import atomic_json, resources_ok
from navstack.road_candidate_policy import RoadCandidateActorCriticPolicy, RoadCandidateVecNormalize
from scripts.audit_native_method_endpoint import audit as endpoint_audit, read, sha
from scripts.carla_steering_audit import guard
from scripts.evaluate_native_rule import require_finished_parent

ARMS = ('initial_weights', 'trained_weights')
HARNESS = ('scripts/evaluate_native_policy_weights.py',
           'scripts/audit_native_policy_weights.py', 'scripts/watch_existing_pair.py')


def initial_policy(config, protocol):
    """Construct exactly the seeded fresh actor without resetting any plant."""
    if (protocol.get('recovery') or protocol.get('jobs') or
            not protocol.get('road_candidate_policy')):
        raise ValueError('Require fresh structured categorical training, no recovery/jobs')
    local = NavigationEnv(config)
    normalizer = RoadCandidateVecNormalize(DummyVecEnv([lambda: local]),
                                           norm_reward=False, gamma=config.gamma)
    try:
        model = PPO(RoadCandidateActorCriticPolicy, normalizer,
            n_steps=protocol['rollout_actions'], batch_size=32,
            n_epochs=protocol['ppo_epochs'], gamma=config.gamma, gae_lambda=.95**.5,
            policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}},
            seed=protocol['seed'], device='cpu')
        if local.reset_count or local.total_physics_steps or model._n_updates:
            raise ValueError('Initialization unexpectedly interacted or optimized')
        # Inference is deliberately detached from the unused local environment.
        model.env = None
        return model
    finally:
        normalizer.close()


def load_normalizer(path):
    with Path(path).open('rb') as stream:
        normalizer = pickle.load(stream)
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def normalization_state(normalizer):
    return (normalizer.obs_rms.mean.copy(), normalizer.obs_rms.var.copy(),
            float(normalizer.obs_rms.count), normalizer.ret_rms.mean.copy(),
            normalizer.ret_rms.var.copy(), float(normalizer.ret_rms.count))


def assert_normalization(normalizer, before):
    for actual, expected in zip(normalization_state(normalizer), before):
        np.testing.assert_array_equal(actual, expected)
    if normalizer.training:
        raise ValueError('Inference normalization was not frozen')


def prepare(parent, method, wall_seconds=7200):
    parent = Path(parent).resolve()
    require_finished_parent(parent)
    if method not in ('H0', 'H') or not 900 <= wall_seconds <= 10800:
        raise ValueError('Use H/H0 and a bounded 900..10800 second evaluation')
    protocol = read(parent/'protocol.json')
    if method not in protocol['methods'] or len(protocol['routes']) != 4:
        raise ValueError('Require all four original training roads')
    if protocol.get('recovery') or protocol.get('jobs') or not protocol.get('road_candidate_policy'):
        raise ValueError('No inherited or altered learner initialization')
    if endpoint_audit(parent, method) != read(parent/method/'endpoint_raw_audit.json'):
        raise ValueError('Parent endpoint raw audit is absent or stale')
    for name, digest in protocol['source_hashes'].items():
        if sha(parent/'source'/name) != digest:
            raise ValueError('Parent frozen source changed')
    if sha(parent/'calibration/calibration.json') != protocol['calibration_sha256']:
        raise ValueError('Parent calibration changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError('Resource guard: '+str(reason))
    folder = parent/method
    saved = read(folder/'config.json')
    config = EnvConfig.from_dict(saved['env'])
    if config.method != 'H' or config.planner.max_seconds is not None or saved['signature'] != signature(config):
        raise ValueError('Require unchanged categorical node-budget semantics')
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_weight_contrast')
    batch.mkdir()
    (batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source', batch/'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration', batch/'calibration')
    for name in HARNESS:
        (batch/'source'/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name, batch/'source'/name)
    for name in ('config.json', 'model.zip', 'normalization.pkl', 'summary.json', 'evaluation.json'):
        shutil.copy2(folder/name, batch/('parent_'+name))
    shutil.copy2(parent/'protocol.json', batch/'parent_protocol.json')
    now = time.time()
    prepared = dict(scope='Initial versus trained actor weights with the SAME trained frozen normalizer; all four KNOWN training roads, no held-out, fresh-normalizer system or independent-seed claim',
        parent_batch=parent.name, parent_root_relative=parent.relative_to(ROOT).as_posix(),
        parent_protocol_sha256=sha(parent/'protocol.json'), source_method=method,
        source_endpoint_raw_audit_sha256=sha(folder/'endpoint_raw_audit.json'),
        source_model_sha256=sha(folder/'model.zip'), normalization_sha256=sha(folder/'normalization.pkl'),
        config_sha256=sha(folder/'config.json'), calibration_sha256=protocol['calibration_sha256'],
        routes=protocol['routes'], config=config.to_dict(), arms=list(ARMS), worker_scopes=list(ARMS),
        created_epoch=now, deadline_epoch=now+wall_seconds, cleanup_owned_carla=True,
        carla_rpc_host=protocol['carla_rpc_host'], carla_rpc_port=protocol['carla_rpc_port'],
        carla_rpc_port_span=protocol['carla_rpc_port_span'],
        training_physics_steps=0, optimizer_updates=0,
        parent_input_hashes={p.name:sha(p) for p in batch.glob('parent_*') if p.is_file()},
        resource_limits=dict(wall_seconds=wall_seconds, disk_min_GiB=12, ram_min_GiB=4,
                             gpu_stop_temperature_C=85, automatic_retries=0),
        parent_source_hashes=protocol['source_hashes'], harness_overrides=list(HARNESS),
        source_hashes={p.relative_to(batch/'source').as_posix():sha(p)
            for p in (batch/'source').rglob('*') if p.is_file() and
            (p.suffix in ('.py','.ps1') or p.is_relative_to(batch/'source/tests/fixtures'))})
    write_json(batch/'protocol.json', prepared)
    write_json(batch/'prepared.json', dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(batch/'protocol.json')))
    print(batch, flush=True)
    return batch


def verify_inputs(batch, protocol):
    if sha(batch/'protocol.json') != read(batch/'prepared.json')['protocol_sha256']:
        raise ValueError('Prepared protocol changed')
    required = {'parent_'+name for name in ('protocol.json','config.json','model.zip',
                                            'normalization.pkl','summary.json','evaluation.json')}
    if set(protocol['parent_input_hashes']) != required or protocol['harness_overrides'] != list(HARNESS):
        raise ValueError('Parent input or harness inventory changed')
    for name, digest in protocol['parent_input_hashes'].items():
        if Path(name).name != name or sha(batch/name) != digest:
            raise ValueError('Frozen parent input changed: '+name)
    for name, digest in protocol['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Execute only the frozen source')
    for name, digest in protocol['parent_source_hashes'].items():
        if name not in HARNESS and protocol['source_hashes'].get(name) != digest:
            raise ValueError('A parent execution or observation source changed')
    for name, key in (('parent_protocol.json','parent_protocol_sha256'),
                      ('parent_model.zip','source_model_sha256'),
                      ('parent_normalization.pkl','normalization_sha256'),
                      ('parent_config.json','config_sha256'),
                      ('calibration/calibration.json','calibration_sha256')):
        if sha(batch/name) != protocol[key]:
            raise ValueError('Frozen contrast input changed: '+name)
    if read(batch/'parent_config.json')['env'] != protocol['config'] or protocol['arms'] != list(ARMS):
        raise ValueError('Contrast semantics or arm inventory changed')
    parent = read(batch/'parent_protocol.json')
    if parent['source_hashes'] != protocol['parent_source_hashes'] or parent['routes'] != protocol['routes']:
        raise ValueError('Parent source inventory or task catalog changed')


def evaluate_arm(env, model, normalizer, cases, output, health):
    before = normalization_state(normalizer)
    weights = {k:v.clone() for k,v in model.policy.state_dict().items()}
    updates = model._n_updates
    episodes, controls = [], 0
    with gzip.open(output/'evaluation_trace.jsonl.gz','wt',encoding='utf-8') as stream:
        for case_id, case in enumerate(cases):
            health()
            obs, _ = env.reset(seed=7301+case_id, options={'case':case})
            write_json(output/f'scene_{case_id:04d}.json',dict(scene=env.scene_evidence,initial=env.initial_scenario))
            decision_times, inference_times, actions = [], [], 0
            while True:
                health()
                started = time.perf_counter()
                action = int(model.predict(normalizer.normalize_obs(obs[None]),deterministic=True)[0][0])
                inferred = time.perf_counter()-started
                obs, _, done, truncated, info = env.step(action)
                elapsed = time.perf_counter()-started
                actions += 1
                controls += info['executed_substeps']
                decision_times.append(elapsed); inference_times.append(inferred)
                stream.write(json.dumps(dict(case_id=case_id,policy_kind=output.name,
                    policy_action=action,inference_seconds=inferred,full_decision_seconds=elapsed,
                    info=info),default=jsonable)+'\n')
                atomic_json(output/'heartbeat.json',dict(epoch=time.time(),pid=os.getpid(),
                    case_id=case_id,actions=actions,physics_steps=controls))
                if done or truncated:
                    episodes.append(dict(case_id=case_id,**info['episode_metrics'],
                        inference_times=timing_summary(inference_times,env.config.decision_dt),
                        full_decision_times=timing_summary(decision_times,env.config.decision_dt)))
                    break
    assert_normalization(normalizer,before)
    if model._n_updates != updates or any(not torch.equal(value,model.policy.state_dict()[name])
                                         for name,value in weights.items()):
        raise ValueError('Evaluation changed policy parameters or optimizer counter')
    return episodes, controls


def run(batch):
    batch = Path(batch).resolve()
    protocol = read(batch/'protocol.json')
    verify_inputs(batch,protocol)
    parent = (batch.parent.parent/protocol['parent_root_relative']).resolve()
    if not parent.is_relative_to(batch.parent):
        raise ValueError('Parent is outside the run archive')
    require_finished_parent(parent)
    if sha(parent/'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    guard(batch,protocol)
    with (batch/'started.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(epoch=time.time()),stream)
    results = []
    try:
        torch.set_num_threads(1)
        config = EnvConfig.from_dict(protocol['config'])
        trained = load_ppo(batch/'parent_model.zip',device='cpu')
        initial = initial_policy(config,read(batch/'parent_protocol.json'))
        actor_change = float(torch.linalg.vector_norm(actor_vector(trained)-actor_vector(initial)))
        np.testing.assert_allclose(actor_change,read(batch/'parent_summary.json')['actor_l2_change'],rtol=1e-6,atol=1e-8)
        cases = [dict(scenario='native_curve',condition=i,native_route=r) for i,r in enumerate(protocol['routes'])]
        initial.save(batch/'initial_model.zip')
        write_json(batch/'initialization.json',dict(actor_l2_change=actor_change,
            matches_parent_training_log=True,initial_model_sha256=sha(batch/'initial_model.zip'),
            optimizer_updates=0,new_controls=0,normalization='Parent trained statistics shared by both weight sets'))
        for arm, model in zip(ARMS,(initial,trained)):
            output = batch/arm; output.mkdir()
            client, env = None, None
            started = time.time()
            try:
                normalizer = load_normalizer(batch/'parent_normalization.pkl')
                client = BridgeClient(batch)
                env = CarlaNavigationEnv(config,client,cases)
                episodes, controls = evaluate_arm(env,model,normalizer,cases,output,lambda:guard(batch,protocol))
                result = dict(arm=arm,episodes=episodes,evaluation_controls=controls,
                    world_ticks=client.total_ticks,reset_ticks=client.total_ticks-controls,
                    normalization_unchanged=True,model_unchanged=True,training_physics_steps=0,
                    optimizer_updates=0,wall_seconds=time.time()-started)
                write_json(output/'evaluation.json',result);results.append(result)
            finally:
                if env is not None:env.close()
                if client is not None:client.close()
        write_json(batch/'summary.json',dict(status='complete',scope=protocol['scope'],arms=results,
            training_physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        write_json(batch/'failure.json',dict(epoch=time.time(),error=repr(exc),completed_arms=results))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare');group.add_argument('--batch')
    parser.add_argument('--method',choices=['H0','H'],default='H')
    parser.add_argument('--wall-seconds',type=int,default=7200)
    args = parser.parse_args()
    prepare(args.prepare,args.method,args.wall_seconds) if args.prepare else run(args.batch)
