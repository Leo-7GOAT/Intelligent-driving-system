"""Audit every executed frame AND deterministic decision of a weight contrast."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from navstack.config import EnvConfig
from navstack.experiments import actor_vector
from navstack.independent_clip_ppo import load_ppo
from navstack.metrics import timing_summary
from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks, verify_initial_actuator
from scripts.audit_native_policy_sequence import observation_env, advance_observation
from scripts.evaluate_native_policy_weights import (ARMS, verify_inputs, initial_policy,
    load_normalizer, normalization_state, assert_normalization)
from scripts.evaluate_native_rule import require_finished_parent


def audit(batch):
    batch = Path(batch).resolve()
    require_finished_parent(batch)
    protocol,summary = read(batch/'protocol.json'),read(batch/'summary.json')
    verify_inputs(batch,protocol)
    if [r['arm'] for r in summary['arms']] != list(ARMS):
        raise ValueError('Missing, duplicated, or reordered weight arms')
    if any(r.get(k) != 0 for r in [protocol,summary,*summary['arms']]
           for k in ('training_physics_steps','optimizer_updates')):
        raise ValueError('Weight evaluation must not claim training or optimizer updates')
    if len(protocol['routes']) != 4:
        raise ValueError('Require all four registered training roads')
    config = EnvConfig.from_dict(protocol['config'])
    torch.set_num_threads(1)
    initial = initial_policy(config,read(batch/'parent_protocol.json'))
    saved_initial = load_ppo(batch/'initial_model.zip',device='cpu')
    for name,value in initial.policy.state_dict().items():
        if not torch.equal(value,saved_initial.policy.state_dict()[name]):
            raise ValueError('Initial weights differ from the registered seeded construction')
    trained = load_ppo(batch/'parent_model.zip',device='cpu')
    actor_change = float(torch.linalg.vector_norm(actor_vector(trained)-actor_vector(initial)))
    np.testing.assert_allclose(actor_change,read(batch/'parent_summary.json')['actor_l2_change'],rtol=1e-6,atol=1e-8)
    initialization = read(batch/'initialization.json')
    if (initialization['initial_model_sha256'] != sha(batch/'initial_model.zip') or
            initialization['actor_l2_change'] != actor_change or
            initialization['optimizer_updates'] != 0 or initialization['new_controls'] != 0):
        raise ValueError('Initialization evidence changed')

    actual,acks,owners,resets = {},{},{},{}
    bridge_hashes = {}
    paths = sorted((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths) != len(ARMS):
        raise ValueError('Require exactly one complete bridge stream per arm, no retry')
    for path in paths:
        rows,error = read_prefix(path)
        if error:raise ValueError('Truncated bridge response stream: '+error)
        bridge_hashes[path.name] = sha(path)
        for wrapper in rows:
            response = wrapper.get('response',{})
            episode,request = response.get('episode_id'),response.get('request_id')
            if episode is not None and request is not None:
                key = (episode,request)
                if key in acks and acks[key] != response:raise ValueError('Conflicting acknowledgement')
                acks[key] = response
            if episode is not None and 'reset_ticks' in response:
                if episode in resets and resets[episode] != response['reset_ticks']:
                    raise ValueError('Conflicting reset evidence')
                resets[episode] = response['reset_ticks']
            if 'vehicle_control' in response:
                key = (episode,response['frame'])
                if key in actual:raise ValueError('Duplicate applied frame')
                actual[key],owners[key] = response,path.name
    used,previous,all_episodes = set(),{},set()
    reports,starts,artifact_hashes = {},{},{}
    for arm,model,result in zip(ARMS,(saved_initial,trained),summary['arms']):
        folder = batch/arm
        if read(folder/'evaluation.json') != result:
            raise ValueError('Arm evaluation differs from whole summary')
        rows,error = read_prefix(folder/'evaluation_trace.jsonl.gz')
        if error:raise ValueError('Truncated evaluation stream: '+error)
        normalizer = load_normalizer(batch/'parent_normalization.pkl')
        norm_before = normalization_state(normalizer)
        cases,episodes = {},set()
        for row in rows:
            if row['policy_kind'] != arm:raise ValueError('Wrong weight arm label')
            case_id,info = row['case_id'],row['info']
            if type(case_id) is not int or not 0 <= case_id < 4:
                raise ValueError('Unexpected case identity')
            if case_id not in cases:
                path = folder/f'scene_{case_id:04d}.json'
                scene = read(path)
                if (scene['initial'] != info['initial_scenario'] or
                        scene['scene']['native_layout']['geometry_id'] != protocol['routes'][case_id]['geometry_id']):
                    raise ValueError('Scene differs from the registered case')
                artifact_hashes[f'{arm}/{path.name}'] = sha(path)
                cases[case_id] = dict(env=observation_env(config,scene['scene'],scene['initial']),
                    actions=0,controls=0,terminal=None,episodes=set(),inference=[],full=[],initial=scene['initial'])
            case = cases[case_id]
            if case['terminal'] is not None or case['initial'] != info['initial_scenario']:
                raise ValueError('Case changed or continued after termination')
            action = row['policy_action']
            if type(action) is not int or not 0 <= action < 31:
                raise ValueError('Invalid categorical action')
            predicted = int(model.predict(normalizer.normalize_obs(case['env']._observation()[None]),deterministic=True)[0][0])
            if predicted != action:raise ValueError('Recorded action differs from saved policy inference')
            for key in ('inference_seconds','full_decision_seconds'):
                if not math.isfinite(row[key]) or row[key] < 0:raise ValueError('Invalid compute measurement')
            if row['full_decision_seconds'] < row['inference_seconds']:raise ValueError('Invalid complete-cycle cost')
            count = _verify_ticks(info['trace'],actual,acks,owners,used,previous,episodes,
                                  protocol['config'].get('actuator_observation','off'))
            if count != info['executed_substeps']:raise ValueError('Substep accounting differs')
            ids = {t['backend_evidence']['episode_id'] for t in info['trace']}
            if protocol['config'].get('actuator_observation','off') != 'off':
                for episode in ids:verify_initial_actuator(case['initial'],episode,acks)
            case['episodes'].update(ids);case['actions'] += 1;case['controls'] += count
            case['inference'].append(row['inference_seconds']);case['full'].append(row['full_decision_seconds'])
            advance_observation(case['env'],row)
            if info['termination_reason']:case['terminal'] = info['episode_metrics']
        expected = {r['case_id']:r for r in result['episodes']}
        if set(cases) != set(range(4)) or set(expected) != set(cases) or len(result['episodes']) != 4:
            raise ValueError('Missing or duplicated case outcomes')
        for case_id,case in cases.items():
            metrics = {k:v for k,v in expected[case_id].items()
                       if k not in ('case_id','inference_times','full_decision_times')}
            if (case['terminal'] != metrics or case['actions'] != metrics['actions'] or
                    case['controls'] != metrics['physics_steps'] or len(case['episodes']) != 1):
                raise ValueError('Incomplete case or terminal accounting')
            for field,key in (('inference','inference_times'),('full','full_decision_times')):
                if timing_summary(case[field],config.decision_dt) != expected[case_id][key]:
                    raise ValueError('Timing summary differs from all measured decisions')
            case['env'].close()
        controls = sum(c['controls'] for c in cases.values())
        reset_ticks = sum(resets[e] for e in episodes)
        if (result['evaluation_controls'] != controls or result['reset_ticks'] != reset_ticks or
                result['world_ticks'] != controls+reset_ticks):raise ValueError('Arm world/control/reset accounting differs')
        assert_normalization(normalizer,norm_before)
        starts[arm] = {i:c['initial'] for i,c in cases.items()}
        reports[arm] = dict(evaluation_controls_verified=controls,reset_ticks_verified=reset_ticks,
            decisions_reproduced=sum(c['actions'] for c in cases.values()),
            termination_counts=dict(Counter(r['termination_reason'] for r in result['episodes'])))
        all_episodes.update(episodes)
        for name in ('evaluation.json','evaluation_trace.jsonl.gz'):
            artifact_hashes[f'{arm}/{name}'] = sha(folder/name)
    if not actual or used != set(actual) or set(resets) != all_episodes or len(all_episodes) != 8:
        raise ValueError('Unaccounted controls, extra reset, or reused episode')
    pairs = []
    for case_id in range(4):
        left,right = [starts[a][case_id] for a in ARMS]
        if {k:v for k,v in left.items() if k!='initial'} != {k:v for k,v in right.items() if k!='initial'}:
            raise ValueError('Paired task or initial actuator evidence differs')
        deltas = {k:right['initial'][k]-v for k,v in left['initial'].items()}
        deltas['yaw'] = math.atan2(math.sin(deltas['yaw']),math.cos(deltas['yaw']))
        pairs.append(dict(case_id=case_id,measured_start_deltas=deltas,
            metrics={r['arm']:next(e for e in r['episodes'] if e['case_id']==case_id) for r in summary['arms']}))
    return dict(scope=protocol['scope'],protocol_sha256=sha(batch/'protocol.json'),
        summary_sha256=sha(batch/'summary.json'),initial_model_sha256=sha(batch/'initial_model.zip'),
        trained_model_sha256=protocol['source_model_sha256'],normalization_sha256=protocol['normalization_sha256'],
        audit_script_sha256=sha(__file__),bridge_hashes=bridge_hashes,artifact_hashes=artifact_hashes,
        all_cases_and_failures_retained=True,arms=reports,paired_cases=pairs,
        total_applied_controls_verified=len(used),world_ticks_verified=len(used)+sum(resets.values()),
        new_controls=0,optimizer_updates=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--batch',required=True)
    parser.add_argument('--output',required=True);args = parser.parse_args()
    result = audit(args.batch)
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))
