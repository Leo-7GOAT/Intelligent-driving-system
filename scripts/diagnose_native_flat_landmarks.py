"""Compare the initial and trained B1 actions at audited measured decision states."""
import argparse
import gzip
import json
import math
from pathlib import Path
import pickle
import sys


def decision_landmarks(rows, tolerance):
    """Select observed pre-action states, never invent an action after termination."""
    goal = rows[0]['info']['initial_scenario']['goal']
    states = [rows[0]['info']['initial_scenario']['initial']]
    states.extend(row['info']['trace'][-1]['state'] for row in rows[:-1])
    distances = [math.hypot(s['x']-goal[0], s['y']-goal[1]) for s in states]
    selected = {'initial':0, 'closest_observed_decision':min(range(len(rows)), key=distances.__getitem__),
                'last_observed_decision':len(rows)-1}
    entries = [i for i, d in enumerate(distances) if d <= tolerance]
    if entries:
        selected['first_position_entry_decision'] = entries[0]
    return selected


def require_same_observation_normalization(norm, initial_norm):
    import numpy as np
    if type(norm) is not type(initial_norm):
        raise ValueError('Normalizer implementation differs')
    for name in ('mean','var','count'):
        np.testing.assert_array_equal(getattr(norm.obs_rms,name),getattr(initial_norm.obs_rms,name))
    for name in ('norm_obs','clip_obs','epsilon'):
        if norm.__dict__[name] != initial_norm.__dict__[name]:
            raise ValueError('Input normalization transform differs')


def run(batch, output):
    batch, output = Path(batch).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve prior landmark evidence')
    sys.path.insert(0, str(batch/'source'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.measured_demo import measured_flat_env, measured_label, advance_measured_flat
    from scripts.audit_native_method_endpoint import read, sha
    folder, origin = batch/'B1', batch/'resume/B1'
    audit, protocol = read(folder/'endpoint_raw_audit.json'), read(batch/'protocol.json')
    if sha(batch/'protocol.json') != audit['protocol_sha256']:
        raise ValueError('Protocol changed after raw audit')
    for name, digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(batch/'source'/name) != digest:
            raise ValueError('Frozen observation or policy implementation changed')
    for name, key in (('config.json','config_sha256'), ('evaluation_trace.jsonl.gz','evaluation_trace_sha256'),
                      ('evaluation.json','evaluation_sha256')):
        if sha(folder/name) != audit[key]:
            raise ValueError('Evaluation input changed after raw audit')
    evaluation = read(folder/'evaluation.json')
    metadata = read(origin/'metadata.json')
    for name, key in (('model.zip','model_sha256'), ('normalization.pkl','normalization_sha256')):
        if sha(folder/name) != evaluation[key]:
            raise ValueError('Evaluated learning artifact changed')
        if sha(origin/name) != metadata[key]:
            raise ValueError('Initial learning artifact differs from checkpoint metadata')
    cfg = EnvConfig.from_dict(read(folder/'config.json')['env'])
    if cfg.method != 'B1' or cfg.flat_hold != 1 or not protocol['freeze_observation_statistics']:
        raise ValueError('Require direct hold1 frozen-statistics B1')
    bridge = batch/'infrastructure'/audit['bridge_response_stream']
    if sha(bridge) != audit['bridge_response_sha256']:
        raise ValueError('Measured bridge stream changed')
    begins = {}
    with gzip.open(bridge,'rt',encoding='utf-8') as stream:
        for line in stream:
            response = json.loads(line)['response']
            if 'reset_ticks' in response:
                begins[response['episode_id']] = response
    cases = {}
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            cases.setdefault(row['case_id'], []).append(row)
    if set(cases) != {e['case_id'] for e in evaluation['episodes']}:
        raise ValueError('Incomplete measured condition inventory')
    torch.set_num_threads(1)
    models = {key:load_ppo(path/'model.zip',device='cpu') for key,path in
              (('initial',origin), ('trained',folder))}
    with (folder/'normalization.pkl').open('rb') as stream:
        norm = pickle.load(stream)
    with (origin/'normalization.pkl').open('rb') as stream:
        initial_norm = pickle.load(stream)
    require_same_observation_normalization(norm,initial_norm)
    norm.training = False
    results = []
    for case_id, rows in sorted(cases.items()):
        selected = decision_landmarks(rows,cfg.goal_tolerance)
        first = rows[0]
        episode = first['info']['trace'][0]['backend_evidence']['episode_id']
        begin = begins[episode]
        env = measured_flat_env(cfg,begin['scene'],first['info']['initial_scenario'],begin=begin)
        try:
            for index,row in enumerate(rows):
                labels = [name for name,i in selected.items() if i == index]
                if labels:
                    raw = env._observation()[None]
                    obs = norm.normalize_obs(raw)
                    actions = {key:model.predict(obs,deterministic=True)[0][0]
                               for key,model in models.items()}
                    error = float(np.max(np.abs(actions['trained']-np.asarray(row['policy_action']))))
                    if error > 1e-6:
                        raise ValueError('Reconstructed executed-policy action differs by more than1e-6')
                    s, goal = env.state, env.goal
                    results.append({'case_id':case_id,'landmarks':labels,'decision_index':index,
                        'simulation_seconds':s.time,'goal_distance_m':math.hypot(s.x-goal[0],s.y-goal[1]),
                        'absolute_speed_mps':abs(s.v),'yaw_error_rad':abs(math.atan2(math.sin(s.yaw-goal[2]),math.cos(s.yaw-goal[2]))),
                        'initial_policy_action':actions['initial'].tolist(),
                        'trained_policy_action':actions['trained'].tolist(),'executed_policy_action_error':error,
                        'trained_minus_initial_action':(actions['trained']-actions['initial']).tolist()})
                for tick in row['info']['trace']:
                    advance_measured_flat(env,tick,measured_label(env,tick))
        finally:
            env.close()
    result = {'scope':__doc__, 'batch':batch.name,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),
        'model_sha256':evaluation['model_sha256'],'initial_model_sha256':sha(origin/'model.zip'),
        'initial_metadata_sha256':sha(origin/'metadata.json'),
        'normalization_sha256':evaluation['normalization_sha256'],'script_sha256':sha(__file__),
        'frozen_observation_statistics_identical':True,
        'cases':len(cases),'reconstructed_decisions':len(results),'results':results,
        'new_controls':0,'optimizer_updates':0,
        'limits':'Same measured states from the trained policy,not executed initial-policy counterfactuals. Post-terminal states are not queried. Action differences do not establish a causal source of failure or driving improvement.',
        'research_acceptance':'UNMET'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(args.batch,args.output)
