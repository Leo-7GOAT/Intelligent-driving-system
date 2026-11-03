"""Assemble all12 old static H/H0 cases from authenticated, nonoverlapping segments."""
import argparse
from collections import Counter
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.independent_clip_ppo import load_ppo
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env
from scripts.evaluate_native_rule import require_finished_parent


def analyze(parent,remainder):
    parent,remainder=Path(parent).resolve(),Path(remainder).resolve()
    require_finished_parent(remainder)
    old,p=read(parent/'protocol.json'),read(remainder/'protocol.json')
    prefix_path=parent/'H0_PPO/completed_prefix_audit.json';prefix=read(prefix_path)
    ids=p['static_remainder_registration']['original_parent_case_ids']
    completed=prefix['completed_case_ids']
    if (set(ids)&set(completed) or set(ids)|set(completed)!=set(range(12))
            or ids!=prefix['remaining_case_ids']):raise ValueError('Missing or repeated original static case')
    if sha(parent/'calibration/calibration.json')!=sha(remainder/'calibration/calibration.json') or old['source_hashes']!=p['source_hashes']:
        raise ValueError('Recovered comparison changes calibration or frozen worker/physical source')
    before,after=registered_native_cases(old),registered_native_cases(p)
    for index,case in zip(ids,after):
        if {k:v for k,v in before[index].items() if k!='condition'}!={k:v for k,v in case.items() if k!='condition'}:
            raise ValueError('Recovered route or hazard differs')
    for name in ('model.zip','normalization.pkl'):
        if sha(parent/'H0_PPO'/name)!=sha(remainder/'H0_PPO'/name):raise ValueError('Frozen H0 learning bytes changed')
    if read(parent/'H0_PPO/config.json')['env']!=read(remainder/'H0_PPO/config.json')['env']:
        raise ValueError('Recovered observation or controller configuration differs')
    for name,key in (('protocol.json','protocol_sha256'),('H0_PPO/model.zip','model_sha256'),
                     ('H0_PPO/evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(parent/name)!=prefix[key]:raise ValueError('Original complete-prefix evidence changed')
    h_eval=read(parent/'H_PPO/evaluation.json');h_raw=read(parent/'H_PPO/endpoint_raw_audit.json')
    if (h_raw['protocol_sha256']!=sha(parent/'protocol.json') or h_raw['evaluation_sha256']!=sha(parent/'H_PPO/evaluation.json')
            or h_raw['evaluation_trace_sha256']!=sha(parent/'H_PPO/evaluation_trace.jsonl.gz')):
        raise ValueError('Original complete H evidence changed')
    r_eval=read(remainder/'H0_PPO/evaluation.json');r_raw=read(remainder/'H0_PPO/endpoint_raw_audit.json')
    if (r_raw['protocol_sha256']!=sha(remainder/'protocol.json') or r_raw['evaluation_sha256']!=sha(remainder/'H0_PPO/evaluation.json')
            or not read(remainder/'strict_delivery.json')['strict_delivery_gate_passed']):
        raise ValueError('Remainder lacks complete authenticated endpoint/control evidence')
    for folder,count in ((parent/'H_PPO',12),(remainder/'H0_PPO',8)):
        starts=read(folder/'measured_start_inference.json')
        if (starts['model_sha256']!=sha(folder/'model.zip') or starts['trace_sha256']!=sha(folder/'evaluation_trace.jsonl.gz')
                or len(starts['results'])!=count or any(r['action']!=r['recorded_action'] for r in starts['results'])):
            raise ValueError('Complete measured-start inference changed')
    # The original torn prefix has no evaluation.json. Authenticate its four
    # actual initial NN actions directly, without inventing a complete endpoint.
    rows,_=read_prefix(parent/'H0_PPO/evaluation_trace.jsonl.gz');begins={}
    for stream in prefix['bridge_streams']:
        path=parent/'infrastructure'/stream['name']
        if sha(path)!=stream['sha256']:raise ValueError('Retained bridge prefix changed')
        wrappers,_=read_prefix(path)
        begins.update({w['response']['episode_id']:w['response'] for w in wrappers if 'scene' in w['response']})
    model=load_ppo(parent/'H0_PPO/model.zip',device='cpu')
    with (parent/'H0_PPO/normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False;cfg=EnvConfig.from_dict(read(parent/'H0_PPO/config.json')['env']);initial_actions=[]
    for case in completed:
        first=next(r for r in rows if r['case_id']==case);episode=first['info']['trace'][0]['backend_evidence']['episode_id'];begin=begins[episode]
        env=observation_env(cfg,begin['scene'],first['info']['initial_scenario'],begin=begin)
        env.world.road_collision_backend=cfg.road_collision_backend
        try:
            action=int(model.predict(norm.normalize_obs(env._observation()[None]),deterministic=True)[0][0])
            if action!=first['policy_action']:raise ValueError('Original prefix NN action differs')
            initial_actions.append(dict(case_id=case,action=action,recorded_action=int(first['policy_action'])))
        finally:env.close()
    h={e['case_id']:e for e in h_eval['episodes']}
    h0={e['case_id']:e['metrics'] for e in prefix['completed_cases']}
    h0.update({original:dict(e,case_id=original) for original,e in zip(ids,r_eval['episodes'])})
    if set(h)!=set(range(12)) or set(h0)!=set(h):raise ValueError('Full12-case paired inventory required')
    results={}
    for method,episodes in (('H',h),('H0',h0)):
        results[method]=dict(cases=12,successes=sum(e['success'] for e in episodes.values()),
            termination_counts=dict(Counter(e['termination_reason'] for e in episodes.values())),
            completed_case_controls=sum(e['physics_steps'] for e in episodes.values()))
    h_rows,_=read_prefix(parent/'H_PPO/evaluation_trace.jsonl.gz')
    remainder_rows,_=read_prefix(remainder/'H0_PPO/evaluation_trace.jsonl.gz')
    common_bounds=[]
    for case in range(12):
        if h[case]['termination_reason']!='out_of_bounds' or h0[case]['termination_reason']!='out_of_bounds':continue
        h_case=[r for r in h_rows if r['case_id']==case]
        h0_case=([r for r in rows if r['case_id']==case] if case in completed else
                 [r for r in remainder_rows if r['case_id']==ids.index(case)])
        detail={};sequences=[];commands=[]
        for method,selected in (('H',h_case),('H0',h0_case)):
            ticks=[t for r in selected for t in r['info']['trace']];last=ticks[-1]
            sequences.append([t['state'] for t in ticks]);commands.append([t['command'] for t in ticks])
            detail[method]=dict(controls=len(ticks),policy_actions=[int(r['policy_action']) for r in selected],
                plan_statuses=[r['info']['plan']['status'] for r in selected],
                controller_status_counts=dict(Counter(t['controller_status'] for t in ticks)),
                final_tracking=last['tracking'],final_road_surface_probe=last['backend_evidence']['road_surface_probe'],
                recorded_metrics={k:selected[-1]['info']['episode_metrics'][k] for k in
                    ('plan_failures','mpc_failures','interventions','lateral_errors_rms','heading_errors_rms')})
        common_bounds.append(dict(case_id=case,detail=detail,
            actual_state_sequences_identical=sequences[0]==sequences[1],actual_command_sequences_identical=commands[0]==commands[1],
            interpretation='This retained failure occurs with stable high-level actions and converged MPC. Goal switching alone cannot explain all static failures;nominal planning/solver success does not prove actual footprint containment.',
            limits='Failure anatomy only;does not identify the exact geometry/controller cause or establish a repaired counterfactual.'))
    return dict(scope=__doc__,script_sha256=sha(Path(__file__)),research_acceptance='UNMET',
        results=results,paired_cases=[dict(case_id=i,H_success=h[i]['success'],H0_success=h0[i]['success'],
            H_termination=h[i]['termination_reason'],H0_termination=h0[i]['termination_reason']) for i in range(12)],
        prefix_start_inference=initial_actions,prefix_audit_sha256=sha(prefix_path),
        remainder_raw_audit_sha256=sha(remainder/'H0_PPO/endpoint_raw_audit.json'),
        H_model_sha256=sha(parent/'H_PPO/model.zip'),H0_model_sha256=sha(parent/'H0_PPO/model.zip'),
        H0_consumed_evaluation_controls_lower_bound=p['static_remainder_registration']['prefix_consumed_controls_lower_bound']+r_raw['evaluation_control_frames_verified'],
        H0_unknown_unflushed_original_tail=True,new_controls=0,new_updates=0,
        shared_bounds_anatomy=common_bounds,
        teacher_comparison='PENDING',reference_context_models='NOT_EVALUATED_HERE',
        limits='One canonical seed old4705-input policies on knownTRAIN12. Combined only disjoint authenticated completed cases;original failures/torn tails retained. No five-seed feedback or PPO benefit claim,no final confirmation.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',required=True)
    p.add_argument('--remainder',required=True);p.add_argument('--output',required=True);a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous recovered comparison')
    result=analyze(a.parent,a.remainder);atomic_json(output,result);print(result['results'])
