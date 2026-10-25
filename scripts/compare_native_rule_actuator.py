"""Matched development-road actuator comparison from fully audited B3 runs."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_rule import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent,feedforward_calibration
from navstack.experiments import write_json


def selected_implementation(path):
    module=ast.parse(Path(path).read_text(encoding='utf-8'))
    functions={}
    for node in module.body:
        if isinstance(node,ast.FunctionDef) and node.name in ('rule_action','candidate_goal_witness'):
            functions[node.name]=ast.dump(node,include_attributes=False)
        if isinstance(node,ast.ClassDef) and node.name=='NavigationEnv':
            for member in node.body:
                if isinstance(member,ast.FunctionDef) and member.name in ('decode_action','_prepare_reference','_route_candidates'):
                    functions[member.name]=ast.dump(member,include_attributes=False)
    if len(functions)!=5:
        raise ValueError('Missing registered rule/catalog/reference implementation')
    return functions


def compare(baseline,candidate,output):
    baseline,candidate,output=Path(baseline),Path(candidate),Path(output)
    if output.exists():raise FileExistsError('Preserve comparison evidence')
    protocols=[]
    for batch in (baseline,candidate):
        require_finished_parent(batch)
        if audit(batch)!=read(batch/'raw_audit.json'):
            raise ValueError('Raw audit is absent or stale')
        protocols.append(read(batch/'protocol.json'))
    old,new=protocols
    old_config={**old['config']};new_config={**new['config']}
    if old_config.pop('actuator_observation','off')!='off' or new_config.pop('actuator_observation',None)!='ff_state_v1':
        raise ValueError('Require legacy baseline and explicit new actuator state')
    if old_config!=new_config or old['routes']!=new['routes']:
        raise ValueError('Task, vehicle or environment differs beyond the declared adapter input')
    if any(p.get('rule_speed_policy','native_score')!='native_score' for p in protocols):
        raise ValueError('Speed selector differs from the registered native-score comparison')
    feedforward_calibration(candidate/'calibration/calibration.json',read(baseline/'calibration/calibration.json'))
    for name in ('planning.py','trajectory.py','mpc.py','subgoals.py','vehicle.py'):
        if sha(baseline/'source/navstack'/name)!=sha(candidate/'source/navstack'/name):
            raise ValueError('Planner/trajectory/controller/vehicle implementation differs: '+name)
    if selected_implementation(baseline/'source/navstack/env.py')!=selected_implementation(candidate/'source/navstack/env.py'):
        raise ValueError('Rule selection or candidate/reference generation changed')
    summaries=[read(b/'summary.json') for b in (baseline,candidate)]
    rows=[]
    for case_id in range(len(old['routes'])):
        scenes=[read(b/f'scene_{case_id:04d}.json')['initial'] for b in (baseline,candidate)]
        stripped=[{k:v for k,v in s.items() if k!='actuator_evidence'} for s in scenes]
        if stripped[0]!=stripped[1]:raise ValueError('Measured starts/tasks differ')
        metrics={}
        for label,summary in zip(('legacy','feedforward'),summaries):
            episode=next(r for r in summary['episodes'] if r['case_id']==case_id)
            metrics[label]={k:episode[k] for k in ('success','termination_reason','physics_steps','progress',
                'speed_errors_rms','jerk_rms','lateral_errors_rms','heading_errors_rms','interventions',
                'plan_failures','mpc_failures','full_decision_times')}
        rows.append({'case_id':case_id,**metrics})
    report={'scope':'Four paired known training roads, single B3 execution per adapter; descriptive low-level engineering evidence, no PPO or population inference',
        'baseline':baseline.name,'candidate':candidate.name,
        'raw_audit_sha256':{b.name:sha(b/'raw_audit.json') for b in (baseline,candidate)},
        'summary_sha256':{b.name:sha(b/'summary.json') for b in (baseline,candidate)},
        'analysis_script_sha256':sha(Path(__file__)),'matched_measured_starts':True,
        'matched_rule_planner_trajectory_mpc':True,'new_controls':0,'optimizer_updates':0,'cases':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,report)
    print(json.dumps({'cases':len(rows),'successes':{label:sum(r[label]['success'] for r in rows) for label in ('legacy','feedforward')}}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();compare(a.baseline,a.candidate,a.output)
