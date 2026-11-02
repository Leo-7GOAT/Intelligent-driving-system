"""Verify actual recorded reset conditions against the declared TRAIN schedule."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_native_method_endpoint import read,sha,read_prefix
from scripts.evaluate_native_rule import require_finished_parent


def verify_order(order,cursor,observed):
    if type(cursor) is not int or cursor<0 or not order:raise ValueError('Invalid saved reset cursor or schedule')
    expected=[order[(cursor+i)%len(order)] for i in range(len(observed))]
    if observed!=expected:raise ValueError('Actual native reset order differs from its registered schedule')
    return expected


def audit(batch,live=False):
    from navstack.native_cases import registered_native_cases
    batch=Path(batch).resolve();p=read(batch/'protocol.json')
    if not live:require_finished_parent(batch)
    if p.get('evaluation_only') or not p.get('training_case_schedule'):raise ValueError('Require scheduled TRAIN-only execution')
    relative='navstack/native_case_schedule.py';source=batch/'source'/relative
    if sha(source)!=p['source_hashes'][relative]:raise ValueError('Frozen schedule implementation changed')
    spec=importlib.util.spec_from_file_location('_executed_schedule',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cases=registered_native_cases(p);expanded=module.scheduled_native_cases(cases,p['training_case_schedule'])
    order=[c['condition'] for c in expanded]
    with (batch/'resume/B1/resume_state.pkl').open('rb') as stream:cursor=pickle.load(stream)['next_case_index']
    scenes=[read(path) for path in sorted((batch/'B1').glob('scene_*.json'))]
    if not scenes:raise ValueError('No actual native reset recorded')
    observed=[s['scene']['condition'] for s in scenes];verify_order(order,cursor,observed)
    controls=Counter();begins={}
    for path in (batch/'infrastructure').glob('*_responses.jsonl.gz'):
        rows,error=read_prefix(path)
        if error and not live:raise ValueError('Incomplete bridge history')
        for wrapper in rows:
            r=wrapper.get('response',{})
            if 'scene' in r:begins[r['frame']]=r
            if 'vehicle_control' in r:controls[r['episode_id']]+=1
    counts=Counter();actual_controls=0
    for scene in scenes:
        begin=begins.get(scene['frame'])
        if begin is None:
            if live:continue
            raise ValueError('Scene lacks measured reset acknowledgement')
        if begin['scene']['condition']!=scene['scene']['condition']:raise ValueError('Measured reset condition differs')
        n=controls[begin['episode_id']]
        if n:counts[scene['scene']['condition']]+=1;actual_controls+=n
    if not live and actual_controls!=read(batch/'B1/summary.json')['physics_steps']:raise ValueError('Scheduled controls differ from actual budget')
    return dict(batch=batch.name,script_sha256=sha(Path(__file__)),protocol_sha256=sha(batch/'protocol.json'),
        scope='LIVE_PREFIX_DIAGNOSTIC' if live else 'COMPLETE_TRAIN_RESET_AND_EXPOSURE_AUDIT',saved_next_case_index=cursor,
        recorded_resets=len(scenes),all_actual_reset_conditions_match=True,unique_inventory_conditions=len(cases),
        schedule_exposures=len(order),actual_control_frames_in_readable_stream=actual_controls,
        actual_controlled_episode_counts=dict(counts),original_basic_episode_counts={str(i):counts[i] for i in module.BASIC},
        repeated_exposures_not_independent_conditions=True,new_controls=0,new_updates=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--batch',required=True);parser.add_argument('--output',required=True);parser.add_argument('--live-prefix',action='store_true')
    args=parser.parse_args();result=audit(args.batch,args.live_prefix)
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)
