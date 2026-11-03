"""Read-only reference reconstruction on recorded static native states."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env,advance_observation


def diagnose(batch,method,case_id,output,limit=16):
    batch,output=Path(batch).resolve(),Path(output)
    if output.exists():raise FileExistsError('Preserve the diagnostic attempt')
    folder=batch/method;protocol=read(batch/'protocol.json')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(ROOT/name)!=digest:
            raise ValueError('Use the frozen observation/planning implementation')
    rows,error=read_prefix(folder/'evaluation_trace.jsonl.gz')
    rows=[r for r in rows if r['case_id']==case_id]
    if not rows or 'episode_metrics' not in rows[-1]['info']:
        raise ValueError('Require a complete recorded case,even if its enclosing batch remains live')
    episode=rows[0]['info']['trace'][0]['backend_evidence']['episode_id']
    begins=[]
    for path in (batch/'infrastructure').glob('*_responses.jsonl.gz'):
        bridge,_=read_prefix(path)
        begins.extend(w['response'] for w in bridge if w.get('response',{}).get('episode_id')==episode and 'scene' in w['response'])
    if len(begins)!=1:raise ValueError('Require one measured reset acknowledgement')
    begin=begins[0];config=read(folder/'config.json')['env']
    if begin['dynamic'] or begin['scene'].get('native_hazard',{}).get('kind')!='static_vehicle':
        raise ValueError('This reconstruction is limited to recorded static cases')
    output.parent.mkdir(parents=True,exist_ok=True)
    snapshot=output.with_suffix('.inputs.json')
    if snapshot.exists():raise FileExistsError('Preserve immutable input snapshot')
    atomic_json(snapshot,dict(rows=rows,begin=begin,config=config))
    env=observation_env(EnvConfig.from_dict(config),begin['scene'],rows[0]['info']['initial_scenario'],begin=begin)
    env.world.road_collision_backend=env.config.road_collision_backend
    env.reference=None;env._reference_selection=None;env._rule_plan_cache=None
    env.episode=dict(plan_times=[],plan_failures=0)
    events=[]
    try:
        for index,row in enumerate(rows[:limit]):
            action=int(row['policy_action']);old=env._reference_selection
            selected=None if action==0 else env.decode_action(action)
            reusable=old is not None and env.reusable_reference(*old)
            mode,detail=env._prepare_reference(action)
            recorded=row['info']['plan']
            if detail is not None and (detail['status']!=recorded['status']
                    or detail.get('expanded_nodes')!=recorded.get('expanded_nodes')):
                raise ValueError('Reconstructed search differs from its recorded result')
            events.append(dict(decision=index,action=action,previous_selection=old,new_selection=selected,
                previous_reference_still_reusable=bool(reusable),switched_selection=old is not None and selected!=old,
                recorded_mode=row['info']['mode'],reconstructed_prepare_mode=mode,
                recorded_plan_status=None if recorded is None else recorded['status'],
                measured_time=env.state.time))
            advance_observation(env,row)
            if mode=='plan_failed':break
    finally:env.close()
    return dict(batch=batch.name,method=method,case_id=case_id,source_sha256=sha(Path(__file__)),
        immutable_input_snapshot_sha256=sha(snapshot),enclosing_live_trace_prefix_error=error,
        reconstruction_actions=len(events),events=events,
        switches_away_from_reusable_reference=sum(e['previous_reference_still_reusable'] and e['switched_selection'] for e in events),
        observed_terminal=rows[-1]['info']['episode_metrics']['termination_reason'],
        new_controls=0,new_updates=0,
        limits='Recorded static-state reconstruction only;no plant stepping or physical counterfactual. A reusable old reference is not proof that retaining it would complete the task. Not a full-batch score or a causal policy comparison.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--method',required=True)
    p.add_argument('--case',type=int,default=0);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=16)
    a=p.parse_args();r=diagnose(a.batch,a.method,a.case,a.output,a.limit);atomic_json(a.output,r)
    print({k:r[k] for k in ('reconstruction_actions','switches_away_from_reusable_reference','observed_terminal')})
