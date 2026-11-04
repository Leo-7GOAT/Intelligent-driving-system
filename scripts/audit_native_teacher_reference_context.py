"""Compare held and freshly reset teacher context on actual basic teacher frames.

No plant step or integration exists. Only measured CARLA states are assigned.
Held commands must reproduce every recorded reference command before any
fresh-context discrepancy can be interpreted. No policy is fitted or driven.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import time


def analyze(teacher,runtime,output):
    teacher,runtime,output=map(lambda p:Path(p).resolve(),(teacher,runtime,output))
    if output.exists():raise FileExistsError('Preserve previous context audit')
    sys.path.insert(0,str(runtime))
    import numpy as np
    from navstack.config import EnvConfig
    from navstack.env import rule_action
    from navstack.imitation import command_action
    from navstack.mpc import MPC
    from navstack.vehicle import Command,VehicleState,brake
    from scripts.audit_native_policy_sequence import observation_env
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    require_finished_parent(teacher)
    protocol=read(teacher/'protocol.json');config=EnvConfig.from_dict(protocol['config'])
    for name,digest in protocol['source_hashes'].items():
        if sha(teacher/'source'/name)!=digest:raise ValueError('Frozen teacher source changed')
    critical=['env.py','planning.py','trajectory.py','mpc.py']
    for name in critical:
        if sha(runtime/'navstack'/name)!=sha(teacher/'source/navstack'/name):
            raise ValueError('Controller/reference implementation differs:'+name)
    rows=[]
    with gzip.open(teacher/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            if row['case_id'] in range(4):rows.append(row)
    if {r['case_id'] for r in rows}!=set(range(4)):raise ValueError('All four original basic teacher conditions required')
    started=time.monotonic();records=[];results=[]
    def build(scene,initial,state,previous):
        env=observation_env(config,scene,initial)
        env.world.road_collision_backend=config.road_collision_backend
        env.plant.state=VehicleState(**state);env.plant.previous_command=Command(**previous)
        env.reference=None;env.controller=MPC(config.vehicle,config.control_dt,config.mpc_horizon)
        env._rule_plan_cache=None;env.episode={'plan_times':[],'plan_failures':0}
        return env
    for case in range(4):
        selected=[r for r in rows if r['case_id']==case];initial=selected[0]['info']['initial_scenario']
        scene=read(teacher/('scene_%04d.json'%case))['scene']
        env=build(scene,initial,initial['initial'],{'acceleration':0.,'steering':0.})
        errors=[];differences=[];mode_counts={}
        try:
            for row in selected:
                if row['info']['mode']!='reference':break
                proposal=rule_action(env)
                if proposal!=row['rule_proposal']:raise ValueError('Held rule proposal differs from the recorded teacher')
                mode,_=env._prepare_reference(int(row['policy_action']))
                if mode!='reference':raise ValueError('Recorded reference is not reconstructed')
                for tick in row['info']['trace']:
                    if time.monotonic()-started>300:raise TimeoutError('Finite offline context audit expired')
                    actual=tick['requested_command']
                    result=env.controller.step(env.state,env.reference,env.plant.previous_command)
                    held={'acceleration':result.command.acceleration,'steering':result.command.steering}
                    discrepancy=max(abs(held[k]-actual[k]) for k in held)
                    if result.status!='solved' or discrepancy>1e-5:
                        raise ValueError('Held-context command does not reproduce actual teacher:case=%d frame=%d error=%g'%(case,tick['backend_frame'],discrepancy))
                    # Golden held-command verification remains exhaustive. The
                    # expensive fresh comparison uses outcome-independent phases
                    # 0/5 of every40controls,locked before this revised run.
                    control_index=len(errors);errors.append(discrepancy)
                    record={'case_id':case,'control_index':control_index,'frame_before':tick['backend_frame']-1,
                        'held_command_max_error':discrepancy,'fresh_compared':control_index%40 in (0,5)}
                    if record['fresh_compared']:
                        fresh=build(scene,initial,vars(env.state),vars(env.plant.previous_command))
                        try:
                            action=rule_action(fresh);fresh_mode,_=fresh._prepare_reference(action)
                            if fresh_mode=='reference':fresh_command=fresh.controller.step(fresh.state,fresh.reference,fresh.plant.previous_command).command
                            else:fresh_command=brake(fresh.state,config.vehicle)
                            delta=command_action(vars(fresh_command),config.vehicle)-command_action(actual,config.vehicle)
                        finally:fresh.close()
                        mode_counts[fresh_mode]=mode_counts.get(fresh_mode,0)+1;differences.append(delta)
                        record.update(fresh_mode=fresh_mode,fresh_minus_actual_normalized_action=delta.tolist())
                    records.append(record)
                    env.plant.state=VehicleState(**tick['state'])
                    env.plant.previous_command=Command(**tick['command']);env.plant.evidence=tick.get('backend_evidence',{})
        finally:env.close()
        delta=np.asarray(differences)
        results.append({'case_id':case,'reference_prefix_controls':len(errors),'fresh_comparison_queries':len(delta),
            'held_max_command_error':max(errors),'fresh_MSE':np.square(delta).mean(axis=0).tolist(),
            'fresh_mean_absolute_difference':np.abs(delta).mean(axis=0).tolist(),'fresh_modes':mode_counts})
        print(json.dumps(results[-1]),flush=True)
    report={'scope':__doc__,'teacher_batch':teacher.name,'runtime_source_manifest':{
        name:sha(runtime/'navstack'/name) for name in critical},
        'teacher_protocol_sha256':sha(teacher/'protocol.json'),'teacher_trace_sha256':sha(teacher/'evaluation_trace.jsonl.gz'),
        'script_sha256':sha(Path(__file__)),'results':results,'records':records,
        'fresh_comparison_selection':'Fixed control_index modulo40 in(0,5);all held reference-prefix commands verified',
        'wall_seconds':time.monotonic()-started,'new_controls':0,'optimizer_updates':0,
        'limits':'Recorded teacher states only. Fresh-context disagreement identifies a different labeling policy,not which policy would succeed in new driving.',
        'research_acceptance':'UNMET'}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('teacher','runtime','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args();analyze(args.teacher,args.runtime,args.output)
