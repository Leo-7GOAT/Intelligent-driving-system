"""Rebuild all four frozen H/H0 known-development inventories without selection."""
import argparse
from collections import Counter
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent


def summarize(execution):
    execution=Path(execution).resolve();done=read(execution/'complete.json');archive=execution.parent
    expected={(m,p) for m in ('H','H0') for p in ('placements30','known24')}
    if {(x['job']['method'],x['job']['phase']) for x in done['results']}!=expected or len(done['results'])!=4:
        raise ValueError('Require all four registered inventories,including every failed condition')
    results={};protocols={}
    for item in done['results']:
        job=item['job'];method,phase=job['method'],job['phase'];batch=Path(item['result']['batch']).resolve()
        if batch.parent!=archive or batch.name!=job['output']:raise ValueError('Registered archive path differs')
        require_finished_parent(batch)
        p=read(batch/'protocol.json');raw=read(batch/'H/endpoint_raw_audit.json');ev=read(batch/'H/evaluation.json')
        count=30 if phase=='placements30' else 24
        if (raw['protocol_sha256']!=sha(batch/'protocol.json') or raw['evaluation_sha256']!=sha(batch/'H/evaluation.json')
                or sha(batch/'H/endpoint_raw_audit.json')!=item['result']['raw_audit_sha256']
                or ev['model_sha256']!=sha(batch/'H/model.zip')
                or ev['model_sha256']!=p['fixed_outcome_development_validation']['frozen_model_sha256']
                or sorted(e['case_id'] for e in ev['episodes'])!=list(range(count))):
            raise ValueError('Raw/model/protocol or full condition inventory differs')
        engine=read(batch/'strict_delivery.json')
        if not engine['strict_delivery_gate_passed'] or engine['counts']['controls']!=raw['evaluation_control_frames_verified']:
            raise ValueError('Engine/frame accounting incomplete')
        for name in ('measured_start_inference.json','static_action_inference.json'):
            proof=read(batch/'H'/name)
            if proof['raw_audit_sha256']!=sha(batch/'H/endpoint_raw_audit.json') or not proof['all_recorded_actions_match']:
                raise ValueError('Missing bound action reconstruction')
        manifest=p['native_case_manifest'] if phase=='placements30' else p['native_validation']['manifest']
        families={};cases=[]
        for case,ep in zip(manifest['cases'],ev['episodes']):
            kind=(case.get('hazard') or {}).get('kind')
            family={None:'curve_parking','static_vehicle':'static_obstacle','crossing_walker':'dynamic_interaction'}[kind]
            stats=families.setdefault(family,dict(cases=0,successes=0,unsafe=0));stats['cases']+=1
            stats['successes']+=ep['success'];stats['unsafe']+=ep['termination_reason'] in ('collision','out_of_bounds')
            cases.append(dict(family=family,**{k:ep[k] for k in ('case_id','success','termination_reason','physics_steps',
                'interventions','plan_failures','mpc_failures','goal_parking_controls','simulation_seconds','progress',
                'control_cycle_times','decision_cycle_times')}))
        label=method+'_'+phase
        results[label]=dict(batch=batch.name,cases=count,successes=sum(e['success'] for e in ev['episodes']),
            termination_counts=dict(Counter(e['termination_reason'] for e in ev['episodes'])),families=families,
            actual_controls=raw['evaluation_control_frames_verified'],
            interventions=sum(e['interventions'] for e in ev['episodes']),
            plan_failures=sum(e['plan_failures'] for e in ev['episodes']),
            shared_parking_controls=sum(e['goal_parking_controls'] for e in ev['episodes']),
            model_sha256=ev['model_sha256'],raw_audit_sha256=sha(batch/'H/endpoint_raw_audit.json'),all_cases=cases)
        protocols[label]=p
    paired={}
    for phase in ('placements30','known24'):
        h,h0=protocols['H_'+phase],protocols['H0_'+phase]
        for key in ('source_hashes','routes','native_case_manifest','native_validation','calibration_sha256'):
            if h.get(key)!=h0.get(key):raise ValueError('Paired task/lowlevel differs:'+key)
        left,right=results['H_'+phase],results['H0_'+phase]
        paired[phase]=dict(success_count_difference=left['successes']-right['successes'],
            controls_difference=left['actual_controls']-right['actual_controls'],
            interventions_difference=left['interventions']-right['interventions'],
            planning_failures_difference=left['plan_failures']-right['plan_failures'])
    return dict(scope=__doc__,research_acceptance='UNMET',independent_training_seeds=1,
        registered_known_entries_each=54,new_final_conditions=0,new_PPO_updates=0,
        results=results,paired=paired,actual_evaluation_controls=sum(r['actual_controls'] for r in results.values()),
        execution_complete_sha256=sha(execution/'complete.json'),script_sha256=sha(Path(__file__)),
        limits='Known development inventories,one continued seed,not54new final conditions or independent-seed inference.Equal completion counts do not establish feedback success-rate gain.Same-weight action counterfactuals are separate from trained-H0driving.Wall timings use a shared workstation with concurrent checks and a disjoint external CARLA instance;not an isolated real-time benchmark.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve completed development reports')
    atomic_json(output,summarize(a.execution))
