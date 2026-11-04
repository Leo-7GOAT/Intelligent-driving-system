"""Register every model-supported proposal, reusing a closed pilot regardless of outcome."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.native_cases import registered_native_cases
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_reference_evaluation import prepare as prepare_endpoint


def eligible_proposals(registration,complete):
    expected={(case,index) for case in registration['cases'] for index in range(len(registration['proposals']))}
    rows=complete['results'];keys=[(r['case_id'],r['proposal_index']) for r in rows]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('Incomplete or duplicate proposal inventory')
    result=[]
    for row in sorted(rows,key=lambda r:(r['case_id'],r['proposal_index'])):
        if row['spec']!=registration['proposals'][row['proposal_index']]:raise ValueError('Proposal specification changed')
        if type(row['eligible_model_proposal']) is not bool:raise ValueError('Explicit model eligibility required')
        if row['eligible_model_proposal']:
            if row.get('status')!='success' or not all(row.get(k) is True for k in
                ('hazard_fully_road_supported','start_free','goal_free','path_samples_free','timed_samples_free')):
                raise ValueError('Model eligibility lacks a complete witness')
            result.append(row)
    return result


def placement_inputs(routes,manifest,rows):
    used=sorted({manifest['cases'][r['case_id']]['route_index'] for r in rows})
    mapping={old:new for new,old in enumerate(used)}
    chosen={**deepcopy(manifest),'cases':[dict(route_index=mapping[manifest['cases'][r['case_id']]['route_index']],
                                           hazard=deepcopy(r['spec'])) for r in rows]}
    selected=[deepcopy(routes[i]) for i in used]
    registered_native_cases(dict(routes=selected,native_case_manifest=chosen))
    return selected,chosen


def prepare(parent,geometry_source,preflight,pilot,output):
    parent,geometry_source,preflight,pilot,output=map(lambda p:Path(p).resolve(),(parent,geometry_source,preflight,pilot,output))
    require_finished_parent(parent);require_finished_parent(pilot)
    if read(parent/'protocol.json')['methods']!=['H']:raise ValueError('This inventory requires the registered H-only policy')
    registration=read(preflight/'registration.json');complete=read(preflight/'complete.json')
    if registration['protocol_sha256']!=sha(geometry_source/'protocol.json'):raise ValueError('Geometry source changed')
    if complete['new_controls'] or complete['new_updates'] or complete['new_final_conditions']:raise ValueError('Require model-only preflight')
    rows=eligible_proposals(registration,complete)
    pp=read(pilot/'protocol.json');identity=pp['placement_transfer_pilot'];witness=identity['model_witness']
    key=(witness['case_id'],witness['proposal_index'])
    if identity['model_preflight_sha256']!=sha(preflight/'complete.json') or witness not in rows:
        raise ValueError('Reused pilot does not bind this proposal inventory')
    raw=read(pilot/'H/endpoint_raw_audit.json');evaluation=read(pilot/'H/evaluation.json')
    if raw['protocol_sha256']!=sha(pilot/'protocol.json') or raw['evaluation_sha256']!=sha(pilot/'H/evaluation.json'):
        raise ValueError('Pilot raw binding changed')
    if len(evaluation['episodes'])!=1 or not read(pilot/'strict_delivery.json')['strict_delivery_gate_passed']:
        raise ValueError('Pilot incomplete')
    for name in ('measured_start_inference.json','static_action_inference.json'):
        proof=read(pilot/'H'/name)
        if not proof['all_recorded_actions_match'] or proof['raw_audit_sha256']!=sha(pilot/'H/endpoint_raw_audit.json'):
            raise ValueError('Pilot inference binding changed')
    if evaluation['model_sha256']!=sha(parent/'H/model.zip') or evaluation['normalization_sha256']!=sha(parent/'H/normalization.pkl'):
        raise ValueError('Reused policy differs')
    # No success predicate: a fully recorded pilot failure must also be retained.
    remaining=[r for r in rows if (r['case_id'],r['proposal_index'])!=key]
    if not remaining:raise ValueError('Nothing new to evaluate')
    prepare_endpoint(parent,output,'static');p=read(output/'protocol.json')
    original=read(geometry_source/'protocol.json')
    for name,digest in original['source_hashes'].items():
        if name.startswith('navstack/') and p['source_hashes'].get(name)!=digest:raise ValueError('Physical/model implementation differs')
    p['routes'],p['native_case_manifest']=placement_inputs(p['routes'],p['native_case_manifest'],remaining)
    p['placement_inventory']=dict(version='model_supported_development_placements_v1',
        builder_sha256=sha(Path(__file__)),
        preflight_registration_sha256=sha(preflight/'registration.json'),preflight_complete_sha256=sha(preflight/'complete.json'),
        selected_keys=[[r['case_id'],r['proposal_index']] for r in rows],
        new_keys=[[r['case_id'],r['proposal_index']] for r in remaining],
        reused_key=list(key),reused_batch=pilot.name,reused_raw_sha256=sha(pilot/'H/endpoint_raw_audit.json'),
        selection='All model-eligible proposals,independent of policy outcomes;reuse complete pilot including any failure',
        new_PPO_updates=0,new_final_conditions=0,
        limits='New development distribution,not a rescore of the old12or independent final confirmation. No performance gain is attributed to learning. All new failures must be retained.')
    p['scope']='Frozen PPO transfer on all remaining model-supported development placements;zero learning'
    p['reference_context_endpoint']['phase']='model_supported_placement_inventory'
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    print(dict(total=len(rows),new=len(remaining),reused=list(key)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('parent','geometry-source','preflight','pilot','output'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();prepare(a.parent,a.geometry_source,a.preflight,a.pilot,a.output)
