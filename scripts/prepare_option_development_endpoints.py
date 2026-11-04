"""Freeze all30existing placements and unchanged known24 for the adapted model."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_reference_evaluation import prepare as prepare_endpoint
from scripts.prepare_native_placement_inventory import eligible_proposals,placement_inputs


def prepare(parent,geometry,preflight,placements,known):
    parent,geometry,preflight,placements,known=map(lambda x:Path(x).resolve(),[parent,geometry,preflight,placements,known])
    registration=read(preflight/'registration.json');complete=read(preflight/'complete.json')
    if registration['protocol_sha256']!=sha(geometry/'protocol.json'):raise ValueError('Preflight geometry source changed')
    if complete['new_controls'] or complete['new_updates'] or complete['new_final_conditions']:raise ValueError('Require model-only original preflight')
    rows=eligible_proposals(registration,complete)
    if len(rows)!=30:raise ValueError('Preserve complete original30eligible placements')
    old=read(geometry/'protocol.json');p0=read(parent/'protocol.json')
    if p0['calibration_sha256']!=old['calibration_sha256']:raise ValueError('Vehicle calibration changed')
    geometry_modules=['world.py','vehicle.py','road_grid.py','road_grid_batch.py','native_hazards.py','native_route.py','lane_projection.py']
    for name in geometry_modules:
        if sha(parent/'source/navstack'/name)!=sha(geometry/'source/navstack'/name):raise ValueError('Actual geometry/model changed:'+name)
    prepare_endpoint(parent,placements,'static');p=read(placements/'protocol.json')
    p['routes'],p['native_case_manifest']=placement_inputs(p['routes'],p['native_case_manifest'],rows)
    p.pop('fixed_lowlevel_training',None)
    p['placement_inventory']=dict(version='all_existing_model_supported_placements_new_policy_v2',
        selected_keys=[[r['case_id'],r['proposal_index']] for r in rows],reused_cases=0,
        preflight_registration_sha256=sha(preflight/'registration.json'),preflight_complete_sha256=sha(preflight/'complete.json'),
        unchanged_actual_geometry_modules=geometry_modules,old_witness_node_budget=registration['planner_node_budget'],
        new_planner_node_budget=32000,new_PPO_updates=0,new_final_conditions=0,
        selection='All30original model-eligible proposals,regardless of all old/new policy outcomes;rerun all because model/lowlevel changed',
        limits='Old geometric witnesses establish the original scenario-construction criterion,not feasibility of every new steering-aware search. Known development conditions,not independent final confirmation or an isolated learning comparison.')
    p['scope']='Adapted fixed-option PPOtransfer on all30unchanged known development placements;zero updates'
    atomic_json(placements/'protocol.json',p);atomic_json(placements/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(placements/'protocol.json')))
    prepare_endpoint(parent,known,'known_validation');q=read(known/'protocol.json');q.pop('fixed_lowlevel_training',None)
    atomic_json(known/'protocol.json',q);atomic_json(known/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(known/'protocol.json')))
    if p['source_hashes']!=q['source_hashes']:raise ValueError('Both endpoints must freeze identical source')
    print(dict(placements=str(placements),known=str(known),cases=54,reused=0,new_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','geometry','preflight','placements','known']:p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.parent,a.geometry,a.preflight,a.placements,a.known)
