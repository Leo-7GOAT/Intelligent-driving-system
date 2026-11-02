"""Select twelve genuinely different TRAIN paths from the saved geometry survey."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_cases import registered_native_cases
from scripts.prepare_native_validation_inventory import separated, manifest, case, static, walker

TARGETS = (-15,15,-30,30,-45,45,-60,60,-75,75,-20,20,-40,40,-65,65)


def select(survey, excluded, count=12):
    selected = []
    for target in TARGETS:
        candidates = sorted((r for r in survey['candidates'] if r['heading_change_degrees']*target > 0),
            key=lambda r: (abs(r['heading_change_degrees']-target), r['geometry_id']))
        for raw in candidates:
            if separated(raw, excluded+selected):
                selected.append({**deepcopy(raw), 'map':survey['map'],
                    'opendrive_sha256':survey['opendrive_sha256'], 'selection_target_degrees':target})
                break
        if len(selected) == count: return selected
    raise ValueError('Insufficient separated paths; never lower separation or duplicate prefixes')


def prepare(survey_path, inventory_path, output):
    survey_path, inventory_path, output = map(Path, (survey_path,inventory_path,output))
    if output.exists(): raise FileExistsError('Preserve the registered geometry-only TRAIN inventory')
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    survey, old=read(survey_path),read(inventory_path)
    if sha(survey_path)!=old['registration']['survey_sha256'] or survey['world_ticks_requested']!=0:
        raise ValueError('Require the original read-only measured map survey')
    excluded=old['routes']+old['native_validation']['routes']
    routes=select(survey,excluded)
    cases=[case(i) for i in range(12)]
    cases += [case(i,static(13.)) for i in range(12)]
    cases += [case(i,walker(10.,1.,2.)) for i in range(12)]
    result=dict(routes=routes,native_case_manifest=manifest(cases),data_role='TRAIN_EXTRA_SUPERVISION',
        registration=dict(version='diverse_native_TRAIN36_geometry_v2',survey_sha256=sha(survey_path),
            excluded_inventory_sha256=sha(inventory_path),minimum_sampled_centreline_distance_m=13.25,
            selection_targets=list(TARGETS),base_paths=12,route_prefix_augmentation=False,
            core_conditions_each=12,physical_controls=0,teacher_labels=0,optimizer_updates=0,
            rationale='Old12TRAINsegments are prefixes of four base paths. Select twelve separated base paths using geometry only.',
            scope='Provisional TRAIN geometry,not driving evidence,feasibility guarantee,validation or final confirmation. No policy results used for selection.'))
    if len(registered_native_cases(result))!=36:raise ValueError('Missing registered TRAIN conditions')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(dict(status='GEOMETRY_REGISTERED_NOT_DRIVEN',base_paths=12,TRAIN_conditions=36,
        validation_conditions=0,physical_controls=0,sha256=sha(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey',required=True);parser.add_argument('--excluded-inventory',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();prepare(args.survey,args.excluded_inventory,args.output)
