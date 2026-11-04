"""Freeze a known-case engineering transfer;no training or scoring changes."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.reference_road_refinement import VERSION,settings
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_reference_evaluation import prepare as prepare_endpoint
from scripts.prepare_native_camera import selected_camera_inputs


def prepare(parent,output,ids):
    output=Path(output).resolve()
    if read(Path(parent)/'protocol.json')['methods']!=['H']:raise ValueError('Frozen H-only pilot required')
    prepare_endpoint(parent,output,'static');p=read(output/'protocol.json')
    p['routes'],p['native_case_manifest'],mapping=selected_camera_inputs(p['routes'],p['native_case_manifest'],ids)
    p['reference_road_refinement']=dict(version=VERSION,max_attempts=3)
    p['refinement_transfer_registration']=dict(original_case_ids=ids,builder_sha256=sha(Path(__file__)),
        physics_and_actual_v4_scoring_unchanged=True,planning_only_constraints=True,
        original_to_subset_route_indices=mapping,new_PPO_updates=0,new_final_conditions=0,
        selection='Prespecified known safe control and original bounds-failure case;retain all outcomes',
        limits='Frozen-policy deployment intervention after a lowlevel change;not learning,independent confirmation or a rescore. Sampled error offsets do not guarantee continuous safety.')
    p['scope']='Known-case frozen PPO engineering transfer with finite native reference-road validation'
    p['reference_context_endpoint']['phase']='reference_road_refinement_pilot'
    for directory in ('navstack','scripts'):
        for source in (ROOT/directory).rglob('*'):
            if source.is_file() and '__pycache__' not in source.parts and source.suffix in ('.py','.ps1'):
                destination=output/'source'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    settings(p);atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--case-ids',nargs='+',type=int,default=[1,4]);a=parser.parse_args();prepare(a.parent,a.output,a.case_ids)
