"""Freeze two known failed conditions after source-bound exact-point SDK probes."""
import argparse
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_exact_compute_recovery import stage
from scripts.audit_native_method_endpoint import read,sha
from navstack.research import atomic_json
from navstack.steering_retiming import VERSION as JOINT_VERSION
from navstack.reference_point_constraints import VERSION
from navstack.reference_road_refinement import settings


def prepare(parent,evidence,output):
    parent,evidence,output=map(Path,[parent,evidence,output])
    inputs=read(evidence/'native_case21_27_exact_point_road_inputs.json')
    road=read(evidence/'native_case21_27_exact_point_road_probe.json')
    if road['input_sha256']!=sha(evidence/'native_case21_27_exact_point_road_inputs.json'):
        raise ValueError('Native SDK road proof changed')
    for case in [21,27]:
        proof=read(evidence/f'native_case{case}_exact_point_runtime_source_bound.json')
        if proof['status']!='success' or not proof['exact_counterexamples'] or proof['nodes']>32000:
            raise ValueError('Require bounded successful point-constraint model probe')
        for name,digest in proof['constraint_source_hashes'].items():
            if sha(ROOT/name)!=digest:raise ValueError('Point-constraint implementation changed')
        selected=[x for x in inputs['references'] if x['key']==['exact_point',case]]
        results=[x for x in road['results'] if x['key']==['exact_point',case]]
        poses=[[x-lat*math.sin(yaw),y+lat*math.cos(yaw),yaw+turn]
            for x,y,yaw in proof['poses'] for lat in (-.05,0,.05) for turn in (-.05,0,.05)]
        if (len(selected)!=1 or selected[0]['poses']!=poses or selected[0]['layout']!=proof['layout']
                or len(results)!=1 or results[0]['non_driving_points']!=0):
            raise ValueError('Require full nine-offset native SDK acceptance of exact candidate poses')
    stage(parent,output,parent/'resume/H',[21,27],28964)
    p=read(output/'protocol.json');p['reference_joint_speed']=dict(version=JOINT_VERSION,spatial_speed_ceiling=.25)
    p['reference_point_constraints']=dict(version=VERSION)
    p['exact_point_pilot_registration']=dict(original_case_ids=[21,27],new_PPO_updates=0,new_final_conditions=0,
        offline_road_sha256=sha(evidence/'native_case21_27_exact_point_road_probe.json'),
        constraint_source_hashes=proof['constraint_source_hashes'],maximum_points=8192,
        scope='Prespecified known failed conditions;frozen PPO engineering transfer,not learning or independent confirmation',
        actual_road_scoring_and_nine_offset_native_validation_unchanged=True)
    settings(p);p['scope']='Frozen PPO engineering transfer with exact native-counterexample planning constraints'
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','evidence','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();prepare(a.parent,a.evidence,a.output)
