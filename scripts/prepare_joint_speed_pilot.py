"""Freeze one previously failed placement for joint-speed engineering transfer."""
import argparse
from pathlib import Path
import sys
import math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_exact_compute_recovery import stage
from scripts.audit_native_method_endpoint import read,sha
from navstack.steering_retiming import VERSION
from navstack.steering_transition import settings
from navstack.research import atomic_json


def prepare(parent,retiming,road_proof,road_input,output,case_ids=(1,),prior_controls=0):
    parent,output=Path(parent),Path(output);r=read(Path(retiming));road=read(Path(road_proof));inputs=read(Path(road_input))
    successful=[x for x in r['results'] if x['search_speed_ceiling']==.25]
    geometry=[x for x in road['results'] if x['key']==['case1',.25]]
    if (len(successful)!=1 or not successful[0]['geometry_unchanged']
            or successful[0]['retimed_seconds']>=45 or len(geometry)!=1 or geometry[0]['non_driving_points']!=0
            or r['module_sha256']!=sha(ROOT/'navstack/steering_retiming.py')
            or road['input_sha256']!=sha(Path(road_input))):
        raise ValueError('Require source-bound model retiming and native SDK road witness')
    selected=[x for x in inputs['references'] if x['key']==['case1',.25]]
    poses=[[x-lat*math.sin(yaw),y+lat*math.cos(yaw),yaw+turn]
        for x,y,yaw in successful[0]['poses'] for lat in (-.05,0,.05) for turn in (-.05,0,.05)]
    scene=read(parent/'H/scene_0002.json')['scene']
    if len(selected)!=1 or selected[0]['poses']!=poses or selected[0]['layout']!=scene['native_layout']:
        raise ValueError('Road witness does not bind exact retimed geometry/layout')
    if list(case_ids) not in ([1],[0,4,21,27]):raise ValueError('Use the bounded prespecified pilot inventories')
    stage(parent,output,parent/'resume/H',list(case_ids),prior_controls)
    p=read(output/'protocol.json');p['reference_joint_speed']=dict(version=VERSION,spatial_speed_ceiling=.25)
    p['joint_speed_pilot_registration']=dict(original_case_ids=list(case_ids),new_PPO_updates=0,new_final_conditions=0,
        offline_retiming_sha256=sha(Path(retiming)),offline_road_sha256=sha(Path(road_proof)),
        scope='Prespecified known development conditions,frozen PPO engineering transfer;not learning gain or independent confirmation',
        search_budget='At most two32000-node searches per refinement attempt;every attempt and cache cost recorded')
    p['scope']='Frozen PPO known-case joint speed/steering reference transfer;unchanged actor control and scoring'
    settings(p);atomic_json(output/'protocol.json',p)
    atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','retiming','road-proof','road-input','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--case-ids',type=int,nargs='+',default=[1]);p.add_argument('--prior-controls',type=int,default=0)
    a=p.parse_args();prepare(a.parent,a.retiming,a.road_proof,a.road_input,a.output,a.case_ids,a.prior_controls)
