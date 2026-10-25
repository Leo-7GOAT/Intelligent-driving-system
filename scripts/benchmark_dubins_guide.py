"""Paired offline search on a previously frame-bound first-wait state."""
import argparse
from copy import deepcopy
import json
import shutil
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.experiments import write_json,signature
from navstack.planning import HybridAStar
from navstack.road_grid import RoadGrid
from navstack.trajectory import parameterize
from navstack.world import World
from scripts.audit_native_method_endpoint import read,sha,read_prefix
from scripts.diagnose_rule_wait import first_wait_state
from scripts.evaluate_native_rule import require_finished_parent


def analyze(batch,diagnostic,output,case_id=3):
    batch,output=Path(batch),Path(output)
    diagnostic=Path(diagnostic) if diagnostic is not None else None
    if output.exists():raise FileExistsError('Preserve search evidence')
    protocol=read(batch/'protocol.json');raw_audit_sha=None
    if diagnostic is not None:
        d=read(diagnostic)
    else:
        from scripts.audit_native_rule import audit
        require_finished_parent(batch)
        if audit(batch)!=read(batch/'raw_audit.json'):raise ValueError('Whole-run raw audit is absent or stale')
        raw_audit_sha=sha(batch/'raw_audit.json');d={'case_id':case_id}
    scene_path=batch/f"scene_{d['case_id']:04d}.json"
    response_path=next((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    for path,key in [(batch/'protocol.json','protocol_sha256'),(scene_path,'scene_sha256'),
        (response_path,'response_sha256'),(batch/'evaluation_trace.jsonl.gz','trace_sha256')]:
        if diagnostic is not None and sha(path)!=d[key]:raise ValueError('Measured diagnostic input changed')
    for name,digest in protocol['source_hashes'].items():
        if name.startswith('navstack/') and sha(batch/'source'/name)!=digest:
            raise ValueError('Frozen observation/planning source changed')
    responses,_=read_prefix(response_path);rows,_=read_prefix(batch/'evaluation_trace.jsonl.gz')
    actual={};acks={}
    for wrapper in responses:
        r=wrapper.get('response',{})
        if r.get('episode_id') is not None and r.get('request_id') is not None:acks[r['episode_id'],r['request_id']]=r
        if 'vehicle_control' in r:actual[r['episode_id'],r['frame']]=r
    base=EnvConfig.from_dict(protocol['config'])
    measured=first_wait_state(rows,d['case_id'],actual,acks,base.actuator_observation)
    if diagnostic is not None and any(measured[k]!=d['selection'][k] for k in ('state','predecision_frame','selected_control_frames_verified')):
        raise ValueError('Selected measured state does not rebind')
    scene=read(scene_path)['scene'];world=World(scene['width'],scene['height'],scene['static'],road=RoadGrid(scene['road_grid']))
    state=measured['state'];pose=tuple(state[k] for k in ('x','y','yaw'));results=[]
    archive=output.with_suffix('')
    archive=archive.with_name(archive.name+'_source')
    archive.mkdir()
    for folder in ('navstack','scripts'):
        for source in (ROOT/folder).glob('*.py'):
            target=archive/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    (archive/'third_party/ompl').mkdir(parents=True)
    shutil.copy2(ROOT/'third_party/ompl/LICENSE',archive/'third_party/ompl/LICENSE')
    for mode in ('euclidean','dubins_v1'):
        config=deepcopy(base);config.planner.heuristic_mode=mode
        plan=HybridAStar(config.vehicle,config.planner).plan(pose,scene['goal'],world)
        record={'mode':mode,'signature':signature(config),'status':plan.status,'expanded_nodes':plan.expanded_nodes,
            'node_budget':config.planner.max_nodes,'failure_reason':plan.failure_reason,'cost':plan.cost,'path':plan.path.tolist()}
        if mode=='euclidean' and diagnostic is not None:
            expected=d['direct_goal_queries'][0]
            if any(record[k]!=expected[k] for k in ('status','expanded_nodes','failure_reason','cost')):
                raise ValueError('Default planning no longer reproduces frozen baseline')
        if plan.status=='success':
            ref=parameterize(plan.path,1.5,state['v'],config.vehicle,state['time'],True)
            if not all(world.free((x,y,yaw),config.vehicle) for x,y,yaw in zip(ref.x,ref.y,ref.yaw)):
                raise ValueError('Resampled reference fails actual body geometry')
            record['reference']={'samples':len(ref.x),'duration_seconds':float(ref.t[-1]),'terminal_speed':float(ref.v[-1])}
        results.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ('path','signature')}),flush=True)
    result={'scope':'Offline search-order intervention at a reverified CARLA state; no physical execution or learned-policy claim',
        'batch':batch.name,'case_id':d['case_id'],'diagnostic_sha256':sha(diagnostic) if diagnostic is not None else None,
        'baseline_raw_audit_sha256':raw_audit_sha,'script_sha256':sha(__file__),
        'source_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted((ROOT/'navstack').glob('*.py'))},
        'new_controls':0,'optimizer_updates':0,'state':state,'goal':scene['goal'],'results':results}
    write_json(output,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--diagnostic')
    p.add_argument('--case',type=int,default=3)
    p.add_argument('--output',required=True);a=p.parse_args();analyze(a.batch,a.diagnostic,a.output,a.case)
