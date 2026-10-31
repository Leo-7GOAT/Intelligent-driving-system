"""Save the first recorded7200-node failure or first successful search,offline."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.vehicle import VehicleState,Command
from scripts.audit_native_method_endpoint import read,sha
from scripts.audit_native_policy_sequence import observation_env


def extract(batch,method,output,success=False,minimum_nodes=1):
    batch,output=Path(batch),Path(output);folder=batch/method
    if output.exists():raise FileExistsError('Preserve recorded query')
    raw=read(folder/'endpoint_raw_audit.json');protocol=read(batch/'protocol.json')
    for name in ('config','env','vehicle','world','planning','road_grid','road_grid_batch','subgoals','native_route','dubins_guide'):
        relative='navstack/'+name+'.py';frozen=batch/'source'/relative
        if sha(frozen)!=protocol['source_hashes'][relative] or frozen.read_text()!= (ROOT/relative).read_text():
            raise ValueError('Query decoder/search source differs: '+name)
    for path,key in ((batch/'protocol.json','protocol_sha256'),(folder/'config.json','config_sha256'),
                     (folder/'evaluation_trace.jsonl.gz','evaluation_trace_sha256')):
        if sha(path)!=raw[key]:raise ValueError('Raw-audited query input differs')
    responses=batch/'infrastructure'/raw['bridge_response_stream']
    if sha(responses)!=raw['bridge_response_sha256']:raise ValueError('Native response stream differs')
    begins={}
    with gzip.open(responses,'rt',encoding='utf-8') as stream:
        for line in stream:
            response=json.loads(line).get('response',{})
            if 'scene' in response:begins[response['episode_id']]=response
    config=EnvConfig.from_dict(read(folder/'config.json')['env'])
    if config.method!='H' or config.planner.max_seconds is not None:raise ValueError('Require fixed-node hierarchical search')
    current=None;before=None;previous=None
    with gzip.open(folder/'evaluation_trace.jsonl.gz','rt',encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            row=json.loads(line);info=row['info'];trace=info['trace']
            if current!=row['case_id']:
                current=row['case_id'];before=info['initial_scenario']['initial'];previous={'acceleration':0.,'steering':0.}
            plan=info['plan']
            selected=(plan is not None and not plan.get('cache_hit',False) and
                (plan['status']=='success' and plan['expanded_nodes']>=minimum_nodes if success
                 else plan.get('failure_reason')=='search_budget' and plan.get('expanded_nodes')==7200))
            if selected:
                begin=begins[trace[0]['backend_evidence']['episode_id']];scene=begin['scene']
                if info['initial_scenario']['dynamic']:raise ValueError('First selected query has moving actors;do not silently choose another')
                env=observation_env(config,scene,info['initial_scenario'],begin=begin)
                try:
                    env.plant.state=VehicleState(**before);env.plant.previous_command=Command(**previous)
                    target,speed=env.decode_action(int(row['policy_action']))
                finally:env.close()
                fixture={'scope':__doc__,'selection':f'First actual successful non-reused search with at least{minimum_nodes}nodes' if success else 'First actual7200-node failure;no timing/outcome-based best-query selection',
                    'batch':batch.name,'method':method,'case_id':current,'trace_row':index,'policy_action':int(row['policy_action']),
                    'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'trace_sha256':raw['evaluation_trace_sha256'],
                    'extractor_sha256':sha(Path(__file__)),'measured_before_state':before,'state':before,
                    'goal':list(target),'selected_speed':speed,'config':config.to_dict(),
                    'world':{k:scene[k] for k in ('width','height','static','road_grid')},
                    'recorded_plan':{k:plan[k] for k in ('status','expanded_nodes','failure_reason')},
                    'new_controls':0,'optimizer_updates':0}
                with output.open('x',encoding='utf-8') as stream:json.dump(fixture,stream,separators=(',',':'),allow_nan=False)
                print(json.dumps({'case_id':current,'row':index,'status':plan['status'],'nodes':plan['expanded_nodes'],'fixture_sha256':sha(output)}))
                return fixture
            before=trace[-1]['state'];previous=trace[-1]['command']
    raise ValueError('No registered query of the requested kind')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('batch','method','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--success',action='store_true')
    parser.add_argument('--minimum-nodes',type=int,default=1)
    args=parser.parse_args();extract(args.batch,args.method,args.output,args.success,args.minimum_nodes)
