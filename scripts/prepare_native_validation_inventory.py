"""Register development train/validation roads using geometry only,no policy calls."""
import argparse
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.native_partitions import VERSION,validate_native_partitions
from navstack.native_cases import VERSION as CASE_VERSION,registered_native_cases

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def geometry_id(points):
    return hashlib.sha256(json.dumps([[round(p[k],3) for k in ('x','y','yaw_degrees_carla')] for p in points]).encode()).hexdigest()
def prefix(route,count):
    result=deepcopy(route);result['points']=result['points'][:count+1]
    result['geometry_id']=geometry_id(result['points'])
    result['length_m']=sum(math.hypot(a['x']-b['x'],a['y']-b['y']) for a,b in zip(result['points'],result['points'][1:]))
    result['heading_change_degrees']=(result['points'][-1]['yaw_degrees_carla']-result['points'][0]['yaw_degrees_carla']+180)%360-180
    return result
def separated(route,others,gap=12.):
    return all(route['geometry_id']!=other['geometry_id'] and all(
        math.hypot(a['x']-b['x'],a['y']-b['y'])>=gap+1.25
        for a in route['points'] for b in other['points']) for other in others)
def manifest(cases):return {'version':CASE_VERSION,'split':'development','cases':cases}
def case(index,hazard=None):return {'route_index':index,'hazard':deepcopy(hazard)}
def static(distance):return {'version':'native_hazard_v1','kind':'static_vehicle','distance_m':distance,'static_geometry':'projected_box_strips12_v1'}
def walker(distance,speed,delay):return {'version':'native_hazard_v1','kind':'crossing_walker','distance_m':distance,'lateral_m':-2.5,'speed_m_s':speed,'start_delay_s':delay}

def prepare(survey_path,parent_path,history_root,output):
    if output.exists():raise FileExistsError('Preserve the registered geometry inventory')
    survey,parent=read(survey_path),read(parent_path)
    historical=[];receipts=[]
    for folder in sorted(history_root.iterdir()):
        path=folder/'protocol.json'
        if not folder.is_dir() or not path.is_file():continue
        value=read(path)
        routes=value.get('routes',[])
        if not isinstance(routes,list):continue
        selected=[r for r in routes if isinstance(r,dict) and all(k in r for k in ('points','map','opendrive_sha256','geometry_id'))
                  and (r['map'],r['opendrive_sha256'])==(survey['map'],survey['opendrive_sha256'])]
        if selected:
            historical.extend(selected);receipts.append({'batch':folder.name,'protocol_sha256':sha(path),'geometries':sorted({r['geometry_id'] for r in selected})})
    originals=deepcopy(parent['routes'])
    if len(originals)!=4:raise ValueError('Require the original four geometry-selected training roads')
    historical.extend(originals)
    choices=[]
    for target in (-30.,-60.,30.,60.):
        candidates=sorted((r for r in survey['candidates'] if r['heading_change_degrees']*target>0),
                          key=lambda r:(abs(r['heading_change_degrees']-target),r['geometry_id']))
        for raw in candidates:
            route={**deepcopy(raw),'map':survey['map'],'opendrive_sha256':survey['opendrive_sha256'],'selection_target_degrees':target}
            if separated(route,historical+choices):choices.append(route);break
        else:raise ValueError('Survey lacks four spatially separated unused registered development roads')
    train_routes=originals+[prefix(r,count) for count in (16,20) for r in originals]
    train_cases=deepcopy(parent['native_case_manifest']['cases'])
    train_cases += [case(i) for i in range(4,12)]
    train_cases += [case(i,static(distance)) for distance in (11.,15.) for i in range(4)]
    train_cases += [case(i,walker(distance,speed,delay)) for distance,speed,delay in ((8.,1.2,0.),(12.,.6,4.)) for i in range(4)]
    val_routes=choices+[prefix(r,16) for r in choices]
    val_cases=[case(i) for i in range(8)]
    val_cases += [case(i,static(distance)) for distance in (12.,14.) for i in range(4)]
    val_cases += [case(i,walker(distance,speed,delay)) for distance,speed,delay in ((9.,.6,0.),(11.,1.2,3.5)) for i in range(4)]
    result={'routes':train_routes,'native_case_manifest':manifest(train_cases),
       'native_validation':{'version':VERSION,'routes':val_routes,'manifest':manifest(val_cases),'minimum_route_separation_m':12.}}
    verification=validate_native_partitions(result)
    assert len(registered_native_cases(result))==36 and len(registered_native_cases(result,'evaluation'))==24
    result['registration']={'scope':'Geometry-only development training/validation inventory. No policy inference,teacher labels,CARLA steps or final confirmation. Not yet activated in a training batch.',
        'survey_sha256':sha(survey_path),'parent_protocol_sha256':sha(parent_path),'historical_protocols':receipts,
        'selection':'Nearest signed heading targets -30,-60,30,60 with geometry-id tie break; sampled centreline separation at least13.25m from every historical registered same-map route and each earlier selected road.',
        'training_core_conditions_each':12,'validation_core_conditions_each':8,
        'historical_exclusion_scope':'Top-level run protocol route inventories,including prepared but unexecuted variants. This does not certify no historical vehicle ever visited a nearby point.',
        'no_policy_results_used_for_road_selection':True,'verification':verification}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'REGISTERED_NOT_ACTIVATED','training':36,'validation':24,'historical_protocols':len(receipts),'output_sha256':sha(output)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--survey',type=Path,required=True);parser.add_argument('--parent-protocol',type=Path,required=True)
    parser.add_argument('--history-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.survey,args.parent_protocol,args.history_root,args.output)
