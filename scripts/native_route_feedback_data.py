"""Augment audited TRAIN observations with errors from measured ego and known map."""
import json
from pathlib import Path
import numpy as np
from navstack.config import EnvConfig
from navstack.native_cases import registered_native_cases
from navstack.route_error import route_error_features
from navstack.vehicle import VehicleState
from scripts.audit_native_method_endpoint import read_prefix,sha
from scripts.fit_broader_measured_actor import validated_arrays
from scripts.fit_broader_native_recovery import recovery_masks
from scripts.fit_diverse_native_actor import balanced_masses
from scripts.evaluate_native_rule import require_finished_parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def measured_begins(batch):
    paths=list((batch/'infrastructure').glob('*_responses.jsonl.gz'))
    if len(paths)!=1:raise ValueError('Require one complete measured bridge stream')
    rows,error=read_prefix(paths[0])
    if error:raise ValueError(error)
    return {x['response']['episode_id']:x['response'] for x in rows if 'scene' in x.get('response',{})}


def teacher_features(batch,vehicle,control_dt):
    rows,error=read_prefix(batch/'evaluation_trace.jsonl.gz')
    if error:raise ValueError(error)
    begins=measured_begins(batch);result=[]
    for case in range(36):
        ticks=[t for r in rows if r['case_id']==case for t in r['info']['trace']]
        begin=begins[ticks[0]['backend_evidence']['episode_id']]
        state=VehicleState(**begin['state']);frame=begin['frame'];route=begin['scene']['native_layout']['path']
        for tick in ticks:
            if tick['backend_frame']!=frame+1 or abs(tick['state']['time']-state.time-control_dt)>1e-6:
                raise ValueError('Measured teacher history discontinuity')
            result.append(route_error_features(state,route,vehicle))
            state=VehicleState(**tick['state']);frame=tick['backend_frame']
    return np.asarray(result,dtype=np.float32)


def load_bundle(workspace,reference_config):
    root=Path(workspace);reference_config=Path(reference_config);cfg=EnvConfig.from_dict(read(reference_config)['env'])
    if cfg.flat_route_observation:raise ValueError('Source data must retain the original observation schema')
    old=root/'runs/20261102T053259603507Z_native_B3_broader_TRAIN_teacher'
    diverse_execution=root/'runs/B3_diverse_TRAIN36_parallel_conversion_20261102'
    if read(diverse_execution/'status.json')['phase']!='ALL36_DIVERSE_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE':
        raise ValueError('Diverse labels are not complete')
    new=root/'runs'/read(diverse_execution/'registration.json')['batch']
    folders=[root/'runs/B3_broader_TRAIN_teacher_execution_v4_20261102/direct_views',diverse_execution/'direct_views']
    parts=[];provenance=[];protocols=[];cases=[]
    for teacher,folder,offset in zip((old,new),folders,(0,4096)):
        require_finished_parent(teacher);p=read(teacher/'protocol.json');m=read(folder/'manifest.json')
        if (p.get('data_role')!='TRAIN_EXTRA_SUPERVISION' or p.get('native_validation') is not None
                or m['reference_config_sha256']!=sha(reference_config)
                or m['sources'][0]['protocol_sha256']!=sha(teacher/'protocol.json')
                or m['sources'][0]['raw_audit_sha256']!=sha(teacher/'raw_audit.json')
                or m['sources'][0]['trace_sha256']!=sha(teacher/'evaluation_trace.jsonl.gz')):
            raise ValueError('Nominal TRAIN source provenance differs')
        c=registered_native_cases(p);families=[x.get('native_hazard',{}).get('kind','curve') for x in c]
        x,y,g,t,h,_=validated_arrays(folder/'measured_demonstrations.npz',m,families)
        if np.any(x[:,599:603]!=0.):raise ValueError('Cannot replace nonconstant source feedback fields')
        features=teacher_features(teacher,cfg.vehicle,cfg.control_dt)
        if features.shape!=(len(x),4):raise ValueError('Teacher observations and measured states differ')
        x[:,599:603]=features;parts.append((x,y,g+offset,t,h));protocols.append(p);cases.append(c)
        provenance.append(dict(batch=teacher.name,manifest_sha256=sha(folder/'manifest.json'),dataset_sha256=m['dataset_sha256']))
    collection=root/'runs/20261102T084415462162Z_native_broader_TRAIN_recovery'
    require_finished_parent(collection);audit=read(collection/'B1/collection_raw_audit.json')
    if (not audit['all_measured_observations_bit_exact'] or audit['dataset_sha256']!=sha(collection/'B1/collection.npz')
            or audit['trace_sha256']!=sha(collection/'B1/collection_trace.jsonl.gz')):raise ValueError('Executed recovery data changed')
    with np.load(collection/'B1/collection.npz',allow_pickle=False) as loaded:data={k:loaded[k] for k in loaded.files}
    t,h=recovery_masks(data,audit['eligible_executed_teacher_labels']);begins=measured_begins(collection)
    rows,error=read_prefix(collection/'B1/collection_trace.jsonl.gz')
    if error:raise ValueError(error)
    x=data['observations']
    if len(rows)!=len(x) or np.any(x[:,599:603]!=0.):raise ValueError('Recovery input/schema inventory differs')
    for index,row in enumerate(rows):
        episode=row['info']['trace'][0]['backend_evidence']['episode_id']
        x[index,599:603]=route_error_features(VehicleState(**row['state_before']),begins[episode]['scene']['native_layout']['path'],cfg.vehicle)
    parts.insert(1,(x,data['actions'],data['groups'],t,h))
    x,y,groups,train,holdout=(np.concatenate([part[i] for part in parts]) for i in range(5))
    bases=protocols[0]['routes'][:4];cells={}
    for i,c in enumerate(cases[0]):
        path=c['native_route']['points'];matches=[j for j,b in enumerate(bases) if path==b['points'][:len(path)]]
        if len(matches)!=1:raise ValueError('Old physical base-path identity differs')
        cell=(matches[0],c.get('native_hazard',{}).get('kind','curve'));cells[i]=cell;cells[2048+i]=cell
    for i,c in enumerate(cases[1]):
        route=protocols[1]['native_case_manifest']['cases'][i]['route_index']
        cells[4096+i]=(4+route,c.get('native_hazard',{}).get('kind','curve'))
    masses=balanced_masses(cells,np.unique(groups[train]))
    if {cells[g][0] for g in masses}!=set(range(16)):raise ValueError('Missing physical geometry coverage')
    return dict(observations=x,actions=y,groups=groups,train=train,holdout=holdout,masses=masses,
        provenance=provenance,recovery_audit_sha256=sha(collection/'B1/collection_raw_audit.json'))
