"""Fixed actor continuation with geometry/family/episode-balanced actual TRAIN labels."""
import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def balanced_masses(group_cells,active_groups):
    """Equal geometry,then family,then episode mass;episode length cannot dominate."""
    active=set(map(int,active_groups))
    if not active or active-set(group_cells):raise ValueError('Missing physical geometry/family identity')
    geometries={group_cells[g][0] for g in active};masses={}
    for geometry in geometries:
        selected=[g for g in active if group_cells[g][0]==geometry]
        families={group_cells[g][1] for g in selected}
        for family in families:
            episodes=[g for g in selected if group_cells[g][1]==family]
            for group in episodes:masses[group]=1/(len(geometries)*len(families)*len(episodes))
    return masses


def run(collection,views,reference_config,output,profile='diverse_geometry_v4'):
    if profile!='diverse_geometry_v4':raise ValueError('Unknown fixed profile')
    collection,views,reference_config,output=map(lambda p:Path(p).resolve(),(collection,views,reference_config,output))
    if output.exists():raise FileExistsError('Never repeat or overwrite the diverse actor fit')
    if any(k.startswith('navstack.') for k in sys.modules):raise RuntimeError('Use a fresh frozen-runtime interpreter')
    sys.path.insert(0,str(collection/'source'))
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    from navstack.native_cases import registered_native_cases
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import sha
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    from scripts.evaluate_native_rule import require_finished_parent
    recipe_file='protocols/native_B1_diverse_geometry_actor_20261102.json';recipe_path=ROOT/recipe_file;p=read(recipe_path)
    helper_path=ROOT/'scripts/fit_broader_measured_actor.py'
    if (p['version']!='diverse_geometry_actor_v4' or p['steps']!=8192 or p['retained_actor_optimizer_steps']!=16384
            or p['fit_script_sha256']!=sha(__file__) or p['helper_script_sha256']!=sha(helper_path)):
        raise ValueError('Fixed diverse fit recipe/source changed')
    require_finished_parent(collection)
    audit=read(collection/'B1/collection_raw_audit.json')
    if (audit['protocol_sha256']!=sha(collection/'protocol.json') or not audit['all_measured_observations_bit_exact']
            or audit['dataset_sha256']!=sha(collection/'B1/collection.npz')
            or not read(collection/'B1/strict_delivery.json')['strict_delivery_gate_passed']):
        raise ValueError('Original executed recovery evidence changed')
    module=importlib.util.spec_from_file_location('diverse_actor_helper',helper_path)
    helper=importlib.util.module_from_spec(module);module.loader.exec_module(helper)
    recovery_path=ROOT/'scripts/fit_broader_native_recovery.py'
    if sha(recovery_path)!=p['recovery_mask_script_sha256']:raise ValueError('Executed-label eligibility helper changed')
    recovery_module=importlib.util.spec_from_file_location('diverse_recovery_masks',recovery_path)
    recovery=importlib.util.module_from_spec(recovery_module);recovery_module.loader.exec_module(recovery)
    def nominal(folder,teacher,offset):
        require_finished_parent(teacher)
        tp=read(teacher/'protocol.json');m=read(folder/'manifest.json');cases=registered_native_cases(tp)
        if (tp.get('native_validation') is not None or tp.get('data_role')!='TRAIN_EXTRA_SUPERVISION'
                or m['reference_config_sha256']!=sha(reference_config)
                or m['sources'][0]['protocol_sha256']!=sha(teacher/'protocol.json')
                or m['sources'][0]['raw_audit_sha256']!=sha(teacher/'raw_audit.json')):
            raise ValueError('Nominal supervision is not hash-bound TRAIN data')
        families=[c.get('native_hazard',{}).get('kind','curve') for c in cases]
        arrays=helper.validated_arrays(folder/'measured_demonstrations.npz',m,families)
        return (*arrays[:2],arrays[2]+offset,*arrays[3:5],cases,tp,m)
    old_teacher=ROOT/'runs/20261102T053259603507Z_native_B3_broader_TRAIN_teacher'
    x,y,groups,train,holdout,old_cases,old_protocol,old_manifest=nominal(views,old_teacher,0)
    with np.load(collection/'B1/collection.npz',allow_pickle=False) as d:
        rt,rh=recovery.recovery_masks(d,audit['eligible_executed_teacher_labels'])
        x=np.concatenate((x,d['observations']));y=np.concatenate((y,d['actions']));groups=np.r_[groups,d['groups']]
        train=np.r_[train,rt];holdout=np.r_[holdout,rh]
    teacher_execution=ROOT/p['diverse_teacher_execution'];state=read(teacher_execution/'status.json')
    if state['phase']!='ALL36_DIVERSE_TRAIN_TEACHER_AND_DIRECT_VIEWS_COMPLETE':raise ValueError('Wait for actual completed diverse data')
    new_teacher=ROOT/'runs'/read(teacher_execution/'registration.json')['batch']
    strict=read(teacher_execution/'strict_delivery.json')
    nx,ny,ng,nt,nh,new_cases,new_protocol,new_manifest=nominal(teacher_execution/'direct_views',new_teacher,4096)
    if (not strict['all_engine_controls_acknowledged'] or strict['verified_controls']!=new_manifest['teacher_unique_controls']
            or new_protocol['teacher_collection']['inventory_sha256']!=p['diverse_inventory_sha256']):
        raise ValueError('Diverse physical dataset/engine inventory differs')
    x=np.concatenate((x,nx));y=np.concatenate((y,ny));groups=np.r_[groups,ng];train=np.r_[train,nt];holdout=np.r_[holdout,nh]
    bases=old_protocol['routes'][:4];cells={}
    for i,c in enumerate(old_cases):
        points=c['native_route']['points']
        matches=[j for j,r in enumerate(bases) if points==r['points'][:len(points)]]
        if len(matches)!=1:raise ValueError('Original prefix/base identity ambiguous')
        cell=(matches[0],c.get('native_hazard',{}).get('kind','curve'))
        cells[i]=cell;cells[2048+i]=cell
    for i,c in enumerate(new_cases):
        route_index=new_protocol['native_case_manifest']['cases'][i]['route_index']
        cells[4096+i]=(4+route_index,c.get('native_hazard',{}).get('kind','curve'))
    active=np.unique(groups[train]);masses=balanced_masses(cells,active)
    if {cells[int(g)][0] for g in active}!=set(range(16)):
        raise ValueError('Require successful nominal coverage of all twelve new base paths')
    prior=ROOT/p['resume_fit']
    for name,digest in p['resume_hashes'].items():
        if sha(prior/name)!=digest:raise ValueError('Actual saved Adam/sampler/checkpoint changed')
    origin=prior/'checkpoint';metadata=verified_checkpoint_metadata(origin)
    if (metadata['model_sha256']!=p['source_model_sha256']
            or read(origin/'config.json')['env']!=read(reference_config)['env']):raise ValueError('Diverse continuation parent/interface differs')
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu')
    with (origin/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    rms=pickle.dumps(norm.obs_rms);output.mkdir()
    shutil.copy2(__file__,output/'executed_fit.py');shutil.copy2(helper_path,output/'executed_helper.py')
    shutil.copy2(recovery_path,output/'executed_mask_helper.py')
    atomic_json(output/'registration.json',dict(version=p['version'],recipe_file=recipe_file,recipe_sha256=sha(recipe_path),
        seed=22,steps=8192,retained_recovery_actor_steps=16384,collection=collection.name,
        collection_audit_sha256=sha(collection/'B1/collection_raw_audit.json'),
        nominal_manifest_sha256=sha(views/'manifest.json'),diverse_manifest_sha256=sha(teacher_execution/'direct_views/manifest.json'),
        source_model_sha256=metadata['model_sha256'],source_normalization_sha256=metadata['normalization_sha256'],
        group_masses={str(g):masses[g] for g in sorted(masses)},group_cells={str(g):cells[g] for g in sorted(masses)},
        sampling='Equal physicalbasegeometry,then family,then episode;rows uniformly within episode. Only TRAINlabels;correlatedholdout excluded.',
        new_controls=0,new_PPO_updates=0,source_hashes={n:sha(output/n) for n in ('executed_fit.py','executed_helper.py','executed_mask_helper.py')}))
    started=time.monotonic()
    def health(row):
        okay,reason=resources_ok(ROOT)
        if not okay or time.monotonic()-started>600:raise RuntimeError('Bounded diverse fit stopped:'+str(reason))
        atomic_json(output/'heartbeat.json',row)
    result,adam,sampler=helper.fit_actor(model,norm.normalize_obs(x),y,groups,train,holdout,seed=22,steps=8192,
        callback=health,steering_scale=.02,group_masses=masses,
        resume_optimizer=torch.load(prior/'auxiliary_Adam.pt',map_location='cpu',weights_only=True),
        resume_sampler=read(prior/'sampler_state.json'))
    if result['actual_Adam_step_each']!=[16384] or pickle.dumps(norm.obs_rms)!=rms:raise ValueError('Adam/RMS retention differs')
    destination=output/'checkpoint';shutil.copytree(origin,destination);model.save(destination/'model.zip')
    metadata.update(model_sha256=sha(destination/'model.zip'),additional_diverse_actor_steps=8192,
        retained_recovery_actor_steps=16384,cumulative_measured_actor_steps=17408,
        diverse_registration_sha256=sha(output/'registration.json'))
    atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
    torch.save(adam.state_dict(),output/'auxiliary_Adam.pt');atomic_json(output/'sampler_state.json',sampler)
    result.update(model_sha256=metadata['model_sha256'],new_controls=0,new_PPO_updates=0,new_auxiliary_Adam_steps=8192,
        cumulative_measured_actor_steps=17408,elapsed_seconds=time.monotonic()-started,research_acceptance='UNMET')
    atomic_json(output/'complete.json',result)
    print(json.dumps(dict(model_sha256=metadata['model_sha256'],new_auxiliary_Adam_steps=8192)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','views','reference-config','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--profile',default='diverse_geometry_v4',choices=['diverse_geometry_v4'])
    args=parser.parse_args();run(args.collection,args.views,args.reference_config,args.output,args.profile)
