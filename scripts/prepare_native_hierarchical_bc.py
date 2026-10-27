"""Matched teacher access for H/H0; supervised initialization is not PPO value."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import VehicleConfig
from navstack.env import NavigationEnv
from navstack.experiments import write_json,signature
from navstack.imitation import fit_categorical_initialization
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from navstack.road_candidate_policy import EndpointClearRoadPolicy,RoadCandidateVecNormalize
from scripts.audit_native_method_endpoint import read,read_prefix,sha
from scripts.audit_native_policy_sequence import observation_env,advance_observation
from scripts.audit_native_rule import audit
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config,learning_schedule,save_training_checkpoint
from scripts.prepare_native_hazard_learning import learning_protocol


def prepare(parent,seed=11,steps=896):
    parent=Path(parent).resolve()
    if type(seed) is not int or not 0<=seed<2**31 or not 1<=steps<=2000:
        raise ValueError('Invalid bounded categorical supervision')
    require_finished_parent(parent)
    if read(parent/'protocol.json').get('native_case_manifest') is not None:
        raise ValueError('Explicit mixed cases require per-frame dynamic hierarchical views; refusing four-case subset supervision')
    if audit(parent)!=read(parent/'raw_audit.json'):raise ValueError('Teacher raw audit changed')
    protocol=learning_protocol(parent,640,1800,seed,['H','H0'],endpoint_filter=True)
    vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    registration=parent/f'measured_hierarchical_bc_seed{seed}_registration.json'
    with ProcessLease(parent/'hierarchical_bc_prepare.lock'):
        if registration.exists():raise FileExistsError('Inspect the registered teaching phase instead of repeating')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_hierarchical_bc')
        batch.mkdir();write_json(registration,{'batch':batch.name,'status':'registered_before_fit'})
        write_json(batch/'fit_registration.json',{'scope':'Same actual B3teacher cases as continuous B1; all four cases and all macro decisions, including deadline. Extra supervision, zero PPO updates.',
            'seed':seed,'supervised_steps_per_method':steps,'teacher_batch':parent.name,
            'teacher_raw_audit_sha256':sha(parent/'raw_audit.json'),'fit_wall_seconds':900,
            'holdout':'Every fifth decision within each case; correlated fit only, no independent validation'})
        deadline=time.time()+900
        def health():
            healthy,reason=resources_ok(ROOT)
            if not healthy or shutil.disk_usage(ROOT).free<12*1024**3 or time.time()>=deadline:
                raise RuntimeError('Categorical teaching resource/deadline guard: '+str(reason))
        try:
            torch.set_num_threads(1)
            rows,error=read_prefix(parent/'evaluation_trace.jsonl.gz')
            if error:raise ValueError(error)
            arrays={};fits={};recovery={};case_counts=[]
            for case in range(4):
                selected=[r for r in rows if r['case_id']==case]
                case_counts.append({'case_id':case,'decisions':len(selected),
                    'controls':sum(len(r['info']['trace']) for r in selected),
                    'termination_reason':read(parent/f'case_{case:04d}.json')['termination_reason']})
            controls=sum(r['controls'] for r in case_counts)
            for method in protocol['methods']:
                health();config=training_config(protocol,method,vehicle)
                observations=[];actions=[];masks=[]
                for case in range(4):
                    selected=[r for r in rows if r['case_id']==case]
                    scene=read(parent/f'scene_{case:04d}.json')['scene']
                    env=observation_env(config,scene,selected[0]['info']['initial_scenario'])
                    env.world.road_collision_backend=config.road_collision_backend
                    for i,row in enumerate(selected):
                        if i%32==0:health()
                        observations.append(env._observation());actions.append(int(row['policy_action']));masks.append(i%5!=0)
                        advance_observation(env,row)
                    env.close()
                observations=np.asarray(observations,dtype=np.float32)
                actions=np.asarray(actions,dtype=np.int64);masks=np.asarray(masks,dtype=bool)
                arrays[method+'_observations']=observations;arrays['actions']=actions;arrays['train_mask']=masks
                raw=DummyVecEnv([lambda:NavigationEnv(config)]);raw.envs[0].case_index=0
                norm=RoadCandidateVecNormalize(raw,norm_reward=False,gamma=config.gamma)
                model=PPO(EndpointClearRoadPolicy,norm,**learning_schedule(protocol,config,32),seed=seed,device='cpu')
                def progress(row):
                    health();write_json(batch/(method+'_fit_heartbeat.json'),{'epoch':time.time(),**row})
                    print(json.dumps({'method':method,**row}),flush=True)
                fits[method]=fit_categorical_initialization(model,norm,observations,actions,masks,steps,progress)
                output=batch/'initialization'/method;output.mkdir(parents=True)
                write_json(output/'config.json',{'env':config.to_dict(),'signature':signature(config),'seed':seed,
                    'backend':'measured_CARLA_teacher_supervised_initialization','scope':'Categorical actor-only teaching, zero PPO updates'})
                metadata={'physics_steps':0,'actions':0,'episodes':0,'optimization_phases':0,'optimizer_steps':0,
                    'world_ticks':0,'lineage_control_steps':0,'charged_control_steps_lower_bound':controls,'supervised_optimizer_steps':steps}
                save_training_checkpoint(output,0,model,norm,raw,metadata)
                shutil.copytree(output/read(output/'latest_checkpoint.json')['folder'],batch/'resume'/method)
                norm.close()
                recovery[method]={'checkpoint':'resume/'+method,'parent_batch':parent.name,
                    'parent_checkpoint':'teacher_supervised_initialization','lineage_controls':0,
                    'consumed_controls_lower_bound':controls,'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False}
            np.savez_compressed(batch/'measured_demonstrations.npz',**arrays)
            (batch/'infrastructure').mkdir();shutil.copytree(parent/'calibration',batch/'calibration')
            for folder in ('navstack','scripts','tests'):
                for source in (ROOT/folder).rglob('*'):
                    if source.is_file() and '__pycache__' not in source.parts and (source.suffix in ('.py','.ps1') or 'fixtures' in source.parts):
                        target=batch/'source'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            now=time.time()
            protocol.update(evaluation_only=True,controlled_budget_per_method=0,method_budgets={'H':0,'H0':0},
                created_epoch=now,deadline_epoch=now+1800,record_initial_parameters=False,
                calibration_sha256=sha(batch/'calibration/calibration.json'),parent_batch=parent.name,recovery=recovery,
                scope='Frozen H/H0 teacher-supervised representation comparison; zero PPO updates, same physical teacher data as B1, not PPO value or independent confirmation',
                extra_supervision={'teacher_batch':parent.name,'teacher_controls':controls,'cases':case_counts,
                    'dataset_sha256':sha(batch/'measured_demonstrations.npz'),'supervised_optimizer_steps_per_method':steps,
                    'methods':fits,'macro_labels_per_method':len(arrays['actions']),
                    'fairness':'Same physical teacher cases; macro versus direct labels and different losses/compute are separately disclosed. No ordinary-PPO or architecture superiority claim.'})
            protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            write_json(batch/'protocol.json',protocol);write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            print(batch,flush=True);return batch
        except BaseException as exc:
            write_json(batch/'fit_failure.json',{'error':repr(exc),'epoch':time.time()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True)
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--steps',type=int,default=896)
    args=parser.parse_args();prepare(args.parent,args.seed,args.steps)
