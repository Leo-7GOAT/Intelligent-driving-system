"""New feedback-learning control with a common physical endpoint mask.

Reuse the completed H actor-only initialization; fit only the newly defined H0.
No completed H fit is repeated and neither method performs PPO here.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
import pickle
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from navstack.config import EnvConfig,VehicleConfig
from navstack.env import NavigationEnv
from navstack.experiments import write_json,signature
from navstack.imitation import fit_categorical_initialization
from navstack.independent_clip_ppo import load_ppo
from navstack.process_lease import ProcessLease
from navstack.research import resources_ok
from navstack.road_candidate_policy import RoadCandidateVecNormalize,endpoint_policy_class
from scripts.audit_native_method_endpoint import audit,read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.native_road_learning import training_config,learning_schedule,save_training_checkpoint
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.audit_native_rule import audit as rule_audit


def prepare(parent):
    parent=Path(parent).resolve();old=read(parent/'protocol.json')
    if (old.get('methods')!=['H','H0'] or not old.get('evaluation_only') or old.get('shared_endpoint_mask_v1')
            or old.get('candidate_head')!='unshared_endpoint_v1' or not old.get('extra_supervision')):
        raise ValueError('Require the completed original actor-only endpoint comparison')
    selected_evaluation_controls=0
    if (parent/'guard_final.json').exists():
        require_finished_parent(parent);verified=audit(parent,'H')
        if verified!=read(parent/'H/endpoint_raw_audit.json'):raise ValueError('Reused H raw audit changed')
        selected_evaluation_controls=verified['evaluation_control_frames_verified']
    else:
        prepared=read(parent/'prepared.json')
        if (prepared.get('status')!='PREPARED_NOT_STARTED' or prepared.get('protocol_sha256')!=sha(parent/'protocol.json')
                or any((parent/name).exists() for name in ('started.json','worker_process.json','guard.json','launcher_execution.json','fit_failure.json','failure.json'))):
            raise ValueError('Unlaunched initialization must be fully prepared and have no active/failed lifecycle')
        for name,digest in old['source_hashes'].items():
            if sha(parent/'source'/name)!=digest:raise ValueError('Prepared source changed')
        teacher_name=old['extra_supervision']['teacher_batch']
        if Path(teacher_name).name!=teacher_name:raise ValueError('Invalid teacher identity')
        teacher=parent.parent/teacher_name;verified_teacher=rule_audit(teacher)
        if (verified_teacher!=read(teacher/'raw_audit.json')
                or read(parent/'fit_registration.json')['teacher_raw_audit_sha256']!=sha(teacher/'raw_audit.json')):
            raise ValueError('Prepared actor fit lacks the verified measured teacher')
    origin=parent/'resume/H';metadata=verified_checkpoint_metadata(origin)
    if metadata['physics_steps'] or metadata['optimizer_steps']:raise ValueError('Do not reuse a PPO-trained H as a fresh initializer')
    dataset=parent/'measured_demonstrations.npz'
    if sha(dataset)!=old['extra_supervision']['dataset_sha256']:raise ValueError('Registered measured teacher views changed')
    steps=old['extra_supervision']['supervised_optimizer_steps_per_method']
    if steps!=metadata['supervised_optimizer_steps'] or not 1<=steps<=2000:raise ValueError('Unmatched actor-supervision budget')
    arrays=np.load(dataset,allow_pickle=False);observations=arrays['H_observations'];actions=arrays['actions'];mask=arrays['train_mask']
    protocol=deepcopy(old);protocol['shared_endpoint_mask_v1']=True
    vehicle=VehicleConfig(**read(parent/'calibration/calibration.json')['vehicle'])
    config=training_config(protocol,'H0',vehicle)
    if (config.to_dict()!=training_config(protocol,'H',vehicle).to_dict()
            or config.to_dict()!=read(origin/'config.json')['env']):raise ValueError('Physical task or common H information changed')
    def health():
        okay,reason=resources_ok(ROOT)
        if not okay or shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError(str(reason))
    health()
    with ProcessLease(parent/'shared_mask_prepare.lock'):
        registration=parent/'shared_endpoint_mask_bc_registration.json'
        if registration.exists():raise FileExistsError('Preserve the registered shared-mask control')
        batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_native_shared_mask_bc');batch.mkdir()
        write_json(registration,{'batch':batch.name,'status':'REGISTERED_BEFORE_FIT'})
        write_json(batch/'fit_registration.json',{'version':'shared_endpoint_mask_v1','seed':old['seed'],
            'wall_seconds':900,'new_supervised_optimizer_steps':{'H':0,'H0':steps},
            'lineage_supervised_optimizer_steps':{'H':steps,'H0':steps},
            'scope':'H actor-only checkpoint reused byte-exact;only H0new internal-feedback ablation fitted. Same physical information/mask/data/limits/labels. No PPO or driving.'})
        deadline=time.time()+900
        try:
            torch.set_num_threads(1)
            raw=DummyVecEnv([lambda:NavigationEnv(config)]);raw.envs[0].case_index=0
            norm=RoadCandidateVecNormalize(raw,norm_reward=False,gamma=config.gamma)
            model=PPO(endpoint_policy_class(protocol,'H0'),norm,**learning_schedule(protocol,config,32),seed=old['seed'],device='cpu')
            h=load_ppo(origin/'model.zip',device='cpu')
            if h.num_timesteps or h._n_updates or h.policy.optimizer.state:raise ValueError('Reused H is not actor-only initialization')
            def progress(row):
                health()
                if time.time()>=deadline:raise TimeoutError('Declared actor-fit wall cap exhausted')
                write_json(batch/'H0_fit_heartbeat.json',{'epoch':time.time(),**row});print(row,flush=True)
            history=fit_categorical_initialization(model,norm,observations,actions,mask,steps,progress)
            for name,value in h.policy.state_dict().items():
                if 'value_net' in name and not torch.equal(value,model.policy.state_dict()[name]):raise ValueError('Paired critic initialization differs')
            with (origin/'normalization.pkl').open('rb') as stream:hnorm=pickle.load(stream)
            np.testing.assert_array_equal(hnorm.obs_rms.mean,norm.obs_rms.mean);np.testing.assert_array_equal(hnorm.obs_rms.var,norm.obs_rms.var)
            if hnorm.obs_rms.count!=norm.obs_rms.count:raise ValueError('Paired training-only normalization count differs')
            destination=batch/'initialization/H0';destination.mkdir(parents=True)
            write_json(destination/'config.json',{'env':config.to_dict(),'signature':signature(config),'seed':old['seed'],
                'scope':'Shared physical endpoint mask;H0actor/critic internally receive zero new feedback features'})
            meta={'physics_steps':0,'actions':0,'episodes':0,'optimization_phases':0,'optimizer_steps':0,'world_ticks':0,
                'lineage_control_steps':0,'charged_control_steps_lower_bound':old['extra_supervision']['teacher_controls'],'supervised_optimizer_steps':steps}
            save_training_checkpoint(destination,0,model,norm,raw,meta)
            shutil.copytree(origin,batch/'resume/H');shutil.copytree(destination/read(destination/'latest_checkpoint.json')['folder'],batch/'resume/H0');norm.close()
            for filename in ('model.zip','normalization.pkl','resume_state.pkl','config.json'):
                if sha(origin/filename)!=sha(batch/'resume/H'/filename):raise ValueError('Reused H checkpoint changed')
            np.savez_compressed(batch/'measured_demonstrations.npz',H_observations=observations,H0_observations=observations,actions=actions,train_mask=mask)
            (batch/'infrastructure').mkdir();shutil.copytree(parent/'calibration',batch/'calibration')
            for folder in ('navstack','scripts','tests'):
                for file in (ROOT/folder).rglob('*'):
                    if file.is_file() and '__pycache__' not in file.parts and (file.suffix in ('.py','.ps1') or 'fixtures' in file.parts):
                        target=batch/'source'/file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
            now=time.time();controls=old['extra_supervision']['teacher_controls']
            protocol.update(parent_batch=parent.name,created_epoch=now,deadline_epoch=now+1800,
                scope='New actor-only H/H0initialization control:common physical endpoint mask;only H0learned feedback ablated. Known12conditions,zeroPPO,not formal feedback benefit.',
                shared_mask_ablation={'version':'shared_endpoint_mask_v1','H_checkpoint_reused':True,'H_source_model_sha256':metadata['model_sha256'],
                    'new_supervised_updates':{'H':0,'H0':steps},'network_feedback_H0':'candidate columns5:8andoutcomes599:605zero inside BOTHactor andcritic;physicalmask from unmodified observed endpoint-clear bits',
                    'normalization_statistics_match':True,'critic_parameters_match':True,
                    'comparison_limit':'Developed after observing earlier12case/seed11results;not unseen confirmation or ordinary from-scratch PPO.'},
                extra_supervision={**old['extra_supervision'],'dataset_sha256':sha(batch/'measured_demonstrations.npz'),
                    'methods':{'H':old['extra_supervision']['methods']['H'],'H0':history},
                    'scope':'Both methods have896lineage actor-supervised updates;H reused,onlyH0new. Common teacher raw observations and labels;only learned feedback access differs.'},
                recovery={m:{'checkpoint':'resume/'+m,'parent_batch':parent.name,'parent_checkpoint':'actor_only_initialization',
                    'lineage_controls':0,'consumed_controls_lower_bound':controls+(selected_evaluation_controls if m=='H' else 0),
                    'retained_optimizer_steps':0,'unknown_unflushed_parent_tail':False} for m in ('H','H0')})
            protocol['source_hashes']={p.relative_to(batch/'source').as_posix():sha(p) for p in (batch/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
            write_json(batch/'protocol.json',protocol);write_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
            print(batch,flush=True);return batch
        except BaseException as exc:
            write_json(batch/'fit_failure.json',{'error':repr(exc),'epoch':time.time()});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',required=True);a=parser.parse_args();prepare(a.parent)
