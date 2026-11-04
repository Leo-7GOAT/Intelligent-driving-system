"""One bounded real-PPO basic-road pilot changing only initial exploration std."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def equal(left,right):
    import torch
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(equal(a,b) for a,b in zip(left,right))
    return left==right


def set_exploration(model,std=.1):
    import math
    import torch
    if (type(std) is not float or std!=.1 or model.num_timesteps<=0 or model._n_updates<=0
            or not model.policy.optimizer.state or tuple(model.policy.log_std.shape)!=(2,)
            or not torch.isfinite(model.policy.log_std).all()):
        raise ValueError('Require the locked .10std intervention on a trained two-axis Gaussian policy')
    parameters={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    optimizer=deepcopy(model.policy.optimizer.state_dict());counters=(model.num_timesteps,model._n_updates)
    old=model.policy.log_std.detach().exp().cpu().tolist()
    with torch.no_grad():model.policy.log_std.fill_(math.log(std))
    if (any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in parameters.items() if k!='log_std')
            or not equal(optimizer,model.policy.optimizer.state_dict())
            or counters!=(model.num_timesteps,model._n_updates)):
        raise ValueError('Exploration intervention changed other weights,optimizer or PPO counters')
    return {'version':'initial_Gaussian_std_010_v1','old_std':old,'new_std':model.policy.log_std.detach().exp().cpu().tolist(),
        'changed_parameter':'log_std','changed_scalar_parameters':2,'other_actor_and_critic_weights_exact':True,
        'PPO_optimizer_moments_exact':True,'PPO_counters_exact':True,'optimizer_updates':0,
        'deterministic_mean_network_unchanged':True,
        'scope':'Explicit initial stochastic exploration intervention before a fresh actual PPOrollout,not a PPOupdate or deterministic-policy performance improvement.'}


def prepare():
    if any(name.startswith('navstack.') for name in sys.modules):raise RuntimeError('Use a fresh interpreter for the frozen runtime')
    parent=ROOT/'runs/20261101T011350364179Z_native_B1_basic_lr_extension'
    sys.path.insert(0,str(parent/'source'))
    from navstack.independent_clip_ppo import load_ppo
    from navstack.research import resources_ok,atomic_json
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.prepare_native_recovery import verified_checkpoint_metadata
    pointer=ROOT/'runs/native_B1_exploration_PPO_registration_20261101.json'
    if pointer.exists():raise FileExistsError('Preserve the registered exploration pilot')
    require_finished_parent(parent)
    old=read(parent/'protocol.json');summary=read(parent/'B1/summary.json');raw=read(parent/'B1/endpoint_raw_audit.json')
    origin=ROOT/'runs/B1_seed22_matched_trained_trajectory_fit_20261101/replay'
    metadata=verified_checkpoint_metadata(origin)
    if (metadata['model_sha256']!='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0'
            or metadata['normalization_sha256']!='db467fb35b0f78a989e8e042b0400036cada30cfad36b41fba727a4f432e5df5'
            or metadata['lineage_control_steps']!=20480 or old['methods']!=['B1']):
        raise ValueError('Use the fixed replay reference,not a selected recovery-fit endpoint')
    reports=['native_B1_executed_recovery_collection_and_probe_20261101.json',
        'native_B1_original_TRAIN_recovery_collection_and_probe_20261101.json',
        'native_B1_original_TRAIN_unprojected_probe_20261101.json']
    outcomes=[read(ROOT/'reports'/name) for name in reports]
    if any(r['pure_policy_probe']['basic_gate_passed'] for r in outcomes):
        raise ValueError('This pilot is registered only after the completed rejected recovery hypotheses')
    spent=outcomes[-1]['all_recovery_profiles_controls']
    if spent!=9010:raise ValueError('Require complete actual cost of all recovery profiles')
    for name,digest in old['source_hashes'].items():
        if sha(parent/'source'/name)!=digest:raise ValueError('Parent frozen source changed')
    okay,reason=resources_ok(ROOT)
    if not okay:raise RuntimeError(str(reason))
    import torch
    torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu')
    intervention=set_exploration(model)
    batch=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_native_B1_exploration_PPO')
    batch.mkdir();(batch/'infrastructure').mkdir()
    shutil.copytree(parent/'source',batch/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent/'calibration',batch/'calibration');shutil.copytree(origin,batch/'resume/B1')
    shutil.copy2(parent/'measured_demonstrations.npz',batch/'measured_demonstrations.npz')
    shutil.copy2(__file__,batch/'source/scripts/prepare_native_exploration_ppo.py')
    destination=batch/'resume/B1';model.save(destination/'model.zip')
    metadata.update(model_sha256=sha(destination/'model.zip'),exploration_intervention=intervention,
        note='Actual trained replay parent;only the2initial Gaussian log_std scalars changed. Optimizer,counters,mean-network,critic,RMS and saved RNG retained. New physical rollout required.')
    atomic_json(destination/'metadata.json',metadata);verified_checkpoint_metadata(destination)
    p=deepcopy(old)
    for name in ('jobs','own_success_replay','native_validation','native_camera_evaluation','native_recovery_collection'):
        p.pop(name,None)
    now=time.time();p.update(scope='Real PPO basic-motion exploration pilot,one known development seed;not a healthy five-seed baseline or final confirmation.',
        evaluation_only=False,controlled_budget_per_method=10240,method_budgets={'B1':10240},
        checkpoint_thresholds=list(range(1280,10241,1280)),created_epoch=now,deadline_epoch=now+7200,
        deterministic_training_route_evaluation=True,freeze_observation_statistics=True,
        resource_limits={**old['resource_limits'],'wall_seconds':7200,'automatic_retries':0},
        native_exploration_PPO={'version':'retained_mean_std010_PPO10240_v1','intervention':intervention,
            'source_model_sha256':sha(origin/'model.zip'),'intervened_model_sha256':metadata['model_sha256'],
            'normalization_sha256':metadata['normalization_sha256'],'new_supervised_updates':0,
            'new_PPO_controls':10240,'initial_deterministic_replay_not_repeated':True,
            'initial_mean_evidence':'reports/native_B1_matched_trained_actor_correction_20261101.json',
            'prior_recovery_controls_charged':spent,'prior_recovery_results':reports,
            'selection':'One final10240control endpoint;all4original TRAINroad outcomes retained. Same physical/reward/normalization/rate/rollout settings;only initial Gaussian std changes.',
            'basic_gate':'At least3/4successANDzero collision/bounds;unchanged. No full-task entry until all5independent courses qualify.'},
        recovery={'B1':{**old['recovery']['B1'],'checkpoint':'resume/B1','parent_batch':parent.name,
            'parent_checkpoint':'B1/'+read(parent/'B1/latest_checkpoint.json')['folder'],
            'lineage_controls':summary['lineage_control_steps'],
            'consumed_controls_lower_bound':summary['charged_control_steps_lower_bound']+raw['evaluation_control_frames_verified']+spent,
            'retained_optimizer_steps':old['recovery']['B1']['retained_optimizer_steps']+summary['optimizer_steps']}})
    p['extra_supervision']['supervised_optimizer_steps']+=512
    p['extra_supervision']['inherited_replay_actor_fit']={'steps':512,'new_this_pilot':0,'model_sha256':sha(origin/'model.zip')}
    p['source_hashes']['scripts/prepare_native_exploration_ppo.py']=sha(batch/'source/scripts/prepare_native_exploration_ppo.py')
    atomic_json(batch/'exploration_intervention.json',{'source_model_sha256':sha(origin/'model.zip'),
        'intervened_model_sha256':metadata['model_sha256'],**intervention})
    atomic_json(batch/'protocol.json',p);atomic_json(batch/'prepared.json',{'status':'PREPARED_NOT_STARTED','protocol_sha256':sha(batch/'protocol.json')})
    atomic_json(pointer,{'batch':batch.name,'protocol_sha256':sha(batch/'protocol.json'),'new_controls_so_far':0})
    return batch


def analyze(batch):
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    batch=Path(batch).resolve();require_finished_parent(batch);folder=batch/'B1'
    p=read(batch/'protocol.json');spec=p['native_exploration_PPO'];s=read(folder/'summary.json')
    r=read(folder/'endpoint_raw_audit.json');e=read(folder/'evaluation.json');i=read(folder/'measured_start_inference.json')
    if (s['physics_steps']!=10240 or s['optimizer_steps']<=0 or s['lineage_control_steps']!=30720
            or r['training_control_frames_verified']!=10240 or r['segment_optimizer_steps']!=s['optimizer_steps']
            or len(i['results'])!=4 or i['model_sha256']!=e['model_sha256']
            or r['protocol_sha256']!=sha(batch/'protocol.json') or r['evaluation_sha256']!=sha(folder/'evaluation.json')
            or sorted(x['case_id'] for x in e['episodes'])!=[0,1,2,3]):raise ValueError('Incomplete actual PPOpilot or endpoint audit')
    unsafe=sum(x['termination_reason'] in ('collision','out_of_bounds') for x in e['episodes'])
    success=sum(x['success'] for x in e['episodes'])
    return {'scope':__doc__,'batch':batch.name,'seed':22,'protocol_sha256':sha(batch/'protocol.json'),
        'intervention':spec['intervention'],'actual_new_PPO_controls':s['physics_steps'],'actual_new_PPO_Adam_updates':s['optimizer_steps'],
        'new_supervised_updates':0,'PPO_lineage_controls':s['lineage_control_steps'],
        'model_sha256':e['model_sha256'],'normalization_sha256':e['normalization_sha256'],
        'training_and_eval_controls_verified':r['total_applied_controls_verified'],
        'endpoint_successes':success,'unsafe_cases':unsafe,'basic_gate_passed':success>=3 and unsafe==0,
        'case_results':[{'case_id':x['case_id'],'termination':x['termination_reason'],'controls':x['physics_steps']} for x in e['episodes']],
        'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json'),'new_final_conditions':0,
        'research_acceptance':'UNMET;one development pilot,not qualified full-task five-seed evidence.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analyze');p.add_argument('--output');a=p.parse_args()
    if a.analyze:
        if not a.output:p.error('--analyze requires --output')
        with Path(a.output).open('x',encoding='utf-8') as stream:json.dump(analyze(a.analyze),stream,indent=2)
    else:print(prepare(),flush=True)
