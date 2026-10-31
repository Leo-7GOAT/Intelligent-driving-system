"""All-row TRAIN fit diagnostics; no validation labels, driving or updates."""
from pathlib import Path
import argparse,json,pickle,sys,time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import torch
from navstack.independent_clip_ppo import load_ppo
from scripts.audit_native_method_endpoint import read,sha
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,help='Fresh JSON output;requires the archived local training/actor inputs')
args=parser.parse_args()
replay=root/'runs/20261031T123039922770Z_native_replay_eval_configuration_recovery'
measured=root/'runs/20261031T094314260006Z_native_measured36_training_view_bc'
registration=read(replay/'protocol.json')['measured_training_view_initialization']
origin=root/'runs'/registration['parent']/'resume/B1'
with np.load(replay/'measured_demonstrations.npz',allow_pickle=False) as data:
    old={k:data[k].copy() for k in ('observations','actions','weights')}
with np.load(measured/'measured_demonstrations.npz',allow_pickle=False) as data:
    full={k:data[k].copy() for k in ('observations','actions','weights')}
count=len(old['actions'])
assert count==3496 and len(full['actions'])==7652
for key in old:np.testing.assert_array_equal(old[key],full[key][:count])
initial=None;results=[];started=time.monotonic()
for label,folder in [('before_additional_fit',origin),('replay_512',replay/'resume/B1'),('measured36_512',measured/'resume/B1')]:
    with (folder/'normalization.pkl').open('rb') as stream:norm=pickle.load(stream)
    norm.training=False
    statistics=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),float(norm.obs_rms.count))
    model=load_ppo(folder/'model.zip',device='cpu');model.policy.set_training_mode(False)
    assert model.num_timesteps==model._n_updates==0 and not model.policy.optimizer.state
    critic={n:p.detach().clone() for n,p in model.policy.named_parameters() if 'value_net' in n}
    assert critic
    if initial is None:initial=critic
    else:assert all(torch.equal(p,initial[n]) for n,p in critic.items())
    blocks={}
    for part,lo,hi in [('original_TRAIN',0,count),('additional_measured36_TRAIN',count,len(full['actions']))]:
        predictions=[]
        for begin in range(lo,hi,128):
            if time.monotonic()-started>180:raise TimeoutError('Bounded read-only fit diagnosis expired')
            features=torch.as_tensor(norm.normalize_obs(full['observations'][begin:min(begin+128,hi)]))
            with torch.no_grad():prediction=model.policy.get_distribution(features).distribution.mean.cpu().numpy()
            predictions.append(prediction)
        predictions=np.concatenate(predictions);labels=full['actions'][lo:hi];error=(predictions-labels)**2
        blocks[part]={'rows':hi-lo,'unclipped_mean_squared_error_each_action':error.mean(axis=0).tolist(),
            'clipped_deployment_mean_squared_error_each_action':((np.clip(predictions,-1,1)-labels)**2).mean(axis=0).tolist(),
            'weighted_unclipped_action_MSE':float(np.average(error.mean(axis=1),weights=full['weights'][lo:hi])),
            'target_action_mean':labels.mean(axis=0).tolist(),'target_action_std':labels.std(axis=0).tolist(),
            'actor_mean':predictions.mean(axis=0).tolist()}
    np.testing.assert_array_equal(statistics[0],norm.obs_rms.mean);np.testing.assert_array_equal(statistics[1],norm.obs_rms.var)
    assert statistics[2]==norm.obs_rms.count
    results.append({'arm':label,'model_sha256':sha(folder/'model.zip'),'normalization_sha256':sha(folder/'normalization.pkl'),'blocks':blocks})
assert len({r['normalization_sha256'] for r in results})==1
reports=[read(root/'reports'/('native_B1_'+label+'_training_view_validation_20261031.json')) for label in ('replay','measured36')]
assert all(r['model_sha256']==m['model_sha256'] for r,m in zip(reports,results[1:]))
report={'scope':__doc__,'script_sha256':sha(Path(__file__)),'dataset_sha256':sha(measured/'measured_demonstrations.npz'),
    'models':results,'complete_validation_results':[{k:r[k] for k in ('batch','cases','successes','termination_counts','raw_controls_verified','model_sha256')} for r in reports],
    'critic_unchanged':True,'normalization_unchanged':True,'new_controls':0,'PPO_updates':0,'supervised_updates':0,
    'limits':'TRAIN fit only,not holdout accuracy. Added-data and replay actors each have512additional fits but different data distributions. Full validation is one development seed. A fit/driving association is not a causal proof of failure mechanism. Extra teacher controls and failed training episodes are retained in source manifests.','research_acceptance':'UNMET'}
output=Path(args.output)
with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
print(json.dumps(report),flush=True)
