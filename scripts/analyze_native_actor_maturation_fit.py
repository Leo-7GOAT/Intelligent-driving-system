"""TRAIN-only all-five before/after actor-maturation fit diagnostic;no fitting/driving."""
import argparse,json,pickle
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from navstack.independent_clip_ppo import load_ppo,actor_critic_parameters
from scripts.audit_native_method_endpoint import read,sha

def run(plan,preparation,output):
 plan,preparation,output=map(Path,(plan,preparation,output));p=read(plan);state=read(preparation/'status.json')
 if [r['seed'] for r in p['parents']]!=[11,22,33,44,55] or [r['seed'] for r in state['batches']]!=[11,22,33,44,55]:raise ValueError('All five fixed seeds required')
 if output.exists():raise FileExistsError('Preserve diagnostic')
 torch.set_num_threads(1);results=[];hashes=set()
 for parent,child in zip(p['parents'],state['batches']):
  before=ROOT/'runs'/parent['batch'];after=ROOT/'runs'/child['batch'];dataset=after/'measured_demonstrations.npz'
  if sha(dataset)!=sha(before/'measured_demonstrations.npz'):raise ValueError('TRAIN input changed')
  hashes.add(sha(dataset))
  with np.load(dataset,allow_pickle=False) as d:
   if not d['train_mask'].all():raise ValueError('TRAIN-only input required')
   observations,labels,weights=[d[k].copy() for k in ('observations','actions','weights')]
  models=[];critics=[]
  for version,batch in (('before',before),('matured',after)):
   folder=batch/'resume/B1';model=load_ppo(folder/'model.zip',device='cpu')
   if model.num_timesteps or model._n_updates or model.policy.optimizer.state:raise ValueError('Zero PPOactor checkpoints only')
   with (folder/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
   moments=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
   predictions=[]
   with torch.no_grad():
    model.policy.set_training_mode(False)
    for start in range(0,len(labels),128):
     obs=torch.as_tensor(norm.normalize_obs(observations[start:start+128]),device=model.device)
     predictions.append(model.policy.get_distribution(obs).distribution.mean.cpu().numpy())
   prediction=np.concatenate(predictions);error=prediction-labels
   blocks={}
   for name,mask in (('all',np.ones(len(labels),bool)),('original',weights==1),('corrective',weights>1)):
    blocks[name]={'rows':int(mask.sum()),'coordinate_mse':np.square(error[mask]).mean(0).tolist(),
     'coordinate_absolute_error_p95':np.quantile(np.abs(error[mask]),.95,axis=0).tolist(),
     'target_std':labels[mask].std(0).tolist(),
     'weighted_coordinate_mse':np.average(np.square(error[mask]),axis=0,weights=weights[mask]).tolist()}
   original_mask=weights==1
   for name,mask in (('moving_original',(observations[:,0]>.5)&original_mask),('slow_original',(observations[:,0]<=.5)&original_mask)):
    blocks[name]={'rows':int(mask.sum()),'coordinate_mse':np.square(error[mask]).mean(0).tolist()}
   stats_unchanged=(np.array_equal(moments[0],norm.obs_rms.mean) and np.array_equal(moments[1],norm.obs_rms.var) and moments[2]==norm.obs_rms.count)
   assert stats_unchanged
   critics.append([v.detach().clone() for v in actor_critic_parameters(model.policy)[1]])
   models.append({'version':version,'batch':batch.name,'model_sha256':sha(folder/'model.zip'),
    'normalization_sha256':sha(folder/'normalization.pkl'),'blocks':blocks})
  assert all(torch.equal(a,b) for a,b in zip(*critics))
  assert models[0]['normalization_sha256']==models[1]['normalization_sha256']
  results.append({'seed':parent['seed'],'models':models,'critic_exact':True})
  print(json.dumps({'seed':parent['seed'],'all_mse_before':models[0]['blocks']['all']['coordinate_mse'],'all_mse_after':models[1]['blocks']['all']['coordinate_mse']}),flush=True)
 assert len(hashes)==1
 result={'scope':__doc__,'script_sha256':sha(Path(__file__)),'plan_sha256':sha(plan),
  'TRAIN_dataset_sha256':next(iter(hashes)),'results':results,'new_controls':0,'PPO_updates':0,'supervised_updates':0,
  'limits':'Training reconstruction error is not holdout accuracy or a causal closed-loop failure explanation. No evaluation observations or labels enter these calculations;all five prepared actors included. More actor fits are not justified merely by this diagnostic.','research_acceptance':'UNMET'}
 with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--preparation',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.plan,a.preparation,a.output)
