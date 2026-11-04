"""Numerically reconstruct a real first TRAIN rollout and compare bounded PPO rates offline."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def scalar_json(value):
    if hasattr(value,'item'):return value.item()
    raise TypeError(type(value).__name__)


def run(batch,output):
    batch,output=Path(batch).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve earlier offline diagnostic')
    output.mkdir(parents=True);shutil.copy2(__file__,output/'executed_replay.py')
    sys.path.insert(0,str(batch/'source'))
    import numpy as np
    import torch
    from navstack.config import EnvConfig
    from navstack.independent_clip_ppo import load_ppo
    from navstack.measured_demo import measured_flat_env,advance_measured_flat,measured_label
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.distributions import kl_divergence
    folder=batch/'B1';origin=batch/'resume/B1';audit=read(folder/'endpoint_raw_audit.json');protocol=read(batch/'protocol.json')
    started=time.monotonic();env=None
    try:
        if digest(batch/'protocol.json')!=audit['protocol_sha256'] or not protocol['freeze_observation_statistics']:
            raise ValueError('Require the original raw-audited frozen-statistics course')
        for name,sha in protocol['source_hashes'].items():
            if digest(batch/'source'/name)!=sha:raise ValueError('Frozen source changed')
        begins={};bridge=batch/'infrastructure'/audit['bridge_response_stream']
        if digest(bridge)!=audit['bridge_response_sha256']:raise ValueError('Measured bridge changed')
        with gzip.open(bridge,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)['response']
                if 'reset_ticks' in r:begins[r['episode_id']]=r
        rows=[]
        for name,sha in audit['transition_hashes'].items():
            path=folder/name
            if digest(path)!=sha:raise ValueError('TRAIN trace changed')
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                for line in stream:
                    if len(rows)==1280:break
                    rows.append(json.loads(line))
        if len(rows)!=1280:raise ValueError('Complete first1280TRAIN controls required')
        cfg=EnvConfig.from_dict(read(origin/'config.json')['env'])
        def create(row):
            begin=begins[row['trace'][0]['backend_evidence']['episode_id']];scene=begin['scene']
            initial={'name':scene['name'],'initial':begin['state'],'goal':scene['goal'],'static':scene['static'],
                'dynamic':begin['dynamic'],'road_sha256':scene['road_grid']['sha256'],
                'actuator_evidence':{k:begin[k] for k in ('actuator_observation','speed_setpoint','actuation_state')}}
            return measured_flat_env(cfg,scene,initial,begin=begin)
        torch.set_num_threads(1);model=load_ppo(origin/'model.zip',device='cpu')
        if model.num_timesteps or model._n_updates or model.policy.optimizer.state or model.use_sde or model.n_steps!=1280:
            raise ValueError('Require the first untouched Gaussian PPOcourse rollout')
        model.policy.set_training_mode(False);model.rollout_buffer.reset()
        with (origin/'normalization.pkl').open('rb') as f:norm=pickle.load(f)
        with (origin/'resume_state.pkl').open('rb') as f:state=pickle.load(f)
        env=create(rows[0]);episode=rows[0]['episode_id'];episode_start=True;maximum=0.
        random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
        for i,row in enumerate(rows):
            if time.monotonic()-started>600:raise TimeoutError('Ten-minute numerical reconstruction budget')
            if row['episode_id']!=episode:
                env.close();env=create(row);episode=row['episode_id'];episode_start=True
            if len(row['trace'])!=1:raise ValueError('Direct hold1measured controls required')
            obs=norm.normalize_obs(env._observation()[None]);tensor=torch.as_tensor(obs)
            recorded=np.asarray(row['policy_action'],dtype=np.float32)[None]
            with torch.no_grad():
                distribution=model.policy.get_distribution(tensor)
                sampled=distribution.get_actions(deterministic=False)
                values=model.policy.predict_values(tensor)
                logp=distribution.log_prob(torch.as_tensor(recorded))
            error=float(np.abs(sampled.numpy()-recorded).max());maximum=max(maximum,error)
            if error>1e-6:raise ValueError('Measured stochastic action reconstruction exceeds1e-6')
            tick=row['trace'][0];rewards=norm.normalize_reward(np.asarray([tick['reward']],dtype=np.float32))
            # Train on the originally recorded UNCLIPPED Gaussian action,never a
            # reconstructed surrogate or the bounded vehicle request.
            model.rollout_buffer.add(obs,recorded,rewards,np.array([episode_start]),values,logp)
            advance_measured_flat(env,tick,measured_label(env,tick));episode_start=row['termination_reason'] is not None
            if (i+1)%128==0:print(json.dumps({'TRAIN_controls_reconstructed':i+1,'max_action_error':maximum,'new_controls':0}),flush=True)
        if episode_start:raise ValueError('This registered first rollout requires nonterminal bootstrap')
        with torch.no_grad():last_values=model.policy.predict_values(torch.as_tensor(norm.normalize_obs(env._observation()[None])))
        model.rollout_buffer.compute_returns_and_advantage(last_values=last_values,dones=np.array([False]))
        buffer=copy.deepcopy(model.rollout_buffer);np_state=np.random.get_state();torch_state=torch.get_rng_state()
        np.savez_compressed(output/'recorded_rollout.npz',**{name:getattr(buffer,name) for name in
            ('observations','actions','rewards','episode_starts','values','log_probs','advantages','returns')})
        with (output/'before_update_rng.pkl').open('xb') as f:pickle.dump({'numpy':np_state,'torch':torch_state},f)
        reference=load_ppo(folder/'checkpoint_000001280/model.zip',device='cpu');results=[]
        for rate in (3e-5,3e-6):
            candidate=load_ppo(origin/'model.zip',device='cpu');candidate.rollout_buffer=copy.deepcopy(buffer)
            candidate.learning_rate=rate;candidate._setup_lr_schedule();candidate._current_progress_remaining=0.
            candidate.policy.optimizer.defaults['lr']=rate
            for group in candidate.policy.optimizer.param_groups:group['lr']=rate
            candidate.set_logger(configure(str(output/('rate_'+str(rate))),[]));steps=[0]
            def hook(*args):steps[0]+=1
            handle=candidate.policy.optimizer.register_step_post_hook(hook)
            np.random.set_state(np_state);torch.set_rng_state(torch_state);candidate.train();handle.remove()
            parameter_error=max(float((v-reference.policy.state_dict()[k]).abs().max()) for k,v in candidate.policy.state_dict().items())
            observations=torch.as_tensor(buffer.observations[:,0]);old_values=[];new_values=[];kls=[]
            model.policy.set_training_mode(False);candidate.policy.set_training_mode(False)
            with torch.no_grad():
                for start in range(0,1280,128):
                    x=observations[start:start+128]
                    kls.append(kl_divergence(model.policy.get_distribution(x),candidate.policy.get_distribution(x)).numpy())
                    old_values.append(model.policy.predict_values(x).numpy().ravel());new_values.append(candidate.policy.predict_values(x).numpy().ravel())
            returns=buffer.returns.ravel()
            row={'learning_rate':rate,'offline_Adam_steps':steps[0],'native_checkpoint_maximum_parameter_error':parameter_error,
                'mean_KL_on_recorded_rollout':float(np.concatenate(kls).mean()),
                'value_MSE_before_on_same_TRAIN_rollout':float(np.square(np.concatenate(old_values)-returns).mean()),
                'value_MSE_after_on_same_TRAIN_rollout':float(np.square(np.concatenate(new_values)-returns).mean()),
                'metrics':dict(candidate.logger.name_to_value)}
            candidate.save(output/('offline_rate_'+str(rate)+'.zip'));results.append(row)
            # Record evidence before asserting:failed replay must remain inspectable.
            (output/'rate_results.json').write_text(json.dumps(results,indent=2,default=scalar_json),encoding='utf-8')
            if rate==3e-5 and (steps[0]!=1 or parameter_error>1e-7):
                raise ValueError('Original-rate replay did not match actual native checkpoint within declared1e-7;do not interpret alternative')
        result={'scope':__doc__,'batch':batch.name,'raw_audit_sha256':digest(folder/'endpoint_raw_audit.json'),
            'script_sha256':digest(__file__),'TRAIN_controls_reconstructed':1280,'max_sampled_action_error':maximum,
            'action_absolute_tolerance':1e-6,'native_parameter_absolute_tolerance':1e-7,
            'rollout_dataset_sha256':digest(output/'recorded_rollout.npz'),'results':results,
            'new_controls':0,'offline_PPO_Adam_steps':sum(r['offline_Adam_steps'] for r in results),
            'limits':'One training seed,one fixed measured TRAINrollout. Numerical reconstruction,not bitwise identity. Original unbounded actions are used. Same-data critic loss/KL and update counts are not driving improvement or heldout evidence. Offline models are diagnostic and never substitute for registered native policies.',
            'wall_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
        (output/'result.json').write_text(json.dumps(result,indent=2,default=scalar_json),encoding='utf-8')
        print(json.dumps({'rates':[{k:r[k] for k in ('learning_rate','offline_Adam_steps','native_checkpoint_maximum_parameter_error','value_MSE_after_on_same_TRAIN_rollout')} for r in results]}),flush=True)
        return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'new_controls':0,'wall_seconds':time.monotonic()-started}),encoding='utf-8');raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.batch,a.output)
