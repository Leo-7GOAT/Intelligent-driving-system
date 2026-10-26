import numpy as np
import torch
import pytest
from stable_baselines3 import PPO
from test_discrete_flat_policy import Probe
from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
from scripts.fit_own_success_policy import fit_actor


def test_auxiliary_success_replay_changes_actor_without_ppo_or_critic_updates():
    torch.set_num_threads(1)
    model=PPO(DiscreteFlatRoadPolicy,Probe(True),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    model.learn(16)  # Populate both actor and critic Adam state before the check.
    actor=model.policy.action_net.weight.detach().clone()
    obs=np.zeros((8,4705),dtype=np.float32)
    actions=np.tile([1,3],(8,1));groups=np.repeat([0,2],4)
    updates=model._n_updates
    result=fit_actor(model,obs,actions,groups,10)
    assert result['actor_optimizer_steps']==10 and result['ppo_updates_added']==0
    assert result['critic_unchanged'] and result['critic_adam_unchanged']
    assert model._n_updates==updates
    assert not torch.equal(actor,model.policy.action_net.weight)
    assert result['last_loss']<result['first_loss']


def test_frozen_statistics_still_allow_actual_ppo_updates():
    from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
    from scripts.native_road_learning import configure_continuation_normalization
    env=VecNormalize(DummyVecEnv([lambda:Probe(True)]),norm_reward=False)
    model=PPO(DiscreteFlatRoadPolicy,env,seed=11,n_steps=8,batch_size=8,n_epochs=2)
    model.learn(8)
    before=(env.obs_rms.mean.copy(),env.obs_rms.var.copy(),env.obs_rms.count)
    actor=model.policy.action_net.weight.detach().clone();updates=model._n_updates
    with pytest.raises(ValueError):configure_continuation_normalization(env,{'freeze_observation_statistics':True},None)
    configure_continuation_normalization(env,{'freeze_observation_statistics':True},{'checkpoint':'saved'})
    model.learn(8,reset_num_timesteps=False)
    np.testing.assert_array_equal(before[0],env.obs_rms.mean);np.testing.assert_array_equal(before[1],env.obs_rms.var)
    assert before[2]==env.obs_rms.count and model._n_updates>updates
    assert not torch.equal(actor,model.policy.action_net.weight)
    env.close()


def test_interleaved_actor_replay_counts_separately_from_actual_ppo_updates():
    torch.set_num_threads(1)
    model=PPO(DiscreteFlatRoadPolicy,Probe(True),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    steps=[0]
    def count(*args):steps[0]+=1
    hook=model.policy.optimizer.register_step_post_hook(count)
    obs=np.zeros((8,4705),dtype=np.float32);actions=np.tile([1,3],(8,1));groups=np.repeat([0,2],4)
    for phase in range(2):
        model.learn(8,reset_num_timesteps=False)
        fit_actor(model,obs,actions,groups,100,seed=phase)
    hook.remove()
    assert model._n_updates==4 and steps[0]==204
    assert steps[0]-200==model._n_updates


def test_frozen_evaluation_drops_inherited_training_only_statistics_flag(tmp_path,monkeypatch):
    import json
    from scripts import prepare_own_success_evaluation as stage
    root=tmp_path/'project';(root/'runs').mkdir(parents=True);(root/'scripts').mkdir()
    parent=tmp_path/'parent';(parent/'source/scripts').mkdir(parents=True);(parent/'calibration').mkdir();(parent/'B1').mkdir()
    artifact=tmp_path/'actor';artifact.mkdir()
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    for name in ('native_road_learning.py','audit_native_method_endpoint.py','watch_existing_pair.py','prepare_own_success_evaluation.py','run_carla_owned_foreground.ps1','fit_own_success_policy.py'):
        (root/'scripts'/name).write_text('# fixture',encoding='utf-8')
    calibration=parent/'calibration/calibration.json';write(calibration,{})
    write(parent/'protocol.json',{'source_hashes':{},'calibration_sha256':stage.sha(calibration),
        'freeze_observation_statistics':True,'resource_limits':{'wall_seconds':900},'routes':[{'id':0}],
        'recovery':{'B1':{'retained_optimizer_steps':4}}})
    write(parent/'B1/summary.json',{'optimizer_steps':2,'lineage_control_steps':20,'charged_control_steps_lower_bound':20})
    write(parent/'B1/endpoint_raw_audit.json',{'fixture':True})
    metadata={'actor_optimizer_steps':10,'samples':8,'condition_counts':{'0':8}}
    for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256'),('resume_state.pkl','resume_state_sha256')]:
        (artifact/name).write_bytes(name.encode());metadata[key]=stage.sha(artifact/name)
    write(artifact/'metadata.json',metadata);write(artifact/'config.json',{})
    monkeypatch.setattr(stage,'ROOT',root);monkeypatch.setattr(stage,'validate',lambda *args:('B1',metadata))
    monkeypatch.setattr(stage,'resources_ok',lambda *args:(True,None))
    batch=stage.prepare(parent,artifact)
    protocol=stage.read(batch/'protocol.json')
    assert protocol['evaluation_only'] and protocol['controlled_budget_per_method']==0
    assert 'freeze_observation_statistics' not in protocol
    assert protocol['recovery']['B1']['retained_optimizer_steps']==16
