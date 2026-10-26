import numpy as np
import torch
import pytest
from stable_baselines3 import PPO
from test_discrete_flat_policy import Probe
from navstack.discrete_flat_policy import DiscreteFlatRoadPolicy
from scripts.fit_own_success_policy import fit_actor


def test_terminal_margin_is_stricter_sample_selection_without_changing_task_gate():
    from scripts.fit_own_success_policy import terminal_quality,has_terminal_margin
    assert has_terminal_margin(terminal_quality({'x':.4,'y':0.,'v':.1,'yaw':2*np.pi+.1},[0.,0.,0.]))
    assert not has_terminal_margin(terminal_quality({'x':.792,'y':0.,'v':.22,'yaw':0.},[0.,0.,0.]))
    assert not has_terminal_margin({'distance':.4,'speed':.16,'yaw_error':0.})
    assert not has_terminal_margin({'distance':.4,'speed':.1,'yaw_error':.21})


def test_speed_smoothing_restores_gradient_for_collapsed_speed_without_smoothing_steering():
    from stable_baselines3.common.distributions import MultiCategoricalDistribution
    from scripts.fit_own_success_policy import actor_replay_loss
    gradients=[]
    for smoothing in (0.,.3):
        logits=torch.tensor([[-12.,12.,-12.,0.,0.,0.,0.,0.,0.,0.]],requires_grad=True)
        distribution=MultiCategoricalDistribution([3,7]).proba_distribution(logits)
        actor_replay_loss(distribution,torch.tensor([[1,3]]),smoothing).backward()
        gradients.append(logits.grad.clone())
    assert abs(float(gradients[0][0,2]))<1e-8 and float(gradients[1][0,2])<-.09
    torch.testing.assert_close(gradients[0][:,3:],gradients[1][:,3:],rtol=0,atol=0)


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


@pytest.mark.parametrize('auxiliary_steps',[10,100])
def test_interleaved_actor_replay_counts_separately_from_actual_ppo_updates(auxiliary_steps):
    torch.set_num_threads(1)
    model=PPO(DiscreteFlatRoadPolicy,Probe(True),seed=11,n_steps=8,batch_size=8,n_epochs=2)
    steps=[0]
    def count(*args):steps[0]+=1
    hook=model.policy.optimizer.register_step_post_hook(count)
    obs=np.zeros((8,4705),dtype=np.float32);actions=np.tile([1,3],(8,1));groups=np.repeat([0,2],4)
    for phase in range(2):
        model.learn(8,reset_num_timesteps=False)
        fit_actor(model,obs,actions,groups,auxiliary_steps,seed=phase)
    hook.remove()
    assert model._n_updates==4 and steps[0]==4+2*auxiliary_steps
    assert steps[0]-2*auxiliary_steps==model._n_updates


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
        'freeze_observation_statistics':True,'own_success_replay':{'steps_per_phase':100,'online_success_updates':True},
        'resource_limits':{'wall_seconds':900},'routes':[{'id':0}],
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
    assert 'own_success_replay' not in protocol
    assert protocol['recovery']['B1']['retained_optimizer_steps']==16


def test_online_memory_uses_actual_ppo_inputs_and_excludes_failed_or_intervened_episodes(tmp_path):
    from scripts.own_success_replay import OwnSuccessMemory
    class Episodes(Probe):
        def __init__(self):super().__init__(True);self.episode=-1
        def reset(self,**kwargs):self.episode+=1;return super().reset(**kwargs)
        def step(self,action):
            obs,reward,done,truncated,info=super().step(action)
            return obs,reward,done,truncated,{'success':self.episode!=0,'trace':[{'intervention':self.episode==2,'controller_status':'direct'}]}
    torch.set_num_threads(1)
    initial=np.zeros((2,4705),dtype=np.float32);actions=np.tile([1,3],(2,1));groups=np.zeros(2,dtype=np.int64)
    memory=OwnSuccessMemory(initial,actions,groups,lambda episode:episode)
    model=PPO(DiscreteFlatRoadPolicy,Episodes(),seed=11,n_steps=8,batch_size=8,n_epochs=1)
    model.learn(24,callback=memory)
    metadata=memory.save(tmp_path/'memory.npz')
    assert metadata['new_own_success_decisions']==8 and metadata['new_own_success_episodes']==1
    saved=np.load(tmp_path/'memory.npz',allow_pickle=False)
    np.testing.assert_array_equal(saved['groups'],[0,0]+[1]*8)
    np.testing.assert_array_equal(saved['normalized_observations'],np.zeros((10,4705)))
    assert (saved['actions'][:,0]<3).all() and (saved['actions'][:,1]<7).all()


def test_curriculum_gate_requires_all_current_goals_same_model_and_same_task(tmp_path,monkeypatch):
    import json
    from scripts import prepare_own_success_evaluation as stage
    parent=tmp_path/'parent';artifact=tmp_path/'artifact';batch=tmp_path/'runs/evaluation';folder=batch/'B1'
    for path in (parent,artifact,folder):path.mkdir(parents=True)
    def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')
    protocol={'routes':[0,1,2,3],'evaluation_only':True,'calibration_sha256':'calibration','source_hashes':{'navstack/env.py':'unchanged'}}
    write(parent/'protocol.json',protocol);write(batch/'protocol.json',protocol)
    write(artifact/'evaluation_registration.json',{'batch':'evaluation'})
    for name in ('model.zip','normalization.pkl'):
        (artifact/name).write_bytes(name.encode());(folder/name).write_bytes(name.encode())
    config={'env':{'goal_tolerance':.8}}
    write(artifact/'config.json',config);write(folder/'config.json',config)
    episodes=[{'case_id':i,'success':True} for i in range(4)]
    write(folder/'evaluation.json',{'episodes':episodes});write(folder/'endpoint_raw_audit.json',{'controls':8})
    monkeypatch.setattr(stage,'ROOT',tmp_path);monkeypatch.setattr(stage,'require_finished_parent',lambda *args:None)
    monkeypatch.setattr(stage,'audit',lambda *args:{'controls':8})
    assert stage.verified_fitted_goal_gate(parent,artifact,'B1')['batch']=='evaluation'
    episodes[1]['success']=False;write(folder/'evaluation.json',{'episodes':episodes})
    with pytest.raises(ValueError,match='All four'):stage.verified_fitted_goal_gate(parent,artifact,'B1')
    episodes[1]['success']=True;write(folder/'evaluation.json',{'episodes':episodes})
    write(folder/'config.json',{'env':{'goal_tolerance':1.}})
    with pytest.raises(ValueError,match='task or controller'):stage.verified_fitted_goal_gate(parent,artifact,'B1')
    write(folder/'config.json',config);(folder/'model.zip').write_bytes(b'other actor')
    with pytest.raises(ValueError,match='saved fitted actor'):stage.verified_fitted_goal_gate(parent,artifact,'B1')


def test_online_quality_gate_keeps_original_success_but_filters_auxiliary_replay(tmp_path):
    from scripts.own_success_replay import OwnSuccessMemory
    class Episodes(Probe):
        def __init__(self):super().__init__(True);self.episode=-1
        def reset(self,**kwargs):self.episode+=1;return super().reset(**kwargs)
        def step(self,action):
            obs,reward,done,truncated,info=super().step(action)
            return obs,reward,done,truncated,{'success':True,'margin':self.episode!=1,'trace':[]}
    torch.set_num_threads(1)
    memory=OwnSuccessMemory(np.zeros((1,4705),np.float32),np.array([[1,3]]),np.array([-1]),
        lambda episode:episode,lambda info:info['margin'])
    model=PPO(DiscreteFlatRoadPolicy,Episodes(),seed=11,n_steps=8,batch_size=8,n_epochs=1)
    model.learn(24,callback=memory)
    result=memory.save(tmp_path/'memory.npz')
    assert result['new_own_success_episodes']==2 and result['successful_episodes_rejected_by_quality']==1
    assert result['new_own_success_decisions']==16
    np.testing.assert_array_equal(memory.arrays()[2],[-1]+[0]*8+[2]*8)
