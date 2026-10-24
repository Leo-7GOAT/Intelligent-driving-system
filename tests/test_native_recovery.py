"""Real PPO checkpoint roundtrip on the lightweight plant; CARLA is tested separately."""
import hashlib
import json
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import signature,write_json
from scripts.native_road_learning import restore_training,save_training_checkpoint
from scripts.prepare_native_recovery import claim_recovery
from scripts.continue_native_learning import verified_endpoint


def test_native_learning_state_checkpoint_roundtrip_and_hash_guard(tmp_path):
    torch.set_num_threads(1)
    cfg=EnvConfig(method='B1',scenario='empty',flat_hold=10)
    raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);raw.seed(11)
    raw.envs[0].case_index=3
    vec=VecNormalize(raw,norm_reward=False,gamma=cfg.gamma)
    model=PPO('MlpPolicy',vec,n_steps=8,batch_size=8,n_epochs=2,seed=11,device='cpu')
    model.learn(16)
    write_json(tmp_path/'config.json',{'env':cfg.to_dict(),'signature':signature(cfg)})
    metadata={'physics_steps':raw.envs[0].total_physics_steps,'episodes':0,'threshold':160}
    save_training_checkpoint(tmp_path,160,model,vec,raw,metadata)
    checkpoint=tmp_path/'checkpoint_000000160'
    assert checkpoint.is_dir() and not (tmp_path/'checkpoint_000000160.partial').exists()
    assert json.loads((tmp_path/'latest_checkpoint.json').read_text())['folder']==checkpoint.name
    expected_rng=torch.get_rng_state().clone()
    new_raw=DummyVecEnv([lambda:NavigationEnv(cfg)])
    loaded,norm,mode=restore_training(checkpoint,new_raw,cfg,99)
    assert loaded._last_obs is None and 'saved RNG' in mode
    assert new_raw.envs[0].case_index==3
    assert torch.equal(torch.get_rng_state(),expected_rng)
    np.testing.assert_array_equal(norm.obs_rms.mean,vec.obs_rms.mean)
    assert loaded._n_updates==model._n_updates
    for name,weight in model.policy.state_dict().items():
        assert torch.equal(weight,loaded.policy.state_dict()[name])
    old_states=model.policy.optimizer.state_dict()['state']
    new_states=loaded.policy.optimizer.state_dict()['state']
    assert old_states.keys()==new_states.keys() and old_states
    for key,state in old_states.items():
        for name,value in state.items():
            assert torch.equal(value,new_states[key][name]) if torch.is_tensor(value) else value==new_states[key][name]
    before=loaded._n_updates;loaded.learn(8,reset_num_timesteps=False)
    assert loaded._n_updates>before
    norm.close();vec.close()
    with (checkpoint/'model.zip').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        restore_training(checkpoint,new_raw,cfg,99)


def test_parent_cannot_be_registered_for_duplicate_recovery(tmp_path):
    claim_recovery(tmp_path,'first_child')
    with pytest.raises(FileExistsError):claim_recovery(tmp_path,'duplicate_child')
    assert json.loads((tmp_path/'recovery_registration.json').read_text())['child_batch']=='first_child'


def test_continuation_uses_evaluated_weights_and_optimizer(tmp_path):
    """Actual PPO/plant; a valid hash alone cannot excuse changed learning state."""
    torch.set_num_threads(1)
    cfg=EnvConfig(method='B1',scenario='empty',flat_hold=10)
    raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);raw.seed(11)
    raw.envs[0].case_index=0
    vec=VecNormalize(raw,norm_reward=False,gamma=cfg.gamma)
    try:
        model=PPO('MlpPolicy',vec,n_steps=8,batch_size=8,n_epochs=2,seed=11,device='cpu')
        model.learn(16)
        steps=raw.envs[0].total_physics_steps
        write_json(tmp_path/'config.json',{'env':cfg.to_dict(),'signature':signature(cfg)})
        metadata={'physics_steps':steps,'lineage_control_steps':steps,'episodes':0,'threshold':steps}
        save_training_checkpoint(tmp_path,steps,model,vec,raw,metadata)
        model.save(tmp_path/'model.zip');vec.save(tmp_path/'normalization.pkl')
        summary={**metadata,'reload_exact':True,'stopped_reason':None,'charged_control_steps_lower_bound':steps}
        write_json(tmp_path/'summary.json',summary)
        def publish_hashes():
            write_json(tmp_path/'evaluation.json',{
                key:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
                for name,key in [('model.zip','model_sha256'),('normalization.pkl','normalization_sha256')]})
        publish_hashes()
        checkpoint,actual,_=verified_endpoint(tmp_path,100000)
        assert checkpoint.name==f'checkpoint_{steps:09d}' and actual['physics_steps']==steps
        # Deliberately modify optimizer state while retaining the exact actor.
        state=next(iter(model.policy.optimizer.state.values()))
        state['exp_avg'].add_(1.)
        model.save(tmp_path/'model.zip');publish_hashes()
        with pytest.raises(ValueError,match='weights or optimizer differ'):
            verified_endpoint(tmp_path,100000)
        with pytest.raises(ValueError,match='extendable'):
            verified_endpoint(tmp_path,steps)
    finally:
        vec.close()
