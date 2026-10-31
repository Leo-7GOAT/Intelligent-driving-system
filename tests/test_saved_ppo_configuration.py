import pickle
from types import SimpleNamespace

import pytest

from scripts import audit_native_ppo_configuration as auditor


def learner(**changes):
    model=SimpleNamespace(num_timesteps=0,_n_updates=0,n_steps=32,n_epochs=10,batch_size=32,
        gamma=.995,gae_lambda=.95,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
        normalize_advantage=True,target_kl=None,lr_schedule=lambda progress:3e-5,
        clip_range=lambda progress:.2,clip_range_vf=None,observation_space=SimpleNamespace(shape=(4705,)),
        action_space=SimpleNamespace(n=31),policy=SimpleNamespace(optimizer=SimpleNamespace(
            state={},param_groups=[{'lr':3e-5}],defaults={'lr':3e-5,'eps':1e-5})))
    model.__dict__.update(changes);return model


def directories(tmp_path):
    result=[]
    for name in ('candidate','reference'):
        folder=tmp_path/name;folder.mkdir()
        with (folder/'normalization.pkl').open('wb') as stream:
            pickle.dump(SimpleNamespace(norm_obs=True,norm_reward=False,clip_obs=10.,clip_reward=10.,epsilon=1e-8),stream)
        result.append(folder)
    return result


def test_saved_optimizer_rate_cannot_hide_behind_matching_declared_schedule(tmp_path,monkeypatch):
    candidate,reference=directories(tmp_path);wrong=learner();wrong.policy.optimizer.param_groups[0]['lr']=3e-4
    monkeypatch.setattr(auditor,'load_ppo',lambda path,**kw:wrong if path.parent==candidate else learner())
    with pytest.raises(ValueError,match='optimizer_group_rates'):auditor.verify(candidate,reference)


def test_future_rollout_or_discount_mismatch_is_rejected(tmp_path,monkeypatch):
    candidate,reference=directories(tmp_path)
    monkeypatch.setattr(auditor,'load_ppo',lambda path,**kw:learner(n_steps=128,gamma=.9) if path.parent==candidate else learner())
    with pytest.raises(ValueError,match='n_steps'):auditor.verify(candidate,reference)


def test_warm_checkpoint_cannot_be_counted_as_pristine_initializer(tmp_path,monkeypatch):
    candidate,_=directories(tmp_path)
    monkeypatch.setattr(auditor,'load_ppo',lambda *args,**kw:learner(num_timesteps=20480))
    with pytest.raises(ValueError,match='zero-PPO'):auditor.profile(candidate)
