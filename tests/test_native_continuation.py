"""Continuation publication and endpoint semantics fail closed."""
import hashlib
import json

import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from navstack.config import EnvConfig
from navstack.env import NavigationEnv
from navstack.experiments import signature, write_json
from scripts.continue_native_learning import (
    ORCHESTRATION_FILES,
    assert_continuation_pointer_available,
    register_prepared_child,
    sha,
    verified_endpoint,
    write_json_atomic,
)
from scripts.native_road_learning import save_training_checkpoint


def _publish_hashes(folder):
    write_json(folder/'evaluation.json',{
        key:hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name,key in [('model.zip','model_sha256'),
                         ('normalization.pkl','normalization_sha256')]})


def _make_endpoint(folder):
    """Use real SB3 persistence so tests cover the saved object semantics."""
    torch.set_num_threads(1)
    cfg=EnvConfig(method='B1',scenario='empty',flat_hold=10)
    raw=DummyVecEnv([lambda:NavigationEnv(cfg)]);raw.seed(23)
    raw.envs[0].case_index=0
    vec=VecNormalize(raw,norm_reward=False,gamma=cfg.gamma)
    try:
        model=PPO('MlpPolicy',vec,n_steps=8,batch_size=8,n_epochs=2,
                  seed=23,device='cpu')
        model.learn(16)
        steps=raw.envs[0].total_physics_steps
        write_json(folder/'config.json',{'env':cfg.to_dict(),'signature':signature(cfg)})
        metadata={'physics_steps':steps,'lineage_control_steps':steps,
                  'episodes':0,'threshold':steps,'optimizer_steps':4}
        save_training_checkpoint(folder,steps,model,vec,raw,metadata)
        model.save(folder/'model.zip');vec.save(folder/'normalization.pkl')
        write_json(folder/'summary.json',{**metadata,'reload_exact':True,
                   'stopped_reason':None,'charged_control_steps_lower_bound':steps})
        _publish_hashes(folder)
    finally:
        vec.close()
    assert verified_endpoint(folder,100000)[1]['physics_steps']==steps
    return steps


def test_parent_registration_requires_complete_matching_child(tmp_path):
    parent=tmp_path/'parent';child=tmp_path/'child'
    parent.mkdir();child.mkdir()
    write_json(child/'protocol.json',{'methods':['H','B1']})
    with pytest.raises(ValueError,match='completely prepared'):
        register_prepared_child(parent,child,100000)
    assert not (parent/'continuation_registration.json').exists()

    write_json_atomic(child/'prepared.json',{
        'status':'PREPARED_UNREGISTERED','protocol_sha256':sha(child/'protocol.json')})
    # A changed protocol invalidates the receipt before the exclusive parent
    # claim, so a staging failure cannot strand the parent.
    write_json(child/'protocol.json',{'methods':['H','H0','B1']})
    with pytest.raises(ValueError,match='does not match'):
        register_prepared_child(parent,child,100000)
    assert not (parent/'continuation_registration.json').exists()

    write_json_atomic(child/'prepared.json',{
        'status':'PREPARED_UNREGISTERED','protocol_sha256':sha(child/'protocol.json')})
    record=register_prepared_child(parent,child,100000)
    assert json.loads((parent/'continuation_registration.json').read_text())==record
    with pytest.raises(FileExistsError):
        register_prepared_child(parent,child,100000)


def test_active_or_unknown_continuation_pointer_cannot_be_hidden(tmp_path):
    runs=tmp_path/'runs';runs.mkdir()
    child=runs/'child';child.mkdir()
    write_json(runs/'native_continuation.json',{'batch':str(child)})
    with pytest.raises(RuntimeError,match='active or non-terminal'):
        assert_continuation_pointer_available(runs)
    write_json(child/'guard_final.json',{'exit_code':True})
    with pytest.raises(RuntimeError,match='active or non-terminal'):
        assert_continuation_pointer_available(runs)
    write_json(child/'guard_final.json',{'exit_code':1,'reason':'worker exited'})
    assert_continuation_pointer_available(runs) is None
    write_json(runs/'native_continuation.json',{'batch':str(runs/'missing')})
    with pytest.raises(RuntimeError,match='active or non-terminal'):
        assert_continuation_pointer_available(runs)


@pytest.mark.parametrize('tamper',('ppo','normalization'))
def test_endpoint_rejects_saved_semantic_tampering(tmp_path,tamper):
    _make_endpoint(tmp_path)
    if tamper=='ppo':
        changed=PPO.load(tmp_path/'model.zip',device='cpu')
        changed.n_epochs+=1
        changed.save(tmp_path/'model.zip')
        expected='PPO or policy semantics'
    else:
        cfg=EnvConfig(method='B1',scenario='empty',flat_hold=10)
        raw=DummyVecEnv([lambda:NavigationEnv(cfg)])
        changed=VecNormalize.load(tmp_path/'normalization.pkl',raw)
        try:
            changed.norm_obs=not changed.norm_obs
            changed.save(tmp_path/'normalization.pkl')
        finally:
            changed.close()
        expected='normalization semantics'
    _publish_hashes(tmp_path)
    with pytest.raises(ValueError,match=expected):
        verified_endpoint(tmp_path,100000)


def test_all_continuation_lifecycle_sources_are_declared():
    assert set(ORCHESTRATION_FILES)=={
        'continue_native_learning.py','watch_existing_pair.py','start_carla.ps1',
        'run_carla_owned_foreground.ps1','stop_owned_carla.ps1'}
