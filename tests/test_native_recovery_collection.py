from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from scripts.collect_native_recovery import VERSION, tensor_digest, validate_collection
from scripts.prepare_native_recovery_collection import patch_worker
from scripts.fit_native_executed_recovery import executed_recovery_arrays,episode_balanced_weights


def inputs():
    ids = ['train'+str(i) for i in range(4)]
    cases = [{'native_route': {'geometry_id': i}} for i in ids]
    config = SimpleNamespace(method='B1', flat_hold=1, flat_longitudinal_mode='acceleration', deadline=45)
    protocol = {'methods': ['B1'], 'evaluation_only': True, 'deterministic_training_route_evaluation': False,
        'method_budgets': {'B1': 0}, 'controlled_budget_per_method': 0,
        'native_recovery_collection': {'version': VERSION, 'teacher_probability': .5, 'block_controls': 10,
            'max_controls': 3600, 'geometry_ids': ids, 'original_TRAIN_route_indices': [4, 5, 6, 7]}}
    return protocol, config, cases


@pytest.mark.parametrize('key,value', [('evaluation_only',False), ('deterministic_training_route_evaluation',True),
    ('method_budgets',{'B1':1}), ('controlled_budget_per_method',1)])
def test_collection_cannot_silently_become_evaluation_or_training(key,value):
    protocol, config, cases = inputs()
    validate_collection(protocol,config,cases)
    protocol[key]=value
    with pytest.raises(ValueError): validate_collection(protocol,config,cases)


@pytest.mark.parametrize('key,value', [('teacher_probability',.9),('block_controls',1),
    ('max_controls',7200),('original_TRAIN_route_indices',[0,1,2,3]),('geometry_ids',['validation']*4)])
def test_collection_cadence_budget_and_route_selection_are_locked(key,value):
    protocol, config, cases = inputs(); protocol['native_recovery_collection'][key]=value
    with pytest.raises(ValueError): validate_collection(protocol,config,cases)


def test_stable_fingerprint_catches_parameter_optimizer_and_rms_changes():
    original={'policy':np.zeros((2,3),dtype=np.float32),'optimizer':{'steps':7},'rms':{'count':42.,'mean':np.zeros(3)}}
    assert tensor_digest(original)==tensor_digest(deepcopy(original))
    for mutate in (lambda x:x['policy'].__setitem__((0,1),1),
                   lambda x:x['optimizer'].__setitem__('steps',8),
                   lambda x:x['rms'].__setitem__('count',43.)):
        changed=deepcopy(original);mutate(changed)
        assert tensor_digest(changed)!=tensor_digest(original)


def test_parent_worker_patch_changes_only_dispatch_and_rejects_double_patch():
    marker='                if protocol.get("deterministic_training_route_evaluation",False):'
    old='before\n'+marker+'\n                    untouched()\nafter\n'
    new=patch_worker(old)
    assert new.startswith('before\n') and new.endswith('\n                    untouched()\nafter\n')
    assert 'RecordedRoadEnv,lambda:guard(batch,protocol)' in new
    with pytest.raises(ValueError):patch_worker(new)
    with pytest.raises(ValueError):patch_worker(old+marker)


def test_fit_uses_only_executed_teacher_controls_from_all_episodes():
    data={'fit_train_mask':np.array([True,True,True,True,False]),
        'teacher_executed':np.array([True,True,True,True,False]),
        'observations':np.arange(10).reshape(5,2),'actions':np.arange(10).reshape(5,2),
        'groups':np.array([2048,2049,2050,2051,2050])}
    obs,labels,groups=executed_recovery_arrays(data,4)
    np.testing.assert_array_equal(obs,data['observations'][:4])
    np.testing.assert_array_equal(labels,data['actions'][:4])
    assert set(groups)==set(range(2048,2052))
    invalid=deepcopy(data);invalid['fit_train_mask'][-1]=True
    with pytest.raises(ValueError):executed_recovery_arrays(invalid,5)
    with pytest.raises(ValueError):executed_recovery_arrays(data,3)
    invalid=deepcopy(data);invalid['groups'][2]=2048
    with pytest.raises(ValueError):executed_recovery_arrays(invalid,4)


def test_timeout_episode_length_does_not_increase_total_sampling_probability():
    groups=np.array([0]*4+[1]*2+[2050]*900)
    weights=episode_balanced_weights(groups)
    np.testing.assert_allclose([weights[groups==g].sum() for g in np.unique(groups)],[1.,1.,1.])
    assert np.all(weights>0)
