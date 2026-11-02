from copy import deepcopy
from types import SimpleNamespace
import json
from pathlib import Path
import numpy as np
import pytest

from scripts.collect_native_recovery import VERSION, tensor_digest, validate_collection
from scripts.prepare_native_recovery_collection import patch_worker
from scripts.fit_native_executed_recovery import executed_recovery_arrays,episode_balanced_weights,actor_fit_lineage,require_probe_fit


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


def test_original_train_collection_requires_explicit_distinct_profile():
    protocol,config,cases=inputs();spec=protocol['native_recovery_collection']
    spec['original_TRAIN_route_indices']=[0,1,2,3]
    with pytest.raises(ValueError):validate_collection(protocol,config,cases)
    spec['route_profile']='original_TRAIN4_v1'
    assert validate_collection(protocol,config,cases) is spec
    spec['route_profile']='validation'
    with pytest.raises(ValueError):validate_collection(protocol,config,cases)


def broader_inputs():
    from navstack.native_cases import registered_native_cases
    p, config, _ = inputs()
    layout = json.loads((Path(__file__).resolve().parents[1] / 'experiments/native_development_train36_validation24_v1.json').read_text(encoding='utf-8'))
    p.update(routes=layout['routes'], native_case_manifest=layout['native_case_manifest'], data_role='TRAIN_EXTRA_SUPERVISION')
    cases = registered_native_cases(p)
    p['native_recovery_collection'].update(route_profile='broader_TRAIN36_v1', max_controls=32400,
        original_TRAIN_case_indices=list(range(36)), TRAIN_case_manifest=deepcopy(p['native_case_manifest']),
        geometry_ids=[c['native_route']['geometry_id'] for c in cases],
        source_lineage_control_steps=51200, exclude_intervened_teacher_labels=True)
    return p, config, cases


def test_broader_collection_includes_all_three_train_families():
    p, config, cases = broader_inputs()
    assert len(cases) == 36
    assert validate_collection(p, config, cases) is p['native_recovery_collection']


@pytest.mark.parametrize('mutation', ['validation', 'partial', 'manifest', 'mask', 'budget', 'lineage', 'role'])
def test_broader_collection_rejects_leakage_selection_and_unexecuted_labels(mutation):
    p, config, cases = broader_inputs(); spec = p['native_recovery_collection']
    if mutation == 'validation': p['native_validation'] = {}
    elif mutation == 'partial': cases = cases[:-1]
    elif mutation == 'manifest': spec['TRAIN_case_manifest']['cases'][4]['hazard'] = None
    elif mutation == 'mask': spec['exclude_intervened_teacher_labels'] = False
    elif mutation == 'budget': spec['max_controls'] = 64000
    elif mutation == 'lineage': spec['source_lineage_control_steps'] = 20480
    else: p['data_role'] = 'VAL'
    with pytest.raises(ValueError): validate_collection(p, config, cases)


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


def test_broader_recovery_fit_excludes_unexecuted_proposals_and_preserves_all_episodes():
    from scripts.fit_broader_native_recovery import recovery_masks
    groups = np.repeat(np.arange(2048, 2084), 6)
    data = dict(groups=groups, fit_train_mask=np.tile([True]*5+[False],36),
        teacher_executed=np.tile([True]*5+[False],36))
    train, holdout = recovery_masks(data, 180)
    assert train.sum() == 144 and holdout.sum() == 36
    assert not (train & holdout).any()
    np.testing.assert_array_equal(train | holdout, data['fit_train_mask'])
    bad = deepcopy(data); bad['teacher_executed'][0] = False
    with pytest.raises(ValueError): recovery_masks(bad,180)
    bad = deepcopy(data); bad['groups'][bad['groups']==2083] = 2082
    with pytest.raises(ValueError): recovery_masks(bad,180)
    with pytest.raises(ValueError): recovery_masks(data,179)


def test_repeated_actor_interventions_retain_cumulative_cost_instead_of_last_segment_only():
    legacy={'additional_actor_fit_steps':512}
    first=actor_fit_lineage(legacy,512)
    assert first['cumulative_actor_fit_steps']==1024
    second=actor_fit_lineage(first,512)
    assert second=={'inherited_actor_fit_steps':1024,'additional_actor_fit_steps':512,'cumulative_actor_fit_steps':1536}


def test_rejected_old_fit_cannot_use_an_unregistered_flag_to_enter_native_probe():
    result={'new_supervised_Adam_proposals':512,'retention_gate':{'passed':False},'TRAIN_retention_screen_applied':False}
    with pytest.raises(ValueError):require_probe_fit(result)
    result['fit_profile']='executed_original_TRAIN_no_projection_v1'
    with pytest.raises(ValueError):require_probe_fit(result)
    result.update(route_profile='original_TRAIN4_v1',retention_reference_model_sha256='4f20570844ff17da318c601893be8523ddf504193d21775f9553a3d5c8530ca0')
    assert require_probe_fit(result) is True
    result['new_supervised_Adam_proposals']=1024
    with pytest.raises(ValueError):require_probe_fit(result)
