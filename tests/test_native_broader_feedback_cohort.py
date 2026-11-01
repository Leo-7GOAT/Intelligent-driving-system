from copy import deepcopy
import pytest
from scripts import prepare_native_broader_feedback_cohort as preparer
from scripts.run_native_broader_feedback_cohort import require_preparation,require_camera_completion


def test_broader_replication_preserves_prior_protocol_and_uses_the_fixed_development_budget(monkeypatch):
    old={'seed':11,'methods':['H','H0'],'evaluation_only':False,'routes':['old'],
         'native_case_manifest':{'cases':[1]},'resource_limits':{'wall_seconds':900},
         'extra_supervision':{'dataset_sha256':'same_teacher'}}
    inventory={'routes':['new'],'native_case_manifest':{'cases':list(range(36))},
               'native_validation':{'manifest':{'cases':list(range(24))}}}
    before=deepcopy(old)
    monkeypatch.setattr(preparer,'validate_native_partitions',lambda p:{'training_conditions':36,'validation_conditions':24})
    p=preparer.broader_protocol(old,inventory,11)
    assert old==before and p['method_budgets']=={'H':20480,'H0':20480}
    assert p['extra_supervision']==old['extra_supervision']
    assert p['broader_feedback_replication']['additional_supervised_updates']==0
    assert p['freeze_observation_statistics'] is True and 'evaluation_only' not in p
    assert p['checkpoint_thresholds'][-1]==20480
    p['native_validation']['manifest']['cases'].pop()
    assert len(inventory['native_validation']['manifest']['cases'])==24
    with pytest.raises(ValueError):preparer.broader_protocol(old,inventory,22)
    monkeypatch.setattr(preparer,'validate_native_partitions',lambda p:{'training_conditions':35,'validation_conditions':24})
    with pytest.raises(ValueError):preparer.broader_protocol(old,inventory,11)


def registered():
    return {'phase':'ALL_FIVE_MATCHED_BROADER_FEEDBACK_PREPARED_NOT_DRIVEN',
        'batches':[{'seed':s,'batch':str(s)} for s in (11,22,33,44,55)],
        'distinct_actor_initializations_each_method':{'H':5,'H0':5}}


def test_no_seed_omission_duplicate_or_shared_policy_can_enter_the_cohort():
    state=registered();assert len(require_preparation(state))==5
    state['batches'].pop()
    with pytest.raises(ValueError):require_preparation(state)
    state=registered();state['batches'][1]['batch']=state['batches'][0]['batch']
    with pytest.raises(ValueError):require_preparation(state)
    state=registered();state['distinct_actor_initializations_each_method']['H0']=1
    with pytest.raises(ValueError):require_preparation(state)


def test_live_or_failed_predecessor_does_not_release_native_resources():
    state={'phase':'ALL_REGISTERED_NATIVE_PPO_RECORDINGS_AND_FRAME_VERIFICATIONS_COMPLETE',
           'completed':[{'method':'H'},{'method':'B1'}]}
    require_camera_completion(state,False)
    with pytest.raises(ValueError):require_camera_completion(state,True)
    state['phase']='WAITING_FOR_OWNED_TRAINING_QUEUES_TO_EXIT'
    with pytest.raises(ValueError):require_camera_completion(state,False)
