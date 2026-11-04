from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts.prepare_native_training_view_pair import evaluation_input_protocol
from scripts.native_road_learning import configure_continuation_normalization


def test_actor_validation_cannot_inherit_learning_only_normalization_flag():
    original={'freeze_observation_statistics':True,'evaluation_only':True,
              'recovery':{'B1':{'checkpoint':'resume/B1'}},'rollout_actions':1280,
              'extra_supervision':{'dataset_sha256':'same'},'native_validation':{'manifest':'same'}}
    before=deepcopy(original)
    fixed=evaluation_input_protocol(original)
    with pytest.raises(ValueError,match='learning continuation'):
        configure_continuation_normalization(SimpleNamespace(),original,original['recovery']['B1'])
    configure_continuation_normalization(SimpleNamespace(),fixed,fixed['recovery']['B1'])
    assert original==before and 'freeze_observation_statistics' not in fixed
    assert fixed=={k:v for k,v in original.items() if k!='freeze_observation_statistics'}


def test_removes_online_learning_interventions_from_zero_update_actor_gate():
    fixed=evaluation_input_protocol({'jobs':[{'name':'B1'}],'own_success_replay':{'steps':10},
                                     'seed':11,'controlled_budget_per_method':0})
    assert fixed=={'seed':11,'controlled_budget_per_method':0}
