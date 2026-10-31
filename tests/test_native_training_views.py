from copy import deepcopy

import pytest

from scripts.build_native_training_views import compatible_configs,sampled_indices


@pytest.mark.parametrize('length',[1,5,6,349,900])
def test_fixed_stride_covers_last_precontrol_without_duplicate_rows(length):
    selected=sampled_indices(length,5)
    assert 0 in selected and length-1 in selected
    assert all(0<=i<length for i in selected)
    assert all(i%5==0 or i==length-1 for i in selected)


@pytest.mark.parametrize('length,stride',[(0,5),(True,5),(10,0),(10,21),(10,False)])
def test_bad_sampling_profiles_fail_before_data_use(length,stride):
    with pytest.raises(ValueError):sampled_indices(length,stride)


def configs():
    h={'method':'H','flat_hold':10,'endpoint_candidate_filter':True,'control_dt':.05,
       'reward':{'collision':-90},'goal_tolerance':.8,'vehicle':{'max_steer':.6}}
    b=deepcopy(h);b.update(method='B1',flat_hold=1,endpoint_candidate_filter=None)
    return h,b


def test_only_declared_h_to_direct_observation_differences_are_allowed():
    compatible_configs(*configs())


@pytest.mark.parametrize('key,value',[('control_dt',.1),('reward',{'collision':-45}),
    ('goal_tolerance',1.5),('vehicle',{'max_steer':.7})])
def test_physics_reward_and_success_gates_cannot_change(key,value):
    h,b=configs();b[key]=value
    with pytest.raises(ValueError,match='physical/task'):compatible_configs(h,b)
