from copy import deepcopy
import pytest
from scripts.prepare_native_basic_motion_ppo import course_protocol
from scripts.run_native_basic_motion_ppo_cohort import require_failed_predecessor,basic_qualification
from scripts.audit_native_ppo_configuration import action_profile
from gymnasium import spaces
from types import SimpleNamespace
import torch


def source():
    return {'methods':['B1'],'native_case_manifest':{'version':'native_development_cases_v1','split':'development',
        'cases':[{'route_index':i%4,'hazard':None if i<4 else {'kind':'old_hazard'}} for i in range(12)]},
        'resource_limits':{'wall_seconds':1800},'initialization_gate_profile':'basic_motion_training_v1',
        'supervised_representation_gates':{'B1':'failed'},'evaluation_only':True}


def test_course_is_explicitly_learning_before_qualification_and_keeps_full_task_identity():
    original=source();saved=deepcopy(original);course=course_protocol(original,'parent')
    assert original==saved
    assert course['evaluation_only'] is False and course['method_budgets']=={'B1':10240}
    assert len(course['native_case_manifest']['cases'])==4
    assert course['basic_motion_learning']['original_full_case_manifest']==original['native_case_manifest']
    assert course['basic_motion_learning']['additional_supervised_updates']==0
    assert 'supervised_representation_gates' not in course and 'initialization_gate_profile' not in course
    assert course['basic_motion_learning']['maximum_basic_unsafe_episodes']==0


@pytest.mark.parametrize('fault',['missing_road','selected_road','other_method','validation','replay','wrong_budget'])
def test_course_cannot_hide_seed_selection_heldout_data_or_unregistered_extra_learning(fault):
    p=source();controls=10240
    if fault=='missing_road':p['native_case_manifest']['cases'].pop(0)
    elif fault=='selected_road':p['native_case_manifest']['cases'][0]['route_index']=1
    elif fault=='other_method':p['methods']=['H']
    elif fault=='validation':p['native_validation']={'heldout':True}
    elif fault=='replay':p['own_success_replay']={'extra_fit':True}
    else:controls=20480
    with pytest.raises(ValueError):course_protocol(p,'parent',controls)


@pytest.mark.parametrize('fault',['not_failed','not_all_gates','started_old_PPO','no_failure_receipt'])
def test_new_learning_course_cannot_overlap_or_shortcut_the_old_gates(fault):
    state={'gates':[{'seed':s,'passed':s!=22} for s in (11,22,33,44,55)],'created':[]};present=True
    if fault=='not_failed':state['gates'][1]['passed']=True
    elif fault=='not_all_gates':state['gates'].pop()
    elif fault=='started_old_PPO':state['created']=[{'seed':11}]
    else:present=False
    with pytest.raises(ValueError):require_failed_predecessor(state,present)


def test_learning_course_does_not_lower_basic_safety_requirement():
    episodes=[{'case_id':i,'success':i<3,'termination_reason':'success' if i<3 else 'deadline'} for i in range(4)]
    assert basic_qualification(episodes)
    episodes[3]['termination_reason']='out_of_bounds'
    assert not basic_qualification(episodes)
    with pytest.raises(ValueError):basic_qualification(episodes[:3])


def test_flat_ppo_profile_covers_continuous_bounds_and_exploration_without_changing_discrete_profiles():
    policy=SimpleNamespace(log_std=torch.tensor([-4.,-4.]))
    assert action_profile(spaces.Discrete(31),policy)=={'action_count':31}
    a=action_profile(spaces.Box(-1.,1.,(2,)),policy)
    assert a['action_shape']==[2] and a['initial_log_std']==[-4.,-4.]
    assert a!=action_profile(spaces.Box(-.5,.5,(2,)),policy)
    assert a!=action_profile(spaces.Box(-1.,1.,(2,)),SimpleNamespace(log_std=torch.tensor([-3.,-4.])))
