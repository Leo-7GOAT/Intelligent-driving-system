from copy import deepcopy
import pytest
from scripts.prepare_native_full_task_after_course import require_all_courses,full_protocol,require_completed_courses_for_extension,retain_optimizer_with_lower_rate
from types import SimpleNamespace
import torch


def qualified():
    return {'phase':'ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING',
            'completed':[{'seed':s,'basic_qualified_after_learning':True} for s in (11,22,33,44,55)]}


@pytest.mark.parametrize('fault',['still_running','omitted_seed','failed_seed','string_boolean'])
def test_full_task_cannot_start_from_incomplete_or_selected_courses(fault):
    state=qualified()
    if fault=='still_running':state['phase']='TRAINING_BASIC_MOTION_WITH_ACTUAL_PPO'
    elif fault=='omitted_seed':state['completed'].pop()
    elif fault=='failed_seed':state['completed'][1]['basic_qualified_after_learning']=False
    else:state['completed'][1]['basic_qualified_after_learning']='True'
    with pytest.raises(ValueError):require_all_courses(state)


def test_restoring_full_inventory_retains_and_charges_the_basic_curriculum():
    full={'cases':list(range(12))};old={'methods':['B1'],'evaluation_only':False,'method_budgets':{'B1':10240},
        'native_case_manifest':{'cases':list(range(4))},'basic_motion_learning':{'original_full_case_manifest':full},'resource_limits':{}}
    summary={'physics_steps':10240,'optimizer_steps':9,'stopped_reason':None,'reload_exact':True}
    saved=deepcopy(old);p=full_protocol(old,'basic',summary,1500)
    assert old==saved and p['native_case_manifest']==full
    assert p['method_budgets']=={'B1':20480} and p['full_task_after_basic_course']['policy_lineage_target_controls']==30720
    assert p['full_task_after_basic_course']['basic_evaluation_controls']==1500
    assert p['full_task_after_basic_course']['additional_supervised_updates']==0
    summary['optimizer_steps']=0
    with pytest.raises(ValueError):full_protocol(old,'basic',summary,1500)


def test_an_extension_requires_every_finished_course_and_keeps_failed_gate_distinct_from_full_entry():
    s=qualified();s['phase']='ALL_FIVE_BASIC_PPO_COURSES_COMPLETE_WITH_FAILED_GATES';s['completed'][0]['basic_qualified_after_learning']=False
    require_completed_courses_for_extension(s)
    with pytest.raises(ValueError):require_all_courses(s)
    s['completed'].pop()
    with pytest.raises(ValueError):require_completed_courses_for_extension(s)


def test_lower_rate_continuation_preserves_actual_trained_Adam_moments_and_weights():
    policy=torch.nn.Linear(2,1);policy.optimizer=torch.optim.Adam(policy.parameters(),lr=3e-5)
    policy(torch.tensor([[1.,2.]])).square().mean().backward();policy.optimizer.step()
    before=deepcopy(policy.state_dict());moments=deepcopy(policy.optimizer.state_dict()['state'])
    model=SimpleNamespace(policy=policy,num_timesteps=1280,_n_updates=1,_setup_lr_schedule=lambda:None)
    change=retain_optimizer_with_lower_rate(model)
    torch.testing.assert_close(policy.state_dict(),before,rtol=0,atol=0)
    torch.testing.assert_close(policy.optimizer.state_dict()['state'],moments,rtol=0,atol=0)
    assert model.learning_rate==3e-6 and policy.optimizer.param_groups[0]['lr']==3e-6
    assert change['new_optimizer_updates']==0


def test_full_stage_charges_both_basic_learning_stages_and_retained_Adam_steps():
    old={'methods':['B1'],'evaluation_only':False,'method_budgets':{'B1':10240},'native_case_manifest':{'cases':list(range(4))},
         'basic_motion_learning':{'original_full_case_manifest':{'cases':list(range(12))}},'resource_limits':{},
         'recovery':{'B1':{'retained_optimizer_steps':9}}}
    summary={'physics_steps':10240,'lineage_control_steps':20480,'optimizer_steps':300,'stopped_reason':None,'reload_exact':True}
    p=full_protocol(old,'extension',summary,1500)
    assert p['full_task_after_basic_course']['policy_lineage_target_controls']==40960
    assert p['full_task_after_basic_course']['basic_actual_Adam_steps']==309
