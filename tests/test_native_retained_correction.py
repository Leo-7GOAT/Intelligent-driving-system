import numpy as np
import pytest
from scripts.probe_native_retained_correction import retention_gate,backtrack_actor


def arrays():
    return np.full((8,2),.1),np.full((8,2),.1),np.zeros((8,2)),np.repeat(np.arange(4),2),np.full((2,2),.1),np.ones((2,2)),np.zeros((2,2))


def test_fixed_gate_requires_every_basic_group_and_query_improvement():
    data=arrays();result=retention_gate(*data)
    assert result['passed']
    assert len(result['basic_teacher_groups'])==4
    data=list(data);data[4]=data[5].copy()
    assert not retention_gate(*data)['passed']


def test_better_pooled_error_cannot_hide_one_forgotten_condition():
    data=list(arrays());data[0]=np.zeros((8,2));data[0][-2:]=.19
    assert np.square(data[0]-data[2]).mean()<np.square(data[1]-data[2]).mean()
    result=retention_gate(*data)
    assert not result['passed']
    assert result['basic_teacher_groups'][3]['passed'] is False


@pytest.mark.parametrize('fault',['missing_group','nonfinite','wrong_shape'])
def test_incomplete_or_invalid_training_metrics_fail_closed(fault):
    data=list(arrays())
    if fault=='missing_group':data[3][-2:]=2
    elif fault=='nonfinite':data[4][0,0]=np.nan
    else:data[1]=data[1][:-1]
    with pytest.raises(ValueError):retention_gate(*data)


def test_backtracking_checks_actual_displacement_and_keeps_other_parameters():
    import torch
    actor=torch.nn.Parameter(torch.tensor([1.]));critic=torch.nn.Parameter(torch.tensor([7.]))
    accepted=[actor.detach().clone()];proposed=[torch.tensor([3.])]
    def check():return {'basic_teacher_groups':[{'passed':float(actor[0].detach())<=2.}]}
    scale,_=backtrack_actor([actor],accepted,proposed,check)
    assert scale==.5 and actor.item()==2. and critic.item()==7.
    assert accepted[0].item()==1. and proposed[0].item()==3.


def test_backtracking_can_reject_all_proposed_displacement():
    import torch
    actor=torch.nn.Parameter(torch.tensor([1.]))
    def check():return {'basic_teacher_groups':[{'passed':actor.item()==1.}]}
    scale,_=backtrack_actor([actor],[torch.tensor([1.])],[torch.tensor([3.])],check)
    assert scale==0. and actor.item()==1.


def test_invalid_last_accepted_state_cannot_be_declared_retained():
    import torch
    actor=torch.nn.Parameter(torch.tensor([1.]))
    with pytest.raises(ValueError,match='Last accepted'):
        backtrack_actor([actor],[torch.tensor([1.])],[torch.tensor([3.])],lambda:{'basic_teacher_groups':[{'passed':False}]})


def recipe(constrained):
    result={'steps_each':512,'additional_supervised_steps_total':512 if constrained else 1024,
        'new_controls':0,'PPO_updates':0,'results':[{'arm':'trajectory_correction'}] if constrained else [{'arm':'replay'},{'arm':'trajectory_correction'}]}
    if constrained:result.update(retained_correction_profile='basic_teacher_group_backtracking_v1',
        retention_gate={'passed':True},new_supervised_updates=512,reused_supervised_updates=0)
    return result


@pytest.mark.parametrize('constrained',[False,True])
def test_legacy_pair_and_new_single_fit_preserve_their_actual_update_counts(constrained):
    from scripts.prepare_native_corrective_probe_pair import require_fit_result
    data=recipe(constrained)
    assert require_fit_result(data)==data


@pytest.mark.parametrize('fault',['failed_gate','missing_gate','extra_arm'])
def test_rejected_retention_fit_cannot_be_staged_for_driving(fault):
    from scripts.prepare_native_corrective_probe_pair import require_fit_result
    data=recipe(True)
    if fault=='failed_gate':data['retention_gate']['passed']=False
    elif fault=='missing_gate':data.pop('retention_gate')
    else:data['results'].insert(0,{'arm':'replay'})
    with pytest.raises(ValueError):require_fit_result(data)


def test_new_paired_label_profile_must_also_pass_its_optional_retention_gate():
    from scripts.prepare_native_corrective_probe_pair import require_fit_result
    data=recipe(False);data['require_basic_retention']=True;data['retention_gate']={'passed':False}
    with pytest.raises(ValueError):require_fit_result(data)
    data['retention_gate']['passed']=True
    assert require_fit_result(data)==data


def test_executed_teacher_only_gate_keeps_the_same_basic_ceilings_without_query_targets():
    data=arrays();gate=retention_gate(*data[:4],basic_only=True)
    assert gate['passed'] and gate['query_constraint_applied'] is False
    assert 'query_MSE' not in gate
    ordinary=retention_gate(*data)
    assert gate['basic_teacher_groups']==ordinary['basic_teacher_groups']


@pytest.mark.parametrize('passed',[True,False])
def test_clean_teacher_single_fit_must_pass_before_native_preparation(passed):
    from scripts.prepare_native_corrective_probe_pair import require_fit_result
    data={'steps_each':512,'additional_supervised_steps_total':512,'new_controls':0,'PPO_updates':0,
        'results':[{'arm':'teacher_only_cleaning'}],'executed_teacher_only_profile':'verified_measured_groups0to11_v1',
        'retention_gate':{'passed':passed},'new_supervised_updates':512,'reused_supervised_updates':0}
    if passed:assert require_fit_result(data)==data
    else:
        with pytest.raises(ValueError):require_fit_result(data)
