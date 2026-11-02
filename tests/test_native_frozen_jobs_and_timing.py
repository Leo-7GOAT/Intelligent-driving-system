from copy import deepcopy
import pytest
from navstack.metrics import timing_summary
from scripts.native_road_learning import registered_jobs
from scripts.audit_native_method_endpoint import verified_decision_timing


def protocol():
    return dict(frozen_policy_comparison={'version':'named_frozen_native_v1'},evaluation_only=True,
        jobs=[dict(name='H',method='H'),dict(name='H_teacher',method='H'),dict(name='H0',method='H0')],
        methods=['H','H_teacher','H0'],controlled_budget_per_method=0,
        method_budgets={'H':0,'H_teacher':0,'H0':0},recovery={'H':{},'H_teacher':{},'H0':{}})


def test_named_policy_comparison_does_not_require_an_unrelated_training_intervention():
    p=protocol();assert registered_jobs(p)==p['jobs']


@pytest.mark.parametrize('fault',['budget','train','missing','duplicate','intervention'])
def test_frozen_comparison_rejects_learning_and_unbound_models(fault):
    p=protocol()
    if fault=='budget':p['method_budgets']['H']=1
    elif fault=='train':p['evaluation_only']=False
    elif fault=='missing':p['recovery'].pop('H')
    elif fault=='duplicate':p['jobs'][1]['name']='H'
    else:p['gradient_clipping_experiment']=True
    with pytest.raises(ValueError):registered_jobs(p)


def test_external_timing_uses_actual_direct_control_period_and_requires_every_clock():
    rows=[dict(policy_prediction_seconds=.002,full_decision_seconds=.08),
          dict(policy_prediction_seconds=.003,full_decision_seconds=.04)]
    episode=dict(policy_prediction_times=timing_summary([.002,.003],.05),full_decision_times=timing_summary([.08,.04],.05))
    assert verified_decision_timing(rows,episode,dict(method='B1',flat_hold=1))=={'policy_prediction_times','full_decision_times'}
    wrong=deepcopy(episode);wrong['full_decision_times']=timing_summary([.08,.04],.5)
    with pytest.raises(ValueError):verified_decision_timing(rows,wrong,dict(method='B1',flat_hold=1))
    wrong_rows=deepcopy(rows);wrong_rows[0].pop('full_decision_seconds')
    with pytest.raises(ValueError):verified_decision_timing(wrong_rows,episode,dict(method='B1'))
    assert verified_decision_timing([{}],{},dict(method='B1'))==set()
