from copy import deepcopy
import pytest
from scripts.prepare_native_pristine_baselines import baseline_protocol
from scripts.run_native_pristine_baselines import require_pristine_preparation,require_training_completion
from scripts.analyze_native_pristine_baselines import compare_cases,complete_cases


def original():
    return {'methods':['H','H0'],'method_budgets':{'H':20480,'H0':20480},
        'native_case_manifest':{'cases':list(range(36))},
        'native_validation':{'manifest':{'cases':list(range(24))}},
        'recovery':{'H':{'lineage_controls':0,'retained_optimizer_steps':0,'checkpoint':'resume/H'},
                    'H0':{'lineage_controls':0,'retained_optimizer_steps':0}},
        'extra_supervision':{'supervised_optimizer_steps_per_method':896,'dataset_sha256':'teacher'},
        'resource_limits':{'wall_seconds':7200},'freeze_observation_statistics':True,
        'broader_feedback_replication':{'requested_new_controls_each_method':20480}}


def test_teacher_only_protocol_keeps_all_conditions_and_initializer_without_extra_updates():
    old=original();before=deepcopy(old);p=baseline_protocol(old,'paired')
    assert old==before and p['methods']==['H'] and p['evaluation_only'] is True
    assert p['controlled_budget_per_method']==0 and p['method_budgets']=={'H':0}
    assert p['recovery']=={'H':old['recovery']['H']}
    assert p['native_case_manifest']==old['native_case_manifest'] and p['native_validation']==old['native_validation']
    assert p['extra_supervision']['dataset_sha256']=='teacher'
    assert p['pristine_policy_baseline']['teacher_supervised_steps_inherited']==896
    assert p['pristine_policy_baseline']['additional_supervised_updates']==0
    assert p['pristine_policy_baseline']['additional_PPO_updates']==0
    assert 'freeze_observation_statistics' not in p and 'broader_feedback_replication' not in p
    p['native_validation']['manifest']['cases'].pop()
    assert len(old['native_validation']['manifest']['cases'])==24


@pytest.mark.parametrize('bad',['trained','optimizer','omitted_cases','wrong_budget','evaluation_parent'])
def test_wrong_lineage_or_incomplete_matched_parent_cannot_be_teacher_only_control(bad):
    old=original()
    if bad=='trained':old['recovery']['H']['lineage_controls']=1
    if bad=='optimizer':old['recovery']['H']['retained_optimizer_steps']=1
    if bad=='omitted_cases':old['native_validation']['manifest']['cases'].pop()
    if bad=='wrong_budget':old['method_budgets']['H']=10240
    if bad=='evaluation_parent':old['evaluation_only']=True
    with pytest.raises(ValueError):baseline_protocol(old,'paired')


def registered():
    return {'phase':'ALL_FIVE_PRISTINE_H_BASELINES_PREPARED_NOT_DRIVEN','distinct_pristine_actor_initializations':5,
        'batches':[{'seed':s,'batch':str(s),'paired_training_batch':'paired'+str(s),'actor_parameter_sha256':str(s)} for s in (11,22,33,44,55)]}


def test_all_pristine_seeds_and_completed_PPO_dependency_are_mandatory():
    state=registered();assert len(require_pristine_preparation(state))==5
    for key in ('batch','paired_training_batch','actor_parameter_sha256'):
        bad=deepcopy(state);bad['batches'][1][key]=bad['batches'][0][key]
        with pytest.raises(ValueError):require_pristine_preparation(bad)
    state['batches'].pop()
    with pytest.raises(ValueError):require_pristine_preparation(state)
    done={'phase':'ALL_FIVE_MATCHED_BROADER_FEEDBACK_PPO_PAIRS_COMPLETE','completed':[{'seed':s} for s in (11,22,33,44,55)]}
    require_training_completion(done,False)
    with pytest.raises(ValueError):require_training_completion(done,True)
    done['completed'].pop()
    with pytest.raises(ValueError):require_training_completion(done,False)


def outcomes():
    return [{'case_id':i,'success':True,'termination_reason':'success','simulation_seconds':10.,'interventions':0} for i in range(24)]


def test_early_failure_cannot_create_a_successful_travel_time_advantage():
    left,right=outcomes(),outcomes()
    left[0].update(success=False,termination_reason='collision',simulation_seconds=1.)
    right[1]['simulation_seconds']=12.
    report=compare_cases(left,right)
    assert report['safe_completion_difference']==-1/24
    assert report['left_unsafe_episodes']==1 and report['right_unsafe_episodes']==0
    assert report['mutually_successful_case_ids']==list(range(1,24))
    assert report['mutually_successful_mean_time_difference_seconds']==-2/23
    right[0].update(success=False,termination_reason='deadline',simulation_seconds=45.)
    assert compare_cases(left,right)['mutually_successful_mean_time_difference_seconds']==-2/23


def test_incomplete_duplicate_or_contradictory_outcomes_are_rejected():
    rows=outcomes();assert len(complete_cases(rows))==24
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError):complete_cases(bad)
    rows[1]['case_id']=0
    with pytest.raises(ValueError):complete_cases(rows)
    rows=outcomes();rows[0]['success']=False
    with pytest.raises(ValueError):complete_cases(rows)
