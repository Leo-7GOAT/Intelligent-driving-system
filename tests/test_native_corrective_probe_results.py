from copy import deepcopy
import pytest
from scripts.run_native_corrective_probes import summarize_probe,require_teacher_completion


def inputs():
    expected={'model_sha256':'model','normalization_sha256':'norm'}
    evaluation={**expected,'episodes':[{'case_id':i,'success':i!=1,
        'termination_reason':'out_of_bounds' if i==1 else 'success','physics_steps':100+i} for i in range(4)]}
    return evaluation,{'physics_steps':0,'optimizer_steps':0},{'training_control_frames_verified':0,'total_applied_controls_verified':406},expected


def test_three_successes_with_bounds_still_fail_unchanged_gate_and_keep_failed_case():
    result=summarize_probe(*inputs())
    assert result['successes']==3 and not result['basic_gate_passed']
    assert result['unsafe_cases']==1 and result['case_results'][1]['termination_reason']=='out_of_bounds'


@pytest.mark.parametrize('change',['missing','duplicate','learned','changed_model'])
def test_incomplete_or_changed_probe_is_rejected(change):
    evaluation,summary,audit,expected=deepcopy(inputs())
    if change=='missing':evaluation['episodes'].pop()
    if change=='duplicate':evaluation['episodes'][3]['case_id']=2
    if change=='learned':summary['optimizer_steps']=1
    if change=='changed_model':evaluation['model_sha256']='other'
    with pytest.raises(ValueError):summarize_probe(evaluation,summary,audit,expected)


def test_failed_or_partial_predecessor_cannot_start_driving():
    state={'phase':'ALL_FIVE_PRISTINE_BASELINES_AND_PPO_ATTRIBUTION_COMPLETE',
        'completed':[{'seed':s} for s in [11,22,33,44,55]]}
    require_teacher_completion(state,False)
    with pytest.raises(ValueError):require_teacher_completion(state,True)
    state['completed'].pop()
    with pytest.raises(ValueError):require_teacher_completion(state,False)
