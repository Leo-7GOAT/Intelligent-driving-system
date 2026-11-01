from copy import deepcopy
import pytest
from scripts.run_native_camera_after_training import require_training_barrier


def failed_gate():
    lower={'phase':'ALL_FIVE_BASIC_PPO_COURSES_COMPLETE_WITH_FAILED_GATES',
        'completed':[{'seed':s,'basic_qualified_after_learning':s!=11} for s in (11,22,33,44,55)]}
    full={'created':[],'completed':[]}
    failure={'created':[],'completed':[],
        'error':"ValueError('All five actually trained basic courses must qualify;no seed omission or gate waiver')"}
    return lower,full,failure


def test_explicit_childless_research_failure_does_not_disguise_infrastructure_failure():
    lower,full,failure=failed_gate()
    assert require_training_barrier(lower,full,None,failure)['full_task_skipped_for_unchanged_failed_gate']
    with pytest.raises(ValueError):require_training_barrier(lower,full,{'error':'worker failed'},failure)
    bad=deepcopy(failure);bad['error']='unknown failure'
    with pytest.raises(ValueError):require_training_barrier(lower,full,None,bad)
    full['created']=[{'batch':'still_owned'}]
    with pytest.raises(ValueError):require_training_barrier(lower,full,None,failure)


def test_qualified_courses_require_the_actual_complete_full_task_cohort():
    lower,full,_=failed_gate()
    lower['phase']='ALL_FIVE_BASIC_PPO_COURSES_QUALIFIED_FOR_FULL_TASK_TRAINING'
    for row in lower['completed']:row['basic_qualified_after_learning']=True
    with pytest.raises(ValueError):require_training_barrier(lower,full,None,None)
    full={'phase':'ALL_FIVE_FULL_TASK_PPO_AFTER_BASIC_COURSE_COMPLETE',
          'completed':[{'seed':s} for s in (11,22,33,44,55)]}
    assert require_training_barrier(lower,full,None,None)['basic_qualified_all_five']
    full['completed'].pop()
    with pytest.raises(ValueError):require_training_barrier(lower,full,None,None)
