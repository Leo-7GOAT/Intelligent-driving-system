from copy import deepcopy
import pytest
from scripts.prepare_native_full_task_after_course import require_all_courses,full_protocol


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
