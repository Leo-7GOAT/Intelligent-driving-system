from copy import deepcopy
import pytest
from scripts.continue_native_hierarchical_pair import continuation_protocol


def inputs():
    return (dict(methods=['H_PPO', 'H0_PPO', 'H_teacher'], evaluation_only=True,
                 jobs=[{'name': 'old'}], frozen_policy_comparison={'version': 'named_frozen_native_v1'},
                 goal_parking_assist=True, planner_nodes=7200, task_progress_weight=.5,
                 source_hashes={'frozen.py': 'unchanged'}),
            dict(routes=list(range(16)), native_case_manifest={'cases': list(range(48))}),
            dict(routes=list(range(8)), manifest={'cases': list(range(24))}))


def test_pair_continuation_keeps_physics_and_all_conditions_without_teacher_updates():
    old, inventory, validation = inputs()
    before = deepcopy((old, inventory, validation))
    p = continuation_protocol(old, inventory, validation)
    assert (old, inventory, validation) == before
    assert p['methods'] == ['H', 'H0']
    assert p['method_budgets'] == {'H': 10240, 'H0': 10240}
    assert p['goal_parking_assist'] and p['planner_nodes'] == 7200
    assert p['task_progress_weight'] == old['task_progress_weight']
    assert p['source_hashes'] == old['source_hashes']
    assert p['freeze_observation_statistics'] and not p['evaluation_only']
    assert p['native_validation'] == validation
    assert 'jobs' not in p and 'frozen_policy_comparison' not in p
    assert p['native_case_manifest']['cases'] == list(range(48))


@pytest.mark.parametrize('fault', ['missing_train', 'missing_eval', 'wrong_source'])
def test_pair_cannot_start_with_missing_conditions_or_another_source(fault):
    old, inventory, validation = inputs()
    if fault == 'missing_train':
        inventory['native_case_manifest']['cases'].pop()
    elif fault == 'missing_eval':
        validation['manifest']['cases'].pop()
    else:
        old['methods'].pop()
    with pytest.raises(ValueError):
        continuation_protocol(old, inventory, validation)
