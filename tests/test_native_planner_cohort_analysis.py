from copy import deepcopy
import pytest
from scripts.analyze_native_planner_cohort import require_complete_cohort, verify_planner_change


@pytest.mark.parametrize('change', ['incomplete', 'omitted', 'duplicate'])
def test_cannot_infer_from_a_running_or_selected_seed_cohort(change):
    state = {'phase': 'ALL_FIVE_REGISTERED_B2_PPO_SEEDS_COMPLETE',
             'completed': [{'seed': s} for s in (11, 22, 33, 44, 55)]}
    if change == 'incomplete': state['phase'] = 'RUNNING_REGISTERED_B2_PPO'
    elif change == 'omitted': state['completed'].pop()
    else: state['completed'][-1]['seed'] = 11
    with pytest.raises(ValueError): require_complete_cohort(state)


@pytest.mark.parametrize('change', ['reward', 'controller', 'legacy_grid'])
def test_planner_ablation_rejects_other_unmatched_capabilities(change):
    h = {'method': 'H', 'planner': {'grid_postprocess': 'legacy'}, 'reward': 1, 'mpc': 20}
    b = deepcopy(h); b['method'] = 'B2'; b['planner']['grid_postprocess'] = 'body_spline_v1'
    verify_planner_change(h, b)
    if change == 'reward': b['reward'] = 2
    elif change == 'controller': b['mpc'] = 30
    else: b['planner']['grid_postprocess'] = 'legacy'
    with pytest.raises(ValueError): verify_planner_change(h, b)
