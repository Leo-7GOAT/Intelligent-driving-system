from copy import deepcopy
import numpy as np
import pytest
from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv, rule_action
from navstack.imitation import RecoveryTeacher


def test_teacher_query_preserves_actual_state_and_reproduces_executed_control():
    cfg = EnvConfig(method='B3', scenario='empty', obstacle_num=0,
                    subgoal_mode='goal_frame_terminal', parking=True,
                    planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))
    teacher = NavigationEnv(cfg)
    teacher.reset(seed=8101, options={'diagnostic_route_length':6.})
    expected = []
    while not teacher.done:
        _, _, _, _, info = teacher.step(rule_action(teacher))
        expected.extend(info['trace'])
    assert info['episode_metrics']['success']
    teacher.close()
    cfg.method, cfg.flat_hold = 'B1', 1
    direct = NavigationEnv(cfg)
    obs, _ = direct.reset(seed=8101, options={'diagnostic_route_length':6.})
    oracle = RecoveryTeacher(direct)
    for index, tick in enumerate(expected):
        state, metrics = deepcopy(direct.state), deepcopy(direct.episode)
        label, audit = oracle.query()
        assert direct.state == state and direct.episode == metrics
        np.testing.assert_array_equal(obs, direct._observation())
        assert audit['control'] == index and audit['query_seconds'] >= 0
        if index == 0:
            with pytest.raises(ValueError, match='exactly once'):
                oracle.query()
        obs, _, _, _, result = direct.step(label)
        actual = result['trace'][0]['state']
        differences = {k: actual[k]-tick['state'][k] for k in actual}
        assert max(abs(v) for v in differences.values()) < 1e-4, (index, differences, audit['requested_command'], tick['requested_command'])
    assert direct.done and result['episode_metrics']['success']
    direct.reset(seed=8102)
    with pytest.raises(ValueError, match='new teacher'):
        oracle.query()
    direct.close()
