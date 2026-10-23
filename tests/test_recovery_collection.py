import numpy as np
from navstack.config import EnvConfig, PlannerConfig
from navstack.imitation import collect_recovery_episode


class ConstantLearner:
    def predict(self, observation, deterministic):
        return np.array([[.5, .1]], dtype=np.float32), None


class IdentityNormalizer:
    def normalize_obs(self, obs):
        return obs.copy()


def test_collection_executes_declared_source_and_keeps_teacher_labels():
    cfg = EnvConfig(method='B1', flat_hold=1, scenario='empty', obstacle_num=0,
                    subgoal_mode='goal_frame_terminal', deadline=45,
                    planner=PlannerConfig(max_seconds=None, analytic_expansion=True, vectorized_primitives=True))
    for probability in (0., 1.):
        rows = []
        result = collect_recovery_episode(cfg, ConstantLearner(), IdentityNormalizer(), 8101,
                    probability, 42, options={'diagnostic_route_length':6.}, callback=rows.append)
        assert len(rows) == result['teacher_queries'] == result['metrics']['physics_steps']
        assert result['teacher_executed'].all() if probability else not result['teacher_executed'].any()
        assert result['metrics']['success'] if probability else not result['metrics']['success']
        for i, row in enumerate(rows):
            expected = row['teacher_action'] if probability else row['learner_action']
            np.testing.assert_array_equal(row['executed_action'], expected)
            np.testing.assert_array_equal(result['actions'][i], row['teacher_action'])
            assert row['info']['trace'][0]['state']['time'] > row['state_before']['time']
        if not probability:
            assert any(not np.array_equal(r['learner_action'], r['teacher_action']) for r in rows)


def test_current_state_teacher_queries_do_not_change_mixture_schedule():
    cfg = EnvConfig(method='B1',flat_hold=1,scenario='empty',obstacle_num=0,deadline=1.,
        subgoal_mode='goal_frame_terminal',planner=PlannerConfig(max_seconds=None,analytic_expansion=True,vectorized_primitives=True))
    rows=[]
    result=collect_recovery_episode(cfg,ConstantLearner(),IdentityNormalizer(),8101,.5,42,
        callback=rows.append,teacher_mode='current_state')
    assert result['teacher_decisions']==result['teacher_queries']==20
    rng=np.random.default_rng(42)
    expected=np.repeat([rng.random()<.5 for _ in range(2)],10)
    np.testing.assert_array_equal(result['teacher_executed'],expected)
