import gzip
import json

import pytest

from navstack.config import EnvConfig, PlannerConfig
from navstack.env import NavigationEnv
from navstack.native_route import sample_road_grid
from navstack.road_grid import RoadGrid
from navstack.vehicle import VehiclePlant, VehicleState
from navstack.world import World
from scripts.evaluate_native_rule import require_finished_parent, evaluate_cases


def test_rule_preparation_requires_normal_parent_and_server_cleanup(tmp_path):
    with pytest.raises(ValueError, match='finish first'):
        require_finished_parent(tmp_path)
    (tmp_path/'infrastructure').mkdir()
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value))
    write('summary.json', {'status': 'complete'})
    write('guard_final.json', {'exit_code': 0, 'reason': None})
    write('infrastructure/server_stop.json', {'status': 'owned_empty_server_stopped'})
    require_finished_parent(tmp_path)
    write('guard_final.json', {'exit_code': 0, 'reason': 'wall_deadline'})
    with pytest.raises(ValueError, match='normally'):
        require_finished_parent(tmp_path)
    write('guard_final.json', {'exit_code': 0, 'reason': None})
    write('failure.json', {'error': 'preserved'})
    with pytest.raises(ValueError, match='normally'):
        require_finished_parent(tmp_path)


class LightweightFixture(NavigationEnv):
    """Exercise runner bookkeeping only; never native/CARLA evidence."""
    def _create_episode(self, options=None):
        road = sample_road_grid({'width': 32., 'height': 32., 'geometry_id': 'rule-fixture'}, lambda x,y: True)
        self.scene_evidence = {'width': 32., 'height': 32., 'road_grid': road}
        return World(32., 32., road=RoadGrid(road)), VehiclePlant(
            self.config.vehicle, VehicleState(6., 16.)), (24., 16., 0.), 'fixture'


def test_rule_runner_preserves_failures_costs_and_all_applied_controls(tmp_path):
    config = EnvConfig(deadline=.5, planner=PlannerConfig(max_seconds=None, max_nodes=32, analytic_expansion=True))
    env = LightweightFixture(config)
    checks = []
    try:
        episodes, controls = evaluate_cases(env, [{}, {}], tmp_path, lambda: checks.append(True))
    finally:
        env.close()
    assert len(episodes) == 2 and len(checks) == 2
    assert controls == sum(r['physics_steps'] for r in episodes) == 20
    assert all(r['termination_reason'] == 'deadline' and not r['success'] for r in episodes)
    assert all(r['rule_selection_times']['count'] == r['actions'] for r in episodes)
    with gzip.open(tmp_path/'evaluation_trace.jsonl.gz', 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    assert sum(len(r['info']['trace']) for r in rows) == controls
    assert all(r['policy_kind'] == 'rule_only_B3' and r['full_decision_seconds'] >= r['selection_seconds'] for r in rows)
    assert {r['case_id'] for r in rows} == {0, 1}
