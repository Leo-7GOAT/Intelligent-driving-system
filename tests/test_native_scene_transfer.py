from copy import deepcopy
import hashlib
import json

import pytest

from navstack.config import VehicleConfig
from scripts.native_road_learning import registered_native_cases, training_config
from scripts.prepare_native_scene_transfer import transfer_protocol


def inputs():
    route = {'map': 'fixture', 'opendrive_sha256': 'fixture-map',
             'points': [{'x': float(i), 'y': 0., 'z': 0., 'yaw_degrees_carla': 0.}
                        for i in range(26)]}
    route['geometry_id'] = hashlib.sha256(json.dumps([
        [round(p[k], 3) for k in ('x', 'y', 'yaw_degrees_carla')]
        for p in route['points']]).encode()).hexdigest()
    old = {'methods': ['H', 'H0', 'B1'], 'routes': [route] * 4,
           'native_hazard': {'version': 'native_hazard_v1', 'kind': 'static_vehicle',
                             'distance_m': 10., 'static_geometry': 'projected_box_strips12_v1'},
           'resource_limits': {'wall_seconds': 7200, 'disk_min_GiB': 12, 'automatic_retries': 0},
           'episode_seconds': 45., 'road_candidate_policy': True,
           'continuation_feedback': True, 'feedback_ablation': True,
           'subgoal_mode': 'route_frame_terminal', 'reference_reuse': True,
           'route_goal_candidate': True, 'task_progress_weight': .5, 'unsafe_event_penalty': -90.,
           'own_success_replay': {'dataset': 'never-load-this.npz'}}
    reference = {'routes': deepcopy(old['routes']),
                 'native_hazard': {'version': 'native_hazard_v1', 'kind': 'crossing_walker',
                                   'distance_m': 10., 'lateral_m': -2.5}}
    lineage = {m: {'checkpoint': 'resume/' + m, 'lineage_controls': 20480,
                   'consumed_controls_lower_bound': 22406, 'retained_optimizer_steps': 640,
                   'unknown_unflushed_parent_tail': True} for m in old['methods']}
    return old, reference, lineage


def test_scene_transfer_keeps_task_semantics_and_lineage_without_training_or_replay():
    old, reference, lineage = inputs()
    before = deepcopy((old, reference, lineage))
    protocol = transfer_protocol(old, reference, lineage, 100., 1800)
    assert (old, reference, lineage) == before
    assert protocol['evaluation_only'] and protocol['controlled_budget_per_method'] == 0
    assert protocol['method_budgets'] == {'H': 0, 'H0': 0, 'B1': 0}
    assert protocol['checkpoint_thresholds'] == [] and 'own_success_replay' not in protocol
    assert protocol['recovery'] == lineage
    assert protocol['created_epoch'] == 100. and protocol['deadline_epoch'] == 1900.
    for method in old['methods']:
        assert training_config(old, method, VehicleConfig()) == training_config(protocol, method, VehicleConfig())
    cases = registered_native_cases(protocol)
    assert len(cases) == 4 and all(c['native_hazard'] == reference['native_hazard'] for c in cases)
    protocol['native_hazard']['lateral_m'] = -3.
    assert reference == before[1]


@pytest.mark.parametrize('change', ['same_scene', 'changed_route', 'invalid_spawn', 'missing_lineage',
                                  'evaluation_parent', 'previous_transfer', 'named_jobs'])
def test_transfer_rejects_unregistered_changes_and_missing_provenance(change):
    old, reference, lineage = inputs()
    if change == 'same_scene': reference['native_hazard'] = deepcopy(old['native_hazard'])
    if change == 'changed_route': reference['routes'][0]['points'][0]['x'] += 1.
    if change == 'invalid_spawn': reference['native_hazard']['distance_m'] = 2.
    if change == 'missing_lineage': lineage.pop('B1')
    if change == 'evaluation_parent': old['evaluation_only'] = True
    if change == 'previous_transfer': old['scene_transfer'] = {'split': 'development'}
    if change == 'named_jobs': old['jobs'] = [{'name': 'different', 'method': 'H'}]
    with pytest.raises(ValueError): transfer_protocol(old, reference, lineage, 100., 1800)
