import ast
from copy import deepcopy
from pathlib import Path
import pytest
from navstack.config import EnvConfig, VehicleConfig
from navstack.experiments import signature
from scripts.prepare_native_shared_brake import deployment_record, environment_overlay, worker_overlay, VERSION


def patched_configuration():
    import scripts.native_road_learning as runtime
    text = worker_overlay(Path(runtime.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'training_config')
    namespace = dict(runtime.__dict__)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<scoped-deployment-config>', 'exec'), namespace)
    return runtime.training_config, namespace['training_config']


@pytest.mark.parametrize('method', ['H', 'H0', 'B1', 'B2'])
@pytest.mark.parametrize('enabled', [False, True])
def test_actual_configuration_overlay_changes_only_brake_flag(method, enabled):
    from test_native_case_manifest import protocol
    original, patched = patched_configuration()
    p = {**protocol(), 'episode_seconds': 45}
    expected = original(p, method, VehicleConfig()).to_dict()
    p.update(evaluation_only=True, controlled_budget_per_method=0,
             deployment_shared_brake={'version': VERSION, 'enabled': enabled})
    actual = patched(p, method, VehicleConfig()).to_dict()
    expected['shared_shield'] = enabled
    assert actual == expected


@pytest.mark.parametrize('change', [
    {'evaluation_only': False}, {'controlled_budget_per_method': 1280},
    {'deployment_shared_brake': {'version': VERSION, 'enabled': 'false'}},
    {'deployment_shared_brake': {'version': 'unknown', 'enabled': True}},
    {'deployment_shared_brake': None},
])
def test_configuration_rejects_training_or_unregistered_safety_activation(change):
    from test_native_case_manifest import protocol
    _, patched = patched_configuration()
    p = {**protocol(), 'episode_seconds': 45, 'evaluation_only': True,
         'controlled_budget_per_method': 0,
         'deployment_shared_brake': {'version': VERSION, 'enabled': True}}
    p.update(change)
    with pytest.raises(ValueError, match='zero-update'):
        patched(p, 'H', VehicleConfig())


def test_common_brake_does_not_disable_native_trajectory_conflict_rejection():
    import navstack.env as module
    text = environment_overlay(Path(module.__file__).read_text(encoding='utf-8-sig'))
    function = next(n for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.FunctionDef) and n.name == '_prepare_reference')
    calls = []
    namespace = dict(module.__dict__)
    namespace['dynamic_safe'] = lambda *args: calls.append(True) or False
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<retained-native-gate>', 'exec'), namespace)
    env = module.NavigationEnv(EnvConfig(obstacle_num=0, parking=False, shared_shield=True))
    try:
        env.reset(seed=17)
        mode, detail = namespace['_prepare_reference'](env, 5)
        assert calls and mode == 'dynamic_brake'
        assert detail['failure_reason'] == 'constant_velocity_spacetime_conflict'
    finally:
        env.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_saved_deployment_configuration_keeps_physics_and_original_record(enabled):
    cfg = EnvConfig(method='B1', flat_hold=1)
    record = {'env': cfg.to_dict(), 'signature': signature(cfg), 'seed': 11}
    original = deepcopy(record)
    result = deployment_record(record, enabled)
    expected = deepcopy(record['env']);expected['shared_shield'] = enabled
    assert result['env'] == expected and record == original
    assert result['signature'] == signature(EnvConfig.from_dict(expected))


def test_rejects_already_wrapped_parent_and_unknown_source_structures():
    cfg = EnvConfig(shared_shield=True)
    with pytest.raises(ValueError, match='native safety'):
        deployment_record({'env': cfg.to_dict()}, True)
    with pytest.raises(ValueError, match='boolean'):
        deployment_record({'env': EnvConfig().to_dict()}, 1)
    with pytest.raises(ValueError, match='structure'):
        environment_overlay('def unrelated(): pass\n')
    with pytest.raises(ValueError, match='configuration function'):
        worker_overlay('def unrelated(): pass\n')


@pytest.mark.parametrize('live,lock_present', [(False, True), (True, True), (False, False), (True, False)])
def test_closed_sibling_failure_requires_verified_process_exit(tmp_path, monkeypatch, live, lock_present):
    import json
    import scripts.prepare_native_shared_brake as module
    (tmp_path/'infrastructure').mkdir()
    evidence = {'guard_final.json': {'exit_code': 1, 'epoch': 1.},
                'guard.json': {'pid': 111}, 'worker_process.json': {'pid': 222},
                'infrastructure/server_stop.json': {'status': 'owned_empty_server_stopped'}}
    for name, value in evidence.items():
        (tmp_path/name).write_text(json.dumps(value))
    if lock_present:
        (tmp_path/'run_registered_native.lock').touch()
    monkeypatch.setattr(module, 'recorded_lifecycle_gone', lambda ids, ended: not live)
    if live:
        with pytest.raises(ValueError, match='still live'):
            module.require_closed_method_parent(tmp_path)
    else:
        # This allows verification of the completed METHOD,not acceptance of its
        # failed sibling;prepare still checks all case/raw/model counters.
        module.require_closed_method_parent(tmp_path)
