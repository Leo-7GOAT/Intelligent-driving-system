"""Fabricated fixtures test evidence rejection; no real CARLA execution is claimed."""
import gzip
import json

import pytest

from navstack.experiments import write_json
from navstack.metrics import timing_summary
from scripts.audit_native_rule import audit, sha


def write_gzip(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')


def fixture(batch):
    for folder in ('source', 'calibration', 'infrastructure'):
        (batch/folder).mkdir()
    (batch/'source/fixture.py').write_text('pass\n')
    write_json(batch/'calibration/calibration.json', {})
    config = {'method': 'H', 'decision_dt': .5}
    write_json(batch/'source_config.json', {'env': config})
    write_json(batch/'protocol.json', {'config': config, 'routes': [{'geometry_id': 'road'}],
        'source_hashes': {'fixture.py': sha(batch/'source/fixture.py')},
        'source_config_sha256': sha(batch/'source_config.json'),
        'calibration_sha256': sha(batch/'calibration/calibration.json'),
        'training_physics_steps': 0, 'optimizer_updates': 0})
    response = {'episode_id': 'episode', 'request_id': 2, 'frame': 31, 'imu_frame': 31,
                'state': {'x': 5., 'time': .05}, 'command': {'acceleration': 0., 'steering': 0.},
                'vehicle_control': {'throttle': 0.}, 'collisions': []}
    write_gzip(batch/'infrastructure/bridge_fixture_responses.jsonl.gz', [
        {'response': {'episode_id': 'episode', 'request_id': 1, 'reset_ticks': 30}}, {'response': response}])
    initial = {'road_sha256': 'fixture_hash'}
    terminal = {'physics_steps': 1, 'actions': 1, 'success': False, 'termination_reason': 'deadline'}
    tick = {'state': response['state'], 'command': response['command'], 'backend_frame': 31,
            'backend_evidence': {k: response[k] for k in ('episode_id', 'request_id', 'frame', 'imu_frame', 'collisions')}}
    row = {'case_id': 0, 'policy_kind': 'rule_only_B3', 'policy_action': 0,
           'selection_seconds': .01, 'full_decision_seconds': .02,
           'info': {'initial_scenario': initial, 'trace': [tick], 'executed_substeps': 1,
                    'termination_reason': 'deadline', 'episode_metrics': terminal}}
    write_gzip(batch/'evaluation_trace.jsonl.gz', [row])
    write_json(batch/'scene_0000.json', {'initial': initial, 'scene': {
        'native_layout': {'geometry_id': 'road'}, 'road_grid': {'sha256': 'fixture_hash'}}})
    write_json(batch/'summary.json', {'status': 'complete', 'policy_kind': 'rule_only_B3',
        'training_physics_steps': 0, 'optimizer_updates': 0, 'evaluation_controls': 1,
        'reset_ticks': 30, 'world_ticks': 31, 'episodes': [{'case_id': 0, **terminal,
            'rule_selection_times': timing_summary([.01], .5), 'full_decision_times': timing_summary([.02], .5)}]})
    return row


def test_rule_audit_preserves_failure_and_zero_training(tmp_path):
    fixture(tmp_path)
    result = audit(tmp_path)
    assert result['evaluation_controls_verified'] == 1
    assert result['world_ticks_verified'] == 31
    assert result['termination_counts'] == {'deadline': 1}
    assert result['training_physics_steps'] == result['optimizer_updates'] == 0


@pytest.mark.parametrize('change,match', [('state','measured CARLA state'), ('duplicate','after termination'),
                                         ('timing','timing summary'), ('training','cannot claim PPO'),
                                         ('inventory','case inventory')])
def test_rule_audit_rejects_corrupted_evidence(tmp_path, change, match):
    row = fixture(tmp_path)
    summary_path = tmp_path/'summary.json'
    summary = json.loads(summary_path.read_text())
    if change == 'state':
        row['info']['trace'][0]['state']['x'] = 99.
        write_gzip(tmp_path/'evaluation_trace.jsonl.gz', [row])
    elif change == 'duplicate':
        write_gzip(tmp_path/'evaluation_trace.jsonl.gz', [row, row])
    elif change == 'timing':
        summary['episodes'][0]['full_decision_times']['p95'] = 0.
    elif change == 'training':
        summary['optimizer_updates'] = 10
    else:
        summary['episodes'] = []
    write_json(summary_path, summary)
    with pytest.raises(ValueError, match=match):
        audit(tmp_path)


def test_rule_audit_rejects_unregistered_speed_selection(tmp_path):
    row = fixture(tmp_path)
    protocol=json.loads((tmp_path/'protocol.json').read_text())
    protocol['rule_speed_policy']='paired_low_speed'
    write_json(tmp_path/'protocol.json',protocol)
    row.update(rule_proposal=2,rule_speed_policy='paired_low_speed',policy_action=2)
    write_gzip(tmp_path/'evaluation_trace.jsonl.gz',[row])
    with pytest.raises(ValueError,match='lower-speed selection'):
        audit(tmp_path)


def test_completed_prefix_keeps_missing_cases_and_requires_terminal_infrastructure_error(tmp_path):
    fixture(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    write_json(tmp_path/'case_0000.json',summary['episodes'][0])
    (tmp_path/'summary.json').unlink()
    protocol=json.loads((tmp_path/'protocol.json').read_text())
    protocol['routes'].append({'geometry_id':'unstarted-road'})
    write_json(tmp_path/'protocol.json',protocol)
    write_json(tmp_path/'failure.json',{'error':'RuntimeError: fixture spawn failure'})
    write_json(tmp_path/'guard_final.json',{'exit_code':1})
    write_json(tmp_path/'infrastructure/server_stop.json',{'status':'owned_empty_server_stopped'})
    path=tmp_path/'infrastructure/bridge_fixture_responses.jsonl.gz'
    with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
    with pytest.raises(FileNotFoundError):audit(tmp_path)
    with pytest.raises(ValueError,match='terminal infrastructure error'):audit(tmp_path,True)
    rows.append({'response':{'ok':False,'infrastructure_failure':True,'error':'spawn failure'}})
    write_gzip(path,rows)
    result=audit(tmp_path,True)
    assert not result['batch_complete'] and result['uncompleted_cases']==[1]
    assert result['registered_cases']==2 and result['cases_verified']==1
    assert result['termination_counts']=={'deadline':1}
    rows.insert(2,rows[1])
    write_gzip(path,rows)
    with pytest.raises(ValueError,match='Duplicate applied bridge frame'):audit(tmp_path,True)
