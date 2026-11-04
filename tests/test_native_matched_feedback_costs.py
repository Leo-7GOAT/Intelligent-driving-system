from copy import deepcopy
import json

import pytest

from scripts.analyze_native_matched_feedback_costs import analyze
from scripts.audit_native_method_endpoint import sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root):
    pairs = []
    for seed in (11, 22, 33, 44, 55):
        proofs = []
        for method in ('H', 'H0'):
            folder = root / 'runs' / str(seed) / method
            episodes = []
            for case in range(24):
                success = not (seed == 44 and method == 'H0' and case == 15)
                seconds = (10 if method == 'H' else 8) if success else 1
                episodes.append({'case_id': case, 'success': success,
                    'termination_reason': 'success' if success else 'collision',
                    'simulation_seconds': seconds, 'interventions': 2 if method == 'H' else 1,
                    'plan_failures': 0, 'mpc_failures': 0})
            summary = {'physics_steps': 20480, 'actions': 2080,
                       'optimizer_steps': 650, 'wall_seconds': 60}
            evaluation = {'model_sha256': method + str(seed), 'episodes': episodes}
            write(folder / 'evaluation.json', evaluation)
            write(folder / 'summary.json', summary)
            write(folder / 'endpoint_raw_audit.json', {
                'evaluation_sha256': sha(folder / 'evaluation.json'),
                'summary_sha256': sha(folder / 'summary.json'),
                'evaluation_control_frames_verified': sum(e['simulation_seconds'] * 20 for e in episodes)})
            proofs.append({'batch': str(seed), 'summary': summary,
                'model_sha256': evaluation['model_sha256'],
                'raw_audit_sha256': sha(folder / 'endpoint_raw_audit.json')})
        pairs.append({'seed': seed, 'source_and_model_proof': proofs})
    cohort = {'cohort_kind': 'matched_broader_36train_24val_v1',
        'seeds': [11, 22, 33, 44, 55], 'unique_condition_count': 24,
        'identical_complete_execution_source_across_all_pairs': True, 'results': pairs}
    path = root / 'cohort.json'
    write(path, cohort)
    return path, cohort


def test_early_failure_is_kept_in_costs_and_excluded_from_paired_speed(tmp_path):
    path, _ = fixture(tmp_path)
    output = tmp_path / 'costs.json'
    report = analyze(path, tmp_path, output)
    assert report['totals']['H']['simulation_seconds'] == 1200
    assert report['totals']['H0']['simulation_seconds'] == 953
    assert report['totals']['H']['training_controls'] == 102400
    assert report['totals']['H0']['actual_Adam_steps'] == 3250
    assert report['totals']['H']['interventions'] == 240
    pair = report['results'][3]['paired_execution_costs']
    assert pair['right_unsafe_episodes'] == 1
    assert 15 not in pair['mutually_successful_case_ids']
    assert pair['mutually_successful_mean_time_difference_seconds'] == 2
    assert report['new_controls_for_analysis'] == report['optimizer_updates_for_analysis'] == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        analyze(path, tmp_path, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('fault', ['partial', 'order', 'audit', 'evaluation', 'summary', 'escape', 'duplicate_case'])
def test_incomplete_or_changed_evidence_fails_before_writing(tmp_path, fault):
    path, cohort = fixture(tmp_path)
    cohort = deepcopy(cohort)
    folder = tmp_path / 'runs/11/H'
    if fault == 'partial':
        cohort['results'].pop()
    elif fault == 'order':
        cohort['results'].reverse()
    elif fault == 'escape':
        cohort['results'][0]['source_and_model_proof'][0]['batch'] = '../../outside'
    elif fault == 'audit':
        (folder / 'endpoint_raw_audit.json').write_text('{}')
    elif fault == 'summary':
        cohort['results'][0]['source_and_model_proof'][0]['summary']['optimizer_steps'] += 1
    else:
        evaluation = json.loads((folder / 'evaluation.json').read_text())
        evaluation['episodes'][0]['case_id'] = 1
        write(folder / 'evaluation.json', evaluation)
        if fault == 'duplicate_case':
            raw_path = folder / 'endpoint_raw_audit.json'
            raw = json.loads(raw_path.read_text())
            raw['evaluation_sha256'] = sha(folder / 'evaluation.json')
            write(raw_path, raw)
            cohort['results'][0]['source_and_model_proof'][0]['raw_audit_sha256'] = sha(raw_path)
    write(path, cohort)
    output = tmp_path / 'costs.json'
    with pytest.raises(ValueError):
        analyze(path, tmp_path, output)
    assert not output.exists()


def test_concurrent_cohort_replacement_cannot_bind_the_wrong_receipt(tmp_path, monkeypatch):
    import scripts.analyze_native_matched_feedback_costs as module
    path, cohort = fixture(tmp_path)
    original_read = module.read
    def changing_read(source):
        result = original_read(source)
        if source == path:
            write(path, {**cohort, 'changed_during_read': True})
        return result
    monkeypatch.setattr(module, 'read', changing_read)
    output = tmp_path / 'costs.json'
    with pytest.raises(ValueError, match='changed while'):
        analyze(path, tmp_path, output)
    assert not output.exists()
