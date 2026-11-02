import json

import pytest

from scripts.prepare_measured_actor_basic_probe import reviewed_fits, prepare, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path):
    write(tmp_path / 'bound_plan.json', dict(parents=[dict(seed=s) for s in (11, 22, 33, 44, 55)]))
    plan_sha = sha(tmp_path / 'bound_plan.json')
    completed = []
    for seed in (11, 22, 33, 44, 55):
        fit = tmp_path / f'seed{seed}'
        checkpoint = fit / 'checkpoint'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'model.zip').write_bytes(b'model' + str(seed).encode())
        (checkpoint / 'normalization.pkl').write_bytes(b'normalizer')
        metadata = dict(model_sha256=sha(checkpoint / 'model.zip'),
            normalization_sha256=sha(checkpoint / 'normalization.pkl'), measured_actor_plan_sha256=plan_sha)
        write(checkpoint / 'metadata.json', metadata)
        write(fit / 'registration.json', dict(plan_sha256=plan_sha))
        write(fit / 'complete.json', dict(seed=seed, model_sha256=metadata['model_sha256'],
            actual_Adam_step_each=[1024], new_controls=0, new_PPO_updates=0,
            PPO_critic_logstd_optimizer_counters_RMS_retained=True))
        completed.append(dict(seed=seed, result_sha256=sha(fit / 'complete.json')))
    write(tmp_path / 'status.json', dict(phase='ALL_FIVE_TRAIN_ONLY_ACTOR_FITS_COMPLETE', completed=completed))
    review = dict(decision='probe_canonical22_on_four_existing_TRAIN_roads',
        fit_status_sha256=sha(tmp_path / 'status.json'),
        fit_result_sha256={str(r['seed']): r['result_sha256'] for r in completed})
    write(tmp_path / 'review.json', review)
    return review


def test_review_requires_every_fixed_fit_and_hash_bound_canonical_choice(tmp_path):
    review = fixture(tmp_path)
    assert reviewed_fits(tmp_path) == review


@pytest.mark.parametrize('fault', ['unfinished', 'changed_review_choice', 'missing_seed', 'changed_model', 'changed_plan', 'changed_result'])
def test_unreviewed_or_changed_fits_cannot_enter_native_preparation(tmp_path, fault):
    fixture(tmp_path)
    if fault == 'unfinished':
        write(tmp_path / 'status.json', dict(phase='TRAIN_ONLY_ACTOR_FIT_RUNNING'))
    elif fault in ('changed_review_choice', 'missing_seed'):
        review = json.loads((tmp_path / 'review.json').read_text())
        if fault == 'changed_review_choice': review['decision'] = 'choose_best_seed_after_results'
        else: review['fit_result_sha256'].pop('55')
        write(tmp_path / 'review.json', review)
    elif fault == 'changed_model': (tmp_path / 'seed44/checkpoint/model.zip').write_bytes(b'changed')
    elif fault == 'changed_plan': write(tmp_path / 'bound_plan.json', dict(changed=True))
    else: write(tmp_path / 'seed11/complete.json', dict(changed=True))
    output = tmp_path / 'unstarted_probe'
    with pytest.raises(ValueError):
        prepare(tmp_path, output)
    assert not output.exists()


def test_existing_probe_output_is_preserved(tmp_path):
    with pytest.raises(FileExistsError):
        prepare(tmp_path / 'not_needed', tmp_path)
