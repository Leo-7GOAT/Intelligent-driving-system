"""Stage a reviewed canonical22 actor on the four original TRAIN roads, zero PPO."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def reviewed_fits(execution):
    """Require all five outcomes and a hash-bound research review, not best-seed selection."""
    execution = Path(execution).resolve()
    status = read(execution / 'status.json')
    if status['phase'] != 'ALL_FIVE_TRAIN_ONLY_ACTOR_FITS_COMPLETE':
        raise ValueError('Review all five completed actor fits before native preparation')
    review = read(execution / 'review.json')
    if (review['decision'] != 'probe_canonical22_on_four_existing_TRAIN_roads'
            or review['fit_status_sha256'] != sha(execution / 'status.json')
            or sorted(int(k) for k in review['fit_result_sha256']) != [11, 22, 33, 44, 55]
            or [r['seed'] for r in status['completed']] != [11, 22, 33, 44, 55]):
        raise ValueError('Require the fixed canonical22 review and every completed seed')
    for row in status['completed']:
        fit = execution / ('seed' + str(row['seed']))
        digest = sha(fit / 'complete.json')
        result = read(fit / 'complete.json')
        registration = read(fit / 'registration.json')
        metadata = verified_checkpoint_metadata(fit / 'checkpoint')
        if (row['result_sha256'] != digest or review['fit_result_sha256'][str(row['seed'])] != digest
                or result['seed'] != row['seed']
                or result['model_sha256'] != metadata['model_sha256']
                or metadata['measured_actor_plan_sha256'] != sha(execution / 'bound_plan.json')
                or registration['plan_sha256'] != sha(execution / 'bound_plan.json')
                or result['actual_Adam_step_each'] != [1024]
                or result['new_controls'] != 0 or result['new_PPO_updates'] != 0
                or result['PPO_critic_logstd_optimizer_counters_RMS_retained'] is not True):
            raise ValueError('Reviewed fixed actor fit/checkpoint evidence changed')
    return review


def prepare(execution, output):
    execution, output = Path(execution).resolve(), Path(output).resolve()
    if output.exists() or (execution / 'basic_probe_registration.json').exists():
        raise FileExistsError('Preserve the existing canonical basic probe')
    reviewed_fits(execution)
    plan = read(execution / 'bound_plan.json')
    parent = Path(next(r for r in plan['parents'] if r['seed'] == 22)['batch'])
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    fit = execution / 'seed22'
    metadata = verified_checkpoint_metadata(fit / 'checkpoint')
    source_config = read(fit / 'checkpoint/config.json')
    original_basic = ROOT / 'runs/20261101T121859752667Z_native_B1_shared_parking_cohort_seed22'
    basic = read(original_basic / 'protocol.json')
    if (old['seed'] != 22 or metadata['lineage_control_steps'] != 51200
            or source_config['env']['method'] != 'B1' or source_config['env']['flat_hold'] != 1
            or source_config['env'] != read(parent / 'B1/config.json')['env']
            or old['routes'][:4] != basic['routes'] or len(basic['routes']) != 4
            or read(parent / 'calibration/calibration.json')['recommended_adapter']['control_delivery'] != 'batch_sync_v1'):
        raise ValueError('Require the same direct policy, synchronous physics and original four TRAIN roads')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen parent source changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 18 * 1024**3:
        raise RuntimeError('Native preparation resource guard: ' + str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_measured_actor_basic22')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(parent / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(fit / 'checkpoint', batch / 'resume/B1')
    shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    evidence = batch / 'measured_actor_evidence'
    evidence.mkdir()
    for name in ('registration.json', 'complete.json', 'executed_fit.py', 'auxiliary_Adam.pt', 'sampler_state.json'):
        shutil.copy2(fit / name, evidence / name)
    for name in ('bound_plan.json', 'review.json', 'status.json'):
        shutil.copy2(execution / name, evidence / name)
    p = deepcopy(old)
    for name in ('native_case_manifest', 'native_validation', 'broader_B1_PPO', 'broader_course_recovery', 'lineage_targets'):
        p.pop(name, None)
    p.pop('freeze_observation_statistics', None)
    now = time.time()
    p.update(routes=deepcopy(basic['routes']), parent_batch=parent.name,
        scope='Canonical22direct actor after extra36TRAINteacher supervision;four original TRAINroads only,zero PPOupdates. Not independent validation,allfivequalification or PPOvalue.',
        evaluation_only=True, controlled_budget_per_method=0, method_budgets={'B1': 0},
        checkpoint_thresholds=[], deterministic_training_route_evaluation=True,
        created_epoch=now, deadline_epoch=now + 1800,
        resource_limits={**old['resource_limits'], 'wall_seconds': 1800, 'automatic_retries': 0})
    summary, raw = read(parent / 'B1/summary.json'), read(parent / 'B1/endpoint_raw_audit.json')
    p['recovery'] = dict(B1={**old['recovery']['B1'], 'checkpoint': 'resume/B1',
        'parent_batch': parent.name, 'parent_checkpoint': 'reviewed_measured_actor_fit/seed22',
        'lineage_controls': 51200,
        'consumed_controls_lower_bound': summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        'retained_optimizer_steps': old['recovery']['B1']['retained_optimizer_steps'] + summary['optimizer_steps'],
        'model_sha256': metadata['model_sha256'], 'normalization_sha256': metadata['normalization_sha256']})
    p['measured_actor_basic_probe'] = dict(version='reviewed_canonical22_basic_v1',
        fit_complete_sha256=sha(fit / 'complete.json'), review_sha256=sha(execution / 'review.json'),
        plan_sha256=sha(execution / 'bound_plan.json'), additional_actor_Adam_steps=1024,
        teacher_dataset_sha256=read(execution / 'seed22/registration.json')['dataset_sha256'],
        teacher_data_role='TRAIN_EXTRA_SUPERVISION', native_teacher_runtime=False,
        gate='Original basic gate:at least3/4successes ANDzero collision/bounds;no automatic full-task promotion.',
        new_PPO_updates=0, new_final_conditions=0)
    p['extra_supervision']['supervised_optimizer_steps'] += 1024
    p['extra_supervision']['additional_broader_measured_actor'] = p['measured_actor_basic_probe']
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    output.mkdir()
    registration = dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        fit_execution=execution.name, source_model_sha256=metadata['model_sha256'],
        new_controls_so_far=0, new_PPO_updates=0, final_conditions=0)
    atomic_json(output / 'registration.json', registration)
    atomic_json(execution / 'basic_probe_registration.json', registration)
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.execution, args.output)
