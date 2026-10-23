"""Resume actual PPO from the fixed final pure-learner aggregation endpoint.

All supervision is inherited and charged, never relabeled as PPO interaction.
The completed parent must pass its full raw-data audit before preparation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json, load_run
from navstack.research import resources_ok
from scripts.train_recovery_demonstrations import read, sha, run


def prepare(parent):
    parent = Path(parent).resolve()
    pointer = ROOT/'runs/aggregated_ppo.json'
    if pointer.exists():
        raise RuntimeError('Inspect registered PPO continuation; no duplicate')
    old, completed = read(parent/'protocol.json'), read(parent/'completed.json')
    audit, guard = read(parent/'analysis/verified_summary.json'), read(parent/'guard_final.json')
    assert audit['complete'] and guard['exit_code'] == 0 and guard['reason'] is None
    assert not (parent/'failure.json').exists()
    assert old['teacher_mode'] == 'current_state' and old['value_weight'] == 0
    assert completed['final_ppo_controls'] == 0 and not old['ppo_milestones']
    assert len(audit['rounds']) == len(old['rounds']) == len(completed['results']) == 2
    for name, digest in old['source_hashes'].items():
        assert sha(parent/'source'/name) == digest
    assert sha(parent/'parent_data/demonstrations.npz') == old['data_sha256']
    for name, digest in audit['input_hashes'].items():
        assert sha(parent/name) == digest
    # Always the final declared endpoint, regardless of whether round one wins.
    checkpoint = parent/'round_02/supervised_checkpoint'
    fit = read(parent/'round_02/supervision.json')
    assert sha(checkpoint/'model.zip') == fit['model_sha256']
    assert sha(checkpoint/'normalization.pkl') == fit['normalization_sha256']
    config = read(checkpoint/'config.json')
    assert config['env'] == old['env']
    model, norm = load_run(checkpoint)
    try:
        assert model.seed == config['seed'] and model._n_updates == 0
        assert not model.policy.optimizer.state
        assert model.use_sde and model.sde_sample_freq == 1
        assert model.target_kl == .03 and model.ent_coef == .001
        assert float(model.lr_schedule(1.)) == .0001
        assert model.n_steps == 2560 and model.n_epochs == 10 and model.batch_size == 64
    finally:
        norm.close()
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_aggregated_ppo')
    batch.mkdir()
    shutil.copytree(checkpoint, batch/'initial')
    # The common runner reads these arrays; rounds=[] prevents any further fit.
    shutil.copytree(parent/'parent_data', batch/'parent_data')
    hashes = {}
    for folder in ('navstack', 'scripts', 'tests'):
        for path in (ROOT/folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT)
            target = batch/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[relative.as_posix()] = sha(path)
    now = time.time()
    protocol = {
        'scope': 'One inherited policy: separately counted on-policy PPO after explicit supervised aggregation; not pure-RL training, an independent seed or final confirmation',
        'created_epoch': now, 'deadline_epoch': now+3600, 'parent_batch': parent.name,
        'parent_audit_sha256': sha(parent/'analysis/verified_summary.json'),
        'source_hashes': hashes, 'initial_hashes': {p.name: sha(p) for p in (batch/'initial').iterdir()},
        'data_sha256': old['data_sha256'], 'env': old['env'], 'seed': config['seed'],
        'seed_note': 'Preserve original PPO model seed; parent supervised permutation seeds are separate, not independent policy seeds.',
        'cases': old['cases'], 'collection_cases': [], 'rounds': [], 'supervised_epochs_per_round': 0,
        'ppo_milestones': [0, 25000, 100000],
        'ppo_options': {'target_kl': .03, 'use_sde': True, 'sde_sample_freq': 1,
                        'log_std_init': -2, 'entropy': .001, 'learning_rate': .0001},
        'zero_update_reference': {key: completed['results'][-1][key]
                                  for key in ('full_task_success', 'retention', 'evaluation_controls')},
        'selection': 'Use final round two even if it regresses; retain zero/25k/100k endpoints and all failures, no best-checkpoint substitution',
        'hypothesis': 'Test whether ordinary task-reward PPO improves the fixed final supervised policy without losing its parking competence; no teacher executes or supplies labels during PPO.',
        'normalization': 'Inherited statistics at initialization, ordinary training-only updates during PPO; frozen for every evaluation.',
        'inherited_costs': {**old['inherited_costs'],
            'pure_learner_collection_controls': completed['new_collection_controls'],
            'pure_learner_teacher_queries': completed['teacher_queries'],
            'pure_learner_supervised_optimizer_steps': completed['new_supervised_optimizer_steps'],
            'pure_learner_evaluation_controls': sum(r['evaluation_controls'] for r in completed['results'])},
        'resource_limits': {'wall_seconds': 3600, 'cpu_workers': 1, 'disk_min_GiB': 12,
                            'ram_min_GiB': 4, 'gpu_stop_temperature_C': 85, 'automatic_retries': 0},
        'disk_free_bytes': shutil.disk_usage(ROOT).free, 'worker_scopes': ['B1']}
    write_json(batch/'protocol.json', protocol)
    write_json(pointer, {'batch': str(batch)})
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent')
    parser.add_argument('--batch')
    args = parser.parse_args()
    prepare(args.parent) if args.parent else run(args.batch)
