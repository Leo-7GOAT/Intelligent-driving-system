"""Two bounded pure-learner aggregation rounds after the fixed-label contrast."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from navstack.experiments import write_json
from navstack.research import resources_ok
from scripts.train_recovery_demonstrations import run


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(parent, relabel):
    parent, relabel = Path(parent).resolve(), Path(relabel).resolve()
    pointer = ROOT/'runs/current_state_aggregation.json'
    if pointer.exists():
        raise RuntimeError('Inspect existing registered aggregation; no duplicate')
    p, result = read(parent/'protocol.json'), read(parent/'completed.json')
    contrast = read(parent/'contrast_analysis/verified_summary.json')
    assert contrast['endpoints'][1]['model_sha256'] == sha(parent/'supervised_checkpoint/model.zip')
    assert p['relabel_batch'] == relabel.name and result['ppo_updates'] == 0
    assert read(relabel/'verification.json')['complete']
    ok, reason = resources_ok(ROOT)
    if not ok or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_current_state_aggregation')
    batch.mkdir()
    (batch/'parent_data').mkdir()
    shutil.copytree(parent/'supervised_checkpoint', batch/'initial')
    arrays = {key: [] for key in ('observations', 'actions', 'returns', 'case_id', 'train_mask')}
    cases, inherited_files = [], {}
    for group in ('original', 'round1', 'round2'):
        source = parent/'data'/(group+'.npz')
        assert sha(source) == p['data_hashes'][group+'.npz'] == read(relabel/'completed.json')['files'][group]['sha256']
        data = np.load(source)
        group_cases = (read(parent/'data/original_protocol.json')['cases'] if group == 'original'
                       else read(relabel/'protocol.json')['collection_cases'])
        mask = np.asarray([group_cases[int(i)]['data_split']=='train' for i in data['case_id']])
        if 'train_mask' in data:
            np.testing.assert_array_equal(mask, data['train_mask'])
        for key in ('observations', 'actions', 'returns'):
            arrays[key].append(data[key])
        arrays['case_id'].append(data['case_id']+len(cases))
        arrays['train_mask'].append(mask)
        cases.extend([{**case, 'source_group': group} for case in group_cases])
        inherited_files[group] = sha(source)
    combined = {key: np.concatenate(value) for key, value in arrays.items()}
    assert len(combined['actions']) == 32538 and int(combined['train_mask'].sum()) == 24831
    np.savez_compressed(batch/'parent_data/demonstrations.npz', **combined)
    write_json(batch/'parent_data/protocol.json', {'cases': cases, 'scope': 'All relabeled ancestor visits, including failures; original split retained'})
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
    protocol = {'scope': 'One-seed current-state supervision on pure learner visits; exploratory continuation, not pure PPO or a causal isolation of coverage from extra optimization',
                'created_epoch': now, 'deadline_epoch': now+7200, 'parent_batch': parent.name,
                'source_hashes': hashes, 'initial_hashes': {x.name: sha(x) for x in (batch/'initial').iterdir()},
                'data_sha256': sha(batch/'parent_data/demonstrations.npz'), 'ancestor_group_hashes': inherited_files,
                'env': p['env'], 'seed': 13, 'supervised_seed_base': 13, 'cases': p['cases'],
                'collection_cases': [{'scenario': scene, 'seed': seed, 'data_split': 'train' if seed <= 8406 else 'holdout'}
                                     for seed in range(8401, 8409) for scene in ('empty', 'static', 'narrow', 'dynamic')],
                'rounds': [{'index': 1, 'teacher_probability': 0.}, {'index': 2, 'teacher_probability': 0.}],
                'teacher_mode': 'current_state', 'value_weight': 0., 'supervised_epochs_per_round': 200,
                'ppo_milestones': [], 'selection': 'Fixed current-state label endpoint initialization, both 200-epoch endpoints retained; no best-checkpoint replacement',
                'hypothesis': 'Label actual pure-policy visits, including recovery and failed trajectories, after label consistency alone did not improve task competence',
                'normalization': 'Original statistics frozen; all inherited and fresh holdout examples excluded from gradients and statistics',
                'supervision_costs': 'Teacher queries never execute; every additional control/query and optimizer step charged separately. No PPO updates in this batch.',
                'inherited_costs': {'model_lineage_supervised_steps': 6400+12800+19400,
                                    'original_teacher_controls': 10817, 'original_direct_replay_controls': 10817,
                                    'mixture_collection_controls': 21721, 'current_state_relabel_controls': 32538,
                                    'mixture_teacher_queries': 21721,
                                    'current_state_relabel_queries': 32538,
                                    'discarded_sibling_supervised_steps': 19400+19400,
                                    'discarded_sibling_ppo_controls': 102400+102400,
                                    'teacher_history_audit_replay_controls': 3120,
                                    'teacher_history_audit_queries': 3120,
                                    'current_state_teacher_competence_controls': 1341,
                                    'pure_learner_diagnostic_controls_and_queries_each': 613,
                                    'excluded_diagnostic_labels': 'The 613-control audit includes heldout/validation visits and is not used as training data',
                                    'other_sibling_development_costs': 'Retained in ancestor protocols; not inherited model updates'},
                'resource_limits': {'wall_seconds': 7200, 'cpu_workers': 1, 'disk_min_GiB': 12, 'ram_min_GiB': 4,
                                    'gpu_stop_temperature_C': 85, 'automatic_retries': 0},
                'disk_free_bytes': shutil.disk_usage(ROOT).free, 'worker_scopes': ['round_01', 'round_02']}
    write_json(batch/'protocol.json', protocol)
    write_json(pointer, {'batch': str(batch)})
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent')
    parser.add_argument('--relabel')
    parser.add_argument('--batch')
    args = parser.parse_args()
    prepare(args.parent, args.relabel) if args.parent else run(args.batch)
