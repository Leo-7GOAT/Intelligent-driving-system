"""One training-normalization contrast from the identical pre-PPO checkpoint.

The completed reduced-exploration run is the control; it is never retrained or
replaced by its best endpoint. This branch has no new teacher data or labels.
"""
import argparse
from datetime import datetime, timezone
import difflib
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.research import resources_ok
from scripts.train_recovery_demonstrations import read, sha


def prepare(parent):
    parent = Path(parent).resolve()
    pointer = ROOT/'runs/frozen_training_normalization.json'
    if pointer.exists():
        raise RuntimeError('Inspect existing normalization contrast; no duplicate')
    old = read(parent/'protocol.json')
    audit = read(parent/'exploration_verification.json')
    guard = read(parent/'guard_final.json')
    assert audit['complete'] and guard['exit_code'] == 0 and guard['reason'] is None
    assert not (parent/'failure.json').exists()
    assert old['milestones'] == [0,25000,100000] and len(old['branches']) == 1
    assert old['ppo_options']['log_std_init'] == -4
    assert not old['ppo_options'].get('freeze_observation_stats', False)
    assert audit['training_audit']['controls'] == 102400
    assert audit['endpoints'][0]['actual_physics_steps'] == 0
    assert audit['endpoints'][-1]['full_task_success'] == 0
    origin = parent/old['branches'][0]['name']
    start = origin/'initial_checkpoint'
    baseline_hashes = old['initial_checkpoint_hashes'][origin.name]
    for name,digest in baseline_hashes.items():
        assert sha(start/name) == digest
    for name,digest in old['source_hashes'].items():
        assert sha(parent/'source'/name) == digest
        if name.startswith('navstack/') and name != 'navstack/research.py':
            assert sha(ROOT/name) == digest
    healthy, reason = resources_ok(ROOT)
    if not healthy or shutil.disk_usage(ROOT).free < 12*1024**3:
        raise RuntimeError(str(reason))
    batch = ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_frozen_training_normalization')
    batch.mkdir()
    name = 'frozen_observation_stats'
    folder = batch/name
    shutil.copytree(start, folder/'initial_checkpoint')
    for filename in ('resolved_config.json','version.json'):
        shutil.copy2(origin/filename, folder/filename)
    options = {**old['ppo_options'], 'freeze_observation_stats': True}
    write_json(folder/'ppo_options.json', options)
    resolved = read(folder/'resolved_config.json')
    resolved['intervention'] = 'Freeze inherited observation mean/variance/count during PPO; identical pre-PPO weights, optimizer, RNG, vehicle and statistics'
    write_json(folder/'resolved_config.json', resolved)
    saved = read(start/'training_summary.json')
    write_json(folder/'latest.json', {**saved, 'checkpoint': str(folder/'initial_checkpoint')})
    hashes = {}
    for source_folder in ('navstack','scripts','tests'):
        for path in (ROOT/source_folder).rglob('*.py'):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(ROOT)
            target = batch/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path,target)
            hashes[relative.as_posix()] = sha(path)
    # Keep the exact code delta reviewable alongside the frozen versions.
    delta = ''.join(difflib.unified_diff(
        (parent/'source/navstack/research.py').read_text(encoding='utf-8').splitlines(True),
        (ROOT/'navstack/research.py').read_text(encoding='utf-8').splitlines(True),
        fromfile='completed_control/research.py', tofile='frozen_stats/research.py'))
    (batch/'training_source.diff').write_text(delta, encoding='utf-8')
    copied = {p.name: sha(p) for p in (folder/'initial_checkpoint').iterdir() if p.is_file()}
    assert copied == baseline_hashes
    now = time.time()
    protocol = {
        'scope': 'One correlated development normalization contrast; inherited supervision disclosed, not pure RL or independent confirmation',
        'intervention_kind': 'freeze_observation_stats',
        'created_epoch': now, 'deadline_epoch': now+3600,
        'parent_batch': parent.name,
        'parent_audit_sha256': sha(parent/'exploration_verification.json'),
        'ledger_baseline_batch': old['parent_batch'],
        'inherited_controls': 0, 'branches': [{'name': name, 'learning_rate': .0001}],
        'milestones': old['milestones'], 'env': old['env'], 'seed': old['seed'],
        'cases': old['cases'], 'ppo_options': options,
        'hypothesis': 'Holding the inherited input transform fixed during PPO may prevent loss of supervised parking competence; endpoint normalization swaps do not test this training-time interaction',
        'intervention': resolved['intervention'],
        'selection': 'Identical original reduced-exploration initialization, never a learned endpoint; retain all0/25k/100k endpoints and failures',
        'expected_zero_result': old['expected_zero_result'],
        'inherited_costs': {**old['inherited_costs'],
            'completed_reduced_exploration_ppo_controls': audit['training_audit']['controls'],
            'completed_reduced_exploration_eval_controls': sum(e['evaluation_controls'] for e in audit['endpoints'])},
        'checkpoint_hashes': baseline_hashes,
        'initial_checkpoint_hashes': {name: copied}, 'source_hashes': hashes,
        'training_source_diff_sha256': sha(batch/'training_source.diff'),
        'resource_limits': {'wall_seconds':3600, 'disk_min_GiB':12, 'ram_min_GiB':4,
                            'cpu_workers':1, 'gpu_stop_temperature_C':85, 'automatic_retries':0},
        'disk_free_bytes': shutil.disk_usage(ROOT).free}
    write_json(batch/'protocol.json', protocol)
    write_json(batch/'initial_pairing_audit.json', {
        'all_initial_checkpoint_files_byte_identical': True,
        'all_other_navstack_modules_unchanged': True,
        'only_ppo_option_added': {'freeze_observation_stats': True},
        'source_delta': 'training_source.diff'})
    write_json(pointer, {'batch': str(batch)})
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    prepare(parser.parse_args().parent)
