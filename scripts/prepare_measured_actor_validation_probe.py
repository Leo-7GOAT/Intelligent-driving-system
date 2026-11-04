"""Measure canonical22 on all24known VAL conditions after its actual four-road gate."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_partitions import validate_native_partitions
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.audit_native_control_delivery import analyze as delivery_audit


def prepare(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or (parent / 'measured_validation_registration.json').exists():
        raise FileExistsError('Preserve the registered frozen-model validation')
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    evaluation, raw, summary = (read(parent / 'B1' / name) for name in ('evaluation.json', 'endpoint_raw_audit.json', 'summary.json'))
    if (old['seed'] != 22 or not old.get('measured_actor_basic_probe')
            or not old['evaluation_only'] or summary['physics_steps'] != 0 or summary['optimizer_steps'] != 0
            or sorted(r['case_id'] for r in evaluation['episodes']) != [0, 1, 2, 3]
            or sum(r['success'] for r in evaluation['episodes']) < 3
            or any(r['termination_reason'] in ('collision', 'out_of_bounds') for r in evaluation['episodes'])
            or raw['protocol_sha256'] != sha(parent / 'protocol.json')
            or not delivery_audit(parent)['strict_delivery_gate_passed']):
        raise ValueError('Require the actual canonical22 basic gate, not offline fit quality')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if metadata['model_sha256'] != evaluation['model_sha256'] or metadata['lineage_control_steps'] != 51200:
        raise ValueError('Fixed reviewed model or original PPO lineage differs')
    reference = read(ROOT / 'runs/20261101T131830295783Z_native_B1_broader_shared_parking_PPO/protocol.json')
    p = deepcopy(old)
    for name in ('routes', 'native_case_manifest', 'native_validation'):
        p[name] = deepcopy(reference[name])
    partitions = validate_native_partitions(p)
    if partitions['training_conditions'] != 36 or partitions['validation_conditions'] != 24:
        raise ValueError('Require the original36TRAIN/24knownVAL inventory')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen physical/policy implementation changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 18 * 1024**3:
        raise RuntimeError('Validation resource guard: ' + str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_measured_actor_validation22')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    for name in ('source', 'calibration'):
        shutil.copytree(parent / name, batch / name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(origin, batch / 'resume/B1')
    shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    shutil.copytree(parent / 'measured_actor_evidence', batch / 'measured_actor_evidence')
    now = time.time()
    p.update(parent_batch=parent.name, created_epoch=now, deadline_epoch=now + 3600,
        resource_limits={**old['resource_limits'], 'wall_seconds': 3600, 'automatic_retries': 0},
        scope='Canonical22fixed direct NN after extra TRAIN supervision;all24KNOWNVALconditions,zero new PPOupdates. Establish a pre-PPO development reference,not final confirmation or allfivehealthybaselines.')
    p['recovery']['B1'].update(parent_batch=parent.name, parent_checkpoint='B1/' + origin.name,
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'],
        model_sha256=metadata['model_sha256'], normalization_sha256=metadata['normalization_sha256'])
    p['measured_actor_validation_probe'] = dict(version='canonical22_post_supervision_known24_v1',
        basic_evaluation_sha256=sha(parent / 'B1/evaluation.json'), basic_raw_audit_sha256=sha(parent / 'B1/endpoint_raw_audit.json'),
        model_sha256=metadata['model_sha256'], cases=24, partition=partitions,
        new_PPO_updates=0, new_final_conditions=0,
        limits='Fixed canonical22 chosen before fitting;no model selection. Extra TRAIN labels,not an isolated PPO effect. All conditions and failures retained.')
    atomic_json(batch / 'protocol.json', p)
    atomic_json(batch / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(batch / 'protocol.json')))
    output.mkdir()
    registration = dict(batch=batch.name, protocol_sha256=sha(batch / 'protocol.json'),
        parent=parent.name, source_model_sha256=metadata['model_sha256'], known_validation_cases=24,
        new_PPO_updates=0, new_controls_so_far=0, final_conditions=0)
    atomic_json(output / 'registration.json', registration)
    atomic_json(parent / 'measured_validation_registration.json', registration)
    print(batch, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.parent, args.output)
