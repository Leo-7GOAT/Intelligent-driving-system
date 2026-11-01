"""Register all existing24development-validation conditions with the unchanged frozen B1 runtime."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.native_partitions import validate_native_partitions
from navstack.research import atomic_json, resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def prepare():
    pointer = ROOT / 'runs/native_B1_shared_parking_validation_probe_registration_20261101.json'
    if pointer.exists():
        raise FileExistsError('Preserve the registered validation probe; no automatic repetition')
    parent = ROOT / 'runs/20261101T114634589144Z_native_B1_shared_parking_full_probe'
    inventory = ROOT / 'runs/20261101T011945179519Z_native_matched_broader_feedback'
    require_finished_parent(parent)
    old = read(parent / 'protocol.json')
    reference = read(inventory / 'protocol.json')
    partition = validate_native_partitions(reference)
    report_path = ROOT / 'reports/native_B1_shared_parking_full_probe_20261101.json'
    report = read(report_path)
    if report['successes'] != 12 or report['protocol_sha256'] != sha(parent / 'protocol.json'):
        raise ValueError('Require the complete audited12-condition endpoint')
    origin = parent / 'B1' / read(parent / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(origin)
    if metadata['model_sha256'] != report['model_sha256']:
        raise ValueError('Frozen endpoint differs')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen worker source changed')
    okay, reason = resources_ok(ROOT)
    if not okay:
        raise RuntimeError(str(reason))
    batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_native_B1_shared_parking_validation_probe')
    batch.mkdir()
    (batch / 'infrastructure').mkdir()
    shutil.copytree(parent / 'source', batch / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(parent / 'calibration', batch / 'calibration')
    shutil.copytree(origin, batch / 'resume/B1')
    if (parent / 'measured_demonstrations.npz').exists():
        shutil.copy2(parent / 'measured_demonstrations.npz', batch / 'measured_demonstrations.npz')
    p = deepcopy(old)
    p.pop('qualified_basic_full_probe')
    now = time.time()
    # This older frozen worker evaluates its registered cases directly. Supply
    # the exact validation inventory there, without adding an unsupported flag.
    p.update(routes=deepcopy(reference['native_validation']['routes']),
             native_case_manifest=deepcopy(reference['native_validation']['manifest']),
             created_epoch=now, deadline_epoch=now + 7200,
             resource_limits={**old['resource_limits'], 'wall_seconds': 7200, 'automatic_retries': 0},
             scope='Frozen trained B1 on all24EXISTING development-validation conditions. Same worker,model,normalizer,vehicle,goal gates and shared assistance;zero learning. Not new final confirmation.',
             known_validation_probe={'version': 'frozen_B1_existing24validation_v1',
                 'inventory_batch': inventory.name, 'inventory_protocol_sha256': sha(inventory / 'protocol.json'),
                 'partition': partition, 'parent_report_sha256': sha(report_path),
                 'source_model_sha256': metadata['model_sha256'], 'normalization_sha256': metadata['normalization_sha256'],
                 'new_PPO_updates': 0, 'new_final_conditions': 0,
                 'selection': 'All24existingvalidation conditions and the same final trained model;no filtering or checkpoint selection.',
                 'limits': 'One seed and historically known developmentvalidation,not five independent policies or a new finalset. The earlier12conditions and all failures remain retained.'})
    summary = read(parent / 'B1/summary.json')
    raw = read(parent / 'B1/endpoint_raw_audit.json')
    p['recovery']['B1'].update(parent_batch=parent.name, parent_checkpoint='B1/' + origin.name,
        lineage_controls=summary['lineage_control_steps'],
        consumed_controls_lower_bound=summary['charged_control_steps_lower_bound'] + raw['evaluation_control_frames_verified'])
    name = 'scripts/prepare_native_shared_parking_validation_probe.py'
    shutil.copy2(__file__, batch / 'source' / name)
    p['source_hashes'][name] = sha(batch / 'source' / name)
    atomic_json(batch / 'protocol.json', p)
    # Check the actual frozen parser, not just the current repository parser.
    command = "import sys,json;sys.path.insert(0,sys.argv[1]);from navstack.native_cases import registered_native_cases;p=json.load(open(sys.argv[2],encoding='utf-8'));assert len(registered_native_cases(p))==24"
    subprocess.run([sys.executable, '-c', command, str(batch / 'source'), str(batch / 'protocol.json')], check=True)
    atomic_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
    atomic_json(pointer, {'batch': batch.name, 'protocol_sha256': sha(batch / 'protocol.json'), 'new_controls_so_far': 0})
    print(batch, flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
