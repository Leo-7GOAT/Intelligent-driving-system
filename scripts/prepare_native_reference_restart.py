"""Explicitly restart a failed, checkpoint-free reference course, charging its loss."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.research import atomic_json
from navstack.reference_context_policy import VERSION
from scripts.audit_native_method_endpoint import read, sha
from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone
from scripts.prepare_native_recovery import claim_recovery, verified_checkpoint_metadata


def prepare(parent, output):
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if output.exists() or (parent / 'recovery_registration.json').exists():
        raise FileExistsError('Preserve the unique registered restart')
    old = read(parent / 'protocol.json')
    guard, worker = read(parent / 'guard_final.json'), read(parent / 'worker_process.json')
    server, stop = read(parent / 'infrastructure/server_start.json'), read(parent / 'infrastructure/server_stop.json')
    watch = read(parent / 'guard.json')
    identities = [worker['pid'], watch['pid'], server['shipping_pid'], server['launcher_pid']]
    if (old.get('reference_context') != {'version': VERSION} or old['methods'] != ['H', 'H0']
            or guard.get('exit_code') != 1 or guard.get('worker_pid') != worker['pid']
            or read(parent / 'failure.json').get('error') != "InfrastructureFailure('bridge exited')"
            or not server.get('owned_by_task') or stop.get('shipping_pid') != server['shipping_pid']
            or stop.get('status') != 'owned_empty_server_stopped'
            or stop.get('post_stop_tcp') != [] or stop.get('post_stop_udp') != []
            or not recorded_lifecycle_gone(identities, stop['epoch'])):
        raise ValueError('Require an ended, safely cleaned bridge-exit failure')
    audit = read(parent / 'interruption_audit.json')
    if (audit['methods']['H']['state'] != 'interrupted' or audit['methods']['H']['checkpoints']
            or audit['methods']['H0']['state'] != 'not_started'):
        raise ValueError('A saved learner must be recovered instead of discarded')
    for name, digest in old['source_hashes'].items():
        if sha(parent / 'source' / name) != digest:
            raise ValueError('Frozen source changed: ' + name)
    for method in old['methods']:
        verified_checkpoint_metadata(parent / old['recovery'][method]['checkpoint'])
    bridge_controls = sum(s['applied_controls'] for s in audit['bridge_streams'].values())
    charged = max(bridge_controls, audit['methods']['H']['consumed_controls_lower_bound'])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.drive.lower() != 'd:' or shutil.disk_usage(output.parent).free < 16 * 1024**3:
        raise RuntimeError('Use D staging with at least 16GiB headroom')
    output.mkdir()
    for directory in ('source', 'resume', 'calibration'):
        shutil.copytree(parent / directory, output / directory, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(parent / 'measured_demonstrations.npz', output / 'measured_demonstrations.npz')
    (output / 'infrastructure').mkdir()
    p = deepcopy(old)
    p['recovery']['H']['consumed_controls_lower_bound'] += charged
    p['recovery']['H']['unknown_unflushed_parent_tail'] = True
    now = time.time()
    p.update(created_epoch=now, deadline_epoch=now + 14400, parent_batch=parent.name,
             checkpoint_thresholds=list(range(320, 10241, 320)),
             resource_limits={**p['resource_limits'], 'wall_seconds': 14400, 'automatic_retries': 0},
             restart_registration=dict(parent=parent.name, reason='bridge_exit_before_first_checkpoint',
                 source_protocol_sha256=sha(parent / 'protocol.json'),
                 interruption_audit_sha256=sha(parent / 'interruption_audit.json'),
                 discarded_controls_lower_bound=charged,
                 discarded_optimizer_steps=audit['methods']['H']['last_optimization']['optimizer_steps'],
                 unknown_unflushed_tail=True, retained_learning_from_failed_course=0,
                 semantics='Same frozen scale1 policy and physical source; fresh course from verified original initializer. All failed cost retained; not a matched consumed-budget comparison.'))
    atomic_json(output / 'protocol.json', p)
    atomic_json(output / 'prepared.json', dict(status='PREPARED_NOT_STARTED', protocol_sha256=sha(output / 'protocol.json')))
    claim_recovery(parent, output.name)
    print(str(output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.parent, args.output)
