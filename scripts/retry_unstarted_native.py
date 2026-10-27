"""One explicit child for a verified startup-only timeout; preserve the failure."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from navstack.experiments import write_json
from navstack.process_lease import ProcessLease, pid_alive
from navstack.research import resources_ok
from scripts.audit_native_method_endpoint import read, sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata


def startup_only_timeout(failed):
    failed = Path(failed)
    for name in ('started.json', 'worker_process.json', 'guard.json', 'guard_final.json', 'failure.json', 'summary.json'):
        if (failed / name).exists():
            raise ValueError('Not a zero-worker startup failure: ' + name)
    if any((failed / 'infrastructure').glob('bridge_*')):
        raise ValueError('Bridge activity is not a zero-control startup failure')
    failure = read(failed / 'launcher_failure.json')
    if failure.get('error') != 'Connected server is not matching-version and empty':
        raise ValueError('Unsupported startup failure; inspect it separately')
    probes = list((failed / 'infrastructure').glob('readiness_probe_*.json'))
    if not probes or any(read(path).get('status') != 'CONNECTION_FAILED' for path in probes):
        raise ValueError('Only recorded connection timeouts qualify, never foreign/nonempty worlds')
    start = read(failed / 'infrastructure/server_start.json')
    stop = read(failed / 'infrastructure/server_stop.json')
    if (not start.get('owned_by_task') or stop.get('status') != 'owned_empty_server_stopped'
            or stop.get('shipping_pid') != start.get('shipping_pid')):
        raise ValueError('Owned server cleanup is unverified')
    probe = Path(stop['probe']).resolve()
    if probe.parent != (failed / 'infrastructure').resolve():
        raise ValueError('Cleanup probe is outside the failed batch')
    observed = read(probe)
    if (observed.get('status') != 'CONNECTED' or observed.get('client_version') != '0.9.15'
            or observed.get('server_version') != '0.9.15' or observed.get('actors') != []):
        raise ValueError('Require the later matching-version empty-server cleanup probe')
    for key in ('shipping_pid', 'launcher_pid'):
        pid = start.get(key)
        if type(pid) is not int or pid <= 0 or pid_alive(pid):
            raise ValueError('Original owned process is live or ambiguous')
    return {'failed_batch': failed.name, 'failure_sha256': sha(failed / 'launcher_failure.json'),
            'cleanup_sha256': sha(failed / 'infrastructure/server_stop.json'),
            'cleanup_probe_sha256': sha(probe), 'worker_started': False, 'applied_controls': 0}


def prepare(failed):
    failed = Path(failed).resolve()
    evidence = startup_only_timeout(failed)
    original = read(failed / 'protocol.json')
    if read(failed / 'prepared.json')['protocol_sha256'] != sha(failed / 'protocol.json'):
        raise ValueError('Prepared protocol changed')
    for method in original['methods']:
        if (failed / method).exists():
            raise ValueError('Method execution evidence must be preserved, not retried as startup')
        verified_checkpoint_metadata(failed / original['recovery'][method]['checkpoint'])
    for name, digest in original['source_hashes'].items():
        if sha(failed / 'source' / name) != digest:
            raise ValueError('Frozen source changed')
    if sha(failed / 'calibration/calibration.json') != original['calibration_sha256']:
        raise ValueError('Calibration changed')
    okay, reason = resources_ok(ROOT)
    if not okay or shutil.disk_usage(ROOT).free < 12 * 1024**3:
        raise RuntimeError(str(reason))
    with ProcessLease(failed / 'startup_retry_prepare.lock'):
        registration = failed / 'startup_retry_registration.json'
        if registration.exists():
            raise FileExistsError('One startup child is already registered; inspect that child')
        batch = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_native_startup_retry')
        batch.mkdir()
        write_json(registration, {'batch': batch.name, 'status': 'registered_before_staging'})
        (batch / 'infrastructure').mkdir()
        for folder in ('source', 'calibration', 'resume'):
            shutil.copytree(failed / folder, batch / folder, ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('parent_protocol.json', 'reference_scene_protocol.json', 'measured_demonstrations.npz'):
            if (failed / name).is_file(): shutil.copy2(failed / name, batch / name)
        for method in original['methods']:
            path = original['recovery'][method]['checkpoint']
            if verified_checkpoint_metadata(batch / path) != verified_checkpoint_metadata(failed / path):
                raise ValueError('Copied checkpoint changed')
        protocol = deepcopy(original)
        now = time.time()
        protocol.update(created_epoch=now, deadline_epoch=now + original['resource_limits']['wall_seconds'],
                        startup_retry={**evidence, 'failed_protocol_sha256': sha(failed / 'protocol.json'),
                                       'preparer_sha256': sha(Path(__file__)), 'retry_limit': 1,
                                       'scope': 'Same saved policies, scenes, runtime source and budgets; zero previous worker/controls. Original failed lifecycle remains untouched.'})
        write_json(batch / 'protocol.json', protocol)
        write_json(batch / 'prepared.json', {'status': 'PREPARED_NOT_STARTED', 'protocol_sha256': sha(batch / 'protocol.json')})
        write_json(batch / 'startup_retry_evidence.json', evidence)
        shutil.copy2(__file__, batch / 'startup_retry_preparer.py')
        write_json(registration, {'batch': batch.name, 'status': 'PREPARED_NOT_STARTED',
                                  'protocol_sha256': sha(batch / 'protocol.json')})
    print(batch, flush=True)
    return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--failed', required=True)
    args = parser.parse_args()
    prepare(args.failed)
