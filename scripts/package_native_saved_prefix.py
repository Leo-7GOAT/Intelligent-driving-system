"""Audit/export a complete readable saved TRAIN prefix from an interrupted job.

No endpoint result, missing frame or gzip footer is synthesized. This profile
supports a prefix whose entire readable control inventory is checkpointed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import read, sha, read_prefix, _verify_ticks
from scripts.audit_native_control_delivery import analyze as delivery_audit
from scripts.analyze_native_feedback_ppo import adam_step_increments
from scripts.prepare_native_recovery import verified_checkpoint_metadata
from scripts.package_native_continuation import portable_metadata, scan_payload, operational_guard_files, SUFFIXES


def audit_prefix(batch):
    import numpy as np
    import pickle
    from navstack.independent_clip_ppo import load_ppo
    batch = Path(batch).resolve()
    p = read(batch / 'protocol.json')
    guard, failure = read(batch / 'guard_final.json'), read(batch / 'failure.json')
    if (p['methods'] != ['B1'] or guard['exit_code'] != 1
            or guard.get('reason') not in (None, 'disk_below_12GiB')
            or failure['error'] != "RuntimeError('Disk below12GiB')"
            or read(batch / 'infrastructure/server_stop.json')['status'] != 'owned_empty_server_stopped'
            or (batch / 'B1/evaluation.json').exists() or (batch / 'B1/summary.json').exists()):
        raise ValueError('Require the preserved disk-interrupted TRAIN-only job; no endpoint claim')
    for name, digest in p['source_hashes'].items():
        if sha(batch / 'source' / name) != digest:
            raise ValueError('Frozen physical/learning source changed')
    checkpoint = batch / 'B1' / read(batch / 'B1/latest_checkpoint.json')['folder']
    metadata = verified_checkpoint_metadata(checkpoint)
    actual, acknowledgements, owners, streams = {}, {}, {}, []
    for path in (batch / 'infrastructure').glob('*_responses.jsonl.gz'):
        rows, error = read_prefix(path)
        if error:
            raise ValueError('Only complete readable streams are supported by this export')
        streams.append(dict(name=path.name, sha256=sha(path)))
        for row in rows:
            r = row['response']
            if r.get('episode_id') is not None and r.get('request_id') is not None:
                key = r['episode_id'], r['request_id']
                if key in acknowledgements and acknowledgements[key] != r:
                    raise ValueError('Conflicting bridge acknowledgement')
                acknowledgements[key] = r
            if 'vehicle_control' in r:
                key = r['episode_id'], r['frame']
                if key in actual:
                    raise ValueError('Duplicate actual control frame')
                actual[key], owners[key] = r, path.name
    used, previous, episodes = set(), {}, set()
    count = 0
    mode = read(batch / 'B1/config.json')['env']['actuator_observation']
    for path in sorted((batch / 'B1').glob('transitions*.gz')):
        rows, error = read_prefix(path)
        if error:
            raise ValueError('Truncated policy transitions cannot enter this complete-prefix profile')
        for row in rows:
            count += _verify_ticks(row['trace'], actual, acknowledgements, owners, used, previous, episodes, mode)
    delivery = delivery_audit(batch)
    if count != len(actual) or count != metadata['physics_steps'] or not delivery['strict_delivery_gate_passed']:
        raise ValueError('Saved prefix omits controls or synchronous engine delivery proof')
    initial = load_ppo(batch / 'resume/B1/model.zip', device='cpu')
    final = load_ppo(checkpoint / 'model.zip', device='cpu')
    if final.num_timesteps - initial.num_timesteps != count:
        raise ValueError('Saved PPO counter differs from raw training')
    adam = adam_step_increments(initial.policy.optimizer.state_dict()['state'],
        final.policy.optimizer.state_dict()['state'], metadata['optimizer_steps'])
    with (batch / 'resume/B1/normalization.pkl').open('rb') as stream:
        before = pickle.load(stream)
    with (checkpoint / 'normalization.pkl').open('rb') as stream:
        after = pickle.load(stream)
    if (not np.array_equal(before.obs_rms.mean, after.obs_rms.mean)
            or not np.array_equal(before.obs_rms.var, after.obs_rms.var)
            or before.obs_rms.count != after.obs_rms.count):
        raise ValueError('Frozen observation RMS changed')
    reference = read(batch / 'verified_saved_prefix.json')
    rebuilt = dict(protocol_sha256=sha(batch / 'protocol.json'), verified_controls=count,
        all_engine_controls_acknowledged=True, checkpoint_sha256=sha(checkpoint / 'model.zip'),
        bridge_streams=streams, saved_Adam=adam, saved_controls=count,
        remaining_controls=p['broader_B1_PPO']['new_actual_PPO_controls'] - count,
        unknown_unflushed_tail_retained=read(batch / 'interruption_audit.json')['methods']['B1']['unknown_unflushed_tail'])
    if any(reference[k] != value for k, value in rebuilt.items()):
        raise ValueError('Saved-prefix reference does not rebuild exactly')
    return dict(scope=__doc__, batch=batch.name, **rebuilt,
        reference_receipt_sha256=sha(batch / 'verified_saved_prefix.json'), reference_fields_exact=True,
        observation_RMS_exact=True, strict_engine_delivery=delivery, new_controls=0, new_optimizer_updates=0)


def package(batch, output):
    batch, output = Path(batch).resolve(), Path(output)
    temporary = output.with_name(output.name + '.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Preserve an existing prefix archive')
    proof = audit_prefix(batch)
    p = read(batch / 'protocol.json')
    selected = set(batch.glob('*.json'))
    for directory in ('source', 'calibration', 'resume', 'B1'):
        selected.update(path for path in (batch / directory).rglob('*')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in SUFFIXES)
    selected.update(path for path in (batch / 'infrastructure').iterdir()
        if path.is_file() and path.suffix in ('.json', '.gz'))
    operational = operational_guard_files(batch)
    selected.update(operational)
    protected = {batch / name for name in ('protocol.json', 'verified_saved_prefix.json', 'interruption_audit.json')}
    protected.update(operational)
    protected.update(batch / 'source' / name for name in p['source_hashes'])
    protected.update(path for directory in ('calibration', 'resume', 'B1')
        for path in (batch / directory).rglob('*.json')
        if directory != 'B1' or path.name in ('config.json', 'metadata.json', 'latest_checkpoint.json'))
    files, transformations = {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = path.relative_to(batch).as_posix()
            original = payload = path.read_bytes()
            changes = []
            if path.suffix == '.json':
                portable = portable_metadata(read(path), changes)
                if changes:
                    if path in protected:
                        raise ValueError('Cannot redact hash-bound evidence: ' + relative)
                    payload = json.dumps(portable, indent=2).encode('utf-8')
                    transformations[relative] = dict(original_sha256=hashlib.sha256(original).hexdigest(), changes=changes)
            name = batch.name + '/' + relative
            scan_payload(name, payload)
            archive.writestr(name, payload)
            files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        manifest = dict(schema='native_saved_training_prefix_v1', batch=batch.name, files=files,
            metadata_transformations=transformations, saved_prefix_proof=proof,
            repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            package_script_sha256=sha(__file__), batch_complete=False, endpoint_results=0,
            not_exported=[path.relative_to(batch).as_posix() for path in sorted(batch.rglob('*'))
                if path.is_file() and path not in selected and '__pycache__' not in path.parts])
        payload = json.dumps(manifest, indent=2).encode('utf-8')
        scan_payload('manifest.json', payload)
        archive.writestr('manifest.json', payload)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files) | {'manifest.json'}:
            raise ValueError('Prefix archive inventory differs')
        for name, row in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != row['sha256']:
                raise ValueError('Prefix archive content differs')
    temporary.replace(output)
    return dict(archive=output.name, bytes=output.stat().st_size, sha256=sha(output), files=len(files),
        saved_controls=proof['saved_controls'], endpoint_results=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        result = audit_prefix(args.batch)
        with Path(args.output).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    else:
        result = package(args.batch, args.output)
    print(json.dumps({k: result[k] for k in ('batch', 'archive', 'saved_controls', 'bytes') if k in result}))
