"""Rebuild measured training labels from a trusted public provenance capsule."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import zipfile


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def reproduce(archive_path, expected_sha256, output):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    started = time.monotonic()
    if digest(archive_path.read_bytes()) != expected_sha256:
        raise ValueError('Archive differs from the published trusted digest')
    if output.exists():
        raise FileExistsError('Preserve existing reproduction output')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP member')
        for name in names:
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or ':' in name
                    or '\\' in name or path.as_posix() != name):
                raise ValueError('Unsafe ZIP path')
        manifest = json.loads(archive.read('manifest.json'))
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Archive inventory mismatch')
        for name, row in manifest['files'].items():
            payload = archive.read(name)
            if len(payload) != row['bytes'] or digest(payload) != row['sha256']:
                raise ValueError('Archive member changed: ' + name)
        output.mkdir(parents=True)
        archive.extractall(output / 'asset')
    asset = output / 'asset'
    batch, views, reference = [asset / manifest[k] for k in
                               ('teacher_batch', 'views', 'reference_config')]
    if any(not p.resolve().is_relative_to(asset.resolve()) for p in (batch, views, reference)):
        raise ValueError('Declared reconstruction path escapes extracted asset')
    environment = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'))
    rebuilt = output / 'rebuilt_training_views'
    command = [sys.executable, str(views / 'executed_view_builder.py'),
               '--batch', str(batch), '--reference-config', str(reference),
               '--output', str(rebuilt), '--wall-seconds', '900']
    # Only this bounded offline child is owned here;no simulator is started.
    with (output / 'reconstruction.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=asset, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, timeout=960)
    if result.returncode:
        raise RuntimeError('Offline reconstruction failed;all logs and partial evidence retained')
    original = json.loads((views / 'manifest.json').read_text(encoding='utf-8'))
    rebuilt_manifest = json.loads((rebuilt / 'manifest.json').read_text(encoding='utf-8'))
    for key in ('cases', 'teacher_training_controls', 'teacher_learning_optimizer_steps',
                'teacher_charged_controls_lower_bound', 'trace_hashes', 'bridge_hashes',
                'training_conditions', 'sampled_rows', 'dataset_sha256'):
        if original[key] != rebuilt_manifest[key]:
            raise ValueError('Rebuilt training provenance differs: ' + key)
    actual = digest((rebuilt / 'measured_training_views.npz').read_bytes())
    if actual != manifest['dataset_sha256']:
        raise ValueError('Rebuilt dataset is not byte identical')
    receipt = {
        'scope': 'Actual offline reconstruction of all measured TRAIN controls and fixed-stride labels;no new driving or fitting',
        'archive_sha256': expected_sha256, 'asset_source_commit': manifest['repository_commit'],
        'verified_files': len(manifest['files']),
        'verified_training_controls': rebuilt_manifest['teacher_training_controls'],
        'training_conditions': rebuilt_manifest['training_conditions'],
        'training_episodes_including_unfinished': len(rebuilt_manifest['cases']),
        'sampled_rows': rebuilt_manifest['sampled_rows'],
        'dataset_sha256': actual, 'dataset_byte_identical': True,
        'all_training_provenance_fields_exact': True,
        'interrupted_validation_not_scored': True,
        'wall_seconds': time.monotonic() - started,
        'new_controls': 0, 'PPO_updates': 0, 'supervised_updates': 0,
        'research_acceptance': 'UNMET',
    }
    (output / 'reproduction.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('archive', 'sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    reproduce(args.archive, args.sha256, args.output)
