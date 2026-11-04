"""Publish complete measured TRAIN provenance, including interrupted validation.

This is deliberately separate from the complete-endpoint exporter. It neither
repairs the interrupted trial nor assigns it a complete evaluation score.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_native_continuation import (
    SUFFIXES, operational_guard_files, portable_metadata, read, scan_payload, sha,
)


def package(batch, views, reference_config, output):
    batch, views, reference_config, output = map(
        lambda p: Path(p).resolve(), (batch, views, reference_config, output))
    temporary = output.with_name(output.name + '.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Preserve existing complete or partial exports')
    protocol = read(batch / 'protocol.json')
    manifest = read(views / 'manifest.json')
    summary = read(batch / 'H/summary.json')
    interruption = read(batch / 'interruption_audit.json')
    prefix = interruption['methods']['H']
    if (manifest['batch'] != batch.name or manifest['method'] != 'H'
            or manifest['new_controls'] or manifest['PPO_updates']
            or manifest['supervised_updates'] or summary['stopped_reason'] is not None
            or not summary['optimizer_steps'] or prefix['unknown_unflushed_tail']
            or prefix['trace_errors'] or prefix['optimization_error']
            or prefix['readable_trace_controls'] != summary['physics_steps']
            or manifest['teacher_training_controls'] != summary['physics_steps']
            or interruption['methods']['H0']['state'] != 'not_started'):
        raise ValueError('Require complete H training with disclosed H0 not started')
    if (read(batch / 'infrastructure/server_stop.json')['status']
            != 'owned_empty_server_stopped'):
        raise ValueError('Require verified owned cleanup')
    bindings = {
        batch / 'protocol.json': manifest['protocol_sha256'],
        batch / 'H/summary.json': manifest['teacher_summary_sha256'],
        reference_config: manifest['reference_config_sha256'],
        views / 'measured_training_views.npz': manifest['dataset_sha256'],
        views / 'executed_view_builder.py': manifest['script_sha256'],
        views / 'executed_tick_verifier.py': manifest['tick_verifier_sha256'],
        batch / 'calibration/calibration.json': protocol['calibration_sha256'],
    }
    bindings.update({batch / 'source' / p: h for p, h in protocol['source_hashes'].items()})
    bindings.update({batch / 'H' / p: h for p, h in manifest['trace_hashes'].items()})
    bindings.update({batch / 'infrastructure' / p: h for p, h in manifest['bridge_hashes'].items()})
    for path, digest in bindings.items():
        if sha(path.read_bytes()) != digest:
            raise ValueError('Bound provenance changed: ' + path.name)
    selected = set(batch.glob('*.json'))
    protected = set(bindings)
    for folder in ('source', 'calibration', 'resume', 'H'):
        paths = {p for p in (batch / folder).rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.suffix in SUFFIXES}
        selected.update(paths)
        # All learning inputs, checkpoints, counters and raw evidence stay exact.
        protected.update(paths)
    selected.update(operational_guard_files(batch))
    protected.update(operational_guard_files(batch))
    selected.update(p for p in (batch / 'infrastructure').iterdir()
                    if p.is_file() and p.suffix in ('.json', '.gz'))
    members = {batch.name + '/' + p.relative_to(batch).as_posix(): p for p in selected}
    target = reference_config.parent.parent
    for folder in (views, target / 'source'):
        for path in folder.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix in SUFFIXES:
                name = (views.name + '/' + path.relative_to(views).as_posix()
                        if folder == views else target.name + '/source/' + path.relative_to(folder).as_posix())
                members[name] = path
                protected.add(path)
    members[target.name + '/B1/config.json'] = reference_config
    files, transformations = {}, {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(members.items()):
            original = payload = path.read_bytes()
            changes = []
            if path.suffix == '.json':
                value = portable_metadata(read(path), changes)
                if changes:
                    if path in protected:
                        raise ValueError('Cannot redact learning/raw input: ' + name)
                    payload = json.dumps(value, indent=2).encode()
                    transformations[name] = {
                        'original_sha256': sha(original), 'export_sha256': sha(payload),
                        'changes': changes,
                    }
            scan_payload(path.name, payload)
            archive.writestr(name, payload)
            files[name] = {'sha256': sha(payload), 'bytes': len(payload)}
        members_packager = 'tools/package_native_training_views.py'
        payload = Path(__file__).read_bytes()
        scan_payload(members_packager, payload)
        archive.writestr(members_packager, payload)
        files[members_packager] = {'sha256': sha(payload), 'bytes': len(payload)}
        for name in ('LICENSE', 'requirements.lock.txt'):
            payload = (ROOT / name).read_bytes()
            scan_payload(name, payload)
            archive.writestr(name, payload)
            files[name] = {'sha256': sha(payload), 'bytes': len(payload)}
        result = {
            'schema_version': 1,
            'scope': 'Complete H TRAIN provenance and every available interrupted validation record;not a complete evaluation or final research evidence',
            'teacher_batch': batch.name, 'views': views.name,
            'reference_config': target.name + '/B1/config.json',
            'teacher_training_controls': summary['physics_steps'],
            'teacher_optimizer_steps': summary['optimizer_steps'],
            'charged_controls_lower_bound': summary['charged_control_steps_lower_bound'],
            'sampled_training_rows': manifest['sampled_rows'],
            'dataset_sha256': manifest['dataset_sha256'],
            'interrupted_validation_preserved': True,
            'unstarted_methods': ['H0'], 'research_acceptance': 'UNMET',
            'metadata_transformations': transformations, 'files': files,
            'packager_sha256': sha(Path(__file__).read_bytes()),
            'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'repository_has_local_changes': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
            'not_exported': [p.relative_to(batch).as_posix() for p in sorted(batch.rglob('*'))
                             if p.is_file() and p not in selected and '__pycache__' not in p.parts],
            'reconstruction': 'Set NAVSTACK_AUDIT_SOURCE_ROOT to extracted teacher/source;run extracted views/executed_view_builder.py with --batch teacher --reference-config reference_config --output a fresh directory --wall-seconds 900. This rechecks all measured TRAIN controls;it does not drive or learn.',
        }
        payload = json.dumps(result, indent=2).encode()
        scan_payload('manifest.json', payload)
        archive.writestr('manifest.json', payload)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist()) != set(files) | {'manifest.json'}:
            raise ValueError('Archive inventory mismatch')
        for name, row in files.items():
            if sha(archive.read(name)) != row['sha256']:
                raise ValueError('Archive content mismatch: ' + name)
    temporary.rename(output)
    receipt = {'files': len(files), 'bytes': output.stat().st_size,
               'sha256': sha(output.read_bytes()), 'training_controls': summary['physics_steps'],
               'sampled_training_rows': manifest['sampled_rows'], 'new_controls': 0, 'PPO_updates': 0}
    print(json.dumps(receipt))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('batch', 'views', 'reference-config', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    package(args.batch, args.views, args.reference_config, args.output)
