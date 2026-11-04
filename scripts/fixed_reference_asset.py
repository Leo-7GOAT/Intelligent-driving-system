"""Publish and independently replay the complete fixed-reference control study."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.package_native_continuation import scan_payload


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def package(batch, partial, references, output):
    batch, partial, references, output = map(Path, (batch, partial, references, output))
    if output.exists() or output.with_suffix('.partial').exists():
        raise FileExistsError('Preserve previous asset or interrupted packaging')
    protocol = read(batch/'protocol.json')
    summary = read(batch/'summary.json')
    prefix = read(partial/'partial_summary.json')
    if (summary['status'] != 'complete' or len(summary['episodes']) != 80
            or read(partial/'failure.json')['completed_episodes'] != len(prefix['episodes'])):
        raise ValueError('Require all completed episodes and the declared failed prefix')
    payloads = {}

    def add(name, path, expected=None):
        payload = Path(path).read_bytes()
        if expected is not None and digest(payload) != expected:
            raise ValueError('Registered input changed: '+name)
        if name in payloads:
            raise ValueError('Duplicate asset member')
        scan_payload(name, payload)
        payloads[name] = payload

    source_rule = ROOT/'runs'/protocol['source_rule_batch']
    for run in (batch, partial):
        registered = read(run/'protocol.json')
        for path in sorted(run.iterdir()):
            if path.is_file() and (path.suffix == '.json' or path.name.endswith('.jsonl.gz')):
                add(run.name+'/'+path.name, path)
        for name, sha in registered['source_hashes'].items():
            original = run/'source'/name
            add(run.name+'/source/'+name, original if original.exists() else ROOT/name, sha)
        for name, sha in registered['source_scene_hashes'].items():
            add(run.name+'/source_scenes/'+name, source_rule/name, sha)
    for path in sorted(references.glob('*.json')):
        add('fixed_references/'+path.name, path,
            protocol['fixed_references_manifest_sha256'] if path.name == 'manifest.json' else None)
    add('calibration.json', source_rule/'calibration/calibration.json', protocol['calibration_sha256'])
    add('analysis/verified.json', ROOT/'reports/fixed_reference_controllers_verified_20261028.json')
    add('analysis/controllers.png', ROOT/'reports/figures/fixed_reference_controllers_20261028.png')
    manifest = {
        'version': 'fixed_reference_asset_v1', 'batch': batch.name, 'partial_batch': partial.name,
        'repository_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'VehiclePlant control comparison only; no CARLA or neural training evidence.',
        'complete_controls': sum(r['physics_steps'] for r in summary['episodes']),
        'partial_controls': sum(r['physics_steps'] for r in prefix['episodes']),
        'partial_status': 'Failed reference check; preserve all traces, never pool with complete scores.',
        'files': {name: {'bytes': len(data), 'sha256': digest(data)} for name, data in payloads.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    output_hash = digest(temporary.read_bytes())
    verify(temporary, output_hash)
    temporary.rename(output)
    result = {'sha256': output_hash, 'bytes': output.stat().st_size, 'files': len(payloads),
              'complete_controls': manifest['complete_controls'], 'partial_controls': manifest['partial_controls']}
    print(json.dumps(result))
    return result


def verify(path, expected):
    if digest(Path(path).read_bytes()) != expected:
        raise ValueError('Archive differs from explicitly trusted release hash')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate archive member')
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or '..' in parsed.parts or ':' in name or '\\' in name or parsed.as_posix() != name:
                raise ValueError('Unsafe archive path')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest['version'] != 'fixed_reference_asset_v1' or set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Unexpected asset inventory')
        for name, row in manifest['files'].items():
            data = archive.read(name)
            if len(data) != row['bytes'] or digest(data) != row['sha256']:
                raise ValueError('Asset member differs from manifest')
        if manifest['batch'] not in {PurePosixPath(name).parts[0] for name in names}:
            raise ValueError('Complete batch missing')
        return manifest


def reproduce(path, expected, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous reproduction')
    manifest = verify(path, expected)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(output/'asset')
    batch = output/'asset'/manifest['batch']
    if not batch.is_relative_to(output/'asset'):
        raise ValueError('Batch escapes asset')
    # Git normalizes text line endings. Execute the exact hash-registered
    # dynamics bytes in an isolated source tree instead of changing any hash
    # or overwriting the user's checkout to match historical CRLF files.
    runtime = output/'runtime'
    for directory in ('navstack', 'scripts'):
        for source in (ROOT/directory).rglob('*.py'):
            target = runtime/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name, expected_source in read(batch/'protocol.json')['source_hashes'].items():
        source = batch/'source'/name
        if digest(source.read_bytes()) != expected_source:
            raise ValueError('Archived experiment source differs from registration')
        shutil.copy2(source, runtime/name)
    for name, arguments in (
        ('analyze_fixed_reference_benchmark.py', ['--batch', batch, '--output', output/'verified.json']),
        ('plot_fixed_reference_benchmark.py', ['--batch', batch, '--output-prefix', output/'controllers']),
    ):
        result = subprocess.run([sys.executable, str(runtime/'scripts'/name), *map(str, arguments)],
                                cwd=runtime, capture_output=True, text=True)
        (output/(name+'.log')).write_text(result.stdout+'\n'+result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError('Reproduction failed: '+name+'; inspect preserved logs')
    if read(output/'verified.json') != read(output/'asset/analysis/verified.json'):
        raise ValueError('Independently rebuilt metrics differ')
    figure = (output/'controllers.png').read_bytes()
    if figure != (output/'asset/analysis/controllers.png').read_bytes():
        raise ValueError('Figure bytes differ; inspect renderer/dependencies')
    result = {'archive_sha256': expected, 'verified_files': len(manifest['files']),
              'replayed_complete_controls': read(output/'verified.json')['raw_control_frames_replayed'],
              'preserved_partial_controls': manifest['partial_controls'],
              'partial_verification': 'Archive/file digests only; excluded from complete-study replay and scores.',
              'figure_byte_identical': True, 'figure_sha256': digest(figure),
              'new_physical_controls': 0, 'optimizer_updates': 0}
    (output/'reproduction.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('package')
    for name in ('batch', 'partial', 'references', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('reproduce')
    for name in ('archive', 'sha256', 'output'):
        p.add_argument('--'+name, required=True)
    args = parser.parse_args()
    if args.mode == 'package':
        package(args.batch, args.partial, args.references, args.output)
    else:
        reproduce(args.archive, args.sha256, args.output)
