"""Rebuild five broader B1 courses and their interrupted prefix from trusted capsules."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.download_native_asset import download
from scripts.reproduce_native_asset import reproduce
from scripts.package_native_saved_prefix import audit_prefix
from scripts.summarize_broader_b1_cohort import summarize
from scripts.reproduce_public_feedback_cohort import cached_download


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_prefix(archive, expected, directory):
    """Restore exact prefix files beside its recovered child, without overwriting."""
    archive, directory = Path(archive), Path(directory)
    if sha(archive) != expected:
        raise ValueError('Prefix archive differs from the trusted SHA256')
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        manifest = json.loads(source.read('manifest.json'))
        batch = manifest['batch']
        if (manifest['schema'] != 'native_saved_training_prefix_v1' or manifest['batch_complete'] is not False
                or manifest['endpoint_results'] != 0 or not re.fullmatch(r'[A-Za-z0-9_.-]+', batch)
                or batch in ('.', '..') or len(names) != len(set(names))
                or set(names) != set(manifest['files']) | {'manifest.json'}):
            raise ValueError('Invalid saved-prefix inventory or endpoint claim')
        for name in names:
            p = PurePosixPath(name)
            if (p.is_absolute() or p.as_posix() != name or '..' in p.parts or ':' in name or '\\' in name
                    or name != 'manifest.json' and p.parts[0] != batch
                    or stat.S_ISLNK(source.getinfo(name).external_attr >> 16)):
                raise ValueError('Unsafe saved-prefix archive path')
        for name, row in manifest['files'].items():
            payload = source.read(name)
            if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Saved-prefix file differs')
        target = directory / batch
        if target.exists() or target.is_symlink():
            raise FileExistsError('Preserve the restored prefix; never overwrite it')
        # The manifest is container metadata, not another batch's file.
        directory.mkdir(parents=True, exist_ok=True)
        for name in manifest['files']:
            source.extract(name, directory)
    return target, manifest


def restore_auditors(archive, expected, directory):
    """Use the actually executed code for each report's script-identity field."""
    archive, directory = Path(archive), Path(directory)
    if sha(archive) != expected or directory.exists():
        raise ValueError('Audit-source archive changed or output already exists')
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        manifest = json.loads(source.read('manifest.json'))
        if (manifest['schema'] != 'native_broader_audit_sources_v1'
                or set(manifest['seed_sources']) != {'11', '22', '33', '44', '55'}
                or len(names) != len(set(names)) or set(names) != set(manifest['files']) | {'manifest.json'}):
            raise ValueError('Executed audit-source inventory differs')
        for name in names:
            p = PurePosixPath(name)
            if (p.is_absolute() or p.as_posix() != name or '..' in p.parts or ':' in name or '\\' in name
                    or name != 'manifest.json' and p.parts[0] != 'snapshots'
                    or stat.S_ISLNK(source.getinfo(name).external_attr >> 16)):
                raise ValueError('Unsafe audit-source path')
        for name, row in manifest['files'].items():
            payload = source.read(name)
            if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Executed audit-source bytes changed')
        for seed, root in manifest['seed_sources'].items():
            name = root + '/scripts/analyze_native_broader_b1_course.py'
            if name not in manifest['files'] or manifest['files'][name]['sha256'] != manifest['seed_auditor_sha256'][seed]:
                raise ValueError('Per-seed executed auditor identity differs')
        directory.mkdir(parents=True)
        source.extractall(directory)
    return manifest


def run(recipe, output, local_artifacts=None, public_download_cache=None):
    if local_artifacts is not None and public_download_cache is not None:
        raise ValueError('Local preflight and prior public transfer evidence are distinct')
    recipe, output = Path(recipe).resolve(), Path(output).resolve()
    spec = read(recipe)
    if ([r['seed'] for r in spec['capsules']] != [11, 22, 33, 44, 55]
            or spec['scope'] != 'all_five_broader_B1_development_v1'):
        raise ValueError('Require every registered independent seed, not a successful subset')
    for row in [*spec['capsules'], spec['saved_prefix44'], spec['audit_sources']]:
        for key in ('batch', 'archive'):
            if key == 'batch' and row is spec['audit_sources']:
                continue
            value = row[key]
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
                raise ValueError('Capsule identities must be portable basenames')
    if output.exists():
        raise FileExistsError('Preserve previous reproduction evidence')
    reference = ROOT / spec['cohort_report']
    if sha(reference) != spec['cohort_report_sha256']:
        raise ValueError('Trusted cohort reference changed')
    expected = read(reference)
    output.mkdir(parents=True)
    workspace = output / 'workspace'
    runs = workspace / 'runs'
    runs.mkdir(parents=True)
    receipts, batches = [], {}
    def obtain(row, label):
        if local_artifacts is not None:
            archive = Path(local_artifacts) / row['archive']
            if sha(archive) != row['archive_sha256']:
                raise ValueError('Local preflight archive changed')
            return archive, dict(mode='LOCAL_PREFLIGHT_NOT_PUBLIC_DOWNLOAD')
        downloaded = output / (label + '_download')
        if public_download_cache is not None:
            cache = Path(public_download_cache) / (label + '_download')
            if cache.exists():
                if not (cache / 'download_receipt.json').exists():
                    download(row['manifest_url'], cache, row['archive_sha256'], 1200, resume=True)
                proof = cached_download(dict(manifest_url=row['manifest_url'], sha256=row['archive_sha256']), cache, downloaded)
                if proof is None: raise ValueError('Prior public transfer belongs to another capsule')
                return downloaded / row['archive'], dict(mode='VERIFIED_PRIOR_UNAUTHENTICATED_PUBLIC_DOWNLOAD', receipt=proof)
        proof = download(row['manifest_url'], downloaded, row['archive_sha256'], 1200)
        return downloaded / row['archive'], dict(mode='UNAUTHENTICATED_PUBLIC_DOWNLOAD', receipt=proof)
    auditor_archive, auditor_transfer = obtain(spec['audit_sources'], 'audit_sources')
    auditor_directory = output / 'executed_audit_sources'
    auditors = restore_auditors(auditor_archive, spec['audit_sources']['archive_sha256'], auditor_directory)
    for row in spec['capsules']:
        seed = row['seed']
        archive, transfer = obtain(row, 'seed' + str(seed))
        rebuilt = output / ('seed' + str(seed) + '_rebuild')
        # Existing capsules retain the figure's original exported name. The
        # reference remains inside the SHA-bound archive; no comparison is skipped.
        reference_figure = rebuilt / 'asset' / row['batch'] / 'analysis/native_endpoints.png'
        reproduce(archive, row['archive_sha256'], rebuilt, reference_figure=reference_figure)
        original = rebuilt / 'asset' / row['batch']
        if not original.is_relative_to(rebuilt / 'asset') or original.name != row['batch']:
            raise ValueError('Restored course path differs')
        batch = runs / row['batch']
        shutil.copytree(original, batch)
        batches[seed] = batch
        receipts.append(dict(seed=seed, transfer=transfer, reproduction=read(rebuilt / 'reproduction.json')))
        print(json.dumps(dict(seed=seed, raw_inference_figure_rebuilt=True)), flush=True)
    prefix_spec = spec['saved_prefix44']
    archive, transfer = obtain(prefix_spec, 'prefix44')
    prefix, prefix_manifest = restore_prefix(archive, prefix_spec['archive_sha256'], runs)
    if prefix.name != read(batches[44] / 'protocol.json')['parent_batch']:
        raise ValueError('Prefix is not the registered parent of recovered44')
    prefix_proof = audit_prefix(prefix)
    if prefix_proof != prefix_manifest['saved_prefix_proof']:
        raise ValueError('Saved TRAIN prefix does not reproduce exactly')
    (output / 'saved_prefix_rebuilt.json').write_text(json.dumps(prefix_proof, indent=2), encoding='utf-8')
    original_execution = runs / 'B1_additional_broader_cohort_20261102'
    staged_execution = runs / 'public_broader_cohort_routing'
    original_execution.mkdir()
    staged_execution.mkdir()
    (workspace / 'reports').mkdir()
    rows, completed = [], []
    for seed in (11, 22, 33, 44, 55):
        batch = batches[seed]
        audit_path = batch / 'analysis/broader_learning_audit.json'
        snapshot = auditor_directory / auditors['seed_sources'][str(seed)]
        script = snapshot / 'scripts/analyze_native_broader_b1_course.py'
        if sha(script) != read(audit_path)['script_sha256']:
            raise ValueError('Original executed analyzer differs from its recorded identity')
        destination = output / f'seed{seed}_learning_rebuilt.json'
        environment = dict(os.environ, PYTHONPATH=str(snapshot), PYTHONIOENCODING='utf-8',
            NAVSTACK_AUDIT_SOURCE_ROOT=str(batch / 'source'))
        result = subprocess.run([sys.executable, str(script), '--batch', str(batch), '--output', str(destination)],
            cwd=snapshot, env=environment, capture_output=True, timeout=600)
        (output / f'seed{seed}_executed_audit.log').write_bytes(result.stdout + b'\n' + result.stderr)
        if result.returncode:
            raise RuntimeError('Original executed learning auditor failed; inspect preserved log')
        rebuilt = read(destination)
        if rebuilt != read(audit_path):
            raise ValueError('Learning/Adam/RMS/recovery audit differs for seed' + str(seed))
        if seed == 22:
            destination = workspace / 'reports/native_B1_broader_seed22_20261102.json'
        else:
            rows.append(dict(seed=seed, batch=batch.name))
            completed.append(dict(seed=seed, batch=batch.name, audit_sha256=sha(audit_path)))
            destination = (original_execution if seed in (11, 33) else staged_execution) / f'seed{seed}_learning_audit.json'
        shutil.copy2(audit_path, destination)
    # Routing metadata is reconstructed from the fixed registered archive list;
    # it is not a fabricated original process receipt or another driving run.
    (staged_execution / 'registration.json').write_text(json.dumps(dict(rows=rows)), encoding='utf-8')
    (staged_execution / 'status.json').write_text(json.dumps(dict(completed=completed,
        scope='Public reconstruction routing only;not an original observer execution')), encoding='utf-8')
    cohort = summarize(workspace, staged_execution)
    if cohort != expected:
        raise ValueError('Full five-seed outcome/cost report differs')
    (output / 'cohort_rebuilt.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
    proof = dict(scope=__doc__, recipe_sha256=sha(recipe),
        reproduction_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        public_download_reproduction=local_artifacts is None, capsules=receipts,
        executed_audit_sources=dict(transfer=auditor_transfer, archive_sha256=spec['audit_sources']['archive_sha256'],
            verified_files=len(auditors['files']), original_script_identities_preserved=True),
        prefix44=dict(transfer=transfer, archive_sha256=prefix_spec['archive_sha256'],
            verified_files=len(prefix_manifest['files']), verified_controls=prefix_proof['verified_controls'],
            saved_Adam=prefix_proof['saved_Adam'], exact_rebuild=True),
        all_learning_audits_exact=True, complete_cohort_exact=True,
        verified_controls=sum(r['reproduction']['methods'][0]['verified_controls'] for r in receipts) + prefix_proof['verified_controls'],
        verified_new_PPO_controls=cohort['effective_new_PPO_controls'],
        verified_new_Adam_steps=cohort['effective_new_Adam_steps'], endpoint_successes=cohort['endpoint_successes'],
        unique_known_validation_conditions=24, new_controls=0, new_optimizer_updates=0,
        limits='Reconstruction of preserved failed development baselines only. No new driving,training,healthy-baseline qualification,hierarchy advantage or final confirmation.')
    (output / 'public_verification.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps(dict(complete_cohort_exact=True, public_download=local_artifacts is None,
                         controls=proof['verified_controls'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-artifacts', help='Explicit local preflight;never counted as public-download reproduction')
    parser.add_argument('--public-download-cache', help='Previous public transfer output;completed receipts and partial parts are rehashed,only missing parts downloaded')
    args = parser.parse_args()
    run(args.recipe, args.output, args.local_artifacts, args.public_download_cache)
