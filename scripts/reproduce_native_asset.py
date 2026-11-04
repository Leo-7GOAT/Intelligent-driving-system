"""Verify a trusted native release ZIP and rebuild all endpoint evidence offline.

The expected archive SHA256 must come from the published release record. Model
loading executes trusted serialized policy data only after every file is hashed.
This is raw-evidence/inference reproduction, not additional CARLA driving.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inference_auditor(protocol):
    context=protocol.get('reference_context')
    if context is None:return 'audit_native_policy_inference.py'
    from navstack.reference_context_policy import VERSION
    from navstack.outcome_context_policy import VERSION as OUTCOME_VERSION
    if context not in ({'version':VERSION}, {'version':OUTCOME_VERSION}):
        raise ValueError('Unsupported registered reference-context schema')
    return 'audit_native_reference_inference.py'


def reproduce(archive_path, expected_sha256, output, reference_figure=None):
    archive_path, output = Path(archive_path).resolve(), Path(output).resolve()
    if digest(archive_path.read_bytes()) != expected_sha256:
        raise ValueError('Archive hash differs from the published trusted digest')
    if output.exists():
        raise FileExistsError('Preserve previous reproduction output')
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entry')
        for name in names:
            path = PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or ':' in name or '\\' in name
                    or path.as_posix() != name):
                raise ValueError('Unsafe ZIP path')
        manifest = json.loads(archive.read('manifest.json'))
        if set(names) != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('ZIP inventory differs from manifest')
        for name, row in manifest['files'].items():
            payload = archive.read(name)
            if len(payload) != row['bytes'] or digest(payload) != row['sha256']:
                raise ValueError('Manifest content mismatch: '+name)
        output.mkdir(parents=True)
        archive.extractall(output/'asset')
    batch = output/'asset'/manifest['batch']
    if not batch.is_relative_to(output/'asset'):
        raise ValueError('Batch path escapes extracted asset')
    protocol = json.loads((batch/'protocol.json').read_text(encoding='utf-8'))
    inference_script=inference_auditor(protocol)
    reference_context=inference_script=='audit_native_reference_inference.py'
    from scripts.native_completed_method_export import validate_exported_inventory
    partial=validate_exported_inventory(batch,protocol,manifest)
    evidence = output/'rebuild'
    evidence.mkdir()
    environment = dict(os.environ, NAVSTACK_AUDIT_SOURCE_ROOT=str(batch/'source'))

    def run(script, *arguments):
        command = [sys.executable, str(ROOT/'scripts'/script), *map(str, arguments)]
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        with (evidence/(script+'.log')).open('a', encoding='utf-8') as log:
            log.write(result.stdout+'\n'+result.stderr)
        if result.returncode:
            raise RuntimeError(script+' failed; inspect preserved logs')

    rows = []
    for method in manifest['methods']:
        raw = evidence/(method+'_raw_audit.json')
        run('audit_native_method_endpoint.py', '--batch', batch, '--method', method, '--output', raw)
        if raw.read_bytes() != (batch/method/'endpoint_raw_audit.json').read_bytes():
            raise ValueError('Raw audit changed: '+method)
        inference = evidence/(method+'_inference.json')
        run(inference_script, '--batch', batch, '--method', method, '--output', inference)
        result = json.loads(inference.read_text(encoding='utf-8'))
        raw_result = json.loads(raw.read_text(encoding='utf-8'))
        rows.append({'method':method, 'raw_audit_sha256':digest(raw.read_bytes()),
            'verified_controls':raw_result['total_applied_controls_verified'],
            'measured_start_actions_reproduced':len(result['results']),
            'normalization_frozen':result['inference_normalization_frozen'],
            'new_controls':result['new_controls'], 'optimizer_updates':result['new_updates'] if reference_context else result['optimizer_updates']})
        if reference_context:
            static_path=evidence/(method+'_static_inference.json')
            run(inference_script,'--batch',batch,'--method',method,'--all-static-actions','--output',static_path)
            static=json.loads(static_path.read_text(encoding='utf-8'))
            if not result['all_recorded_actions_match'] or not static['all_recorded_actions_match']:
                raise ValueError('Reference-context action reconstruction failed')
            rows[-1].update(static_cases_reproduced=len(static['results']),
                static_actions_reproduced=sum(r['decisions'] for r in static['results']))
        print(json.dumps(rows[-1]), flush=True)
    engine=None
    if reference_context:
        engine_path=evidence/'strict_delivery.json'
        run('audit_native_control_delivery.py','--batch',batch,'--output',engine_path)
        engine=json.loads(engine_path.read_text(encoding='utf-8'))
        if (not engine['strict_delivery_gate_passed']
                or engine['counts']['controls']!=sum(r['verified_controls'] for r in rows)):
            raise ValueError('Reference-context engine-delivery verification failed')
    figure = evidence/'frozen_endpoints.png'
    selection=[argument for method in manifest['methods'] for argument in ('--method',method)] if partial else []
    run('plot_native_endpoints.py', '--batch', batch, '--output', figure,*selection)
    expected_figure = Path(reference_figure).resolve() if reference_figure is not None else batch/'analysis/frozen_endpoints.png'
    if not expected_figure.is_file():
        raise FileNotFoundError('Figure baseline missing; supply the versioned repository figure with --reference-figure')
    if figure.read_bytes() != expected_figure.read_bytes():
        raise ValueError('Rebuilt figure differs; inspect renderer/dependencies before claiming reproduction')
    receipt = {'scope':'Downloaded trusted asset raw audit, measured-start inference and plot reproduction; no new driving',
        'archive_sha256':expected_sha256,'asset_source_commit':manifest['repository_commit'],
        'reproduction_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'verified_files':len(manifest['files']), 'methods':rows,
        'figure_sha256':digest(figure.read_bytes()), 'figure_byte_identical':True,
        'figure_baseline':'explicit_reference' if reference_figure is not None else 'packaged_analysis'}
    if engine is not None:
        receipt.update(reference_context_reproduced=True,strict_engine_delivery_reproduced=True,
            engine_controls_verified=engine['counts']['controls'],
            scope='Trusted asset raw frames,reference-context starts/all static actions,engine delivery and figure reproduced;zero new driving')
    if partial:
        receipt.update(completed_method_export=manifest['completed_method_export'],
            incomplete_sibling_files_verified=sum(len(names) for names in manifest['retained_incomplete_method_files'].values()),
            batch_complete=False,scope='Completed-method raw/inference/figure reproduction inside a retained interrupted batch;all incomplete sibling file bytes verified;no sibling performance score or new driving')
    (output/'reproduction.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-figure', help='Explicit versioned figure baseline when the archive has no analysis figure')
    args = parser.parse_args()
    reproduce(args.archive, args.sha256, args.output,args.reference_figure)
