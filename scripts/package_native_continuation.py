"""Export native endpoints with unchanged model and raw bytes.

Auxiliary JSON paths are redacted with a transformation ledger. Inputs used by
the raw endpoint auditor cannot be transformed; their original hashes must hold.
Whole-batch export rejects incomplete runs. An explicit completed-method profile
retains every incomplete sibling file and failure;it never declares batch success.
No successful-case subset or runtime installation is exported.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_native_method_endpoint import audit

PRIVATE = re.compile(rb'(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]|/(?:home|Users)/[A-Za-z0-9_.-]+/|\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{40,}|\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
SUFFIXES = {'.py','.ps1','.json','.jsonl','.gz','.csv','.png','.zip','.npz','.pkl','.txt','.md'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def operational_guard_files(batch):
    """Bind the actual operational override separately from policy source."""
    batch = Path(batch).resolve()
    execution_path = batch/'launcher_execution.json'
    execution = read(execution_path) if execution_path.exists() else {}
    if not execution.get('guard_current_override'):
        if (batch/'launcher_override').exists():
            raise ValueError('Undeclared operational guard snapshot')
        return set()
    manifest_path = batch/'launcher_override/manifest.json'
    if sha(manifest_path.read_bytes()) != execution['guard_manifest_sha256']:
        raise ValueError('Operational guard manifest differs')
    entries = read(manifest_path)['files']
    if not isinstance(entries, dict) or not 1 <= len(entries) <= 512:
        raise ValueError('Invalid operational guard file inventory')
    root = batch/'launcher_override/source'
    selected = {manifest_path, execution_path}
    permitted_scripts = {'watch_existing_pair.py', 'stop_owned_carla.ps1',
                         'probe_carla37.py', 'run_registered_native.ps1'}
    for relative, expected in entries.items():
        path = PurePosixPath(relative)
        if (path.is_absolute() or path.as_posix() != relative or '\\' in relative or '..' in path.parts or len(path.parts) != 2
                or not ((path.parts[0] == 'navstack' and path.suffix == '.py')
                        or (path.parts[0] == 'scripts' and path.name in permitted_scripts))):
            raise ValueError('Operational source path is not a registered code file')
        source = root/relative
        if not source.resolve().is_relative_to(root.resolve()):
            raise ValueError('Operational source escapes its snapshot')
        if not source.is_file() or source.stat().st_size > 5*1024**2 or sha(source.read_bytes()) != expected:
            raise ValueError('Operational source differs: '+relative)
        selected.add(source)
    guard = batch/execution['guard_source_relative']
    if (execution['guard_source_relative'] != 'launcher_override/source/scripts/watch_existing_pair.py'
            or guard not in selected or sha(guard.read_bytes()) != execution['guard_source_sha256']):
        raise ValueError('Executed guard source differs')
    return selected


def package_guard_snapshot(batch, output, parent_archive_sha256):
    """Supplement an earlier capsule without rewriting its raw evidence."""
    batch, output = Path(batch).resolve(), Path(output)
    if not re.fullmatch('[0-9a-f]{64}', parent_archive_sha256):
        raise ValueError('Require the original capsule SHA256')
    guard = read(batch/'guard_final.json')
    if type(guard.get('exit_code')) is not int or guard['exit_code'] != 0 or guard.get('reason') is not None:
        raise ValueError('Require completed guarded execution')
    if read(batch/'infrastructure/server_stop.json').get('status') != 'owned_empty_server_stopped':
        raise ValueError('Require owned cleanup evidence')
    selected = operational_guard_files(batch)
    if not selected:
        raise ValueError('No operational guard override to supplement')
    temporary = output.with_name(output.name+'.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Preserve existing guard archive')
    output.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            payload = path.read_bytes()
            scan_payload(path.name, payload)
            name = batch.name+'/'+path.relative_to(batch).as_posix()
            archive.writestr(name, payload)
            files[name] = {'bytes': len(payload), 'sha256': sha(payload)}
        manifest = {'scope': 'Exact operational guard supplement; no policy, model, metrics or raw control changes',
                    'batch': batch.name, 'parent_archive_sha256': parent_archive_sha256,
                    'files': files, 'packager_sha256': sha(Path(__file__).read_bytes())}
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    temporary.rename(output)
    result = {'files': len(files), 'bytes': output.stat().st_size, 'sha256': sha(output.read_bytes()),
              'parent_archive_sha256': parent_archive_sha256, 'new_controls': 0, 'optimizer_updates': 0}
    print(json.dumps(result))
    return result


def portable_metadata(value, changes, pointer=''):
    if isinstance(value, dict):
        return {k:portable_metadata(v,changes,pointer+'/'+k.replace('~','~0').replace('/','~1')) for k,v in value.items()}
    if isinstance(value, list):
        return [portable_metadata(v,changes,pointer+'/'+str(i)) for i,v in enumerate(value)]
    if isinstance(value, str) and re.match(r'''^["']?(?:[A-Za-z]:[\\/]|/(?:home|Users)/)''',value):
        digest = sha(value.encode())
        changes.append({'json_pointer':pointer,'original_value_sha256':digest})
        return 'origin_metadata/'+digest[:24]
    if isinstance(value,str) and value.lstrip().startswith(('{','[')):
        try:
            nested=json.loads(value)
        except json.JSONDecodeError:
            nested=None
        if nested is not None:
            before=len(changes)
            nested=portable_metadata(nested,changes,pointer+'/$embedded_json')
            if len(changes)!=before:
                return json.dumps(nested,ensure_ascii=False)
    if isinstance(value,str) and re.search(
            r'(?i)[A-Z]:[\\/]+Users[\\/]+[A-Za-z0-9_.-]+[\\/]|/(?:home|Users)/[A-Za-z0-9_.-]+/', value):
        digest=sha(value.encode())
        changes.append({'json_pointer':pointer,'original_value_sha256':digest,
                        'redaction_scope':'entire auxiliary metadata string containing a private home path'})
        return 'origin_metadata/'+digest[:24]
    return value


def scan_payload(name, payload, depth=0):
    if depth > 8:
        raise ValueError('Unexpected archive nesting')
    if name.endswith('.png'):
        return
    if name.endswith('.gz'):
        scan_payload(name[:-3],gzip.decompress(payload),depth+1)
        return
    if name.endswith(('.zip','.npz')):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.namelist():
                scan_payload(member,archive.read(member),depth+1)
        return
    if name.endswith('.npy'):
        array=np.load(io.BytesIO(payload),allow_pickle=False)
        if array.dtype.hasobject:raise ValueError('Opaque object arrays cannot be published')
        if array.dtype.kind in ('U','S'):
            for value in array.reshape(-1):
                decoded=value.encode('utf-8') if isinstance(value,str) else bytes(value)
                scan_payload(name+'.decoded',decoded,depth+1)
    if PRIVATE.search(payload):
        raise ValueError('Private data candidate in '+name)
    # SB3 embeds cloudpickle bytes in base64 JSON. Scan those without loading or
    # executing the pickle; a plain-text scan would miss encoded source paths.
    if name.endswith('.json') or name == 'data':
        def encoded(value):
            if isinstance(value,dict):
                if ':serialized:' in value:
                    raw=base64.b64decode(value[':serialized:'],validate=True)
                    if PRIVATE.search(raw):
                        raise ValueError('Encoded private data candidate in '+name)
                for item in value.values(): encoded(item)
            elif isinstance(value,list):
                for item in value: encoded(item)
        encoded(json.loads(payload))


def package(batch, output, completed_method=None):
    batch,output=Path(batch).resolve(),Path(output)
    temporary=output.with_name(output.name+'.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Preserve existing complete or partial asset')
    protocol=read(batch/'protocol.json')
    summary=read(batch/'summary.json') if (batch/'summary.json').exists() else {}
    registered_methods=protocol['methods']
    if not registered_methods or len(registered_methods)!=len(set(registered_methods)):
        raise ValueError('Registered jobs must be nonempty and unique')
    profile=None
    if completed_method is None:
        methods=registered_methods
        if (batch/'failure.json').exists() or summary.get('status')!='complete':
            raise ValueError('Require a normally completed batch')
        if sorted(row.get('job_name',row['method']) for row in summary['methods'])!=sorted(methods):
            raise ValueError('Require every registered endpoint, without selection')
        guard=read(batch/'guard_final.json')
        if type(guard.get('exit_code')) is not int or guard['exit_code']!=0 or guard.get('reason') is not None:
            raise ValueError('Require normal guard completion')
    else:
        from scripts.native_completed_method_export import completed_method_profile
        profile=completed_method_profile(batch,protocol,completed_method)
        methods=[completed_method]
    if read(batch/'infrastructure/server_stop.json').get('status')!='owned_empty_server_stopped':
        raise ValueError('Require verified owned-server cleanup')
    for relative,digest in protocol['source_hashes'].items():
        if sha((batch/'source'/relative).read_bytes())!=digest:
            raise ValueError('Frozen source changed')
    endpoint_audits={}
    for method in methods:
        rebuilt=audit(batch,method)
        if rebuilt!=read(batch/method/'endpoint_raw_audit.json'):
            raise ValueError('Endpoint raw audit no longer matches: '+method)
        endpoint_audits[method]=rebuilt
    selected=set(batch.glob('*.json'))
    supervised_dataset=batch/'measured_demonstrations.npz'
    if supervised_dataset.is_file():
        if sha(supervised_dataset.read_bytes())!=protocol.get('extra_supervision',{}).get('dataset_sha256'):
            raise ValueError('Supervised dataset is unregistered or its bytes changed')
        selected.add(supervised_dataset)
    for name in ('source','calibration','resume','analysis',*registered_methods):
        selected.update(p for p in (batch/name).rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and p.suffix in SUFFIXES)
    selected.update(p for p in (batch/'infrastructure').iterdir() if p.is_file() and p.suffix in ('.json','.gz'))
    operational_sources = operational_guard_files(batch)
    selected.update(operational_sources)
    # These files are directly hash-bound by endpoint and continuation auditors.
    protected={batch/'protocol.json',batch/'summary.json'}
    if profile:
        protected.update(batch/name for name in profile['failure_evidence_sha256'])
        protected.update(batch/name for rows in profile['incomplete_file_sha256'].values() for name in rows)
        for method in registered_methods:
            for path in (batch/method).rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix not in SUFFIXES:
                    raise ValueError('Incomplete method file must not be silently omitted: '+path.name)
    protected.update(operational_sources)
    protected.update(batch/'source'/relative for relative in protocol['source_hashes'])
    protected.update((batch/'calibration').rglob('*.json'))
    protected.update((batch/'resume').rglob('*.json'))
    for method in registered_methods:
        folder=batch/method
        protected.update(folder/name for name in ('config.json','summary.json','evaluation.json',
                         'endpoint_raw_audit.json','latest_checkpoint.json'))
        protected.update(folder.glob('checkpoint_*/metadata.json'))
        protected.update(folder.glob('checkpoint_*/config.json'))
    files,transformations={},{}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative=path.relative_to(batch).as_posix()
            original=payload=path.read_bytes()
            changes=[]
            if path.suffix=='.json':
                value=portable_metadata(read(path),changes)
                if changes:
                    if path in protected:
                        raise ValueError('Cannot redact hash-bound input: '+relative)
                    payload=json.dumps(value,indent=2,ensure_ascii=False).encode()
            scan_payload(path.name,payload)
            name=batch.name+'/'+relative
            archive.writestr(name,payload)
            files[name]={'sha256':sha(payload),'bytes':len(payload)}
            if changes:
                transformations[relative]={'original_sha256':sha(original),'export_sha256':sha(payload),
                    'changes':changes,'meaning':'Absolute metadata paths replaced by non-resolving origin identifiers; metrics unchanged'}
        extras={'LICENSE':(ROOT/'LICENSE').read_bytes(),
                'requirements.lock.txt':(ROOT/'requirements.lock.txt').read_bytes(),
                'README.txt':(
                    ('COMPLETED METHOD ONLY; ORIGINAL BATCH INTERRUPTED. ALL INCOMPLETE SIBLING FILES RETAINED.\n' if profile else '')+
                    'DEVELOPMENT EVIDENCE, NOT RESEARCH ACCEPTANCE OR A FINAL MODEL.\n'
                    'All registered methods, failures and controls in this segment are retained. Model/norm/raw bytes are unchanged.\n'
                    'Source, saved checkpoints and inherited resume checkpoints are included; install dependencies independently.\n'
                    'Historical ancestor interaction costs and unknown tails remain charged in protocol/summary; upstream raw traces are not duplicated here.\n'
                    'Auxiliary metadata path redactions are explicit in manifest.json; critical raw-audit inputs cannot be redacted.\n'
                    'No simulator binaries, complete maps, environments or credentials are included. Derived road-cell evidence is included.\n'
                    'Use the repository source commit recorded below with scripts/audit_native_method_endpoint.py for each METHOD.\n'
                    'Raw audit: python scripts/audit_native_method_endpoint.py --batch EXTRACTED_BATCH --method METHOD --output NEW_AUDIT.json\n'
                    'For measured-start inference set NAVSTACK_AUDIT_SOURCE_ROOT to the absolute EXTRACTED_BATCH/source path.\n'
                    'Then: python scripts/audit_native_policy_inference.py --batch EXTRACTED_BATCH --method METHOD --output NEW_INFERENCE.json\n'
                    'That command loads the saved model and freezes normalization; it performs no plant integration or simulator ticks.\n'
                    'Raw states come from CARLA; no downloaded-asset replay should be called a new drive.\n').encode()}
        for name,payload in extras.items():
            scan_payload(name,payload)
            archive.writestr(name,payload)
            files[name]={'sha256':sha(payload),'bytes':len(payload)}
        manifest={'schema_version':1,'scope':'Complete registered native development segment; all failures retained; not final research evidence',
            'batch':batch.name,'methods':methods,'files':files,'metadata_transformations':transformations,
            'packager_sha256':sha(Path(__file__).read_bytes()),
            'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'repository_has_local_changes':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
            'endpoint_auditor_sha256':sha((ROOT/'scripts/audit_native_method_endpoint.py').read_bytes()),
            'verified_controls':{m:a['total_applied_controls_verified'] for m,a in endpoint_audits.items()},
            'not_exported':[p.relative_to(batch).as_posix() for p in sorted(batch.rglob('*'))
                            if p.is_file() and p not in selected and '__pycache__' not in p.parts]}
        if profile:
            manifest.update(schema_version=2,scope=profile['scope'],completed_method_export=profile,
                retained_incomplete_method_files={method:sorted(name for name in files if name.startswith(batch.name+'/'+method+'/'))
                    for method in profile['incomplete_methods']})
        manifest_bytes=json.dumps(manifest,indent=2).encode()
        scan_payload('manifest.json',manifest_bytes)
        archive.writestr('manifest.json',manifest_bytes)
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist())!=set(files)|{'manifest.json'}:
            raise ValueError('Archive inventory mismatch')
        for name,row in files.items():
            if sha(archive.read(name))!=row['sha256']:
                raise ValueError('Archive content mismatch')
    temporary.replace(output)
    result={'files':len(files),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),
            'verified_controls':manifest['verified_controls'],'metadata_files_transformed':len(transformations)}
    print(json.dumps(result))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--completed-method',help='Explicit complete method in an interrupted batch;all incomplete sibling data retained')
    parser.add_argument('--guard-only',action='store_true',help='Supplement a previously published capsule with its exact operational guard snapshot')
    parser.add_argument('--parent-archive-sha256')
    args=parser.parse_args()
    if args.guard_only:
        if args.completed_method:parser.error('--completed-method cannot be combined with --guard-only')
        package_guard_snapshot(args.batch,args.output,args.parent_archive_sha256 or '')
    else:
        if args.parent_archive_sha256:parser.error('--parent-archive-sha256 requires --guard-only')
        package(args.batch,args.output,args.completed_method)
