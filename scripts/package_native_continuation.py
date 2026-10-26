"""Export every registered native endpoint with unchanged model and raw bytes.

Auxiliary JSON paths are redacted with a transformation ledger. Inputs used by
the raw endpoint auditor cannot be transformed; their original hashes must hold.
No incomplete batch, successful-case subset or runtime installation is exported.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path
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
            return value
        before=len(changes)
        nested=portable_metadata(nested,changes,pointer+'/$embedded_json')
        if len(changes)!=before:
            return json.dumps(nested,ensure_ascii=False)
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


def package(batch, output):
    batch,output=Path(batch).resolve(),Path(output)
    temporary=output.with_name(output.name+'.partial')
    if output.exists() or temporary.exists():
        raise FileExistsError('Preserve existing complete or partial asset')
    protocol,summary=read(batch/'protocol.json'),read(batch/'summary.json')
    methods=protocol['methods']
    if not methods or len(methods)!=len(set(methods)):
        raise ValueError('Registered jobs must be nonempty and unique')
    if (batch/'failure.json').exists() or summary.get('status')!='complete':
        raise ValueError('Require a normally completed batch')
    if sorted(row.get('job_name',row['method']) for row in summary['methods'])!=sorted(methods):
        raise ValueError('Require every registered endpoint, without selection')
    guard=read(batch/'guard_final.json')
    if type(guard.get('exit_code')) is not int or guard['exit_code']!=0 or guard.get('reason') is not None:
        raise ValueError('Require normal guard completion')
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
    for name in ('source','calibration','resume','analysis',*methods):
        selected.update(p for p in (batch/name).rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and p.suffix in SUFFIXES)
    selected.update(p for p in (batch/'infrastructure').iterdir() if p.is_file() and p.suffix in ('.json','.gz'))
    # These files are directly hash-bound by endpoint and continuation auditors.
    protected={batch/'protocol.json',batch/'summary.json'}
    protected.update(batch/'source'/relative for relative in protocol['source_hashes'])
    protected.update((batch/'calibration').rglob('*.json'))
    protected.update((batch/'resume').rglob('*.json'))
    for method in methods:
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
    args=parser.parse_args()
    package(args.batch,args.output)
