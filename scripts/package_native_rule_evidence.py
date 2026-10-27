"""Export complete rule development batches and explicitly interrupted prefixes.

All registered conditions and failures are retained. Runtime/environment files
are excluded; private auxiliary metadata is transformed with a hash ledger.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_native_rule import audit
from scripts.package_native_continuation import portable_metadata,scan_payload,sha,SUFFIXES,read


def package(batches,output):
    output=Path(output);temporary=output.with_suffix(output.suffix+'.partial')
    if output.exists() or temporary.exists():raise FileExistsError('Preserve existing complete or partial asset')
    files={};changes={};audits={};payloads={};invalid_auxiliary_json=[]
    for batch in map(Path,batches):
        partial=not (batch/'summary.json').exists()
        rebuilt=audit(batch,allow_completed_prefix=partial)
        receipt=batch/('completed_prefix_raw_audit.json' if partial else 'raw_audit.json')
        stored=read(receipt)
        if not {'protocol_sha256','trace_sha256','evaluation_controls_verified','cases_verified','termination_counts'}<=set(stored):
            raise ValueError('Original raw audit is missing required evidence')
        if any(rebuilt.get(key)!=value for key,value in stored.items()):
            raise ValueError('Prior raw audit changed')
        guard=read(batch/'guard_final.json')
        if (guard.get('reason') is not None or guard.get('exit_code')!=(1 if partial else 0)
                or read(batch/'infrastructure/server_stop.json')['status']!='owned_empty_server_stopped'):
            raise ValueError('Require verified complete or explicitly failed owned lifecycle')
        audits[batch.name]=rebuilt;protocol=read(batch/'protocol.json')
        protected={batch/'protocol.json',batch/'source_config.json',receipt}
        protected.update(batch/'source'/name for name in protocol['source_hashes'])
        protected.update((batch/'calibration').glob('*.json'))
        protected.update(batch.glob('scene_*.json'));protected.update(batch.glob('case_*.json'))
        protected.update(p for p in (batch/'summary.json',batch/'failure.json') if p.exists())
        for path in sorted(batch.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix not in SUFFIXES:continue
            relative=batch.name+'/'+path.relative_to(batch).as_posix()
            original=payload=path.read_bytes();edits=[];empty_failed_output=False
            if path.suffix=='.json':
                try:value=portable_metadata(read(path),edits)
                except json.JSONDecodeError as exc:
                    if path in protected or original.strip():raise
                    # A failed preflight may leave its exclusive output empty.
                    # Preserve those bytes and label them, never fabricate JSON.
                    invalid_auxiliary_json.append({'file':relative,'error':str(exc),'sha256':sha(original)})
                    value=None;empty_failed_output=True
                if edits:
                    if path in protected:raise ValueError('Cannot redact hash-bound raw input: '+relative)
                    payload=json.dumps(value,indent=2,ensure_ascii=False).encode()
            scan_payload(path.stem+'.txt' if empty_failed_output else path.name,payload)
            if relative in payloads:raise ValueError('Duplicate batch/file selection')
            payloads[relative]=payload;files[relative]={'sha256':sha(payload),'bytes':len(payload)}
            if edits:changes[relative]={'original_sha256':sha(original),'export_sha256':sha(payload),'changes':edits}
    manifest={'schema_version':1,'scope':'All selected rule-only development conditions, including failures and declared incomplete prefixes; no PPO, final confirmation or independent-seed claim',
        'audits':audits,'files':files,'metadata_transformations':changes,
        'invalid_auxiliary_json_preserved':invalid_auxiliary_json,
        'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'repository_has_local_changes':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        'packager_sha256':sha(Path(__file__).read_bytes()),'new_controls':0,'optimizer_updates':0}
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED) as archive:
        for name,payload in payloads.items():archive.writestr(name,payload)
        archive.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temporary) as archive:
        if set(archive.namelist())!=set(files)|{'manifest.json'}:raise ValueError('Archive inventory mismatch')
        for name,row in files.items():
            if sha(archive.read(name))!=row['sha256']:raise ValueError('Archive byte hash mismatch')
    temporary.replace(output)
    result={'files':len(files),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),'verified_controls':sum(a['evaluation_controls_verified'] for a in audits.values()),'batches':list(audits)}
    print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--batches',nargs='+',required=True);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();package(args.batches,args.output)
