"""Publish every completed registered pair, then rebuild the full public cohort."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def require_rows(rows):
    if ([r.get('seed') for r in rows]!=[11,22,33,44,55]
            or len({r.get('batch') for r in rows})!=5
            or any(not r.get('protocol_sha256') for r in rows)):
        raise ValueError('Every registered independent pair in its fixed order is required')
    return rows


def missing_objects(rows,assets):
    missing=[]
    for row in rows:
        remote=assets.get(row['name'])
        if remote is None:missing.append(row)
        elif (remote['state']!='uploaded' or remote['size']!=row['bytes']
                or remote['digest']!='sha256:'+row['sha256']):
            raise ValueError('Existing remote object differs; preserve it without overwrite or deletion')
    return missing


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json,resources_ok
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.evaluate_native_rule import require_finished_parent
    from scripts.native_asset_parts import split
    from scripts.package_native_continuation import scan_payload
    plan=read(execution/'plan.json');rows=require_rows(plan['batches']);hashes=read(execution/'source_manifest.json')
    artifact=Path(plan['artifact_directory']);artifact.mkdir();package_checkout=Path(plan['package_checkout'])
    reconstruction_checkout=Path(plan['reconstruction_checkout']);python=root/'runs/prerelease_20261022/clean_checkout/.venv/Scripts/python.exe'
    repo='Leo-7GOAT/Intelligent-driving-system';tag='v0.2.0-dev.20261101-feedback';gh='C:/Program Files/GitHub CLI/gh.exe'
    env=dict(os.environ,HTTPS_PROXY='http://127.0.0.1:7890',HTTP_PROXY='http://127.0.0.1:7890',GODEBUG='http2client=0');env.pop('NAVSTACK_AUDIT_SOURCE_ROOT',None)
    completed=[];failures=[];owner=None;analysis_owner=None;deadline=time.monotonic()+43200
    def status(phase,**extra):atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time(),'completed':completed,'network_failures':failures,**extra})
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen publication implementation changed')
        for checkout,commit in ((package_checkout,plan['package_commit']),(reconstruction_checkout,plan['reconstruction_commit'])):
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()!=commit:raise ValueError('Dedicated independent checkout changed')
            if subprocess.check_output(['git','status','--porcelain'],cwd=checkout,text=True).strip():raise ValueError('Dedicated source checkout must remain clean')
    def call(command,label,cap=900,required=True,cwd=None):
        remaining=min(cap,deadline-time.monotonic())
        if remaining<=0:raise TimeoutError('Finite twelve-hour publication deadline exhausted')
        try:
            result=subprocess.run(list(map(str,command)),cwd=cwd or reconstruction_checkout,env=env,capture_output=True,text=True,timeout=remaining)
            okay=result.returncode==0;output=result.stdout;log=output+'\n'+result.stderr
        except subprocess.TimeoutExpired:okay=False;output='';log='Owned publication command timed out;no native worker affected'
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:stream.write(log)
        if required and not okay:raise RuntimeError('Publication command failed:'+label)
        return okay,output
    def remote(label):
        _,text=call([gh,'api','repos/'+repo+'/releases/tags/'+tag,'--jq','[.assets[]|{name,size,state,digest}]'],label,90)
        return {r['name']:r for r in json.loads(text)}
    def upload(rows,directory,label):
        todo=missing_objects(rows,remote(label+'_remote_before'))
        for attempt in (1,2,3):
            def send(row):
                okay,_=call([gh,'release','upload',tag,Path(directory)/row['name'],'--repo',repo],label+'_'+row['name']+'_attempt'+str(attempt),150,False)
                return row['name'],okay
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for name,okay in pool.map(send,todo):
                    if not okay:failures.append({'asset':name,'attempt':attempt})
                    status('UPLOADING_VERIFIED_COMPLETED_PAIR',asset=name,success=okay)
            assets=remote(label+'_remote_after'+str(attempt));todo=missing_objects(rows,assets)
            if not todo:return assets
        raise RuntimeError('Finite upload attempts exhausted;retain remote objects and local evidence')
    def wait_receipt(path,handle,label):
        while True:
            verify()
            if time.monotonic()>deadline:raise TimeoutError('Finite owned dependency wait exhausted')
            if path.exists():
                try:return read(path)
                except (json.JSONDecodeError,PermissionError):pass
            if handle.poll() is not None:raise RuntimeError('Exact owned dependency ended without required completed receipt:'+label)
            status('WAITING_FOR_EXACT_OWNED_COMPLETED_PAIR',dependency=label);time.sleep(10)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'epoch':time.time(),'plan_sha256':sha(execution/'plan.json')},stream)
            for key in ('training_process','analysis_process'):
                identity=plan[key];handle=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
                if key=='training_process':owner=handle
                else:analysis_owner=handle
            call([gh,'run','watch',str(plan['pipeline_CI_run']),'--repo',repo,'--exit-status','--interval','15'],'publication_code_CI',900)
            _,ci=call([gh,'run','view',str(plan['pipeline_CI_run']),'--repo',repo,'--json','headSha,status,conclusion'],'publication_code_CI_identity',90)
            if json.loads(ci)!={'headSha':plan['reconstruction_commit'],'status':'completed','conclusion':'success'}:
                raise ValueError('Exact publication/reconstruction source CI did not pass')
            _,release=call([gh,'release','view',tag,'--repo',repo,'--json','isPrerelease,targetCommitish'],'release_identity',90)
            if json.loads(release)!={'isPrerelease':True,'targetCommitish':plan['package_commit']}:raise ValueError('Existing development release source differs')
            _,ref=call([gh,'api','repos/'+repo+'/git/ref/tags/'+tag],'release_tag_identity',90)
            if json.loads(ref)['object']['sha']!=plan['package_commit']:raise ValueError('Release tag moved')
            common_source=None;caches=[];capsules=[]
            for row in rows:
                seed=row['seed'];label='seed'+str(seed);batch=root/'runs'/row['batch']
                endpoint=wait_receipt(root/'reports'/('native_matched_broader_feedback_seed'+str(seed)+'_endpoint_20261101.json'),owner,label)
                require_finished_parent(batch);p=read(batch/'protocol.json')
                if (sha(batch/'queued_protocol.json')!=row['protocol_sha256'] or p['seed']!=seed
                        or p['methods']!=['H','H0'] or p['method_budgets']!={'H':20480,'H0':20480}
                        or len(p['native_case_manifest']['cases'])!=36 or len(p['native_validation']['manifest']['cases'])!=24):
                    raise ValueError('Registered pair task,inventory or budget changed')
                if common_source is None:common_source=p['source_hashes']
                if p['source_hashes']!=common_source:raise ValueError('Complete worker source differs across registered pairs')
                for name,digest in common_source.items():
                    if sha(batch/'source'/name)!=digest:raise ValueError('Frozen worker source changed:'+name)
                for method in ('H','H0'):
                    saved=endpoint['methods'][method];folder=batch/method
                    if (sha(folder/'endpoint_raw_audit.json')!=saved['raw_audit_sha256']
                            or sha(folder/'model.zip')!=saved['model_sha256'] or sha(folder/'normalization.pkl')!=saved['normalization_sha256']
                            or saved['cases']!=24 or saved['training_controls']<20480 or saved['actual_Adam_steps']<=0):
                        raise ValueError('Completed actual PPO/raw/normalization proof differs')
                okay,reason=resources_ok(root)
                if not okay:raise RuntimeError(str(reason))
                if seed==11:
                    first=root/'reports/native_matched_broader_seed11_public_verification_20261101.json'
                    if sha(first)!=plan['first_public_receipt_sha256']:raise ValueError('Original public receipt changed')
                    proof=read(first)
                    if not proof['unauthenticated_download_verified'] or not proof['figure_byte_identical']:raise ValueError('First capsule was not publicly reproduced')
                    archive_hash=proof['archive_sha256'];url=proof['public_manifest_url'];cache=Path(plan['first_public_download_cache'])
                    if [r['raw_audit_sha256'] for r in proof['methods']]!=[endpoint['methods'][m]['raw_audit_sha256'] for m in ('H','H0')]:raise ValueError('First capsule has different raw data')
                else:
                    verify();folder=artifact/label;folder.mkdir();status('PACKAGING_COMPLETED_ACTUAL_PPO_PAIR',seed=seed)
                    call([python,package_checkout/'scripts/plot_native_endpoints.py','--batch',batch,'--output',batch/'analysis/frozen_endpoints.png'],label+'_figure',300)
                    archive=folder/('native_matched_broader_H_H0_seed'+str(seed)+'_20261101_repro.zip')
                    call([python,package_checkout/'scripts/package_native_continuation.py','--batch',batch,'--output',archive],label+'_package',900,cwd=package_checkout)
                    archive_hash=sha(archive)
                    with zipfile.ZipFile(archive) as z:manifest=json.loads(z.read('manifest.json'))
                    if manifest['repository_commit']!=plan['package_commit'] or manifest.get('repository_has_local_changes') is not False:
                        raise ValueError('Archive packager source is not the pinned clean checkout')
                    parts=folder/'parts';mp=split(archive,parts,part_bytes=1024**2);m=read(mp)
                    upload(m['parts'],parts,label+'_parts')
                    metadata=[{'name':mp.name,'bytes':mp.stat().st_size,'sha256':sha(mp)}]
                    upload(metadata,parts,label+'_manifest_LAST')
                    url='https://github.com/'+repo+'/releases/download/'+tag+'/'+mp.name;cache=folder/'public_download';rebuild=folder/'public_reconstruction'
                    status('DOWNLOADING_PUBLIC_COMPLETED_PAIR',seed=seed)
                    call([python,package_checkout/'scripts/download_native_asset.py','--manifest-url',url,'--sha256',archive_hash,'--output',cache],label+'_public_download',1200)
                    call([python,package_checkout/'scripts/reproduce_native_asset.py','--archive',cache/archive.name,'--sha256',archive_hash,'--output',rebuild],label+'_public_reconstruction',1200)
                    proof=read(rebuild/'reproduction.json')
                    if (not proof['figure_byte_identical'] or [r['method'] for r in proof['methods']]!=['H','H0']
                            or any(r['measured_start_actions_reproduced']!=24 for r in proof['methods'])
                            or sum(r['verified_controls'] for r in proof['methods'])!=sum(r['raw_controls_verified'] for r in endpoint['methods'].values())):
                        raise ValueError('Publicly reconstructed complete pair differs')
                    proof.update(public_manifest_url=url,unauthenticated_download_verified=True,seed=seed,batch=batch.name,known_unique_validation_conditions=24,research_acceptance='UNMET;development pair,not final confirmation')
                    payload=json.dumps(proof,indent=2).encode();scan_payload(label+'_public_receipt.json',payload)
                    with (root/'reports'/('native_matched_broader_seed'+str(seed)+'_public_verification_20261101.json')).open('xb') as stream:stream.write(payload)
                caches.append(cache);capsules.append({'slot':label,'seed':seed,'batch':batch.name,'methods':['H','H0'],'manifest_url':url,'sha256':archive_hash})
                completed.append({'seed':seed,'batch':batch.name,'public_manifest_url':url,'archive_sha256':archive_hash});status('REGISTERED_PAIR_PUBLICLY_RECONSTRUCTED',seed=seed)
            wait_receipt(root/'runs'/plan['analysis_execution']/'complete.json',analysis_owner,'all_five_analysis')
            references=artifact/'references';references.mkdir()
            names={'pairs_manifest':'protocols/native_matched_broader_feedback_pairs_20261101.json',
                'expected_report':'reports/native_matched_broader_feedback_cohort_20261101.json',
                'expected_figure':'reports/figures/native_matched_broader_feedback_cohort_20261101.png'}
            for name in names.values():
                target=references/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
            recipe={'scope':'All five actually trained common-harness H/H0development pairs from public capsules;24known validation conditions,not120unique final cases.',
                'cohort_kind':'matched_broader_36train_24val_v1','seeds':[11,22,33,44,55],**names,'capsules':capsules}
            recipe_path=artifact/'public_native_matched_broader_feedback_cohort_20261101.json';atomic_json(recipe_path,recipe)
            status('INDEPENDENT_FULL_PUBLIC_COHORT_RECONSTRUCTION')
            command=[python,reconstruction_checkout/'scripts/reproduce_public_feedback_cohort.py','--recipe',recipe_path,'--reference-root',references,'--output',artifact/'full_public_cohort_reconstruction']
            for cache in caches:command.extend(['--public-download-cache',cache])
            call(command,'all_five_public_reconstruction',3600)
            result=read(artifact/'full_public_cohort_reconstruction/reproduction.json')
            if (result['independent_training_seeds']!=5 or result['unique_known_development_conditions']!=24
                    or not result['five_distinct_final_policy_tensors_each_method'] or not result['all_reference_cohort_fields_identical']
                    or not result['figure_byte_identical']):raise ValueError('Complete independent public five-seed proof missing')
            public_metadata=artifact/'public_metadata';public_metadata.mkdir()
            for path in [recipe_path,*[references/name for name in names.values()]]:
                target=public_metadata/path.name;shutil.copy2(path,target);scan_payload(target.name,target.read_bytes())
            upload([{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in public_metadata.iterdir()],public_metadata,'all_five_metadata')
            receipt={**result,'release':'https://github.com/'+repo+'/releases/tag/'+tag,
                'public_metadata_urls':{p.name:'https://github.com/'+repo+'/releases/download/'+tag+'/'+p.name for p in public_metadata.iterdir()},
                'network_failures':failures,'new_controls':0,'new_optimizer_updates':0,'research_acceptance':'UNMET;known development cohort,no healthy B1or final confirmation claim'}
            for path,data in ((root/'protocols/public_native_matched_broader_feedback_cohort_20261101.json',recipe),
                    (root/'reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json',receipt)):
                payload=json.dumps(data,indent=2).encode();scan_payload(path.name,payload)
                with path.open('xb') as stream:stream.write(payload)
            atomic_json(execution/'complete.json',{'seeds':[11,22,33,44,55],'receipt_sha256':sha(root/'reports/native_public_matched_broader_feedback_five_seed_verification_20261101.json'),'new_controls':0,'new_updates':0})
            status('ALL_FIVE_MATCHED_BROADER_PAIRS_PUBLICLY_PUBLISHED_AND_RECONSTRUCTED')
    except BaseException as error:
        atomic_json(execution/'failure.json',{'error':repr(error),'completed':completed,'network_failures':failures,'epoch':time.time()});raise
    finally:
        for handle in (owner,analysis_owner):
            if handle is not None:handle.kernel.CloseHandle(handle.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)
