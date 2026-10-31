"""Rebuild all five DEVELOPMENT feedback pairs from six public evidence capsules."""
import argparse
import concurrent.futures
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.analyze_native_feedback_cohort import run as analyze_cohort,resolve_batch
from scripts.download_native_asset import download,release_url
from scripts.reproduce_native_asset import reproduce
from scripts.native_asset_parts import validate_manifest


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def validate_recipe(recipe):
    if recipe['seeds']!=[11,22,33,44,55] or len(recipe['capsules'])!=6:
        raise ValueError('The complete canonical five-seed/six-capsule inventory is required')
    slots=[r['slot'] for r in recipe['capsules']];batches=[r['batch'] for r in recipe['capsules']]
    if len(set(slots))!=6 or len(set(batches))!=6:
        raise ValueError('Repeated capsule slots/batches are not independent inputs')
    for row in recipe['capsules']:
        for key in ('slot','batch'):
            if not re.fullmatch(r'[A-Za-z0-9_.-]+',row[key]) or row[key] in ('.','..'):
                raise ValueError('Safe direct-child capsule identifiers required')
        release_url(row['manifest_url'])
        if not re.fullmatch(r'[0-9a-f]{64}',row['sha256']):raise ValueError('Explicit trusted capsule hash required')
        if not row['methods'] or len(set(row['methods']))!=len(row['methods']) or set(row['methods'])-{'H','H0'}:
            raise ValueError('Only explicit complete H/H0method inventories are replayed')
        if row.get('reference_figure') is not None:resolve_batch(ROOT,row['reference_figure'])
    for key in ('pairs_manifest','expected_report','expected_figure'):resolve_batch(ROOT,recipe[key])


def model_proof(inputs,pairs):
    import pickle
    import numpy as np
    import torch
    from navstack.independent_clip_ppo import load_ppo
    torch.set_num_threads(1);rows=[];signatures={'H':set(),'H0':set()}
    for pair in pairs:
        for method,key in (('H','h'),('H0','h0')):
            batch=resolve_batch(inputs,pair[key]);folder=batch/method
            protocol=read(batch/'protocol.json');summary=read(folder/'summary.json')
            origin=batch/protocol['recovery'][method]['checkpoint']
            model=load_ppo(folder/'model.zip',device='cpu')
            steps={int(s['step']) for s in model.policy.optimizer.state.values() if 'step' in s}
            if steps!={summary['optimizer_steps']} or not steps or summary['optimizer_steps']<=0:
                raise ValueError('Actual per-parameter Adam steps differ from the audited learner')
            if model.num_timesteps!=summary['actions']:raise ValueError('Saved PPOdecision count differs')
            h=hashlib.sha256()
            for name,value in sorted(model.policy.state_dict().items()):
                array=value.detach().cpu().numpy()
                h.update(name.encode()+b'\0'+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
            signatures[method].add(h.hexdigest())
            with (origin/'normalization.pkl').open('rb') as stream:before=pickle.load(stream)
            with (folder/'normalization.pkl').open('rb') as stream:after=pickle.load(stream)
            left,right=[obj.__dict__['obs_rms'] for obj in (before,after)]
            for attribute in ('mean','var'):
                a,b=getattr(left,attribute),getattr(right,attribute)
                if a.dtype!=b.dtype:raise ValueError('Observation statistic dtype changed')
                np.testing.assert_array_equal(a,b)
            if left.count!=right.count:raise ValueError('Observation statistic count changed')
            rows.append({'seed':pair['seed'],'method':method,'batch':batch.name,
                'model_sha256':sha(folder/'model.zip'),'policy_tensor_sha256':h.hexdigest(),
                'training_controls':summary['physics_steps'],'actual_Adam_steps':max(steps),
                'nominal_SB3_updates':model._n_updates,'normalization_statistics_exact':True})
    if any(len(values)!=5 for values in signatures.values()):raise ValueError('Final trained policy tensors are not distinct within each method')
    return rows


def cached_download(row,cache,target):
    """Reuse only an exact archive with its matching original public-download receipt."""
    cache,target=Path(cache),Path(target)
    receipt=read(cache/'download_receipt.json')
    if receipt.get('manifest_url')!=row['manifest_url']:return None
    name=row['manifest_url'].rsplit('/',1)[1].removesuffix('.parts.json')
    manifest=read(cache/(name+'.parts.json'));parts=validate_manifest(manifest,row['sha256'])
    archive=cache/name
    if (receipt.get('archive_byte_exact') is not True or receipt.get('archive')!=name
            or receipt.get('sha256')!=row['sha256'] or receipt.get('bytes')!=manifest['bytes']
            or receipt.get('parts')!=len(parts) or manifest.get('archive')!=name
            or archive.stat().st_size!=manifest['bytes'] or sha(archive)!=row['sha256']):
        raise ValueError('Cached public archive/receipt fails its original trusted hash')
    target.mkdir(parents=True,exist_ok=False)
    for source in (archive,cache/(name+'.parts.json'),cache/'download_receipt.json'):
        shutil.copy2(source,target/source.name)
    return {**receipt,'transfer_source':'verified_prior_public_download',
            'original_download_receipt_sha256':sha(cache/'download_receipt.json')}


def run(recipe_path,output,public_download_caches=()):
    recipe_path,output=Path(recipe_path).resolve(),Path(output).resolve()
    recipe=read(recipe_path);validate_recipe(recipe)
    if output.exists():raise FileExistsError('Preserve previous public-cohort reconstruction')
    ancestor=output.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<4*1024**3:raise OSError('Four GiBfree space required for checked archives,raw inputs and rebuilds')
    output.mkdir(parents=True);started=time.monotonic();capsules=[]
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for folder in ('navstack','scripts') for p in (ROOT/folder).glob('*.py')}
    (output/'source_manifest.json').write_text(json.dumps(source_hashes,indent=2),encoding='utf-8')
    def verify_source():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=source_commit:
            raise ValueError('Keep the independent source checkout fixed during reconstruction')
        for name,digest in source_hashes.items():
            if sha(ROOT/name)!=digest:raise ValueError('Reconstruction source changed:'+name)
    def get(row):
        target=output/'downloads'/row['slot']
        result=None
        for cache in public_download_caches:
            result=cached_download(row,cache,target)
            if result is not None:break
        if result is None:
            result=download(row['manifest_url'],target,row['sha256'],wall_seconds=1200)
            result['transfer_source']='new_unauthenticated_public_download'
        return row,target/result['archive'],result
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
            downloads=list(workers.map(get,recipe['capsules']))
        inputs=output/'inputs';(inputs/'runs').mkdir(parents=True)
        for row,archive,download_receipt in downloads:
            verify_source();target=output/'capsules'/row['slot'];log=output/(row['slot']+'_reconstruction.log')
            figure=resolve_batch(ROOT,row['reference_figure']) if row.get('reference_figure') else None
            with log.open('x',encoding='utf-8') as stream,redirect_stdout(stream):
                reproduce(archive,row['sha256'],target,figure)
            receipt=read(target/'reproduction.json')
            if [r['method'] for r in receipt['methods']]!=row['methods']:
                raise ValueError('Published complete-method inventory differs from locked recipe')
            source=target/'asset'/row['batch']
            if not source.is_dir():raise ValueError('Wrong archived training batch')
            shutil.copytree(source,inputs/'runs'/row['batch'],ignore=shutil.ignore_patterns('__pycache__'))
            capsules.append({'slot':row['slot'],'archive_sha256':row['sha256'],
                'public_manifest_url':row['manifest_url'],'unauthenticated_download':True,
                'transfer_source':download_receipt['transfer_source'],
                'original_download_receipt_sha256':download_receipt.get('original_download_receipt_sha256'),
                'methods':receipt['methods'],'figure_byte_identical':receipt['figure_byte_identical'],
                'retained_incomplete_sibling_files':receipt.get('incomplete_sibling_files_verified',0)})
        verify_source();pairs_path=resolve_batch(ROOT,recipe['pairs_manifest'])
        with (output/'cohort_analysis.log').open('x',encoding='utf-8') as stream,redirect_stdout(stream):
            actual=analyze_cohort(pairs_path,output/'cohort',inputs_root=inputs)
        if actual!=read(resolve_batch(ROOT,recipe['expected_report'])):
            raise ValueError('Full five-seed report differs from the published reference;inspect retained outputs')
        proofs=model_proof(inputs,read(pairs_path)['pairs'])
        from scripts.plot_native_feedback_cohort import plot
        figure=output/'feedback_five_seed.png'
        with (output/'cohort_figure.log').open('x',encoding='utf-8') as stream,redirect_stdout(stream):
            plot(output/'cohort/cohort.json',figure)
        if sha(figure)!=sha(resolve_batch(ROOT,recipe['expected_figure'])):
            raise ValueError('Public-input five-seed figure differs from the versioned reference')
        verify_source()
        result={'scope':__doc__,'recipe_sha256':sha(recipe_path),'source_commit':source_commit,
            'public_capsules':capsules,'all_reference_cohort_fields_identical':True,
            'independent_training_seeds':5,'unique_known_development_conditions':12,
            'H_successes':actual['H_successes'],'H0_successes':actual['H0_successes'],
            'episodes_each_method':actual['episodes_per_method'],'statistics':actual['statistics'],
            'model_and_optimizer_proof':proofs,'five_distinct_final_policy_tensors_each_method':True,
            'figure_byte_identical':True,'figure_sha256':sha(figure),'wall_seconds':time.monotonic()-started,
            'new_controls':0,'PPO_updates':0,'supervised_updates':0,'research_acceptance':'UNMET'}
        (output/'reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('H_successes','H0_successes','statistics','figure_byte_identical','wall_seconds')}),flush=True)
        return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'new_controls':0,'source_commit':source_commit},indent=2),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',default=str(ROOT/'protocols/public_native_feedback_cohort_20261031.json'))
    parser.add_argument('--output',required=True)
    parser.add_argument('--public-download-cache',action='append',default=[],
                        help='Optional existing public-download directory;exact archive,manifest and original receipt are rechecked')
    args=parser.parse_args();run(args.recipe,args.output,args.public_download_cache)
