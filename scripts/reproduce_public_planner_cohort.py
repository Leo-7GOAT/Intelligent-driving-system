"""Reconstruct every H/B2DEVELOPMENT cohort field and figure from ten public capsules."""
import argparse
import concurrent.futures
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.reproduce_public_feedback_cohort import read,sha,cached_download
from scripts.download_native_asset import download,release_url
from scripts.reproduce_native_asset import reproduce
from scripts.analyze_native_feedback_cohort import resolve_batch
from scripts.analyze_native_planner_cohort import run as analyze


def validate_recipe(recipe):
    if recipe['seeds']!=[11,22,33,44,55] or len(recipe['capsules'])!=10:raise ValueError('All five H/B2pairs required')
    rows=recipe['capsules']
    if len({r['slot'] for r in rows})!=10 or len({r['batch'] for r in rows})!=10:raise ValueError('Repeated public capsule')
    for method in ('H','B2'):
        if sorted(r['seed'] for r in rows if r['analysis_method']==method)!=recipe['seeds']:
            raise ValueError('No omitted/duplicated independently trained seed')
    import re
    for r in rows:
        release_url(r['manifest_url'])
        if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',r[k]) or r[k] in ('.','..') for k in ('slot','batch')):
            raise ValueError('Safe direct-child identifiers required')
        if not re.fullmatch('[0-9a-f]{64}',r['sha256']) or r['analysis_method'] not in r['methods']:
            raise ValueError('Trusted archive hash and complete method inventory required')
        if r.get('reference_figure'):resolve_batch(ROOT,r['reference_figure'])
    for k in ('pairs_manifest','execution_state','expected_report','expected_figure'):resolve_batch(ROOT,recipe[k])


def model_proof(inputs,rows):
    import numpy as np
    import pickle
    import torch
    from navstack.independent_clip_ppo import load_ppo
    torch.set_num_threads(1);results=[];signatures={'H':set(),'B2':set()}
    for row in rows:
        method=row['analysis_method'];batch=inputs/'runs'/row['batch'];folder=batch/method
        protocol=read(batch/'protocol.json');summary=read(folder/'summary.json');model=load_ppo(folder/'model.zip',device='cpu')
        steps={int(v['step']) for v in model.policy.optimizer.state.values() if 'step' in v}
        if steps!={summary['optimizer_steps']} or model.num_timesteps!=summary['actions'] or summary['optimizer_steps']<=0:
            raise ValueError('Actual saved Adam/action state differs from raw-audited training')
        origin=batch/protocol['recovery'][method]['checkpoint']
        with (origin/'normalization.pkl').open('rb') as f:before=pickle.load(f)
        with (folder/'normalization.pkl').open('rb') as f:after=pickle.load(f)
        for name in ('mean','var'):np.testing.assert_array_equal(getattr(before.obs_rms,name),getattr(after.obs_rms,name))
        if before.obs_rms.count!=after.obs_rms.count:raise ValueError('Observation statistics changed')
        h=hashlib.sha256()
        for name,value in sorted(model.policy.state_dict().items()):
            a=value.detach().cpu().numpy();h.update(name.encode()+str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        signatures[method].add(h.hexdigest())
        results.append({'seed':row['seed'],'method':method,'model_sha256':sha(folder/'model.zip'),
            'policy_tensor_sha256':h.hexdigest(),'actual_Adam_steps':max(steps),'normalization_statistics_exact':True})
    if any(len(s)!=5 for s in signatures.values()):raise ValueError('Five distinct trained policies each method required')
    return results


def run(recipe_path,output,caches=()):
    recipe_path,output=Path(recipe_path).resolve(),Path(output).resolve();recipe=read(recipe_path);validate_recipe(recipe)
    if output.exists():raise FileExistsError('Preserve earlier reconstruction')
    ancestor=output.parent
    while not ancestor.exists():ancestor=ancestor.parent
    if shutil.disk_usage(ancestor).free<6*1024**3:raise OSError('Six GiBfree space required')
    output.mkdir(parents=True);started=time.monotonic();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={p.relative_to(ROOT).as_posix():sha(p) for folder in ('scripts','navstack') for p in (ROOT/folder).glob('*.py')}
    (output/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    def fixed():
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=commit:raise ValueError('Source checkout advanced during reconstruction')
        if any(sha(ROOT/k)!=v for k,v in sources.items()):raise ValueError('Reconstruction code changed')
    def get(row):
        target=output/'downloads'/row['slot'];receipt=None
        for cache in caches:
            receipt=cached_download(row,cache,target)
            if receipt is not None:break
        if receipt is None:
            receipt=download(row['manifest_url'],target,row['sha256']);receipt['transfer_source']='new_unauthenticated_public_download'
        return row,target/receipt['archive'],receipt
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:downloads=list(pool.map(get,recipe['capsules']))
        inputs=output/'inputs';(inputs/'runs').mkdir(parents=True);capsules=[]
        for row,archive,receipt in downloads:
            fixed();target=output/'capsules'/row['slot'];figure=resolve_batch(ROOT,row['reference_figure']) if row.get('reference_figure') else None
            with (output/(row['slot']+'_reconstruction.log')).open('x',encoding='utf-8') as log,redirect_stdout(log):
                reproduce(archive,row['sha256'],target,figure)
            verification=read(target/'reproduction.json')
            if [r['method'] for r in verification['methods']]!=row['methods']:raise ValueError('Public methods differ')
            shutil.copytree(target/'asset'/row['batch'],inputs/'runs'/row['batch'])
            capsules.append({'slot':row['slot'],'public_manifest_url':row['manifest_url'],'archive_sha256':row['sha256'],
                'transfer_source':receipt['transfer_source'],'original_download_receipt_sha256':receipt.get('original_download_receipt_sha256'),
                'verified_controls':sum(r['verified_controls'] for r in verification['methods']),'figure_byte_identical':True})
        state=resolve_batch(ROOT,recipe['execution_state']);execution=output/'registered_execution';execution.mkdir();shutil.copy2(state,execution/'status.json')
        with (output/'cohort_analysis.log').open('x',encoding='utf-8') as log,redirect_stdout(log):
            actual=analyze(resolve_batch(ROOT,recipe['pairs_manifest']),execution,output/'cohort.json',inputs_root=inputs)
        if actual!=read(resolve_batch(ROOT,recipe['expected_report'])):raise ValueError('Full paired report differs from original published reference')
        proofs=model_proof(inputs,recipe['capsules'])
        from scripts.plot_native_planner_cohort import plot
        with (output/'cohort_figure.log').open('x',encoding='utf-8') as log,redirect_stdout(log):plot(output/'cohort.json',output/'planner_cohort.png')
        if sha(output/'planner_cohort.png')!=sha(resolve_batch(ROOT,recipe['expected_figure'])):raise ValueError('Main planner figure differs')
        fixed();result={'scope':__doc__,'source_commit':commit,'recipe_sha256':sha(recipe_path),'public_capsules':capsules,
            'all_reference_cohort_fields_identical':True,'model_optimizer_and_normalizer_proofs':proofs,
            'five_distinct_policy_tensors_each_method':True,'successes':actual['successes'],'statistics':actual['statistics'],
            'figure_byte_identical':True,'figure_sha256':sha(output/'planner_cohort.png'),'wall_seconds':time.monotonic()-started,
            'new_controls':0,'PPO_updates':0,'research_acceptance':'UNMET'}
        (output/'reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'successes':result['successes'],'figure_byte_identical':True,'wall_seconds':result['wall_seconds']}),flush=True)
        return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'source_commit':commit,'new_controls':0},indent=2),encoding='utf-8');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recipe',default=str(ROOT/'protocols/public_native_planner_cohort_20261101.json'))
    p.add_argument('--output',required=True);p.add_argument('--public-download-cache',action='append',default=[])
    a=p.parse_args();run(a.recipe,a.output,a.public_download_cache)
