"""Resume only the two untouched registered learners in two disjoint CARLA slots.

Native physics, initial weights, data, budgets and original failure receipts are
preserved. This coordinator never launches final confirmation or deferred pilots.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navstack.process_lease import ProcessLease
from navstack.research import atomic_json
from scripts.run_independent_outcome_cohort import completed_phase, resource_admission, runtime_paths
from scripts.audit_native_method_endpoint import read, sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata

SLOTS={44:2110,55:2120}


def labels(seed):
    return [f'seed{seed}_baseline',*[f'seed{seed}_{m}_{phase}' for m in ('H','H0') for phase in ('training','endpoint')]]


def retire_barrier(archive, output):
    archive,output=Path(archive).resolve(),Path(output).resolve()
    source=archive/'20261104_independent_outcome_seed44_zero_PPO_known24'
    target=output/'retired_three_hour_admission_barrier'
    if not source.exists():return None
    if output.parent!=archive or source.resolve().parent!=archive or target.parent!=output or target.exists():
        raise ValueError('Barrier paths must remain in the declared archive/output')
    if sorted(p.name for p in source.iterdir())!=['admission_blocked_by_user_deadline.json']:
        raise ValueError('Not an empty,never-started admission barrier;preserve native evidence')
    note=read(source/'admission_blocked_by_user_deadline.json')
    if note.get('status')!='NOT_STARTED_ADMISSION_BARRIER' or note.get('native_controls')!=0 or note.get('native_jobs')!=0:
        raise ValueError('Barrier is not explicitly zero-native')
    original=sha(source/'admission_blocked_by_user_deadline.json')
    source.rename(target)
    return dict(original_batch=source.name,retained_at=target.name,original_sha256=original,native_jobs=0)


def check_idle_ports():
    held=[]
    try:
        for base in SLOTS.values():
            for port in range(base,base+3):
                s=socket.socket();held.append(s)
                if os.name=='nt':s.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
                s.bind(('127.0.0.1',port))
    finally:
        for s in held:s.close()


def preflight(cohort,archive,output):
    cohort,archive,output=map(lambda p:Path(p).resolve(),(cohort,archive,output))
    if output.exists():raise FileExistsError('Preserve previous continuation attempts')
    if output.parent!=archive:raise ValueError('Use a fresh direct archive child')
    if (archive/'STOP').exists():raise ValueError('Archive STOP requires its recorded owner to be resolved')
    plan=read(cohort/'registration.json');resources=resource_admission(archive);check_idle_ports()
    python,carla_python=runtime_paths(plan,ROOT/'.venv/Scripts/python.exe',ROOT/'.carla-python37/runtime/python.exe')
    for name,digest in plan['source_hashes'].items():
        if sha(cohort/'source'/name)!=digest:raise ValueError('Frozen native source differs:'+name)
    original={}
    for seed in (11,22,33):
        for label in labels(seed):
            if not completed_phase(cohort,archive,label,True):raise ValueError('Earlier seed is not closed:'+label)
            original[label]=sha(cohort/'execution'/(label+'_complete.json'))
    initializers=[]
    import io,torch,zipfile
    for seed in SLOTS:
        row=next(x for x in plan['initializer_results'] if x['seed']==seed);states=[]
        for method in ('H','H0'):
            p=cohort/row['arms'][method]['path'];metadata=verified_checkpoint_metadata(p)
            for key in ('model_sha256','normalization_sha256','resume_state_sha256'):
                if metadata[key]!=row['arms'][method][key]:raise ValueError('Registered initializer changed')
            with zipfile.ZipFile(p/'model.zip') as z:
                states.append(torch.load(io.BytesIO(z.read('policy.pth')),map_location='cpu',weights_only=True))
            if metadata['physics_steps'] or metadata['optimizer_steps']:raise ValueError('Require pristine registered new-course starts')
        if states[0].keys()!=states[1].keys() or not all(torch.equal(states[0][k],states[1][k]) for k in states[0]):
            raise ValueError('H/H0paired initial weights differ')
        if any(row['arms']['H'][k]!=row['arms']['H0'][k] for k in ('normalization_sha256','resume_state_sha256','optimizer_sha256')):
            raise ValueError('H/H0paired initial Adam,RNG or normalization differ')
        initializers.append(dict(seed=seed,port=SLOTS[seed],paired_weights_Adam_RNG_RMS_equal=True))
    output.mkdir()
    for folder in ('navstack','scripts'):
        shutil.copytree(ROOT/folder,output/'operational_source'/folder,ignore=shutil.ignore_patterns('__pycache__'))
    code={p.relative_to(output/'operational_source').as_posix():sha(p) for p in (output/'operational_source').rglob('*') if p.is_file()}
    registration=dict(version='remaining_two_independent_outcome_pairs_v1',cohort=str(cohort),archive=str(archive),
        user_authorization='User explicitly allows several hours to complete seed44/55under original budgets',
        seeds=list(SLOTS),ports=SLOTS,new_controls_per_arm=30720,max_rollout_overshoot=319,
        native_jobs=10,training_jobs=4,evaluation_jobs=6,automatic_retries=0,per_native_job_wall_seconds=14400,
        maximum_simultaneous_servers=2,resources=resources,original_registration_sha256=sha(cohort/'registration.json'),
        previous_completed_markers=original,initializers=initializers,operational_source_hashes=code,
        repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),created_epoch=time.time())
    registration.update(runtime_python=python,runtime_carla_python=carla_python,
        runtime_override_note='Explicit executables from existing workspace;original registration strings and SHA retained unchanged')
    atomic_json(output/'registration.json',registration)
    print(json.dumps(dict(status='PREFLIGHT_COMPLETE_NOT_STARTED',completed_phases_reused=len(original),remaining_seeds=list(SLOTS))),flush=True)


def run(output):
    output=Path(output).resolve();p=read(output/'registration.json');cohort,archive=Path(p['cohort']),Path(p['archive'])
    if (output/'started.json').exists():raise FileExistsError('Do not duplicate an existing continuation')
    if sha(cohort/'registration.json')!=p['original_registration_sha256']:raise ValueError('Original cohort changed')
    for name,digest in p['operational_source_hashes'].items():
        if sha(output/'operational_source'/name)!=digest:raise ValueError('Operational snapshot changed')
    with ProcessLease(cohort/'supplement_remaining_pairs.lock'):
        resource_admission(archive);check_idle_ports()
        atomic_json(output/'retired_barrier.json',retire_barrier(archive,output))
        atomic_json(output/'started.json',dict(pid=os.getpid(),epoch=time.time(),seeds=p['seeds']))
        errors=[]
        def one(seed):
            execution=f'execution_supplement_seed{seed}_20261105'
            command=[sys.executable,str(output/'operational_source/scripts/run_independent_outcome_cohort.py'),'run',
                '--cohort',str(cohort),'--archive',str(archive),'--resume-completed','--execution-name',execution,
                '--seeds',str(seed),'--rpc-port',str(SLOTS[seed]),
                '--python',p['runtime_python'],'--carla-python',p['runtime_carla_python']]
            with (output/f'seed{seed}_driver.log').open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                atomic_json(output/f'seed{seed}_driver_process.json',dict(pid=child.pid,epoch=time.time(),execution=execution))
                code=child.wait()
            if code:raise RuntimeError(f'seed{seed}driver exited{code};preserve failed attempt,no automatic retry')
            return dict(seed=seed,execution=execution,exit_code=0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(one,s):s for s in p['seeds']}
            for future in as_completed(futures):
                try:atomic_json(output/f'seed{futures[future]}_complete.json',future.result())
                except BaseException as error:
                    errors.append(dict(seed=futures[future],error=repr(error),epoch=time.time()))
                    atomic_json(output/'failures.json',errors)
        for label,digest in p['previous_completed_markers'].items():
            if sha(cohort/'execution'/(label+'_complete.json'))!=digest:raise ValueError('A prior completed receipt changed')
        if errors:raise RuntimeError('Supplement contains failures;no complete cohort claim')
        for seed in (11,22,33,44,55):
            for label in labels(seed):
                if not completed_phase(cohort,archive,label,True):raise ValueError('Full five-seed cohort is not closed')
        complete=dict(seeds=[11,22,33,44,55],status='ALL_REGISTERED_JOBS_CLOSED',research_acceptance='UNMET')
        atomic_json(cohort/'execution/complete.json',complete)
        atomic_json(output/'complete.json',dict(**complete,epoch=time.time(),previous_completed_receipts_unchanged=True))
        print('Both remaining pairs and all25registered phases closed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('preflight','run'))
    p.add_argument('--cohort');p.add_argument('--archive');p.add_argument('--output',required=True);a=p.parse_args()
    if a.phase=='preflight':
        if not a.cohort or not a.archive:p.error('preflight requires cohort and archive')
        preflight(a.cohort,a.archive,a.output)
    else:run(a.output)
