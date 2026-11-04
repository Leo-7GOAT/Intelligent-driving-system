"""Preserve completed cases and finish a stopped development queue with exact shortcuts."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.reference_compute import VERSION
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.prepare_native_recovery import verified_checkpoint_metadata,claim_recovery
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone


def stage(template,output,source_model,ids=None,prior_evaluation_controls=0):
    p=deepcopy(read(template/'protocol.json'))
    if output.exists():raise FileExistsError('Preserve every frozen recovery child')
    if ids is not None:
        original=p['native_case_manifest'];used=sorted({original['cases'][i]['route_index'] for i in ids})
        mapping={old:new for new,old in enumerate(used)}
        p['routes']=[deepcopy(p['routes'][i]) for i in used]
        p['native_case_manifest']={**deepcopy(original),'cases':[dict(deepcopy(original['cases'][i]),
            route_index=mapping[original['cases'][i]['route_index']]) for i in ids]}
    p.pop('freeze_observation_statistics',None);p.pop('fixed_lowlevel_training',None)
    p['reference_compute_profile']={'version':VERSION}
    p['exact_compute_recovery']=dict(original_case_ids=ids,source_template=template.name,
        source_protocol_sha256=sha(template/'protocol.json'),new_PPO_updates=0,new_final_conditions=0,
        scope='Same model/tasks;exact spatial search memo and no-progress refinement shortcuts only. No rescoring of retained completed cases.')
    output.mkdir();(output/'infrastructure').mkdir();shutil.copytree(template/'calibration',output/'calibration')
    shutil.copytree(source_model,output/'resume/H');meta=verified_checkpoint_metadata(output/'resume/H')
    if sha(source_model/'model.zip')!=sha(template/'resume/H/model.zip') or sha(source_model/'normalization.pkl')!=sha(template/'resume/H/normalization.pkl'):
        raise ValueError('Recovery must retain exact registered model and normalization bytes')
    p['recovery']['H'].update(checkpoint='resume/H',model_sha256=meta['model_sha256'],normalization_sha256=meta['normalization_sha256'])
    p['recovery']['H']['consumed_controls_lower_bound']+=prior_evaluation_controls
    p['exact_compute_recovery'].update(prior_evaluation_controls_charged=prior_evaluation_controls,
        accounting='Sibling evaluation costs are enumerated separately;branch lineage counters are lower bounds,not additive project totals')
    for directory in ['navstack','scripts']:
        for f in (ROOT/directory).rglob('*'):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.ps1'):
                target=output/'source'/f.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
    p['source_hashes']={f.relative_to(output/'source').as_posix():sha(f) for f in (output/'source').rglob('*') if f.is_file()}
    now=time.time();p.update(created_epoch=now,deadline_epoch=now+p['resource_limits']['wall_seconds'])
    atomic_json(output/'protocol.json',p);atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))


def run(parent,known,identity,proof,execution):
    parent,known,execution=map(lambda p:Path(p).resolve(),[parent,known,execution]);identity=read(Path(identity));proof=read(Path(proof))
    for name,digest in proof['compute_source_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Compute equivalence source changed:'+name)
    if (proof['protocol_sha256']!=sha(parent/'protocol.json') or proof['model_sha256']!=sha(parent/'H/model.zip')
            or proof['actual_control_frames_bound']!=900 or len(proof['results'])!=90
            or any(not all(r[k] for k in ['requested_command_exact','constraint_cells_exact','all_new_reference_poses_match_actual_queries']) for r in proof['results'])):
        raise ValueError('Require the complete recorded equivalence proof')
    execution.mkdir();atomic_json(execution/'registration.json',dict(parent=parent.name,known_template=known.name,identity=identity,
        compute_profile=VERSION,new_training=0,new_final_conditions=0,automatic_retries=0))
    def status(label):atomic_json(execution/'status.json',dict(phase=label,pid=os.getpid(),epoch=time.time()))
    def call(args,label,timeout=2400,env=None):
        status(label)
        with (execution/(label+'.log')).open('xb') as stream:
            subprocess.run(list(map(str,args)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout,env=env)
    audits=[]
    try:
        status('waiting_existing_coordinator')
        try:owner=OwnedProcess(**identity,allow_terminate=False)
        except (OSError,RuntimeError) as error:
            if isinstance(error,RuntimeError) and str(error)!='Owned process handle unavailable':raise
            final=read(parent/'guard_final.json')
            if not recorded_lifecycle_gone([identity['pid']],final['epoch']):raise
            owner=None
        if owner is not None:
            try:
                while owner.poll() is None:
                    if time.time()>read(parent/'protocol.json')['deadline_epoch']+2400:raise TimeoutError('Original coordinator wait expired;do not restart it')
                    time.sleep(5)
            finally:owner.kernel.CloseHandle(owner.handle)
        env=dict(os.environ,NAVSTACK_AUDIT_SOURCE_ROOT=str(parent/'source'),PYTHONIOENCODING='utf-8')
        if (parent/'H/evaluation.json').exists() and read(parent/'guard_final.json')['exit_code']==0:
            require_finished_parent(parent)
            if len(read(parent/'H/evaluation.json')['episodes'])!=30:raise ValueError('Unexpected completed inventory')
            completed=list(range(30));remaining=[];prefix=None
            prior_controls=read(parent/'H/endpoint_raw_audit.json')['evaluation_control_frames_verified']
        else:
            prefix=parent/'H/exact_compute_recovery_prefix_raw_audit.json'
            call([sys.executable,ROOT/'scripts/audit_native_partial_evaluation.py','--batch',parent,'--method','H',
                '--development-stop','--output',prefix],'prefix_raw',env=env)
            receipt=read(prefix);completed=receipt['completed_case_ids'];remaining=receipt['remaining_case_ids']
            prior_controls=receipt['all_recorded_actual_controls_verified']
            for flag,name in [([], 'compute_prefix_start_inference.json'),(['--all-static-actions'],'compute_prefix_static_inference.json')]:
                log=(execution/(name+'.log')).open('xb')
                child=subprocess.Popen(list(map(str,[sys.executable,ROOT/'scripts/audit_native_reference_inference.py',
                    '--batch',parent,'--method','H','--prefix-audit',prefix,*flag,'--output',parent/'H'/name])),
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
                audits.append((child,log,name))
            atomic_json(execution/'prefix_audit_processes.json',dict(processes=[dict(pid=c.pid,name=n) for c,_,n in audits]))
        atomic_json(execution/'retained_prefix.json',dict(completed_case_ids=completed,remaining_case_ids=remaining,
            original_failed_or_partial_cases_retained=True,prefix_raw_sha256=None if prefix is None else sha(prefix)))
        source_model=parent/'resume/H';batches=[];counts=[]
        if remaining:
            target=parent.parent/(parent.name+'_exact_compute_tail');stage(parent,target,source_model,remaining,prior_controls)
            batches.append(target);counts.append(len(remaining))
        target=known.parent/(known.name+'_exact_compute');stage(known,target,source_model,prior_evaluation_controls=prior_controls)
        batches.append(target);counts.append(24)
        claim_recovery(parent,execution.name)
        atomic_json(execution/'children_prepared.json',dict(batches=list(map(str,batches)),counts=counts))
        call([sys.executable,ROOT/'scripts/run_registered_validation_queue.py','--batches',*batches,'--counts',*counts,
            '--execution',execution/'tail_and_known_queue'],'tail_and_known',sum(read(b/'protocol.json')['resource_limits']['wall_seconds']+600 for b in batches)+7200)
        for child,log,name in audits:
            try:
                if child.wait(timeout=2400)!=0:raise ValueError('Completed-prefix inference failed:'+name)
            finally:log.close()
        atomic_json(execution/'complete.json',dict(retained_completed_case_ids=completed,
            children=read(execution/'tail_and_known_queue/complete.json'),new_training=0,new_final_conditions=0,
            research_acceptance='UNMET',limits='Known development recovery using two disclosed exact-compute implementations;not an isolated learning comparison or independent final confirmation. All discarded partial controls remain charged.'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','known','identity','proof','execution']:p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.parent,a.known,a.identity,a.proof,a.execution)
