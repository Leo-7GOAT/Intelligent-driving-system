"""Queue one H-only scale intervention after the exact audited endpoint lifecycle."""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.owned_process import OwnedProcess
from navstack.research import atomic_json
from scripts.audit_native_method_endpoint import read,sha
from scripts.evaluate_native_rule import require_finished_parent
from scripts.prepare_native_reference_learning import prepare
from scripts.run_native_reference_course import run as run_course


def source_hashes():
    return {p.relative_to(ROOT).as_posix():sha(p) for directory in ('navstack','scripts')
            for p in sorted((ROOT/directory).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.ps1')}


def completed_predecessor(predecessor,baseline):
    complete=read(Path(predecessor)/'complete.json')
    for phase,count in (('static',12),('known_validation',24)):
        measured=complete['results'][phase];batch=Path(measured['batch'])
        require_finished_parent(batch)
        if not read(batch/'strict_delivery.json')['strict_delivery_gate_passed']:
            raise ValueError('Predecessor engine-delivery verification failed')
        for method in ('H','H0'):
            result=measured['results'][method];model_sha=sha(Path(baseline)/method/'model.zip')
            if (result['cases']!=count or result['model_sha256']!=model_sha
                    or result['raw_audit_sha256']!=sha(batch/method/'endpoint_raw_audit.json')):
                raise ValueError('Predecessor inventory/model/raw binding changed')
            for name in ('measured_start_inference.json','static_action_inference.json'):
                inference=read(batch/method/name)
                if not inference['all_recorded_actions_match'] or inference['model_sha256']!=model_sha:
                    raise ValueError('Predecessor NN reconstruction failed')
    return complete


def run(initializers,baseline,predecessor,identity_file,output,execution):
    initializers,baseline,predecessor,output,execution=map(lambda p:Path(p).resolve(),
        (initializers,baseline,predecessor,output,execution))
    proof=read(initializers/'complete.json')
    if (set(proof['results'])!={'H'} or proof.get('gain_scale')!=100.
            or proof['status']!='INITIALIZERS_PREPARED_NOT_NATIVE_TRAINED'):
        raise ValueError('Require the prepared H-only scale100 initializer')
    if output.exists():raise FileExistsError('Preserve every registered native course')
    execution.mkdir()
    identity=read(identity_file);owner=OwnedProcess(**identity,allow_terminate=False)
    sources=source_hashes();initializer_sha=sha(initializers/'complete.json')
    initializer_files={p.relative_to(initializers).as_posix():sha(p) for p in sorted(initializers.rglob('*')) if p.is_file()}
    registration=dict(dependency=identity,initializer_proof_sha256=initializer_sha,
        source_hashes=sources,initializer_files=initializer_files,new_training_methods=['H'],requested_controls=10240,
        gain_scale=100.,checkpoint_every_controls=320,new_auxiliary_updates=0,new_final_conditions=0,
        automatic_retries=0,selection='Final fixed requested-budget endpoint;full12static and24known-validation conditions',
        motivation='Completed scale1 H static3/12 plus measured reusable-reference logit deficit;rescale the trainable term without forcing actions. Existing H0 training is not repeated.',
        limits='One seed development intervention;all prior infrastructure losses and tuning costs remain charged. No guaranteed improvement or matched total-development-cost claim.')
    atomic_json(execution/'registration.json',registration)
    try:
        atomic_json(execution/'status.json',dict(phase='WAITING_FOR_EXACT_REFERENCE_ENDPOINT_COORDINATOR',pid=os.getpid(),epoch=time.time()))
        deadline=time.monotonic()+28800
        while owner.poll() is None:
            if time.monotonic()>deadline:raise TimeoutError('Finite predecessor wait exhausted')
            time.sleep(10)
        if owner.poll()!=0:raise ValueError('Predecessor failed;no new course or automatic retry')
        previous=completed_predecessor(predecessor,baseline)
        atomic_json(execution/'predecessor_complete.json',previous)
        current_initializers={p.relative_to(initializers).as_posix():sha(p) for p in sorted(initializers.rglob('*')) if p.is_file()}
        if source_hashes()!=sources or current_initializers!=initializer_files or sha(initializers/'complete.json')!=initializer_sha:
            raise ValueError('Prospectively registered source or initializer changed while waiting')
        prepare(output,initializers)
        p=read(output/'protocol.json')
        if p['methods']!=['H'] or p['method_budgets']!={'H':10240} or p['checkpoint_thresholds']!=list(range(320,10241,320)):
            raise ValueError('Selected fixed-budget course differs from registration')
        p['reference_scale_followup_registration']=dict(registration_sha256=sha(execution/'registration.json'),
            predecessor_complete_sha256=sha(predecessor/'complete.json'),
            baseline_H_static_successes=previous['results']['static']['results']['H']['successes'])
        atomic_json(output/'protocol.json',p)
        atomic_json(output/'prepared.json',dict(status='PREPARED_NOT_STARTED',protocol_sha256=sha(output/'protocol.json')))
        atomic_json(execution/'status.json',dict(phase='SELECTED_H_COURSE',pid=os.getpid(),epoch=time.time(),batch=str(output)))
        run_course(output,execution/'course')
        atomic_json(execution/'complete.json',dict(batch=str(output),course=read(execution/'course/complete.json'),research_acceptance='UNMET'))
    except BaseException as error:
        atomic_json(execution/'failure.json',dict(error=repr(error),epoch=time.time()));raise
    finally:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('initializers','baseline','predecessor','identity-file','output','execution'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run(a.initializers,a.baseline,a.predecessor,a.identity_file,a.output,a.execution)
