"""Read-only finite all-five feedback analysis after exact owned training exits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(execution):
    execution=Path(execution).resolve();root=execution.parent.parent;source=execution/'source'
    sys.path.insert(0,str(source))
    from navstack.owned_process import OwnedProcess
    from navstack.process_lease import ProcessLease
    from navstack.research import atomic_json
    from scripts.audit_native_method_endpoint import read,sha
    from scripts.run_native_pristine_baselines import require_training_completion
    from scripts.evaluate_native_rule import require_finished_parent
    plan=read(execution/'plan.json');hashes=read(execution/'source_manifest.json');owner=None
    def verify():
        for name,digest in hashes.items():
            if sha(source/name)!=digest:raise ValueError('Frozen analysis implementation changed')
        if sha(execution/'pairs.json')!=plan['pairs_sha256']:raise ValueError('Registered pair list changed')
    def status(phase):atomic_json(execution/'status.json',{'phase':phase,'pid':os.getpid(),'epoch':time.time()})
    def call(command,label):
        with (execution/(label+'.log')).open('x',encoding='utf-8') as stream:
            subprocess.run(list(map(str,command)),cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1200)
    try:
        verify()
        with ProcessLease(execution/'pipeline.lock'):
            with (execution/'started.json').open('x',encoding='utf-8') as stream:json.dump({'pid':os.getpid(),'epoch':time.time()},stream)
            identity=plan['dependency_process']
            owner=OwnedProcess(identity['pid'],identity['creation_epoch'],identity['executable'],allow_terminate=False)
            until=time.monotonic()+43200
            while owner.poll() is None:
                if time.monotonic()>until:raise TimeoutError('Finite owned-cohort wait exhausted')
                status('WAITING_FOR_ALL_FIVE_EXACT_OWNED_TRAINED_FEEDBACK_PAIRS');time.sleep(10)
            dependency=root/'runs'/plan['dependency_execution'];prior=read(dependency/'status.json')
            require_training_completion(prior,(dependency/'failure.json').exists())
            for row in prior['completed']:require_finished_parent(root/'runs'/row['batch'])
            verify();status('REBUILDING_ALL_FIVE_BROADER_FEEDBACK_STATISTICS')
            output=root/plan['analysis_output'];figure=root/plan['figure_output']
            call([sys.executable,source/'scripts/analyze_native_feedback_cohort.py','--pairs-manifest',execution/'pairs.json',
                '--inputs-root',root,'--output',output],'all_five_analysis')
            call([sys.executable,source/'scripts/plot_native_feedback_cohort.py','--report',output/'cohort.json','--output',figure],'all_five_figure')
            result=read(output/'cohort.json')
            if result['unique_condition_count']!=24 or result['episodes_per_method']!=120:raise ValueError('Incomplete broader inventory')
            with (root/'reports'/plan['report_output']).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
            atomic_json(execution/'complete.json',{'report_sha256':sha(output/'cohort.json'),'figure_sha256':sha(figure),
                'seeds':result['seeds'],'unique_known_development_conditions':24,'new_controls':0,'new_optimizer_updates':0,'research_acceptance':'UNMET'})
            status('ALL_FIVE_BROADER_FEEDBACK_STATISTICS_AND_FIGURE_COMPLETE')
    except BaseException as error:
        atomic_json(execution/'failure.json',{'error':repr(error),'epoch':time.time()});raise
    finally:
        if owner is not None:owner.kernel.CloseHandle(owner.handle)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execution',required=True)
    run(parser.parse_args().execution)
