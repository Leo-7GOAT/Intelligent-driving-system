import gzip
import json
import pytest
from scripts.run_native_control_cadence_cohort import cadence_trace_counts


def write(path,rows):
    with gzip.open(path,'wt',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')


def tick(recomputed,status='solved',hold=2):
    return {'controller_status':status,'diagnostics':{'controller_recomputed':recomputed,'controller_hold_steps':hold,
        'control_seconds':.02 if recomputed else 0.,'qp_seconds':.01 if recomputed else 0.}}


def test_native_trace_distinguishes_actual_MPCcomputations_holds_and_safety_frames(tmp_path):
    path=tmp_path/'trace.gz';write(path,[{'info':{'mode':'reference','executed_substeps':2,'trace':[tick(True),tick(False)]}},
        {'info':{'mode':'reference','executed_substeps':2,'trace':[tick(True,'failed'),tick(False,'failed')]}},
        {'info':{'mode':'dynamic_brake','executed_substeps':1,'trace':[{}]}}])
    actual=cadence_trace_counts(path,'H',2)
    assert actual=={'physics_controls':5,'policy_decisions':3,'MPCrecomputations':2,'held_MPCframes':2,
        'MPCfailed_recomputations':1,'held_failed_MPCframes':1,'native_nonreference_frames':1,'MPCrecomputation_seconds':[.02,.02]}


def test_direct_policy_decisions_do_not_become_MPCsolves(tmp_path):
    path=tmp_path/'trace.gz';write(path,[{'info':{'mode':'flat','executed_substeps':2,'trace':[{},{}]}},
        {'info':{'mode':'flat','executed_substeps':1,'trace':[{}]}}])
    actual=cadence_trace_counts(path,'B1',2)
    assert actual['physics_controls']==3 and actual['policy_decisions']==2 and actual['MPCrecomputations']==0


@pytest.mark.parametrize('info',[
    {'mode':'reference','executed_substeps':2,'trace':[tick(True)]},
    {'mode':'reference','executed_substeps':1,'trace':[tick(True,hold=5)]},
    {'mode':'reference','executed_substeps':1,'trace':[tick('true')]},
])
def test_incomplete_or_unregistered_cadence_evidence_is_not_scored(info,tmp_path):
    path=tmp_path/'trace.gz';write(path,[{'info':info}])
    with pytest.raises(ValueError):cadence_trace_counts(path,'H',2)
