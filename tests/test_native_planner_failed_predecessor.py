import json
from pathlib import Path
import pytest
from scripts.run_native_planner_cohort import closed_failure_dependency, sha
from scripts import prepare_native_evaluation_recovery as recovery


def fixture(tmp_path, monkeypatch):
    dependency=tmp_path/'runs/failed_B1';dependency.mkdir(parents=True)
    failure=dependency/'failure.json';failure.write_text(json.dumps({'created':[],'error':'basic motion failed'}))
    batch=tmp_path/'runs/closed_gate';(batch/'infrastructure').mkdir(parents=True)
    (batch/'summary.json').write_text(json.dumps({'status':'complete'}))
    (batch/'guard_final.json').write_text(json.dumps({'exit_code':0,'reason':None}))
    (batch/'infrastructure/server_stop.json').write_text(json.dumps({'status':'owned_empty_server_stopped'}))
    monkeypatch.setattr(recovery,'recorded_lifecycle_gone',lambda *args:True)
    return {'dependency_execution':'failed_B1','closed_failed_predecessor':{
        'failure_sha256':sha(failure),'closed_native_batches':['closed_gate'],
        'process_pids':[11,22,33],'closed_epoch':100}}


def test_default_dependency_still_requires_success(tmp_path):
    assert closed_failure_dependency(tmp_path,{}) is False


def test_closed_failure_releases_only_independent_B2(tmp_path,monkeypatch):
    assert closed_failure_dependency(tmp_path,fixture(tmp_path,monkeypatch)) is True


def test_live_predecessor_cannot_release_simulator(tmp_path,monkeypatch):
    plan=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(recovery,'recorded_lifecycle_gone',lambda *args:False)
    with pytest.raises(ValueError,match='still live'):
        closed_failure_dependency(tmp_path,plan)


def test_failed_cleanup_cannot_release_simulator(tmp_path,monkeypatch):
    plan=fixture(tmp_path,monkeypatch)
    (tmp_path/'runs/closed_gate/guard_final.json').write_text(json.dumps({'exit_code':1,'reason':'failed'}))
    with pytest.raises(ValueError,match='normally'):
        closed_failure_dependency(tmp_path,plan)


def test_retained_failure_must_match_and_have_no_learning_orphans(tmp_path,monkeypatch):
    plan=fixture(tmp_path,monkeypatch)
    failure=tmp_path/'runs/failed_B1/failure.json'
    failure.write_text(json.dumps({'created':[{'seed':22,'batch':'unclosed'}]}))
    plan['closed_failed_predecessor']['failure_sha256']=sha(failure)
    with pytest.raises(ValueError,match='orphan'):
        closed_failure_dependency(tmp_path,plan)
