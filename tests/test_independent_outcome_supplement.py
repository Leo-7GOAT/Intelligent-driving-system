import hashlib
import json
import pytest

from scripts.run_independent_outcome_cohort import completed_phase, resumed_selection, stage
from scripts.supplement_independent_outcome_cohort import retire_barrier


@pytest.mark.parametrize('seeds,namespace,reuse,port',[
    ([44,44],'execution_new',True,2110),([77],'execution_new',True,2110),
    ([44],'execution_new',False,2110),([44],'../execution_new',True,2110),
    ([44],'execution_new',True,2100),([True],'execution_new',True,2110),
])
def test_invalid_continuation_cannot_select_or_replace_existing_work(seeds,namespace,reuse,port):
    with pytest.raises(ValueError):resumed_selection(seeds,namespace,reuse,port)


def test_subset_preserves_registered_order_and_accepts_disjoint_second_slot():
    assert resumed_selection([55,44],'execution_new',True,2120)==(44,55)
    assert resumed_selection(None,None,False,2110)==(11,22,33,44,55)


def test_completed_receipt_must_still_bind_the_original_protocol(tmp_path,monkeypatch):
    import scripts.run_independent_outcome_cohort as m
    cohort=tmp_path/'cohort';(cohort/'execution').mkdir(parents=True)
    batch=tmp_path/'batch';batch.mkdir();protocol=batch/'protocol.json';protocol.write_text('{}')
    (cohort/'execution/seed11_baseline_complete.json').write_text(json.dumps(dict(batch='batch',protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())))
    monkeypatch.setattr(m,'require_finished_parent',lambda p:None)
    assert completed_phase(cohort,tmp_path,'seed11_baseline',True)
    assert not completed_phase(cohort,tmp_path,'seed11_baseline',False)
    protocol.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):completed_phase(cohort,tmp_path,'seed11_baseline',True)


def test_only_never_started_barrier_can_be_retired(tmp_path):
    barrier=tmp_path/'20261104_independent_outcome_seed44_zero_PPO_known24';barrier.mkdir()
    output=tmp_path/'continuation';output.mkdir()
    note=dict(status='NOT_STARTED_ADMISSION_BARRIER',native_controls=0,native_jobs=0)
    (barrier/'admission_blocked_by_user_deadline.json').write_text(json.dumps(note))
    (barrier/'worker_process.json').write_text('{}')
    with pytest.raises(ValueError,match='never-started'):retire_barrier(tmp_path,output)
    (barrier/'worker_process.json').unlink()
    receipt=retire_barrier(tmp_path,output)
    assert receipt['native_jobs']==0 and not barrier.exists()
    assert json.loads((output/'retired_three_hour_admission_barrier/admission_blocked_by_user_deadline.json').read_text())==note


def test_unregistered_port_is_rejected_before_staging_anything(tmp_path):
    with pytest.raises(ValueError,match='disjoint'):
        stage(tmp_path,tmp_path,{},'H','unused',training=True,checkpoint=tmp_path,rpc_port=2100)
    assert not list(tmp_path.iterdir())


def test_explicit_runtime_paths_preserve_old_registration_and_fail_before_missing_runtime(tmp_path):
    from scripts.run_independent_outcome_cohort import runtime_paths
    plan=dict(runtime_python='unavailable_old_path',runtime_carla_python='unavailable_old_client')
    before=dict(plan)
    with pytest.raises(ValueError,match='executables'):runtime_paths(plan)
    py=tmp_path/'learning.exe';client=tmp_path/'client.exe';py.touch();client.touch()
    assert runtime_paths(plan,py,client)==(str(py.resolve()),str(client.resolve()))
    assert plan==before
