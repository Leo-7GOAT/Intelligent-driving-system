import pytest
from scripts.prepare_native_evaluation_recovery import recorded_lifecycle_gone


@pytest.mark.parametrize('creation,expected',[(950.,True),(900.,False),(901.,False),(899.,False),(None,False),(1003.,False)])
def test_reused_pid_is_distinguished_only_by_verified_later_creation(monkeypatch,creation,expected):
    monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:pid==123)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.time.time',lambda:1001.)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.windows_process_creation_epoch',lambda pid:creation)
    assert recorded_lifecycle_gone((123,124),900.) is expected


def test_one_reused_pid_cannot_hide_an_original_live_or_unknown_process(monkeypatch):
    monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:True)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.time.time',lambda:1001.)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.windows_process_creation_epoch',lambda pid:950. if pid==123 else 899.)
    assert not recorded_lifecycle_gone((123,124),900.)


@pytest.mark.parametrize('ended',[None,True,float('nan'),1002.])
def test_unknown_or_invalid_exit_time_never_authorizes_live_pid_recovery(monkeypatch,ended):
    monkeypatch.setattr('navstack.process_lease.pid_alive',lambda pid:True)
    monkeypatch.setattr('scripts.prepare_native_evaluation_recovery.time.time',lambda:1001.)
    assert not recorded_lifecycle_gone((123,),ended)
