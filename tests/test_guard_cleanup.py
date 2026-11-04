"""Fault injection only: these checks do not claim an actual CARLA shutdown."""
import json
import subprocess

import pytest

from scripts import watch_existing_pair as guard


@pytest.mark.parametrize('failure',[
    subprocess.TimeoutExpired('powershell',60,output=b'bridge drain pending',stderr=b'probe timeout'),
    FileNotFoundError('PowerShell unavailable'),
])
def test_cleanup_failure_has_durable_receipt_without_retry(tmp_path,monkeypatch,failure):
    calls=[]
    def fail(*args,**kwargs):
        calls.append((args,kwargs))
        raise failure
    monkeypatch.setattr(guard.subprocess,'run',fail)
    receipt=guard.cleanup_owned_server(tmp_path,'powershell')
    saved=json.loads((tmp_path/'server_cleanup.json').read_text(encoding='utf-8'))
    assert saved==receipt and saved['returncode'] is None
    assert type(failure).__name__ in saved['error'] and len(calls)==1
    if isinstance(failure,subprocess.TimeoutExpired):
        assert saved['stdout']=='bridge drain pending' and saved['stderr']=='probe timeout'


def test_nonzero_cleanup_exit_is_not_reported_as_success(tmp_path,monkeypatch):
    monkeypatch.setattr(guard.subprocess,'run',lambda *args,**kwargs:
        subprocess.CompletedProcess(args[0],1,'','Server is not verified empty and matching'))
    receipt=guard.cleanup_owned_server(tmp_path,'powershell')
    assert receipt['returncode']==1 and 'not verified empty' in receipt['stderr']
