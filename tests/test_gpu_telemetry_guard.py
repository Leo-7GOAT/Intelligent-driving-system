import subprocess
import pytest
from scripts.watch_existing_pair import query_gpu_temperature


def test_one_transient_timeout_gets_one_bounded_retry_without_changing_temperature_gate():
    calls=[]
    def probe(command,**kwargs):
        calls.append((command,kwargs))
        if len(calls)==1:raise subprocess.TimeoutExpired(command,10)
        return subprocess.CompletedProcess(command,0,'45\n','')
    result=query_gpu_temperature(probe)
    assert result=={'status':'ok','temperature_c':45,
        'attempts':[{'status':'timeout','timeout_seconds':10},{'status':'ok'}]}
    assert len(calls)==2 and all(call[1]['timeout']==10 for call in calls)


@pytest.mark.parametrize('failure',['timeout','nonzero','malformed','empty','missing'])
def test_two_failed_probes_remain_unavailable_with_no_unbounded_retry(failure):
    calls=[]
    def probe(command,**kwargs):
        calls.append(command)
        if failure=='timeout':raise subprocess.TimeoutExpired(command,10)
        if failure=='missing':raise FileNotFoundError('nvidia-smi')
        return subprocess.CompletedProcess(command,1 if failure=='nonzero' else 0,
            'not-a-temperature' if failure=='malformed' else '', '')
    result=query_gpu_temperature(probe)
    assert result['status']=='unavailable' and result['temperature_c'] is None and len(calls)==2


def test_hot_reading_is_returned_immediately_so_the_existing_85C_guard_can_stop():
    calls=[]
    def probe(command,**kwargs):
        calls.append(command);return subprocess.CompletedProcess(command,0,'45\n85\n','')
    result=query_gpu_temperature(probe)
    assert result['status']=='ok' and result['temperature_c']==85 and len(calls)==1
