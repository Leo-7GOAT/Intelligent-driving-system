"""Metadata gate only; these tests never launch CARLA or a training process."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

POWERSHELL=shutil.which('pwsh')
pytestmark=pytest.mark.skipif(sys.platform!='win32' or not POWERSHELL,reason='Windows owned-launcher metadata gate')
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_registered_native.ps1'


@pytest.mark.parametrize('probe,expected',[
    ({'status':'CONNECTION_FAILED'},False),
    ({'status':'CONNECTED','client_version':'0.9.15','server_version':'0.9.15','actors':[]},True),
    ({'status':'CONNECTED','client_version':'0.9.15','server_version':'0.9.14','actors':[]},None),
    ({'status':'CONNECTED','client_version':'0.9.15','server_version':'0.9.15','actors':[{'id':42}]},None),
    ({'status':'UNKNOWN'},None),
])
def test_readiness_timeout_retries_but_foreign_or_unknown_world_is_rejected(tmp_path,probe,expected):
    probe_path=tmp_path/'probe.json';probe_path.write_text(json.dumps(probe))
    driver=tmp_path/'function_only.ps1'
    driver.write_text('''param([string]$Source,[string]$ProbePath)
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Test-NativeReadinessProbe'},$true)
if($null -eq $function){throw 'Missing readiness classifier'}
. ([scriptblock]::Create($function.Extent.Text))
$probe=Get-Content -LiteralPath $ProbePath -Raw | ConvertFrom-Json
@{ready=(Test-NativeReadinessProbe -Probe $probe)} | ConvertTo-Json -Compress
''',encoding='utf-8')
    result=subprocess.run([POWERSHELL,'-NoProfile','-File',str(driver),'-Source',str(SCRIPT),
        '-ProbePath',str(probe_path)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    if expected is None:assert result.returncode!=0
    else:
        assert result.returncode==0,result.stderr
        assert json.loads(result.stdout)['ready'] is expected


def fixture(tmp_path):
    batch=tmp_path/'batch with spaces';source=batch/'source/scripts';source.mkdir(parents=True)
    (source/'native_road_learning.py').write_text('# inert fixture, never executed\n')
    (batch/'calibration').mkdir();(batch/'calibration/calibration.json').write_text('{}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    protocol={'calibration_sha256':sha(batch/'calibration/calibration.json'),
        'source_hashes':{'scripts/native_road_learning.py':sha(source/'native_road_learning.py')},
        'created_epoch':1.,'deadline_epoch':2.,'carla_rpc_port':2200,'resource_limits':{'wall_seconds':1800},
        'routes':[{'points':[[.12345678901234567,-100.96441650390625,3.141592653589793]]}]}
    (batch/'protocol.json').write_text(json.dumps(protocol))
    return batch,protocol


def validate(batch,current_guard=False):
    return subprocess.run([POWERSHELL,'-NoProfile','-File',str(SCRIPT),'-Batch',str(batch),
        '-Python',sys.executable,'-CarlaPython',sys.executable,'-ValidateOnly',
        *(['-UseCurrentGuard'] if current_guard else [])],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)


def test_explicit_operational_guard_validation_still_writes_nothing(tmp_path):
    batch,_=fixture(tmp_path)
    before={p.relative_to(batch):p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    result=validate(batch,current_guard=True)
    assert result.returncode==0,result.stderr
    assert before=={p.relative_to(batch):p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    assert not (batch/'launcher_override').exists()


def test_operational_snapshot_is_separate_hash_bound_and_never_overwrites(tmp_path):
    origin=tmp_path/'guard source';(origin/'navstack').mkdir(parents=True);(origin/'scripts').mkdir()
    files=['navstack/__init__.py','scripts/watch_existing_pair.py','scripts/stop_owned_carla.ps1','scripts/probe_carla37.py','scripts/run_registered_native.ps1']
    for name in files:(origin/name).write_text('# inert fixture '+name,encoding='utf-8')
    driver=tmp_path/'snapshot_only.ps1'
    driver.write_text('''param([string]$Source,[string]$Origin,[string]$Destination)
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'New-NativeGuardSnapshot'},$true)
. ([scriptblock]::Create($function.Extent.Text))
New-NativeGuardSnapshot -SourceRoot $Origin -Destination $Destination | ConvertTo-Json -Compress
''',encoding='utf-8')
    destination=tmp_path/'operational snapshot'
    command=[POWERSHELL,'-NoProfile','-File',str(driver),'-Source',str(SCRIPT),'-Origin',str(origin),'-Destination',str(destination)]
    first=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert first.returncode==0,first.stderr
    manifest=json.loads((destination/'manifest.json').read_text(encoding='utf-8-sig'))
    assert set(manifest['files'])==set(files) and manifest['temperature_stop_c']==85
    for name in files:
        assert (destination/'source'/name).read_bytes()==(origin/name).read_bytes()
        assert manifest['files'][name]==hashlib.sha256((origin/name).read_bytes()).hexdigest()
    second=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert second.returncode!=0 and 'Preserve existing' in second.stderr


def test_launcher_metadata_gate_is_read_only_and_does_not_claim_server_readiness(tmp_path):
    batch,_=fixture(tmp_path)
    before={p.relative_to(batch):p.read_bytes() for p in batch.rglob('*') if p.is_file()}
    result=validate(batch)
    assert result.returncode==0,result.stderr
    output=json.loads(result.stdout)
    assert output['status']=='INPUTS_VALIDATED_NOT_SERVER_PROBED' and output['source_files']==1
    assert before=={p.relative_to(batch):p.read_bytes() for p in batch.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['missing_calibration','changed_source','started','escaping_source','prepared_changed'])
def test_launcher_rejects_bad_inputs_before_any_lifecycle(tmp_path,fault):
    batch,p=fixture(tmp_path)
    if fault=='missing_calibration':p.pop('calibration_sha256')
    elif fault=='changed_source':(batch/'source/scripts/native_road_learning.py').write_text('changed')
    elif fault=='started':(batch/'started.json').write_text('{}')
    elif fault=='prepared_changed':(batch/'prepared.json').write_text(json.dumps({'protocol_sha256':'a'*64}))
    else:p['source_hashes']={'../../outside.py':'a'*64}
    (batch/'protocol.json').write_text(json.dumps(p))
    result=validate(batch)
    assert result.returncode!=0
    assert not (batch/'infrastructure').exists() and not (batch/'worker_process.json').exists()


def test_control_worker_kind_validates_metadata_without_any_server_or_process_launch(tmp_path):
    batch,p=fixture(tmp_path)
    path=batch/'source/scripts/native_curve_control.py';path.write_text('# inert control fixture,never executed\n')
    p['source_hashes']['scripts/native_curve_control.py']=hashlib.sha256(path.read_bytes()).hexdigest()
    (batch/'protocol.json').write_text(json.dumps(p))
    result=subprocess.run([POWERSHELL,'-NoProfile','-File',str(SCRIPT),'-Batch',str(batch),'-Kind','control',
        '-Python',sys.executable,'-CarlaPython',sys.executable,'-ValidateOnly'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['worker']=='native_curve_control.py'
    assert not (batch/'infrastructure').exists() and not (batch/'worker_process.json').exists()


def test_queue_activation_preserves_all_nonclock_inputs_and_original_bytes(tmp_path):
    batch,original=fixture(tmp_path)
    before=(batch/'protocol.json').read_bytes()
    command=[POWERSHELL,'-NoProfile','-File',str(SCRIPT),'-Batch',str(batch),
             '-Python',sys.executable,'-CarlaPython',sys.executable,'-ActivateQueued','-ActivateOnly']
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='ACTIVATED_NOT_STARTED'
    assert (batch/'queued_protocol.json').read_bytes()==before
    after=json.loads((batch/'protocol.json').read_text(encoding='utf-8-sig'))
    assert after['deadline_epoch']-after['created_epoch']==1800
    assert {k:v for k,v in original.items() if k not in ('created_epoch','deadline_epoch')}=={k:v for k,v in after.items() if k not in ('created_epoch','deadline_epoch')}
    assert not (batch/'infrastructure').exists() and not (batch/'worker_process.json').exists()
    repeated=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert repeated.returncode!=0 and (batch/'queued_protocol.json').read_bytes()==before
