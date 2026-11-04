"""Extract only the pure peer classifier;never start/stop a simulator."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

pytestmark=pytest.mark.skipif(sys.platform!='win32' or not shutil.which('pwsh'),reason='PowerShell classifier')
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_carla_owned_foreground.ps1'


@pytest.mark.parametrize('commands,accepted',[
    (['CarlaUE4 -carla-rpc-port=2100'],True),
    (['CarlaUE4 -carla-rpc-port=2100','CarlaUE4 -carla-rpc-port=2100'],True),
    (['CarlaUE4 -carla-rpc-port=2111'],False),
    (['CarlaUE4'],False),
    (['CarlaUE4 -carla-rpc-port=2100 -carla-rpc-port=2200'],False),
    (['CarlaUE4 -carla-rpc-port=2100','CarlaUE4 -carla-rpc-port=2200'],False),
])
def test_only_one_explicit_disjoint_peer_span_is_admitted(tmp_path,commands,accepted):
    data=tmp_path/'peers.json';data.write_text(json.dumps([dict(ProcessId=i,CommandLine=c) for i,c in enumerate(commands)]))
    driver=tmp_path/'classify.ps1'
    driver.write_text('''param([string]$Source,[string]$Data)
$ErrorActionPreference='Stop'
$tokens=$null;$errors=$null
$ast=[System.Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors)
if($errors.Count){throw 'Launcher must parse'}
$function=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Get-DisjointCarlaPeers'},$true)
. ([scriptblock]::Create($function.Extent.Text))
$peers=@(Get-Content -LiteralPath $Data -Raw | ConvertFrom-Json)
@(Get-DisjointCarlaPeers -Processes $peers -RequestedPorts @(2110,2111,2112)) | ConvertTo-Json -Compress
''',encoding='utf-8')
    result=subprocess.run([shutil.which('pwsh'),'-NoProfile','-File',str(driver),'-Source',str(SCRIPT),'-Data',str(data)],
        capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=20)
    assert (result.returncode==0) is accepted,result.stderr
