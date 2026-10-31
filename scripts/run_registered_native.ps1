param(
    [Parameter(Mandatory=$true)][string]$Batch,
    [ValidateSet('learning','rule','control')][string]$Kind='learning',
    [string]$Python,
    [string]$CarlaPython,
    [switch]$ValidateOnly,
    [switch]$ActivateQueued,
    [switch]$ActivateOnly,
    [switch]$UseCurrentGuard
)
function New-NativeGuardSnapshot {
    param([Parameter(Mandatory=$true)][string]$SourceRoot,[Parameter(Mandatory=$true)][string]$Destination)
    if(Test-Path -LiteralPath $Destination){throw 'Preserve existing operational guard snapshot'}
    $snapshotFiles=@(Get-ChildItem -LiteralPath (Join-Path $SourceRoot 'navstack') -Filter '*.py' -File -Recurse)
    foreach($snapshotName in @('watch_existing_pair.py','stop_owned_carla.ps1','run_registered_native.ps1')) {
        $snapshotFiles+=Get-Item -LiteralPath (Join-Path $SourceRoot ('scripts/'+$snapshotName)) -ErrorAction Stop
    }
    $snapshotRoot=Join-Path $Destination 'source'
    $snapshotHashes=@{}
    foreach($snapshotFile in $snapshotFiles) {
        $snapshotRelative=[IO.Path]::GetRelativePath($SourceRoot,$snapshotFile.FullName)
        if($snapshotRelative.StartsWith('..')){throw 'Operational guard source escapes project'}
        $snapshotTarget=Join-Path $snapshotRoot $snapshotRelative
        New-Item -ItemType Directory -Path (Split-Path -Parent $snapshotTarget) -Force | Out-Null
        Copy-Item -LiteralPath $snapshotFile.FullName -Destination $snapshotTarget -ErrorAction Stop
        $snapshotHashes[$snapshotRelative.Replace('\','/')]=(Get-FileHash -LiteralPath $snapshotTarget -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $snapshotManifest=Join-Path $Destination 'manifest.json'
    @{scope='Operational watchdog override only;worker policy,physical config and frozen source remain unchanged';files=$snapshotHashes;gpu_probe_attempts=2;gpu_timeout_each_seconds=10;temperature_stop_c=85} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $snapshotManifest -Encoding utf8
    return @{script=(Join-Path $snapshotRoot 'scripts/watch_existing_pair.py');manifest_sha256=(Get-FileHash -LiteralPath $snapshotManifest -Algorithm SHA256).Hash.ToLowerInvariant()}
}
function Test-NativeReadinessProbe {
    param([Parameter(Mandatory=$true)]$Probe)
    # probe_carla37 reports connection timeouts as JSON with process exit0.
    # A still-starting owned server is retryable within the bounded loop.
    if($Probe.status -eq 'CONNECTION_FAILED'){return $false}
    if($Probe.status -ne 'CONNECTED'){throw 'Unexpected readiness probe status'}
    if($Probe.client_version -ne '0.9.15' -or $Probe.server_version -ne '0.9.15' -or @($Probe.actors).Count -ne 0){throw 'Connected server is not matching-version and empty'}
    return $true
}
$ErrorActionPreference='Stop'
if($PSVersionTable.PSVersion.Major -lt 7){throw 'PowerShell7 is required for owned native lifecycle handling'}
if($ActivateOnly -and -not $ActivateQueued){throw 'ActivateOnly requires explicit ActivateQueued'}
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskBatch=(Resolve-Path -LiteralPath $Batch).Path
$taskProtocolPath=Join-Path $taskBatch 'protocol.json'
$taskProtocol=Get-Content -LiteralPath $taskProtocolPath -Raw | ConvertFrom-Json
$taskSource=(Resolve-Path -LiteralPath (Join-Path $taskBatch 'source')).Path
if (-not $Python) { $Python=Join-Path $taskRoot '.venv/Scripts/python.exe' }
if (-not $CarlaPython) { $CarlaPython=$env:NAVSTACK_CARLA_PYTHON }
if (-not $CarlaPython) { $CarlaPython=Join-Path $taskRoot '.carla-python37/runtime/python.exe' }
$taskPython=(Resolve-Path -LiteralPath $Python).Path
$taskClient=(Resolve-Path -LiteralPath $CarlaPython).Path
$taskWorkerName=switch($Kind){'rule'{'evaluate_native_rule.py'} 'control'{'native_curve_control.py'} default{'native_road_learning.py'}}
$taskWorkerScript=Join-Path $taskSource ('scripts/'+$taskWorkerName)
$taskGuardScript=Join-Path $taskSource 'scripts/watch_existing_pair.py'
if($UseCurrentGuard -and -not(Test-Path -LiteralPath (Join-Path $taskRoot 'scripts/watch_existing_pair.py') -PathType Leaf)){throw 'Explicit current watchdog source missing'}
foreach($taskRequired in @('calibration_sha256','source_hashes','deadline_epoch','created_epoch','carla_rpc_port','resource_limits')) {
    if ($taskProtocol.PSObject.Properties.Name -notcontains $taskRequired) { throw "Missing registered input: $taskRequired" }
}
if ($taskProtocol.carla_rpc_port -lt 1024 -or $taskProtocol.carla_rpc_port -gt 65533) { throw 'Invalid RPC port' }
$taskWall=[double]$taskProtocol.resource_limits.wall_seconds
if ([double]::IsNaN($taskWall) -or [double]::IsInfinity($taskWall) -or $taskWall -lt 300 -or $taskWall -gt 21600) { throw 'Invalid bounded runtime window' }
foreach($taskHash in $taskProtocol.source_hashes.PSObject.Properties) {
    $taskFile=[IO.Path]::GetFullPath((Join-Path $taskSource $taskHash.Name))
    if (-not $taskFile.StartsWith($taskSource+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Source entry escapes frozen tree' }
    if ((Get-FileHash -LiteralPath $taskFile -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskHash.Value) { throw "Frozen source changed: $($taskHash.Name)" }
}
if ((Get-FileHash -LiteralPath (Join-Path $taskBatch 'calibration/calibration.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskProtocol.calibration_sha256) { throw 'Calibration changed' }
if (-not (Test-Path -LiteralPath $taskWorkerScript -PathType Leaf)) { throw 'Registered worker missing' }
$taskPreparedPath=Join-Path $taskBatch 'prepared.json'
if(Test-Path -LiteralPath $taskPreparedPath) {
    $taskPreparedValue=Get-Content -LiteralPath $taskPreparedPath -Raw | ConvertFrom-Json
    if($taskPreparedValue.protocol_sha256 -and $taskPreparedValue.protocol_sha256 -ne (Get-FileHash -LiteralPath $taskProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()){throw 'Prepared protocol hash changed'}
}
foreach($taskEvidence in @('started.json','worker_process.json','guard_final.json','launcher_failure.json','launcher_execution.json','infrastructure/server_start.claim','infrastructure/server_start.json')) {
    if (Test-Path -LiteralPath (Join-Path $taskBatch $taskEvidence)) { throw "Preserve existing lifecycle: $taskEvidence; prepare a recovery child instead" }
}
if ($ValidateOnly) {
    @{status='INPUTS_VALIDATED_NOT_SERVER_PROBED';batch=(Split-Path -Leaf $taskBatch);source_files=@($taskProtocol.source_hashes.PSObject.Properties).Count;worker=$taskWorkerName} | ConvertTo-Json -Compress
    exit 0
}
$taskLease=[IO.File]::Open((Join-Path $taskBatch 'run_registered_native.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
try {
    # The handle is held through supervision. Another launcher cannot enter
    # activation or clean up this launcher's server after losing a startup race.
    $taskLease.SetLength(0)
    $taskLeaseBytes=[Text.Encoding]::UTF8.GetBytes("pid=$PID`nepoch=$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())`n")
    $taskLease.Write($taskLeaseBytes,0,$taskLeaseBytes.Length);$taskLease.Flush($true)
if ($ActivateQueued) {
    foreach($taskSnapshot in @('queued_protocol.json','queued_prepared.json','activation.json')) {
        if(Test-Path -LiteralPath (Join-Path $taskBatch $taskSnapshot)){throw 'Preserve existing queue activation evidence'}
    }
    $taskOldProtocolHash=(Get-FileHash -LiteralPath $taskProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Copy-Item -LiteralPath $taskProtocolPath -Destination (Join-Path $taskBatch 'queued_protocol.json') -ErrorAction Stop
    $taskPrepared=Join-Path $taskBatch 'prepared.json'
    if(Test-Path -LiteralPath $taskPrepared){Copy-Item -LiteralPath $taskPrepared -Destination (Join-Path $taskBatch 'queued_prepared.json') -ErrorAction Stop}
    $taskQueuedCreated=$taskProtocol.created_epoch
    $taskNow=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000
    $taskProtocol.created_epoch=$taskNow
    $taskProtocol.deadline_epoch=$taskNow+$taskProtocol.resource_limits.wall_seconds
    $taskProtocol | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $taskProtocolPath -Encoding utf8
    $taskNewProtocolHash=(Get-FileHash -LiteralPath $taskProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant()
    @{status='ACTIVATED_BEFORE_ANY_LIFECYCLE';original_protocol_sha256=$taskOldProtocolHash;activated_protocol_sha256=$taskNewProtocolHash;queued_created_epoch=$taskQueuedCreated;activated_epoch=$taskNow;wall_seconds=$taskProtocol.resource_limits.wall_seconds;policy_or_task_changes=0} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBatch 'activation.json') -Encoding utf8
    @{status='ACTIVATED_NOT_STARTED';protocol_sha256=$taskNewProtocolHash} | ConvertTo-Json | Set-Content -LiteralPath $taskPrepared -Encoding utf8
}
if($ActivateOnly){@{status='ACTIVATED_NOT_STARTED';controls=0;policy_or_task_changes=0} | ConvertTo-Json -Compress;return}
if ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -ge $taskProtocol.deadline_epoch) { throw 'Registered deadline expired; use explicit queued activation only before first lifecycle' }
if ((Get-PSDrive -Name ([IO.Path]::GetPathRoot($taskBatch).TrimEnd('\').TrimEnd(':'))).Free -lt 12GB) { throw 'Disk below12GiB before launch' }
$env:NAVSTACK_CARLA_PYTHON=$taskClient
$taskPwsh=(Get-Command pwsh).Source
$taskInfrastructure=Join-Path $taskBatch 'infrastructure'
New-Item -ItemType Directory -Path $taskInfrastructure -Force | Out-Null
$taskServerHost=$null
try {
    $taskGuardManifest=$null
    if($UseCurrentGuard) {
        $taskGuardSnapshot=New-NativeGuardSnapshot -SourceRoot $taskRoot -Destination (Join-Path $taskBatch 'launcher_override')
        $taskGuardScript=$taskGuardSnapshot.script;$taskGuardManifest=$taskGuardSnapshot.manifest_sha256
    }
    @{wrapper_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();wrapper='scripts/run_registered_native.ps1';protocol_sha256=(Get-FileHash -LiteralPath $taskProtocolPath -Algorithm SHA256).Hash.ToLowerInvariant();epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();guard_current_override=[bool]$UseCurrentGuard;guard_source_relative=[IO.Path]::GetRelativePath($taskBatch,$taskGuardScript).Replace('\','/');guard_source_sha256=(Get-FileHash -LiteralPath $taskGuardScript -Algorithm SHA256).Hash.ToLowerInvariant();guard_manifest_sha256=$taskGuardManifest} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBatch 'launcher_execution.json') -Encoding utf8
    $taskServerScript=Join-Path $taskSource 'scripts/run_carla_owned_foreground.ps1'
    $taskServerHost=Start-Process -FilePath $taskPwsh -ArgumentList @('-NoProfile','-File',('"'+$taskServerScript+'"'),'-RpcPort',[string]$taskProtocol.carla_rpc_port,'-RenderOffScreen','-OutputDirectory',('"'+$taskInfrastructure+'"')) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskInfrastructure 'server_host_stdout.log') -RedirectStandardError (Join-Path $taskInfrastructure 'server_host_stderr.log')
    $taskReady=$false
    for($taskAttempt=0;$taskAttempt -lt 20;$taskAttempt++) {
        if($taskServerHost.HasExited){throw 'Owned server launcher ended before readiness'}
        if(Test-Path -LiteralPath (Join-Path $taskInfrastructure 'server_start.json')) {
            $taskProbe=Join-Path $taskInfrastructure ('readiness_probe_{0:D2}.json' -f $taskAttempt)
            & $taskClient (Join-Path $taskSource 'scripts/probe_carla37.py') --port $taskProtocol.carla_rpc_port --output $taskProbe >> (Join-Path $taskInfrastructure 'readiness_stdout.log') 2>> (Join-Path $taskInfrastructure 'readiness_stderr.log')
            if($LASTEXITCODE -eq 0) {
                $taskObserved=Get-Content -LiteralPath $taskProbe -Raw | ConvertFrom-Json
                if(Test-NativeReadinessProbe -Probe $taskObserved){$taskReady=$true;break}
            }
        }
        Start-Sleep -Seconds 1
    }
    if(-not $taskReady){throw 'Owned server readiness budget exhausted'}
    $taskLaunch=Start-Process -FilePath $taskPython -ArgumentList @('-u',('"'+$taskWorkerScript+'"'),'--batch',('"'+$taskBatch+'"')) -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskBatch 'stdout.log') -RedirectStandardError (Join-Path $taskBatch 'stderr.log')
    $taskWorker=$null
    for($taskAttempt=0;$taskAttempt -lt 40;$taskAttempt++) {
        $taskCandidates=@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -and $_.CommandLine.Contains($taskWorkerScript) -and $_.CommandLine.Contains($taskBatch) -and ($_.ParentProcessId -eq $taskLaunch.Id -or $_.ProcessId -eq $taskLaunch.Id) })
        $taskChildren=@($taskCandidates | Where-Object { $_.ParentProcessId -eq $taskLaunch.Id })
        if($taskChildren.Count -eq 1){$taskWorker=$taskChildren[0];break}
        if($taskCandidates.Count -eq 1 -and $taskCandidates[0].ExecutablePath -notlike '*\.venv\Scripts\python.exe'){$taskWorker=$taskCandidates[0];break}
        if($taskLaunch.HasExited){$taskLaunch.WaitForExit();throw "Worker ended before guard adoption; launcher exit code=$($taskLaunch.ExitCode)"}
        Start-Sleep -Milliseconds 100
    }
    if($null -eq $taskWorker){throw 'Could not identify exactly one task-launched worker'}
    @{pid=$taskWorker.ProcessId;executable=$taskWorker.ExecutablePath;command=$taskWorker.CommandLine;creation_epoch=([DateTimeOffset]$taskWorker.CreationDate.ToUniversalTime()).ToUnixTimeMilliseconds()/1000;verified_at=[DateTimeOffset]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBatch 'worker_process.json') -Encoding utf8
    & $taskPython -u $taskGuardScript --batch $taskBatch --identity (Join-Path $taskBatch 'worker_process.json') --powershell $taskPwsh > (Join-Path $taskBatch 'guard_stdout.log') 2> (Join-Path $taskBatch 'guard_stderr.log')
    if($LASTEXITCODE -ne 0){throw 'Guard failed; inspect its immutable logs and actual worker identity'}
    $taskFinal=Get-Content -LiteralPath (Join-Path $taskBatch 'guard_final.json') -Raw | ConvertFrom-Json
    if($taskFinal.exit_code -ne 0 -or $null -ne $taskFinal.reason){throw 'Worker ended nonnormally; inspect guard_final.json'}
    $taskStop=Get-Content -LiteralPath (Join-Path $taskInfrastructure 'server_stop.json') -Raw | ConvertFrom-Json
    if($taskStop.status -ne 'owned_empty_server_stopped'){throw 'Owned cleanup not verified'}
    @{status='COMPLETE_AND_OWNED_SERVER_STOPPED';batch=(Split-Path -Leaf $taskBatch)} | ConvertTo-Json -Compress
} catch {
    @{status='LAUNCH_OR_SUPERVISION_FAILURE';epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();error=$_.Exception.Message} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskBatch 'launcher_failure.json') -Encoding utf8
    if((Test-Path -LiteralPath (Join-Path $taskInfrastructure 'server_start.json')) -and -not(Test-Path -LiteralPath (Join-Path $taskInfrastructure 'server_stop.json'))) {
        # This helper refuses live bridges or nonempty/foreign worlds. Never
        # escalate a failed probe to an unverified process/actor kill.
        & (Join-Path $taskSource 'scripts/stop_owned_carla.ps1') -Batch $taskBatch -BridgeDrainSeconds 30 > (Join-Path $taskBatch 'launcher_cleanup_stdout.log') 2> (Join-Path $taskBatch 'launcher_cleanup_stderr.log')
    }
    throw
}
} finally {
    $taskLease.Dispose()
}
