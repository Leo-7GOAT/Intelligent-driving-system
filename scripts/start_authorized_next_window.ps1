param(
    [Parameter(Mandatory=$true)][string]$Batch,
    [Parameter(Mandatory=$true)][ValidateRange(0.1,24)][double]$Hours,
    [switch]$Inspect
)
# Invoke only after the user explicitly schedules another finite local window.
# This script is not called by a watcher, scheduler, or the current batch finalizer.
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskBatch = (Resolve-Path -LiteralPath $Batch).Path
$taskName = Split-Path $taskBatch -Leaf
$taskSpecPath = Join-Path $taskBatch 'batch.json'
$taskSpec = Get-Content -LiteralPath $taskSpecPath -Raw -Encoding utf8 | ConvertFrom-Json
$taskNow = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$taskStopPath = Join-Path $taskBatch 'STOP'
$taskStop = if (Test-Path -LiteralPath $taskStopPath) { (Get-Content -LiteralPath $taskStopPath -Raw).Trim() } else { '' }
$taskProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(python|powershell|pwsh)\.exe$' -and
    $_.CommandLine -and $_.CommandLine.Contains($taskName)
})
# Generic user/technical STOP must not be cleared automatically.
$taskAllowedStops = @('', 'six_hour_batch_budget', 'six_hour_budget', 'wall_budget_reached')
$taskReady = $taskNow -ge $taskSpec.deadline_epoch -and $taskProcesses.Count -eq 0 -and $taskAllowedStops.Contains($taskStop)
if ($Inspect) {
    [pscustomobject]@{ready=$taskReady;old_deadline=$taskSpec.deadline_epoch;stop=$taskStop;active_pids=@($taskProcesses.ProcessId);proposed_hours=$Hours;mutations_performed=$false} | ConvertTo-Json -Depth 4
    return
}
if ($taskNow -lt $taskSpec.deadline_epoch) { throw 'The original window has not ended. No window was changed.' }
if ($taskProcesses.Count -gt 0) { throw ('Batch processes are still active: ' + ($taskProcesses.ProcessId -join ', ')) }
if (-not $taskAllowedStops.Contains($taskStop)) { throw ('STOP is not a normal budget boundary; investigate first: ' + $taskStop) }
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskBridgePython = Join-Path $taskRoot '.carla-python37\runtime\python.exe'
& $taskPython (Join-Path $PSScriptRoot 'verify_batch_integrity.py') --batch $taskBatch
if ($LASTEXITCODE -ne 0) { throw 'Frozen code or accounting check failed. No window was changed.' }

$taskServerProcesses = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^CarlaUE4' })
$taskListeners = @(Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue)
$taskReuseServer = $false
if ($taskServerProcesses.Count -gt 0 -or $taskListeners.Count -gt 0) {
    $taskServerRecord = Get-Content -LiteralPath (Join-Path $taskBatch 'infrastructure\server_current.json') -Raw -Encoding utf8 | ConvertFrom-Json
    $taskOwnedLauncher = @($taskServerProcesses | Where-Object { $_.ProcessId -eq $taskServerRecord.launcher_pid -and $_.ExecutablePath -eq 'D:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe' })
    if ($taskOwnedLauncher.Count -ne 1) { throw 'Existing server identity is not established as this batch owner; no duplicate server or new window started.' }
    $taskProbe = Join-Path $taskBatch ('infrastructure\next_window_identity_' + $taskNow + '.json')
    & $taskBridgePython (Join-Path $PSScriptRoot 'probe_carla37.py') --output $taskProbe
    $taskState = Get-Content -LiteralPath $taskProbe -Raw -Encoding utf8 | ConvertFrom-Json
    if ($taskState.status -ne 'CONNECTED' -or @($taskState.actors).Count -ne 0 -or $taskState.client_version -ne '0.9.15' -or $taskState.server_version -ne '0.9.15') { throw 'Existing server is not verified empty, version-matched and ready.' }
    $taskReuseServer = $true
}

$taskArchive = [IO.Path]::GetFullPath((Join-Path $taskBatch ('budget_windows\ended_' + $taskSpec.started_epoch + '_' + $taskNow)))
if (-not $taskArchive.StartsWith($taskBatch + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive path escaped the batch.' }
New-Item -ItemType Directory -Path $taskArchive | Out-Null
foreach ($taskFile in @('batch.json','budget_guard.json','finalizer_status.json','run_status.md','research_results.md','failure_analysis.md')) {
    $taskSourceFile = Join-Path $taskBatch $taskFile
    if (Test-Path -LiteralPath $taskSourceFile) { Copy-Item -LiteralPath $taskSourceFile -Destination (Join-Path $taskArchive $taskFile) }
}
$taskAnalysisArchive = Join-Path $taskArchive 'analysis'
New-Item -ItemType Directory -Path $taskAnalysisArchive | Out-Null
Get-ChildItem -LiteralPath (Join-Path $taskBatch 'analysis') -File | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $taskAnalysisArchive $_.Name)
}
if (Test-Path -LiteralPath $taskStopPath) { Move-Item -LiteralPath $taskStopPath -Destination (Join-Path $taskArchive 'STOP') }
$taskWindow = @{}
foreach ($taskProperty in $taskSpec.PSObject.Properties) { $taskWindow[$taskProperty.Name] = $taskProperty.Value }
$taskWindow.started_epoch = $taskNow
$taskWindow.deadline_epoch = $taskNow + [long]($Hours * 3600)
$taskWindow.wall_budget_seconds = [long]($Hours * 3600)
$taskWindow.status = 'explicit_next_window'
$taskWindow.previous_window_archive = $taskArchive
$taskWindow.authorization = 'New finite window explicitly invoked by the user; original window preserved; same frozen confirmation and cumulative checkpoints.'
$taskUtf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($taskSpecPath, ($taskWindow | ConvertTo-Json -Depth 8), $taskUtf8)

$taskReceipts = @()
foreach ($taskScript in @('batch_budget_guard.ps1','finalize_research_batch.py')) {
    $taskLogBase = Join-Path $taskBatch ('next_window_' + $taskNow + '_' + $taskScript)
    if ($taskScript.EndsWith('.ps1')) {
        $taskExecutable = 'powershell.exe'
        $taskArguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"' + (Join-Path $PSScriptRoot $taskScript) + '"'),'-Batch',('"' + $taskBatch + '"'))
    } else {
        $taskExecutable = $taskPython
        $taskArguments = @(('"' + (Join-Path $PSScriptRoot $taskScript) + '"'),'--batch',('"' + $taskBatch + '"'))
    }
    $taskChild = Start-Process -FilePath $taskExecutable -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput ($taskLogBase + '.out') -RedirectStandardError ($taskLogBase + '.err')
    $taskReceipts += @{script=$taskScript;launcher_pid=$taskChild.Id}
}
if (-not $taskReuseServer) {
    $taskServerOutput = Join-Path $taskBatch ('infrastructure\window_server_' + $taskNow)
    & (Join-Path $PSScriptRoot 'start_carla.ps1') -OutputDirectory $taskServerOutput
    $taskStartup = Get-Content -LiteralPath (Join-Path $taskServerOutput 'server_start.json') -Raw -Encoding utf8
    [IO.File]::WriteAllText((Join-Path $taskBatch 'infrastructure\server_current.json'), $taskStartup.TrimStart([char]0xFEFF), $taskUtf8)
    $taskReadyServer = $false
    for ($taskAttempt=0; $taskAttempt -lt 20; $taskAttempt++) {
        $taskProbe = Join-Path $taskServerOutput ('probe_' + $taskAttempt + '.json')
        & $taskBridgePython (Join-Path $PSScriptRoot 'probe_carla37.py') --output $taskProbe
        $taskState = Get-Content -LiteralPath $taskProbe -Raw -Encoding utf8 | ConvertFrom-Json
        if ($taskState.status -eq 'CONNECTED' -and $taskState.client_version -eq '0.9.15' -and $taskState.server_version -eq '0.9.15') { $taskReadyServer = $true; break }
        Start-Sleep -Seconds 2
    }
    if (-not $taskReadyServer) { throw 'Owned server started but readiness failed; inspect preserved probes. New finite window guard remains active.' }
}
& (Join-Path $PSScriptRoot 'resume_current_batch.ps1') -Batch $taskBatch
[IO.File]::WriteAllText((Join-Path $taskBatch ('next_window_' + $taskNow + '_receipt.json')), ($taskReceipts | ConvertTo-Json -Depth 5), $taskUtf8)
