param(
    [Parameter(Mandatory=$true)][string]$Batch,
    [switch]$Inspect
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$taskBatch = (Resolve-Path -LiteralPath $Batch).Path
$taskSpec = Get-Content -LiteralPath (Join-Path $taskBatch 'batch.json') -Raw | ConvertFrom-Json
$taskFreeze = Get-Content -LiteralPath (Join-Path $taskBatch 'development_freeze.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($taskFreeze.protocol -ne 'formal_v5') { throw 'This resume entry point requires the registered formal_v5 protocol.' }
$taskSource = (Resolve-Path -LiteralPath (Join-Path $taskBatch 'source_formal_v5')).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine.Contains((Split-Path $taskBatch -Leaf))
})
$taskJobs = @(
    @{name='primary'; script='parallel_research.py'; extra=@('--queue', ('"' + (Join-Path $taskRoot 'configs\formal_v5_queue.json') + '"'), '--source-root', ('"' + $taskSource + '"'), '--workers', '6')},
    @{name='final'; script='final_evaluation_queue.py'; extra=@('--source-root', ('"' + $taskSource + '"'), '--version', 'formal_v5', '--workers', '2')},
    @{name='carla'; script='carla_formal_pipeline.py'; extra=@()},
    @{name='supplementary'; script='supplementary_pipeline.py'; extra=@('--source-root', ('"' + $taskSource + '"'), '--queue', ('"' + (Join-Path $taskRoot 'configs\supplementary_v5_queue.json') + '"'))},
    @{name='rule'; script='evaluate_rule_manifest.py'; extra=@('--source-root', ('"' + $taskSource + '"'))}
)
if (-not $Inspect) {
    if (Test-Path -LiteralPath (Join-Path $taskBatch 'STOP')) { throw 'STOP is present. Its cause must be resolved before resuming; this script never removes it.' }
    if ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() -ge $taskSpec.deadline_epoch - 180) { throw 'The original wall budget is exhausted or has fewer than 180 seconds. This script never extends it.' }
}
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
foreach ($taskJob in $taskJobs) {
    if ($taskJob.name -eq 'carla' -and @($taskProcesses | Where-Object { $_.CommandLine.Contains('carla_v5_development_finetune.py') }).Count -gt 0) {
        Write-Output 'carla: development owner active; it will resume the formal queue afterward'
        continue
    }
    $taskExisting = @($taskProcesses | Where-Object { $_.CommandLine.Contains($taskJob.script) })
    if ($taskExisting.Count -gt 0) {
        Write-Output ($taskJob.name + ': already running; PIDs ' + (($taskExisting.ProcessId) -join ', '))
        continue
    }
    $taskArguments = @(('"' + (Join-Path $PSScriptRoot $taskJob.script) + '"'), '--batch', ('"' + $taskBatch + '"')) + $taskJob.extra
    if ($Inspect) { Write-Output ($taskJob.name + ': stopped; resume command ' + $taskPython + ' ' + ($taskArguments -join ' ')); continue }
    $taskStamp = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArguments -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskBatch ('resume_' + $taskJob.name + '_' + $taskStamp + '.out')) -RedirectStandardError (Join-Path $taskBatch ('resume_' + $taskJob.name + '_' + $taskStamp + '.err'))
    Write-Output ($taskJob.name + ': launched PID ' + $taskProcess.Id)
}
