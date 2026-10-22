param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference = 'Stop'
$batchPath = (Resolve-Path -LiteralPath $Batch).Path
$projectPath = Split-Path -Parent $PSScriptRoot
$record = Get-Content -LiteralPath (Join-Path $batchPath 'infrastructure/server_start.json') -Raw | ConvertFrom-Json
if (-not $record.owned_by_task) { throw 'Server ownership not established' }
$bridges = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*carla_bridge37.py*' })
if ($bridges.Count -gt 0) { throw 'A bridge is still live; refusing to stop server' }
$probePath = Join-Path $batchPath ('infrastructure/stop_probe_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')+'.json')
$clientPython = if ($env:NAVSTACK_CARLA_PYTHON) { $env:NAVSTACK_CARLA_PYTHON } else { Join-Path $projectPath '.carla-python37/runtime/python.exe' }
if (-not (Test-Path -LiteralPath $clientPython -PathType Leaf)) { throw 'Isolated CARLA client interpreter is missing' }
& $clientPython (Join-Path $PSScriptRoot 'probe_carla37.py') --output $probePath
if ($LASTEXITCODE -ne 0) { throw 'Read-only probe failed' }
$probe = Get-Content -LiteralPath $probePath -Raw | ConvertFrom-Json
if ($probe.status -ne 'CONNECTED' -or $probe.server_version -ne '0.9.15' -or @($probe.actors).Count -ne 0) { throw 'Server is not verified empty and matching' }
$expectedExe = [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $record.executable) 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'))
$started = if ($record.started_utc -is [DateTime]) { [DateTimeOffset]$record.started_utc } else { [DateTimeOffset]::Parse($record.started_utc) }
$owned = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ParentProcessId -eq $record.launcher_pid -and $_.ExecutablePath -eq $expectedExe -and
    ([DateTimeOffset]$_.CreationDate) -ge $started.AddSeconds(-5) -and ([DateTimeOffset]$_.CreationDate) -le $started.AddMinutes(2)
})
if ($owned.Count -ne 1) { throw 'Server PID, executable, parent or creation time identity ambiguous' }
Stop-Process -Id $owned[0].ProcessId
@{status='owned_empty_server_stopped';shipping_pid=$owned[0].ProcessId;epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();probe=$probePath} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchPath 'infrastructure/server_stop.json') -Encoding utf8
