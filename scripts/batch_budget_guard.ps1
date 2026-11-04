param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference='Stop'
$batchPath=(Resolve-Path -LiteralPath $Batch).Path
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$config=Get-Content -LiteralPath (Join-Path $batchPath 'batch.json') -Raw | ConvertFrom-Json
$deadline=[DateTimeOffset]::FromUnixTimeSeconds([long]$config.deadline_epoch)
$record=@{pid=$PID;deadline_utc=$deadline.ToString('o');status='watching';started_utc=[DateTimeOffset]::UtcNow.ToString('o')}
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchPath 'budget_guard.json')
while ([DateTimeOffset]::UtcNow -lt $deadline.AddSeconds(-60)) { Start-Sleep -Seconds 5 }
'six_hour_batch_budget' | Set-Content -LiteralPath (Join-Path $batchPath 'STOP')
while ([DateTimeOffset]::UtcNow -lt $deadline) { Start-Sleep -Seconds 2 }
# Only stop the server started by this batch, and only after a read-only empty-world check.
$probe=Join-Path $batchPath 'infrastructure\budget_end_probe.json'
& (Join-Path $root '.carla-python37\runtime\python.exe') (Join-Path $root 'scripts\probe_carla37.py') --output $probe
$state=Get-Content -LiteralPath $probe -Raw | ConvertFrom-Json
$currentServer=Join-Path $batchPath 'infrastructure\server_current.json'
if (-not (Test-Path -LiteralPath $currentServer)) { $currentServer=Join-Path $batchPath 'infrastructure\server_start.json' }
$startup=Get-Content -LiteralPath $currentServer -Raw | ConvertFrom-Json
$launcher=Get-CimInstance Win32_Process -Filter ('ProcessId='+$startup.launcher_pid)
if ($state.status -eq 'CONNECTED' -and @($state.actors).Count -eq 0 -and $launcher -and $launcher.ExecutablePath -eq 'D:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe') {
    $children=Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $startup.launcher_pid -and $_.ExecutablePath -eq 'D:\CARLA_0.9.15\WindowsNoEditor\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe' }
    foreach ($child in $children) { Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue }
    Stop-Process -Id $startup.launcher_pid -ErrorAction SilentlyContinue
    $record.status='budget_reached_owned_idle_server_stopped'
} else { $record.status='budget_reached_server_retained_identity_or_active_actors_require_review' }
$record.finished_utc=[DateTimeOffset]::UtcNow.ToString('o')
$record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $batchPath 'budget_guard.json')
