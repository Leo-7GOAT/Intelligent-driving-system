param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference='Stop'
$batchPath=(Resolve-Path -LiteralPath $Batch).Path
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$lockPath=Join-Path $batchPath 'carla_tick_owner.lock'
if (-not (Test-Path -LiteralPath $lockPath)) { Write-Output 'No stale lock to recover'; exit 0 }
$ownerId=[int](Get-Content -LiteralPath $lockPath -Raw).Trim()
if (Get-CimInstance Win32_Process -Filter ('ProcessId='+$ownerId)) { throw 'Recorded owner PID still exists; refusing lock recovery' }
$bridges=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*carla_bridge37.py*' })
if ($bridges.Count -ne 0) { throw 'A CARLA bridge process exists; refusing lock recovery' }
$stamp=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$probe=Join-Path $batchPath ('infrastructure/stale_lock_probe_'+$stamp+'.json')
& (Join-Path $root '.carla-python37/runtime/python.exe') (Join-Path $root 'scripts/probe_carla37.py') --output $probe
if ($LASTEXITCODE -ne 0) { throw 'Read-only server probe failed' }
$state=Get-Content -LiteralPath $probe -Raw | ConvertFrom-Json
if ($state.status -ne 'CONNECTED' -or $state.client_version -ne '0.9.15' -or $state.server_version -ne '0.9.15' -or @($state.actors).Count -ne 0) { throw 'Server is not verified empty with matching 0.9.15; refusing recovery' }
$archive=Join-Path $batchPath ('infrastructure/stale_tick_owner_'+$ownerId+'_'+$stamp+'.lock')
Move-Item -LiteralPath $lockPath -Destination $archive
$receipt=@{owner_pid=$ownerId;owner_process_absent=$true;bridge_processes_absent=$true;probe=$probe;archived_lock=$archive;recovered_utc=[DateTimeOffset]::UtcNow.ToString('o');note='Only an absent owner plus verified empty matching world permits this recovery. Original lock and failed experiments preserved. No new retry or training is started.'}
$json=$receipt | ConvertTo-Json -Depth 4
[IO.File]::WriteAllText((Join-Path $batchPath 'infrastructure/stale_lock_recovery.json'),$json,(New-Object Text.UTF8Encoding($false)))
Write-Output $json
