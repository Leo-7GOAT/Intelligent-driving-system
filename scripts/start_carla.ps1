param(
    [string]$Install = 'D:\CARLA_0.9.15\WindowsNoEditor',
    [int]$RpcPort = 2000,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
$exe = Join-Path $Install 'CarlaUE4.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing installed server: $exe" }
$existing = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^CarlaUE4' })
$listeners = @(Get-NetTCPConnection -LocalPort $RpcPort -State Listen -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0 -or $listeners.Count -gt 0) {
    @{status='EXISTING_SERVER_REQUIRES_IDENTITY_CHECK';processes=$existing | Select-Object ProcessId,ExecutablePath,CommandLine;listeners=$listeners | Select-Object LocalPort,OwningProcess} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'server_existing.json') -Encoding utf8
    throw 'Existing CARLA process or port listener detected; no duplicate server started.'
}
$logPath = Join-Path ([System.IO.Path]::GetTempPath()) ('codex_carla_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')+'.log')
$serverArgs = @('-quality-level=Low','-windowed','-ResX=640','-ResY=360',"-carla-rpc-port=$RpcPort",'-nosplash','-unattended','-stdout','-FullStdOutLogOutput',"-abslog=`"$logPath`"")
$process = Start-Process -FilePath $exe -ArgumentList $serverArgs -WorkingDirectory $Install -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resolvedOutput 'server_stdout.log') -RedirectStandardError (Join-Path $resolvedOutput 'server_stderr.log')
@{status='STARTED_NOT_YET_READY';started_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$process.Id;executable=$exe;arguments=$serverArgs;rpc_port=$RpcPort;log=$logPath;owned_by_task=$true} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'server_start.json') -Encoding utf8
Write-Output "Started installed CARLA launcher PID $($process.Id), RPC $RpcPort; verify RPC readiness separately."
