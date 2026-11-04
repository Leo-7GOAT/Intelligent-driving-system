param(
    [string]$Install = 'D:\CARLA_0.9.15\WindowsNoEditor',
    [int]$RpcPort = 2000,
    [switch]$RenderOffScreen,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
if ($RpcPort -lt 1024 -or $RpcPort -gt 65533) { throw 'CARLA RPC base port must be in 1024..65533' }
$rpcPorts = @($RpcPort..($RpcPort + 2))
$exe = Join-Path $Install 'CarlaUE4.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing installed server: $exe" }
$existing = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^CarlaUE4' })
$tcpOwners = @($rpcPorts | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue })
$udpOwners = @($rpcPorts | ForEach-Object { Get-NetUDPEndpoint -LocalPort $_ -ErrorAction SilentlyContinue })
if ($existing.Count -gt 0 -or $tcpOwners.Count -gt 0 -or $udpOwners.Count -gt 0) {
    @{status='EXISTING_SERVER_OR_PORT_OWNER_REQUIRES_IDENTITY_CHECK';processes=$existing | Select-Object ProcessId,ExecutablePath,CommandLine;rpc_port_span=$rpcPorts;tcp=$tcpOwners | Select-Object LocalAddress,LocalPort,State,OwningProcess;udp=$udpOwners | Select-Object LocalAddress,LocalPort,OwningProcess} |
        ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'server_existing.json') -Encoding utf8
    throw 'Existing CARLA process or TCP/UDP port owner detected; no duplicate server started.'
}
$logPath = Join-Path ([System.IO.Path]::GetTempPath()) ('codex_carla_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')+'.log')
$serverArgs = @('-quality-level=Low','-windowed','-ResX=640','-ResY=360',"-carla-rpc-port=$RpcPort",'-nosplash','-unattended','-stdout','-FullStdOutLogOutput',"-abslog=`"$logPath`"")
if ($RenderOffScreen) { $serverArgs += '-RenderOffScreen' }
$process = Start-Process -FilePath $exe -ArgumentList $serverArgs -WorkingDirectory $Install -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $resolvedOutput 'server_stdout.log') -RedirectStandardError (Join-Path $resolvedOutput 'server_stderr.log')
@{status='STARTED_NOT_YET_READY';started_utc=[DateTime]::UtcNow.ToString('o');launcher_pid=$process.Id;executable=$exe;arguments=$serverArgs;rpc_host='127.0.0.1';rpc_port=$RpcPort;rpc_port_span=$rpcPorts;log=$logPath;owned_by_task=$true} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'server_start.json') -Encoding utf8
Write-Output "Started installed CARLA launcher PID $($process.Id), RPC $RpcPort; verify RPC readiness separately."
