param(
    [string]$Install = 'D:\CARLA_0.9.15\WindowsNoEditor',
    [int]$RpcPort = 2000,
    [switch]$RenderOffScreen,
    [switch]$NullRHI,
    [switch]$DirectShipping,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
if ($RpcPort -lt 1024 -or $RpcPort -gt 65533) { throw 'CARLA RPC base port must be in 1024..65533' }
$rpcPorts = @($RpcPort..($RpcPort + 2))
$exe = if ($DirectShipping) { Join-Path $Install 'CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe' } else { Join-Path $Install 'CarlaUE4.exe' }
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing installed server: $exe" }
$existing = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^CarlaUE4' })
$tcpOwners = @($rpcPorts | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue })
$udpOwners = @($rpcPorts | ForEach-Object { Get-NetUDPEndpoint -LocalPort $_ -ErrorAction SilentlyContinue })
if ($existing.Count -or $tcpOwners.Count -or $udpOwners.Count) { throw 'Diagnostic refuses an existing CARLA process or TCP/UDP port owner' }
$serverArgs = @()
if ($DirectShipping) { $serverArgs += 'CarlaUE4' }
$serverArgs += @('-quality-level=Low','-windowed','-ResX=640','-ResY=360',"-carla-rpc-port=$RpcPort",'-nosplash','-unattended','-stdout','-FullStdOutLogOutput')
if ($RenderOffScreen) { $serverArgs += '-RenderOffScreen' }
if ($NullRHI) { $serverArgs += '-NullRHI' }
$stdout = Join-Path $resolvedOutput 'diagnostic_stdout.log'
$stderr = Join-Path $resolvedOutput 'diagnostic_stderr.log'
$started = [DateTimeOffset]::UtcNow
$process = Start-Process -FilePath $exe -ArgumentList $serverArgs -WorkingDirectory $Install -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
@{status='STARTED_DIAGNOSTIC';started_utc=$started.ToString('o');pid=$process.Id;executable=$exe;arguments=$serverArgs;rpc_host='127.0.0.1';rpc_port=$RpcPort;rpc_port_span=$rpcPorts} |
    ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'diagnostic_start.json') -Encoding utf8
$process.WaitForExit()
@{
    status = 'PROCESS_EXITED_BEFORE_RPC_READINESS'
    started_utc = $started.ToString('o')
    ended_utc = [DateTimeOffset]::UtcNow.ToString('o')
    pid = $process.Id
    exit_code = $process.ExitCode
    executable = $exe
    arguments = $serverArgs
    rpc_host = '127.0.0.1'
    rpc_port = $RpcPort
    rpc_port_span = $rpcPorts
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutput 'diagnostic_exit.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $resolvedOutput 'diagnostic_exit.json') -Raw
