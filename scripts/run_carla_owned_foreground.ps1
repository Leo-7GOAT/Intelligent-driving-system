param(
    [string]$Install = 'D:\CARLA_0.9.15\WindowsNoEditor',
    [int]$RpcPort = 2000,
    [switch]$RenderOffScreen,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw 'This owned foreground launcher requires PowerShell 7 (pwsh) for ProcessStartInfo.ArgumentList.'
}

function Write-AtomicJson([string]$Path,[object]$Value) {
    $temporary = "$Path.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
    $Value | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path
}

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
$evidenceNames = @('server_start.claim','server_start.json','server_start_failure.json','server_exit.json','server_stdout.log','server_stderr.log')
$existingEvidence = @($evidenceNames | Where-Object { Test-Path -LiteralPath (Join-Path $resolvedOutput $_) })
if ($existingEvidence.Count) { throw ('Startup evidence already exists; use a new immutable batch: ' + ($existingEvidence -join ', ')) }
$claimPath = Join-Path $resolvedOutput 'server_start.claim'
$claimStream = [System.IO.File]::Open($claimPath,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::None)
try {
    $claimBytes = [System.Text.Encoding]::UTF8.GetBytes("host_pid=$PID`nclaimed_utc=$([DateTimeOffset]::UtcNow.ToString('o'))`n")
    $claimStream.Write($claimBytes,0,$claimBytes.Length)
    $claimStream.Flush($true)
} finally {
    $claimStream.Dispose()
}
if ($RpcPort -lt 1024 -or $RpcPort -gt 65533) { throw 'CARLA RPC base port must be in 1024..65533' }
$rpcPorts = @($RpcPort..($RpcPort + 2))
$wrapper = Join-Path $Install 'CarlaUE4.exe'
$shipping = Join-Path $Install 'CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe'
if (-not (Test-Path -LiteralPath $wrapper -PathType Leaf) -or -not (Test-Path -LiteralPath $shipping -PathType Leaf)) {
    throw 'Installed CARLA wrapper or shipping executable is missing'
}
$existing = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^CarlaUE4' })
$tcpOwners = @($rpcPorts | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue } | Where-Object { $_.OwningProcess -ne 0 })
$udpOwners = @($rpcPorts | ForEach-Object { Get-NetUDPEndpoint -LocalPort $_ -ErrorAction SilentlyContinue })
if ($existing.Count -or $tcpOwners.Count -or $udpOwners.Count) { throw 'Existing CARLA process or TCP/UDP port owner detected' }
$stdout = Join-Path $resolvedOutput 'server_stdout.log'
$stderr = Join-Path $resolvedOutput 'server_stderr.log'
$serverArgs = @('CarlaUE4','-quality-level=Low','-windowed','-ResX=640','-ResY=360',"-carla-rpc-port=$RpcPort",'-nosplash','-unattended','-stdout','-FullStdOutLogOutput')
if ($RenderOffScreen) { $serverArgs += '-RenderOffScreen' }
$started = [DateTimeOffset]::UtcNow
$process = $null
$stdoutFile = $null
$stderrFile = $null
$serverExit = $null
try {
    # Open evidence sinks before creating the process. Any failure before
    # Start() therefore cannot leave an unowned simulator behind.
    $stdoutFile = [System.IO.File]::Open($stdout,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    $stderrFile = [System.IO.File]::Open($stderr,[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $shipping
    $startInfo.WorkingDirectory = $Install
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in $serverArgs) { [void]$startInfo.ArgumentList.Add($argument) }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    if (-not $process.Start()) { throw 'CARLA Shipping process did not start' }
    $shippingStarted = [DateTimeOffset]$process.StartTime.ToUniversalTime()
    $shippingIdentity = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
    $hostIdentity = Get-CimInstance Win32_Process -Filter "ProcessId=$PID"
    if ($null -eq $shippingIdentity -or $shippingIdentity.ParentProcessId -ne $PID -or
        [System.IO.Path]::GetFullPath($shippingIdentity.ExecutablePath) -ne [System.IO.Path]::GetFullPath($shipping)) {
        throw 'Started CARLA process identity could not be established'
    }
    Write-AtomicJson (Join-Path $resolvedOutput 'server_start.json') @{
        status='STARTED_NOT_YET_READY';started_utc=$started.ToString('o');launcher_pid=$PID
        launcher_creation_utc=([DateTimeOffset]$hostIdentity.CreationDate).ToUniversalTime().ToString('o')
        launcher_executable=$hostIdentity.ExecutablePath;executable=$wrapper;launched_executable=$shipping
        shipping_pid=$process.Id;shipping_parent_pid=$shippingIdentity.ParentProcessId
        shipping_creation_utc=$shippingStarted.ToString('o');shipping_command_line=$shippingIdentity.CommandLine
        arguments=$serverArgs;rpc_host='127.0.0.1';rpc_port=$RpcPort;rpc_port_span=$rpcPorts
        client_python=$env:NAVSTACK_CARLA_PYTHON
        stdout=$stdout;stderr=$stderr;owned_by_task=$true;foreground_host=$true
    }
    $stdoutCopy = $process.StandardOutput.BaseStream.CopyToAsync($stdoutFile)
    $stderrCopy = $process.StandardError.BaseStream.CopyToAsync($stderrFile)
    $process.WaitForExit()
    [System.Threading.Tasks.Task]::WaitAll([System.Threading.Tasks.Task[]]@($stdoutCopy,$stderrCopy))
    $serverExit = $process.ExitCode
} catch {
    $failure = $_
    if ($null -ne $process) {
        try {
            if (-not $process.HasExited) { $process.Kill();$process.WaitForExit() }
            $serverExit = $process.ExitCode
        } catch {}
    }
    Write-AtomicJson (Join-Path $resolvedOutput 'server_start_failure.json') @{
        status='STARTUP_HOST_FAILURE';epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
        launcher_pid=$PID;shipping_pid=if($null -ne $process){$process.Id}else{$null}
        error_type=$failure.Exception.GetType().FullName;error=$failure.Exception.Message
    }
    if (-not (Test-Path -LiteralPath (Join-Path $resolvedOutput 'server_exit.json'))) {
        Write-AtomicJson (Join-Path $resolvedOutput 'server_exit.json') @{
            status='SERVER_PROCESS_ABORTED_BY_HOST';launcher_pid=$PID
            shipping_pid=if($null -ne $process){$process.Id}else{$null}
            exit_code=$serverExit;ended_utc=[DateTimeOffset]::UtcNow.ToString('o')
        }
    }
    throw
} finally {
    if ($null -ne $stdoutFile) { $stdoutFile.Dispose() }
    if ($null -ne $stderrFile) { $stderrFile.Dispose() }
}
Write-AtomicJson (Join-Path $resolvedOutput 'server_exit.json') @{
    status='SERVER_PROCESS_EXITED';launcher_pid=$PID;shipping_pid=$process.Id
    exit_code=$serverExit;ended_utc=[DateTimeOffset]::UtcNow.ToString('o')
}
exit $serverExit
