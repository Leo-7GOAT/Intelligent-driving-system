param([Parameter(Mandatory=$true)][string]$Batch,[ValidateRange(0,30)][int]$BridgeDrainSeconds=0)
$ErrorActionPreference = 'Stop'
$batchPath = (Resolve-Path -LiteralPath $Batch).Path
$projectPath = Split-Path -Parent $PSScriptRoot
$record = Get-Content -LiteralPath (Join-Path $batchPath 'infrastructure/server_start.json') -Raw | ConvertFrom-Json
if (-not $record.owned_by_task) { throw 'Server ownership not established' }
$stopPath = Join-Path $batchPath 'infrastructure/server_stop.json'
if (Test-Path -LiteralPath $stopPath) { throw 'Immutable server stop receipt already exists' }
$rpcHost = if ($record.PSObject.Properties.Name -contains 'rpc_host') { [string]$record.rpc_host } else { '127.0.0.1' }
$rpcPort = if ($record.PSObject.Properties.Name -contains 'rpc_port') { [int]$record.rpc_port } else { 2000 }
function Get-RelevantBridges {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -like '*carla_bridge37.py*' -and
        ($_.CommandLine.Contains($batchPath) -or
         $_.CommandLine -match ('--port(?:\s+|=)' + $rpcPort + '(?:\s|$)') -or
         ($rpcPort -eq 2000 -and $_.CommandLine -notmatch '--port'))
    })
}
$bridges = @(Get-RelevantBridges)
$drainDeadline = [DateTime]::UtcNow.AddSeconds($BridgeDrainSeconds)
while ($bridges.Count -gt 0 -and [DateTime]::UtcNow -lt $drainDeadline) {
    Start-Sleep -Seconds 2
    $bridges = @(Get-RelevantBridges)
}
if ($bridges.Count -gt 0) { throw 'A bridge is still live; refusing to stop server' }
$probePath = Join-Path $batchPath ('infrastructure/stop_probe_'+[DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')+'.json')
$clientCandidates = @($env:NAVSTACK_CARLA_PYTHON, $record.client_python)
$ancestor = $batchPath
while ($ancestor) {
    $clientCandidates += Join-Path $ancestor '.carla-python37/runtime/python.exe'
    $ancestor = Split-Path -Parent $ancestor
}
$clientCandidates += Join-Path $projectPath '.carla-python37/runtime/python.exe'
$clientPython = $clientCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } | Select-Object -First 1
if (-not $clientPython) { throw 'Isolated CARLA client interpreter is missing' }
if (-not (Test-Path -LiteralPath $clientPython -PathType Leaf)) { throw 'Isolated CARLA client interpreter is missing' }
& $clientPython (Join-Path $PSScriptRoot 'probe_carla37.py') --host $rpcHost --port $rpcPort --output $probePath
if ($LASTEXITCODE -ne 0) { throw 'Read-only probe failed' }
$probe = Get-Content -LiteralPath $probePath -Raw | ConvertFrom-Json
if ($probe.status -ne 'CONNECTED' -or $probe.server_version -ne '0.9.15' -or @($probe.actors).Count -ne 0) { throw 'Server is not verified empty and matching' }
$expectedExe = if ($record.PSObject.Properties.Name -contains 'launched_executable') {
    [System.IO.Path]::GetFullPath([string]$record.launched_executable)
} else {
    [System.IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $record.executable) 'CarlaUE4/Binaries/Win64/CarlaUE4-Win64-Shipping.exe'))
}
$startedText = if ($record.PSObject.Properties.Name -contains 'shipping_creation_utc') { $record.shipping_creation_utc } else { $record.started_utc }
$started = [DateTimeOffset]::Parse([string]$startedText)
$candidates = if ($record.PSObject.Properties.Name -contains 'shipping_pid') {
    @(Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$record.shipping_pid)" -ErrorAction SilentlyContinue)
} else {
    @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $record.launcher_pid })
}
$owned = @($candidates | Where-Object {
    $_.ParentProcessId -eq $record.launcher_pid -and
    [System.IO.Path]::GetFullPath($_.ExecutablePath) -eq $expectedExe -and
    ([DateTimeOffset]$_.CreationDate) -ge $started.AddSeconds(-2) -and
    ([DateTimeOffset]$_.CreationDate) -le $started.AddSeconds(2) -and
    $_.CommandLine -like "*-carla-rpc-port=$rpcPort*"
})
if ($owned.Count -ne 1) { throw 'Server PID, executable, parent or creation time identity ambiguous' }
$shippingPid = [int]$owned[0].ProcessId
Stop-Process -Id $shippingPid
$exitDeadline = [DateTimeOffset]::UtcNow.AddSeconds(20)
do {
    $remaining = Get-CimInstance Win32_Process -Filter "ProcessId=$shippingPid" -ErrorAction SilentlyContinue
    if ($null -eq $remaining) { break }
    Start-Sleep -Milliseconds 250
} while ([DateTimeOffset]::UtcNow -lt $exitDeadline)
if ($null -ne (Get-CimInstance Win32_Process -Filter "ProcessId=$shippingPid" -ErrorAction SilentlyContinue)) {
    throw 'Verified CARLA Shipping process did not exit after stop request'
}
$serverExitPath = Join-Path $batchPath 'infrastructure/server_exit.json'
if ($record.PSObject.Properties.Name -contains 'foreground_host') {
    $receiptDeadline = [DateTimeOffset]::UtcNow.AddSeconds(10)
    $exitReceipt = $null
    do {
        try { $exitReceipt = Get-Content -LiteralPath $serverExitPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop } catch { $exitReceipt = $null }
        if ($null -ne $exitReceipt) { break }
        Start-Sleep -Milliseconds 250
    } while ([DateTimeOffset]::UtcNow -lt $receiptDeadline)
    if ($null -eq $exitReceipt -or $exitReceipt.shipping_pid -ne $shippingPid -or
        $null -eq $exitReceipt.exit_code -or -not $exitReceipt.ended_utc) {
        throw 'Foreground host did not record a matching complete server exit'
    }
    [void][DateTimeOffset]::Parse([string]$exitReceipt.ended_utc)
}
$span = if ($record.PSObject.Properties.Name -contains 'rpc_port_span') { @($record.rpc_port_span) } else { @($rpcPort..($rpcPort + 2)) }
$postTcp = @($span | ForEach-Object { Get-NetTCPConnection -LocalPort $_ -ErrorAction SilentlyContinue } |
    Select-Object LocalAddress,LocalPort,State,OwningProcess)
$postUdp = @($span | ForEach-Object { Get-NetUDPEndpoint -LocalPort $_ -ErrorAction SilentlyContinue } |
    Select-Object LocalAddress,LocalPort,OwningProcess)
$temporary = "$stopPath.tmp-$PID-$([Guid]::NewGuid().ToString('N'))"
@{status='owned_empty_server_stopped';shipping_pid=$shippingPid;epoch=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds();
  probe=$probePath;server_exit=$serverExitPath;post_stop_tcp=$postTcp;post_stop_udp=$postUdp} |
    ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
Move-Item -LiteralPath $temporary -Destination $stopPath
