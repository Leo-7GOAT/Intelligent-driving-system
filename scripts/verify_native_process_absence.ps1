param([Parameter(Mandatory=$true)][string]$Batch)
$ErrorActionPreference='Stop'
$taskBatch=(Resolve-Path -LiteralPath $Batch).Path
$taskOutput=Join-Path $taskBatch 'process_absence_receipt.json'
if(Test-Path -LiteralPath $taskOutput){throw 'Preserve existing absence receipt'}
if(Test-Path -LiteralPath (Join-Path $taskBatch 'guard_final.json')){throw 'Use the existing terminal guard evidence'}
$taskWorker=Get-Content -LiteralPath (Join-Path $taskBatch 'worker_process.json') -Raw | ConvertFrom-Json
$taskGuard=Get-Content -LiteralPath (Join-Path $taskBatch 'guard.json') -Raw | ConvertFrom-Json
$taskServer=Get-Content -LiteralPath (Join-Path $taskBatch 'infrastructure/server_start.json') -Raw | ConvertFrom-Json
$taskProtocol=Get-Content -LiteralPath (Join-Path $taskBatch 'protocol.json') -Raw | ConvertFrom-Json
if(-not $taskServer.owned_by_task -or $taskGuard.worker_pid -ne $taskWorker.pid){throw 'Ownership/worker records disagree'}
$taskIds=@{worker_pid=[int]$taskWorker.pid;guard_pid=[int]$taskGuard.pid;shipping_pid=[int]$taskServer.shipping_pid;launcher_pid=[int]$taskServer.launcher_pid}
foreach($taskId in $taskIds.Values){
 if($taskId -le 0){throw 'Invalid recorded process ID'}
 if(Get-CimInstance Win32_Process -Filter "ProcessId=$taskId"){throw 'A recorded PID is present; do not assume exit or terminate it'}
}
$taskPort=[int]$taskProtocol.carla_rpc_port
if($taskServer.rpc_port -ne $taskPort){throw 'Server/protocol ports disagree'}
$taskSpan=@($taskPort,($taskPort+1),($taskPort+2))
$taskBridges=@(Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object {
 $_.CommandLine -like '*carla_bridge37.py*' -and
 ($_.CommandLine.Contains($taskBatch) -or $_.CommandLine -match ('--port(?:\s+|=)'+$taskPort+'(?:\s|$)'))
})
if($taskBridges.Count){throw 'A related bridge remains present'}
$taskTcp=@(Get-NetTCPConnection | Where-Object {$taskSpan -contains $_.LocalPort})
$taskUdp=@(Get-NetUDPEndpoint | Where-Object {$taskSpan -contains $_.LocalPort})
if($taskTcp.Count -or $taskUdp.Count){throw 'The owned port span is occupied; preserve unrelated owners'}
$taskHashes=@{}
foreach($taskName in @('protocol.json','worker_process.json','guard.json','infrastructure/server_start.json')){
 $taskHashes[$taskName]=(Get-FileHash -LiteralPath (Join-Path $taskBatch $taskName) -Algorithm SHA256).Hash.ToLowerInvariant()
}
$taskReceipt=@{status='verified_worker_guard_server_absent';reason='external_process_loss_cause_unknown';
 observed_epoch=([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()/1000);process_ids=$taskIds;
 protocol_port_span=$taskSpan;tcp_endpoints=@();udp_endpoints=@();related_bridges=@();source_hashes=$taskHashes;
 scope='Read-only OS absence verification. No exit code, normal cleanup, interruption time or cause is inferred; no process is killed.'}
$taskJson=$taskReceipt | ConvertTo-Json -Depth 8
$taskBytes=[System.Text.Encoding]::UTF8.GetBytes($taskJson)
$taskStream=[System.IO.File]::Open($taskOutput,[System.IO.FileMode]::CreateNew)
try{$taskStream.Write($taskBytes,0,$taskBytes.Length)}finally{$taskStream.Dispose()}
$taskReceipt | ConvertTo-Json -Depth 8
