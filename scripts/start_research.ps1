param([Parameter(Mandatory=$true)][string]$Batch,[string]$Queue='configs/development_queue.json',[string]$SourceRoot='')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$batchPath=(Resolve-Path -LiteralPath $Batch).Path
if (Test-Path -LiteralPath (Join-Path $batchPath 'STOP')) { throw 'STOP exists. Inspect the reason and budget before explicitly removing it to resume.' }
$python=Join-Path $root '.venv\Scripts\python.exe'
$arguments=@(('"'+(Join-Path $root 'scripts\batch_watchdog.py')+'"'),'--batch',('"'+$batchPath+'"'),'--queue',('"'+(Resolve-Path -LiteralPath $Queue).Path+'"'))
if ($SourceRoot) { $arguments+=@('--source-root',('"'+(Resolve-Path -LiteralPath $SourceRoot).Path+'"')) }
$process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
Write-Output ('watchdog PID: '+$process.Id)
