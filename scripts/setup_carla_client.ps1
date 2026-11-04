param([string]$Install='D:\CARLA_0.9.15\WindowsNoEditor')
$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$isolated=Join-Path $root '.carla-python37'
$runtime=Join-Path $isolated 'runtime'
$python=Join-Path $runtime 'python.exe'
$wheel=Join-Path $Install 'PythonAPI\carla\dist\carla-0.9.15-cp37-cp37m-win_amd64.whl'
if (-not (Test-Path -LiteralPath $wheel)) { throw 'The installed 0.9.15 cp37 Windows wheel is missing.' }
if (-not (Test-Path -LiteralPath $python)) {
    New-Item -ItemType Directory -Path $isolated -Force | Out-Null
    $archive=Join-Path $isolated 'python-3.7.9-embed-amd64.zip'
    if (-not (Test-Path -LiteralPath $archive)) { Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.7.9/python-3.7.9-embed-amd64.zip' -OutFile $archive }
    Expand-Archive -LiteralPath $archive -DestinationPath $runtime
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($wheel,(Join-Path $runtime 'Lib\site-packages'))
    @('python37.zip','.','Lib/site-packages','import site') | Set-Content -LiteralPath (Join-Path $runtime 'python37._pth') -Encoding ascii
}
$metadata=& $python -c 'import carla,sys,json;print(json.dumps({"python":sys.version,"executable":sys.executable,"carla_package":carla.__file__}))'
if ($LASTEXITCODE -ne 0) { throw 'Isolated CARLA client import failed.' }
@{runtime=($metadata | ConvertFrom-Json);local_wheel=$wheel;wheel_sha256=(Get-FileHash -LiteralPath $wheel -Algorithm SHA256).Hash;main_environment_modified=$false} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $isolated 'runtime_manifest.json') -Encoding utf8
Write-Output $metadata
