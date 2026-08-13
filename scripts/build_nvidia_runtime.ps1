param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$runtimeRoot = Join-Path $ProjectRoot "build\vendor-python312\runtime"
$marker = Join-Path $runtimeRoot ".chronophoto-nvidia-runtime-2.2.0"
$worker = Join-Path $ProjectRoot "src\chronophoto\gpu\nvidia_worker.py"

if (-not (Test-Path -LiteralPath $marker)) {
    $downloadRoot = Join-Path $ProjectRoot "build\vendor-python312"
    New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
    $pythonZip = Join-Path $downloadRoot "python.zip"
    $getPip = Join-Path $downloadRoot "get-pip.py"
    Invoke-WebRequest `
        -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" `
        -OutFile $pythonZip
    if (Test-Path -LiteralPath $runtimeRoot) {
        Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
    }
    Expand-Archive -LiteralPath $pythonZip -DestinationPath $runtimeRoot -Force
    $pathFile = Join-Path $runtimeRoot "python312._pth"
    (Get-Content -LiteralPath $pathFile) -replace '^#import site$', 'import site' |
        Set-Content -LiteralPath $pathFile -Encoding ascii
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip
    & (Join-Path $runtimeRoot "python.exe") $getPip
    & (Join-Path $runtimeRoot "python.exe") -m pip install `
        "PyNvVideoCodec==2.2.0" `
        "cupy-cuda12x==13.6.0" `
        "nvidia-cuda-runtime-cu12==12.9.79" `
        "nvidia-cuda-nvrtc-cu12==12.9.86"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    New-Item -ItemType File -Force -Path $marker | Out-Null
}

Copy-Item -LiteralPath $worker -Destination (Join-Path $runtimeRoot "nvidia_worker.py") -Force
$sitePackages = Join-Path $runtimeRoot "Lib\site-packages"
foreach ($path in @(
    (Join-Path $sitePackages "pip"),
    (Join-Path $sitePackages "pip-*.dist-info"),
    (Join-Path $sitePackages "samples"),
    (Join-Path $sitePackages "benchmarks"),
    (Join-Path $sitePackages "nvidia\cuda_nvrtc\bin\nvrtc64_120_0.alt.dll"),
    (Join-Path $runtimeRoot "Scripts")
)) {
    Get-Item -Path $path -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force
}
Get-ChildItem -LiteralPath $runtimeRoot -Directory -Filter "__pycache__" -Recurse |
    Remove-Item -Recurse -Force
Write-Output "Bundled NVIDIA runtime ready: $runtimeRoot"
