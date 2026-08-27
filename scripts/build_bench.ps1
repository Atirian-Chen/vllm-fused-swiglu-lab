param([string]$BuildDir = "build")
$ErrorActionPreference = "Stop"
$source = Join-Path (Get-Location) "csrc\packed_swiglu_bench.cu"
$out = Join-Path (Get-Location) "$BuildDir\packed_swiglu_bench.exe"
$asciiRoot = Join-Path $env:TEMP "vllm_fused_swiglu_lab_build"
$asciiOut = Join-Path $asciiRoot "packed_swiglu_bench.exe"
New-Item -ItemType Directory -Force -Path (Split-Path $out),$asciiRoot | Out-Null
$nvcc = (Get-Command nvcc.exe -ErrorAction Stop).Source
& $nvcc -O3 -lineinfo -std=c++17 $source -o $asciiOut
if ($LASTEXITCODE -ne 0) { throw "NVCC build failed with exit code $LASTEXITCODE" }
Copy-Item -LiteralPath $asciiOut -Destination $out -Force
Write-Output "Built $BuildDir\packed_swiglu_bench.exe"
