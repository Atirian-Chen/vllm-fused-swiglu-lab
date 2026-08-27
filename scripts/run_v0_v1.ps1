param(
  [int]$Tokens = 128,
  [int]$Intermediate = 8960,
  [int]$Warmup = 200,
  [int]$Iters = 200,
  [string]$Dtype = "fp16",
  [string]$Output = "results/kernel.json"
)
$ErrorActionPreference = "Stop"
$exe = Join-Path (Get-Location) "build\Release\packed_swiglu_bench.exe"
if (!(Test-Path $exe)) { $exe = Join-Path (Get-Location) "build\packed_swiglu_bench.exe" }
if (!(Test-Path $exe)) { throw "Build first with scripts/build_bench.ps1" }
$json = & $exe --tokens $Tokens --intermediate $Intermediate --warmup $Warmup --iters $Iters --dtype $Dtype
$json | Set-Content -Encoding UTF8 $Output
$json

