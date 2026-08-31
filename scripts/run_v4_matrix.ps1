param([string]$Output = "results/kernel_matrix.json")
$ErrorActionPreference = "Stop"
$exe = Join-Path (Get-Location) "build\Release\packed_swiglu_bench.exe"
if (!(Test-Path $exe)) { $exe = Join-Path (Get-Location) "build\packed_swiglu_bench.exe" }
if (!(Test-Path $exe)) { throw "Build first with scripts/build_bench.ps1" }
$records = @()
foreach ($dtype in @("fp16", "bf16", "fp32")) {
  foreach ($tokens in @(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)) {
    $line = & $exe --tokens $tokens --intermediate 8960 --warmup 200 --iters 200 --dtype $dtype
    $records += ($line | ConvertFrom-Json)
  }
}
$records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $Output -Encoding UTF8
Get-Content -LiteralPath $Output
