param(
  [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
  [string]$ModelCache = "..\vllm-serving-lab\.cache\huggingface",
  [string]$OutputDir = "results\service_matrix",
  [string]$StockImage = "vllm/vllm-openai:v0.10.2",
  [string]$FusedImage = "vllm-fused-swiglu:v0.1",
  [int]$Rounds = 2,
  [int]$MaxModelLen = 1536,
  [int]$MaxNumSeqs = 8
)
$ErrorActionPreference = "Stop"

function Invoke-Docker([string[]]$DockerArgs) {
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) { throw "docker failed: $($DockerArgs -join ' ')" }
}

function Wait-Health([int]$Port) {
  for ($i = 0; $i -lt 120; $i++) {
    try {
      $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
      if ($response.StatusCode -eq 200) { return }
    } catch { }
    Start-Sleep -Seconds 2
  }
  throw "vLLM service on port $Port did not become healthy"
}

$cases = @(
  [pscustomobject]@{ Name="decode_t1";  Prompt=16;  Output=96; Concurrency=1; Requests=8 },
  [pscustomobject]@{ Name="decode_t4";  Prompt=16;  Output=96; Concurrency=4; Requests=16 },
  [pscustomobject]@{ Name="decode_t8";  Prompt=16;  Output=96; Concurrency=8; Requests=24 },
  [pscustomobject]@{ Name="prefill_small";  Prompt=64;  Output=8; Concurrency=1; Requests=8 },
  [pscustomobject]@{ Name="prefill_medium"; Prompt=256; Output=8; Concurrency=4; Requests=16 },
  [pscustomobject]@{ Name="prefill_large";  Prompt=768; Output=8; Concurrency=4; Requests=12 },
  [pscustomobject]@{ Name="mixed"; Prompt=256; Output=64; Concurrency=4; Requests=16 }
)

$projectRoot = (Get-Location).Path
$cachePath = (Resolve-Path (Join-Path $projectRoot $ModelCache)).Path
$outputPath = Join-Path $projectRoot $OutputDir
$rawPath = Join-Path $outputPath "raw"
$containerRawPath = (($OutputDir -replace '\\', '/') + "/raw")
New-Item -ItemType Directory -Force -Path $rawPath | Out-Null
$containerName = "vllm-swiglu-matrix"
$mount = "${cachePath}:/root/.cache/huggingface"
$common = @("--model", $Model, "--max-model-len", "$MaxModelLen", "--max-num-seqs", "$MaxNumSeqs", "--gpu-memory-utilization", "0.55")

for ($round = 0; $round -lt $Rounds; $round++) {
  $implementations = if ($round % 2 -eq 0) { @("stock", "fused") } else { @("fused", "stock") }
  foreach ($implementation in $implementations) {
    $existing = & docker ps -a --format "{{.Names}}"
    if ($existing -contains $containerName) { Invoke-Docker @("rm", "-f", $containerName) | Out-Null }
    $image = if ($implementation -eq "stock") { $StockImage } else { $FusedImage }
    $envArgs = if ($implementation -eq "fused") { @("-e", "VLLM_USE_FUSED_SILU_MUL=1") } else { @() }
    Invoke-Docker (@("run", "-d", "--name", $containerName, "--gpus", "all", "-p", "8000:8000", "-v", $mount, "-e", "HF_HUB_OFFLINE=1") + $envArgs + @($image) + $common) | Out-Null
    Wait-Health 8000

    $reverseCases = (($round % 2 -eq 1) -xor ($implementation -eq "fused"))
    $orderedCases = @($cases)
    if ($reverseCases) { [array]::Reverse($orderedCases) }
    foreach ($case in $orderedCases) {
      $fileName = "round{0:D2}_{1}_{2}.json" -f ($round + 1), $implementation, $case.Name
      Invoke-Docker @(
        "run", "--rm", "--network", "host", "-v", "${projectRoot}:/workspace",
        "-e", "PYTHONPATH=/workspace/python", "--entrypoint", "python3", $StockImage,
        "-m", "vllm_fused_swiglu_lab.service_benchmark",
        "--base-url", "http://127.0.0.1:8000", "--model", $Model,
        "--implementation", $implementation,
        "--concurrency", "$($case.Concurrency)", "--requests", "$($case.Requests)",
        "--prompt-tokens", "$($case.Prompt)", "--output-tokens", "$($case.Output)",
        "--warmup-requests", "1", "--output", "/workspace/$containerRawPath/$fileName"
      ) | Out-Null
      $filePath = Join-Path $rawPath $fileName
      $payload = Get-Content -Raw $filePath | ConvertFrom-Json
      $payload | Add-Member -NotePropertyName case -NotePropertyValue $case.Name
      $payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $filePath
      Write-Host ("round={0} implementation={1} case={2} ttft={3:N2}ms throughput={4:N2}tok/s" -f ($round + 1), $implementation, $case.Name, $payload.summary.ttft_p50_ms, $payload.summary.output_tok_s)
    }
    Invoke-Docker @("rm", "-f", $containerName) | Out-Null
  }
}

python scripts/summarize_service_matrix.py --input-dir $rawPath --output (Join-Path $outputPath "summary.json")
