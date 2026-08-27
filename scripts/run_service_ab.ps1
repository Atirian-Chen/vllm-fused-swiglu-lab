param(
  [string]$Model = "Qwen/Qwen2.5-1.5B-Instruct",
  [int]$Concurrency = 8,
  [int]$Requests = 60,
  [int]$OutputTokens = 64,
  [string]$ModelCache = "..\vllm-serving-lab\.cache\huggingface",
  [string]$OutputDir = "results",
  [string]$StockImage = "vllm/vllm-openai:v0.10.2",
  [string]$FusedImage = "vllm-fused-swiglu:v0.1",
  [int]$MaxModelLen = 1024,
  [int]$MaxNumSeqs = 8,
  [switch]$KeepContainers
)
$ErrorActionPreference = "Stop"

function Invoke-Docker([string[]]$DockerArgs) {
  & docker @DockerArgs
  if ($LASTEXITCODE -ne 0) { throw "docker failed: $($DockerArgs -join ' ')" }
}

function Wait-Health([int]$Port) {
  for ($i = 0; $i -lt 90; $i++) {
    try {
      $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -UseBasicParsing -TimeoutSec 3
      if ($response.StatusCode -eq 200) { return }
    } catch { }
    Start-Sleep -Seconds 2
  }
  throw "vLLM service on port $Port did not become healthy"
}

$projectRoot = (Get-Location).Path
$cachePath = (Resolve-Path (Join-Path $projectRoot $ModelCache)).Path
$outputPath = Join-Path $projectRoot $OutputDir
$containerOutputDir = $OutputDir -replace '\\', '/'
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

foreach ($name in @("vllm-stock-ab", "vllm-fused-ab")) {
  $existing = & docker ps -a --format "{{.Names}}"
  if ($existing -contains $name) { Invoke-Docker @("rm", "-f", $name) | Out-Null }
}

$common = @("--model", $Model, "--max-model-len", "$MaxModelLen", "--max-num-seqs", "$MaxNumSeqs", "--gpu-memory-utilization", "0.55")
$mount = "${cachePath}:/root/.cache/huggingface"

Invoke-Docker (@("run", "-d", "--name", "vllm-stock-ab", "--gpus", "all", "-p", "8000:8000", "-v", $mount, "-e", "HF_HUB_OFFLINE=1", $StockImage) + $common)
Wait-Health 8000
Invoke-Docker @("run", "--rm", "--network", "host", "-v", "${projectRoot}:/workspace", "-e", "PYTHONPATH=/workspace/python", "--entrypoint", "python3", $StockImage, "-m", "vllm_fused_swiglu_lab.service_benchmark", "--base-url", "http://127.0.0.1:8000", "--model", $Model, "--implementation", "stock", "--concurrency", "$Concurrency", "--requests", "$Requests", "--output-tokens", "$OutputTokens", "--output", "/workspace/$containerOutputDir/service_stock.json")
& docker stop vllm-stock-ab | Out-Null

Invoke-Docker (@("run", "-d", "--name", "vllm-fused-ab", "--gpus", "all", "-p", "8001:8000", "-v", $mount, "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_USE_FUSED_SILU_MUL=1", $FusedImage) + $common)
Wait-Health 8001
Invoke-Docker @("run", "--rm", "--network", "host", "-v", "${projectRoot}:/workspace", "-e", "PYTHONPATH=/workspace/python", "--entrypoint", "python3", $FusedImage, "-m", "vllm_fused_swiglu_lab.service_benchmark", "--base-url", "http://127.0.0.1:8001", "--model", $Model, "--implementation", "fused", "--concurrency", "$Concurrency", "--requests", "$Requests", "--output-tokens", "$OutputTokens", "--output", "/workspace/$containerOutputDir/service_fused.json")

$stock = Get-Content (Join-Path $outputPath "service_stock.json") -Raw | ConvertFrom-Json
$fused = Get-Content (Join-Path $outputPath "service_fused.json") -Raw | ConvertFrom-Json
$summary = [ordered]@{
  model = $Model
  requests = $Requests
  concurrency = $Concurrency
  output_tokens = $OutputTokens
  stock = $stock.summary
  fused = $fused.summary
  delta_fused_minus_stock = [ordered]@{
    ttft_p50_ms = $fused.summary.ttft_p50_ms - $stock.summary.ttft_p50_ms
    latency_p95_ms = $fused.summary.latency_p95_ms - $stock.summary.latency_p95_ms
    output_tok_s = $fused.summary.output_tok_s - $stock.summary.output_tok_s
  }
}
$summary | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $outputPath "service_ab_summary.json") -Encoding UTF8
$summary | ConvertTo-Json -Depth 8

if (!$KeepContainers) {
  foreach ($name in @("vllm-stock-ab", "vllm-fused-ab")) {
    $existing = & docker ps -a --format "{{.Names}}"
    if ($existing -contains $name) { Invoke-Docker @("rm", "-f", $name) | Out-Null }
  }
}
