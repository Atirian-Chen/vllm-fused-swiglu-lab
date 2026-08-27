from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


async def _one(client, url, model, prompt, output_tokens):
    started = time.perf_counter()
    first = None
    pieces = []
    usage = {}
    async with client.stream("POST", url, json={
        "model": model, "prompt": prompt, "max_tokens": output_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            data = json.loads(payload)
            if first is None and data.get("choices"):
                first = time.perf_counter()
            pieces.extend(c.get("text", "") for c in data.get("choices", []))
            usage = data.get("usage") or usage
    ended = time.perf_counter()
    return {"ttft_ms": ((first or ended) - started) * 1000,
            "latency_ms": (ended - started) * 1000,
            "output_tokens": usage.get("completion_tokens", len(pieces))}


async def _run(args):
    prompts = [f"Summarize request {i} about inference scheduling and CUDA kernels." for i in range(args.requests)]
    sem = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def guarded(i):
            async with sem:
                try:
                    r = await _one(client, args.base_url.rstrip("/") + "/v1/completions", args.model, prompts[i], args.output_tokens)
                    r["ok"] = True; return r
                except Exception as e:
                    return {"ok": False, "error": str(e)}
        results = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
    wall_time = time.perf_counter() - started
    good = [r for r in results if r["ok"]]
    latencies = sorted(r["latency_ms"] for r in good)
    summary = {
        "implementation": args.implementation, "requests": len(results),
        "success_rate": len(good) / len(results),
        "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in good) if good else None,
        "latency_p95_ms": latencies[max(0, int(len(latencies) * .95) - 1)] if latencies else None,
        "output_tok_s": sum(r["output_tokens"] for r in good) / wall_time if wall_time else 0,
        "wall_time_s": wall_time,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"summary": summary, "requests": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def benchmark():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=60)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--implementation", choices=["stock", "fused"], default="stock")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    benchmark()

