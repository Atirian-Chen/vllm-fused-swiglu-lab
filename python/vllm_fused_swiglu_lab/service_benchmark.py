from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx


def _make_prompt(index: int, target_tokens: int) -> str:
    """Build a deterministic prompt whose Qwen token count is close to target_tokens."""
    prefix = f"Request {index}:"
    return prefix + " benchmark" * max(1, target_tokens - 4)


async def _one(client, url, model, prompt, output_tokens):
    started = time.perf_counter()
    first = None
    pieces = []
    usage = {}
    async with client.stream("POST", url, json={
        "model": model, "prompt": prompt, "max_tokens": output_tokens,
        "temperature": 0, "ignore_eos": True,
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
    output_count = usage.get("completion_tokens", len(pieces))
    latency_ms = (ended - started) * 1000
    ttft_ms = ((first or ended) - started) * 1000
    return {
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "tpot_ms": ((latency_ms - ttft_ms) / (output_count - 1)
                    if output_count > 1 else None),
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": output_count,
        "total_tokens": usage.get("total_tokens"),
    }


async def _run(args):
    prompts = [_make_prompt(i, args.prompt_tokens) for i in range(args.requests)]
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        url = args.base_url.rstrip("/") + "/v1/completions"
        for i in range(args.warmup_requests):
            await _one(client, url, args.model,
                       _make_prompt(100000 + i, args.prompt_tokens),
                       args.output_tokens)

        started = time.perf_counter()

        async def guarded(i):
            async with sem:
                try:
                    r = await _one(client, url, args.model, prompts[i], args.output_tokens)
                    r["ok"] = True; return r
                except Exception as e:
                    return {"ok": False, "error": str(e)}
        results = await asyncio.gather(*(guarded(i) for i in range(args.requests)))
    wall_time = time.perf_counter() - started
    good = [r for r in results if r["ok"]]
    latencies = sorted(r["latency_ms"] for r in good)
    ttfts = sorted(r["ttft_ms"] for r in good)
    tpots = sorted(r["tpot_ms"] for r in good if r["tpot_ms"] is not None)
    prompt_counts = [r["prompt_tokens"] for r in good if r["prompt_tokens"] is not None]
    input_tokens = sum(prompt_counts)
    output_tokens = sum(r["output_tokens"] for r in good)
    summary = {
        "implementation": args.implementation, "requests": len(results),
        "concurrency": args.concurrency,
        "target_prompt_tokens": args.prompt_tokens,
        "actual_prompt_tokens_p50": statistics.median(prompt_counts) if prompt_counts else None,
        "requested_output_tokens": args.output_tokens,
        "success_rate": len(good) / len(results),
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
        "ttft_p95_ms": ttfts[max(0, int(len(ttfts) * .95) - 1)] if ttfts else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": latencies[max(0, int(len(latencies) * .95) - 1)] if latencies else None,
        "tpot_p50_ms": statistics.median(tpots) if tpots else None,
        "input_tok_s": input_tokens / wall_time if wall_time else 0,
        "output_tok_s": output_tokens / wall_time if wall_time else 0,
        "total_tok_s": (input_tokens + output_tokens) / wall_time if wall_time else 0,
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
    p.add_argument("--prompt-tokens", type=int, default=32,
                   help="Approximate prompt length; actual server token count is recorded")
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--warmup-requests", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--implementation", choices=["stock", "fused"], default="stock")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    benchmark()
