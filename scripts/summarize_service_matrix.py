from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[max(0, int(len(values) * p) - 1)]


def aggregate(files: list[Path]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        grouped[(payload["case"], payload["summary"]["implementation"])].append(payload)

    rows = []
    for case in sorted({key[0] for key in grouped}):
        implementations = {}
        for implementation in ("stock", "fused"):
            runs = grouped[(case, implementation)]
            requests = [request for run in runs for request in run["requests"] if request["ok"]]
            wall_time = sum(run["summary"]["wall_time_s"] for run in runs)
            prompt_tokens = [r["prompt_tokens"] for r in requests if r["prompt_tokens"] is not None]
            output_tokens = sum(r["output_tokens"] for r in requests)
            implementations[implementation] = {
                "runs": len(runs),
                "requests": len(requests),
                "success_rate": len(requests) / sum(len(run["requests"]) for run in runs),
                "concurrency": runs[0]["summary"]["concurrency"],
                "target_prompt_tokens": runs[0]["summary"]["target_prompt_tokens"],
                "actual_prompt_tokens_p50": statistics.median(prompt_tokens),
                "requested_output_tokens": runs[0]["summary"]["requested_output_tokens"],
                "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in requests),
                "ttft_p95_ms": percentile([r["ttft_ms"] for r in requests], .95),
                "latency_p50_ms": statistics.median(r["latency_ms"] for r in requests),
                "latency_p95_ms": percentile([r["latency_ms"] for r in requests], .95),
                "tpot_p50_ms": statistics.median(r["tpot_ms"] for r in requests if r["tpot_ms"] is not None),
                "output_tok_s": output_tokens / wall_time,
                "input_tok_s": sum(prompt_tokens) / wall_time,
            }
        stock, fused = implementations["stock"], implementations["fused"]
        rows.append({
            "case": case,
            "stock": stock,
            "fused": fused,
            "delta_fused_vs_stock_pct": {
                "ttft_p50": (fused["ttft_p50_ms"] / stock["ttft_p50_ms"] - 1) * 100,
                "latency_p95": (fused["latency_p95_ms"] / stock["latency_p95_ms"] - 1) * 100,
                "tpot_p50": (fused["tpot_p50_ms"] / stock["tpot_p50_ms"] - 1) * 100,
                "output_throughput": (fused["output_tok_s"] / stock["output_tok_s"] - 1) * 100,
            },
        })
    return {"rows": rows, "notes": "Pooled request metrics from alternating stock/fused service passes."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files = sorted(Path(args.input_dir).glob("*.json"))
    result = aggregate(files)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
