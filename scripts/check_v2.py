from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fused_swiglu import silu_and_mul


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/v2_correctness.json")
    args = parser.parse_args()

    torch.manual_seed(0)
    intermediate = 8960
    records = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        x = torch.randn(3, 2 * intermediate, device="cuda", dtype=dtype)
        y = silu_and_mul(x)
        reference = torch.nn.functional.silu(x[:, :intermediate]) * x[:, intermediate:]
        records[name] = {
            "shape": list(y.shape),
            "max_abs_error": float((y - reference).abs().max().item()),
        }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
