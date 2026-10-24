from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase1_runner import run_phase1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MM-HOPE Phase-1 offline baselines")
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--models", nargs="+", default=["B03", "B04", "B05", "B06"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase1(
        args.workspace_root,
        models=args.models,
        seeds=args.seeds,
        encoder_batch_size=args.encoder_batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        force_cache=args.force_cache,
        cache_only=args.cache_only,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
