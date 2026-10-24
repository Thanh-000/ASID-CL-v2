from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase2_runner import run_phase2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MM-HOPE Phase-2 N01/N02 mechanism isolation"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--memory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase2(
        args.workspace_root,
        seeds=args.seeds,
        memory_learning_rate=args.memory_learning_rate,
        bootstrap_replicates=args.bootstrap_replicates,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
