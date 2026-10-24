from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase2_confirm import CONFIRMATION_SEEDS, run_phase2_confirmation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run held-out five-seed MM-HOPE Phase-2 confirmation"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase2_confirmation(
        args.workspace_root,
        seeds=args.seeds,
        bootstrap_replicates=args.bootstrap_replicates,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
