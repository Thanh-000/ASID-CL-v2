from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase4_robustness import run_phase4_r01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable MM-HOPE Phase-4 R01 joint-split robustness"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["C01", "C02", "C03", "C04", "C05", "C06"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[2027, 2028, 2029, 2030, 2031],
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        default=["O1", "O2", "O3", "O4", "O5", "O6"],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase4_r01(
        args.workspace_root,
        methods=args.methods,
        seeds=args.seeds,
        orders=args.orders,
        bootstrap_replicates=args.bootstrap_replicates,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "completed_run_count": payload["completed_run_count"],
                "expected_run_count": payload["expected_run_count"],
                "decisions": payload["decisions"],
                "artifacts": payload["artifacts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
