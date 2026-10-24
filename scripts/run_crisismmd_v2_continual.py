from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.crisismmd_v2_continual import run_crisismmd_v2_event_continual


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrisisMMD event-domain continual learning.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--methods", nargs="+", default=["C01", "C02", "C05", "C06"])
    parser.add_argument("--orders", nargs="+", default=["O1", "O2", "O3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_crisismmd_v2_event_continual(
        args.workspace_root,
        methods=args.methods,
        order_names=args.orders,
        seeds=args.seeds,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
