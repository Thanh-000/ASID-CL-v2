from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.crisismmd_v2_runner import run_crisismmd_v2_offline_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrisisMMD v2.0 external benchmark.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--task", choices=("humanitarian", "informative", "damage"), default="humanitarian")
    parser.add_argument("--all-labels", action="store_true")
    parser.add_argument("--models", nargs="+", default=["E01", "E02", "E03", "E04", "E05", "E06"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029, 2030, 2031])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    result = run_crisismmd_v2_offline_suite(
        args.workspace_root,
        task=args.task,
        agreed_labels=not args.all_labels,
        models=args.models,
        seeds=args.seeds,
        device=args.device,
        encoder_batch_size=args.encoder_batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
