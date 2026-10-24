from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.gates import write_gate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MM-HOPE Phase-0 correctness gates")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/mm_hope_phase0_gate_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parent
    payload = write_gate_report(workspace_root, workspace_root / args.output)
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
