from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase3_diagnostics import run_phase3_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed MM-HOPE Phase-3 continual-learning results"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase3_diagnostics(args.workspace_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
