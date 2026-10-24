from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase3_diagnostics import run_continual_diagnostics
from experiments.mm_hope.phase3_protocol import load_phase4_r01_protocol
from experiments.mm_hope.phase4_robustness import R01_PREFIX


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze completed MM-HOPE Phase-4 R01 robustness results"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.workspace_root.resolve()
    payload = run_continual_diagnostics(
        root,
        protocol=load_phase4_r01_protocol(root),
        source_prefix=R01_PREFIX,
        output_prefix=R01_PREFIX,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
