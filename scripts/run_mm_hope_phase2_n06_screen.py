from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase2_control_screen import run_phase2_control_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen N06 extra-pass and state-matched fairness controls"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase2_control_screen(
        args.workspace_root,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
