from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.phase3_ewc_extension import run_phase3_ewc_scale_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend the validation-only Phase-3 EWC lambda scale screen"
    )
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_phase3_ewc_scale_extension(args.workspace_root, device=args.device)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
