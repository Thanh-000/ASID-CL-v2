from __future__ import annotations

import json
from pathlib import Path

from experiments.mm_hope.phase2_control_diagnostics import (
    write_phase2_control_diagnostics,
)


def main() -> None:
    payload = write_phase2_control_diagnostics(Path(__file__).resolve().parent)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
