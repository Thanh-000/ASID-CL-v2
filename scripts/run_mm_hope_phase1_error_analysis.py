from __future__ import annotations

import json
from pathlib import Path

from experiments.mm_hope.data import load_joint_v2_dataset
from experiments.mm_hope.error_analysis import write_stable_error_analysis


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = load_joint_v2_dataset(root)
    predictions_path = root / "analysis_outputs" / "mm_hope_phase1_predictions.csv"
    payload = write_stable_error_analysis(
        root,
        dataset,
        predictions_path=predictions_path,
    )
    phase1_metrics_path = root / "analysis_outputs" / "mm_hope_phase1_metrics.json"
    with phase1_metrics_path.open("r", encoding="utf-8") as handle:
        phase1_payload = json.load(handle)
    phase1_payload["stable_error_analysis"] = payload
    phase1_payload.setdefault("artifacts", {}).update(payload["artifacts"])
    with phase1_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(phase1_payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
