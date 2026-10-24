from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.mm_hope.data import load_joint_v2_dataset
from experiments.mm_hope.phase1_config import CLASS_TO_INDEX
from experiments.mm_hope.phase1_runner import (
    _load_b02_predictions,
    compute_pairwise_rows,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = load_joint_v2_dataset(root)
    path = root / "analysis_outputs" / "mm_hope_phase1_predictions.csv"
    predictions = pd.read_csv(path)
    predictions = predictions[predictions["evaluation_split"] == "test"]
    seeds = sorted(int(value) for value in predictions["seed"].unique())
    test_frame = dataset.split("test").sort_values("row_id", kind="stable")
    row_ids = test_frame["row_id"].to_numpy(dtype=np.int64)
    labels = np.asarray(
        [CLASS_TO_INDEX[value] for value in test_frame["predicted_crisis_type"]],
        dtype=np.int64,
    )
    predictions_by_model: dict[str, list[np.ndarray]] = {}
    for experiment_id in sorted(predictions["experiment_id"].unique()):
        predictions_by_model[experiment_id] = []
        for seed in seeds:
            selected = predictions[
                (predictions["experiment_id"] == experiment_id)
                & (predictions["seed"] == seed)
            ].sort_values("row_id", kind="stable")
            if not np.array_equal(selected["row_id"].to_numpy(dtype=np.int64), row_ids):
                raise ValueError(f"prediction alignment failed for {experiment_id}/{seed}")
            predictions_by_model[experiment_id].append(
                np.asarray([CLASS_TO_INDEX[value] for value in selected["predicted_label"]])
            )
    b02 = _load_b02_predictions(root, row_ids)
    predictions_by_model["B02"] = [b02.copy() for _ in seeds]
    rows = compute_pairwise_rows(
        labels,
        predictions_by_model,
        test_frame["component_id"].astype(str).to_numpy(),
        bootstrap_replicates=2000,
    )
    pairwise_path = root / "analysis_outputs" / "mm_hope_phase1_pairwise_inference.csv"
    pd.DataFrame(rows).to_csv(pairwise_path, index=False)

    metrics_path = root / "analysis_outputs" / "mm_hope_phase1_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["pairwise_inference"] = rows
    lookup = {row["comparison"]: row for row in rows}
    payload["decisions"]["B05_vs_preregistered_text_ci_excludes_zero"] = bool(
        lookup["late_fusion_vs_frozen_text"]["ci_lo"] > 0.0
    )
    payload["decisions"][
        "B05_vs_observed_best_unimodal_image_ci_excludes_zero"
    ] = bool(lookup["late_fusion_vs_best_unimodal_image"]["ci_lo"] > 0.0)
    payload["decisions"].pop("B05_modality_gain_ci_excludes_zero", None)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
