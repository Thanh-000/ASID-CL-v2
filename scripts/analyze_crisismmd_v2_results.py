from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "analysis_outputs" / "crisismmd_v2_runs"
OFFLINE_VARIANTS = (
    "humanitarian_agreed",
    "informative_agreed",
    "humanitarian_all",
    "damage_all",
)
CONTINUAL_CONTRASTS = (
    ("C02", "C01"),
    ("C05", "C01"),
    ("C06", "C02"),
    ("C06", "C05"),
)
CONTINUAL_METRICS = (
    "final_task_average_macro_f1",
    "final_overall_macro_f1",
    "forgetting",
    "bwt",
)


def main() -> None:
    offline_rows: list[dict[str, object]] = []
    offline_seed_frames: list[pd.DataFrame] = []
    contrast_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    for variant in OFFLINE_VARIANTS:
        directory = RUN_ROOT / variant
        summary = pd.read_csv(directory / "summary.csv")
        seed_frame = summary.copy()
        seed_frame.insert(0, "variant", variant)
        offline_seed_frames.append(seed_frame)
        inference = json.loads(
            (directory / "cluster_bootstrap_inference.json").read_text("utf-8")
        )
        for experiment_id, group in summary.groupby("experiment_id"):
            estimate = inference["models"][experiment_id]
            offline_rows.append(
                {
                    "variant": variant,
                    "experiment_id": experiment_id,
                    "runs": len(group),
                    "val_macro_f1_mean": float(group["val_macro_f1"].mean()),
                    "test_macro_f1_mean": float(group["test_macro_f1"].mean()),
                    "test_macro_f1_sd": float(group["test_macro_f1"].std(ddof=1)),
                    "cluster_ci_lo": estimate["ci_lo"],
                    "cluster_ci_hi": estimate["ci_hi"],
                    "test_balanced_accuracy_mean": float(
                        group["test_balanced_accuracy"].mean()
                    ),
                    "test_accuracy_mean": float(group["test_accuracy"].mean()),
                    "test_worst_class_f1_mean": float(
                        group["test_worst_class_f1"].mean()
                    ),
                    "test_ece_15_mean": float(group["test_ece_15"].mean()),
                    "train_seconds_mean": float(group["train_seconds"].mean()),
                    "parameter_count": int(group["parameter_count"].iloc[0]),
                }
            )
        for name, values in inference["contrasts"].items():
            contrast_rows.append(
                {
                    "scope": variant,
                    "contrast": name,
                    "metric": "test_macro_f1",
                    **values,
                }
            )
        for path in directory.glob("E??_seed????.json"):
            result = json.loads(path.read_text("utf-8"))
            for row in result["evaluation"]["test"]["per_class"]:
                per_class_rows.append(
                    {
                        "variant": variant,
                        "experiment_id": result["experiment_id"],
                        "seed": result["seed"],
                        **row,
                    }
                )
            prediction_path = path.with_name(path.stem + "_predictions.npz")
            predictions = np.load(prediction_path)
            true_labels = predictions["labels"].astype(np.int64)
            predicted_labels = predictions["predictions"].astype(np.int64)
            class_count = len(result["labels"])
            matrix = np.zeros((class_count, class_count), dtype=np.int64)
            np.add.at(matrix, (true_labels, predicted_labels), 1)
            for true_index, true_name in enumerate(result["labels"]):
                support = int(matrix[true_index].sum())
                for predicted_index, predicted_name in enumerate(result["labels"]):
                    count = int(matrix[true_index, predicted_index])
                    confusion_rows.append(
                        {
                            "variant": variant,
                            "experiment_id": result["experiment_id"],
                            "seed": result["seed"],
                            "true_class": true_name,
                            "predicted_class": predicted_name,
                            "count": count,
                            "true_support": support,
                        }
                    )

    continual = pd.read_csv(RUN_ROOT / "event_continual" / "summary.csv")
    continual_method_rows = []
    continual_event_rows: list[dict[str, object]] = []
    continual_curve_rows: list[dict[str, object]] = []
    for method_id, group in continual.groupby("method_id"):
        continual_method_rows.append(
            {
                "method_id": method_id,
                "runs": len(group),
                **{
                    f"{metric}_mean": float(group[metric].mean())
                    for metric in CONTINUAL_METRICS
                },
                **{
                    f"{metric}_sd": float(group[metric].std(ddof=1))
                    for metric in CONTINUAL_METRICS
                },
                "train_seconds_mean": float(group["train_seconds"].mean()),
                "additional_evaluation_memory_bytes": int(
                    group["additional_evaluation_memory_bytes"].iloc[0]
                ),
            }
        )
    for path in (RUN_ROOT / "event_continual").glob("C??_O?_seed????.json"):
        result = json.loads(path.read_text("utf-8"))
        for row in result["history"][-1]["events"]:
            continual_event_rows.append(
                {
                    "method_id": result["method_id"],
                    "order_name": result["order_name"],
                    "seed": result["seed"],
                    **row,
                }
            )
        for state in result["history"]:
            for row in state["events"]:
                continual_curve_rows.append(
                    {
                        "method_id": result["method_id"],
                        "order_name": result["order_name"],
                        "seed": result["seed"],
                        "stream_step": state["stream_step"],
                        "trained_event": state.get("trained_event", "pretraining_baseline"),
                        **row,
                    }
                )

    for left, right in CONTINUAL_CONTRASTS:
        paired = continual[continual["method_id"].isin((left, right))].pivot(
            index=["order_name", "seed"], columns="method_id"
        )
        if len(paired) != 15:
            raise ValueError(f"incomplete continual pairing: {left} vs {right}")
        for metric in CONTINUAL_METRICS:
            differences = (
                paired[metric][left].to_numpy() - paired[metric][right].to_numpy()
            )
            estimate = _paired_bootstrap(differences)
            contrast_rows.append(
                {
                    "scope": "event_continual",
                    "contrast": f"{left}_minus_{right}",
                    "metric": metric,
                    **estimate,
                }
            )

    offline_frame = pd.DataFrame(offline_rows)
    offline_seed_frame = pd.concat(offline_seed_frames, ignore_index=True)
    continual_frame = pd.DataFrame(continual_method_rows)
    contrasts_frame = pd.DataFrame(contrast_rows)
    per_class_frame = pd.DataFrame(per_class_rows)
    per_class_summary = (
        per_class_frame.groupby(["variant", "experiment_id", "class"], as_index=False)
        .agg(
            support=("support", "first"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
        )
    )
    continual_event_frame = pd.DataFrame(continual_event_rows)
    continual_event_summary = (
        continual_event_frame.groupby(["method_id", "event"], as_index=False)
        .agg(
            runs=("seed", "size"),
            n=("n", "first"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
        )
    )
    confusion_frame = pd.DataFrame(confusion_rows)
    confusion_summary = (
        confusion_frame.groupby(
            ["variant", "experiment_id", "true_class", "predicted_class"],
            as_index=False,
        )
        .agg(count=("count", "sum"), true_support=("true_support", "sum"))
    )
    confusion_summary["row_rate"] = (
        confusion_summary["count"] / confusion_summary["true_support"]
    )
    continual_order_summary = (
        continual.groupby(["method_id", "order_name"], as_index=False)
        .agg(
            runs=("seed", "size"),
            task_average_mean=("final_task_average_macro_f1", "mean"),
            task_average_sd=("final_task_average_macro_f1", "std"),
            overall_mean=("final_overall_macro_f1", "mean"),
            overall_sd=("final_overall_macro_f1", "std"),
            forgetting_mean=("forgetting", "mean"),
            forgetting_sd=("forgetting", "std"),
            bwt_mean=("bwt", "mean"),
            bwt_sd=("bwt", "std"),
            fwt_mean=("fwt", "mean"),
            fwt_sd=("fwt", "std"),
            train_seconds_mean=("train_seconds", "mean"),
        )
    )
    continual_curve_frame = pd.DataFrame(continual_curve_rows)
    output = ROOT / "analysis_outputs"
    offline_frame.to_csv(output / "crisismmd_v2_offline_model_summary.csv", index=False)
    continual_frame.to_csv(output / "crisismmd_v2_continual_method_summary.csv", index=False)
    contrasts_frame.to_csv(output / "crisismmd_v2_all_contrasts.csv", index=False)
    per_class_summary.to_csv(
        output / "crisismmd_v2_per_class_summary.csv", index=False
    )
    continual_event_summary.to_csv(
        output / "crisismmd_v2_continual_event_summary.csv", index=False
    )
    offline_seed_frame.to_csv(
        output / "crisismmd_v2_offline_seed_results.csv", index=False
    )
    confusion_summary.to_csv(
        output / "crisismmd_v2_offline_confusion_summary.csv", index=False
    )
    continual.to_csv(
        output / "crisismmd_v2_continual_seed_order_results.csv", index=False
    )
    continual_order_summary.to_csv(
        output / "crisismmd_v2_continual_order_summary.csv", index=False
    )
    continual_curve_frame.to_csv(
        output / "crisismmd_v2_continual_learning_curves.csv", index=False
    )
    payload = {
        "offline": offline_rows,
        "continual": continual_method_rows,
        "contrasts": contrast_rows,
        "per_class": per_class_summary.to_dict(orient="records"),
        "continual_events": continual_event_summary.to_dict(orient="records"),
        "continual_orders": continual_order_summary.to_dict(orient="records"),
        "completed_runs": {
            "offline": int(sum(row["runs"] for row in offline_rows)),
            "continual": int(len(continual)),
            "total": int(sum(row["runs"] for row in offline_rows) + len(continual)),
        },
    }
    (output / "crisismmd_v2_full_experiment_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["completed_runs"], indent=2))


def _paired_bootstrap(values: np.ndarray, replicates: int = 10_000) -> dict[str, object]:
    differences = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(2026)
    samples = rng.choice(differences, size=(replicates, len(differences)), replace=True)
    bootstrap = samples.mean(axis=1)
    observed = abs(float(differences.mean()))
    sign_flip = np.asarray(
        [
            abs(float(np.mean(differences * np.asarray(signs))))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return {
        "delta": float(differences.mean()),
        "ci_lo": float(np.quantile(bootstrap, 0.025)),
        "ci_hi": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_gt_zero": float(np.mean(bootstrap > 0)),
        "exact_sign_flip_p_value": float(np.mean(sign_flip >= observed - 1e-12)),
        "paired_deltas": differences.tolist(),
    }


if __name__ == "__main__":
    main()
