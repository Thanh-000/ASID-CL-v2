from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis_outputs"
METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "mcc",
    "worst_class_f1",
    "ece_15",
    "brier_score",
    "nll",
)
CL_METRICS = (
    "final_average_macro_f1",
    "average_forgetting",
    "backward_transfer",
    "forward_transfer",
    "average_incremental_macro_f1",
    "all_task_auc_macro_f1",
    "final_overall_macro_f1",
    "final_overall_balanced_accuracy",
    "final_overall_accuracy",
    "final_overall_mcc",
    "final_overall_ece_15",
)


def _model_summary(frame: pd.DataFrame, model_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in frame.groupby(model_column, sort=True):
        row: dict[str, object] = {model_column: model, "runs": len(group)}
        for metric in METRICS:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean())
                row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        for column in ("train_seconds", "parameter_count", "parameter_bytes"):
            if column in group:
                row[f"{column}_mean"] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    phase1 = pd.read_csv(OUTPUT / "mm_hope_phase1_results.csv")
    phase1_test = phase1[phase1["evaluation_split"] == "test"].copy()
    phase1_test.to_csv(OUTPUT / "uk_crisismmd_phase1_seed_results.csv", index=False)
    phase1_summary = _model_summary(phase1_test, "experiment_id")
    phase1_summary.to_csv(OUTPUT / "uk_crisismmd_phase1_model_summary.csv", index=False)

    phase1_class = pd.read_csv(OUTPUT / "mm_hope_phase1_per_class.csv")
    phase1_class = phase1_class[phase1_class["evaluation_split"] == "test"]
    phase1_class_summary = (
        phase1_class.groupby(["experiment_id", "class"], as_index=False)
        .agg(
            runs=("seed", "size"),
            support=("support", "first"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
        )
    )
    phase1_class_summary.to_csv(
        OUTPUT / "uk_crisismmd_phase1_per_class_summary.csv", index=False
    )

    phase2 = pd.read_csv(OUTPUT / "mm_hope_phase2_n06_combined_results.csv")
    phase2_test = phase2[phase2["evaluation_split"] == "test"].copy()
    phase2_test.to_csv(OUTPUT / "uk_crisismmd_phase2_seed_results.csv", index=False)
    phase2_summary = _model_summary(phase2_test, "experiment_id")
    resource_columns = (
        "fast_delta_bytes",
        "runtime_fast_state_bytes",
        "evaluation_state_bytes",
        "total_evaluation_parameter_bytes",
        "runtime_extra_bytes",
        "memory_updates_observed",
        "memory_updates_expected",
    )
    for column in resource_columns:
        if column not in phase2_test:
            continue
        values = phase2_test.groupby("experiment_id")[column].mean()
        phase2_summary[f"{column}_mean"] = phase2_summary["experiment_id"].map(values)
    phase2_summary.to_csv(OUTPUT / "uk_crisismmd_phase2_model_summary.csv", index=False)

    phase2_class = pd.read_csv(OUTPUT / "mm_hope_phase2_n06_combined_per_class.csv")
    phase2_class = phase2_class[phase2_class["evaluation_split"] == "test"]
    phase2_class_summary = (
        phase2_class.groupby(["experiment_id", "class"], as_index=False)
        .agg(
            runs=("seed", "size"),
            support=("support", "first"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
        )
    )
    phase2_class_summary.to_csv(
        OUTPUT / "uk_crisismmd_phase2_per_class_summary.csv", index=False
    )

    phase3 = pd.read_csv(OUTPUT / "mm_hope_phase3_confirmatory_results.csv")
    if len(phase3) != 210:
        raise ValueError(f"expected 210 Phase-3 runs, found {len(phase3)}")
    phase3_rows: list[dict[str, object]] = []
    for model, group in phase3.groupby("model_key", sort=True):
        row: dict[str, object] = {"model_key": model, "runs": len(group)}
        for metric in CL_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        for column in (
            "task_train_seconds",
            "additional_evaluation_memory_bytes",
            "outer_optimizer_state_bytes",
            "fast_runtime_state_bytes",
            "replay_storage_bytes",
            "ewc_state_bytes",
        ):
            row[f"{column}_mean"] = float(group[column].mean())
        phase3_rows.append(row)
    phase3_summary = pd.DataFrame(phase3_rows)
    phase3_summary.to_csv(OUTPUT / "uk_crisismmd_phase3_method_summary.csv", index=False)

    phase3_seed_summary = (
        phase3.groupby(["model_key", "seed"], as_index=False)
        .agg(
            orders=("order_name", "size"),
            final_average_macro_f1=("final_average_macro_f1", "mean"),
            average_forgetting=("average_forgetting", "mean"),
            backward_transfer=("backward_transfer", "mean"),
            final_overall_macro_f1=("final_overall_macro_f1", "mean"),
            task_train_seconds=("task_train_seconds", "mean"),
        )
    )
    phase3_seed_summary.to_csv(
        OUTPUT / "uk_crisismmd_phase3_seed_summary.csv", index=False
    )

    phase3_order_summary = (
        phase3.groupby(["model_key", "order_name"], as_index=False)
        .agg(
            seeds=("seed", "size"),
            final_average_macro_f1_mean=("final_average_macro_f1", "mean"),
            final_average_macro_f1_sd=("final_average_macro_f1", "std"),
            average_forgetting_mean=("average_forgetting", "mean"),
            average_forgetting_sd=("average_forgetting", "std"),
            final_overall_macro_f1_mean=("final_overall_macro_f1", "mean"),
            final_overall_macro_f1_sd=("final_overall_macro_f1", "std"),
            task_train_seconds_mean=("task_train_seconds", "mean"),
        )
    )
    phase3_order_summary.to_csv(
        OUTPUT / "uk_crisismmd_phase3_order_summary.csv", index=False
    )

    payload = {
        "phase1": {
            "runs": len(phase1_test),
            "models": phase1_summary.to_dict(orient="records"),
        },
        "phase2": {
            "runs": len(phase2_test),
            "models": phase2_summary.to_dict(orient="records"),
        },
        "phase3": {
            "runs": len(phase3),
            "models": phase3_summary.to_dict(orient="records"),
        },
    }
    (OUTPUT / "uk_crisismmd_full_results_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value["runs"] for key, value in payload.items()}, indent=2))


if __name__ == "__main__":
    main()
