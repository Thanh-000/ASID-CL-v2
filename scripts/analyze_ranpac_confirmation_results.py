from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from experiments.mm_hope.gpu_study_data import load_gpu_study_dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE = ROOT / "MM_HOPE_results (9).zip"
BASE = "continual_baseline_s_frozen_v1"
SUITE = "ranpac_class_balanced_ridge"
CONTROL = "balanced_p1024_rel1e-1_beta0p00_cpu_fp64"
PRIMARY = "balanced_p1024_rel1e-1_beta0p50_cpu_fp64"


def _json(zf: zipfile.ZipFile, relative: str) -> dict[str, Any]:
    return json.loads(zf.read(f"{BASE}/{relative}").decode("utf-8"))


def _csv(zf: zipfile.ZipFile, relative: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(f"{BASE}/{relative}")))


def _canonical_sha256(payload: dict[str, Any], digest_field: str) -> str:
    clean = dict(payload)
    clean.pop(digest_field, None)
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _calibrated_confidence(frame: pd.DataFrame, temperature: float) -> float:
    probability_columns = [column for column in frame if column.startswith("prob_")]
    probabilities = np.clip(frame[probability_columns].to_numpy(float), 1e-12, 1.0)
    logits = np.log(probabilities) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return float(calibrated.max(axis=1).mean())


def _validate_member_names(zf: zipfile.ZipFile) -> None:
    for name in zf.namelist():
        member = PurePosixPath(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe ZIP member: {name}")


def _load_and_validate(
    zf: zipfile.ZipFile, workspace: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[tuple[str, str, int], pd.DataFrame],
    dict[str, Any],
]:
    _validate_member_names(zf)
    contract = _json(zf, "representation_contract.json")
    lock = _json(zf, f"{SUITE}/confirmation_lock.json")
    inference = _json(zf, f"{SUITE}/confirmation/confirmation_inference.json")
    if _canonical_sha256(contract, "contract_sha256") != contract["contract_sha256"]:
        raise ValueError("representation contract SHA-256 is invalid")
    if _canonical_sha256(lock, "lock_sha256") != lock["lock_sha256"]:
        raise ValueError("confirmation lock SHA-256 is invalid")
    if lock["representation_contract_sha256"] != contract["contract_sha256"]:
        raise ValueError("lock and representation contract do not match")

    seeds = tuple(int(seed) for seed in lock["projection_seeds"])
    candidates = tuple(lock["candidate_ids"])
    if candidates != (CONTROL, PRIMARY):
        raise ValueError(f"unexpected locked candidates: {candidates}")

    datasets: dict[str, Any] = {}
    results: dict[tuple[str, str, int], dict[str, Any]] = {}
    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    for dataset_key, locked_dataset in lock["datasets"].items():
        dataset = load_gpu_study_dataset(workspace, dataset_key)
        if dataset.fingerprint != locked_dataset["dataset_fingerprint"]:
            raise ValueError(f"local dataset fingerprint mismatch: {dataset_key}")
        test = dataset.frame.loc[dataset.frame["split"] == "test"].copy()
        test["row_id"] = test["row_id"].astype(str)
        test["group_id"] = test["group_id"].astype(str)
        test_ids = set(test["row_id"].astype(str))
        test_lookup = test.set_index("row_id")
        datasets[dataset_key] = dataset
        for candidate in candidates:
            for seed in seeds:
                relative = (
                    f"{SUITE}/confirmation/{dataset_key}/{candidate}/"
                    f"seed_{seed}"
                )
                result = _json(zf, f"{relative}/result.json")
                prediction = _csv(zf, f"{relative}/test_predictions.csv")
                prediction["row_id"] = prediction["row_id"].astype(str)
                prediction["group_id"] = prediction["group_id"].astype(str)
                if result.get("status") != "complete" or result.get("stage") != "confirmation":
                    raise ValueError(f"incomplete confirmation result: {relative}")
                if not result.get("test_accessed"):
                    raise ValueError(f"test-access marker missing: {relative}")
                expected = {
                    "dataset": dataset_key,
                    "candidate_id": candidate,
                    "dataset_fingerprint": dataset.fingerprint,
                    "representation_contract_sha256": contract["contract_sha256"],
                    "ranpac_class_balanced_lock_sha256": lock["lock_sha256"],
                }
                for field, value in expected.items():
                    if result.get(field) != value:
                        raise ValueError(f"{relative}: invalid {field}")
                if int(result["config"]["seed"]) != seed:
                    raise ValueError(f"{relative}: invalid config.seed")
                if len(prediction) != len(test) or set(prediction["row_id"].astype(str)) != test_ids:
                    raise ValueError(f"prediction/test row mismatch: {relative}")
                raw_targets = pd.to_numeric(prediction["target_index"], errors="raise").to_numpy()
                targets = raw_targets.astype(int)
                if not np.array_equal(raw_targets, targets):
                    raise ValueError(f"non-integral target labels: {relative}")
                expected_targets = test_lookup.loc[prediction["row_id"], "label_index"].to_numpy(int)
                expected_groups = test_lookup.loc[prediction["row_id"], "group_id"].to_numpy(str)
                if not np.array_equal(targets, expected_targets):
                    raise ValueError(f"prediction/local targets differ: {relative}")
                if not np.array_equal(prediction["group_id"].to_numpy(str), expected_groups):
                    raise ValueError(f"prediction/local groups differ: {relative}")
                results[(dataset_key, candidate, seed)] = result
                predictions[(dataset_key, candidate, seed)] = prediction.sort_values(
                    "row_id", kind="stable"
                ).reset_index(drop=True)
    return contract, lock, inference, results, predictions, datasets


def _overall_rows(
    lock: dict[str, Any],
    results: dict[tuple[str, str, int], dict[str, Any]],
    predictions: dict[tuple[str, str, int], pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_key in lock["datasets"]:
        for candidate in lock["candidate_ids"]:
            selected = [
                results[(dataset_key, candidate, int(seed))]
                for seed in lock["projection_seeds"]
            ]
            raw_confidence = []
            calibrated_confidence = []
            for result in selected:
                prediction = predictions[(dataset_key, candidate, int(result["config"]["seed"]))]
                probability_columns = [
                    column for column in prediction if column.startswith("prob_")
                ]
                raw_confidence.append(
                    float(prediction[probability_columns].max(axis=1).mean())
                )
                calibrated_confidence.append(
                    _calibrated_confidence(prediction, result["calibration_temperature"])
                )
            raw_metrics = [result["test_metrics"] for result in selected]
            calibrated_metrics = [result["test_calibrated_metrics"] for result in selected]
            persistent_bytes = [
                sum(int(value) for value in result["persistent_state_final"].values())
                for result in selected
            ]
            rows.append(
                {
                    "dataset": dataset_key,
                    "candidate_id": candidate,
                    "candidate_role": lock["candidate_roles"][candidate],
                    "seeds": len(selected),
                    "test_examples": int(selected[0]["test_examples"]),
                    "accuracy_mean": np.mean([metric["accuracy"] for metric in raw_metrics]),
                    "macro_f1_mean": np.mean([metric["macro_f1"] for metric in raw_metrics]),
                    "macro_f1_sd": np.std([metric["macro_f1"] for metric in raw_metrics], ddof=1),
                    "balanced_accuracy_mean": np.mean(
                        [metric["balanced_accuracy"] for metric in raw_metrics]
                    ),
                    "worst_class_f1_mean": np.mean(
                        [metric["worst_class_f1"] for metric in raw_metrics]
                    ),
                    "mcc_mean": np.mean([metric["mcc"] for metric in raw_metrics]),
                    "raw_nll_mean": np.mean([metric["nll"] for metric in raw_metrics]),
                    "calibrated_nll_mean": np.mean(
                        [metric["nll"] for metric in calibrated_metrics]
                    ),
                    "raw_ece_15_mean": np.mean([metric["ece_15"] for metric in raw_metrics]),
                    "calibrated_ece_15_mean": np.mean(
                        [metric["ece_15"] for metric in calibrated_metrics]
                    ),
                    "raw_brier_mean": np.mean(
                        [metric["brier_score"] for metric in raw_metrics]
                    ),
                    "calibrated_brier_mean": np.mean(
                        [metric["brier_score"] for metric in calibrated_metrics]
                    ),
                    "temperature_mean": np.mean(
                        [result["calibration_temperature"] for result in selected]
                    ),
                    "raw_confidence_mean": np.mean(raw_confidence),
                    "calibrated_confidence_mean": np.mean(calibrated_confidence),
                    "persistent_state_mib": np.mean(persistent_bytes) / 2**20,
                    "training_seconds_mean": np.mean(
                        [result["training_seconds"] for result in selected]
                    ),
                    "online_peak_vram_gib_mean": np.mean(
                        [result["online_peak_gpu_memory_bytes"] for result in selected]
                    )
                    / 2**30,
                    "feature_extraction_peak_vram_gib_mean": np.mean(
                        [result["feature_extraction_peak_gpu_memory_bytes"] for result in selected]
                    )
                    / 2**30,
                }
            )
    return rows


def _per_class_deltas(per_class: pd.DataFrame) -> pd.DataFrame:
    metrics = ["f1", "precision", "recall", "predicted_n"]
    grouped = per_class.groupby(["dataset", "candidate_id", "class"], as_index=False).agg(
        support=("support", "first"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
    )
    control = grouped.loc[grouped["candidate_id"] == CONTROL].drop(columns="candidate_id")
    primary = grouped.loc[grouped["candidate_id"] == PRIMARY].drop(columns="candidate_id")
    merged = control.merge(primary, on=["dataset", "class", "support"], suffixes=("_control", "_primary"))
    for metric in metrics:
        merged[f"{metric}_delta"] = (
            merged[f"{metric}_mean_primary"] - merged[f"{metric}_mean_control"]
        )
    paired = per_class.pivot(
        index=["dataset", "seed", "class"], columns="candidate_id", values="f1"
    ).reset_index()
    paired["delta"] = paired[PRIMARY] - paired[CONTROL]
    signs = paired.groupby(["dataset", "class"], as_index=False).agg(
        positive_seeds=("delta", lambda values: int((values > 1e-12).sum())),
        zero_seeds=("delta", lambda values: int((np.abs(values) <= 1e-12).sum())),
        negative_seeds=("delta", lambda values: int((values < -1e-12).sum())),
    )
    return merged.merge(signs, on=["dataset", "class"]).sort_values(
        ["dataset", "f1_delta"], ascending=[True, False]
    )


def _paired_frames(
    dataset_key: str,
    seed: int,
    predictions: dict[tuple[str, str, int], pd.DataFrame],
) -> pd.DataFrame:
    control = predictions[(dataset_key, CONTROL, seed)][
        ["row_id", "group_id", "target_index", "prediction_index"]
    ].rename(columns={"prediction_index": "control_prediction"})
    primary = predictions[(dataset_key, PRIMARY, seed)][
        ["row_id", "target_index", "prediction_index"]
    ].rename(
        columns={
            "target_index": "primary_target",
            "prediction_index": "primary_prediction",
        }
    )
    paired = control.merge(primary, on="row_id", validate="one_to_one")
    if not np.array_equal(paired["target_index"], paired["primary_target"]):
        raise ValueError(f"candidate targets differ: {dataset_key}, seed {seed}")
    return paired.drop(columns="primary_target")


def _transition_outputs(
    lock: dict[str, Any],
    datasets: dict[str, Any],
    predictions: dict[tuple[str, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    transition_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for dataset_key, dataset in datasets.items():
        labels = dataset.labels
        sample_correct: dict[str, list[np.ndarray]] = {CONTROL: [], PRIMARY: []}
        sample_predictions: dict[str, list[np.ndarray]] = {CONTROL: [], PRIMARY: []}
        for seed in lock["projection_seeds"]:
            paired = _paired_frames(dataset_key, int(seed), predictions)
            target = paired["target_index"].to_numpy(int)
            control_prediction = paired["control_prediction"].to_numpy(int)
            primary_prediction = paired["primary_prediction"].to_numpy(int)
            control_correct = control_prediction == target
            primary_correct = primary_prediction == target
            sample_correct[CONTROL].append(control_correct)
            sample_correct[PRIMARY].append(primary_correct)
            sample_predictions[CONTROL].append(control_prediction)
            sample_predictions[PRIMARY].append(primary_prediction)
            categories = {
                "both_correct": control_correct & primary_correct,
                "rescued_by_primary": ~control_correct & primary_correct,
                "harmed_by_primary": control_correct & ~primary_correct,
                "both_wrong_same_prediction": (~control_correct & ~primary_correct & (control_prediction == primary_prediction)),
                "both_wrong_changed_prediction": (~control_correct & ~primary_correct & (control_prediction != primary_prediction)),
            }
            for category, mask in categories.items():
                transition_rows.append(
                    {
                        "dataset": dataset_key,
                        "seed": int(seed),
                        "scope": "overall",
                        "label": "all",
                        "category": category,
                        "count": int(mask.sum()),
                        "rate": float(mask.mean()),
                    }
                )
            for class_index, label in enumerate(labels):
                class_mask = target == class_index
                for category, mask in {
                    "rescued_by_primary": ~control_correct & primary_correct,
                    "harmed_by_primary": control_correct & ~primary_correct,
                }.items():
                    transition_rows.append(
                        {
                            "dataset": dataset_key,
                            "seed": int(seed),
                            "scope": "class",
                            "label": label,
                            "category": category,
                            "count": int((class_mask & mask).sum()),
                            "rate": float((class_mask & mask).sum() / class_mask.sum()),
                        }
                    )
            for true_index, true_label in enumerate(labels):
                for predicted_index, predicted_label in enumerate(labels):
                    mask = target == true_index
                    control_count = int((mask & (control_prediction == predicted_index)).sum())
                    primary_count = int((mask & (primary_prediction == predicted_index)).sum())
                    confusion_rows.append(
                        {
                            "dataset": dataset_key,
                            "seed": int(seed),
                            "true_class": true_label,
                            "predicted_class": predicted_label,
                            "support": int(mask.sum()),
                            "control_count": control_count,
                            "primary_count": primary_count,
                            "count_delta": primary_count - control_count,
                        }
                    )
        control_frequency = np.mean(np.stack(sample_correct[CONTROL]), axis=0)
        primary_frequency = np.mean(np.stack(sample_correct[PRIMARY]), axis=0)
        modal_consistency = {}
        for candidate in (CONTROL, PRIMARY):
            matrix = np.stack(sample_predictions[candidate])
            consistency = [
                np.bincount(matrix[:, index], minlength=len(labels)).max() / matrix.shape[0]
                for index in range(matrix.shape[1])
            ]
            modal_consistency[candidate] = float(np.mean(consistency))
        stability[dataset_key] = {
            "samples_improved_correct_frequency": int((primary_frequency > control_frequency).sum()),
            "samples_harmed_correct_frequency": int((primary_frequency < control_frequency).sum()),
            "samples_unchanged_correct_frequency": int((primary_frequency == control_frequency).sum()),
            "modal_prediction_consistency_control": modal_consistency[CONTROL],
            "modal_prediction_consistency_primary": modal_consistency[PRIMARY],
        }
    transitions = pd.DataFrame(transition_rows).groupby(
        ["dataset", "scope", "label", "category"], as_index=False
    ).agg(mean_count=("count", "mean"), mean_rate=("rate", "mean"))
    confusions = pd.DataFrame(confusion_rows).groupby(
        ["dataset", "true_class", "predicted_class"], as_index=False
    ).agg(
        support=("support", "first"),
        control_count_mean=("control_count", "mean"),
        primary_count_mean=("primary_count", "mean"),
        count_delta_mean=("count_delta", "mean"),
    )
    return transitions, confusions, stability


def _slice_outputs(
    lock: dict[str, Any],
    datasets: dict[str, Any],
    predictions: dict[tuple[str, str, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    leave_one_out_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for dataset_key, dataset in datasets.items():
        test = dataset.frame.loc[dataset.frame["split"] == "test"].copy()
        metadata = test.set_index(test["row_id"].astype(str))
        unit_column = "event_name" if dataset_key == "crisismmd_humanitarian" else "group_id"
        units = sorted(test[unit_column].astype(str).unique())
        class_indices = np.arange(len(dataset.labels))
        unit_seed_rows: list[dict[str, Any]] = []
        for seed in lock["projection_seeds"]:
            paired = _paired_frames(dataset_key, int(seed), predictions).set_index("row_id")
            paired[unit_column] = metadata.loc[paired.index, unit_column].astype(str).to_numpy()
            if dataset_key == "uk_crisis_type":
                paired["source_query"] = metadata.loc[paired.index, "source_query"].astype(str).to_numpy()
            for unit in units:
                inside = paired[unit_column] == unit
                outside = ~inside
                target = paired["target_index"].to_numpy(int)
                control_prediction = paired["control_prediction"].to_numpy(int)
                primary_prediction = paired["primary_prediction"].to_numpy(int)
                descriptor = unit
                query_count = 0
                if dataset_key == "uk_crisis_type":
                    queries = sorted(paired.loc[inside, "source_query"].unique())
                    descriptor = " | ".join(queries)
                    query_count = len(queries)
                unit_seed_rows.append(
                    {
                        "dataset": dataset_key,
                        "unit_type": "event" if dataset_key == "crisismmd_humanitarian" else "component_group",
                        "unit_id": unit,
                        "descriptor": descriptor,
                        "source_query_count": query_count,
                        "seed": int(seed),
                        "n": int(inside.sum()),
                        "control_accuracy": float((control_prediction[inside] == target[inside]).mean()),
                        "primary_accuracy": float((primary_prediction[inside] == target[inside]).mean()),
                        "loo_macro_f1_control": float(
                            f1_score(
                                target[outside],
                                control_prediction[outside],
                                labels=class_indices,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                        "loo_macro_f1_primary": float(
                            f1_score(
                                target[outside],
                                primary_prediction[outside],
                                labels=class_indices,
                                average="macro",
                                zero_division=0,
                            )
                        ),
                    }
                )
        unit_frame = pd.DataFrame(unit_seed_rows)
        unit_frame["accuracy_delta"] = unit_frame["primary_accuracy"] - unit_frame["control_accuracy"]
        unit_frame["loo_macro_f1_delta"] = (
            unit_frame["loo_macro_f1_primary"] - unit_frame["loo_macro_f1_control"]
        )
        group_rows.extend(
            unit_frame.groupby(
                ["dataset", "unit_type", "unit_id", "descriptor", "source_query_count"],
                as_index=False,
            )
            .agg(
                n=("n", "first"),
                control_accuracy_mean=("control_accuracy", "mean"),
                primary_accuracy_mean=("primary_accuracy", "mean"),
                accuracy_delta_mean=("accuracy_delta", "mean"),
            )
            .to_dict("records")
        )
        leave_one_out_rows.extend(
            unit_frame.groupby(
                ["dataset", "unit_type", "unit_id", "descriptor", "source_query_count"],
                as_index=False,
            )
            .agg(
                n_omitted=("n", "first"),
                loo_macro_f1_delta_mean=("loo_macro_f1_delta", "mean"),
                loo_macro_f1_delta_min=("loo_macro_f1_delta", "min"),
                loo_macro_f1_delta_max=("loo_macro_f1_delta", "max"),
                positive_seeds=("loo_macro_f1_delta", lambda values: int((values > 0).sum())),
            )
            .to_dict("records")
        )

        if dataset_key == "uk_crisis_type":
            prediction_rows = []
            for seed in lock["projection_seeds"]:
                paired = _paired_frames(dataset_key, int(seed), predictions).set_index("row_id")
                paired = paired.join(
                    metadata[["label_name", "quality_stratum", "quality_flag_count"]],
                    how="left",
                )
                paired["seed"] = int(seed)
                prediction_rows.append(paired.reset_index())
            all_predictions = pd.concat(prediction_rows, ignore_index=True)
            slices = [
                ("quality_stratum", ["quality_stratum"]),
                ("quality_flag_count", ["quality_flag_count"]),
                ("class_x_quality", ["label_name", "quality_stratum"]),
            ]
            for slice_type, columns in slices:
                for keys, frame in all_predictions.groupby(columns, dropna=False):
                    key_values = keys if isinstance(keys, tuple) else (keys,)
                    control_accuracy = float(
                        (frame["control_prediction"] == frame["target_index"]).mean()
                    )
                    primary_accuracy = float(
                        (frame["primary_prediction"] == frame["target_index"]).mean()
                    )
                    quality_rows.append(
                        {
                            "dataset": dataset_key,
                            "slice_type": slice_type,
                            "slice_value": " | ".join(str(value) for value in key_values),
                            "n_examples": int(frame["row_id"].nunique()),
                            "control_accuracy": control_accuracy,
                            "primary_accuracy": primary_accuracy,
                            "accuracy_delta": primary_accuracy - control_accuracy,
                        }
                    )
    return pd.DataFrame(group_rows), pd.DataFrame(leave_one_out_rows), pd.DataFrame(quality_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the locked RanPAC confirmation archive.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis_outputs")
    args = parser.parse_args()
    archive = args.archive.resolve()
    workspace = args.workspace_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as zf:
        contract, lock, inference, results, predictions, datasets = _load_and_validate(
            zf, workspace
        )
        summary = _csv(zf, f"{SUITE}/confirmation/confirmation_summary.csv")
        per_class = _csv(zf, f"{SUITE}/confirmation/confirmation_per_class.csv")
        runs = _csv(zf, f"{SUITE}/confirmation/confirmation_runs.csv")
        member_count = len(zf.namelist())

    expected_runs = len(lock["datasets"]) * len(lock["candidate_ids"]) * len(
        lock["projection_seeds"]
    )
    if len(results) != expected_runs or len(runs) != expected_runs:
        raise ValueError("confirmation archive does not contain the locked run matrix")
    if len(summary) != len(lock["datasets"]) * len(lock["candidate_ids"]):
        raise ValueError("confirmation summary has an unexpected row count")

    overall = pd.DataFrame(_overall_rows(lock, results, predictions))
    summary_check = overall.merge(
        summary[["dataset", "candidate_id", "test_macro_f1_mean", "test_accuracy_mean"]],
        on=["dataset", "candidate_id"],
        validate="one_to_one",
    )
    if not np.allclose(summary_check["macro_f1_mean"], summary_check["test_macro_f1_mean"]):
        raise ValueError("derived Macro-F1 summary differs from the archived summary")
    if not np.allclose(summary_check["accuracy_mean"], summary_check["test_accuracy_mean"]):
        raise ValueError("derived accuracy summary differs from the archived summary")
    per_class_delta = _per_class_deltas(per_class)
    transitions, confusions, stability = _transition_outputs(lock, datasets, predictions)
    groups, leave_one_out, quality = _slice_outputs(lock, datasets, predictions)
    contrast_rows = []
    for dataset_key, dataset_inference in inference["datasets"].items():
        contrast = dataset_inference["primary_vs_control"]
        deltas = np.asarray(contrast["seed_deltas"], dtype=np.float64)
        contrast_rows.append(
            {
                "dataset": dataset_key,
                "primary_candidate": contrast["primary_candidate"],
                "control_candidate": contrast["control_candidate"],
                "paired_seeds": len(deltas),
                "delta_mean": float(deltas.mean()),
                "delta_sd": float(deltas.std(ddof=1)),
                "paired_effect_dz": float(deltas.mean() / deltas.std(ddof=1)),
                "delta_median": float(np.median(deltas)),
                "delta_min": float(deltas.min()),
                "delta_max": float(deltas.max()),
                "positive_seeds": int((deltas > 0).sum()),
                "group_bootstrap_ci_lo": contrast["ci_lo"],
                "group_bootstrap_ci_hi": contrast["ci_hi"],
                "bootstrap_probability_gt_zero": contrast[
                    "bootstrap_probability_gt_zero"
                ],
                "exact_sign_flip_p_value": contrast["exact_sign_flip_p_value"],
                "holm_adjusted_sign_flip_p_value": contrast[
                    "holm_adjusted_sign_flip_p_value"
                ],
            }
        )
    contrasts = pd.DataFrame(contrast_rows)

    overall.to_csv(output / "ranpac_confirmation_overall.csv", index=False)
    contrasts.to_csv(output / "ranpac_confirmation_paired_contrasts.csv", index=False)
    per_class_delta.to_csv(output / "ranpac_confirmation_per_class_deltas.csv", index=False)
    transitions.to_csv(output / "ranpac_confirmation_transitions.csv", index=False)
    confusions.to_csv(output / "ranpac_confirmation_confusion_deltas.csv", index=False)
    groups.to_csv(output / "ranpac_confirmation_group_sensitivity.csv", index=False)
    leave_one_out.to_csv(output / "ranpac_confirmation_leave_one_group_out.csv", index=False)
    quality.to_csv(output / "ranpac_confirmation_quality_sensitivity.csv", index=False)

    integrity = {
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _file_sha256(archive),
        "zip_members": member_count,
        "unsafe_members": 0,
        "contract_sha256": contract["contract_sha256"],
        "lock_sha256": lock["lock_sha256"],
        "dataset_fingerprints": {
            key: dataset.fingerprint for key, dataset in datasets.items()
        },
        "projection_seeds": lock["projection_seeds"],
        "candidate_ids": lock["candidate_ids"],
        "expected_runs": expected_runs,
        "validated_result_prediction_pairs": len(results),
        "summary_rows": len(summary),
        "per_class_rows": len(per_class),
        "run_rows": len(runs),
        "status": "valid",
    }
    (output / "ranpac_confirmation_integrity.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    diagnostics = {
        "integrity": integrity,
        "official_inference": inference,
        "sample_stability": stability,
        "test_group_counts": {
            key: int(dataset.frame.loc[dataset.frame["split"] == "test", "group_id"].nunique())
            for key, dataset in datasets.items()
        },
        "test_class_support": {
            key: {
                str(label): int(count)
                for label, count in dataset.frame.loc[
                    dataset.frame["split"] == "test", "label_name"
                ].value_counts().sort_index().items()
            }
            for key, dataset in datasets.items()
        },
    }
    (output / "ranpac_confirmation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
