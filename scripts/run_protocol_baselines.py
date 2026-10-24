import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


OUT = Path("analysis_outputs")
MANIFEST_PATH = OUT / "dataset_version_manifest.csv"
ASSIGNMENTS_PATH = OUT / "dataset_version_split_assignments.csv"
CONSTRAINED_ASSIGNMENTS_PATH = OUT / "constrained_source_query_split_assignments.csv"
JOINT_ASSIGNMENTS_PATH = OUT / "joint_query_post_split_assignments.csv"
SEED = 2026
BOOTSTRAP_REPLICATES = 300

CLASSES = [
    "major_accidents",
    "natural_disaster",
    "pandemic_health",
    "political_crisis",
    "social_unrest",
    "strikes",
]

PROTOCOLS = [
    ("v2_duplicate_safe", "balanced_original_post_group"),
    ("v2_duplicate_safe", "balanced_source_query_group"),
    ("v2_duplicate_safe", "constrained_source_query_group"),
    ("v2_duplicate_safe", "joint_source_query_original_post_group"),
    ("v3_post_safe_recommended", "balanced_original_post_group"),
    ("v3_post_safe_recommended", "balanced_source_query_group"),
    ("v3_post_safe_recommended", "constrained_source_query_group"),
    ("v3_post_safe_recommended", "joint_source_query_original_post_group"),
]

QUERY_STOPWORDS = {
    "and",
    "or",
    "not",
    "uk",
    "the",
    "a",
    "an",
    "in",
    "of",
    "to",
    "for",
    "on",
    "near",
}


def mask_query_terms(text, query):
    text = "" if pd.isna(text) else str(text)
    query = "" if pd.isna(query) else str(query)
    terms = {
        token.lower().lstrip("#")
        for token in re.findall(r"#?[A-Za-z][A-Za-z0-9_'-]{2,}", query)
        if token.lower().lstrip("#") not in QUERY_STOPWORDS
    }
    if not terms:
        return text
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])#?(?:"
        + "|".join(sorted((re.escape(term) for term in terms), key=len, reverse=True))
        + r")(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", pattern.sub(" <query_term> ", text)).strip()


def load_data():
    manifest = pd.read_csv(MANIFEST_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)
    constrained_assignments = pd.read_csv(CONSTRAINED_ASSIGNMENTS_PATH)
    joint_assignments = pd.read_csv(JOINT_ASSIGNMENTS_PATH)
    assignments = pd.concat(
        [assignments, constrained_assignments, joint_assignments],
        ignore_index=True,
    )
    manifest["text"] = manifest["text"].fillna("").astype(str)
    manifest["platform"] = manifest["platform"].fillna("<missing>").astype(str)
    manifest["source_query"] = manifest["source_query"].fillna("<missing>").astype(str)
    manifest["query_masked_text"] = [
        mask_query_terms(text, query)
        for text, query in zip(manifest["text"], manifest["source_query"])
    ]
    return manifest, assignments


def make_model(model_name):
    categorical = {
        "platform_only": ["platform"],
        "source_query_only": ["source_query"],
    }
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if model_name in categorical:
        prep = ColumnTransformer(
            [
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical[model_name],
                )
            ]
        )
        return Pipeline(
            [
                ("features", prep),
                ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
            ]
        )

    text_col = "query_masked_text" if model_name == "text_word_char_query_masked" else "text"
    transformers = []
    if model_name in {
        "text_word",
        "text_word_char",
        "text_word_char_query_masked",
        "text_word_char_platform",
        "text_word_char_platform_query",
    }:
        transformers.append(
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=50000,
                    sublinear_tf=True,
                ),
                text_col,
            )
        )
    if model_name in {
        "text_char",
        "text_word_char",
        "text_word_char_query_masked",
        "text_word_char_platform",
        "text_word_char_platform_query",
    }:
        transformers.append(
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=80000,
                    sublinear_tf=True,
                ),
                text_col,
            )
        )
    if model_name in {"text_word_char_platform", "text_word_char_platform_query"}:
        cat_cols = ["platform"]
        if model_name == "text_word_char_platform_query":
            cat_cols.append("source_query")
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            )
        )

    prep = ColumnTransformer(transformers)
    return Pipeline(
        [
            ("features", prep),
            ("classifier", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
        ]
    )


def metric_values(y_true, y_pred):
    _, recalls, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred, labels=CLASSES)),
    }


def stratified_bootstrap_indices(y_true, rng):
    y_true = np.asarray(y_true)
    sampled = []
    for cls in CLASSES:
        idx = np.flatnonzero(y_true == cls)
        if len(idx):
            sampled.append(rng.choice(idx, size=len(idx), replace=True))
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out


def bootstrap_ci(y_true, y_pred, replicates=BOOTSTRAP_REPLICATES):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    values = {name: [] for name in metric_values(y_true, y_pred)}
    for _ in range(replicates):
        idx = stratified_bootstrap_indices(y_true, rng)
        metrics = metric_values(y_true[idx], y_pred[idx])
        for name, value in metrics.items():
            values[name].append(value)
    return {
        name: {
            "lo": float(np.quantile(vals, 0.025)),
            "hi": float(np.quantile(vals, 0.975)),
        }
        for name, vals in values.items()
    }


def evaluate_predictions(y_true, y_pred, include_ci):
    metrics = metric_values(y_true, y_pred)
    if include_ci:
        ci = bootstrap_ci(y_true, y_pred)
        for name, interval in ci.items():
            metrics[f"{name}_ci_lo"] = interval["lo"]
            metrics[f"{name}_ci_hi"] = interval["hi"]
    return metrics


def per_class_rows(version, split_name, evaluation_split, model_name, y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
    predicted_counts = pd.Series(y_pred).value_counts()
    rows = []
    for i, cls in enumerate(CLASSES):
        rows.append(
            {
                "version": version,
                "split_name": split_name,
                "evaluation_split": evaluation_split,
                "model": model_name,
                "class": cls,
                "support": int(support[i]),
                "predicted_n": int(predicted_counts.get(cls, 0)),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
        )
    return rows


def paired_bootstrap_delta(y_true, pred_a, pred_b, metric_name="macro_f1"):
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    rng = np.random.default_rng(SEED + 11)
    deltas = []
    for _ in range(BOOTSTRAP_REPLICATES):
        idx = stratified_bootstrap_indices(y_true, rng)
        a = metric_values(y_true[idx], pred_a[idx])[metric_name]
        b = metric_values(y_true[idx], pred_b[idx])[metric_name]
        deltas.append(a - b)
    return {
        "delta": metric_values(y_true, pred_a)[metric_name]
        - metric_values(y_true, pred_b)[metric_name],
        "ci_lo": float(np.quantile(deltas, 0.025)),
        "ci_hi": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_gt_zero": float(np.mean(np.asarray(deltas) > 0)),
    }


def mcnemar_exact(y_true, pred_a, pred_b):
    y_true = np.asarray(y_true)
    correct_a = np.asarray(pred_a) == y_true
    correct_b = np.asarray(pred_b) == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    p_value = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant_n": discordant,
        "exact_p_value": p_value,
    }


def build_pairwise_rows(version, split_name, predictions):
    comparisons = [
        ("text_word_char", "text_word_char_query_masked", "raw_text_vs_query_masked"),
        ("text_word_char_platform_query", "text_word_char_platform", "query_metadata_ablation"),
        ("text_word_char", "source_query_only", "semantic_text_vs_query_only"),
    ]
    rows = []
    for model_a, model_b, comparison in comparisons:
        if model_a not in predictions or model_b not in predictions:
            continue
        true_a, pred_a = predictions[model_a]
        true_b, pred_b = predictions[model_b]
        if not np.array_equal(true_a, true_b):
            raise ValueError(f"Prediction alignment mismatch: {model_a} vs {model_b}")
        delta = paired_bootstrap_delta(true_a, pred_a, pred_b)
        mcnemar = mcnemar_exact(true_a, pred_a, pred_b)
        rows.append(
            {
                "version": version,
                "split_name": split_name,
                "comparison": comparison,
                "model_a": model_a,
                "model_b": model_b,
                "delta_macro_f1_a_minus_b": delta["delta"],
                "delta_macro_f1_ci_lo": delta["ci_lo"],
                "delta_macro_f1_ci_hi": delta["ci_hi"],
                "bootstrap_probability_delta_gt_zero": delta[
                    "bootstrap_probability_gt_zero"
                ],
                **mcnemar,
            }
        )
    return rows


def main():
    manifest, assignments = load_data()
    model_names = [
        "majority",
        "platform_only",
        "source_query_only",
        "text_word",
        "text_char",
        "text_word_char",
        "text_word_char_query_masked",
        "text_word_char_platform",
        "text_word_char_platform_query",
    ]

    result_rows = []
    class_rows = []
    prediction_rows = []
    pairwise_rows = []
    confusion_payload = {}

    for version, split_name in PROTOCOLS:
        assigned = assignments[
            (assignments["version"] == version)
            & (assignments["split_name"] == split_name)
        ].copy()
        data = manifest.merge(assigned[["row_id", "split"]], on="row_id", how="inner")
        train = data[data["split"] == "train"].copy()
        if train["predicted_crisis_type"].nunique() != len(CLASSES):
            raise ValueError(f"Training split lacks classes: {version} / {split_name}")

        test_predictions = {}
        protocol_key = f"{version}::{split_name}"
        confusion_payload[protocol_key] = {}

        for model_name in model_names:
            model = make_model(model_name)
            feature_cols = ["text", "query_masked_text", "platform", "source_query"]
            model.fit(train[feature_cols], train["predicted_crisis_type"])

            for evaluation_split in ["val", "test"]:
                evaluation = data[data["split"] == evaluation_split].copy()
                y_true = evaluation["predicted_crisis_type"].to_numpy()
                y_pred = model.predict(evaluation[feature_cols])
                metrics = evaluate_predictions(
                    y_true,
                    y_pred,
                    include_ci=evaluation_split == "test",
                )
                result_rows.append(
                    {
                        "version": version,
                        "split_name": split_name,
                        "evaluation_split": evaluation_split,
                        "model": model_name,
                        "train_n": int(len(train)),
                        "evaluation_n": int(len(evaluation)),
                        "train_classes": int(train["predicted_crisis_type"].nunique()),
                        "evaluation_classes": int(
                            evaluation["predicted_crisis_type"].nunique()
                        ),
                        **metrics,
                    }
                )
                class_rows.extend(
                    per_class_rows(
                        version,
                        split_name,
                        evaluation_split,
                        model_name,
                        y_true,
                        y_pred,
                    )
                )
                for row_id, true_label, predicted_label in zip(
                    evaluation["row_id"], y_true, y_pred
                ):
                    prediction_rows.append(
                        {
                            "row_id": int(row_id),
                            "version": version,
                            "split_name": split_name,
                            "evaluation_split": evaluation_split,
                            "model": model_name,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                            "correct": bool(true_label == predicted_label),
                        }
                    )
                if evaluation_split == "test":
                    test_predictions[model_name] = (y_true.copy(), np.asarray(y_pred).copy())
                    confusion_payload[protocol_key][model_name] = confusion_matrix(
                        y_true,
                        y_pred,
                        labels=CLASSES,
                    ).tolist()

        pairwise_rows.extend(
            build_pairwise_rows(version, split_name, test_predictions)
        )

    results = pd.DataFrame(result_rows)
    per_class = pd.DataFrame(class_rows)
    predictions = pd.DataFrame(prediction_rows)
    pairwise = pd.DataFrame(pairwise_rows)

    results.to_csv(OUT / "protocol_baseline_results.csv", index=False)
    per_class.to_csv(OUT / "protocol_baseline_per_class.csv", index=False)
    predictions.to_csv(OUT / "protocol_baseline_predictions.csv", index=False)
    pairwise.to_csv(OUT / "protocol_baseline_pairwise_tests.csv", index=False)

    payload = {
        "configuration": {
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "classes": CLASSES,
            "protocols": [
                {"version": version, "split_name": split_name}
                for version, split_name in PROTOCOLS
            ],
            "training_rule": "Fixed hyperparameters; fit on train only; val and test are reported separately.",
            "query_masking_rule": "Replace query-derived alphabetic tokens of length >=3, excluding logical/generic stopwords, with <query_term>.",
        },
        "test_results": results[results["evaluation_split"] == "test"].to_dict(
            orient="records"
        ),
        "pairwise_tests": pairwise.to_dict(orient="records"),
        "confusion_matrices": {
            "class_order": CLASSES,
            "matrices": confusion_payload,
        },
    }
    with (OUT / "protocol_baseline_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    compact_cols = [
        "version",
        "split_name",
        "model",
        "evaluation_n",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "macro_f1_ci_lo",
        "macro_f1_ci_hi",
    ]
    compact = (
        results[results["evaluation_split"] == "test"][compact_cols]
        .sort_values(["version", "split_name", "macro_f1"], ascending=[True, True, False])
    )
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
