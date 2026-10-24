import json
import math
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import jensenshannon
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("UK_CrisisMMD_Public")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(123)


def load_all():
    records = [json.loads(line) for line in (ROOT / "public_dataset.jsonl").open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    df["text"] = df["text"].fillna("").astype(str)
    df["has_image"] = df["image_path"].notna() & df["image_path"].astype(str).ne("")
    df["modality"] = np.where(df["has_image"], "text+image", "text-only")
    df["char_len"] = df["text"].str.len()
    df["word_count"] = df["text"].map(lambda s: len(re.findall(r"\b\w+\b", s)))
    df["hashtag_count"] = df["hashtags"].map(lambda x: len(x) if isinstance(x, list) else 0)
    df["raw_http"] = df["text"].str.contains("http", case=False, regex=False).astype(int)
    df["raw_www"] = df["text"].str.contains("www", case=False, regex=False).astype(int)
    df["has_mojibake"] = df["text"].str.contains(r"[âÃðÂ�]", regex=True).astype(int)
    df["log_char_len"] = np.log1p(df["char_len"])
    df["log_word_count"] = np.log1p(df["word_count"])
    df["log_hashtag_count"] = np.log1p(df["hashtag_count"])

    whitespace_re = re.compile(r"\s+")
    url_re = re.compile(r"https?\s*:\s*/\s*/\s*\S+|https?://\S+|www\.\S+", re.I)
    mention_re = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,20}")
    nonword_re = re.compile(r"[^\w#@]+", re.UNICODE)

    def normalize_text(s):
        s = s.lower()
        s = url_re.sub(" <url> ", s)
        s = mention_re.sub(" <user> ", s)
        s = nonword_re.sub(" ", s)
        return whitespace_re.sub(" ", s).strip()

    df["normalized_text"] = df["text"].map(normalize_text)

    mapping = pd.read_csv(ROOT / "image_mapping.csv")
    mapping["orig_stem"] = (
        mapping["original_filename"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True).str.replace(r"_\d+$", "", regex=True)
    )
    df = df.merge(mapping[["new_rel_path", "orig_stem", "original_crawl_cat"]], left_on="image_path", right_on="new_rel_path", how="left")

    image_rows = []
    for p in (ROOT / "images").rglob("*.jpg"):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            image_rows.append(
                {
                    "image_path": rel,
                    "width": im.width,
                    "height": im.height,
                    "bytes": p.stat().st_size,
                    "aspect_ratio": im.width / im.height,
                    "megapixels": im.width * im.height / 1_000_000,
                    "sha256": h,
                }
            )
    img = pd.DataFrame(image_rows)
    df = df.merge(img, on="image_path", how="left")
    for col in ["width", "height", "bytes", "aspect_ratio", "megapixels"]:
        df[col] = df[col].astype(float)
    df["log_width"] = np.log1p(df["width"])
    df["log_height"] = np.log1p(df["height"])
    df["log_bytes"] = np.log1p(df["bytes"])
    df["log_megapixels"] = np.log1p(df["megapixels"])
    return df


def bootstrap_metric_ci(y_true, y_pred, metric, n_boot=500):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals.append(metric(y_true[idx], y_pred[idx]))
    return {
        "mean": round(float(np.mean(vals)), 6),
        "lo": round(float(np.quantile(vals, 0.025)), 6),
        "hi": round(float(np.quantile(vals, 0.975)), 6),
    }


def make_pipeline(numeric, categorical, class_weight=None, binary=False):
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    preprocess = ColumnTransformer(transformers)
    base_clf = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        solver="liblinear",
    )
    clf = base_clf if binary else OneVsRestClassifier(base_clf)
    return Pipeline([("prep", preprocess), ("clf", clf)])


def cv_predict_table(df, target, numeric, categorical, name, group_col=None, class_weight=None):
    cols = list(dict.fromkeys(numeric + categorical + [target] + ([group_col] if group_col else [])))
    data = df[cols].dropna().copy()
    y = data[target].astype(str).to_numpy()
    X = data[numeric + categorical]
    classes = np.unique(y)
    if group_col:
        splitter = GroupKFold(n_splits=5).split(X, y, data[group_col].astype(str).to_numpy())
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X, y)

    if name.startswith("dummy"):
        model = DummyClassifier(strategy="most_frequent")
    else:
        model = make_pipeline(numeric, categorical, class_weight=class_weight, binary=len(classes) == 2)

    pred = np.empty(len(data), dtype=object)
    proba_positive = np.full(len(data), np.nan)
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter, 1):
        m = model
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        pred[te] = p
        if len(classes) == 2 and not name.startswith("dummy"):
            probs = m.predict_proba(X.iloc[te])
            cls = list(m.classes_)
            pos_idx = cls.index(sorted(classes)[-1])
            proba_positive[te] = probs[:, pos_idx]
        fold_rows.append(
            {
                "fold": fold,
                "n_test": int(len(te)),
                "accuracy": round(float(accuracy_score(y[te], p)), 6),
                "balanced_accuracy": round(float(balanced_accuracy_score(y[te], p)), 6),
                "macro_f1": round(float(f1_score(y[te], p, average="macro", zero_division=0)), 6),
                "weighted_f1": round(float(f1_score(y[te], p, average="weighted", zero_division=0)), 6),
            }
        )
    overall = {
        "name": name,
        "target": target,
        "n": int(len(data)),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "group_col": group_col,
        "class_weight": str(class_weight),
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 6),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y, pred, average="weighted", zero_division=0)), 6),
        "macro_f1_ci": bootstrap_metric_ci(y, pred, lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)),
        "balanced_accuracy_ci": bootstrap_metric_ci(y, pred, lambda yt, yp: balanced_accuracy_score(yt, yp)),
        "folds": fold_rows,
    }
    if len(classes) == 2 and not np.isnan(proba_positive).all():
        positive = sorted(classes)[-1]
        y_bin = (y == positive).astype(int)
        overall["positive_class_for_auc"] = positive
        overall["roc_auc"] = round(float(roc_auc_score(y_bin, proba_positive)), 6)
    return overall


def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    return float(jensenshannon(p, q, base=2.0) ** 2)


def split_once(df, strategy, target, group_col=None, test_size=0.2, seed=0):
    y = df[target].astype(str).to_numpy()
    if strategy == "stratified_random":
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(df, y))
    elif strategy == "group_shuffle":
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(df, y, df[group_col].astype(str).to_numpy()))
    else:
        raise ValueError(strategy)
    return tr, te


def split_diagnostics(df, target="predicted_crisis_type", n_iter=100):
    labeled = df[df[target].notna()].copy().reset_index(drop=True)
    classes = sorted(labeled[target].astype(str).unique())
    all_props = labeled[target].astype(str).value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
    groups_to_check = {
        "source_query": "source_query",
        "normalized_text": "normalized_text",
        "image_sha": "sha256",
        "original_post": "orig_stem",
    }
    strategies = [
        ("stratified_random", None),
        ("source_query_group", "source_query"),
        ("normalized_text_group", "normalized_text"),
        ("image_sha_group", "sha256"),
        ("original_post_group", "orig_stem"),
    ]
    rows = []
    for strategy_name, group_col in strategies:
        for i in range(n_iter):
            if group_col and labeled[group_col].notna().sum() < len(labeled):
                data = labeled[labeled[group_col].notna()].reset_index(drop=True)
            else:
                data = labeled
            if strategy_name == "stratified_random":
                tr, te = split_once(data, "stratified_random", target, seed=1000 + i)
            else:
                tr, te = split_once(data, "group_shuffle", target, group_col=group_col, seed=1000 + i)
            train = data.iloc[tr]
            test = data.iloc[te]
            train_props = train[target].astype(str).value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
            test_props = test[target].astype(str).value_counts(normalize=True).reindex(classes, fill_value=0).to_numpy()
            rec = {
                "strategy": strategy_name,
                "iteration": i + 1,
                "n_total_used": int(len(data)),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "test_size_rate": round(float(len(test) / len(data)), 6),
                "js_train_test": round(js_divergence(train_props, test_props), 6),
                "js_test_overall": round(js_divergence(test_props, all_props), 6),
                "max_abs_class_prop_diff_train_test": round(float(np.max(np.abs(train_props - test_props))), 6),
                "missing_classes_in_test": int(sum(test[target].astype(str).value_counts().reindex(classes, fill_value=0) == 0)),
            }
            train_class_counts = train[target].astype(str).value_counts().reindex(classes, fill_value=0)
            test_class_counts = test[target].astype(str).value_counts().reindex(classes, fill_value=0)
            for cls in classes:
                rec[f"test_count_{cls}"] = int(test_class_counts[cls])
                rec[f"test_prop_{cls}"] = round(float(test_class_counts[cls] / len(test)), 6)
                rec[f"train_count_{cls}"] = int(train_class_counts[cls])
            for label, col in groups_to_check.items():
                subset_train = set(train[col].dropna().astype(str))
                subset_test = test[col].dropna().astype(str)
                if len(subset_test) == 0:
                    leak_rate = np.nan
                    unseen_rate = np.nan
                else:
                    leak_rate = subset_test.isin(subset_train).mean()
                    unseen_rate = 1 - leak_rate
                rec[f"{label}_test_rows_seen_in_train_rate"] = round(float(leak_rate), 6) if not np.isnan(leak_rate) else None
                rec[f"{label}_test_rows_unseen_in_train_rate"] = round(float(unseen_rate), 6) if not np.isnan(unseen_rate) else None
            rows.append(rec)
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby("strategy")
        .agg(
            n_total_used_mean=("n_total_used", "mean"),
            n_test_mean=("n_test", "mean"),
            js_train_test_mean=("js_train_test", "mean"),
            js_train_test_p95=("js_train_test", lambda s: s.quantile(0.95)),
            js_test_overall_mean=("js_test_overall", "mean"),
            max_abs_class_prop_diff_mean=("max_abs_class_prop_diff_train_test", "mean"),
            max_abs_class_prop_diff_p95=("max_abs_class_prop_diff_train_test", lambda s: s.quantile(0.95)),
            missing_classes_in_test_mean=("missing_classes_in_test", "mean"),
            source_query_leak_mean=("source_query_test_rows_seen_in_train_rate", "mean"),
            normalized_text_leak_mean=("normalized_text_test_rows_seen_in_train_rate", "mean"),
            image_sha_leak_mean=("image_sha_test_rows_seen_in_train_rate", "mean"),
            original_post_leak_mean=("original_post_test_rows_seen_in_train_rate", "mean"),
        )
        .reset_index()
    )
    return raw, summary


def main():
    df = load_all()
    labeled = df[df["predicted_crisis_type"].notna()].copy()
    surface_num = ["log_char_len", "log_word_count", "log_hashtag_count", "raw_http", "raw_www", "has_mojibake"]
    image_num = ["log_width", "log_height", "log_bytes", "aspect_ratio", "log_megapixels"]

    baselines = []
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", [], [], "dummy_majority_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", surface_num, [], "text_surface_to_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", [], ["platform"], "platform_to_crisis_logreg"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", image_num, [], "image_metadata_to_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", image_num, ["platform"], "image_metadata_plus_platform_to_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", surface_num + image_num, ["platform"], "surface_image_platform_to_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", [], ["source_query"], "source_query_onehot_to_crisis"))
    baselines.append(cv_predict_table(labeled, "predicted_crisis_type", [], ["source_query"], "source_query_onehot_groupheldout_to_crisis", group_col="source_query"))

    baselines.append(cv_predict_table(labeled, "predicted_informativeness", [], [], "dummy_majority_informativeness"))
    baselines.append(cv_predict_table(labeled, "predicted_informativeness", surface_num, [], "text_surface_to_informativeness"))
    baselines.append(cv_predict_table(labeled, "predicted_informativeness", image_num, [], "image_metadata_to_informativeness"))
    baselines.append(cv_predict_table(labeled, "predicted_informativeness", [], ["predicted_crisis_type"], "crisis_type_to_informativeness_logreg"))
    baselines.append(cv_predict_table(labeled, "predicted_informativeness", [], ["source_query"], "source_query_onehot_to_informativeness"))
    baselines.append(
        cv_predict_table(
            labeled,
            "predicted_informativeness",
            surface_num + image_num,
            ["platform", "predicted_crisis_type"],
            "surface_image_platform_crisis_to_informativeness",
        )
    )

    df["has_image_label"] = np.where(df["has_image"], "has_image", "text_only")
    baselines.append(cv_predict_table(df, "has_image_label", surface_num, ["platform"], "platform_surface_to_image_availability"))
    baselines.append(cv_predict_table(df, "has_image_label", [], ["platform"], "platform_to_image_availability_logreg"))

    baseline_rows = []
    for b in baselines:
        row = {
            "name": b["name"],
            "target": b["target"],
            "n": b["n"],
            "group_col": b["group_col"],
            "accuracy": b["accuracy"],
            "balanced_accuracy": b["balanced_accuracy"],
            "macro_f1": b["macro_f1"],
            "weighted_f1": b["weighted_f1"],
            "macro_f1_ci_lo": b["macro_f1_ci"]["lo"],
            "macro_f1_ci_hi": b["macro_f1_ci"]["hi"],
            "balanced_accuracy_ci_lo": b["balanced_accuracy_ci"]["lo"],
            "balanced_accuracy_ci_hi": b["balanced_accuracy_ci"]["hi"],
        }
        if "roc_auc" in b:
            row["roc_auc"] = b["roc_auc"]
            row["positive_class_for_auc"] = b["positive_class_for_auc"]
        baseline_rows.append(row)
    baseline_df = pd.DataFrame(baseline_rows)

    split_raw, split_summary = split_diagnostics(df, n_iter=50)

    output = {
        "artifact_model_baselines": baselines,
        "split_diagnostics_summary": split_summary.to_dict(orient="records"),
        "notes": {
            "model": "LogisticRegression with 5-fold CV; numeric features standardized; categorical features one-hot encoded.",
            "split_diagnostics": "50 Monte Carlo 80/20 splits per strategy. Leakage rates mean percent of test rows whose group value is present in train.",
        },
    }
    with (OUT / "model_artifact_split_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    baseline_df.to_csv(OUT / "model_artifact_baselines.csv", index=False)
    split_raw.to_csv(OUT / "split_diagnostics_raw.csv", index=False)
    split_summary.to_csv(OUT / "split_diagnostics_summary.csv", index=False)

    compact = {
        "top_model_baselines": baseline_df.sort_values(["target", "macro_f1"], ascending=[True, False]).groupby("target").head(5).to_dict(orient="records"),
        "split_summary": split_summary.to_dict(orient="records"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
