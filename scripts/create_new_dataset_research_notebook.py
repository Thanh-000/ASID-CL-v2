import json
from pathlib import Path


NOTEBOOK_PATH = Path("new_dataset_research_experiment_from_scratch.ipynb")


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    md(
        """# New Dataset Research Experiment Notebook — From Scratch

Notebook này xây dựng lại thí nghiệm nghiên cứu cho dataset mới từ đầu, độc lập với các notebook cũ.

## Mục tiêu

1. Tải/đọc dataset mới.
2. Chuẩn hóa schema và tách labeled subset.
3. Audit label availability, imbalance, modality confounding, duplicate/conflict.
4. Phân tích source-query artifact và crisis type × informativeness dependency.
5. Chạy baseline nhẹ, tái lập nhanh: TF-IDF + Logistic Regression.
6. Chạy robust splits: random, text-group-held-out, source-query-held-out.
7. Chạy artifact-only và text ablation baselines.
8. Chạy low-evidence proxy.
9. Xuất bảng CSV và Markdown report-ready findings.

## Lưu ý khoa học

- Notebook này dùng baseline nhẹ để nghiên cứu cấu trúc dataset và artifact.  
- Baseline học thuật mạnh hơn nên là `CrisisTransformers CT-M1-Complete fine-tuning`.  
- Không dùng `image_path`/modality làm feature supervised vì dataset bị modality-label confounding.  
- `source_query` được dùng để audit/heldout, không nên dùng như feature chính nếu mục tiêu là semantic generalization.
"""
    ),
    code(
        """# ============================================================
# 1. Setup
# ============================================================

import os
import re
import json
import math
import gzip
import zipfile
import tarfile
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)
pd.set_option("display.max_colwidth", 180)

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("./new_dataset_research_from_scratch")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"
for p in [DATA_DIR, OUT_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

VERSION_ID = "new_dataset_research_from_scratch_v1"
print("VERSION_ID:", VERSION_ID)
print("BASE_DIR:", BASE_DIR.resolve())
"""
    ),
    code(
        """# ============================================================
# 2. Dataset source configuration
# ============================================================

# Nếu bạn đã có file dataset local, đặt đường dẫn ở đây.
# Ví dụ: LOCAL_DATASET_PATH = r"C:\\path\\to\\public_dataset.jsonl"
LOCAL_DATASET_PATH = ""

# URL lấy từ các notebook nghiên cứu trước. Nếu URL hết hạn, hãy dùng LOCAL_DATASET_PATH.
NEW_DATASET_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1SIbVIKp_mCsageSqqOu-D-jlCnlleOJj&export=download&authuser=0"
)

EXPECTED_COLUMNS = [
    "platform",
    "text",
    "image_path",
    "source_query",
    "hashtags",
    "predicted_crisis_type",
    "predicted_informativeness",
]

print("LOCAL_DATASET_PATH:", LOCAL_DATASET_PATH or "<not set>")
print("Expected columns:", EXPECTED_COLUMNS)
"""
    ),
    code(
        """# ============================================================
# 3. Robust file readers and dataset loader
# ============================================================

def save_csv(df, name, index=False):
    path = OUT_DIR / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print("saved:", path)
    return path

def save_markdown(text, name):
    path = OUT_DIR / name
    path.write_text(text, encoding="utf-8")
    print("saved:", path)
    return path

def read_any_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".csv" or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suffix in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path, lines=True)
        except Exception:
            return pd.read_json(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

def download_if_needed(url, out_path):
    import urllib.request
    out_path = Path(out_path)
    if out_path.exists() and out_path.stat().st_size > 0:
        print("Using existing file:", out_path)
        return out_path
    print("Downloading:", url)
    try:
        urllib.request.urlretrieve(url, out_path)
        print("Downloaded:", out_path, "size=", out_path.stat().st_size)
        return out_path
    except Exception as e:
        print("Download failed:", e)
        return None

def find_candidate_tables(root):
    root = Path(root)
    exts = {".csv", ".tsv", ".txt", ".jsonl", ".json", ".xlsx", ".xls", ".parquet"}
    rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            rows.append({"path": str(p), "name": p.name, "size": p.stat().st_size})
    return pd.DataFrame(rows).sort_values("size", ascending=False) if rows else pd.DataFrame()

def standardize_new_dataset(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Handle generic/no-header files that arrive as 7 unnamed columns.
    if len(df.columns) >= 7 and not set(EXPECTED_COLUMNS).issubset(df.columns):
        lower_cols = [str(c).lower() for c in df.columns]
        generic = all(c.startswith("unnamed") or c.isdigit() for c in lower_cols[:7])
        if generic:
            df = df.iloc[:, :7].copy()
            df.columns = EXPECTED_COLUMNS

    # If columns are present but order is odd, keep only expected + extras.
    missing = [c for c in ["text", "predicted_crisis_type", "predicted_informativeness"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}; available={list(df.columns)}")

    for c in EXPECTED_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    df["text"] = df["text"].fillna("").astype(str)
    df["predicted_crisis_type"] = df["predicted_crisis_type"].fillna("").astype(str).str.strip().str.lower().str.replace(r"\\s+", "_", regex=True)
    df["predicted_informativeness"] = df["predicted_informativeness"].fillna("").astype(str).str.strip().str.lower().str.replace(r"\\s+", "_", regex=True)
    df["source_query"] = df["source_query"].fillna("").astype(str)
    df["hashtags"] = df["hashtags"].fillna("").astype(str)
    df["platform"] = df["platform"].fillna("").astype(str).str.lower()
    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["row_id"] = np.arange(len(df))
    return df

dataset_path = Path(LOCAL_DATASET_PATH) if LOCAL_DATASET_PATH else None
if dataset_path and dataset_path.exists():
    raw_df = read_any_table(dataset_path)
else:
    downloaded = download_if_needed(NEW_DATASET_URL, DATA_DIR / "new_dataset_download")
    if downloaded is None:
        raise RuntimeError("Could not download dataset. Set LOCAL_DATASET_PATH manually.")
    # Try direct read first; if it fails, ask user to provide local path.
    try:
        raw_df = read_any_table(downloaded)
    except Exception as e:
        raise RuntimeError(f"Downloaded file could not be read directly: {e}. Set LOCAL_DATASET_PATH manually.")

df = standardize_new_dataset(raw_df)
print("Loaded rows:", len(df))
display(df.head())
save_csv(pd.DataFrame({"column": df.columns}), "00_loaded_columns.csv")
"""
    ),
    code(
        """# ============================================================
# 4. Basic dataset audit
# ============================================================

def is_missing_label(s):
    s = s.fillna("").astype(str).str.strip().str.lower()
    return s.eq("") | s.isin(["nan", "none", "null", "__missing__", "missing"])

crisis_missing = is_missing_label(df["predicted_crisis_type"])
info_missing = is_missing_label(df["predicted_informativeness"])
labeled_mask = (~crisis_missing) & (~info_missing)
labeled_df = df[labeled_mask].copy()
unlabeled_df = df[~labeled_mask].copy()

audit = pd.DataFrame([
    {"metric": "full_rows", "value": len(df)},
    {"metric": "labeled_rows", "value": len(labeled_df)},
    {"metric": "unlabeled_rows", "value": len(unlabeled_df)},
    {"metric": "labeled_percent", "value": 100 * len(labeled_df) / max(len(df), 1)},
    {"metric": "num_columns", "value": df.shape[1]},
    {"metric": "has_timestamp", "value": "timestamp" in df.columns},
    {"metric": "has_event_id", "value": "event_id" in df.columns},
    {"metric": "num_source_queries_labeled", "value": labeled_df["source_query"].nunique()},
])

display(audit)
save_csv(audit, "01_dataset_basic_audit.csv")

print("Supervised experiments must use labeled_df only:", labeled_df.shape)
"""
    ),
    code(
        """# ============================================================
# 5. Label distribution, imbalance, entropy, majority baselines
# ============================================================

def entropy_bits(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())

def majority_baselines(counts):
    counts = list(counts)
    n = sum(counts)
    k = len(counts)
    maj = max(counts)
    maj_acc = maj / n
    precision = maj / n
    recall = 1.0
    maj_f1 = 2 * precision * recall / (precision + recall)
    return {
        "majority_accuracy": maj_acc,
        "majority_macro_f1": maj_f1 / k,
        "majority_weighted_f1": (maj / n) * maj_f1,
    }

def distribution_report(series, task_name):
    vc = series.value_counts()
    out = vc.rename_axis("label").reset_index(name="count")
    out["percent"] = 100 * out["count"] / out["count"].sum()
    counts = out["count"].tolist()
    ent = entropy_bits(counts)
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 0
    summary = {
        "task_name": task_name,
        "num_classes": len(counts),
        "total": sum(counts),
        "majority_class": out.iloc[0]["label"],
        "majority_count": int(out.iloc[0]["count"]),
        "minority_class": out.iloc[-1]["label"],
        "minority_count": int(out.iloc[-1]["count"]),
        "imbalance_ratio": float(out.iloc[0]["count"] / out.iloc[-1]["count"]),
        "entropy_bits": ent,
        "normalized_entropy": ent / max_ent if max_ent else np.nan,
        "effective_num_classes": 2 ** ent,
    }
    summary.update(majority_baselines(counts))
    return out, pd.DataFrame([summary])

crisis_dist, crisis_summary = distribution_report(labeled_df["predicted_crisis_type"], "crisis_type")
info_dist, info_summary = distribution_report(labeled_df["predicted_informativeness"], "informativeness")
imbalance_summary = pd.concat([crisis_summary, info_summary], ignore_index=True)

display(crisis_dist)
display(info_dist)
display(imbalance_summary)

save_csv(crisis_dist, "02_crisis_type_distribution.csv")
save_csv(info_dist, "03_informativeness_distribution.csv")
save_csv(imbalance_summary, "04_imbalance_entropy_majority_baselines.csv")
"""
    ),
    code(
        """# ============================================================
# 6. Modality/platform confounding audit
# ============================================================

work = df.copy()
work["label_state"] = np.where(labeled_mask, "labeled", "unlabeled")
work["has_image"] = work["image_path"].fillna("").astype(str).str.strip().ne("")
work["modality"] = np.where(work["has_image"], "text+image", "text")

platform_dist = work["platform"].value_counts(dropna=False).rename_axis("platform").reset_index(name="count")
platform_dist["percent"] = 100 * platform_dist["count"] / len(work)

modality_dist = work["modality"].value_counts().rename_axis("modality").reset_index(name="count")
modality_dist["percent"] = 100 * modality_dist["count"] / len(work)

label_modality = pd.crosstab(work["label_state"], work["modality"], margins=True)

display(platform_dist)
display(modality_dist)
display(label_modality)

save_csv(platform_dist, "05_platform_distribution.csv")
save_csv(modality_dist, "06_modality_distribution.csv")
save_csv(label_modality.reset_index(), "07_label_state_by_modality.csv")
"""
    ),
    code(
        """# ============================================================
# 7. Crisis type × informativeness dependency
# ============================================================

cross = pd.crosstab(
    labeled_df["predicted_crisis_type"],
    labeled_df["predicted_informativeness"],
    margins=True
)
row_pct = pd.crosstab(
    labeled_df["predicted_crisis_type"],
    labeled_df["predicted_informativeness"],
    normalize="index"
) * 100

display(cross)
display(row_pct)

# Chi-square and Cramer's V.
try:
    from scipy.stats import chi2_contingency
    mat = pd.crosstab(labeled_df["predicted_crisis_type"], labeled_df["predicted_informativeness"])
    chi2, p, dof, expected = chi2_contingency(mat)
    n = mat.to_numpy().sum()
    cramers_v = math.sqrt(chi2 / (n * min(mat.shape[0] - 1, mat.shape[1] - 1)))
    dep = pd.DataFrame([{"chi_square": chi2, "p_value": p, "dof": dof, "cramers_v": cramers_v}])
except Exception as e:
    dep = pd.DataFrame([{"chi_square": np.nan, "p_value": np.nan, "dof": np.nan, "cramers_v": np.nan, "error": str(e)}])

display(dep)
save_csv(cross.reset_index(), "08_crisis_type_by_informativeness_counts.csv")
save_csv(row_pct.reset_index(), "09_crisis_type_by_informativeness_row_percent.csv")
save_csv(dep, "10_crisis_informativeness_dependency_stats.csv")
"""
    ),
    code(
        """# ============================================================
# 8. Duplicate, conflict, and weak-label audit
# ============================================================

def normalize_text(s):
    return re.sub(r"\\s+", " ", str(s).lower()).strip()

labeled_df["text_norm"] = labeled_df["text"].map(normalize_text)
df["text_norm"] = df["text"].map(normalize_text)

dup_summary = pd.DataFrame([
    {"field": "text_norm_full", "duplicate_count": int(df["text_norm"].duplicated().sum()), "duplicate_rate": float(df["text_norm"].duplicated().mean())},
    {"field": "text_norm_labeled", "duplicate_count": int(labeled_df["text_norm"].duplicated().sum()), "duplicate_rate": float(labeled_df["text_norm"].duplicated().mean())},
    {"field": "image_path_full", "duplicate_count": int(df["image_path"].duplicated().sum()), "duplicate_rate": float(df["image_path"].duplicated().mean())},
])

conflict_rows = []
for text_key, g in labeled_df.groupby("text_norm"):
    crisis_labels = sorted(g["predicted_crisis_type"].dropna().unique())
    info_labels = sorted(g["predicted_informativeness"].dropna().unique())
    if len(crisis_labels) > 1 or len(info_labels) > 1:
        conflict_rows.append({
            "text_norm": text_key,
            "rows": len(g),
            "crisis_labels": "|".join(crisis_labels),
            "informativeness_labels": "|".join(info_labels),
        })
conflicts = pd.DataFrame(conflict_rows)
conflict_summary = pd.DataFrame([{
    "unique_text_keys_labeled": labeled_df["text_norm"].nunique(),
    "conflict_keys": len(conflicts),
    "conflict_rate": len(conflicts) / max(labeled_df["text_norm"].nunique(), 1),
}])

display(dup_summary)
display(conflict_summary)
display(conflicts.head(20))

save_csv(dup_summary, "11_duplicate_summary.csv")
save_csv(conflict_summary, "12_label_conflict_summary.csv")
save_csv(conflicts, "13_label_conflict_examples.csv")
"""
    ),
    code(
        """# ============================================================
# 9. Source-query artifact audit
# ============================================================

query_rows = []
for q, g in labeled_df.groupby("source_query"):
    if str(q).strip() == "":
        continue
    vc = g["predicted_crisis_type"].value_counts()
    top_label = vc.index[0]
    top_count = int(vc.iloc[0])
    query_rows.append({
        "source_query": q,
        "rows": len(g),
        "top_crisis_type": top_label,
        "top_count": top_count,
        "top_purity": top_count / len(g),
        "num_crisis_types": g["predicted_crisis_type"].nunique(),
    })

query_purity = pd.DataFrame(query_rows).sort_values(["rows", "top_purity"], ascending=[False, False])
query_summary = pd.DataFrame([{
    "num_labeled_source_queries": len(query_purity),
    "queries_ge_20_rows": int((query_purity["rows"] >= 20).sum()) if len(query_purity) else 0,
    "queries_100pct_pure_ge_20_rows": int(((query_purity["rows"] >= 20) & (query_purity["top_purity"] >= 1.0)).sum()) if len(query_purity) else 0,
    "mean_top_purity": float(query_purity["top_purity"].mean()) if len(query_purity) else np.nan,
    "median_top_purity": float(query_purity["top_purity"].median()) if len(query_purity) else np.nan,
}])

display(query_summary)
display(query_purity.head(30))

save_csv(query_summary, "14_source_query_artifact_summary.csv")
save_csv(query_purity, "15_source_query_purity_table.csv")
"""
    ),
    code(
        """# ============================================================
# 10. Split utilities for baselines
# ============================================================

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

def make_random_split(d, label_col):
    dd = d.copy()
    dd = dd[dd[label_col].map(dd[label_col].value_counts()).ge(2)]
    train, test = train_test_split(dd, test_size=0.25, random_state=SEED, stratify=dd[label_col])
    return train, test, "ok"

def make_group_split(d, label_col, group_col):
    dd = d.copy()
    dd = dd[dd[label_col].map(dd[label_col].value_counts()).ge(2)]
    dd = dd[dd[group_col].fillna("").astype(str).str.len() > 0]
    if len(dd) < 100 or dd[group_col].nunique() < 3 or dd[label_col].nunique() < 2:
        return None, None, "insufficient_group_data"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, te_idx = next(splitter.split(dd, dd[label_col], groups=dd[group_col]))
    train, test = dd.iloc[tr_idx].copy(), dd.iloc[te_idx].copy()
    train_labels = set(train[label_col])
    test = test[test[label_col].isin(train_labels)].copy()
    if len(test) < 20 or test[label_col].nunique() < 2:
        return None, None, "insufficient_test_after_unseen_label_filter"
    return train, test, "ok"

def train_eval_tfidf_lr(train, test, input_col, label_col, run_name):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)),
    ])
    model.fit(train[input_col].fillna("").astype(str), train[label_col].astype(str))
    pred = model.predict(test[input_col].fillna("").astype(str))
    metrics = compute_metrics(test[label_col].astype(str), pred)
    result = {
        "run_name": run_name,
        "input_col": input_col,
        "label_col": label_col,
        "rows_train": len(train),
        "rows_test": len(test),
        "num_train_labels": train[label_col].nunique(),
        "num_test_labels": test[label_col].nunique(),
    }
    result.update(metrics)
    report = pd.DataFrame(classification_report(test[label_col].astype(str), pred, output_dict=True, zero_division=0)).T.reset_index().rename(columns={"index": "label"})
    report["run_name"] = run_name
    return result, report
"""
    ),
    code(
        """# ============================================================
# 11. Feature construction and baseline suite
# ============================================================

exp_df = labeled_df.copy()
exp_df["text_only"] = exp_df["text"].fillna("").astype(str)
exp_df["source_query_only"] = exp_df["source_query"].fillna("").astype(str)
exp_df["hashtag_only"] = exp_df["hashtags"].fillna("").astype(str)
exp_df["platform_only"] = exp_df["platform"].fillna("").astype(str)
exp_df["text_plus_hashtags"] = exp_df["text_only"] + " " + exp_df["hashtag_only"]
exp_df["full_text_hashtags_source"] = exp_df["text_only"] + " " + exp_df["hashtag_only"] + " " + exp_df["source_query_only"]
exp_df["text_without_hashtags"] = exp_df["text_only"].str.replace(r"#\\w+", " ", regex=True)

feature_cols = [
    "text_only",
    "text_without_hashtags",
    "text_plus_hashtags",
    "full_text_hashtags_source",
    "source_query_only",
    "hashtag_only",
    "platform_only",
]

target_cols = ["predicted_crisis_type", "predicted_informativeness"]
split_specs = [
    ("random", None),
    ("text_group_heldout", "text_norm"),
    ("source_query_heldout", "source_query"),
]

baseline_rows = []
per_class_reports = []

for label_col in target_cols:
    for feature_col in feature_cols:
        # Skip empty artifact-only columns.
        if exp_df[feature_col].fillna("").astype(str).str.strip().eq("").mean() > 0.95:
            continue
        for split_name, group_col in split_specs:
            if split_name == "random":
                train, test, status = make_random_split(exp_df, label_col)
            else:
                train, test, status = make_group_split(exp_df, label_col, group_col)
            run_name = f"{label_col}__{feature_col}__{split_name}"
            if status != "ok":
                baseline_rows.append({"run_name": run_name, "label_col": label_col, "input_col": feature_col, "split": split_name, "status": status})
                continue
            try:
                res, rep = train_eval_tfidf_lr(train, test, feature_col, label_col, run_name)
                res["split"] = split_name
                res["status"] = "ok"
                baseline_rows.append(res)
                per_class_reports.append(rep)
            except Exception as e:
                baseline_rows.append({"run_name": run_name, "label_col": label_col, "input_col": feature_col, "split": split_name, "status": f"failed: {e}"})

baseline_results = pd.DataFrame(baseline_rows)
per_class_results = pd.concat(per_class_reports, ignore_index=True) if per_class_reports else pd.DataFrame()

display(baseline_results.sort_values(["label_col", "split", "macro_f1"], ascending=[True, True, False]).head(50))
save_csv(baseline_results, "16_tfidf_lr_baseline_results.csv")
save_csv(per_class_results, "17_tfidf_lr_per_class_reports.csv")
"""
    ),
    code(
        """# ============================================================
# 12. Low-evidence proxy
# ============================================================

def low_evidence_split(d, label_col, group_col="source_query", budget=0.10, min_group_size=20):
    train_parts, test_parts = [], []
    for _, g in d.groupby(group_col):
        g = g.sort_values("row_id").copy()
        if len(g) < min_group_size:
            continue
        n_train = max(1, int(np.ceil(len(g) * budget)))
        if n_train >= len(g):
            continue
        train_parts.append(g.iloc[:n_train])
        test_parts.append(g.iloc[n_train:])
    if not train_parts or not test_parts:
        return pd.DataFrame(), pd.DataFrame(), "insufficient_groups"
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    train = train[train[label_col].map(train[label_col].value_counts()).ge(2)]
    test = test[test[label_col].isin(set(train[label_col]))]
    if len(train) < 60 or len(test) < 40 or train[label_col].nunique() < 2 or test[label_col].nunique() < 2:
        return train, test, "insufficient_low_evidence_after_filter"
    return train, test, "ok"

low_rows = []
low_reports = []
for label_col in target_cols:
    for budget in [0.10, 0.20, 0.40]:
        train, test, status = low_evidence_split(exp_df, label_col, group_col="source_query", budget=budget)
        run_name = f"{label_col}__text_only__low_evidence_source_query_{budget}"
        if status != "ok":
            low_rows.append({"run_name": run_name, "label_col": label_col, "budget": budget, "status": status, "rows_train": len(train), "rows_test": len(test)})
            continue
        try:
            res, rep = train_eval_tfidf_lr(train, test, "text_only", label_col, run_name)
            res["budget"] = budget
            res["status"] = "ok"
            low_rows.append(res)
            low_reports.append(rep)
        except Exception as e:
            low_rows.append({"run_name": run_name, "label_col": label_col, "budget": budget, "status": f"failed: {e}"})

low_results = pd.DataFrame(low_rows)
low_per_class = pd.concat(low_reports, ignore_index=True) if low_reports else pd.DataFrame()
display(low_results)
save_csv(low_results, "18_low_evidence_proxy_results.csv")
save_csv(low_per_class, "19_low_evidence_per_class_reports.csv")
"""
    ),
    code(
        """# ============================================================
# 13. Report-ready findings
# ============================================================

lines = []
lines.append("# New Dataset Research Findings")
lines.append("")
lines.append("## Dataset characterization")
lines.append("")
lines.append(f"- Full rows: {len(df):,}")
lines.append(f"- Labeled rows: {len(labeled_df):,} ({100 * len(labeled_df) / len(df):.3f}%)")
lines.append(f"- Unlabeled rows: {len(unlabeled_df):,}")
lines.append("- Supervised experiments use labeled subset only.")
lines.append("")

if len(imbalance_summary):
    lines.append("## Imbalance and majority baselines")
    lines.append("")
    lines.append(imbalance_summary.to_markdown(index=False))
    lines.append("")

if len(query_summary):
    lines.append("## Source-query artifact")
    lines.append("")
    lines.append(query_summary.to_markdown(index=False))
    lines.append("")

if len(baseline_results):
    ok = baseline_results[baseline_results["status"].eq("ok")].copy()
    if len(ok):
        lines.append("## TF-IDF Logistic Regression baseline summary")
        lines.append("")
        best = ok.sort_values("macro_f1", ascending=False).head(10)
        lines.append(best[["label_col", "input_col", "split", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy"]].to_markdown(index=False))
        lines.append("")

        crisis_ok = ok[ok["label_col"].eq("predicted_crisis_type")]
        if len(crisis_ok):
            lines.append("## Crisis-type robustness gaps")
            lines.append("")
            pivot = crisis_ok.pivot_table(index="input_col", columns="split", values="macro_f1", aggfunc="mean").reset_index()
            if {"random", "source_query_heldout"}.issubset(pivot.columns):
                pivot["gap_random_minus_source_query"] = pivot["random"] - pivot["source_query_heldout"]
            lines.append(pivot.to_markdown(index=False))
            lines.append("")

if len(low_results):
    lines.append("## Low-evidence proxy")
    lines.append("")
    lines.append(low_results.to_markdown(index=False))
    lines.append("")

lines.append("## Interpretation")
lines.append("")
lines.append("- Random split should be interpreted as learnability under familiar distribution.")
lines.append("- Source-query-heldout is more important for artifact-aware generalization.")
lines.append("- Low-evidence split is an early-warning proxy, not real-time early warning.")
lines.append("- Accuracy is secondary because crisis type and informativeness are imbalanced.")
lines.append("- `image_path`/modality should not be used as supervised feature because label availability is modality-confounded.")

report_text = "\\n".join(lines)
print(report_text[:3000])
save_markdown(report_text, "20_report_ready_findings.md")
"""
    ),
    code(
        """# ============================================================
# 14. Final artifact index
# ============================================================

artifact_rows = []
for p in sorted(OUT_DIR.glob("*")):
    if p.is_file():
        artifact_rows.append({"artifact": p.name, "path": str(p), "size": p.stat().st_size})

artifact_index = pd.DataFrame(artifact_rows)
display(artifact_index)
save_csv(artifact_index, "21_final_artifact_index.csv")
"""
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Created {NOTEBOOK_PATH.resolve()}")
