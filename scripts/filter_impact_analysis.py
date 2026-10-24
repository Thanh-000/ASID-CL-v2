import json
import math
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2_contingency
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


ROOT = Path("UK_CrisisMMD_Public")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)


def normalize_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"https?\s*:\s*/\s*/\s*\S+|https?://\S+|www\.\S+", " <url> ", s, flags=re.I)
    s = re.sub(r"(?<!\w)@[A-Za-z0-9_]{1,20}", " <user> ", s)
    s = re.sub(r"[^\w#@]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def load_dataset():
    rows = [json.loads(line) for line in (ROOT / "public_dataset.jsonl").open(encoding="utf-8") if line.strip()]
    df = pd.DataFrame(rows)
    df["row_id"] = np.arange(len(df))
    df["text"] = df["text"].fillna("").astype(str)
    df["has_image"] = df["image_path"].notna() & df["image_path"].astype(str).ne("")
    df["has_label"] = df["predicted_crisis_type"].notna()
    df["normalized_text"] = df["text"].map(normalize_text)
    df["has_mojibake"] = df["text"].str.contains(r"[âÃðÂ�]", regex=True)
    df["word_count"] = df["text"].map(lambda x: len(re.findall(r"\b\w+\b", x)))
    df["char_len"] = df["text"].str.len()

    mapping = pd.read_csv(ROOT / "image_mapping.csv")
    mapping["orig_stem"] = (
        mapping["original_filename"].astype(str)
        .str.replace(r"\.[A-Za-z0-9]+$", "", regex=True)
        .str.replace(r"_\d+$", "", regex=True)
    )
    df = df.merge(
        mapping[["new_rel_path", "predicted_type", "original_crawl_cat", "original_filename", "orig_stem"]],
        left_on="image_path",
        right_on="new_rel_path",
        how="left",
    )

    img_rows = []
    for p in (ROOT / "images").rglob("*.jpg"):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        rec = {"image_path": rel, "image_exists": True, "image_corrupt": False}
        try:
            data = p.read_bytes()
            rec["sha256"] = hashlib.sha256(data).hexdigest()
            with Image.open(p) as im:
                rec["width"] = int(im.width)
                rec["height"] = int(im.height)
                rec["mode"] = im.mode
                rec["megapixels"] = im.width * im.height / 1_000_000
        except Exception:
            rec["image_corrupt"] = True
        img_rows.append(rec)
    img = pd.DataFrame(img_rows)
    df = df.merge(img, on="image_path", how="left")
    df["image_exists"] = df["image_exists"].fillna(False).astype(bool)
    df["image_corrupt"] = df["image_corrupt"].fillna(False).astype(bool)
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df["low_resolution_any_side_lt224"] = df["has_image"] & ((df["width"] < 224) | (df["height"] < 224))
    df["non_rgb_or_l_mode"] = df["has_image"] & ~df["mode"].isin(["RGB", "L"])
    df["path_folder"] = df["image_path"].fillna("").astype(str).map(
        lambda p: p.replace("\\", "/").split("/")[1] if len(p.replace("\\", "/").split("/")) >= 3 else None
    )
    df["path_label_mismatch"] = df["has_image"] & (df["path_folder"] != df["predicted_crisis_type"])
    df["crawl_pred_mismatch"] = df["has_image"] & (df["original_crawl_cat"].astype(str) != df["predicted_crisis_type"].astype(str))

    labeled = df[df["has_label"]].copy()
    text_conflict_keys = (
        labeled.groupby("normalized_text")["predicted_crisis_type"].nunique().loc[lambda s: s > 1].index
    )
    df["normalized_text_conflict_key"] = df["normalized_text"].isin(set(text_conflict_keys))

    orig_multilabel_keys = (
        labeled.dropna(subset=["orig_stem"])
        .groupby("orig_stem")["predicted_crisis_type"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    df["original_post_multilabel"] = df["orig_stem"].isin(set(orig_multilabel_keys))

    # Duplicate-beyond-first flags are deterministic by original row order.
    for col, flag in [
        ("sha256", "duplicate_sha_beyond_first"),
        ("normalized_text", "duplicate_normalized_text_beyond_first"),
        ("orig_stem", "duplicate_original_post_beyond_first"),
    ]:
        df[flag] = False
        mask = df[col].notna() & df["has_label"]
        df.loc[mask, flag] = df.loc[mask].duplicated(col, keep="first")
    return df


def entropy_bits(series):
    vc = series.value_counts()
    p = vc / vc.sum()
    return float(-(p * np.log2(p)).sum())


def cramers_v(df, row, col):
    ct = pd.crosstab(df[row], df[col])
    if min(ct.shape) < 2:
        return None
    arr = ct.to_numpy(dtype=float)
    arr = arr[arr.sum(axis=1) > 0, :]
    arr = arr[:, arr.sum(axis=0) > 0]
    if min(arr.shape) < 2:
        return None
    chi2 = chi2_contingency(arr, correction=False)[0]
    n = arr.sum()
    return float(math.sqrt(chi2 / (n * min(arr.shape[0] - 1, arr.shape[1] - 1))))


def majority_stats(y):
    y = pd.Series(y).astype(str)
    majority = y.value_counts().idxmax()
    pred = np.array([majority] * len(y), dtype=object)
    return {
        "majority_class": majority,
        "majority_accuracy": round(float(accuracy_score(y, pred)), 6),
        "majority_macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 6),
    }


def lookup_cv(df, target, feature, group_heldout=False):
    data = df[[target, feature]].dropna().copy()
    if len(data) < 20 or data[target].nunique() < 2:
        return {"accuracy": None, "macro_f1": None}
    y = data[target].astype(str).to_numpy()
    x = data[feature].astype(str)
    if group_heldout:
        n_groups = x.nunique()
        if n_groups < 5:
            return {"accuracy": None, "macro_f1": None}
        splitter = GroupKFold(n_splits=5).split(data, y, x)
    else:
        min_class = pd.Series(y).value_counts().min()
        n_splits = int(min(5, min_class))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(data, y)
    pred = np.empty(len(data), dtype=object)
    for tr, te in splitter:
        train = data.iloc[tr]
        default = train[target].astype(str).value_counts().idxmax()
        mapping = train.groupby(feature)[target].agg(lambda s: s.astype(str).value_counts().idxmax()).to_dict()
        pred[te] = [str(mapping.get(v, default)) for v in data.iloc[te][feature].astype(str)]
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 6),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 6),
    }


def summarize_subset(name, subset, raw_n):
    n = len(subset)
    crisis_vc = subset["predicted_crisis_type"].value_counts()
    info_vc = subset["predicted_informativeness"].value_counts()
    min_count = int(crisis_vc.min()) if len(crisis_vc) else 0
    max_count = int(crisis_vc.max()) if len(crisis_vc) else 0
    rec = {
        "scenario": name,
        "n": int(n),
        "removed_vs_raw": int(raw_n - n),
        "retained_pct": round(float(n / raw_n * 100), 3) if raw_n else 0,
        "classes": int(crisis_vc.size),
        "min_class_count": min_count,
        "max_class_count": max_count,
        "imbalance_ratio": round(float(max_count / min_count), 6) if min_count else None,
        "crisis_entropy_bits": round(entropy_bits(subset["predicted_crisis_type"]), 6) if n else None,
        "informativeness_entropy_bits": round(entropy_bits(subset["predicted_informativeness"]), 6) if n and subset["predicted_informativeness"].notna().any() else None,
        "source_query_x_crisis_v": round(cramers_v(subset, "source_query", "predicted_crisis_type"), 6) if n else None,
        "platform_x_crisis_v": round(cramers_v(subset, "platform", "predicted_crisis_type"), 6) if n else None,
        "crisis_x_info_v": round(cramers_v(subset, "predicted_crisis_type", "predicted_informativeness"), 6) if n else None,
        "unique_source_queries": int(subset["source_query"].nunique()),
        "unique_sha": int(subset["sha256"].nunique()),
        "unique_normalized_text": int(subset["normalized_text"].nunique()),
        "unique_original_posts": int(subset["orig_stem"].nunique()),
        "mojibake_rows": int(subset["has_mojibake"].sum()),
        "crawl_pred_mismatch_rows": int(subset["crawl_pred_mismatch"].sum()),
        "low_res_rows": int(subset["low_resolution_any_side_lt224"].sum()),
        "text_conflict_rows": int(subset["normalized_text_conflict_key"].sum()),
        "orig_multilabel_rows": int(subset["original_post_multilabel"].sum()),
    }
    rec.update({f"count_{k}": int(v) for k, v in crisis_vc.to_dict().items()})
    rec.update({f"pct_{k}": round(float(v / n * 100), 3) for k, v in crisis_vc.to_dict().items()})
    rec.update(majority_stats(subset["predicted_crisis_type"]))
    sq_random = lookup_cv(subset, "predicted_crisis_type", "source_query", group_heldout=False)
    sq_group = lookup_cv(subset, "predicted_crisis_type", "source_query", group_heldout=True)
    rec["source_query_lookup_random_macro_f1"] = sq_random["macro_f1"]
    rec["source_query_lookup_random_accuracy"] = sq_random["accuracy"]
    rec["source_query_lookup_group_macro_f1"] = sq_group["macro_f1"]
    rec["source_query_lookup_group_accuracy"] = sq_group["accuracy"]
    return rec


def main():
    df = load_dataset()
    labeled = df[df["has_label"]].copy()
    raw_n = len(labeled)

    hard_invalid = (
        ~labeled["image_exists"]
        | labeled["image_corrupt"]
        | labeled["path_label_mismatch"]
    )
    integrity_risk = hard_invalid | labeled["low_resolution_any_side_lt224"] | labeled["non_rgb_or_l_mode"]
    conflict_risk = integrity_risk | labeled["normalized_text_conflict_key"]
    duplicate_safe = conflict_risk | labeled["duplicate_sha_beyond_first"] | labeled["duplicate_normalized_text_beyond_first"]
    post_safe = duplicate_safe | labeled["duplicate_original_post_beyond_first"]
    conservative = post_safe | labeled["crawl_pred_mismatch"] | labeled["has_mojibake"] | labeled["original_post_multilabel"]
    label_precision = integrity_risk | labeled["normalized_text_conflict_key"] | labeled["original_post_multilabel"] | labeled["crawl_pred_mismatch"]

    scenarios = {
        "raw_labeled": pd.Series(False, index=labeled.index),
        "hard_invalid_only_removed": hard_invalid,
        "integrity_clean": integrity_risk,
        "label_conflict_clean": conflict_risk,
        "duplicate_safe": duplicate_safe,
        "post_safe_recommended": post_safe,
        "label_precision_no_crawl_mismatch": label_precision,
        "conservative_high_precision": conservative,
    }

    summaries = []
    removal_rows = []
    for name, mask in scenarios.items():
        kept = labeled.loc[~mask].copy()
        summaries.append(summarize_subset(name, kept, raw_n))
        removed = labeled.loc[mask].copy()
        removal_rows.append(
            {
                "scenario": name,
                "removed": int(len(removed)),
                "retained": int(len(kept)),
                "removed_pct": round(float(len(removed) / raw_n * 100), 3),
                "removed_strikes": int((removed["predicted_crisis_type"] == "strikes").sum()),
                "removed_social_unrest": int((removed["predicted_crisis_type"] == "social_unrest").sum()),
                "removed_pandemic_health": int((removed["predicted_crisis_type"] == "pandemic_health").sum()),
                "removed_political_crisis": int((removed["predicted_crisis_type"] == "political_crisis").sum()),
                "removed_natural_disaster": int((removed["predicted_crisis_type"] == "natural_disaster").sum()),
                "removed_major_accidents": int((removed["predicted_crisis_type"] == "major_accidents").sum()),
            }
        )

    flag_counts = []
    flags = [
        "image_exists",
        "image_corrupt",
        "path_label_mismatch",
        "low_resolution_any_side_lt224",
        "non_rgb_or_l_mode",
        "normalized_text_conflict_key",
        "duplicate_sha_beyond_first",
        "duplicate_normalized_text_beyond_first",
        "duplicate_original_post_beyond_first",
        "original_post_multilabel",
        "crawl_pred_mismatch",
        "has_mojibake",
    ]
    for flag in flags:
        if flag == "image_exists":
            count = int((~labeled[flag]).sum())
            desc = "missing_image_file"
        else:
            count = int(labeled[flag].sum())
            desc = flag
        flag_counts.append({"flag": desc, "count": count, "pct_labeled": round(float(count / raw_n * 100), 3)})

    summary_df = pd.DataFrame(summaries)
    removal_df = pd.DataFrame(removal_rows)
    flags_df = pd.DataFrame(flag_counts)

    summary_df.to_csv(OUT / "filter_impact_summary.csv", index=False)
    removal_df.to_csv(OUT / "filter_removal_by_class.csv", index=False)
    flags_df.to_csv(OUT / "filter_quality_flag_counts.csv", index=False)

    payload = {
        "quality_flag_counts": flag_counts,
        "scenario_summary": summaries,
        "scenario_removal_by_class": removal_rows,
        "scenario_definitions": {
            "hard_invalid_only_removed": "Remove missing/corrupt images and path-folder/label mismatch.",
            "integrity_clean": "Hard invalid + low-resolution any side <224 + unsupported image mode.",
            "label_conflict_clean": "Integrity clean + remove normalized text keys with conflicting crisis labels.",
            "duplicate_safe": "Label conflict clean + remove duplicate SHA beyond first + duplicate normalized text beyond first.",
            "post_safe_recommended": "Duplicate safe + keep only first image per original post id.",
            "label_precision_no_crawl_mismatch": "Integrity clean + remove text conflict + original-post multilabel + original crawl vs predicted mismatch.",
            "conservative_high_precision": "Post safe + remove crawl/pred mismatch + mojibake + original-post multilabel.",
        },
    }
    with (OUT / "filter_impact_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "flags": flag_counts,
        "scenarios": [
            {
                "scenario": r["scenario"],
                "n": r["n"],
                "retained_pct": r["retained_pct"],
                "imbalance_ratio": r["imbalance_ratio"],
                "source_query_x_crisis_v": r["source_query_x_crisis_v"],
                "source_query_lookup_random_macro_f1": r["source_query_lookup_random_macro_f1"],
                "source_query_lookup_group_macro_f1": r["source_query_lookup_group_macro_f1"],
                "counts": {k.replace("count_", ""): v for k, v in r.items() if k.startswith("count_")},
            }
            for r in summaries
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
