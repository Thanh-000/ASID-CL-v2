import json
import math
import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import jensenshannon


ROOT = Path("UK_CrisisMMD_Public")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(2026)


CLASSES = [
    "major_accidents",
    "natural_disaster",
    "pandemic_health",
    "political_crisis",
    "social_unrest",
    "strikes",
]


def normalize_text(s):
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"https?\s*:\s*/\s*/\s*\S+|https?://\S+|www\.\S+", " <url> ", s, flags=re.I)
    s = re.sub(r"(?<!\w)@[A-Za-z0-9_]{1,20}", " <user> ", s)
    s = re.sub(r"[^\w#@]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def load_labeled():
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
    labeled["normalized_text_conflict_key"] = labeled["normalized_text"].isin(set(text_conflict_keys))
    orig_multilabel_keys = (
        labeled.dropna(subset=["orig_stem"])
        .groupby("orig_stem")["predicted_crisis_type"]
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )
    labeled["original_post_multilabel"] = labeled["orig_stem"].isin(set(orig_multilabel_keys))

    for col, flag in [
        ("sha256", "duplicate_sha_beyond_first"),
        ("normalized_text", "duplicate_normalized_text_beyond_first"),
        ("orig_stem", "duplicate_original_post_beyond_first"),
    ]:
        labeled[flag] = labeled.duplicated(col, keep="first")
    return labeled


def add_versions(df):
    hard_invalid = ~df["image_exists"] | df["image_corrupt"] | df["path_label_mismatch"]
    integrity_risk = hard_invalid | df["low_resolution_any_side_lt224"] | df["non_rgb_or_l_mode"]
    label_conflict = integrity_risk | df["normalized_text_conflict_key"]
    duplicate_safe_drop = label_conflict | df["duplicate_sha_beyond_first"] | df["duplicate_normalized_text_beyond_first"]
    post_safe_drop = duplicate_safe_drop | df["duplicate_original_post_beyond_first"]
    conservative_drop = post_safe_drop | df["crawl_pred_mismatch"] | df["has_mojibake"] | df["original_post_multilabel"]
    df = df.copy()
    df["include_v0_raw_labeled"] = True
    df["include_v1_integrity_clean"] = ~integrity_risk
    df["include_v2_duplicate_safe"] = ~duplicate_safe_drop
    df["include_v3_post_safe_recommended"] = ~post_safe_drop
    df["include_v4_conservative_sensitivity"] = ~conservative_drop
    return df


def js_divergence(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return None
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2)


def class_counts(df):
    return df["predicted_crisis_type"].value_counts().reindex(CLASSES, fill_value=0)


def retention_by(df, group_col, version_col, min_raw=1):
    raw = df.groupby(group_col).size().rename("raw_n")
    kept = df[df[version_col]].groupby(group_col).size().rename("kept_n")
    out = pd.concat([raw, kept], axis=1).fillna(0).reset_index()
    out["kept_n"] = out["kept_n"].astype(int)
    out["raw_n"] = out["raw_n"].astype(int)
    out = out[out["raw_n"] >= min_raw].copy()
    out["retention_rate"] = out["kept_n"] / out["raw_n"]
    return out.sort_values(["retention_rate", "raw_n"], ascending=[True, False])


def leakage_rates(train, test):
    rates = {}
    for name, col in [
        ("source_query", "source_query"),
        ("normalized_text", "normalized_text"),
        ("image_sha", "sha256"),
        ("original_post", "orig_stem"),
    ]:
        train_set = set(train[col].dropna().astype(str))
        test_values = test[col].dropna().astype(str)
        rates[f"{name}_test_seen_in_train_rate"] = float(test_values.isin(train_set).mean()) if len(test_values) else None
    return rates


def split_score(train, val, test, overall_counts, target_val=0.1, target_test=0.2):
    train_counts = class_counts(train).to_numpy()
    val_counts = class_counts(val).to_numpy()
    test_counts = class_counts(test).to_numpy()
    total = overall_counts.to_numpy()
    score = 0
    score += 3.0 * (js_divergence(test_counts, total) or 0)
    score += 2.0 * (js_divergence(val_counts, total) or 0)
    score += abs(len(test) / overall_counts.sum() - target_test)
    score += abs(len(val) / overall_counts.sum() - target_val)
    score += 0.5 * np.max(np.abs(test_counts / max(1, len(test)) - total / total.sum()))
    return float(score)


def greedy_group_split(df, group_col, seed=0, val_frac=0.1, test_frac=0.2):
    rng = np.random.default_rng(seed)
    group_stats = []
    for g, sub in df.groupby(group_col, dropna=False):
        counts = class_counts(sub)
        group_stats.append({"group": g, "n": len(sub), "counts": counts})
    rng.shuffle(group_stats)
    group_stats = sorted(group_stats, key=lambda r: (r["n"], rng.random()), reverse=True)

    total_counts = class_counts(df)
    target_test = total_counts * test_frac
    target_val = total_counts * val_frac
    test_groups = set()
    val_groups = set()
    test_counts = pd.Series(0, index=CLASSES, dtype=float)
    val_counts = pd.Series(0, index=CLASSES, dtype=float)
    test_n = 0
    val_n = 0
    target_test_n = len(df) * test_frac
    target_val_n = len(df) * val_frac

    def distance(counts, n, target_counts, target_n):
        class_term = float(((counts - target_counts).abs() / np.maximum(target_counts, 1)).mean())
        size_term = abs(n - target_n) / max(target_n, 1)
        return class_term + size_term

    for item in group_stats:
        c = item["counts"].astype(float)
        place_test = distance(test_counts + c, test_n + item["n"], target_test, target_test_n)
        keep_test = distance(test_counts, test_n, target_test, target_test_n)
        place_val = distance(val_counts + c, val_n + item["n"], target_val, target_val_n)
        keep_val = distance(val_counts, val_n, target_val, target_val_n)

        if test_n < target_test_n and (place_test - keep_test) <= (place_val - keep_val):
            test_groups.add(item["group"])
            test_counts += c
            test_n += item["n"]
        elif val_n < target_val_n:
            val_groups.add(item["group"])
            val_counts += c
            val_n += item["n"]

    split = pd.Series("train", index=df.index)
    split[df[group_col].isin(test_groups)] = "test"
    split[df[group_col].isin(val_groups)] = "val"
    # If group is null for some rows, pandas isin handles actual np.nan poorly. Keep those rows in train by default.
    return split


def evaluate_split(df, split_name, split_series, version_name, group_col):
    train = df[split_series == "train"]
    val = df[split_series == "val"]
    test = df[split_series == "test"]
    overall = class_counts(df)
    rec = {
        "version": version_name,
        "split_name": split_name,
        "group_col": group_col,
        "n": int(len(df)),
        "train_n": int(len(train)),
        "val_n": int(len(val)),
        "test_n": int(len(test)),
        "train_pct": round(len(train) / len(df) * 100, 3),
        "val_pct": round(len(val) / len(df) * 100, 3),
        "test_pct": round(len(test) / len(df) * 100, 3),
        "test_js_vs_overall": round(js_divergence(class_counts(test), overall), 6) if len(test) else None,
        "val_js_vs_overall": round(js_divergence(class_counts(val), overall), 6) if len(val) else None,
        "test_max_abs_prop_diff": round(float(np.max(np.abs(class_counts(test) / max(1, len(test)) - overall / len(df)))), 6) if len(test) else None,
        "val_max_abs_prop_diff": round(float(np.max(np.abs(class_counts(val) / max(1, len(val)) - overall / len(df)))), 6) if len(val) else None,
        "missing_test_classes": int((class_counts(test) == 0).sum()),
        "missing_val_classes": int((class_counts(val) == 0).sum()),
    }
    for k, v in leakage_rates(train, test).items():
        rec[f"test_{k}"] = round(v, 6) if v is not None else None
    for split_label, sub in [("train", train), ("val", val), ("test", test)]:
        counts = class_counts(sub)
        for cls, count in counts.items():
            rec[f"{split_label}_count_{cls}"] = int(count)
    return rec


def make_iid_split(df, seed=0, val_frac=0.1, test_frac=0.2):
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for cls, sub in df.groupby("predicted_crisis_type"):
        idx = sub.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_frac))
        n_val = int(round(len(idx) * val_frac))
        split.loc[idx[:n_test]] = "test"
        split.loc[idx[n_test:n_test + n_val]] = "val"
    return split


def repeated_best_group_split(df, group_col, version_name, split_name, seeds=3):
    overall = class_counts(df)
    best = None
    best_series = None
    for seed in range(seeds):
        series = greedy_group_split(df, group_col, seed=seed)
        train = df[series == "train"]
        val = df[series == "val"]
        test = df[series == "test"]
        score = split_score(train, val, test, overall)
        if best is None or score < best:
            best = score
            best_series = series
    rec = evaluate_split(df, split_name, best_series, version_name, group_col)
    rec["greedy_score"] = round(best, 6)
    rec["seed_search_n"] = seeds
    return best_series, rec


def cl_stream_summary(df, version_name):
    rows = []
    for stream_col in ["predicted_crisis_type", "source_query", "platform", "original_crawl_cat"]:
        vc = df[stream_col].value_counts()
        rows.append({
            "version": version_name,
            "stream_col": stream_col,
            "num_streams": int(vc.size),
            "min_n": int(vc.min()) if len(vc) else 0,
            "median_n": float(vc.median()) if len(vc) else 0,
            "max_n": int(vc.max()) if len(vc) else 0,
            "streams_ge_20": int((vc >= 20).sum()),
            "streams_ge_50": int((vc >= 50).sum()),
            "streams_ge_100": int((vc >= 100).sum()),
        })
    return rows


def main():
    df = add_versions(load_labeled())
    version_cols = {
        "v0_raw_labeled": "include_v0_raw_labeled",
        "v1_integrity_clean": "include_v1_integrity_clean",
        "v2_duplicate_safe": "include_v2_duplicate_safe",
        "v3_post_safe_recommended": "include_v3_post_safe_recommended",
        "v4_conservative_sensitivity": "include_v4_conservative_sensitivity",
    }

    manifest_cols = [
        "row_id",
        "platform",
        "text",
        "image_path",
        "source_query",
        "predicted_crisis_type",
        "predicted_informativeness",
        "normalized_text",
        "sha256",
        "orig_stem",
        "original_crawl_cat",
        "crawl_pred_mismatch",
        "normalized_text_conflict_key",
        "original_post_multilabel",
        "duplicate_sha_beyond_first",
        "duplicate_normalized_text_beyond_first",
        "duplicate_original_post_beyond_first",
    ] + list(version_cols.values())
    df[manifest_cols].to_csv(OUT / "dataset_version_manifest.csv", index=False)

    retention_rows = []
    top_query_loss_rows = []
    for version, col in version_cols.items():
        subset = df[df[col]].copy()
        counts = class_counts(subset)
        raw_counts = class_counts(df)
        for cls in CLASSES:
            retention_rows.append({
                "version": version,
                "dimension": "class",
                "name": cls,
                "raw_n": int(raw_counts[cls]),
                "kept_n": int(counts[cls]),
                "retention_rate": round(float(counts[cls] / raw_counts[cls]), 6) if raw_counts[cls] else None,
            })
        platform_ret = retention_by(df, "platform", col)
        for _, r in platform_ret.iterrows():
            retention_rows.append({
                "version": version,
                "dimension": "platform",
                "name": r["platform"],
                "raw_n": int(r["raw_n"]),
                "kept_n": int(r["kept_n"]),
                "retention_rate": round(float(r["retention_rate"]), 6),
            })
        query_ret = retention_by(df, "source_query", col, min_raw=20)
        query_ret.insert(0, "version", version)
        top_query_loss_rows.extend(query_ret.head(30).to_dict(orient="records"))

    retention_df = pd.DataFrame(retention_rows)
    retention_df.to_csv(OUT / "dataset_version_retention_by_class_platform.csv", index=False)
    pd.DataFrame(top_query_loss_rows).to_csv(OUT / "dataset_version_top_query_retention_losses.csv", index=False)

    split_rows = []
    assignment_frames = []
    split_version_order = [
        "v0_raw_labeled",
        "v2_duplicate_safe",
        "v3_post_safe_recommended",
    ]
    split_groups_by_version = {
        "v0_raw_labeled": [
            ("orig_stem", "balanced_original_post_group"),
            ("sha256", "balanced_image_sha_group"),
            ("normalized_text", "balanced_normalized_text_group"),
            ("source_query", "balanced_source_query_group"),
        ],
        "v2_duplicate_safe": [
            ("orig_stem", "balanced_original_post_group"),
            ("source_query", "balanced_source_query_group"),
        ],
        "v3_post_safe_recommended": [
            ("orig_stem", "balanced_original_post_group"),
            ("source_query", "balanced_source_query_group"),
        ],
    }

    for version in split_version_order:
        col = version_cols[version]
        sub = df[df[col]].copy()
        if len(sub) < 100:
            continue
        iid = make_iid_split(sub, seed=42)
        split_rows.append(evaluate_split(sub, "iid_stratified", iid, version, "none"))
        assignment_frames.append(pd.DataFrame({"row_id": sub["row_id"], "version": version, "split_name": "iid_stratified", "split": iid.values}))

        for group_col, split_name in split_groups_by_version[version]:
            if sub[group_col].nunique(dropna=True) >= 5:
                series, rec = repeated_best_group_split(sub, group_col, version, split_name, seeds=3)
                split_rows.append(rec)
                assignment_frames.append(pd.DataFrame({"row_id": sub["row_id"], "version": version, "split_name": split_name, "split": series.values}))

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(OUT / "dataset_version_split_protocol_summary.csv", index=False)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    assignments.to_csv(OUT / "dataset_version_split_assignments.csv", index=False)

    cl_rows = []
    for version, col in version_cols.items():
        cl_rows.extend(cl_stream_summary(df[df[col]].copy(), version))
    cl_df = pd.DataFrame(cl_rows)
    cl_df.to_csv(OUT / "dataset_version_cl_stream_feasibility.csv", index=False)

    payload = {
        "version_counts": {
            version: int(df[col].sum()) for version, col in version_cols.items()
        },
        "retention_by_class_platform": retention_rows,
        "top_query_retention_losses": top_query_loss_rows[:100],
        "split_protocol_summary": split_rows,
        "cl_stream_feasibility": cl_rows,
    }
    with (OUT / "dataset_version_split_protocol_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    compact = {
        "version_counts": payload["version_counts"],
        "best_splits_excerpt": split_df[
            [
                "version",
                "split_name",
                "n",
                "train_n",
                "val_n",
                "test_n",
                "test_js_vs_overall",
                "test_max_abs_prop_diff",
                "test_source_query_test_seen_in_train_rate",
                "test_normalized_text_test_seen_in_train_rate",
                "test_image_sha_test_seen_in_train_rate",
                "test_original_post_test_seen_in_train_rate",
            ]
        ].to_dict(orient="records")[:30],
        "cl_streams": cl_rows,
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
