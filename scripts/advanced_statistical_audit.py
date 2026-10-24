import json
import math
import re
import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.metrics import accuracy_score, f1_score, mutual_info_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


ROOT = Path("UK_CrisisMMD_Public")
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(42)


def load_dataset():
    records = []
    with (ROOT / "public_dataset.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["has_image"] = df["image_path"].notna() & df["image_path"].astype(str).ne("")
    df["modality"] = np.where(df["has_image"], "text+image", "text-only")
    df["has_crisis_label"] = df["predicted_crisis_type"].notna()
    df["has_info_label"] = df["predicted_informativeness"].notna()
    df["text"] = df["text"].fillna("").astype(str)

    url_re = re.compile(r"https?\s*:\s*/\s*/\s*\S+|https?://\S+|www\.\S+", re.I)
    mention_re = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,20}")
    nonword_re = re.compile(r"[^\w#@]+", re.UNICODE)
    whitespace_re = re.compile(r"\s+")

    def normalize_text(s):
        s = s.lower()
        s = url_re.sub(" <url> ", s)
        s = mention_re.sub(" <user> ", s)
        s = nonword_re.sub(" ", s)
        return whitespace_re.sub(" ", s).strip()

    df["normalized_text"] = df["text"].map(normalize_text)
    df["char_len"] = df["text"].str.len()
    df["word_count"] = df["text"].map(lambda x: len(re.findall(r"\b\w+\b", x)))
    df["hashtag_count"] = df["hashtags"].map(lambda x: len(x) if isinstance(x, list) else 0)
    return df


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return {"n": 0, "count": int(k), "p": None, "lo": None, "hi": None}
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {
        "n": int(n),
        "count": int(k),
        "p": round(float(p), 6),
        "lo": round(float(max(0, center - half)), 6),
        "hi": round(float(min(1, center + half)), 6),
    }


def entropy_bits(labels):
    counts = np.array(list(Counter(labels).values()), dtype=float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def cramers_v_from_table(table):
    arr = np.asarray(table, dtype=float)
    arr = arr[arr.sum(axis=1) > 0, :]
    arr = arr[:, arr.sum(axis=0) > 0]
    if arr.sum() == 0 or min(arr.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(arr, correction=False)[0]
    n = arr.sum()
    r, c = arr.shape
    return float(math.sqrt(chi2 / (n * min(r - 1, c - 1))))


def contingency_stats(df, row, col, name):
    ct = pd.crosstab(df[row], df[col])
    chi2, p, dof, expected = chi2_contingency(ct.to_numpy(), correction=False)
    n = expected.sum()
    row_prop = ct.to_numpy().sum(axis=1, keepdims=True) / n
    col_prop = ct.to_numpy().sum(axis=0, keepdims=True) / n
    pearson = (ct.to_numpy() - expected) / np.sqrt(expected)
    adjusted = (ct.to_numpy() - expected) / np.sqrt(expected * (1 - row_prop) * (1 - col_prop))
    contrib = ((ct.to_numpy() - expected) ** 2 / expected) / chi2
    cells = []
    for i, r in enumerate(ct.index):
        for j, c in enumerate(ct.columns):
            cells.append(
                {
                    "test": name,
                    "row": str(r),
                    "col": str(c),
                    "observed": int(ct.iloc[i, j]),
                    "expected": round(float(expected[i, j]), 3),
                    "pearson_residual": round(float(pearson[i, j]), 3),
                    "adjusted_residual": round(float(adjusted[i, j]), 3),
                    "chi2_contribution_pct": round(float(contrib[i, j] * 100), 3),
                    "direction": "over" if ct.iloc[i, j] > expected[i, j] else "under",
                }
            )
    cells = sorted(cells, key=lambda x: abs(x["adjusted_residual"]), reverse=True)
    return {
        "name": name,
        "rows": list(map(str, ct.index)),
        "cols": list(map(str, ct.columns)),
        "chi2": round(float(chi2), 6),
        "dof": int(dof),
        "p_value": float(p),
        "cramers_v": round(cramers_v_from_table(ct.to_numpy()), 6),
        "top_adjusted_residuals": cells[:20],
        "table": ct.reset_index().to_dict(orient="records"),
    }, cells


def bootstrap_ci(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    return {
        "mean": round(float(np.mean(values)), 6),
        "lo": round(float(np.quantile(values, alpha / 2)), 6),
        "hi": round(float(np.quantile(values, 1 - alpha / 2)), 6),
    }


def bootstrap_entropy_ci(labels, n_boot=2000):
    labels = np.asarray(labels)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals.append(entropy_bits(labels[idx]))
    return bootstrap_ci(vals)


def bootstrap_cramers_ci(df, row, col, n_boot=1000):
    categories_row = sorted(df[row].dropna().astype(str).unique())
    categories_col = sorted(df[col].dropna().astype(str).unique())
    vals = []
    arr = df[[row, col]].dropna().astype(str).to_numpy()
    n = len(arr)
    for _ in range(n_boot):
        sample = arr[RNG.integers(0, n, n)]
        tmp = pd.DataFrame(sample, columns=[row, col])
        ct = pd.crosstab(
            pd.Categorical(tmp[row], categories=categories_row),
            pd.Categorical(tmp[col], categories=categories_col),
            dropna=False,
        )
        vals.append(cramers_v_from_table(ct.to_numpy()))
    return bootstrap_ci(vals)


def odds_ratio_ci(a, b, c, d):
    # a=exposed+outcome, b=exposed+no outcome, c=unexposed+outcome, d=unexposed+no outcome
    aa, bb, cc, dd = map(float, [a, b, c, d])
    if min(aa, bb, cc, dd) == 0:
        aa += 0.5
        bb += 0.5
        cc += 0.5
        dd += 0.5
    log_or = math.log((aa * dd) / (bb * cc))
    se = math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    lo = math.exp(log_or - 1.959963984540054 * se)
    hi = math.exp(log_or + 1.959963984540054 * se)
    return {
        "odds_ratio": round(math.exp(log_or), 6),
        "ci95_lo": round(lo, 6),
        "ci95_hi": round(hi, 6),
    }


def holm_adjust(p_values):
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adj[idx] = running
    return adj.tolist()


def cliffs_delta_from_mwu(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.mannwhitneyu(x, y, alternative="two-sided")
    delta = 2 * float(res.statistic) / (len(x) * len(y)) - 1
    return float(delta), float(res.pvalue)


def two_group_test(df, feature, group, a, b):
    x = df.loc[df[group] == a, feature].dropna().to_numpy()
    y = df.loc[df[group] == b, feature].dropna().to_numpy()
    delta, p = cliffs_delta_from_mwu(x, y)
    return {
        "feature": feature,
        "group": group,
        "a": str(a),
        "b": str(b),
        "n_a": int(len(x)),
        "n_b": int(len(y)),
        "median_a": round(float(np.median(x)), 6),
        "median_b": round(float(np.median(y)), 6),
        "mann_whitney_p": p,
        "cliffs_delta_a_vs_b": round(delta, 6),
    }


def kruskal_group_test(df, feature, group):
    clean = df[[feature, group]].dropna()
    groups = [(g, v[feature].to_numpy()) for g, v in clean.groupby(group)]
    h, p = stats.kruskal(*[v for _, v in groups])
    n = len(clean)
    k = len(groups)
    eps2 = (h - k + 1) / (n - k) if n > k else np.nan
    pairwise = []
    raw_p = []
    for ga, gb in combinations([g for g, _ in groups], 2):
        rec = two_group_test(clean, feature, group, ga, gb)
        pairwise.append(rec)
        raw_p.append(rec["mann_whitney_p"])
    if raw_p:
        adj = holm_adjust(raw_p)
        for rec, p_adj in zip(pairwise, adj):
            rec["holm_p"] = p_adj
    summaries = []
    for g, vals in groups:
        summaries.append(
            {
                "group": str(g),
                "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "p25": round(float(np.quantile(vals, 0.25)), 6),
                "p75": round(float(np.quantile(vals, 0.75)), 6),
            }
        )
    return {
        "feature": feature,
        "group": group,
        "kruskal_h": round(float(h), 6),
        "p_value": float(p),
        "epsilon_squared": round(float(eps2), 6),
        "group_summaries": summaries,
        "top_pairwise_by_abs_delta": sorted(pairwise, key=lambda r: abs(r["cliffs_delta_a_vs_b"]), reverse=True)[:12],
    }


def lookup_cv_baseline(df, target, features, name, group_col=None, n_splits=5):
    cols = []
    for col in features + [target] + ([group_col] if group_col else []):
        if col and col not in cols:
            cols.append(col)
    data = df[cols].dropna().copy()
    y = data[target].astype(str).to_numpy()
    x = data[features].astype(str)
    if group_col:
        groups = data[group_col].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=n_splits).split(x, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42).split(x, y)

    preds = np.empty(len(data), dtype=object)
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splitter, 1):
        train = data.iloc[train_idx]
        test = data.iloc[test_idx]
        default = train[target].astype(str).value_counts().idxmax()
        mapping = {}
        grouped = train.groupby(features)[target].agg(lambda s: s.astype(str).value_counts().idxmax())
        if len(features) == 1:
            for key, val in grouped.items():
                mapping[(str(key),)] = str(val)
        else:
            for key, val in grouped.items():
                mapping[tuple(map(str, key))] = str(val)
        y_pred = []
        for _, row in test[features].astype(str).iterrows():
            key = tuple(row[f] for f in features)
            y_pred.append(mapping.get(key, default))
        preds[test_idx] = y_pred
        y_test = y[test_idx]
        fold_metrics.append(
            {
                "fold": fold,
                "n": int(len(test_idx)),
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
                "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 6),
                "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 6),
            }
        )
    overall = {
        "accuracy": round(float(accuracy_score(y, preds)), 6),
        "macro_f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y, preds, average="weighted", zero_division=0)), 6),
    }
    boot = []
    n = len(y)
    for _ in range(2000):
        idx = RNG.integers(0, n, n)
        boot.append(f1_score(y[idx], preds[idx], average="macro", zero_division=0))
    return {
        "name": name,
        "target": target,
        "features": features,
        "group_col": group_col,
        "n": int(len(data)),
        "overall": overall,
        "macro_f1_bootstrap_ci": bootstrap_ci(boot),
        "fold_metrics": fold_metrics,
    }


def information_coefficients(x, y):
    x = pd.Series(x).astype(str)
    y = pd.Series(y).astype(str)
    mi_nats = mutual_info_score(x, y)
    mi_bits = mi_nats / math.log(2)
    hx = entropy_bits(x)
    hy = entropy_bits(y)
    return {
        "mutual_information_bits": round(float(mi_bits), 6),
        "entropy_x_bits": round(float(hx), 6),
        "entropy_y_bits": round(float(hy), 6),
        "theils_u_x_to_y": round(float(mi_bits / hy), 6) if hy > 0 else None,
        "theils_u_y_to_x": round(float(mi_bits / hx), 6) if hx > 0 else None,
        "symmetrical_uncertainty": round(float(2 * mi_bits / (hx + hy)), 6) if hx + hy > 0 else None,
    }


def image_dataframe():
    records = []
    for p in (ROOT / "images").rglob("*.jpg"):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            records.append(
                {
                    "image_path": rel,
                    "folder_label": p.parent.name,
                    "width": im.width,
                    "height": im.height,
                    "megapixels": im.width * im.height / 1_000_000,
                    "aspect_ratio": im.width / im.height,
                    "bytes": p.stat().st_size,
                    "sha256": h,
                }
            )
    return pd.DataFrame(records)


def effective_cluster_metrics(counts):
    counts = np.asarray(list(counts), dtype=float)
    p = counts / counts.sum()
    h_bits = -(p * np.log2(p)).sum()
    return {
        "clusters": int(len(counts)),
        "n": int(counts.sum()),
        "mean_size": round(float(counts.mean()), 6),
        "median_size": round(float(np.median(counts)), 6),
        "max_size": int(counts.max()),
        "hhi": round(float((p * p).sum()), 6),
        "simpson_effective_clusters": round(float(1 / (p * p).sum()), 6),
        "entropy_effective_clusters": round(float(2**h_bits), 6),
    }


def main():
    df = load_dataset()
    labeled = df[df["has_crisis_label"] & df["has_info_label"]].copy()
    img_df = image_dataframe()
    labeled_img = labeled.merge(img_df, on="image_path", how="left")

    proportion_cis = []
    for label, subset, col in [
        ("platform_overall", df, "platform"),
        ("modality_overall", df, "modality"),
        ("crisis_labeled", labeled, "predicted_crisis_type"),
        ("informativeness_labeled", labeled, "predicted_informativeness"),
    ]:
        vc = subset[col].value_counts()
        n = int(vc.sum())
        for cat, k in vc.items():
            rec = {"scope": label, "category": str(cat)}
            rec.update(wilson_ci(int(k), n))
            proportion_cis.append(rec)

    # Informativeness rates and odds ratios per crisis type.
    info_or = []
    for crisis, sub in labeled.groupby("predicted_crisis_type"):
        a = int((sub["predicted_informativeness"] == "uninformative").sum())
        b = int((sub["predicted_informativeness"] == "informative").sum())
        rest = labeled[labeled["predicted_crisis_type"] != crisis]
        c = int((rest["predicted_informativeness"] == "uninformative").sum())
        d = int((rest["predicted_informativeness"] == "informative").sum())
        rec = {
            "crisis_type": str(crisis),
            "uninformative": a,
            "informative": b,
            "rate_ci": wilson_ci(a, a + b),
        }
        rec.update(odds_ratio_ci(a, b, c, d))
        info_or.append(rec)

    # Missingness mechanism: image availability by platform.
    tw = df[df["platform"] == "twitter"]
    fb = df[df["platform"] == "facebook"]
    image_or_twitter_vs_facebook = odds_ratio_ci(
        int(tw["has_image"].sum()),
        int((~tw["has_image"]).sum()),
        int(fb["has_image"].sum()),
        int((~fb["has_image"]).sum()),
    )
    risk_t = float(tw["has_image"].mean())
    risk_f = float(fb["has_image"].mean())
    image_or_twitter_vs_facebook.update(
        {
            "twitter_image_rate": round(risk_t, 6),
            "facebook_image_rate": round(risk_f, 6),
            "risk_ratio": round(risk_t / risk_f, 6),
            "risk_difference": round(risk_t - risk_f, 6),
        }
    )

    contingencies = []
    residual_rows = []
    for row, col, name, data in [
        ("predicted_crisis_type", "predicted_informativeness", "crisis_x_informativeness", labeled),
        ("platform", "modality", "platform_x_modality", df),
        ("platform", "predicted_crisis_type", "platform_x_crisis", labeled),
        ("source_query", "predicted_crisis_type", "source_query_x_crisis", labeled),
    ]:
        rec, cells = contingency_stats(data, row, col, name)
        contingencies.append(rec)
        residual_rows.extend(cells)

    information = {
        "source_query_to_crisis": information_coefficients(labeled["source_query"], labeled["predicted_crisis_type"]),
        "platform_to_crisis": information_coefficients(labeled["platform"], labeled["predicted_crisis_type"]),
        "crisis_to_informativeness": information_coefficients(labeled["predicted_crisis_type"], labeled["predicted_informativeness"]),
        "platform_to_modality": information_coefficients(df["platform"], df["modality"]),
    }

    bootstrap = {
        "crisis_entropy_bits": bootstrap_entropy_ci(labeled["predicted_crisis_type"].astype(str).to_numpy()),
        "informativeness_entropy_bits": bootstrap_entropy_ci(labeled["predicted_informativeness"].astype(str).to_numpy()),
        "crisis_x_informativeness_cramers_v": bootstrap_cramers_ci(labeled, "predicted_crisis_type", "predicted_informativeness"),
        "platform_x_modality_cramers_v": bootstrap_cramers_ci(df, "platform", "modality"),
        "platform_x_crisis_cramers_v": bootstrap_cramers_ci(labeled, "platform", "predicted_crisis_type"),
        "source_query_x_crisis_cramers_v": bootstrap_cramers_ci(labeled, "source_query", "predicted_crisis_type"),
    }

    distribution_tests = {
        "word_count_by_modality": two_group_test(df, "word_count", "modality", "text+image", "text-only"),
        "word_count_by_platform": two_group_test(df, "word_count", "platform", "twitter", "facebook"),
        "word_count_by_crisis": kruskal_group_test(labeled, "word_count", "predicted_crisis_type"),
        "char_len_by_crisis": kruskal_group_test(labeled, "char_len", "predicted_crisis_type"),
        "megapixels_by_crisis": kruskal_group_test(labeled_img, "megapixels", "predicted_crisis_type"),
        "aspect_ratio_by_crisis": kruskal_group_test(labeled_img, "aspect_ratio", "predicted_crisis_type"),
        "megapixels_by_platform_labeled": two_group_test(labeled_img, "megapixels", "platform", "twitter", "facebook"),
    }

    baselines = [
        lookup_cv_baseline(labeled, "predicted_crisis_type", ["source_query"], "crisis_from_source_query_random_cv"),
        lookup_cv_baseline(
            labeled,
            "predicted_crisis_type",
            ["source_query"],
            "crisis_from_source_query_groupkfold_by_source_query",
            group_col="source_query",
        ),
        lookup_cv_baseline(labeled, "predicted_crisis_type", ["platform"], "crisis_from_platform_random_cv"),
        lookup_cv_baseline(
            labeled,
            "predicted_crisis_type",
            ["source_query", "platform"],
            "crisis_from_source_query_plus_platform_random_cv",
        ),
        lookup_cv_baseline(
            labeled,
            "predicted_informativeness",
            ["predicted_crisis_type"],
            "informativeness_from_crisis_type_random_cv",
        ),
        lookup_cv_baseline(
            labeled,
            "predicted_informativeness",
            ["source_query"],
            "informativeness_from_source_query_random_cv",
        ),
        lookup_cv_baseline(df, "modality", ["platform"], "modality_from_platform_random_cv"),
    ]

    source_counts = labeled["source_query"].value_counts()
    text_counts = df["normalized_text"].value_counts()
    sha_counts = img_df["sha256"].value_counts()
    mapping = pd.read_csv(ROOT / "image_mapping.csv")
    mapping["orig_stem"] = (
        mapping["original_filename"].astype(str).str.replace(r"\.[A-Za-z0-9]+$", "", regex=True).str.replace(r"_\d+$", "", regex=True)
    )
    orig_counts = mapping["orig_stem"].value_counts()

    cluster_metrics = {
        "source_query_labeled": effective_cluster_metrics(source_counts.values),
        "normalized_text_all_nonempty": effective_cluster_metrics(text_counts[text_counts.index != ""].values),
        "image_sha_exact": effective_cluster_metrics(sha_counts.values),
        "original_post_from_mapping": effective_cluster_metrics(orig_counts.values),
    }

    duplicate_cis = {
        "duplicate_normalized_text_rows_beyond_first": wilson_ci(int((text_counts - 1).clip(lower=0).sum()), len(df)),
        "duplicate_sha_files_beyond_first": wilson_ci(int((sha_counts - 1).clip(lower=0).sum()), len(img_df)),
        "multi_image_original_posts": wilson_ci(int((orig_counts > 1).sum()), int(len(orig_counts))),
    }

    out = {
        "proportion_confidence_intervals": proportion_cis,
        "informativeness_odds_ratios_by_crisis": info_or,
        "image_availability_twitter_vs_facebook": image_or_twitter_vs_facebook,
        "contingency_tests": contingencies,
        "information_coefficients": information,
        "bootstrap_confidence_intervals": bootstrap,
        "distribution_tests": distribution_tests,
        "lookup_artifact_baselines": baselines,
        "cluster_effective_sample_metrics": cluster_metrics,
        "duplicate_confidence_intervals": duplicate_cis,
    }

    with (OUT / "advanced_statistical_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    pd.DataFrame(proportion_cis).to_csv(OUT / "advanced_proportion_confidence_intervals.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "advanced_chi_square_residuals.csv", index=False)
    pd.DataFrame(info_or).to_csv(OUT / "advanced_informativeness_odds_ratios.csv", index=False)
    pd.DataFrame(
        [
            {
                "name": b["name"],
                "target": b["target"],
                "features": "+".join(b["features"]),
                "group_col": b["group_col"],
                "n": b["n"],
                **b["overall"],
                "macro_f1_ci_lo": b["macro_f1_bootstrap_ci"]["lo"],
                "macro_f1_ci_hi": b["macro_f1_bootstrap_ci"]["hi"],
            }
            for b in baselines
        ]
    ).to_csv(OUT / "advanced_artifact_lookup_baselines.csv", index=False)

    compact = {
        "image_availability_or": image_or_twitter_vs_facebook,
        "top_residuals": {c["name"]: c["top_adjusted_residuals"][:6] for c in contingencies},
        "bootstrap": bootstrap,
        "baselines": [
            {
                "name": b["name"],
                "overall": b["overall"],
                "macro_f1_ci": b["macro_f1_bootstrap_ci"],
            }
            for b in baselines
        ],
        "distribution_key_tests": {
            k: v
            for k, v in distribution_tests.items()
            if k in ["word_count_by_modality", "word_count_by_platform", "word_count_by_crisis", "megapixels_by_crisis"]
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
