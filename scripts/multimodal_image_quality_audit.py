import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from scipy.ndimage import laplace, sobel
from scipy.stats import kruskal
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path("UK_CrisisMMD_Public")
OUT = Path("analysis_outputs")
MANIFEST_PATH = OUT / "dataset_version_manifest.csv"
ASSIGNMENTS_PATH = OUT / "dataset_version_split_assignments.csv"
CONSTRAINED_ASSIGNMENTS_PATH = OUT / "constrained_source_query_split_assignments.csv"
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

GEOMETRY_FEATURES = [
    "log_width",
    "log_height",
    "log_megapixels",
    "log_file_bytes",
    "aspect_ratio",
    "bytes_per_pixel",
]

QUALITY_FEATURES = [
    "luminance_mean",
    "luminance_std",
    "luminance_entropy",
    "saturation_mean",
    "saturation_std",
    "colorfulness",
    "sharpness_laplacian_var",
    "edge_density",
    "dark_clip_fraction",
    "bright_clip_fraction",
    "red_mean",
    "green_mean",
    "blue_mean",
]

ALL_VISUAL_FEATURES = GEOMETRY_FEATURES + QUALITY_FEATURES

PROTOCOLS = [
    ("v2_duplicate_safe", "balanced_original_post_group"),
    ("v2_duplicate_safe", "constrained_source_query_group"),
    ("v3_post_safe_recommended", "balanced_original_post_group"),
    ("v3_post_safe_recommended", "constrained_source_query_group"),
]


def bits_to_hex(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return f"{value:016x}"


def phash_hex(gray):
    arr = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    coeff = dctn(arr, type=2, norm="ortho")[:8, :8]
    threshold = np.median(coeff.ravel()[1:])
    return bits_to_hex((coeff > threshold).ravel())


def dhash_hex(gray):
    arr = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    return bits_to_hex((arr[:, 1:] > arr[:, :-1]).ravel())


def entropy_from_uint8(values):
    counts = np.bincount(values.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_image_features(row):
    path = ROOT / str(row["image_path"]).replace("/", "\\")
    with Image.open(path) as image:
        width, height = image.size
        rgb_image = image.convert("RGB")
        gray_image = image.convert("L")
        metric_rgb_image = rgb_image.copy()
        metric_rgb_image.thumbnail((256, 256), Image.Resampling.LANCZOS)
        metric_gray_image = metric_rgb_image.convert("L")
        rgb = np.asarray(metric_rgb_image, dtype=np.float32)
        gray = np.asarray(metric_gray_image, dtype=np.float32)
        gray_uint8 = np.asarray(metric_gray_image, dtype=np.uint8)

    pixels = max(width * height, 1)
    rgb_01 = rgb / 255.0
    maximum = rgb_01.max(axis=2)
    minimum = rgb_01.min(axis=2)
    saturation = np.divide(
        maximum - minimum,
        maximum,
        out=np.zeros_like(maximum),
        where=maximum > 0,
    )
    rg = rgb[:, :, 0] - rgb[:, :, 1]
    yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
    colorfulness = math.sqrt(float(rg.var() + yb.var())) + 0.3 * math.sqrt(
        float(rg.mean() ** 2 + yb.mean() ** 2)
    )
    gradient_x = sobel(gray, axis=1, mode="reflect")
    gradient_y = sobel(gray, axis=0, mode="reflect")
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    file_bytes = path.stat().st_size

    return {
        "row_id": int(row["row_id"]),
        "image_path": row["image_path"],
        "platform": row["platform"],
        "predicted_crisis_type": row["predicted_crisis_type"],
        "predicted_informativeness": row["predicted_informativeness"],
        "source_query": row["source_query"],
        "sha256": row["sha256"],
        "width": int(width),
        "height": int(height),
        "megapixels": float(pixels / 1_000_000),
        "file_bytes": int(file_bytes),
        "aspect_ratio": float(width / max(height, 1)),
        "bytes_per_pixel": float(file_bytes / pixels),
        "log_width": float(np.log1p(width)),
        "log_height": float(np.log1p(height)),
        "log_megapixels": float(np.log1p(pixels / 1_000_000)),
        "log_file_bytes": float(np.log1p(file_bytes)),
        "luminance_mean": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "luminance_entropy": entropy_from_uint8(gray_uint8),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "colorfulness": float(colorfulness),
        "sharpness_laplacian_var": float(laplace(gray, mode="reflect").var()),
        "edge_density": float((gradient_magnitude > 100.0).mean()),
        "dark_clip_fraction": float((gray <= 5).mean()),
        "bright_clip_fraction": float((gray >= 250).mean()),
        "red_mean": float(rgb[:, :, 0].mean()),
        "green_mean": float(rgb[:, :, 1].mean()),
        "blue_mean": float(rgb[:, :, 2].mean()),
        "phash": phash_hex(gray_image),
        "dhash": dhash_hex(gray_image),
    }


def add_quality_flags(features):
    out = features.copy()
    blur_cutoff = float(out["sharpness_laplacian_var"].quantile(0.05))
    low_entropy_cutoff = float(out["luminance_entropy"].quantile(0.05))
    out["flag_very_dark"] = out["luminance_mean"] < 35
    out["flag_very_bright"] = out["luminance_mean"] > 220
    out["flag_low_contrast"] = out["luminance_std"] < 20
    out["flag_relative_blur_bottom_5pct"] = (
        out["sharpness_laplacian_var"] <= blur_cutoff
    )
    out["flag_low_entropy_bottom_5pct"] = (
        out["luminance_entropy"] <= low_entropy_cutoff
    )
    out["flag_heavy_shadow_clipping"] = out["dark_clip_fraction"] > 0.25
    out["flag_heavy_highlight_clipping"] = out["bright_clip_fraction"] > 0.25
    out["flag_near_grayscale"] = out["saturation_mean"] < 0.03
    flag_cols = [col for col in out.columns if col.startswith("flag_")]
    out["quality_flag_count"] = out[flag_cols].sum(axis=1)
    out["has_any_quality_flag"] = out["quality_flag_count"] > 0
    thresholds = {
        "relative_blur_bottom_5pct_cutoff": blur_cutoff,
        "low_entropy_bottom_5pct_cutoff": low_entropy_cutoff,
        "very_dark_luminance_mean_lt": 35,
        "very_bright_luminance_mean_gt": 220,
        "low_contrast_luminance_std_lt": 20,
        "heavy_clipping_fraction_gt": 0.25,
        "near_grayscale_saturation_mean_lt": 0.03,
    }
    return out, thresholds


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def find_perceptual_duplicates(features):
    hashes_p = [int(value, 16) for value in features["phash"]]
    hashes_d = [int(value, 16) for value in features["dhash"]]
    aspects = features["aspect_ratio"].to_numpy(float)
    sha = features["sha256"].astype(str).to_numpy()
    labels = features["predicted_crisis_type"].astype(str).to_numpy()
    platforms = features["platform"].astype(str).to_numpy()
    row_ids = features["row_id"].to_numpy(int)

    union_find = UnionFind(len(features))
    pair_rows = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            p_distance = bin(hashes_p[i] ^ hashes_p[j]).count("1")
            if p_distance > 6:
                continue
            d_distance = bin(hashes_d[i] ^ hashes_d[j]).count("1")
            if d_distance > 10:
                continue
            aspect_log_gap = abs(math.log(max(aspects[i], 1e-9) / max(aspects[j], 1e-9)))
            if aspect_log_gap > 0.08:
                continue
            union_find.union(i, j)
            pair_rows.append(
                {
                    "row_id_a": int(row_ids[i]),
                    "row_id_b": int(row_ids[j]),
                    "phash_distance": int(p_distance),
                    "dhash_distance": int(d_distance),
                    "aspect_log_gap": float(aspect_log_gap),
                    "exact_sha_match": bool(sha[i] == sha[j]),
                    "same_crisis_label": bool(labels[i] == labels[j]),
                    "same_platform": bool(platforms[i] == platforms[j]),
                }
            )

    roots = [union_find.find(i) for i in range(len(features))]
    root_counts = pd.Series(roots).value_counts()
    duplicate_roots = set(root_counts[root_counts > 1].index)
    root_to_group = {
        root: f"phash_group_{index:04d}"
        for index, root in enumerate(sorted(duplicate_roots), 1)
    }
    component = [root_to_group.get(root) for root in roots]
    component_size = [int(root_counts[root]) if root in duplicate_roots else 1 for root in roots]
    out_features = features.copy()
    out_features["perceptual_duplicate_group"] = component
    out_features["perceptual_component_size"] = component_size
    out_features["in_perceptual_duplicate_group"] = (
        out_features["perceptual_component_size"] > 1
    )
    return out_features, pd.DataFrame(pair_rows)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def association_tests(features):
    rows = []
    for grouping in ["predicted_crisis_type", "platform", "predicted_informativeness"]:
        for feature in ALL_VISUAL_FEATURES:
            groups = [
                sub[feature].dropna().to_numpy()
                for _, sub in features.groupby(grouping)
                if len(sub[feature].dropna()) > 0
            ]
            if len(groups) < 2:
                continue
            statistic, p_value = kruskal(*groups)
            n = sum(len(group) for group in groups)
            k = len(groups)
            epsilon_squared = max(0.0, float((statistic - k + 1) / max(n - k, 1)))
            rows.append(
                {
                    "grouping": grouping,
                    "feature": feature,
                    "groups": k,
                    "n": n,
                    "kruskal_h": float(statistic),
                    "p_value": float(p_value),
                    "epsilon_squared": epsilon_squared,
                }
            )
    out = pd.DataFrame(rows)
    out["holm_p_value_within_grouping"] = np.nan
    for grouping, index in out.groupby("grouping").groups.items():
        out.loc[index, "holm_p_value_within_grouping"] = holm_adjust(
            out.loc[index, "p_value"].to_numpy()
        )
    return out.sort_values(["grouping", "epsilon_squared"], ascending=[True, False])


def make_visual_model(model_name):
    if model_name == "majority":
        return DummyClassifier(strategy="most_frequent")

    if model_name == "geometry_only":
        numeric = GEOMETRY_FEATURES
        categorical = []
    elif model_name == "quality_only":
        numeric = QUALITY_FEATURES
        categorical = []
    elif model_name == "all_lowlevel":
        numeric = ALL_VISUAL_FEATURES
        categorical = []
    elif model_name == "all_lowlevel_platform":
        numeric = ALL_VISUAL_FEATURES
        categorical = ["platform"]
    else:
        raise ValueError(model_name)

    transformers = [
        (
            "numeric",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric,
        )
    ]
    if categorical:
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)
        )
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def fixed_balanced_accuracy(y_true, y_pred, labels):
    _, recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return float(np.mean(recall))


def metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": fixed_balanced_accuracy(y_true, y_pred, labels),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def stratified_bootstrap_ci(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = []
        for label in labels:
            indices = np.flatnonzero(y_true == label)
            if len(indices):
                sampled.append(rng.choice(indices, size=len(indices), replace=True))
        index = np.concatenate(sampled)
        rng.shuffle(index)
        values.append(metrics(y_true[index], y_pred[index], labels)["macro_f1"])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def run_baselines(features, assignments):
    result_rows = []
    per_class_rows = []
    model_names = [
        "majority",
        "geometry_only",
        "quality_only",
        "all_lowlevel",
        "all_lowlevel_platform",
    ]
    input_cols = ALL_VISUAL_FEATURES + ["platform"]

    for version, split_name in PROTOCOLS:
        assigned = assignments[
            (assignments["version"] == version)
            & (assignments["split_name"] == split_name)
        ]
        data = features.merge(assigned[["row_id", "split"]], on="row_id", how="inner")
        train = data[data["split"] == "train"]
        test = data[data["split"] == "test"]
        for model_name in model_names:
            model = make_visual_model(model_name)
            model.fit(train[input_cols], train["predicted_crisis_type"])
            y_true = test["predicted_crisis_type"].to_numpy()
            y_pred = model.predict(test[input_cols])
            values = metrics(y_true, y_pred, CLASSES)
            ci_lo, ci_hi = stratified_bootstrap_ci(y_true, y_pred, CLASSES)
            result_rows.append(
                {
                    "task": "crisis_type",
                    "version": version,
                    "split_name": split_name,
                    "model": model_name,
                    "train_n": int(len(train)),
                    "test_n": int(len(test)),
                    **values,
                    "macro_f1_ci_lo": ci_lo,
                    "macro_f1_ci_hi": ci_hi,
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true,
                y_pred,
                labels=CLASSES,
                zero_division=0,
            )
            for index, label in enumerate(CLASSES):
                per_class_rows.append(
                    {
                        "task": "crisis_type",
                        "version": version,
                        "split_name": split_name,
                        "model": model_name,
                        "class": label,
                        "support": int(support[index]),
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(f1[index]),
                    }
                )

    version = "v2_duplicate_safe"
    split_name = "balanced_original_post_group"
    assigned = assignments[
        (assignments["version"] == version)
        & (assignments["split_name"] == split_name)
    ]
    data = features.merge(assigned[["row_id", "split"]], on="row_id", how="inner")
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    info_labels = ["informative", "uninformative"]
    for model_name in model_names:
        model = make_visual_model(model_name)
        model.fit(train[input_cols], train["predicted_informativeness"])
        y_true = test["predicted_informativeness"].to_numpy()
        y_pred = model.predict(test[input_cols])
        values = metrics(y_true, y_pred, info_labels)
        ci_lo, ci_hi = stratified_bootstrap_ci(y_true, y_pred, info_labels)
        result_rows.append(
            {
                "task": "informativeness",
                "version": version,
                "split_name": split_name,
                "model": model_name,
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                **values,
                "macro_f1_ci_lo": ci_lo,
                "macro_f1_ci_hi": ci_hi,
            }
        )
    return pd.DataFrame(result_rows), pd.DataFrame(per_class_rows)


def perceptual_leakage(features, assignments):
    rows = []
    grouped = features[
        features["perceptual_duplicate_group"].notna()
    ][["row_id", "perceptual_duplicate_group"]]
    for version, split_name in PROTOCOLS:
        assigned = assignments[
            (assignments["version"] == version)
            & (assignments["split_name"] == split_name)
        ]
        data = assigned.merge(grouped, on="row_id", how="left")
        train_groups = set(
            data.loc[data["split"] == "train", "perceptual_duplicate_group"].dropna()
        )
        test = data[data["split"] == "test"]
        test_grouped = test["perceptual_duplicate_group"].notna()
        leaked = test["perceptual_duplicate_group"].isin(train_groups)
        rows.append(
            {
                "version": version,
                "split_name": split_name,
                "test_n": int(len(test)),
                "test_rows_in_perceptual_group": int(test_grouped.sum()),
                "test_rows_with_perceptual_group_seen_in_train": int(leaked.sum()),
                "test_perceptual_leakage_rate_all_rows": float(leaked.mean()),
                "test_perceptual_leakage_rate_grouped_rows": float(
                    leaked[test_grouped].mean()
                )
                if test_grouped.any()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def quality_summary(features):
    flag_cols = [col for col in features.columns if col.startswith("flag_")]
    rows = []
    for dimension in ["overall", "predicted_crisis_type", "platform"]:
        groups = [("all", features)] if dimension == "overall" else features.groupby(dimension)
        for name, sub in groups:
            row = {
                "dimension": dimension,
                "name": name,
                "n": int(len(sub)),
                "has_any_quality_flag_rate": float(sub["has_any_quality_flag"].mean()),
                "mean_quality_flag_count": float(sub["quality_flag_count"].mean()),
            }
            for flag in flag_cols:
                row[f"{flag}_rate"] = float(sub[flag].mean())
            for feature in [
                "luminance_mean",
                "luminance_std",
                "luminance_entropy",
                "saturation_mean",
                "colorfulness",
                "sharpness_laplacian_var",
                "edge_density",
            ]:
                row[f"{feature}_median"] = float(sub[feature].median())
            rows.append(row)
    return pd.DataFrame(rows)


def quality_filter_impact(features, manifest):
    version_masks = {
        "all_labeled": pd.Series(True, index=features.index),
    }
    manifest_flags = manifest[
        ["row_id", "include_v2_duplicate_safe", "include_v3_post_safe_recommended"]
    ]
    data = features.merge(manifest_flags, on="row_id", how="left")
    version_masks = {
        "all_labeled": pd.Series(True, index=data.index),
        "v2_duplicate_safe": data["include_v2_duplicate_safe"].astype(bool),
        "v3_post_safe_recommended": data[
            "include_v3_post_safe_recommended"
        ].astype(bool),
    }
    exposure_flags = (
        data["flag_very_dark"]
        | data["flag_very_bright"]
        | data["flag_low_contrast"]
        | data["flag_heavy_shadow_clipping"]
        | data["flag_heavy_highlight_clipping"]
    )
    relative_quality_flags = (
        data["flag_relative_blur_bottom_5pct"]
        | data["flag_low_entropy_bottom_5pct"]
    )
    policies = {
        "none": pd.Series(True, index=data.index),
        "remove_any_quality_flag": ~data["has_any_quality_flag"],
        "remove_extreme_exposure_or_contrast": ~exposure_flags,
        "remove_relative_blur_or_low_entropy": ~relative_quality_flags,
        "perceptual_dedup_keep_first": pd.Series(True, index=data.index),
    }
    rows = []
    for version, version_mask in version_masks.items():
        version_data = data[version_mask]
        raw_counts = version_data["predicted_crisis_type"].value_counts()
        for policy, policy_mask in policies.items():
            if policy == "perceptual_dedup_keep_first":
                local_keep = (
                    ~version_data.duplicated(
                        "perceptual_duplicate_group", keep="first"
                    )
                    | version_data["perceptual_duplicate_group"].isna()
                )
                kept = version_data[local_keep]
            else:
                kept = data[version_mask & policy_mask]
            kept_counts = kept["predicted_crisis_type"].value_counts()
            rows.append(
                {
                    "version": version,
                    "policy": policy,
                    "dimension": "overall",
                    "name": "all",
                    "raw_n": int(len(version_data)),
                    "kept_n": int(len(kept)),
                    "retention_rate": float(len(kept) / max(len(version_data), 1)),
                }
            )
            for cls in CLASSES:
                raw_n = int(raw_counts.get(cls, 0))
                kept_n = int(kept_counts.get(cls, 0))
                rows.append(
                    {
                        "version": version,
                        "policy": policy,
                        "dimension": "class",
                        "name": cls,
                        "raw_n": raw_n,
                        "kept_n": kept_n,
                        "retention_rate": float(kept_n / raw_n) if raw_n else None,
                    }
                )
    return pd.DataFrame(rows)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    assignments = pd.concat(
        [
            pd.read_csv(ASSIGNMENTS_PATH),
            pd.read_csv(CONSTRAINED_ASSIGNMENTS_PATH),
        ],
        ignore_index=True,
    )

    cache_path = OUT / "image_content_quality_features_unclustered.csv"
    if cache_path.exists():
        features = pd.read_csv(cache_path)
        print(f"Loaded cached visual features: {len(features)}", flush=True)
    else:
        feature_rows = []
        for index, (_, row) in enumerate(manifest.iterrows(), 1):
            feature_rows.append(extract_image_features(row))
            if index % 500 == 0:
                print(f"Extracted visual features: {index}/{len(manifest)}", flush=True)
        features = pd.DataFrame(feature_rows)
        features.to_csv(cache_path, index=False)
    features, thresholds = add_quality_flags(features)
    features, duplicate_pairs = find_perceptual_duplicates(features)

    associations = association_tests(features)
    quality = quality_summary(features)
    filter_impact = quality_filter_impact(features, manifest)
    leakage = perceptual_leakage(features, assignments)
    baselines, per_class = run_baselines(features, assignments)

    group_summary_rows = []
    duplicate_grouped = features[features["perceptual_duplicate_group"].notna()]
    for group, sub in duplicate_grouped.groupby("perceptual_duplicate_group"):
        group_summary_rows.append(
            {
                "perceptual_duplicate_group": group,
                "n": int(len(sub)),
                "crisis_classes": int(sub["predicted_crisis_type"].nunique()),
                "informativeness_classes": int(
                    sub["predicted_informativeness"].nunique()
                ),
                "platforms": int(sub["platform"].nunique()),
                "source_queries": int(sub["source_query"].nunique()),
                "contains_exact_sha_duplicate": bool(sub["sha256"].duplicated().any()),
                "row_ids": "|".join(map(str, sorted(sub["row_id"].tolist()))),
            }
        )
    group_summary = pd.DataFrame(group_summary_rows)

    features.to_csv(OUT / "image_content_quality_features.csv", index=False)
    quality.to_csv(OUT / "image_content_quality_summary.csv", index=False)
    filter_impact.to_csv(OUT / "image_quality_filter_impact.csv", index=False)
    associations.to_csv(OUT / "image_content_association_tests.csv", index=False)
    duplicate_pairs.to_csv(OUT / "perceptual_duplicate_pairs.csv", index=False)
    group_summary.to_csv(OUT / "perceptual_duplicate_groups.csv", index=False)
    leakage.to_csv(OUT / "perceptual_duplicate_split_leakage.csv", index=False)
    baselines.to_csv(OUT / "image_content_baselines.csv", index=False)
    per_class.to_csv(OUT / "image_content_baseline_per_class.csv", index=False)

    payload = {
        "configuration": {
            "seed": SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "perceptual_duplicate_rule": "pHash Hamming <= 6, dHash Hamming <= 10, and absolute log aspect-ratio gap <= 0.08.",
            "quality_flag_thresholds": thresholds,
            "semantic_alignment_limitation": "No pretrained vision-language model was available locally; these features measure intrinsic/low-level visual properties, not semantic image-text alignment.",
        },
        "counts": {
            "images": int(len(features)),
            "quality_flagged_images": int(features["has_any_quality_flag"].sum()),
            "perceptual_duplicate_pairs": int(len(duplicate_pairs)),
            "perceptual_duplicate_groups": int(len(group_summary)),
            "rows_in_perceptual_duplicate_groups": int(
                features["in_perceptual_duplicate_group"].sum()
            ),
            "cross_crisis_perceptual_groups": int(
                (group_summary["crisis_classes"] > 1).sum()
            )
            if len(group_summary)
            else 0,
        },
        "quality_summary": quality.to_dict(orient="records"),
        "quality_filter_impact": filter_impact.to_dict(orient="records"),
        "top_associations": associations.groupby("grouping").head(10).to_dict(
            orient="records"
        ),
        "perceptual_leakage": leakage.to_dict(orient="records"),
        "baseline_results": baselines.to_dict(orient="records"),
    }
    with (OUT / "image_content_quality_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    compact = {
        "counts": payload["counts"],
        "perceptual_leakage": payload["perceptual_leakage"],
        "best_crisis_baselines": (
            baselines[baselines["task"] == "crisis_type"]
            .sort_values(
                ["version", "split_name", "macro_f1"],
                ascending=[True, True, False],
            )
            .groupby(["version", "split_name"])
            .head(2)
            .to_dict(orient="records")
        ),
        "informativeness_baselines": baselines[
            baselines["task"] == "informativeness"
        ].to_dict(orient="records"),
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
