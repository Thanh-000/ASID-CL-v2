import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon


OUT = Path("analysis_outputs")
MANIFEST_PATH = OUT / "dataset_version_manifest.csv"
SEED = 2026

CLASSES = [
    "major_accidents",
    "natural_disaster",
    "pandemic_health",
    "political_crisis",
    "social_unrest",
    "strikes",
]

VERSIONS = {
    "v2_duplicate_safe": "include_v2_duplicate_safe",
    "v3_post_safe_recommended": "include_v3_post_safe_recommended",
}

SPLITS = ["train", "val", "test"]
TARGET_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
SIZE_BOUNDS = {
    "train": (0.55, 0.82),
    "val": (0.05, 0.22),
    "test": (0.13, 0.35),
}


def js_divergence(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return None
    return float(jensenshannon(a / a.sum(), b / b.sum(), base=2.0) ** 2)


def build_group_table(df):
    query_values = sorted(df["source_query"].fillna("<missing>").astype(str).unique())
    rows = []
    for query in query_values:
        sub = df[df["source_query"].fillna("<missing>").astype(str) == query]
        rec = {"source_query": query, "n": int(len(sub))}
        counts = sub["predicted_crisis_type"].value_counts()
        for cls in CLASSES:
            rec[cls] = int(counts.get(cls, 0))
        rows.append(rec)
    return pd.DataFrame(rows)


def solve_assignment(group_table):
    group_n = len(group_table)
    split_n = len(SPLITS)
    class_n = len(CLASSES)
    x_n = group_n * split_n

    class_slack_start = x_n
    class_slack_n = split_n * class_n * 2
    size_slack_start = class_slack_start + class_slack_n
    size_slack_n = split_n * 2
    variable_n = size_slack_start + size_slack_n

    def x_idx(group_i, split_i):
        return group_i * split_n + split_i

    def class_slack_idx(split_i, class_i, positive):
        offset = (split_i * class_n + class_i) * 2
        return class_slack_start + offset + (0 if positive else 1)

    def size_slack_idx(split_i, positive):
        return size_slack_start + split_i * 2 + (0 if positive else 1)

    objective = np.zeros(variable_n, dtype=float)
    rng = np.random.default_rng(SEED)
    objective[:x_n] = rng.uniform(0, 1e-8, size=x_n)

    total_n = int(group_table["n"].sum())
    total_class = group_table[CLASSES].sum()
    for split_i, split in enumerate(SPLITS):
        target_fraction = TARGET_FRACTIONS[split]
        for class_i, cls in enumerate(CLASSES):
            target = max(float(total_class[cls] * target_fraction), 1.0)
            weight = 1.0 / target
            objective[class_slack_idx(split_i, class_i, True)] = weight
            objective[class_slack_idx(split_i, class_i, False)] = weight
        size_weight = 2.0 / max(total_n * target_fraction, 1.0)
        objective[size_slack_idx(split_i, True)] = size_weight
        objective[size_slack_idx(split_i, False)] = size_weight

    equality_rows = []
    equality_lower = []
    equality_upper = []

    for group_i in range(group_n):
        row = {}
        for split_i in range(split_n):
            row[x_idx(group_i, split_i)] = 1.0
        equality_rows.append(row)
        equality_lower.append(1.0)
        equality_upper.append(1.0)

    for split_i, split in enumerate(SPLITS):
        fraction = TARGET_FRACTIONS[split]
        for class_i, cls in enumerate(CLASSES):
            row = {}
            for group_i, count in enumerate(group_table[cls].to_numpy()):
                if count:
                    row[x_idx(group_i, split_i)] = float(count)
            row[class_slack_idx(split_i, class_i, True)] = -1.0
            row[class_slack_idx(split_i, class_i, False)] = 1.0
            target = float(total_class[cls] * fraction)
            equality_rows.append(row)
            equality_lower.append(target)
            equality_upper.append(target)

        row = {}
        for group_i, count in enumerate(group_table["n"].to_numpy()):
            row[x_idx(group_i, split_i)] = float(count)
        row[size_slack_idx(split_i, True)] = -1.0
        row[size_slack_idx(split_i, False)] = 1.0
        target = float(total_n * fraction)
        equality_rows.append(row)
        equality_lower.append(target)
        equality_upper.append(target)

    inequality_rows = []
    inequality_lower = []
    inequality_upper = []

    for split_i, split in enumerate(SPLITS):
        lower_fraction, upper_fraction = SIZE_BOUNDS[split]
        row = {}
        for group_i, count in enumerate(group_table["n"].to_numpy()):
            row[x_idx(group_i, split_i)] = float(count)
        inequality_rows.append(row)
        inequality_lower.append(float(total_n * lower_fraction))
        inequality_upper.append(float(total_n * upper_fraction))

        for cls in CLASSES:
            row = {}
            for group_i, count in enumerate(group_table[cls].to_numpy()):
                if count:
                    row[x_idx(group_i, split_i)] = float(count)
            cls_total = int(total_class[cls])
            if split == "train":
                minimum = max(3, int(np.floor(cls_total * 0.35)))
            elif split == "test":
                minimum = max(2, int(np.floor(cls_total * 0.04)))
            else:
                minimum = max(1, int(np.floor(cls_total * 0.02)))
            inequality_rows.append(row)
            inequality_lower.append(float(minimum))
            inequality_upper.append(np.inf)

    def sparse_rows(rows):
        data = []
        row_idx = []
        col_idx = []
        for i, row in enumerate(rows):
            for j, value in row.items():
                row_idx.append(i)
                col_idx.append(j)
                data.append(value)
        return csr_matrix((data, (row_idx, col_idx)), shape=(len(rows), variable_n))

    constraints = [
        LinearConstraint(
            sparse_rows(equality_rows),
            np.asarray(equality_lower),
            np.asarray(equality_upper),
        ),
        LinearConstraint(
            sparse_rows(inequality_rows),
            np.asarray(inequality_lower),
            np.asarray(inequality_upper),
        ),
    ]
    lower_bounds = np.zeros(variable_n)
    upper_bounds = np.full(variable_n, np.inf)
    upper_bounds[:x_n] = 1.0
    integrality = np.zeros(variable_n, dtype=int)
    integrality[:x_n] = 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed: status={result.status}, message={result.message}")

    assignment = []
    for group_i in range(group_n):
        values = [result.x[x_idx(group_i, split_i)] for split_i in range(split_n)]
        assignment.append(SPLITS[int(np.argmax(values))])
    return assignment, result


def summarize(df, version, result):
    overall = (
        df["predicted_crisis_type"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_numpy()
    )
    row = {
        "version": version,
        "split_name": "constrained_source_query_group",
        "n": int(len(df)),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
        "source_query_overlap_train_test": 0.0,
    }
    for split in SPLITS:
        sub = df[df["split"] == split]
        counts = (
            sub["predicted_crisis_type"]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
        )
        row[f"{split}_n"] = int(len(sub))
        row[f"{split}_pct"] = float(len(sub) / len(df))
        row[f"{split}_js_vs_overall"] = js_divergence(counts.to_numpy(), overall)
        row[f"{split}_missing_classes"] = int((counts == 0).sum())
        row[f"{split}_source_queries"] = int(sub["source_query"].nunique())
        for cls in CLASSES:
            row[f"{split}_count_{cls}"] = int(counts[cls])
    return row


def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    assignment_frames = []
    summary_rows = []
    group_frames = []

    for version, include_col in VERSIONS.items():
        subset = manifest[manifest[include_col].astype(bool)].copy()
        group_table = build_group_table(subset)
        assignments, result = solve_assignment(group_table)
        group_table["version"] = version
        group_table["split_name"] = "constrained_source_query_group"
        group_table["split"] = assignments
        group_frames.append(group_table)

        query_to_split = dict(zip(group_table["source_query"], group_table["split"]))
        subset["split"] = subset["source_query"].fillna("<missing>").astype(str).map(
            query_to_split
        )
        assignment_frames.append(
            subset[["row_id", "split"]].assign(
                version=version,
                split_name="constrained_source_query_group",
            )[["row_id", "version", "split_name", "split"]]
        )
        summary_rows.append(summarize(subset, version, result))

    assignments_out = pd.concat(assignment_frames, ignore_index=True)
    summaries_out = pd.DataFrame(summary_rows)
    groups_out = pd.concat(group_frames, ignore_index=True)
    assignments_out.to_csv(
        OUT / "constrained_source_query_split_assignments.csv", index=False
    )
    summaries_out.to_csv(
        OUT / "constrained_source_query_split_summary.csv", index=False
    )
    groups_out.to_csv(
        OUT / "constrained_source_query_group_assignments.csv", index=False
    )
    with (OUT / "constrained_source_query_split_metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "method": "Mixed-integer linear optimization with query-disjoint assignment, fixed class coverage constraints, and weighted L1 deviation from 70/10/20 class and size targets.",
                "seed": SEED,
                "summaries": summary_rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(summaries_out.to_string(index=False))


if __name__ == "__main__":
    main()
