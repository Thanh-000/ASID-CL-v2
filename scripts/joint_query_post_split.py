from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from constrained_source_query_split import (
    CLASSES,
    SEED,
    SPLITS,
    js_divergence,
    solve_assignment,
)


OUT = Path("analysis_outputs")
MANIFEST_PATH = OUT / "dataset_version_manifest.csv"

VERSIONS = {
    "v2_duplicate_safe": "include_v2_duplicate_safe",
    "v3_post_safe_recommended": "include_v3_post_safe_recommended",
}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.lower().isin({"1", "true", "yes"})


def build_components(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    union_find = UnionFind()
    for row in df.itertuples(index=False):
        query_node = f"query::{row.source_query}"
        post_node = f"post::{row.orig_stem}"
        union_find.union(query_node, post_node)

    root_to_rows: dict[str, list[int]] = defaultdict(list)
    for idx, row in df.iterrows():
        root_to_rows[union_find.find(f"query::{row['source_query']}")].append(idx)

    row_component: dict[int, str] = {}
    group_rows: list[dict[str, object]] = []
    for indices in root_to_rows.values():
        sub = df.loc[indices]
        queries = sorted(sub["source_query"].fillna("<missing>").astype(str).unique())
        posts = sorted(sub["orig_stem"].fillna("<missing>").astype(str).unique())
        digest_input = "\n".join([*(f"q:{item}" for item in queries), *(f"p:{item}" for item in posts)])
        component_id = "qp_" + hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:12]
        counts = sub["predicted_crisis_type"].value_counts()
        record: dict[str, object] = {
            "component_id": component_id,
            "n": int(len(sub)),
            "source_queries": len(queries),
            "original_posts": len(posts),
        }
        for cls in CLASSES:
            record[cls] = int(counts.get(cls, 0))
        group_rows.append(record)
        for idx in indices:
            row_component[idx] = component_id

    enriched = df.copy()
    enriched["component_id"] = pd.Series(row_component)
    group_table = pd.DataFrame(group_rows).sort_values("component_id").reset_index(drop=True)
    return enriched, group_table


def seen_test_rate(df: pd.DataFrame, field: str) -> tuple[int, float]:
    train_values = set(df.loc[df["split"] == "train", field].dropna().astype(str))
    test_values = df.loc[df["split"] == "test", field].fillna("").astype(str)
    seen = int(test_values.isin(train_values).sum())
    return seen, float(seen / len(test_values)) if len(test_values) else float("nan")


def summarize(df: pd.DataFrame, version: str, result, component_count: int) -> dict[str, object]:
    overall = (
        df["predicted_crisis_type"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_numpy()
    )
    summary: dict[str, object] = {
        "version": version,
        "split_name": "joint_source_query_original_post_group",
        "n": int(len(df)),
        "components": component_count,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective_value": float(result.fun),
    }
    for split in SPLITS:
        sub = df[df["split"] == split]
        counts = (
            sub["predicted_crisis_type"]
            .value_counts()
            .reindex(CLASSES, fill_value=0)
        )
        summary[f"{split}_n"] = int(len(sub))
        summary[f"{split}_pct"] = float(len(sub) / len(df))
        summary[f"{split}_js_vs_overall"] = js_divergence(counts.to_numpy(), overall)
        summary[f"{split}_missing_classes"] = int((counts == 0).sum())
        for cls in CLASSES:
            summary[f"{split}_count_{cls}"] = int(counts[cls])

    for label, field in {
        "source_query": "source_query",
        "original_post": "orig_stem",
        "image_sha": "sha256",
        "normalized_text": "normalized_text",
    }.items():
        seen, rate = seen_test_rate(df, field)
        summary[f"test_seen_{label}_rows"] = seen
        summary[f"test_seen_{label}_rate"] = rate
    return summary


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    assignment_frames: list[pd.DataFrame] = []
    group_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for version, include_col in VERSIONS.items():
        subset = manifest[as_bool(manifest[include_col])].copy()
        enriched, group_table = build_components(subset)
        assignments, result = solve_assignment(group_table)
        group_table["version"] = version
        group_table["split_name"] = "joint_source_query_original_post_group"
        group_table["split"] = assignments
        group_frames.append(group_table)

        component_to_split = dict(zip(group_table["component_id"], group_table["split"]))
        enriched["split"] = enriched["component_id"].map(component_to_split)
        assignment_frames.append(
            enriched[["row_id", "component_id", "split"]].assign(
                version=version,
                split_name="joint_source_query_original_post_group",
            )[["row_id", "version", "split_name", "component_id", "split"]]
        )
        summaries.append(summarize(enriched, version, result, len(group_table)))

    assignments_out = pd.concat(assignment_frames, ignore_index=True)
    groups_out = pd.concat(group_frames, ignore_index=True)
    summaries_out = pd.DataFrame(summaries)

    assignments_out.to_csv(OUT / "joint_query_post_split_assignments.csv", index=False)
    groups_out.to_csv(OUT / "joint_query_post_group_assignments.csv", index=False)
    summaries_out.to_csv(OUT / "joint_query_post_split_summary.csv", index=False)
    with (OUT / "joint_query_post_split_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": (
                    "Connected components over the bipartite source_query-original_post graph, "
                    "followed by the constrained MILP class/size assignment used by the "
                    "query-disjoint protocol."
                ),
                "seed": SEED,
                "summaries": summaries,
            },
            handle,
            ensure_ascii=True,
            indent=2,
        )
    print(summaries_out.to_string(index=False))


if __name__ == "__main__":
    main()
