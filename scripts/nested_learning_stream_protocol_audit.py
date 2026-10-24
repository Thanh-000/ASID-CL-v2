from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


SCHEMES = {
    "class_incremental": ("predicted_crisis_type", "primary proxy class-incremental CL"),
    "crawl_domain_incremental": ("original_crawl_cat", "secondary domain-incremental CL"),
    "platform_incremental": ("platform", "two-domain diagnostic only"),
    "crawl_platform_incremental": ("crawl_platform", "fine-grained domain stress test"),
    "raw_query_incremental": ("source_query", "feasibility diagnostic; not a main stream"),
}

VERSION_FLAGS = {
    "v2_duplicate_safe": "include_v2_duplicate_safe",
    "v3_post_safe_recommended": "include_v3_post_safe_recommended",
}

IDENTIFIERS = {
    "source_query": "source_query",
    "original_post": "orig_stem",
    "image_sha": "sha256",
    "normalized_text": "normalized_text",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_true(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def median(values: Iterable[int]) -> float:
    items = list(values)
    return float(statistics.median(items)) if items else float("nan")


def normalized_entropy(labels: Iterable[str]) -> float:
    counts = Counter(labels)
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
    return entropy / math.log2(len(counts))


def load_split_maps(output_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    maps: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    standard = read_csv(output_dir / "dataset_version_split_assignments.csv")
    constrained = read_csv(output_dir / "constrained_source_query_split_assignments.csv")
    joint = read_csv(output_dir / "joint_query_post_split_assignments.csv")
    for row in [*standard, *constrained, *joint]:
        version = row["version"]
        split_name = row["split_name"]
        if version not in VERSION_FLAGS:
            continue
        if split_name not in {
            "balanced_original_post_group",
            "constrained_source_query_group",
            "joint_source_query_original_post_group",
        }:
            continue
        maps[(version, split_name)][row["row_id"]] = row["split"]
    return maps


def add_derived_fields(row: dict[str, str]) -> dict[str, str]:
    enriched = dict(row)
    enriched["crawl_platform"] = f"{row['original_crawl_cat']}::{row['platform']}"
    return enriched


def test_seen_rate(
    rows: list[dict[str, str]], split_map: dict[str, str], field: str
) -> tuple[int, int, float]:
    train_values = {
        row[field]
        for row in rows
        if split_map.get(row["row_id"]) == "train" and row.get(field, "")
    }
    test_rows = [row for row in rows if split_map.get(row["row_id"]) == "test"]
    seen = sum(bool(row.get(field, "")) and row[field] in train_values for row in test_rows)
    total = len(test_rows)
    return seen, total, seen / total if total else float("nan")


def recommend_scheme(
    scheme: str,
    num_streams: int,
    viable_streams: int,
    min_train: int,
    min_test: int,
) -> str:
    if scheme == "class_incremental":
        if num_streams == 6 and viable_streams == 6:
            return "USE_PRIMARY_PROXY"
        return "USE_WITH_CLASS_SCARCITY_WARNING"
    if scheme == "crawl_domain_incremental":
        return "USE_SECONDARY" if viable_streams >= 5 else "SENSITIVITY_ONLY"
    if scheme == "platform_incremental":
        return "DIAGNOSTIC_ONLY"
    if scheme == "crawl_platform_incremental":
        return "SENSITIVITY_ONLY" if min_train >= 5 and min_test >= 2 else "TOO_SPARSE"
    if scheme == "raw_query_incremental":
        return "REJECT_AS_MAIN_STREAM"
    return "REVIEW"


def audit(
    manifest: list[dict[str, str]],
    split_maps: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    leakage_payload: dict[str, object] = {}

    for version, flag in VERSION_FLAGS.items():
        version_rows = [add_derived_fields(row) for row in manifest if is_true(row.get(flag))]
        for protocol in (
            "balanced_original_post_group",
            "constrained_source_query_group",
            "joint_source_query_original_post_group",
        ):
            split_map = split_maps[(version, protocol)]
            if len(split_map) != len(version_rows):
                raise ValueError(
                    f"Split coverage mismatch for {version}/{protocol}: "
                    f"{len(split_map)} assignments vs {len(version_rows)} rows"
                )

            leakage_key = f"{version}/{protocol}"
            leakage_payload[leakage_key] = {}
            for name, field in IDENTIFIERS.items():
                seen, total, rate = test_seen_rate(version_rows, split_map, field)
                leakage_payload[leakage_key][name] = {
                    "test_rows_seen_in_train": seen,
                    "test_rows": total,
                    "rate": rate,
                }

            for scheme, (stream_field, role) in SCHEMES.items():
                grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
                for row in version_rows:
                    grouped[row[stream_field]].append(row)

                scheme_details: list[dict[str, object]] = []
                for stream_id in sorted(grouped):
                    rows = grouped[stream_id]
                    split_counts = Counter(split_map[row["row_id"]] for row in rows)
                    train_labels = [
                        row["predicted_crisis_type"]
                        for row in rows
                        if split_map[row["row_id"]] == "train"
                    ]
                    test_labels = [
                        row["predicted_crisis_type"]
                        for row in rows
                        if split_map[row["row_id"]] == "test"
                    ]
                    train_n = split_counts["train"]
                    val_n = split_counts["val"]
                    test_n = split_counts["test"]
                    viable = train_n >= 20 and test_n >= 10
                    row_out: dict[str, object] = {
                        "version": version,
                        "protocol": protocol,
                        "scheme": scheme,
                        "stream_id": stream_id,
                        "role": role,
                        "total_n": len(rows),
                        "train_n": train_n,
                        "val_n": val_n,
                        "test_n": test_n,
                        "train_target_classes": len(set(train_labels)),
                        "test_target_classes": len(set(test_labels)),
                        "train_label_entropy_norm": round(normalized_entropy(train_labels), 6),
                        "test_label_entropy_norm": round(normalized_entropy(test_labels), 6),
                        "viable_train20_test10": viable,
                    }
                    detail_rows.append(row_out)
                    scheme_details.append(row_out)

                train_sizes = [int(row["train_n"]) for row in scheme_details]
                val_sizes = [int(row["val_n"]) for row in scheme_details]
                test_sizes = [int(row["test_n"]) for row in scheme_details]
                viable_streams = sum(bool(row["viable_train20_test10"]) for row in scheme_details)
                complete_streams = sum(
                    int(row["train_n"]) > 0
                    and int(row["val_n"]) > 0
                    and int(row["test_n"]) > 0
                    for row in scheme_details
                )
                min_train = min(train_sizes, default=0)
                min_test = min(test_sizes, default=0)
                summary_rows.append(
                    {
                        "version": version,
                        "protocol": protocol,
                        "scheme": scheme,
                        "role": role,
                        "num_streams": len(scheme_details),
                        "complete_streams": complete_streams,
                        "viable_train20_test10": viable_streams,
                        "train_min": min_train,
                        "train_median": median(train_sizes),
                        "train_max": max(train_sizes, default=0),
                        "val_min": min(val_sizes, default=0),
                        "val_median": median(val_sizes),
                        "val_max": max(val_sizes, default=0),
                        "test_min": min_test,
                        "test_median": median(test_sizes),
                        "test_max": max(test_sizes, default=0),
                        "recommendation": recommend_scheme(
                            scheme,
                            len(scheme_details),
                            viable_streams,
                            min_train,
                            min_test,
                        ),
                    }
                )

    payload = {
        "scope": {
            "versions": list(VERSION_FLAGS),
            "protocols": [
                "balanced_original_post_group",
                "constrained_source_query_group",
                "joint_source_query_original_post_group",
            ],
            "viability_rule": "train_n >= 20 and test_n >= 10 per stream",
            "chronology_available": False,
        },
        "leakage": leakage_payload,
        "summary": summary_rows,
        "interpretation": {
            "primary": (
                "Class-incremental crisis streams are feasible as a proxy, but are not "
                "chronological event streams."
            ),
            "secondary": (
                "Original crawl category can define a domain-incremental sensitivity stream."
            ),
            "reject": (
                "Raw source-query streams are too fragmented and collection-defined for the "
                "main continual-learning claim."
            ),
        },
    }
    return summary_rows, detail_rows, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit UK-CrisisMMD stream protocols for Nested Learning experiments."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root containing analysis_outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = root / "analysis_outputs"
    manifest = read_csv(output_dir / "dataset_version_manifest.csv")
    split_maps = load_split_maps(output_dir)
    summary_rows, detail_rows, payload = audit(manifest, split_maps)

    summary_path = output_dir / "nested_learning_stream_protocol_summary.csv"
    detail_path = output_dir / "nested_learning_stream_protocol_details.csv"
    metrics_path = output_dir / "nested_learning_stream_protocol_metrics.json"
    write_csv(summary_path, summary_rows)
    write_csv(detail_path, detail_rows)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {detail_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
