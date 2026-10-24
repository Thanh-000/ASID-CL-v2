from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations


def _audit_variant(
    inventory: pd.DataFrame,
    annotations: pd.DataFrame,
    variant: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    merged = annotations[
        [
            "row_id",
            "image_id",
            "tweet_id",
            "split",
            "label_name",
            "event_name",
            "image_abspath",
        ]
    ].merge(
        inventory[["row_id", "sha256", "file_bytes"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    missing = merged["sha256"].isna()
    for index, path_text in merged.loc[missing, "image_abspath"].items():
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"{variant}: missing media file {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        merged.at[index, "sha256"] = digest.hexdigest()
        merged.at[index, "file_bytes"] = path.stat().st_size

    records: list[dict[str, object]] = []
    for sha256, group in merged.groupby("sha256", sort=True):
        if len(group) < 2:
            continue
        splits = sorted(group["split"].unique())
        labels = sorted(group["label_name"].unique())
        records.append(
            {
                "variant": variant,
                "sha256": sha256,
                "rows": len(group),
                "unique_tweets": group["tweet_id"].nunique(),
                "unique_events": group["event_name"].nunique(),
                "splits": ";".join(splits),
                "split_count": len(splits),
                "labels": ";".join(labels),
                "label_count": len(labels),
                "row_ids": ";".join(group["row_id"]),
            }
        )
    groups = pd.DataFrame.from_records(records)
    if groups.empty:
        groups = pd.DataFrame(
            columns=[
                "variant",
                "sha256",
                "rows",
                "unique_tweets",
                "unique_events",
                "splits",
                "split_count",
                "labels",
                "label_count",
                "row_ids",
            ]
        )

    cross_split = groups[groups["split_count"] > 1]
    conflicting = groups[groups["label_count"] > 1]
    summary: dict[str, object] = {
        "variant": variant,
        "annotation_rows": len(merged),
        "unique_content_hashes": merged["sha256"].nunique(),
        "duplicate_groups": len(groups),
        "duplicate_rows": int(groups["rows"].sum()) if len(groups) else 0,
        "cross_split_duplicate_groups": len(cross_split),
        "cross_split_duplicate_rows": int(cross_split["rows"].sum())
        if len(cross_split)
        else 0,
        "label_conflict_duplicate_groups": len(conflicting),
        "label_conflict_duplicate_rows": int(conflicting["rows"].sum())
        if len(conflicting)
        else 0,
    }
    return summary, groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact image-content duplication across CrisisMMD v2 splits."
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("external_data/crisismmd_v2")
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("analysis_outputs/crisismmd_v2_media_inventory.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("analysis_outputs")
    )
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory, dtype={"row_id": str, "sha256": str})
    variants = (
        (
            "humanitarian_agreed",
            args.data_root / "crisismmd_datasplit_agreed_label",
            "humanitarian",
            True,
        ),
        (
            "informative_agreed",
            args.data_root / "crisismmd_datasplit_agreed_label",
            "informative",
            True,
        ),
        (
            "humanitarian_all",
            args.data_root / "crisismmd_datasplit_all",
            "humanitarian",
            False,
        ),
        (
            "damage_all_official",
            args.data_root / "crisismmd_datasplit_all",
            "damage",
            False,
        ),
    )

    summaries: list[dict[str, object]] = []
    group_frames: list[pd.DataFrame] = []
    for name, root, task, agreed in variants:
        dataset = load_crisismmd_v2_annotations(
            root,
            task=task,
            agreed_labels=agreed,
            media_root=args.data_root / "media" / "CrisisMMD_v2.0",
            strict_official=True,
        )
        summary, groups = _audit_variant(inventory, dataset.frame, name)
        summaries.append(summary)
        group_frames.append(groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "crisismmd_v2_exact_duplicate_split_audit.json"
    csv_path = args.output_dir / "crisismmd_v2_exact_duplicate_groups.csv"
    json_path.write_text(
        json.dumps({"variants": summaries}, indent=2), encoding="utf-8"
    )
    pd.concat(group_frames, ignore_index=True).to_csv(csv_path, index=False)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "variants": summaries}, indent=2))


if __name__ == "__main__":
    main()
