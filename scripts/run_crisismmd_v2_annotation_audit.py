from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official CrisisMMD v2.0 annotations.")
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("external_data/crisismmd_v2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/crisismmd_v2_annotation_audit.json"),
    )
    args = parser.parse_args()

    reports: dict[str, object] = {}
    task_modes = (
        ("humanitarian", True),
        ("informative", True),
        ("humanitarian", False),
        ("informative", False),
        ("damage", False),
    )
    for task, agreed in task_modes:
        dataset = load_crisismmd_v2_annotations(
            args.annotation_root,
            task=task,
            agreed_labels=agreed,
            require_images=False,
        )
        frame = dataset.frame
        key = f"{task}_{'agreed' if agreed else 'all'}"
        reports[key] = {
            "provenance": dataset.provenance.as_dict(),
            "labels": list(dataset.labels),
            "events": sorted(frame["event_name"].unique().tolist()),
            "unique_tweets": int(frame["tweet_id"].nunique()),
            "duplicate_image_rows_by_tweet": int(len(frame) - frame["tweet_id"].nunique()),
            "label_counts_by_split": {
                split: {
                    label: int(count)
                    for label, count in frame.loc[frame["split"] == split, "label_name"]
                    .value_counts()
                    .sort_index()
                    .items()
                }
                for split in ("train", "val", "test")
            },
            "cross_split_overlap": {
                column: _cross_split_overlap(frame, column)
                for column in ("tweet_id", "image_id", "tweet_text")
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"wrote {args.output} ({len(reports)} task variants)")


def _cross_split_overlap(frame, column: str) -> int:
    return int((frame.groupby(column, dropna=False)["split"].nunique() > 1).sum())


if __name__ == "__main__":
    main()
