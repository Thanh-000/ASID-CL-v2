from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all CrisisMMD v2.0 media files.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.workspace_root.resolve()
    media_root = root / "external_data" / "crisismmd_v2" / "media"
    dataset = load_crisismmd_v2_annotations(
        root / "external_data" / "crisismmd_v2",
        task="humanitarian",
        agreed_labels=False,
        media_root=media_root,
        require_images=True,
    )
    rows = []
    for position, source in dataset.frame.iterrows():
        path = Path(source["image_abspath"])
        error = ""
        width = height = 0
        image_format = mode = ""
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image_format = str(image.format or "")
                mode = str(image.mode)
        except Exception as exc:  # Preserve every decode failure in the audit artifact.
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "row_id": source["row_id"],
                "tweet_id": source["tweet_id"],
                "image_id": source["image_id"],
                "event_name": source["event_name"],
                "image_relative_path": source["image"],
                "file_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
                "decode_error": error,
            }
        )
        if (position + 1) % 1000 == 0:
            print(f"media audit: {position + 1}/{len(dataset.frame)}", flush=True)
    inventory = pd.DataFrame(rows)
    annotated = set(dataset.frame["image"].astype(str))
    extracted_root = media_root / "CrisisMMD_v2.0"
    physical = {
        path.relative_to(extracted_root).as_posix()
        for path in (extracted_root / "data_image").rglob("*")
        if path.is_file()
    }
    duplicate_rows = int(inventory["sha256"].duplicated(keep=False).sum())
    duplicate_groups = int(
        (inventory.groupby("sha256").size() > 1).sum()
    )
    summary = {
        "dataset": dataset.provenance.as_dict(),
        "annotated_images": len(annotated),
        "physical_images": len(physical),
        "missing_annotated_images": len(annotated - physical),
        "extra_physical_images": len(physical - annotated),
        "decode_errors": int((inventory["decode_error"] != "").sum()),
        "zero_dimension_images": int(
            ((inventory["width"] <= 0) | (inventory["height"] <= 0)).sum()
        ),
        "exact_duplicate_rows": duplicate_rows,
        "exact_duplicate_groups": duplicate_groups,
        "formats": {
            str(key): int(value)
            for key, value in inventory["format"].value_counts().sort_index().items()
        },
        "file_bytes": {
            "total": int(inventory["file_bytes"].sum()),
            "minimum": int(inventory["file_bytes"].min()),
            "median": float(inventory["file_bytes"].median()),
            "maximum": int(inventory["file_bytes"].max()),
        },
    }
    output = root / "analysis_outputs"
    inventory.to_csv(output / "crisismmd_v2_media_inventory.csv", index=False)
    (output / "crisismmd_v2_media_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
