from __future__ import annotations

import json
from pathlib import Path

from experiments.mm_hope.alignment import write_crossmodal_alignment_analysis
from experiments.mm_hope.data import load_joint_v2_dataset
from experiments.mm_hope.encoders import load_embedding_cache
from experiments.mm_hope.phase1_config import EncoderConfig


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset = load_joint_v2_dataset(root)
    cache = load_embedding_cache(root, dataset, EncoderConfig())
    if cache is None:
        raise FileNotFoundError("Phase-1 encoder cache is missing")
    payload = write_crossmodal_alignment_analysis(root, dataset, cache)

    phase1_metrics_path = root / "analysis_outputs" / "mm_hope_phase1_metrics.json"
    if phase1_metrics_path.is_file():
        with phase1_metrics_path.open("r", encoding="utf-8") as handle:
            phase1_payload = json.load(handle)
        phase1_payload["crossmodal_alignment"] = payload
        phase1_payload.setdefault("artifacts", {}).update(
            {
                "crossmodal_alignment_rows": payload["artifacts"]["rows"],
                "crossmodal_alignment_summary": payload["artifacts"]["summary"],
            }
        )
        with phase1_metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(phase1_payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
