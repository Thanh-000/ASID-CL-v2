#!/usr/bin/env python3
"""
Audit Single-Tower vs Dual-Tower LoRA Adaptation Gain on Real CLIP ViT-B/16 Features
(using the real extracted embeddings in analysis_outputs/h1_g1_crisismmd/).

Compares:
  1. Both Towers Frozen (Img_frozen + Txt_frozen)
  2. Vision-Only First-Session LoRA (Img_first + Txt_frozen)  [Single-Tower Vision]
  3. Text-Only First-Session LoRA   (Img_frozen + Txt_first)  [Single-Tower Text]
  4. Dual-Tower First-Session LoRA  (Img_first + Txt_first)   [RanPAC Dual-Tower]
  5. Vision-Only Joint LoRA         (Img_oracle + Txt_frozen) [Single-Tower Vision Ceiling]
  6. Text-Only Joint LoRA           (Img_frozen + Txt_oracle) [Single-Tower Text Ceiling]
  7. Dual-Tower Joint LoRA          (Img_oracle + Txt_oracle) [Dual-Tower Ceiling]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations, regroup_crisismmd_by_tweet  # noqa: E402
from mm_flex_merge.models import BlockAnalyticCL  # noqa: E402

SEEDS = [2026, 2027, 2028, 2029, 2030]


def main():
    ds = regroup_crisismmd_by_tweet(load_crisismmd_v2_annotations(
        ROOT / "external_data" / "crisismmd_v2", task="damage", agreed_labels=False,
        media_root=ROOT / "external_data" / "crisismmd_v2" / "media", require_images=True
    ))
    fr = ds.frame[ds.frame["split"] != "test"].reset_index(drop=True)
    y = torch.tensor(fr["label_index"].to_numpy(np.int64))
    is_tr = fr["split"].to_numpy() == "train"
    is_va = fr["split"].to_numpy() == "val"
    C = 3

    f_fz = np.load(ROOT / "analysis_outputs" / "h1_g1_crisismmd" / "features_frozen.npz")
    f_fs = np.load(ROOT / "analysis_outputs" / "h1_g1_crisismmd" / "features_first_session.npz")
    f_or = np.load(ROOT / "analysis_outputs" / "h1_g1_crisismmd" / "features_joint_oracle.npz")

    arms = {
        "1_both_frozen": (f_fz["img"], f_fz["txt"]),
        "2_vision_only_first_session": (f_fs["img"], f_fz["txt"]),
        "3_text_only_first_session": (f_fz["img"], f_fs["txt"]),
        "4_dual_tower_first_session": (f_fs["img"], f_fs["txt"]),
        "5_vision_only_adapted": (f_or["img"], f_fz["txt"]),
        "6_text_only_adapted": (f_fz["img"], f_or["txt"]),
        "7_dual_tower_adapted": (f_or["img"], f_or["txt"]),
    }

    out = {}
    for name, (img_np, txt_np) in arms.items():
        img = torch.from_numpy(img_np)
        txt = torch.from_numpy(txt_np)
        out[name] = {}
        for beta in [0.0, 0.5]:
            seed_scores = []
            for s in SEEDS:
                h = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=beta, seed=s)
                h.update({"img": img[is_tr], "txt": txt[is_tr]}, y[is_tr])
                pred = h.predict({"img": img[is_va], "txt": txt[is_va]}).numpy()
                yv = y[is_va].numpy()
                per_cls = f1_score(yv, pred, labels=[0, 1, 2], average=None, zero_division=0)
                seed_scores.append({
                    "macro_f1": float(np.mean(per_cls)),
                    "worst_class_f1": float(np.min(per_cls)),
                    "balanced_acc": float(balanced_accuracy_score(yv, pred)),
                    "acc": float(accuracy_score(yv, pred)),
                })
            out[name][f"beta_{beta}"] = {
                k: {
                    "mean": float(np.mean([r[k] for r in seed_scores])),
                    "sd": float(np.std([r[k] for r in seed_scores], ddof=1)),
                }
                for k in seed_scores[0]
            }
        b5 = out[name]["beta_0.5"]
        print(
            f"{name:<30} | Macro-F1={b5['macro_f1']['mean']:.4f}±{b5['macro_f1']['sd']:.4f} | "
            f"BalAcc={b5['balanced_acc']['mean']:.4f} | WorstF1={b5['worst_class_f1']['mean']:.4f} | "
            f"Acc={b5['acc']['mean']:.4f}"
        )

    out_path = ROOT / "analysis_outputs" / "single_vs_dual_tower_lora_audit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
