#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Ablation Grid for 2-Modality Block Analytic Classifier:
  1. Projection dimension P in {256, 512, 1024, 2048} (at beta=0.5, rel_ridge=0.1)
  2. Relative ridge gamma in {0.001, 0.01, 0.1, 1.0, 10.0} (at P=1024, beta=0.5)
  3. Class-balance exponent beta in {0.0, 0.25, 0.5, 0.75, 1.0} (at P=1024, rel_ridge=0.1)
Evaluated across 5 seeds (2027..2031) on UK-CrisisMMD and CrisisMMD v2.0 Humanitarian.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mm_flex_merge.models import BlockAnalyticCL  # noqa: E402
from run_cmml_mil_2modality_audit import load_humanitarian_tasks, load_uk_tasks  # noqa: E402

SEEDS = [2027, 2028, 2029, 2030, 2031]


def eval_config(tasks, test_img, test_txt, test_y, C: int, *, P: int, ridge: float, beta: float):
    f1s, wf1s, baccs = [], [], []
    yt = test_y.numpy()
    for s in SEEDS:
        m = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=P, ridge=ridge, class_balance_beta=beta, seed=s)
        for td in tasks:
            m.update({"img": td["train_img"], "txt": td["train_txt"]}, td["train_y"])
        pred = m.predict({"img": test_img, "txt": test_txt}).numpy()
        per_cls = f1_score(yt, pred, labels=list(range(C)), average=None, zero_division=0)
        f1s.append(float(np.mean(per_cls)))
        wf1s.append(float(np.min(per_cls)))
        baccs.append(float(balanced_accuracy_score(yt, pred)))
    return {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_sd": float(np.std(f1s, ddof=1)),
        "worst_class_f1_mean": float(np.mean(wf1s)),
        "balanced_acc_mean": float(np.mean(baccs)),
    }


def main():
    datasets = {
        "uk_crisismmd": load_uk_tasks(),
        "crisismmd_humanitarian": load_humanitarian_tasks(),
    }
    out = {}
    for name, (tasks, ti, tt, ty, C) in datasets.items():
        print(f"\n=== {name.upper()} ===")
        out[name] = {"P_ablation": {}, "ridge_ablation": {}, "beta_ablation": {}}
        for P in [256, 512, 1024, 2048]:
            r = eval_config(tasks, ti, tt, ty, C, P=P, ridge=0.1, beta=0.5)
            out[name]["P_ablation"][f"P_{P}"] = r
            print(f"  P={P:<4} | Macro-F1={r['macro_f1_mean']:.4f}±{r['macro_f1_sd']:.4f} | WorstF1={r['worst_class_f1_mean']:.4f}")
        for ridge in [0.001, 0.01, 0.1, 1.0, 10.0]:
            r = eval_config(tasks, ti, tt, ty, C, P=1024, ridge=ridge, beta=0.5)
            out[name]["ridge_ablation"][f"ridge_{ridge}"] = r
            print(f"  γ={ridge:<5} | Macro-F1={r['macro_f1_mean']:.4f}±{r['macro_f1_sd']:.4f} | WorstF1={r['worst_class_f1_mean']:.4f}")
        for beta in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = eval_config(tasks, ti, tt, ty, C, P=1024, ridge=0.1, beta=beta)
            out[name]["beta_ablation"][f"beta_{beta}"] = r
            print(f"  β={beta:<4} | Macro-F1={r['macro_f1_mean']:.4f}±{r['macro_f1_sd']:.4f} | WorstF1={r['worst_class_f1_mean']:.4f}")

    p = ROOT / "analysis_outputs" / "block_analytic_sensitivity_grid.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
