#!/usr/bin/env python3
"""
Feasibility probe D3 (exploratory, VAL ONLY on UK-CrisisMMD).

Modality-orthogonal decomposition (Frisch-Waugh-Lovell) under monotone missingness:
every training sample has the image, only a fraction q also has text
(also the modality-incremental case: text "arrives later" for some samples).

  img model   : ridge on phi_img using ALL samples
  residual    : txt_perp = phi_txt - phi_img @ B  (B fitted on paired samples)
                W_r = ridge(txt_perp -> Y - phi_img W_img) on paired samples
  full logits : phi_img W_img + txt_perp W_r    (orthogonal increment, img model untouched)

Baselines: img-only, complete-case joint, zero-fill joint (dummy input, PAL-style),
late fusion (avg logits of img model on all + txt model on paired).
Test: both modalities present (val).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "src")]

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402
from mm_flex_merge.models import BlockAnalyticCL  # noqa: E402

C, RIDGE = 6, 0.1
SEEDS = [2026, 2027, 2028, 2029, 2030]
QS = [0.1, 0.3, 0.5, 1.0]


def ridge(X, Y):
    G = X.T @ X
    lam = RIDGE * max(float(torch.trace(G)) / G.shape[0], 1e-12)
    return torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), X.T @ Y)


def mf1(y, logits):
    return float(f1_score(y, logits.argmax(-1).numpy(), labels=list(range(C)), average="macro", zero_division=0))


def main():
    protocol = load_phase3_protocol(ROOT)
    tds = uk.build_task_data(protocol, "O1")
    xi = torch.cat([t["train_img"] for t in tds]); xt = torch.cat([t["train_txt"] for t in tds])
    y = torch.cat([t["train_y"] for t in tds])
    vi = torch.cat([t["val_img"] for t in tds]); vt = torch.cat([t["val_txt"] for t in tds])
    vy = torch.cat([t["val_y"] for t in tds]).numpy()
    Y = F.one_hot(y, C).double()
    rows = []
    for seed in SEEDS:
        m = BlockAnalyticCL({"img": xi.shape[1], "txt": xt.shape[1]}, C, seed=seed)
        Pi, Pt = m.features({"img": xi}, ["img"]), m.features({"txt": xt}, ["txt"])
        Qi, Qt = m.features({"img": vi}, ["img"]), m.features({"txt": vt}, ["txt"])
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(y), generator=g)
        for q in QS:
            paired = perm[: max(C, int(q * len(y)))]
            W_img = ridge(Pi, Y)
            img_only = Qi @ W_img
            W_cc = ridge(torch.cat([Pi[paired], Pt[paired]], 1), Y[paired])
            cc = torch.cat([Qi, Qt], 1) @ W_cc
            Pt_zf = torch.zeros_like(Pt); Pt_zf[paired] = Pt[paired]
            W_zf = ridge(torch.cat([Pi, Pt_zf], 1), Y)
            zf = torch.cat([Qi, Qt], 1) @ W_zf
            W_txt = ridge(Pt[paired], Y[paired])
            late = 0.5 * (img_only + Qt @ W_txt)
            B = ridge(Pi[paired], Pt[paired])
            R = Y[paired] - Pi[paired] @ W_img
            W_r = ridge(Pt[paired] - Pi[paired] @ B, R)
            orth = img_only + (Qt - Qi @ B) @ W_r
            r = {"seed": seed, "q": q,
                 "img_only_all": mf1(vy, img_only), "complete_case_joint": mf1(vy, cc),
                 "zero_fill_joint": mf1(vy, zf), "late_fusion": mf1(vy, late),
                 "orthogonal_increment": mf1(vy, orth)}
            rows.append(r)
    summary = {}
    for q in QS:
        sub = [r for r in rows if r["q"] == q]
        summary[str(q)] = {k: float(np.mean([r[k] for r in sub])) for k in sub[0] if k not in ("seed", "q")}
        print(f"q={q}: " + " | ".join(f"{k} {v:.4f}" for k, v in summary[str(q)].items()))
    out = ROOT / "analysis_outputs" / "feasibility_probe_orthogonal_modality.json"
    out.write_text(json.dumps({"status": "exploratory, val only", "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"📁 {out}")


if __name__ == "__main__":
    main()
