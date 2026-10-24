#!/usr/bin/env python3
"""
Feasibility probe D2 (exploratory, VAL ONLY on UK-CrisisMMD).

Block-structured RanPAC: separate random projections per modality,
    phi(x) = [relu(W_img x_img), relu(W_txt x_txt)]
with sufficient statistics G = sum phi phi^T, S = sum phi y^T.

Questions:
  Q1  Missing modality: does solving ridge on the available-modality sub-block of G
      (no retraining) beat zero-filling the missing modality in the full model?
  Q2  Is the sub-block solution identical to a dedicated single-modality RanPAC?
  Q3  Exact deletion: does subtracting a task's statistics equal retraining without it?
  Q4  Order invariance: are final predictions identical across task orders?
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402

C, P, RIDGE_REL = 6, 1024, 0.1
SEEDS = [2026, 2027, 2028, 2029, 2030]


class BlockRanPAC:
    def __init__(self, seed):
        g = torch.Generator().manual_seed(seed)
        self.W = {"img": torch.randn(512, P, generator=g, dtype=torch.float64),
                  "txt": torch.randn(512, P, generator=g, dtype=torch.float64)}
        self.G = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
        self.S = torch.zeros(2 * P, C, dtype=torch.float64)

    def phi(self, img, txt):
        a = F.relu(F.normalize(img.double(), dim=-1) @ self.W["img"])
        b = F.relu(F.normalize(txt.double(), dim=-1) @ self.W["txt"])
        return torch.cat([a, b], dim=1)

    def stats(self, img, txt, y):
        f = self.phi(img, txt)
        return f.T @ f, f.T @ F.one_hot(y, C).double()

    def add(self, img, txt, y, sign=1.0):
        g, s = self.stats(img, txt, y)
        self.G += sign * g
        self.S += sign * s

    def solve(self, block):
        sl = {"both": slice(0, 2 * P), "img": slice(0, P), "txt": slice(P, 2 * P)}[block]
        G, S = self.G[sl, sl], self.S[sl]
        lam = RIDGE_REL * G.diagonal().mean()
        return sl, torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), S)

    def logits(self, img, txt, block="both"):
        sl, W = self.solve(block)
        return self.phi(img, txt)[:, sl] @ W


def mf1(y, logits):
    return float(f1_score(y, logits.argmax(-1).numpy(), labels=list(range(C)), average="macro", zero_division=0))


def main():
    protocol = load_phase3_protocol(ROOT)
    rows = []
    for seed in SEEDS:
        td_o1 = uk.build_task_data(protocol, "O1")
        m = BlockRanPAC(seed)
        for td in td_o1:
            m.add(td["train_img"], td["train_txt"], td["train_y"])
        vi = torch.cat([t["val_img"] for t in td_o1])
        vt = torch.cat([t["val_txt"] for t in td_o1])
        vy = torch.cat([t["val_y"] for t in td_o1]).numpy()
        zi, zt = torch.zeros_like(vi), torch.zeros_like(vt)
        r = {
            "full": mf1(vy, m.logits(vi, vt)),
            "img_missing_zero_fill": mf1(vy, m.logits(zi, vt)),
            "img_missing_block_solve": mf1(vy, m.logits(zi, vt, "txt")),
            "txt_missing_zero_fill": mf1(vy, m.logits(vi, zt)),
            "txt_missing_block_solve": mf1(vy, m.logits(vi, zt, "img")),
        }
        # Q2: dedicated text-only model (same projection) vs sub-block
        ded = BlockRanPAC(seed)
        for td in td_o1:
            f = ded.phi(td["train_img"], td["train_txt"])[:, P:]
            ded.G[P:, P:] += f.T @ f
            ded.S[P:] += f.T @ F.one_hot(td["train_y"], C).double()
        r["q2_max_logit_gap_block_vs_dedicated"] = float((m.logits(zi, vt, "txt") - ded.logits(zi, vt, "txt")).abs().max())
        # Q3: subtract task 0 vs retrain without task 0
        retrain = BlockRanPAC(seed)
        for td in td_o1[1:]:
            retrain.add(td["train_img"], td["train_txt"], td["train_y"])
        m.add(td_o1[0]["train_img"], td_o1[0]["train_txt"], td_o1[0]["train_y"], sign=-1.0)
        r["q3_max_logit_gap_delete_vs_retrain"] = float((m.logits(vi, vt) - retrain.logits(vi, vt)).abs().max())
        # Q4: order invariance (O1 vs O4)
        a, b = BlockRanPAC(seed), BlockRanPAC(seed)
        for td in td_o1:
            a.add(td["train_img"], td["train_txt"], td["train_y"])
        for td in uk.build_task_data(protocol, "O4"):
            b.add(td["train_img"], td["train_txt"], td["train_y"])
        r["q4_max_logit_gap_O1_vs_O4"] = float((a.logits(vi, vt) - b.logits(vi, vt)).abs().max())
        rows.append(r)
        print(f"seed {seed}: " + " | ".join(f"{k} {v:.4g}" for k, v in r.items()))

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("\nTrung bình (VAL):")
    for k, v in summary.items():
        print(f"   {k:<40}: {v:.4g}")
    out = ROOT / "analysis_outputs" / "feasibility_probe_block_ranpac.json"
    out.write_text(json.dumps({"status": "exploratory, val only", "rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"📁 {out}")


if __name__ == "__main__":
    main()
