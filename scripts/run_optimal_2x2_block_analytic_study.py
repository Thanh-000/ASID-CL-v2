#!/usr/bin/env python3
"""
Deep Mathematical & Empirical Study of the Optimal 2x2 Block Analytic Classifier
for 2 Modalities (Image I + Text T) on UK-CrisisMMD and CrisisMMD v2.0 Humanitarian.

Investigates 4 mathematical optimizations over the standard 2x2 Block Analytic baseline:
  1. Asymmetric Modality-Specific Tikhonov Regularization (gamma_I != gamma_T):
     Penalizes the noisier modality (Text) differently from the cleaner modality (Image).
  2. Damped Cross-Modal Schur Coupling (rho in [0, 1] on off-diagonal block G_IT):
     G(rho) = [[G_II + Lambda_I, rho * G_IT], [rho * G_TI, G_TT + Lambda_T]].
     Tests rho=0 (Block-Diagonal decouple), rho=1 (Full Schur coupling), and optimal rho* in (0, 1).
  3. Block Schur Complement Inversion Equivalence:
     Verifies exact machine-precision equivalence (< 1e-14) between 2P x 2P direct solve
     and P x P Block Schur Complement inversion.
  4. Multi-Head Sub-Block + Joint Logit Fusion (W_both(rho) + alpha_I W_I + alpha_T W_T):
     Combines the marginal unimodal sub-block heads with the Schur-coupled joint head,
     hyperparameters selected strictly on the Validation split before evaluating on Locked Test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402
from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402

SEEDS = [2027, 2028, 2029, 2030, 2031]
P = 1024


def load_uk_full():
    protocol = load_phase3_protocol(ROOT)
    td_list = uk.build_task_data(protocol, "O1")
    tr_i = torch.cat([t["train_img"] for t in td_list]).double()
    tr_t = torch.cat([t["train_txt"] for t in td_list]).double()
    tr_y = torch.cat([t["train_y"] for t in td_list]).long()
    va_i = torch.cat([t["val_img"] for t in td_list]).double()
    va_t = torch.cat([t["val_txt"] for t in td_list]).double()
    va_y = torch.cat([t["val_y"] for t in td_list]).long()
    te_i = torch.cat([t["test_img"] for t in td_list]).double()
    te_t = torch.cat([t["test_txt"] for t in td_list]).double()
    te_y = torch.cat([t["test_y"] for t in td_list]).long()
    return (tr_i, tr_t, tr_y), (va_i, va_t, va_y), (te_i, te_t, te_y), 6


def load_humanitarian_full():
    cache = np.load(ROOT / "analysis_outputs" / "mm_hope_phase1_cache" / "clip_embeddings_f96e9e018d610664.npz")
    ds = load_crisismmd_v2_annotations(
        ROOT / "external_data" / "crisismmd_v2",
        task="humanitarian",
        agreed_labels=True,
        media_root=ROOT / "external_data" / "crisismmd_v2" / "media",
        require_images=False,
    )
    frame = ds.frame
    rid2i = {str(r): i for i, r in enumerate(cache["row_ids"])}
    keep = [str(r) in rid2i for r in frame["row_id"]]
    frame = frame[keep].reset_index(drop=True)
    idx = [rid2i[str(r)] for r in frame["row_id"]]
    img = torch.from_numpy(cache["image_embeddings"][idx]).double()
    txt = torch.from_numpy(cache["text_embeddings"][idx]).double()
    y = torch.from_numpy(frame["label_index"].to_numpy(dtype=np.int64).copy()).long()
    sp = frame["split"].to_numpy()
    return (
        (img[sp == "train"], txt[sp == "train"], y[sp == "train"]),
        (img[sp == "val"], txt[sp == "val"], y[sp == "val"]),
        (img[sp == "test"], txt[sp == "test"], y[sp == "test"]),
        len(ds.labels),
    )


class OptimalBlock2x2Analytic:
    """
    Generalized 2x2 Block Analytic Classifier with:
      - Per-modality feature normalization & ReLU random expansion (dim P per modality)
      - Class-balanced sufficient statistics (exponent beta)
      - Asymmetric modality ridge (gamma_I, gamma_T)
      - Damped cross-modal Schur coupling (rho in [0, 1])
      - Exact P x P Block Schur Complement solver
    """

    def __init__(self, C: int, seed: int, P: int = 1024, beta: float = 0.5):
        self.C = C
        self.P = P
        self.beta = beta
        g_i = torch.Generator().manual_seed(seed + 101)
        g_t = torch.Generator().manual_seed(seed + 202)
        self.W_i = torch.randn(512, P, generator=g_i, dtype=torch.float64)
        self.W_t = torch.randn(512, P, generator=g_t, dtype=torch.float64)

        self.cgrams = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
        self.csums = torch.zeros(2 * P, C, dtype=torch.float64)
        self.counts = torch.zeros(C, dtype=torch.long)

    def phi(self, img: torch.Tensor, txt: torch.Tensor):
        pi = F.relu(F.normalize(img, dim=-1) @ self.W_i)
        pt = F.relu(F.normalize(txt, dim=-1) @ self.W_t)
        return pi, pt

    def fit(self, img: torch.Tensor, txt: torch.Tensor, y: torch.Tensor):
        pi, pt = self.phi(img, txt)
        f = torch.cat([pi, pt], dim=1)
        for c in y.unique().tolist():
            m = y == c
            fc = f[m]
            self.cgrams[c] += fc.T @ fc
            self.csums[:, c] += fc.sum(0)
            self.counts[c] += int(m.sum())

        w = torch.zeros(self.C, dtype=torch.float64)
        obs = self.counts > 0
        rel = self.counts[obs].double() / self.counts[obs].double().mean()
        w[obs] = rel.pow(-self.beta)
        w[obs] /= w[obs].mean()

        G = torch.einsum("c,cij->ij", w, self.cgrams)
        S = self.csums * w.unsqueeze(0)
        self.G_II = G[:self.P, :self.P]
        self.G_TT = G[self.P:, self.P:]
        self.G_IT = G[:self.P, self.P:]
        self.S_I = S[:self.P]
        self.S_T = S[self.P:]
        self.tr_I = max(float(torch.trace(self.G_II)) / self.P, 1e-12)
        self.tr_T = max(float(torch.trace(self.G_TT)) / self.P, 1e-12)
        self.tr_joint = max(float(torch.trace(G)) / (2 * self.P), 1e-12)

    def solve_unimodal_heads(self, gamma_I: float, gamma_T: float):
        eye = torch.eye(self.P, dtype=torch.float64)
        A_I = self.G_II + (gamma_I * self.tr_I) * eye
        A_T = self.G_TT + (gamma_T * self.tr_T) * eye
        W_I = torch.linalg.solve(A_I, self.S_I)
        W_T = torch.linalg.solve(A_T, self.S_T)
        return W_I, W_T

    def solve_joint_direct(self, gamma_I: float, gamma_T: float, rho: float = 1.0, isotropic_scale: bool = False):
        eye = torch.eye(self.P, dtype=torch.float64)
        lam_I = (gamma_I * (self.tr_joint if isotropic_scale else self.tr_I)) * eye
        lam_T = (gamma_T * (self.tr_joint if isotropic_scale else self.tr_T)) * eye
        G_top = torch.cat([self.G_II + lam_I, rho * self.G_IT], dim=1)
        G_bot = torch.cat([rho * self.G_IT.T, self.G_TT + lam_T], dim=1)
        G_full = torch.cat([G_top, G_bot], dim=0)
        S_full = torch.cat([self.S_I, self.S_T], dim=0)
        return torch.linalg.solve(G_full, S_full)

    def solve_joint_schur(self, gamma_I: float, gamma_T: float, rho: float = 1.0):
        """
        Solves the 2P x 2P block system using only P x P inversions via the Block Schur Complement:
          A_T = G_TT + lam_T I
          K_{I|T} = rho * G_IT @ A_T^{-1}
          Schur_I = (G_II + lam_I I) - rho * K_{I|T} @ G_TI
          W_I_joint = Schur_I^{-1} (S_I - K_{I|T} S_T)
          W_T_joint = A_T^{-1} (S_T - rho * G_TI W_I_joint)
        """
        eye = torch.eye(self.P, dtype=torch.float64)
        A_I = self.G_II + (gamma_I * self.tr_I) * eye
        A_T = self.G_TT + (gamma_T * self.tr_T) * eye
        # Solve A_T^{-1} [G_TI, S_T] in one P x P Cholesky/LU call
        rhs_T = torch.cat([self.G_IT.T, self.S_T], dim=1)
        sol_T = torch.linalg.solve(A_T, rhs_T)
        A_T_inv_G_TI = sol_T[:, :self.P]
        W_T_marginal = sol_T[:, self.P:]

        Schur_I = A_I - (rho ** 2) * (self.G_IT @ A_T_inv_G_TI)
        rhs_I = self.S_I - rho * (self.G_IT @ W_T_marginal)
        W_I_joint = torch.linalg.solve(Schur_I, rhs_I)
        W_T_joint = W_T_marginal - rho * (A_T_inv_G_TI @ W_I_joint)
        return torch.cat([W_I_joint, W_T_joint], dim=0), W_T_marginal


def metrics(y_true: np.ndarray, logits: torch.Tensor, C: int):
    pred = logits.argmax(dim=-1).numpy()
    per_cls = f1_score(y_true, pred, labels=list(range(C)), average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_cls)),
        "worst_class_f1": float(np.min(per_cls)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "acc": float(accuracy_score(y_true, pred)),
    }


def run_dataset_study(name: str, tr, va, te, C: int):
    print(f"\n==================== {name.upper()} (train={len(tr[2])}, val={len(va[2])}, test={len(te[2])}) ====================")
    models = []
    schur_gaps = []
    for s in SEEDS:
        m = OptimalBlock2x2Analytic(C, s, P=P, beta=0.5)
        m.fit(*tr)
        models.append(m)
        w_dir = m.solve_joint_direct(0.1, 0.1, rho=0.7)
        w_sch, _ = m.solve_joint_schur(0.1, 0.1, rho=0.7)
        schur_gaps.append(float((w_dir - w_sch).abs().max()))

    max_schur_gap = max(schur_gaps)
    print(f"  [Theorem Check] Max difference Direct (2P x 2P) vs Block Schur (P x P): {max_schur_gap:.3e}")

    y_va = va[2].numpy()
    y_te = te[2].numpy()

    def eval_arm(fn):
        va_scores, te_scores = [], []
        for m in models:
            pi_va, pt_va = m.phi(va[0], va[1])
            pi_te, pt_te = m.phi(te[0], te[1])
            lg_va, lg_te = fn(m, (pi_va, pt_va), (pi_te, pt_te))
            va_scores.append(metrics(y_va, lg_va, C))
            te_scores.append(metrics(y_te, lg_te, C))
        return {
            split: {
                k: {
                    "mean": float(np.mean([r[k] for r in rows])),
                    "sd": float(np.std([r[k] for r in rows], ddof=1)),
                }
                for k in rows[0]
            }
            for split, rows in [("val", va_scores), ("test", te_scores)]
        }

    # Baseline 1: Standard Isotropic Joint Block Analytic (gamma_I = gamma_T = 0.1, rho = 1.0, isotropic trace)
    res_baseline = eval_arm(lambda m, f_va, f_te: (
        torch.cat(f_va, 1) @ m.solve_joint_direct(0.1, 0.1, rho=1.0, isotropic_scale=True),
        torch.cat(f_te, 1) @ m.solve_joint_direct(0.1, 0.1, rho=1.0, isotropic_scale=True),
    ))

    # Mechanism 1: Block-Diagonal Decoupled (rho = 0.0, equivalent to independent W_I + W_T)
    res_rho0 = eval_arm(lambda m, f_va, f_te: (
        torch.cat(f_va, 1) @ m.solve_joint_schur(0.1, 0.1, rho=0.0)[0],
        torch.cat(f_te, 1) @ m.solve_joint_schur(0.1, 0.1, rho=0.0)[0],
    ))

    # Mechanism 2: Asymmetric Modality Ridge (grid search (gamma_I, gamma_T) on VAL only)
    best_asym_cfg, best_asym_val = None, -1.0
    for g_i in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        for g_t in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
            cur = eval_arm(lambda m, f_va, f_te, gi=g_i, gt=g_t: (
                torch.cat(f_va, 1) @ m.solve_joint_schur(gi, gt, rho=1.0)[0],
                torch.cat(f_te, 1) @ m.solve_joint_schur(gi, gt, rho=1.0)[0],
            ))
            if cur["val"]["macro_f1"]["mean"] > best_asym_val:
                best_asym_val = cur["val"]["macro_f1"]["mean"]
                best_asym_cfg = (g_i, g_t)
                res_asym = cur

    # Mechanism 3: Damped Cross-Modal Schur Coupling (grid search rho in [0, 1] on VAL)
    gi_star, gt_star = best_asym_cfg
    best_rho_cfg, best_rho_val = None, -1.0
    rho_curve = {}
    for rho in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cur = eval_arm(lambda m, f_va, f_te, r=rho: (
            torch.cat(f_va, 1) @ m.solve_joint_schur(gi_star, gt_star, rho=r)[0],
            torch.cat(f_te, 1) @ m.solve_joint_schur(gi_star, gt_star, rho=r)[0],
        ))
        rho_curve[f"rho_{rho}"] = cur["test"]["macro_f1"]["mean"]
        if cur["val"]["macro_f1"]["mean"] > best_rho_val:
            best_rho_val = cur["val"]["macro_f1"]["mean"]
            best_rho_cfg = rho
            res_damped = cur

    # Mechanism 4: Optimal Multi-Head Schur + Marginal Sub-Block Logit Fusion
    # Combines W_joint(gi*, gt*, rho*) with marginal sub-block heads (alpha_I * z_I + alpha_T * z_T)
    best_ens_cfg, best_ens_val = None, -1.0
    for a_i in [0.0, 0.25, 0.5, 1.0]:
        for a_t in [0.0, 0.25, 0.5, 1.0]:
            def _fn(m, f_va, f_te, ai=a_i, at=a_t):
                W_j, _ = m.solve_joint_schur(gi_star, gt_star, rho=best_rho_cfg)
                W_i, W_t = m.solve_unimodal_heads(gi_star, gt_star)
                lg_va = (torch.cat(f_va, 1) @ W_j) + ai * (f_va[0] @ W_i) + at * (f_va[1] @ W_t)
                lg_te = (torch.cat(f_te, 1) @ W_j) + ai * (f_te[0] @ W_i) + at * (f_te[1] @ W_t)
                return lg_va, lg_te
            cur = eval_arm(_fn)
            if cur["val"]["macro_f1"]["mean"] > best_ens_val:
                best_ens_val = cur["val"]["macro_f1"]["mean"]
                best_ens_cfg = (a_i, a_t)
                res_optimal = cur

    arms = {
        "1_standard_block_analytic_gamma0.1_rho1.0": res_baseline,
        "2_block_diagonal_decoupled_rho0.0": res_rho0,
        f"3_asymmetric_ridge_gI{gi_star}_gT{gt_star}": res_asym,
        f"4_damped_schur_coupling_rho{best_rho_cfg}": res_damped,
        f"5_optimal_schur_subblock_fusion_aI{best_ens_cfg[0]}_aT{best_ens_cfg[1]}": res_optimal,
    }
    for k, v in arms.items():
        t = v["test"]
        print(
            f"  {k:<55} | Test Macro-F1={t['macro_f1']['mean']:.4f}±{t['macro_f1']['sd']:.4f} | "
            f"BalAcc={t['balanced_acc']['mean']:.4f} | WorstF1={t['worst_class_f1']['mean']:.4f} | Acc={t['acc']['mean']:.4f}"
        )
    print(f"  [Rho Curve on Test]: {rho_curve}")
    return {
        "max_schur_inversion_gap": max_schur_gap,
        "selected_hyperparams_on_val": {
            "gamma_I": gi_star,
            "gamma_T": gt_star,
            "rho": best_rho_cfg,
            "alpha_I": best_ens_cfg[0],
            "alpha_T": best_ens_cfg[1],
        },
        "rho_curve_test_macro_f1": rho_curve,
        "arms": arms,
    }


def main():
    uk_data = load_uk_full()
    hm_data = load_humanitarian_full()
    out = {
        "uk_crisismmd": run_dataset_study("uk_crisismmd", *uk_data),
        "crisismmd_humanitarian": run_dataset_study("crisismmd_humanitarian", *hm_data),
    }
    p = ROOT / "analysis_outputs" / "optimal_2x2_block_analytic_study.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {p}")


if __name__ == "__main__":
    main()
