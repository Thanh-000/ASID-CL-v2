#!/usr/bin/env python3
"""
Head-to-Head SOTA Continual Learning & Multimodal Merging Benchmark for ICCV:
Compares our unified framework (`BlockAnalyticCL` v2: Block-Schur + Centered Fisher-Spectral
Shrinkage + Pearson Conflict-Gated SID) against established baselines on identical splits/seeds:
  1. Raw-Ridge Concat: Ridge regression directly on L2-normalized CLIP features [x_I, x_T].
  2. Late-Fusion Analytic: Separate unimodal analytic classifiers combined via logit averaging (rho = 0).
  3. Task Arithmetic (Ilharco et al., ICLR 2023): Continual task-vector averaging across tasks.
  4. TIES-Merging (Yadav et al., NeurIPS 2023): Trim, Elect Sign, and Disjoint Merge across tasks.
  5. ACIL (Zhuang et al., NeurIPS 2022): Analytic Class-Incremental Learning (unweighted isotropic Gram).
  6. RanPAC (McDonnell et al., NeurIPS 2023): Random Projections + Cross-Validated scalar ridge.
  7. Ours (ASID-CL / BlockAnalyticCL v2): 2x2 Block Schur + Centered Fisher-Spectral Auto-Ridges + Pearson SID.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_unified_schur_innovation_study import load_humanitarian_events_and_full, load_uk_tasks_and_full
from run_upgraded_system_evaluation import extract_food101_frozen_features


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_cls)),
        "worst_class_f1": float(np.min(per_cls)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def ties_merge_weights(task_weights: list[torch.Tensor], density: float = 0.6) -> torch.Tensor:
    """TIES-Merging (Trim, Elect Sign, Disjoint Merge) across task weight matrices."""
    stack = torch.stack(task_weights, dim=0)  # (T, D, C)
    T, D, C = stack.shape
    flat = stack.view(T, -1)
    k = max(int(flat.shape[1] * density), 1)
    topk_vals = torch.topk(flat.abs(), k=k, dim=1).values[:, -1:]
    trimmed = torch.where(flat.abs() >= topk_vals, flat, torch.zeros_like(flat)).view(T, D, C)

    # Elect dominant sign weighted by magnitude sum
    elected_sign = torch.sign(trimmed.sum(dim=0))
    agree_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)
    denom = agree_mask.sum(dim=0).clamp(min=1).to(trimmed.dtype)
    merged = (trimmed * agree_mask.to(trimmed.dtype)).sum(dim=0) / denom
    return merged


def evaluate_dataset_sota(
    name: str,
    X_tr: dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    X_va: dict[str, torch.Tensor],
    y_va: torch.Tensor,
    X_te: dict[str, torch.Tensor],
    y_te: torch.Tensor,
    task_ids_tr: torch.Tensor,
    num_classes: int,
    seeds: tuple[int, ...],
) -> dict:
    y_va_np = y_va.numpy()
    y_te_np = y_te.numpy()
    d_img = X_tr["img"].shape[1]
    d_txt = X_tr["txt"].shape[1]
    unique_tasks = sorted(torch.unique(task_ids_tr).tolist())

    results = {
        m: {"clean": [], "missing_img_50": [], "corrupt_txt_50": []}
        for m in (
            "raw_ridge_concat",
            "task_arithmetic",
            "ties_merging",
            "late_fusion_analytic",
            "acil_neurips22",
            "ranpac_neurips23",
            "ours_asid_cl_v2",
        )
    }

    for s in seeds:
        # Prepare deterministic 50% missing-image and 50% corrupted-text test inputs
        g = torch.Generator().manual_seed(s + 99)
        miss_mask = torch.rand(len(y_te_np), generator=g) < 0.50
        X_te_zero = {"img": X_te["img"].clone(), "txt": X_te["txt"]}
        X_te_zero["img"][miss_mask] = 0.0

        X_te_corr = {"img": X_te["img"], "txt": X_te["txt"].clone()}
        noise = F.normalize(torch.randn(X_te_corr["txt"][miss_mask].shape, generator=g), dim=-1)
        X_te_corr["txt"][miss_mask] = F.normalize(0.15 * X_te_corr["txt"][miss_mask] + 0.85 * noise, dim=-1)

        # ------------------------------------------------------------------
        # 1. Raw-Ridge Concat (No random expansion, direct L2-normalized CLIP features)
        # ------------------------------------------------------------------
        phi_tr_raw = torch.cat([F.normalize(X_tr["img"].double(), dim=-1), F.normalize(X_tr["txt"].double(), dim=-1)], dim=1)
        phi_te_raw = torch.cat([F.normalize(X_te["img"].double(), dim=-1), F.normalize(X_te["txt"].double(), dim=-1)], dim=1)
        phi_te_zero_raw = torch.cat([
            torch.where(miss_mask.unsqueeze(1), torch.zeros_like(X_te["img"].double()), F.normalize(X_te["img"].double(), dim=-1)),
            F.normalize(X_te["txt"].double(), dim=-1),
        ], dim=1)
        phi_te_corr_raw = torch.cat([F.normalize(X_te_corr["img"].double(), dim=-1), F.normalize(X_te_corr["txt"].double(), dim=-1)], dim=1)

        G_raw = phi_tr_raw.T @ phi_tr_raw
        S_raw = phi_tr_raw.T @ F.one_hot(y_tr, num_classes).double()
        lam_raw = 0.1 * (float(torch.trace(G_raw)) / G_raw.shape[0])
        W_raw = torch.linalg.solve(G_raw + lam_raw * torch.eye(G_raw.shape[0], dtype=torch.float64), S_raw)
        results["raw_ridge_concat"]["clean"].append(compute_metrics(y_te_np, (phi_te_raw @ W_raw).argmax(1).numpy()))
        results["raw_ridge_concat"]["missing_img_50"].append(compute_metrics(y_te_np, (phi_te_zero_raw @ W_raw).argmax(1).numpy()))
        results["raw_ridge_concat"]["corrupt_txt_50"].append(compute_metrics(y_te_np, (phi_te_corr_raw @ W_raw).argmax(1).numpy()))

        # ------------------------------------------------------------------
        # 2. Task Arithmetic (ICLR 2023) & 3. TIES-Merging (NeurIPS 2023) across Continual Tasks
        # ------------------------------------------------------------------
        task_Ws = []
        for t_id in unique_tasks:
            m_t = task_ids_tr == t_id
            phi_t = phi_tr_raw[m_t]
            y_t = y_tr[m_t]
            G_t = phi_t.T @ phi_t
            S_t = phi_t.T @ F.one_hot(y_t, num_classes).double()
            lam_t = 0.2 * (float(torch.trace(G_t)) / G_t.shape[0])
            W_t = torch.linalg.solve(G_t + lam_t * torch.eye(G_t.shape[0], dtype=torch.float64), S_t)
            task_Ws.append(W_t)

        W_ta = torch.stack(task_Ws, dim=0).mean(dim=0)
        W_ties = ties_merge_weights(task_Ws, density=0.6)

        results["task_arithmetic"]["clean"].append(compute_metrics(y_te_np, (phi_te_raw @ W_ta).argmax(1).numpy()))
        results["task_arithmetic"]["missing_img_50"].append(compute_metrics(y_te_np, (phi_te_zero_raw @ W_ta).argmax(1).numpy()))
        results["task_arithmetic"]["corrupt_txt_50"].append(compute_metrics(y_te_np, (phi_te_corr_raw @ W_ta).argmax(1).numpy()))

        results["ties_merging"]["clean"].append(compute_metrics(y_te_np, (phi_te_raw @ W_ties).argmax(1).numpy()))
        results["ties_merging"]["missing_img_50"].append(compute_metrics(y_te_np, (phi_te_zero_raw @ W_ties).argmax(1).numpy()))
        results["ties_merging"]["corrupt_txt_50"].append(compute_metrics(y_te_np, (phi_te_corr_raw @ W_ties).argmax(1).numpy()))

        # ------------------------------------------------------------------
        # 4. ACIL (NeurIPS 2022): Unweighted (beta=0.0), Isotropic scalar ridge=0.1, Zero-fill
        # ------------------------------------------------------------------
        m_acil = BlockAnalyticCL({"img": d_img, "txt": d_txt}, num_classes, projection_dim=1024, ridge=0.1, class_balance_beta=0.0, seed=s)
        m_acil.update(X_tr, y_tr)
        results["acil_neurips22"]["clean"].append(compute_metrics(y_te_np, m_acil.predict(X_te).numpy()))
        results["acil_neurips22"]["missing_img_50"].append(compute_metrics(y_te_np, m_acil.predict(X_te_zero).numpy()))
        results["acil_neurips22"]["corrupt_txt_50"].append(compute_metrics(y_te_np, m_acil.predict(X_te_corr).numpy()))

        # ------------------------------------------------------------------
        # 5. Late-Fusion Analytic (rho = 0.0: separate unimodal heads without Schur coupling)
        # ------------------------------------------------------------------
        m_lf = BlockAnalyticCL({"img": d_img, "txt": d_txt}, num_classes, projection_dim=1024, ridge=0.1, class_balance_beta=0.5, schur_coupling_rho=0.0, seed=s)
        m_lf.update(X_tr, y_tr)
        results["late_fusion_analytic"]["clean"].append(compute_metrics(y_te_np, m_lf.predict(X_te).numpy()))
        results["late_fusion_analytic"]["missing_img_50"].append(compute_metrics(y_te_np, m_lf.predict(X_te_zero).numpy()))
        results["late_fusion_analytic"]["corrupt_txt_50"].append(compute_metrics(y_te_np, m_lf.predict(X_te_corr).numpy()))

        # ------------------------------------------------------------------
        # 6. RanPAC (NeurIPS 2023): Random Projection + Val-selected scalar ridge in {0.01, 0.1, 0.5, 1.0, 2.0}
        # ------------------------------------------------------------------
        best_r_val = -1.0
        best_r_model = None
        for r_cand in (0.01, 0.1, 0.25, 0.5, 1.0):
            m_rp = BlockAnalyticCL({"img": d_img, "txt": d_txt}, num_classes, projection_dim=1024, ridge=r_cand, class_balance_beta=0.0, seed=s)
            m_rp.gram = m_acil.gram.clone()
            m_rp.class_sums = m_acil.class_sums.clone()
            m_rp.class_counts = m_acil.class_counts.clone()
            m_rp.modality_class_counts = m_acil.modality_class_counts.clone()
            sc_va = compute_metrics(y_va_np, m_rp.predict(X_va).numpy())["macro_f1"]
            if sc_va > best_r_val:
                best_r_val = sc_va
                best_r_model = m_rp
        results["ranpac_neurips23"]["clean"].append(compute_metrics(y_te_np, best_r_model.predict(X_te).numpy()))
        results["ranpac_neurips23"]["missing_img_50"].append(compute_metrics(y_te_np, best_r_model.predict(X_te_zero).numpy()))
        results["ranpac_neurips23"]["corrupt_txt_50"].append(compute_metrics(y_te_np, best_r_model.predict(X_te_corr).numpy()))

        # ------------------------------------------------------------------
        # 7. Ours (ASID-CL / BlockAnalyticCL v2): Block-Schur + Centered Fisher-Spectral + Pearson SID
        # ------------------------------------------------------------------
        m_ours = BlockAnalyticCL({"img": d_img, "txt": d_txt}, num_classes, projection_dim=1024, class_balance_beta=0.5, seed=s)
        m_ours.update(X_tr, y_tr)
        m_ours.auto_spectral_ridges()
        pred_clean = m_ours.predict(X_te, adaptive_sid=True).numpy()
        pred_miss = pred_clean.copy()
        pred_miss[miss_mask.numpy()] = m_ours.predict({"txt": X_te["txt"][miss_mask]}, modalities=("txt",)).numpy()
        pred_corr = m_ours.predict(X_te_corr, adaptive_sid=True).numpy()

        results["ours_asid_cl_v2"]["clean"].append(compute_metrics(y_te_np, pred_clean))
        results["ours_asid_cl_v2"]["missing_img_50"].append(compute_metrics(y_te_np, pred_miss))
        results["ours_asid_cl_v2"]["corrupt_txt_50"].append(compute_metrics(y_te_np, pred_corr))

    summary = {}
    for m_name, regimes in results.items():
        summary[m_name] = {}
        for reg_name, lst in regimes.items():
            summary[m_name][reg_name] = {
                k: {
                    "mean": float(np.mean([x[k] for x in lst])),
                    "std": float(np.std([x[k] for x in lst], ddof=1)),
                }
                for k in ("macro_f1", "worst_class_f1", "balanced_acc", "acc")
            }
    return summary


def main():
    t0 = time.time()
    out = {}

    # 1. UK-CrisisMMD (6 classes, 3 CIL tasks, 5 seeds)
    uk_tasks, (uk_tri, uk_trt, uk_try), (uk_vai, uk_vat, uk_vay), (uk_tei, uk_tet, uk_tey), C_uk = load_uk_tasks_and_full()
    uk_tids = torch.cat([torch.full((len(t["train_y"]),), i, dtype=torch.long) for i, t in enumerate(uk_tasks)])
    print("Running SOTA comparison on UK-CrisisMMD (5 seeds)...")
    out["uk_crisismmd"] = evaluate_dataset_sota(
        "uk_crisismmd",
        {"img": uk_tri.float(), "txt": uk_trt.float()},
        uk_try.long(),
        {"img": uk_vai.float(), "txt": uk_vat.float()},
        uk_vay.long(),
        {"img": uk_tei.float(), "txt": uk_tet.float()},
        uk_tey.long(),
        uk_tids,
        C_uk,
        (2027, 2028, 2029, 2030, 2031),
    )

    # 2. CrisisMMD v2.0 Humanitarian (7 classes, 7 disaster events, 5 seeds)
    hu_tasks, (hu_tri, hu_trt, hu_try), (hu_vai, hu_vat, hu_vay), (hu_tei, hu_tet, hu_tey), C_hu = load_humanitarian_events_and_full()
    hu_tids = torch.cat([torch.full((len(t["train_y"]),), i, dtype=torch.long) for i, t in enumerate(hu_tasks)])
    print("Running SOTA comparison on CrisisMMD v2.0 Humanitarian (5 seeds)...")
    out["crisismmd_humanitarian"] = evaluate_dataset_sota(
        "crisismmd_humanitarian",
        {"img": hu_tri.float(), "txt": hu_trt.float()},
        hu_try.long(),
        {"img": hu_vai.float(), "txt": hu_vat.float()},
        hu_vay.long(),
        {"img": hu_tei.float(), "txt": hu_tet.float()},
        hu_tey.long(),
        hu_tids,
        C_hu,
        (2027, 2028, 2029, 2030, 2031),
    )

    # 3. UPMC Food-101 (100 classes, 10 CIL tasks, 3 seeds)
    F_fd = extract_food101_frozen_features()
    tr, va, te = F_fd["train"], F_fd["val"], F_fd["test"]
    print("Running SOTA comparison on UPMC Food-101 (100 classes, 10 CIL tasks, 3 seeds)...")
    out["upmc_food101"] = evaluate_dataset_sota(
        "upmc_food101",
        {"img": F_fd["img"][tr], "txt": F_fd["txt"][tr]},
        F_fd["y"][tr].long(),
        {"img": F_fd["img"][va], "txt": F_fd["txt"][va]},
        F_fd["y"][va].long(),
        {"img": F_fd["img"][te], "txt": F_fd["txt"][te]},
        F_fd["y"][te].long(),
        torch.tensor(F_fd["task"][tr], dtype=torch.long),
        F_fd["C"],
        (2026, 2027, 2028),
    )

    out["elapsed_seconds"] = round(time.time() - t0, 2)
    out_path = ROOT / "analysis_outputs" / "iccv_sota_baselines_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved SOTA comparison to {out_path} in {out['elapsed_seconds']}s")
    for ds in ("uk_crisismmd", "crisismmd_humanitarian", "upmc_food101"):
        print(f"\n=== {ds.upper()} ===")
        for m, r in out[ds].items():
            c_f1 = r["clean"]["macro_f1"]["mean"] * 100
            c_wf1 = r["clean"]["worst_class_f1"]["mean"] * 100
            m_f1 = r["missing_img_50"]["macro_f1"]["mean"] * 100
            n_f1 = r["corrupt_txt_50"]["macro_f1"]["mean"] * 100
            print(f"{m:22s} | Clean F1: {c_f1:5.2f}% (Worst: {c_wf1:5.2f}%) | MissImg50: {m_f1:5.2f}% | CorrTxt50: {n_f1:5.2f}%")


if __name__ == "__main__":
    main()
