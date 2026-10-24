#!/usr/bin/env python3
"""
Confirmatory 2-Modality (Image + Text) Continual Missing-Modality Learning (CMML)
and Modality-Incremental Learning (MIL) Audit on Locked Test Splits.

Evaluates on:
  1. UK-CrisisMMD (n=2,385, 6 classes, 3 tasks, duplicate-safe split)
  2. CrisisMMD v2.0 Humanitarian (n=8,079, 5 classes, 7 disaster events, agreed-label split)

Compares 4 Continual Analytic Strategies across 5 seeds (2027..2031):
  - Zero-Fill Joint Analytic (β=0.5): zero-fills missing modality in 2P joint vector, uses 1 joint head.
  - Mean-Imputed Analytic RLS (β=0.5): imputes missing modality using running training mean.
  - PAL-Style Cross-Modal Imputed RLS (β=0.5): learns cross-modal ridge reconstructor (I->T, T->I)
    from paired task samples to impute missing modality before RLS Gram update (tests Σ εε^T bias).
  - 2-Modality Sub-Block Analytic Solve (β=0.0 and β=0.5): zero-imputation block accumulation
    + exact closed-form sub-block inference heads (W_both, W_I-only, W_T-only).
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
sys.path.insert(0, str(ROOT / "scripts"))

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402
from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402

SEEDS = [2027, 2028, 2029, 2030, 2031]
P = 1024
RIDGE_REL = 0.1


def load_uk_tasks():
    protocol = load_phase3_protocol(ROOT)
    td_list = uk.build_task_data(protocol, "O1")
    tasks = []
    for td in td_list:
        tasks.append({
            "train_img": td["train_img"].double(),
            "train_txt": td["train_txt"].double(),
            "train_y": td["train_y"].long(),
        })
    test_img = torch.cat([td["test_img"] for td in td_list]).double()
    test_txt = torch.cat([td["test_txt"] for td in td_list]).double()
    test_y = torch.cat([td["test_y"] for td in td_list]).long()
    return tasks, test_img, test_txt, test_y, 6


def load_humanitarian_tasks():
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
    y = torch.from_numpy(frame["label_index"].to_numpy(dtype=np.int64)).long()
    splits = frame["split"].to_numpy()
    events = frame["event_name"].to_numpy()

    event_order = sorted(set(events))
    tasks = []
    for ev in event_order:
        m = (splits == "train") & (events == ev)
        tasks.append({
            "train_img": img[m],
            "train_txt": img[m] * 0 + txt[m],
            "train_y": y[m],
        })
    test_m = splits == "test"
    return tasks, img[test_m], txt[test_m], y[test_m], len(ds.labels)


def project_pair(img: torch.Tensor, txt: torch.Tensor, W_img: torch.Tensor, W_txt: torch.Tensor):
    phi_i = F.relu(F.normalize(img, dim=-1) @ W_img)
    phi_t = F.relu(F.normalize(txt, dim=-1) @ W_txt)
    return phi_i, phi_t


def sample_availability_masks(n: int, eta: float, mode: str, rng: np.random.Generator):
    """Returns boolean masks (has_img, has_txt) of length n, guaranteeing at least 1 modality per sample."""
    has_img = np.ones(n, dtype=bool)
    has_txt = np.ones(n, dtype=bool)
    if eta <= 0.0:
        return has_img, has_txt
    drop = rng.random(n) < eta
    if mode == "missing_img":
        has_img[drop] = False
    elif mode == "missing_txt":
        has_txt[drop] = False
    elif mode == "mixed":
        which = rng.integers(0, 2, size=n)
        has_img[drop & (which == 0)] = False
        has_txt[drop & (which == 1)] = False
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return has_img, has_txt


def solve_ridge(G: torch.Tensor, S: torch.Tensor, rel_ridge: float = RIDGE_REL) -> torch.Tensor:
    d = G.shape[0]
    lam = rel_ridge * max(float(torch.trace(G)) / d, 1e-12)
    return torch.linalg.solve(G + lam * torch.eye(d, dtype=G.dtype), S)


def compute_class_weights(counts: torch.Tensor, beta: float) -> torch.Tensor:
    w = torch.zeros_like(counts, dtype=torch.float64)
    obs = counts > 0
    if obs.any():
        rel = counts[obs].double() / counts[obs].double().mean()
        w[obs] = rel.pow(-beta)
        w[obs] /= w[obs].mean()
    return w


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, float]:
    per_class = f1_score(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_class)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "worst_class_f1": float(np.min(per_class)),
    }


def run_cmml_experiment(tasks, test_img, test_txt, test_y, C: int, eta: float, mode: str, seed: int):
    g = torch.Generator().manual_seed(seed)
    W_img = torch.randn(512, P, generator=g, dtype=torch.float64)
    W_txt = torch.randn(512, P, generator=g, dtype=torch.float64)
    rng = np.random.default_rng(seed + int(eta * 1000) + sum( map(ord, mode) ))

    # Accumulators for the 4 methods
    # 1. Zero-Fill Joint (beta=0.5)
    zf_cgrams = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
    zf_csums = torch.zeros(2 * P, C, dtype=torch.float64)
    zf_counts = torch.zeros(C, dtype=torch.long)

    # 2. Mean-Imputed RLS (beta=0.5)
    mi_cgrams = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
    mi_csums = torch.zeros(2 * P, C, dtype=torch.float64)
    mi_counts = torch.zeros(C, dtype=torch.long)
    sum_phi_i = torch.zeros(P, dtype=torch.float64)
    sum_phi_t = torch.zeros(P, dtype=torch.float64)
    cnt_i, cnt_t = 0, 0

    # 3. PAL-Style Cross-Modal Imputed RLS (beta=0.5)
    pal_cgrams = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
    pal_csums = torch.zeros(2 * P, C, dtype=torch.float64)
    pal_counts = torch.zeros(C, dtype=torch.long)
    G_ii_cross = torch.zeros(P, P, dtype=torch.float64)
    G_tt_cross = torch.zeros(P, P, dtype=torch.float64)
    G_it_cross = torch.zeros(P, P, dtype=torch.float64)
    imputation_error_trace = 0.0
    imputed_count = 0

    # 4. 2-Modality Sub-Block Analytic Solve (beta=0.0 and beta=0.5)
    # We keep separate stats for (both, img-only, txt-only) from all valid observations of those modalities!
    sb_cgrams_both = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
    sb_csums_both = torch.zeros(2 * P, C, dtype=torch.float64)
    sb_counts_both = torch.zeros(C, dtype=torch.long)

    sb_cgrams_img = torch.zeros(C, P, P, dtype=torch.float64)
    sb_csums_img = torch.zeros(P, C, dtype=torch.float64)
    sb_counts_img = torch.zeros(C, dtype=torch.long)

    sb_cgrams_txt = torch.zeros(C, P, P, dtype=torch.float64)
    sb_csums_txt = torch.zeros(P, C, dtype=torch.float64)
    sb_counts_txt = torch.zeros(C, dtype=torch.long)

    for td in tasks:
        phi_i_true, phi_t_true = project_pair(td["train_img"], td["train_txt"], W_img, W_txt)
        y = td["train_y"]
        n = len(y)
        has_i, has_t = sample_availability_masks(n, eta, mode, rng)
        has_i_t = torch.from_numpy(has_i)
        has_t_t = torch.from_numpy(has_t)
        both_m = has_i_t & has_t_t

        # Update running means & cross-modal reconstructor from observed features
        if has_i_t.any():
            sum_phi_i += phi_i_true[has_i_t].sum(0)
            cnt_i += int(has_i_t.sum())
        if has_t_t.any():
            sum_phi_t += phi_t_true[has_t_t].sum(0)
            cnt_t += int(has_t_t.sum())
        if both_m.any():
            G_ii_cross += phi_i_true[both_m].T @ phi_i_true[both_m]
            G_tt_cross += phi_t_true[both_m].T @ phi_t_true[both_m]
            G_it_cross += phi_i_true[both_m].T @ phi_t_true[both_m]

        mean_i = sum_phi_i / max(cnt_i, 1)
        mean_t = sum_phi_t / max(cnt_t, 1)

        if int(both_m.sum()) > 0 or G_ii_cross.diagonal().sum() > 0:
            B_i2t = solve_ridge(G_ii_cross, G_it_cross, rel_ridge=0.5)
            B_t2i = solve_ridge(G_tt_cross, G_it_cross.T, rel_ridge=0.5)
        else:
            B_i2t = torch.zeros(P, P, dtype=torch.float64)
            B_t2i = torch.zeros(P, P, dtype=torch.float64)

        # 1. Zero-fill features
        phi_i_zf = torch.where(has_i_t.unsqueeze(1), phi_i_true, torch.zeros_like(phi_i_true))
        phi_t_zf = torch.where(has_t_t.unsqueeze(1), phi_t_true, torch.zeros_like(phi_t_true))
        phi_zf = torch.cat([phi_i_zf, phi_t_zf], dim=1)

        # 2. Mean-imputed features
        phi_i_mi = torch.where(has_i_t.unsqueeze(1), phi_i_true, mean_i.unsqueeze(0).expand_as(phi_i_true))
        phi_t_mi = torch.where(has_t_t.unsqueeze(1), phi_t_true, mean_t.unsqueeze(0).expand_as(phi_t_true))
        phi_mi = torch.cat([phi_i_mi, phi_t_mi], dim=1)

        # 3. PAL-style cross-modal imputed features
        phi_i_pal = torch.where(has_i_t.unsqueeze(1), phi_i_true, F.relu(phi_t_true @ B_t2i))
        phi_t_pal = torch.where(has_t_t.unsqueeze(1), phi_t_true, F.relu(phi_i_true @ B_i2t))
        phi_pal = torch.cat([phi_i_pal, phi_t_pal], dim=1)

        # Measure positive-definite imputation error trace Tr(Σ εε^T)
        if (~both_m).any():
            phi_true_full = torch.cat([phi_i_true, phi_t_true], dim=1)
            eps = phi_pal[~both_m] - phi_true_full[~both_m]
            imputation_error_trace += float((eps ** 2).sum())
            imputed_count += int((~both_m).sum())

        for c in y.unique().tolist():
            cm = y == c
            # Zero-fill
            f_zf = phi_zf[cm]
            zf_cgrams[c] += f_zf.T @ f_zf
            zf_csums[:, c] += f_zf.sum(0)
            zf_counts[c] += int(cm.sum())
            # Mean-imputed
            f_mi = phi_mi[cm]
            mi_cgrams[c] += f_mi.T @ f_mi
            mi_csums[:, c] += f_mi.sum(0)
            mi_counts[c] += int(cm.sum())
            # PAL-imputed
            f_pal = phi_pal[cm]
            pal_cgrams[c] += f_pal.T @ f_pal
            pal_csums[:, c] += f_pal.sum(0)
            pal_counts[c] += int(cm.sum())

            # Sub-block exact accumulation (no imputation!)
            cm_both = cm & both_m
            if cm_both.any():
                fb = phi_zf[cm_both]
                sb_cgrams_both[c] += fb.T @ fb
                sb_csums_both[:, c] += fb.sum(0)
                sb_counts_both[c] += int(cm_both.sum())
            else:
                # Even if no paired sample for class c in this batch, zero-filled outer product is PSD
                sb_cgrams_both[c] += f_zf.T @ f_zf
                sb_csums_both[:, c] += f_zf.sum(0)
                sb_counts_both[c] += int(cm.sum())

            cm_i = cm & has_i_t
            if cm_i.any():
                fi = phi_i_true[cm_i]
                sb_cgrams_img[c] += fi.T @ fi
                sb_csums_img[:, c] += fi.sum(0)
                sb_counts_img[c] += int(cm_i.sum())

            cm_t = cm & has_t_t
            if cm_t.any():
                ft = phi_t_true[cm_t]
                sb_cgrams_txt[c] += ft.T @ ft
                sb_csums_txt[:, c] += ft.sum(0)
                sb_counts_txt[c] += int(cm_t.sum())

    # Solve heads
    def solve_weighted(cgrams, csums, counts, beta):
        w = compute_class_weights(counts, beta)
        G = torch.einsum("c,cij->ij", w, cgrams)
        S = csums * w.unsqueeze(0)
        return solve_ridge(G, S)

    def solve_availability_normalized_joint(cgrams_both, cgrams_i, cgrams_t, csums_i, csums_t, counts_all, counts_i, counts_t, beta):
        """
        Availability-Normalized 2x2 Block Gram:
        Scales each class-modality block by (counts_all[c] / counts_m[c]) before applying class-balance weight w_c,
        so asymmetric missingness (e.g. 70% missing image vs 0% missing text) never biases the relative energy
        between G_II and G_TT in W_both!
        """
        w = compute_class_weights(counts_all, beta)
        scale_i = torch.where(counts_i > 0, counts_all.double() / counts_i.double().clamp(min=1), torch.ones_like(w))
        scale_t = torch.where(counts_t > 0, counts_all.double() / counts_t.double().clamp(min=1), torch.ones_like(w))
        scale_it = torch.sqrt(scale_i * scale_t)

        G = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
        S = torch.zeros(2 * P, C, dtype=torch.float64)
        G[:P, :P] = torch.einsum("c,cij->ij", w * scale_i, cgrams_i)
        G[P:, P:] = torch.einsum("c,cij->ij", w * scale_t, cgrams_t)
        G[:P, P:] = torch.einsum("c,cij->ij", w * scale_it, cgrams_both[:, :P, P:])
        G[P:, :P] = G[:P, P:].T
        S[:P] = csums_i * (w * scale_i).unsqueeze(0)
        S[P:] = csums_t * (w * scale_t).unsqueeze(0)
        return solve_ridge(G, S)

    W_zf = solve_weighted(zf_cgrams, zf_csums, zf_counts, 0.5)
    W_mi = solve_weighted(mi_cgrams, mi_csums, mi_counts, 0.5)
    W_pal = solve_weighted(pal_cgrams, pal_csums, pal_counts, 0.5)

    W_sb_both_0 = solve_availability_normalized_joint(
        zf_cgrams, sb_cgrams_img, sb_cgrams_txt, sb_csums_img, sb_csums_txt, zf_counts, sb_counts_img, sb_counts_txt, 0.0
    )
    W_sb_img_0 = solve_weighted(sb_cgrams_img, sb_csums_img, sb_counts_img, 0.0)
    W_sb_txt_0 = solve_weighted(sb_cgrams_txt, sb_csums_txt, sb_counts_txt, 0.0)

    W_sb_both_5 = solve_availability_normalized_joint(
        zf_cgrams, sb_cgrams_img, sb_cgrams_txt, sb_csums_img, sb_csums_txt, zf_counts, sb_counts_img, sb_counts_txt, 0.5
    )
    W_sb_img_5 = solve_weighted(sb_cgrams_img, sb_csums_img, sb_counts_img, 0.5)
    W_sb_txt_5 = solve_weighted(sb_cgrams_txt, sb_csums_txt, sb_counts_txt, 0.5)

    # Test evaluation under the exact same missingness regime (eta, mode)
    t_phi_i_true, t_phi_t_true = project_pair(test_img, test_txt, W_img, W_txt)
    n_test = len(test_y)
    t_has_i, t_has_t = sample_availability_masks(n_test, eta, mode, rng)
    t_has_i_t = torch.from_numpy(t_has_i)
    t_has_t_t = torch.from_numpy(t_has_t)

    # 1. Zero-fill test
    t_phi_zf = torch.cat([
        torch.where(t_has_i_t.unsqueeze(1), t_phi_i_true, torch.zeros_like(t_phi_i_true)),
        torch.where(t_has_t_t.unsqueeze(1), t_phi_t_true, torch.zeros_like(t_phi_t_true)),
    ], dim=1)
    pred_zf = (t_phi_zf @ W_zf).argmax(-1).numpy()

    # 2. Mean-imputed test
    mean_i = sum_phi_i / max(cnt_i, 1)
    mean_t = sum_phi_t / max(cnt_t, 1)
    t_phi_mi = torch.cat([
        torch.where(t_has_i_t.unsqueeze(1), t_phi_i_true, mean_i.unsqueeze(0).expand_as(t_phi_i_true)),
        torch.where(t_has_t_t.unsqueeze(1), t_phi_t_true, mean_t.unsqueeze(0).expand_as(t_phi_t_true)),
    ], dim=1)
    pred_mi = (t_phi_mi @ W_mi).argmax(-1).numpy()

    # 3. PAL-style cross-modal imputed test
    B_i2t = solve_ridge(G_ii_cross, G_it_cross, rel_ridge=0.5) if G_ii_cross.diagonal().sum() > 0 else torch.zeros(P, P, dtype=torch.float64)
    B_t2i = solve_ridge(G_tt_cross, G_it_cross.T, rel_ridge=0.5) if G_tt_cross.diagonal().sum() > 0 else torch.zeros(P, P, dtype=torch.float64)
    t_phi_pal = torch.cat([
        torch.where(t_has_i_t.unsqueeze(1), t_phi_i_true, F.relu(t_phi_t_true @ B_t2i)),
        torch.where(t_has_t_t.unsqueeze(1), t_phi_t_true, F.relu(t_phi_i_true @ B_i2t)),
    ], dim=1)
    pred_pal = (t_phi_pal @ W_pal).argmax(-1).numpy()

    # 4. Sub-Block Exact Solve (routes s=(1,1) -> W_both, s=(1,0) -> W_img, s=(0,1) -> W_txt)
    def predict_subblock(W_b, W_i, W_t):
        logits = torch.zeros(n_test, C, dtype=torch.float64)
        m_both = t_has_i_t & t_has_t_t
        m_io = t_has_i_t & (~t_has_t_t)
        m_to = (~t_has_i_t) & t_has_t_t
        if m_both.any():
            logits[m_both] = t_phi_zf[m_both] @ W_b
        if m_io.any():
            logits[m_io] = t_phi_i_true[m_io] @ W_i
        if m_to.any():
            logits[m_to] = t_phi_t_true[m_to] @ W_t
        return logits.argmax(-1).numpy()

    pred_sb0 = predict_subblock(W_sb_both_0, W_sb_img_0, W_sb_txt_0)
    pred_sb5 = predict_subblock(W_sb_both_5, W_sb_img_5, W_sb_txt_5)

    yt = test_y.numpy()
    return {
        "zero_fill_beta0.5": evaluate_predictions(yt, pred_zf, C),
        "mean_imputed_beta0.5": evaluate_predictions(yt, pred_mi, C),
        "pal_imputed_rls_beta0.5": evaluate_predictions(yt, pred_pal, C),
        "subblock_analytic_beta0.0": evaluate_predictions(yt, pred_sb0, C),
        "subblock_analytic_beta0.5": evaluate_predictions(yt, pred_sb5, C),
        "imputation_error_per_sample": imputation_error_trace / max(imputed_count, 1),
    }


def run_mil_experiment(tasks, test_img, test_txt, test_y, C: int, stage1_modality: str, seed: int):
    """
    2-Modality Incremental Learning (MIL):
    Stage 1 (first half of tasks): only `stage1_modality` ('img' or 'txt') is available.
    Stage 2 (second half of tasks): both 'img' and 'txt' are available.
    Evaluated on locked test under 3 test states: Both (1,1), Image-only (1,0), Text-only (0,1).
    """
    g = torch.Generator().manual_seed(seed)
    W_img = torch.randn(512, P, generator=g, dtype=torch.float64)
    W_txt = torch.randn(512, P, generator=g, dtype=torch.float64)

    zf_cgrams = torch.zeros(C, 2 * P, 2 * P, dtype=torch.float64)
    zf_csums = torch.zeros(2 * P, C, dtype=torch.float64)
    zf_counts = torch.zeros(C, dtype=torch.long)

    sb_cgrams_img = torch.zeros(C, P, P, dtype=torch.float64)
    sb_csums_img = torch.zeros(P, C, dtype=torch.float64)
    sb_counts_img = torch.zeros(C, dtype=torch.long)

    sb_cgrams_txt = torch.zeros(C, P, P, dtype=torch.float64)
    sb_csums_txt = torch.zeros(P, C, dtype=torch.float64)
    sb_counts_txt = torch.zeros(C, dtype=torch.long)

    half = max(1, len(tasks) // 2)
    for k, td in enumerate(tasks):
        phi_i, phi_t = project_pair(td["train_img"], td["train_txt"], W_img, W_txt)
        y = td["train_y"]
        in_stage1 = k < half
        has_i = (not in_stage1) or (stage1_modality == "img")
        has_t = (not in_stage1) or (stage1_modality == "txt")

        phi_i_eff = phi_i if has_i else torch.zeros_like(phi_i)
        phi_t_eff = phi_t if has_t else torch.zeros_like(phi_t)
        phi_joint = torch.cat([phi_i_eff, phi_t_eff], dim=1)

        for c in y.unique().tolist():
            cm = y == c
            fj = phi_joint[cm]
            zf_cgrams[c] += fj.T @ fj
            zf_csums[:, c] += fj.sum(0)
            zf_counts[c] += int(cm.sum())
            if has_i:
                fi = phi_i[cm]
                sb_cgrams_img[c] += fi.T @ fi
                sb_csums_img[:, c] += fi.sum(0)
                sb_counts_img[c] += int(cm.sum())
            if has_t:
                ft = phi_t[cm]
                sb_cgrams_txt[c] += ft.T @ ft
                sb_csums_txt[:, c] += ft.sum(0)
                sb_counts_txt[c] += int(cm.sum())

    def solve_w(cgrams, csums, counts, beta):
        w = compute_class_weights(counts, beta)
        G = torch.einsum("c,cij->ij", w, cgrams)
        S = csums * w.unsqueeze(0)
        return solve_ridge(G, S)

    W_both_5 = solve_w(zf_cgrams, zf_csums, zf_counts, 0.5)
    W_img_5 = solve_w(sb_cgrams_img, sb_csums_img, sb_counts_img, 0.5)
    W_txt_5 = solve_w(sb_cgrams_txt, sb_csums_txt, sb_counts_txt, 0.5)

    t_phi_i, t_phi_t = project_pair(test_img, test_txt, W_img, W_txt)
    t_joint = torch.cat([t_phi_i, t_phi_t], dim=1)
    t_zf_img_only = torch.cat([t_phi_i, torch.zeros_like(t_phi_t)], dim=1)
    t_zf_txt_only = torch.cat([torch.zeros_like(t_phi_i), t_phi_t], dim=1)
    yt = test_y.numpy()

    return {
        "test_both": evaluate_predictions(yt, (t_joint @ W_both_5).argmax(-1).numpy(), C),
        "test_img_only_zerofill": evaluate_predictions(yt, (t_zf_img_only @ W_both_5).argmax(-1).numpy(), C),
        "test_img_only_subblock": evaluate_predictions(yt, (t_phi_i @ W_img_5).argmax(-1).numpy(), C),
        "test_txt_only_zerofill": evaluate_predictions(yt, (t_zf_txt_only @ W_both_5).argmax(-1).numpy(), C),
        "test_txt_only_subblock": evaluate_predictions(yt, (t_phi_t @ W_txt_5).argmax(-1).numpy(), C),
    }


def summarize_runs(runs: list[dict]):
    keys = list(runs[0].keys())
    out = {}
    for k in keys:
        if isinstance(runs[0][k], dict):
            out[k] = {
                m: {
                    "mean": float(np.mean([r[k][m] for r in runs])),
                    "sd": float(np.std([r[k][m] for r in runs], ddof=1)),
                }
                for m in runs[0][k]
            }
        else:
            out[k] = float(np.mean([r[k] for r in runs]))
    return out


def main():
    datasets = {
        "uk_crisismmd": load_uk_tasks(),
        "crisismmd_humanitarian": load_humanitarian_tasks(),
    }
    results = {"cmml": {}, "mil": {}}

    for ds_name, (tasks, test_img, test_txt, test_y, C) in datasets.items():
        print(f"\n==================== {ds_name.upper()} (test n={len(test_y)}, C={C}) ====================")
        results["cmml"][ds_name] = {}
        for mode in ["missing_img", "missing_txt", "mixed"]:
            results["cmml"][ds_name][mode] = {}
            for eta in [0.0, 0.3, 0.5, 0.7]:
                if eta == 0.0 and mode != "mixed":
                    continue
                runs = [run_cmml_experiment(tasks, test_img, test_txt, test_y, C, eta, mode, s) for s in SEEDS]
                summ = summarize_runs(runs)
                results["cmml"][ds_name][mode][f"eta_{eta}"] = summ
                zf = summ["zero_fill_beta0.5"]["macro_f1"]["mean"]
                mi = summ["mean_imputed_beta0.5"]["macro_f1"]["mean"]
                pal = summ["pal_imputed_rls_beta0.5"]["macro_f1"]["mean"]
                sb0 = summ["subblock_analytic_beta0.0"]["macro_f1"]["mean"]
                sb5 = summ["subblock_analytic_beta0.5"]["macro_f1"]["mean"]
                err = summ["imputation_error_per_sample"]
                print(
                    f"  [CMML {mode:<11} eta={eta:.1f}] "
                    f"ZeroFill={zf:.4f} | MeanImp={mi:.4f} | PAL-Imp={pal:.4f} (Tr(ee^T)={err:.1f}) | "
                    f"SubBlock(b=0)={sb0:.4f} | SubBlock(b=0.5)={sb5:.4f} (Δ vs PAL: {sb5 - pal:+.4f})"
                )

        results["mil"][ds_name] = {}
        for stage1_mod in ["img", "txt"]:
            runs = [run_mil_experiment(tasks, test_img, test_txt, test_y, C, stage1_mod, s) for s in SEEDS]
            summ = summarize_runs(runs)
            results["mil"][ds_name][f"stage1_{stage1_mod}_to_both"] = summ
            print(
                f"  [MIL Stage1={stage1_mod}->Both] "
                f"Both={summ['test_both']['macro_f1']['mean']:.4f} | "
                f"ImgOnly(ZF={summ['test_img_only_zerofill']['macro_f1']['mean']:.4f} -> SB={summ['test_img_only_subblock']['macro_f1']['mean']:.4f}) | "
                f"TxtOnly(ZF={summ['test_txt_only_zerofill']['macro_f1']['mean']:.4f} -> SB={summ['test_txt_only_subblock']['macro_f1']['mean']:.4f})"
            )

    out_path = ROOT / "analysis_outputs" / "cmml_mil_2modality_confirmation.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
