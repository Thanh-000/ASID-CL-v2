#!/usr/bin/env python3
"""
Comprehensive Empirical Evaluation of the Upgraded Current System (`BlockAnalyticCL` v2):
  - Self-Governing Continual Policy (`auto_continual_policy`)
  - Empirical-Bayes Auto-Spectral Ridges (`auto_spectral_ridges`)
  - Adaptive Schur-Innovation Decomposition (`schur_innovation_decomposition` + `adaptive_sid=True`)
  - Perturbative Drift-Guarded Block-Gram Transport (`transport_gram(adaptive_drift_guard=True)`)
  - Availability-Normalized Sub-Block Missing-Modality & Machine Unlearning

Evaluated on real features across:
  1. UK-CrisisMMD (6 classes, 3 CIL tasks, 5 seeds)
  2. CrisisMMD v2.0 Humanitarian (7 classes, 7 disaster events, 5 seeds)
  3. UPMC Food-101 (100 classes, 10 CIL tasks, 3,600 samples from real CLIP ViT-B/16 cache, 3 seeds)
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


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_cls)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "worst_class_f1": float(np.min(per_cls)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


@torch.no_grad()
def extract_food101_frozen_features() -> dict[str, torch.Tensor | np.ndarray]:
    feat_cache = ROOT / "data" / "upmc_food101" / "frozen_clip_features_3600.pt"
    if feat_cache.exists():
        return torch.load(feat_cache, map_location="cpu", weights_only=False)

    from run_food101_local import TOP_LAYERS, build_top_model, encode_from_prefix
    pref_path = ROOT / "data" / "upmc_food101" / f"prefix_cache_L{TOP_LAYERS[0]}_3600.pt"
    D = torch.load(pref_path, map_location="cpu", weights_only=False)
    m, _, _ = build_top_model(lora=False)
    all_idx = torch.arange(len(D["y"]))
    I, T = [], []
    for b in range(0, len(all_idx), 256):
        i, t = encode_from_prefix(m, D, all_idx[b:b + 256])
        I.append(i.cpu())
        T.append(t.cpu())
    out = {
        "img": torch.cat(I, 0),
        "txt": torch.cat(T, 0),
        "y": D["y"],
        "train": D["train"],
        "val": D["val"],
        "test": D["test"],
        "task": D["task"],
        "C": int(D["C"]),
    }
    torch.save(out, feat_cache)
    return out


def eval_food101_10task_system(F_food: dict, seeds=(2026, 2027, 2028)) -> dict:
    tr = F_food["train"]
    va = F_food["val"]
    te = F_food["test"]
    y_tr, y_va, y_te = F_food["y"][tr], F_food["y"][va].numpy(), F_food["y"][te].numpy()
    t_tr, t_va = F_food["task"][tr], F_food["task"][va]
    X_tr = {"img": F_food["img"][tr], "txt": F_food["txt"][tr]}
    X_va = {"img": F_food["img"][va], "txt": F_food["txt"][va]}
    X_te = {"img": F_food["img"][te], "txt": F_food["txt"][te]}
    C = F_food["C"]

    # 1. Compare v1 (static ridge, standard solve) vs v2 (Auto-EB + Adaptive SID) across 10 CIL sessions
    v1_traj, v2_traj = [], []
    v1_test, v2_test = [], []
    v1_missing_img, v2_missing_img = [], []
    v1_noisy_txt, v2_noisy_txt = [], []

    for s in seeds:
        m_v1 = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, ridge=0.1, class_balance_beta=0.5, seed=s)
        m_v2 = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, ridge=0.1, class_balance_beta=0.5, seed=s)
        seen = torch.zeros(C, dtype=torch.bool)
        s_v1_traj, s_v2_traj = [], []
        for k in range(10):
            mask_k = torch.tensor(t_tr == k)
            xk = {"img": X_tr["img"][mask_k], "txt": X_tr["txt"][mask_k]}
            yk = y_tr[mask_k]
            seen[yk.unique()] = True
            m_v1.update(xk, yk)
            m_v2.update(xk, yk)
            m_v2.auto_spectral_ridges()

            va_k = t_va <= k
            lg1 = m_v1.logits({"img": X_va["img"][va_k], "txt": X_va["txt"][va_k]})
            lg1[:, ~seen] = -float("inf")
            s_v1_traj.append(float((lg1.argmax(1).numpy() == y_va[va_k]).mean()))

            lg2 = m_v2.logits({"img": X_va["img"][va_k], "txt": X_va["txt"][va_k]}, adaptive_sid=True)
            lg2[:, ~seen] = -float("inf")
            s_v2_traj.append(float((lg2.argmax(1).numpy() == y_va[va_k]).mean()))

        v1_traj.append(s_v1_traj)
        v2_traj.append(s_v2_traj)
        v1_test.append(metrics(y_te, m_v1.predict(X_te, adaptive_sid=False).numpy()))
        v2_test.append(metrics(y_te, m_v2.predict(X_te, adaptive_sid=True).numpy()))

        # Missing image stress test on 100 classes (50% missing image at test time)
        g = torch.Generator().manual_seed(s + 99)
        miss_mask = torch.rand(len(y_te), generator=g) < 0.50
        # Zero-fill vs Sub-block routing
        X_te_zero = {"img": X_te["img"].clone(), "txt": X_te["txt"]}
        X_te_zero["img"][miss_mask] = 0.0
        pred_zf = m_v1.predict(X_te_zero, adaptive_sid=False).numpy()
        pred_sub = m_v2.predict(X_te, adaptive_sid=True).numpy()
        pred_sub[miss_mask.numpy()] = m_v2.predict({"txt": X_te["txt"][miss_mask]}, modalities=("txt",)).numpy()
        v1_missing_img.append(metrics(y_te, pred_zf))
        v2_missing_img.append(metrics(y_te, pred_sub))

        # Soft text corruption (50% of test samples have corrupted text)
        X_te_corr = {"img": X_te["img"], "txt": X_te["txt"].clone()}
        noise = F.normalize(torch.randn_like(X_te_corr["txt"][miss_mask]), dim=-1)
        X_te_corr["txt"][miss_mask] = F.normalize(0.15 * X_te_corr["txt"][miss_mask] + 0.85 * noise, dim=-1)
        v1_noisy_txt.append(metrics(y_te, m_v1.predict(X_te_corr, adaptive_sid=False).numpy()))
        v2_noisy_txt.append(metrics(y_te, m_v2.predict(X_te_corr, adaptive_sid=True).numpy()))

    # 2. Evaluate Drift-Guarded Transport vs Unguarded Transport across 10 tasks under small & severe drift
    drift_regimes = {}
    for drift_label, rot_strength in (("small_nullspace_drift_0.12", 0.08), ("severe_naive_drift_1.20", 0.95)):
        unguarded_scores, guarded_scores, stale_scores = [], [], []
        for s in seeds:
            m_stale = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            m_ung = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            m_grd = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)

            I_cur = X_tr["img"].clone()
            T_cur = X_tr["txt"].clone()
            I_te_cur = X_te["img"].clone()
            T_te_cur = X_te["txt"].clone()

            for k in range(10):
                mask_k = torch.tensor(t_tr == k)
                if k > 0:
                    g = torch.Generator().manual_seed(s * 100 + k)
                    A_i = torch.randn(512, 512, generator=g) / (512 ** 0.5)
                    A_t = torch.randn(512, 512, generator=g) / (512 ** 0.5)
                    M_i = torch.eye(512) + (rot_strength / 9.0) * (A_i - A_i.T)
                    M_t = torch.eye(512) + (rot_strength / 9.0) * (A_t - A_t.T)

                    old_k = {"img": I_cur[mask_k].clone(), "txt": T_cur[mask_k].clone()}
                    I_cur = F.normalize(I_cur @ M_i, dim=-1)
                    T_cur = F.normalize(T_cur @ M_t, dim=-1)
                    I_te_cur = F.normalize(I_te_cur @ M_i, dim=-1)
                    T_te_cur = F.normalize(T_te_cur @ M_t, dim=-1)
                    new_k = {"img": I_cur[mask_k].clone(), "txt": T_cur[mask_k].clone()}

                    m_ung.transport_gram(old_k, new_k, adaptive_drift_guard=False)
                    m_grd.transport_gram(old_k, new_k, max_drift=0.35, adaptive_drift_guard=True)

                cur_k = {"img": I_cur[mask_k], "txt": T_cur[mask_k]}
                yk = y_tr[mask_k]
                m_stale.update(cur_k, yk)
                m_ung.update(cur_k, yk)
                m_grd.update(cur_k, yk)

            m_stale.auto_spectral_ridges()
            m_ung.auto_spectral_ridges()
            m_grd.auto_spectral_ridges()
            X_te_final = {"img": I_te_cur, "txt": T_te_cur}
            stale_scores.append(metrics(y_te, m_stale.predict(X_te_final, adaptive_sid=True).numpy()))
            unguarded_scores.append(metrics(y_te, m_ung.predict(X_te_final, adaptive_sid=True).numpy()))
            guarded_scores.append(metrics(y_te, m_grd.predict(X_te_final, adaptive_sid=True).numpy()))

        drift_regimes[drift_label] = {
            "stale_acc": float(np.mean([x["acc"] for x in stale_scores])),
            "unguarded_transport_acc": float(np.mean([x["acc"] for x in unguarded_scores])),
            "guarded_transport_acc": float(np.mean([x["acc"] for x in guarded_scores])),
        }

    def _agg(lst):
        return {
            k: {
                "mean": float(np.mean([r[k] for r in lst])),
                "std": float(np.std([r[k] for r in lst], ddof=1)),
            }
            for k in ("acc", "macro_f1", "worst_class_f1", "balanced_acc")
        }

    return {
        "v1_standard_test": _agg(v1_test),
        "v2_upgraded_test": _agg(v2_test),
        "v1_val_trajectory_mean": [round(float(x), 4) for x in np.mean(v1_traj, axis=0)],
        "v2_val_trajectory_mean": [round(float(x), 4) for x in np.mean(v2_traj, axis=0)],
        "missing_50pct_img_v1_zerofill": _agg(v1_missing_img),
        "missing_50pct_img_v2_subblock": _agg(v2_missing_img),
        "corrupted_50pct_txt_v1_static": _agg(v1_noisy_txt),
        "corrupted_50pct_txt_v2_sid": _agg(v2_noisy_txt),
        "drift_guard_ablation": drift_regimes,
    }


def eval_crisis_datasets_v1_vs_v2(seeds=(2027, 2028, 2029, 2030, 2031)) -> dict:
    out = {}
    for ds_name, loader in (
        ("uk_crisismmd", load_uk_tasks_and_full),
        ("crisismmd_humanitarian", load_humanitarian_events_and_full),
    ):
        _, (tr_i, tr_t, tr_y), _, (te_i, te_t, te_y), C = loader()
        X_tr = {"img": tr_i.float(), "txt": tr_t.float()}
        y_tr = tr_y.long()
        X_te = {"img": te_i.float(), "txt": te_t.float()}
        y_te = te_y.numpy()

        v0_unweighted, v1_balanced, v2_schur, v2_upgraded = [], [], [], []
        v1_corr_txt, v2_corr_txt = [], []
        last_ridges = {}
        for s in seeds:
            m0 = BlockAnalyticCL({"img": X_tr["img"].shape[1], "txt": X_tr["txt"].shape[1]}, C, projection_dim=1024, class_balance_beta=0.0, seed=s)
            m0.update(X_tr, y_tr)
            v0_unweighted.append(metrics(y_te, m0.predict(X_te, adaptive_sid=False).numpy()))

            m1 = BlockAnalyticCL({"img": X_tr["img"].shape[1], "txt": X_tr["txt"].shape[1]}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            m1.update(X_tr, y_tr)
            v1_balanced.append(metrics(y_te, m1.predict(X_te, adaptive_sid=False).numpy()))

            m2 = BlockAnalyticCL({"img": X_tr["img"].shape[1], "txt": X_tr["txt"].shape[1]}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            m2.update(X_tr, y_tr)
            last_ridges = m2.auto_spectral_ridges()
            v2_schur.append(metrics(y_te, m2.predict(X_te, adaptive_sid=False).numpy()))
            v2_upgraded.append(metrics(y_te, m2.predict(X_te, adaptive_sid=True).numpy()))

            g = torch.Generator().manual_seed(s + 99)
            mask = torch.rand(len(y_te), generator=g) < 0.50
            X_corr = {"img": X_te["img"], "txt": X_te["txt"].clone()}
            noise = F.normalize(torch.randn(X_corr["txt"][mask].shape, generator=g), dim=-1)
            X_corr["txt"][mask] = F.normalize(0.15 * X_corr["txt"][mask] + 0.85 * noise, dim=-1)
            v1_corr_txt.append(metrics(y_te, m1.predict(X_corr, adaptive_sid=False).numpy()))
            v2_corr_txt.append(metrics(y_te, m2.predict(X_corr, adaptive_sid=True).numpy()))

        def _agg(lst):
            return {
                k: {
                    "mean": float(np.mean([r[k] for r in lst])),
                    "std": float(np.std([r[k] for r in lst], ddof=1)),
                }
                for k in ("macro_f1", "worst_class_f1", "balanced_acc", "acc")
            }

        out[ds_name] = {
            "v0_unweighted_beta0.0": _agg(v0_unweighted),
            "v1_balanced_beta0.5": _agg(v1_balanced),
            "v2_auto_eb_schur": _agg(v2_schur),
            "v2_upgraded_auto_eb_sid": _agg(v2_upgraded),
            "corrupted_50pct_txt_v1_static": _agg(v1_corr_txt),
            "corrupted_50pct_txt_v2_sid": _agg(v2_corr_txt),
            "auto_ridges_sample": last_ridges,
        }
    return out


def main():
    t0 = time.time()
    print("Extracting / loading UPMC Food-101 frozen CLIP features (3,600 samples, 100 classes)...")
    F_food = extract_food101_frozen_features()
    print("Evaluating Upgraded System (v2) vs Baseline System (v1) on UPMC Food-101 (10 tasks)...")
    food_eval = eval_food101_10task_system(F_food)
    print("Evaluating Upgraded System (v2) vs v1/v0 on UK-CrisisMMD & CrisisMMD Humanitarian...")
    crisis_eval = eval_crisis_datasets_v1_vs_v2()

    policies = {
        "food101_cil": BlockAnalyticCL.auto_continual_policy(g1_gain_pts=-0.08, incremental_mode="cil"),
        "crisismmd_damage_dil": BlockAnalyticCL.auto_continual_policy(g1_gain_pts=5.42, incremental_mode="dil"),
        "generic_domain_shifted_cil": BlockAnalyticCL.auto_continual_policy(g1_gain_pts=4.20, incremental_mode="cil"),
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "auto_continual_policies": policies,
        "food101_100class_10task_evaluation": food_eval,
        "crisis_benchmarks_evolution": crisis_eval,
    }
    out_path = ROOT / "analysis_outputs" / "upgraded_system_evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
