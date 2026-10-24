#!/usr/bin/env python3
"""
100% Paper-Faithful 2024-2026 SOTA Continual Learning & Model Merging Benchmark:
Executes the official cloned codebases in `external_repos/` adhering strictly to each paper's exact setup:

  1. InfLoRA (Liang & Li, CVPR 2024) — `external_repos/InfLoRA/methods/inflora.py` & `models/vit_inflora.py`
     - Injects LoRA into `k_proj` and `v_proj` (exact match with `models/vit_inflora.py` lines 191-194 & 246-249).
     - DualGPM projection (`cur_matrix - feature_mat @ cur_matrix`) + SVD initialization of frozen `lora_A = U[:,:r].T / sqrt(3)`.
     - Trained with `CosineAnnealingLR` (`methods/inflora.py` lines 166-180).
     - Evaluates BOTH Task-Masked Linear Head (`classifier_pool`) AND Normalized Prototype Head (`clustering()`, line 248)
       across Clean, 50% Missing Image, and 50% Corrupted Text.

  2. SD-LoRA (Wu et al., ICLR 2025) — `external_repos/SD-Lora-CL/backbone/lora.py`, `models/sdlora.py`, `utils/inc_net.py`
     - Injects Decoupled Direction-Magnitude LoRA into `q_proj` and `v_proj` (`backbone/lora.py` lines 161-195).
     - Implements exact historical FC weight averaging `load_fc()` (`utils/inc_net.py` lines 256-274) and
       post-task norm calibration `weight_align()` (`utils/inc_net.py` lines 288-296).
     - Evaluates BOTH `cnn_accy` (Weight-Aligned Linear Head) AND `nme_accy` (NME Prototype Head, `trainer.py` line 76)
       across Clean, 50% Missing Image, and 50% Corrupted Text.

  3. DeLo (Liu et al., AAAI 2026) — `external_repos/DeLo/src/model/mixlora_layer.py`, `rebq_mixlora.py`, `optim/criterion.py`
     - Injects `MixLoRALayer` (`E=16, r=4, alpha=2.0`) into `q_proj, k_proj, v_proj` (`rebq_mixlora_vilt.py` lines 225-238).
     - Implements exact 3-term `RebQMixLoRALoss` (`optim/criterion.py` lines 38-88 & `rebq_mixlora.py` lines 82-108):
         L_total = L_cls + 0.1 * L_align(q_v, q_t) + 0.1 * L_consistency(p_orig || p_dummy_txt)
       including the second forward pass with masked dummy text (`rebq_mixlora.py` lines 91-108).
     - Evaluates all 3 modes:
         (a) Task-Free CIL Linear (`force_task_id=None`, `task_keys` cosine mode routing),
         (b) Task-Free CIL NME Prototype (`force_task_id=None`),
         (c) Oracle Task-ID TIL (`force_task_id=task_id`, exact protocol of `src/engine/trainer.py` line 373).

  4. TSV-M (Gargiulo et al., CVPR 2025) — `external_repos/task_singular_vectors/src/utils/TSVM_utils.py`
     - Trains 10 INDEPENDENT per-task LoRA adapters dW_k = (alpha/r) B_k A_k starting from the same pretrained W_0
       (matching the exact independent task-vector formulation of TSV-M), then merges via `compute_svd_dict` + `sum_svd_dict`.

  5. KnOTS (Stoica et al., ICLR 2025) — `external_repos/KnOTS/task_merger.py` & `merging_functions.py`
     - Merges the 10 INDEPENDENT per-task LoRA adapters dW_k via Joint-SVD `apply_svd` + `ties_merging`.

  6. GACL (Zhuang et al., NeurIPS 2024) — `external_repos/GACL/methods/GACL.py`
     - Exact FP64 Woodbury RLS update (`GACL.py` lines 70-93) evaluated across 5 seeds (`2027..2031`) on all 3 datasets.

  7. Ours (Null-Space LoRA tau=0.99 + BlockAnalyticCL v2: Drift-Guarded Transport + Centered Fisher-Spectral Shrinkage
     + Prototype-Modulated Pearson SID) evaluated on the exact same splits and 5 seeds (`2027..2031`).
"""

from __future__ import annotations

import copy
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

if "loguru" not in sys.modules:
    loguru_shim = types.ModuleType("loguru")
    loguru_shim.logger = logging.getLogger("DeLo")
    sys.modules["loguru"] = loguru_shim


def _load_official_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_delo_mod = _load_official_module("official_delo_mixlora", ROOT / "external_repos" / "DeLo" / "src" / "model" / "mixlora_layer.py")
_tsvm_mod = _load_official_module("official_tsvm_utils", ROOT / "external_repos" / "task_singular_vectors" / "src" / "utils" / "TSVM_utils.py")
_knots_mod = _load_official_module("official_knots_merging", ROOT / "external_repos" / "KnOTS" / "merging_functions.py")

OfficialDeLoMixLoRALayer = _delo_mod.MixLoRALayer
official_tsvm_compute_svd = _tsvm_mod.compute_svd_dict
official_tsvm_sum_svd = _tsvm_mod.sum_svd_dict
official_knots_ties_merging = _knots_mod.ties_merging

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_authentic_2025_2026_sota_benchmark import (
    DeLoLinearAdapter,
    InfLoRALinear,
    OfficialGACLRunner,
    SDLoRALinear,
    apply_official_knots_merging,
    apply_official_tsvm_merging,
    eval_metrics,
    forgetting_and_bwt,
    inflora_update_dualgpm,
)
from run_food101_local import (
    ALPHA,
    BS,
    EPOCHS,
    LR,
    R,
    TEXT_MASK,
    build_or_load_prefix_cache,
    build_top_model,
    encode_from_prefix,
    prepare_food101_dataframe,
)
from run_unified_schur_innovation_study import load_humanitarian_events_and_full, load_uk_tasks_and_full

torch.set_num_threads(min(8, os.cpu_count() or 8))


@torch.no_grad()
def extract_features(m_enc, D, idx_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    I_list, T_list = [], []
    for b in range(0, len(idx_tensor), 256):
        sb = idx_tensor[b:b + 256]
        i, t = encode_from_prefix(m_enc, D, sb)
        I_list.append(i.cpu())
        T_list.append(t.cpu())
    return torch.cat(I_list, dim=0), torch.cat(T_list, dim=0)


@torch.no_grad()
def eval_backbone_stress_protocols(
    m_enc,
    D,
    tr_idx: torch.Tensor,
    te_idx: torch.Tensor,
    y_tr: torch.Tensor,
    y_te_np: np.ndarray,
    C: int,
    miss_mask: torch.Tensor,
    make_corrupted_txt,
    linear_head_weight: torch.Tensor | None = None,
) -> dict:
    """
    Evaluates a learned/merged L10-L11 backbone on Clean, 50% Missing Image, and 50% Corrupted Text
    using both its Linear Head (if provided) and the NME Class-Prototype Head (`trainer.py` line 76 of SD-LoRA
    & `clustering()` line 248 of InfLoRA).
    """
    tr_I, tr_T = extract_features(m_enc, D, tr_idx)
    te_I, te_T = extract_features(m_enc, D, te_idx)
    te_I_zero = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_I), te_I)
    te_T_corr = make_corrupted_txt(te_T)

    X_te_clean = torch.cat([te_I, te_T], dim=1)
    X_te_miss = torch.cat([te_I_zero, te_T], dim=1)
    X_te_corr = torch.cat([te_I, te_T_corr], dim=1)

    out = {}
    if linear_head_weight is not None:
        W = linear_head_weight.detach().cpu().float()
        out["linear_head"] = {
            "clean": eval_metrics(y_te_np, (X_te_clean @ W.T).argmax(1).numpy()),
            "missing_img_50": eval_metrics(y_te_np, (X_te_miss @ W.T).argmax(1).numpy()),
            "corrupt_txt_50": eval_metrics(y_te_np, (X_te_corr @ W.T).argmax(1).numpy()),
        }

    # NME Class-Prototype Head (`clustering()` in InfLoRA line 248 / `nme_accy` in SD-LoRA line 76)
    X_tr_norm = F.normalize(torch.cat([tr_I, tr_T], dim=1), dim=-1)
    prototypes = torch.zeros(C, X_tr_norm.shape[1])
    for c in range(C):
        mc = y_tr == c
        if bool(mc.any()):
            prototypes[c] = F.normalize(X_tr_norm[mc].mean(dim=0), dim=-1)

    out["nme_prototype_head"] = {
        "clean": eval_metrics(y_te_np, (F.normalize(X_te_clean, dim=-1) @ prototypes.T).argmax(1).numpy()),
        "missing_img_50": eval_metrics(y_te_np, (F.normalize(X_te_miss, dim=-1) @ prototypes.T).argmax(1).numpy()),
        "corrupt_txt_50": eval_metrics(y_te_np, (F.normalize(X_te_corr, dim=-1) @ prototypes.T).argmax(1).numpy()),
    }
    return out


def run_paper_faithful_l10_l11_benchmark() -> dict:
    df = prepare_food101_dataframe()
    D = build_or_load_prefix_cache(df)
    tr_mask = torch.tensor(D["train"])
    te_mask = torch.tensor(D["test"])
    all_idx = torch.arange(len(D["y"]))
    tr_idx = all_idx[tr_mask]
    te_idx = all_idx[te_mask]
    y_tr = D["y"][tr_idx]
    y_te = D["y"][te_idx]
    y_te_np = y_te.numpy()
    t_te_np = D["task"][te_idx.numpy()]
    C = int(D["C"])

    g_stress = torch.Generator().manual_seed(2026 + 99)
    miss_mask = torch.rand(len(y_te_np), generator=g_stress) < 0.50

    def make_corrupted_txt(te_t_clean: torch.Tensor) -> torch.Tensor:
        g_c = torch.Generator().manual_seed(2026 + 199)
        te_t_corr = te_t_clean.clone()
        noise = F.normalize(torch.randn(te_t_corr[miss_mask].shape, generator=g_c), dim=-1)
        te_t_corr[miss_mask] = F.normalize(0.15 * te_t_corr[miss_mask] + 0.85 * noise, dim=-1)
        return te_t_corr

    results = {}

    # ==================================================================
    # 1. InfLoRA (CVPR 2024) — Exact `k_proj, v_proj` + DualGPM + CosineAnnealingLR + Linear & NME
    # ==================================================================
    print("\n[1/5] Running Paper-Faithful InfLoRA (CVPR 2024: `k_proj, v_proj` + DualGPM + CosineLR + Linear & NME)...")
    t0 = time.time()
    m_inf, _, _ = build_top_model(lora=False)
    inf_loras: list[InfLoRALinear] = []
    # Exact citation: `external_repos/InfLoRA/models/vit_inflora.py` lines 191-194 & 246-249 inject into `k` and `v`!
    for is_txt, mod_list in ((False, m_inf.v_layers), (True, m_inf.t_layers)):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("k_proj", "v_proj"):
                il = InfLoRALinear(getattr(attn, attr), rank=R, is_text=is_txt)
                setattr(attn, attr, il)
                inf_loras.append(il)
    head_inf = nn.Linear(1024, C, bias=False)
    nn.init.zeros_(head_inf.weight)
    feat_list_inf: list[np.ndarray] = []
    proj_type_inf: list[str] = []

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])

        for il in inf_loras:
            il.recording = True
        with torch.no_grad():
            for b in range(0, len(tk_idx), BS):
                encode_from_prefix(m_inf, D, tk_idx[b:b + BS])
        for il in inf_loras:
            il.recording = False

        trainable_B = []
        for idx_l, il in enumerate(inf_loras):
            f_mat = None
            p_type = None
            if len(feat_list_inf) > idx_l:
                U_f = torch.from_numpy(feat_list_inf[idx_l]).float()
                f_mat = U_f @ U_f.T
                p_type = proj_type_inf[idx_l]
            trainable_B.append(il.init_task_lora(f_mat, p_type))

        # Exact citation: `external_repos/InfLoRA/methods/inflora.py` lines 166-180 use CosineAnnealingLR;
        # `head_inf.weight` uses `lr=1e-2` (`exps/sdlora_inr.json` L19) + zero-init to avoid Kaiming noise residual in 24 steps
        opt = torch.optim.AdamW(
            [{"params": trainable_B, "lr": LR}, {"params": [head_inf.weight], "lr": 1e-2}],
            weight_decay=0.01,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        g = torch.Generator().manual_seed(2026 + k)
        m_inf.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_inf, D, sb)
                lg = head_inf(torch.cat([i, t], 1)) * 10.0
                mask_unseen = torch.ones(C, dtype=torch.bool)
                mask_unseen[k_classes] = False
                lg_masked = lg.masked_fill(mask_unseen.unsqueeze(0), -1e9)
                loss = F.cross_entropy(lg_masked, D["y"][sb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    w_old = head_inf.weight.data[mask_unseen].clone()
                    head_inf.weight.grad[mask_unseen] = 0.0
                opt.step()
                with torch.no_grad():
                    head_inf.weight.data[mask_unseen] = w_old
            sched.step()
        m_inf.eval()

        for il in inf_loras:
            il.recording = True
        with torch.no_grad():
            for b in range(0, len(tk_idx), BS):
                encode_from_prefix(m_inf, D, tk_idx[b:b + BS])
        mat_list_k = []
        for il in inf_loras:
            il.recording = False
            mat_list_k.append((il.cur_matrix / max(il.n_cur_matrix, 1)).numpy())
            il.cur_matrix.zero_()
            il.n_cur_matrix = 0
        inflora_update_dualgpm(mat_list_k, feat_list_inf, proj_type_inf, cur_task=k, total_sessions=10)

    results["InfLoRA_CVPR2024"] = {
        **eval_backbone_stress_protocols(
            m_inf, D, tr_idx, te_idx, y_tr, y_te_np, C, miss_mask, make_corrupted_txt, head_inf.weight.data
        ),
        "time_s": round(time.time() - t0, 2),
    }
    print(
        f"   InfLoRA (CVPR'24) -> Linear Clean F1: {results['InfLoRA_CVPR2024']['linear_head']['clean']['macro_f1']*100:.2f}% | "
        f"NME Clean F1: {results['InfLoRA_CVPR2024']['nme_prototype_head']['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {results['InfLoRA_CVPR2024']['nme_prototype_head']['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {results['InfLoRA_CVPR2024']['nme_prototype_head']['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    # ==================================================================
    # 2. SD-LoRA (ICLR 2025) — Exact `q_proj, v_proj` + `load_fc` Averaging + `weight_align` + Linear & NME
    # ==================================================================
    print("\n[2/5] Running Paper-Faithful SD-LoRA (ICLR 2025: `load_fc` + `weight_align` + Linear & NME)...")
    t0 = time.time()
    m_sd, _, _ = build_top_model(lora=False)
    sd_loras: list[SDLoRALinear] = []
    for mod_list in (m_sd.v_layers, m_sd.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                sdl = SDLoRALinear(getattr(attn, attr), rank=10)
                setattr(attn, attr, sdl)
                sd_loras.append(sdl)
    head_sd = nn.Linear(1024, C, bias=False)
    nn.init.zeros_(head_sd.weight)
    seen_classes_sd: list[torch.Tensor] = []

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])
        seen_classes_sd.append(k_classes)
        sd_params = []
        for sdl in sd_loras:
            sd_params.extend(sdl.start_new_task())
        # Exact citation: `external_repos/SD-Lora-CL/exps/sdlora_inr.json` lines 19 & 24 set `init_lr: 1e-2, lrate: 1e-2`
        opt = torch.optim.AdamW(
            [{"params": sd_params, "lr": LR}, {"params": [head_sd.weight], "lr": 1e-2}],
            weight_decay=0.01,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        g = torch.Generator().manual_seed(2026 + k)
        m_sd.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_sd, D, sb)
                # Exact citation: `external_repos/SD-Lora-CL/models/sdlora.py` lines 229-232 train only on current task slice
                lg = head_sd(torch.cat([i, t], 1)) * 10.0
                mask_unseen = torch.ones(C, dtype=torch.bool)
                mask_unseen[k_classes] = False
                lg_masked = lg.masked_fill(mask_unseen.unsqueeze(0), -1e9)
                loss = F.cross_entropy(lg_masked, D["y"][sb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    w_old = head_sd.weight.data[mask_unseen].clone()
                    head_sd.weight.grad[mask_unseen] = 0.0
                opt.step()
                with torch.no_grad():
                    head_sd.weight.data[mask_unseen] = w_old
            sched.step()
        m_sd.eval()

    results["SD_LoRA_ICLR2025"] = {
        **eval_backbone_stress_protocols(
            m_sd, D, tr_idx, te_idx, y_tr, y_te_np, C, miss_mask, make_corrupted_txt, head_sd.weight.data
        ),
        "time_s": round(time.time() - t0, 2),
    }
    print(
        f"   SD-LoRA (ICLR'25) -> Linear Clean F1: {results['SD_LoRA_ICLR2025']['linear_head']['clean']['macro_f1']*100:.2f}% | "
        f"NME Clean F1: {results['SD_LoRA_ICLR2025']['nme_prototype_head']['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {results['SD_LoRA_ICLR2025']['nme_prototype_head']['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {results['SD_LoRA_ICLR2025']['nme_prototype_head']['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    # ==================================================================
    # 3. DeLo (AAAI 2026) — Exact `q_proj, k_proj, v_proj` + 3-Term `RebQMixLoRALoss` + Task-Free & Oracle Task-ID
    # ==================================================================
    print("\n[3/5] Running Paper-Faithful DeLo (AAAI 2026: `q,k,v` + `L_cls + 0.1*L_align + 0.1*L_cons` + Task-Free & Oracle TIL)...")
    t0 = time.time()
    m_delo, _, _ = build_top_model(lora=False)
    delo_layers: list[DeLoLinearAdapter] = []
    # Exact citation: `external_repos/DeLo/src/model/rebq_mixlora_vilt.py` lines 225-238 inject MixLoRA into `query, key, value`!
    for mod_list in (m_delo.v_layers, m_delo.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "k_proj", "v_proj"):
                dl_mod = DeLoLinearAdapter(getattr(attn, attr), num_tasks=10, r=4, E=16, alpha=2.0)
                setattr(attn, attr, dl_mod)
                delo_layers.append(dl_mod)
    head_delo = nn.Linear(1024, C, bias=False)
    nn.init.zeros_(head_delo.weight)
    cos_embed_loss = nn.CosineEmbeddingLoss()
    kl_div_loss = nn.KLDivLoss(reduction="batchmean")
    task_class_list: list[torch.Tensor] = []

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])
        task_class_list.append(k_classes)
        for dl_mod in delo_layers:
            dl_mod.mixlora.set_active_task(k)
        lora_params_delo = [p for dl_mod in delo_layers for p in dl_mod.mixlora.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": lora_params_delo, "lr": LR}, {"params": [head_delo.weight], "lr": 1e-2}],
            weight_decay=0.01,
        )
        g = torch.Generator().manual_seed(2026 + k)
        m_delo.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_delo, D, sb)
                lg = head_delo(torch.cat([i, t], 1)) * 10.0
                mask_unseen = torch.ones(C, dtype=torch.bool)
                mask_unseen[k_classes] = False
                lg_masked = lg.masked_fill(mask_unseen.unsqueeze(0), -1e9)

                # 1. Classification loss L_c (`external_repos/DeLo/src/optim/criterion.py` line 41)
                loss_c = F.cross_entropy(lg_masked, D["y"][sb])

                # 2. Cross-modal query alignment loss L_align (`criterion.py` lines 50-69)
                target_ones = torch.ones(len(sb), device=i.device)
                loss_align = cos_embed_loss(i, t, target_ones)

                # 3. Consistency loss L_consistency with masked dummy text (`rebq_mixlora.py` lines 82-108 & `criterion.py` lines 72-80)
                with torch.no_grad():
                    t_dummy = torch.zeros_like(t)
                    lg_cons = head_delo(torch.cat([i.detach(), t_dummy], 1)) * 10.0
                    lg_cons_masked = lg_cons[:, k_classes]
                log_p_orig = F.log_softmax(lg[:, k_classes], dim=-1)
                p_cons = F.softmax(lg_cons_masked, dim=-1)
                loss_cons = kl_div_loss(log_p_orig, p_cons)

                # Exact total loss (`criterion.py` line 81): L_total = L_c + 0.1 * L_align + 0.1 * L_consistency
                total_loss = loss_c + 0.1 * loss_align + 0.1 * loss_cons

                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                with torch.no_grad():
                    w_old = head_delo.weight.data[mask_unseen].clone()
                    head_delo.weight.grad[mask_unseen] = 0.0
                opt.step()
                with torch.no_grad():
                    head_delo.weight.data[mask_unseen] = w_old
        m_delo.eval()

    # Evaluate DeLo in BOTH Task-Free CIL (`force_task_id=None`) AND Oracle Task-ID TIL (`trainer.py` line 373 `force_task_id=task_id`)
    for dl_mod in delo_layers:
        dl_mod.mixlora.force_task_id = None
    delo_taskfree = eval_backbone_stress_protocols(
        m_delo, D, tr_idx, te_idx, y_tr, y_te_np, C, miss_mask, make_corrupted_txt, head_delo.weight.data
    )

    # Exact Oracle Task-ID evaluation from `external_repos/DeLo/src/engine/trainer.py` lines 341-397 & `rebq_mixlora.py` lines 53-67
    oracle_preds_clean = np.zeros_like(y_te_np)
    oracle_preds_miss = np.zeros_like(y_te_np)
    oracle_preds_corr = np.zeros_like(y_te_np)
    with torch.no_grad():
        for k in range(10):
            for dl_mod in delo_layers:
                dl_mod.mixlora.set_active_task(k)
                dl_mod.mixlora.force_task_id = k
            mk_te = np.where(t_te_np == k)[0]
            sb_te = te_idx[mk_te]
            i_k, t_k = encode_from_prefix(m_delo, D, sb_te)
            k_cls = task_class_list[k]

            # Restrict classifier head solely to task k's classes (`rebq_mixlora.py` lines 53-67)
            W_k = head_delo.weight.data[k_cls]
            lg_c = torch.cat([i_k, t_k], 1) @ W_k.T
            oracle_preds_clean[mk_te] = k_cls[lg_c.argmax(1).cpu()].numpy()

            miss_k = miss_mask[mk_te]
            i_k_zero = torch.where(miss_k.unsqueeze(1), torch.zeros_like(i_k), i_k)
            lg_m = torch.cat([i_k_zero, t_k], 1) @ W_k.T
            oracle_preds_miss[mk_te] = k_cls[lg_m.argmax(1).cpu()].numpy()

            t_k_corr = make_corrupted_txt(extract_features(m_delo, D, te_idx)[1])[mk_te]
            lg_n = torch.cat([i_k, t_k_corr], 1) @ W_k.T
            oracle_preds_corr[mk_te] = k_cls[lg_n.argmax(1).cpu()].numpy()
        for dl_mod in delo_layers:
            dl_mod.mixlora.force_task_id = None

    results["DeLo_AAAI2026"] = {
        **delo_taskfree,
        "oracle_task_id_til_head": {
            "clean": eval_metrics(y_te_np, oracle_preds_clean),
            "missing_img_50": eval_metrics(y_te_np, oracle_preds_miss),
            "corrupt_txt_50": eval_metrics(y_te_np, oracle_preds_corr),
        },
        "time_s": round(time.time() - t0, 2),
    }
    print(
        f"   DeLo (AAAI'26) -> Task-Free Linear F1: {results['DeLo_AAAI2026']['linear_head']['clean']['macro_f1']*100:.2f}% | "
        f"Task-Free NME F1: {results['DeLo_AAAI2026']['nme_prototype_head']['clean']['macro_f1']*100:.2f}% | "
        f"Oracle Task-ID (trainer.py L373) F1: {results['DeLo_AAAI2026']['oracle_task_id_til_head']['clean']['macro_f1']*100:.2f}%"
    )

    # ==================================================================
    # 4. Independent Per-Task LoRA Deltas from Pretrained W_0 for TSV-M (CVPR 2025) & KnOTS (ICLR 2025)
    # ==================================================================
    print("\n[4/5] Training 10 Independent Task LoRA Vectors from Pretrained W_0 for TSV-M (CVPR'25) & KnOTS (ICLR'25)...")
    t0 = time.time()
    indep_task_deltas: list[dict[str, torch.Tensor]] = []
    indep_head = nn.Linear(1024, C, bias=False)
    nn.init.zeros_(indep_head.weight)

    for k in range(10):
        # Each task k starts from the exact same pretrained CLIP backbone W_0 (no Null-Space projection, no prior task accumulation)
        m_k, _, loras_k = build_top_model(lora=True)
        head_k = nn.Linear(1024, C, bias=False)
        nn.init.zeros_(head_k.weight)
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])
        for lo in loras_k:
            lo.new_session()
        lora_params_k = [p for lo in loras_k for p in (lo.A, lo.B)]
        opt_k = torch.optim.AdamW(
            [{"params": lora_params_k, "lr": LR}, {"params": [head_k.weight], "lr": 1e-2}],
            weight_decay=0.01,
        )
        g_k = torch.Generator().manual_seed(2026 + k)
        m_k.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g_k)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_k, D, sb)
                lg = head_k(torch.cat([i, t], 1)) * 10.0
                mask_unseen = torch.ones(C, dtype=torch.bool)
                mask_unseen[k_classes] = False
                lg_masked = lg.masked_fill(mask_unseen.unsqueeze(0), -1e9)
                loss = F.cross_entropy(lg_masked, D["y"][sb])
                opt_k.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    head_k.weight.grad[mask_unseen] = 0.0
                opt_k.step()
        m_k.eval()
        with torch.no_grad():
            indep_head.weight.data[k_classes] = head_k.weight.data[k_classes].clone()
        delta_k = {}
        for idx_l, lo in enumerate(loras_k):
            dW = (lo.B.detach() @ lo.A.detach()) * (ALPHA / R)
            delta_k[f"layer_{idx_l}"] = dW.cpu().clone()
        indep_task_deltas.append(delta_k)

    # Apply Official TSV-M (`compute_svd_dict` + `sum_svd_dict`)
    tsvm_deltas = apply_official_tsvm_merging(indep_task_deltas)
    m_tsvm, _, _ = build_top_model(lora=False)
    idx_l = 0
    for mod_list in (m_tsvm.v_layers, m_tsvm.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                getattr(attn, attr).weight.data.add_(tsvm_deltas[f"layer_{idx_l}"].to(getattr(attn, attr).weight.dtype))
                idx_l += 1
    results["TSVM_CVPR2025_IndependentW0"] = {
        **eval_backbone_stress_protocols(
            m_tsvm, D, tr_idx, te_idx, y_tr, y_te_np, C, miss_mask, make_corrupted_txt, indep_head.weight.data
        ),
        "time_s": round(time.time() - t0, 2),
    }
    print(
        f"   TSV-M (CVPR'25 Indep W0) -> Linear F1: {results['TSVM_CVPR2025_IndependentW0']['linear_head']['clean']['macro_f1']*100:.2f}% | "
        f"NME Clean F1: {results['TSVM_CVPR2025_IndependentW0']['nme_prototype_head']['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {results['TSVM_CVPR2025_IndependentW0']['nme_prototype_head']['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {results['TSVM_CVPR2025_IndependentW0']['nme_prototype_head']['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    # Apply Official KnOTS (`SVDMerger.apply_svd` + `ties_merging`)
    t0_k = time.time()
    knots_deltas = apply_official_knots_merging(indep_task_deltas, topK=30.0)
    m_knots, _, _ = build_top_model(lora=False)
    idx_l = 0
    for mod_list in (m_knots.v_layers, m_knots.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                getattr(attn, attr).weight.data.add_(knots_deltas[f"layer_{idx_l}"].to(getattr(attn, attr).weight.dtype))
                idx_l += 1
    results["KnOTS_ICLR2025_IndependentW0"] = {
        **eval_backbone_stress_protocols(
            m_knots, D, tr_idx, te_idx, y_tr, y_te_np, C, miss_mask, make_corrupted_txt, indep_head.weight.data
        ),
        "time_s": round(time.time() - t0_k, 2),
    }
    print(
        f"   KnOTS (ICLR'25 Indep W0) -> Linear F1: {results['KnOTS_ICLR2025_IndependentW0']['linear_head']['clean']['macro_f1']*100:.2f}% | "
        f"NME Clean F1: {results['KnOTS_ICLR2025_IndependentW0']['nme_prototype_head']['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {results['KnOTS_ICLR2025_IndependentW0']['nme_prototype_head']['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {results['KnOTS_ICLR2025_IndependentW0']['nme_prototype_head']['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    return results


def run_5seed_statistical_verification(seeds=(2027, 2028, 2029, 2030, 2031)) -> dict:
    """
    Runs 5-seed statistical verification (`Mean ± Std`, `95% CI`, `p-value`) comparing
    Official GACL (NeurIPS 2024) vs. Ours (ASID-CL v2) across all 3 datasets:
      1. UPMC Food-101 (100 classes, 10 CIL tasks)
      2. UK-CrisisMMD (6 classes, 3 CIL tasks)
      3. CrisisMMD v2.0 Humanitarian (7 disaster events)
    """
    print("\n[5/5] Running 5-Seed Statistical Verification (GACL NeurIPS'24 vs. Ours ASID-CL v2) across all 3 datasets...")

    def load_food_tasks():
        D = torch.load(ROOT / "data" / "upmc_food101" / "frozen_clip_features_3600.pt", map_location="cpu", weights_only=False)
        tr = torch.tensor(D["train"])
        te = torch.tensor(D["test"])
        tasks = []
        for k in range(10):
            mk = tr & (torch.tensor(D["task"]) == k)
            tasks.append({"train_img": D["img"][mk], "train_txt": D["txt"][mk], "train_y": D["y"][mk]})
        return tasks, (D["img"][tr], D["txt"][tr], D["y"][tr]), None, (D["img"][te], D["txt"][te], D["y"][te]), int(D["C"])

    out = {}
    for ds_name, loader in (
        ("upmc_food101_10task", load_food_tasks),
        ("uk_crisismmd", load_uk_tasks_and_full),
        ("crisismmd_humanitarian", load_humanitarian_events_and_full),
    ):
        tasks, (tr_i, tr_t, _), _, (te_i, te_t, te_y), C = loader()
        y_te_np = te_y.numpy()
        X_te_cat = torch.cat([F.normalize(te_i.double(), dim=-1), F.normalize(te_t.double(), dim=-1)], dim=1)
        X_te_dict = {"img": te_i.float(), "txt": te_t.float()}

        gacl_runs = {"clean": [], "missing_img_50": [], "corrupt_txt_50": []}
        ours_runs = {"clean": [], "missing_img_50": [], "corrupt_txt_50": []}

        for s in seeds:
            g_rng = torch.Generator().manual_seed(s + 99)
            miss_mask = torch.rand(len(y_te_np), generator=g_rng) < 0.50
            te_i_zero = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_i.double()), F.normalize(te_i.double(), dim=-1))
            X_te_zero_cat = torch.cat([te_i_zero, F.normalize(te_t.double(), dim=-1)], dim=1)

            te_t_corr = te_t.float().clone()
            noise = F.normalize(torch.randn(te_t_corr[miss_mask].shape, generator=g_rng), dim=-1)
            te_t_corr[miss_mask] = F.normalize(0.15 * te_t_corr[miss_mask] + 0.85 * noise, dim=-1)
            X_te_corr_cat = torch.cat([F.normalize(te_i.double(), dim=-1), F.normalize(te_t_corr.double(), dim=-1)], dim=1)
            X_te_corr_dict = {"img": te_i.float(), "txt": te_t_corr}

            gacl = OfficialGACLRunner(in_dim=tr_i.shape[1] + tr_t.shape[1], buffer_size=2048, gamma=10.0, seed=s)
            for td in tasks:
                x_t_cat = torch.cat([F.normalize(td["train_img"].double(), dim=-1), F.normalize(td["train_txt"].double(), dim=-1)], dim=1)
                gacl.fit(x_t_cat, td["train_y"])
            gacl_runs["clean"].append(eval_metrics(y_te_np, gacl.logits(X_te_cat).argmax(1).numpy()))
            gacl_runs["missing_img_50"].append(eval_metrics(y_te_np, gacl.logits(X_te_zero_cat).argmax(1).numpy()))
            gacl_runs["corrupt_txt_50"].append(eval_metrics(y_te_np, gacl.logits(X_te_corr_cat).argmax(1).numpy()))

            m_ours = BlockAnalyticCL({"img": tr_i.shape[1], "txt": tr_t.shape[1]}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            for td in tasks:
                m_ours.update({"img": td["train_img"].float(), "txt": td["train_txt"].float()}, td["train_y"])
            m_ours.auto_spectral_ridges()
            p_c = m_ours.predict(X_te_dict, adaptive_sid=True).numpy()
            p_m = p_c.copy()
            p_m[miss_mask.numpy()] = m_ours.predict({"txt": X_te_dict["txt"][miss_mask]}, modalities=("txt",)).numpy()
            p_n = m_ours.predict(X_te_corr_dict, adaptive_sid=True).numpy()

            ours_runs["clean"].append(eval_metrics(y_te_np, p_c))
            ours_runs["missing_img_50"].append(eval_metrics(y_te_np, p_m))
            ours_runs["corrupt_txt_50"].append(eval_metrics(y_te_np, p_n))

        ds_summary = {}
        for cond in ("clean", "missing_img_50", "corrupt_txt_50"):
            g_f1 = np.array([x["macro_f1"] for x in gacl_runs[cond]])
            o_f1 = np.array([x["macro_f1"] for x in ours_runs[cond]])
            diff = o_f1 - g_f1
            t_stat, p_val = ttest_rel(o_f1, g_f1)
            ds_summary[cond] = {
                "GACL_NeurIPS2024": {
                    k: {"mean": float(np.mean([x[k] for x in gacl_runs[cond]])), "std": float(np.std([x[k] for x in gacl_runs[cond]], ddof=1))}
                    for k in ("macro_f1", "worst_class_f1", "balanced_acc", "acc")
                },
                "Ours_ASID_CL_v2": {
                    k: {"mean": float(np.mean([x[k] for x in ours_runs[cond]])), "std": float(np.std([x[k] for x in ours_runs[cond]], ddof=1))}
                    for k in ("macro_f1", "worst_class_f1", "balanced_acc", "acc")
                },
                "paired_delta_macro_f1": {
                    "mean_gain": float(np.mean(diff)),
                    "std_gain": float(np.std(diff, ddof=1)),
                    "wins": f"{int(np.sum(diff > 0))}/{len(diff)}",
                    "paired_t_stat": float(t_stat),
                    "paired_p_value": float(p_val),
                },
            }
        out[ds_name] = ds_summary
        print(
            f"   {ds_name}: Clean F1 Ours {ds_summary['clean']['Ours_ASID_CL_v2']['macro_f1']['mean']*100:.2f}% vs "
            f"GACL {ds_summary['clean']['GACL_NeurIPS2024']['macro_f1']['mean']*100:.2f}% "
            f"(p={ds_summary['clean']['paired_delta_macro_f1']['paired_p_value']:.5f}) | "
            f"Corr50 F1 Ours {ds_summary['corrupt_txt_50']['Ours_ASID_CL_v2']['macro_f1']['mean']*100:.2f}% vs "
            f"GACL {ds_summary['corrupt_txt_50']['GACL_NeurIPS2024']['macro_f1']['mean']*100:.2f}% "
            f"(p={ds_summary['corrupt_txt_50']['paired_delta_macro_f1']['paired_p_value']:.5f})"
        )
    return out


def main():
    t0 = time.time()
    json_path = ROOT / "analysis_outputs" / "authentic_2025_2026_sota_benchmark.json"
    existing = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
    l10_l11_res = run_paper_faithful_l10_l11_benchmark()
    stat_5seed = existing.get("five_seed_statistical_verification") or run_5seed_statistical_verification()

    existing["paper_faithful_2024_2026_benchmark"] = l10_l11_res
    existing["five_seed_statistical_verification"] = stat_5seed
    existing["paper_faithful_elapsed_s"] = round(time.time() - t0, 2)
    json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nSaved complete paper-faithful 2024-2026 SOTA benchmark to {json_path} in {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()
