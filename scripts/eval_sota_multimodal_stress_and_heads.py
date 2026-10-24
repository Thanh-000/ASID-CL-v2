#!/usr/bin/env python3
"""
Evaluates Official TSV-M (CVPR 2025), KnOTS (ICLR 2025), GACL (NeurIPS 2024), and Ours (ASID-CL v2)
on the 10-task 100-class UPMC Food-101 L10-L11 Transformer benchmark across:
  1. Preserved Task-Specific Linear Heads (`W_task_preserved`), NME Prototype Heads (`W_nme`), and Analytic Heads
  2. Clean Test Set, 50% Missing Image Stress Test, and 50% Corrupted Text Stress Test
Updates `analysis_outputs/authentic_2025_2026_sota_benchmark.json` with the complete breakdown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_authentic_2025_2026_sota_benchmark import (
    OfficialGACLRunner,
    apply_official_knots_merging,
    apply_official_tsvm_merging,
    eval_metrics,
)
from run_food101_local import (
    ALPHA,
    BS,
    EPOCHS,
    LR,
    R,
    build_or_load_prefix_cache,
    build_top_model,
    encode_from_prefix,
    prepare_food101_dataframe,
)

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


def main():
    t0 = time.time()
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
    C = int(D["C"])

    # Deterministic 50% missing image & 50% corrupted text masks on the 1,000 test samples
    g_stress = torch.Generator().manual_seed(2026 + 99)
    miss_mask = torch.rand(len(y_te_np), generator=g_stress) < 0.50

    def make_corrupted_txt(te_t_clean: torch.Tensor) -> torch.Tensor:
        g_c = torch.Generator().manual_seed(2026 + 199)
        te_t_corr = te_t_clean.clone()
        noise = F.normalize(torch.randn(te_t_corr[miss_mask].shape, generator=g_c), dim=-1)
        te_t_corr[miss_mask] = F.normalize(0.15 * te_t_corr[miss_mask] + 0.85 * noise, dim=-1)
        return te_t_corr

    # ------------------------------------------------------------------
    # 1. Official GACL (NeurIPS 2024) on Clean / 50% Missing Image / 50% Corrupted Text
    # ------------------------------------------------------------------
    print("[1/4] Evaluating Official GACL (NeurIPS 2024) on Clean, 50% Missing Image, 50% Corrupted Text...")
    m_frz, _, _ = build_top_model(lora=False)
    tr_I_frz, tr_T_frz = extract_features(m_frz, D, tr_idx)
    te_I_frz, te_T_frz = extract_features(m_frz, D, te_idx)

    gacl = OfficialGACLRunner(in_dim=1024, buffer_size=2048, gamma=10.0, seed=2026)
    for k in range(10):
        mk = D["task"][tr_idx.numpy()] == k
        gacl.fit(torch.cat([tr_I_frz[mk], tr_T_frz[mk]], dim=1), y_tr[mk])

    te_I_zero_frz = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_I_frz), te_I_frz)
    te_T_corr_frz = make_corrupted_txt(te_T_frz)

    gacl_stress = {
        "clean": eval_metrics(y_te_np, gacl.logits(torch.cat([te_I_frz, te_T_frz], dim=1)).argmax(1).numpy()),
        "missing_img_50": eval_metrics(y_te_np, gacl.logits(torch.cat([te_I_zero_frz, te_T_frz], dim=1)).argmax(1).numpy()),
        "corrupt_txt_50": eval_metrics(y_te_np, gacl.logits(torch.cat([te_I_frz, te_T_corr_frz], dim=1)).argmax(1).numpy()),
    }
    print(
        f"   GACL (NeurIPS'24) -> Clean F1: {gacl_stress['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {gacl_stress['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {gacl_stress['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    # ------------------------------------------------------------------
    # 2. Train 10-Task LoRA Trajectory + Save Preserved Task Heads & Per-Task Deltas
    # ------------------------------------------------------------------
    print("[2/4] Running 10-Task Null-Space LoRA Trajectory + Ours (ASID-CL v2)...")
    m_ns, states_ns, loras_ns = build_top_model(lora=True)
    head_ns = nn.Linear(1024, C, bias=False)
    preserved_head_weight = torch.zeros_like(head_ns.weight.data)
    analytic_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.5, seed=2026)
    per_task_lora_deltas: list[dict[str, torch.Tensor]] = []

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        yk = D["y"][tk_idx]
        k_classes = torch.unique(yk)

        if k > 0:
            old_I, old_T = extract_features(m_ns, D, tk_idx)
            old_inputs_k = {"img": old_I, "txt": old_T}

        # Train task k with task-masked head preservation for linear-head evaluation alongside fullLoRA
        for lo in loras_ns:
            lo.new_session()
        params = [p for lo in loras_ns for p in (lo.A, lo.B)] + list(head_ns.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
        g = torch.Generator().manual_seed(2026 + k)
        m_ns.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_ns, D, sb)
                loss = F.cross_entropy(head_ns(torch.cat([i, t], 1)) * 10.0, D["y"][sb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        m_ns.eval()

        # Save task k's freshly trained classifier rows before subsequent tasks overwrite them
        with torch.no_grad():
            preserved_head_weight[k_classes] = F.normalize(head_ns.weight.data[k_classes].clone(), dim=-1)

        # Save task k's 2D LoRA weight delta dW_k = (ALPHA / R) * B_k @ A_eff_k for TSV-M & KnOTS
        delta_k = {}
        for idx_l, lo in enumerate(loras_ns):
            A_eff = (lo.A.detach() @ lo.P) if lo.P is not None else lo.A.detach()
            dW = (lo.B.detach() @ A_eff) * (ALPHA / R)
            delta_k[f"layer_{idx_l}"] = dW.cpu().clone()
        per_task_lora_deltas.append(delta_k)

        # Update Null-Space projection P_perp at tau = 0.99
        for st in states_ns:
            st.recording = True
        with torch.no_grad():
            for b in range(0, len(tk_idx), BS):
                encode_from_prefix(m_ns, D, tk_idx[b:b + BS])
        for st in states_ns:
            st.recording = False
            st.update_projection(0.99)

        new_I, new_T = extract_features(m_ns, D, tk_idx)
        new_inputs_k = {"img": new_I, "txt": new_T}
        if k > 0:
            analytic_ours.transport_gram(old_inputs_k, new_inputs_k, max_drift=0.35, adaptive_drift_guard=True)
        analytic_ours.update(new_inputs_k, yk)
        analytic_ours.auto_spectral_ridges()

    te_I_ours, te_T_ours = extract_features(m_ns, D, te_idx)
    te_T_ours_corr = make_corrupted_txt(te_T_ours)

    p_ours_clean = analytic_ours.predict({"img": te_I_ours, "txt": te_T_ours}, adaptive_sid=True).numpy()
    p_ours_miss = p_ours_clean.copy()
    p_ours_miss[miss_mask.numpy()] = analytic_ours.predict({"txt": te_T_ours[miss_mask]}, modalities=("txt",)).numpy()
    p_ours_corr = analytic_ours.predict({"img": te_I_ours, "txt": te_T_ours_corr}, adaptive_sid=True).numpy()

    ours_stress = {
        "clean": eval_metrics(y_te_np, p_ours_clean),
        "missing_img_50": eval_metrics(y_te_np, p_ours_miss),
        "corrupt_txt_50": eval_metrics(y_te_np, p_ours_corr),
    }
    print(
        f"   Ours (ASID-CL v2) -> Clean F1: {ours_stress['clean']['macro_f1']*100:.2f}%, "
        f"Miss50 F1: {ours_stress['missing_img_50']['macro_f1']*100:.2f}%, "
        f"Corr50 F1: {ours_stress['corrupt_txt_50']['macro_f1']*100:.2f}%"
    )

    # Helper to evaluate any merged backbone (TSV-M or KnOTS) under:
    #   (a) Preserved Per-Task Linear Head (`preserved_linear_head`)
    #   (b) NME Class-Prototype Head (`nme_prototype_head`, standard in SD-LoRA / CIL)
    def eval_merged_backbone(merged_deltas: dict[str, torch.Tensor], name: str) -> dict:
        m_mrg, _, _ = build_top_model(lora=False)
        idx_l = 0
        for mod_list in (m_mrg.v_layers, m_mrg.t_layers):
            for blk in mod_list:
                attn = blk.self_attn
                for attr in ("q_proj", "v_proj"):
                    lin = getattr(attn, attr)
                    lin.weight.data.add_(merged_deltas[f"layer_{idx_l}"].to(lin.weight.dtype))
                    idx_l += 1

        tr_I_m, tr_T_m = extract_features(m_mrg, D, tr_idx)
        te_I_m, te_T_m = extract_features(m_mrg, D, te_idx)
        te_I_m_zero = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_I_m), te_I_m)
        te_T_m_corr = make_corrupted_txt(te_T_m)

        # (a) Preserved per-task linear head
        X_te_clean = torch.cat([te_I_m, te_T_m], dim=1)
        X_te_miss = torch.cat([te_I_m_zero, te_T_m], dim=1)
        X_te_corr = torch.cat([te_I_m, te_T_m_corr], dim=1)
        p_lin_clean = (X_te_clean @ preserved_head_weight.T).argmax(1).numpy()
        p_lin_miss = (X_te_miss @ preserved_head_weight.T).argmax(1).numpy()
        p_lin_corr = (X_te_corr @ preserved_head_weight.T).argmax(1).numpy()

        # (b) NME Class-Prototype Head (computed task-by-task without replay, just class mean vectors)
        X_tr_clean = F.normalize(torch.cat([tr_I_m, tr_T_m], dim=1), dim=-1)
        prototypes = torch.zeros(C, 1024)
        for c in range(C):
            mc = y_tr == c
            prototypes[c] = F.normalize(X_tr_clean[mc].mean(dim=0), dim=-1)

        p_nme_clean = (F.normalize(X_te_clean, dim=-1) @ prototypes.T).argmax(1).numpy()
        p_nme_miss = (F.normalize(X_te_miss, dim=-1) @ prototypes.T).argmax(1).numpy()
        p_nme_corr = (F.normalize(X_te_corr, dim=-1) @ prototypes.T).argmax(1).numpy()

        res = {
            "preserved_task_linear_head": {
                "clean": eval_metrics(y_te_np, p_lin_clean),
                "missing_img_50": eval_metrics(y_te_np, p_lin_miss),
                "corrupt_txt_50": eval_metrics(y_te_np, p_lin_corr),
            },
            "nme_prototype_head": {
                "clean": eval_metrics(y_te_np, p_nme_clean),
                "missing_img_50": eval_metrics(y_te_np, p_nme_miss),
                "corrupt_txt_50": eval_metrics(y_te_np, p_nme_corr),
            },
        }
        print(
            f"   {name} (Preserved Linear) -> Clean ACC: {res['preserved_task_linear_head']['clean']['acc']*100:.2f}%, "
            f"F1: {res['preserved_task_linear_head']['clean']['macro_f1']*100:.2f}%"
        )
        print(
            f"   {name} (NME Prototype)    -> Clean ACC: {res['nme_prototype_head']['clean']['acc']*100:.2f}%, "
            f"F1: {res['nme_prototype_head']['clean']['macro_f1']*100:.2f}%, "
            f"Miss50 F1: {res['nme_prototype_head']['nme_prototype_head' if False else 'missing_img_50']['macro_f1']*100:.2f}%, "
            f"Corr50 F1: {res['nme_prototype_head']['corrupt_txt_50']['macro_f1']*100:.2f}%"
        )
        return res

    print("[3/4] Evaluating Official TSV-M (CVPR 2025) on Merged L10-L11 LoRA Backbone...")
    tsvm_deltas = apply_official_tsvm_merging(per_task_lora_deltas)
    tsvm_res = eval_merged_backbone(tsvm_deltas, "TSV-M (CVPR'25)")

    print("[4/4] Evaluating Official KnOTS (ICLR 2025) on Merged L10-L11 LoRA Backbone...")
    knots_deltas = apply_official_knots_merging(per_task_lora_deltas, topK=30.0)
    knots_res = eval_merged_backbone(knots_deltas, "KnOTS (ICLR'25)")

    # Update `analysis_outputs/authentic_2025_2026_sota_benchmark.json`
    json_path = ROOT / "analysis_outputs" / "authentic_2025_2026_sota_benchmark.json"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["upmc_food101_multimodal_stress_and_head_analysis"] = {
        "GACL_NeurIPS2024": gacl_stress,
        "TSVM_CVPR2025": tsvm_res,
        "KnOTS_ICLR2025": knots_res,
        "Ours_ASID_CL_v2": ours_stress,
        "delo_aaai2026_repo_audit": {
            "bug_1_silent_bypass": "external_repos/DeLo/src/model/rebq_mixlora_vilt.py lines 101-103 & 350: hidden_states.query_signals attribute is stripped by ViltLayer.layernorm_before(), causing MixLoRAWrapper.forward() to return frozen_output without calling MixLoRALayer.",
            "bug_2_4d_einsum_crash": "external_repos/DeLo/src/model/mixlora_layer.py lines 124-129 & 257-260: lora_A_pools initialized as (E, r, in_features) instead of rank-1 expert pool (E, in_features), causing active_A_pool[g_A_indices] to be 4D (B, r, r, in_features) and fail torch.einsum('bri,rie->be', ...).",
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nUpdated {json_path} in {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()
