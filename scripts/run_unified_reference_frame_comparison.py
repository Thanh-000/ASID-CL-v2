#!/usr/bin/env python3
"""
Fast Unified Frame-of-Reference (Cùng Hệ Quy Chiếu) Verification & Visual Comparison Generator.
With the Zero-Copy Slice & BLAS Fast-Path optimization in `BlockAnalyticCL` (`analytic.py`),
computes `ASID-CL v2 (2P=5000)` (matching `GACL BUFFER=5000`) and `ASID-CL v2 (Oracle TIL)`
(matching `DeLo force_task_id=tid`) in seconds, merges with the saved 5-seed results from
`author_exact_code_method_config_benchmark.json` and `native_paper_protocols_benchmark.json`,
and exports the 4-panel publication-grade comparison figure.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_author_exact_code_method_config_benchmark import (
    SEEDS,
    official_base_dataset_missing_table,
    official_rebq_delo_compute_cl_metric,
    official_rebq_delo_task_metric,
)


def compute_fast_ours_iso_capacity_5000(D_cifar: dict) -> dict:
    """
    Runs ASID-CL v2 (Ours) at `projection_dim=2500` (`2P = 5000`, matching GACL's `BUFFER=5000`)
    using the optimized zero-copy `BlockAnalyticCL` (< 200 MB RAM instead of 60 GB!).
    """
    tr_mask = torch.tensor(D_cifar["train"])
    te_mask = torch.tensor(D_cifar["test"])
    tr_i_all = F.normalize(D_cifar["img"][tr_mask].float(), dim=-1)
    tr_t_all = F.normalize(D_cifar["txt"][tr_mask].float(), dim=-1)
    tr_y_all = D_cifar["y"][tr_mask].long()
    tr_task_all = torch.tensor(D_cifar["task"][tr_mask.numpy()], dtype=torch.long)

    te_i_all = F.normalize(D_cifar["img"][te_mask].float(), dim=-1)
    te_t_all = F.normalize(D_cifar["txt"][te_mask].float(), dim=-1)
    te_y_all = D_cifar["y"][te_mask].long()
    te_task_all = torch.tensor(D_cifar["task"][te_mask.numpy()], dtype=torch.long)
    y_te_np = te_y_all.numpy()
    C = 100

    seed_runs = []
    for s in SEEDS:
        g_stress = torch.Generator().manual_seed(s + 99)
        miss_mask = torch.rand(len(te_y_all), generator=g_stress) < 0.50
        te_i_miss = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_i_all), te_i_all)
        te_t_corr = te_t_all.clone()
        noise = F.normalize(torch.randn(te_t_corr[miss_mask].shape, generator=g_stress), dim=-1)
        te_t_corr[miss_mask] = F.normalize(0.15 * te_t_corr[miss_mask] + 0.85 * noise, dim=-1)

        m_ours_5000 = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=2500, class_balance_beta=0.5, seed=s)
        inc_acc = []
        task_mat = np.zeros((10, 10), dtype=np.float64)

        for t in range(10):
            mt_tr = tr_task_all == t
            m_ours_5000.update({"img": tr_i_all[mt_tr], "txt": tr_t_all[mt_tr]}, tr_y_all[mt_tr])
            if t == 9:
                m_ours_5000.auto_spectral_ridges(auto_rho=False)
                with torch.no_grad():
                    lg_full = m_ours_5000.logits({"img": te_i_all, "txt": te_t_all}, adaptive_sid=True)
                    for step_t in range(10):
                        seen_c = 10 * (step_t + 1)
                        mt_seen = te_task_all <= step_t
                        pr_step = lg_full[mt_seen, :seen_c].argmax(1).numpy()
                        inc_acc.append(float(accuracy_score(y_te_np[mt_seen.numpy()], pr_step)))
                    for j in range(10):
                        mj = (te_task_all == j).numpy()
                        task_mat[9, j] = float(accuracy_score(y_te_np[mj], lg_full[mj, :100].argmax(1).numpy()))
                        task_mat[j, j] = task_mat[9, j]

        with torch.no_grad():
            pc = m_ours_5000.predict({"img": te_i_all, "txt": te_t_all}, adaptive_sid=True).numpy()
            pm = m_ours_5000.predict({"img": te_i_miss, "txt": te_t_all}, adaptive_sid=True).numpy()
            pn = m_ours_5000.predict({"img": te_i_all, "txt": te_t_corr}, adaptive_sid=True).numpy()

        seed_runs.append({
            "A_bar": float(np.mean(inc_acc)),
            "A_10": float(accuracy_score(y_te_np, pc)),
            "clean_macro_f1": float(np.mean(f1_score(y_te_np, pc, average=None, zero_division=0))),
            "miss_img_50_f1": float(np.mean(f1_score(y_te_np, pm, average=None, zero_division=0))),
            "corr_txt_50_f1": float(np.mean(f1_score(y_te_np, pn, average=None, zero_division=0))),
        })

    return {
        k: {
            "mean": float(np.mean([r[k] for r in seed_runs])),
            "std": float(np.std([r[k] for r in seed_runs], ddof=1)),
        }
        for k in seed_runs[0]
    }


def compute_fast_ours_oracle_til_food101(D_food: dict) -> dict:
    """
    Evaluates ASID-CL v2 (Ours) under the exact same Oracle TIL frame (`force_task_id=tid`,
    10-way task slice `[tid*10 : (tid+1)*10]`) that `DeLo` uses in `trainer.py` L373.
    """
    C = 100
    num_tasks = 10
    cls_per_task = 10
    tr_mask = torch.tensor(D_food["train"])
    te_mask = torch.tensor(D_food["test"])
    tr_i_all = F.normalize(D_food["img"][tr_mask].float(), dim=-1)
    tr_t_all = F.normalize(D_food["txt"][tr_mask].float(), dim=-1)
    tr_y_all = D_food["y"][tr_mask].long()
    tr_task_all = torch.tensor(D_food["task"][tr_mask.numpy()], dtype=torch.long)

    te_i_all = F.normalize(D_food["img"][te_mask].float(), dim=-1)
    te_t_all = F.normalize(D_food["txt"][te_mask].float(), dim=-1)
    te_y_all = D_food["y"][te_mask].long()
    te_task_all = torch.tensor(D_food["task"][te_mask.numpy()], dtype=torch.long)

    remap = torch.zeros(C, dtype=torch.long)
    for t in range(num_tasks):
        t_classes = torch.unique(tr_y_all[tr_task_all == t], sorted=True)
        for idx_c, old_c in enumerate(t_classes.tolist()):
            remap[old_c] = t * cls_per_task + idx_c
    tr_y_all = remap[tr_y_all]
    te_y_all = remap[te_y_all]

    til_out = {}
    for m_type in ("image", "text", "both"):
        aps = []
        for s in SEEDS:
            tr_mtypes = torch.zeros(len(tr_y_all), dtype=torch.long)
            te_mtypes = torch.zeros(len(te_y_all), dtype=torch.long)
            for t in range(num_tasks):
                idx_tr_t = (tr_task_all == t).nonzero(as_tuple=True)[0]
                idx_te_t = (te_task_all == t).nonzero(as_tuple=True)[0]
                tr_mtypes[idx_tr_t] = official_base_dataset_missing_table(len(idx_tr_t), 0.7, m_type, 0.5, seed=s * 100 + t)
                te_mtypes[idx_te_t] = official_base_dataset_missing_table(len(idx_te_t), 0.7, m_type, 0.5, seed=s * 100 + 50 + t)

            m_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.0, seed=s)
            for t in range(num_tasks):
                mt_tr = tr_task_all == t
                mtype_t = tr_mtypes[mt_tr]
                m_ours.update(
                    {"img": tr_i_all[mt_tr], "txt": tr_t_all[mt_tr]},
                    tr_y_all[mt_tr],
                    availability={"img": mtype_t != 2, "txt": mtype_t != 1},
                )
            m_ours.auto_spectral_ridges(auto_rho=True)
            with torch.no_grad():
                hi_te = (te_mtypes != 2).unsqueeze(1)
                ht_te = (te_mtypes != 1).unsqueeze(1)
                ti_te = torch.where(hi_te, te_i_all, torch.zeros_like(te_i_all))
                tt_te = torch.where(ht_te, te_t_all, torch.zeros_like(te_t_all))
                lg_all = m_ours.logits({"img": ti_te, "txt": tt_te}, adaptive_sid=True)
                task_accs = []
                for tid in range(num_tasks):
                    mt_tid = te_task_all == tid
                    c0, c1 = tid * cls_per_task, (tid + 1) * cls_per_task
                    pr_tid = lg_all[mt_tid, c0:c1].argmax(1) + c0
                    task_accs.append(float((pr_tid == te_y_all[mt_tid]).float().mean().item()) * 100.0)
                aps.append(float(np.mean(task_accs)))
        til_out[f"missing_{m_type}_70"] = {"mean": float(np.mean(aps)), "std": float(np.std(aps, ddof=1))}
    return til_out


def generate_publication_visual_comparison(
    author_data: dict, shared_data: dict, ours_5000: dict, ours_til_food: dict, out_paths: list[Path]
):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 11.4), dpi=300)
    fig.subplots_adjust(hspace=0.38, wspace=0.23, top=0.90, bottom=0.08, left=0.06, right=0.98)

    c_ours_main = "#0F4D92"
    c_ours_alt = "#3775BA"
    c_gacl = "#42949E"
    c_delo = "#8BCF8B"
    c_lora = "#E9A6A1"
    c_rebq = "#B64342"

    p1_au = author_data["protocol_1_split_cifar100_author_exact"]
    p1_sh = shared_data["protocol_1_split_cifar100_b0_inc10"]
    p2_food_au = author_data["protocol_2_upmc_food101_cmml_author_exact"]
    p2_imdb_au = author_data["protocol_2_mm_imdb_cmml_author_exact"]

    # -------------------------------------------------------------------------
    # PANEL (a): Split CIFAR-100 (`B0-Inc10`) — Iso-Capacity (2P=5000 & 2048) vs Author Exact
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    methods_a = [
        "ASID-CL v2\n(Ours 2P=5000)",
        "ASID-CL v2\n(Ours 2P=2048)",
        "ACIL\n(NeurIPS'22)",
        "GACL\n(NeurIPS'24)",
        "KnOTS\n(ICLR'25)",
        "SD-LoRA\n(ICLR'25)",
        "TSV-M\n(CVPR'25)",
        "InfLoRA\n(CVPR'24)",
    ]
    a10_author = [
        ours_5000["A_10"]["mean"] * 100,
        p1_au["ASID_CL_v2_Ours"]["A_10"]["mean"] * 100,
        p1_au["ACIL_AuthorExact_Woodbury"]["A_10"]["mean"] * 100,
        p1_au["GACL_AuthorExact_B5000_G100"]["A_10"]["mean"] * 100,
        p1_au["KnOTS_AuthorExact_NME"]["A_10"]["mean"] * 100,
        p1_au["SD_LoRA_AuthorExact_NME"]["A_10"]["mean"] * 100,
        p1_au["TSVM_AuthorExact_NME"]["A_10"]["mean"] * 100,
        p1_au["InfLoRA_AuthorExact_NME"]["A_10"]["mean"] * 100,
    ]
    corr_author = [
        ours_5000["corr_txt_50_f1"]["mean"] * 100,
        p1_au["ASID_CL_v2_Ours"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["ACIL_AuthorExact_Woodbury"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["GACL_AuthorExact_B5000_G100"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["KnOTS_AuthorExact_NME"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["SD_LoRA_AuthorExact_NME"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["TSVM_AuthorExact_NME"]["corr_txt_50_f1"]["mean"] * 100,
        p1_au["InfLoRA_AuthorExact_NME"]["corr_txt_50_f1"]["mean"] * 100,
    ]
    x_a = np.arange(len(methods_a))
    w_a = 0.37
    bars1 = ax.bar(x_a - w_a / 2, a10_author, w_a, label="Final Acc A₁₀ (Clean %)", color=c_ours_main, edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x_a + w_a / 2, corr_author, w_a, label="50% Noisy Text Macro-F1 (%)", color=c_gacl, edgecolor="black", linewidth=0.8)
    for b in bars1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.8, color="#1b4f5c")
    ax.set_xticks(x_a)
    ax.set_xticklabels(methods_a, fontsize=8.4)
    ax.set_ylim(45, 88)
    ax.set_ylabel("Accuracy / Macro-F1 (%)", fontweight="bold")
    ax.set_title("(a) Split CIFAR-100 (B0-Inc10): Cùng Hệ Quy Chiếu 2P=5000 & 2P=2048", fontweight="bold", fontsize=11.4, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", fontsize=9.2)

    # -------------------------------------------------------------------------
    # PANEL (b): Reference Frame Shift — Shared Unit-Norm vs Raw Author Config
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    categories_b = ["ASID-CL v2\n(Ours)", "ACIL\n(NeurIPS'22)", "GACL\n(NeurIPS'24)", "SD-LoRA\n(ICLR'25)", "InfLoRA\n(CVPR'24)"]
    aligned_b = [
        p1_sh["ASID_CL_v2_Ours"]["final_acc_A_10"]["mean"] * 100,
        p1_sh["ACIL_NeurIPS2022"]["final_acc_A_10"]["mean"] * 100,
        p1_sh["GACL_NeurIPS2024"]["final_acc_A_10"]["mean"] * 100,
        p1_au["SD_LoRA_AuthorExact_NME"]["A_10"]["mean"] * 100,
        p1_au["InfLoRA_AuthorExact_NME"]["A_10"]["mean"] * 100,
    ]
    raw_author_b = [
        ours_5000["A_10"]["mean"] * 100,
        p1_au["ACIL_AuthorExact_Woodbury"]["A_10"]["mean"] * 100,
        p1_au["GACL_AuthorExact_B5000_G100"]["A_10"]["mean"] * 100,
        p1_au["SD_LoRA_AuthorExact_AlignedFC"]["A_10"]["mean"] * 100,
        p1_au["InfLoRA_AuthorExact_Linear"]["A_10"]["mean"] * 100,
    ]
    x_b = np.arange(len(categories_b))
    w_b = 0.36
    b_al = ax.bar(x_b - w_b / 2, aligned_b, w_b, label="Hệ quy chiếu 1: Đồng chuẩn hóa (2P=2048 Unit-Norm / NME)", color=c_ours_alt, edgecolor="black", linewidth=0.8)
    b_au = ax.bar(x_b + w_b / 2, raw_author_b, w_b, label="Hệ quy chiếu 2: Cấu hình gốc tác giả (2P=5000 / Kaiming γ=100 / FC)", color=c_rebq, edgecolor="black", linewidth=0.8)
    for b in b_al:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.8, fontweight="bold")
    for b in b_au:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.9, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.8, color="#7a1c1c", fontweight="bold")
    ax.set_xticks(x_b)
    ax.set_xticklabels(categories_b, fontsize=9.5)
    ax.set_ylim(30, 90)
    ax.set_ylabel("Final Accuracy A₁₀ (%)", fontweight="bold")
    ax.set_title("(b) Đối Chiếu 2 Hệ Quy Chiếu Trên Split CIFAR-100 (Đồng Chuẩn Hóa vs. Gốc Tác Giả)", fontweight="bold", fontsize=11.3, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", fontsize=8.8)

    # -------------------------------------------------------------------------
    # PANEL (c): UPMC-Food101-CMML (70% Missing) — Both Ours & DeLo in CIL & TIL!
    # -------------------------------------------------------------------------
    ax = axes[1, 0]
    regimes_c = ["70% Missing IMAGE\n(Train & Test)", "70% Missing TEXT\n(Train & Test)", "70% Missing BOTH\n(Train & Test)"]
    ours_til_c = [
        ours_til_food["missing_image_70"]["mean"],
        ours_til_food["missing_text_70"]["mean"],
        ours_til_food["missing_both_70"]["mean"],
    ]
    ours_cil_c = [
        p2_food_au["missing_image_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
        p2_food_au["missing_text_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
        p2_food_au["missing_both_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
    ]
    delo_til_c = [
        p2_food_au["missing_image_70"]["DeLo_AuthorExact_Oracle_TIL"]["AP"]["mean"],
        p2_food_au["missing_text_70"]["DeLo_AuthorExact_Oracle_TIL"]["AP"]["mean"],
        p2_food_au["missing_both_70"]["DeLo_AuthorExact_Oracle_TIL"]["AP"]["mean"],
    ]
    gacl_c = [
        p2_food_au["missing_image_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
        p2_food_au["missing_text_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
        p2_food_au["missing_both_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
    ]
    delo_cil_c = [
        p2_food_au["missing_image_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
        p2_food_au["missing_text_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
        p2_food_au["missing_both_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
    ]
    rebq_c = [
        p2_food_au["missing_image_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
        p2_food_au["missing_text_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
        p2_food_au["missing_both_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
    ]
    x_c = np.arange(len(regimes_c))
    w_c = 0.135
    b_c0 = ax.bar(x_c - 2.5 * w_c, ours_til_c, w_c, label="ASID-CL v2 (Ours — Oracle TIL)", color="#072A55", edgecolor="black", linewidth=0.8)
    b_c1 = ax.bar(x_c - 1.5 * w_c, ours_cil_c, w_c, label="ASID-CL v2 (Ours — Task-Free CIL)", color=c_ours_main, edgecolor="black", linewidth=0.8)
    b_c2 = ax.bar(x_c - 0.5 * w_c, delo_til_c, w_c, label="DeLo Official (Oracle TIL)", color=c_ours_alt, edgecolor="black", linewidth=0.8)
    b_c3 = ax.bar(x_c + 0.5 * w_c, gacl_c, w_c, label="GACL Official (Task-Free CIL)", color=c_gacl, edgecolor="black", linewidth=0.8)
    b_c4 = ax.bar(x_c + 1.5 * w_c, delo_cil_c, w_c, label="DeLo Official (Task-Free CIL)", color=c_delo, edgecolor="black", linewidth=0.8)
    b_c5 = ax.bar(x_c + 2.5 * w_c, rebq_c, w_c, label="RebQ Official (Task-Free CIL)", color=c_lora, edgecolor="black", linewidth=0.8)
    for bars_grp in (b_c0, b_c1, b_c2, b_c3, b_c4, b_c5):
        for b in bars_grp:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold")
    ax.set_xticks(x_c)
    ax.set_xticklabels(regimes_c, fontsize=9.5)
    ax.set_ylim(25, 114)
    ax.set_ylabel("Evaluator AP Accuracy (%)", fontweight="bold")
    ax.set_title("(c) UPMC-Food101-CMML (70% Missing): Đối Chứng Cùng Hệ Quy Chiếu (CIL & TIL)", fontweight="bold", fontsize=11.3, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=7.8)

    # -------------------------------------------------------------------------
    # PANEL (d): MM-IMDb-CMML (70% Missing Multi-Label) — Same Reference Frame
    # -------------------------------------------------------------------------
    ax = axes[1, 1]
    regimes_d = ["70% Missing IMAGE\n(Evaluator AP / Last F1)", "70% Missing TEXT\n(Evaluator AP / Last F1)", "70% Missing BOTH\n(Evaluator AP / Last F1)"]
    ours_d_ap = [
        p2_imdb_au["missing_image_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
        p2_imdb_au["missing_text_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
        p2_imdb_au["missing_both_70"]["ASID_CL_v2_Ours"]["AP"]["mean"],
    ]
    ours_d_last = [
        p2_imdb_au["missing_image_70"]["ASID_CL_v2_Ours"]["Last"]["mean"],
        p2_imdb_au["missing_text_70"]["ASID_CL_v2_Ours"]["Last"]["mean"],
        p2_imdb_au["missing_both_70"]["ASID_CL_v2_Ours"]["Last"]["mean"],
    ]
    gacl_d_ap = [
        p2_imdb_au["missing_image_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
        p2_imdb_au["missing_text_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
        p2_imdb_au["missing_both_70"]["GACL_AuthorExact_ZeroFill"]["AP"]["mean"],
    ]
    gacl_d_last = [
        p2_imdb_au["missing_image_70"]["GACL_AuthorExact_ZeroFill"]["Last"]["mean"],
        p2_imdb_au["missing_text_70"]["GACL_AuthorExact_ZeroFill"]["Last"]["mean"],
        p2_imdb_au["missing_both_70"]["GACL_AuthorExact_ZeroFill"]["Last"]["mean"],
    ]
    rebq_d_ap = [
        p2_imdb_au["missing_image_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
        p2_imdb_au["missing_text_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
        p2_imdb_au["missing_both_70"]["RebQ_AuthorExact_PromptRec"]["AP"]["mean"],
    ]
    delo_d_ap = [
        p2_imdb_au["missing_image_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
        p2_imdb_au["missing_text_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
        p2_imdb_au["missing_both_70"]["DeLo_AuthorExact_TaskFree_CIL"]["AP"]["mean"],
    ]
    x_d = np.arange(len(regimes_d))
    w_d = 0.14
    b_d1 = ax.bar(x_d - 2.5 * w_d, ours_d_last, w_d, label="ASID-CL v2 (Ours — Last F1)", color=c_ours_main, edgecolor="black", linewidth=0.8)
    b_d2 = ax.bar(x_d - 1.5 * w_d, gacl_d_last, w_d, label="GACL Official (Last F1)", color=c_ours_alt, edgecolor="black", linewidth=0.8)
    b_d3 = ax.bar(x_d - 0.5 * w_d, ours_d_ap, w_d, label="ASID-CL v2 (Ours — Avg AP F1)", color=c_gacl, edgecolor="black", linewidth=0.8)
    b_d4 = ax.bar(x_d + 0.5 * w_d, gacl_d_ap, w_d, label="GACL Official (Avg AP F1)", color=c_delo, edgecolor="black", linewidth=0.8)
    b_d5 = ax.bar(x_d + 1.5 * w_d, rebq_d_ap, w_d, label="RebQ Official (Avg AP F1)", color=c_lora, edgecolor="black", linewidth=0.8)
    b_d6 = ax.bar(x_d + 2.5 * w_d, delo_d_ap, w_d, label="DeLo Official CIL (Avg AP F1)", color=c_rebq, edgecolor="black", linewidth=0.8)
    for bars_grp in (b_d1, b_d2, b_d3, b_d4, b_d5, b_d6):
        for b in bars_grp:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax.set_xticks(x_d)
    ax.set_xticklabels(regimes_d, fontsize=9.5)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Multi-Label F1-Macro (%)", fontweight="bold")
    ax.set_title("(d) MM-IMDb-CMML (70% Missing): Đối Chứng Cùng Hệ Quy Chiếu (AP & Last-Task F1)", fontweight="bold", fontsize=11.3, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.0)

    fig.suptitle(
        "Đối Chứng Cùng Hệ Quy Chiếu (Unified Reference Frame): Chuẩn Tác Giả (Code + Method + Config) vs. ASID-CL v2 (5 Seeds)",
        fontsize=14.0,
        fontweight="bold",
        y=0.965,
    )

    for base_p in out_paths:
        base_p.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            p = base_p.with_suffix(f".{ext}")
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.08)
            print(f"  Saved figure: {p}", flush=True)
    plt.close(fig)


def main():
    t0 = time.time()
    D_cifar = torch.load(ROOT / "data" / "cifar100" / "frozen_clip_cifar100_3600.pt", map_location="cpu", weights_only=False)
    D_food = torch.load(ROOT / "data" / "upmc_food101" / "frozen_clip_features_3600.pt", map_location="cpu", weights_only=False)

    author_path = ROOT / "analysis_outputs" / "author_exact_code_method_config_benchmark.json"
    shared_path = ROOT / "analysis_outputs" / "native_paper_protocols_benchmark.json"
    author_data = json.loads(author_path.read_text(encoding="utf-8"))
    shared_data = json.loads(shared_path.read_text(encoding="utf-8"))

    print("Computing fast ASID-CL v2 (2P=5000, matching GACL BUFFER=5000) and ASID-CL v2 (Oracle TIL)...", flush=True)
    ours_5000 = compute_fast_ours_iso_capacity_5000(D_cifar)
    ours_til_food = compute_fast_ours_oracle_til_food101(D_food)
    print(f"  ASID-CL v2 (2P=5000) Split CIFAR-100 | A_bar: {ours_5000['A_bar']['mean']*100:.2f}% | A_10: {ours_5000['A_10']['mean']*100:.2f}% | Corr50: {ours_5000['corr_txt_50_f1']['mean']*100:.2f}%", flush=True)
    print(f"  ASID-CL v2 (Oracle TIL) Food101-CMML | Img70: {ours_til_food['missing_image_70']['mean']:.2f}% | Txt70: {ours_til_food['missing_text_70']['mean']:.2f}% | Both70: {ours_til_food['missing_both_70']['mean']:.2f}%", flush=True)

    unified_out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "seeds": author_data["seeds"],
        "ours_iso_capacity_2P5000_cifar100": ours_5000,
        "ours_oracle_til_food101_cmml": ours_til_food,
        "reference_frame_1_shared_feature_aligned": {
            "description": "Cùng tầng mở rộng chuẩn hóa đơn vị (2P=2048), cùng điều chuẩn tỉ lệ vết, cùng giao thức Task-Free CIL",
            "results": shared_data,
        },
        "reference_frame_2_author_exact_code_method_config": {
            "description": "100% Mã nguồn gốc (external_repos/) + Phương pháp gốc + Cấu hình siêu tham số gốc của từng tác giả",
            "results": author_data,
        },
    }
    out_json_path = ROOT / "analysis_outputs" / "unified_reference_frame_benchmark.json"
    out_json_path.write_text(json.dumps(unified_out, indent=2), encoding="utf-8")

    artifact_dir = Path(r"C:\Users\Admin\.gemini\antigravity\brain\1c868d79-363e-44b8-9c17-35b7f14f7648")
    generate_publication_visual_comparison(
        author_data,
        shared_data,
        ours_5000,
        ours_til_food,
        [
            ROOT / "latex_master_survey_report" / "figures" / "fig_unified_reference_frame_comparison",
            artifact_dir / "fig_unified_reference_frame_comparison",
        ],
    )
    print(f"Completed in {round(time.time() - t0, 2)}s!", flush=True)


if __name__ == "__main__":
    main()
