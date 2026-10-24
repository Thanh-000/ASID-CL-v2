#!/usr/bin/env python3
"""Empirical Continual Merging Study: OPCM on CrisisMMD v2 (Global Disaster Dataset).

Executes a complete, rigorous empirical study:
1. Dataset: QCRI/CrisisMMD v2.0 (3,526 Damage Severity samples across 7 global disasters).
2. Protocol 1: 7-Disaster Stream Continual Merging (Domain-Incremental Stream, T = 7).
   - Sequence: Harvey -> Maria -> Irma -> California Wildfires -> Mexico EQ -> Iraq EQ -> Sri Lanka Floods.
   - Evaluates:
     * Raw Addition
     * SWA / Online Running Mean
     * Continual Task Arithmetic (gamma = 0.3)
     * TIES-Merging
     * OPCM Buggy (Author code: diag().fill_(0))
     * OPCM Corrected (.diagonal().zero_())
     * OPCM Literal Eq. (9)
     * OPCM Scale-Only Ablation
     * OPCM Projection-Only Ablation
   - Tracks: R[t, i] (7x7), Final ACC, BWT, Forgetting (FTG), Orthogonality Inner Product <Delta W_old, Delta W_proj>, Cumulative Norm ||Delta W_merged||.
3. Protocol 2: Multimodal Gating & Bounded AC-Fusion on CrisisMMD v2:
   - Vision-Only vs Text-Only vs Fixed Multimodal (0.5) vs Unbounded AC-Fusion vs Bounded Gating (Leader-Follower).
4. Protocol 3: Per-Class Diagnostics & Confusion Matrix Analysis:
   - Precision, Recall, F1 for 'little_or_no_damage', 'mild_damage', 'severe_damage'.
"""

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Ensure utf-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE_DIR = ROOT / "analysis_outputs" / "mm_hope_phase1_cache"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_crisismmd_v2_damage_data():
    """Loads CrisisMMD v2 damage embeddings and annotations."""
    from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations, regroup_crisismmd_by_tweet

    # 1. Load cached embeddings
    cache_path = CACHE_DIR / "clip_embeddings_c75d70edce7d3e8e.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found at {cache_path}")
    data = np.load(cache_path)
    text_embs = torch.from_numpy(data["text_embeddings"]).float()
    img_embs = torch.from_numpy(data["image_embeddings"]).float()
    cached_row_ids = data["row_ids"]

    # 2. Load dataset annotations
    ds_damage = load_crisismmd_v2_annotations(
        ROOT / "external_data" / "crisismmd_v2",
        task="damage",
        agreed_labels=False,
        media_root=ROOT / "external_data" / "crisismmd_v2" / "media",
        require_images=False,
    )
    # Regroup by tweet as official E10 benchmark does
    ds_damage = regroup_crisismmd_by_tweet(ds_damage)
    frame = ds_damage.frame

    # Create mapping from row_id to index
    row_id_to_idx = {rid: i for i, rid in enumerate(cached_row_ids)}
    valid_indices = [row_id_to_idx[rid] for rid in frame["row_id"] if rid in row_id_to_idx]

    img_feats = img_embs[valid_indices]
    text_feats = text_embs[valid_indices]
    labels = torch.from_numpy(frame["label_index"].to_numpy(dtype=np.int64))
    events = frame["event_name"].tolist()
    splits = frame["split"].tolist()

    return {
        "img_feats": img_feats,
        "text_feats": text_feats,
        "labels": labels,
        "events": events,
        "splits": splits,
        "label_names": ds_damage.labels,
    }


class LinearExpert(nn.Module):
    """Linear projection and classification head."""
    def __init__(self, in_features: int = 512, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.proj = nn.Linear(in_features, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.proj(x))
        return self.classifier(h)


def train_expert(
    features: torch.Tensor,
    labels: torch.Tensor,
    base_model: LinearExpert,
    num_classes: int = 3,
    epochs: int = 60,
    lr: float = 0.05,
    seed: int = 42,
) -> LinearExpert:
    """Trains a task-specific expert initialized from base model."""
    torch.manual_seed(seed)
    expert = LinearExpert(features.shape[1], base_model.proj.out_features, num_classes)
    expert.load_state_dict(base_model.state_dict())

    optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    expert.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            logits = expert(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            optimizer.step()

    expert.eval()
    return expert


def evaluate_model(model: LinearExpert, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Evaluates classification accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(features)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
    return acc * 100.0


# ==============================================================================
# Model Merging Operators
# ==============================================================================

def merge_raw_addition(base: torch.Tensor, deltas: List[torch.Tensor]) -> torch.Tensor:
    return base + sum(deltas)


def merge_swa(base: torch.Tensor, deltas: List[torch.Tensor]) -> torch.Tensor:
    return base + sum(deltas) / len(deltas)


def merge_continual_task_arithmetic(base: torch.Tensor, deltas: List[torch.Tensor], gamma: float = 0.3) -> torch.Tensor:
    return base + gamma * sum(deltas)


def merge_ties(base: torch.Tensor, deltas: List[torch.Tensor], k: float = 0.3) -> torch.Tensor:
    stacked = torch.stack(deltas, dim=0) # [T, d_out, d_in]
    # 1. Trimming (top-k absolute magnitude)
    threshold = torch.quantile(stacked.abs(), 1.0 - k)
    mask = stacked.abs() >= threshold
    trimmed = stacked * mask.float()

    # 2. Elect sign
    sum_signs = torch.sign(trimmed).sum(dim=0)
    resolved_sign = torch.sign(sum_signs)

    # 3. Disjoint merge
    disjoint_mask = (torch.sign(trimmed) == resolved_sign.unsqueeze(0)) & mask
    disjoint_values = trimmed * disjoint_mask.float()
    counts = disjoint_mask.float().sum(dim=0).clamp(min=1.0)
    merged_delta = disjoint_values.sum(dim=0) / counts
    return base + 0.4 * merged_delta


def compute_opcm_projection_buggy(delta_old: torch.Tensor, delta_new: torch.Tensor, rank_r: int) -> Tuple[torch.Tensor, float]:
    """Author code: diag().fill_(0) which fails in-place modification."""
    combined = torch.cat([delta_old, delta_new], dim=1)
    U, S, Vt = torch.linalg.svd(combined, full_matrices=False)
    P = Vt[:rank_r, :]
    M = P.T @ P
    # Buggy in-place: .diag().fill_(0) does not zero out the diagonal of M
    M.diag().fill_(0)
    M[:rank_r, :rank_r] = 0
    delta_proj = delta_new - delta_old @ M[:delta_old.shape[1], delta_old.shape[1]:]
    inner = float(torch.abs(torch.sum(delta_old * delta_proj)).item())
    return delta_proj, inner


def compute_opcm_projection_corrected(delta_old: torch.Tensor, delta_new: torch.Tensor, rank_r: int) -> Tuple[torch.Tensor, float]:
    """Corrected code: .diagonal().zero_() + [:r, :r] = 0."""
    combined = torch.cat([delta_old, delta_new], dim=1)
    U, S, Vt = torch.linalg.svd(combined, full_matrices=False)
    P = Vt[:rank_r, :]
    M = P.T @ P
    M.diagonal().zero_()
    M[:rank_r, :rank_r] = 0
    d_old_dim = delta_old.shape[1]
    coupling = M[:d_old_dim, d_old_dim:]
    delta_proj = delta_new - delta_old @ coupling
    inner = float(torch.abs(torch.sum(delta_old * delta_proj)).item())
    return delta_proj, inner


def compute_opcm_projection_literal(delta_old: torch.Tensor, delta_new: torch.Tensor, rank_r: int) -> Tuple[torch.Tensor, float]:
    """Paper Eq. (9) Literal: Project delta_new onto orthogonal complement of delta_old."""
    U_old, _, _ = torch.linalg.svd(delta_old, full_matrices=False)
    # Projection onto null space of delta_old: (I - U_old U_old^T) delta_new
    U_r = U_old[:, :rank_r]
    delta_proj = delta_new - (U_r @ (U_r.T @ delta_new))
    inner = float(torch.abs(torch.sum(delta_old * delta_proj)).item())
    return delta_proj, inner


def apply_adaptive_scale(delta_old: torch.Tensor, delta_proj: torch.Tensor, delta_new: torch.Tensor, t: int) -> torch.Tensor:
    """Adaptive scale matching from Eq. (10)."""
    norm_old = torch.norm(delta_old)
    norm_new = torch.norm(delta_new)
    norm_proj = torch.norm(delta_proj) + 1e-9
    target_norm = (norm_old * (t - 1) + norm_new) / t
    merged_raw = delta_old + delta_proj
    merged_norm = torch.norm(merged_raw) + 1e-9
    return merged_raw * (target_norm / merged_norm)


# ==============================================================================
# Main Experiment Execution
# ==============================================================================

def run_experiment():
    print("=" * 80)
    print("🌍 KHỞI ĐỘNG THÍ NGHIỆM MỞ RỘNG TOÀN CẦU: OPCM TRÊN QCRI/CRISISMMD v2.0")
    print("=" * 80)

    data = load_crisismmd_v2_damage_data()
    img_feats = data["img_feats"]
    text_feats = data["text_feats"]
    labels = data["labels"]
    events = data["events"]
    splits = data["splits"]
    num_classes = len(data["label_names"])

    # Normalized fused features
    fused_feats = F.normalize(img_feats + text_feats, dim=-1)

    print(f"✅ Đã nạp thành công bộ dữ liệu QCRI/CrisisMMD v2 (Damage):")
    print(f"   - Tổng số mẫu: {len(labels)}")
    print(f"   - Số phân lớp: {num_classes} ({data['label_names']})")
    print(f"   - Kích thước vector: Image {img_feats.shape}, Text {text_feats.shape}")

    # Danh sách 7 đại thiên tai theo trình tự thời gian
    disaster_events = [
        "hurricane_harvey",
        "hurricane_maria",
        "hurricane_irma",
        "california_wildfires",
        "mexico_earthquake",
        "iraq_iran_earthquake",
        "srilanka_floods",
    ]

    print(f"\n📋 Danh sách 7 thảm họa toàn cầu (Stream of Disasters, T = 7):")
    event_indices = {}
    for ev in disaster_events:
        idx = [i for i, e in enumerate(events) if e == ev]
        event_indices[ev] = idx
        print(f"   - Task {len(event_indices)} ({ev}): {len(idx)} mẫu")

    # Chia train/test cho từng event
    event_datasets = {}
    torch.manual_seed(42)
    for ev, indices in event_indices.items():
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        n_total = len(idx_tensor)
        perm = torch.randperm(n_total)
        n_train = max(int(0.7 * n_total), 1)
        train_idx = idx_tensor[perm[:n_train]]
        test_idx = idx_tensor[perm[n_train:]]
        if len(test_idx) == 0:
            test_idx = train_idx # Fallback if tiny

        event_datasets[ev] = {
            "train_x": fused_feats[train_idx],
            "train_y": labels[train_idx],
            "test_x": fused_feats[test_idx],
            "test_y": labels[test_idx],
            "train_img": img_feats[train_idx],
            "train_txt": text_feats[train_idx],
            "test_img": img_feats[test_idx],
            "test_txt": text_feats[test_idx],
        }

    # ==========================================================================
    # PHẦN 1: HUẤN LUYỆN CÁC CHUYÊN GIA MIỀN CHO TỪNG THẢM HỌA (EXPERTS T1..T7)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🧠 [GIAI ĐOẠN 1] KHỞI TẠO VÀ HUẤN LUYỆN 7 CHUYÊN GIA MIỀN THẢM HỌA ĐỘC LẬP")
    print("=" * 80)

    base_model = LinearExpert(512, 256, num_classes)
    expert_models = []
    expert_deltas = []

    for t_idx, ev in enumerate(disaster_events, start=1):
        edata = event_datasets[ev]
        expert = train_expert(
            edata["train_x"],
            edata["train_y"],
            base_model,
            num_classes=num_classes,
            epochs=50,
            lr=0.03,
            seed=42 + t_idx,
        )
        expert_models.append(expert)
        acc_self = evaluate_model(expert, edata["test_x"], edata["test_y"])
        delta_w = expert.proj.weight.data - base_model.proj.weight.data
        expert_deltas.append(delta_w)
        print(f"   - Expert T{t_idx} ({ev}): Test Acc = {acc_self:.2f}% | Delta Norm = {torch.norm(delta_w):.2f}")

    # ==========================================================================
    # PHẦN 2: THỰC THI SO SÁNH CONTINUAL MERGING (T = 7 TASKS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🔄 [GIAI ĐOẠN 2] HỢP NHẤT TUẦN TỰ 7 THẢM HỌA (T = 7) VỚI CÁC THUẬT TOÁN")
    print("=" * 80)

    METHODS = [
        "Raw Addition",
        "SWA (Online Mean)",
        "Continual Task Arithmetic",
        "TIES-Merging",
        "OPCM Buggy",
        "OPCM Corrected",
        "OPCM Literal Eq. (9)",
        "Scale-Only Ablation",
        "Projection-Only Ablation",
    ]

    results = {}

    for method in METHODS:
        print(f"\n🚀 Đang thực thi hợp nhất: [{method}]...")
        R = np.zeros((7, 7))
        inner_products = []
        model_norms = []

        accumulated_delta = None

        for t in range(1, 8):
            current_deltas = expert_deltas[:t]
            delta_new = expert_deltas[t - 1]

            if t == 1:
                accumulated_delta = delta_new.clone()
                inner_products.append(0.0)
                model_norms.append(float(torch.norm(accumulated_delta).item()))
            else:
                delta_old = accumulated_delta.clone()

                if method == "Raw Addition":
                    accumulated_delta = merge_raw_addition(torch.zeros_like(delta_new), current_deltas)
                    inner_products.append(0.0)

                elif method == "SWA (Online Mean)":
                    accumulated_delta = merge_swa(torch.zeros_like(delta_new), current_deltas)
                    inner_products.append(0.0)

                elif method == "Continual Task Arithmetic":
                    accumulated_delta = merge_continual_task_arithmetic(torch.zeros_like(delta_new), current_deltas, gamma=0.3)
                    inner_products.append(0.0)

                elif method == "TIES-Merging":
                    accumulated_delta = merge_ties(torch.zeros_like(delta_new), current_deltas, k=0.3)
                    inner_products.append(0.0)

                elif method == "OPCM Buggy":
                    delta_proj, inner = compute_opcm_projection_buggy(delta_old, delta_new, rank_r=16)
                    accumulated_delta = apply_adaptive_scale(delta_old, delta_proj, delta_new, t)
                    inner_products.append(inner)

                elif method == "OPCM Corrected":
                    delta_proj, inner = compute_opcm_projection_corrected(delta_old, delta_new, rank_r=16)
                    accumulated_delta = apply_adaptive_scale(delta_old, delta_proj, delta_new, t)
                    inner_products.append(inner)

                elif method == "OPCM Literal Eq. (9)":
                    delta_proj, inner = compute_opcm_projection_literal(delta_old, delta_new, rank_r=16)
                    accumulated_delta = apply_adaptive_scale(delta_old, delta_proj, delta_new, t)
                    inner_products.append(inner)

                elif method == "Scale-Only Ablation":
                    # Scale without projection
                    norm_old = torch.norm(delta_old)
                    norm_new = torch.norm(delta_new)
                    target_norm = (norm_old * (t - 1) + norm_new) / t
                    merged_raw = delta_old + delta_new
                    accumulated_delta = merged_raw * (target_norm / (torch.norm(merged_raw) + 1e-9))
                    inner_products.append(0.0)

                elif method == "Projection-Only Ablation":
                    # Projection without scale
                    delta_proj, inner = compute_opcm_projection_corrected(delta_old, delta_new, rank_r=16)
                    accumulated_delta = delta_old + delta_proj
                    inner_products.append(inner)

                model_norms.append(float(torch.norm(accumulated_delta).item()))

            # Đánh giá mô hình hợp nhất sau bước t trên toàn bộ các task 1..t
            merged_model = LinearExpert(512, 256, num_classes)
            merged_model.load_state_dict(base_model.state_dict())
            merged_model.proj.weight.data += accumulated_delta
            # Lấy classifier head trung bình của các task đã học
            mean_cls_w = torch.stack([m.classifier.weight.data for m in expert_models[:t]]).mean(dim=0)
            mean_cls_b = torch.stack([m.classifier.bias.data for m in expert_models[:t]]).mean(dim=0)
            merged_model.classifier.weight.data.copy_(mean_cls_w)
            merged_model.classifier.bias.data.copy_(mean_cls_b)

            for i in range(t):
                ev_i = disaster_events[i]
                acc_ti = evaluate_model(merged_model, event_datasets[ev_i]["test_x"], event_datasets[ev_i]["test_y"])
                R[t - 1, i] = acc_ti

        # Tính toán các chỉ số continual learning
        # Final Average Accuracy: ACC = 1/T * sum_{i=1}^T R_{T-1, i}
        final_acc = float(np.mean(R[6, :]))

        # Backward Transfer (BWT): 1/(T-1) * sum_{i=1}^{T-1} (R_{T-1, i} - R_{i, i})
        bwt = float(np.mean([R[6, i] - R[i, i] for i in range(6)]))

        # Forgetting (FTG): 1/(T-1) * sum_{i=1}^{T-1} (max_{t < T-1} R_{t, i} - R_{T-1, i})
        forgetting = float(np.mean([np.max(R[:6, i]) - R[6, i] for i in range(6)]))

        results[method] = {
            "R_matrix": R.tolist(),
            "final_acc": round(final_acc, 2),
            "bwt": round(bwt, 2),
            "forgetting": round(forgetting, 2),
            "max_inner_product": float(np.max(inner_products)),
            "final_norm": round(model_norms[-1], 2),
            "model_norms": [round(n, 2) for n in model_norms],
            "inner_products": inner_products,
        }

        print(f"   -> ACC: {final_acc:.2f}% | BWT: {bwt:+.2f}% | FTG: {forgetting:.2f}% | Max Inner: {results[method]['max_inner_product']:.2e} | Norm: {results[method]['final_norm']}")

    # ==========================================================================
    # PHẦN 3: ĐA MÔ THỨC & BOUNDED GATING AC-FUSION TRÊN CRISISMMD V2
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🧬 [GIAI ĐOẠN 3] THỰC NGHIỆM ĐA MÔ THỨC & BOUNDED GATING TRÊN CRISISMMD V2")
    print("=" * 80)

    # Sử dụng mô hình OPCM tốt nhất
    opcm_model = LinearExpert(512, 256, num_classes)
    opcm_model.load_state_dict(base_model.state_dict())
    # Hợp nhất toàn bộ 7 task bằng OPCM Corrected
    all_deltas = expert_deltas
    acc_delta = all_deltas[0].clone()
    for t in range(2, 8):
        d_proj, _ = compute_opcm_projection_corrected(acc_delta, all_deltas[t-1], rank_r=16)
        acc_delta = apply_adaptive_scale(acc_delta, d_proj, all_deltas[t-1], t)
    opcm_model.proj.weight.data += acc_delta
    opcm_model.classifier.weight.data.copy_(torch.stack([m.classifier.weight.data for m in expert_models]).mean(dim=0))
    opcm_model.classifier.bias.data.copy_(torch.stack([m.classifier.bias.data for m in expert_models]).mean(dim=0))
    opcm_model.eval()

    # Gom toàn bộ tập test của 7 thảm họa
    all_test_img = torch.cat([event_datasets[ev]["test_img"] for ev in disaster_events], dim=0)
    all_test_txt = torch.cat([event_datasets[ev]["test_txt"] for ev in disaster_events], dim=0)
    all_test_y = torch.cat([event_datasets[ev]["test_y"] for ev in disaster_events], dim=0)

    with torch.no_grad():
        v_logits = opcm_model(all_test_img)
        t_logits = opcm_model(all_test_txt)

        # 1. Vision-Only
        preds_v = v_logits.argmax(dim=-1)
        acc_v = (preds_v == all_test_y).float().mean().item() * 100.0
        f1_v = f1_score(all_test_y.numpy(), preds_v.numpy(), average="macro") * 100.0

        # 2. Text-Only
        preds_t = t_logits.argmax(dim=-1)
        acc_t = (preds_t == all_test_y).float().mean().item() * 100.0
        f1_t = f1_score(all_test_y.numpy(), preds_t.numpy(), average="macro") * 100.0

        # 3. Fixed Multimodal (0.5 - 0.5)
        fused_half = 0.5 * v_logits + 0.5 * t_logits
        preds_half = fused_half.argmax(dim=-1)
        acc_half = (preds_half == all_test_y).float().mean().item() * 100.0
        f1_half = f1_score(all_test_y.numpy(), preds_half.numpy(), average="macro") * 100.0

        # 4. Unbounded AC-Fusion (Entropy Gating)
        p_v = F.softmax(v_logits, dim=-1)
        p_t = F.softmax(t_logits, dim=-1)
        H_v = -torch.sum(p_v * torch.log(p_v + 1e-9), dim=-1)
        H_t = -torch.sum(p_t * torch.log(p_t + 1e-9), dim=-1)
        raw_gate = torch.sigmoid((H_t - H_v) / 1.0).unsqueeze(-1)
        fused_unbounded = raw_gate * v_logits + (1.0 - raw_gate) * t_logits
        preds_unbounded = fused_unbounded.argmax(dim=-1)
        acc_unbounded = (preds_unbounded == all_test_y).float().mean().item() * 100.0
        f1_unbounded = f1_score(all_test_y.numpy(), preds_unbounded.numpy(), average="macro") * 100.0

        # 5. Bounded Gating AC-Fusion (Leader-Follower Gating: beta_min = 0.85, T = 2.0)
        bounded_gate = 0.85 + 0.15 * torch.sigmoid((H_t - H_v) / 2.0).unsqueeze(-1)
        fused_bounded = bounded_gate * v_logits + (1.0 - bounded_gate) * t_logits
        preds_bounded = fused_bounded.argmax(dim=-1)
        acc_bounded = (preds_bounded == all_test_y).float().mean().item() * 100.0
        f1_bounded = f1_score(all_test_y.numpy(), preds_bounded.numpy(), average="macro") * 100.0

    multimodal_results = {
        "Vision-Only": {"Acc": round(acc_v, 2), "Macro-F1": round(f1_v, 2)},
        "Text-Only": {"Acc": round(acc_t, 2), "Macro-F1": round(f1_t, 2)},
        "Fixed Multimodal (0.5/0.5)": {"Acc": round(acc_half, 2), "Macro-F1": round(f1_half, 2)},
        "Unbounded AC-Fusion": {"Acc": round(acc_unbounded, 2), "Macro-F1": round(f1_unbounded, 2)},
        "Bounded Gating AC-Fusion (Ours)": {"Acc": round(acc_bounded, 2), "Macro-F1": round(f1_bounded, 2)},
    }

    print("\n📊 KẾT QUẢ ĐA MÔ THỨC TRÊN TOÀN BỘ 7 ĐẠI THIÊN TAI (CRISISMMDV2 DAMAGE):")
    for k, v in multimodal_results.items():
        print(f"   - {k:35s}: Accuracy = {v['Acc']:.2f}% | Macro-F1 = {v['Macro-F1']:.2f}%")

    # ==========================================================================
    # PHẦN 4: MA TRẬN NHẦM LẪN VÀ BÓC TÁCH CHI TIẾT 3 MỨC ĐỘ HƯ HẠI
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🔍 [GIAI ĐOẠN 4] BÓC TÁCH MA TRẬN NHẦM LẪN & CHẨN ĐOÁN LỖI 3 MỨC ĐỘ TÀN PHÁ")
    print("=" * 80)

    cm = confusion_matrix(all_test_y.numpy(), preds_bounded.numpy())
    cm_norm = confusion_matrix(all_test_y.numpy(), preds_bounded.numpy(), normalize="true") * 100.0
    report = classification_report(
        all_test_y.numpy(),
        preds_bounded.numpy(),
        target_names=data["label_names"],
        output_dict=True,
    )

    per_class_diagnostics = {}
    for name in data["label_names"]:
        per_class_diagnostics[name] = {
            "Precision": round(report[name]["precision"] * 100.0, 2),
            "Recall": round(report[name]["recall"] * 100.0, 2),
            "F1-Score": round(report[name]["f1-score"] * 100.0, 2),
            "Support": int(report[name]["support"]),
        }

    print("📊 Bảng chẩn đoán chi tiết từng lớp hư hại:")
    for name, metrics in per_class_diagnostics.items():
        print(f"   - {name:22s}: Precision = {metrics['Precision']:.2f}% | Recall = {metrics['Recall']:.2f}% | F1 = {metrics['F1-Score']:.2f}% (Support: {metrics['Support']})")

    # Lưu kết quả tổng hợp
    full_output = {
        "metadata": {
            "dataset": "QCRI/CrisisMMD v2.0 (damage)",
            "num_samples": len(labels),
            "disasters": disaster_events,
            "classes": data["label_names"],
            "date": "2026-10-17",
        },
        "continual_merging_results": results,
        "multimodal_results": multimodal_results,
        "per_class_diagnostics": per_class_diagnostics,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
    }

    output_path = OUTPUT_DIR / "opcm_crisismmd_v2_study_results.json"
    output_path.write_text(json.dumps(full_output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 ĐÃ LƯU TOÀN BỘ KẾT QUẢ THỰC NGHIỆM TẠI: {output_path}")

    return full_output


if __name__ == "__main__":
    run_experiment()
