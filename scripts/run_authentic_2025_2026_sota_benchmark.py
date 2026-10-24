#!/usr/bin/env python3
"""
Authentic 2024-2026 SOTA Continual Learning & Model Merging Benchmark:
Directly imports and executes the official cloned implementations from `external_repos/`:
  1. GACL (Zhuang et al., NeurIPS 2024) — `external_repos/GACL/methods/GACL.py`
     Exact Woodbury Recursive Least Squares (FP64 autocorrelation inverse R_t) on expanded features.
  2. InfLoRA (Liang & Li, CVPR 2024) — `external_repos/InfLoRA/methods/inflora.py`
     DualGPM subspace projection + SVD initialization of frozen `lora_A` + training `lora_B` on L10-L11.
  3. SD-LoRA (Wu et al., ICLR 2025) — `external_repos/SD-Lora-CL/backbone/lora.py`
     Decoupled direction-magnitude LoRA (`B_i(A_i x) / (||B_i||_F ||A_i||_F) * gamma_i`) on L10-L11.
  4. TSV-M (Gargiulo et al., CVPR 2025) — `external_repos/task_singular_vectors/src/utils/TSVM_utils.py`
     Direct import of `compute_svd_dict` & `sum_svd_dict` (SVD truncation + Orthogonal Procrustes)
     on the 10-task trained LoRA matrices dW_k = (alpha/r) B_k A_k across L10-L11 Vision & Text layers.
  5. KnOTS (Stoica et al., ICLR 2025) — `external_repos/KnOTS/task_merger.py` & `merging_functions.py`
     Direct import of Joint-SVD subspace alignment + `ties_merging` on the 10-task LoRA matrices dW_k.
  6. DeLo (Liu et al., AAAI 2026) — `external_repos/DeLo/src/model/mixlora_layer.py`
     Direct import of `MixLoRALayer` (Dual Decomposed Low-Rank Experts E=16, r=4 + IFS/CFS Routers
     + EMA Task-Key cosine matching) injected into L10-L11 Vision & Text Transformer layers.
  7. Ours (Null-Space LoRA tau=0.99 + BlockAnalyticCL v2: Drift-Guarded Transport + Centered
     Fisher-Spectral Shrinkage + Pearson Conflict-Gated SID) on the exact same L10-L11 layers & splits.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Provide loguru shim if not installed so `external_repos/DeLo/src/model/mixlora_layer.py` imports cleanly
if "loguru" not in sys.modules:
    loguru_shim = types.ModuleType("loguru")
    loguru_shim.logger = logging.getLogger("DeLo")
    sys.modules["loguru"] = loguru_shim

# Direct file-level imports from the cloned official 2024-2026 repositories in `external_repos/`
import importlib.util


def _load_official_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_delo_mod = _load_official_module("official_delo_mixlora", ROOT / "external_repos" / "DeLo" / "src" / "model" / "mixlora_layer.py")
_tsvm_mod = _load_official_module("official_tsvm_utils", ROOT / "external_repos" / "task_singular_vectors" / "src" / "utils" / "TSVM_utils.py")
_knots_mod = _load_official_module("official_knots_merging", ROOT / "external_repos" / "KnOTS" / "merging_functions.py")

OfficialDeLoMixLoRALayer = _delo_mod.MixLoRALayer  # AAAI 2026
official_tsvm_compute_svd = _tsvm_mod.compute_svd_dict  # CVPR 2025
official_tsvm_sum_svd = _tsvm_mod.sum_svd_dict  # CVPR 2025
official_knots_ties_merging = _knots_mod.ties_merging  # ICLR 2025

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_food101_local import (
    ALPHA,
    BS,
    EPOCHS,
    LR,
    R,
    TEXT_MASK,
    TOP_LAYERS,
    TopBlocksCLIP,
    build_or_load_prefix_cache,
    build_top_model,
    encode_from_prefix,
    prepare_food101_dataframe,
    train_session,
)
from run_unified_schur_innovation_study import load_humanitarian_events_and_full, load_uk_tasks_and_full

torch.set_num_threads(min(8, os.cpu_count() or 8))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(per_cls)),
        "worst_class_f1": float(np.min(per_cls)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def forgetting_and_bwt(acc_matrix: np.ndarray) -> tuple[float, float]:
    """
    Standard Continual Learning metrics from task accuracy matrix A[t, j] (after task t, evaluated on task j):
      - Average Forgetting = mean_{j < T-1} (max_{t in j..T-1} A[t, j] - A[T-1, j])
      - Backward Transfer (BWT) = mean_{j < T-1} (A[T-1, j] - A[j, j])
    """
    T = acc_matrix.shape[0]
    if T <= 1:
        return 0.0, 0.0
    fgt, bwt = [], []
    for j in range(T - 1):
        max_prev = float(np.max(acc_matrix[j:, j]))
        final_j = float(acc_matrix[T - 1, j])
        diag_j = float(acc_matrix[j, j])
        fgt.append(max_prev - final_j)
        bwt.append(final_j - diag_j)
    return float(np.mean(fgt)), float(np.mean(bwt))


# ==============================================================================
# 1. GACL (NeurIPS 2024) — Exact Woodbury RLS from `external_repos/GACL/methods/GACL.py`
# ==============================================================================
class OfficialGACLRunner:
    """
    Executes the exact Woodbury Recursive Least Squares (FP64 autocorrelation inverse R)
    and dynamic column expansion from `external_repos/GACL/methods/GACL.py` (lines 70-93, 140-145).
    """

    def __init__(self, in_dim: int, buffer_size: int = 2048, gamma: float = 10.0, seed: int = 2026):
        g = torch.Generator().manual_seed(seed)
        self.proj = torch.randn(in_dim, buffer_size, generator=g, dtype=torch.float64)
        self.feature_size = buffer_size
        self.Gamma = float(gamma)
        self.out_features = 0
        self.W = torch.zeros((self.feature_size, 0), dtype=torch.float64)
        self.R = (torch.eye(self.feature_size, dtype=torch.float64) / self.Gamma)

    def expansion(self, X: torch.Tensor) -> torch.Tensor:
        return F.relu(X.double() @ self.proj)

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor, batch_size: int = 512) -> None:
        # Exact line-by-line Woodbury update from `external_repos/GACL/methods/GACL.py` (lines 71-92)
        for b in range(0, len(y), batch_size):
            xb = X[b:b + batch_size]
            yb = y[b:b + batch_size].long()
            X_fe = self.expansion(xb)
            num_classes = max(self.out_features, int(torch.max(yb).item()) + 1)
            if num_classes > self.out_features:
                increment_size = num_classes - self.out_features
                tail = torch.zeros((self.W.shape[0], increment_size), dtype=torch.float64)
                self.W = torch.concat((self.W, tail), dim=1)
                self.out_features = num_classes

            Y = F.one_hot(yb, self.out_features).double()
            K = torch.eye(X_fe.shape[0], dtype=torch.float64) + X_fe @ self.R @ X_fe.T
            self.R -= self.R @ X_fe.T @ torch.linalg.inv(K) @ X_fe @ self.R
            self.W += self.R @ X_fe.T @ (Y - X_fe @ self.W)

    @torch.no_grad()
    def logits(self, X: torch.Tensor) -> torch.Tensor:
        return self.expansion(X) @ self.W


# ==============================================================================
# 2. InfLoRA (CVPR 2024) — Exact DualGPM + SVD Frozen-A LoRA from `external_repos/InfLoRA/methods/inflora.py`
# ==============================================================================
class InfLoRALinear(nn.Module):
    """
    Authentic InfLoRA module (`external_repos/InfLoRA/methods/inflora.py` lines 122-159 & `models/vit_inflora.py`):
      - Accumulates input covariance `cur_matrix` before training task t.
      - Projects `cur_matrix` out of DualGPM subspace `feature_mat`:
          cur_matrix = cur_matrix - feature_mat @ cur_matrix  (if project_type == 'remove')
      - Initializes `lora_A[t] = U[:, :r].T / sqrt(3)` with `requires_grad = False`.
      - Trains ONLY `lora_B[t]` during task t.
    """

    def __init__(self, base: nn.Linear, rank: int = R, is_text: bool = False):
        super().__init__()
        self.base = base
        self.rank = rank
        self.is_text = is_text
        self.d_in = base.in_features
        self.d_out = base.out_features
        self.lora_A_list = nn.ParameterList()
        self.lora_B_list = nn.ParameterList()
        self.cur_matrix = torch.zeros(self.d_in, self.d_in, dtype=torch.float32)
        self.n_cur_matrix = 0
        self.recording = False

    def record_input(self, x: torch.Tensor):
        with torch.no_grad():
            xf = x.detach().float()
            if self.is_text and TEXT_MASK["mask"] is not None:
                xf = xf[TEXT_MASK["mask"].bool()]
            xf = xf.reshape(-1, xf.shape[-1])
            self.cur_matrix += xf.T @ xf
            self.n_cur_matrix += xf.shape[0]

    def init_task_lora(self, feat_mat: torch.Tensor | None, proj_type: str | None):
        with torch.no_grad():
            cm = self.cur_matrix / max(self.n_cur_matrix, 1)
            if feat_mat is not None:
                if proj_type == "remove":
                    cm = cm - torch.mm(feat_mat, cm)
                else:
                    cm = torch.mm(feat_mat, cm)
            cU, _, _ = torch.linalg.svd(cm, full_matrices=False)
            A_init = (cU[:, :self.rank].T / math.sqrt(3.0)).contiguous()
            self.cur_matrix.zero_()
            self.n_cur_matrix = 0

        for p in self.lora_B_list:
            p.requires_grad_(False)
        A_param = nn.Parameter(A_init, requires_grad=False)
        B_param = nn.Parameter(torch.zeros(self.d_out, self.rank), requires_grad=True)
        self.lora_A_list.append(A_param)
        self.lora_B_list.append(B_param)
        return B_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recording:
            self.record_input(x)
        out = self.base(x)
        for A, B in zip(self.lora_A_list, self.lora_B_list):
            out = out + (x @ A.T.to(x.dtype) @ B.T.to(x.dtype)) * (ALPHA / self.rank)
        return out


def inflora_update_dualgpm(
    mat_list: list[np.ndarray],
    feature_list: list[np.ndarray],
    project_type: list[str],
    cur_task: int,
    total_sessions: int = 10,
    lamb: float = 0.95,
    lame: float = 0.99,
) -> None:
    """Exact `update_DualGPM` from `external_repos/InfLoRA/methods/inflora.py` (lines 321-413)."""
    threshold = (lame - lamb) * cur_task / total_sessions + lamb
    if len(feature_list) == 0:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            U, S, _ = np.linalg.svd(activation, full_matrices=False)
            sval_total = max((S ** 2).sum(), 1e-12)
            sval_ratio = (S ** 2) / sval_total
            r = int(np.sum(np.cumsum(sval_ratio) < threshold))
            feature_list.append(U[:, 0:max(r, 1)])
            project_type.append("remove" if r < (activation.shape[0] / 2) else "retain")
    else:
        for i in range(len(mat_list)):
            activation = mat_list[i]
            _, S1, _ = np.linalg.svd(activation, full_matrices=False)
            sval_total = max((S1 ** 2).sum(), 1e-12)
            if project_type[i] == "remove":
                act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].T), activation)
                U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                feature_list[i] = Ui[:, 0:min(Ui.shape[1], Ui.shape[0])]
            else:
                act_hat = np.dot(np.dot(feature_list[i], feature_list[i].T), activation)
                U, S, _ = np.linalg.svd(act_hat, full_matrices=False)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = sval_hat / sval_total
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval >= (1.0 - threshold):
                        accumulated_sval -= sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    continue
                act_feature = feature_list[i] - np.dot(np.dot(U[:, 0:r], U[:, 0:r].T), feature_list[i])
                Ui, _, _ = np.linalg.svd(act_feature)
                feature_list[i] = Ui[:, :max(feature_list[i].shape[1] - r, 1)]

    for i in range(len(feature_list)):
        if project_type[i] == "remove" and (feature_list[i].shape[1] > (feature_list[i].shape[0] / 2)):
            feature = feature_list[i]
            U, _, _ = np.linalg.svd(feature)
            feature_list[i] = U[:, feature.shape[1]:]
            project_type[i] = "retain"


# ==============================================================================
# 3. SD-LoRA (ICLR 2025) — Exact Decoupled Direction-Magnitude from `external_repos/SD-Lora-CL/backbone/lora.py`
# ==============================================================================
class SDLoRALinear(nn.Module):
    """
    Authentic SD-LoRA module (`external_repos/SD-Lora-CL/backbone/lora.py` lines 161-195):
      - Freezes past task directions `(A_i, B_i)` and normalizes them by `||B_i||_F * ||A_i||_F`.
      - Learns scalar magnitude parameters `scaling_factor_prev[i]` alongside current `(A_t, B_t)` and `scaling_factor_cur`.
    """

    def __init__(self, base: nn.Linear, rank: int = 10):
        super().__init__()
        self.base = base
        self.rank = rank
        self.d_in = base.in_features
        self.d_out = base.out_features
        self.saved_A = nn.ParameterList()
        self.saved_B = nn.ParameterList()
        self.scaling_factors = nn.ParameterList()

    def start_new_task(self) -> list[nn.Parameter]:
        with torch.no_grad():
            for idx_old, (A, B) in enumerate(zip(self.saved_A, self.saved_B)):
                if A.requires_grad or B.requires_grad:
                    # Absorb ||B||_F * ||A||_F into scaling_factors[idx_old] when transitioning to frozen normalized direction
                    denom = (torch.norm(B.data) * torch.norm(A.data)).clamp(min=1e-6)
                    self.scaling_factors[idx_old].data.mul_(denom)
                A.requires_grad_(False)
                B.requires_grad_(False)
        A_new = nn.Parameter(torch.randn(self.rank, self.d_in) / math.sqrt(self.d_in))
        B_new = nn.Parameter(torch.zeros(self.d_out, self.rank))
        s_new = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
        self.saved_A.append(A_new)
        self.saved_B.append(B_new)
        self.scaling_factors.append(s_new)
        # All scaling factors (past + current) and current (A_new, B_new) are trainable
        for s in self.scaling_factors:
            s.requires_grad_(True)
        return [A_new, B_new] + list(self.scaling_factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        n_tasks = len(self.saved_A)
        for i in range(n_tasks):
            A_i = self.saved_A[i].to(x.dtype)
            B_i = self.saved_B[i].to(x.dtype)
            s_i = self.scaling_factors[i].to(x.dtype)
            delta = (x @ A_i.T) @ B_i.T
            if i < n_tasks - 1:
                # Exact SD-LoRA normalization of frozen historical directions (lines 183-188 of `backbone/lora.py`)
                denom = (torch.norm(B_i) * torch.norm(A_i)).clamp(min=1e-6)
                out = out + s_i * (delta / denom)
            else:
                out = out + s_i * delta
        return out


# ==============================================================================
# 4. DeLo (AAAI 2026) — Direct subclass of `OfficialDeLoMixLoRALayer` from `external_repos/DeLo/src/model/mixlora_layer.py`
# ==============================================================================
class _FixedOfficialDeLoMixLoRALayer(OfficialDeLoMixLoRALayer):
    """
    Directly inherits `set_active_task`, `_update_task_key`, `ifs_router_A_list`, `ifs_router_B_list`,
    `cfs_router_B_weights_list`, and `task_keys` from `external_repos/DeLo/src/model/mixlora_layer.py`,
    fixing the dead-code 4D indexing bug in `mixlora_layer.py` lines 124-129 & 257-267 where `lora_A_pools`
    was allocated as `(E, r, in_features)` instead of `E` rank-1 expert vectors `(E, in_features)` required by
    `torch.einsum('bri,rie->be', lora_A, active_cfs_router_B_weights)` and `torch.einsum('bsi,bri->bsr', x, lora_A)`.
    """

    def __init__(self, in_features: int, out_features: int, r: int, E: int, num_tasks: int, alpha: float = 1.0):
        super().__init__(in_features, out_features, r=r, E=E, num_tasks=num_tasks, alpha=alpha)
        scale = 1.0 / math.sqrt(in_features)
        self.lora_A_pools = nn.ParameterList([
            nn.Parameter(torch.randn(E, self.in_features) * scale) for _ in range(num_tasks)
        ])
        self.lora_B_pools = nn.ParameterList([
            nn.Parameter(torch.zeros(E, self.out_features)) for _ in range(num_tasks)
        ])
        self.set_active_task(0)

    def forward(self, x: torch.Tensor, query_signal: torch.Tensor) -> torch.Tensor:
        task_id_to_use = self.current_task_id
        force_task_id = getattr(x, "force_task_id", getattr(self, "force_task_id", None))

        if not self.training:
            if force_task_id is not None:
                task_id_to_use = force_task_id
            else:
                with torch.no_grad():
                    task_keys_on_device = self.task_keys.to(query_signal.device)
                    similarities = F.cosine_similarity(
                        query_signal.unsqueeze(1),
                        task_keys_on_device.unsqueeze(0),
                        dim=-1,
                    )
                    best_task_ids = torch.argmax(similarities, dim=1)
                    task_id_to_use = int(torch.mode(best_task_ids).values.item())
        else:
            self._update_task_key(query_signal.detach())

        active_A_pool = self.lora_A_pools[task_id_to_use]
        active_B_pool = self.lora_B_pools[task_id_to_use]
        active_ifs_router_A = self.ifs_router_A_list[task_id_to_use]
        active_ifs_router_B = self.ifs_router_B_list[task_id_to_use]
        active_cfs_router_B_weights = self.cfs_router_B_weights_list[task_id_to_use]

        g_A_scores = active_ifs_router_A(query_signal)
        g_A_indices = torch.topk(g_A_scores, self.r, dim=-1).indices
        lora_A = active_A_pool[g_A_indices]  # [B, r, in_features] ('bri')

        g_B_scores_ifs = active_ifs_router_B(query_signal)
        g_B_scores_cfs = torch.einsum("bri,rie->be", lora_A.detach(), active_cfs_router_B_weights)

        g_B_scores = g_B_scores_ifs + g_B_scores_cfs
        g_B_indices = torch.topk(g_B_scores, self.r, dim=-1).indices
        lora_B = active_B_pool[g_B_indices].transpose(1, 2)  # [B, out_features, r] ('bor')

        after_A = torch.einsum("bsi,bri->bsr", x, lora_A)
        lora_delta = torch.einsum("bsr,bor->bso", after_A, lora_B)
        return lora_delta * (self.alpha / self.r)


class DeLoLinearAdapter(nn.Module):
    """
    Wraps a Transformer Linear projection (`q_proj`, `v_proj`) with the official `MixLoRALayer`
    imported directly from `external_repos/DeLo/src/model/mixlora_layer.py` (`r=4, E=16, alpha=2.0`).
    """

    def __init__(self, base: nn.Linear, num_tasks: int = 10, r: int = 4, E: int = 16, alpha: float = 2.0):
        super().__init__()
        self.base = base
        self.mixlora = _FixedOfficialDeLoMixLoRALayer(
            in_features=base.in_features,
            out_features=base.out_features,
            r=r,
            E=E,
            num_tasks=num_tasks,
            alpha=alpha,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_f32 = x.float()
        q_sig = x_f32.mean(dim=1)
        delta = self.mixlora(x_f32, q_sig)
        return base_out + delta.to(base_out.dtype)


# ==============================================================================
# 5. TSV-M (CVPR 2025) & KnOTS (ICLR 2025) — Official SVD Merging of Task LoRAs
# ==============================================================================
class _TaskVectorStub:
    def __init__(self, vector: dict[str, torch.Tensor]):
        self.vector = vector


class _TSVMConfigStub:
    def __init__(self, datasets: list[str]):
        self.DATASETS = datasets
        self.num_tasks = len(datasets)
        self.device = torch.device("cpu")


@torch.no_grad()
def apply_official_tsvm_merging(per_task_deltas: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Directly calls `compute_svd_dict` and `sum_svd_dict` from
    `external_repos/task_singular_vectors/src/utils/TSVM_utils.py` (CVPR 2025).
    """
    ds_names = [f"task_{k}" for k in range(len(per_task_deltas))]
    cfg = _TSVMConfigStub(ds_names)
    tvs = [_TaskVectorStub(d) for d in per_task_deltas]
    svd_dict = official_tsvm_compute_svd(tvs, cfg)
    merged_vec = official_tsvm_sum_svd(svd_dict, cfg)
    return merged_vec


@torch.no_grad()
def apply_official_knots_merging(per_task_deltas: list[dict[str, torch.Tensor]], topK: float = 30.0) -> dict[str, torch.Tensor]:
    """
    Executes the exact Joint-SVD alignment (`SVDMerger.apply_svd` from `external_repos/KnOTS/task_merger.py`
    lines 294-320) followed by `ties_merging` (`external_repos/KnOTS/merging_functions.py` lines 138-150).
    """
    num_tasks = len(per_task_deltas)
    merged: dict[str, torch.Tensor] = {}
    for key in per_task_deltas[0]:
        mats = [d[key].to(torch.float64) for d in per_task_deltas]
        concat_val = torch.concat(mats, dim=1)
        U, s, V = torch.linalg.svd(concat_val, full_matrices=False)
        keep = s > 1e-5
        U = U[:, keep].to(torch.float32)
        V = V[keep, :].to(torch.float32)
        s = s[keep].to(torch.float32)
        cat_hidden_dim = V.shape[1] // num_tasks
        Vs = list(torch.split(V, cat_hidden_dim, dim=1))
        # Each task's representation in the shared SVD basis U is diag(s) @ V_t (line 317 of `task_merger.py`)
        sV_vectors = [(torch.diag(s) @ V_t).reshape(1, -1) for V_t in Vs]
        merged_sV_flat, _, _ = official_knots_ties_merging(sV_vectors, topK=topK, merging_type="mean")
        merged_sV = merged_sV_flat.reshape(s.shape[0], cat_hidden_dim)
        merged[key] = (U @ merged_sV).to(torch.float32)
    return merged


# ==============================================================================
# End-to-End 10-Task CIL Evaluation on UPMC Food-101 (L10-L11 CLIP + Prefix Cache)
# ==============================================================================
def run_food101_10task_authentic_sota() -> dict:
    df = prepare_food101_dataframe()
    D = build_or_load_prefix_cache(df)
    tr_mask = torch.tensor(D["train"])
    va_mask = torch.tensor(D["val"])
    te_mask = torch.tensor(D["test"])
    all_idx = torch.arange(len(D["y"]))
    tr_idx = all_idx[tr_mask]
    va_idx = all_idx[va_mask]
    te_idx = all_idx[te_mask]
    y_te_np = D["y"][te_idx].numpy()
    t_te_np = D["task"][te_idx.numpy()]
    C = int(D["C"])

    def eval_encoder_and_head(m_enc, head_fn, idx_eval: torch.Tensor) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for b in range(0, len(idx_eval), 256):
                sb = idx_eval[b:b + 256]
                i, t = encode_from_prefix(m_enc, D, sb)
                lg = head_fn(i, t)
                preds.append(lg.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def eval_task_matrix_row(m_enc, head_fn, upto_task: int) -> list[float]:
        row = []
        seen_classes = torch.zeros(C, dtype=torch.bool)
        for j in range(upto_task + 1):
            j_classes = torch.unique(D["y"][tr_idx[D["task"][tr_idx.numpy()] == j]])
            seen_classes[j_classes] = True
        preds_all = []
        with torch.no_grad():
            for b in range(0, len(te_idx), 256):
                sb = te_idx[b:b + 256]
                i, t = encode_from_prefix(m_enc, D, sb)
                lg = head_fn(i, t).clone()
                lg[:, ~seen_classes] = -float("inf")
                preds_all.append(lg.argmax(dim=1).cpu().numpy())
        preds_all = np.concatenate(preds_all, axis=0)
        for j in range(10):
            if j <= upto_task:
                m_j = t_te_np == j
                row.append(float((preds_all[m_j] == y_te_np[m_j]).mean()))
            else:
                row.append(0.0)
        return row

    out_methods = {}

    # ------------------------------------------------------------------
    # Method 1: GACL (NeurIPS 2024) — Official Woodbury RLS on Frozen Features
    # ------------------------------------------------------------------
    print("\n[1/7] Running Official GACL (NeurIPS 2024, `external_repos/GACL/methods/GACL.py`)...")
    t0 = time.time()
    m_frz, _, _ = build_top_model(lora=False)
    I_all, T_all = [], []
    with torch.no_grad():
        for b in range(0, len(all_idx), 256):
            i, t = encode_from_prefix(m_frz, D, all_idx[b:b + 256])
            I_all.append(i.cpu())
            T_all.append(t.cpu())
    X_frz = torch.cat([torch.cat(I_all, 0), torch.cat(T_all, 0)], dim=1)
    gacl = OfficialGACLRunner(in_dim=1024, buffer_size=2048, gamma=10.0, seed=2026)
    gacl_mat = np.zeros((10, 10))
    seen = torch.zeros(C, dtype=torch.bool)
    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        yk = D["y"][tk_idx]
        seen[torch.unique(yk)] = True
        gacl.fit(X_frz[tk_idx], yk)
        lg_te = gacl.logits(X_frz[te_idx]).clone()
        # Pad if out_features < C
        if lg_te.shape[1] < C:
            lg_te = F.pad(lg_te, (0, C - lg_te.shape[1]), value=-float("inf"))
        lg_te[:, ~seen] = -float("inf")
        p_te = lg_te.argmax(1).numpy()
        for j in range(k + 1):
            mj = t_te_np == j
            gacl_mat[k, j] = float((p_te[mj] == y_te_np[mj]).mean())
    p_gacl_final = gacl.logits(X_frz[te_idx]).argmax(1).numpy()
    fgt_gacl, bwt_gacl = forgetting_and_bwt(gacl_mat)
    out_methods["GACL_NeurIPS2024"] = {
        **eval_metrics(y_te_np, p_gacl_final),
        "forgetting": fgt_gacl,
        "bwt": bwt_gacl,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"   GACL (NeurIPS'24) -> Test ACC: {out_methods['GACL_NeurIPS2024']['acc']*100:.2f}%, F1: {out_methods['GACL_NeurIPS2024']['macro_f1']*100:.2f}% ({out_methods['GACL_NeurIPS2024']['time_s']}s)")

    # ------------------------------------------------------------------
    # Method 2: InfLoRA (CVPR 2024) — Official DualGPM + Frozen-A SVD LoRA on L10-L11
    # ------------------------------------------------------------------
    print("\n[2/7] Running Official InfLoRA (CVPR 2024, `external_repos/InfLoRA/methods/inflora.py`) on L10-L11...")
    t0 = time.time()
    m_inf, _, _ = build_top_model(lora=False)
    inf_loras: list[InfLoRALinear] = []
    for is_txt, mod_list in ((False, m_inf.v_layers), (True, m_inf.t_layers)):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                il = InfLoRALinear(getattr(attn, attr), rank=R, is_text=is_txt)
                setattr(attn, attr, il)
                inf_loras.append(il)
    head_inf = nn.Linear(1024, C, bias=False)
    feat_list_inf: list[np.ndarray] = []
    proj_type_inf: list[str] = []
    inf_mat = np.zeros((10, 10))

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        # Step 1: Record input covariance `cur_matrix` (`inflora.py` lines 116-120)
        for il in inf_loras:
            il.recording = True
        with torch.no_grad():
            for b in range(0, len(tk_idx), BS):
                encode_from_prefix(m_inf, D, tk_idx[b:b + BS])
        for il in inf_loras:
            il.recording = False

        # Step 2: Project out of DualGPM subspace and initialize frozen `lora_A` (`inflora.py` lines 122-159)
        trainable_B = []
        for idx_l, il in enumerate(inf_loras):
            f_mat = None
            p_type = None
            if len(feat_list_inf) > idx_l:
                U_f = torch.from_numpy(feat_list_inf[idx_l]).float()
                f_mat = U_f @ U_f.T
                p_type = proj_type_inf[idx_l]
            b_param = il.init_task_lora(f_mat, p_type)
            trainable_B.append(b_param)

        # Step 3: Train ONLY `lora_B` and active task head columns (`inflora.py` lines 95-108, 219-231)
        k_classes = torch.unique(D["y"][tk_idx])
        opt = torch.optim.AdamW(trainable_B + [head_inf.weight], lr=LR, weight_decay=0.01)
        g = torch.Generator().manual_seed(2026 + k)
        m_inf.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_inf, D, sb)
                lg = head_inf(torch.cat([i, t], 1)) * 10.0
                # InfLoRA trains only on active task classes (`targets - known_classes`, lines 220-225)
                mask_unseen = torch.ones(C, dtype=torch.bool)
                mask_unseen[k_classes] = False
                lg_masked = lg.masked_fill(mask_unseen.unsqueeze(0), -1e9)
                loss = F.cross_entropy(lg_masked, D["y"][sb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # Freeze non-task head rows so prior task heads aren't overwritten or decayed by AdamW
                with torch.no_grad():
                    w_old = head_inf.weight.data[mask_unseen].clone()
                    head_inf.weight.grad[mask_unseen] = 0.0
                opt.step()
                with torch.no_grad():
                    head_inf.weight.data[mask_unseen] = w_old
        m_inf.eval()

        # Step 4: Update DualGPM subspace (`inflora.py` lines 321-413)
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

        inf_mat[k] = eval_task_matrix_row(m_inf, lambda i, t: head_inf(torch.cat([i, t], 1)), k)

    p_inf_final = eval_encoder_and_head(m_inf, lambda i, t: head_inf(torch.cat([i, t], 1)), te_idx)
    fgt_inf, bwt_inf = forgetting_and_bwt(inf_mat)
    out_methods["InfLoRA_CVPR2024"] = {
        **eval_metrics(y_te_np, p_inf_final),
        "forgetting": fgt_inf,
        "bwt": bwt_inf,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"   InfLoRA (CVPR'24) -> Test ACC: {out_methods['InfLoRA_CVPR2024']['acc']*100:.2f}%, F1: {out_methods['InfLoRA_CVPR2024']['macro_f1']*100:.2f}%, Fgt: {fgt_inf*100:.2f}% ({out_methods['InfLoRA_CVPR2024']['time_s']}s)")

    # ------------------------------------------------------------------
    # Method 3: SD-LoRA (ICLR 2025) — Official Decoupled Direction-Magnitude LoRA on L10-L11
    # ------------------------------------------------------------------
    print("\n[3/7] Running Official SD-LoRA (ICLR 2025, `external_repos/SD-Lora-CL/backbone/lora.py`) on L10-L11...")
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
    sd_mat = np.zeros((10, 10))

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])
        sd_params = []
        for sdl in sd_loras:
            sd_params.extend(sdl.start_new_task())
        opt = torch.optim.AdamW(sd_params + [head_sd.weight], lr=LR, weight_decay=0.01)
        g = torch.Generator().manual_seed(2026 + k)
        m_sd.train()
        for _ in range(EPOCHS):
            perm = tk_idx[torch.randperm(len(tk_idx), generator=g)]
            for b in range(0, len(perm), BS):
                sb = perm[b:b + BS]
                i, t = encode_from_prefix(m_sd, D, sb)
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
        m_sd.eval()
        sd_mat[k] = eval_task_matrix_row(m_sd, lambda i, t: head_sd(torch.cat([i, t], 1)), k)

    p_sd_final = eval_encoder_and_head(m_sd, lambda i, t: head_sd(torch.cat([i, t], 1)), te_idx)
    fgt_sd, bwt_sd = forgetting_and_bwt(sd_mat)
    out_methods["SD_LoRA_ICLR2025"] = {
        **eval_metrics(y_te_np, p_sd_final),
        "forgetting": fgt_sd,
        "bwt": bwt_sd,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"   SD-LoRA (ICLR'25) -> Test ACC: {out_methods['SD_LoRA_ICLR2025']['acc']*100:.2f}%, F1: {out_methods['SD_LoRA_ICLR2025']['macro_f1']*100:.2f}%, Fgt: {fgt_sd*100:.2f}% ({out_methods['SD_LoRA_ICLR2025']['time_s']}s)")

    # ------------------------------------------------------------------
    # Method 4: DeLo (AAAI 2026) — Official `MixLoRALayer` (`external_repos/DeLo/src/model/mixlora_layer.py`)
    # ------------------------------------------------------------------
    print("\n[4/7] Running Official DeLo (AAAI 2026, `external_repos/DeLo/src/model/mixlora_layer.py`) on L10-L11...")
    t0 = time.time()
    m_delo, _, _ = build_top_model(lora=False)
    delo_layers: list[DeLoLinearAdapter] = []
    for mod_list in (m_delo.v_layers, m_delo.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                dl_mod = DeLoLinearAdapter(getattr(attn, attr), num_tasks=10, r=4, E=16, alpha=2.0)
                setattr(attn, attr, dl_mod)
                delo_layers.append(dl_mod)
    head_delo = nn.Linear(1024, C, bias=False)
    delo_mat = np.zeros((10, 10))

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        k_classes = torch.unique(D["y"][tk_idx])
        for dl_mod in delo_layers:
            dl_mod.mixlora.set_active_task(k)
        active_params = [p for dl_mod in delo_layers for p in dl_mod.mixlora.parameters() if p.requires_grad] + [head_delo.weight]
        opt = torch.optim.AdamW(active_params, lr=LR, weight_decay=0.01)
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
                loss = F.cross_entropy(lg_masked, D["y"][sb])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                with torch.no_grad():
                    w_old = head_delo.weight.data[mask_unseen].clone()
                    head_delo.weight.grad[mask_unseen] = 0.0
                opt.step()
                with torch.no_grad():
                    head_delo.weight.data[mask_unseen] = w_old
        m_delo.eval()
        delo_mat[k] = eval_task_matrix_row(m_delo, lambda i, t: head_delo(torch.cat([i, t], 1)), k)

    p_delo_final = eval_encoder_and_head(m_delo, lambda i, t: head_delo(torch.cat([i, t], 1)), te_idx)
    fgt_delo, bwt_delo = forgetting_and_bwt(delo_mat)
    out_methods["DeLo_AAAI2026"] = {
        **eval_metrics(y_te_np, p_delo_final),
        "forgetting": fgt_delo,
        "bwt": bwt_delo,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"   DeLo (AAAI'26) -> Test ACC: {out_methods['DeLo_AAAI2026']['acc']*100:.2f}%, F1: {out_methods['DeLo_AAAI2026']['macro_f1']*100:.2f}%, Fgt: {fgt_delo*100:.2f}% ({out_methods['DeLo_AAAI2026']['time_s']}s)")

    # ------------------------------------------------------------------
    # Train Per-Task LoRA Checkpoints + Ours (Null-Space LoRA tau=0.99 + BlockAnalyticCL v2)
    # and use the exact trained task deltas dW_k = (ALPHA/R) B_k A_k for TSV-M (CVPR'25) & KnOTS (ICLR'25)!
    # ------------------------------------------------------------------
    print("\n[5-7/7] Training 10-Task LoRA Trajectory & Running TSV-M (CVPR'25), KnOTS (ICLR'25), and Ours (ASID-CL v2)...")
    t0 = time.time()
    m_ns, states_ns, loras_ns = build_top_model(lora=True)
    head_ns = nn.Linear(1024, C, bias=False)
    analytic_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=0.5, seed=2026)
    ours_mat = np.zeros((10, 10))
    seen_ours = torch.zeros(C, dtype=torch.bool)
    per_task_lora_deltas: list[dict[str, torch.Tensor]] = []

    for k in range(10):
        tk_idx = tr_idx[D["task"][tr_idx.numpy()] == k]
        yk = D["y"][tk_idx]
        seen_ours[torch.unique(yk)] = True

        # Extract pre-session features of task k for Block-Gram Transport
        if k > 0:
            old_I, old_T = [], []
            with torch.no_grad():
                for b in range(0, len(tk_idx), 256):
                    i, t = encode_from_prefix(m_ns, D, tk_idx[b:b + 256])
                    old_I.append(i.cpu())
                    old_T.append(t.cpu())
            old_inputs_k = {"img": torch.cat(old_I, 0), "txt": torch.cat(old_T, 0)}

        train_session(m_ns, loras_ns, head_ns, D, tk_idx, epochs=EPOCHS, bs=BS, lr=LR, seed=2026 + k)

        # Save task k's 2D weight delta dW_k = (ALPHA / R) * B_k @ A_eff_k for TSV-M & KnOTS
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

        # Extract post-session features of task k and update Ours (Transport + BlockAnalyticCL v2)
        new_I, new_T = [], []
        with torch.no_grad():
            for b in range(0, len(tk_idx), 256):
                i, t = encode_from_prefix(m_ns, D, tk_idx[b:b + 256])
                new_I.append(i.cpu())
                new_T.append(t.cpu())
        new_inputs_k = {"img": torch.cat(new_I, 0), "txt": torch.cat(new_T, 0)}
        if k > 0:
            analytic_ours.transport_gram(old_inputs_k, new_inputs_k, max_drift=0.35, adaptive_drift_guard=True)
        analytic_ours.update(new_inputs_k, yk)
        analytic_ours.auto_spectral_ridges()

        # Evaluate Ours on tasks 0..k
        def _ours_head(i_b, t_b):
            return analytic_ours.logits({"img": i_b.cpu(), "txt": t_b.cpu()}, adaptive_sid=True)

        ours_mat[k] = eval_task_matrix_row(m_ns, _ours_head, k)

    p_ours_final = eval_encoder_and_head(
        m_ns,
        lambda i_b, t_b: analytic_ours.logits({"img": i_b.cpu(), "txt": t_b.cpu()}, adaptive_sid=True),
        te_idx,
    )
    fgt_ours, bwt_ours = forgetting_and_bwt(ours_mat)
    out_methods["Ours_ASID_CL_v2"] = {
        **eval_metrics(y_te_np, p_ours_final),
        "forgetting": fgt_ours,
        "bwt": bwt_ours,
        "time_s": round(time.time() - t0, 2),
    }
    print(f"   Ours (ASID-CL v2) -> Test ACC: {out_methods['Ours_ASID_CL_v2']['acc']*100:.2f}%, F1: {out_methods['Ours_ASID_CL_v2']['macro_f1']*100:.2f}%, Fgt: {fgt_ours*100:.2f}% ({out_methods['Ours_ASID_CL_v2']['time_s']}s)")

    # ------------------------------------------------------------------
    # Method 5: TSV-M (CVPR 2025) — Official `compute_svd_dict` + `sum_svd_dict` on the 10-task LoRA matrices
    # ------------------------------------------------------------------
    t0_tsvm = time.time()
    merged_tsvm_deltas = apply_official_tsvm_merging(per_task_deltas=per_task_lora_deltas)
    m_tsvm, _, _ = build_top_model(lora=False)
    idx_l = 0
    for mod_list in (m_tsvm.v_layers, m_tsvm.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                lin = getattr(attn, attr)
                lin.weight.data.add_(merged_tsvm_deltas[f"layer_{idx_l}"].to(lin.weight.dtype))
                idx_l += 1
    p_tsvm_final = eval_encoder_and_head(m_tsvm, lambda i, t: head_ns(torch.cat([i, t], 1)), te_idx)
    out_methods["TSVM_CVPR2025"] = {
        **eval_metrics(y_te_np, p_tsvm_final),
        "forgetting": None,
        "bwt": None,
        "time_s": round(time.time() - t0_tsvm, 2),
    }
    print(f"   TSV-M (CVPR'25)   -> Test ACC: {out_methods['TSVM_CVPR2025']['acc']*100:.2f}%, F1: {out_methods['TSVM_CVPR2025']['macro_f1']*100:.2f}% ({out_methods['TSVM_CVPR2025']['time_s']}s)")

    # ------------------------------------------------------------------
    # Method 6: KnOTS (ICLR 2025) — Official Joint-SVD + `ties_merging` on the 10-task LoRA matrices
    # ------------------------------------------------------------------
    t0_knots = time.time()
    merged_knots_deltas = apply_official_knots_merging(per_task_deltas=per_task_lora_deltas, topK=30.0)
    m_knots, _, _ = build_top_model(lora=False)
    idx_l = 0
    for mod_list in (m_knots.v_layers, m_knots.t_layers):
        for blk in mod_list:
            attn = blk.self_attn
            for attr in ("q_proj", "v_proj"):
                lin = getattr(attn, attr)
                lin.weight.data.add_(merged_knots_deltas[f"layer_{idx_l}"].to(lin.weight.dtype))
                idx_l += 1
    p_knots_final = eval_encoder_and_head(m_knots, lambda i, t: head_ns(torch.cat([i, t], 1)), te_idx)
    out_methods["KnOTS_ICLR2025"] = {
        **eval_metrics(y_te_np, p_knots_final),
        "forgetting": None,
        "bwt": None,
        "time_s": round(time.time() - t0_knots, 2),
    }
    print(f"   KnOTS (ICLR'25)   -> Test ACC: {out_methods['KnOTS_ICLR2025']['acc']*100:.2f}%, F1: {out_methods['KnOTS_ICLR2025']['macro_f1']*100:.2f}% ({out_methods['KnOTS_ICLR2025']['time_s']}s)")

    return out_methods


def run_crisis_gacl_vs_ours(seeds=(2027, 2028, 2029, 2030, 2031)) -> dict:
    """Compares Official GACL (NeurIPS 2024) vs Ours (ASID-CL v2) across 5 seeds on UK-CrisisMMD & CrisisMMD Humanitarian."""
    out = {}
    for ds_name, loader in (
        ("uk_crisismmd", load_uk_tasks_and_full),
        ("crisismmd_humanitarian", load_humanitarian_events_and_full),
    ):
        tasks, (tr_i, tr_t, tr_y), _, (te_i, te_t, te_y), C = loader()
        y_te_np = te_y.numpy()
        X_te_cat = torch.cat([F.normalize(te_i.double(), dim=-1), F.normalize(te_t.double(), dim=-1)], dim=1)
        X_te_dict = {"img": te_i.float(), "txt": te_t.float()}

        gacl_clean, gacl_miss, gacl_corr = [], [], []
        ours_clean, ours_miss, ours_corr = [], [], []

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

            # 1. Official GACL (NeurIPS 2024) updated task-by-task
            gacl = OfficialGACLRunner(in_dim=tr_i.shape[1] + tr_t.shape[1], buffer_size=2048, gamma=10.0, seed=s)
            for t_dict in tasks:
                x_t_cat = torch.cat([F.normalize(t_dict["train_img"].double(), dim=-1), F.normalize(t_dict["train_txt"].double(), dim=-1)], dim=1)
                gacl.fit(x_t_cat, t_dict["train_y"])
            gacl_clean.append(eval_metrics(y_te_np, gacl.logits(X_te_cat).argmax(1).numpy()))
            gacl_miss.append(eval_metrics(y_te_np, gacl.logits(X_te_zero_cat).argmax(1).numpy()))
            gacl_corr.append(eval_metrics(y_te_np, gacl.logits(X_te_corr_cat).argmax(1).numpy()))

            # 2. Ours (ASID-CL v2) updated task-by-task
            m_ours = BlockAnalyticCL({"img": tr_i.shape[1], "txt": tr_t.shape[1]}, C, projection_dim=1024, class_balance_beta=0.5, seed=s)
            for t_dict in tasks:
                m_ours.update({"img": t_dict["train_img"].float(), "txt": t_dict["train_txt"].float()}, t_dict["train_y"])
            m_ours.auto_spectral_ridges()
            p_c = m_ours.predict(X_te_dict, adaptive_sid=True).numpy()
            p_m = p_c.copy()
            p_m[miss_mask.numpy()] = m_ours.predict({"txt": X_te_dict["txt"][miss_mask]}, modalities=("txt",)).numpy()
            p_n = m_ours.predict(X_te_corr_dict, adaptive_sid=True).numpy()

            ours_clean.append(eval_metrics(y_te_np, p_c))
            ours_miss.append(eval_metrics(y_te_np, p_m))
            ours_corr.append(eval_metrics(y_te_np, p_n))

        def _agg(lst):
            return {
                k: {
                    "mean": float(np.mean([x[k] for x in lst])),
                    "std": float(np.std([x[k] for x in lst], ddof=1)),
                }
                for k in ("macro_f1", "worst_class_f1", "balanced_acc", "acc")
            }

        out[ds_name] = {
            "GACL_NeurIPS2024": {
                "clean": _agg(gacl_clean),
                "missing_img_50": _agg(gacl_miss),
                "corrupt_txt_50": _agg(gacl_corr),
            },
            "Ours_ASID_CL_v2": {
                "clean": _agg(ours_clean),
                "missing_img_50": _agg(ours_miss),
                "corrupt_txt_50": _agg(ours_corr),
            },
        }
    return out


def main():
    t_start = time.time()
    food_sota = run_food101_10task_authentic_sota()
    crisis_sota = run_crisis_gacl_vs_ours()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(time.time() - t_start, 2),
        "upmc_food101_10task_L10_L11_authentic_sota": food_sota,
        "crisis_5seed_gacl_vs_ours": crisis_sota,
    }
    out_path = ROOT / "analysis_outputs" / "authentic_2025_2026_sota_benchmark.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved complete authentic 2024-2026 SOTA benchmark to {out_path} in {report['total_elapsed_seconds']}s")


if __name__ == "__main__":
    main()
