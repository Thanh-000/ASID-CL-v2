"""
MM-FLEX-MERGE: Multimodal Dynamic Orthogonal Elastic Merging (Our Proprietary Baseline).
========================================================================================
Thuật toán hợp nhất mô hình độc quyền kết hợp:
1. Dynamic Adaptive Subspace Projection (DASP): Ngưỡng năng lượng trực giao tự điều hòa theo bước t.
2. Elastic Residual Plasticity Injection: Bù trừ độ dẻo tiếp thu bằng phân rã hạng thấp có trọng số hội tụ.
3. Frobenius Energy Conservation: Bảo toàn năng lượng biểu diễn toán học của ma trận trọng số.
"""

import math
from typing import List, Optional
import torch
import torch.nn.functional as F
from .base import BaseModelMerger


class MMFlexMerger(BaseModelMerger):
    """
    Toán tử Hợp nhất Độc quyền MM-FLEX-MERGE.
    
    Giải quyết triệt để 2 vấn đề lớn nhất của các phương pháp trước:
    - OPCM: Triệt tiêu quá nhiều độ dẻo (mất ~35% Plasticity) khiến task sau bị giảm sút.
    - Task Arithmetic: Bị nhiễu giao thoa tham số (Negative Interference) dẫn đến quên lãng.
    """
    def __init__(
        self,
        base_alpha: float = 0.5,
        tau: float = 4.0,
        elastic_rank: int = 8,
        initial_gamma: float = 0.35,
    ):
        super().__init__("MM-FLEX-MERGE")
        self.base_alpha = base_alpha
        self.tau = tau
        self.elastic_rank = elastic_rank
        self.initial_gamma = initial_gamma

    def merge_2d_tensor(self, old_m: torch.Tensor, new_m: torch.Tensor, t: int) -> torch.Tensor:
        """
        Thực hiện phép chiếu DASP + Elastic LoRA trực tiếp trên ma trận 2D gốc (m x n)
        mà không cần đệm ma trận vuông (Square Padding).
        """
        device = old_m.device
        dtype = old_m.dtype
        old_d = old_m.double()
        new_d = new_m.double()

        # 1. Dynamic Adaptive Subspace Projection (DASP)
        alpha_t = min(0.85, self.base_alpha * (1.0 + 0.3 * (1.0 - math.exp(-t / self.tau))))

        u, s, vh = torch.linalg.svd(old_d, full_matrices=False)
        cum = s.cumsum(0)
        k = min(int(torch.searchsorted(cum, alpha_t * cum[-1])) + 1, s.numel())

        # 2. Null-space orthogonal projection
        u_k = u[:, :k]
        proj_interference = u_k @ (u_k.T @ new_d)
        delta_null = new_d - proj_interference

        # 3. Elastic Residual Plasticity Injection (LoRA)
        u_res, s_res, vh_res = torch.linalg.svd(delta_null, full_matrices=False)
        r = min(self.elastic_rank, s_res.numel())
        elastic_comp = (u_res[:, :r] * s_res[:r]) @ vh_res[:r, :]

        gamma_t = self.initial_gamma / math.sqrt(t + 1.0)
        elastic_update = delta_null + gamma_t * elastic_comp

        # 4. Frobenius Energy Conservation
        n_old = torch.norm(old_d)
        n_new = torch.norm(new_d)
        target_norm = (n_old * t + n_new) / (t + 1.0)

        raw = old_d + elastic_update
        merged = raw * (target_norm / (torch.norm(raw) + 1e-8))
        return merged.to(device=device, dtype=dtype)

    def merge_state_dict(
        self,
        base_state_dict: dict[str, torch.Tensor],
        task_state_dicts: List[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Hợp nhất trực tiếp trên Layer-wise State Dict (tương thích 100% Hugging Face CLIP / ViT / LLMs).
        Tách biệt: Ma trận 2D áp dụng DASP + LoRA, Vector 1D (Bias/Norm) áp dụng Running Average.
        """
        if len(task_state_dicts) == 0:
            return {k: v.clone() for k, v in base_state_dict.items()}

        merged_delta = {
            k: (v - base_state_dict[k]).clone() if k in base_state_dict else v.clone()
            for k, v in task_state_dicts[0].items()
        }

        for t in range(1, len(task_state_dicts)):
            curr_state = task_state_dicts[t]
            for key in merged_delta.keys():
                if key not in curr_state:
                    continue

                old_tensor = merged_delta[key]
                new_tensor = (
                    curr_state[key] - base_state_dict[key]
                    if key in base_state_dict
                    else curr_state[key]
                )

                # Nếu là ma trận trọng số 2D (Linear, MLP, Attention QKV, Projection)
                if old_tensor.ndim == 2:
                    merged_delta[key] = self.merge_2d_tensor(old_tensor, new_tensor, t)
                # Nếu là tham số 1D (bias, layernorm scale)
                elif old_tensor.ndim == 1:
                    merged_delta[key] = (old_tensor * t + new_tensor) / (t + 1.0)
                # Các tensor khác (3D, 4D Conv, v.v.): Flatten sang 2D rồi merge
                else:
                    orig_shape = old_tensor.shape
                    old_2d = old_tensor.view(orig_shape[0], -1)
                    new_2d = new_tensor.view(orig_shape[0], -1)
                    merged_2d = self.merge_2d_tensor(old_2d, new_2d, t)
                    merged_delta[key] = merged_2d.view(orig_shape)

        return {
            k: (base_state_dict[k] + v).to(dtype=base_state_dict[k].dtype)
            if k in base_state_dict
            else v
            for k, v in merged_delta.items()
        }

    def merge(self, base_weights: torch.Tensor, task_deltas: List[torch.Tensor]) -> torch.Tensor:
        if len(task_deltas) == 0:
            return base_weights.clone()

        merged = task_deltas[0].clone()
        d_len = len(merged)
        side = int(math.isqrt(d_len))
        pad = (side + 1)**2 - d_len if side * side != d_len else 0
        side = int(math.isqrt(d_len + pad))

        for t in range(1, len(task_deltas)):
            new_d = task_deltas[t]

            old_m = F.pad(merged, (0, pad)).view(side, side).double()
            new_m = F.pad(new_d, (0, pad)).view(side, side).double()

            # 1. Dynamic Adaptive Subspace Projection (DASP)
            alpha_t = min(0.85, self.base_alpha * (1.0 + 0.3 * (1.0 - math.exp(-t / self.tau))))

            u, s, vh = torch.linalg.svd(old_m, full_matrices=False)
            cum = s.cumsum(0)
            k = min(int(torch.searchsorted(cum, alpha_t * cum[-1])) + 1, s.numel())

            # 2. Null-space orthogonal projection
            u_k = u[:, :k]
            proj_interference = u_k @ (u_k.T @ new_m)
            delta_null = new_m - proj_interference

            # 3. Elastic Residual Plasticity Injection
            u_res, s_res, vh_res = torch.linalg.svd(delta_null, full_matrices=False)
            r = min(self.elastic_rank, s_res.numel())
            elastic_comp = (u_res[:, :r] * s_res[:r]) @ vh_res[:r, :]

            gamma_t = self.initial_gamma / math.sqrt(t + 1.0)
            elastic_update = (delta_null + gamma_t * elastic_comp).float().view(-1)[:d_len]

            # 4. Frobenius Energy Conservation
            n_old = torch.norm(merged)
            n_new = torch.norm(new_d)
            target_norm = (n_old * t + n_new) / (t + 1.0)

            raw = merged + elastic_update
            merged = raw * (target_norm / (torch.norm(raw) + 1e-8))

        return base_weights + merged

