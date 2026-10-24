"""
NUFILT: Null-Space Filtered Model Merging with Residual Plasticity (ICLR 2026).
"""

import math
from typing import List
import torch
import torch.nn.functional as F
from .base import BaseModelMerger


class NUFILTMerger(BaseModelMerger):
    """
    Toán tử NUFILT: Chiếu delta mới lên Null-Space của tri thức cũ và
    bù đắp độ dẻo (Plasticity) thông qua xấp xỉ hạng thấp LoRA.
    """
    def __init__(self, alpha: float = 0.5, lora_rank: int = 8, lora_scale: float = 0.3):
        super().__init__("NUFILT")
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.lora_scale = lora_scale

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

            # Phân rã SVD để tìm không gian con chiếm ưu thế
            u, s, vh = torch.linalg.svd(old_m, full_matrices=False)
            cum = s.cumsum(0)
            k = min(int(torch.searchsorted(cum, self.alpha * cum[-1])) + 1, s.numel())

            # 1. Null-space projection: loại bỏ hình chiếu lên U_k
            u_k = u[:, :k]
            new_null = new_m - u_k @ (u_k.T @ new_m)

            # 2. Plasticity LoRA residual
            u_res, s_res, vh_res = torch.linalg.svd(new_null, full_matrices=False)
            r = min(self.lora_rank, s_res.numel())
            lora_residual = (u_res[:, :r] * s_res[:r]) @ vh_res[:r, :]

            update = (new_null + self.lora_scale * lora_residual).float().view(-1)[:d_len]

            # 3. Chuẩn hóa độ lớn Frobenius Norm
            n_old = torch.norm(merged)
            n_new = torch.norm(new_d)
            target = (n_old * t + n_new) / (t + 1)
            raw = merged + update
            merged = raw * (target / (torch.norm(raw) + 1e-8))

        return base_weights + merged
