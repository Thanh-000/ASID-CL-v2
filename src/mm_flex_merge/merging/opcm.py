"""
OPCM: Orthogonal Projection for Continual Model Merging (Tang et al., 2025).
"""

import math
from typing import Dict, List, Optional, Set
import torch
import torch.nn.functional as F
from .base import BaseModelMerger


class OPCMMerger(BaseModelMerger):
    """
    Toán tử OPCM: Chiếu vector tham số Delta mới lên không gian trực giao của
    các task trước đó qua phân rã SVD và chuẩn hóa năng lượng Frobenius.
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__("OPCM")
        self.alpha = alpha

    def merge(self, base_weights: torch.Tensor, task_deltas: List[torch.Tensor]) -> torch.Tensor:
        """LEGACY (không trung thực với OPCM gốc): SVD trên vector phẳng reshape vuông.

        Dùng `merge_state_dict` cho mọi thực nghiệm báo cáo.
        """
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

            # Phân rã giá trị suy biến SVD trên ma trận tri thức tích lũy
            u, s, vh = torch.linalg.svd(old_m, full_matrices=False)
            cum = s.cumsum(0)
            rank = min(int(torch.searchsorted(cum, self.alpha * cum[-1])) + 1, s.numel())

            # Chiếu trực giao: Loại bỏ các thành phần nằm trong không gian của task cũ
            coeff = u.T @ new_m @ vh.T
            coeff[:rank, :rank] = 0
            coeff.diagonal().zero_()
            proj = (u @ coeff @ vh).float().view(-1)[:d_len]

            # Chuẩn hóa Frobenius Norm bảo toàn độ lớn
            n_old = torch.norm(merged)
            n_new = torch.norm(new_d)
            target = (n_old * t + n_new) / (t + 1)
            raw = merged + proj
            merged = raw * (target / (torch.norm(raw) + 1e-8))

        return base_weights + merged

    def merge_state_dict(
        self,
        base_state_dict: Dict[str, torch.Tensor],
        task_state_dicts: List[Dict[str, torch.Tensor]],
        linear_weight_keys: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Port trung thực của `OPCMForCLIP.run` (fusion_bench/method/opcm/opcm.py).

        - Trọng số `nn.Linear`: SVD (full_matrices=True) trên merged task vector của
          chính layer đó, xóa khối [:split_rank, :split_rank] của task vector mới.
        - Tham số khác (bias, LayerNorm): cộng trực tiếp.
        - Sau mỗi bước: rescale toàn cục để ||merged_tv|| = trung bình ||tv_i||.

        Ghi chú: code gốc gọi `projected_task_tv.diag().fill_(0)`; `Tensor.diag()`
        trả về bản sao nên lệnh này không có hiệu lực -> đường chéo KHÔNG bị xóa.
        Port này tái hiện đúng hành vi thực tế đó.

        `linear_weight_keys`: tên các trọng số Linear. Mặc định: key 2 chiều kết thúc
        bằng ".weight".
        """
        if len(task_state_dicts) == 0:
            return {k: v.clone() for k, v in base_state_dict.items()}

        keys = list(base_state_dict.keys())
        if linear_weight_keys is None:
            linear_weight_keys = {
                k for k in keys if k.endswith(".weight") and base_state_dict[k].ndim == 2
            }

        def global_norm(sd: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.linalg.norm(torch.cat([sd[k].flatten() for k in keys]))

        pre = {k: base_state_dict[k].double() for k in keys}
        merged_tv = {k: task_state_dicts[0][k].double() - pre[k] for k in keys}
        all_norms = [global_norm(merged_tv)]
        previous_lambda = 1.0

        for task_sd in task_state_dicts[1:]:
            task_tv = {k: task_sd[k].double() - pre[k] for k in keys}
            all_norms.append(global_norm(task_tv))
            avg_norm = torch.stack(all_norms).mean()

            new_tv = {}
            for k in keys:
                if k in linear_weight_keys:
                    u, s, vh = torch.linalg.svd(merged_tv[k], full_matrices=True)
                    v = vh.T
                    split_rank = int((s.cumsum(dim=0) / s.sum() > self.alpha).float().argmax().item())
                    projected = u.T @ task_tv[k] @ v
                    projected[:split_rank, :split_rank] = 0
                    cleaned = u @ projected @ v.T
                    new_tv[k] = previous_lambda * merged_tv[k] + cleaned
                else:
                    new_tv[k] = previous_lambda * merged_tv[k] + task_tv[k]

            current_norm = global_norm(new_tv)
            previous_lambda = float(current_norm / avg_norm)
            merged_tv = {k: new_tv[k] * (avg_norm / current_norm) for k in keys}

        return {
            k: (pre[k] + merged_tv[k]).to(base_state_dict[k].dtype) for k in keys
        }
