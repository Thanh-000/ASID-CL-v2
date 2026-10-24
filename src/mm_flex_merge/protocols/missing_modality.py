"""
Missing Modality Stress Test Protocol.
"""

from typing import Dict, List, Tuple, Callable
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


class MissingModalityProtocol:
    """
    Quy chuẩn Khảo sát Tính Vững bền khi Khuyết thiếu Phương thức:
    - Mô phỏng tỷ lệ mất ảnh hiện trường (Missing Image Rates): 0%, 10%, 20%, 30%, 50%.
    - Đo lường đường cong suy giảm (Degradation Curve).
    - Tính toán khoảng cách suy giảm (Delta Robustness Drop).
    """
    @staticmethod
    def run_stress_test(
        eval_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[float, float]],
        test_img: torch.Tensor,
        test_txt: torch.Tensor,
        test_y: torch.Tensor,
        missing_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
        seed: int = 42,
    ) -> Dict[str, float]:
        torch.manual_seed(seed)
        results = {}
        baseline_acc = None

        for ratio in missing_ratios:
            corrupted_img = test_img.clone()
            n_samples = len(corrupted_img)
            n_drop = int(n_samples * ratio)

            if n_drop > 0:
                drop_indices = torch.randperm(n_samples)[:n_drop]
                corrupted_img[drop_indices] = 0.0 # Che mờ ảnh hỏng

            acc, f1 = eval_fn(corrupted_img, test_txt, test_y)
            pct_label = f"missing_{int(ratio * 100)}pct"
            results[f"{pct_label}_acc"] = float(acc)
            results[f"{pct_label}_f1"] = float(f1)

            if ratio == 0.0:
                baseline_acc = acc

        # Tính khoảng cách suy giảm
        if baseline_acc is not None:
            results["delta_drop_30pct"] = float(baseline_acc - results.get("missing_30pct_acc", baseline_acc))
            results["delta_drop_50pct"] = float(baseline_acc - results.get("missing_50pct_acc", baseline_acc))

        return results
