"""
Scientific Evaluation Metrics Protocol for Continual Learning & Model Merging.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class ContinualMergingProtocol:
    """
    Quy chuẩn Đo lường Khoa học cho Continual Model Merging:
    - R_{i, j}: Ma trận hiệu năng đo trên task i sau khi hợp nhất task j.
    - Final Average Accuracy & Macro-F1.
    - Backward Transfer (BWT): Đo lường mức độ duy trì tri thức quá khứ.
    - Forgetting Metric: Tỷ lệ quên lãng trung bình của các task trước.
    - Plasticity Index: Khả năng tiếp thu tri thức mới của task hiện tại.
    """
    @staticmethod
    def evaluate_predictions(targets: np.ndarray, predictions: np.ndarray) -> Tuple[float, float]:
        acc = float(accuracy_score(targets, predictions) * 100.0)
        macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0) * 100.0)
        return acc, macro_f1

    @staticmethod
    def compute_bwt(single_task_accs: List[float], final_merged_accs: List[float]) -> float:
        """
        BWT = (1 / (T - 1)) * sum_{i=1}^{T-1} (R_{i, T} - R_{i, i})
        BWT >= 0: Chuyển giao tích cực (Positive Transfer).
        BWT < 0: Suy hao quá khứ (Catastrophic Forgetting).
        """
        T = len(single_task_accs)
        if T <= 1:
            return 0.0
        diffs = [final_merged_accs[i] - single_task_accs[i] for i in range(T - 1)]
        return float(np.mean(diffs))

    @staticmethod
    def compute_forgetting(single_task_accs: List[float], final_merged_accs: List[float]) -> float:
        """
        Forgetting = max(0, -BWT)
        """
        bwt = ContinualMergingProtocol.compute_bwt(single_task_accs, final_merged_accs)
        return float(-bwt if bwt < 0 else 0.0)

    @staticmethod
    def compute_plasticity(single_task_accs: List[float], final_merged_accs: List[float]) -> float:
        """
        Tỷ số bảo toàn độ dẻo của task cuối: R_{T, T_merged} / R_{T, single}
        """
        if len(single_task_accs) == 0:
            return 1.0
        return float(final_merged_accs[-1] / (single_task_accs[-1] + 1e-8))
