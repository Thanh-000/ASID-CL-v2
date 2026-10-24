"""
Statistical Significance and Hypothesis Testing Protocol.
"""

from typing import Dict, List, Tuple
import numpy as np
import scipy.stats as stats


class StatisticalSignificanceProtocol:
    """
    Quy chuẩn Kiểm định Thống kê Khoa học Core A* (NeurIPS/ICLR/CVPR):
    - Tính toán kỳ vọng Mean +/- Sai số chuẩn Std qua 5 Random Seeds.
    - Chạy Paired t-test hai phía giữa phương pháp đề xuất và từng baseline.
    - Báo cáo t-statistic, p-value và mức ý nghĩa thống kê (alpha = 0.05, 0.01, 0.001).
    - Tính kích thước hiệu ứng Cohen's d (Effect Size).
    """
    @staticmethod
    def aggregate_runs(values: List[float]) -> Tuple[float, float]:
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        return mean_val, std_val

    @staticmethod
    def paired_t_test(proposed_values: List[float], baseline_values: List[float]) -> Dict[str, float]:
        diff = np.asarray(proposed_values, dtype=np.float64) - np.asarray(baseline_values, dtype=np.float64)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

        if std_diff < 1e-12:
            if abs(mean_diff) <= 1e-12:
                t_stat, p_val, cohen_d = 0.0, 1.0, 0.0
            else:
                t_stat = float(np.sign(mean_diff) * np.inf)
                p_val = 0.0
                cohen_d = float(mean_diff / 1e-8)
        else:
            t_res, p_res = stats.ttest_rel(proposed_values, baseline_values)
            t_stat = float(t_res)
            p_val = float(p_res)
            cohen_d = float(mean_diff / (std_diff + 1e-8))

        return {
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohen_d,
            "statistically_significant_p05": bool(p_val < 0.05),
            "statistically_significant_p01": bool(p_val < 0.01),
            "statistically_significant_p001": bool(p_val < 0.001),
        }
