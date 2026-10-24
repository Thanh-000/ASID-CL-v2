"""
Task Arithmetic and Simple Weight Averaging (SWA) Mergers.
"""

from typing import List
import torch
from .base import BaseModelMerger


class TaskArithmeticMerger(BaseModelMerger):
    """
    Task Arithmetic (Ilharco et al., 2023).
    w_merged = w_base + scaling_factor * sum(Delta_t)
    """
    def __init__(self, scaling_factor: float = 0.3):
        super().__init__("Task_Arithmetic")
        self.scaling_factor = scaling_factor

    def merge(self, base_weights: torch.Tensor, task_deltas: List[torch.Tensor]) -> torch.Tensor:
        if len(task_deltas) == 0:
            return base_weights.clone()
        sum_delta = sum(task_deltas)
        return base_weights + self.scaling_factor * sum_delta


class SWAMerger(BaseModelMerger):
    """
    Simple Weight Averaging (SWA / Uniform Averaging).
    w_merged = w_base + (1 / T) * sum(Delta_t)
    """
    def __init__(self):
        super().__init__("Simple_Weight_Averaging")

    def merge(self, base_weights: torch.Tensor, task_deltas: List[torch.Tensor]) -> torch.Tensor:
        if len(task_deltas) == 0:
            return base_weights.clone()
        avg_delta = sum(task_deltas) / float(len(task_deltas))
        return base_weights + avg_delta
