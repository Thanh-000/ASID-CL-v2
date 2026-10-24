"""
Base class for Continual Model Merging algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import torch


class BaseModelMerger(ABC):
    """Lớp cơ sở trừu tượng cho tất cả toán tử Continual Model Merging."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def merge(self, base_weights: torch.Tensor, task_deltas: List[torch.Tensor]) -> torch.Tensor:
        """Hợp nhất các vector tham số phẳng."""
        pass

    def merge_state_dict(
        self,
        base_state_dict: dict[str, torch.Tensor],
        task_state_dicts: List[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """
        Hợp nhất trực tiếp trên Layer-wise State Dict (từng tensor ma trận 2D tự nhiên).
        Mặc định có thể được ghi đè bởi các lớp con kế thừa.
        """
        raise NotImplementedError(f"{self.name} does not implement merge_state_dict yet.")

