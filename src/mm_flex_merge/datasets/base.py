"""
Base Dataset Abstraction for Cached Embedding Datasets.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from ..utils.io import load_npz_cache


class BaseEmbeddingDataset(ABC):
    """
    Lớp cơ sở trừu tượng cho Dataset dạng Pre-Extracted Embedding Cache.

    Mỗi task/event là một file .npz chứa img_emb, txt_emb, labels.
    Subclasses chỉ cần define danh sách tasks và đường dẫn cache.
    """

    def __init__(self, cache_dir: str, embedding_dim: int = 512, device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.embedding_dim = embedding_dim
        self.device = device

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

    @abstractmethod
    def get_task_names(self) -> List[str]:
        """Trả về danh sách tên task theo thứ tự stream."""
        pass

    @abstractmethod
    def get_cache_filename(self, task_name: str, split: str) -> str:
        """Trả về tên file cache .npz cho task và split (train/test)."""
        pass

    def load_task(
        self,
        task_name: str,
        split: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load embedding cache cho 1 task.

        Returns
        -------
        img_emb : torch.Tensor [N, D]
        txt_emb : torch.Tensor [N, D]
        labels  : torch.Tensor [N]
        """
        filename = self.get_cache_filename(task_name, split)
        cache_path = self.cache_dir / filename
        data = load_npz_cache(str(cache_path), device=self.device)
        return data["img_emb"], data["txt_emb"], data["labels"]

    def load_stream(
        self,
        split: str = "train",
        task_order: Optional[List[str]] = None,
    ) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Load toàn bộ task stream theo thứ tự.

        Returns
        -------
        List of (task_name, img_emb, txt_emb, labels)
        """
        tasks = task_order or self.get_task_names()
        stream = []
        for name in tasks:
            img, txt, y = self.load_task(name, split)
            stream.append((name, img, txt, y))
        return stream
