"""
UK-CrisisMMD Cached Embedding Dataset.
"""

from typing import List

from .base import BaseEmbeddingDataset


class UKCrisisMMDDataset(BaseEmbeddingDataset):
    """
    UK-CrisisMMD Dataset: 6 disaster events, pre-extracted CLIP embeddings.

    Mỗi event tạo thành 1 task trong Continual Learning stream.
    Cache format: {event_name}_{split}.npz
    """

    # Thứ tự stream mặc định (theo thời gian sự kiện)
    DEFAULT_TASKS = [
        "event_1",
        "event_2",
        "event_3",
        "event_4",
        "event_5",
        "event_6",
    ]

    def __init__(self, cache_dir: str, embedding_dim: int = 512, device: str = "cpu"):
        super().__init__(cache_dir, embedding_dim, device)

    def get_task_names(self) -> List[str]:
        return self.DEFAULT_TASKS.copy()

    def get_cache_filename(self, task_name: str, split: str) -> str:
        return f"{task_name}_{split}.npz"
