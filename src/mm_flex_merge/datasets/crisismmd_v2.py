"""
CrisisMMD-v2 Cached Embedding Dataset.
"""

from typing import List

from .base import BaseEmbeddingDataset


class CrisisMMDv2Dataset(BaseEmbeddingDataset):
    """
    CrisisMMD-v2 Dataset: 7 disaster events, pre-extracted CLIP embeddings.

    External benchmark dataset for cross-dataset generalization validation.
    Cache format: {event_name}_{split}.npz
    """

    DEFAULT_TASKS = [
        "hurricane_harvey",
        "hurricane_irma",
        "hurricane_maria",
        "mexico_earthquake",
        "iraq_iran_earthquake",
        "california_wildfires",
        "srilanka_floods",
    ]

    def __init__(self, cache_dir: str, embedding_dim: int = 512, device: str = "cpu"):
        super().__init__(cache_dir, embedding_dim, device)

    def get_task_names(self) -> List[str]:
        return self.DEFAULT_TASKS.copy()

    def get_cache_filename(self, task_name: str, split: str) -> str:
        return f"{task_name}_{split}.npz"
