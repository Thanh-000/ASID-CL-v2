"""
Deterministic Seeding Utilities for Reproducible Experiments.
"""

import random
import numpy as np
import torch


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Thiết lập seed toàn cục cho tái hiện hoàn toàn trên CPU.

    Đặt seed cho: Python random, NumPy, PyTorch CPU.
    Trên CUDA (nếu có) sẽ bật cudnn.deterministic và tắt cudnn.benchmark.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
