"""
NPZ Cache I/O Helpers for Pre-Extracted CLIP Embeddings.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def load_npz_cache(
    cache_path: str,
    keys: Tuple[str, ...] = ("img_emb", "txt_emb", "labels"),
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load pre-extracted embeddings từ file .npz và chuyển về PyTorch tensors.

    Parameters
    ----------
    cache_path : str
        Đường dẫn tuyệt đối hoặc tương đối tới file .npz.
    keys : tuple of str
        Tên các array cần đọc từ .npz (mặc định: img_emb, txt_emb, labels).
    device : str
        Device đích (mặc định: "cpu").

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping từ tên key sang tensor tương ứng.
    """
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")

    result = {}
    with np.load(str(path), allow_pickle=False) as data:
        for k in keys:
            if k not in data:
                raise KeyError(f"Key '{k}' not found in {path}. Available: {list(data.keys())}")
            arr = data[k]
            dtype = torch.long if arr.dtype in (np.int32, np.int64) else torch.float32
            result[k] = torch.tensor(arr, dtype=dtype, device=device)

    return result


def save_npz_cache(
    cache_path: str,
    **arrays: np.ndarray,
) -> None:
    """
    Lưu pre-extracted embeddings ra file .npz.

    Parameters
    ----------
    cache_path : str
        Đường dẫn đích. Thư mục cha sẽ được tạo tự động.
    **arrays : np.ndarray
        Mapping tên → numpy array cần lưu.
    """
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)
