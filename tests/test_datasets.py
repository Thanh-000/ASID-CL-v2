"""
Unit Tests for Dataset Loaders and Utility Modules.
"""

import pytest
import tempfile
from pathlib import Path

import numpy as np
import torch

from mm_flex_merge.utils.seed import set_deterministic_seed
from mm_flex_merge.utils.io import load_npz_cache, save_npz_cache


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

class TestDeterministicSeed:
    def test_reproducibility(self):
        """Same seed must produce same random values."""
        set_deterministic_seed(123)
        a = torch.randn(10)
        set_deterministic_seed(123)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        set_deterministic_seed(1)
        a = torch.randn(10)
        set_deterministic_seed(2)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# NPZ I/O
# ---------------------------------------------------------------------------

class TestNPZIO:
    def test_save_and_load_round_trip(self, tmp_path):
        cache_file = str(tmp_path / "test_cache.npz")
        img = np.random.randn(20, 512).astype(np.float32)
        txt = np.random.randn(20, 512).astype(np.float32)
        labels = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)

        save_npz_cache(cache_file, img_emb=img, txt_emb=txt, labels=labels)
        loaded = load_npz_cache(cache_file)

        assert torch.allclose(loaded["img_emb"], torch.tensor(img))
        assert torch.allclose(loaded["txt_emb"], torch.tensor(txt))
        assert torch.equal(loaded["labels"], torch.tensor(labels, dtype=torch.long))

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_npz_cache("/nonexistent/path.npz")

    def test_load_missing_key_raises(self, tmp_path):
        cache_file = str(tmp_path / "partial.npz")
        np.savez(cache_file, img_emb=np.zeros((5, 512)))
        with pytest.raises(KeyError):
            load_npz_cache(cache_file, keys=("img_emb", "txt_emb", "labels"))

    def test_save_creates_parent_dirs(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c" / "cache.npz")
        save_npz_cache(deep_path, data=np.zeros(10))
        assert Path(deep_path).exists()
