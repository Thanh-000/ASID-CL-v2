"""
Unit Tests for Merging Operators.

Tests all merger classes: MMFlexMerger, OPCMMerger, NUFILTMerger,
TaskArithmeticMerger, SWAMerger against the BaseModelMerger contract.
"""

import pytest
import torch

from mm_flex_merge.merging import (
    BaseModelMerger,
    MMFlexMerger,
    OPCMMerger,
    NUFILTMerger,
    TaskArithmeticMerger,
    SWAMerger,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_weights():
    """Base weight vector (512-dim, like flattened CLIP head)."""
    torch.manual_seed(0)
    return torch.randn(512)


@pytest.fixture
def task_deltas():
    """3 task delta vectors for sequential merging."""
    torch.manual_seed(1)
    return [torch.randn(512) for _ in range(3)]


@pytest.fixture
def state_dicts():
    """3 task state dicts mimicking a small 2-layer head."""
    torch.manual_seed(2)
    dicts = []
    for _ in range(3):
        sd = {
            "linear1.weight": torch.randn(64, 128),
            "linear1.bias": torch.randn(64),
            "linear2.weight": torch.randn(3, 64),
            "linear2.bias": torch.randn(3),
        }
        dicts.append(sd)
    return dicts


# ---------------------------------------------------------------------------
# Common contract tests
# ---------------------------------------------------------------------------

ALL_MERGERS = [
    ("MMFlexMerger", MMFlexMerger),
    ("OPCMMerger", OPCMMerger),
    ("NUFILTMerger", NUFILTMerger),
    ("TaskArithmeticMerger", TaskArithmeticMerger),
    ("SWAMerger", SWAMerger),
]


@pytest.mark.parametrize("name,cls", ALL_MERGERS)
def test_merger_inherits_base(name, cls):
    """All mergers must be subclasses of BaseModelMerger."""
    assert issubclass(cls, BaseModelMerger)


@pytest.mark.parametrize("name,cls", ALL_MERGERS)
def test_merge_returns_same_shape(name, cls, base_weights, task_deltas):
    """merge() must return a tensor with the same shape as base_weights."""
    merger = cls()
    result = merger.merge(base_weights, task_deltas)
    assert result.shape == base_weights.shape


@pytest.mark.parametrize("name,cls", ALL_MERGERS)
def test_merge_empty_deltas_returns_base(name, cls, base_weights):
    """merge() with empty task_deltas must return a clone of base_weights."""
    merger = cls()
    result = merger.merge(base_weights, [])
    assert torch.allclose(result, base_weights)


@pytest.mark.parametrize("name,cls", ALL_MERGERS)
def test_merge_no_nan(name, cls, base_weights, task_deltas):
    """merge() output must be free of NaN values."""
    merger = cls()
    result = merger.merge(base_weights, task_deltas)
    assert not torch.isnan(result).any(), f"{name} produced NaN"


# ---------------------------------------------------------------------------
# MMFlexMerger-specific tests
# ---------------------------------------------------------------------------

class TestMMFlexMerger:
    def test_merge_state_dict_keys_preserved(self, state_dicts):
        merger = MMFlexMerger()
        base_sd = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
        merged = merger.merge_state_dict(base_sd, state_dicts)
        assert set(merged.keys()) == set(state_dicts[0].keys())

    def test_merge_state_dict_shapes_preserved(self, state_dicts):
        merger = MMFlexMerger()
        base_sd = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
        merged = merger.merge_state_dict(base_sd, state_dicts)
        for key in merged:
            assert merged[key].shape == state_dicts[0][key].shape, f"Shape mismatch at {key}"

    def test_merge_state_dict_no_nan(self, state_dicts):
        merger = MMFlexMerger()
        base_sd = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
        merged = merger.merge_state_dict(base_sd, state_dicts)
        for key, val in merged.items():
            assert not torch.isnan(val).any(), f"NaN at {key}"

    def test_merge_2d_tensor_shape(self):
        merger = MMFlexMerger()
        old = torch.randn(64, 128)
        new = torch.randn(64, 128)
        result = merger.merge_2d_tensor(old, new, t=1)
        assert result.shape == (64, 128)

    def test_merge_state_dict_empty_returns_base(self, state_dicts):
        merger = MMFlexMerger()
        base_sd = state_dicts[0]
        merged = merger.merge_state_dict(base_sd, [])
        for key in base_sd:
            assert torch.allclose(merged[key], base_sd[key])

    def test_merge_state_dict_base_shift_invariance(self, state_dicts):
        """Merging (W_base + dW_t) relative to W_base must equal W_base + merge(0, dW_t)."""
        torch.manual_seed(99)
        merger = MMFlexMerger()
        zero_base = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}
        nonzero_base = {k: torch.randn_like(v) * 5.0 for k, v in state_dicts[0].items()}
        shifted_tasks = [
            {k: nonzero_base[k] + sd[k] for k in sd}
            for sd in state_dicts
        ]
        merged_zero = merger.merge_state_dict(zero_base, state_dicts)
        merged_shifted = merger.merge_state_dict(nonzero_base, shifted_tasks)
        for key in zero_base:
            assert torch.allclose(merged_shifted[key], nonzero_base[key] + merged_zero[key], atol=1e-5)
