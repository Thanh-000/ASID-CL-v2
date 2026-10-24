"""
Unit Tests for Model Architectures (Heads & Routers).
"""

import pytest
import torch

from mm_flex_merge.models import (
    DualStreamCrossGatedExpert,
    LinearExpert,
    UnimodalExpert,
    DualCentroidRouter,
)


# ---------------------------------------------------------------------------
# DualStreamCrossGatedExpert
# ---------------------------------------------------------------------------

class TestDualStreamCrossGatedExpert:
    @pytest.fixture
    def expert(self):
        return DualStreamCrossGatedExpert(in_features=512, hidden_dim=256, num_classes=3)

    def test_forward_shape(self, expert):
        img = torch.randn(8, 512)
        txt = torch.randn(8, 512)
        out = expert(img, txt)
        assert out.shape == (8, 3)

    def test_forward_no_nan(self, expert):
        img = torch.randn(4, 512)
        txt = torch.randn(4, 512)
        out = expert(img, txt)
        assert not torch.isnan(out).any()

    def test_missing_image_recovery(self, expert):
        """When image embeddings are zeroed, output should still be valid."""
        img = torch.zeros(4, 512)
        txt = torch.randn(4, 512)
        out = expert(img, txt)
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_missing_text_recovery(self, expert):
        """When text embeddings are zeroed, output should still be valid."""
        img = torch.randn(4, 512)
        txt = torch.zeros(4, 512)
        out = expert(img, txt)
        assert out.shape == (4, 3)
        assert not torch.isnan(out).any()

    def test_extract_and_load_flat_params(self, expert):
        flat = expert.extract_flat_parameters()
        assert flat.ndim == 1
        total_params = sum(p.numel() for p in expert.parameters())
        assert flat.numel() == total_params

        # Round-trip: load back and check
        expert.load_flat_parameters(flat)
        flat2 = expert.extract_flat_parameters()
        assert torch.allclose(flat, flat2)


# ---------------------------------------------------------------------------
# LinearExpert
# ---------------------------------------------------------------------------

class TestLinearExpert:
    def test_forward_shape(self):
        expert = LinearExpert(in_features=512, hidden_dim=256, num_classes=3)
        x = torch.randn(8, 512)
        out = expert(x)
        assert out.shape == (8, 3)


# ---------------------------------------------------------------------------
# UnimodalExpert
# ---------------------------------------------------------------------------

class TestUnimodalExpert:
    def test_vision_forward(self):
        expert = UnimodalExpert(modality="vision")
        out = expert(img=torch.randn(4, 512))
        assert out.shape == (4, 3)

    def test_text_forward(self):
        expert = UnimodalExpert(modality="text")
        out = expert(txt=torch.randn(4, 512))
        assert out.shape == (4, 3)

    def test_wrong_modality_raises(self):
        expert = UnimodalExpert(modality="vision")
        with pytest.raises(ValueError):
            expert(img=None, txt=torch.randn(4, 512))

    def test_invalid_modality_init_raises(self):
        with pytest.raises(ValueError):
            UnimodalExpert(modality="audio")


# ---------------------------------------------------------------------------
# DualCentroidRouter
# ---------------------------------------------------------------------------

class TestDualCentroidRouter:
    @pytest.fixture
    def router(self):
        return DualCentroidRouter(num_tasks=3, dim=512)

    def test_route_output_shape(self, router):
        img = torch.randn(8, 512)
        txt = torch.randn(8, 512)
        probs = router.route(img, txt)
        assert probs.shape == (8, 3)

    def test_route_sums_to_one(self, router):
        img = torch.randn(4, 512)
        txt = torch.randn(4, 512)
        probs = router.route(img, txt)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_opt_out_zeroes_probability(self, router):
        """After opt-out, target task probability must be effectively zero."""
        # Update centroids first so routing is non-trivial
        for i in range(3):
            router.update_centroids(i, torch.randn(10, 512), torch.randn(10, 512))

        router.opt_out_task(1)
        img = torch.randn(4, 512)
        txt = torch.randn(4, 512)
        probs = router.route(img, txt)
        assert probs[:, 1].max().item() < 1e-6

    def test_restore_task(self, router):
        router.opt_out_task(1)
        assert router.is_opted_out(1)
        router.restore_task(1)
        assert not router.is_opted_out(1)

    def test_update_centroids(self, router):
        train_img = torch.randn(20, 512)
        train_txt = torch.randn(20, 512)
        router.update_centroids(0, train_img, train_txt)
        # Centroid should be non-zero after update
        assert router.image_centroids[0].norm().item() > 0.0

    def test_route_missing_modality_fallback(self):
        """When one modality is zeroed out, routing must match dedicated unimodal centroid similarity."""
        torch.manual_seed(7)
        router = DualCentroidRouter(num_tasks=3, dim=32, default_bias=0.0)
        for i in range(3):
            router.update_centroids(i, torch.randn(10, 32), torch.randn(10, 32))

        txt_query = router.text_centroids[2:3].clone()
        zero_img = torch.zeros_like(txt_query)
        probs = router.route(zero_img, txt_query)
        assert int(probs.argmax(dim=-1).item()) == 2
