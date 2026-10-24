"""Exactness properties of BlockAnalyticCL (the Direction-D core)."""

import pytest
import torch
import torch.nn.functional as F

from mm_flex_merge.models import BlockAnalyticCL

DIMS = {"img": 16, "txt": 12}
C, P = 4, 32


def _data(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = {"img": torch.randn(n, DIMS["img"], generator=g), "txt": torch.randn(n, DIMS["txt"], generator=g)}
    y = torch.randint(0, C, (n,), generator=g)
    return x, y


def _model(beta=0.0, dims=DIMS):
    return BlockAnalyticCL(dims, C, projection_dim=P, ridge=0.1, class_balance_beta=beta, seed=7)


def _tasks():
    return [_data(n, s) for n, s in [(40, 1), (25, 2), (60, 3)]]


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_block_solution_equals_dedicated_single_modality_model(beta):
    full, ded = _model(beta), _model(beta, {"txt": DIMS["txt"]})
    for x, y in _tasks():
        full.update(x, y)
        ded.update({"txt": x["txt"]}, y)
    xq, _ = _data(30, 99)
    gap = (full.logits({"txt": xq["txt"]}, ["txt"]) - ded.logits({"txt": xq["txt"]})).abs().max()
    assert gap < 1e-9


@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_delete_equals_retrain(beta):
    tasks = _tasks()
    model, retrain = _model(beta), _model(beta)
    for x, y in tasks:
        model.update(x, y)
    for x, y in tasks[1:]:
        retrain.update(x, y)
    # task-level deletion
    model.delete(*tasks[0])
    xq, _ = _data(30, 99)
    assert (model.logits(xq) - retrain.logits(xq)).abs().max() < 1e-8
    # sample-level deletion
    x, y = tasks[1]
    model.delete({k: v[:5] for k, v in x.items()}, y[:5])
    retrain2 = _model(beta)
    retrain2.update({k: v[5:] for k, v in x.items()}, y[5:])
    retrain2.update(*tasks[2])
    assert (model.logits(xq) - retrain2.logits(xq)).abs().max() < 1e-8
    assert torch.equal(model.class_counts, retrain2.class_counts)


def test_order_invariance():
    tasks = _tasks()
    a, b = _model(0.5), _model(0.5)
    for x, y in tasks:
        a.update(x, y)
    for x, y in reversed(tasks):
        b.update(x, y)
    xq, _ = _data(30, 99)
    assert (a.logits(xq) - b.logits(xq)).abs().max() < 1e-9


def test_zero_beta_matches_closed_form_ridge():
    model = _model(0.0)
    xs, ys = zip(*_tasks())
    for x, y in zip(xs, ys):
        model.update(x, y)
    phi = torch.cat([model.features(x) for x in xs])
    Y = F.one_hot(torch.cat(ys), C).double()
    G = phi.T @ phi
    lam = 0.1 * torch.trace(G) / G.shape[0]
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), phi.T @ Y)
    assert (model.solve() - W).abs().max() < 1e-9


def test_class_balance_upweights_minority():
    model = _model(1.0)
    x, _ = _data(10, 5)
    model.update(x, torch.tensor([0] * 8 + [1] * 2))
    w = model.class_weights()
    assert w[1] > w[0] and w[2] == 0 and w[3] == 0
    assert float(w[:2].mean()) == pytest.approx(1.0)


def test_over_deletion_and_unknown_modality_rejected():
    model = _model()
    x, y = _data(5, 1)
    model.update(x, y)
    with pytest.raises(ValueError):
        model.delete(*_data(50, 2))
    with pytest.raises(ValueError):
        model.solve(["audio"])


def test_block_schur_complement_equals_direct_solve():
    model = _model(0.5)
    for x, y in _tasks():
        model.update(x, y)
    w_direct = model.solve()
    w_schur, w_img, w_txt = model.solve_schur()
    assert (w_direct - w_schur).abs().max() < 1e-12
    assert (model.solve(["img"]) - w_img).abs().max() < 1e-12
    assert (model.solve(["txt"]) - w_txt).abs().max() < 1e-12


def test_asymmetric_ridge_and_partial_modality_update():
    model = BlockAnalyticCL(
        DIMS, C, projection_dim=P, ridge={"img": 0.25, "txt": 1.0}, class_balance_beta=0.5, schur_coupling_rho=0.8, seed=7
    )
    x1, y1 = _data(30, 1)
    x2, y2 = _data(30, 2)
    # Stage 1: only image present (MIL Stage 1)
    model.update({"img": x1["img"]}, y1)
    # Stage 2: both image and text present (MIL Stage 2)
    model.update(x2, y2)
    xq, _ = _data(20, 99)
    # Auto-detects available modalities when only one is passed at test time
    lg_both = model.logits(xq)
    lg_img = model.logits({"img": xq["img"]})
    lg_txt = model.logits({"txt": xq["txt"]})
    assert lg_both.shape == (20, C)
    assert lg_img.shape == (20, C)
    assert lg_txt.shape == (20, C)


def test_schur_innovation_decomposition_and_auto_spectral_ridges():
    model = _model(0.5)
    for x, y in _tasks():
        model.update(x, y)
    ridges = model.auto_spectral_ridges(gamma_base=0.35)
    assert "img" in ridges and "txt" in ridges
    assert 0.02 <= ridges["img"] <= 5.0 and 0.02 <= ridges["txt"] <= 5.0

    w_1_o, w_2_o, dw_1_s, dw_2_s, w_both_sid = model.schur_innovation_decomposition()
    w_schur, _, _ = model.solve_schur()
    assert (w_both_sid - w_schur).abs().max() < 1e-12

    xq, _ = _data(16, 123)
    lg_sid = model.logits(xq, adaptive_sid=True)
    assert lg_sid.shape == (16, C)


def test_block_gram_pushforward_transport():
    model = _model(0.5)
    x1, y1 = _data(40, 1)
    model.update(x1, y1)
    x2_old, y2 = _data(40, 2)
    # Apply slight linear drift on input features
    x2_new = {"img": x2_old["img"] * 1.05, "txt": x2_old["txt"] * 0.95}
    ops = model.transport_gram(x2_old, x2_new)
    assert ops["img"].shape == (P, P) and ops["txt"].shape == (P, P)
    model.update(x2_new, y2)
    assert model.logits(x2_new).shape == (40, C)


def test_transport_gram_drift_guard():
    model = _model(0.5)
    x1, y1 = _data(40, 1)
    model.update(x1, y1)
    gram_before = model.gram.clone()
    x2_old, _ = _data(40, 2)
    # Apply large non-perturbative drift (naive LoRA collapse regime: delta_rel > 0.35)
    x2_huge_drift = {"img": -x2_old["img"] * 2.0, "txt": -x2_old["txt"] * 2.0}
    ops = model.transport_gram(x2_old, x2_huge_drift, max_drift=0.35, adaptive_drift_guard=True)
    eye = torch.eye(P, dtype=torch.float64)
    assert torch.allclose(ops["img"], eye)
    assert torch.allclose(ops["txt"], eye)
    assert torch.allclose(model.gram, gram_before)


def test_auto_continual_policy():
    p_frozen = BlockAnalyticCL.auto_continual_policy(g1_gain_pts=-0.08, incremental_mode="cil")
    assert p_frozen["regime"] == "frozen_analytic_sid"
    assert p_frozen["lora_enabled"] is False
    assert p_frozen["adaptive_sid"] is True

    p_dil = BlockAnalyticCL.auto_continual_policy(g1_gain_pts=5.42, incremental_mode="dil")
    assert p_dil["regime"] == "dil_vision_nullspace_frozen_text"
    assert p_dil["tau"] == 0.95
    assert p_dil["towers"] == ("vision",)

    p_cil = BlockAnalyticCL.auto_continual_policy(g1_gain_pts=3.50, incremental_mode="cil")
    assert p_cil["regime"] == "cil_dual_nullspace_transport_sid"
    assert p_cil["tau"] == 0.99
    assert p_cil["towers"] == ("vision", "text")


def test_zero_filled_missing_modality_exact_equivalence_in_sid():
    model = _model(0.5)
    x_tr, y_tr = _data(80, 10)
    model.update(x_tr, y_tr)
    model.auto_spectral_ridges()

    x_te, _ = _data(20, 20)
    x_mixed = {"img": x_te["img"].clone(), "txt": x_te["txt"].clone()}
    x_mixed["img"][:8] = 0.0
    x_mixed["txt"][8:14] = 0.0

    lg_sid = model.logits(x_mixed, adaptive_sid=True)
    lg_txt_only = model.logits({"txt": x_te["txt"][:8]}, modalities=("txt",))
    lg_img_only = model.logits({"img": x_te["img"][8:14]}, modalities=("img",))
    assert torch.allclose(lg_sid[:8], lg_txt_only, atol=1e-12)
    assert torch.allclose(lg_sid[8:14], lg_img_only, atol=1e-12)


def test_partial_modality_weighted_gram_math_scaling():
    # Force _use_exact_class_grams = False to exercise _weighted_gram_and_sums
    model = BlockAnalyticCL(
        DIMS, C, projection_dim=P, ridge=0.1, class_balance_beta=0.5, seed=42
    )
    model._use_exact_class_grams = False
    x1, y1 = _data(30, 1)
    x2, y2 = _data(30, 2)
    model.update({"img": x1["img"]}, y1)
    model.update(x2, y2)
    W = model.solve(("img", "txt"))
    assert W.shape == (2 * P, C)
    assert not torch.isnan(W).any()





