"""Unit tests for Stage 1 Null-Space Uncentered-Gram Orthogonal SessionLoRA (`nullspace_lora.py`)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mm_flex_merge.models import TEXT_MASK, AttnState, SessionLoRA


def test_o1_merged_forward_matches_sequential_sum_across_sessions():
    """Verifies O(1) merged `delta_weight` forward pass matches sequential loop over past sessions."""
    torch.manual_seed(42)
    d_in, d_out, r, alpha = 32, 24, 4, 8.0
    base = nn.Linear(d_in, d_out, bias=True)
    for p in base.parameters():
        p.requires_grad_(False)

    st = AttnState(d_in, is_text=False)
    lora = SessionLoRA(base, st, r=r, alpha=alpha)

    x_test = torch.randn(16, 5, d_in)

    for session_idx in range(4):
        lora.new_session()
        with torch.no_grad():
            lora.B.copy_(torch.randn_like(lora.B) * 0.1)
        st.recording = True
        _ = st.hook(base, (torch.randn(10, 3, d_in),))
        st.recording = False
        st.update_projection(tau=0.90)

    # Compare O(1) merged forward against explicit sequential loop over `lora.past`
    out_o1 = lora(x_test)

    out_seq = base(x_test)
    scale = alpha / r
    for i in range(0, len(lora.past), 2):
        out_seq = out_seq + (x_test @ lora.past[i].T @ lora.past[i + 1].T) * scale
    A_curr = lora.A if lora.P is None else lora.A @ lora.P
    out_seq = out_seq + (x_test @ A_curr.T @ lora.B.T) * scale

    assert torch.allclose(out_o1, out_seq, atol=1e-5)
    # Also verify full `merged_weight()` matches `F.linear(x, lora.merged_weight(), base.bias)`
    out_full_merged = nn.functional.linear(x_test, lora.merged_weight(), base.bias)
    assert torch.allclose(out_o1, out_full_merged, atol=1e-5)


def test_null_space_orthogonality_and_tau_one_rank_bound():
    """Verifies `dW_2 @ X_1^T == 0` when projecting onto the full span of Session 1 (`tau = 1.0`)."""
    torch.manual_seed(123)
    d_in, d_out = 16, 12
    base = nn.Linear(d_in, d_out, bias=False)
    st = AttnState(d_in, is_text=False)
    lora = SessionLoRA(base, st, r=4, alpha=8.0)

    # Session 1 spans a 6-dimensional subspace of R^16
    basis_1 = torch.linalg.qr(torch.randn(d_in, 6))[0]
    x_1 = torch.randn(20, 6) @ basis_1.T  # (20, 16)

    lora.new_session()
    with torch.no_grad():
        lora.B.copy_(torch.randn_like(lora.B))

    st.recording = True
    st.hook(base, (x_1,))
    st.recording = False

    # Test both tau=0.999999 (captures the 6D subspace) and tau=1.0 (rank bound <= d_in)
    st.update_projection(tau=0.999999)
    assert st.rank == 6

    # Session 2 initialized with null-space projector P_perp^{(1)}
    lora.new_session()
    with torch.no_grad():
        lora.B.copy_(torch.randn_like(lora.B))

    A_eff_2 = lora.A @ lora.P
    dW_2 = (lora.B @ A_eff_2) * lora.scale

    # Session 2's parameter update produces zero interference on Session 1's inputs x_1
    interference = x_1 @ dW_2.T
    assert torch.allclose(interference, torch.zeros_like(interference), atol=1e-5)

    # Edge case: tau=1.0 on full-rank Gram must clamp st.rank <= d_in
    st.recording = True
    st.hook(base, (torch.randn(64, d_in),))
    st.recording = False
    st.update_projection(tau=1.0)
    assert st.rank <= d_in
    assert st.proj.shape == (d_in, d_in)


def test_reversible_unmerge_session_and_text_mask():
    """Verifies `unmerge_session(k)` exactly removes session k and `TEXT_MASK` filters padding tokens."""
    torch.manual_seed(77)
    d_in, d_out = 12, 8
    base = nn.Linear(d_in, d_out)
    st = AttnState(d_in, is_text=True)
    lora = SessionLoRA(base, st, r=4, alpha=8.0)

    # Verify TEXT_MASK filtering inside hook
    x_tok = torch.ones(2, 4, d_in)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])  # 3 valid tokens total
    TEXT_MASK["mask"] = mask
    st.recording = True
    st.hook(base, (x_tok,))
    st.recording = False
    TEXT_MASK["mask"] = None
    assert torch.allclose(st.gram, torch.full((d_in, d_in), 3.0))

    # Commit 3 sessions, then unmerge session 1 (middle session)
    session_deltas = []
    for _ in range(3):
        lora.new_session()
        with torch.no_grad():
            lora.B.copy_(torch.randn_like(lora.B))
        A_eff = lora.A.detach() if lora.P is None else (lora.A.detach() @ lora.P)
        session_deltas.append((lora.B.detach() @ A_eff) * lora.scale)

    # Move 3rd session into `past` by starting a 4th session with B=0
    lora.new_session()
    assert len(lora.past) == 6

    # Unmerge session index 1 (the second session)
    lora.unmerge_session(1)
    assert len(lora.past) == 4
    expected_delta = session_deltas[0] + session_deltas[2]
    assert torch.allclose(lora.delta_weight, expected_delta, atol=1e-6)
