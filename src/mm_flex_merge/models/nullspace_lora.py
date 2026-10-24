"""
Stage 1: Null-Space Uncentered-Gram Orthogonal SessionLoRA.

Implements:
- `AttnState`: Uncentered input Gram matrix accumulator (`C^{(t)} = sum X_t^T X_t`)
  and exact orthogonal complement projector (`P_perp^{(t)} = I - V_t V_t^T`) via SVD/EVD.
- `SessionLoRA`: Continual low-rank adapter (`A_eff,t = A_t P_perp^{(t-1)}`) with
  exact O(1) constant-time additive parameter-space merging (`W_merged = W_0 + sum dW_k`)
  and reversible session un-merging (`unmerge_session`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_MASK: dict[str, torch.Tensor | None] = {"mask": None}


class AttnState:
    """Tracks the uncentered input activation Gram matrix and null-space projector."""

    def __init__(
        self,
        d: int,
        is_text: bool = False,
        device: str | torch.device = "cpu",
        text_mask_holder: dict[str, torch.Tensor | None] | None = None,
    ) -> None:
        self.gram = torch.zeros(d, d, dtype=torch.float32, device=device)
        self.proj: torch.Tensor | None = None
        self.is_text = bool(is_text)
        self.recording = False
        self.rank = 0
        self._mask_holder = TEXT_MASK if text_mask_holder is None else text_mask_holder

    def hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if not self.recording:
            return
        dev_type = "cuda" if self.gram.is_cuda else "cpu"
        with torch.amp.autocast(dev_type, enabled=False):
            x = inputs[0].detach()
            mask = self._mask_holder.get("mask")
            if self.is_text and mask is not None:
                x = x[mask.bool()]
            x = x.reshape(-1, x.shape[-1]).float().to(self.gram.device)
            self.gram += x.T @ x

    def update_projection(self, tau: float | None) -> None:
        if tau is None:
            return
        g = torch.nan_to_num(self.gram.double().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        g = 0.5 * (g + g.T)
        evals, evecs = torch.linalg.eigh(g)
        evals, evecs = evals.flip(0).clamp(min=0.0), evecs.flip(1)
        tot = evals.sum().clamp(min=1e-12)
        k = min(int((torch.cumsum(evals, 0) / tot < tau).sum().item()) + 1, evals.numel())
        U = evecs[:, :k].float()
        self.proj = (torch.eye(U.shape[0], dtype=torch.float32) - U @ U.T).to(self.gram.device)
        self.rank = k


class SessionLoRA(nn.Module):
    """
    Null-Space Projected Continual LoRA with O(1) constant-time merged forward pass.

    Completed sessions are folded into `delta_weight` (`sum_{k=1}^{t-1} (alpha/r) B_k A_{eff,k}`)
    so inference and subsequent session training execute in constant O(1) time regardless
    of the number of tasks `T`, while retaining `(A_{eff,k}, B_k)` in `self.past` for
    reversible session un-merging (`unmerge_session`).
    """

    def __init__(
        self,
        base: nn.Linear,
        owner: AttnState,
        r: int = 16,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.owner = owner
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r
        self.past = nn.ParameterList()
        self.A: nn.Parameter | None = None
        self.B: nn.Parameter | None = None
        self.P: torch.Tensor | None = None
        self.register_buffer(
            "delta_weight",
            torch.zeros_like(base.weight, dtype=torch.float32),
            persistent=False,
        )
        self._has_merged = False

    def _commit_active_to_past(self) -> None:
        if self.A is None or self.B is None:
            return
        A_eff = (self.A.detach() @ self.P) if self.P is not None else self.A.detach()
        B_det = self.B.detach()
        self.past.append(nn.Parameter(A_eff, requires_grad=False))
        self.past.append(nn.Parameter(B_det, requires_grad=False))
        self.delta_weight.add_((B_det.float() @ A_eff.float()) * self.scale)
        self._has_merged = True

    def new_session(self) -> None:
        if self.A is not None:
            self._commit_active_to_past()
        self.P = None if self.owner.proj is None else self.owner.proj.clone().to(self.base.weight.device)
        d_in, d_out = self.base.in_features, self.base.out_features
        dev = self.base.weight.device
        self.A = nn.Parameter(torch.randn(self.r, d_in, device=dev) / d_in ** 0.5)
        self.B = nn.Parameter(torch.zeros(d_out, self.r, device=dev))

    def rebuild_merged_delta(self) -> None:
        """Recomputes `delta_weight` from `self.past` (e.g., after un-merging a session)."""
        self.delta_weight.zero_()
        for i in range(0, len(self.past), 2):
            A_eff = self.past[i].detach().float()
            B_det = self.past[i + 1].detach().float()
            self.delta_weight.add_((B_det @ A_eff) * self.scale)
        self._has_merged = len(self.past) > 0

    def unmerge_session(self, session_idx: int) -> None:
        """Removes a committed past session `session_idx` in O(1) without retraining."""
        num_past = len(self.past) // 2
        if session_idx < 0:
            session_idx += num_past
        if not (0 <= session_idx < num_past):
            raise IndexError(f"session_idx={session_idx} out of range for {num_past} committed sessions.")
        drop = {2 * session_idx, 2 * session_idx + 1}
        self.past = nn.ParameterList([p for idx, p in enumerate(self.past) if idx not in drop])
        self.rebuild_merged_delta()

    def merged_weight(self) -> torch.Tensor:
        """Returns the full effective weight `W_0 + sum_{k=1}^t (alpha/r) B_k A_{eff,k}`."""
        w = self.base.weight.float() + self.delta_weight
        if self.A is not None and self.B is not None:
            A_eff = self.A if self.P is None else self.A @ self.P
            w = w + (self.B.float() @ A_eff.float()) * self.scale
        return w.to(self.base.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_merged:
            w_past = self.base.weight + self.delta_weight.to(self.base.weight.dtype)
            out = F.linear(x, w_past, self.base.bias)
        else:
            out = self.base(x)
        if self.A is None or self.B is None:
            return out
        A = self.A if self.P is None else self.A @ self.P
        return out + (x @ A.T.to(x.dtype) @ self.B.T.to(x.dtype)) * self.scale
