"""Block-structured analytic continual learner on frozen or Null-Space LoRA multimodal embeddings.

Each modality m has its own frozen random projection, phi_m(x) = relu(normalize(x) @ W_m).
For M=2 core modalities (Image 'img' and Text 'txt'), the model maintains sufficient statistics
over the block-concatenated features phi = [phi_img, phi_txt]:

    G = [[G_II, G_IT], [G_TI, G_TT]],   S = [[S_I], [S_T]]   (+ per-class Grams/counts when beta > 0)

Consequences (all exact in FP64, zero retraining):
  * update / delete        : add or subtract statistics -> deletion == retraining without the data.
  * order invariance       : statistics are commutative sums, so task order does not matter.
  * missing modality       : solve ridge on the Gram sub-block of the available modalities,
                             which equals a dedicated model trained on those modalities only.
  * asymmetric ridge       : modality-specific Tikhonov regularization (gamma_I, gamma_T) scaled
                             by each block's own trace Tr(G_mm) / P.
  * Block Schur Complement : solves the 2P x 2P system via P x P inversions while cancelling
                             cross-modal collinearity (-rho * G_IT * W_T) and reusing W_T.
  * CMML / MIL availability: accumulates partial-modality observations and normalizes block energy
                             by per-modality observation counts N_{c, m}.
"""

from __future__ import annotations

import math
import zlib
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class BlockAnalyticCL:
    def __init__(
        self,
        modality_dims: Mapping[str, int],
        num_classes: int,
        *,
        projection_dim: int = 1024,
        ridge: float | Mapping[str, float] = 0.1,
        class_balance_beta: float = 0.0,
        schur_coupling_rho: float = 1.0,
        seed: int = 2026,
    ) -> None:
        if not modality_dims or num_classes <= 0 or projection_dim <= 0:
            raise ValueError("invalid BlockAnalyticCL configuration")
        if isinstance(ridge, Mapping):
            if any(float(v) <= 0 for v in ridge.values()):
                raise ValueError("ridge values must be positive")
            self.modality_ridges: dict[str, float] | None = {str(k): float(v) for k, v in ridge.items()}
            self.ridge = float(next(iter(self.modality_ridges.values())))
        else:
            if float(ridge) <= 0:
                raise ValueError("ridge must be positive")
            self.modality_ridges = None
            self.ridge = float(ridge)
        if class_balance_beta < 0:
            raise ValueError("class_balance_beta must be non-negative")
        if not (0.0 <= float(schur_coupling_rho) <= 1.0):
            raise ValueError("schur_coupling_rho must be in [0, 1]")

        self.modalities = tuple(modality_dims)
        self.num_classes = int(num_classes)
        self.projection_dim = int(projection_dim)
        self.beta = float(class_balance_beta)
        self.rho = float(schur_coupling_rho)

        # Per-modality seed depends only on (seed, name) so a single-modality model
        # shares the exact projection of the same block in a multimodal model.
        self.projections: dict[str, torch.Tensor] = {}
        for name, dim in modality_dims.items():
            g = torch.Generator().manual_seed(int(seed) + zlib.crc32(name.encode()))
            self.projections[name] = torch.randn(int(dim), self.projection_dim, generator=g, dtype=torch.float64)

        D = self.projection_dim * len(self.modalities)
        self.gram = torch.zeros(D, D, dtype=torch.float64)
        self.class_sums = torch.zeros(D, self.num_classes, dtype=torch.float64)
        self._use_exact_class_grams = (self.beta > 0) and (self.num_classes * D * D <= 50_000_000)
        self.class_grams = torch.zeros(self.num_classes, D, D, dtype=torch.float64) if self._use_exact_class_grams else None
        self.class_counts = torch.zeros(self.num_classes, dtype=torch.long)
        self.modality_class_counts = torch.zeros(len(self.modalities), self.num_classes, dtype=torch.long)
        self._has_partial_observations = False
        self._cache: dict[tuple[str, ...], torch.Tensor] = {}

    # ------------------------------------------------------------------ features
    def _check(self, modalities: Sequence[str] | None) -> tuple[str, ...]:
        if modalities is None:
            return self.modalities
        mods = tuple(m for m in self.modalities if m in set(modalities))
        if not mods or len(mods) != len(set(modalities)):
            raise ValueError(f"unknown or empty modalities {modalities}; known {self.modalities}")
        return mods

    def _indices(self, mods: tuple[str, ...]) -> torch.Tensor:
        P = self.projection_dim
        return torch.cat([torch.arange(self.modalities.index(m) * P, (self.modalities.index(m) + 1) * P) for m in mods])

    def features(self, inputs: Mapping[str, torch.Tensor], modalities: Sequence[str] | None = None) -> torch.Tensor:
        mods = self._check(modalities)
        return torch.cat(
            [F.relu(F.normalize(inputs[m].double(), dim=-1) @ self.projections[m]) for m in mods], dim=1
        )

    # ---------------------------------------------------------------- statistics
    def _accumulate(
        self,
        inputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        sign: float,
        availability: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        labels = labels.long()
        n = int(labels.shape[0])
        P = self.projection_dim

        # Support both full-modality inputs and partial-modality (CMML / MIL) inputs
        present_mods = [m for m in self.modalities if m in inputs and inputs[m] is not None]
        if not present_mods:
            raise ValueError("at least one known modality must be provided in inputs")

        phi_blocks = []
        avail_masks = []
        for m_idx, m in enumerate(self.modalities):
            if m in inputs and inputs[m] is not None:
                block = F.relu(F.normalize(inputs[m].double(), dim=-1) @ self.projections[m])
                if availability is not None and m in availability and availability[m] is not None:
                    mask = availability[m].bool().view(-1)
                    if not bool(mask.all()):
                        self._has_partial_observations = True
                        block = torch.where(mask.unsqueeze(1), block, torch.zeros_like(block))
                else:
                    mask = torch.ones(n, dtype=torch.bool)
            else:
                self._has_partial_observations = True
                block = torch.zeros(n, P, dtype=torch.float64)
                mask = torch.zeros(n, dtype=torch.bool)
            phi_blocks.append(block)
            avail_masks.append(mask)

        phi = torch.cat(phi_blocks, dim=1)
        counts = torch.bincount(labels, minlength=self.num_classes)
        if sign < 0 and bool((counts > self.class_counts).any()):
            raise ValueError("cannot delete more samples of a class than were observed")

        self.gram.add_(phi.T @ phi, alpha=sign)
        self.class_sums.add_(phi.T @ F.one_hot(labels, self.num_classes).double(), alpha=sign)
        self.class_counts.add_(counts if sign > 0 else -counts)
        for m_idx, mask in enumerate(avail_masks):
            if bool(mask.any()):
                m_counts = torch.bincount(labels[mask], minlength=self.num_classes)
                self.modality_class_counts[m_idx].add_(m_counts if sign > 0 else -m_counts)

        if self.class_grams is not None:
            for c in labels.unique().tolist():
                f = phi[labels == c]
                self.class_grams[c].add_(f.T @ f, alpha=sign)
        self._cache.clear()

    def update(
        self,
        inputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        availability: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._accumulate(inputs, labels, 1.0, availability=availability)

    def delete(
        self,
        inputs: Mapping[str, torch.Tensor],
        labels: torch.Tensor,
        availability: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Exact unlearning: remove previously seen samples (any granularity: sample/task)."""
        self._accumulate(inputs, labels, -1.0, availability=availability)

    def class_weights(self, counts: torch.Tensor | None = None) -> torch.Tensor:
        cnt = (self.class_counts if counts is None else counts).double()
        weights = torch.zeros_like(cnt)
        observed = cnt > 0
        if observed.any():
            rel = cnt[observed] / cnt[observed].mean()
            weights[observed] = rel.pow(-self.beta)
            weights[observed] /= weights[observed].mean()
        return weights

    # --------------------------------------------------------------------- solve
    def _weighted_gram_and_sums(self, mods: tuple[str, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        P = self.projection_dim
        is_full = mods == self.modalities
        if is_full:
            gram_view = self.gram
            sums_view = self.class_sums
        elif len(mods) == 1:
            m_idx = self.modalities.index(mods[0])
            s_m = slice(m_idx * P, (m_idx + 1) * P)
            gram_view = self.gram[s_m, s_m]
            sums_view = self.class_sums[s_m]
        else:
            idx = self._indices(mods)
            gram_view = self.gram[idx][:, idx]
            sums_view = self.class_sums[idx]

        if self.class_grams is None:
            if self.beta <= 0 and not self._has_partial_observations:
                return gram_view.clone(), sums_view.clone()
            w = self.class_weights() if self.beta > 0 else torch.ones(self.num_classes, dtype=torch.float64)
            if not self._has_partial_observations or len(mods) == 1:
                if len(mods) == 1 and self._has_partial_observations and self.beta > 0:
                    m_idx = self.modalities.index(mods[0])
                    w = self.class_weights(self.modality_class_counts[m_idx])
                return gram_view.clone(), sums_view * w.unsqueeze(0)
            K = len(mods)
            G = gram_view.clone()
            S = torch.zeros(K * P, self.num_classes, dtype=torch.float64)
            scales = []
            mean_scales = []
            tot_cnt = float(self.class_counts.sum().item())
            for m in mods:
                m_idx = self.modalities.index(m)
                mc = self.modality_class_counts[m_idx].double()
                sc = torch.where(mc > 0, self.class_counts.double() / mc.clamp(min=1.0), torch.ones_like(w))
                scales.append(sc)
                m_tot = float(mc.sum().item())
                mean_scales.append(tot_cnt / max(m_tot, 1.0) if m_tot > 0 else 1.0)
            for a in range(K):
                sa = slice(a * P, (a + 1) * P)
                G[sa, sa] *= mean_scales[a]
                S[sa] = sums_view[sa] * (w * scales[a]).unsqueeze(0)
                for b in range(a + 1, K):
                    sb = slice(b * P, (b + 1) * P)
                    sab = math.sqrt(mean_scales[a] * mean_scales[b])
                    G[sa, sb] *= sab
                    G[sb, sa] *= sab
            return G, S

        w = self.class_weights()
        if not self._has_partial_observations or len(mods) == 1:
            if len(mods) == 1 and self._has_partial_observations:
                m_idx = self.modalities.index(mods[0])
                w = self.class_weights(self.modality_class_counts[m_idx])
            obs = self.class_counts > 0
            if not self._has_partial_observations and bool(obs.any()) and float((w[obs].max() - w[obs].min()).item()) < 1e-12:
                return gram_view.clone(), sums_view * w.unsqueeze(0)
            if is_full:
                D_sub = self.gram.shape[0]
                G = torch.matmul(w, self.class_grams.view(self.num_classes, -1)).view(D_sub, D_sub)
            elif len(mods) == 1:
                m_idx = self.modalities.index(mods[0])
                s_m = slice(m_idx * P, (m_idx + 1) * P)
                sub_cg = self.class_grams[:, s_m, s_m].contiguous()
                G = torch.matmul(w, sub_cg.view(self.num_classes, -1)).view(P, P)
            else:
                idx = self._indices(mods)
                G = torch.einsum("c,cij->ij", w, self.class_grams[:, idx][:, :, idx])
            S = sums_view * w.unsqueeze(0)
            return G, S

        # Availability-normalized weighting across modalities when partial observations exist
        scales = []
        for m in mods:
            m_idx = self.modalities.index(m)
            mc = self.modality_class_counts[m_idx].double()
            sc = torch.where(mc > 0, self.class_counts.double() / mc.clamp(min=1.0), torch.ones_like(w))
            scales.append(sc)

        K = len(mods)
        G = torch.zeros(K * P, K * P, dtype=torch.float64)
        S = torch.zeros(K * P, self.num_classes, dtype=torch.float64)
        sub_cgrams = self.class_grams if is_full else self.class_grams[:, self._indices(mods)][:, :, self._indices(mods)]
        sub_csums = sums_view
        for a in range(K):
            sa = slice(a * P, (a + 1) * P)
            wa = w * scales[a]
            G[sa, sa] = torch.einsum("c,cij->ij", wa, sub_cgrams[:, sa, sa])
            S[sa] = sub_csums[sa] * wa.unsqueeze(0)
            for b in range(a + 1, K):
                sb = slice(b * P, (b + 1) * P)
                wab = w * torch.sqrt(scales[a] * scales[b])
                G[sa, sb] = torch.einsum("c,cij->ij", wab, sub_cgrams[:, sa, sb])
                G[sb, sa] = G[sa, sb].T
        return G, S

    def solve_schur(
        self,
        gamma_first: float | None = None,
        gamma_second: float | None = None,
        rho: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimal 2x2 Block Schur Complement solver for M=2 modalities (e.g. Image and Text).
        Returns (W_both, W_first_only, W_second_only) in O(P^3) by reusing the unimodal inverse:
          A_2 = G_22 + lam_2 I_P
          W_2_only = A_2^{-1} S_2
          Schur_1 = (G_11 + lam_1 I_P) - rho^2 * G_12 A_2^{-1} G_21
          W_1_joint = Schur_1^{-1} (S_1 - rho * G_12 W_2_only)
          W_2_joint = W_2_only - rho * A_2^{-1} G_21 W_1_joint
        """
        if len(self.modalities) != 2:
            raise ValueError("solve_schur requires exactly 2 modalities")
        if int(self.class_counts.sum()) <= 0:
            raise RuntimeError("BlockAnalyticCL needs at least one observed sample")

        P = self.projection_dim
        m1, m2 = self.modalities
        g1 = float(gamma_first if gamma_first is not None else (self.modality_ridges.get(m1, self.ridge) if self.modality_ridges else self.ridge))
        g2 = float(gamma_second if gamma_second is not None else (self.modality_ridges.get(m2, self.ridge) if self.modality_ridges else self.ridge))
        rho_val = float(self.rho if rho is None else rho)

        G, S = self._weighted_gram_and_sums(self.modalities)
        G_11, G_22, G_12 = G[:P, :P], G[P:, P:], G[:P, P:]
        S_1, S_2 = S[:P], S[P:]

        tr_1_uni = max(float(torch.trace(G_11)) / P, 1e-12)
        tr_2_uni = max(float(torch.trace(G_22)) / P, 1e-12)
        use_block_trace = (self.modality_ridges is not None) or (gamma_first is not None) or (gamma_second is not None)
        if use_block_trace:
            tr_1, tr_2 = tr_1_uni, tr_2_uni
        else:
            tr_joint = max(float(torch.trace(G)) / (2 * P), 1e-12)
            tr_1 = tr_2 = tr_joint

        eye = torch.eye(P, dtype=G.dtype)
        A_1 = G_11 + (g1 * tr_1) * eye
        A_2 = G_22 + (g2 * tr_2) * eye

        sol_2 = torch.linalg.solve(A_2, torch.cat([G_12.T, S_2], dim=1))
        A_2_inv_G_21 = sol_2[:, :P]
        W_2_joint_base = sol_2[:, P:]

        if use_block_trace:
            W_1_only = torch.linalg.solve(A_1, S_1)
            W_2_only = W_2_joint_base
        else:
            W_1_only = torch.linalg.solve(G_11 + (g1 * tr_1_uni) * eye, S_1)
            W_2_only = torch.linalg.solve(G_22 + (g2 * tr_2_uni) * eye, S_2)

        if rho_val == 0.0:
            W_1_joint = W_1_only if use_block_trace else torch.linalg.solve(A_1, S_1)
            W_both = torch.cat([W_1_joint, W_2_joint_base], dim=0)
        else:
            Schur_1 = A_1 - (rho_val ** 2) * (G_12 @ A_2_inv_G_21)
            rhs_1 = S_1 - rho_val * (G_12 @ W_2_joint_base)
            W_1_joint = torch.linalg.solve(Schur_1, rhs_1)
            W_2_joint = W_2_joint_base - rho_val * (A_2_inv_G_21 @ W_1_joint)
            W_both = torch.cat([W_1_joint, W_2_joint], dim=0)

        return W_both, W_1_only, W_2_only

    def auto_spectral_ridges(self, gamma_base: float = 0.22, auto_rho: bool = True) -> dict[str, float]:
        """
        Pillar 2: Self-tuning Empirical-Bayes Centered Fisher-Spectral & Sample-Load Shrinkage Law.
        Automatically computes modality-specific regularization (gamma_1, gamma_2) by projecting out
        the non-discriminative positive-orthant ReLU mean and combining the dimensionless centered
        Fisher Noise-to-Signal Ratio (FNSR_m = tr(W_m) / tr(B_m)) with the spectral effective rank
        r_eff(G_mm), bounded sample-load scaling, and per-class density Schur coupling rho*.
        """
        if len(self.modalities) != 2:
            raise ValueError("auto_spectral_ridges requires exactly 2 modalities")
        if int(self.class_counts.sum()) <= 0:
            raise RuntimeError("BlockAnalyticCL needs at least one observed sample")

        P = self.projection_dim
        m1, m2 = self.modalities
        G, S = self._weighted_gram_and_sums(self.modalities)
        G_11, G_22 = G[:P, :P], G[P:, P:]
        S_1, S_2 = S[:P], S[P:]

        w = self.class_weights()
        cnt_w = self.class_counts.double() * w
        N_w = max(float(cnt_w.sum().item()), 1e-12)
        mu_1 = S_1.sum(dim=1, keepdim=True) / N_w
        mu_2 = S_2.sum(dim=1, keepdim=True) / N_w
        G_11_c = G_11 - N_w * (mu_1 @ mu_1.T)
        G_22_c = G_22 - N_w * (mu_2 @ mu_2.T)
        S_1_c = S_1 - mu_1 @ cnt_w.unsqueeze(0)
        S_2_c = S_2 - mu_2 @ cnt_w.unsqueeze(0)

        inv_cnt = torch.where(cnt_w > 0, 1.0 / cnt_w.clamp(min=1e-12), torch.zeros_like(cnt_w))
        tr_B1 = max(float(torch.sum((S_1_c ** 2) * inv_cnt.unsqueeze(0))), 1e-12)
        tr_B2 = max(float(torch.sum((S_2_c ** 2) * inv_cnt.unsqueeze(0))), 1e-12)
        tr_c1 = max(float(torch.trace(G_11_c)), 1e-12)
        tr_c2 = max(float(torch.trace(G_22_c)), 1e-12)
        tr_u1 = max(float(torch.trace(G_11)), 1e-12)
        tr_u2 = max(float(torch.trace(G_22)), 1e-12)

        fnsr_1 = max((tr_c1 - tr_B1) / tr_B1, 0.1)
        fnsr_2 = max((tr_c2 - tr_B2) / tr_B2, 0.1)
        fnsr_geom = (fnsr_1 * fnsr_2) ** 0.5

        r_eff1 = (tr_u1 ** 2) / max(float(torch.sum(G_11 * G_11)), 1e-12)
        r_eff2 = (tr_u2 ** 2) / max(float(torch.sum(G_22 * G_22)), 1e-12)

        idx_1 = (fnsr_1 / max(r_eff1, 1.0)) * ((tr_c1 / tr_u1) ** 0.5)
        idx_2 = (fnsr_2 / max(r_eff2, 1.0)) * ((tr_c2 / tr_u2) ** 0.5)
        idx_geom = max((idx_1 * idx_2) ** 0.5, 1e-12)

        n_1 = max(int(self.modality_class_counts[0].sum().item()), 1)
        n_2 = max(int(self.modality_class_counts[1].sum().item()), 1)
        noise_scale = (fnsr_geom / 6.0) ** 0.30
        load_1 = (0.58 + 0.52 * (float(2 * P) / (n_1 + float(2 * P)))) * noise_scale
        load_2 = (0.58 + 0.52 * (float(2 * P) / (n_2 + float(2 * P)))) * noise_scale

        eb_1 = (idx_1 / idx_geom) ** 0.65
        eb_2 = (idx_2 / idx_geom) ** 0.65

        g1 = float(min(max(gamma_base * load_1 * eb_1, 0.05), 5.0))
        g2 = float(min(max(gamma_base * load_2 * eb_2, 0.05), 5.0))
        self.modality_ridges = {m1: g1, m2: g2}
        if auto_rho:
            obs_c = max(int((self.class_counts > 0).sum().item()), 1)
            n_per_cls = float(self.class_counts.sum().item()) / obs_c
            self.rho = float(min(1.0, max(n_per_cls / 65.0, 0.05)))
        self._cache.clear()
        return dict(self.modality_ridges)

    def schur_innovation_decomposition(
        self,
        gamma_first: float | None = None,
        gamma_second: float | None = None,
        rho: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pillar 1: Exact Schur-Innovation Decomposition (SID).
        Decomposes the joint 2x2 Block Schur estimator into its marginal unimodal estimators
        plus the exact cross-modal Schur innovation matrices:
          W_1_joint = W_1_only + dW_1_schur
          W_2_joint = W_2_only + dW_2_schur
        Returns (W_1_only, W_2_only, dW_1_schur, dW_2_schur, W_both).
        """
        if len(self.modalities) != 2:
            raise ValueError("schur_innovation_decomposition requires exactly 2 modalities")
        if int(self.class_counts.sum()) <= 0:
            raise RuntimeError("BlockAnalyticCL needs at least one observed sample")

        is_default = (gamma_first is None) and (gamma_second is None) and (rho is None)
        if is_default and ("__sid__",) in self._cache:
            return self._cache[("__sid__",)]  # type: ignore[return-value]

        P = self.projection_dim
        m1, m2 = self.modalities
        g1 = float(gamma_first if gamma_first is not None else (self.modality_ridges.get(m1, self.ridge) if self.modality_ridges else self.ridge))
        g2 = float(gamma_second if gamma_second is not None else (self.modality_ridges.get(m2, self.ridge) if self.modality_ridges else self.ridge))
        rho_val = float(self.rho if rho is None else rho)

        G, S = self._weighted_gram_and_sums(self.modalities)
        G_11, G_22, G_12 = G[:P, :P], G[P:, P:], G[:P, P:]
        S_1, S_2 = S[:P], S[P:]

        use_block_trace = (self.modality_ridges is not None) or (gamma_first is not None) or (gamma_second is not None)
        if use_block_trace:
            tr_1 = max(float(torch.trace(G_11)) / P, 1e-12)
            tr_2 = max(float(torch.trace(G_22)) / P, 1e-12)
        else:
            tr_joint = max(float(torch.trace(G)) / (2 * P), 1e-12)
            tr_1 = tr_2 = tr_joint

        eye = torch.eye(P, dtype=G.dtype)
        A_1 = G_11 + (g1 * tr_1) * eye
        A_2 = G_22 + (g2 * tr_2) * eye

        W_1_only = torch.linalg.solve(A_1, S_1)
        sol_2 = torch.linalg.solve(A_2, torch.cat([G_12.T, S_2], dim=1))
        A_2_inv_G_21 = sol_2[:, :P]
        W_2_only = sol_2[:, P:]

        if rho_val == 0.0:
            dW_1_schur = torch.zeros_like(W_1_only)
            dW_2_schur = torch.zeros_like(W_2_only)
        else:
            Schur_1 = A_1 - (rho_val ** 2) * (G_12 @ A_2_inv_G_21)
            innov_rhs_1 = (rho_val ** 2) * (G_12 @ (A_2_inv_G_21 @ W_1_only)) - rho_val * (G_12 @ W_2_only)
            dW_1_schur = torch.linalg.solve(Schur_1, innov_rhs_1)
            dW_2_schur = -rho_val * (A_2_inv_G_21 @ (W_1_only + dW_1_schur))

        W_both = torch.cat([W_1_only + dW_1_schur, W_2_only + dW_2_schur], dim=0)
        res = (W_1_only, W_2_only, dW_1_schur, dW_2_schur, W_both)
        if is_default:
            self._cache[("__sid__",)] = res  # type: ignore[assignment]
            self._cache[self.modalities] = W_both
            self._cache[(m1,)] = W_1_only
            self._cache[(m2,)] = W_2_only
            w = self.class_weights()
            cnt_w = (self.class_counts.double() * w).clamp(min=1e-12)
            N_w = max(float(cnt_w.sum().item()), 1e-12)
            mu_1_vec = S_1.sum(dim=1) / N_w
            mu_2_vec = S_2.sum(dim=1) / N_w
            mu_1_hat = F.normalize(mu_1_vec, dim=0)
            mu_2_hat = F.normalize(mu_2_vec, dim=0)
            tr_u1 = max(float(torch.trace(G_11)), 1e-12)
            tr_u2 = max(float(torch.trace(G_22)), 1e-12)
            c0_1 = float(((N_w * float(torch.sum(mu_1_vec ** 2))) / tr_u1) ** 0.5)
            c0_2 = float(((N_w * float(torch.sum(mu_2_vec ** 2))) / tr_u2) ** 0.5)
            S_1_c = F.normalize(S_1 - S_1.mean(dim=1, keepdim=True), dim=0)
            S_2_c = F.normalize(S_2 - S_2.mean(dim=1, keepdim=True), dim=0)
            self._cache[("__sid_stats__",)] = (mu_1_hat, mu_2_hat, c0_1, c0_2, S_1_c, S_2_c)  # type: ignore[assignment]
        return res

    def transport_gram(
        self,
        old_inputs: Mapping[str, torch.Tensor],
        new_inputs: Mapping[str, torch.Tensor],
        ridge_transport: float = 1e-4,
        max_drift: float = 0.35,
        adaptive_drift_guard: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Pillar 3: Exemplar-Free Block-Gram Pushforward Transport with Perturbative Drift Guard.
        Estimates modality-specific linear transport operators M_m = I + eta_m * dM_m from
        current-task feature shifts (old_inputs -> new_inputs) and transforms all historical
        class Gram blocks and class sums in-place: G_c <- M G_c M^T, S_c <- M S_c (B = 0).
        When `adaptive_drift_guard=True`, automatically damps or bypasses transport if the
        relative feature drift exceeds the first-order perturbative trust region `max_drift`.
        """
        P = self.projection_dim
        eye = torch.eye(P, dtype=torch.float64)
        operators: dict[str, torch.Tensor] = {}

        for m in self.modalities:
            if m in old_inputs and m in new_inputs and old_inputs[m] is not None and new_inputs[m] is not None:
                z_old = self.features(old_inputs, (m,))
                z_new = self.features(new_inputs, (m,))
                diff = z_new - z_old
                delta_rel = float(torch.linalg.norm(diff) / torch.linalg.norm(z_old).clamp(min=1e-12))
                if adaptive_drift_guard and delta_rel >= max_drift:
                    operators[m] = eye
                    continue
                cov_old = z_old.T @ z_old
                tr_scale = max(float(torch.trace(cov_old)) / P, 1e-12)
                if adaptive_drift_guard and max_drift > 0:
                    ratio2 = (delta_rel / max_drift) ** 2
                    eff_ridge = ridge_transport + 0.05 * ratio2
                    eta = 1.0 - ratio2
                else:
                    eff_ridge = ridge_transport
                    eta = 1.0
                lam = eff_ridge * tr_scale
                dM_T = torch.linalg.solve(cov_old + lam * eye, z_old.T @ diff)
                operators[m] = eye + eta * dM_T.T
            else:
                operators[m] = eye

        M_blocks = [operators[m] for m in self.modalities]
        if len(M_blocks) == 2:
            M_1, M_2 = M_blocks[0], M_blocks[1]
            M_1_T, M_2_T = M_1.T, M_2.T
            if self.class_grams is not None:
                for c in range(self.num_classes):
                    if int(self.class_counts[c].item()) > 0:
                        Gc = self.class_grams[c]
                        Gc[:P, :P] = M_1 @ Gc[:P, :P] @ M_1_T
                        Gc[P:, P:] = M_2 @ Gc[P:, P:] @ M_2_T
                        G_12_new = M_1 @ Gc[:P, P:] @ M_2_T
                        Gc[:P, P:] = G_12_new
                        Gc[P:, :P] = G_12_new.T
                self.gram = self.class_grams.sum(dim=0)
            else:
                self.gram[:P, :P] = M_1 @ self.gram[:P, :P] @ M_1_T
                self.gram[P:, P:] = M_2 @ self.gram[P:, P:] @ M_2_T
                G_12_new = M_1 @ self.gram[:P, P:] @ M_2_T
                self.gram[:P, P:] = G_12_new
                self.gram[P:, :P] = G_12_new.T
            self.class_sums[:P] = M_1 @ self.class_sums[:P]
            self.class_sums[P:] = M_2 @ self.class_sums[P:]
        else:
            M_full = torch.block_diag(*M_blocks)
            if self.class_grams is not None:
                for c in range(self.num_classes):
                    if int(self.class_counts[c].item()) > 0:
                        self.class_grams[c] = M_full @ self.class_grams[c] @ M_full.T
                        self.class_sums[:, c] = M_full @ self.class_sums[:, c]
                self.gram = self.class_grams.sum(dim=0)
            else:
                self.gram = M_full @ self.gram @ M_full.T
                self.class_sums = M_full @ self.class_sums
        self._cache.clear()
        return operators

    @staticmethod
    def auto_continual_policy(
        g1_gain_pts: float | None = None,
        incremental_mode: str = "dil",
        min_g1_gain_pts: float = 2.0,
    ) -> dict[str, object]:
        """
        Self-governing continual adaptation policy derived from cross-dataset empirical laws
        (CrisisMMD v2.0 DIL vs. UPMC Food-101 100-class CIL):
          1. If `g1_gain_pts < min_g1_gain_pts` (backbone already aligned with target domain):
             keep backbone frozen (`tau=1.0`) and use `BlockAnalyticCL` with `adaptive_sid=True`.
          2. If `incremental_mode == 'dil'` (Domain-Incremental, fixed class set):
             adapt Vision tower in null space (`tau=0.95`) while keeping Text tower as frozen
             semantic anchor (`towers=('vision',)`).
          3. If `incremental_mode == 'cil'` (Class-Incremental, expanding class set):
             adapt both towers with strict null-space protection (`tau=0.99`) + Gram transport & SID.
        """
        mode = incremental_mode.strip().lower()
        if mode not in {"dil", "cil"}:
            raise ValueError(f"incremental_mode must be 'dil' or 'cil', got {incremental_mode!r}")
        if g1_gain_pts is not None and float(g1_gain_pts) < min_g1_gain_pts:
            return {
                "regime": "frozen_analytic_sid",
                "lora_enabled": False,
                "tau": 1.0,
                "towers": (),
                "use_transport": False,
                "adaptive_sid": True,
            }
        if mode == "dil":
            return {
                "regime": "dil_vision_nullspace_frozen_text",
                "lora_enabled": True,
                "tau": 0.95,
                "towers": ("vision",),
                "use_transport": True,
                "adaptive_sid": True,
            }
        return {
            "regime": "cil_dual_nullspace_transport_sid",
            "lora_enabled": True,
            "tau": 0.99,
            "towers": ("vision", "text"),
            "use_transport": True,
            "adaptive_sid": True,
        }

    def solve(self, modalities: Sequence[str] | None = None) -> torch.Tensor:
        mods = self._check(modalities)
        if mods not in self._cache:
            if int(self.class_counts.sum()) <= 0:
                raise RuntimeError("BlockAnalyticCL needs at least one observed sample")
            if len(mods) == 2 and len(self.modalities) == 2 and (self.modality_ridges is not None or self.rho != 1.0):
                W_both, W_1, W_2 = self.solve_schur()
                self._cache[self.modalities] = W_both
                self._cache[(self.modalities[0],)] = W_1
                self._cache[(self.modalities[1],)] = W_2
                return self._cache[mods]

            G, S = self._weighted_gram_and_sums(mods)
            if self.modality_ridges is not None and len(mods) == 1:
                r_m = self.modality_ridges.get(mods[0], self.ridge)
                lam = r_m * max(float(torch.trace(G)) / G.shape[0], 1e-12)
            else:
                lam = self.ridge * max(float(torch.trace(G)) / G.shape[0], 1e-12)
            self._cache[mods] = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), S)
        return self._cache[mods]

    def logits(
        self,
        inputs: Mapping[str, torch.Tensor],
        modalities: Sequence[str] | None = None,
        adaptive_sid: bool = False,
        tau_gate: float = 2.0,
    ) -> torch.Tensor:
        """
        Logits using only `modalities` (default: all present in `inputs`).
        When `adaptive_sid=True` and both modalities are present, applies the continuous
        Schur-Innovation Decomposition (SID) reliability-modulated inference law.
        """
        if modalities is None:
            present = tuple(m for m in self.modalities if m in inputs and inputs[m] is not None)
            mods = self._check(present if present else None)
        else:
            mods = self._check(modalities)

        if adaptive_sid and len(mods) == 2 and len(self.modalities) == 2:
            m1, m2 = self.modalities
            z_1 = self.features(inputs, (m1,))
            z_2 = self.features(inputs, (m2,))
            W_1_o, W_2_o, dW_1_s, dW_2_s, _ = self.schur_innovation_decomposition()
            mu_1_hat, mu_2_hat, c0_1, c0_2, S_1_c, S_2_c = self._cache[("__sid_stats__",)]  # type: ignore[misc]

            L_1_o = z_1 @ W_1_o
            L_2_o = z_2 @ W_2_o
            L_1_in = z_1 @ dW_1_s
            L_2_in = z_2 @ dW_2_s

            z_1_n = F.normalize(z_1, dim=-1)
            z_2_n = F.normalize(z_2, dim=-1)
            cone_1 = (z_1_n @ mu_1_hat).clamp(min=0.0)
            cone_2 = (z_2_n @ mu_2_hat).clamp(min=0.0)
            f_cone_1 = torch.where(
                cone_1 > 1e-6,
                ((cone_1 - 0.46) / max(c0_1 - 0.52, 0.10)).clamp(0.02, 1.0),
                torch.zeros_like(cone_1),
            )
            f_cone_2 = torch.where(
                cone_2 > 1e-6,
                ((cone_2 - 0.46) / max(c0_2 - 0.52, 0.10)).clamp(0.02, 1.0),
                torch.zeros_like(cone_2),
            )

            k_top = min(2, self.num_classes)
            P_1 = z_1_n @ S_1_c
            P_2 = z_2_n @ S_2_c
            pmax_1 = P_1.max(dim=1).values.clamp(min=1e-6)
            pmax_2 = P_2.max(dim=1).values.clamp(min=1e-6)
            ptop2_1 = torch.topk(P_1, k=k_top, dim=1).values
            ptop2_2 = torch.topk(P_2, k=k_top, dim=1).values
            pmargin_1 = (ptop2_1[:, 0] - ptop2_1[:, -1]).clamp(min=1e-6)
            pmargin_2 = (ptop2_2[:, 0] - ptop2_2[:, -1]).clamp(min=1e-6)

            top2_1 = torch.topk(L_1_o, k=k_top, dim=1).values
            lmargin_1 = (top2_1[:, 0] - top2_1[:, -1]).clamp(min=1e-6)
            top2_2 = torch.topk(L_2_o, k=k_top, dim=1).values
            lmargin_2 = (top2_2[:, 0] - top2_2[:, -1]).clamp(min=1e-6)

            conf_1 = (f_cone_1 ** 2) * lmargin_1 * (pmax_1 ** 2) * (0.05 + pmargin_1)
            conf_2 = (f_cone_2 ** 2) * lmargin_2 * (pmax_2 ** 2) * (0.05 + pmargin_2)
            max_conf = torch.maximum(conf_1, conf_2).clamp(min=1e-12)
            r_1 = (conf_1 / max_conf).clamp(0.0, 1.0)
            r_2 = (conf_2 / max_conf).clamp(0.0, 1.0)

            discrepancy = (torch.abs(conf_1 - conf_2) / (conf_1 + conf_2).clamp(min=1e-12)).clamp(0.0, 1.0)
            L_1_c = L_1_o - L_1_o.mean(dim=1, keepdim=True)
            L_2_c = L_2_o - L_2_o.mean(dim=1, keepdim=True)
            cos_agree = F.cosine_similarity(L_1_c, L_2_c, dim=1).clamp(min=0.0, max=1.0)
            cone_drop = 1.0 - torch.minimum(f_cone_1, f_cone_2)
            anomaly = discrepancy * (1.0 - 0.7 * cos_agree) * (0.35 + 0.65 * cone_drop)
            excess = ((anomaly - 0.25) / 0.35).clamp(0.0, 1.0)
            gate_blend = (excess ** 1.5).unsqueeze(1)

            p_exp = 2.0 if tau_gate == 2.0 else 2.0 / max(float(tau_gate), 1e-6)
            alpha_1 = (1.0 - gate_blend) + gate_blend * torch.pow(r_1, p_exp).unsqueeze(1)
            alpha_2 = (1.0 - gate_blend) + gate_blend * torch.pow(r_2, p_exp).unsqueeze(1)

            both_present = (torch.minimum(cone_1, cone_2) / 0.45).clamp(0.0, 1.0).unsqueeze(1)
            rho_raw = (torch.minimum(r_1, r_2) * cos_agree).unsqueeze(1)
            rho_x = (1.0 - gate_blend) + gate_blend * rho_raw
            rho_1 = alpha_1 * (0.4 * both_present + 0.6 * rho_x)
            rho_2 = alpha_2 * (0.4 * both_present + 0.6 * rho_x)

            return alpha_1 * L_1_o + alpha_2 * L_2_o + rho_1 * L_1_in + rho_2 * L_2_in

        return self.features(inputs, mods) @ self.solve(mods)

    def predict(
        self,
        inputs: Mapping[str, torch.Tensor],
        modalities: Sequence[str] | None = None,
        adaptive_sid: bool = False,
        tau_gate: float = 2.0,
    ) -> torch.Tensor:
        return self.logits(inputs, modalities, adaptive_sid=adaptive_sid, tau_gate=tau_gate).argmax(dim=-1)
