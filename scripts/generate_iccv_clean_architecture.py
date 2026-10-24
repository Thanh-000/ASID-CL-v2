#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Grade Visual Architecture & Continual Merging Pipeline Generator (v9)
=================================================================================
Addresses all 4 visual critique points from the user on iccv_panel_a_architecture.png:
  1. Image 0 Fix ("adapt tức là sao, hình biểu diễn chưa đủ trực quan"):
     - Replaces vague "Adapt ΔM_t" text boxes with iconic visual cards showing:
       * Multimodal Stream D_t with visual [Image frame + Text lines] mini-icons
       * Task-t Learner card containing a visual LoRA bottleneck icon (A_t ▷◁ B_t)
         and a mini 2x2 colored Gram matrix grid (G_t), linked vertically by the
         orthogonal subspace bus P_perp^(t-1) and horizontally into the progressive
         merging spine (M_1 -> M_{1:2} = M_1 ⊕ ΔM_2 -> ... -> M_{1:n}).
  2. Image 1 Fix ("các mũi tên lệch không hợp vị trí"):
     - Sets shrinkA=0.0, shrinkB=0.0 on all FancyArrowPatch instances so orthogonal
       90-degree corners and T-junctions have ZERO pixel gaps or misalignments.
     - Replaces the single off-center inter-stage line with two aligned horizontal
       buses connecting Stage 1 directly into Stage 2A (Weight Merge) and Stage 2B
       (2x2 Block-Gram Merge).
  3. Image 2 Fix ("các khối trong còn sơ sài, và mũi tên biểu diễn chưa khoa học"):
     - Enriches Stage 2 with internal visual schematics:
       * Stage 2A (Backbone Weight Merging): Stacked LoRA plates (W_0 ⊕ ΔW_1 ⊕ ... ⊕ ΔW_n)
         -> Dual Vision/Text Encoder trapezoids (φ_I, φ_T) -> Segmented Feature Tensor
         Bars (z_I, z_T).
       * Stage 2B (2x2 Block-Gram & Schur Solver): Visual 2x2 Matrix Sum
         (G_{1:n-1} ⊕ ΔG_n = G_{1:n}^*, with ⊖ G_k unmerge) -> Block-Schur Solver
         S_m(ρ*)^-1 -> 3 aligned Classifier Weight Matrix icons ([W_I^o], [ΔW^s], [W_T^o]).
  4. Image 3 Fix ("chưa đủ visual cho các thông tin này và còn đơn điệu"):
     - Enriches Stage 3 (Conflict-Gated Late Fusion) with visual mini logit bar charts
       inside Image Logits L_I^o, Schur Synergy ΔL^s (with Conflict Gate g(x) badge),
       and Text Logits L_T^o, converging via gap-free orthogonal routing into the
       Late Fusion node (⊕) and a visual Fused Class Probability Distribution bar chart
       for y_hat(x).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle, FancyArrowPatch


# ==============================================================================
# 1. GLOBAL TYPOGRAPHY & MATHTEXT CONFIGURATION
# ==============================================================================
def configure_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "font.size": 6.0,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.2,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def init_mm_axis(ax, w_mm, h_mm):
    ax.set_xlim(0.0, w_mm)
    ax.set_ylim(0.0, h_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    ax._boxed_texts = []


def register_bounded_text(ax, txt_obj, x, y, w, h, pad_x=0.6):
    if not hasattr(ax, "_boxed_texts"):
        ax._boxed_texts = []
    ax._boxed_texts.append((txt_obj, x, y, w, h, pad_x))


# ==============================================================================
# 2. PRECISION GEOMETRIC & VISUAL ICON PRIMITIVES (ZERO ARROW SHRINK / GAPS)
# ==============================================================================
def draw_rbox(ax, x, y, w, h, fill="#FFFFFF", edge="#1E293B", r=1.1, lw=0.85, ls="-", zorder=2, alpha=1.0):
    patch = FancyBboxPatch(
        (x + r, y + r),
        max(0.1, w - 2 * r),
        max(0.1, h - 2 * r),
        boxstyle=f"round,pad={r},rounding_size={r}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_box_text(ax, x, y, w, h, title, sub=None, fill="#FFFFFF", edge="#1E293B",
                  t_col="#0F172A", s_col="#334155", fs_t=4.8, fs_s=4.1, r=1.0, lw=0.85,
                  ls="-", zorder=2, pad_x=0.6):
    draw_rbox(ax, x, y, w, h, fill=fill, edge=edge, r=r, lw=lw, ls=ls, zorder=zorder)
    cx = x + w / 2.0
    if sub is None or sub == "":
        t1 = ax.text(cx, y + h / 2.0, title, ha="center", va="center",
                     fontsize=fs_t, fontweight="bold", color=t_col, zorder=zorder + 2)
        register_bounded_text(ax, t1, x, y, w, h, pad_x)
    else:
        t1 = ax.text(cx, y + h * 0.65, title, ha="center", va="center",
                     fontsize=fs_t, fontweight="bold", color=t_col, zorder=zorder + 2)
        t2 = ax.text(cx, y + h * 0.28, sub, ha="center", va="center",
                     fontsize=fs_s, fontweight="normal", color=s_col, zorder=zorder + 2)
        register_bounded_text(ax, t1, x, y, w, h, pad_x)
        register_bounded_text(ax, t2, x, y, w, h, pad_x)


def draw_arrow_mm(ax, p1, p2, color="#334155", lw=0.95, ls="-", mut=5.2, zorder=4):
    """Strictly straight arrow with shrinkA=0, shrinkB=0 so endpoints never pull back or misalign."""
    arr = FancyArrowPatch(
        p1, p2,
        connectionstyle="arc3,rad=0.0",
        arrowstyle="-|>",
        mutation_scale=mut,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def draw_ortho_arrow(ax, pts, color="#334155", lw=0.95, ls="-", mut=5.2, zorder=4):
    """Gap-free 90-degree orthogonal polyline with exact corner joins and terminal arrowhead."""
    if len(pts) >= 2:
        # Draw continuous polyline up to slightly before the final tip so the shaft is 100% seamless
        p_prev = np.array(pts[-2], dtype=float)
        p_end = np.array(pts[-1], dtype=float)
        vec = p_end - p_prev
        dist = np.linalg.norm(vec)
        if dist > 0.8:
            p_stop = p_end - (vec / dist) * 0.6
        else:
            p_stop = p_prev
        poly_xs = [p[0] for p in pts[:-1]] + [p_stop[0]]
        poly_ys = [p[1] for p in pts[:-1]] + [p_stop[1]]
        ax.plot(poly_xs, poly_ys, color=color, linewidth=lw, linestyle=ls,
                solid_joinstyle="round", solid_capstyle="butt", zorder=zorder)
        draw_arrow_mm(ax, tuple(p_prev), tuple(p_end), color=color, lw=lw, ls=ls, mut=mut, zorder=zorder)


def draw_op_node(ax, cx, cy, r=1.6, sym=r"$\oplus$", fill="#FEF9C3", edge="#A16207",
                 col="#713F12", fs=5.8, lw=0.9, zorder=6):
    c = Circle((cx, cy), radius=r, facecolor=fill, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(c)
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, fontweight="bold", color=col, zorder=zorder + 1)
    return c


def draw_2x2_gram_grid(ax, x, y, w, h, l11=r"$G_{II}$", l12=r"$G_{IT}$", l21=r"$G_{TI}$", l22=r"$G_{TT}$",
                       c11="#DCFCE7", c12="#FEF3C7", c21="#FEF3C7", c22="#DBEAFE", fs=3.9, zorder=3, show_text=True):
    hw, hh = w / 2.0, h / 2.0
    cells = [
        (x, y + hh, c11, l11),
        (x + hw, y + hh, c12, l12),
        (x, y, c21, l21),
        (x + hw, y, c22, l22),
    ]
    for cx, cy, fill, lbl in cells:
        ax.add_patch(Rectangle((cx, cy), hw, hh, facecolor=fill, edgecolor="#334155", linewidth=0.6, zorder=zorder))
        if show_text and lbl:
            t_cell = ax.text(cx + hw / 2.0, cy + hh / 2.0, lbl, ha="center", va="center",
                             fontsize=fs, fontweight="bold", color="#0F172A", zorder=zorder + 1)
            register_bounded_text(ax, t_cell, cx, cy, hw, hh, pad_x=0.2)
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#0F172A", linewidth=0.85, zorder=zorder + 2))


def draw_mini_lora_bowtie(ax, x, y, w, h, fill="#DCFCE7", edge="#15803D", zorder=4):
    """Draws a miniature LoRA bottleneck icon [A ▷ | ◁ B] inside a card."""
    cy = y + h / 2.0
    w_trap = w * 0.40
    w_mid = w * 0.10
    gap = w * 0.05
    # Down-projection trapezoid A
    p_a = [[x, y], [x + w_trap, y + h * 0.24], [x + w_trap, y + h * 0.76], [x, y + h]]
    ax.add_patch(Polygon(p_a, closed=True, facecolor=fill, edgecolor=edge, lw=0.6, zorder=zorder))
    # Red orthogonal null-space bottleneck bar
    bx = x + w_trap + gap * 0.5
    ax.add_patch(Rectangle((bx, y + h * 0.20), w_mid, h * 0.60, facecolor="#FCA5A5", edgecolor="#DC2626", lw=0.6, zorder=zorder + 1))
    # Up-projection trapezoid B
    ux = bx + w_mid + gap * 0.5
    p_b = [[ux, y + h * 0.24], [x + w, y], [x + w, y + h], [ux, y + h * 0.76]]
    ax.add_patch(Polygon(p_b, closed=True, facecolor=fill, edgecolor=edge, lw=0.6, zorder=zorder))


def draw_mini_logit_bars(ax, x, y, w, h, heights=(0.35, 0.88, 0.52, 0.25),
                         bar_col="#15803D", hi_idx=1, hi_col="#16A34A", bg="#FFFFFF", edge="#CBD5E1", zorder=4):
    """Draws a miniature class-logit histogram icon inside a Late Fusion card."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=bg, edgecolor=edge, linewidth=0.55, zorder=zorder))
    n = len(heights)
    pad = w * 0.10
    bw = (w - 2 * pad) / (n * 1.25)
    step = (w - 2 * pad) / n
    for i, hr in enumerate(heights):
        bx = x + pad + i * step + (step - bw) * 0.5
        bh = max(0.4, (h - 0.8) * hr)
        col = hi_col if i == hi_idx else bar_col
        alpha = 1.0 if i == hi_idx else 0.55
        ax.add_patch(Rectangle((bx, y + 0.3), bw, bh, facecolor=col, edgecolor="none", alpha=alpha, zorder=zorder + 1))


def draw_segmented_vector(ax, x, y, w, h, n_segs=4, fill="#86EFAC", edge="#15803D", zorder=4):
    """Draws a vertical segmented feature vector tensor bar z_m."""
    sh = h / float(n_segs)
    for i in range(n_segs):
        alpha = 0.95 if i % 2 == 0 else 0.65
        ax.add_patch(Rectangle((x, y + i * sh), w, sh, facecolor=fill, edgecolor=edge, lw=0.55, alpha=alpha, zorder=zorder))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=edge, lw=0.8, zorder=zorder + 1))


def draw_mini_image_icon(ax, x, y, w, h, edge="#15803D", zorder=4):
    """Draws a miniature RGB photo thumbnail (sky, golden sun, twin mountain peaks) inside Data Stream D_t."""
    # Outer white photo frame
    draw_rbox(ax, x, y, w, h, fill="#FFFFFF", edge=edge, r=0.45, lw=0.65, zorder=zorder)
    pad = 0.35
    ix, iy, iw, ih = x + pad, y + pad, w - 2 * pad, h - 2 * pad
    # Sky canvas
    ax.add_patch(Rectangle((ix, iy), iw, ih, facecolor="#E0F2FE", edgecolor="none", zorder=zorder + 1))
    # Golden sun disk in upper-right
    sun_r = min(iw, ih) * 0.16
    ax.add_patch(Circle((ix + iw * 0.74, iy + ih * 0.73), radius=sun_r,
                        facecolor="#FBBF24", edgecolor="#D97706", lw=0.4, zorder=zorder + 2))
    # Back mountain peak (lighter green)
    m_back = [[ix + iw * 0.32, iy], [ix + iw * 0.68, iy + ih * 0.62], [ix + iw, iy]]
    ax.add_patch(Polygon(m_back, closed=True, facecolor="#4ADE80", edgecolor="#15803D", lw=0.45, zorder=zorder + 3))
    # Front mountain peak (rich emerald green)
    m_front = [[ix, iy], [ix + iw * 0.38, iy + ih * 0.76], [ix + iw * 0.76, iy]]
    ax.add_patch(Polygon(m_front, closed=True, facecolor="#16A34A", edgecolor="#14532D", lw=0.45, zorder=zorder + 4))
    # Crisp inner frame border
    ax.add_patch(Rectangle((ix, iy), iw, ih, facecolor="none", edgecolor="#94A3B8", lw=0.35, zorder=zorder + 5))


def draw_mini_text_icon(ax, x, y, w, h, edge="#1D4ED8", zorder=4):
    """Draws a miniature text-prompt document card (header band + stylized token lines) inside Data Stream D_t."""
    draw_rbox(ax, x, y, w, h, fill="#FFFFFF", edge=edge, r=0.45, lw=0.65, zorder=zorder)
    pad = 0.38
    ix, iy, iw, ih = x + pad, y + pad, w - 2 * pad, h - 2 * pad
    # Top blue prompt header band
    ax.add_patch(Rectangle((ix, iy + ih * 0.74), iw, ih * 0.26, facecolor="#DBEAFE", edgecolor="none", zorder=zorder + 1))
    ax.add_patch(Rectangle((ix + iw * 0.08, iy + ih * 0.82), iw * 0.45, ih * 0.10,
                           facecolor="#1D4ED8", edgecolor="none", zorder=zorder + 2))
    # Stylized horizontal text token bars
    line_specs = [
        (0.54, 0.08, 0.36, "#2563EB"),
        (0.54, 0.50, 0.40, "#60A5FA"),
        (0.32, 0.08, 0.78, "#3B82F6"),
        (0.10, 0.08, 0.56, "#60A5FA"),
    ]
    for ly_rel, lx_rel, lw_rel, lcol in line_specs:
        ax.add_patch(Rectangle((ix + iw * lx_rel, iy + ih * ly_rel), iw * lw_rel, ih * 0.11,
                               facecolor=lcol, edgecolor="none", zorder=zorder + 2))


def draw_mini_checkpoint_icon(ax, x, y, w, h, edge="#1E3A8A", fill_w="#BBF7D0", zorder=4):
    """Draws a miniature merged neural checkpoint icon (stacked weight layers + 2x2 Gram grid)."""
    # Left half: 3 stacked neural weight plates
    lw_w = w * 0.48
    sh = (h - 0.6) / 3.0
    cols = ["#E2E8F0", fill_w, "#86EFAC"]
    for i in range(3):
        ax.add_patch(Rectangle((x, y + i * (sh + 0.25)), lw_w, sh,
                               facecolor=cols[i], edgecolor=edge, lw=0.45, zorder=zorder + i))
    # Right half: mini 2x2 Gram matrix
    gx = x + w * 0.56
    gw = w * 0.44
    draw_2x2_gram_grid(ax, gx, y, gw, h, show_text=False, zorder=zorder + 1)


def draw_signed_synergy_bars(ax, x, y, w, h, vals=(-0.35, 0.88, -0.48, -0.22), zorder=4):
    """Draws a zero-centered signed residual bar chart (+ boost / - suppression) for Schur Synergy ΔL^s."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.55, zorder=zorder))
    y_zero = y + h * 0.42
    ax.plot([x + 0.2, x + w - 0.2], [y_zero, y_zero], color="#64748B", lw=0.55, ls="--", zorder=zorder + 1)
    n = len(vals)
    pad = w * 0.10
    bw = (w - 2 * pad) / (n * 1.25)
    step = (w - 2 * pad) / n
    max_up = h - (y_zero - y) - 0.5
    max_dn = (y_zero - y) - 0.4
    for i, v in enumerate(vals):
        bx = x + pad + i * step + (step - bw) * 0.5
        if v >= 0:
            bh = max(0.35, max_up * v)
            ax.add_patch(Rectangle((bx, y_zero), bw, bh, facecolor="#9333EA", edgecolor="none", zorder=zorder + 2))
        else:
            bh = max(0.35, max_dn * abs(v))
            ax.add_patch(Rectangle((bx, y_zero - bh), bw, bh, facecolor="#F43F5E", edgecolor="none", alpha=0.85, zorder=zorder + 2))


def draw_mini_sigmoid_gate(ax, x, y, w, h, zorder=4):
    """Draws a miniature Sigmoid gating curve g(x) = σ(τ·s(x)) with conflict/synergy zones."""
    ax.add_patch(Rectangle((x, y), w * 0.5, h, facecolor="#FFE4E6", edgecolor="none", zorder=zorder))
    ax.add_patch(Rectangle((x + w * 0.5, y), w * 0.5, h, facecolor="#DCFCE7", edgecolor="none", zorder=zorder))
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#94A3B8", lw=0.45, zorder=zorder + 1))
    # Axes lines
    ax.plot([x + w * 0.5, x + w * 0.5], [y + 0.2, y + h - 0.2], color="#94A3B8", lw=0.4, ls=":", zorder=zorder + 1)
    ts = np.linspace(-4.5, 4.5, 30)
    xs = x + 0.25 + (ts + 4.5) / 9.0 * (w - 0.5)
    ys = y + 0.3 + (1.0 / (1.0 + np.exp(-ts))) * (h - 0.6)
    ax.plot(xs, ys, color="#DC2626", lw=0.85, zorder=zorder + 2)
    # Active operating point dot in synergy zone
    px = x + 0.25 + (2.0 + 4.5) / 9.0 * (w - 0.5)
    py = y + 0.3 + (1.0 / (1.0 + np.exp(-2.0))) * (h - 0.6)
    ax.add_patch(Circle((px, py), radius=0.32, facecolor="#16A34A", edgecolor="#FFFFFF", lw=0.35, zorder=zorder + 3))


def draw_mini_cosine_gauge(ax, x, y, w, h, zorder=4):
    """Draws a miniature angular cosine alignment gauge between L_I^o (green) and L_T^o (blue)."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#94A3B8", lw=0.45, zorder=zorder))
    ox, oy = x + w * 0.22, y + h * 0.22
    # Two vector rays from origin (ox, oy)
    v1_x, v1_y = ox + w * 0.62, oy + h * 0.22
    v2_x, v2_y = ox + w * 0.36, oy + h * 0.66
    ax.plot([ox, v1_x], [oy, v1_y], color="#1D4ED8", lw=0.85, solid_capstyle="round", zorder=zorder + 2)
    ax.plot([ox, v2_x], [oy, v2_y], color="#15803D", lw=0.85, solid_capstyle="round", zorder=zorder + 2)
    # Angle arc between the two rays
    ang = np.linspace(0.34, 1.07, 16)
    r_arc = min(w, h) * 0.38
    ax.plot(ox + r_arc * np.cos(ang), oy + r_arc * np.sin(ang), color="#DC2626", lw=0.65, ls="--", zorder=zorder + 1)
    ax.add_patch(Circle((ox, oy), radius=0.25, facecolor="#0F172A", edgecolor="none", zorder=zorder + 3))


def draw_stacked_fusion_bars(ax, x, y, w, h, zorder=4):
    """Draws a 3-color stacked bar chart showing additive Late Fusion: α_I L_I^o + α_T L_T^o + ρ(x) ΔL^s."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=0.6, zorder=zorder))
    # 4 classes: (img_part, txt_part, schur_part)
    stacks = [
        (0.14, 0.10, 0.00),
        (0.36, 0.30, 0.28),  # Winning class ★ with strong cross-modal Schur Synergy boost!
        (0.26, 0.12, 0.00),
        (0.08, 0.18, 0.00),
    ]
    n = len(stacks)
    pad = w * 0.10
    bw = (w - 2 * pad) / (n * 1.22)
    step = (w - 2 * pad) / n
    usable_h = h - 0.9
    base_y = y + 0.35
    for i, (h_i, h_t, h_s) in enumerate(stacks):
        bx = x + pad + i * step + (step - bw) * 0.5
        alpha = 1.0 if i == 1 else 0.55
        bi = usable_h * h_i
        bt = usable_h * h_t
        bs = usable_h * h_s
        # Bottom: Image contribution (Green)
        ax.add_patch(Rectangle((bx, base_y), bw, bi, facecolor="#16A34A", edgecolor="none", alpha=alpha, zorder=zorder + 1))
        # Middle: Text contribution (Blue)
        ax.add_patch(Rectangle((bx, base_y + bi), bw, bt, facecolor="#2563EB", edgecolor="none", alpha=alpha, zorder=zorder + 2))
        # Top: Schur Synergy boost (Purple)
        if bs > 0:
            ax.add_patch(Rectangle((bx, base_y + bi + bt), bw, bs, facecolor="#9333EA", edgecolor="#581C87", lw=0.4, zorder=zorder + 3))
    # Dashed decision threshold line showing how Schur Synergy pushes Class 2 decisively above confusers
    y_th = base_y + usable_h * 0.68
    ax.plot([x + 0.3, x + w - 0.3], [y_th, y_th], color="#DC2626", lw=0.5, ls="--", zorder=zorder + 4)


def auto_fit_all_boxed_texts(fig, max_iters=14):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iters):
        changed = False
        for ax in fig.axes:
            for (txt, bx, by, bw, bh, pad_x) in getattr(ax, "_boxed_texts", []):
                bb_disp = txt.get_window_extent(renderer=renderer)
                bb_data = bb_disp.transformed(ax.transData.inverted())
                max_w = bw - 2.0 * pad_x
                if bb_data.width > max_w and txt.get_fontsize() > 3.1:
                    ratio = max_w / max(bb_data.width, 1e-6)
                    new_fs = max(3.1, txt.get_fontsize() * min(0.93, ratio * 0.96))
                    txt.set_fontsize(new_fs)
                    changed = True
        if not changed:
            break
        fig.canvas.draw()

    remaining = 0
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for (txt, bx, by, bw, bh, pad_x) in getattr(ax, "_boxed_texts", []):
            bb_data = txt.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            if bb_data.width > (bw - 2.0 * pad_x) * 1.03:
                remaining += 1
    return remaining


def add_panel_letter(ax, letter, x_rel=-0.015, y_rel=1.03):
    ax.text(x_rel, y_rel, letter, transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="bottom", ha="left", color="#0F172A")


# ==============================================================================
# 3. FIGURE 1: 3-STAGE VISUAL ARCHITECTURE (v9 — RICH VISUAL SCHEMATICS)
# ==============================================================================
def build_figure1_scientific_architecture(out_dirs):
    configure_rcparams()
    fig = plt.figure(figsize=(180.0 / 25.4, 116.0 / 25.4))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.42, 0.92],
        width_ratios=[1.0, 1.05, 1.0],
        hspace=0.17, wspace=0.10,
        left=0.015, right=0.985, top=0.955, bottom=0.035
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    add_panel_letter(ax_a, "a", x_rel=0.002, y_rel=1.01)
    add_panel_letter(ax_b, "b", x_rel=-0.02, y_rel=1.02)
    add_panel_letter(ax_c, "c", x_rel=-0.02, y_rel=1.02)
    add_panel_letter(ax_d, "d", x_rel=-0.02, y_rel=1.02)

    # ==========================================================================
    # PANEL A: 3-TIER VISUAL ARCHITECTURE (176 mm x 58 mm)
    # ==========================================================================
    init_mm_axis(ax_a, 176.0, 58.0)

    # 3 Stage Containers with crisp headers
    # Stage 1: x = 1.0 .. 59.0
    draw_rbox(ax_a, 1.0, 1.2, 58.0, 55.2, fill="#F8FAFC", edge="#94A3B8", r=1.8, lw=0.85, ls="--", zorder=1)
    s1_hdr = ax_a.text(30.0, 53.8, r"Stage 1: Continual Task Stream ($t = 1 \rightarrow n$)",
                       ha="center", va="center", fontsize=4.9, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_a, s1_hdr, 1.0, 51.5, 58.0, 4.2, pad_x=1.0)

    # Stage 2: x = 64.4 .. 119.5
    draw_rbox(ax_a, 64.4, 1.2, 55.1, 55.2, fill="#F8FAFC", edge="#94A3B8", r=1.8, lw=0.85, ls="--", zorder=1)
    s2_hdr = ax_a.text(91.95, 53.8, r"Stage 2: Dual-Space Model Merging ($\mathcal{M}_1 \oplus \mathcal{M}_2 \oplus \cdots \oplus \mathcal{M}_n \rightarrow \mathcal{M}_{1:n}^*$)",
                       ha="center", va="center", fontsize=4.8, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_a, s2_hdr, 64.4, 51.5, 55.1, 4.2, pad_x=0.8)

    # Stage 3: x = 123.0 .. 175.0
    draw_rbox(ax_a, 123.0, 1.2, 52.0, 55.2, fill="#F8FAFC", edge="#94A3B8", r=1.8, lw=0.85, ls="--", zorder=1)
    s3_hdr = ax_a.text(149.0, 53.8, "Stage 3: Late Fusion & Classification (SID v2)",
                       ha="center", va="center", fontsize=4.9, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_a, s3_hdr, 123.0, 51.5, 52.0, 4.2, pad_x=1.0)

    # --------------------------------------------------------------------------
    # STAGE 1: SEQUENTIAL TASK MODEL LEARNING (t = 1 -> n)
    # Each task t = 1, 2, ..., n learns its own Task-Specific Model M_t = (ΔW_t, G_t)
    # on Multimodal Stream D_t (Img + Txt), then ALL Task Models M_1..M_n flow
    # into Stage 2 to be merged into the single Unified Merged Model M_{1:n}^*!
    # --------------------------------------------------------------------------
    y1, y2, y_dots, yn = 43.2, 28.2, 18.7, 9.2
    bh = 10.2

    # 3 Spacious Sub-column headers inside Stage 1
    ax_a.text(4.8, 49.6, "Time", ha="center", va="center", fontsize=4.0, fontweight="bold", color="#475569")
    ax_a.text(17.5, 49.6, "Multimodal Stream", ha="center", va="center", fontsize=4.0, fontweight="bold", color="#475569")
    ax_a.text(42.8, 49.6, r"Task-Specific Model $\mathcal{M}_t = (\Delta W_t, G_t)$", ha="center", va="center", fontsize=4.0, fontweight="bold", color="#475569")

    rows_cfg = [
        (y1, "$t=1$", r"$\mathcal{D}_1$ (Task 1)", r"Task-1 Model $\mathcal{M}_1$", r"LoRA $\Delta W_1$", r"Gram $G_1$",
         "#EFF6FF", "#DBEAFE", "#1D4ED8", "#1E3A8A"),
        (y2, "$t=2$", r"$\mathcal{D}_2$ (Task 2)", r"Task-2 Model $\mathcal{M}_2$", r"LoRA $\Delta W_2$", r"Gram $G_2$",
         "#F0FDF4", "#DCFCE7", "#15803D", "#14532D"),
        (yn, "$t=n$", r"$\mathcal{D}_n$ (Task $n$)", r"Task-$n$ Model $\mathcal{M}_n$", r"LoRA $\Delta W_n$", r"Gram $G_n$",
         "#FEF2F2", "#FEE2E2", "#DC2626", "#991B1B"),
    ]

    for (ry, t_lbl, d_lbl, m_hdr, w_lbl, g_lbl, bg_lt, bg_dk, edge_c, txt_c) in rows_cfg:
        # 1. Time Pill (x = 2.0..7.6)
        draw_rbox(ax_a, 2.0, ry - 2.3, 5.6, 4.6, fill=bg_dk, edge=edge_c, r=0.75, lw=0.85, zorder=5)
        ax_a.text(4.8, ry, t_lbl, ha="center", va="center", fontsize=4.1, fontweight="bold", color=txt_c, zorder=6)

        # 2. Multimodal Data Stream Card D_t with RGB Photo & Text Document icons (x = 9.4..25.6)
        draw_rbox(ax_a, 9.4, ry - bh / 2, 16.2, bh, fill=bg_lt, edge=edge_c, r=0.95, lw=0.85, zorder=2)
        t_d = ax_a.text(17.5, ry + 3.2, f"Stream {d_lbl}", ha="center", va="center",
                        fontsize=4.0, fontweight="bold", color=txt_c, zorder=4)
        register_bounded_text(ax_a, t_d, 9.4, ry + 1.6, 16.2, 3.2, pad_x=0.4)
        draw_mini_image_icon(ax_a, 10.5, ry - 2.6, 6.2, 3.9, edge="#15803D", zorder=3)
        ax_a.text(13.6, ry - 3.9, "Img", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#14532D", zorder=5)
        draw_mini_text_icon(ax_a, 18.3, ry - 2.6, 6.2, 3.9, edge="#1D4ED8", zorder=3)
        ax_a.text(21.4, ry - 3.9, "Txt", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#1E3A8A", zorder=5)

        draw_arrow_mm(ax_a, (25.6, ry), (28.4, ry), color=edge_c, lw=0.95, mut=4.8)

        # 3. Prominent Task-t Model Card M_t = (LoRA ΔW_t | 2x2 Gram G_t) (x = 28.4..57.2)
        draw_rbox(ax_a, 28.4, ry - bh / 2, 28.8, bh, fill=bg_lt, edge=edge_c, r=0.95, lw=0.9, zorder=2)
        t_mh = ax_a.text(42.8, ry + 3.3, m_hdr, ha="center", va="center", fontsize=4.0, fontweight="bold", color=txt_c, zorder=5)
        register_bounded_text(ax_a, t_mh, 28.4, ry + 1.8, 28.8, 3.0, pad_x=0.5)

        # Left module inside Model M_t: Null-Space LoRA Bowtie (x = 29.8..42.2)
        draw_mini_lora_bowtie(ax_a, 29.8, ry - 2.2, 12.2, 3.6, fill=bg_dk, edge=edge_c, zorder=4)
        t_w = ax_a.text(35.9, ry - 3.8, w_lbl, ha="center", va="center", fontsize=3.5, fontweight="bold", color=txt_c, zorder=4)
        register_bounded_text(ax_a, t_w, 29.0, ry - 5.0, 14.0, 2.6, pad_x=0.3)

        # Divider line inside M_t card
        ax_a.plot([44.4, 44.4], [ry - 4.2, ry + 1.6], color=edge_c, lw=0.55, ls=":", alpha=0.7, zorder=3)

        # Right module inside Model M_t: 2x2 Block-Gram Matrix Grid (x = 47.6..53.4)
        draw_2x2_gram_grid(ax_a, 47.6, ry - 2.2, 5.8, 3.6, show_text=False, zorder=4)
        t_g = ax_a.text(50.5, ry - 3.8, g_lbl, ha="center", va="center", fontsize=3.5, fontweight="bold", color=txt_c, zorder=4)
        register_bounded_text(ax_a, t_g, 45.0, ry - 5.0, 11.4, 2.6, pad_x=0.3)

    # Vertical Temporal Progression Arrows across Stage 1 (t = 1 -> t = 2 -> ... -> t = n)
    for cx, col_v, ls_v in [
        (4.8, "#0F172A", "-"),
        (17.5, "#64748B", "--"),
        (35.9, "#DC2626", "--"),
    ]:
        y_top_bot = y1 - (2.3 if cx < 6 else bh / 2)
        y_mid_top = y2 + (2.3 if cx < 6 else bh / 2)
        y_mid_bot = y2 - (2.3 if cx < 6 else bh / 2)
        y_bot_top = yn + (2.3 if cx < 6 else bh / 2)

        draw_arrow_mm(ax_a, (cx, y_top_bot), (cx, y_mid_top), color=col_v, lw=0.9, ls=ls_v, mut=4.5)
        draw_arrow_mm(ax_a, (cx, y_mid_bot), (cx, y_dots + 1.6), color=col_v, lw=0.9, ls=ls_v, mut=4.5)
        ax_a.text(cx, y_dots, r"$\vdots$", ha="center", va="center", fontsize=6.0, fontweight="bold", color=col_v)
        draw_arrow_mm(ax_a, (cx, y_dots - 1.6), (cx, y_bot_top), color=col_v, lw=0.9, ls=ls_v, mut=4.5)

    # Null-space projection badges on the vertical red LoRA subspace bus (x = 35.9)
    ax_a.text(37.1, (y1 + y2) / 2, r"$P_{\perp}^{(1)}$", ha="left", va="center", fontsize=3.9, fontweight="bold", color="#991B1B")
    ax_a.text(37.1, y_dots, r"$P_{\perp}^{(t)}$", ha="left", va="center", fontsize=3.9, fontweight="bold", color="#991B1B")

    # --------------------------------------------------------------------------
    # INTER-STAGE CONNECTION (STAGE 1 -> STAGE 2): FIX FOR IMAGE 0
    # ONE single, bold, centered horizontal arrow from Stage 1 into Stage 2!
    # --------------------------------------------------------------------------
    draw_arrow_mm(ax_a, (59.0, 26.2), (64.4, 26.2), color="#0F172A", lw=1.4, mut=6.2, zorder=6)

    # --------------------------------------------------------------------------
    # STAGE 2: EXPLICIT VISUAL MODEL MERGING -> UNIFIED MERGED MODEL M_{1:n}^*
    #   Left Column (x = 65.8..90.0):
    #     - Top (1. Weight Merging): Stacks Base W_0 + Blue ΔW_1 + Green ΔW_2 + Red ΔW_n -> (⊕) -> W_{1:n}^*
    #     - Bottom (2. Gram Merging): Sums Blue G_1 ⊕ Green G_2 ⊕ Red G_n = G_{1:n}^*
    #   Right Column (x = 92.4..118.2):
    #     - Prominent Gold Container: "Unified Merged Model M_{1:n}^*"
    #       with 3 symmetric horizontal rows (y_top=39.2, y_mid=24.4, y_bot=9.6)
    #       matching Stage 3 1-to-1 so all inter-stage arrows are 100% straight & symmetric!
    # --------------------------------------------------------------------------
    # --- STAGE 2 LEFT-TOP: 1. ORTHOGONAL WEIGHT MERGING (STACKED MULTI-TASK LORA) ---
    draw_rbox(ax_a, 65.8, 28.2, 24.2, 20.2, fill="#F0FDF4", edge="#15803D", r=1.1, lw=0.85, zorder=2)
    t_2a = ax_a.text(77.9, 46.5, "1. Orthogonal Weight Merge", ha="center", va="center",
                     fontsize=4.2, fontweight="bold", color="#14532D", zorder=4)
    register_bounded_text(ax_a, t_2a, 65.8, 44.6, 24.2, 3.6, pad_x=0.5)

    # 4 Visual Stacked Task Weight Modules on the left: Base W_0 + Blue ΔW_1 + Green ΔW_2 + Red ΔW_n
    draw_box_text(ax_a, 67.0, 41.6, 9.8, 2.5, title=r"Base $W_0$",
                  fill="#E2E8F0", edge="#475569", t_col="#1E293B", fs_t=3.3, r=0.4, lw=0.6, zorder=3)
    draw_mini_lora_bowtie(ax_a, 67.0, 38.5, 5.6, 2.3, fill="#DBEAFE", edge="#1D4ED8", zorder=4)
    ax_a.text(74.8, 39.65, r"$\Delta W_1$", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#1E3A8A", zorder=5)
    draw_mini_lora_bowtie(ax_a, 67.0, 35.4, 5.6, 2.3, fill="#DCFCE7", edge="#15803D", zorder=4)
    ax_a.text(74.8, 36.55, r"$\Delta W_2$", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#14532D", zorder=5)
    draw_mini_lora_bowtie(ax_a, 67.0, 32.3, 5.6, 2.3, fill="#FEE2E2", edge="#DC2626", zorder=4)
    ax_a.text(74.8, 33.45, r"$\Delta W_n$", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#991B1B", zorder=5)

    for wy, wcol in [(42.85, "#475569"), (39.65, "#1D4ED8"), (36.55, "#15803D"), (33.45, "#DC2626")]:
        ax_a.plot([76.8, 78.6], [wy, wy], color=wcol, lw=0.75, zorder=4)
    ax_a.plot([78.6, 78.6], [33.45, 42.85], color="#15803D", lw=0.9, zorder=4)
    draw_op_node(ax_a, 81.2, 38.2, r=1.25, sym=r"$\oplus$", fill="#DCFCE7", edge="#15803D", col="#14532D", fs=4.6, zorder=5)
    ax_a.plot([78.6, 79.95], [38.2, 38.2], color="#15803D", lw=0.9, zorder=4)

    draw_box_text(ax_a, 83.8, 34.8, 5.2, 6.8, title="Merged\n" + r"$W_{1:n}^*$",
                  fill="#86EFAC", edge="#15803D", t_col="#14532D", fs_t=3.5, r=0.5, lw=0.75, zorder=4, pad_x=0.25)
    draw_arrow_mm(ax_a, (82.45, 38.2), (83.8, 38.2), color="#15803D", lw=0.8, mut=3.8, zorder=5)

    t_2a_sub = ax_a.text(77.9, 29.9, r"$W_{1:n}^* = W_0 \oplus \sum_{t=1}^n \Delta W_t \;\;(\Delta W_t \perp X_{\mathrm{old}})$",
                         ha="center", va="center", fontsize=3.6, fontweight="bold", color="#166534", zorder=4)
    register_bounded_text(ax_a, t_2a_sub, 65.8, 28.4, 24.2, 3.0, pad_x=0.5)

    # --- STAGE 2 LEFT-BOTTOM: 2. 2x2 BLOCK-GRAM MERGING (G_1 ⊕ G_2 ⊕ G_n = G_{1:n}^*) ---
    draw_rbox(ax_a, 65.8, 3.0, 24.2, 23.4, fill="#EFF6FF", edge="#1D4ED8", r=1.1, lw=0.85, zorder=2)
    t_2b = ax_a.text(77.9, 24.0, r"2. $2\times 2$ Block-Gram Merge", ha="center", va="center",
                     fontsize=4.2, fontweight="bold", color="#1E3A8A", zorder=4)
    register_bounded_text(ax_a, t_2b, 65.8, 22.0, 24.2, 3.6, pad_x=0.5)

    draw_2x2_gram_grid(ax_a, 66.6, 13.0, 3.4, 4.0, c11="#DBEAFE", c22="#93C5FD", show_text=False, zorder=4)
    ax_a.text(68.3, 18.3, r"$G_1$", ha="center", va="center", fontsize=3.4, fontweight="bold", color="#1E3A8A", zorder=5)
    ax_a.text(71.05, 15.0, r"$\oplus$", ha="center", va="center", fontsize=3.7, fontweight="bold", color="#854D0E", zorder=5)

    draw_2x2_gram_grid(ax_a, 72.1, 13.0, 3.4, 4.0, c11="#DCFCE7", c22="#86EFAC", show_text=False, zorder=4)
    ax_a.text(73.8, 18.3, r"$G_2$", ha="center", va="center", fontsize=3.4, fontweight="bold", color="#14532D", zorder=5)
    ax_a.text(76.55, 15.0, r"$\oplus$", ha="center", va="center", fontsize=3.7, fontweight="bold", color="#854D0E", zorder=5)

    draw_2x2_gram_grid(ax_a, 77.6, 13.0, 3.4, 4.0, c11="#FEE2E2", c22="#FCA5A5", show_text=False, zorder=4)
    ax_a.text(79.3, 18.3, r"$G_n$", ha="center", va="center", fontsize=3.4, fontweight="bold", color="#991B1B", zorder=5)
    ax_a.text(82.05, 15.0, r"$=$", ha="center", va="center", fontsize=4.2, fontweight="bold", color="#1E3A8A", zorder=5)

    draw_2x2_gram_grid(ax_a, 82.9, 12.2, 5.8, 5.8,
                       l11=r"$G_{II}^*$", l12=r"$G_{IT}^*$", l21=r"$G_{TI}^*$", l22=r"$G_{TT}^*$",
                       c11="#86EFAC", c12="#FCD34D", c21="#FCD34D", c22="#93C5FD", fs=3.2, zorder=4)
    ax_a.text(85.4, 19.3, r"Merged $G_{1:n}^*$", ha="center", va="center", fontsize=3.2, fontweight="bold", color="#581C87", zorder=5)

    draw_box_text(ax_a, 67.2, 4.2, 21.4, 5.6,
                  title=r"$G_{1:n}^* = \sum_{t=1}^n M_t G_t M_t^\top \;\mid\; \ominus G_k$",
                  fill="#FEFCE8", edge="#CA8A04", t_col="#854D0E", fs_t=3.6, r=0.6, lw=0.65, zorder=3, pad_x=0.4)

    # --- STAGE 2 RIGHT COLUMN: SYMMETRIC "UNIFIED MERGED MODEL M_{1:n}^*" CONTAINER ---
    # Organized into 3 symmetric horizontal rows (y_top=39.2, y_mid=24.4, y_bot=9.6)
    # that align 100% straight across with Stage 3's 3 Logit Cards!
    y_top, y_mid, y_bot = 39.2, 24.4, 9.6
    ch = 10.2

    draw_rbox(ax_a, 92.4, 3.0, 25.8, 45.4, fill="#FEFCE8", edge="#CA8A04", r=1.3, lw=1.1, zorder=2)
    draw_rbox(ax_a, 93.4, 44.2, 23.8, 3.4, fill="#FEF08A", edge="#CA8A04", r=0.6, lw=0.75, zorder=3)
    t_um = ax_a.text(105.3, 45.9, r"Unified Merged Model $\mathcal{M}_{1:n}^*$", ha="center", va="center",
                     fontsize=4.0, fontweight="bold", color="#713F12", zorder=5)
    register_bounded_text(ax_a, t_um, 93.4, 44.2, 23.8, 3.4, pad_x=0.4)

    # Arrows from Weight Merge (90.0, 38.2) and Gram Merge (90.0, 15.1) into Unified Merged Model (92.4)
    draw_arrow_mm(ax_a, (90.0, 38.2), (92.4, 38.2), color="#15803D", lw=1.0, mut=4.2, zorder=5)
    draw_arrow_mm(ax_a, (90.0, 15.1), (92.4, 15.1), color="#1D4ED8", lw=1.0, mut=4.2, zorder=5)

    # Row 1 inside Merged Model (y_top = 39.2, Green — Vision Branch: Vision φ_I -> Head W_I^o)
    draw_rbox(ax_a, 93.4, y_top - 4.2, 23.8, 8.4, fill="#DCFCE7", edge="#15803D", r=0.8, lw=0.8, zorder=3)
    p_vi = [[94.2, y_top - 3.0], [104.2, y_top - 2.2], [104.2, y_top + 2.2], [94.2, y_top + 3.0]]
    ax_a.add_patch(Polygon(p_vi, closed=True, facecolor="#86EFAC", edgecolor="#15803D", lw=0.65, zorder=4))
    ax_a.text(99.2, y_top, r"Vision $\phi_I(W_I^*)$", ha="center", va="center", fontsize=3.3, fontweight="bold", color="#14532D", zorder=5)
    draw_arrow_mm(ax_a, (104.2, y_top), (106.6, y_top), color="#15803D", lw=0.75, mut=3.6, zorder=5)
    draw_box_text(ax_a, 106.6, y_top - 2.8, 9.8, 5.6, title=r"Head $W_I^{\circ}$", sub=r"($z_I$)",
                  fill="#FFFFFF", edge="#15803D", t_col="#14532D", s_col="#166534", fs_t=3.5, fs_s=3.1, r=0.5, lw=0.7, zorder=4, pad_x=0.25)

    # Row 2 inside Merged Model (y_mid = 24.4, Purple — Cross-Modal Schur Branch: Schur S_m^-1 -> Schur ΔW^s)
    draw_rbox(ax_a, 93.4, y_mid - 4.6, 23.8, 9.2, fill="#F3E8FF", edge="#7E22CE", r=0.8, lw=0.8, zorder=3)
    draw_box_text(ax_a, 94.2, y_mid - 3.4, 10.0, 6.8,
                  title="Schur Solver", sub=r"$\mathcal{S}_m(\rho^*)^{-1}$",
                  fill="#FFFFFF", edge="#7E22CE", t_col="#581C87", s_col="#6B21A8", fs_t=3.4, fs_s=3.2, r=0.5, lw=0.7, zorder=4, pad_x=0.25)
    draw_arrow_mm(ax_a, (104.2, y_mid), (106.6, y_mid), color="#7E22CE", lw=0.75, mut=3.6, zorder=5)
    draw_box_text(ax_a, 106.6, y_mid - 2.8, 9.8, 5.6, title=r"Schur $\Delta W^{\mathrm{s}}$", sub=r"($z_I, z_T$)",
                  fill="#E9D5FF", edge="#7E22CE", t_col="#581C87", s_col="#6B21A8", fs_t=3.5, fs_s=3.1, r=0.5, lw=0.7, zorder=4, pad_x=0.25)

    # Row 3 inside Merged Model (y_bot = 9.6, Blue — Text Branch: Text φ_T -> Head W_T^o)
    draw_rbox(ax_a, 93.4, y_bot - 4.2, 23.8, 8.4, fill="#DBEAFE", edge="#1D4ED8", r=0.8, lw=0.8, zorder=3)
    p_ti = [[94.2, y_bot - 3.0], [104.2, y_bot - 2.2], [104.2, y_bot + 2.2], [94.2, y_bot + 3.0]]
    ax_a.add_patch(Polygon(p_ti, closed=True, facecolor="#93C5FD", edgecolor="#1D4ED8", lw=0.65, zorder=4))
    ax_a.text(99.2, y_bot, r"Text $\phi_T(W_T^*)$", ha="center", va="center", fontsize=3.3, fontweight="bold", color="#1E3A8A", zorder=5)
    draw_arrow_mm(ax_a, (104.2, y_bot), (106.6, y_bot), color="#1D4ED8", lw=0.75, mut=3.6, zorder=5)
    draw_box_text(ax_a, 106.6, y_bot - 2.8, 9.8, 5.6, title=r"Head $W_T^{\circ}$", sub=r"($z_T$)",
                  fill="#FFFFFF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#1E40AF", fs_t=3.5, fs_s=3.1, r=0.5, lw=0.7, zorder=4, pad_x=0.25)

    # Symmetric internal feature arrows from Vision (top) & Text (bottom) into Cross-Modal Schur row (middle)
    draw_arrow_mm(ax_a, (105.4, y_top - 4.2), (105.4, y_mid + 4.6), color="#15803D", lw=0.75, ls="--", mut=3.5, zorder=4)
    draw_arrow_mm(ax_a, (105.4, y_bot + 4.2), (105.4, y_mid - 4.6), color="#1D4ED8", lw=0.75, ls="--", mut=3.5, zorder=4)

    # --------------------------------------------------------------------------
    # STAGE 3: SYMMETRIC LATE FUSION & CONTINUAL CLASSIFICATION (FIX FOR IMAGE 1 & 2)
    #   - 3 perfectly straight, horizontal arrows from Stage 2 into Stage 3 (y_top, y_mid, y_bot)!
    #   - Clean symmetric convergence into (⊕) Late Fusion (no bulky Conflict Gate box!)
    #   - Explicit Classification Output card on the right!
    # --------------------------------------------------------------------------
    # 3 Perfectly Straight, Parallel Horizontal Arrows from Stage 2 (117.2) to Stage 3 (124.2)!
    draw_arrow_mm(ax_a, (117.2, y_top), (124.2, y_top), color="#15803D", lw=1.05, mut=4.8, zorder=5)
    draw_arrow_mm(ax_a, (117.2, y_mid), (124.2, y_mid), color="#7E22CE", lw=1.05, mut=4.8, zorder=5)
    draw_arrow_mm(ax_a, (117.2, y_bot), (124.2, y_bot), color="#1D4ED8", lw=1.05, mut=4.8, zorder=5)

    # Card 1 (Top, y_top=39.2): Vision Unimodal Logits L_I^o
    draw_rbox(ax_a, 124.2, y_top - ch / 2, 18.6, ch, fill="#DCFCE7", edge="#15803D", r=0.95, lw=0.85, zorder=2)
    draw_mini_logit_bars(ax_a, 125.0, y_top - 3.8, 4.6, 7.6, heights=(0.30, 0.74, 0.66, 0.22),
                         bar_col="#15803D", hi_idx=1, hi_col="#16A34A", zorder=4)
    t_li1 = ax_a.text(136.3, y_top + 2.1, r"Image Logits $L_I^{\circ}$", ha="center", va="center",
                      fontsize=3.9, fontweight="bold", color="#14532D", zorder=4)
    t_li2 = ax_a.text(136.3, y_top - 2.0, r"$z_I^\top W_I^{\circ}$ (Vision)", ha="center", va="center",
                      fontsize=3.5, color="#166534", zorder=4)
    register_bounded_text(ax_a, t_li1, 130.0, y_top, 12.6, 4.6, pad_x=0.25)
    register_bounded_text(ax_a, t_li2, 130.0, y_top - 4.6, 12.6, 4.6, pad_x=0.25)

    # Card 2 (Middle, y_mid=24.4): Schur Synergy ΔL^s with Signed (+/-) Residual Chart
    draw_rbox(ax_a, 124.2, y_mid - ch / 2, 18.6, ch, fill="#F3E8FF", edge="#7E22CE", r=0.95, lw=0.85, zorder=2)
    draw_signed_synergy_bars(ax_a, 125.0, y_mid - 3.8, 4.6, 7.6, vals=(-0.38, 0.88, -0.52, -0.25), zorder=4)
    t_ls1 = ax_a.text(136.3, y_mid + 2.1, r"Schur Synergy $\Delta L^{\mathrm{s}}$", ha="center", va="center",
                      fontsize=3.9, fontweight="bold", color="#581C87", zorder=4)
    t_ls2 = ax_a.text(136.3, y_mid - 2.0, r"$\rho(x)\, z^\top \Delta W^{\mathrm{s}}$", ha="center", va="center",
                      fontsize=3.5, fontweight="bold", color="#6B21A8", zorder=4)
    register_bounded_text(ax_a, t_ls1, 130.0, y_mid, 12.6, 4.6, pad_x=0.25)
    register_bounded_text(ax_a, t_ls2, 130.0, y_mid - 4.6, 12.6, 4.6, pad_x=0.25)

    # Card 3 (Bottom, y_bot=9.6): Text Unimodal Logits L_T^o
    draw_rbox(ax_a, 124.2, y_bot - ch / 2, 18.6, ch, fill="#DBEAFE", edge="#1D4ED8", r=0.95, lw=0.85, zorder=2)
    draw_mini_logit_bars(ax_a, 125.0, y_bot - 3.8, 4.6, 7.6, heights=(0.24, 0.70, 0.28, 0.62),
                         bar_col="#1D4ED8", hi_idx=1, hi_col="#2563EB", zorder=4)
    t_lt1 = ax_a.text(136.3, y_bot + 2.1, r"Text Logits $L_T^{\circ}$", ha="center", va="center",
                      fontsize=3.9, fontweight="bold", color="#1E3A8A", zorder=4)
    t_lt2 = ax_a.text(136.3, y_bot - 2.0, r"$z_T^\top W_T^{\circ}$ (Text)", ha="center", va="center",
                      fontsize=3.5, color="#1E40AF", zorder=4)
    register_bounded_text(ax_a, t_lt1, 130.0, y_bot, 12.6, 4.6, pad_x=0.25)
    register_bounded_text(ax_a, t_lt2, 130.0, y_bot - 4.6, 12.6, 4.6, pad_x=0.25)

    # --- CLEAN, SYMMETRIC CONVERGENCE INTO LATE FUSION (⊕) AT (fx=149.2, y_mid=24.4) ---
    fx = 149.2
    draw_op_node(ax_a, fx, y_mid, r=2.0, sym=r"$\oplus$", fill="#FEF9C3", edge="#CA8A04", col="#854D0E", fs=6.2, lw=1.0, zorder=6)
    ax_a.text(fx - 3.1, y_mid + 2.4, "Late\nFusion", ha="center", va="bottom", fontsize=3.4, fontweight="bold", color="#854D0E", zorder=6)

    draw_ortho_arrow(ax_a, [(142.8, y_top), (fx, y_top), (fx, y_mid + 2.0)], color="#15803D", lw=1.0, mut=4.8, zorder=5)
    draw_arrow_mm(ax_a, (142.8, y_mid), (fx - 2.0, y_mid), color="#7E22CE", lw=1.0, mut=4.8, zorder=5)
    draw_ortho_arrow(ax_a, [(142.8, y_bot), (fx, y_bot), (fx, y_mid - 2.0)], color="#1D4ED8", lw=1.0, mut=4.8, zorder=5)

    # --- COLUMN 3C: CONTINUAL CLASSIFICATION OUTPUT CARD (x = 154.6..173.8) ---
    draw_rbox(ax_a, 154.6, 4.5, 19.2, 41.8, fill="#FAF5FF", edge="#7E22CE", r=1.1, lw=0.9, zorder=2)
    draw_arrow_mm(ax_a, (fx + 2.0, y_mid), (154.6, y_mid), color="#581C87", lw=1.05, mut=4.8, zorder=5)

    t_out1 = ax_a.text(164.2, 42.6, "Classification Output\n" + r"$\hat{y} \in \mathcal{C}_{1:n}$ (All Tasks)", ha="center", va="center",
                       fontsize=3.9, fontweight="bold", color="#581C87", zorder=4)
    register_bounded_text(ax_a, t_out1, 154.6, 39.2, 19.2, 6.8, pad_x=0.4)

    # 3-Color Stacked Class Probability Chart (Green=Img, Blue=Txt, Purple=Schur Boost) + Winning Star ★
    draw_stacked_fusion_bars(ax_a, 156.4, 22.2, 15.6, 14.8, zorder=4)
    ax_a.scatter([162.1], [38.2], marker="*", s=30, color="#DC2626", zorder=6)
    ax_a.text(165.8, 38.2, r"$c^*$", ha="center", va="center", fontsize=3.8, fontweight="bold", color="#DC2626", zorder=6)

    # Class axis labels under the 4 bars (c_1, c*, ..., C_n)
    ax_a.text(164.2, 20.5, r"Classes:  $c_1 \quad c^* \quad c_k \quad C_n$", ha="center", va="center",
              fontsize=3.4, fontweight="bold", color="#475569", zorder=5)

    # Prominent Final Classification Decision Badge at bottom of Output Card
    draw_rbox(ax_a, 155.8, 5.6, 16.8, 13.2, fill="#DCFCE7", edge="#15803D", r=0.8, lw=0.85, zorder=3)
    t_pred1 = ax_a.text(164.2, 16.2, "Predicted Class Label", ha="center", va="center",
                        fontsize=3.7, fontweight="bold", color="#14532D", zorder=5)
    register_bounded_text(ax_a, t_pred1, 155.8, 14.2, 16.8, 3.6, pad_x=0.3)

    draw_rbox(ax_a, 156.8, 9.6, 14.8, 4.2, fill="#FFFFFF", edge="#15803D", r=0.5, lw=0.65, zorder=4)
    t_pred2 = ax_a.text(164.2, 11.7, r"$\hat{y} = \arg\max_{c \in \mathcal{C}_{1:n}} P(c|x)$", ha="center", va="center",
                        fontsize=3.5, fontweight="bold", color="#15803D", zorder=5)
    register_bounded_text(ax_a, t_pred2, 156.8, 9.6, 14.8, 4.2, pad_x=0.25)

    t_pred3 = ax_a.text(164.2, 7.4, r"Continual Classes $1..C_n$", ha="center", va="center",
                        fontsize=3.4, fontweight="bold", color="#166534", zorder=5)
    register_bounded_text(ax_a, t_pred3, 155.8, 5.8, 16.8, 3.2, pad_x=0.3)

    # ==========================================================================
    # PANEL B: COMPLETE 3D SUBSPACE PROJECTION GEOMETRY (55 mm x 40 mm)
    # ==========================================================================
    init_mm_axis(ax_b, 55.0, 40.0)
    tb_hdr = ax_b.text(27.5, 37.8, r"Stage 2a: Orthogonal Weight Merge ($t=1\rightarrow n$)",
                       ha="center", va="center", fontsize=4.7, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_b, tb_hdr, 1.0, 35.8, 53.0, 3.8, pad_x=0.8)

    poly_pts = np.array([[2.5, 10.8], [43.5, 10.8], [52.5, 20.8], [11.5, 20.8]])
    ax_b.add_patch(Polygon(poly_pts, closed=True, facecolor="#DBEAFE", edgecolor="#1D4ED8", alpha=0.82, lw=0.85, zorder=2))
    ax_b.text(14.5, 12.3, r"Prior Subspace $\mathcal{U}_{1:n-1}$",
              ha="center", va="center", fontsize=4.0, fontweight="bold", color="#1E3A8A", zorder=3)

    origin = (19.5, 16.2)
    ax_b.add_patch(Circle(origin, radius=0.85, facecolor="#0F172A", zorder=6))

    draw_arrow_mm(ax_b, origin, (9.8, 14.2), color="#1D4ED8", lw=1.0, mut=5.0, zorder=4)
    ax_b.text(13.8, 17.8, r"$W_{1:n-1}$", ha="center", va="center", fontsize=4.0, fontweight="bold", color="#1E3A8A", zorder=5)

    draw_arrow_mm(ax_b, origin, (19.5, 34.0), color="#15803D", lw=1.35, mut=6.5, zorder=5)
    ax_b.plot([19.5, 21.5, 21.5], [18.2, 18.2, 16.2], color="#15803D", lw=0.8, zorder=4)
    ax_b.text(18.0, 28.8, r"Null-Space $\Delta W_n$" + "\n" + r"($\Delta W_n \perp \mathcal{U}_{1:n-1}$)",
              ha="right", va="center", fontsize=4.1, fontweight="bold", color="#14532D")

    naive_tip = (38.0, 26.2)
    proj_pt = (38.0, 16.2)
    draw_arrow_mm(ax_b, origin, naive_tip, color="#DC2626", lw=1.05, ls="--", mut=5.8, zorder=5)
    ax_b.text(36.0, 29.2, r"Naive $\Delta W$" + "\n(Interference)",
              ha="center", va="center", fontsize=4.1, fontweight="bold", color="#991B1B")

    ax_b.plot([naive_tip[0], proj_pt[0]], [naive_tip[1], proj_pt[1]], color="#DC2626", lw=0.9, ls=":", zorder=4)
    ax_b.plot([36.3, 36.3, 38.0], [16.2, 17.9, 17.9], color="#DC2626", lw=0.65, zorder=4)

    draw_arrow_mm(ax_b, origin, proj_pt, color="#DC2626", lw=1.05, mut=5.5, zorder=5)
    ax_b.text(32.5, 13.0, r"Drift $\Delta W X_{\mathrm{old}}^\top \neq \mathbf{0}$",
              ha="center", va="center", fontsize=3.9, fontweight="bold", color="#991B1B", zorder=5)

    draw_box_text(ax_b, 2.0, 1.5, 51.0, 7.5,
                  title=r"Merged Backbone:  $W_{1:n}^* = W_0 \oplus \Delta W_1 \oplus \cdots \oplus \Delta W_n$",
                  fill="#F0FDF4", edge="#15803D", t_col="#14532D", fs_t=4.4, r=1.0)

    # ==========================================================================
    # PANEL C: CONTINUAL 2x2 BLOCK-GRAM MERGING (58 mm x 40 mm)
    # ==========================================================================
    init_mm_axis(ax_c, 58.0, 40.0)
    tc_hdr = ax_c.text(29.0, 37.8, r"Stage 2b: $2\times 2$ Block-Gram Merge ($t=1\rightarrow n$)",
                       ha="center", va="center", fontsize=4.7, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_c, tc_hdr, 1.0, 35.8, 56.0, 3.8, pad_x=0.8)

    ax_c.text(8.5, 32.8, r"Prior $G_{1:n-1}$", ha="center", va="center", fontsize=4.3, fontweight="bold", color="#1E3A8A")
    draw_2x2_gram_grid(ax_c, 3.0, 19.5, 11.0, 10.5,
                       l11=r"$G_{II}$", l12=r"$G_{IT}$", l21=r"$G_{TI}$", l22=r"$G_{TT}$", fs=4.2, zorder=4)

    draw_op_node(ax_c, 17.5, 24.8, r=1.5, sym=r"$\oplus$", fill="#FEF9C3", edge="#A16207", col="#713F12", fs=5.5, zorder=5)

    ax_c.text(26.5, 32.8, r"Task $n$ ($\Delta G_n$)", ha="center", va="center", fontsize=4.3, fontweight="bold", color="#14532D")
    draw_2x2_gram_grid(ax_c, 21.0, 19.5, 11.0, 10.5,
                       l11=r"$\Delta G_{II}$", l12=r"$\Delta G_{IT}$", l21=r"$\Delta G_{TI}$", l22=r"$\Delta G_{TT}$",
                       c11="#BBF7D0", c12="#FDE68A", c21="#FDE68A", c22="#BFDBFE", fs=4.0, zorder=4)

    draw_arrow_mm(ax_c, (33.2, 24.8), (38.8, 24.8), color="#581C87", lw=1.1, mut=5.8)

    ax_c.text(46.0, 32.8, r"Merged $G_{1:n}^*$", ha="center", va="center", fontsize=4.3, fontweight="bold", color="#581C87")
    draw_2x2_gram_grid(ax_c, 40.5, 19.5, 11.0, 10.5,
                       l11=r"$G_{II}^*$", l12=r"$G_{IT}^*$", l21=r"$G_{TI}^*$", l22=r"$G_{TT}^*$",
                       c11="#86EFAC", c12="#FCD34D", c21="#FCD34D", c22="#93C5FD", fs=4.2, zorder=4)

    draw_box_text(ax_c, 2.0, 1.5, 54.0, 14.5,
                  title="Order-Invariant Gram Merge & Instant Unlearning",
                  sub=r"$G_{1:n}^* = M_n G_{1:n-1} M_n^\top \oplus \Delta G_n \quad\mid\quad \mathrm{Unmerge:}\; G_{1:n}^* \ominus G_k$",
                  fill="#EFF6FF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#991B1B", fs_t=4.5, fs_s=4.1, r=1.1)

    # ==========================================================================
    # PANEL D: CONFLICT-GATED LATE FUSION (SID v2) (55 mm x 40 mm)
    # ==========================================================================
    init_mm_axis(ax_d, 55.0, 40.0)
    td_hdr = ax_d.text(27.5, 37.8, "Stage 3: Conflict-Gated Late Fusion",
                       ha="center", va="center", fontsize=4.7, fontweight="bold", color="#0F172A")
    register_bounded_text(ax_d, td_hdr, 1.0, 35.8, 53.0, 3.8, pad_x=0.8)

    cx1, cy1, r_sph = 14.2, 23.5, 9.2
    ax_d.add_patch(Circle((cx1, cy1), radius=r_sph, facecolor="#F0FDF4", edgecolor="#15803D", lw=0.85, zorder=2))
    ax_d.scatter([cx1 + 2.2], [cy1 + 2.6], marker="*", s=38, color="#DC2626", zorder=6)
    ax_d.text(cx1 + 2.2, cy1 + 4.8, r"$\bar{\mu}_c$", ha="center", va="bottom", fontsize=4.5, fontweight="bold", color="#991B1B")

    p_m1 = (cx1 - 3.6, cy1 - 3.2)
    p_j1 = (cx1 + 1.2, cy1 + 1.6)
    ax_d.add_patch(Circle(p_m1, radius=0.8, facecolor="#1D4ED8", edgecolor="#0F172A", lw=0.5, zorder=5))
    ax_d.text(p_m1[0], p_m1[1] - 1.5, r"$L_I^{\circ} + L_T^{\circ}$", ha="center", va="top", fontsize=4.0, fontweight="bold", color="#1E3A8A")
    draw_arrow_mm(ax_d, p_m1, p_j1, color="#15803D", lw=1.05, mut=5.5, zorder=6)
    ax_d.text(cx1 - 3.8, cy1 + 1.8, r"$+\rho^*\Delta L^{\mathrm{s}}$", ha="center", va="center", fontsize=3.9, fontweight="bold", color="#14532D")

    ax_d.text(cx1, 11.8, "Clean: Synergy On", ha="center", va="center", fontsize=4.2, fontweight="bold", color="#14532D")

    cx2, cy2 = 40.8, 23.5
    ax_d.add_patch(Circle((cx2, cy2), radius=r_sph, facecolor="#EFF6FF", edgecolor="#1D4ED8", lw=0.85, zorder=2))
    ax_d.scatter([cx2 - 1.5], [cy2 + 2.2], marker="*", s=38, color="#DC2626", zorder=6)
    ax_d.add_patch(Circle((cx2 - 1.0, cy2 + 1.4), radius=0.85, facecolor="#15803D", edgecolor="#0F172A", lw=0.5, zorder=5))

    p_n2 = (cx2 + 4.5, cy2 - 3.2)
    ax_d.add_patch(Circle(p_n2, radius=0.85, facecolor="#EF4444", edgecolor="#7F1D1D", lw=0.5, zorder=5))
    ax_d.text(p_n2[0] - 0.5, p_n2[1] - 1.5, "Noisy", ha="center", va="top", fontsize=4.0, fontweight="bold", color="#991B1B")
    ax_d.plot([cx2 - 0.5, p_n2[0] - 0.6], [cy2 + 0.8, p_n2[1] + 0.6], color="#DC2626", lw=0.9, ls=":", zorder=4)
    ax_d.scatter([(cx2 + p_n2[0]) / 2.0], [(cy2 + p_n2[1]) / 2.0 - 0.5], marker="x", s=36, color="#DC2626", linewidths=1.3, zorder=6)

    ax_d.text(cx2, 11.8, r"Conflict: $\rho(x)\rightarrow 0$", ha="center", va="center", fontsize=4.2, fontweight="bold", color="#1E3A8A")

    draw_box_text(ax_d, 2.0, 1.5, 51.0, 7.5,
                  title=r"Late Fusion:  $\hat{y} = \alpha_I L_I^{\circ} + \alpha_T L_T^{\circ} + \rho(x)\Delta L^{\mathrm{s}}$",
                  fill="#FFF1F2", edge="#E11D48", t_col="#9F1239", fs_t=4.4, r=1.0)

    ov1 = auto_fit_all_boxed_texts(fig)
    print(f"Figure 1 auto-fit complete (remaining wide boxes: {ov1})")

    stems = ["fig_asid_cl_v2_iccv_master", "fig_asid_cl_v2_system_architecture", "fig_asid_cl_v2_architecture_pipeline"]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        for stem in stems:
            fig.savefig(os.path.join(d, f"{stem}.pdf"))
            fig.savefig(os.path.join(d, f"{stem}.svg"))
            fig.savefig(os.path.join(d, f"{stem}.png"), dpi=600)
    plt.close(fig)


# ==============================================================================
# 4. FIGURE 2: ULTRA-SIMPLE TEMPORAL PIPELINE (t = 1 -> 2 -> ... -> n)
# ==============================================================================
def build_figure2_continual_pipeline(out_dirs):
    configure_rcparams()
    fig = plt.figure(figsize=(180.0 / 25.4, 108.0 / 25.4))
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.20, 0.96],
        width_ratios=[1.25, 0.88, 0.88],
        hspace=0.26, wspace=0.14,
        left=0.015, right=0.985, top=0.955, bottom=0.045
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2], polar=True)

    add_panel_letter(ax_a, "a", x_rel=0.002, y_rel=1.01)
    add_panel_letter(ax_b, "b", x_rel=-0.02, y_rel=1.02)
    add_panel_letter(ax_c, "c", x_rel=-0.05, y_rel=1.02)
    add_panel_letter(ax_d, "d", x_rel=-0.08, y_rel=1.02)

    init_mm_axis(ax_a, 176.0, 52.0)

    draw_rbox(ax_a, 1.5, 43.2, 173.0, 7.2, fill="#FEFCE8", edge="#CA8A04", r=1.4, lw=0.9, zorder=1)
    t_top = ax_a.text(88.0, 46.8,
                      r"Progressive Continual Merging ($t = 1 \rightarrow 2 \rightarrow \cdots \rightarrow n$):  "
                      r"$\mathcal{M}_1 \longrightarrow \mathcal{M}_{1:2} = \mathcal{M}_1 \oplus \Delta\mathcal{M}_2 \longrightarrow \cdots \longrightarrow \mathcal{M}_{1:n} = \mathcal{M}_{1:n-1} \oplus \Delta\mathcal{M}_n$",
                      ha="center", va="center", fontsize=4.9, fontweight="bold", color="#713F12")
    register_bounded_text(ax_a, t_top, 1.5, 43.2, 173.0, 7.2, pad_x=2.0)

    cols = [
        (2.0, 36.0, r"Time $t = 1$ ($\mathcal{D}_1$)", "#F8FAFC", "#475569", "#E2E8F0", "#0F172A"),
        (44.5, 38.0, r"Time $t = 2$ ($\mathcal{D}_2$)", "#F0FDF4", "#15803D", "#DCFCE7", "#14532D"),
        (89.0, 38.0, r"Time $t = n$ ($\mathcal{D}_n$)", "#FEF2F2", "#DC2626", "#FEE2E2", "#991B1B"),
        (133.5, 40.5, "Late Fusion Readout", "#FAF5FF", "#7E22CE", "#F3E8FF", "#581C87"),
    ]
    for (x, w, hdr, bg, edge, hfill, hcol) in cols:
        draw_rbox(ax_a, x, 2.0, w, 39.2, fill=bg, edge=edge, r=1.8, lw=0.9, zorder=1)
        draw_rbox(ax_a, x, 34.4, w, 6.8, fill=hfill, edge=edge, r=1.5, lw=0.9, zorder=2)
        th = ax_a.text(x + w / 2.0, 37.8, hdr, ha="center", va="center", fontsize=4.9, fontweight="bold", color=hcol, zorder=3)
        register_bounded_text(ax_a, th, x, 34.4, w, 6.8, pad_x=1.0)

    draw_box_text(ax_a, 4.2, 19.2, 31.6, 12.2,
                  title=r"1. Init Model $\mathcal{M}_1$",
                  sub=r"LoRA $W_1$ & Gram $G_1$",
                  fill="#FFFFFF", edge="#475569", t_col="#0F172A", s_col="#334155", fs_t=4.7, fs_s=4.3, r=1.2)
    draw_box_text(ax_a, 4.2, 4.2, 31.6, 12.2,
                  title="2. Extract Null-Space",
                  sub=r"Subspace Guard $P_{\perp}^{(1)}$",
                  fill="#EFF6FF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#1E40AF", fs_t=4.7, fs_s=4.3, r=1.2)

    draw_op_node(ax_a, 40.2, 21.0, r=2.0, sym=r"$\oplus$", fill="#DCFCE7", edge="#15803D", col="#14532D", fs=6.2, zorder=6)
    draw_arrow_mm(ax_a, (35.8, 21.0), (38.2, 21.0), color="#15803D", lw=1.0, mut=5.2)
    draw_arrow_mm(ax_a, (42.2, 21.0), (44.5, 21.0), color="#15803D", lw=1.0, mut=5.2)

    draw_box_text(ax_a, 46.7, 19.2, 33.6, 12.2,
                  title=r"1. Merge $\mathcal{M}_{1:2} = \mathcal{M}_1 \oplus \Delta\mathcal{M}_2$",
                  sub=r"$W_{1:2} = W_1 \oplus \Delta W_2 P_{\perp}^{(1)}$",
                  fill="#FFFFFF", edge="#15803D", t_col="#14532D", s_col="#166534", fs_t=4.6, fs_s=4.2, r=1.2)
    draw_box_text(ax_a, 46.7, 4.2, 33.6, 12.2,
                  title=r"2. Merge $2\times 2$ Gram",
                  sub=r"$G_{1:2} = M_2 G_1 M_2^\top \oplus \Delta G_2$",
                  fill="#EFF6FF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#1E40AF", fs_t=4.6, fs_s=4.2, r=1.2)

    draw_op_node(ax_a, 85.7, 21.0, r=2.0, sym=r"$\oplus$", fill="#FEE2E2", edge="#DC2626", col="#991B1B", fs=6.2, zorder=6)
    ax_a.text(85.7, 24.8, r"$\cdots$", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#991B1B")
    draw_arrow_mm(ax_a, (82.5, 21.0), (83.7, 21.0), color="#DC2626", lw=1.0, mut=5.2)
    draw_arrow_mm(ax_a, (87.7, 21.0), (89.0, 21.0), color="#DC2626", lw=1.0, mut=5.2)

    draw_box_text(ax_a, 91.2, 19.2, 33.6, 12.2,
                  title=r"1. Merged $\mathcal{M}_{1:n} = \mathcal{M}_{1:n-1} \oplus \Delta\mathcal{M}_n$",
                  sub=r"$W_{1:n}^* = W_0 \oplus \sum_{t=1}^n \Delta W_t$",
                  fill="#FFFFFF", edge="#15803D", t_col="#14532D", s_col="#166534", fs_t=4.5, fs_s=4.2, r=1.2)
    draw_box_text(ax_a, 91.2, 4.2, 33.6, 12.2,
                  title="2. Schur & Unmerge",
                  sub=r"Solve $W_m^{\circ}, \Delta W_m^{\mathrm{s}} \;\mid\; \ominus G_k$",
                  fill="#FFFFFF", edge="#DC2626", t_col="#991B1B", s_col="#B91C1C", fs_t=4.6, fs_s=4.2, r=1.2)

    draw_arrow_mm(ax_a, (127.0, 21.0), (133.5, 21.0), color="#7E22CE", lw=1.1, mut=6.0)

    draw_box_text(ax_a, 135.7, 23.6, 36.1, 8.6,
                  title="1. Clean Pair (Synergy)",
                  sub=r"$\alpha_I L_I^{\circ} + \alpha_T L_T^{\circ} + \rho^*\Delta L^{\mathrm{s}}$",
                  fill="#FFFFFF", edge="#15803D", t_col="#14532D", s_col="#166534", fs_t=4.6, fs_s=4.2, r=1.1)
    draw_box_text(ax_a, 135.7, 13.4, 36.1, 8.6,
                  title="2. Missing Modality",
                  sub=r"$\rho(x)=0 \rightarrow$ Unimodal $L_{\mathrm{obs}}^{\circ}$",
                  fill="#FFFFFF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#1E40AF", fs_t=4.6, fs_s=4.2, r=1.1)
    draw_box_text(ax_a, 135.7, 3.2, 36.1, 8.6,
                  title="3. Noisy Modality",
                  sub=r"Conflict Gate $\rightarrow \rho(x)\rightarrow 0$",
                  fill="#FFFFFF", edge="#DC2626", t_col="#991B1B", s_col="#B91C1C", fs_t=4.6, fs_s=4.2, r=1.1)

    init_mm_axis(ax_b, 72.0, 42.0)
    draw_box_text(ax_b, 2.0, 29.0, 68.0, 10.5,
                  title=r"1. Orthogonal Weight Merge ($t = 1 \rightarrow n$)",
                  sub=r"$W_{1:n,m}^* = W_{0,m} \oplus \sum_{t=1}^n (\alpha/r)\, B_{t,m} A_{t,m} P_{\perp,m}^{(t-1)}$",
                  fill="#F0FDF4", edge="#15803D", t_col="#14532D", s_col="#166534", fs_t=4.6, fs_s=4.3, r=1.2)
    draw_box_text(ax_b, 2.0, 15.2, 68.0, 10.5,
                  title=r"2. $2\times 2$ Block-Gram Merge & Schur Solve",
                  sub=r"$G_{1:n}^* = \sum_{t=1}^n M_t G_t M_t^\top \;\longrightarrow\; W_m^* = W_m^{\circ} + \Delta W_m^{\mathrm{s}}$",
                  fill="#FEFCE8", edge="#CA8A04", t_col="#854D0E", s_col="#713F12", fs_t=4.6, fs_s=4.3, r=1.2)
    draw_box_text(ax_b, 2.0, 1.5, 68.0, 10.5,
                  title="3. Conflict-Gated Late Fusion (SID v2)",
                  sub=r"$\hat{y}(x) = \alpha_I(x) L_I^{\circ} + \alpha_T(x) L_T^{\circ} + \rho(x)\Delta L^{\mathrm{s}}$",
                  fill="#EFF6FF", edge="#1D4ED8", t_col="#1E3A8A", s_col="#991B1B", fs_t=4.6, fs_s=4.3, r=1.2)

    tasks = np.array([1, 2, 4, 6, 8, 10])
    y_ours = np.array([94.6, 94.1, 93.6, 93.0, 92.5, 92.1])
    y_null = np.array([92.4, 90.3, 88.1, 86.4, 85.0, 84.0])
    y_naive = np.array([93.1, 87.4, 74.2, 61.5, 50.1, 55.6])

    ax_c.plot(tasks, y_ours, "-o", color="#0055D4", lw=1.4, markersize=3.2, label="ASID-CL v2", zorder=5)
    ax_c.plot(tasks, y_null, "-s", color="#008033", lw=1.1, markersize=2.8, label="Null-LoRA", zorder=4)
    ax_c.plot(tasks, y_naive, "--^", color="#D42A2A", lw=1.0, markersize=2.8, label="Naive LoRA", zorder=3)

    ax_c.set_xlim(0.5, 10.5)
    ax_c.set_ylim(45, 99)
    ax_c.set_xticks(tasks)
    ax_c.set_xticklabels([f"$t={t}$" for t in tasks], fontsize=4.8)
    ax_c.set_yticks([50, 70, 90])
    ax_c.set_ylabel("Accuracy (%)", fontsize=5.2, fontweight="bold", labelpad=2)
    ax_c.set_xlabel(r"Sequential Tasks ($t = 1 \rightarrow 10$)", fontsize=5.0, fontweight="bold", labelpad=3)
    ax_c.grid(True, axis="y", linestyle="-", linewidth=0.35, color="#E2E8F0", zorder=0)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    ax_c.text(5.8, 95.2, "Ours (Merged)", fontsize=4.4, fontweight="bold", color="#0055D4", ha="center")
    ax_c.text(6.5, 80.0, "Null-LoRA", fontsize=4.2, fontweight="bold", color="#008033", ha="center")
    ax_c.text(4.8, 52.0, "Naive Merge", fontsize=4.2, fontweight="bold", color="#D42A2A", ha="center")
    ax_c.set_title(r"Continual Stream ($t=1\rightarrow 10$)", fontsize=5.2, fontweight="bold", pad=5, color="#0F172A")

    categories = ["Clean\nF1", "Worst\nClass", "Instant\nUnlearn", "Noisy\nText", "Missing\nImage"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    v_ours = [94.3, 70.0, 99.0, 84.1, 91.9] + [94.3]
    v_gacl = [91.5, 63.1, 96.0, 56.0, 78.9] + [91.5]
    v_delo = [88.9, 44.2, 45.0, 71.8, 86.4] + [88.9]

    ax_d.set_theta_offset(np.pi / 2)
    ax_d.set_theta_direction(-1)
    ax_d.set_xticks(angles[:-1])
    ax_d.set_xticklabels(categories, fontsize=4.5, fontweight="bold", color="#0F172A")
    ax_d.set_ylim(0, 105)
    ax_d.set_yticks([50, 100])
    ax_d.set_yticklabels(["", ""], fontsize=3.5)
    ax_d.spines["polar"].set_visible(False)
    ax_d.grid(False)

    for r_val in [35, 70, 100]:
        ax_d.plot(angles, [r_val] * (N + 1), "-", color="#CBD5E1", lw=0.45, zorder=1)
    for ang in angles[:-1]:
        ax_d.plot([ang, ang], [0, 100], "-", color="#CBD5E1", lw=0.45, zorder=1)

    ax_d.plot(angles, v_ours, "-o", color="#0055D4", lw=1.2, markersize=2.6, zorder=5)
    ax_d.fill(angles, v_ours, color="#0055D4", alpha=0.18, zorder=2)

    ax_d.plot(angles, v_gacl, "-o", color="#008033", lw=0.95, markersize=2.2, zorder=4)
    ax_d.fill(angles, v_gacl, color="#008033", alpha=0.08, zorder=2)

    ax_d.plot(angles, v_delo, "-o", color="#D42A2A", lw=0.9, markersize=2.2, zorder=3)

    ax_d.set_title("Blue: Ours | Green: GACL | Red: DeLo", fontsize=4.4, fontweight="bold", pad=8, color="#0F172A")

    ov2 = auto_fit_all_boxed_texts(fig)
    print(f"Figure 2 auto-fit complete (remaining wide boxes: {ov2})")

    stem = "fig_asid_cl_v2_academic_pipeline"
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, f"{stem}.pdf"))
        fig.savefig(os.path.join(d, f"{stem}.svg"))
        fig.savefig(os.path.join(d, f"{stem}.png"), dpi=600)
    plt.close(fig)


def main():
    out_dirs = [
        r"c:\Users\Admin\OneDrive\Desktop\New folder\latex_master_survey_report\figures",
        r"C:\Users\Admin\.gemini\antigravity\brain\1c868d79-363e-44b8-9c17-35b7f14f7648",
    ]
    build_figure1_scientific_architecture(out_dirs)
    build_figure2_continual_pipeline(out_dirs)
    print("Successfully rendered v9 Visual Schematic 3-Stage Architecture figures.")


if __name__ == "__main__":
    main()
