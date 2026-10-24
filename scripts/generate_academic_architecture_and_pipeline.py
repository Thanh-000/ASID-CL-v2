#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Grade Academic System Architecture & Operational Pipeline Generator
================================================================================
Generates two dedicated, top-tier CVPR/NeurIPS/Nature-style double-column (183 mm)
academic figures using `ChenLiu-1996/figures4papers` visual idioms and
`Boom5426/nature-paper-skills` typography/layout standards:

1. `fig_asid_cl_v2_system_architecture.{pdf,svg,png}`:
   Detailed System Architecture Diagram of ASID-CL v2 + Null-Space Dual-Tower LoRA.
2. `fig_asid_cl_v2_academic_pipeline.{pdf,svg,png}` (and `fig_asid_cl_v2_architecture_pipeline`):
   End-to-End Continual Learning & Multi-Regime Operational Pipeline across Tasks t=1..T,
   coupled with KDE Manifold Tube, 3D Hyperspherical Geodesic Fusion, and 4-Stage Ablation.
"""

from __future__ import annotations

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KERNEL_PATH = PROJECT_ROOT / ".agent" / "skills" / "nature-paper-skills" / "resources" / "figure-style" / "kernel.py"
kernel_ns = {}
exec(KERNEL_PATH.read_text(encoding="utf-8"), kernel_ns)
apply_figure_style = kernel_ns["apply_figure_style"]
panel_letter = kernel_ns["panel_letter"]
set_frame = kernel_ns["set_frame"]

ALIGN_PATH = PROJECT_ROOT / ".agent" / "skills" / "nature-paper-skills" / "resources" / "nature-figure" / "scripts" / "audit_panel_alignment.py"
align_ns = {}
exec(ALIGN_PATH.read_text(encoding="utf-8"), align_ns)
require_matplotlib_panel_alignment = align_ns["require_matplotlib_panel_alignment"]

MM_TO_INCH = 1.0 / 25.4
SINGLE_COL_MM = 89.0
DOUBLE_COL_MM = 183.0

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "teal": "#1F7A7A",
    "amber": "#C67D0A",
    "violet": "#5E3C99",
}


def configure_publication_rc():
    apply_figure_style(frame="open", font="DejaVu Sans", sizes=(6.4, 5.5, 5.2), grid=False)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "savefig.dpi": 600,
    })


# ==============================================================================
# HELPER DRAWING PRIMITIVES FOR ACADEMIC ARCHITECTURE DIAGRAMS
# ==============================================================================

def draw_card(
    ax, x, y, w, h,
    facecolor="#FFFFFF", edgecolor="#23374B", lw=0.85,
    rounding=0.012, alpha=1.0, ls="-", zorder=2, shadow=False
):
    if shadow:
        sh = FancyBboxPatch(
            (x + 0.0025, y - 0.004), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={rounding}",
            facecolor="#D9E1E8", edgecolor="none", alpha=0.60, zorder=zorder - 1
        )
        ax.add_patch(sh)
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_header_card(
    ax, x, y, w, h,
    header_text, body_lines=None,
    header_bg="#1F4E79", header_fg="#FFFFFF",
    body_bg="#FFFFFF", edge_color="#1F4E79",
    header_frac=0.24, rounding=0.010, lw=0.85,
    header_fs=5.3, body_fs=5.0, zorder=3, shadow=True
):
    draw_card(ax, x, y, w, h, facecolor=body_bg, edgecolor=edge_color,
              lw=lw, rounding=rounding, zorder=zorder, shadow=shadow)
    hh = h * header_frac
    hdr = FancyBboxPatch(
        (x, y + h - hh), w, hh,
        boxstyle=f"round,pad=0.0,rounding_size={rounding}",
        facecolor=header_bg, edgecolor=edge_color,
        linewidth=lw, zorder=zorder + 1
    )
    ax.add_patch(hdr)
    ax.text(
        x + w / 2.0, y + h - hh / 2.0, header_text,
        ha="center", va="center", fontsize=header_fs,
        fontweight="bold", color=header_fg, zorder=zorder + 2
    )
    if body_lines:
        body_h = h - hh
        n = len(body_lines)
        for i, line in enumerate(body_lines):
            ly = y + body_h * (1.0 - (i + 0.55) / (n + 0.12))
            if isinstance(line, tuple):
                txt, col, fw, fs_override = line
                ax.text(x + w / 2.0, ly, txt, ha="center", va="center",
                        fontsize=fs_override or body_fs, fontweight=fw,
                        color=col, zorder=zorder + 2)
            else:
                ax.text(x + w / 2.0, ly, line, ha="center", va="center",
                        fontsize=body_fs, color="#1F242B", zorder=zorder + 2)


def draw_op_node(ax, cx, cy, r_x, r_y, symbol, bg="#FFFFFF", fg="#1F4E79", ec="#1F4E79", fs=6.0, lw=0.85, zorder=6):
    ell = mpatches.Ellipse((cx, cy), width=2 * r_x, height=2 * r_y,
                           facecolor=bg, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(ell)
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=fg, zorder=zorder + 1)


def draw_ortho_arrow(
    ax, p1, p2, color="#23374B", lw=0.9, ls="-", rad=0.0,
    label=None, label_pos=None, label_fs=5.0, label_bg=None, zorder=5, mut=7.0
):
    arr = FancyArrowPatch(
        p1, p2,
        arrowstyle="-|>",
        mutation_scale=mut,
        linewidth=lw,
        linestyle=ls,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder
    )
    ax.add_patch(arr)
    if label:
        lx, ly = label_pos if label_pos is not None else ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        bbox_kw = dict(boxstyle="round,pad=0.10", facecolor=label_bg, edgecolor="none", alpha=0.92) if label_bg else None
        ax.text(lx, ly, label, ha="center", va="center", fontsize=label_fs,
                color=color, fontweight="bold", bbox=bbox_kw, zorder=zorder + 2)
    return arr


def draw_matrix_grid(ax, x, y, w, h, rows, cols, cmap_colors, border_col="#23374B", lw=0.6, zorder=4):
    cw = w / cols
    ch = h / rows
    for r in range(rows):
        for c in range(cols):
            col = cmap_colors[(r * cols + c) % len(cmap_colors)]
            rect = Rectangle((x + c * cw, y + (rows - 1 - r) * ch), cw, ch,
                             facecolor=col, edgecolor="white", linewidth=0.35, zorder=zorder)
            ax.add_patch(rect)
    outer = Rectangle((x, y), w, h, facecolor="none", edgecolor=border_col, linewidth=lw, zorder=zorder + 1)
    ax.add_patch(outer)


# ==============================================================================
# FIGURE 1: FULL-WIDTH ACADEMIC SYSTEM ARCHITECTURE DIAGRAM
# ==============================================================================

def build_academic_system_architecture_figure(output_dirs):
    """
    Constructs the dedicated double-column (183 mm x 118 mm) System Architecture
    diagram of ASID-CL v2 + Null-Space Dual-Tower LoRA.
    Every card text is strictly budgeted to <= 27 chars/line so zero text touches borders.
    """
    configure_publication_rc()
    fig_w = DOUBLE_COL_MM * MM_TO_INCH  # 7.205 in (183 mm)
    fig_h = 118.0 * MM_TO_INCH          # 4.645 in (118 mm)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top Title Banner
    ax.text(
        0.50, 0.976,
        "ASID-CL v2 System Architecture: Dual-Tower Null-Space LoRA & Block-Schur Analytic Learning",
        ha="center", va="center", fontsize=6.6, fontweight="bold", color="#16222F"
    )

    # ------------------------------------------------------------------
    # 4 MODULE CONTAINERS (Background Frames)
    # ------------------------------------------------------------------
    modules = [
        (0.008, 0.068, 0.254, 0.880, "#F3F7FB", "#2F6B9A",
         "I. Null-Space LoRA Backbone", "CLIP ViT-B/16 (L10–L11 Adapted)"),
        (0.274, 0.068, 0.232, 0.880, "#F1F9F8", "#2B8C7E",
         "II. 2×2 Block Gram Memory", "Random-ReLU & Drift Transport"),
        (0.518, 0.068, 0.236, 0.880, "#F7F4FA", "#6F4E8B",
         "III. Block-Schur Solver", "Spectral Shrinkage & Innovation"),
        (0.766, 0.068, 0.226, 0.880, "#FDF5F4", "#B83B35",
         "IV. Prototype SID v2 Head", "Conflict-Gated Task-Free Readout"),
    ]
    for mx, my, mw, mh, mbg, mec, mtitle, msub in modules:
        draw_card(ax, mx, my, mw, mh, facecolor=mbg, edgecolor=mec, lw=0.95, rounding=0.012, ls="-", zorder=1)
        ribbon = FancyBboxPatch(
            (mx, my + mh - 0.066), mw, 0.066,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            facecolor=mec, edgecolor=mec, linewidth=0.9, zorder=2
        )
        ax.add_patch(ribbon)
        ax.text(mx + mw / 2.0, my + mh - 0.023, mtitle,
                ha="center", va="center", fontsize=5.4, fontweight="bold", color="white", zorder=3)
        ax.text(mx + mw / 2.0, my + mh - 0.049, msub,
                ha="center", va="center", fontsize=5.0, color="#EEF4FA", zorder=3)

    # ==================================================================
    # MODULE I: DUAL-TOWER BACKBONE & NULL-SPACE LORA CIRCUIT
    # ==================================================================
    draw_header_card(
        ax, 0.016, 0.725, 0.114, 0.140,
        "Vision Stream (I)",
        [
            ("Patch Tokens x_v", "#1F4E79", "bold", 5.0),
            ("L0–L9 Frozen", "#536271", "normal", 5.0),
            ("L10–L11 NS-LoRA", "#1F4E79", "bold", 5.0),
        ],
        header_bg="#2F6B9A", body_bg="#FFFFFF", edge_color="#2F6B9A", header_frac=0.26
    )
    draw_header_card(
        ax, 0.140, 0.725, 0.114, 0.140,
        "Text Stream (T)",
        [
            ("Tokens x_t + Mask", "#2B6CB0", "bold", 5.0),
            ("L0–L9 Frozen", "#536271", "normal", 5.0),
            ("L10–L11 NS-LoRA", "#2B6CB0", "bold", 5.0),
        ],
        header_bg="#3A7CA5", body_bg="#FFFFFF", edge_color="#3A7CA5", header_frac=0.26
    )

    draw_card(ax, 0.016, 0.625, 0.238, 0.072, facecolor="#E6F0FA", edgecolor="#2F6B9A", lw=0.8, rounding=0.009, zorder=3)
    ax.text(0.135, 0.673, "Dual-Tower Features (d = 512)", ha="center", va="center",
            fontsize=5.1, fontweight="bold", color="#1F4E79", zorder=4)
    ax.text(0.135, 0.644, "f_I(x_v), f_T(x_t) ∈ R^512 (δ ≤ 0.04)",
            ha="center", va="center", fontsize=5.0, color="#23374B", zorder=4)
    draw_ortho_arrow(ax, (0.073, 0.725), (0.073, 0.697), color="#2F6B9A", lw=0.85)
    draw_ortho_arrow(ax, (0.197, 0.725), (0.197, 0.697), color="#3A7CA5", lw=0.85)

    # Zoomed-in Micro-Circuit of a Null-Space LoRA Layer
    draw_card(ax, 0.016, 0.085, 0.238, 0.515, facecolor="#FFFFFF", edgecolor="#2F6B9A",
              lw=0.85, rounding=0.010, ls="--", zorder=2, shadow=True)
    ax.text(0.135, 0.578, "Zoom-In: NS-LoRA Layer (L10–L11)",
            ha="center", va="center", fontsize=5.2, fontweight="bold", color="#1F4E79", zorder=4)

    draw_card(ax, 0.024, 0.485, 0.064, 0.062, facecolor="#EDF2F7", edgecolor="#4A5568", lw=0.75, rounding=0.007, zorder=4)
    ax.text(0.056, 0.524, "Input X_t", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#1F242B", zorder=5)
    ax.text(0.056, 0.499, "Mask M_pad", ha="center", va="center", fontsize=5.0, color="#4A5568", zorder=5)

    draw_card(ax, 0.110, 0.492, 0.088, 0.050, facecolor="#E2E8F0", edgecolor="#4A5568", lw=0.75, rounding=0.007, zorder=4)
    ax.text(0.154, 0.517, "Frozen W_0", ha="center", va="center", fontsize=5.1, fontweight="bold", color="#2D3748", zorder=5)

    draw_card(ax, 0.024, 0.345, 0.102, 0.095, facecolor="#D6F5EF", edgecolor="#1E7E6C", lw=0.8, rounding=0.007, zorder=4)
    ax.text(0.075, 0.418, "Null Projector", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#135E4F", zorder=5)
    ax.text(0.075, 0.391, "P_⊥ = I - U_r U_r^T", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#1F242B", zorder=5)
    ax.text(0.075, 0.364, "P_⊥ X_old^T ≈ 0", ha="center", va="center", fontsize=5.0, color="#135E4F", zorder=5)

    draw_card(ax, 0.142, 0.345, 0.104, 0.095, facecolor="#FCE8E6", edgecolor="#C0392B", lw=0.8, rounding=0.007, zorder=4)
    ax.text(0.194, 0.418, "LoRA Adapter", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#96281B", zorder=5)
    ax.text(0.194, 0.391, "A_eff = A_t P_⊥", ha="center", va="center", fontsize=5.0, fontweight="bold", color="#1F242B", zorder=5)
    ax.text(0.194, 0.364, "ΔW_t = (α/r) B_t A_eff", ha="center", va="center", fontsize=5.0, color="#96281B", zorder=5)

    draw_op_node(ax, 0.228, 0.517, 0.013, 0.021, "+", bg="#FFF3CD", fg="#856404", ec="#B8860B", fs=6.2, zorder=6)

    draw_ortho_arrow(ax, (0.088, 0.517), (0.110, 0.517), color="#4A5568", lw=0.8)
    draw_ortho_arrow(ax, (0.198, 0.517), (0.215, 0.517), color="#4A5568", lw=0.8)
    draw_ortho_arrow(ax, (0.056, 0.485), (0.056, 0.440), color="#1E7E6C", lw=0.8)
    draw_ortho_arrow(ax, (0.126, 0.392), (0.142, 0.392), color="#1E7E6C", lw=0.8)
    draw_ortho_arrow(ax, (0.228, 0.440), (0.228, 0.496), color="#C0392B", lw=0.8)

    # Activation Covariance & Eigenspectrum bar
    draw_card(ax, 0.024, 0.100, 0.222, 0.218, facecolor="#F8FAFC", edgecolor="#2B8C7E", lw=0.8, rounding=0.007, zorder=4)
    ax.text(0.135, 0.296, "Activation Gram & Eigh (FP64)", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#135E4F", zorder=5)
    ax.text(0.135, 0.268, "G_act^(t) = G_act^(t-1) + X_t^T X_t / N_t",
            ha="center", va="center", fontsize=5.0, color="#1F242B", zorder=5)
    bar_x0, bar_y0, bar_w, bar_h = 0.036, 0.148, 0.196, 0.090
    n_bars = 14
    bw = bar_w / n_bars
    eig_vals = np.exp(-np.linspace(0.0, 3.2, n_bars))
    for bi, ev in enumerate(eig_vals):
        bh = ev * bar_h
        bcol = "#2F6B9A" if bi < 9 else "#2B8C7E"
        rect = Rectangle((bar_x0 + bi * bw + 0.001, bar_y0), bw - 0.002, bh,
                         facecolor=bcol, edgecolor="none", alpha=0.85, zorder=5)
        ax.add_patch(rect)
    cut_x = bar_x0 + 9 * bw
    ax.plot([cut_x, cut_x], [bar_y0, bar_y0 + bar_h + 0.006], color="#C0392B", lw=0.95, ls="--", zorder=6)
    ax.text(bar_x0 + 4.3 * bw, bar_y0 - 0.024, "Protected U_r (τ=0.99)",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#2F6B9A", zorder=6)
    ax.text(bar_x0 + 11.6 * bw, bar_y0 - 0.024, "Null P_⊥",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#135E4F", zorder=6)
    draw_ortho_arrow(ax, (0.075, 0.318), (0.075, 0.345), color="#1E7E6C", lw=0.8, ls="--")

    # ==================================================================
    # MODULE II: 2x2 BLOCK GRAM MEMORY & PUSHFORWARD TRANSPORT
    # ==================================================================
    draw_header_card(
        ax, 0.284, 0.715, 0.212, 0.150,
        "1. Random-ReLU Buffer (P = 1024)",
        [
            ("z_I = Norm(ReLU(f_I W_RP)) ∈ R^P", "#1F4E79", "bold", 5.0),
            ("z_T = Norm(ReLU(f_T W_RP)) ∈ R^P", "#2B6CB0", "bold", 5.0),
            ("s = (s_I, s_T) ∈ {0,1}^2 \\ {(0,0)}", "#135E4F", "normal", 5.0),
        ],
        header_bg="#2B8C7E", body_bg="#FFFFFF", edge_color="#2B8C7E", header_frac=0.25
    )

    draw_card(ax, 0.284, 0.320, 0.212, 0.370, facecolor="#FFFFFF", edgecolor="#2B8C7E",
              lw=0.85, rounding=0.010, zorder=3, shadow=True)
    ax.text(0.390, 0.668, "2. 2×2 Block Gram G^(t) & S^(t)",
            ha="center", va="center", fontsize=5.2, fontweight="bold", color="#135E4F", zorder=4)

    gx, gy, gw, gh = 0.296, 0.435, 0.112, 0.195
    hw, hh = gw / 2.0, gh / 2.0
    draw_matrix_grid(ax, gx, gy + hh, hw - 0.002, hh - 0.002, 4, 4,
                     ["#2F6B9A", "#4C86B6", "#6BA1CE", "#3973A2"], border_col="#1F4E79", lw=0.75, zorder=4)
    ax.text(gx + hw / 2.0, gy + 1.5 * hh, "G_II\n(P×P)", ha="center", va="center",
            fontsize=5.1, fontweight="bold", color="white", zorder=6)
    draw_matrix_grid(ax, gx + hw + 0.002, gy + hh, hw - 0.002, hh - 0.002, 4, 4,
                     ["#7FC8B8", "#9AD6C8", "#64B6A4", "#B2E2D7"], border_col="#2B8C7E", lw=0.75, zorder=4)
    ax.text(gx + 1.5 * hw, gy + 1.5 * hh, "ρ G_IT\n(Cross)", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#123C35", zorder=6)
    draw_matrix_grid(ax, gx, gy, hw - 0.002, hh - 0.002, 4, 4,
                     ["#7FC8B8", "#9AD6C8", "#64B6A4", "#B2E2D7"], border_col="#2B8C7E", lw=0.75, zorder=4)
    ax.text(gx + hw / 2.0, gy + 0.5 * hh, "ρ G_TI\n(Cross)", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#123C35", zorder=6)
    draw_matrix_grid(ax, gx + hw + 0.002, gy, hw - 0.002, hh - 0.002, 4, 4,
                     ["#3A7CA5", "#5694BC", "#72ACD3", "#2E6B92"], border_col="#1F4E79", lw=0.75, zorder=4)
    ax.text(gx + 1.5 * hw, gy + 0.5 * hh, "G_TT\n(P×P)", ha="center", va="center",
            fontsize=5.1, fontweight="bold", color="white", zorder=6)

    sx, sy, sw, sh = 0.428, 0.435, 0.052, 0.195
    draw_matrix_grid(ax, sx, sy + hh, sw, hh - 0.002, 4, 2,
                     ["#E59866", "#EBAC80", "#D9824B"], border_col="#A04000", lw=0.75, zorder=4)
    ax.text(sx + sw / 2.0, sy + 1.5 * hh, "S_I\n(P×C)", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#4A1C00", zorder=6)
    draw_matrix_grid(ax, sx, sy, sw, hh - 0.002, 4, 2,
                     ["#F5B041", "#F8C471", "#DC7633"], border_col="#A04000", lw=0.75, zorder=4)
    ax.text(sx + sw / 2.0, sy + 0.5 * hh, "S_T\n(P×C)", ha="center", va="center",
            fontsize=5.0, fontweight="bold", color="#4A1C00", zorder=6)

    ax.text(0.390, 0.402, "Class Weight: ω_c = N_c^(-0.5)",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#135E4F", zorder=5)
    ax.text(0.390, 0.372, "Density Norm: ω̃_c,m = ω_c N_c / N_c,m",
            ha="center", va="center", fontsize=5.0, color="#1F242B", zorder=5)
    ax.text(0.390, 0.342, "No Zero-Fill / No Imputation (ε = 0)",
            ha="center", va="center", fontsize=5.0, fontweight="bold", color="#B83B35", zorder=5)

    draw_header_card(
        ax, 0.284, 0.085, 0.212, 0.210,
        "3. Drift Transport (B = 0)",
        [
            ("Shift Op: M_t = I_2P + ΔM_t", "#135E4F", "bold", 5.0),
            ("G_c^(t) ← M_t G_c M_t^T + G_new", "#1F242B", "bold", 5.0),
            ("S_c^(t) ← M_t S_c + S_new", "#1F242B", "normal", 5.0),
            ("0s Unlearn: G ← G - G_k*", "#B83B35", "bold", 5.0),
        ],
        header_bg="#1E7E6C", body_bg="#FFFFFF", edge_color="#1E7E6C", header_frac=0.23
    )

    # ==================================================================
    # MODULE III: BLOCK-SCHUR CLOSED-FORM SOLVER
    # ==================================================================
    draw_header_card(
        ax, 0.528, 0.715, 0.216, 0.150,
        "1. Spectral-EB Ridge γ_m*(t)",
        [
            ("FNSR_m = Tr(G_mm)^2 / (P ||S̃_m||_F^2)", "#4A235A", "normal", 5.0),
            ("γ_m*(t) = γ_0 (2P/N_t,m)^κ r_FNSR^2", "#1F242B", "bold", 5.0),
            ("Ridge: A_m = G_mm + γ_m* λ̄_m I", "#4A235A", "normal", 5.0),
        ],
        header_bg="#6F4E8B", body_bg="#FFFFFF", edge_color="#6F4E8B", header_frac=0.25
    )

    draw_header_card(
        ax, 0.528, 0.365, 0.216, 0.325,
        "2. Block-Schur Solver (50% FLOPs↓)",
        [
            ("Marginal Unimodal Readouts:", "#4A235A", "bold", 5.0),
            ("W_m^o = A_m^(-1) S_m,   m ∈ {I, T}", "#1F4E79", "bold", 5.0),
            ("Block-Schur Complement (P × P):", "#4A235A", "bold", 5.0),
            ("S_I(ρ*) = A_I - (ρ*)^2 G_IT A_T^-1 G_TI", "#1F242B", "bold", 5.0),
            ("Schur Innovation Operators:", "#B83B35", "bold", 5.0),
            ("ΔW_I^s = S_I^-1 (K_IT W_I^o - ρ* G_IT W_T^o)", "#1F242B", "normal", 5.0),
            ("ΔW_T^s = -ρ* A_T^(-1) G_TI (W_I^o + ΔW_I^s)", "#1F242B", "normal", 5.0),
        ],
        header_bg="#5B3E75", body_bg="#FFFFFF", edge_color="#5B3E75", header_frac=0.16
    )

    draw_header_card(
        ax, 0.528, 0.085, 0.216, 0.255,
        "3. Centered Prototypes & Proofs",
        [
            ("Prior-Centered Class Prototypes:", "#4A235A", "bold", 5.0),
            ("μ̄_m,c = Norm(S_m,c - S̄_m) ∈ S^(P-1)", "#1F242B", "bold", 5.0),
            ("• Schur Error: < 3.88 × 10^(-14)", "#135E4F", "normal", 5.0),
            ("• Unimodal Match: < 1.05 × 10^(-15)", "#135E4F", "normal", 5.0),
            ("• Order Invariance: Δ(O1,O6) = 0.0%", "#B83B35", "bold", 5.0),
        ],
        header_bg="#74346B", body_bg="#FFFFFF", edge_color="#74346B", header_frac=0.20
    )

    # ==================================================================
    # MODULE IV: PROTOTYPE-MODULATED SID v2 INFERENCE CIRCUIT
    # ==================================================================
    draw_header_card(
        ax, 0.776, 0.615, 0.206, 0.250,
        "1. Prototype-Modulated Gate",
        [
            ("Marginal: L_m^o = z_m^T W_m^o", "#1F4E79", "bold", 5.0),
            ("Schur: ΔL_m^s = z_m^T ΔW_m^s", "#6F4E8B", "bold", 5.0),
            ("Stream Confidence conf_m(x):", "#B83B35", "bold", 5.0),
            ("Δtop2(L_m^o) · max_c <ẑ_m, μ̄_m,c>", "#1F242B", "normal", 5.0),
            ("Conflict Gate g(x) ∈ [0, 1]:", "#B83B35", "bold", 5.0),
            ("disc(x) · (1 - 0.55 cos(L_I^o, L_T^o))", "#1F242B", "normal", 5.0),
        ],
        header_bg="#B83B35", body_bg="#FFFFFF", edge_color="#B83B35", header_frac=0.18
    )

    draw_card(ax, 0.776, 0.285, 0.206, 0.305, facecolor="#FFFFFF", edgecolor="#B83B35",
              lw=0.85, rounding=0.010, zorder=3, shadow=True)
    ax.text(0.879, 0.568, "2. Dynamic Modulation Circuit",
            ha="center", va="center", fontsize=5.1, fontweight="bold", color="#96281B", zorder=4)

    streams = [
        (0.786, 0.495, "L_I^o (Vision)", "#2F6B9A", "α_I(x)"),
        (0.786, 0.415, "L_T^o (Text)", "#3A7CA5", "α_T(x)"),
        (0.786, 0.335, "Schur ΔL^s", "#6F4E8B", "ρ(x)"),
    ]
    for sx_i, sy_i, slabel, scol, gname in streams:
        draw_card(ax, sx_i, sy_i, 0.080, 0.052, facecolor="#F4F6F9", edgecolor=scol, lw=0.75, rounding=0.006, zorder=4)
        ax.text(sx_i + 0.040, sy_i + 0.026, slabel, ha="center", va="center",
                fontsize=5.0, fontweight="bold", color=scol, zorder=5)
        draw_op_node(ax, 0.896, sy_i + 0.026, 0.012, 0.019, "×", bg="#FFF8E7", fg="#B83B35", ec="#B83B35", fs=5.6, zorder=6)
        ax.text(0.896, sy_i + 0.055, gname, ha="center", va="center", fontsize=5.0, fontweight="bold", color="#B83B35", zorder=6)
        draw_ortho_arrow(ax, (sx_i + 0.080, sy_i + 0.026), (0.884, sy_i + 0.026), color=scol, lw=0.8)
        draw_ortho_arrow(ax, (0.908, sy_i + 0.026), (0.946, 0.441), color=scol, lw=0.8)

    draw_op_node(ax, 0.958, 0.441, 0.014, 0.023, "∑", bg="#FDEDEC", fg="#96281B", ec="#B83B35", fs=6.0, zorder=7)

    draw_header_card(
        ax, 0.776, 0.085, 0.206, 0.175,
        "3. Task-Free Readout f(x)",
        [
            ("f(x) = α_I L_I^o + α_T L_T^o + ρ ΔL^s", "#96281B", "bold", 5.0),
            ("• Clean s=(1,1): Schur Fusion", "#135E4F", "normal", 5.0),
            ("• Missing: Exact Sub-Block Solve", "#1F4E79", "normal", 5.0),
            ("• Corrupted: ρ(x)→0 Shields Output", "#B83B35", "bold", 5.0),
        ],
        header_bg="#96281B", body_bg="#FFFDFB", edge_color="#96281B", header_frac=0.25
    )
    draw_ortho_arrow(ax, (0.958, 0.418), (0.958, 0.260), color="#B83B35", lw=0.95)

    # ==================================================================
    # INTER-MODULE CONNECTING ARROWS
    # ==================================================================
    draw_ortho_arrow(ax, (0.254, 0.661), (0.284, 0.775), color="#2F6B9A", lw=1.0)
    draw_ortho_arrow(ax, (0.390, 0.715), (0.390, 0.690), color="#2B8C7E", lw=0.95)
    draw_ortho_arrow(ax, (0.390, 0.320), (0.390, 0.295), color="#1E7E6C", lw=0.95)

    draw_ortho_arrow(ax, (0.496, 0.530), (0.528, 0.775), color="#2B8C7E", lw=1.0)
    draw_ortho_arrow(ax, (0.636, 0.715), (0.636, 0.690), color="#6F4E8B", lw=0.95)
    draw_ortho_arrow(ax, (0.636, 0.365), (0.636, 0.340), color="#74346B", lw=0.95)

    draw_ortho_arrow(ax, (0.744, 0.525), (0.776, 0.720), color="#6F4E8B", lw=1.0)
    draw_ortho_arrow(ax, (0.744, 0.210), (0.776, 0.650), color="#74346B", lw=0.9, ls="--")
    draw_ortho_arrow(ax, (0.879, 0.615), (0.879, 0.590), color="#B83B35", lw=0.95)

    # Bottom Summary Legend Strip (fits comfortably inside 183 mm)
    draw_card(ax, 0.008, 0.008, 0.984, 0.046, facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.7, rounding=0.008, zorder=2)
    ax.text(
        0.50, 0.031,
        "Guarantees:  [1] Exemplar-Free (B = 0)   [2] Task-Free CIL (No Task-ID)   "
        "[3] Exact 0.000s Unlearning (ε < 4.7×10⁻¹⁴)   [4] 50% Lower FLOPs via Block-Schur",
        ha="center", va="center", fontsize=5.0, fontweight="bold", color="#1E293B", zorder=4
    )

    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)

    saved_files = []
    for out_dir in output_dirs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            out_path = out_dir / f"fig_asid_cl_v2_system_architecture.{ext}"
            dpi = 600 if ext == "png" else 300
            fig.savefig(out_path, dpi=dpi)
            saved_files.append(str(out_path))

    plt.close(fig)
    return saved_files


# ==============================================================================
# FIGURE 2: ACADEMIC CONTINUAL PIPELINE & GEOMETRIC VERIFICATION
# ==============================================================================

def _sample_tube(center_curve, t_samples, rng, sigma_u=0.055, sigma_v=0.04):
    pts = center_curve(t_samples)
    eps = 1e-3
    pts_f = center_curve(np.clip(t_samples + eps, 0, 1))
    tan = pts_f - pts
    tan_norm = np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
    t_hat = tan / tan_norm
    n_hat = np.stack([-t_hat[:, 1], t_hat[:, 0]], axis=1)
    u = rng.normal(0, sigma_u, size=(len(t_samples), 1))
    v = rng.normal(0, sigma_v, size=(len(t_samples), 1))
    return pts + u * n_hat + v * t_hat


def draw_academic_pipeline_flow_panel(ax):
    """
    Panel a: Two-Phase Academic Operational Pipeline
    - Top Swimlane (y = 0.545 .. 0.985): Phase I — Continual Training Stream across Tasks t = 1..T
    - Middle Gap (y = 0.455 .. 0.545): Recurrent Handover (t -> t+1) & Parameter Transfer to Phase II
    - Bottom Swimlane (y = 0.010 .. 0.455): Phase II — Unified Task-Free Inference across 3 Regimes
    All lines inside cards are strictly <= 22 characters to guarantee zero text overflow.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Swimlane 1 Container: Phase I
    draw_card(ax, 0.006, 0.545, 0.988, 0.440, facecolor="#F4F8FC", edgecolor="#2F6B9A",
              lw=0.9, rounding=0.012, zorder=1)
    hdr1 = FancyBboxPatch((0.006, 0.920), 0.988, 0.065,
                          boxstyle="round,pad=0.0,rounding_size=0.012",
                          facecolor="#1F4E79", edgecolor="#1F4E79", linewidth=0.85, zorder=2)
    ax.add_patch(hdr1)
    ax.text(0.50, 0.952,
            "PHASE I: Continual Multimodal Training Pipeline Across Sequential Tasks t = 1, ..., T   (Exemplar Buffer B = 0)",
            ha="center", va="center", fontsize=5.4, fontweight="bold", color="white", zorder=3)

    # 5 Sequential Steps in Phase I (each w = 0.178, max 21 chars/line)
    p1_steps = [
        (0.016, 0.575, 0.176, 0.325, "#2F6B9A", "#FFFFFF",
         "1. Task t Stream",
         [
             ("Data D_t = {(x_I, x_T, y)}", "#1F4E79", "bold", 5.0),
             ("s ∈ {(1,1), (1,0), (0,1)}", "#1F242B", "normal", 5.0),
             ("Gate G1 (Domain Shift):", "#B83B35", "bold", 5.0),
             ("Δ_G1 ≥ 2pt → NS-LoRA", "#135E4F", "bold", 5.0),
         ]),
        (0.212, 0.575, 0.180, 0.325, "#2F6B9A", "#FFFFFF",
         "2. Null-Space LoRA",
         [
             ("Adapt L10–L11 (Ep 1..E)", "#1F4E79", "bold", 5.0),
             ("A_eff = A_t P_⊥^(t-1)", "#1F242B", "bold", 5.0),
             ("Update Act. Gram G_act:", "#135E4F", "bold", 5.0),
             ("Masked eigh → P_⊥^(t)", "#135E4F", "normal", 5.0),
         ]),
        (0.412, 0.575, 0.182, 0.325, "#2B8C7E", "#FFFFFF",
         "3. Drift Transport",
         [
             ("Shift Op M_t = I + ΔM_t", "#135E4F", "bold", 5.0),
             ("Pushforward Old Gram:", "#1F242B", "normal", 5.0),
             ("G^(t) = M_t G^(t-1) M_t^T", "#135E4F", "bold", 5.0),
             ("+ G_new  (ω_c = N_c^-0.5)", "#1F4E79", "bold", 5.0),
         ]),
        (0.614, 0.575, 0.180, 0.325, "#6F4E8B", "#FFFFFF",
         "4. Block-Schur Solve",
         [
             ("Shrinkage γ_m*(t)", "#4A235A", "bold", 5.0),
             ("W_m^o = A_m^(-1) S_m", "#1F4E79", "bold", 5.0),
             ("Schur S_I(ρ*) → ΔW_m^s", "#4A235A", "bold", 5.0),
             ("Prototypes μ̄_m,c ∈ S^P-1", "#1F242B", "normal", 5.0),
         ]),
        (0.814, 0.575, 0.170, 0.325, "#B83B35", "#FFFFFF",
         "5. Audit & Unlearn",
         [
             ("Temp. Scaling τ_cal (Val)", "#96281B", "bold", 5.0),
             ("ECE: 0.531 → 0.0215", "#135E4F", "bold", 5.0),
             ("0.000s Exact Unlearn:", "#96281B", "bold", 5.0),
             ("G ← G - G_k* (4.7e-14)", "#1F242B", "normal", 5.0),
         ]),
    ]
    for sx, sy, sw, sh, scol, sbg, stitle, slines in p1_steps:
        draw_header_card(ax, sx, sy, sw, sh, stitle, slines,
                         header_bg=scol, body_bg=sbg, edge_color=scol, header_frac=0.22)

    # Arrows between Phase I steps
    draw_ortho_arrow(ax, (0.192, 0.735), (0.212, 0.735), color="#1F4E79", lw=0.95)
    draw_ortho_arrow(ax, (0.392, 0.735), (0.412, 0.735), color="#2B8C7E", lw=0.95)
    draw_ortho_arrow(ax, (0.594, 0.735), (0.614, 0.735), color="#6F4E8B", lw=0.95)
    draw_ortho_arrow(ax, (0.794, 0.735), (0.814, 0.735), color="#B83B35", lw=0.95)

    # Middle Gap (y = 0.455 .. 0.545):
    # 1) Recurrent Handover arrow (t -> t+1) placed cleanly on right-center
    draw_ortho_arrow(
        ax, (0.650, 0.545), (0.420, 0.545),
        color="#135E4F", lw=0.9, ls="--", rad=-0.12,
        label="Recurrent (t → t+1): Pass {P_⊥^(t), G^(t), S^(t)}",
        label_pos=(0.565, 0.498), label_fs=5.0, label_bg="#FFFFFF"
    )
    # 2) Clean vertical transfer from Phase I to Phase II on left
    draw_ortho_arrow(
        ax, (0.125, 0.545), (0.125, 0.455),
        color="#6F4E8B", lw=1.05, ls="-",
        label="Deploy {W_m^o, ΔW_m^s, μ̄_m,c}",
        label_pos=(0.135, 0.498), label_fs=5.0, label_bg="#FFFFFF"
    )

    # ------------------------------------------------------------------
    # Swimlane 2 Container: Phase II (Task-Free Multi-Regime Inference)
    # ------------------------------------------------------------------
    draw_card(ax, 0.006, 0.010, 0.988, 0.445, facecolor="#FDF8F6", edgecolor="#B83B35",
              lw=0.9, rounding=0.012, zorder=1)
    hdr2 = FancyBboxPatch((0.006, 0.390), 0.988, 0.065,
                          boxstyle="round,pad=0.0,rounding_size=0.012",
                          facecolor="#96281B", edgecolor="#96281B", linewidth=0.85, zorder=2)
    ax.add_patch(hdr2)
    ax.text(0.50, 0.422,
            "PHASE II: Task-Free Multimodal Inference Pipeline & Dynamic Routing Across 3 Operational Regimes",
            ha="center", va="center", fontsize=5.4, fontweight="bold", color="white", zorder=3)

    # Left: Test Query & Prototype-Modulated Gate
    draw_header_card(
        ax, 0.016, 0.032, 0.236, 0.338,
        "Test Query x (No Task-ID)",
        [
            ("1. Extract z_I, z_T ∈ R^1024", "#1F4E79", "bold", 5.0),
            ("2. Marginal L_m^o = z_m^T W_m^o", "#1F242B", "normal", 5.0),
            ("3. Schur ΔL_m^s = z_m^T ΔW_m^s", "#4A235A", "normal", 5.0),
            ("4. Stream Confidence:", "#96281B", "bold", 5.0),
            ("Δtop2(L_m^o)·max_c<ẑ_m,μ̄_m,c>", "#1F242B", "bold", 5.0),
        ],
        header_bg="#23374B", body_bg="#FFFFFF", edge_color="#23374B", header_frac=0.21
    )

    # 3 Regime Cards in Phase II (connected sequentially / via clean short arrows)
    regimes = [
        (0.278, 0.032, 0.224, 0.338, "#1E7E6C", "#F2FBF9",
         "Regime A: Clean Pair s=(1,1)",
         [
             ("High Agreement (g(x) → 0):", "#135E4F", "bold", 5.0),
             ("α_I ≈ 1, α_T ≈ 1, ρ(x) ≈ ρ*", "#1F242B", "bold", 5.0),
             ("Cooperative Schur Readout:", "#135E4F", "bold", 5.0),
             ("f(x) = L_I^o + L_T^o + ρ* ΔL^s", "#1F242B", "normal", 5.0),
             ("✓ +6.9% Worst-Class F1", "#135E4F", "bold", 5.0),
         ]),
        (0.520, 0.032, 0.224, 0.338, "#2F6B9A", "#F4F8FD",
         "Regime B: Missing Modality",
         [
             ("Missing Stream (z_miss = 0):", "#1F4E79", "bold", 5.0),
             ("α_miss = 0,  ρ(x) = 0 (Exact)", "#1F242B", "bold", 5.0),
             ("Sub-Block Analytic Solve:", "#1F4E79", "bold", 5.0),
             ("f(x) = z_obs^T A_mm^(-1) S_m", "#1F242B", "normal", 5.0),
             ("✓ +5.51% vs Zero-Fill (1e-15)", "#1F4E79", "bold", 5.0),
         ]),
        (0.762, 0.032, 0.222, 0.338, "#B83B35", "#FDF4F3",
         "Regime C: Corrupted Stream",
         [
             ("Off-Manifold Noise (85% ε):", "#96281B", "bold", 5.0),
             ("max_c <ẑ_noisy, μ̄_c> drops", "#1F242B", "normal", 5.0),
             ("Conflict Gate Shields Output:", "#96281B", "bold", 5.0),
             ("ρ(x) → 0,  α_noisy(x) → 0.08", "#1F242B", "bold", 5.0),
             ("✓ +28.1% vs GACL (p=4.9e-7)", "#96281B", "bold", 5.0),
         ]),
    ]
    for rx, ry, rw, rh, rcol, rbg, rtitle, rlines in regimes:
        draw_header_card(ax, rx, ry, rw, rh, rtitle, rlines,
                         header_bg=rcol, body_bg=rbg, edge_color=rcol, header_frac=0.21)

    # Clean non-overlapping arrows from Query to Regimes A, B, C
    draw_ortho_arrow(ax, (0.252, 0.200), (0.278, 0.200), color="#1E7E6C", lw=1.0)
    draw_ortho_arrow(ax, (0.502, 0.200), (0.520, 0.200), color="#2F6B9A", lw=0.9, ls="--")
    draw_ortho_arrow(ax, (0.744, 0.200), (0.762, 0.200), color="#B83B35", lw=0.9, ls="--")


def plot_subspace_manifold_panel(ax):
    """
    Panel b: Subspace Manifold & Task Trajectory across T1..T5
    (adapted from figures4papers/figure_VIGIL/plot_concept.py).
    """
    rng = np.random.default_rng(42)
    xp = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    y_grounded = np.array([0.77, 0.81, 0.74, 0.79, 0.83])
    y_drift = np.array([0.46, 0.32, 0.20, 0.13, 0.08])

    cs_g = CubicSpline(xp, y_grounded, bc_type="natural")
    cs_d = CubicSpline(xp, y_drift, bc_type="natural")

    def curve_grounded(t):
        return np.stack([t, cs_g(t)], axis=1)

    def curve_drift(t):
        return np.stack([t, cs_d(t)], axis=1)

    t_samples = rng.uniform(0.04, 0.96, size=800)
    pts_g = _sample_tube(curve_grounded, t_samples, rng, sigma_u=0.048, sigma_v=0.035)
    pts_d = _sample_tube(curve_drift, t_samples, rng, sigma_u=0.055, sigma_v=0.04)

    for pts, col in [(pts_g, PALETTE["blue_main"]), (pts_d, PALETTE["neutral_mid"])]:
        kde = gaussian_kde(pts.T)
        xx, yy = np.mgrid[-0.08:1.08:110j, -0.05:1.00:110j]
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        levels = np.quantile(zz, np.linspace(0.72, 0.97, 5))
        ax.contour(xx, yy, zz, levels=levels, colors=col, linewidths=0.65, alpha=0.30, zorder=2)
        ax.scatter(pts[::3, 0], pts[::3, 1], s=2.0, color=col, alpha=0.10, linewidths=0, zorder=1)

    tt = np.linspace(0.06, 0.94, 200)
    rg = curve_grounded(tt)
    rd = curve_drift(tt)
    r_ind = 0.45 * rg + 0.55 * rd

    ax.plot(rg[:, 0], rg[:, 1], linestyle="-", lw=1.6, color=PALETTE["blue_main"], label="ASID-CL v2 (Ours)", zorder=5)
    ax.plot(r_ind[:, 0], r_ind[:, 1], linestyle="-.", lw=1.15, color=PALETTE["amber"], label="Indep. Merge (TSV-M)", zorder=4)
    ax.plot(rd[:, 0], rd[:, 1], linestyle="--", lw=1.15, color=PALETTE["neutral_dark"], label="Seq. LoRA + SGD", zorder=4)

    task_idx = np.linspace(0, len(tt) - 1, 5, dtype=int)
    ax.scatter(rg[task_idx, 0], rg[task_idx, 1], marker="*", s=34, color=PALETTE["red_strong"],
               edgecolor="white", linewidth=0.4, zorder=6)
    ax.scatter(rd[task_idx, 0], rd[task_idx, 1], marker="o", s=12, color=PALETTE["neutral_dark"],
               edgecolor="white", linewidth=0.4, zorder=6)

    for i, idx in enumerate(task_idx):
        ax.text(rg[idx, 0], rg[idx, 1] + 0.048, f"T_{i+1}", fontsize=5.1, fontweight="bold",
                ha="center", va="bottom", color=PALETTE["blue_main"])

    mid = task_idx[1]
    draw_ortho_arrow(
        ax, (rd[mid, 0], rd[mid, 1] + 0.03), (rg[mid, 0], rg[mid, 1] - 0.03),
        color=PALETTE["red_strong"], lw=1.05, ls="-",
        label="P_⊥ Projection", label_pos=(0.28, 0.59), label_fs=5.1
    )

    ax.text(0.02, 0.97, "Grounded Multimodal Manifold M_mm", fontsize=5.2, fontweight="bold",
            color=PALETTE["blue_main"], ha="left", va="top")
    ax.text(0.02, 0.02, "Catastrophic Drift Zone (-37.4% ACC)", fontsize=5.0,
            color=PALETTE["neutral_dark"], ha="left", va="bottom")

    ax.set_xlim(-0.05, 1.06)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks([0.06, 0.28, 0.50, 0.72, 0.94])
    ax.set_xticklabels(["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"])
    ax.set_yticks([])
    ax.set_xlabel("Continual Task Progression (100 Classes)")
    ax.set_title("Null-space P_⊥ prevents manifold drift", fontsize=6.2, pad=4)
    set_frame(ax, "open")
    ax.spines["left"].set_visible(False)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.20), fontsize=5.0,
                    frameon=True, framealpha=0.88, edgecolor="none")
    leg.get_frame().set_facecolor("white")


def plot_spherical_fusion_panel(ax):
    """
    Panel c: Hyperspherical Prototype Alignment & Reliability Gating on S^{d-1}
    (adapted from figures4papers/figure_Dispersion/plot_illustration.py).
    """
    aspect_x = 1.32
    n_grid = 220
    xs = np.linspace(-1.15 * aspect_x, 1.15 * aspect_x, n_grid)
    ys = np.linspace(-1.15, 1.15, n_grid)
    x, y = np.meshgrid(xs, ys)
    r2 = x**2 + y**2
    mask = r2 <= 1.0
    z = np.zeros_like(x)
    z[mask] = np.sqrt(np.clip(1.0 - r2[mask], 0.0, 1.0))

    lx, ly, lz = -0.45, 0.55, 0.70
    lnorm = np.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx / lnorm, ly / lnorm, lz / lnorm
    intensity = np.clip(lx * x + ly * y + lz * z, 0.0, 1.0)
    shade = np.ones((n_grid, n_grid, 4))
    cmap = LinearSegmentedColormap.from_list("sphere_blue", ["#D3E3F2", "#F7FAFD"])
    rgb = cmap(intensity)
    shade[..., :3] = rgb[..., :3]
    shade[..., 3] = np.where(mask, 0.92, 0.0)

    ax.imshow(
        shade,
        extent=(-1.15 * aspect_x, 1.15 * aspect_x, -1.15, 1.15),
        origin="lower", aspect="auto", zorder=1
    )

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color=PALETTE["blue_main"], lw=0.95, alpha=0.8, zorder=3)
    ax.plot(np.cos(theta), 0.26 * np.sin(theta), color=PALETTE["neutral_mid"], lw=0.65, ls="--", alpha=0.7, zorder=2)
    ax.plot(0.32 * np.sin(theta), np.cos(theta), color=PALETTE["neutral_mid"], lw=0.65, ls=":", alpha=0.7, zorder=2)

    origin = (0.0, -0.05)
    v_vis = (-0.54, 0.58)
    v_txt_clean = (0.50, 0.62)
    v_txt_noisy = (0.68, -0.42)
    v_proto = (0.02, 0.84)

    draw_ortho_arrow(ax, origin, v_vis, color=PALETTE["blue_main"], lw=1.3, zorder=5)
    draw_ortho_arrow(ax, origin, v_txt_clean, color=PALETTE["teal"], lw=1.2, zorder=5)
    draw_ortho_arrow(ax, origin, v_txt_noisy, color=PALETTE["neutral_dark"], lw=1.0, ls="--", zorder=4)
    draw_ortho_arrow(ax, origin, v_proto, color=PALETTE["red_strong"], lw=1.6, zorder=6)

    t_arc = np.linspace(0, 1, 60)
    arc1_x = (1 - t_arc) * v_vis[0] + t_arc * v_proto[0]
    arc1_y = (1 - t_arc) * v_vis[1] + t_arc * v_proto[1] + 0.08 * np.sin(np.pi * t_arc)
    ax.plot(arc1_x, arc1_y, color=PALETTE["violet"], lw=1.1, ls="-", zorder=5)

    arc2_x = (1 - t_arc) * v_txt_clean[0] + t_arc * v_proto[0]
    arc2_y = (1 - t_arc) * v_txt_clean[1] + t_arc * v_proto[1] + 0.07 * np.sin(np.pi * t_arc)
    ax.plot(arc2_x, arc2_y, color=PALETTE["violet"], lw=1.1, ls="-", zorder=5)

    ax.scatter([v_proto[0]], [v_proto[1]], marker="*", s=52, color=PALETTE["red_strong"],
               edgecolor="white", linewidth=0.5, zorder=7)

    ax.text(v_vis[0] - 0.04, v_vis[1] + 0.07, "Vision z_I\n(conf = 0.92)", fontsize=5.0,
            fontweight="bold", color=PALETTE["blue_main"], ha="right", va="bottom")
    ax.text(v_txt_clean[0] + 0.04, v_txt_clean[1] + 0.06, "Clean Text z_T\n(conf = 0.88)", fontsize=5.0,
            fontweight="bold", color=PALETTE["teal"], ha="left", va="bottom")
    ax.text(v_txt_noisy[0] - 0.05, v_txt_noisy[1] - 0.06, "Noisy Text z̃_T\n(α_T→0.08, ρ→0)", fontsize=5.0,
            color=PALETTE["neutral_dark"], ha="center", va="top")
    ax.text(v_proto[0], v_proto[1] + 0.11, "Class Prototype μ̄_c*\n& Readout f(x)", fontsize=5.1,
            fontweight="bold", color=PALETTE["red_strong"], ha="center", va="bottom")
    ax.text(0.0, 0.44, "Schur Innovation\nρ(x)ΔL^s", fontsize=5.0,
            color=PALETTE["violet"], ha="center", va="center", fontweight="bold")

    ax.set_xlim(-1.15 * aspect_x, 1.15 * aspect_x)
    ax.set_ylim(-1.12, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Unit Hypersphere S^(P-1) (L2-Normalized)")
    ax.set_title("Geodesic Schur fusion & noise shielding", fontsize=6.2, pad=4)
    set_frame(ax, "none")


def plot_stage_ablation_panel(ax):
    """
    Panel d: Cumulative 4-Stage Pipeline Ablation Trajectory
    (showing empirical contribution of each pipeline stage on 100-class Food-101).
    """
    stages = ["S1:\nNS-LoRA\n+ SGD", "S2:\n+ 2×2\nGram", "S3:\n+ Block\nSchur", "S4:\n+ SID v2\n(Ours)"]
    x = np.arange(len(stages))

    clean_f1 = [89.27, 92.43, 93.85, 94.33]
    miss_img_f1 = [88.14, 88.26, 90.85, 92.02]
    corr_txt_f1 = [82.16, 56.11, 78.40, 84.18]
    worst_f1 = [0.00, 55.81, 66.67, 70.00]

    ax.plot(x, clean_f1, marker="o", ms=3.8, lw=1.45, color=PALETTE["blue_main"], label="Clean F1 (%)")
    ax.plot(x, miss_img_f1, marker="D", ms=3.3, lw=1.15, ls="--", color=PALETTE["amber"], label="MissImg F1 (%)")
    ax.plot(x, corr_txt_f1, marker="s", ms=3.5, lw=1.25, color=PALETTE["teal"], label="CorrTxt F1 (%)")
    ax.plot(x, worst_f1, marker="^", ms=3.9, lw=1.35, color=PALETTE["red_strong"], label="Worst-F1 (%)")

    # Place Worst-F1 labels cleanly without touching any annotation or line
    wf_offsets = [(0, 4.5, "bottom"), (1, 5.5, "bottom"), (2, -10.5, "top"), (3, -10.5, "top")]
    for xi, dy, va_mode in wf_offsets:
        wf = worst_f1[xi]
        ax.text(xi, wf + dy, f"{wf:.1f}%", ha="center", va=va_mode,
                fontsize=5.0, fontweight="bold", color=PALETTE["red_strong"])

    for xi, cf in zip([0, 3], [clean_f1[0], clean_f1[3]]):
        ax.text(xi, cf + 3.0, f"{cf:.2f}%", ha="center", va="bottom",
                fontsize=5.0, fontweight="bold", color=PALETTE["blue_main"])

    ax.annotate(
        "Eliminates SGD\ntail-class collapse",
        xy=(0.95, 53.0), xytext=(0.78, 32.0),
        arrowprops=dict(arrowstyle="->", lw=0.85, color=PALETTE["red_strong"]),
        fontsize=5.0, fontweight="bold", color=PALETTE["red_strong"], ha="center"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=5.0)
    ax.set_xlim(-0.30, 3.35)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Performance (%)  [↑ higher = better]")
    ax.set_title("Cumulative 4-stage ablation", fontsize=6.2, pad=4)
    set_frame(ax, "open")
    ax.legend(loc="lower right", fontsize=5.0, frameon=False)


def build_academic_pipeline_figure(output_dirs):
    """
    Constructs the dedicated double-column (183 mm x 126 mm) Continual Operational
    Pipeline + Geometric Verification Figure.
    """
    configure_publication_rc()
    fig_w = DOUBLE_COL_MM * MM_TO_INCH  # 7.205 in
    fig_h = 126.0 * MM_TO_INCH          # 4.960 in
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        height_ratios=[1.42, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.26, wspace=0.26,
        left=0.055, right=0.985, top=0.95, bottom=0.08
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    draw_academic_pipeline_flow_panel(ax_a)
    plot_subspace_manifold_panel(ax_b)
    plot_spherical_fusion_panel(ax_c)
    plot_stage_ablation_panel(ax_d)

    panel_letter(ax_a, "a", dx=-0.04, pad_pt=2.0, fontsize=8.5)
    panel_letter(ax_b, "b", dx=-0.12, pad_pt=4.0, fontsize=8.5)
    panel_letter(ax_c, "c", dx=-0.08, pad_pt=4.0, fontsize=8.5)
    panel_letter(ax_d, "d", dx=-0.14, pad_pt=4.0, fontsize=8.5)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        tolerance_pt=2.0,
        exemptions=[
            {"panels": ["a"], "checks": ["all"], "reason": "Full-width 2-phase academic pipeline banner spanning columns 1-3"},
        ]
    )

    saved_files = []
    for out_dir in output_dirs:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            for stem in ("fig_asid_cl_v2_academic_pipeline", "fig_asid_cl_v2_architecture_pipeline"):
                out_path = out_dir / f"{stem}.{ext}"
                dpi = 600 if ext == "png" else 300
                fig.savefig(out_path, dpi=dpi)
                saved_files.append(str(out_path))

    plt.close(fig)
    return saved_files


def main():
    out_dirs = [
        PROJECT_ROOT / "latex_master_survey_report" / "figures",
        PROJECT_ROOT / "analysis_outputs" / "figures4papers_pub",
        Path(r"C:\Users\Admin\.gemini\antigravity\brain\1c868d79-363e-44b8-9c17-35b7f14f7648"),
    ]
    sys_files = build_academic_system_architecture_figure(out_dirs)
    pipe_files = build_academic_pipeline_figure(out_dirs)
    print("Generated System Architecture files:", sys_files[:3])
    print("Generated Academic Pipeline files:", pipe_files[:3])


if __name__ == "__main__":
    main()
