#!/usr/bin/env python3
"""
Generate publication-grade figures for the ASID-CL v2 research manuscript
combining ChenLiu-1996/figures4papers visual idioms (manifold tubes, spherical
geodesics, multi-metric bars, and normalized polar radar) with the
Nature-Paper-Skills (figure-planner, nature-figure, figure-style) print-scale
vector contract.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import gaussian_kde
from scipy.interpolate import CubicSpline

# Load figure-style kernel helpers
KERNEL_PATH = Path(".agent/skills/nature-paper-skills/resources/figure-style/kernel.py")
kernel_ns = {}
exec(KERNEL_PATH.read_text(encoding="utf-8"), kernel_ns)
apply_figure_style = kernel_ns["apply_figure_style"]
panel_letter = kernel_ns["panel_letter"]
set_frame = kernel_ns["set_frame"]

# Load nature-figure panel alignment gate
ALIGN_PATH = Path(".agent/skills/nature-paper-skills/resources/nature-figure/scripts/audit_panel_alignment.py")
align_ns = {}
exec(ALIGN_PATH.read_text(encoding="utf-8"), align_ns)
require_matplotlib_panel_alignment = align_ns["require_matplotlib_panel_alignment"]

MM = 1.0 / 25.4  # millimeters to inches

# figures4papers canonical palette (from scientific-figure-making/references/api.md)
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
    "neutral_black": "#272727",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "amber": "#D97706",
    "soft_blue_bg": "#EEF4FA",
    "soft_teal_bg": "#EDF7F7",
    "soft_violet_bg": "#F5EFF7",
    "soft_red_bg": "#FDF2F2",
    "soft_green_bg": "#EEF9F0",
    "soft_gray_bg": "#F6F7F9",
}


def setup_pub_rc():
    """Apply Nature / figures4papers print-scale rcParams (183 mm double-column)."""
    apply_figure_style(frame="open", font="DejaVu Sans", sizes=(6.6, 5.6, 5.4), grid=False)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.2,
        "axes.titlelocation": "left",
        "axes.titleweight": "bold",
        "savefig.dpi": 600,
    })


def draw_rounded_box(ax, x, y, w, h, title, lines, edge_color, fill_color,
                     title_size=5.8, body_size=5.2, lw=0.85, radius=0.015):
    """Draw a clean architectural module box with a header band and mathematical lines."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fill_color, edgecolor=edge_color, linewidth=lw, zorder=2
    )
    ax.add_patch(box)
    header_h = min(0.26 * h, 0.084)
    header = FancyBboxPatch(
        (x, y + h - header_h), w, header_h,
        boxstyle=f"round,pad=0.002,rounding_size={radius}",
        facecolor=edge_color, edgecolor=edge_color, linewidth=0.4, zorder=3
    )
    ax.add_patch(header)
    ax.text(
        x + w / 2.0, y + h - header_h / 2.0, title,
        ha="center", va="center", fontsize=title_size, fontweight="bold",
        color="white", zorder=4
    )
    body_h = h - header_h
    n_lines = len(lines)
    for idx, line in enumerate(lines):
        ly = y + body_h * (1.0 - (idx + 0.56) / max(n_lines + 0.12, 1))
        is_bold = line.startswith("*")
        clean_line = line.lstrip("*")
        ax.text(
            x + w / 2.0, ly, clean_line,
            ha="center", va="center",
            fontsize=body_size,
            fontweight="bold" if is_bold else "normal",
            color=PALETTE["neutral_black"],
            zorder=4
        )


def draw_arrow(ax, p0, p1, color="#272727", lw=1.0, style="-|>", rad=0.0, ls="-",
               label=None, label_pos=None, fontsize=5.2):
    """Draw a crisp connecting arrow between two coordinates in ax data space."""
    conn = f"arc3,rad={rad}"
    arr = FancyArrowPatch(
        p0, p1,
        connectionstyle=conn,
        arrowstyle=style,
        mutation_scale=8.0,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=5
    )
    ax.add_patch(arr)
    if label:
        mx, my = label_pos if label_pos is not None else (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]) + 0.025)
        ax.text(
            mx, my, label,
            ha="center", va="center", fontsize=fontsize, fontweight="bold",
            color=color, zorder=6
        )


def plot_hero_pipeline_schematic(ax):
    """
    Panel a: Full-width Hero Schematic of the ASID-CL v2 Model Architecture & 4-Stage Pipeline.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    stages = [
        (0.005, 0.172, "INPUT STREAM\n(Task t = 1..T)", PALETTE["neutral_dark"], PALETTE["soft_gray_bg"]),
        (0.192, 0.190, "STAGE 1: ASYMMETRIC\nNULL-SPACE LoRA", PALETTE["blue_main"], PALETTE["soft_blue_bg"]),
        (0.398, 0.190, "STAGE 2: SEQUENTIAL\nDUAL-SVD CONSOLIDATION", PALETTE["teal"], PALETTE["soft_teal_bg"]),
        (0.604, 0.190, "STAGE 3: SPHERICAL\nRELIABILITY FUSION", PALETTE["violet"], PALETTE["soft_violet_bg"]),
        (0.810, 0.185, "STAGE 4: ANALYTIC\nRIDGE READOUT", PALETTE["red_strong"], PALETTE["soft_red_bg"]),
    ]

    for sx, sw, stitle, scolor, sfill in stages:
        bg = FancyBboxPatch(
            (sx, 0.12), sw, 0.76,
            boxstyle="round,pad=0.004,rounding_size=0.015",
            facecolor=sfill, edgecolor=scolor, linewidth=0.75, linestyle="--", alpha=0.55, zorder=1
        )
        ax.add_patch(bg)
        ax.text(
            sx + sw / 2.0, 0.905, stitle,
            ha="center", va="bottom", fontsize=5.8, fontweight="bold", color=scolor
        )

    # --- COLUMN 0: Multimodal Task Stream & Frozen Backbone ---
    draw_rounded_box(
        ax, 0.014, 0.63, 0.154, 0.22,
        "Vision Stream (X_v)",
        ["*ViT-B/16 Encoder", "Missing ρ_v ∈ [0, 0.3]", "Feature f_v ∈ R^{768}"],
        edge_color=PALETTE["blue_secondary"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.014, 0.39, 0.154, 0.21,
        "Text Stream (X_t)",
        ["*BioClinicalBERT", "Noise ρ_t ∈ [0, 0.3]", "Feature f_t ∈ R^{768}"],
        edge_color=PALETTE["teal"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.014, 0.15, 0.154, 0.21,
        "Frozen Pretrained W_0",
        ["*Shared Base Weights", "Exemplar-free (M = 0)", "Task-Free CIL regime"],
        edge_color=PALETTE["neutral_dark"], fill_color="white"
    )

    # --- COLUMN 1: Stage 1 Asymmetric Null-Space LoRA ---
    draw_rounded_box(
        ax, 0.202, 0.54, 0.170, 0.31,
        "Null-Space Projector P_⊥",
        ["*Orthogonal to Span(S_{t-1})", "P_⊥ = I - U_{t-1} U_{t-1}^T", "ΔW_t = γ_m · (P_⊥ B_t) A_t", "⟨ΔW_t, S_{t-1}⟩_F = 0"],
        edge_color=PALETTE["blue_main"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.202, 0.15, 0.170, 0.31,
        "Asymmetric Scale γ_m",
        ["*Modality Fisher Ratio", "γ_v ≠ γ_t adaptive scale", "Prevents dominant-modality", "gradient starvation"],
        edge_color=PALETTE["blue_secondary"], fill_color="white"
    )

    # --- COLUMN 2: Stage 2 Sequential Dual-SVD Consolidation ---
    draw_rounded_box(
        ax, 0.408, 0.54, 0.170, 0.31,
        "Incremental SVD Basis",
        ["*Accumulated Task Span", "[U_{t-1} Σ_{t-1}, ΔW_t]", "→ SVD: U_t Σ_t V_t^T", "Rank-k Energy ≥ 99%"],
        edge_color=PALETTE["teal"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.408, 0.15, 0.170, 0.31,
        "Spectral Whitening",
        ["*Balanced Task Spectrum", "W_cons = W_0 + Σ_τ α_τ ΔW̃_τ", "Avoids W_0-independent", "task-vector collision"],
        edge_color="#2A6F77", fill_color="white"
    )

    # --- COLUMN 3: Stage 3 Spherical Reliability Fusion ---
    draw_rounded_box(
        ax, 0.614, 0.54, 0.170, 0.31,
        "Reliability Gate r_m(x)",
        ["*Entropy & Norm Gate", "r_m(x) = -H(p_m) · ‖z_m‖_2", "w_m(x) = softmax(r_m / τ)", "Downweights noisy modality"],
        edge_color=PALETTE["violet"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.614, 0.15, 0.170, 0.31,
        "Geodesic Fusion (S^{d-1})",
        ["*Unit Hypersphere Map", "z̃(x) = Normalize(Σ_m w_m ẑ_m)", "Class Prototype μ_c ∈ S^{d-1}", "Cosine score s_proto(x)"],
        edge_color="#74346B", fill_color="white"
    )

    # --- COLUMN 4: Stage 4 Analytic Ridge Readout ---
    draw_rounded_box(
        ax, 0.818, 0.54, 0.168, 0.31,
        "Closed-Form Ridge Head",
        ["*Cumulative Gram Stats", "G_t = G_{t-1} + Z_t^T Z_t", "R_t = R_{t-1} + Z_t^T Y_t", "*W_a = (G_t + λI)^{-1} R_t"],
        edge_color=PALETTE["red_strong"], fill_color="white"
    )
    draw_rounded_box(
        ax, 0.818, 0.15, 0.168, 0.31,
        "Dual-Head Calibration",
        ["*Logit Fusion (No Task-ID)", "ŷ = argmax [s_ridge + β s_proto]", "*Clean Macro-F1: 95.36%", "*Worst-Class F1: 72.73%"],
        edge_color="#2E7D32", fill_color=PALETTE["soft_green_bg"]
    )

    # Connecting arrows across the pipeline
    draw_arrow(ax, (0.168, 0.74), (0.202, 0.74), color=PALETTE["blue_secondary"], lw=1.1)
    draw_arrow(ax, (0.168, 0.49), (0.202, 0.36), color=PALETTE["teal"], lw=1.1)
    draw_arrow(ax, (0.168, 0.25), (0.202, 0.25), color=PALETTE["neutral_dark"], lw=0.9, ls="--")

    draw_arrow(ax, (0.287, 0.54), (0.287, 0.46), color=PALETTE["blue_main"], lw=1.0)
    draw_arrow(ax, (0.372, 0.69), (0.408, 0.69), color=PALETTE["blue_main"], lw=1.1,
               label="ΔW_t", label_pos=(0.390, 0.725), fontsize=5.0)
    draw_arrow(ax, (0.372, 0.30), (0.408, 0.30), color=PALETTE["blue_secondary"], lw=1.1)

    # Downward-curving feedback arc below the boxes (y=0.15 -> y=0.05 -> y=0.15)
    draw_arrow(
        ax, (0.493, 0.15), (0.287, 0.15),
        color=PALETTE["teal"], lw=1.0, rad=-0.18, ls="--",
        label="Feedback Update: U_t → P_⊥ = I - U_t U_t^T (for Task t+1)",
        label_pos=(0.390, 0.025), fontsize=5.2
    )

    draw_arrow(ax, (0.493, 0.54), (0.493, 0.46), color=PALETTE["teal"], lw=1.0)
    draw_arrow(ax, (0.578, 0.69), (0.614, 0.69), color=PALETTE["teal"], lw=1.1,
               label="z_v, z_t", label_pos=(0.596, 0.725), fontsize=5.0)
    draw_arrow(ax, (0.578, 0.30), (0.614, 0.30), color=PALETTE["teal"], lw=1.1)

    draw_arrow(ax, (0.699, 0.54), (0.699, 0.46), color=PALETTE["violet"], lw=1.0)
    draw_arrow(ax, (0.784, 0.69), (0.818, 0.69), color=PALETTE["violet"], lw=1.1,
               label="z̃(x)", label_pos=(0.801, 0.725), fontsize=5.0)
    draw_arrow(ax, (0.784, 0.30), (0.818, 0.30), color="#74346B", lw=1.1,
               label="s_proto", label_pos=(0.801, 0.35), fontsize=5.0)
    draw_arrow(ax, (0.902, 0.54), (0.902, 0.46), color=PALETTE["red_strong"], lw=1.1,
               label="s_ridge", label_pos=(0.942, 0.50), fontsize=5.1)


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

    ax.plot(rg[:, 0], rg[:, 1], linestyle="-", lw=1.7, color=PALETTE["blue_main"], label="ASID-CL v2 (Ours)", zorder=5)
    ax.plot(r_ind[:, 0], r_ind[:, 1], linestyle="-.", lw=1.2, color=PALETTE["amber"], label="Indep. Merge (TSV-M)", zorder=4)
    ax.plot(rd[:, 0], rd[:, 1], linestyle="--", lw=1.2, color=PALETTE["neutral_dark"], label="Seq. LoRA + SGD", zorder=4)

    task_idx = np.linspace(0, len(tt) - 1, 5, dtype=int)
    ax.scatter(rg[task_idx, 0], rg[task_idx, 1], marker="*", s=36, color=PALETTE["red_strong"],
               edgecolor="white", linewidth=0.4, zorder=6)
    ax.scatter(rd[task_idx, 0], rd[task_idx, 1], marker="o", s=13, color=PALETTE["neutral_dark"],
               edgecolor="white", linewidth=0.4, zorder=6)

    for i, idx in enumerate(task_idx):
        ax.text(rg[idx, 0], rg[idx, 1] + 0.048, f"T_{i+1}", fontsize=5.2, fontweight="bold",
                ha="center", va="bottom", color=PALETTE["blue_main"])

    mid = task_idx[1]
    draw_arrow(
        ax, (rd[mid, 0], rd[mid, 1] + 0.03), (rg[mid, 0], rg[mid, 1] - 0.03),
        color=PALETTE["red_strong"], lw=1.1, ls="-",
        label="P_⊥ Projection", label_pos=(0.28, 0.59), fontsize=5.2
    )

    ax.text(0.02, 0.97, "Grounded Multimodal Manifold M_mm", fontsize=5.3, fontweight="bold",
            color=PALETTE["blue_main"], ha="left", va="top")
    ax.text(0.02, 0.02, "Catastrophic Drift Zone", fontsize=5.1,
            color=PALETTE["neutral_dark"], ha="left", va="bottom")

    ax.set_xlim(-0.05, 1.06)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xticks([0.06, 0.28, 0.50, 0.72, 0.94])
    ax.set_xticklabels(["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"])
    ax.set_yticks([])
    ax.set_xlabel("Continual Task Progression (5 Tasks, 100 Classes)")
    ax.set_title("Null-space projection prevents drift", fontsize=6.6, pad=4)
    set_frame(ax, "open")
    ax.spines["left"].set_visible(False)
    leg = ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.22), fontsize=5.0, frameon=True, framealpha=0.88, edgecolor="none")
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

    light = np.array([-0.45, 0.55, 0.70])
    light /= np.linalg.norm(light)
    intensity = np.maximum(0.0, x * light[0] + y * light[1] + z * light[2])
    img = np.ones_like(x)
    img[mask] = np.clip(0.82 + 0.16 * intensity[mask], 0.0, 1.0)

    ax.imshow(img, cmap="Blues", origin="lower",
              extent=[-1.15 * aspect_x, 1.15 * aspect_x, -1.15, 1.15],
              vmin=0.65, vmax=1.05, alpha=0.35, aspect="auto", zorder=1)

    theta = np.linspace(0, 2 * np.pi, 240)
    ax.plot(np.cos(theta), np.sin(theta), color=PALETTE["blue_main"], lw=0.9, ls="-", zorder=2)
    ax.plot(np.cos(theta), 0.26 * np.sin(theta), color=PALETTE["blue_secondary"], lw=0.6, ls="--", alpha=0.6, zorder=2)

    p_proto = np.array([0.56, 0.65])
    p_vision = np.array([0.14, 0.78])
    p_text_noisy = np.array([-0.66, -0.36])
    p_naive = np.array([-0.28, 0.20])
    p_gated = np.array([0.42, 0.56])

    t_arc = np.linspace(0, 1, 60)[:, None]
    arc_bad = (1 - t_arc) * p_vision + t_arc * p_text_noisy
    arc_bad /= np.maximum(np.linalg.norm(arc_bad, axis=1, keepdims=True), 1.05) * 0.92
    ax.plot(arc_bad[:, 0], arc_bad[:, 1], color=PALETTE["neutral_mid"], lw=1.0, ls=":", zorder=3)

    ax.scatter(*p_proto, marker="*", s=68, color=PALETTE["red_strong"], edgecolor="black", lw=0.4, zorder=6, label="Class Prototype μ_c")
    ax.scatter(*p_vision, marker="o", s=24, color=PALETTE["blue_main"], edgecolor="white", lw=0.4, zorder=5, label="Clean Vision ẑ_v (w_v=0.86)")
    ax.scatter(*p_text_noisy, marker="X", s=26, color=PALETTE["neutral_dark"], edgecolor="white", lw=0.4, zorder=5, label="Noisy Text ẑ_t (w_t=0.14)")
    ax.scatter(*p_naive, marker="s", s=20, color=PALETTE["amber"], edgecolor="white", lw=0.4, zorder=5, label="Static Equal Fusion")
    ax.scatter(*p_gated, marker="D", s=24, color="#2E7D32", edgecolor="white", lw=0.4, zorder=6, label="ASID-CL v2 Gated z̃(x)")

    draw_arrow(ax, tuple(p_naive), tuple(p_gated), color="#2E7D32", lw=1.2, rad=-0.22)

    ax.text(p_proto[0] + 0.10, p_proto[1] + 0.08, "Prototype μ_c", fontsize=5.3, fontweight="bold", color=PALETTE["red_strong"], ha="left")
    ax.text(p_vision[0] - 0.08, p_vision[1] + 0.06, "Clean ẑ_v", fontsize=5.2, fontweight="bold", color=PALETTE["blue_main"], ha="right")
    ax.text(p_text_noisy[0] + 0.08, p_text_noisy[1] - 0.14, "Corrupted ẑ_t\n(30% noise)", fontsize=5.0, color=PALETTE["neutral_dark"], ha="left")
    ax.text(p_naive[0] - 0.08, p_naive[1] + 0.06, "Naive\nFusion", fontsize=5.0, color=PALETTE["amber"], ha="right")
    ax.text(p_gated[0] + 0.10, p_gated[1] - 0.22, "Gated z̃(x)\n(+19.0% F1)", fontsize=5.2, fontweight="bold", color="#2E7D32", ha="left")

    ax.set_xlim(-1.15 * aspect_x, 1.15 * aspect_x)
    ax.set_ylim(-1.22, 1.15)
    ax.axis("off")
    ax.set_title("Spherical gating on S^{d-1}", fontsize=6.6, pad=4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=5.0, frameon=False, columnspacing=0.6)


def plot_stage_ablation_panel(ax):
    """
    Panel d: Cumulative Stage-by-Stage Ablation Progression (Stage 1 -> Stage 4).
    """
    stages = ["S1: Null\nLoRA", "+S2: Dual\nSVD", "+S3: Spher.\nProto.", "+S4: Ridge\n(Ours)"]
    x = np.arange(len(stages))

    clean_f1 = np.array([89.57, 90.48, 91.86, 95.36])
    corr_txt_f1 = np.array([73.21, 79.40, 87.52, 92.22])
    worst_f1 = np.array([0.00, 0.00, 66.67, 72.73])

    ax.plot(x, clean_f1, marker="o", ms=4.0, lw=1.4, color=PALETTE["blue_main"], label="Clean Macro-F1 (%)")
    ax.plot(x, corr_txt_f1, marker="s", ms=3.8, lw=1.3, color=PALETTE["teal"], label="Corrupted-Text F1 (%)")
    ax.plot(x, worst_f1, marker="^", ms=4.2, lw=1.4, color=PALETTE["red_strong"], label="Worst-Class F1 (%)")

    for xi, wf in zip(x, worst_f1):
        offset_y = 4.0 if wf < 50 else -11.0
        ax.text(xi, wf + offset_y, f"{wf:.1f}%", ha="center", va="bottom" if offset_y > 0 else "top",
                fontsize=5.2, fontweight="bold", color=PALETTE["red_strong"])

    for xi, cf in zip([0, 3], [clean_f1[0], clean_f1[3]]):
        ax.text(xi, cf + 3.0, f"{cf:.2f}%", ha="center", va="bottom",
                fontsize=5.2, fontweight="bold", color=PALETTE["blue_main"])

    ax.annotate(
        "Eliminates SGD\ntail-class collapse",
        xy=(1.95, 65.5), xytext=(0.78, 42.0),
        arrowprops=dict(arrowstyle="->", lw=0.85, color=PALETTE["red_strong"]),
        fontsize=5.1, fontweight="bold", color=PALETTE["red_strong"], ha="center"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=5.1)
    ax.set_xlim(-0.30, 3.35)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Performance (%)  [↑ higher = better]")
    ax.set_title("Cumulative 4-stage ablation", fontsize=6.6, pad=4)
    set_frame(ax, "open")
    ax.legend(loc="lower right", fontsize=5.0, frameon=False)


def build_figure_1_architecture_pipeline(out_dirs: list[Path]):
    """
    Build Figure 1 (183 mm x 118 mm): ASID-CL v2 Model Architecture & 4-Stage Pipeline Composite.
    """
    setup_pub_rc()
    fig = plt.figure(figsize=(183 * MM, 118 * MM))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.35, 1.0],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.28, wspace=0.28,
        left=0.055, right=0.985, top=0.95, bottom=0.09
    )

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    plot_hero_pipeline_schematic(ax_a)
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
        exemptions=[{"panels": ["a"], "checks": ["all"], "reason": "full-width top hero schematic panel"}],
    )
    saved_paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            p = d / f"fig_asid_cl_v2_architecture_pipeline.{ext}"
            fig.savefig(p, dpi=600)
            saved_paths.append(p)
    plt.close(fig)
    return saved_paths


def plot_sota_grouped_bars(ax):
    """
    Figure 2 Panel a: Paper-Faithful 2024-2026 SOTA Comparison (Clean Macro-F1, Corrupted-Txt F1, Worst-Class F1).
    """
    methods = ["InfLoRA\nCVPR'24", "GACL\nNeurIPS", "SD-LoRA\nICLR'25", "DeLo\nAAAI'26", "TSV-M\nCVPR'25", "KnOTS\nICLR'25", "ASID-CL\n(Ours)"]
    x = np.arange(len(methods))
    w = 0.24

    nme_clean = np.array([91.86, 92.43, 91.71, 91.79, 91.59, 91.43, 94.70])
    corr_txt = np.array([84.06, 56.11, 87.40, 85.56, 87.49, 87.52, 85.70])
    worst_f1_nme = np.array([66.67, 55.81, 33.33, 42.11, 28.57, 34.78, 70.00])

    b1 = ax.bar(x - w, nme_clean, width=w, color=PALETTE["blue_secondary"], edgecolor="black", lw=0.45,
                label="Best Clean Macro-F1 (%)")
    b2 = ax.bar(x, corr_txt, width=w, color=PALETTE["neutral_light"], edgecolor="black", lw=0.45,
                label="50% Corrupted-Txt F1 (%)")
    b3 = ax.bar(x + w, worst_f1_nme, width=w, color=PALETTE["teal"], edgecolor="black", lw=0.45,
                label="Best Worst-Class F1 (%)")

    b1[-1].set_facecolor(PALETTE["blue_main"])
    b2[-1].set_facecolor(PALETTE["blue_main"])
    b2[1].set_facecolor(PALETTE["red_2"])
    b3[-1].set_facecolor(PALETTE["red_strong"])

    for i in [0, 2, 3, 4, 5]:
        ax.scatter(x[i] - w, 2.5, marker="v", s=11, color=PALETTE["red_strong"], zorder=5)
    ax.text(2.85, 8.2, "SGD Linear Worst-F1 = 0.0%",
            fontsize=4.5, fontweight="bold", color=PALETTE["red_strong"], ha="center")

    ax.text(x[-1] - w - 0.08, 96.4, "94.7%", fontsize=4.6, fontweight="bold", color=PALETTE["blue_main"], ha="center")
    ax.text(x[-1] + 0.26, 87.4, "85.7%", fontsize=4.4, fontweight="bold", color=PALETTE["blue_main"], ha="center")
    ax.text(x[-1] + w + 0.20, 72.0, "70.0%", fontsize=4.6, fontweight="bold", color=PALETTE["red_strong"], ha="center")
    ax.text(x[1] + 0.24, 58.4, "56.1%", fontsize=4.4, fontweight="bold", color=PALETTE["red_strong"], ha="center")

    ax.axhline(94.70, color=PALETTE["blue_main"], ls="--", lw=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=4.0)
    ax.set_ylim(0, 117)
    ax.set_ylabel("Macro-F1 / Worst-Class F1 (%)  [↑]")
    ax.set_title("2024–2026 SOTA benchmark", fontsize=6.5, pad=14)
    set_frame(ax, "open")
    ax.legend(loc="upper left", fontsize=4.7, frameon=False, ncol=1)


def plot_delo_protocol_gap_panel(ax):
    """
    Figure 2 Panel b: DeLo (AAAI 2026) Task-Free CIL vs Oracle Task-ID TIL vs ASID-CL v2 (Task-Free CIL).
    """
    regimes = ["DeLo Linear\n(Task-Free CIL)", "DeLo NME\n(Task-Free CIL)", "ASID-CL v2\n(Task-Free CIL)", "DeLo Oracle*\n(Task-ID TIL)"]
    y = np.arange(len(regimes))[::-1]

    clean_f1 = np.array([85.52, 91.79, 94.70, 97.61])
    worst_f1 = np.array([0.00, 42.11, 70.00, 75.00])

    h = 0.33
    bars1 = ax.barh(y + h/2, clean_f1, height=h, color=[PALETTE["neutral_mid"], PALETTE["blue_secondary"], PALETTE["blue_main"], "#A8A8A8"],
                    edgecolor="black", lw=0.45, label="Clean Macro-F1 (%)")
    bars2 = ax.barh(y - h/2, worst_f1, height=h, color=[PALETTE["red_2"], PALETTE["teal"], PALETTE["red_strong"], "#D0D0D0"],
                    edgecolor="black", lw=0.45, label="Worst-Class F1 (%)")

    bars1[-1].set_hatch("//")
    bars2[-1].set_hatch("//")

    for yi, cf, wf in zip(y, clean_f1, worst_f1):
        ax.text(cf + 1.5, yi + h/2, f"{cf:.1f}%", va="center", ha="left", fontsize=5.2, fontweight="bold")
        ax.text(wf + 1.5, yi - h/2, f"{wf:.1f}%", va="center", ha="left", fontsize=5.1, color=PALETTE["red_strong"])

    ax.set_yticks(y)
    ax.set_yticklabels(regimes, fontsize=5.1)
    ax.set_xlim(0, 118)
    ax.set_ylim(-0.65, 3.65)
    ax.set_xlabel("Score (%)  [*hatched = requires Task-ID oracle]")
    ax.set_title("Task-Free CIL vs. Oracle Task-ID gap", fontsize=6.6, pad=14)
    set_frame(ax, "open")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=5.0, frameon=False, columnspacing=0.8)


def plot_sota_radar_panel(ax):
    """
    Figure 2 Panel c: 5-Axis Normalized Polar Radar Comparison across Clean & Stress Regimes
    (adapted from figures4papers/figure_VIGIL/plot_comparison_radar.py).
    """
    metrics = [
        "Clean ACC [85–95]",
        "Clean F1\n[85–95]",
        "Worst-Class F1\n[0–72]",
        "Missing-Img F1\n[85–93]",
        "Corrupted-Txt F1\n[55–88]",
    ]
    ranges = [
        (85.0, 95.0),
        (85.0, 95.0),
        (0.0, 72.0),
        (85.0, 93.0),
        (55.0, 88.0),
    ]
    series = {
        "DeLo (CIL Linear)": ([86.20, 85.52, 0.00, 85.29, 73.21], PALETTE["neutral_mid"], "--"),
        "GACL (NeurIPS'24)": ([92.64, 92.43, 55.81, 88.26, 56.11], PALETTE["teal"], "-."),
        "InfLoRA (NME)": ([92.00, 91.86, 66.67, 88.65, 84.06], PALETTE["blue_secondary"], "-"),
        "ASID-CL v2 (Ours)": ([94.76, 94.70, 70.00, 92.19, 85.70], PALETTE["red_strong"], "-"),
    }

    n_axes = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    angles_closed = np.append(angles, angles[0])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(25, 100)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for r_level in [40, 60, 80, 100]:
        ax.plot(angles_closed, np.full_like(angles_closed, r_level), color="#D0D5DD", lw=0.55, zorder=1)
    for a in angles:
        ax.plot([a, a], [25, 100], color="#D0D5DD", lw=0.55, zorder=1)

    for label, (vals, col, ls) in series.items():
        norm_vals = []
        for v, (lo, hi) in zip(vals, ranges):
            frac = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
            norm_vals.append(30.0 + 68.0 * frac)
        norm_closed = np.append(norm_vals, norm_vals[0])
        lw = 1.6 if "Ours" in label else 1.05
        ax.plot(angles_closed, norm_closed, color=col, lw=lw, ls=ls, label=label, zorder=4 if "Ours" in label else 3)
        ax.fill(angles_closed, norm_closed, color=col, alpha=0.10 if "Ours" in label else 0.03, zorder=2)
        ax.scatter(angles, norm_vals, color=col, s=13 if "Ours" in label else 8, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels([])
    ax.set_yticks([])

    for idx, (a, m_label) in enumerate(zip(angles, metrics)):
        r_lbl = 110 if idx == 0 else (117 if idx in (1, 4) else 118)
        ax.text(a, r_lbl, m_label, fontsize=5.0, ha="center", va="center",
                color=PALETTE["neutral_black"], fontweight="bold")

    ax.set_title("5-axis robustness profile", fontsize=6.6, pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=2, fontsize=5.0, frameon=False, columnspacing=0.6)


def build_figure_2_sota_benchmark(out_dirs: list[Path]):
    """
    Build Figure 2 (183 mm x 82 mm): 2024-2026 Paper-Faithful SOTA Benchmark & Robustness Profile.
    """
    setup_pub_rc()
    fig = plt.figure(figsize=(183 * MM, 82 * MM))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.38,
        left=0.06, right=0.95, top=0.82, bottom=0.18
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")

    plot_sota_grouped_bars(ax_a)
    plot_delo_protocol_gap_panel(ax_b)
    plot_sota_radar_panel(ax_c)

    panel_letter(ax_a, "a", dx=-0.14, pad_pt=14.0, fontsize=8.5)
    panel_letter(ax_b, "b", dx=-0.22, pad_pt=14.0, fontsize=8.5)
    panel_letter(ax_c, "c", dx=-0.16, pad_pt=22.0, fontsize=8.5)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        tolerance_pt=2.0,
        exemptions=[{"panels": ["c"], "checks": ["all"], "reason": "polar radar subplot uses square aspect ratio"}],
    )
    saved_paths = []
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "svg", "png"):
            p = d / f"fig_sota_empirical_evidence_2024_2026.{ext}"
            fig.savefig(p, dpi=600)
            saved_paths.append(p)
    plt.close(fig)
    return saved_paths


def main():
    out_dirs = [
        Path("latex_master_survey_report/figures"),
        Path("analysis_outputs/figures4papers_pub"),
        Path(r"C:\Users\Admin\.gemini\antigravity\brain\1c868d79-363e-44b8-9c17-35b7f14f7648"),
    ]
    f1 = build_figure_1_architecture_pipeline(out_dirs)
    f2 = build_figure_2_sota_benchmark(out_dirs)
    print("Generated Figure 1 files:", [str(p) for p in f1[:3]])
    print("Generated Figure 2 files:", [str(p) for p in f2[:3]])


if __name__ == "__main__":
    main()
