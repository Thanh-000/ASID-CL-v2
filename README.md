# ASID-CL v2: Adaptive Spectral-Isotropic Decoupled Continual Learning

[![Open In Colab (Full End-to-End T4 Suite)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thanh-000/ASID-CL-v2/blob/main/notebooks/ASID_CL_v2_Full_EndToEnd_Colab_T4.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **ASID-CL v2 (`BlockAnalyticCL` & `SessionLoRA`)**: a unified two-stage continual multimodal learning framework combining **Stage 1 Null-Space Orthogonal LoRA** with **Stage 2 Exact 2×2 Block Schur Complement Analytic Continual Learning**, **Centered Finite-Sample Noise-to-Signal Ratio (FNSR) GCV Ridge Selection**, **Exemplar-Free Block-Gram Pushforward Transport**, and **0-1-2 Cone-Calibrated Adaptive Spectral-Isotropic Decoupling (SID)**.

---

## 🚀 1-Click Full End-to-End Execution on Google Colab (Tesla T4 GPU)

Click the badge below to launch the self-contained **Full End-to-End Colab T4 Notebook** directly in Google Colab (no manual file upload required):

👉 **[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Thanh-000/ASID-CL-v2/blob/main/notebooks/ASID_CL_v2_Full_EndToEnd_Colab_T4.ipynb)**

Or clone this repository directly inside a Google Colab cell:
```bash
!git clone https://github.com/Thanh-000/ASID-CL-v2.git
%cd ASID-CL-v2
!pip install -e .
```

---

## ✨ Core Algorithmic Innovations (`src/mm_flex_merge/models/`)

| Module | File | Key Mathematical Mechanism |
| :--- | :--- | :--- |
| **Stage 1: Null-Space LoRA** | [`src/mm_flex_merge/models/nullspace_lora.py`](src/mm_flex_merge/models/nullspace_lora.py) | Accumulates uncentered layer-wise activation Gram matrices (`C^(t) = sum X_t^T X_t`), extracts the exact orthogonal complement projector (`P_perp = I - U_k U_k^T`), and merges completed session adapters in constant `O(1)` inference time (`W_merged = W_0 + sum dW_k`) with reversible session un-merging (`unmerge_session`). |
| **Stage 2: 2×2 Block Schur Analytic Solver** | [`src/mm_flex_merge/models/analytic.py`](src/mm_flex_merge/models/analytic.py) | Decouples multimodal `(2P × 2P)` Ridge Regression into two `(P × P)` symmetric positive-definite Cholesky solves via the exact 2×2 Block Schur Complement (`solve_schur`) with zero-copy slice views and BLAS matrix-vector fast paths. |
| **Centered FNSR GCV & Cross-Modal Discounting** | [`src/mm_flex_merge/models/analytic.py`](src/mm_flex_merge/models/analytic.py) | Closed-form Generalized Cross-Validation (`auto_spectral_ridges`) using centered within-class vs. between-class spectral trace ratios and off-diagonal cross-modal discounting (`rho`). |
| **Exemplar-Free Block-Gram Transport** | [`src/mm_flex_merge/models/analytic.py`](src/mm_flex_merge/models/analytic.py) | Transports historical Gram & cross-covariance statistics `(G, S)` across backbone updates (`theta_{t-1} -> theta_t`) via regularized linear pushforward maps (`transport_gram`) without storing raw exemplars. |
| **0-1-2 Cone-Calibrated Adaptive SID** | [`src/mm_flex_merge/models/analytic.py`](src/mm_flex_merge/models/analytic.py) | Dynamic sample-wise reliability gating (`logits(..., adaptive_sid=True)`) using spherical concentration (`||z||_2`) and training-cone cosine alignment to suppress missing or corrupted modalities at test time. |

---

## 📂 Repository Structure

```text
ASID-CL-v2/
├── src/mm_flex_merge/              # Core installable Python package
│   ├── models/
│   │   ├── analytic.py             # Stage 2: BlockAnalyticCL (Schur 2x2, Centered FNSR, Gram Transport, Adaptive SID)
│   │   ├── nullspace_lora.py       # Stage 1: AttnState & SessionLoRA (Null-Space Orthogonal LoRA)
│   │   ├── heads.py                # Multimodal expert classification heads
│   │   └── routers.py              # Centroid & task routing modules
│   ├── merging/                    # Continual parameter-space merging operators
│   ├── engine/                     # Training and evaluation engines
│   ├── protocols/                  # Continual, missing-modality, and statistical protocols
│   └── utils/                      # Reproducibility & diagnostic utilities
├── notebooks/
│   ├── ASID_CL_v2_Full_EndToEnd_Colab_T4.ipynb  # Self-contained Full End-to-End Colab T4 suite
│   ├── H1_gates_colab_proplus.ipynb             # Gate H1-G1 / H1-G2 / H1-G3 verification notebook
│   └── H1_G1_adaptation_gain_food101.ipynb      # Adaptation gain probe on UPMC-Food101
├── scripts/                        # End-to-end benchmark runners, Colab builders & statistical audits
├── tests/                          # 85 unit & mathematical property tests (pytest)
├── configs/                        # Experiment YAML configurations
└── pyproject.toml                  # Package metadata and dependencies
```

---

## 🛠️ Local Installation & Unit Testing

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Run the complete 85-test verification suite
pytest tests/ -q
```

---

## 📜 License

Released under the [MIT License](LICENSE).
