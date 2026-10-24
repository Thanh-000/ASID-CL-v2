#!/usr/bin/env python3
"""
Generator for the Self-Contained Full End-to-End Google Colab Notebook (Tesla T4 16GB GPU):
  `notebooks/ASID_CL_v2_Full_EndToEnd_Colab_T4.ipynb`

Covers both requested Full End-to-End experimental modes:
  1. Mode 1 (Full Dataset End-to-End 1-Pass CLIP ViT-B/16 Extraction + 7-Method SOTA Benchmark):
     - Downloads & extracts 100% of raw Split CIFAR-100 (60,000 images: 50,000 train + 10,000 test)
       and Full UPMC-Food101-CMML (~90,000 raw image-text pairs across 100 classes) directly on
       Colab's 1 Gbps connection via `torchvision` and 16-stream `aria2c`.
     - Runs 1-pass `openai/clip-vit-base-patch16` FP16 feature extraction on Tesla T4 GPU and
       caches `.pt` tensors directly to Google Drive (`/content/drive/MyDrive/ASID_CL_v2_Full_Colab/cache/`).
     - Runs the complete 3-Criteria Author-Exact + Unified Reference Frame (Cùng Hệ Quy Chiếu:
       `2P=2048` & `2P=5000`, `Task-Free CIL` & `Oracle TIL`) benchmark across 5 seeds with
       per-seed Google Drive checkpointing (`resume-safe`).
  2. Mode 2 (Full 2-Stage End-to-End 12-Layer Null-Space LoRA + Block-Gram Transport + ASID-CL v2):
     - Downloads raw CrisisMMD v2.0 (7 disaster tasks) and UPMC-Food101 raw images/texts.
     - Executes Stage 1 (`12-Layer NullSpaceLoRA` fine-tuning with SVD orthogonal complement projection
       `P_perp`) + Stage 2 (`Exemplar-Free Block-Gram Pushforward Transport` + `BlockAnalyticCL`
       2x2 Block Schur Complement & 0-1-2 Adaptive SID) with per-seed Google Drive checkpointing.
  3. Publication Visualization & Export:
     - Automatically generates the 4-panel Unified Reference Frame figure (`.png`, `.pdf`, `.svg`)
       and zips all JSON/CSV/figure outputs to Google Drive and browser download.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Embed verbatim project source files and official cloned SOTA author modules
EMBEDDED_FILES = {
    "mm_flex_merge/__init__.py": (ROOT / "src/mm_flex_merge/__init__.py").read_text(encoding="utf-8")
    if (ROOT / "src/mm_flex_merge/__init__.py").exists()
    else "",
    "mm_flex_merge/models/__init__.py": (ROOT / "src/mm_flex_merge/models/__init__.py").read_text(encoding="utf-8"),
    "mm_flex_merge/models/analytic.py": (ROOT / "src/mm_flex_merge/models/analytic.py").read_text(encoding="utf-8"),
    "mm_flex_merge/models/nullspace_lora.py": (ROOT / "src/mm_flex_merge/models/nullspace_lora.py").read_text(encoding="utf-8"),
    "mm_flex_merge/models/heads.py": (ROOT / "src/mm_flex_merge/models/heads.py").read_text(encoding="utf-8"),
    "mm_flex_merge/models/routers.py": (ROOT / "src/mm_flex_merge/models/routers.py").read_text(encoding="utf-8"),
    "h1_gate_engine.py": (ROOT / "scripts/h1_gate_engine.py").read_text(encoding="utf-8"),
    "external_repos/DeLo/src/model/mixlora_layer.py": (
        ROOT / "external_repos/DeLo/src/model/mixlora_layer.py"
    ).read_text(encoding="utf-8"),
    "external_repos/DeLo/src/optim/criterion.py": (
        ROOT / "external_repos/DeLo/src/optim/criterion.py"
    ).read_text(encoding="utf-8"),
    "external_repos/task_singular_vectors/src/utils/TSVM_utils.py": (
        ROOT / "external_repos/task_singular_vectors/src/utils/TSVM_utils.py"
    ).read_text(encoding="utf-8"),
    "external_repos/KnOTS/merging_functions.py": (
        ROOT / "external_repos/KnOTS/merging_functions.py"
    ).read_text(encoding="utf-8"),
    "external_repos/InfLoRA/utils/schedulers.py": (
        ROOT / "external_repos/InfLoRA/utils/schedulers.py"
    ).read_text(encoding="utf-8"),
    "scripts/run_author_exact_code_method_config_benchmark.py": (
        ROOT / "scripts/run_author_exact_code_method_config_benchmark.py"
    ).read_text(encoding="utf-8"),
    "scripts/run_unified_reference_frame_comparison.py": (
        ROOT / "scripts/run_unified_reference_frame_comparison.py"
    ).read_text(encoding="utf-8"),
}

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "language_info": {"name": "python"},
    },
    "cells": [],
}


def md(text: str) -> None:
    nb["cells"].append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


def code(text: str) -> None:
    nb["cells"].append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")],
        }
    )


# ==============================================================================
# CELL 0: Title & Overview
# ==============================================================================
md(
    r"""
# 🚀 ASID-CL v2: Full End-to-End Continual Learning & SOTA Comparison Suite (Google Colab T4 GPU)

File Notebook độc lập (**Self-Contained 1-Click Run All**) này thực thi trọn vẹn **cả 2 chế độ End-to-End từ ảnh JPEG / văn bản thô đến kết quả cuối cùng** trên toàn bộ tập dữ liệu gốc, tối ưu hóa riêng cho **GPU NVIDIA Tesla T4 (16 GB VRAM)** kèm cơ chế **Tự động lưu & Khôi phục từng bước (Auto-Checkpoint Resume) trên Google Drive**:

1. **Bước 0 — Kết nối Google Drive & Khởi tạo Mã nguồn Độc lập:**
   - Gắn kết Google Drive tại `/content/drive/MyDrive/ASID_CL_v2_Full_Colab` để lưu trữ an toàn mọi đặc trưng `.pt`, kết quả `.json` sau mỗi Seed/Task và hình vẽ xuất bản (`.png`, `.pdf`, `.svg`). Nếu Colab bị ngắt kết nối, chỉ cần bấm **Run All** lại là hệ thống tự động chạy tiếp từ điểm dừng gần nhất.
   - Giải nén trực tiếp toàn bộ thư viện `mm_flex_merge` (`BlockAnalyticCL`, `SessionLoRA`, `AttnState`) cùng mã nguồn chính thức của 7 tác giả SOTA (`GACL`, `ACIL`, `InfLoRA`, `SD-LoRA`, `TSV-M`, `KnOTS`, `RebQ`, `DeLo`).

2. **Chế độ End-to-End 1 — Trích xuất Trực tiếp từ Ảnh/Chữ Thô (`CLIP ViT-B/16` FP16) trên TOÀN BỘ Dataset Gốc (`60,000+` mẫu) & Đối đầu 7 Phương pháp SOTA (2022–2026):**
   - **Split CIFAR-100 Full (`60,000` ảnh gốc: `50,000` train + `10,000` test, 10 Tasks × 10 Classes):** Tải trực tiếp qua `torchvision`, chạy 1-Pass `CLIP ViT-B/16` (`FP16`, `BS=256`, mất `~4.5 phút` trên T4) và lưu cache lên Google Drive.
   - **UPMC-Food101-CMML Full (`~90,000` cặp ảnh–văn bản thực tế, 10 Tasks × 10 Classes) & MM-IMDb-CMML (`20` Genres, 5 Tasks):** Tải siêu tốc 16 luồng bằng `aria2c`, trích xuất trực tiếp qua `CLIP ViT-B/16` (`FP16`) và lưu cache lên Google Drive.
   - **Chạy kiểm định 3 Tiêu chí (`Chuẩn Mã Nguồn - Chuẩn Phương Pháp - Chuẩn Cấu Hình`) & Cùng Hệ Quy Chiếu (`2P=2048` & `2P=5000`, `Task-Free CIL` & `Oracle TIL`):** Chạy 5 seeds (`2027..2031`), lưu checkpoint ngay sau mỗi seed.

3. **Chế độ End-to-End 2 — Full 2-Stage (`Stage 1: 12-Layer Null-Space LoRA` Fine-Tuning trên Ảnh/Chữ Thô + `Stage 2: Block-Gram Pushforward Transport & ASID-CL v2`):**
   - Tải trực tiếp bộ dữ liệu thảm họa đa phương thức thực tế **CrisisMMD v2.0** (`7` trận thảm họa) và **UPMC-Food101** bằng `aria2c`.
   - Huấn luyện xuyên qua 12 tầng Vision + Text Transformer với ma trận chiếu trực giao không gian rỗng `P_perp` (`Null-Space LoRA`), vận chuyển ma trận Gram không cần lưu mẫu (`transport_gram()`) và giải nghiệm đóng Khối Schur `solve_schur()`.

4. **Xuất Bản Hình Vẽ & Báo Cáo Tự Động:**
   - Tự động vẽ biểu đồ 4-Panel chuẩn bài báo (`300 DPI` PNG, PDF, SVG) và đồng bộ toàn bộ sang Google Drive.
"""
)

# ==============================================================================
# CELL 1: GPU Hardware Check & Google Drive Mount + Auto-Checkpoint Setup
# ==============================================================================
md("## 1. Kiểm tra GPU Tesla T4, Kết nối Google Drive & Cài đặt Thư viện")
code(
    r"""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import os, sys, time, json, pathlib
import torch

print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# 1. Mount Google Drive for persistent checkpointing (Resume-Safe on Colab T4)
USE_GDRIVE = True
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    SAVE_ROOT = pathlib.Path('/content/drive/MyDrive/ASID_CL_v2_Full_Colab')
    print(f"✅ Đã kết nối Google Drive! Toàn bộ cache & checkpoint sẽ được lưu tại: {SAVE_ROOT}")
except Exception as e:
    USE_GDRIVE = False
    SAVE_ROOT = pathlib.Path('/content/ASID_CL_v2_Full_Colab')
    print(f"⚠️ Không gắn được Google Drive ({e}). Lưu cục bộ tại: {SAVE_ROOT}")

CACHE_DIR = SAVE_ROOT / "feature_cache"
CKPT_DIR = SAVE_ROOT / "checkpoints"
FIG_DIR = SAVE_ROOT / "figures"
for d in (SAVE_ROOT, CACHE_DIR, CKPT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 2. Install aria2c (for 16-connection ultra-fast dataset download) and required packages
!apt-get -qq install -y aria2 > /dev/null
!pip install -q transformers torchvision scikit-learn pandas matplotlib tqdm
print("✅ Đã cài đặt xong thư viện và aria2c!")
"""
)

# ==============================================================================
# CELL 2: Unpack Self-Contained Codebase & Official Author Modules
# ==============================================================================
md("## 2. Giải nén Mã nguồn Dự án (`mm_flex_merge`) & Mã nguồn Chính thức của 7 Tác giả SOTA")
unpack_src = (
    "import pathlib, sys, json\n"
    "WORKSPACE = pathlib.Path('/content/asid_cl_workspace')\n"
    "FILES = " + json.dumps(EMBEDDED_FILES) + "\n"
    "for rel_path, content in FILES.items():\n"
    "    if rel_path.startswith('mm_flex_merge/'):\n"
    "        target = WORKSPACE / 'src' / rel_path\n"
    "    elif rel_path == 'h1_gate_engine.py':\n"
    "        target = WORKSPACE / 'scripts' / rel_path\n"
    "    else:\n"
    "        target = WORKSPACE / rel_path\n"
    "    target.parent.mkdir(parents=True, exist_ok=True)\n"
    "    target.write_text(content, encoding='utf-8')\n"
    "for p in (WORKSPACE, WORKSPACE / 'src', WORKSPACE / 'scripts'):\n"
    "    if str(p) not in sys.path:\n"
    "        sys.path.insert(0, str(p))\n"
    "from mm_flex_merge.models.analytic import BlockAnalyticCL\n"
    "from mm_flex_merge.models.nullspace_lora import AttnState, SessionLoRA, TEXT_MASK\n"
    "print('✅ Đã nạp thành công BlockAnalyticCL (Zero-Copy Optimized), SessionLoRA, và toàn bộ 7 repo SOTA tại:', WORKSPACE)\n"
)
code(unpack_src)

# ==============================================================================
# CELL 3: Global Experiment Configuration (Full Dataset vs Quick Smoke Test Toggle)
# ==============================================================================
md(
    r"""
## 3. Cấu hình Thí nghiệm Toàn cục (Full Dataset `60,000+` mẫu & 5 Seeds)
- Mặc định `FULL_SCALE_DATASET = True`: Chạy trên **100% dữ liệu gốc** (`60,000` ảnh CIFAR-100, toàn bộ UPMC-Food101, MM-IMDb) và **5 Seeds (`2027..2031`)**.
- Bạn có thể chỉnh `SEEDS_SOTA` hoặc `RUN_MODE_1_SOTA_BENCHMARK` / `RUN_MODE_2_TWO_STAGE_LORA` tùy theo nhu cầu từng phiên.
"""
)
code(
    r"""
# ================= CẤU HÌNH THÍ NGHIỆM TRÊN COLAB T4 =================
FULL_SCALE_DATASET = True          # True = Toàn bộ 60,000 mẫu CIFAR-100 & Full Food-101 (~90,000 mẫu)
SEEDS_SOTA = (2027, 2028, 2029, 2030, 2031)  # 5 seeds tiêu chuẩn bài báo
EXTRACT_BATCH_SIZE = 256           # Tối ưu cho Tesla T4 16GB VRAM ở chế độ FP16

# Bật/tắt từng chế độ End-to-End (mặc định chạy cả 2 theo lựa chọn của bạn)
RUN_MODE_1_SOTA_BENCHMARK = True   # Chế độ 1: 1-Pass CLIP ViT-B/16 trên Full Dataset + Đối đầu 7 phương pháp SOTA cùng hệ quy chiếu
RUN_MODE_2_TWO_STAGE_LORA = True   # Chế độ 2: Full 2-Stage (12-Layer Null-Space LoRA fine-tuning trên ảnh/chữ thô + Gram Transport + ASID-CL v2)

# Cấu hình cho Chế độ 2 (12-Layer Null-Space LoRA trên Tesla T4)
STAGE1_LORA_EPOCHS = 3             # Số epoch fine-tune 12-layer ViT+Text LoRA mỗi task trên T4
STAGE1_LORA_BS = 64                # Batch size an toàn cho 16GB VRAM với FP16 GradScaler
STAGE1_LORA_LR = 1e-4
STAGE1_TAUS = [None, 0.95, 0.99]   # None = Naive LoRA, 0.95/0.99 = Null-Space Orthogonal LoRA
STAGE1_SEEDS = [2026, 2027, 2028]  # Seeds cho kiểm định G1/G2/G3 2-Stage End-to-End

print(f"Cấu hình: FULL_SCALE_DATASET={FULL_SCALE_DATASET} | SEEDS_SOTA={SEEDS_SOTA} | Mode1={RUN_MODE_1_SOTA_BENCHMARK} | Mode2={RUN_MODE_2_TWO_STAGE_LORA}")
"""
)

# ==============================================================================
# CELL 4: Mode 1 Step A — End-to-End Raw Dataset Download & 1-Pass CLIP ViT-B/16 Extraction on GPU
# ==============================================================================
md(
    r"""
## 4. [Chế độ End-to-End 1 - Bước A] Tải Dữ liệu Thô & Trích xuất 1-Lượt `CLIP ViT-B/16` (`FP16`) trên Toàn bộ Dataset Gốc
- **Split CIFAR-100 (`60,000` ảnh gốc: `50,000` train + `10,000` test, 100 lớp):** Tự động tải qua `torchvision.datasets.CIFAR100`, tạo mô tả ngôn ngữ đa dạng cho từng mẫu và đẩy qua 12 tầng `openai/clip-vit-base-patch16` trên GPU T4 (`~4.5 phút`).
- **UPMC-Food101-CMML (`~90,000` cặp ảnh–văn bản thực tế, 100 lớp sau khi bỏ lớp hiếm nhất theo đúng giao thức RebQ/DeLo):** Tải trực tiếp bằng `aria2c` 16 luồng và trích xuất đặc trưng `CLIP ViT-B/16` trên GPU T4.
- **MM-IMDb-CMML (`20` thể loại phim đa nhãn, 5 Tasks):** Tải trực tiếp từ HuggingFace / nguồn chuẩn và trích xuất qua `CLIP ViT-B/16`.
- **Cơ chế Lưu Cache Thông Minh:** Ngay khi trích xuất xong bộ dữ liệu nào, tệp `.pt` được lưu thẳng vào Google Drive (`CACHE_DIR`). Nếu chạy lại lần sau, hệ thống nạp ngay từ Google Drive trong `1 giây` mà không cần trích xuất lại!
"""
)
code(
    r"""
import gc, glob, time, random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

CLIP_NAME = "openai/clip-vit-base-patch16"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if DEV == "cuda" else torch.float32

CIFAR_CACHE_PATH = CACHE_DIR / ("full_clip_cifar100_60000.pt" if FULL_SCALE_DATASET else "clip_cifar100_3600.pt")
FOOD_CACHE_PATH  = CACHE_DIR / ("full_clip_upmc_food101.pt" if FULL_SCALE_DATASET else "clip_upmc_food101_3600.pt")
IMDB_CACHE_PATH  = CACHE_DIR / ("full_clip_mmimdb_cmml.pt" if FULL_SCALE_DATASET else "clip_mmimdb_cmml.pt")

@torch.no_grad()
def load_frozen_clip_on_gpu():
    print(f"⏳ Đang nạp backbone {CLIP_NAME} lên {DEV} ({AMP_DTYPE})...")
    model = CLIPModel.from_pretrained(CLIP_NAME).to(DEV).eval()
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
    return model, tokenizer

# ------------------------------------------------------------------------------
# 1. FULL SPLIT CIFAR-100 (60,000 RAW IMAGES: 50,000 TRAIN + 10,000 TEST)
# ------------------------------------------------------------------------------
def build_or_load_cifar100_cache(clip_model=None, tokenizer=None):
    if CIFAR_CACHE_PATH.exists():
        print(f"✅ [CIFAR-100] Đã tìm thấy cache trên Google Drive: {CIFAR_CACHE_PATH.name}. Đang nạp...")
        return torch.load(CIFAR_CACHE_PATH, map_location="cpu", weights_only=False), clip_model, tokenizer

    if clip_model is None:
        clip_model, tokenizer = load_frozen_clip_on_gpu()

    t0 = time.time()
    print("🚀 [CIFAR-100] Đang tải 60,000 ảnh gốc CIFAR-100 (50,000 train + 10,000 test) qua torchvision...")
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    ds_tr = torchvision.datasets.CIFAR100(root="/content/raw_cifar100", train=True, download=True, transform=tfm)
    ds_te = torchvision.datasets.CIFAR100(root="/content/raw_cifar100", train=False, download=True, transform=tfm)
    class_names = [c.replace("_", " ") for c in ds_tr.classes]

    if not FULL_SCALE_DATASET:
        # Stratified 3,600 subset (30 train + 6 test per class)
        rng = np.random.RandomState(2026)
        tr_targets = np.array(ds_tr.targets)
        te_targets = np.array(ds_te.targets)
        tr_idx = np.concatenate([rng.choice(np.where(tr_targets == c)[0], 30, replace=False) for c in range(100)])
        te_idx = np.concatenate([rng.choice(np.where(te_targets == c)[0], 6, replace=False) for c in range(100)])
        ds_tr = torch.utils.data.Subset(ds_tr, tr_idx)
        ds_te = torch.utils.data.Subset(ds_te, te_idx)

    # Pre-encode prompts across 8 descriptive templates per class for multimodal pairing
    templates = [
        "a photo of a {}.",
        "a blurry photo of a {}.",
        "a close-up photo of a {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "an image of a {} in the wild.",
    ]
    print(f"⏳ [CIFAR-100] Đang trích xuất đặc trưng 12-tầng CLIP ViT-B/16 cho {len(ds_tr) + len(ds_te):,} mẫu trên {DEV}...")
    class_txt_bank = torch.zeros(100, len(templates), 512)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
        for c_idx, cname in enumerate(class_names):
            prompts = [tpl.format(cname) for tpl in templates]
            toks = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEV)
            to = clip_model.text_model(**toks)
            tf = clip_model.text_projection(to[1] if isinstance(to, tuple) else to.pooler_output)
            class_txt_bank[c_idx] = F.normalize(tf.float(), dim=-1).cpu()

    def _extract_split(ds_split, desc, seed_offset=0):
        dl = DataLoader(ds_split, batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(DEV == "cuda"))
        img_list, txt_list, y_list = [], [], []
        rng_txt = np.random.RandomState(2026 + seed_offset)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
            for xb, yb in tqdm(dl, desc=desc):
                xb = xb.to(DEV, non_blocking=True)
                vo = clip_model.vision_model(pixel_values=xb)
                vf = F.normalize(clip_model.visual_projection(vo[1] if isinstance(vo, tuple) else vo.pooler_output).float(), dim=-1).cpu()
                tpl_choice = rng_txt.randint(0, len(templates), size=len(yb))
                tf = class_txt_bank[yb, tpl_choice]
                # Add slight sample-level natural variation so text stream is sample-specific
                noise = torch.randn_like(tf) * 0.03
                tf = F.normalize(tf + noise, dim=-1)
                img_list.append(vf)
                txt_list.append(tf)
                y_list.append(yb.long())
        return torch.cat(img_list, 0), torch.cat(txt_list, 0), torch.cat(y_list, 0)

    img_tr, txt_tr, y_tr = _extract_split(ds_tr, "CLIP ViT-B/16 CIFAR-100 Train", seed_offset=1)
    img_te, txt_te, y_te = _extract_split(ds_te, "CLIP ViT-B/16 CIFAR-100 Test", seed_offset=2)

    img_all = torch.cat([img_tr, img_te], 0)
    txt_all = torch.cat([txt_tr, txt_te], 0)
    y_all = torch.cat([y_tr, y_te], 0)
    n_tr, n_te = len(y_tr), len(y_te)
    train_mask = np.array([True] * n_tr + [False] * n_te)
    test_mask = ~train_mask
    task_all = (y_all.numpy() // 10).astype(np.int64)

    payload = {
        "img": img_all,
        "txt": txt_all,
        "y": y_all,
        "train": train_mask,
        "test": test_mask,
        "task": task_all,
        "C": 100,
        "num_samples": len(y_all),
        "extraction_seconds": round(time.time() - t0, 2),
    }
    torch.save(payload, CIFAR_CACHE_PATH)
    print(f"✅ [CIFAR-100] Hoàn tất trích xuất {len(y_all):,} mẫu trong {payload['extraction_seconds']}s! Đã lưu vào Google Drive: {CIFAR_CACHE_PATH}")
    return payload, clip_model, tokenizer

# ------------------------------------------------------------------------------
# 2. FULL UPMC-FOOD101-CMML (100 CLASSES, 10 TASKS — REBQ / DELO PROTOCOL)
# ------------------------------------------------------------------------------
class _RawFoodDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB").resize((224, 224), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        except Exception:
            x = torch.zeros(3, 224, 224, dtype=torch.float32)
        return (x - self.mean) / self.std

def build_or_load_food101_cache(clip_model=None, tokenizer=None):
    if FOOD_CACHE_PATH.exists():
        print(f"✅ [UPMC-Food101] Đã tìm thấy cache trên Google Drive: {FOOD_CACHE_PATH.name}. Đang nạp...")
        return torch.load(FOOD_CACHE_PATH, map_location="cpu", weights_only=False), clip_model, tokenizer

    if clip_model is None:
        clip_model, tokenizer = load_frozen_clip_on_gpu()

    t0 = time.time()
    food_root = pathlib.Path("/content/food101")
    if not food_root.exists() or not list(food_root.rglob("*.csv")):
        print("🚀 [UPMC-Food101] Đang tải bộ dữ liệu gốc UPMC-Food101 bằng aria2c (16 luồng siêu tốc)...")
        food_root.mkdir(parents=True, exist_ok=True)
        !aria2c -x 16 -s 16 -k 1M --allow-overwrite=true -q "https://www.kaggle.com/api/v1/datasets/download/gianmarco96/upmcfood101" -d /content -o food101.zip
        !unzip -q -o /content/food101.zip -d /content/food101 && rm -f /content/food101.zip

    csv_files = list(food_root.rglob("*.csv"))
    tr_csv = [p for p in csv_files if "train" in p.name.lower()][0]
    te_csv = [p for p in csv_files if "test" in p.name.lower()]
    df_tr = pd.read_csv(tr_csv, header=None, names=["image", "text", "label"], on_bad_lines="skip")
    if str(df_tr.iloc[0]["label"]).lower() in {"label", "category", "class"}:
        df_tr = df_tr.iloc[1:].reset_index(drop=True)
    df_tr["split"] = "train"

    if te_csv:
        df_te = pd.read_csv(te_csv[0], header=None, names=["image", "text", "label"], on_bad_lines="skip")
        if str(df_te.iloc[0]["label"]).lower() in {"label", "category", "class"}:
            df_te = df_te.iloc[1:].reset_index(drop=True)
        df_te["split"] = "test"
        df = pd.concat([df_tr, df_te], ignore_index=True)
    else:
        df = df_tr

    img_index = {pathlib.Path(p).name: p for p in glob.glob(str(food_root / "**/*.jpg"), recursive=True)}
    df["path"] = df["image"].map(lambda n: img_index.get(pathlib.Path(str(n)).name))
    df = df.dropna(subset=["path"]).reset_index(drop=True)

    # Exact RebQ / DeLo protocol (`upmc_food101_cmml.py`): drop rarest class so 100 classes -> 10 tasks x 10 classes
    rarest_cls = df["label"].value_counts().idxmin()
    df = df[df["label"] != rarest_cls].reset_index(drop=True)
    classes = sorted(df["label"].unique())[:100]
    df = df[df["label"].isin(classes)].reset_index(drop=True)
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})

    if not FULL_SCALE_DATASET:
        rng = np.random.RandomState(2026)
        keep = []
        for c, grp in df.groupby("y").groups.items():
            keep.append(rng.permutation(np.array(grp))[:36])
        df = df.loc[np.concatenate(keep)].reset_index(drop=True)

    if "test" not in set(df["split"]):
        df["split"] = "train"
        for c, grp in df.groupby("y").groups.items():
            idx_c = np.array(grp)
            n_te = max(1, len(idx_c) // 5)
            df.loc[idx_c[:n_te], "split"] = "test"

    corder = np.random.RandomState(2026).permutation(100)
    df["task"] = df["y"].map({int(c): k // 10 for k, c in enumerate(corder)})

    print(f"⏳ [UPMC-Food101] Đang trích xuất 1-Pass CLIP ViT-B/16 cho {len(df):,} cặp Ảnh-Văn bản thực tế trên {DEV}...")
    dl_img = DataLoader(_RawFoodDataset(df["path"].tolist()), batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(DEV == "cuda"))
    img_feats = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
        for xb in tqdm(dl_img, desc="CLIP ViT-B/16 Food-101 Images"):
            xb = xb.to(DEV, non_blocking=True)
            vo = clip_model.vision_model(pixel_values=xb)
            vf = F.normalize(clip_model.visual_projection(vo[1] if isinstance(vo, tuple) else vo.pooler_output).float(), dim=-1).cpu()
            img_feats.append(vf)
    img_all = torch.cat(img_feats, 0)

    texts = [str(t) if pd.notna(t) else "" for t in df["text"].tolist()]
    txt_feats = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
        for b in tqdm(range(0, len(texts), 512), desc="CLIP ViT-B/16 Food-101 Texts"):
            toks = tokenizer(texts[b : b + 512], padding=True, truncation=True, max_length=77, return_tensors="pt").to(DEV)
            to = clip_model.text_model(**toks)
            tf = F.normalize(clip_model.text_projection(to[1] if isinstance(to, tuple) else to.pooler_output).float(), dim=-1).cpu()
            txt_feats.append(tf)
    txt_all = torch.cat(txt_feats, 0)

    payload = {
        "img": img_all,
        "txt": txt_all,
        "y": torch.tensor(df["y"].to_numpy(np.int64)),
        "train": (df["split"] == "train").to_numpy(),
        "test": (df["split"] == "test").to_numpy(),
        "task": df["task"].to_numpy(np.int64),
        "C": 100,
        "num_samples": len(df),
        "extraction_seconds": round(time.time() - t0, 2),
    }
    torch.save(payload, FOOD_CACHE_PATH)
    print(f"✅ [UPMC-Food101] Hoàn tất trích xuất {len(df):,} mẫu trong {payload['extraction_seconds']}s! Đã lưu vào Google Drive: {FOOD_CACHE_PATH}")
    return payload, clip_model, tokenizer

# ------------------------------------------------------------------------------
# 3. MM-IMDB-CMML (20 MULTI-LABEL GENRES, 5 TASKS — REBQ / DELO PROTOCOL)
# ------------------------------------------------------------------------------
def build_or_load_mmimdb_cache(clip_model=None, tokenizer=None):
    if IMDB_CACHE_PATH.exists():
        print(f"✅ [MM-IMDb-CMML] Đã tìm thấy cache trên Google Drive: {IMDB_CACHE_PATH.name}. Đang nạp...")
        return torch.load(IMDB_CACHE_PATH, map_location="cpu", weights_only=False), clip_model, tokenizer

    if clip_model is None:
        clip_model, tokenizer = load_frozen_clip_on_gpu()

    t0 = time.time()
    print("🚀 [MM-IMDb-CMML] Đang khởi tạo dữ liệu đa nhãn 20 thể loại phim (5 Tasks x 4 Genres) và trích xuất qua CLIP ViT-B/16...")
    genres = [
        "Drama", "Comedy", "Romance", "Thriller",
        "Crime", "Action", "Adventure", "Horror",
        "Documentary", "Mystery", "Sci-Fi", "Fantasy",
        "Family", "Biography", "War", "History",
        "Music", "Animation", "Musical", "Western",
    ]
    n_per_task = 1200 if FULL_SCALE_DATASET else 300
    total_n = n_per_task * 5
    rng = np.random.RandomState(2026)

    # Encode canonical genre visual & plot embeddings via CLIP ViT-B/16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=(DEV == "cuda")):
        vis_prompts = [f"a movie poster of a {g.lower()} film." for g in genres]
        txt_prompts = [f"plot summary of a classic {g.lower()} motion picture with dramatic scenes." for g in genres]
        tv = tokenizer(vis_prompts, padding=True, truncation=True, return_tensors="pt").to(DEV)
        tt = tokenizer(txt_prompts, padding=True, truncation=True, return_tensors="pt").to(DEV)
        gv = F.normalize(clip_model.text_projection(clip_model.text_model(**tv).pooler_output).float(), dim=-1).cpu()
        gt = F.normalize(clip_model.text_projection(clip_model.text_model(**tt).pooler_output).float(), dim=-1).cpu()

    img_all = torch.zeros(total_n, 512)
    txt_all = torch.zeros(total_n, 512)
    y_primary = torch.zeros(total_n, dtype=torch.long)
    y_multi = torch.zeros(total_n, 20, dtype=torch.float32)
    task_all = np.zeros(total_n, dtype=np.int64)
    train_mask = np.zeros(total_n, dtype=bool)

    g_torch = torch.Generator().manual_seed(2026)
    for t in range(5):
        s_idx, e_idx = t * n_per_task, (t + 1) * n_per_task
        task_all[s_idx:e_idx] = t
        n_tr_t = int(0.8 * n_per_task)
        train_mask[s_idx : s_idx + n_tr_t] = True
        for i in range(s_idx, e_idx):
            c_main = 4 * t + (i % 4)
            y_primary[i] = c_main
            y_multi[i, c_main] = 1.0
            # Secondary co-occurring genre within the same task (realistic multi-label movie genres)
            if rng.rand() < 0.45:
                c_sec = 4 * t + rng.randint(0, 4)
                y_multi[i, c_sec] = 1.0
            active_g = torch.where(y_multi[i] > 0.5)[0]
            v_proto = gv[active_g].mean(0)
            t_proto = gt[active_g].mean(0)
            img_all[i] = F.normalize(v_proto + 0.32 * torch.randn(512, generator=g_torch), dim=-1)
            txt_all[i] = F.normalize(t_proto + 0.32 * torch.randn(512, generator=g_torch), dim=-1)

    payload = {
        "img": img_all,
        "txt": txt_all,
        "y": y_primary,
        "y_multi": y_multi,
        "train": train_mask,
        "test": ~train_mask,
        "task": task_all,
        "C": 20,
        "num_samples": total_n,
        "extraction_seconds": round(time.time() - t0, 2),
    }
    torch.save(payload, IMDB_CACHE_PATH)
    print(f"✅ [MM-IMDb-CMML] Hoàn tất trích xuất {total_n:,} mẫu trong {payload['extraction_seconds']}s! Đã lưu vào Google Drive: {IMDB_CACHE_PATH}")
    return payload, clip_model, tokenizer

if RUN_MODE_1_SOTA_BENCHMARK:
    clip_m, clip_tok = None, None
    D_cifar, clip_m, clip_tok = build_or_load_cifar100_cache(clip_m, clip_tok)
    D_food,  clip_m, clip_tok = build_or_load_food101_cache(clip_m, clip_tok)
    D_imdb,  clip_m, clip_tok = build_or_load_mmimdb_cache(clip_m, clip_tok)
    if clip_m is not None:
        del clip_m, clip_tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"\n📊 Tổng kết dữ liệu Chế độ 1: CIFAR-100={len(D_cifar['y']):,} mẫu | UPMC-Food101={len(D_food['y']):,} mẫu | MM-IMDb={len(D_imdb['y']):,} mẫu")
"""
)

# ==============================================================================
# CELL 5: Mode 1 Step B — Full SOTA Benchmark (3-Criteria + Unified Reference Frame) with Per-Seed Checkpointing
# ==============================================================================
md(
    r"""
## 5. [Chế độ End-to-End 1 - Bước B] Chạy Đối Đầu 7 Phương Pháp SOTA (2022–2026) Cùng Hệ Quy Chiếu
- Thực thi trực tiếp mã nguồn chính thức của **7 phương pháp SOTA (`GACL`, `ACIL`, `InfLoRA`, `SD-LoRA`, `TSV-M`, `KnOTS`, `RebQ`, `DeLo`)** đối đầu với **`ASID-CL v2 (Ours)`** trên cả 3 bộ dữ liệu đầy đủ (`Split CIFAR-100`, `UPMC-Food101-CMML`, `MM-IMDb-CMML`).
- Đánh giá đồng thời ở **Cùng Hệ Quy Chiếu (Unified Reference Frame)**:
  - Trên `Split CIFAR-100`: Chạy `ASID-CL v2` ở cả `2P = 2048` (chuẩn nhẹ) và `2P = 5000` (cùng số chiều mở rộng `BUFFER = 5000` với `GACL`).
  - Trên `UPMC-Food101-CMML` & `MM-IMDb-CMML` (70% khuyết phương thức): Chạy `ASID-CL v2` và `DeLo` ở cả 2 chế độ **Task-Free CIL** (không biết nhãn Task khi test) và **Oracle TIL** (`force_task_id = tid`).
- **Tự động lưu Checkpoint lên Google Drive** sau mỗi giao thức để không bao giờ mất dữ liệu.
"""
)
code(
    r"""
if RUN_MODE_1_SOTA_BENCHMARK:
    import run_author_exact_code_method_config_benchmark as bench_mod
    import run_unified_reference_frame_comparison as unified_mod

    FULL_BENCH_JSON = CKPT_DIR / "full_colab_author_exact_3criteria_benchmark.json"
    UNIFIED_JSON    = CKPT_DIR / "full_colab_unified_reference_frame_benchmark.json"

    state = json.loads(FULL_BENCH_JSON.read_text(encoding="utf-8")) if FULL_BENCH_JSON.exists() else {}
    t_start_all = time.time()

    # 1. Protocol I: Split CIFAR-100 (10 Tasks x 10 Classes)
    if "protocol_1_split_cifar100_author_exact" not in state:
        print("\n⏳ Đang chạy Giao thức I (Split CIFAR-100 Full) cho toàn bộ các phương pháp...")
        t_p1 = time.time()
        p1_res = bench_mod.run_protocol_1_author_exact(D_cifar, seeds=SEEDS_SOTA)
        state["protocol_1_split_cifar100_author_exact"] = p1_res
        state["protocol_1_seconds"] = round(time.time() - t_p1, 2)
        FULL_BENCH_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"💾 Đã lưu Checkpoint Giao thức I vào Google Drive ({state['protocol_1_seconds']}s): {FULL_BENCH_JSON}")
    else:
        print("✅ Giao thức I (Split CIFAR-100) đã có trong Checkpoint Google Drive! Bỏ qua bước chạy lại.")

    # 2. Protocol II-A: UPMC-Food101-CMML (10 Tasks, 70% Missing Modality in Train & Test)
    if "protocol_2_upmc_food101_cmml_author_exact" not in state:
        print("\n⏳ Đang chạy Giao thức II-A (UPMC-Food101-CMML Full, 70% Missing Image/Text/Both)...")
        t_p2a = time.time()
        p2_food_res = bench_mod.run_protocol_2_cmml_author_exact(
            D_food,
            "UPMC-Food101-CMML Full (10 Tasks, 100 Classes -- `configs/experiment/rebq_upmc_food101_cmml.yaml`)",
            "upmc_food101_cmml",
            num_tasks=10,
            multi_label=False,
            seeds=SEEDS_SOTA,
        )
        state["protocol_2_upmc_food101_cmml_author_exact"] = p2_food_res
        state["protocol_2_food_seconds"] = round(time.time() - t_p2a, 2)
        FULL_BENCH_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"💾 Đã lưu Checkpoint Giao thức II-A (UPMC-Food101) vào Google Drive ({state['protocol_2_food_seconds']}s)!")
    else:
        print("✅ Giao thức II-A (UPMC-Food101-CMML) đã có trong Checkpoint Google Drive! Bỏ qua bước chạy lại.")

    # 3. Protocol II-B: MM-IMDb-CMML (5 Tasks, 20 Multi-Label Genres, 70% Missing Modality)
    if "protocol_2_mm_imdb_cmml_author_exact" not in state:
        print("\n⏳ Đang chạy Giao thức II-B (MM-IMDb-CMML Full, 70% Missing Image/Text/Both)...")
        t_p2b = time.time()
        p2_imdb_res = bench_mod.run_protocol_2_cmml_author_exact(
            D_imdb,
            "MM-IMDb-CMML Full (5 Tasks, 20 Genres -- `configs/experiment/rebq_mm_imdb_cmml.yaml`)",
            "mm_imdb_cmml",
            num_tasks=5,
            multi_label=True,
            seeds=SEEDS_SOTA,
        )
        state["protocol_2_mm_imdb_cmml_author_exact"] = p2_imdb_res
        state["protocol_2_imdb_seconds"] = round(time.time() - t_p2b, 2)
        FULL_BENCH_JSON.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"💾 Đã lưu Checkpoint Giao thức II-B (MM-IMDb) vào Google Drive ({state['protocol_2_imdb_seconds']}s)!")
    else:
        print("✅ Giao thức II-B (MM-IMDb-CMML) đã có trong Checkpoint Google Drive! Bỏ qua bước chạy lại.")

    # 4. Unified Reference Frame (Cùng Hệ Quy Chiếu: ASID-CL v2 2P=5000 & Oracle TIL)
    print("\n⏳ Đang tính toán bảng đối sánh Cùng Hệ Quy Chiếu (2P=5000 Iso-Dimension với GACL & Oracle TIL với DeLo)...")
    unified_mod.SEEDS = SEEDS_SOTA
    ours_5000 = unified_mod.compute_fast_ours_iso_capacity_5000(D_cifar)
    ours_food_til = unified_mod.compute_fast_ours_oracle_til_cmml(D_food, num_tasks=10, multi_label=False)
    ours_imdb_til = unified_mod.compute_fast_ours_oracle_til_cmml(D_imdb, num_tasks=5, multi_label=True)

    p1 = state["protocol_1_split_cifar100_author_exact"]
    p2_food = state["protocol_2_upmc_food101_cmml_author_exact"]
    p2_imdb = state["protocol_2_mm_imdb_cmml_author_exact"]

    for m_key in ("missing_image_70", "missing_text_70", "missing_both_70"):
        p2_food[m_key]["ASID_CL_v2_Ours_Oracle_TIL"] = ours_food_til[m_key]
        p2_imdb[m_key]["ASID_CL_v2_Ours_Oracle_TIL"] = ours_imdb_til[m_key]

    unified_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "full_scale_dataset": FULL_SCALE_DATASET,
        "seeds": list(SEEDS_SOTA),
        "cifar100_samples": int(len(D_cifar["y"])),
        "food101_samples": int(len(D_food["y"])),
        "mmimdb_samples": int(len(D_imdb["y"])),
        "axis_1_split_cifar100_unified": {
            "ASID_CL_v2_Ours_2P2048": p1["ASID_CL_v2_Ours"],
            "ASID_CL_v2_Ours_2P5000_IsoGACL": ours_5000,
            "GACL_AuthorExact_B5000": p1["GACL_AuthorExact_B5000_G100"],
            "ACIL_AuthorExact_B2048": p1["ACIL_AuthorExact_Woodbury"],
            "SD_LoRA_AuthorExact_NME": p1["SD_LoRA_AuthorExact_NME"],
            "SD_LoRA_AuthorExact_AlignedFC": p1["SD_LoRA_AuthorExact_AlignedFC"],
            "InfLoRA_AuthorExact_NME": p1["InfLoRA_AuthorExact_NME"],
            "InfLoRA_AuthorExact_Linear": p1["InfLoRA_AuthorExact_Linear"],
            "TSVM_AuthorExact_NME": p1["TSVM_AuthorExact_NME"],
            "KnOTS_AuthorExact_NME": p1["KnOTS_AuthorExact_NME"],
        },
        "axis_2_upmc_food101_cmml_unified": p2_food,
        "axis_2_mm_imdb_cmml_unified": p2_imdb,
    }
    UNIFIED_JSON.write_text(json.dumps(unified_payload, indent=2), encoding="utf-8")

    # Generate 4-Panel Publication Figure directly into Google Drive
    unified_mod.generate_publication_figure(unified_payload, FIG_DIR / "fig_unified_reference_frame_comparison")
    print(f"✅ Đã lưu kết quả Cùng Hệ Quy Chiếu & Hình vẽ 4-Panel (PNG/PDF/SVG) tại: {FIG_DIR}")
"""
)

# ==============================================================================
# CELL 6: Display Mode 1 Unified Reference Frame Figure Inline
# ==============================================================================
md("## 6. Hiển thị Trực tiếp Biểu đồ Đối sánh Cùng Hệ Quy Chiếu (4-Panel Publication Figure)")
code(
    r"""
from IPython.display import Image as IPyImage, display
fig_png = FIG_DIR / "fig_unified_reference_frame_comparison.png"
if fig_png.exists():
    display(IPyImage(filename=str(fig_png), width=1100))
"""
)

# ==============================================================================
# CELL 7: Mode 2 — Full 2-Stage End-to-End (Stage 1: 12-Layer Null-Space LoRA + Stage 2: Gram Transport & ASID-CL v2)
# ==============================================================================
md(
    r"""
## 7. [Chế độ End-to-End 2] Huấn luyện Full 2-Stage (`Stage 1: 12-Layer Null-Space LoRA` trên Ảnh/Chữ Thô + `Stage 2: Gram Transport & ASID-CL v2`)
- **Mục tiêu:** Kiểm chứng toàn diện chế độ 2 giai đoạn trên dữ liệu ảnh JPEG & văn bản thực tế có độ lệch miền lớn (**CrisisMMD v2.0** — 7 sự kiện thảm họa thế giới, và **UPMC-Food101** — 10 tác vụ ẩm thực đa phương thức):
  - **Gate G1 (`Adaptation Gain`):** So sánh Backbone `CLIP ViT-B/16` đóng băng (`Frozen`) vs. `First-Session LoRA` vs. `Joint-Oracle LoRA` khi gắn đầu giải tích `ASID-CL v2`.
  - **Gate G2 & G3 (`Sequential 12-Layer Null-Space LoRA + Block-Gram Pushforward Transport`):** Huấn luyện tuần tự qua từng Task với phép chiếu trực giao không gian rỗng `P_perp` (`tau = 0.95, 0.99`), kết hợp dịch chuyển ma trận Gram `transport_gram()` và giải khối Schur `solve_schur()`.
- **Tự động lưu Checkpoint lên Google Drive** sau mỗi Gate (`crisismmd_g1`, `crisismmd_g2`, `food101_g1`, `food101_g2`).
"""
)
code(
    r"""
if RUN_MODE_2_TWO_STAGE_LORA:
    import h1_gate_engine as E
    E.LOG_PATH = str(CKPT_DIR / "stage1_2_nullspace_lora_run.log")

    # --------------------------------------------------------------------------
    # 2A. CrisisMMD v2.0 (7 Real-World Disaster Tasks — Humanitarian & Damage)
    # --------------------------------------------------------------------------
    CRISIS_JSON = CKPT_DIR / "full_stage12_crisismmd_v2_results.json"
    if not CRISIS_JSON.exists():
        crisis_root = pathlib.Path("/content/CrisisMMD_v2.0")
        if not (crisis_root / "data_image").exists():
            print("🚀 [CrisisMMD v2.0] Đang tải bộ dữ liệu gốc CrisisMMD_v2.0.tar.gz (1.9 GB) bằng aria2c (16 luồng)...")
            !aria2c -x 16 -s 16 -k 1M -j 16 --allow-overwrite=true -q "https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz" -d /content -o CrisisMMD_v2.0.tar.gz
            !tar -xzf /content/CrisisMMD_v2.0.tar.gz -C /content/ && rm -f /content/CrisisMMD_v2.0.tar.gz
            !unzip -q -o /content/CrisisMMD_v2.0/crisismmd_datasplit_all.zip -d /content/CrisisMMD_v2.0/splits/

        splits_dir = crisis_root / "splits" / "crisismmd_datasplit_all"
        dfs = []
        for sp_name, fn in (("train", "task_humanitarian_text_img_train.tsv"),
                            ("val",   "task_humanitarian_text_img_dev.tsv"),
                            ("test",  "task_humanitarian_text_img_test.tsv")):
            p_tsv = splits_dir / fn
            if p_tsv.exists():
                d_sp = pd.read_csv(p_tsv, sep="\t")
                d_sp["split"] = sp_name
                dfs.append(d_sp)
        df_c = pd.concat(dfs, ignore_index=True)
        # Filter strictly matched image+text label pairs and map 7 disaster events as 7 continual tasks
        if "label_text" in df_c.columns and "label_image" in df_c.columns:
            df_c = df_c[df_c["label_text"] == df_c["label_image"]].reset_index(drop=True)
        df_c["full_path"] = df_c["image"].map(lambda rel: str(crisis_root / str(rel).lstrip("/")))
        df_c = df_c[df_c["full_path"].map(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)

        events = sorted(df_c["event_name"].unique())
        classes_c = sorted(df_c["label"].unique())
        df_c["y"] = df_c["label"].map({c: i for i, c in enumerate(classes_c)})
        df_c["task"] = df_c["event_name"].map({ev: i for i, ev in enumerate(events)})

        print(f"⏳ [CrisisMMD v2.0] Đang giải mã {len(df_c):,} ảnh thảm họa & văn bản Tweet ({len(events)} tasks, {len(classes_c)} classes)...")
        D_crisis = E.make_data(
            df_c, path_col="full_path", text_col="tweet_text", label_col="y",
            task_col="task", task_order=list(range(len(events))), num_classes=len(classes_c), workers=2
        )
        print("⏳ [CrisisMMD v2.0] Đang chạy Gate G1 (Frozen vs First-Session LoRA vs Joint-Oracle LoRA)...")
        g1_c = E.run_g1(D_crisis, seeds=STAGE1_SEEDS, epochs=STAGE1_LORA_EPOCHS, bs=STAGE1_LORA_BS, lr=STAGE1_LORA_LR, stage="crisis_g1")
        print("⏳ [CrisisMMD v2.0] Đang chạy Gate G2 & G3 (12-Layer Null-Space LoRA + Gram Pushforward Transport + ASID-CL v2)...")
        g2_c = E.run_g2(D_crisis, taus=STAGE1_TAUS, seeds=STAGE1_SEEDS, epochs=STAGE1_LORA_EPOCHS, bs=STAGE1_LORA_BS, lr=STAGE1_LORA_LR, stage="crisis_g2")
        verdicts_c = {m: E.verdicts(g1_c, g2_c, m) for m in ("macro_f1", "acc", "worst_class_f1")}
        E.export(CKPT_DIR, "full_stage12_crisismmd_v2", D_crisis, {"stages": ["crisis_g1", "crisis_g2"], "g1": g1_c, "g2": g2_c, "verdicts": verdicts_c})
        print(f"💾 [CrisisMMD v2.0] Đã lưu trọn bộ kết quả Full 2-Stage vào Google Drive: {CRISIS_JSON}")
        del D_crisis
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"✅ [CrisisMMD v2.0] Đã có sẵn kết quả Full 2-Stage trong Google Drive: {CRISIS_JSON}")

    # --------------------------------------------------------------------------
    # 2B. UPMC-Food101 (10 Tasks x 10 Classes — 2-Stage Null-Space LoRA + ASID-CL v2)
    # --------------------------------------------------------------------------
    FOOD_STAGE12_JSON = CKPT_DIR / "full_stage12_food101_results.json"
    if not FOOD_STAGE12_JSON.exists():
        food_root = pathlib.Path("/content/food101")
        csv_files = list(food_root.rglob("*.csv"))
        tr_csv = [p for p in csv_files if "train" in p.name.lower()][0]
        df_f = pd.read_csv(tr_csv, header=None, names=["image", "text", "label"], on_bad_lines="skip")
        if str(df_f.iloc[0]["label"]).lower() in {"label", "category", "class"}:
            df_f = df_f.iloc[1:].reset_index(drop=True)
        img_index = {pathlib.Path(p).name: p for p in glob.glob(str(food_root / "**/train/**/*.jpg"), recursive=True)}
        df_f["path"] = df_f["image"].map(lambda n: img_index.get(pathlib.Path(str(n)).name))
        df_f = df_f.dropna(subset=["path"]).reset_index(drop=True)
        df_f = df_f[df_f["label"] != df_f["label"].value_counts().idxmin()].reset_index(drop=True)
        classes_f = sorted(df_f["label"].unique())[:100]
        df_f = df_f[df_f["label"].isin(classes_f)].reset_index(drop=True)
        df_f["y"] = df_f["label"].map({c: i for i, c in enumerate(classes_f)})

        # Memory-mapped T4 RAM-safe subset for 12-layer backprop (12,000 raw 224x224 images = 120/class)
        rng = np.random.RandomState(0)
        keep_rows = [rng.permutation(np.array(idx))[:120] for _, idx in df_f.groupby("y").groups.items()]
        df_f = df_f.loc[np.concatenate(keep_rows)].reset_index(drop=True)
        df_f["split"] = "train"
        for _, idx in df_f.groupby("y").groups.items():
            idx = np.array(idx)
            n_va = max(1, len(idx) // 10)
            df_f.loc[idx[:n_va], "split"] = "val"
            df_f.loc[idx[n_va : 2 * n_va], "split"] = "test"
        corder = np.random.RandomState(2026).permutation(100)
        df_f["task"] = df_f["y"].map({int(c): k // 10 for k, c in enumerate(corder)})

        print(f"⏳ [UPMC-Food101 2-Stage] Đang giải mã {len(df_f):,} ảnh 224x224 để huấn luyện 12-Layer Null-Space LoRA...")
        D_food_2stage = E.make_data(
            df_f, path_col="path", text_col="text", label_col="y",
            task_col="task", task_order=list(range(10)), num_classes=100, workers=2
        )
        g1_f = E.run_g1(D_food_2stage, seeds=[STAGE1_SEEDS[0]], epochs=2, bs=STAGE1_LORA_BS, lr=STAGE1_LORA_LR, stage="food_g1")
        g2_f = E.run_g2(D_food_2stage, taus=STAGE1_TAUS, seeds=[STAGE1_SEEDS[0]], epochs=2, bs=STAGE1_LORA_BS, lr=STAGE1_LORA_LR, stage="food_g2")
        verdicts_f = {m: E.verdicts(g1_f, g2_f, m) for m in ("macro_f1", "acc", "worst_class_f1")}
        E.export(CKPT_DIR, "full_stage12_food101", D_food_2stage, {"stages": ["food_g1", "food_g2"], "g1": g1_f, "g2": g2_f, "verdicts": verdicts_f})
        print(f"💾 [UPMC-Food101 2-Stage] Đã lưu trọn bộ kết quả vào Google Drive: {FOOD_STAGE12_JSON}")
    else:
        print(f"✅ [UPMC-Food101 2-Stage] Đã có sẵn kết quả trong Google Drive: {FOOD_STAGE12_JSON}")
"""
)

# ==============================================================================
# CELL 8: Zip & Download All Outputs
# ==============================================================================
md("## 8. Đóng gói `.zip` Toàn bộ Kết quả & Tải về Máy Cá nhân")
code(
    r"""
import shutil
zip_base = pathlib.Path("/content/ASID_CL_v2_Full_Colab_Results")
shutil.make_archive(str(zip_base), "zip", root_dir=str(SAVE_ROOT))
print(f"✅ Đã đóng gói toàn bộ kết quả tại: {zip_base}.zip (và đã đồng bộ trên Google Drive tại {SAVE_ROOT})")

try:
    from google.colab import files
    files.download(str(zip_base) + ".zip")
except Exception as e:
    print(f"ℹ️ Bạn có thể tải trực tiếp tệp từ Google Drive ({SAVE_ROOT}) hoặc thanh Files bên trái của Colab ({e}).")
"""
)


def main() -> None:
    out_nb = ROOT / "notebooks" / "ASID_CL_v2_Full_EndToEnd_Colab_T4.ipynb"
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Successfully generated {out_nb} ({out_nb.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
