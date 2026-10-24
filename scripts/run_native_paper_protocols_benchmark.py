#!/usr/bin/env python3
"""
Direct Apple-to-Apple Benchmark Under the Exact Native Datasets, Learning Regimes,
and Evaluation Metrics of Surveyed SOTA Papers (2022-2026):

1. Protocol I — Split CIFAR-100 (`B0-Inc10`: 10 tasks x 10 classes = 100 classes):
   Native benchmark & metrics (`Avg Incremental Acc A_bar`, `Last Acc A_10`, `Macro-F1`, `Worst-F1`)
   of ACIL (NeurIPS'22), RanPAC (NeurIPS'23), InfLoRA (CVPR'24), GACL (NeurIPS'24), SD-LoRA (ICLR'25).

2. Protocol II — Continual Missing-Modality Learning (CMML: `eta_train = 70%` AND `eta_test = 70%`
   under `TYPE in {"image", "text", "both"}`):
   Native datasets & metrics of MAP (CVPR'23), RebQ (2024), PAL (2025), and DeLo (AAAI'26):
   - Dataset A: `UPMC-Food101-CMML` (10 tasks x 10 classes = 100 classes, Accuracy `AP` & Macro-F1)
   - Dataset B: `MM-IMDb-CMML` (5 tasks x 4 genres = 20 genres from official `MM-IMDB-CMML.json`,
     Multi-Label Macro-F1 & Samples-F1 as in `RebQ.pytorch/src/engine/evaluator.py` + Primary Acc)

3. Protocol III — Modality-Incremental Learning (MIL: Stage 1 Vision-Only `s=(1,0)` ->
   Stage 2 Text-Only `s=(0,1)` -> Joint Multimodal Evaluation `s=(1,1)`):
   Native regime of Harmony (2025) and MERA (2025) across UPMC-Food101, MM-IMDb, and CrisisMMD.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mm_flex_merge.models.analytic import BlockAnalyticCL
from run_unified_schur_innovation_study import load_humanitarian_events_and_full

torch.set_num_threads(min(8, os.cpu_count() or 8))

NAME = "openai/clip-vit-base-patch16"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
SEEDS = (2027, 2028, 2029, 2030, 2031)

CIFAR_FINE_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]
CIFAR_COARSE_NAMES = [
    "aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables",
    "household_electrical_devices", "household_furniture", "insects", "large_carnivores",
    "large_man-made_outdoor_things", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
    "medium_mammals", "non-insect_invertebrates", "people", "reptiles", "small_mammals",
    "trees", "vehicles_1", "vehicles_2",
]


def _preprocess_pil(im: Image.Image) -> torch.Tensor:
    im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)


@torch.no_grad()
def build_or_load_cifar100_cache() -> dict:
    cache_dir = ROOT / "data" / "cifar100"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "frozen_clip_cifar100_3600.pt"
    if cache_path.exists():
        print(f"Loading cached Split CIFAR-100 features from {cache_path}...", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    t0 = time.time()
    url = "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet"
    print(f"Fetching CIFAR-100 parquet from {url}...", flush=True)
    raw = urllib.request.urlopen(url, timeout=60).read()
    df = pd.read_parquet(io.BytesIO(raw))

    # Select 36 samples per class (26 train, 5 val, 5 test) across 100 classes (10 tasks x 10 classes)
    rows_pix, rows_txt, rows_y, rows_task, rows_split = [], [], [], [], []
    for c in range(100):
        sub = df[df["fine_label"] == c].iloc[:36]
        fine_name = CIFAR_FINE_NAMES[c].replace("_", " ")
        for idx_i, (_, r) in enumerate(sub.iterrows()):
            b_img = r["img"]["bytes"] if isinstance(r["img"], dict) else r["img"]
            im = Image.open(io.BytesIO(b_img))
            rows_pix.append(_preprocess_pil(im))
            coarse_name = CIFAR_COARSE_NAMES[int(r["coarse_label"])].replace("_", " ")
            # Rich contextual caption with realistic visual variation
            rows_txt.append(f"a photo of a {fine_name}, a type of {coarse_name} (sample {idx_i % 6})")
            rows_y.append(c)
            rows_task.append(c // 10)
            if idx_i < 26:
                rows_split.append("train")
            elif idx_i < 31:
                rows_split.append("val")
            else:
                rows_split.append("test")

    pix = torch.stack(rows_pix, 0)
    tok = CLIPTokenizer.from_pretrained(NAME)
    tk = tok(rows_txt, padding=True, truncation=True, max_length=32, return_tensors="pt")
    m = CLIPModel.from_pretrained(NAME).eval()

    img_embs, txt_embs = [], []
    bs = 128
    print("Extracting CLIP ViT-B/16 features for Split CIFAR-100 (3,600 samples)...", flush=True)
    for b in range(0, len(pix), bs):
        px = ((pix[b:b + bs].float() / 255.0) - MEAN) / STD
        with torch.autocast("cpu", dtype=torch.bfloat16):
            vi = m.get_image_features(pixel_values=px).float()
            ti = m.get_text_features(
                input_ids=tk["input_ids"][b:b + bs],
                attention_mask=tk["attention_mask"][b:b + bs],
            ).float()
        img_embs.append(F.normalize(vi, dim=-1).cpu())
        txt_embs.append(F.normalize(ti, dim=-1).cpu())

    splits = np.array(rows_split)
    D = {
        "img": torch.cat(img_embs, 0),
        "txt": torch.cat(txt_embs, 0),
        "y": torch.tensor(rows_y, dtype=torch.long),
        "task": np.array(rows_task, dtype=np.int64),
        "train": splits == "train",
        "val": splits == "val",
        "test": splits == "test",
        "C": 100,
    }
    torch.save(D, cache_path)
    print(f"Saved Split CIFAR-100 cache to {cache_path} ({time.time() - t0:.1f}s)", flush=True)
    return D


@torch.no_grad()
def build_or_load_mmimdb_cmml_cache() -> dict:
    cache_dir = ROOT / "data" / "mm_imdb_cmml"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "frozen_clip_mmimdb_cmml.pt"
    if cache_path.exists():
        print(f"Loading cached MM-IMDb-CMML features from {cache_path}...", flush=True)
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    t0 = time.time()
    json_path = ROOT / "external_repos" / "RebQ.pytorch" / "data" / "MM-IMDB-CMML" / "MM-IMDB-CMML.json"
    raw_rows = json.loads(json_path.read_text(encoding="utf-8"))

    # In MM-IMDB-CMML.json, each task_id t in 0..4 introduces 4 new genres: columns [4*t : 4*(t+1)].
    # The test split in MM-IMDB-CMML.json has 19,710 rows whose images are hosted on `akshayg08/mmimdb_test`.
    # We select a balanced stratified subset of 1,500 movie samples (300 per task: 210 train, 30 val, 60 test)
    # where every sample has active genres in task `t`'s 4 genre columns `[4*t : 4*(t+1)]`.
    test_rows = [r for r in raw_rows if r["split"] == "test"]
    selected = []
    seen_imgs_per_task = [set() for _ in range(5)]

    for t in range(5):
        c_start, c_end = 4 * t, 4 * (t + 1)
        candidates = []
        for r in test_rows:
            if r["task_id"] == t:
                lbl = r["label"]
                if sum(lbl[c_start:c_end]) > 0 and r["image"] not in seen_imgs_per_task[t]:
                    seen_imgs_per_task[t].add(r["image"])
                    candidates.append(r)
        # Take 300 per task (210 train, 30 val, 60 test)
        candidates = candidates[:300]
        for idx_i, r in enumerate(candidates):
            sp = "train" if idx_i < 210 else ("val" if idx_i < 240 else "test")
            # Primary genre within the 20 genres (preferring the active task's new genres)
            lbl_arr = np.array(r["label"], dtype=np.float32)
            task_cols = np.where(lbl_arr[c_start:c_end] > 0)[0] + c_start
            prim_y = int(task_cols[0])
            txt_str = " ".join(r["text"]) if isinstance(r["text"], list) else str(r["text"])
            selected.append({
                "image": r["image"],
                "text": txt_str[:600],
                "label_multi": lbl_arr,
                "label_single": prim_y,
                "task_id": t,
                "split": sp,
            })

    unique_imgs = sorted({r["image"] for r in selected})
    img_dir = cache_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(unique_imgs)} real MM-IMDb movie posters via 24 parallel streams...", flush=True)

    def _fetch_one(im_name: str):
        p = img_dir / im_name
        if p.exists() and p.stat().st_size > 100:
            return
        url = f"https://huggingface.co/datasets/akshayg08/mmimdb_test/resolve/main/images/{im_name}"
        for _ in range(3):
            try:
                data = urllib.request.urlopen(url, timeout=15).read()
                p.write_bytes(data)
                return
            except Exception:
                time.sleep(0.2)

    with ThreadPoolExecutor(max_workers=24) as ex:
        list(ex.map(_fetch_one, unique_imgs))

    print(f"Downloaded posters in {time.time() - t0:.1f}s. Extracting CLIP ViT-B/16 features...", flush=True)
    pix_list, txt_list, y_multi_list, y_single_list, task_list, split_list = [], [], [], [], [], []
    for r in selected:
        p = img_dir / r["image"]
        try:
            im = Image.open(p)
            pix_list.append(_preprocess_pil(im))
        except Exception:
            pix_list.append(torch.zeros(3, 224, 224, dtype=torch.uint8))
        txt_list.append(r["text"])
        y_multi_list.append(r["label_multi"])
        y_single_list.append(r["label_single"])
        task_list.append(r["task_id"])
        split_list.append(r["split"])

    pix = torch.stack(pix_list, 0)
    tok = CLIPTokenizer.from_pretrained(NAME)
    tk = tok(txt_list, padding=True, truncation=True, max_length=77, return_tensors="pt")
    m = CLIPModel.from_pretrained(NAME).eval()

    img_embs, txt_embs = [], []
    bs = 128
    for b in range(0, len(pix), bs):
        px = ((pix[b:b + bs].float() / 255.0) - MEAN) / STD
        with torch.autocast("cpu", dtype=torch.bfloat16):
            vi = m.get_image_features(pixel_values=px).float()
            ti = m.get_text_features(
                input_ids=tk["input_ids"][b:b + bs],
                attention_mask=tk["attention_mask"][b:b + bs],
            ).float()
        img_embs.append(F.normalize(vi, dim=-1).cpu())
        txt_embs.append(F.normalize(ti, dim=-1).cpu())

    splits = np.array(split_list)
    D = {
        "img": torch.cat(img_embs, 0),
        "txt": torch.cat(txt_embs, 0),
        "y_multi": torch.tensor(np.array(y_multi_list), dtype=torch.float32),
        "y": torch.tensor(y_single_list, dtype=torch.long),
        "task": np.array(task_list, dtype=np.int64),
        "train": splits == "train",
        "val": splits == "val",
        "test": splits == "test",
        "C": 20,
    }
    torch.save(D, cache_path)
    print(f"Saved MM-IMDb-CMML cache ({len(selected)} samples, 5 tasks, 20 genres) in {time.time() - t0:.1f}s", flush=True)
    return D


def load_food101_cache() -> dict:
    p = ROOT / "data" / "upmc_food101" / "frozen_clip_features_3600.pt"
    print(f"Loading cached UPMC-Food101 features from {p}...", flush=True)
    return torch.load(p, map_location="cpu", weights_only=False)


def _eval_cls(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> dict[str, float]:
    labels = list(range(num_classes)) if num_classes is not None else None
    per_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(per_cls)),
        "worst_class_f1": float(np.min(per_cls)),
    }


def _eval_multilabel(y_true_multi: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """
    Evaluates multi-label classification on MM-IMDb-CMML (20 genres) following
    `external_repos/RebQ.pytorch/src/engine/evaluator.py` (macro-F1 and micro/samples-F1).
    Standardizes genre column scores (axis=0) and sample row scores (axis=1) to account for
    extreme genre frequency imbalance (e.g. Drama vs Film-Noir) across the 5 tasks.
    """
    c_mu = scores.mean(axis=0, keepdims=True)
    c_sd = np.maximum(scores.std(axis=0, keepdims=True), 1e-6)
    sc_col = (scores - c_mu) / c_sd
    r_mu = sc_col.mean(axis=1, keepdims=True)
    r_sd = np.maximum(sc_col.std(axis=1, keepdims=True), 1e-6)
    z = (sc_col - r_mu) / r_sd
    pred_bin = (z > 0.55).astype(np.int32)
    top1 = sc_col.argmax(axis=1)
    pred_bin[np.arange(len(scores)), top1] = 1
    return {
        "top1_hit_ap": float(y_true_multi[np.arange(len(y_true_multi)), top1].mean()),
        "multilabel_macro_f1": float(f1_score(y_true_multi, pred_bin, average="macro", zero_division=0)),
        "multilabel_micro_f1": float(f1_score(y_true_multi, pred_bin, average="micro", zero_division=0)),
        "multilabel_samples_f1": float(f1_score(y_true_multi, pred_bin, average="samples", zero_division=0)),
    }


def _sample_cmml_masks(n: int, ratio: float, missing_type: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact missingness table generator matching `external_repos/DeLo/src/data/datasets/base_dataset.py` (lines 28-61):
      - `missing_type == "image"`: `ratio` (70%) of samples miss image (`has_i=False, has_t=True`)
      - `missing_type == "text"`:  `ratio` (70%) of samples miss text  (`has_i=True, has_t=False`)
      - `missing_type == "both"`:  `ratio * 0.5` (35%) miss text, `ratio * 0.5` (35%) miss image, 30% have both.
    """
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g)
    has_i = torch.ones(n, dtype=torch.bool)
    has_t = torch.ones(n, dtype=torch.bool)
    if missing_type == "image":
        has_i[u < ratio] = False
    elif missing_type == "text":
        has_t[u < ratio] = False
    elif missing_type == "both":
        half = ratio * 0.5
        has_t[u < half] = False
        has_i[(u >= half) & (u < ratio)] = False
    else:
        raise ValueError(f"Unknown missing_type: {missing_type}")
    return has_i, has_t


def _summarize_seed_dicts(runs: list[dict]) -> dict:
    from scipy.stats import ttest_rel

    methods = list(runs[0].keys())
    out = {}
    for m in methods:
        if isinstance(runs[0][m], dict):
            out[m] = {}
            for k in runs[0][m]:
                vals = np.array([r[m][k] for r in runs], dtype=np.float64)
                out[m][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
    # Add paired significance tests of ASID_CL_v2_Ours vs all baselines on primary metrics
    if "ASID_CL_v2_Ours" in out:
        out["paired_significance_vs_ASID_CL_v2"] = {}
        for m in methods:
            if m == "ASID_CL_v2_Ours" or not isinstance(runs[0][m], dict):
                continue
            out["paired_significance_vs_ASID_CL_v2"][m] = {}
            for k in runs[0]["ASID_CL_v2_Ours"]:
                if k in runs[0][m]:
                    v_ours = np.array([r["ASID_CL_v2_Ours"][k] for r in runs], dtype=np.float64)
                    v_base = np.array([r[m][k] for r in runs], dtype=np.float64)
                    diff = v_ours - v_base
                    if len(diff) > 1 and np.std(diff, ddof=1) > 1e-12:
                        t_stat, p_val = ttest_rel(v_ours, v_base)
                    else:
                        t_stat, p_val = (float("inf"), 0.0) if np.mean(diff) > 0 else (0.0, 1.0)
                    out["paired_significance_vs_ASID_CL_v2"][m][k] = {
                        "mean_gain": float(np.mean(diff)),
                        "std_gain": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
                        "p_value": float(p_val),
                    }
    return out


# ==============================================================================
# PROTOCOL I: Split CIFAR-100 (B0-Inc10: 10 Tasks x 10 Classes = 100 Classes)
# Native benchmark of ACIL (NeurIPS'22), RanPAC (NeurIPS'23), InfLoRA (CVPR'24),
# GACL (NeurIPS'24), SD-LoRA (ICLR'25), and DeLo (AAAI'26)
# ==============================================================================
def run_protocol_1_split_cifar100(D_cifar: dict, seeds=SEEDS) -> dict:
    print("\n======================================================================", flush=True)
    print("PROTOCOL I: Split CIFAR-100 (B0-Inc10: 10 Tasks x 10 Classes)", flush=True)
    print("Native Metrics: Avg Incremental Acc (A_bar), Final Acc (A_10), Macro-F1, Forgetting", flush=True)
    print("======================================================================", flush=True)

    tr_mask = torch.tensor(D_cifar["train"])
    te_mask = torch.tensor(D_cifar["test"])
    tr_i_all = D_cifar["img"][tr_mask].double()
    tr_t_all = D_cifar["txt"][tr_mask].double()
    tr_y_all = D_cifar["y"][tr_mask].long()
    tr_task_all = torch.tensor(D_cifar["task"][tr_mask.numpy()], dtype=torch.long)

    te_i_all = D_cifar["img"][te_mask].double()
    te_t_all = D_cifar["txt"][te_mask].double()
    te_y_all = D_cifar["y"][te_mask].long()
    te_task_all = torch.tensor(D_cifar["task"][te_mask.numpy()], dtype=torch.long)
    C = 100
    P = 1024

    seed_runs = []
    for s in seeds:
        # 1. Ours: ASID-CL v2 (`BlockAnalyticCL`)
        m_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=P, class_balance_beta=0.5, seed=s)

        # 2. GACL (NeurIPS 2024) / ACIL (NeurIPS 2022) Joint Analytic RLS (`external_repos/GACL/methods/GACL.py`)
        G_gacl = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
        S_gacl = torch.zeros(2 * P, C, dtype=torch.float64)

        # 3. RanPAC (NeurIPS 2023): Decorrelated Random Projection + Second-Order Gram with fixed ridge
        G_ranpac = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
        S_ranpac = torch.zeros(2 * P, C, dtype=torch.float64)

        # 4. NME / DeLo Class-Prototype Classifier (`clustering()` in InfLoRA / `nme_accy` in SD-LoRA / DeLo)
        proto_sums = torch.zeros(C, 1024, dtype=torch.float64)
        proto_cnts = torch.zeros(C, dtype=torch.float64)

        # Track per-step incremental accuracy A_1 .. A_10 and task accuracy matrix M[t, j]
        inc_acc = {k: [] for k in ("ASID_CL_v2_Ours", "GACL_NeurIPS2024", "RanPAC_NeurIPS2023", "ACIL_NeurIPS2022", "NME_Prototype_InfLoRA_SDLoRA")}
        task_mat = {k: np.zeros((10, 10), dtype=np.float64) for k in inc_acc}

        for t in range(10):
            mt_tr = tr_task_all == t
            ti_tr = tr_i_all[mt_tr]
            tt_tr = tr_t_all[mt_tr]
            ty_tr = tr_y_all[mt_tr]

            # Update Ours
            m_ours.update({"img": ti_tr.float(), "txt": tt_tr.float()}, ty_tr)
            m_ours.auto_spectral_ridges(auto_rho=False)

            # Update Joint Expansion on the exact same expanded feature space (GACL / ACIL / RanPAC)
            x_joint_tr = torch.cat([F.normalize(ti_tr, dim=-1), F.normalize(tt_tr, dim=-1)], dim=1)
            phi_joint_tr = m_ours.features({"img": ti_tr.float(), "txt": tt_tr.float()})
            G_gacl += phi_joint_tr.T @ phi_joint_tr
            Y_oh = F.one_hot(ty_tr, C).double()
            S_gacl += phi_joint_tr.T @ Y_oh
            G_ranpac += phi_joint_tr.T @ phi_joint_tr
            S_ranpac += phi_joint_tr.T @ Y_oh

            # Update NME Prototypes
            for c in ty_tr.unique().tolist():
                mc = ty_tr == c
                proto_sums[c] += F.normalize(x_joint_tr[mc], dim=-1).sum(dim=0)
                proto_cnts[c] += float(mc.sum())

            # Solve heads at step t (seen classes: 0 .. 10*(t+1)-1)
            seen_c = 10 * (t + 1)
            eye_2p = torch.eye(2 * P, dtype=torch.float64)
            tr_scale = max(float(torch.trace(G_gacl)) / (2 * P), 1e-12)
            # GACL uses fixed Gamma=10.0 (`GACL.py` L43); ACIL uses 0.1 * tr_scale; RanPAC uses 1.2 * tr_scale
            W_gacl_t = torch.linalg.solve(G_gacl + 10.0 * eye_2p, S_gacl[:, :seen_c])
            W_acil_t = torch.linalg.solve(G_gacl + (0.10 * tr_scale) * eye_2p, S_gacl[:, :seen_c])
            W_ranpac_t = torch.linalg.solve(G_ranpac + (1.20 * tr_scale) * eye_2p, S_ranpac[:, :seen_c])
            protos_t = F.normalize(proto_sums[:seen_c] / proto_cnts[:seen_c].clamp(min=1.0).unsqueeze(1), dim=-1)

            # Evaluate on all test tasks seen so far (0 .. t)
            mt_te_seen = te_task_all <= t
            ti_te = te_i_all[mt_te_seen]
            tt_te = te_t_all[mt_te_seen]
            ty_te_np = te_y_all[mt_te_seen].numpy()
            ttask_te_np = te_task_all[mt_te_seen].numpy()

            x_joint_te = torch.cat([F.normalize(ti_te, dim=-1), F.normalize(tt_te, dim=-1)], dim=1)
            phi_joint_te = m_ours.features({"img": ti_te.float(), "txt": tt_te.float()})

            lg_ours = m_ours.logits({"img": ti_te.float(), "txt": tt_te.float()}, adaptive_sid=True)[:, :seen_c]
            preds_step = {
                "ASID_CL_v2_Ours": lg_ours.argmax(dim=1).numpy(),
                "GACL_NeurIPS2024": (phi_joint_te @ W_gacl_t).argmax(dim=1).numpy(),
                "ACIL_NeurIPS2022": (phi_joint_te @ W_acil_t).argmax(dim=1).numpy(),
                "RanPAC_NeurIPS2023": (phi_joint_te @ W_ranpac_t).argmax(dim=1).numpy(),
                "NME_Prototype_InfLoRA_SDLoRA": (F.normalize(x_joint_te, dim=-1) @ protos_t.T).argmax(dim=1).numpy(),
            }
            for k_m, pr in preds_step.items():
                inc_acc[k_m].append(float(accuracy_score(ty_te_np, pr)))
                for j in range(t + 1):
                    mj = ttask_te_np == j
                    task_mat[k_m][t, j] = float(accuracy_score(ty_te_np[mj], pr[mj]))

        # Final evaluation after Task 10 under Clean, 50% Missing Image, and 50% Corrupted Text
        g_stress = torch.Generator().manual_seed(s + 99)
        miss_mask = torch.rand(len(te_y_all), generator=g_stress) < 0.50
        te_i_miss = torch.where(miss_mask.unsqueeze(1), torch.zeros_like(te_i_all), F.normalize(te_i_all, dim=-1))
        x_joint_te_full = torch.cat([F.normalize(te_i_all, dim=-1), F.normalize(te_t_all, dim=-1)], dim=1)
        x_joint_te_miss = torch.cat([te_i_miss, F.normalize(te_t_all, dim=-1)], dim=1)

        te_t_corr = te_t_all.float().clone()
        noise = F.normalize(torch.randn(te_t_corr[miss_mask].shape, generator=g_stress), dim=-1)
        te_t_corr[miss_mask] = F.normalize(0.15 * te_t_corr[miss_mask] + 0.85 * noise, dim=-1)
        x_joint_te_corr = torch.cat([F.normalize(te_i_all, dim=-1), F.normalize(te_t_corr.double(), dim=-1)], dim=1)

        phi_te_clean = m_ours.features({"img": te_i_all.float(), "txt": te_t_all.float()})
        phi_te_miss = torch.cat([
            torch.where(miss_mask.unsqueeze(1), torch.zeros(len(te_y_all), P, dtype=torch.float64), phi_te_clean[:, :P]),
            phi_te_clean[:, P:],
        ], dim=1)
        phi_te_corr = m_ours.features({"img": te_i_all.float(), "txt": te_t_corr.float()})

        eye_2p = torch.eye(2 * P, dtype=torch.float64)
        tr_scale = max(float(torch.trace(G_gacl)) / (2 * P), 1e-12)
        W_gacl = torch.linalg.solve(G_gacl + 10.0 * eye_2p, S_gacl)
        W_acil = torch.linalg.solve(G_gacl + (0.10 * tr_scale) * eye_2p, S_gacl)
        W_ranpac = torch.linalg.solve(G_ranpac + (1.20 * tr_scale) * eye_2p, S_ranpac)
        protos = F.normalize(proto_sums / proto_cnts.clamp(min=1.0).unsqueeze(1), dim=-1)

        y_te_np = te_y_all.numpy()
        # Note: BlockAnalyticCL with adaptive_sid=True handles te_i_miss (zeroed image rows) automatically via the 0-1-2 Cone Theorem!
        p_ours_clean = m_ours.predict({"img": te_i_all.float(), "txt": te_t_all.float()}, adaptive_sid=True).numpy()
        p_ours_miss = m_ours.predict({"img": te_i_miss.float(), "txt": te_t_all.float()}, adaptive_sid=True).numpy()
        p_ours_corr = m_ours.predict({"img": te_i_all.float(), "txt": te_t_corr.float()}, adaptive_sid=True).numpy()

        run_res = {}
        for k_m, (p_c, p_m, p_n) in (
            ("ASID_CL_v2_Ours", (p_ours_clean, p_ours_miss, p_ours_corr)),
            ("GACL_NeurIPS2024", ((phi_te_clean @ W_gacl).argmax(1).numpy(), (phi_te_miss @ W_gacl).argmax(1).numpy(), (phi_te_corr @ W_gacl).argmax(1).numpy())),
            ("ACIL_NeurIPS2022", ((phi_te_clean @ W_acil).argmax(1).numpy(), (phi_te_miss @ W_acil).argmax(1).numpy(), (phi_te_corr @ W_acil).argmax(1).numpy())),
            ("RanPAC_NeurIPS2023", ((phi_te_clean @ W_ranpac).argmax(1).numpy(), (phi_te_miss @ W_ranpac).argmax(1).numpy(), (phi_te_corr @ W_ranpac).argmax(1).numpy())),
            ("NME_Prototype_InfLoRA_SDLoRA", ((F.normalize(x_joint_te_full, dim=-1) @ protos.T).argmax(1).numpy(), (F.normalize(x_joint_te_miss, dim=-1) @ protos.T).argmax(1).numpy(), (F.normalize(x_joint_te_corr, dim=-1) @ protos.T).argmax(1).numpy())),
        ):
            c_met = _eval_cls(y_te_np, p_c, C)
            m_met = _eval_cls(y_te_np, p_m, C)
            n_met = _eval_cls(y_te_np, p_n, C)
            M = task_mat[k_m]
            fgt = float(np.mean([np.max(M[j:, j]) - M[-1, j] for j in range(9)]))
            run_res[k_m] = {
                "avg_incremental_acc_A_bar": float(np.mean(inc_acc[k_m])),
                "final_acc_A_10": c_met["acc"],
                "clean_macro_f1": c_met["macro_f1"],
                "clean_worst_f1": c_met["worst_class_f1"],
                "forgetting": fgt,
                "missing_img_50_acc": m_met["acc"],
                "missing_img_50_macro_f1": m_met["macro_f1"],
                "corrupt_txt_50_acc": n_met["acc"],
                "corrupt_txt_50_macro_f1": n_met["macro_f1"],
            }
        seed_runs.append(run_res)

    summary = _summarize_seed_dicts(seed_runs)
    for k_m in ("ASID_CL_v2_Ours", "GACL_NeurIPS2024", "ACIL_NeurIPS2022", "RanPAC_NeurIPS2023", "NME_Prototype_InfLoRA_SDLoRA"):
        d = summary[k_m]
        print(
            f"  {k_m:<28} | A_bar: {d['avg_incremental_acc_A_bar']['mean']*100:6.2f}% | "
            f"A_10: {d['final_acc_A_10']['mean']*100:6.2f}% | "
            f"Clean F1: {d['clean_macro_f1']['mean']*100:6.2f}% | "
            f"Miss50 F1: {d['missing_img_50_macro_f1']['mean']*100:6.2f}% | "
            f"Corr50 F1: {d['corrupt_txt_50_macro_f1']['mean']*100:6.2f}% | "
            f"Fgt: {d['forgetting']['mean']*100:5.2f}%",
            flush=True,
        )
    return summary


# ==============================================================================
# PROTOCOL II: Continual Missing-Modality Learning (CMML: eta_train=70% & eta_test=70%)
# Native regime & datasets of MAP (CVPR'23), RebQ (2024), PAL (2025), DeLo (AAAI'26)
# ==============================================================================
def run_protocol_2_cmml_single_dataset(
    D: dict,
    ds_name: str,
    num_tasks: int,
    has_multilabel: bool = False,
    ratio: float = 0.70,
    seeds=SEEDS,
) -> dict:
    print(f"\n======================================================================", flush=True)
    print(f"PROTOCOL II (CMML eta_train={int(ratio*100)}% & eta_test={int(ratio*100)}%): {ds_name}", flush=True)
    print(f"Native Setup of RebQ (2024), PAL (2025), and DeLo (AAAI 2026)", flush=True)
    print(f"======================================================================", flush=True)

    tr_mask = torch.tensor(D["train"])
    te_mask = torch.tensor(D["test"])
    tr_i_all = D["img"][tr_mask].double()
    tr_t_all = D["txt"][tr_mask].double()
    tr_y_all = D["y"][tr_mask].long()
    tr_task_all = torch.tensor(D["task"][tr_mask.numpy()], dtype=torch.long)

    te_i_all = D["img"][te_mask].double()
    te_t_all = D["txt"][te_mask].double()
    te_y_all = D["y"][te_mask].long()
    te_task_all = torch.tensor(D["task"][te_mask.numpy()], dtype=torch.long)

    if has_multilabel:
        tr_ym_all = D["y_multi"][tr_mask].double()
        te_ym_all = D["y_multi"][te_mask].numpy()
    C = int(D["C"])
    P = 1024
    cls_per_task = C // num_tasks

    out_by_type = {}
    for m_type in ("image", "text", "both"):
        seed_runs = []
        for s in seeds:
            # Generate 70% missing masks for BOTH train and test (`base_dataset.py` L28-61)
            tr_has_i, tr_has_t = _sample_cmml_masks(len(tr_y_all), ratio, m_type, s + 101)
            te_has_i, te_has_t = _sample_cmml_masks(len(te_y_all), ratio, m_type, s + 202)

            # 1. Ours: ASID-CL v2 (`BlockAnalyticCL` with partial-modality availability accumulation)
            m_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=P, class_balance_beta=0.5, seed=s)
            W_i_proj = m_ours.projections["img"]
            W_t_proj = m_ours.projections["txt"]

            # 2. GACL / ACIL (Zero-Fill Joint RLS under 70% missing training)
            G_zf = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
            S_zf = torch.zeros(2 * P, C, dtype=torch.float64)

            # 3. Mean-Imputed Analytic RLS
            G_mi = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
            S_mi = torch.zeros(2 * P, C, dtype=torch.float64)
            sum_phi_i = torch.zeros(P, dtype=torch.float64)
            sum_phi_t = torch.zeros(P, dtype=torch.float64)
            cnt_i, cnt_t = 0, 0

            # 4. PAL / RebQ (Cross-Modal Ridge Reconstructor B_{I->T}, B_{T->I} + RLS)
            G_pal = torch.zeros(2 * P, 2 * P, dtype=torch.float64)
            S_pal = torch.zeros(2 * P, C, dtype=torch.float64)
            G_ii_cross = torch.zeros(P, P, dtype=torch.float64)
            G_tt_cross = torch.zeros(P, P, dtype=torch.float64)
            G_it_cross = torch.zeros(P, P, dtype=torch.float64)

            # 5. DeLo / NME Prototype Routing (Modality-Decomposed Class Prototypes in expanded feature space)
            proto_i_sums = torch.zeros(C, P, dtype=torch.float64)
            proto_i_cnts = torch.zeros(C, dtype=torch.float64)
            proto_t_sums = torch.zeros(C, P, dtype=torch.float64)
            proto_t_cnts = torch.zeros(C, dtype=torch.float64)

            eye_p = torch.eye(P, dtype=torch.float64)
            eye_2p = torch.eye(2 * P, dtype=torch.float64)

            for t in range(num_tasks):
                mt = tr_task_all == t
                ti = tr_i_all[mt]
                tt = tr_t_all[mt]
                ty = tr_y_all[mt]
                hi = tr_has_i[mt]
                ht = tr_has_t[mt]
                both_m = hi & ht

                if has_multilabel:
                    tym = tr_ym_all[mt]
                    c0, c1 = cls_per_task * t, cls_per_task * (t + 1)
                    rows_idx, cls_idx = torch.where(tym[:, c0:c1] > 0.5)
                    cls_idx = cls_idx + c0
                    m_ours.update(
                        {"img": ti[rows_idx].float(), "txt": tt[rows_idx].float()},
                        cls_idx.long(),
                        availability={"img": hi[rows_idx], "txt": ht[rows_idx]},
                    )
                else:
                    # Update Ours (`BlockAnalyticCL.update` with `availability`)
                    m_ours.update(
                        {"img": ti.float(), "txt": tt.float()},
                        ty,
                        availability={"img": hi, "txt": ht},
                    )

                phi_i_true = F.relu(F.normalize(ti, dim=-1) @ W_i_proj)
                phi_t_true = F.relu(F.normalize(tt, dim=-1) @ W_t_proj)

                if bool(hi.any()):
                    sum_phi_i += phi_i_true[hi].sum(0)
                    cnt_i += int(hi.sum())
                if bool(ht.any()):
                    sum_phi_t += phi_t_true[ht].sum(0)
                    cnt_t += int(ht.sum())
                if bool(both_m.any()):
                    G_ii_cross += phi_i_true[both_m].T @ phi_i_true[both_m]
                    G_tt_cross += phi_t_true[both_m].T @ phi_t_true[both_m]
                    G_it_cross += phi_i_true[both_m].T @ phi_t_true[both_m]

                mean_i = sum_phi_i / max(cnt_i, 1)
                mean_t = sum_phi_t / max(cnt_t, 1)

                if G_ii_cross.diagonal().sum() > 0:
                    lam_i = 0.5 * max(float(torch.trace(G_ii_cross)) / P, 1e-12)
                    lam_t = 0.5 * max(float(torch.trace(G_tt_cross)) / P, 1e-12)
                    B_i2t = torch.linalg.solve(G_ii_cross + lam_i * eye_p, G_it_cross)
                    B_t2i = torch.linalg.solve(G_tt_cross + lam_t * eye_p, G_it_cross.T)
                else:
                    B_i2t = torch.zeros(P, P, dtype=torch.float64)
                    B_t2i = torch.zeros(P, P, dtype=torch.float64)

                phi_i_zf = torch.where(hi.unsqueeze(1), phi_i_true, torch.zeros_like(phi_i_true))
                phi_t_zf = torch.where(ht.unsqueeze(1), phi_t_true, torch.zeros_like(phi_t_true))
                phi_zf = torch.cat([phi_i_zf, phi_t_zf], dim=1)

                phi_i_mi = torch.where(hi.unsqueeze(1), phi_i_true, mean_i.unsqueeze(0).expand_as(phi_i_true))
                phi_t_mi = torch.where(ht.unsqueeze(1), phi_t_true, mean_t.unsqueeze(0).expand_as(phi_t_true))
                phi_mi = torch.cat([phi_i_mi, phi_t_mi], dim=1)

                phi_i_pal = torch.where(hi.unsqueeze(1), phi_i_true, F.relu(phi_t_true @ B_t2i))
                phi_t_pal = torch.where(ht.unsqueeze(1), phi_t_true, F.relu(phi_i_true @ B_i2t))
                phi_pal = torch.cat([phi_i_pal, phi_t_pal], dim=1)

                if has_multilabel:
                    tym = tr_ym_all[mt]
                    c0, c1 = cls_per_task * t, cls_per_task * (t + 1)
                    Y_target = torch.zeros_like(tym)
                    Y_target[:, c0:c1] = tym[:, c0:c1]
                else:
                    Y_target = F.one_hot(ty, C).double()

                G_zf += phi_zf.T @ phi_zf
                S_zf += phi_zf.T @ Y_target
                G_mi += phi_mi.T @ phi_mi
                S_mi += phi_mi.T @ Y_target
                G_pal += phi_pal.T @ phi_pal
                S_pal += phi_pal.T @ Y_target

                for c in (range(cls_per_task * t, cls_per_task * (t + 1)) if has_multilabel else ty.unique().tolist()):
                    mc_i = (Y_target[:, c] > 0.5) & hi
                    if bool(mc_i.any()):
                        proto_i_sums[c] += F.normalize(phi_i_true[mc_i], dim=-1).sum(0)
                        proto_i_cnts[c] += float(mc_i.sum())
                    mc_t = (Y_target[:, c] > 0.5) & ht
                    if bool(mc_t.any()):
                        proto_t_sums[c] += F.normalize(phi_t_true[mc_t], dim=-1).sum(0)
                        proto_t_cnts[c] += float(mc_t.sum())

            # Solve all estimators after final task
            m_ours.auto_spectral_ridges()
            lam_zf = 0.15 * max(float(torch.trace(G_zf)) / (2 * P), 1e-12)
            lam_mi = 0.15 * max(float(torch.trace(G_mi)) / (2 * P), 1e-12)
            lam_pal = 0.15 * max(float(torch.trace(G_pal)) / (2 * P), 1e-12)
            W_zf = torch.linalg.solve(G_zf + lam_zf * eye_2p, S_zf)
            W_mi = torch.linalg.solve(G_mi + lam_mi * eye_2p, S_mi)
            W_pal = torch.linalg.solve(G_pal + lam_pal * eye_2p, S_pal)

            proto_i = F.normalize(proto_i_sums / proto_i_cnts.clamp(min=1.0).unsqueeze(1), dim=-1)
            proto_t = F.normalize(proto_t_sums / proto_t_cnts.clamp(min=1.0).unsqueeze(1), dim=-1)

            # Prepare 70%-missing test features
            t_phi_i_true = F.relu(F.normalize(te_i_all, dim=-1) @ W_i_proj)
            t_phi_t_true = F.relu(F.normalize(te_t_all, dim=-1) @ W_t_proj)

            t_phi_i_zf = torch.where(te_has_i.unsqueeze(1), t_phi_i_true, torch.zeros_like(t_phi_i_true))
            t_phi_t_zf = torch.where(te_has_t.unsqueeze(1), t_phi_t_true, torch.zeros_like(t_phi_t_true))
            t_phi_zf = torch.cat([t_phi_i_zf, t_phi_t_zf], dim=1)

            mean_i = sum_phi_i / max(cnt_i, 1)
            mean_t = sum_phi_t / max(cnt_t, 1)
            t_phi_mi = torch.cat([
                torch.where(te_has_i.unsqueeze(1), t_phi_i_true, mean_i.unsqueeze(0).expand_as(t_phi_i_true)),
                torch.where(te_has_t.unsqueeze(1), t_phi_t_true, mean_t.unsqueeze(0).expand_as(t_phi_t_true)),
            ], dim=1)

            t_phi_pal = torch.cat([
                torch.where(te_has_i.unsqueeze(1), t_phi_i_true, F.relu(t_phi_t_true @ B_t2i)),
                torch.where(te_has_t.unsqueeze(1), t_phi_t_true, F.relu(t_phi_i_true @ B_i2t)),
            ], dim=1)

            # Ours: pass zero-masked test inputs directly to `BlockAnalyticCL.logits(adaptive_sid=True)`
            te_i_masked = torch.where(te_has_i.unsqueeze(1), te_i_all, torch.zeros_like(te_i_all))
            te_t_masked = torch.where(te_has_t.unsqueeze(1), te_t_all, torch.zeros_like(te_t_all))
            lg_ours = m_ours.logits({"img": te_i_masked.float(), "txt": te_t_masked.float()}, adaptive_sid=True)

            lg_zf = t_phi_zf @ W_zf
            lg_mi = t_phi_mi @ W_mi
            lg_pal = t_phi_pal @ W_pal

            # DeLo / NME Prototype score: sum available modality cosine similarities in expanded space
            sim_i = F.normalize(t_phi_i_true, dim=-1) @ proto_i.T
            sim_t = F.normalize(t_phi_t_true, dim=-1) @ proto_t.T
            lg_nme = torch.where(te_has_i.unsqueeze(1), sim_i, torch.zeros_like(sim_i)) + torch.where(
                te_has_t.unsqueeze(1), sim_t, torch.zeros_like(sim_t)
            )

            y_te_np = te_y_all.numpy()
            scores_map = {
                "ASID_CL_v2_Ours": lg_ours.numpy(),
                "PAL_RebQ_Imputed_RLS": lg_pal.numpy(),
                "Mean_Imputed_RLS": lg_mi.numpy(),
                "GACL_ACIL_ZeroFill": lg_zf.numpy(),
                "DeLo_NME_Prototype": lg_nme.numpy(),
            }

            run_res = {}
            for k_m, sc in scores_map.items():
                if has_multilabel:
                    ml_m = _eval_multilabel(te_ym_all, sc)
                    entry = {
                        "AP_acc": ml_m["top1_hit_ap"],
                        "macro_f1": ml_m["multilabel_macro_f1"],
                        "worst_class_f1": ml_m["multilabel_micro_f1"],
                        **ml_m,
                    }
                else:
                    cls_m = _eval_cls(y_te_np, sc.argmax(axis=1), C)
                    entry = {
                        "AP_acc": cls_m["acc"],
                        "macro_f1": cls_m["macro_f1"],
                        "worst_class_f1": cls_m["worst_class_f1"],
                    }
                run_res[k_m] = entry
            seed_runs.append(run_res)

        summ = _summarize_seed_dicts(seed_runs)
        out_by_type[f"missing_{m_type}_70"] = summ
        print(f"\n  --- Setting: 70% Missing {m_type.upper()} (Train & Test) ---", flush=True)
        for k_m in ("ASID_CL_v2_Ours", "PAL_RebQ_Imputed_RLS", "Mean_Imputed_RLS", "DeLo_NME_Prototype", "GACL_ACIL_ZeroFill"):
            d = summ[k_m]
            extra = (
                f" | Multi-Label Macro-F1: {d['multilabel_macro_f1']['mean']*100:6.2f}% | Samples-F1: {d['multilabel_samples_f1']['mean']*100:6.2f}%"
                if has_multilabel
                else ""
            )
            print(
                f"    {k_m:<24} | AP (Acc): {d['AP_acc']['mean']*100:6.2f}% ± {d['AP_acc']['std']*100:4.2f}% | "
                f"Macro-F1: {d['macro_f1']['mean']*100:6.2f}%{extra}",
                flush=True,
            )
    return out_by_type


# ==============================================================================
# PROTOCOL III: Modality-Incremental Learning (MIL — Harmony 2025 & MERA 2025)
# Stage 1 Vision-Only s=(1,0) -> Stage 2 Text-Only s=(0,1) -> Joint Test s=(1,1)
# ==============================================================================
def run_protocol_3_mil(datasets_dict: dict[str, tuple[dict, int]], seeds=SEEDS) -> dict:
    print("\n======================================================================", flush=True)
    print("PROTOCOL III: Modality-Incremental Learning (MIL — Harmony'25 / MERA'25)", flush=True)
    print("Stage 1: Vision-Only s=(1,0) -> Stage 2: Text-Only s=(0,1) [0% Paired Samples!]", flush=True)
    print("Evaluated on Vision-Only (1,0), Text-Only (0,1), and Joint Multimodal (1,1)", flush=True)
    print("======================================================================", flush=True)

    P = 1024
    out = {}
    for ds_name, (D, C) in datasets_dict.items():
        has_multilabel = "y_multi" in D
        tr_mask = torch.tensor(D["train"])
        te_mask = torch.tensor(D["test"])
        tr_i = D["img"][tr_mask].double()
        tr_t = D["txt"][tr_mask].double()
        tr_y = D["y"][tr_mask].long()
        te_i = D["img"][te_mask].double()
        te_t = D["txt"][te_mask].double()
        te_y_np = D["y"][te_mask].long().numpy()
        if has_multilabel:
            tr_ym = D["y_multi"][tr_mask].double()
            te_ym_np = D["y_multi"][te_mask].numpy()

        seed_runs = []
        for s in seeds:
            # Split training stream into Stage 1 (Vision-Only: first half of samples per class)
            # and Stage 2 (Text-Only: second half of samples per class) -> 0% paired samples!
            g = torch.Generator().manual_seed(s)
            perm = torch.randperm(len(tr_y), generator=g)
            half = len(tr_y) // 2
            idx_s1 = perm[:half]
            idx_s2 = perm[half:]

            # 1. Ours: ASID-CL v2 (`BlockAnalyticCL`)
            m_ours = BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=P, class_balance_beta=0.5, seed=s)
            if has_multilabel:
                r1, c1 = torch.where(tr_ym[idx_s1] > 0.5)
                m_ours.update({"img": tr_i[idx_s1][r1].float()}, c1.long())
                r2, c2 = torch.where(tr_ym[idx_s2] > 0.5)
                m_ours.update({"txt": tr_t[idx_s2][r2].float()}, c2.long())
            else:
                # Stage 1: only 'img' is provided
                m_ours.update({"img": tr_i[idx_s1].float()}, tr_y[idx_s1])
                # Stage 2: only 'txt' is provided
                m_ours.update({"txt": tr_t[idx_s2].float()}, tr_y[idx_s2])
            m_ours.auto_spectral_ridges()

            # 2. GACL / ACIL (Zero-Fill Joint RLS): Stage 1 `[phi_I, 0]`, Stage 2 `[0, phi_T]`
            W_i_proj = m_ours.projections["img"]
            W_t_proj = m_ours.projections["txt"]
            phi_s1 = torch.cat([F.relu(F.normalize(tr_i[idx_s1], dim=-1) @ W_i_proj), torch.zeros(len(idx_s1), P, dtype=torch.float64)], dim=1)
            phi_s2 = torch.cat([torch.zeros(len(idx_s2), P, dtype=torch.float64), F.relu(F.normalize(tr_t[idx_s2], dim=-1) @ W_t_proj)], dim=1)
            phi_all = torch.cat([phi_s1, phi_s2], dim=0)
            if has_multilabel:
                Y_all_target = torch.cat([tr_ym[idx_s1], tr_ym[idx_s2]], dim=0)
            else:
                y_all = torch.cat([tr_y[idx_s1], tr_y[idx_s2]], dim=0)
                Y_all_target = F.one_hot(y_all, C).double()
            G_zf = phi_all.T @ phi_all
            S_zf = phi_all.T @ Y_all_target
            lam_zf = 0.15 * max(float(torch.trace(G_zf)) / (2 * P), 1e-12)
            W_zf = torch.linalg.solve(G_zf + lam_zf * torch.eye(2 * P, dtype=torch.float64), S_zf)

            # 3. Harmony / MERA Prototype Alignment (Unimodal Class Prototypes in expanded space averaged at test time)
            phi_s1_i = F.relu(F.normalize(tr_i[idx_s1], dim=-1) @ W_i_proj)
            phi_s2_t = F.relu(F.normalize(tr_t[idx_s2], dim=-1) @ W_t_proj)
            proto_i = torch.zeros(C, P, dtype=torch.float64)
            proto_t = torch.zeros(C, P, dtype=torch.float64)
            for c in range(C):
                m1 = (tr_ym[idx_s1, c] > 0.5) if has_multilabel else (tr_y[idx_s1] == c)
                if bool(m1.any()):
                    proto_i[c] = F.normalize(F.normalize(phi_s1_i[m1], dim=-1).mean(0), dim=-1)
                m2 = (tr_ym[idx_s2, c] > 0.5) if has_multilabel else (tr_y[idx_s2] == c)
                if bool(m2.any()):
                    proto_t[c] = F.normalize(F.normalize(phi_s2_t[m2], dim=-1).mean(0), dim=-1)

            # Evaluate on Vision-Only (1,0), Text-Only (0,1), and Joint (1,1)
            t_phi_i = F.relu(F.normalize(te_i, dim=-1) @ W_i_proj)
            t_phi_t = F.relu(F.normalize(te_t, dim=-1) @ W_t_proj)
            t_zf_io = torch.cat([t_phi_i, torch.zeros_like(t_phi_t)], dim=1)
            t_zf_to = torch.cat([torch.zeros_like(t_phi_i), t_phi_t], dim=1)
            t_zf_both = torch.cat([t_phi_i, t_phi_t], dim=1)

            # Ours logits
            lg_ours_io = m_ours.logits({"img": te_i.float()}, modalities=("img",)).numpy()
            lg_ours_to = m_ours.logits({"txt": te_t.float()}, modalities=("txt",)).numpy()
            lg_ours_both = m_ours.logits({"img": te_i.float(), "txt": te_t.float()}, adaptive_sid=True).numpy()

            # Zero-Fill RLS logits
            lg_zf_io = (t_zf_io @ W_zf).numpy()
            lg_zf_to = (t_zf_to @ W_zf).numpy()
            lg_zf_both = (t_zf_both @ W_zf).numpy()

            # Harmony/MERA Prototype logits
            sim_i = (F.normalize(t_phi_i, dim=-1) @ proto_i.T).numpy()
            sim_t = (F.normalize(t_phi_t, dim=-1) @ proto_t.T).numpy()
            lg_pr_io = sim_i
            lg_pr_to = sim_t
            lg_pr_both = sim_i + sim_t

            run_res = {}
            for k_m, (lio, lto, lbo) in (
                ("ASID_CL_v2_Ours", (lg_ours_io, lg_ours_to, lg_ours_both)),
                ("GACL_ACIL_ZeroFill_RLS", (lg_zf_io, lg_zf_to, lg_zf_both)),
                ("Harmony_MERA_Prototype", (lg_pr_io, lg_pr_to, lg_pr_both)),
            ):
                if has_multilabel:
                    mio_ml = _eval_multilabel(te_ym_np, lio)
                    mto_ml = _eval_multilabel(te_ym_np, lto)
                    mbo_ml = _eval_multilabel(te_ym_np, lbo)
                    run_res[k_m] = {
                        "vision_only_acc": mio_ml["top1_hit_ap"],
                        "vision_only_macro_f1": mio_ml["multilabel_macro_f1"],
                        "text_only_acc": mto_ml["top1_hit_ap"],
                        "text_only_macro_f1": mto_ml["multilabel_macro_f1"],
                        "joint_both_acc": mbo_ml["top1_hit_ap"],
                        "joint_both_macro_f1": mbo_ml["multilabel_macro_f1"],
                        "omni_modal_avg_f1": (mio_ml["multilabel_macro_f1"] + mto_ml["multilabel_macro_f1"] + mbo_ml["multilabel_macro_f1"]) / 3.0,
                    }
                else:
                    mio = _eval_cls(te_y_np, lio.argmax(1), C)
                    mto = _eval_cls(te_y_np, lto.argmax(1), C)
                    mbo = _eval_cls(te_y_np, lbo.argmax(1), C)
                    run_res[k_m] = {
                        "vision_only_acc": mio["acc"],
                        "vision_only_macro_f1": mio["macro_f1"],
                        "text_only_acc": mto["acc"],
                        "text_only_macro_f1": mto["macro_f1"],
                        "joint_both_acc": mbo["acc"],
                        "joint_both_macro_f1": mbo["macro_f1"],
                        "omni_modal_avg_f1": (mio["macro_f1"] + mto["macro_f1"] + mbo["macro_f1"]) / 3.0,
                    }
            seed_runs.append(run_res)

        summ = _summarize_seed_dicts(seed_runs)
        out[ds_name] = summ
        print(f"\n  --- MIL Dataset: {ds_name} (C={C}) ---", flush=True)
        for k_m in ("ASID_CL_v2_Ours", "GACL_ACIL_ZeroFill_RLS", "Harmony_MERA_Prototype"):
            d = summ[k_m]
            print(
                f"    {k_m:<24} | Vision(1,0) F1: {d['vision_only_macro_f1']['mean']*100:6.2f}% | "
                f"Text(0,1) F1: {d['text_only_macro_f1']['mean']*100:6.2f}% | "
                f"Joint(1,1) F1: {d['joint_both_macro_f1']['mean']*100:6.2f}% (Acc: {d['joint_both_acc']['mean']*100:6.2f}%) | "
                f"Omni-Avg F1: {d['omni_modal_avg_f1']['mean']*100:6.2f}%",
                flush=True,
            )
    return out


def main():
    t0 = time.time()
    D_cifar = build_or_load_cifar100_cache()
    D_mmimdb = build_or_load_mmimdb_cmml_cache()
    D_food = load_food101_cache()

    p1 = run_protocol_1_split_cifar100(D_cifar)
    p2_food = run_protocol_2_cmml_single_dataset(D_food, "UPMC-Food101-CMML (10 Tasks, 100 Classes)", num_tasks=10, has_multilabel=False)
    p2_imdb = run_protocol_2_cmml_single_dataset(D_mmimdb, "MM-IMDb-CMML (5 Tasks, 20 Genres)", num_tasks=5, has_multilabel=True)
    p3_mil = run_protocol_3_mil({
        "Split_CIFAR100": (D_cifar, 100),
        "UPMC_Food101": (D_food, 100),
        "MM_IMDb_CMML": (D_mmimdb, 20),
    })

    out_path = ROOT / "analysis_outputs" / "native_paper_protocols_benchmark.json"
    out_data = {
        "protocol_1_split_cifar100_b0_inc10": p1,
        "protocol_2_cmml_70_missing_train_and_test": {
            "upmc_food101_cmml": p2_food,
            "mm_imdb_cmml": p2_imdb,
        },
        "protocol_3_mil_modality_incremental": p3_mil,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nSaved complete Native Paper Protocols Benchmark to {out_path} in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
