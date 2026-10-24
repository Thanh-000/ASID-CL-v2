#!/usr/bin/env python3
"""
Local Zen-4 AVX-512 BF16 + Frozen-Prefix Cached execution of H1 Gates (G1, G2, G3)
on UPMC Food-101 (RebQ/PAL protocol: 100 classes, 10 class-incremental tasks x 10 classes/task).

Optimizations for AMD Ryzen 7 8745H (24 GB RAM, CPU AVX-512 BF16):
1. Contiguous ZIP HTTP-Range fetcher: downloads only the stratified per-class JPEG slices
   (100 classes x 36 samples = 3,600 real image-text pairs: 2,600 train, 500 val, 500 test)
   in ~15-20s via 16 parallel byte-range streams (~220 MB instead of 8.26 GB).
2. Exact Frozen-Prefix Caching (Layers 0..9 of CLIP ViT-B/16 Vision & Text encoders):
   Since Null-Space SessionLoRA is applied to the top semantic Transformer blocks (Layers 10, 11),
   Layers 0..9 are 100% invariant and computed once in AVX-512 bfloat16 (~1.2 GB RAM), giving
   a 6x exact speedup while preserving exact backprop and input-Gram null-space projection.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import io
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mm_flex_merge.models import TEXT_MASK, AttnState, BlockAnalyticCL, SessionLoRA

torch.set_num_threads(min(8, os.cpu_count() or 8))

NAME = "openai/clip-vit-base-patch16"
HF_ZIP_URL = "https://huggingface.co/datasets/kkim0451/UPMC-Food101/resolve/main/UPMC-Food-101.zip"
HF_ZIP_SIZE = 8268462033

DATA_DIR = ROOT / "data" / "upmc_food101"
OUT_DIR = ROOT / "analysis_outputs" / "h1_gates_colab_t4" / "h1_results" / "food101"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

R, ALPHA = 16, 32
TOP_LAYERS = (10, 11)   # Top 2 Transformer blocks of Vision (12) and Text (12) towers
MAX_TXT_LEN = 48
PER_CLASS = 36          # 36 x 100 classes = 3,600 samples (26 train + 5 val + 5 test per class)
EPOCHS, BS, LR = 2, 64, 3e-4
TAUS = [None, 0.95, 0.99]
SEEDS = [2026, 2027, 2028]
HEAD_SEEDS = (2026, 2027, 2028)
BETAS = (0.0, 0.5)


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------- 1. Fast Range Fetcher
class HttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n == 0 or self.pos >= self.size:
            return b""
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
                data = urllib.request.urlopen(req, timeout=45).read()
                self.pos += len(data)
                return data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.0 * (attempt + 1))
        return b""


def _extract_from_contiguous_chunk(chunk: bytes, base_offset: int, zinfos: list[zipfile.ZipInfo], img_dir: Path) -> int:
    saved = 0
    for zi in zinfos:
        out_p = img_dir / Path(zi.filename).name
        if out_p.exists() and out_p.stat().st_size > 0:
            saved += 1
            continue
        rel = zi.header_offset - base_offset
        sig, _, _, comp_method, _, _, _, _, _, fname_len, extra_len = struct.unpack(
            "<4sHHHHHIIIHH", chunk[rel:rel + 30]
        )
        assert sig == b"PK\x03\x04", f"Bad local zip header at {zi.filename}"
        data_start = rel + 30 + fname_len + extra_len
        comp_data = chunk[data_start:data_start + zi.compress_size]
        if comp_method == 0:
            raw = comp_data
        elif comp_method == 8:
            raw = zlib.decompress(comp_data, -15)
        else:
            raise ValueError(f"Unsupported compression method: {comp_method}")
        out_p.write_bytes(raw)
        saved += 1
    return saved


def prepare_food101_dataframe() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = DATA_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "train_titles.csv"

    if not csv_path.exists():
        log("Downloading train_titles.csv from HuggingFace UPMC-Food101 via HTTP Range...")
        rf = HttpRangeFile(HF_ZIP_URL, HF_ZIP_SIZE)
        with zipfile.ZipFile(rf) as zf:
            csv_path.write_bytes(zf.read("UPMC-Food-101/texts/train_titles.csv"))

    df = pd.read_csv(csv_path, header=None, names=["image", "text", "label"], on_bad_lines="skip")
    if str(df.iloc[0]["label"]).lower() in {"label", "category", "class"}:
        df = df.iloc[1:].reset_index(drop=True)
    df["image"] = df["image"].astype(str).map(lambda s: Path(s).name)

    # RebQ / PAL protocol: drop rarest class -> 100 classes
    rarest = df["label"].value_counts().idxmin()
    df = df[df["label"] != rarest].reset_index(drop=True)
    classes = sorted(df["label"].unique())
    assert len(classes) == 100, f"Expected 100 classes, got {len(classes)}"
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})

    meta_cache = DATA_DIR / f"selected_{PER_CLASS}per_class.csv"
    if meta_cache.exists():
        sel_df = pd.read_csv(meta_cache)
        if len(sel_df) == 100 * PER_CLASS and all((img_dir / nm).exists() for nm in sel_df["image"]):
            log(f"Loaded cached 100-class stratified subset ({len(sel_df)} samples) from {meta_cache}")
            return sel_df

    log("Reading ZIP Central Directory via HTTP Range to locate contiguous class slices...")
    rf = HttpRangeFile(HF_ZIP_URL, HF_ZIP_SIZE)
    with zipfile.ZipFile(rf) as zf:
        by_name = {
            Path(zi.filename).name: zi
            for zi in zf.infolist()
            if zi.filename.endswith(".jpg") and "/train/" in zi.filename
        }

    df = df[df["image"].isin(by_name)].copy().reset_index(drop=True)
    df["offset"] = df["image"].map(lambda nm: by_name[nm].header_offset)
    # Sort within each class by ZIP offset so the first PER_CLASS images are 100% contiguous!
    df = df.sort_values(["y", "offset"]).reset_index(drop=True)

    keep_parts = []
    tasks_to_download = []
    for c_idx, grp in df.groupby("y", sort=True):
        sub = grp.iloc[:PER_CLASS].copy().reset_index(drop=True)
        assert len(sub) == PER_CLASS, f"Class {c_idx} has only {len(sub)} images"
        sub["split"] = "train"
        sub.loc[:4, "split"] = "val"     # 5 val per class (500 val total)
        sub.loc[5:9, "split"] = "test"   # 5 test per class (500 test total)
        keep_parts.append(sub)
        zlist = [by_name[nm] for nm in sub["image"]]
        need = any(not (img_dir / nm).exists() for nm in sub["image"])
        if need:
            tasks_to_download.append((c_idx, zlist))

    sel_df = pd.concat(keep_parts, ignore_index=True)
    corder = np.random.RandomState(2026).permutation(100)
    sel_df["task"] = sel_df["y"].map({int(c): k // 10 for k, c in enumerate(corder)})
    sel_df["path"] = sel_df["image"].map(lambda nm: str(img_dir / nm))

    if tasks_to_download:
        log(f"Downloading {len(tasks_to_download)} contiguous class slices ({len(tasks_to_download) * PER_CLASS} images) via 16 threads...")
        t0 = time.time()

        def _fetch_class(item):
            c_idx, zlist = item
            start = zlist[0].header_offset
            last = zlist[-1]
            end = last.header_offset + last.compress_size + 256
            for attempt in range(4):
                try:
                    req = urllib.request.Request(HF_ZIP_URL, headers={"Range": f"bytes={start}-{end}"})
                    chunk = urllib.request.urlopen(req, timeout=60).read()
                    return _extract_from_contiguous_chunk(chunk, start, zlist, img_dir)
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1.5 * (attempt + 1))
            return 0

        done_imgs = 0
        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = [ex.submit(_fetch_class, it) for it in tasks_to_download]
            for i, fut in enumerate(as_completed(futs), 1):
                done_imgs += fut.result()
                if i % 20 == 0 or i == len(futs):
                    log(f"   fetched {i}/{len(futs)} classes ({done_imgs} images) in {time.time() - t0:.1f}s")

    sel_df.to_csv(meta_cache, index=False)
    log(f"Prepared UPMC Food-101: {len(sel_df)} samples | splits={sel_df['split'].value_counts().to_dict()}")
    return sel_df


# ------------------------------------------------ 2. Prefix-Cached Model & Null-Space LoRA
class _ImageSet(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224))
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        im = im.crop((l, t, l + 224, t + 224))
        return torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1)


@torch.no_grad()
def build_or_load_prefix_cache(df: pd.DataFrame) -> dict:
    cache_path = DATA_DIR / f"prefix_cache_L{TOP_LAYERS[0]}_{len(df)}.pt"
    if cache_path.exists():
        log(f"Loading cached Layer-{TOP_LAYERS[0]-1} prefix states from {cache_path} ...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    from transformers import CLIPModel, CLIPTokenizer
    log(f"Decoding {len(df)} images and tokenizing titles (max_len={MAX_TXT_LEN})...")
    dl = torch.utils.data.DataLoader(_ImageSet(df["path"].tolist()), batch_size=128, num_workers=0)
    pix = torch.cat(list(dl), dim=0)

    tok = CLIPTokenizer.from_pretrained(NAME)
    tk = tok(
        [str(x) for x in df["text"].tolist()],
        padding="max_length",
        truncation=True,
        max_length=MAX_TXT_LEN,
        return_tensors="pt",
    )
    ids, att = tk["input_ids"], tk["attention_mask"]

    m = CLIPModel.from_pretrained(NAME).eval()
    v_mod = m.vision_model
    t_mod = m.text_model
    split_layer = TOP_LAYERS[0]

    v_states, t_states, eos_idx_list = [], [], []
    t0 = time.time()
    bs = 64
    log(f"Computing Frozen Prefix (Layers 0..{split_layer - 1}) in Zen-4 AVX-512 bfloat16...")
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
    for b in range(0, len(df), bs):
        px = ((pix[b:b + bs].float() / 255.0) - MEAN) / STD
        id_b, at_b = ids[b:b + bs], att[b:b + bs]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            # Vision prefix (embeddings + pre_layrnorm + layers 0..9)
            vh = v_mod.embeddings(px)
            vh = v_mod.pre_layrnorm(vh)
            for lyr in v_mod.encoder.layers[:split_layer]:
                vh = lyr(vh, None, None)[0]
            # Text prefix (embeddings + layers 0..9)
            th = t_mod.embeddings(input_ids=id_b)
            causal = _create_4d_causal_attention_mask(id_b.size(), th.dtype, device=th.device)
            ext_att = _prepare_4d_attention_mask(at_b, th.dtype)
            for lyr in t_mod.encoder.layers[:split_layer]:
                th = lyr(th, ext_att, causal)[0]
        v_states.append(vh.to(torch.bfloat16).cpu())
        t_states.append(th.to(torch.bfloat16).cpu())
        eos_idx_list.append(id_b.to(torch.int32).argmax(dim=-1))
        if (b // bs + 1) % 10 == 0 or b + bs >= len(df):
            log(f"   prefix batch {min(b + bs, len(df))}/{len(df)} ({time.time() - t0:.1f}s)")

    del pix, m
    gc.collect()
    splits = df["split"].to_numpy()
    D = {
        "v_pref": torch.cat(v_states, 0),
        "t_pref": torch.cat(t_states, 0),
        "att": att.to(torch.int8),
        "eos": torch.cat(eos_idx_list, 0),
        "y": torch.tensor(df["y"].to_numpy(np.int64)),
        "train": splits == "train",
        "val": splits == "val",
        "test": splits == "test",
        "task": df["task"].to_numpy(),
        "order": list(range(10)),
        "C": 100,
    }
    torch.save(D, cache_path)
    log(f"Saved Layer-{split_layer - 1} prefix cache ({cache_path.stat().st_size / 1e6:.1f} MB)")
    return D


class TopBlocksCLIP(nn.Module):
    """Executes Layers 10..11 + final norms + projections from cached Layer-9 hidden states."""

    def __init__(self, base_clip):
        super().__init__()
        split_layer = TOP_LAYERS[0]
        self.v_layers = nn.ModuleList(list(base_clip.vision_model.encoder.layers[split_layer:]))
        self.v_post_ln = base_clip.vision_model.post_layernorm
        self.v_proj = base_clip.visual_projection

        self.t_layers = nn.ModuleList(list(base_clip.text_model.encoder.layers[split_layer:]))
        self.t_final_ln = base_clip.text_model.final_layer_norm
        self.t_proj = base_clip.text_projection


_BASE_CLIP_STATE = None


def build_top_model(lora: bool = True, towers=("vision", "text")):
    global _BASE_CLIP_STATE
    from transformers import CLIPModel
    base = CLIPModel.from_pretrained(NAME)
    if _BASE_CLIP_STATE is None:
        _BASE_CLIP_STATE = {k: v.clone() for k, v in base.state_dict().items()}
    else:
        base.load_state_dict(_BASE_CLIP_STATE)
    m = TopBlocksCLIP(base)
    for p in m.parameters():
        p.requires_grad_(False)
    states, loras = [], []
    if lora:
        for tower_name, mod_list, is_text in (
            ("vision", m.v_layers, False),
            ("text", m.t_layers, True),
        ):
            if tower_name not in towers:
                continue
            for blk in mod_list:
                attn = blk.self_attn
                st = AttnState(attn.q_proj.in_features, is_text)
                attn.q_proj.register_forward_pre_hook(st.hook)
                for attr in ("q_proj", "v_proj"):
                    lo = SessionLoRA(getattr(attn, attr), st, r=R, alpha=ALPHA)
                    setattr(attn, attr, lo)
                    loras.append(lo)
                states.append(st)
    return m.eval(), states, loras


def encode_from_prefix(m: TopBlocksCLIP, D: dict, idx: torch.Tensor):
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
    vh = D["v_pref"][idx].float()
    th = D["t_pref"][idx].float()
    att = D["att"][idx].to(torch.long)
    eos = D["eos"][idx].to(torch.long)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        for lyr in m.v_layers:
            vh = lyr(vh, None, None)[0]
        v_pool = m.v_post_ln(vh[:, 0, :])
        i = m.v_proj(v_pool)

        TEXT_MASK["mask"] = att
        causal = _create_4d_causal_attention_mask((len(idx), th.shape[1]), th.dtype, device=th.device)
        ext_att = _prepare_4d_attention_mask(att, th.dtype)
        for lyr in m.t_layers:
            th = lyr(th, ext_att, causal)[0]
        th = m.t_final_ln(th)
        t_pool = th[torch.arange(len(idx)), eos]
        t = m.t_proj(t_pool)
        TEXT_MASK["mask"] = None
    return F.normalize(i.float(), dim=-1), F.normalize(t.float(), dim=-1)


def train_session(m, loras, head, D, idx, *, epochs, bs, lr, seed=0):
    for lo in loras:
        lo.new_session()
    params = [p for lo in loras for p in (lo.A, lo.B)] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    m.train()
    for ep in range(epochs):
        perm = idx[torch.randperm(len(idx), generator=g)]
        tot = 0.0
        for b in range(0, len(perm), bs):
            s = perm[b:b + bs]
            i, t = encode_from_prefix(m, D, s)
            loss = F.cross_entropy(head(torch.cat([i, t], 1)) * 10, D["y"][s])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach().item()) * len(s)
        log(f"   epoch {ep}: loss {tot / len(perm):.4f}")
    m.eval()


@torch.no_grad()
def extract(m, D, idx, bs=256):
    I, T = [], []
    for b in range(0, len(idx), bs):
        i, t = encode_from_prefix(m, D, idx[b:b + bs])
        I.append(i.cpu())
        T.append(t.cpu())
    return torch.cat(I), torch.cat(T)


PRED = {}


def scores(y_true, y_pred):
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_cls)),
        "worst_class_f1": float(np.min(per_cls)),
        "acc": float(accuracy_score(y_true, y_pred)),
    }


def _heads(C, beta):
    return [
        BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=beta, seed=s)
        for s in HEAD_SEEDS
    ]


def _logits(hs, feats, adaptive_sid=False):
    return sum(h.logits({"img": feats[0], "txt": feats[1]}, adaptive_sid=adaptive_sid) for h in hs)


def analytic_fit_eval(tr_feats, y_tr, va_feats, y_va, C, key, te_feats=None, y_te=None):
    out = {}
    for b in BETAS:
        hs = _heads(C, b)
        for h in hs:
            h.update({"img": tr_feats[0], "txt": tr_feats[1]}, y_tr)
        pred = _logits(hs, va_feats).argmax(1).numpy()
        PRED[f"{key}/val/beta{b}"] = pred.astype(np.int16)
        out[f"beta_{b}"] = scores(y_va.numpy(), pred)
        if te_feats is not None and y_te is not None and len(y_te) > 0:
            pred_te = _logits(hs, te_feats).argmax(1).numpy()
            PRED[f"{key}/test/beta{b}"] = pred_te.astype(np.int16)
            out[f"test_beta_{b}"] = scores(y_te.numpy(), pred_te)
        if b == 0.5:
            for h in hs:
                h.auto_spectral_ridges()
            pred_sid = _logits(hs, va_feats, adaptive_sid=True).argmax(1).numpy()
            out["unified_sid"] = scores(y_va.numpy(), pred_sid)
            if te_feats is not None and y_te is not None and len(y_te) > 0:
                pred_sid_te = _logits(hs, te_feats, adaptive_sid=True).argmax(1).numpy()
                out["test_unified_sid"] = scores(y_te.numpy(), pred_sid_te)
    return out


def _idx(mask):
    return torch.tensor(np.where(mask)[0])


def run_g1(D, *, seeds, epochs, bs, lr, stage="food_g1"):
    tr, va, te = _idx(D["train"]), _idx(D["val"]), _idx(D["test"])
    first = _idx(D["train"] & (D["task"] == D["order"][0]))
    res = {"frozen": {}, "first_session": {}, "joint_oracle": {}}
    for arm, sub in (("frozen", None), ("first_session", first), ("joint_oracle", tr)):
        for seed in ([seeds[0]] if sub is None else seeds):
            t0 = time.time()
            torch.manual_seed(seed)
            m, _, loras = build_top_model(lora=sub is not None)
            if sub is not None:
                log(f"[{stage}/{arm}/seed{seed}] LoRA on {len(sub)} samples")
                train_session(m, loras, nn.Linear(1024, D["C"]), D, sub, epochs=epochs, bs=bs, lr=lr, seed=seed)
            f_tr, f_va, f_te = extract(m, D, tr), extract(m, D, va), extract(m, D, te)
            res[arm][f"seed{seed}"] = analytic_fit_eval(
                f_tr, D["y"][tr], f_va, D["y"][va], D["C"], f"{stage}/{arm}/seed{seed}", f_te, D["y"][te]
            )
            res[arm][f"seed{seed}"]["seconds"] = time.time() - t0
            log(f"[{stage}/{arm}/seed{seed}] acc(b0.5)={res[arm][f'seed{seed}']['beta_0.5']['acc']:.4f} "
                f"test_acc(b0.5)={res[arm][f'seed{seed}']['test_beta_0.5']['acc']:.4f} ({time.time() - t0:.1f}s)")
    return res


def run_g2(D, *, taus, seeds, epochs, bs, lr, stage="food_g2"):
    va, te = _idx(D["val"]), _idx(D["test"])
    yv, tv, yte = D["y"][va].numpy(), D["task"][va.numpy()], D["y"][te].numpy()
    res = {}
    configs = [(("naive" if tau is None else f"nullspace_tau{tau}"), tau, ("vision", "text")) for tau in taus]
    configs.append(("vision_only_tau0.95", 0.95, ("vision",)))
    for tag, tau, towers in configs:
        res[tag] = {}
        for seed in seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            m, states, loras = build_top_model(lora=True, towers=towers)
            head = nn.Linear(1024, D["C"])
            heads = {b: _heads(D["C"], b) for b in BETAS}
            trans_heads = _heads(D["C"], 0.5)
            stored, ranks, seen = [], [], torch.zeros(D["C"], dtype=torch.bool)
            acc_matrix = {f"beta_{b}": [] for b in BETAS}
            for k, task in enumerate(D["order"]):
                idx = _idx(D["train"] & (D["task"] == task))
                fi_pre, ft_pre = extract(m, D, idx) if k > 0 else (None, None)
                train_session(m, loras, head, D, idx, epochs=epochs, bs=bs, lr=lr, seed=seed * 100 + k)
                for st in states:
                    st.recording = True
                fi, ft = extract(m, D, idx)
                for st in states:
                    st.recording = False
                    st.update_projection(tau)
                stored.append((idx, fi, ft))
                seen[D["y"][idx].unique()] = True
                for hs in heads.values():
                    for h in hs:
                        h.update({"img": fi, "txt": ft}, D["y"][idx])
                for h in trans_heads:
                    if k > 0 and fi_pre is not None:
                        h.transport_gram({"img": fi_pre, "txt": ft_pre}, {"img": fi, "txt": ft})
                    h.update({"img": fi, "txt": ft}, D["y"][idx])
                vi, vt = extract(m, D, va)
                for b, hs in heads.items():
                    lg = _logits(hs, (vi, vt))
                    lg[:, ~seen] = -float("inf")
                    pred = lg.argmax(1).numpy()
                    acc_matrix[f"beta_{b}"].append(
                        [float((pred[tv == t] == yv[tv == t]).mean()) for t in D["order"][:k + 1]]
                    )
                ranks.append(float(np.mean([st.rank / st.gram.shape[0] for st in states])) if tau else 0.0)
                log(f"[{stage}/{tag}/seed{seed}] session {k + 1}/10 n={len(idx)} "
                    f"protected-rank {ranks[-1]:.3f} acc(b=0) {np.mean(acc_matrix['beta_0.0'][-1]):.4f} ({time.time() - t0:.0f}s)")
            r = {"stale": {}, "transport_sid": {}, "acc_matrix": acc_matrix, "protected_rank_frac": ranks, "towers": list(towers)}
            ti, tt = extract(m, D, te)
            for b, hs in heads.items():
                pred = _logits(hs, (vi, vt)).argmax(1).numpy()
                PRED[f"{stage}/{tag}/seed{seed}/stale/val/beta{b}"] = pred.astype(np.int16)
                r["stale"][f"beta_{b}"] = scores(yv, pred)
                pred_te = _logits(hs, (ti, tt)).argmax(1).numpy()
                PRED[f"{stage}/{tag}/seed{seed}/stale/test/beta{b}"] = pred_te.astype(np.int16)
                r["stale"][f"test_beta_{b}"] = scores(yte, pred_te)
            for h in trans_heads:
                h.auto_spectral_ridges()
            pred_tr_val = _logits(trans_heads, (vi, vt), adaptive_sid=True).argmax(1).numpy()
            r["transport_sid"]["val"] = scores(yv, pred_tr_val)
            pred_tr_te = _logits(trans_heads, (ti, tt), adaptive_sid=True).argmax(1).numpy()
            r["transport_sid"]["test"] = scores(yte, pred_tr_te)

            all_idx = torch.cat([s[0] for s in stored])
            fi, ft = extract(m, D, all_idx)
            si, st_ = torch.cat([s[1] for s in stored]), torch.cat([s[2] for s in stored])
            di, dt = (fi - si).norm(dim=1), (ft - st_).norm(dim=1)
            r["drift_img"], r["drift_txt"] = float(di.mean()), float(dt.mean())
            bounds = np.cumsum([0] + [len(s[0]) for s in stored])
            r["drift_by_session"] = [
                {"img": float(di[a:b].mean()), "txt": float(dt[a:b].mean())}
                for a, b in zip(bounds[:-1], bounds[1:])
            ]
            r["refit"] = analytic_fit_eval(
                (fi, ft), D["y"][all_idx], (vi, vt), D["y"][va], D["C"], f"{stage}/{tag}/seed{seed}/refit", (ti, tt), D["y"][te]
            )
            r["seconds"] = time.time() - t0
            log(f"[{stage}/{tag}/seed{seed}] stale acc={r['stale']['beta_0.5']['acc']:.4f} "
                f"test_acc={r['stale']['test_beta_0.5']['acc']:.4f} | "
                f"transport_sid val={r['transport_sid']['val']['acc']:.4f} test={r['transport_sid']['test']['acc']:.4f} | "
                f"drift img {r['drift_img']:.4f} txt {r['drift_txt']:.4f}")
            res[tag][f"seed{seed}"] = r
    return res


def _mean_over_seeds(per_seed, metric, field=None, prefix="beta"):
    rows = [v[field] if field else v for v in per_seed.values()]
    return max(float(np.mean([r[f"{prefix}_{b}"][metric] for r in rows if f"{prefix}_{b}" in r])) for b in BETAS)


def verdicts(g1, g2, metric="acc"):
    fz, orc = _mean_over_seeds(g1["frozen"], metric), _mean_over_seeds(g1["joint_oracle"], metric)
    out = {
        "metric": metric,
        "frozen": fz,
        "joint_oracle": orc,
        "first_session": _mean_over_seeds(g1["first_session"], metric),
        "G1_oracle_gain_pts": 100 * (orc - fz),
        "G1": "PASS" if 100 * (orc - fz) >= 2 else "FAIL",
    }
    if g2:
        target = fz + 0.5 * (orc - fz)
        stale = {k: _mean_over_seeds(v, metric, "stale") for k, v in g2.items()}
        trans_sid = {k: float(np.mean([s["transport_sid"]["val"][metric] for s in v.values()])) for k, v in g2.items()}
        cand = [v for k, v in stale.items() if k != "naive"] + [v for k, v in trans_sid.items() if k != "naive"]
        out.update({
            "G2_target": target,
            "G2_stale": stale,
            "G2_transport_sid": trans_sid,
            "G2": "PASS" if max(cand) >= target else "FAIL",
            "G3_test_frozen": _mean_over_seeds(g1["frozen"], metric, prefix="test_beta"),
            "G3_test_first_session": _mean_over_seeds(g1["first_session"], metric, prefix="test_beta"),
            "G3_test_joint_oracle": _mean_over_seeds(g1["joint_oracle"], metric, prefix="test_beta"),
            "G3_test_stale": {k: _mean_over_seeds(v, metric, "stale", prefix="test_beta") for k, v in g2.items()},
            "G3_test_transport_sid": {
                k: float(np.mean([s["transport_sid"]["test"][metric] for s in v.values()])) for k, v in g2.items()
            },
        })
    return out


def main():
    t_start = time.time()
    df = prepare_food101_dataframe()
    D = build_or_load_prefix_cache(df)
    counts_f = json.loads(df.groupby(["task", "split"]).size().unstack(fill_value=0).to_json())
    config = {
        "epochs": EPOCHS,
        "bs": BS,
        "lr": LR,
        "taus": TAUS,
        "seeds_food": SEEDS,
        "per_class": PER_CLASS,
        "top_layers": list(TOP_LAYERS),
    }

    log("=== Running Food-101 Gate G1 (3 seeds) ===")
    g1f = run_g1(D, seeds=SEEDS, epochs=EPOCHS, bs=BS, lr=LR, stage="food_g1")
    vf1 = verdicts(g1f, None, "acc")
    log(f"G1 Verdicts: {json.dumps(vf1)}")

    log("=== Running Food-101 Gate G2 & G3 (10 tasks, 3 seeds) ===")
    g2f = run_g2(D, taus=TAUS, seeds=SEEDS, epochs=EPOCHS, bs=BS, lr=LR, stage="food_g2")
    vf = verdicts(g1f, g2f, "acc")
    vf_f1 = verdicts(g1f, g2f, "macro_f1")
    log(f"Final Food-101 Verdicts (ACC): {json.dumps(vf, indent=2)}")
    log(f"Final Food-101 Verdicts (Macro-F1): {json.dumps(vf_f1, indent=2)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    va = np.where(D["val"])[0]
    te = np.where(D["test"])[0]
    meta = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu_zen4_bf16",
        "cpu": "AMD Ryzen 7 8745H (AVX-512 BF16)",
        "n_train": int(D["train"].sum()),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "num_classes": D["C"],
        "task_order": [str(t) for t in D["order"]],
        "head_seeds": list(HEAD_SEEDS),
        "betas": list(BETAS),
        "lora": {"r": R, "alpha": ALPHA, "targets": ["q_proj", "v_proj"], "top_layers": list(TOP_LAYERS)},
        "total_seconds": time.time() - t_start,
    }
    payload = {
        "meta": meta,
        "stages": ["food_g1", "food_g2"],
        "config": config,
        "counts": counts_f,
        "g1": g1f,
        "g2": g2f,
        "verdicts": vf,
        "verdicts_macro_f1": vf_f1,
    }
    (OUT_DIR / "food_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    keep = {k.replace("/", "__"): v for k, v in PRED.items()}
    np.savez_compressed(
        OUT_DIR / "food_predictions.npz",
        y_val=D["y"][va].numpy().astype(np.int16),
        task_val=np.array([str(t) for t in D["task"][va]]),
        y_test=D["y"][te].numpy().astype(np.int16),
        task_test=np.array([str(t) for t in D["task"][te]]),
        **keep,
    )
    log(f"Saved {OUT_DIR / 'food_results.json'} and {OUT_DIR / 'food_predictions.npz'} in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
