#!/usr/bin/env python3
"""
Build the Colab notebook for gate H1-G1 (VALIDATION ONLY).

Question: on UPMC Food-101 (RebQ/PAL protocol: 100 classes, 10 steps), does adapting the
CLIP towers with LoRA add >= 2 points over the frozen backbone when the head is the
block analytic learner? If even the oracle (joint LoRA on all train classes) adds < 2
points, H1 (null-space adaptation) is pointless and we fall back to H2.

Arms (features exported for train/val only; the test split is never read):
  frozen        : CLIP ViT-B/16, no adaptation
  first_session : LoRA on task-1 classes only, then frozen (APER-style, CL-valid)
  joint_oracle  : LoRA on all train classes (NOT CL-valid, ceiling of adaptation gain)

Output: /content/h1_g1/features_<arm>.npz (img, txt, y, split) + inline analytic eval.
Run locally afterwards with mm_flex_merge.models.BlockAnalyticCL for the final numbers.
"""

import json
from pathlib import Path

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4", "provenance": []},
                   "language_info": {"name": "python"}},
      "cells": []}


def md(s):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s.strip() + "\n"})


def code(s):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": s.strip() + "\n"})


md(r"""
# Gate H1-G1: adaptation gain on UPMC Food-101 (VAL ONLY)
- Protocol: RebQ/PAL. 100 classes (drop the rarest class), 10 steps × 10 classes.
- **Never reads the test split.** Val = 10% of train, stratified, seed 0.
- Arms: `frozen`, `first_session` (LoRA on task 1 only), `joint_oracle` (LoRA on all classes, ceiling only).
- Gate: if `joint_oracle − frozen` < 2 points of final accuracy, H1 is dropped.
""")

code(r"""
!nvidia-smi
!pip -q install peft transformers kaggle scikit-learn
""")

code(r"""
# Kaggle credentials: upload kaggle.json when prompted (or set KAGGLE_USERNAME/KAGGLE_KEY)
import os, pathlib
if not (pathlib.Path.home() / ".kaggle/kaggle.json").exists() and "KAGGLE_KEY" not in os.environ:
    from google.colab import files
    up = files.upload()
    pathlib.Path.home().joinpath(".kaggle").mkdir(exist_ok=True)
    pathlib.Path.home().joinpath(".kaggle/kaggle.json").write_bytes(up["kaggle.json"])
    os.chmod(pathlib.Path.home() / ".kaggle/kaggle.json", 0o600)
if not pathlib.Path("/content/food101").exists():
    !kaggle datasets download -d gianmarco96/upmcfood101 -p /content/food101 --unzip -q
    !ls /content/food101
""")

code(r"""
# Load train titles robustly (original file has no header: image_name, text, label)
import pandas as pd, numpy as np, glob
from pathlib import Path
ROOT = Path("/content/food101")
csv = next(iter(glob.glob(str(ROOT / "**/train_titles.csv"), recursive=True)))
df = pd.read_csv(csv, header=None, names=["image", "text", "label"], on_bad_lines="skip")
if str(df.iloc[0]["label"]).lower() in {"label", "category"}:
    df = df.iloc[1:].reset_index(drop=True)
img_index = {Path(p).name: p for p in glob.glob(str(ROOT / "**/train/**/*.jpg"), recursive=True)}
df["path"] = df["image"].map(lambda n: img_index.get(Path(str(n)).name))
print("rows", len(df), "| missing images", df["path"].isna().sum())
df = df.dropna(subset=["path"]).reset_index(drop=True)
counts = df["label"].value_counts()
df = df[df["label"] != counts.idxmin()].reset_index(drop=True)          # RebQ: drop rarest class
classes = sorted(df["label"].unique()); assert len(classes) == 100, len(classes)
df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
rng = np.random.RandomState(0)
df["split"] = "train"
for c, idx in df.groupby("y").groups.items():
    idx = rng.permutation(np.array(idx)); df.loc[idx[: max(1, len(idx) // 10)], "split"] = "val"
order = np.random.RandomState(2026).permutation(100)                     # seed-0 class order
task1 = set(order[:10].tolist())
print(df["split"].value_counts().to_dict(), "| task-1 classes:", sorted(task1))
""")

code(r"""
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
DEV = "cuda"; NAME = "openai/clip-vit-base-patch16"
proc = CLIPProcessor.from_pretrained(NAME)

class DS(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d.reset_index(drop=True)
    def __len__(self): return len(self.d)
    def __getitem__(self, i):
        r = self.d.iloc[i]
        try: im = Image.open(r["path"]).convert("RGB")
        except Exception: im = Image.new("RGB", (224, 224))
        return im, str(r["text"])[:300], int(r["y"])

def collate(b):
    ims, txts, ys = zip(*b)
    x = proc(images=list(ims), text=list(txts), return_tensors="pt", padding=True, truncation=True, max_length=77)
    return x, torch.tensor(ys)

def encode(model, x):
    x = {k: v.to(DEV) for k, v in x.items()}
    i = model.get_image_features(pixel_values=x["pixel_values"])
    t = model.get_text_features(input_ids=x["input_ids"], attention_mask=x["attention_mask"])
    return F.normalize(i.float(), dim=-1), F.normalize(t.float(), dim=-1)

def lora_finetune(sub, epochs=2, lr=1e-4, bs=64):
    m = CLIPModel.from_pretrained(NAME)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"])
    m = get_peft_model(m, cfg).to(DEV); m.print_trainable_parameters()
    head = nn.Linear(1024, 100).to(DEV)
    params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    dl = torch.utils.data.DataLoader(DS(sub), batch_size=bs, shuffle=True, num_workers=4, collate_fn=collate)
    for ep in range(epochs):
        m.train(); tot = 0
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                i, t = encode(m, x)
                loss = F.cross_entropy(head(torch.cat([i, t], 1)) * 10, y.to(DEV))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        print(f"epoch {ep}: loss {tot / len(dl):.4f}")
    return m.eval()

@torch.no_grad()
def export(model, arm):
    d = df.reset_index(drop=True)
    dl = torch.utils.data.DataLoader(DS(d), batch_size=256, num_workers=4, collate_fn=collate)
    I, T = [], []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            i, t = encode(model, x)
        I.append(i.cpu()); T.append(t.cpu())
    out = Path("/content/h1_g1"); out.mkdir(exist_ok=True)
    np.savez_compressed(out / f"features_{arm}.npz", img=torch.cat(I).numpy(), txt=torch.cat(T).numpy(),
                        y=d["y"].values, split=d["split"].values, order=order)
    print("saved", arm)
""")

code(r"""
export(CLIPModel.from_pretrained(NAME).to(DEV).eval(), "frozen")
train = df[df["split"] == "train"]
export(lora_finetune(train[train["y"].isin(task1)]), "first_session")
export(lora_finetune(train), "joint_oracle")
""")

code(r"""
# Inline analytic eval (block random projection + relative ridge, same as BlockAnalyticCL)
def analytic_eval(arm, P=2048, ridge=0.1, seed=2026):
    z = np.load(f"/content/h1_g1/features_{arm}.npz", allow_pickle=True)
    g = torch.Generator().manual_seed(seed)
    W = {m: torch.randn(512, P, generator=g, dtype=torch.float64, device=DEV) for m in ("img", "txt")}
    phi = lambda m, idx: F.relu(torch.tensor(z[m][idx], dtype=torch.float64, device=DEV) @ W[m])
    y = torch.tensor(z["y"].astype(int), device=DEV); tr = z["split"] == "train"; va = ~tr
    feats = lambda idx: torch.cat([phi("img", idx), phi("txt", idx)], 1)
    G = torch.zeros(2 * P, 2 * P, dtype=torch.float64, device=DEV); S = torch.zeros(2 * P, 100, dtype=torch.float64, device=DEV)
    Xv, yv = feats(va), y[torch.tensor(va, device=DEV)]
    accs = []
    for k in range(10):
        cls = torch.tensor(order[: 10 * (k + 1)], device=DEV)
        idx = tr & np.isin(z["y"], order[10 * k: 10 * (k + 1)])
        X = feats(idx); Y = F.one_hot(y[torch.tensor(idx, device=DEV)], 100).double()
        G += X.T @ X; S += X.T @ Y
        lam = ridge * torch.trace(G) / G.shape[0]
        Wout = torch.linalg.solve(G + lam * torch.eye(2 * P, dtype=G.dtype, device=DEV), S)
        mask = torch.isin(yv, cls)
        logits = Xv[mask] @ Wout
        unseen = torch.ones(100, dtype=torch.bool, device=DEV); unseen[cls] = False
        logits[:, unseen] = -float("inf")
        accs.append((logits.argmax(1) == yv[mask]).float().mean().item())
    return {"final_acc": accs[-1], "avg_acc": float(np.mean(accs))}

res = {arm: analytic_eval(arm) for arm in ["frozen", "first_session", "joint_oracle"]}
for a, r in res.items(): print(f"{a:<14} final {r['final_acc']*100:.2f} | avg {r['avg_acc']*100:.2f}")
gain = (res["joint_oracle"]["final_acc"] - res["frozen"]["final_acc"]) * 100
print(f"\nH1-G1 oracle adaptation gain = {gain:+.2f} points -> {'PASS' if gain >= 2 else 'FAIL'} (threshold 2)")
json.dump(res, open("/content/h1_g1/g1_val_results.json", "w"), indent=2)
""")

code(r"""
!cd /content && zip -q -r h1_g1.zip h1_g1
from google.colab import files; files.download("/content/h1_g1.zip")
""")

out = Path(__file__).resolve().parent.parent / "notebooks" / "H1_G1_adaptation_gain_food101.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(nb['cells'])} cells)")
