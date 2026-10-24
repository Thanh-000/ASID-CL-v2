#!/usr/bin/env python3
"""
Gate H1-G1 on REAL data, CPU-only (exploratory, VALIDATION ONLY).

CrisisMMD v2 damage (3 classes), leak-safe tweet-grouped split (regroup_crisismmd_by_tweet).
The test split is dropped before anything else and never touched.

Arms (CLIP ViT-B/16, LoRA r=16 on q_proj/v_proj of BOTH towers, implemented manually):
  frozen        : no adaptation
  first_session : LoRA on the first event (order of seed 2026) only, then frozen (CL-valid)
  joint_oracle  : LoRA on all train data (NOT CL-valid; ceiling of adaptation gain)
Head: BlockAnalyticCL (block random projection + relative ridge), domain-incremental over 7 events.
Gate: joint_oracle - frozen >= 2 Macro-F1 points on val, else H1 is pointless.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations, regroup_crisismmd_by_tweet  # noqa: E402
from mm_flex_merge.models import BlockAnalyticCL  # noqa: E402

NAME = "openai/clip-vit-base-patch16"
OUT = ROOT / "analysis_outputs" / "h1_g1_crisismmd"
SEEDS = [2026, 2027, 2028, 2029, 2030]
EPOCHS, BS, LR = 2, 32, 1e-4
torch.set_num_threads(8)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def load_clip(lora):
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(NAME)
    for p in m.parameters():
        p.requires_grad_(False)
    if lora:
        for name, mod in list(m.named_modules()):
            for attr in ("q_proj", "v_proj"):
                if isinstance(getattr(mod, attr, None), nn.Linear):
                    setattr(mod, attr, LoRALinear(getattr(mod, attr)))
    return m


def prepare():
    ds = regroup_crisismmd_by_tweet(load_crisismmd_v2_annotations(
        ROOT / "external_data" / "crisismmd_v2", task="damage", agreed_labels=False,
        media_root=ROOT / "external_data" / "crisismmd_v2" / "media", require_images=False))
    f = ds.frame
    f = f[f["split"] != "test"].reset_index(drop=True)          # test never touched
    from transformers import CLIPProcessor
    proc = CLIPProcessor.from_pretrained(NAME)
    ims = []
    for p in f["image_abspath"]:
        try:
            ims.append(Image.open(p).convert("RGB"))
        except Exception:
            ims.append(Image.new("RGB", (224, 224)))
    pix = torch.cat([proc(images=ims[i:i + 256], return_tensors="pt")["pixel_values"] for i in range(0, len(ims), 256)])
    tok = proc.tokenizer([str(t) for t in f["tweet_text"]], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
    return f, pix, tok["input_ids"], tok["attention_mask"]


def encode(m, pix, ids, att):
    i = m.get_image_features(pixel_values=pix)
    t = m.get_text_features(input_ids=ids, attention_mask=att)
    return F.normalize(i, dim=-1), F.normalize(t, dim=-1)


def finetune(idx, pix, ids, att, y, tag):
    m = load_clip(lora=True)
    head = nn.Linear(1024, 3)
    params = [p for p in m.parameters() if p.requires_grad] + list(head.parameters())
    print(f"[{tag}] trainable params {sum(p.numel() for p in params):,} | samples {len(idx)}")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    for ep in range(EPOCHS):
        m.train(); t0 = time.time(); tot = 0.0
        perm = idx[torch.randperm(len(idx), generator=g)]
        for b in range(0, len(perm), BS):
            s = perm[b:b + BS]
            i, t = encode(m, pix[s], ids[s], att[s])
            loss = F.cross_entropy(head(torch.cat([i, t], 1)) * 10, y[s])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(s)
        print(f"[{tag}] epoch {ep}: loss {tot / len(perm):.4f} ({time.time() - t0:.0f}s)")
    return m.eval()


@torch.no_grad()
def extract(m, pix, ids, att):
    I, T = [], []
    for b in range(0, len(pix), 64):
        i, t = encode(m, pix[b:b + 64], ids[b:b + 64], att[b:b + 64])
        I.append(i); T.append(t)
    return torch.cat(I), torch.cat(T)


def evaluate(img, txt, f, beta):
    y = torch.tensor(f["label_index"].to_numpy(np.int64))
    ev = f["event_name"].to_numpy(); tr = (f["split"] == "train").to_numpy(); va = ~tr
    events = sorted(set(ev))
    out = {}
    for seed in SEEDS:
        order = list(np.random.default_rng(seed).permutation(events))
        m = BlockAnalyticCL({"img": 512, "txt": 512}, 3, projection_dim=1024, class_balance_beta=beta, seed=seed)
        steps = []
        for e in order:
            s = torch.tensor(np.where(tr & (ev == e))[0])
            m.update({"img": img[s], "txt": txt[s]}, y[s])
            pv = m.predict({"img": img[va], "txt": txt[va]}).numpy()
            steps.append(f1_score(y[va].numpy(), pv, average="macro"))
        out[seed] = {"final_f1": steps[-1], "avg_f1": float(np.mean(steps)),
                     "final_acc": accuracy_score(y[va].numpy(), pv)}
    return {k: float(np.mean([r[k] for r in out.values()])) for k in ("final_f1", "avg_f1", "final_acc")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f, pix, ids, att = prepare()
    y = torch.tensor(f["label_index"].to_numpy(np.int64))
    tr = torch.tensor((f["split"] == "train").to_numpy())
    print(f"data: train {int(tr.sum())} | val {int((~tr).sum())} | prep {time.time() - t0:.0f}s")
    first_event = list(np.random.default_rng(2026).permutation(sorted(set(f["event_name"]))))[0]
    arms = {
        "frozen": lambda: load_clip(lora=False).eval(),
        "first_session": lambda: finetune(torch.where(tr & torch.tensor((f["event_name"] == first_event).to_numpy()))[0], pix, ids, att, y, "first_session"),
        "joint_oracle": lambda: finetune(torch.where(tr)[0], pix, ids, att, y, "joint_oracle"),
    }
    results = {}
    for arm, build in arms.items():
        npz = OUT / f"features_{arm}.npz"
        if npz.exists():
            z = np.load(npz); img, txt = torch.from_numpy(z["img"]), torch.from_numpy(z["txt"])
        else:
            img, txt = extract(build(), pix, ids, att)
            np.savez_compressed(npz, img=img.numpy(), txt=txt.numpy())
        results[arm] = {f"beta_{b}": evaluate(img, txt, f, b) for b in (0.0, 0.5)}
        print(f"{arm:<14} " + " | ".join(f"β={b}: F1 {r['final_f1']:.4f} avg {r['avg_f1']:.4f} acc {r['final_acc']:.4f}"
                                         for b, r in zip((0.0, 0.5), results[arm].values())) + f"  ({time.time() - t0:.0f}s)")
    best = lambda a: max(r["final_f1"] for r in results[a].values())
    gain = (best("joint_oracle") - best("frozen")) * 100
    fs_gain = (best("first_session") - best("frozen")) * 100
    verdict = "PASS" if gain >= 2 else "FAIL"
    print(f"\nH1-G1 (val): oracle gain {gain:+.2f} F1 pts, first-session gain {fs_gain:+.2f} -> {verdict} (threshold 2)")
    json.dump({"status": "exploratory, val only, test dropped", "first_event": first_event, "epochs": EPOCHS,
               "results": results, "oracle_gain_pts": gain, "first_session_gain_pts": fs_gain, "verdict": verdict},
              open(OUT / "g1_val_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
