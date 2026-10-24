#!/usr/bin/env python3
"""
Gate H1-G2 on REAL data, CPU-only (exploratory, VALIDATION ONLY; test split dropped).

Sequential LoRA adaptation of CLIP ViT-B/16 (both towers) over the 7 CrisisMMD v2 events,
with the block analytic head accumulating statistics of each event's features computed
right after that event (no replay, statistics never recomputed).

For every attention module, session k trains a fresh LoRA pair (A_k, B_k) on q_proj / v_proj with
    A_k_eff = A_k @ (I - U U^T)
where U spans the top-energy (tau) eigenvectors of the accumulated uncentered Gram of that
module's inputs over all previous sessions (inputs recorded with the model as it was right after
their own session). Then Delta W_k x_old ~ 0, so old features (and hence the stored analytic
statistics) are approximately preserved.

Configs: naive (no projection), null-space tau in {0.95, 0.99}.
Metrics on val: final Macro-F1 with stale statistics (CL-valid), final Macro-F1 with statistics
refit on final-model features (non-CL reference), feature drift of old train samples.
Reference (G1, val): frozen 0.572, joint_oracle 0.619. Gate: CL-valid F1 >= 0.595.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
from run_h1_g1_crisismmd_cpu import NAME, prepare  # noqa: E402
from mm_flex_merge.models import BlockAnalyticCL  # noqa: E402

OUT = ROOT / "analysis_outputs" / "h1_g2_crisismmd"
EPOCHS, BS, LR, R, ALPHA = 2, 32, 1e-4, 16, 32
CONFIGS = [("naive", None), ("nullspace_tau0.95", 0.95), ("nullspace_tau0.99", 0.99)]
torch.set_num_threads(8)
TEXT_MASK = {"mask": None}


class SessionLoRA(nn.Module):
    """q/v projection with frozen past LoRA pairs + one trainable pair whose input side is projected."""

    def __init__(self, base, owner):
        super().__init__()
        self.base, self.owner = base, owner
        self.past = nn.ParameterList()
        self.A = self.B = self.P = None

    def new_session(self):
        if self.A is not None:
            A_eff = (self.A.detach() @ self.P) if self.P is not None else self.A.detach()
            self.past.append(nn.Parameter(A_eff, requires_grad=False))
            self.past.append(nn.Parameter(self.B.detach(), requires_grad=False))
        self.P = None if self.owner.proj is None else self.owner.proj.clone()   # frozen for this session
        d_in, d_out = self.base.in_features, self.base.out_features
        self.A = nn.Parameter(torch.randn(R, d_in) / d_in ** 0.5)
        self.B = nn.Parameter(torch.zeros(d_out, R))

    def forward(self, x):
        out = self.base(x)
        for i in range(0, len(self.past), 2):
            out = out + (x @ self.past[i].T @ self.past[i + 1].T) * (ALPHA / R)
        A = self.A if self.P is None else self.A @ self.P
        return out + (x @ A.T @ self.B.T) * (ALPHA / R)


class AttnState:
    def __init__(self, d, is_text):
        self.gram = torch.zeros(d, d, dtype=torch.float64)
        self.proj, self.is_text, self.recording = None, is_text, False

    def hook(self, module, inputs):
        if not self.recording:
            return
        x = inputs[0].detach()
        if self.is_text and TEXT_MASK["mask"] is not None:
            x = x[TEXT_MASK["mask"].bool()]
        x = x.reshape(-1, x.shape[-1]).double()
        self.gram += x.T @ x

    def update_projection(self, tau):
        if tau is None:
            return
        evals, evecs = torch.linalg.eigh(self.gram)
        evals, evecs = evals.flip(0).clamp(min=0), evecs.flip(1)
        k = int((torch.cumsum(evals, 0) / evals.sum() < tau).sum().item()) + 1
        U = evecs[:, :k].float()
        self.proj = torch.eye(U.shape[0]) - U @ U.T
        self.rank = k


def build(tau):
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(NAME)
    for p in m.parameters():
        p.requires_grad_(False)
    states, loras = [], []
    for name, mod in list(m.named_modules()):
        if isinstance(getattr(mod, "q_proj", None), nn.Linear):
            st = AttnState(mod.q_proj.in_features, name.startswith("text_model"))
            mod.q_proj.register_forward_pre_hook(st.hook)
            for attr in ("q_proj", "v_proj"):
                lo = SessionLoRA(getattr(mod, attr), st)
                setattr(mod, attr, lo); loras.append(lo)
            states.append(st)
    return m, states, loras


def encode(m, pix, ids, att):
    i = m.get_image_features(pixel_values=pix)
    TEXT_MASK["mask"] = att
    t = m.get_text_features(input_ids=ids, attention_mask=att)
    TEXT_MASK["mask"] = None
    return F.normalize(i, dim=-1), F.normalize(t, dim=-1)


@torch.no_grad()
def forward_all(m, idx, pix, ids, att):
    I, T = [], []
    for b in range(0, len(idx), 64):
        s = idx[b:b + 64]
        i, t = encode(m, pix[s], ids[s], att[s]); I.append(i); T.append(t)
    return torch.cat(I), torch.cat(T)


def run_config(tag, tau, f, pix, ids, att, y, order):
    t0 = time.time()
    m, states, loras = build(tau)
    head = nn.Linear(1024, 3)
    ev = f["event_name"].to_numpy(); tr = (f["split"] == "train").to_numpy()
    va_idx = torch.tensor(np.where(~tr)[0])
    heads = {b: BlockAnalyticCL({"img": 512, "txt": 512}, 3, class_balance_beta=b, seed=2026) for b in (0.0, 0.5)}
    stored, ranks = {}, []
    g = torch.Generator().manual_seed(0)
    for k, e in enumerate(order):
        idx = torch.tensor(np.where(tr & (ev == e))[0])
        for lo in loras:
            lo.new_session()
        params = [p for lo in loras for p in (lo.A, lo.B)] + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
        m.train()
        for _ in range(EPOCHS):
            perm = idx[torch.randperm(len(idx), generator=g)]
            for b in range(0, len(perm), BS):
                s = perm[b:b + BS]
                i, t = encode(m, pix[s], ids[s], att[s])
                loss = F.cross_entropy(head(torch.cat([i, t], 1)) * 10, y[s])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        for st in states:
            st.recording = True
        img, txt = forward_all(m, idx, pix, ids, att)      # features for stats + Gram of this session's inputs
        for st in states:
            st.recording = False
            st.update_projection(tau)
        stored[e] = (idx, img, txt)
        for h in heads.values():
            h.update({"img": img, "txt": txt}, y[idx])
        if tau is not None:
            ranks.append(float(np.mean([st.rank / st.gram.shape[0] for st in states])))
        print(f"[{tag}] session {k + 1}/7 {e} n={len(idx)} | mean protected rank frac "
              f"{ranks[-1] if ranks else 0:.3f} | {time.time() - t0:.0f}s")
    vi, vt = forward_all(m, va_idx, pix, ids, att)
    yv = y[va_idx].numpy()
    res = {"stale_f1": {}, "refit_f1": {}}
    for b, h in heads.items():
        res["stale_f1"][str(b)] = f1_score(yv, h.predict({"img": vi, "txt": vt}).numpy(), average="macro")
    all_idx = torch.cat([stored[e][0] for e in order])
    fi, ft = forward_all(m, all_idx, pix, ids, att)
    si = torch.cat([stored[e][1] for e in order]); stt = torch.cat([stored[e][2] for e in order])
    res["drift_img"] = float((fi - si).norm(dim=1).mean()); res["drift_txt"] = float((ft - stt).norm(dim=1).mean())
    for b in (0.0, 0.5):
        h = BlockAnalyticCL({"img": 512, "txt": 512}, 3, class_balance_beta=b, seed=2026)
        h.update({"img": fi, "txt": ft}, y[all_idx])
        res["refit_f1"][str(b)] = f1_score(yv, h.predict({"img": vi, "txt": vt}).numpy(), average="macro")
    res["protected_rank_frac_per_session"] = ranks
    res["seconds"] = time.time() - t0
    print(f"[{tag}] stale F1 {res['stale_f1']} | refit F1 {res['refit_f1']} | drift img {res['drift_img']:.4f} txt {res['drift_txt']:.4f}")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f, pix, ids, att = prepare()
    y = torch.tensor(f["label_index"].to_numpy(np.int64))
    order = list(np.random.default_rng(2026).permutation(sorted(set(f["event_name"]))))
    print("event order:", order)
    path = OUT / "g2_val_results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for tag, tau in CONFIGS:
        if tag in results:
            continue
        results[tag] = run_config(tag, tau, f, pix, ids, att, y, order)
        path.write_text(json.dumps({**results}, indent=2))
    best = {k: max(v["stale_f1"].values()) for k, v in results.items()}
    print("\nCL-valid final val Macro-F1 (best beta):", {k: round(v, 4) for k, v in best.items()})
    print("reference: frozen 0.572 | joint_oracle 0.619 | gate >= 0.595")


if __name__ == "__main__":
    main()
