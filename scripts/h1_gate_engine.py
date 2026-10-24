"""
Self-contained engine for gates H1-G1 / H1-G2 (used locally and embedded verbatim in the Colab notebook).

G1: adaptation gain of LoRA (both CLIP towers) vs frozen, analytic head (BlockAnalyticCL).
G2: sequential LoRA over tasks; each session's LoRA input side is projected onto the null space of
    the accumulated input Gram of previous sessions, analytic statistics are never recomputed.
All evaluation is on the validation split; the caller must drop the test split beforehand.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from mm_flex_merge.models import TEXT_MASK, AttnState, BlockAnalyticCL, SessionLoRA

NAME = "openai/clip-vit-base-patch16"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
R, ALPHA = 16, 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    AMP = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
else:
    AMP = None


LOG_PATH = None    # set by the notebook to a file inside the exported results folder


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ----------------------------------------------------------------------------- data
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


def decode_images(paths, workers=0, bs=256, mmap_path=None):
    import tempfile
    from pathlib import Path
    w_eff = min(max(int(workers or 0), 0), 2)
    dl = torch.utils.data.DataLoader(_ImageSet(paths), batch_size=bs, num_workers=w_eff)
    n = len(paths)
    if mmap_path is not None or n > 1000:
        p = Path(mmap_path) if mmap_path else Path(tempfile.gettempdir()) / f"pix_mmap_{n}.bin"
        mm = np.memmap(str(p), dtype=np.uint8, mode="w+", shape=(n, 3, 224, 224))
        out = torch.from_numpy(mm)
    else:
        out = torch.empty(n, 3, 224, 224, dtype=torch.uint8)
    k = 0
    for b in dl:
        out[k:k + len(b)] = b
        k += len(b)
    return out


def tokenize(texts, bs=4096):
    from transformers import CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(NAME)
    ids_l, att_l = [], []
    for i in range(0, len(texts), bs):
        t = tok([str(x) for x in texts[i:i + bs]], padding="max_length", truncation=True, max_length=77, return_tensors="pt")
        ids_l.append(t["input_ids"].to(torch.int32))
        att_l.append(t["attention_mask"].to(torch.int8))
    return torch.cat(ids_l, 0), torch.cat(att_l, 0)


def make_data(frame, *, path_col, text_col, label_col, task_col, task_order, num_classes, workers=8):
    """frame may contain train/val or train/val/test rows (column 'split')."""
    assert set(frame["split"]) <= {"train", "val", "test"}, "unexpected split names"
    log(f"decoding {len(frame)} images ...")
    pix = decode_images(frame[path_col].tolist(), workers=workers)
    ids, att = tokenize(frame[text_col].tolist())
    splits = frame["split"].to_numpy()
    return {"pix": pix, "ids": ids, "att": att,
            "y": torch.tensor(frame[label_col].to_numpy(np.int64)),
            "train": splits == "train",
            "val": splits == "val",
            "test": splits == "test",
            "task": frame[task_col].to_numpy(), "order": list(task_order), "C": int(num_classes)}


# ---------------------------------------------------------------------------- model
def build(lora=True, towers=("vision", "text")):
    from transformers import CLIPModel
    m = CLIPModel.from_pretrained(NAME, low_cpu_mem_usage=True).to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    states, loras = [], []
    if lora:
        for name, mod in list(m.named_modules()):
            if isinstance(getattr(mod, "q_proj", None), nn.Linear):
                is_text = name.startswith("text_model")
                tower_name = "text" if is_text else "vision"
                if tower_name not in towers:
                    continue
                st = AttnState(mod.q_proj.in_features, is_text, device=DEV)
                mod.q_proj.register_forward_pre_hook(st.hook)
                for attr in ("q_proj", "v_proj"):
                    lo = SessionLoRA(getattr(mod, attr), st, r=R, alpha=ALPHA)
                    setattr(mod, attr, lo)
                    loras.append(lo)
                states.append(st)
    return m.eval(), states, loras


def encode(m, D, idx):
    pix = ((D["pix"][idx].to(DEV).float() / 255.0) - MEAN.to(DEV)) / STD.to(DEV)
    ids, att = D["ids"][idx].to(DEV, dtype=torch.long), D["att"][idx].to(DEV, dtype=torch.long)
    with torch.autocast("cuda", dtype=AMP, enabled=AMP is not None):
        vo = m.vision_model(pixel_values=pix)
        i = m.visual_projection(vo[1] if isinstance(vo, tuple) else vo.pooler_output)
        TEXT_MASK["mask"] = att
        to = m.text_model(input_ids=ids, attention_mask=att)
        t = m.text_projection(to[1] if isinstance(to, tuple) else to.pooler_output)
        TEXT_MASK["mask"] = None
    return F.normalize(i.float(), dim=-1), F.normalize(t.float(), dim=-1)


def train_session(m, loras, head, D, idx, *, epochs, bs, lr, seed=0):
    for lo in loras:
        lo.new_session()
    params = [p for lo in loras for p in (lo.A, lo.B)] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    use_scaler = (DEV == "cuda" and AMP == torch.float16)
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=use_scaler)
        if hasattr(torch, "amp")
        else torch.cuda.amp.GradScaler(init_scale=1024.0, enabled=use_scaler)
    )
    g = torch.Generator().manual_seed(seed)
    m.train()
    for ep in range(epochs):
        perm = idx[torch.randperm(len(idx), generator=g)]
        tot = 0.0
        for b in range(0, len(perm), bs):
            s = perm[b:b + bs]
            i, t = encode(m, D, s)
            with torch.autocast("cuda", dtype=AMP, enabled=AMP is not None):
                loss = F.cross_entropy(head(torch.cat([i, t], 1)) * 10, D["y"][s].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach().item()) * len(s)
        log(f"   epoch {ep}: loss {tot / len(perm):.4f}")
    m.eval()
    if DEV == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def extract(m, D, idx, bs=256):
    I, T = [], []
    for b in range(0, len(idx), bs):
        i, t = encode(m, D, idx[b:b + bs])
        I.append(i.cpu())
        T.append(t.cpu())
    return torch.cat(I), torch.cat(T)


# ----------------------------------------------------------------------- evaluation
PRED = {}          # per-sample validation predictions, exported to predictions.npz for offline analysis
HEAD_SEEDS = (2026, 2027, 2028)
BETAS = (0.0, 0.5)


def scores(y_true, y_pred):
    per_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    return {"macro_f1": float(np.mean(per_cls)),
            "worst_class_f1": float(np.min(per_cls)),
            "acc": float(accuracy_score(y_true, y_pred))}


def _heads(C, beta):
    return [BlockAnalyticCL({"img": 512, "txt": 512}, C, projection_dim=1024, class_balance_beta=beta, seed=s)
            for s in HEAD_SEEDS]


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


def run_g1(D, *, seeds, epochs, bs, lr, stage="g1"):
    tr, va = _idx(D["train"]), _idx(D["val"])
    has_te = bool(D.get("test") is not None and np.any(D["test"]))
    te = _idx(D["test"]) if has_te else None
    first = _idx(D["train"] & (D["task"] == D["order"][0]))
    res = {"frozen": {}, "first_session": {}, "joint_oracle": {}}
    for arm, sub in (("frozen", None), ("first_session", first), ("joint_oracle", tr)):
        for seed in ([seeds[0]] if sub is None else seeds):
            t0 = time.time()
            torch.manual_seed(seed)
            m, _, loras = build(lora=sub is not None)
            if sub is not None:
                log(f"[{stage}/{arm}/seed{seed}] LoRA on {len(sub)} samples")
                train_session(m, loras, nn.Linear(1024, D["C"]).to(DEV), D, sub, epochs=epochs, bs=bs, lr=lr, seed=seed)
            f_tr, f_va = extract(m, D, tr), extract(m, D, va)
            f_te = extract(m, D, te) if has_te else None
            y_te = D["y"][te] if has_te else None
            res[arm][f"seed{seed}"] = analytic_fit_eval(
                f_tr, D["y"][tr], f_va, D["y"][va], D["C"], f"{stage}/{arm}/seed{seed}", f_te, y_te
            )
            res[arm][f"seed{seed}"]["seconds"] = time.time() - t0
            log(f"[{stage}/{arm}/seed{seed}] {res[arm][f'seed{seed}']}")
            del m
            if DEV == "cuda":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    return res


def run_g2(D, *, taus, seeds, epochs, bs, lr, stage="g2", extra_tower_configs=(("vision_only_tau0.95", 0.95, ("vision",)),)):
    va = _idx(D["val"])
    has_te = bool(D.get("test") is not None and np.any(D["test"]))
    te = _idx(D["test"]) if has_te else None
    yv, tv = D["y"][va].numpy(), D["task"][va.numpy()]
    yte = D["y"][te].numpy() if has_te else None
    res = {}
    configs = [(("naive" if tau is None else f"nullspace_tau{tau}"), tau, ("vision", "text")) for tau in taus]
    if extra_tower_configs:
        configs.extend(extra_tower_configs)
    for tag, tau, towers in configs:
        res[tag] = {}
        for seed in seeds:
            t0 = time.time()
            torch.manual_seed(seed)
            m, states, loras = build(lora=True, towers=towers)
            head = nn.Linear(1024, D["C"]).to(DEV)
            heads = {b: _heads(D["C"], b) for b in BETAS}
            trans_heads = _heads(D["C"], 0.5)
            stored, ranks, seen = [], [], torch.zeros(D["C"], dtype=torch.bool)
            acc_matrix = {f"beta_{b}": [] for b in BETAS}      # row k: acc on val of task j<=k after session k
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
                    acc_matrix[f"beta_{b}"].append([float((pred[tv == t] == yv[tv == t]).mean()) for t in D["order"][:k + 1]])
                ranks.append(float(np.mean([st.rank / st.gram.shape[0] for st in states])) if tau else 0.0)
                log(f"[{stage}/{tag}/seed{seed}] session {k + 1}/{len(D['order'])} n={len(idx)} "
                    f"protected-rank {ranks[-1]:.3f} acc(b=0) {np.mean(acc_matrix['beta_0.0'][-1]):.4f} ({time.time() - t0:.0f}s)")
            r = {"stale": {}, "transport_sid": {}, "acc_matrix": acc_matrix, "protected_rank_frac": ranks, "towers": list(towers)}
            ti, tt = extract(m, D, te) if has_te else (None, None)
            for b, hs in heads.items():
                pred = _logits(hs, (vi, vt)).argmax(1).numpy()
                PRED[f"{stage}/{tag}/seed{seed}/stale/val/beta{b}"] = pred.astype(np.int16)
                r["stale"][f"beta_{b}"] = scores(yv, pred)
                if has_te:
                    pred_te = _logits(hs, (ti, tt)).argmax(1).numpy()
                    PRED[f"{stage}/{tag}/seed{seed}/stale/test/beta{b}"] = pred_te.astype(np.int16)
                    r["stale"][f"test_beta_{b}"] = scores(yte, pred_te)
            for h in trans_heads:
                h.auto_spectral_ridges()
            pred_tr_val = _logits(trans_heads, (vi, vt), adaptive_sid=True).argmax(1).numpy()
            r["transport_sid"]["val"] = scores(yv, pred_tr_val)
            if has_te:
                pred_tr_te = _logits(trans_heads, (ti, tt), adaptive_sid=True).argmax(1).numpy()
                r["transport_sid"]["test"] = scores(yte, pred_tr_te)
            all_idx = torch.cat([s[0] for s in stored])
            fi, ft = extract(m, D, all_idx)
            si, st_ = torch.cat([s[1] for s in stored]), torch.cat([s[2] for s in stored])
            di, dt = (fi - si).norm(dim=1), (ft - st_).norm(dim=1)
            r["drift_img"], r["drift_txt"] = float(di.mean()), float(dt.mean())
            bounds = np.cumsum([0] + [len(s[0]) for s in stored])
            r["drift_by_session"] = [{"img": float(di[a:b].mean()), "txt": float(dt[a:b].mean())}
                                     for a, b in zip(bounds[:-1], bounds[1:])]
            r["refit"] = analytic_fit_eval((fi, ft), D["y"][all_idx], (vi, vt), D["y"][va], D["C"],
                                           f"{stage}/{tag}/seed{seed}/refit", (ti, tt) if has_te else None, D["y"][te] if has_te else None)
            r["seconds"] = time.time() - t0
            log(f"[{stage}/{tag}/seed{seed}] stale {r['stale']} | refit {r['refit']} | "
                f"drift img {r['drift_img']:.4f} txt {r['drift_txt']:.4f}")
            res[tag][f"seed{seed}"] = r
            del m
            if DEV == "cuda":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    return res


def _mean_over_seeds(per_seed, metric, field=None, prefix="beta"):
    """Best beta (by seed-mean) of the seed-mean metric."""
    rows = [v[field] if field else v for v in per_seed.values()]
    return max(float(np.mean([r[f"{prefix}_{b}"][metric] for r in rows if f"{prefix}_{b}" in r])) for b in BETAS)


def verdicts(g1, g2, metric):
    fz, orc = _mean_over_seeds(g1["frozen"], metric), _mean_over_seeds(g1["joint_oracle"], metric)
    out = {"metric": metric, "frozen": fz, "joint_oracle": orc,
           "first_session": _mean_over_seeds(g1["first_session"], metric),
           "G1_oracle_gain_pts": 100 * (orc - fz), "G1": "PASS" if 100 * (orc - fz) >= 2 else "FAIL"}
    if g2:
        target = fz + 0.5 * (orc - fz)
        stale = {k: _mean_over_seeds(v, metric, "stale") for k, v in g2.items()}
        trans_sid = {k: float(np.mean([s["transport_sid"]["val"][metric] for s in v.values()]))
                     for k, v in g2.items() if "transport_sid" in next(iter(v.values()))}
        cand = [v for k, v in stale.items() if k != "naive"] + [v for k, v in trans_sid.items() if k != "naive"]
        out.update({"G2_target": target, "G2_stale": stale, "G2_transport_sid": trans_sid,
                    "G2": "PASS" if max(cand) >= target else "FAIL"})
        first_seed = next(iter(next(iter(g2.values())).values()))
        if "test_beta_0.5" in first_seed.get("stale", {}):
            out["G3_test_frozen"] = _mean_over_seeds(g1["frozen"], metric, prefix="test_beta")
            out["G3_test_first_session"] = _mean_over_seeds(g1["first_session"], metric, prefix="test_beta")
            out["G3_test_joint_oracle"] = _mean_over_seeds(g1["joint_oracle"], metric, prefix="test_beta")
            out["G3_test_stale"] = {k: _mean_over_seeds(v, metric, "stale", prefix="test_beta") for k, v in g2.items()}
            out["G3_test_transport_sid"] = {
                k: float(np.mean([s["transport_sid"]["test"][metric] for s in v.values()]))
                for k, v in g2.items() if "test" in next(iter(v.values())).get("transport_sid", {})
            }
    return out


def export(out_dir, name, D, payload):
    """results JSON + per-sample validation/test predictions (for paired bootstrap offline)."""
    import json
    import platform
    out_dir.mkdir(parents=True, exist_ok=True)
    va = np.where(D["val"])[0]
    te = np.where(D["test"])[0] if D.get("test") is not None else np.array([], dtype=int)
    meta = {"torch": torch.__version__, "python": platform.python_version(), "device": DEV,
            "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", "amp": str(AMP),
            "n_train": int(D["train"].sum()), "n_val": int(len(va)), "n_test": int(len(te)), "num_classes": D["C"],
            "task_order": [str(t) for t in D["order"]], "head_seeds": list(HEAD_SEEDS), "betas": list(BETAS),
            "lora": {"r": R, "alpha": ALPHA, "targets": ["q_proj", "v_proj"], "towers": ["vision", "text"]}}
    (out_dir / f"{name}_results.json").write_text(json.dumps({"meta": meta, **payload}, indent=2))
    keep = {k.replace("/", "__"): v for k, v in PRED.items() if k.split("/")[0] in payload["stages"]}
    extra = {}
    if len(te) > 0:
        extra["y_test"] = D["y"][te].numpy().astype(np.int16)
        extra["task_test"] = np.array([str(t) for t in D["task"][te]])
    np.savez_compressed(out_dir / f"{name}_predictions.npz", y_val=D["y"][va].numpy().astype(np.int16),
                        task_val=np.array([str(t) for t in D["task"][va]]), **extra, **keep)
