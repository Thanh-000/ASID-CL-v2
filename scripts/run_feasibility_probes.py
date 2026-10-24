#!/usr/bin/env python3
"""
Feasibility probes for choosing the next research direction (exploratory, NOT confirmatory).

P1 (UK-CrisisMMD, class-incremental, VAL ONLY): can a better task router close the
    oracle–router gap? Compare task-centroid router (current) vs class-prototype router
    (task score = max cosine to the class means owned by the task).
P2 (CrisisMMD v2 'damage', domain-incremental over 7 events): does the E0 ranking
    (prototype methods > router-ensemble > merging) hold in a different CL regime?
    Seeded per-event 60/20/20 split, event order shuffled per seed, val-only selection.
    Caveat: random within-event split (no near-duplicate grouping) -> treat as a probe.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402
from experiments.mm_hope.continual_baselines import RanPACClassifier, RanPACConfig  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402
from mm_flex_merge.merging import OPCMMerger  # noqa: E402

SEEDS = [2026, 2027, 2028, 2029, 2030]
V2_EVENTS = ["hurricane_harvey", "hurricane_maria", "hurricane_irma", "california_wildfires",
             "mexico_earthquake", "iraq_iran_earthquake", "srilanka_floods"]


def fused(img, txt):
    return F.normalize(torch.cat([F.normalize(txt, dim=-1), F.normalize(img, dim=-1)], dim=-1), dim=-1)


def pooled(task_data, split):
    keys = ("img", "txt", "y")
    out = [torch.cat([td[f"{split}_{k}"] for td in task_data]) for k in keys]
    tid = torch.cat([torch.full((len(td[f"{split}_y"]),), i) for i, td in enumerate(task_data)])
    return (*out, tid)


def macro_f1(y, p, num_classes):
    return float(f1_score(y, p, labels=list(range(num_classes)), average="macro", zero_division=0))


def ensemble_with_routes(experts, route_w, img, txt):
    with torch.no_grad():
        logits = torch.stack([e(img, txt) for e in experts], dim=1)
        return (route_w.unsqueeze(-1) * logits).sum(1).argmax(-1).numpy()


# ---------------------------------------------------------------------------
# P1: router probe on UK val
# ---------------------------------------------------------------------------

def class_prototype_routes(task_data, img, txt, temperature):
    feats = fused(img, txt)
    scores = []
    for td in task_data:
        tr = fused(td["train_img"], td["train_txt"])
        protos = torch.stack([F.normalize(tr[td["train_y"] == c].mean(0), dim=-1) for c in td["train_y"].unique()])
        scores.append((feats @ protos.T).max(dim=1).values)
    return F.softmax(torch.stack(scores, dim=1) / temperature, dim=-1)


def probe_router(protocol):
    rows = []
    for order in ["O1", "O2", "O3", "O4", "O5", "O6"]:
        task_data = uk.build_task_data(protocol, order)
        for seed in SEEDS:
            torch.manual_seed(seed)
            experts = [uk.train_expert(td["train_img"], td["train_txt"], td["train_y"], num_classes=6, mode="gated", seed=seed) for td in task_data]
            v_img, v_txt, v_y, v_tid = pooled(task_data, "val")
            yv = v_y.numpy()
            res = {}
            # current router (bias = 0, the val-preferred value)
            _, rw = uk.ensemble_predict(experts, uk.build_router(task_data, 0.0), v_img, v_txt)
            res["task_centroid"] = (macro_f1(yv, ensemble_with_routes(experts, rw, v_img, v_txt), 6), float((rw.argmax(1) == v_tid).float().mean()))
            for temp in (0.01, 0.05):
                rw = class_prototype_routes(task_data, v_img, v_txt, temp)
                res[f"class_proto_T{temp}"] = (macro_f1(yv, ensemble_with_routes(experts, rw, v_img, v_txt), 6), float((rw.argmax(1) == v_tid).float().mean()))
            with torch.no_grad():
                oracle = torch.cat([experts[i](td["val_img"], td["val_txt"]).argmax(-1) for i, td in enumerate(task_data)]).numpy()
            res["oracle"] = (macro_f1(yv, oracle, 6), 1.0)
            ncm_pred = None
            rows.append(res)
    summary = {k: {"val_macro_f1": float(np.mean([r[k][0] for r in rows])), "val_routing_acc": float(np.mean([r[k][1] for r in rows]))} for k in rows[0]}
    return summary


# ---------------------------------------------------------------------------
# P2: CrisisMMD v2 domain-incremental
# ---------------------------------------------------------------------------

def load_v2():
    from experiments.mm_hope.crisismmd_v2 import load_crisismmd_v2_annotations, regroup_crisismmd_by_tweet
    data = np.load(ROOT / "analysis_outputs" / "mm_hope_phase1_cache" / "clip_embeddings_c75d70edce7d3e8e.npz")
    ds = regroup_crisismmd_by_tweet(load_crisismmd_v2_annotations(
        ROOT / "external_data" / "crisismmd_v2", task="damage", agreed_labels=False,
        media_root=ROOT / "external_data" / "crisismmd_v2" / "media", require_images=False))
    frame = ds.frame
    rid2i = {r: i for i, r in enumerate(data["row_ids"])}
    keep = [rid in rid2i for rid in frame["row_id"]]
    frame = frame[keep].reset_index(drop=True)
    idx = [rid2i[r] for r in frame["row_id"]]
    img = torch.from_numpy(data["image_embeddings"][idx]).float()
    txt = torch.from_numpy(data["text_embeddings"][idx]).float()
    y = torch.from_numpy(frame["label_index"].to_numpy(dtype=np.int64))
    return img, txt, y, frame["event_name"].to_numpy(), list(ds.labels)


def v2_task_data(img, txt, y, events, order, seed):
    rng = np.random.default_rng(seed)
    task_data = []
    for ev in order:
        idx = np.where(events == ev)[0]
        rng.shuffle(idx)
        n = len(idx)
        cut1, cut2 = int(0.6 * n), int(0.8 * n)
        td = {}
        for split, sl in (("train", idx[:cut1]), ("val", idx[cut1:cut2]), ("test", idx[cut2:])):
            s = torch.from_numpy(sl)
            td[f"{split}_img"], td[f"{split}_txt"], td[f"{split}_y"] = img[s], txt[s], y[s]
        task_data.append(td)
    return task_data


def probe_v2():
    img, txt, y, events, labels = load_v2()
    C = len(labels)
    print(f"   v2 damage: {len(y)} mẫu, {C} lớp {labels}, phân bố {np.bincount(y.numpy(), minlength=C).tolist()}")
    names = ["NCM", "RanPAC", "Task Arithmetic", "OPCM", "MM-FLEX router-ensemble", "Oracle event-ID experts", "Joint head (ref)"]
    scores = {n: [] for n in names}
    routing = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        order = list(np.random.default_rng(seed).permutation(V2_EVENTS))
        task_data = v2_task_data(img, txt, y, events, order, seed)
        v_img, v_txt, v_y, _ = pooled(task_data, "val")
        t_img, t_txt, t_y, t_tid = pooled(task_data, "test")
        yv, yt = v_y.numpy(), t_y.numpy()
        vf, tf = fused(v_img, v_txt), fused(t_img, t_txt)

        def vf1(p):
            return macro_f1(yv, p, C)

        # NCM
        sums, cnt = torch.zeros(C, 1024), torch.zeros(C)
        for td in task_data:
            f = fused(td["train_img"], td["train_txt"])
            sums.index_add_(0, td["train_y"], f)
            cnt += torch.bincount(td["train_y"], minlength=C).float()
        means = F.normalize(sums / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        scores["NCM"].append(macro_f1(yt, (tf @ means.T).argmax(-1).numpy(), C))

        # RanPAC
        def rp(beta):
            m = RanPACClassifier(RanPACConfig(feature_dim=1024, num_classes=C, projection_dim=1024, ridge=1e-1,
                                              relative_ridge=True, solve_dtype="float64", solve_device="cpu",
                                              class_balance_beta=beta, seed=seed))
            for td in task_data:
                m.update(fused(td["train_img"], td["train_txt"]), td["train_y"])
            return m
        models = {b: rp(b) for b in (0.0, 0.25, 0.5, 0.75, 1.0)}
        b = max(models, key=lambda k: vf1(models[k].predict(vf).numpy()))
        scores["RanPAC"].append(macro_f1(yt, models[b].predict(tf).numpy(), C))

        # experts
        base_m = uk.MultimodalHead(512, 256, num_classes=C, mode="gated")
        base_w, base_sd = uk.extract_flat(base_m), {k: v.detach().clone() for k, v in base_m.state_dict().items()}
        experts = [uk.train_expert(td["train_img"], td["train_txt"], td["train_y"], num_classes=C, mode="gated", seed=seed) for td in task_data]
        deltas = [uk.extract_flat(e) - base_w for e in experts]
        m_eval = uk.MultimodalHead(512, 256, num_classes=C, mode="gated")

        def hp(model, a, t):
            with torch.no_grad():
                return model(a, t).argmax(-1).numpy()

        def flat(w):
            uk.load_flat(m_eval, w)
            return m_eval

        lam = max(uk.TA_LAMBDA_GRID, key=lambda l: vf1(hp(flat(base_w + l * sum(deltas)), v_img, v_txt)))
        scores["Task Arithmetic"].append(macro_f1(yt, hp(flat(base_w + lam * sum(deltas)), t_img, t_txt), C))

        sds = [e.state_dict() for e in experts]
        op = {a: OPCMMerger(alpha=a).merge_state_dict(base_sd, sds) for a in uk.OPCM_ALPHA_GRID}

        def sdm(s):
            m_eval.load_state_dict(s)
            return m_eval
        a = max(op, key=lambda k: vf1(hp(sdm(op[k]), v_img, v_txt)))
        scores["OPCM"].append(macro_f1(yt, hp(sdm(op[a]), t_img, t_txt), C))

        bias = max(uk.ROUTER_BIAS_GRID, key=lambda bb: vf1(uk.ensemble_predict(experts, uk.build_router(task_data, bb), v_img, v_txt)[0].argmax(-1).numpy()))
        logits, rw = uk.ensemble_predict(experts, uk.build_router(task_data, bias), t_img, t_txt)
        scores["MM-FLEX router-ensemble"].append(macro_f1(yt, logits.argmax(-1).numpy(), C))
        routing.append(float((rw.argmax(1) == t_tid).float().mean()))

        with torch.no_grad():
            oracle = torch.cat([experts[i](td["test_img"], td["test_txt"]).argmax(-1) for i, td in enumerate(task_data)]).numpy()
        scores["Oracle event-ID experts"].append(macro_f1(yt, oracle, C))

        tr = [torch.cat([td[f"train_{k}"] for td in task_data]) for k in ("img", "txt", "y")]
        joint = uk.train_expert(*tr, num_classes=C, mode="gated", seed=seed)
        scores["Joint head (ref)"].append(macro_f1(yt, hp(joint, t_img, t_txt), C))
        print(f"   seed {seed}: " + " | ".join(f"{n.split(' ')[0]} {scores[n][-1]:.3f}" for n in names))

    summary = {n: {"macro_f1_mean": float(np.mean(v)), "macro_f1_std": float(np.std(v, ddof=1))} for n, v in scores.items()}
    summary["MM-FLEX router-ensemble"]["routing_acc"] = float(np.mean(routing))
    return summary, labels


def main():
    t0 = time.time()
    print("P1 — Router probe (UK-CrisisMMD, chỉ VAL):")
    p1 = probe_router(load_phase3_protocol(ROOT))
    for k, v in p1.items():
        print(f"   {k:<18}: val Macro-F1 {v['val_macro_f1']:.4f} | routing acc {v['val_routing_acc']:.4f}")

    print("\nP2 — CrisisMMD v2 domain-incremental (test, 5 seed, order xáo theo seed):")
    p2, labels = probe_v2()
    for k, v in p2.items():
        extra = f" | routing acc {v['routing_acc']:.4f}" if "routing_acc" in v else ""
        print(f"   {k:<26}: Macro-F1 {v['macro_f1_mean']:.4f} ± {v['macro_f1_std']:.4f}{extra}")

    out = ROOT / "analysis_outputs" / "feasibility_probes.json"
    out.write_text(json.dumps({"p1_uk_router_val": p1, "p2_v2_domain_incremental": p2, "v2_labels": labels,
                               "status": "exploratory", "runtime_s": time.time() - t0}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n📁 {out} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
