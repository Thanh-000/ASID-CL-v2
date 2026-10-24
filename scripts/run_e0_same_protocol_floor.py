#!/usr/bin/env python3
"""
E0 — Same-protocol floor check (gate G0 of method_repositioning.md).

So sánh MM-FLEX (router-ensemble) với các baseline sàn có khả năng xóa chính xác
(NCM, RanPAC) và các baseline single-model (TA, OPCM) trên CÙNG protocol:
  - UK-CrisisMMD, frozen CLIP cache, 6 order (O1..O6) x 5 seed
  - suy luận task-free, đánh giá sau task cuối trên test GỘP của mọi task
  - siêu tham số chọn trên val gộp; test chỉ dùng cho báo cáo cuối
  - metric chính: pooled Macro-F1 (6 lớp); kèm Acc, worst-class F1, routing acc

Cổng G0: nếu RanPAC - MM-FLEX >= 0.05 pooled Macro-F1 -> Plan B.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_mm_flex_merge_uk_crisismmd_study as uk  # noqa: E402  (tái dùng helper đã sửa P0)
from experiments.mm_hope.continual_baselines import RanPACClassifier, RanPACConfig  # noqa: E402
from experiments.mm_hope.phase3_protocol import load_phase3_protocol  # noqa: E402
from mm_flex_merge.merging import OPCMMerger  # noqa: E402

ORDERS = ["O1", "O2", "O3", "O4", "O5", "O6"]
SEEDS = [2026, 2027, 2028, 2029, 2030]
NUM_CLASSES = uk.NUM_CLASSES
RANPAC_BETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  # cấu hình khóa từ note 42
G0_MARGIN = 0.05


def fused_features(img: torch.Tensor, txt: torch.Tensor) -> torch.Tensor:
    """Biểu diễn giống frozen_baseline_runner (note 42): norm(cat(norm(text), norm(image)))."""
    return F.normalize(torch.cat([F.normalize(txt, dim=-1), F.normalize(img, dim=-1)], dim=-1), dim=-1)


def pooled(task_data, split):
    img = torch.cat([td[f"{split}_img"] for td in task_data])
    txt = torch.cat([td[f"{split}_txt"] for td in task_data])
    y = torch.cat([td[f"{split}_y"] for td in task_data])
    task_id = torch.cat([torch.full((len(td[f"{split}_y"]),), i) for i, td in enumerate(task_data)])
    return img, txt, y, task_id


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    per_class = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(per_class.mean()),
        "worst_class_f1": float(per_class.min()),
    }


# ---------------------------------------------------------------------------
# Methods. Mỗi hàm trả về dự đoán (numpy) trên split được yêu cầu.
# ---------------------------------------------------------------------------

class NCM:
    """Nearest class mean trên fused CLIP feature; xóa chính xác = trừ tổng."""

    def __init__(self):
        self.sums = torch.zeros(NUM_CLASSES, 1024)
        self.counts = torch.zeros(NUM_CLASSES)

    def update(self, feats, y):
        self.sums.index_add_(0, y, feats)
        self.counts += torch.bincount(y, minlength=NUM_CLASSES).float()

    def predict(self, feats):
        seen = self.counts > 0
        means = F.normalize(self.sums / self.counts.clamp(min=1).unsqueeze(1), dim=-1)
        sim = feats @ means.T
        sim[:, ~seen] = -float("inf")
        return sim.argmax(dim=-1)


def ranpac_model(task_data, beta, seed):
    model = RanPACClassifier(
        RanPACConfig(
            feature_dim=1024,
            num_classes=NUM_CLASSES,
            projection_dim=1024,
            ridge=1e-1,
            relative_ridge=True,
            solve_dtype="float64",
            solve_device="cpu",
            class_balance_beta=beta,
            seed=seed,
        )
    )
    for td in task_data:  # cập nhật tuần tự theo order
        model.update(fused_features(td["train_img"], td["train_txt"]), td["train_y"])
    return model


def run_one(protocol, order, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    task_data = uk.build_task_data(protocol, order)
    T = len(task_data)
    v_img, v_txt, v_y, v_tid = pooled(task_data, "val")
    t_img, t_txt, t_y, t_tid = pooled(task_data, "test")
    v_feat, t_feat = fused_features(v_img, v_txt), fused_features(t_img, t_txt)
    yv, yt = v_y.numpy(), t_y.numpy()
    out, chosen = {}, {}

    def val_f1(pred):
        return metrics(yv, pred)["macro_f1"]

    # --- Tầng C: NCM, RanPAC (xóa chính xác bằng thống kê đủ) ---
    ncm = NCM()
    for td in task_data:
        ncm.update(fused_features(td["train_img"], td["train_txt"]), td["train_y"])
    out["NCM"] = metrics(yt, ncm.predict(t_feat).numpy())

    rp = {b: ranpac_model(task_data, b, seed) for b in RANPAC_BETA_GRID}
    beta = max(RANPAC_BETA_GRID, key=lambda b: val_f1(rp[b].predict(v_feat).numpy()))
    out["RanPAC (class-balanced)"] = metrics(yt, rp[beta].predict(t_feat).numpy())
    chosen["ranpac_beta"] = beta

    # --- Experts dùng chung cho tầng A và B ---
    base_m = uk.MultimodalHead(512, 256, num_classes=NUM_CLASSES, mode="gated")
    base_w = uk.extract_flat(base_m)
    base_sd = {k: v.detach().clone() for k, v in base_m.state_dict().items()}
    experts = [
        uk.train_expert(td["train_img"], td["train_txt"], td["train_y"], num_classes=NUM_CLASSES, mode="gated", seed=seed)
        for td in task_data
    ]
    deltas = [uk.extract_flat(e) - base_w for e in experts]
    m_eval = uk.MultimodalHead(512, 256, num_classes=NUM_CLASSES, mode="gated")

    def head_pred(model, img, txt):
        with torch.no_grad():
            return model(img, txt).argmax(dim=-1).numpy()

    def flat(w):
        uk.load_flat(m_eval, w)
        return m_eval

    def sd(state):
        m_eval.load_state_dict(state)
        return m_eval

    # --- Tầng A: single merged model ---
    lam = max(uk.TA_LAMBDA_GRID, key=lambda l: val_f1(head_pred(flat(base_w + l * sum(deltas)), v_img, v_txt)))
    out["Task Arithmetic"] = metrics(yt, head_pred(flat(base_w + lam * sum(deltas)), t_img, t_txt))
    chosen["ta_lambda"] = lam

    expert_sds = [e.state_dict() for e in experts]
    opcm = {a: OPCMMerger(alpha=a).merge_state_dict(base_sd, expert_sds) for a in uk.OPCM_ALPHA_GRID}
    alpha = max(uk.OPCM_ALPHA_GRID, key=lambda a: val_f1(head_pred(sd(opcm[a]), v_img, v_txt)))
    out["OPCM"] = metrics(yt, head_pred(sd(opcm[alpha]), t_img, t_txt))
    chosen["opcm_alpha"] = alpha

    # --- Tầng B: multi-expert ---
    def ens_pred(bias, img, txt):
        logits, route_w = uk.ensemble_predict(experts, uk.build_router(task_data, bias), img, txt)
        return logits.argmax(dim=-1).numpy(), route_w.argmax(dim=-1).numpy()

    bias = max(uk.ROUTER_BIAS_GRID, key=lambda b: val_f1(ens_pred(b, v_img, v_txt)[0]))
    pred, route = ens_pred(bias, t_img, t_txt)
    out["MM-FLEX (router-ensemble)"] = metrics(yt, pred)
    out["MM-FLEX (router-ensemble)"]["routing_acc"] = float((route == t_tid.numpy()).mean())
    chosen["router_bias"] = bias

    with torch.no_grad():
        oracle = torch.cat([experts[i](td["test_img"], td["test_txt"]).argmax(-1) for i, td in enumerate(task_data)])
    out["Oracle task-ID experts"] = metrics(yt, oracle.numpy())

    # --- Tham chiếu: joint (không continual) ---
    tr_img = torch.cat([td["train_img"] for td in task_data])
    tr_txt = torch.cat([td["train_txt"] for td in task_data])
    tr_y = torch.cat([td["train_y"] for td in task_data])
    joint = uk.train_expert(tr_img, tr_txt, tr_y, num_classes=NUM_CLASSES, mode="gated", seed=seed)
    out["Joint head (non-continual ref)"] = metrics(yt, head_pred(joint, t_img, t_txt))

    params = {
        "single_head": int(sum(p.numel() for p in base_m.parameters())),
        "experts_total": int(T * sum(p.numel() for p in base_m.parameters())),
        "ranpac_state": int(rp[beta].persistent_state_bytes // 4),
    }
    return out, chosen, params


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def main():
    t0 = time.time()
    protocol = load_phase3_protocol(ROOT)
    runs = []
    for order in ORDERS:
        for seed in SEEDS:
            res, chosen, params = run_one(protocol, order, seed)
            runs.append({"order": order, "seed": seed, "results": res, "chosen": chosen, "params": params})
            print(f"{order} seed {seed}: " + " | ".join(f"{k.split(' ')[0]} {v['macro_f1']:.3f}" for k, v in res.items()))

    methods = list(runs[0]["results"].keys())
    summary = {}
    print("\n" + "=" * 96)
    print(f"{'Method':<32} | {'Pooled Macro-F1':<17} | {'Acc':<15} | {'Worst-class F1':<15}")
    print("-" * 96)
    for m in methods:
        f1s = np.array([r["results"][m]["macro_f1"] for r in runs])
        accs = np.array([r["results"][m]["acc"] for r in runs])
        worst = np.array([r["results"][m]["worst_class_f1"] for r in runs])
        summary[m] = {
            "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std(ddof=1)),
            "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1)),
            "worst_class_f1_mean": float(worst.mean()),
        }
        print(f"{m:<32} | {f1s.mean():.4f} ± {f1s.std(ddof=1):.4f} | {accs.mean():.4f} ± {accs.std(ddof=1):.3f} | {worst.mean():.4f}")

    flex = np.array([r["results"]["MM-FLEX (router-ensemble)"]["macro_f1"] for r in runs])
    rng = np.random.default_rng(0)
    comparisons, pvals = {}, {}
    for m in ["RanPAC (class-balanced)", "NCM", "Task Arithmetic", "OPCM"]:
        other = np.array([r["results"][m]["macro_f1"] for r in runs])
        diff = flex - other
        # bootstrap theo cụm order (6 cụm) để tôn trọng phụ thuộc giữa các seed cùng order
        by_order = np.array([diff[[i for i, r in enumerate(runs) if r["order"] == o]].mean() for o in ORDERS])
        boots = [rng.choice(by_order, size=len(by_order), replace=True).mean() for _ in range(10000)]
        pvals[m] = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
        comparisons[m] = {
            "mean_diff_flex_minus_other": float(diff.mean()),
            "order_cluster_bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": int((diff > 0).sum()), "n": int(len(diff)),
        }
    for m, p in holm(pvals).items():
        comparisons[m]["wilcoxon_p"] = pvals[m]
        comparisons[m]["holm_p"] = p

    print("\nMM-FLEX − baseline (pooled Macro-F1), 30 cặp (6 order × 5 seed):")
    for m, c in comparisons.items():
        lo, hi = c["order_cluster_bootstrap_ci95"]
        print(f"   vs {m:<24}: Δ = {c['mean_diff_flex_minus_other']:+.4f}  CI95 [{lo:+.4f}, {hi:+.4f}]  thắng {c['wins']}/{c['n']}  Holm p = {c['holm_p']:.4g}")

    gap = summary["RanPAC (class-balanced)"]["macro_f1_mean"] - summary["MM-FLEX (router-ensemble)"]["macro_f1_mean"]
    decision = "PLAN B (benchmark/analysis)" if gap >= G0_MARGIN else "PLAN A (method paper) tiếp tục"
    print(f"\nCổng G0: RanPAC − MM-FLEX = {gap:+.4f} (ngưỡng {G0_MARGIN}) -> {decision}")

    report = {
        "experiment": "E0_same_protocol_floor",
        "protocol": {"orders": ORDERS, "seeds": SEEDS, "metric": "pooled macro-F1 over 6 classes, task-free inference, final step",
                     "hparam_selection": "pooled val macro-F1", "ranpac": "p1024, relative ridge 1e-1, fp64 cpu, beta on val"},
        "summary": summary, "comparisons_vs_mm_flex": comparisons,
        "g0": {"ranpac_minus_flex": gap, "margin": G0_MARGIN, "decision": decision},
        "runs": runs, "runtime_seconds": time.time() - t0,
    }
    out = ROOT / "analysis_outputs" / "e0_same_protocol_floor.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"📁 {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
