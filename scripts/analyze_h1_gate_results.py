#!/usr/bin/env python3
"""
Offline analysis of H1 gate results exported by notebooks/H1_gates_colab_proplus.ipynb. VALIDATION ONLY.

Usage:
    python scripts/analyze_h1_gate_results.py h1_results.zip [--out analysis_outputs/h1_gates] [--boot 2000]

Per dataset (<ds>/<name>_results.json + <name>_predictions.npz) it writes to --out:
    report.md          G1/G2 tables (mean ± std over LoRA seeds), verdicts, paired bootstrap CIs
    summary.json       the same numbers, machine-readable
    <name>_*.png       accuracy over sessions, drift by session, protected rank
The analytic-head beta is chosen per arm on validation (as in the gates), so all CIs are EXPLORATORY.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def metric_fn(name, C):
    """Fast acc / macro-F1 (sklearn semantics: average over labels present in y_true or y_pred)."""
    def f(y, p):
        if name == "acc":
            return float((y == p).mean())
        cm = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
        den = cm.sum(0) + cm.sum(1)
        return float((2 * np.diag(cm)[den > 0] / den[den > 0]).mean())
    return f


def ms(xs):
    xs = 100 * np.asarray(xs, dtype=float)
    return f"{xs.mean():.2f} ± {xs.std():.2f}" if len(xs) > 1 else f"{xs[0]:.2f}"


def best_beta(per_seed, metric, betas, field=None):
    rows = [v[field] if field else v for v in per_seed.values()]
    return max(betas, key=lambda b: np.mean([r[f"beta_{b}"][metric] for r in rows]))


def forgetting(A):
    T = len(A)
    if T < 2:
        return 0.0
    return float(np.nanmean([np.nanmax([A[k][j] for k in range(j, T - 1)]) - A[T - 1][j] for j in range(T - 1)]))


def paired_boot(y, pa, pb, fn, B, rng):
    """Difference of seed-averaged metrics (a − b), same resample for every seed."""
    obs = np.mean([fn(y, a) for a in pa]) - np.mean([fn(y, b) for b in pb])
    d = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, len(y), len(y))
        d[i] = np.mean([fn(y[idx], a[idx]) for a in pa]) - np.mean([fn(y[idx], b[idx]) for b in pb])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(obs), "ci95": [float(lo), float(hi)], "p_le_0": float((d <= 0).mean())}


def analyse(res_path, out, B):
    R = json.loads(res_path.read_text())
    name = res_path.name.replace("_results.json", "")
    npz = np.load(res_path.with_name(f"{name}_predictions.npz"))
    meta, betas, metric, C = R["meta"], R["meta"]["betas"], R["verdicts"]["metric"], R["meta"]["num_classes"]
    y = npz["y_val"].astype(np.int64)
    fn, rng = metric_fn(metric, C), np.random.default_rng(0)
    st_g1 = R["stages"][0]
    st_g2 = R["stages"][1] if len(R["stages"]) > 1 else None

    def preds(stage, arm, seeds, b, field=None):
        return [npz["__".join([stage, arm, s] + ([field] if field else []) + [f"beta{b}"])].astype(np.int64)
                for s in seeds]

    S = {"meta": meta, "verdicts": R["verdicts"], "g1": {}, "g2": {}, "bootstrap": {}}
    L = [f"## {name} (metric: {metric}, n_val={meta['n_val']}, n_train={meta['n_train']}, GPU={meta['gpu']})", "",
         f"Task order: {', '.join(meta['task_order'])}", "", "### Verdicts", "", "```json",
         json.dumps(R["verdicts"], indent=2), "```", "", "### G1: adaptation gain (analytic head on all train)", "",
         "| Arm | " + " | ".join(f"β={b} {metric}" for b in betas) + " | best β | seeds |", "|---|" + "---|" * (len(betas) + 2)]
    arms = {}
    for arm, per in R["g1"].items():
        bb = best_beta(per, metric, betas)
        arms[arm] = (st_g1, arm, list(per), bb, None)
        S["g1"][arm] = {f"beta_{b}": [v[f"beta_{b}"][metric] for v in per.values()] for b in betas} | {"best_beta": bb}
        L.append(f"| {arm} | " + " | ".join(ms(S["g1"][arm][f"beta_{b}"]) for b in betas) + f" | {bb} | {len(per)} |")

    if st_g2:
        L += ["", "### G2: sequential LoRA (stale = analytic stats never recomputed)", "",
              "| Config | stale | refit | stale−refit | final Avg Acc | Avg inc. Acc | Forgetting | drift img | drift txt | protected rank |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
        for tag, per in R["g2"].items():
            bs, br = best_beta(per, metric, betas, "stale"), best_beta(per, metric, betas, "refit")
            arms[f"{tag}/stale"] = (st_g2, tag, list(per), bs, "stale")
            arms[f"{tag}/refit"] = (st_g2, tag, list(per), br, "refit")
            stale = [v["stale"][f"beta_{bs}"][metric] for v in per.values()]
            refit = [v["refit"][f"beta_{br}"][metric] for v in per.values()]
            mats = [v["acc_matrix"][f"beta_{bs}"] for v in per.values()]
            curve = np.mean([[np.nanmean(r) for r in A] for A in mats], 0)
            row = {"stale": stale, "refit": refit, "stale_beta": bs, "refit_beta": br,
                   "final_avg_acc": [float(np.nanmean(A[-1])) for A in mats],
                   "avg_inc_acc": [float(np.mean([np.nanmean(r) for r in A])) for A in mats],
                   "forgetting": [forgetting(A) for A in mats],
                   "drift_img": [v["drift_img"] for v in per.values()], "drift_txt": [v["drift_txt"] for v in per.values()],
                   "protected_rank_final": [v["protected_rank_frac"][-1] for v in per.values()],
                   "acc_curve": curve.tolist()}
            S["g2"][tag] = row
            L.append(f"| {tag} | {ms(stale)} | {ms(refit)} | {ms(np.subtract(stale, refit))} | {ms(row['final_avg_acc'])} | "
                     f"{ms(row['avg_inc_acc'])} | {ms(row['forgetting'])} | {np.mean(row['drift_img']):.4f} | "
                     f"{np.mean(row['drift_txt']):.4f} | {np.mean(row['protected_rank_final']):.3f} |")
            k = np.arange(1, len(curve) + 1)
            ax[0].plot(k, 100 * curve, marker="o", label=tag)
            dbs = [v["drift_by_session"] for v in per.values()]
            ax[1].plot(k, [np.mean([d[j]["img"] + d[j]["txt"] for d in dbs]) for j in range(len(k))], marker="o", label=tag)
            ax[2].plot(k, np.mean([v["protected_rank_frac"] for v in per.values()], 0), marker="o", label=tag)
        for a, t in zip(ax, ["acc on seen tasks (%) after session k", "final drift img+txt of session-k features",
                             "protected-rank fraction"]):
            a.set_title(t, fontsize=10)
            a.set_xlabel("session")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{name}_g2_curves.png", dpi=150)
        plt.close(fig)
        L += ["", f"![G2 curves]({name}_g2_curves.png)"]

    pairs = [("joint_oracle", "frozen"), ("first_session", "frozen")]
    if st_g2:
        for tag in R["g2"]:
            pairs += [(f"{tag}/stale", "frozen"), (f"{tag}/stale", f"{tag}/refit")]
            if tag != "naive":
                pairs.append((f"{tag}/stale", "naive/stale"))
    L += ["", f"### Paired bootstrap (B={B}, Δ in points, seed-averaged, EXPLORATORY: β chosen on val)", "",
          "| A − B | Δ | CI95 | P(Δ≤0) |", "|---|---|---|---|"]
    for a, b in pairs:
        if a in arms and b in arms:
            r = paired_boot(y, preds(*arms[a][:4], arms[a][4]), preds(*arms[b][:4], arms[b][4]), fn, B, rng)
            S["bootstrap"][f"{a} - {b}"] = r
            L.append(f"| {a} − {b} | {100 * r['diff']:+.2f} | [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}] | "
                     f"{r['p_le_0']:.3f} |")
    return name, S, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="h1_results.zip or extracted h1_results directory")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis_outputs" / "h1_gates")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.src
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(args.out / "raw")
        src = args.out / "raw"
    found = sorted(src.rglob("*_results.json"))
    assert found, f"no *_results.json under {src}"
    summary, lines = {}, ["# H1 gates: offline analysis (VALIDATION ONLY)", ""]
    cfg = next(src.rglob("config.json"), None)
    if cfg:
        lines += ["```json", cfg.read_text(), "```", ""]
    for p in found:
        name, S, L = analyse(p, args.out, args.boot)
        summary[name] = S
        lines += L + [""]
        print(f"{name}: {json.dumps(S['verdicts'])}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
