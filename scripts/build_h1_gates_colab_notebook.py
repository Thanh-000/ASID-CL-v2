#!/usr/bin/env python3
"""
Build ONE self-contained Colab T4 / Pro+ notebook running gates H1-G1 and H1-G2 on
UPMC Food-101 (RebQ/PAL protocol: 100 classes, 10 class-incremental tasks).

No Google Drive: repo code (BlockAnalyticCL, gate engine) is embedded verbatim,
results are written to /content/h1_results and downloaded as zip after each stage.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED = {
    "mm_flex_merge/__init__.py": "",
    "mm_flex_merge/models/__init__.py": "from .analytic import BlockAnalyticCL\n",
    "mm_flex_merge/models/analytic.py": (ROOT / "src/mm_flex_merge/models/analytic.py").read_text(encoding="utf-8"),
    "h1_gate_engine.py": (ROOT / "scripts/h1_gate_engine.py").read_text(encoding="utf-8"),
}

nb = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "language_info": {"name": "python"}},
      "cells": []}


def md(s):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": s.strip() + "\n"})


def code(s):
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": s.strip() + "\n"})


md("""
# H1 Gates on UPMC Food-101 (RebQ/PAL Protocol: 100 Classes, 10 Tasks)
- **Dataset:** **UPMC Food-101** (100 classes after dropping rarest class per RebQ/PAL protocol, 10 class-incremental tasks × 10 classes/task).
- **Hardware:** Optimized for **Tesla T4 GPU** (`FP16 + GradScaler(init_scale=1024.0)`, `BS=64`, `np.memmap` disk-backed image tensor so System RAM stays `< 2.5 GB`).
- **Outputs:** Written to `/content/h1_results/food101` (`food_results.json`, `food_predictions.npz`, `run.log`), zipped and downloaded automatically after `G1` and after `G2`.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
import os, sys; print("CPUs:", os.cpu_count())
""")

files_src = "import pathlib\nFILES = " + json.dumps(EMBED) + "\n" + \
    "for p, s in FILES.items():\n    q = pathlib.Path('/content/h1code') / p\n    q.parent.mkdir(parents=True, exist_ok=True)\n    q.write_text(s)\n" + \
    "import sys; sys.path.insert(0, '/content/h1code')\nimport h1_gate_engine as E\nprint('engine on', E.DEV, '| amp', E.AMP)"
code(files_src)

code("""
# ---- experiment configuration ----
EPOCHS, BS, LR = 2, 64, 1e-4
TAUS = [None, 0.95, 0.99]           # None = naive sequential LoRA
SEEDS_FOOD = [2026]                 # add 2027, 2028 if time allows
WORKERS = min(os.cpu_count() or 2, 4)
import json, pathlib, numpy as np, torch
OUT = pathlib.Path('/content/h1_results'); OUT.mkdir(exist_ok=True)
E.LOG_PATH = str(OUT / 'run.log')
CONFIG = dict(epochs=EPOCHS, bs=BS, lr=LR, taus=TAUS, seeds_food=SEEDS_FOOD,
              g1_rule='oracle - frozen >= 2 pts', g2_rule='best null-space stale/transport >= frozen + 0.5*(oracle - frozen)')
(OUT / 'config.json').write_text(json.dumps(CONFIG, indent=2))
from google.colab import files
def download():
    !cd /content && rm -f h1_results.zip && zip -q -r h1_results.zip h1_results
    try: files.download('/content/h1_results.zip')
    except Exception as e: print('download failed (results still in /content/h1_results):', e)
""")

md("## 1. Download & Prepare UPMC Food-101 (RebQ/PAL Protocol)")
code("""
# Zero-config multi-connection aria2c download from Kaggle (16 streams, no KAGGLE_USERNAME required)
if not pathlib.Path('/content/food101').exists() or not list(pathlib.Path('/content/food101').rglob('*.csv')):
    !apt-get -qq install -y aria2 > /dev/null
    pathlib.Path('/content/food101').mkdir(parents=True, exist_ok=True)
    !aria2c -x 16 -s 16 -k 1M --allow-overwrite=true -q "https://www.kaggle.com/api/v1/datasets/download/gianmarco96/upmcfood101" -d /content -o food101.zip
    !unzip -q -o /content/food101.zip -d /content/food101 && rm -f /content/food101.zip
!find /content/food101 -maxdepth 3 -type d | head -20; find /content/food101 -name '*.csv'
""")

code("""
import gc, json, pathlib, glob, numpy as np, pandas as pd, torch
root = pathlib.Path('/content/food101')
csv = [p for p in glob.glob(str(root / '**/*.csv'), recursive=True) if 'train' in pathlib.Path(p).name.lower()][0]
df = pd.read_csv(csv, header=None, names=['image', 'text', 'label'], on_bad_lines='skip')
if str(df.iloc[0]['label']).lower() in {'label', 'category', 'class'}: df = df.iloc[1:].reset_index(drop=True)
index = {pathlib.Path(p).name: p for p in glob.glob(str(root / '**/train/**/*.jpg'), recursive=True)}
df['path'] = df['image'].map(lambda n: index.get(pathlib.Path(str(n)).name))
print('raw train rows', len(df), '| missing images', int(df['path'].isna().sum()), '| csv', csv)
df = df.dropna(subset=['path']).reset_index(drop=True)
df = df[df['label'] != df['label'].value_counts().idxmin()].reset_index(drop=True)   # RebQ: drop rarest class
classes = sorted(df['label'].unique()); assert len(classes) == 100, len(classes)
df['y'] = df['label'].map({c: i for i, c in enumerate(classes)})
PER_CLASS = 120   # T4 12.7GB RAM-safe stratified subset: 120/class x 100 classes = 12,000 samples (1.8 GB tensor)
rng = np.random.RandomState(0); keep_rows = []
for c, idx in df.groupby('y').groups.items():
    idx = rng.permutation(np.array(idx))[:PER_CLASS]
    keep_rows.append(idx)
df = df.loc[np.concatenate(keep_rows)].reset_index(drop=True)
df['split'] = 'train'
for c, idx in df.groupby('y').groups.items():                                         # 80% train, 10% val, 10% test
    idx = np.array(idx); n_va = max(1, len(idx) // 10)
    df.loc[idx[:n_va], 'split'] = 'val'
    df.loc[idx[n_va:2 * n_va], 'split'] = 'test'
corder = np.random.RandomState(2026).permutation(100)
df['task'] = df['y'].map({int(c): k // 10 for k, c in enumerate(corder)})
del index; gc.collect()
print(df['split'].value_counts().to_dict(), '| total rows:', len(df))
D_food = E.make_data(df, path_col='path', text_col='text', label_col='y', task_col='task',
                     task_order=list(range(10)), num_classes=100, workers=WORKERS)
""")

md("## 2. Run Gate G1 & Gate G2 on UPMC Food-101")
code("""
# G1: frozen / first-session LoRA / joint-oracle LoRA
import json, pathlib, numpy as np, torch
counts_f = json.loads(df.groupby(['task', 'split']).size().unstack(fill_value=0).to_json())
g1f = E.run_g1(D_food, seeds=SEEDS_FOOD, epochs=EPOCHS, bs=BS, lr=LR, stage='food_g1')
vf = E.verdicts(g1f, None, 'acc'); print(json.dumps(vf, indent=2))
E.export(OUT / 'food101', 'food', D_food, {'stages': ['food_g1'], 'config': CONFIG, 'counts': counts_f,
                                          'g1': g1f, 'verdicts': vf})
download()
""")

code("""
# G2: 10 class-incremental tasks (naive vs null-space LoRA + BlockAnalyticCL & Transport-SID)
import json, pathlib, numpy as np, torch
g2f = E.run_g2(D_food, taus=TAUS, seeds=SEEDS_FOOD, epochs=EPOCHS, bs=BS, lr=LR, stage='food_g2')
vf = E.verdicts(g1f, g2f, 'acc'); print(json.dumps(vf, indent=2))
E.export(OUT / 'food101', 'food', D_food, {'stages': ['food_g1', 'food_g2'], 'config': CONFIG, 'counts': counts_f,
                                          'g1': g1f, 'g2': g2f, 'verdicts': vf})
download()
""")

md("## 3. Summary")
code("""
for p in sorted(OUT.glob('*/*_results.json')):
    print(p.name, json.dumps(json.loads(p.read_text())['verdicts'], indent=2))
!ls -la /content/h1_results/*
download()
""")

out = ROOT / "notebooks" / "H1_gates_colab_proplus.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out} ({len(nb['cells'])} cells, {out.stat().st_size / 1024:.0f} KB)")
