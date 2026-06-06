#!/usr/bin/env python3
"""Figures for the single-poller persistent-kernel sweep.

Reads the tidy CSV from sweep_single_poller.sh and emits:
  poll_vs_blocks.png  - poll_wait vs blocks, per dataset, config-0/1/2 + fits
  decomp_n5000.png    - first(compute+rtt) vs rest(fan-in) stacked, at n=5000
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "bench/results/sweep/single_poller_sweep.csv"
OUTDIR = "bench/results/sweep"

LABELS = {
    "config0": "config-0 baseline (per-block PCIe poll)",
    "config1": "config-1 single-poller (kills a)",
    "config2": "config-2 +StageFeatures (kills b)",
}
COLORS = {"config0": "#d62728", "config1": "#1f77b4", "config2": "#2ca02c"}

# rows[(dataset, config)] = list of (blocks, first_us, rest_us, poll_us)
rows = defaultdict(list)
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows[(r["dataset"], r["config"])].append(
            (int(r["blocks"]), int(r["first_ns"]) / 1e3,
             int(r["rest_ns"]) / 1e3, int(r["poll_ns"]) / 1e3))
for k in rows:
    rows[k].sort()

datasets = []
for (ds, _cfg) in rows:
    if ds not in datasets:
        datasets.append(ds)
feats = {"higgs": 28, "epsilon": 2000}


def fit(x, y):
    m, b = np.polyfit(x, y, 1)
    return m, b


# ---- Slope report (term attribution) -------------------------------------
slope_rows = [("dataset", "features", "config", "slope_us_per_block",
               "intercept_us", "poll_at_20blk_us")]
print(f"{'dataset':9} {'config':8} {'slope(us/blk)':>14} {'intercept(us)':>14} "
      f"{'poll@20blk(us)':>14}")
for ds in datasets:
    for cfg in ("config0", "config1", "config2"):
        pts = rows[(ds, cfg)]
        if not pts:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[3] for p in pts])
        m, b = fit(x, y)
        at20 = next((p[3] for p in pts if p[0] == 20), float("nan"))
        print(f"{ds:9} {cfg:8} {m:14.3f} {b:14.2f} {at20:14.2f}")
        slope_rows.append((ds, feats.get(ds, ""), cfg, f"{m:.4f}", f"{b:.4f}",
                           f"{at20:.3f}"))
with open(f"{OUTDIR}/single_poller_slopes.csv", "w", newline="") as f:
    csv.writer(f).writerows(slope_rows)
print(f"wrote {OUTDIR}/single_poller_slopes.csv")


# ---- Figure 1: poll_wait vs blocks ---------------------------------------
fig, axes = plt.subplots(1, len(datasets), figsize=(13, 5.2), squeeze=False)
for ax, ds in zip(axes[0], datasets):
    for cfg in ("config0", "config1", "config2"):
        pts = rows[(ds, cfg)]
        if not pts:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[3] for p in pts])
        m, b = fit(x, y)
        c = COLORS[cfg]
        ax.plot(x, y, "o", color=c, zorder=3)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, m * xs + b, "-", color=c,
                label=f"{LABELS[cfg]}\n  slope={m:.2f} µs/block")
    ax.set_title(f"{ds}  ({feats.get(ds, '?')} features)")
    ax.set_xlabel("persistent blocks  (= ceil(n_trees / 256))")
    ax.set_ylabel("host poll_wait per request (µs)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Single-poller persistent kernel: poll_wait vs grid width", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/poll_vs_blocks.png", dpi=130)
print(f"wrote {OUTDIR}/poll_vs_blocks.png")

# ---- Figure 2: first/rest decomposition at n=5000 (20 blocks) -------------
fig, axes = plt.subplots(1, len(datasets), figsize=(11, 5.2), squeeze=False)
cfgs = ["config0", "config1", "config2"]
for ax, ds in zip(axes[0], datasets):
    firsts, rests = [], []
    for cfg in cfgs:
        pt = [p for p in rows[(ds, cfg)] if p[0] == 20]
        firsts.append(pt[0][1] if pt else 0.0)
        rests.append(pt[0][2] if pt else 0.0)
    xpos = np.arange(len(cfgs))
    ax.bar(xpos, firsts, color="#4c72b0", label="first: request+features+compute+1st result")
    ax.bar(xpos, rests, bottom=firsts, color="#dd8452", label="rest: done-flag fan-in (term c)")
    for i, (fst, rst) in enumerate(zip(firsts, rests)):
        ax.text(i, fst + rst + 0.4, f"{fst + rst:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["config-0", "config-1", "config-2"])
    ax.set_title(f"{ds}  n=5000 (20 blocks), {feats.get(ds, '?')} features")
    ax.set_ylabel("host poll_wait per request (µs)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("poll_wait decomposition at 20 blocks", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUTDIR}/decomp_n5000.png", dpi=130)
print(f"wrote {OUTDIR}/decomp_n5000.png")
