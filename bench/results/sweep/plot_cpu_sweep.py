#!/usr/bin/env python3
"""CPU sanity sweep (AVX2 build): latency vs thread count, branch vs bitmask.
Reads cpuavx_<dataset>_n*.json (9-rep median). Path-chase (branch) visits depth
nodes/tree; bitmask (QuickScorer) evaluates all 63 internal nodes -- ~10x more
work at depth 6, so branch wins on CPU."""
import json, re, glob, os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
THREADS = ["nt", "t1", "t2", "t4", "t8", "t16"]
XPOS = [1, 1, 2, 4, 8, 16]  # nothread plotted at x=1 (inline single thread)


def parse(ds, n):
    d = {}
    for b in json.load(open(f"{HERE}/cpuavx_{ds}_n{n}.json"))["benchmarks"]:
        nm = b["name"]
        if "real_time_median" not in nm or "/compact/" not in nm:
            continue
        m = re.match(r"(branch|bitmask)/compact/(nothread|threads:(\d+))/", nm)
        if not m:
            continue
        key = "nt" if m.group(2) == "nothread" else "t" + m.group(3)
        d[(m.group(1), key)] = b["real_time"] / 1000.0
    return d


datasets = [("higgs", 28), ("epsilon", 2000)]
ns = [500, 1000, 2000, 5000]

# ---- console table ----
for ds, _ in datasets:
    print(f"\n=== {ds} CPU (AVX), median µs ===")
    print(f"{'':14} " + " ".join(f"{k:>8}" for k in THREADS) + "   best")
    for n in ns:
        d = parse(ds, n)
        for w in ("branch", "bitmask"):
            vals = [d.get((w, k)) for k in THREADS]
            row = " ".join(f"{v:8.2f}" if v else f"{'-':>8}" for v in vals)
            best = min(v for v in vals if v)
            print(f"n={n:<5} {w:7} {row}  {best:.2f}")

# ---- figure: thread scaling at n=5000 ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), squeeze=False)
for ax, (ds, feat) in zip(axes[0], datasets):
    d = parse(ds, 5000)
    xs = XPOS[1:]  # threads 1..16 (skip the nothread duplicate at x=1)
    for w, color in (("branch", "#1f77b4"), ("bitmask", "#d62728")):
        ys = [d.get((w, k)) for k in THREADS[1:]]
        ax.plot(xs, ys, marker="o", color=color, label=w)
    nt_b = d.get(("branch", "nt"))
    ax.axhline(nt_b, ls=":", color="#1f77b4", alpha=0.5,
               label=f"branch nothread ({nt_b:.0f})")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_title(f"{ds}  n=5000  ({feat} features)")
    ax.set_xlabel("worker threads")
    ax.set_ylabel("latency per inference (µs, real_time)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
fig.suptitle("CPU sweep (AVX2): thread scaling, branch (path-chase) vs bitmask "
             "(QuickScorer)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = f"{HERE}/cpu_thread_sweep.png"
fig.savefig(png, dpi=130)
print("\nwrote", png)
