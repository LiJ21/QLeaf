#!/usr/bin/env python3
"""Compute-phase breakdown (feat / walk / reduce) for the persistent kernel,
n=5000, blk0, clock64 @ ~1.94 GHz. feat = smem feature load, walk = depth-6
path-chase, reduce = cub BlockReduce (incl. straggler sync)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GHZ = 1.94
def us(c): return c / (GHZ * 1e3)

# (feat, walk, reduce) cyc/req, blk0
data = {
    "higgs": {"config-0\n(mapped)": (1653, 4678, 2240),
              "config-2\n(SP+staged)": (636, 5561, 971)},
    "epsilon": {"config-0\n(mapped)": (10332, 9209, 1501),
                "config-2\n(SP+staged)": (884, 3999, 874)},
}
phases = [("feat (smem feature load)", "#4c72b0"),
          ("walk (depth-6 path-chase)", "#dd8452"),
          ("reduce (cub BlockReduce)", "#55a868")]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), squeeze=False)
for ax, (ds, cfgs) in zip(axes[0], data.items()):
    labels = list(cfgs)
    bottoms = [0.0] * len(labels)
    for i, (ph, c) in enumerate(phases):
        vals = [us(cfgs[l][i]) for l in labels]
        ax.bar(labels, vals, bottom=bottoms, color=c, label=ph)
        for j, v in enumerate(vals):
            if v > 0.4:
                ax.text(j, bottoms[j] + v / 2, f"{v:.1f}", ha="center",
                        va="center", fontsize=9, color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for j, l in enumerate(labels):
        ax.text(j, bottoms[j] + 0.15, f"{bottoms[j]:.1f} µs", ha="center", fontsize=10, fontweight="bold")
    ax.set_title(f"{ds}  (n=5000, 20 blocks, blk0)")
    ax.set_ylabel("device compute cyc/req → µs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
fig.suptitle("Compute breakdown: walk dominates, reduce is cheap (~0.5µs)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = "bench/results/sweep/decomp_compute_n5000.png"
fig.savefig(png, dpi=130)
print("wrote", png)
for ds, cfgs in data.items():
    for l, (f, w, r) in cfgs.items():
        print(f"  {ds:8} {l.split(chr(10))[0]:10} feat={us(f):5.2f} walk={us(w):5.2f} "
              f"reduce={us(r):5.2f}  (walk={100*w/(f+w+r):.0f}% reduce={100*r/(f+w+r):.0f}%)")
