#!/usr/bin/env python3
"""Device per-request decomposition: config-0 (per-block done-flags) vs
crossreduce:1 (device atomicAdd reduction -> one done_flag). Both non-single-
poller, n=5000, 20 blocks. Measured with QLEAF_PROF_PERSIST (build-prof-bt256),
clock64 cyc/req for blk0, ~1.94 GHz. wait = idle spin between requests (reflects
the host round-trip / per-request period); compute = feature load + walks +
reduce; publish = result write-back + done-flag(s)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GHZ = 1.94
def us(c): return c / (GHZ * 1e3)

# blk0 cyc/req: (wait, compute, publish)
data = {
    "higgs": {"config-0\n(N done-flags)": (36802, 8819, 7653),
              "crossreduce:1\n(1 done-flag)": (53792, 9528, 7523)},
    "epsilon": {"config-0\n(N done-flags)": (40541, 21369, 8359),
                "crossreduce:1\n(1 done-flag)": (60178, 24216, 10485)},
}
phases = [("wait (idle: per-request round-trip)", "#bbbbbb"),
          ("compute (features + walks + reduce)", "#2ca02c"),
          ("publish (result + done-flag)", "#8c564b")]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), squeeze=False)
for ax, (ds, cfgs) in zip(axes[0], data.items()):
    labels = list(cfgs)
    bottoms = [0.0, 0.0]
    for i, (ph, c) in enumerate(phases):
        vals = [us(cfgs[l][i]) for l in labels]
        ax.bar(labels, vals, bottom=bottoms, color=c, label=ph)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for j, l in enumerate(labels):
        tot = us(sum(cfgs[l]))
        ax.text(j, tot + 0.6, f"{tot:.1f} µs", ha="center", fontsize=10, fontweight="bold")
        # annotate the wait delta
    dw = us(cfgs[labels[1]][0] - cfgs[labels[0]][0])
    ax.annotate(f"+{dw:.1f} µs wait", (1, us(cfgs[labels[1]][0]) / 2),
                ha="center", fontsize=9, color="black")
    ax.set_title(f"{ds}  (n=5000, 20 blocks)")
    ax.set_ylabel("device cyc/req → µs (blk0)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Why crossreduce:1 is slower than config-0: the cost is in wait "
             "(longer round-trip), not publish", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = "bench/results/sweep/decomp_crossreduce_n5000.png"
fig.savefig(png, dpi=130)
print("wrote", png)
for ds, cfgs in data.items():
    for l, (w, c, p) in cfgs.items():
        print(f"  {ds:8} {l.split(chr(10))[0]:14} wait={us(w):6.2f} compute={us(c):6.2f} "
              f"publish={us(p):5.2f} period={us(w+c+p):6.2f} µs")
