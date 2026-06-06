#!/usr/bin/env python3
"""Epsilon (2000 feat): block-size (BT) sweep x stage vs no-stage, fair rotating
inputs, end-to-end median real_time (RTX 3060). grid = ceil(n/BT)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BTS = [256, 512, 1024]
# data[n] = (config-1 no-stage [bt256,512,1024], config-2 stage [...], cpu_best)
data = {
    2000: ([12.22, 10.86, 10.83], [12.96, 11.16, 10.97], 3.88),
    5000: ([20.99, 14.53, 12.84], [14.54, 12.68, 12.19], 10.93),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), squeeze=False)
x = np.arange(len(BTS))
w = 0.38
for ax, (n, (ns, st, cpu)) in zip(axes[0], data.items()):
    ax.bar(x - w / 2, ns, w, color="#4c72b0", label="config-1 (no stage)")
    ax.bar(x + w / 2, st, w, color="#2ca02c", label="config-2 (stage)")
    for i in range(len(BTS)):
        ax.text(x[i] - w / 2, ns[i] + 0.2, f"{ns[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, st[i] + 0.2, f"{st[i]:.1f}", ha="center", fontsize=8)
    ax.axhline(cpu, ls="--", color="#d62728", lw=1.6, label=f"best CPU ({cpu:.1f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"BT={b}\n(grid={-(-n // b)})" for b in BTS])
    ax.set_title(f"epsilon  n={n}  (2000 features)")
    ax.set_ylabel("end-to-end latency (µs, real_time)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8.5)
fig.suptitle("Epsilon: block-size sweep x stage/no-stage (fair rotating inputs, "
             "RTX 3060)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
png = "bench/results/sweep/epsilon_bt_stage.png"
fig.savefig(png, dpi=130)
print("wrote", png)
