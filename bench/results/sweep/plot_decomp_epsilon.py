#!/usr/bin/env python3
"""Timing decomposition for single-poller epsilon @ n=5000 (20 blocks, 2000
features). Measured with the QLEAF_PROF_PERSIST build (build-prof-bt256),
--iters 60000. Device cycles -> us via clock cross-calibrated from
device-total-cycles / host-request-period (~1.94 GHz, consistent across both
configs)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GHZ = 1.94  # cross-calibrated SM clock under load


def us(cyc):
    return cyc / (GHZ * 1e3)


# measured (ns for host; cycles for device leader=blk0 / follower=blk19)
data = {
    "config-1\n(stage=0)": dict(
        host=dict(first=18217, rest=2367, poll=20584),
        leader=dict(wait=7825, compute=28706, publish=3963),
        follower=dict(wait=10345, compute=27298, publish=2837)),
    "config-2\n(stage=1)": dict(
        host=dict(first=12745, rest=2575, poll=15321),
        leader=dict(wait=21506, compute=5859, publish=2583),
        follower=dict(wait=21405, compute=5489, publish=3046)),
}

fig, (axh, axd) = plt.subplots(1, 2, figsize=(12.5, 5.4))

# ---- Panel A: host poll_wait = first + rest ----
labels = list(data)
first = [data[k]["host"]["first"] / 1e3 for k in labels]
rest = [data[k]["host"]["rest"] / 1e3 for k in labels]
axh.bar(labels, first, color="#4c72b0",
        label="first: request round-trip + feature load + compute + 1st result")
axh.bar(labels, rest, bottom=first, color="#dd8452",
        label="rest: done-flag fan-in of the other 19 blocks (term c)")
for i, (f, r) in enumerate(zip(first, rest)):
    axh.text(i, f + r + 0.3, f"{f + r:.1f} µs", ha="center", fontsize=10)
axh.set_ylabel("host poll_wait per request (µs)")
axh.set_title("Host view: poll_wait")
axh.legend(fontsize=8, loc="upper right")
axh.grid(True, axis="y", alpha=0.3)

# ---- Panel B: device follower per-request phases ----
phases = ["wait (spin: idle gap + staging on crit. path)",
          "compute (feature load + 5000-tree walks + reduce)",
          "publish (partial + fence + done flag)"]
colors = ["#bbbbbb", "#2ca02c", "#8c564b"]
keys = ["wait", "compute", "publish"]
bottom = [0.0, 0.0]
for ph, c, k in zip(phases, colors, keys):
    vals = [us(data[lbl]["follower"][k]) for lbl in labels]
    axd.bar(labels, vals, bottom=bottom, color=c, label=ph)
    bottom = [b + v for b, v in zip(bottom, vals)]
# annotate compute specifically (the staging win)
for i, lbl in enumerate(labels):
    cv = us(data[lbl]["follower"]["compute"])
    axd.text(i, us(data[lbl]["follower"]["wait"]) + cv / 2,
             f"compute\n{cv:.1f} µs", ha="center", fontsize=9, color="white")
axd.set_ylabel("device follower (blk 19) cycles → µs per request")
axd.set_title(f"Device view: per-block phases  (clk≈{GHZ} GHz)")
axd.legend(fontsize=8, loc="upper left")
axd.grid(True, axis="y", alpha=0.3)

fig.suptitle("Timing decomposition: single-poller epsilon @ n=5000 "
             "(20 blocks, 2000 features, RTX 3060)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
png = "bench/results/sweep/sp_decomp_epsilon_n5000.png"
fig.savefig(png, dpi=130)
print("wrote", png)

# ---- console table ----
print(f"\n{'':14} {'host first':>10} {'host rest':>10} {'host poll':>10}"
      f" | {'dev wait':>9} {'dev comp':>9} {'dev pub':>8}  (follower, µs)")
for lbl in labels:
    h, fo = data[lbl]["host"], data[lbl]["follower"]
    print(f"{lbl.strip().replace(chr(10),' '):14} {h['first']/1e3:>10.2f} "
          f"{h['rest']/1e3:>10.2f} {h['poll']/1e3:>10.2f} | "
          f"{us(fo['wait']):>9.2f} {us(fo['compute']):>9.2f} {us(fo['publish']):>8.2f}")
