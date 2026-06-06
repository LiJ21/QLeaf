#!/usr/bin/env python3
"""Parse worker_bench GPU sweep JSONs (cuda/* benchmarks across tree counts) into
a tidy CSV and plot real_time vs n, separately for oneshot and persistent, with
one curve per kernel setup (flag combination). Buffer fixed to `compact` for the
plot (tree buffer kept in the CSV)."""
import glob, json, os, re, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(
    r"cuda/(?P<mode>oneshot|persistent)/(?P<buffer>compact|tree)/bt:(?P<bt>\d+)/"
    r"transposed:(?P<T>\d)/bitmask:(?P<B>\d)/cached:(?P<C>\d)/mapped:(?P<M>\d)"
    r"(?:/crossreduce:(?P<CR>\d))?/features:(?P<feat>\d+)/real_time")

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "cuda_n*.json"))):
    n = int(re.search(r"cuda_n(\d+)\.json", f).group(1))
    data = json.load(open(f))
    for b in data["benchmarks"]:
        m = PAT.search(b["name"])
        if not m:
            continue
        d = m.groupdict()
        rows.append(dict(
            n=n, mode=d["mode"], buffer=d["buffer"],
            T=int(d["T"]), B=int(d["B"]), M=int(d["M"]),
            CR=(int(d["CR"]) if d["CR"] is not None else -1),
            time_us=b["real_time"] / 1000.0))

# ---- tidy CSV ----
csv_path = os.path.join(HERE, "gpu_sweep_tidy.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["n", "mode", "buffer", "T", "B", "M", "CR", "time_us"])
    w.writeheader()
    for r in sorted(rows, key=lambda r: (r["mode"], r["buffer"], r["T"], r["B"], r["M"], r["CR"], r["n"])):
        w.writerow(r)
print(f"wrote {csv_path}  ({len(rows)} rows)")

# ---- plot (compact buffer) ----
ns = sorted({r["n"] for r in rows})

def setup_label(r, mode):
    s = f"T{r['T']} B{r['B']} M{r['M']}"
    if mode == "persistent":
        s += f" C{r['CR']}"
    return s

def series(mode):
    out = {}
    for r in rows:
        if r["mode"] != mode or r["buffer"] != "compact":
            continue
        out.setdefault(setup_label(r, mode), {})[r["n"]] = r["time_us"]
    return dict(sorted(out.items()))

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True)
for ax, mode in zip(axes, ["oneshot", "persistent"]):
    ser = series(mode)
    cmap = plt.get_cmap("tab20")
    for i, (label, pts) in enumerate(ser.items()):
        xs = [x for x in ns if x in pts]
        ys = [pts[x] for x in xs]
        # best (lowest) curve drawn thick
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=cmap(i % 20), label=label)
    ax.set_title(f"CUDA {mode}  ({len(ser)} setups, buffer=compact)")
    ax.set_xlabel("n (number of trees)")
    ax.set_ylabel("latency per inference  (µs, real_time)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, ncol=2, title="T=transposed B=bitmask\nM=mapped C=crossreduce",
              title_fontsize=8, loc="upper left")

fig.suptitle("qleaf GPU worker_bench: latency vs tree count (higgs d6, 28 feat, RTX 3060)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
png = os.path.join(HERE, "gpu_sweep.png")
fig.savefig(png, dpi=130)
print(f"wrote {png}")

# ---- console summary: best setup per (mode, n) ----
print("\nBest (lowest-latency) compact setup per mode per n:")
for mode in ["oneshot", "persistent"]:
    print(f"\n  {mode}:")
    print(f"    {'n':>6}  {'best µs':>9}  setup")
    for n in ns:
        cand = [r for r in rows if r["mode"] == mode and r["buffer"] == "compact" and r["n"] == n]
        best = min(cand, key=lambda r: r["time_us"])
        print(f"    {n:>6}  {best['time_us']:>9.2f}  {setup_label(best, mode)}")

# ---- oneshot vs persistent best, side by side ----
print("\nOneshot vs persistent (best compact setup) µs:")
print(f"    {'n':>6}  {'oneshot':>9}  {'persistent':>11}  {'faster':>10}")
for n in ns:
    bo = min((r for r in rows if r["mode"]=="oneshot" and r["buffer"]=="compact" and r["n"]==n), key=lambda r: r["time_us"])
    bp = min((r for r in rows if r["mode"]=="persistent" and r["buffer"]=="compact" and r["n"]==n), key=lambda r: r["time_us"])
    faster = "oneshot" if bo["time_us"] < bp["time_us"] else "persistent"
    print(f"    {n:>6}  {bo['time_us']:>9.2f}  {bp['time_us']:>11.2f}  {faster:>10}")

# ============================================================================
# BT sweep: does a larger block size flatten persistent's scaling with n?
# bt256 reuses cuda_n*.json; bt512/bt1024 from bt<BT>_n*.json. One block holds
# BT trees, so blocks = ceil(n / BT) -- larger BT => fewer blocks to coordinate.
# ============================================================================
bt_rows = []
for f in sorted(glob.glob(os.path.join(HERE, "cuda_n*.json")) +
                glob.glob(os.path.join(HERE, "bt512_n*.json")) +
                glob.glob(os.path.join(HERE, "bt1024_n*.json"))):
    n_f = int(re.search(r"_n(\d+)\.json$", os.path.basename(f)).group(1))
    for b in json.load(open(f))["benchmarks"]:
        m = PAT.search(b["name"])
        if not m:
            continue
        d = m.groupdict()
        bt_rows.append(dict(n=n_f, mode=d["mode"], buffer=d["buffer"], bt=int(d["bt"]),
                            T=int(d["T"]), B=int(d["B"]), M=int(d["M"]),
                            CR=int(d["CR"]) if d["CR"] is not None else -1,
                            time_us=b["real_time"] / 1000.0))

bts = sorted({r["bt"] for r in bt_rows})

def bt_best(mode, bt, n):
    cand = [r for r in bt_rows if r["mode"] == mode and r["buffer"] == "compact"
            and r["bt"] == bt and r["n"] == n]
    return min(cand, key=lambda r: r["time_us"]) if cand else None

def bt_lbl(r):
    s = f"T{r['T']}B{r['B']}M{r['M']}"
    return s + (f"C{r['CR']}" if r["mode"] == "persistent" else "")

fig2, ax2 = plt.subplots(figsize=(9, 6.5))
cmap2 = plt.get_cmap("viridis")
for i, bt in enumerate(bts):
    xs = [n for n in ns if bt_best("persistent", bt, n)]
    ys = [bt_best("persistent", bt, n)["time_us"] for n in xs]
    ax2.plot(xs, ys, marker="o", lw=2,
             color=cmap2(i / max(1, len(bts) - 1)), label=f"persistent BT={bt}")
xs = [n for n in ns if bt_best("oneshot", 256, n)]
ys = [bt_best("oneshot", 256, n)["time_us"] for n in xs]
ax2.plot(xs, ys, marker="s", lw=2, ls="--", color="crimson", label="oneshot BT=256 (ref)")
ax2.set_title("Does larger BT flatten persistent scaling?\n"
              "(best compact setup per point, higgs d6, RTX 3060)")
ax2.set_xlabel("n (number of trees)")
ax2.set_ylabel("latency per inference  (µs, real_time)")
ax2.set_xticks(ns)
ax2.grid(True, ls=":", alpha=0.6)
ax2.legend()
fig2.tight_layout()
png2 = os.path.join(HERE, "persistent_bt_sweep.png")
fig2.savefig(png2, dpi=130)
print("\nwrote", png2)

print("\nBest persistent latency (µs) per BT  [blk = ceil(n/BT) resident blocks]:")
print("     n " + "".join(f"| BT={bt:<4}              " for bt in bts) + "| oneshotBT256")
for n in ns:
    line = f"  {n:>4} "
    for bt in bts:
        r = bt_best("persistent", bt, n)
        blk = -(-n // bt)
        line += f"| {r['time_us']:6.1f} ({blk:>2}blk {bt_lbl(r)}) " if r else "|  --                "
    o = bt_best("oneshot", 256, n)
    line += f"| {o['time_us']:6.1f}"
    print(line)
