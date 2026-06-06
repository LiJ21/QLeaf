#!/usr/bin/env python3
"""Single-poller end-to-end latency sweep (real_time vs n), per dataset.

Reuses plot_sweep.py's benchmark-name parsing, extended for the optional
/sp:<d>/stage:<d> suffix. Reads sp_<dataset>_n*.json (clean, non-prof gbench
JSON). Draws, per dataset, real_time vs n for:
  - oneshot best (compact)         -- reference
  - persistent baseline config-0   -- T1 B0 C0 M1 CR0, no single-poller
  - single-poller                  -- config-1 (no stage) for higgs,
                                      config-2 (with stage) for epsilon
since StageFeatures only pays off for wide feature vectors.
"""
import glob, json, os, re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = re.compile(
    r"cuda/(?P<mode>oneshot|persistent)/(?P<buffer>compact|tree)/bt:(?P<bt>\d+)/"
    r"transposed:(?P<T>\d)/bitmask:(?P<B>\d)/cached:(?P<C>\d)/mapped:(?P<M>\d)"
    r"(?:/crossreduce:(?P<CR>\d))?"
    r"(?:/sp:(?P<SP>\d)/stage:(?P<STG>\d)(?:/seqflag:(?P<SEQ>\d))?)?"
    r"/features:(?P<feat>\d+)/real_time_median")
CPU_PAT = re.compile(
    r"(?P<worker>branch|bitmask)/(?P<buffer>compact|tree)/"
    r"(?:nothread|threads:(?P<th>\d+))/features:(?P<feat>\d+)/real_time_median")

# which single-poller variant wins per dataset (stage = StageFeatures)
SP_STAGE = {"higgs": 0, "epsilon": 1}


def load(dataset):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, f"sp_{dataset}_n*.json"))):
        n = int(re.search(r"_n(\d+)\.json$", f).group(1))
        for b in json.load(open(f))["benchmarks"]:
            us = b["real_time"] / 1000.0
            m = PAT.search(b["name"])
            if m and m["buffer"] == "compact":
                d = m.groupdict()
                rows.append(dict(
                    n=n, mode=d["mode"], T=int(d["T"]), B=int(d["B"]),
                    C=int(d["C"]), M=int(d["M"]),
                    CR=int(d["CR"]) if d["CR"] else 0,
                    SP=int(d["SP"]) if d["SP"] else 0,
                    STG=int(d["STG"]) if d["STG"] else 0,
                    SEQ=int(d["SEQ"]) if d["SEQ"] else 0, us=us))
                continue
            c = CPU_PAT.search(b["name"])
            if c and c["buffer"] == "compact":
                rows.append(dict(n=n, mode="cpu", us=us))
    return rows


def cpu_best(rows, n):  # min over branch/bitmask x thread counts
    return min((r["us"] for r in rows if r["mode"] == "cpu" and r["n"] == n),
              default=None)


def oneshot_best(rows, n):
    cand = [r for r in rows if r["mode"] == "oneshot" and r["n"] == n]
    return min((r["us"] for r in cand), default=None)


def persistent_base(rows, n):  # config-0: T1 B0 C0 M1 CR0, no SP
    cand = [r for r in rows if r["mode"] == "persistent" and r["SP"] == 0
            and r["T"] == 1 and r["B"] == 0 and r["C"] == 0 and r["M"] == 1
            and r["CR"] == 0 and r["n"] == n]
    return min((r["us"] for r in cand), default=None)


def single_poller(rows, n, stage):
    # Best single-poller at n across all stage/seqflag variants (stage arg kept
    # for signature compat but ignored -- we report the per-n optimum).
    cand = [r["us"] for r in rows if r.get("SP") == 1 and r["n"] == n]
    return min(cand, default=None)


import sys
_all = [("higgs", 28), ("epsilon", 2000)]
_want = set(sys.argv[1:])  # optional dataset filter, e.g. `plot_sp_latency.py higgs`
datasets = [d for d in _all if d[0] in _want] if _want else _all
fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5.4),
                         squeeze=False)
print(f"{'dataset':9} {'n':>5} {'cpu(best)':>10} {'oneshot':>9} {'persist-0':>10} "
      f"{'single-poll':>12} {'speedup vs persist-0':>20}")
for ax, (ds, feat) in zip(axes[0], datasets):
    rows = load(ds)
    ns = sorted({r["n"] for r in rows})
    stage = SP_STAGE[ds]
    c = [cpu_best(rows, n) for n in ns]
    o = [oneshot_best(rows, n) for n in ns]
    p = [persistent_base(rows, n) for n in ns]
    s = [single_poller(rows, n, stage) for n in ns]
    sp_label = "single-poller (best of stage/seqflag)"
    ax.plot(ns, c, marker="^", ls=":", color="#1f77b4", label="CPU (best of branch/bitmask × threads)")
    ax.plot(ns, o, marker="s", ls="--", color="#7f7f7f", label="oneshot (best)")
    ax.plot(ns, p, marker="o", color="#d62728", label="persistent baseline (config-0)")
    ax.plot(ns, s, marker="o", color="#2ca02c", lw=2.2, label=sp_label)
    for n, pv, sv in zip(ns, p, s):
        if pv and sv:
            ax.annotate(f"{pv / sv:.2f}×", (n, sv), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#2ca02c")
    ax.set_title(f"{ds}  ({feat} features)")
    ax.set_xlabel("n (number of trees)")
    ax.set_ylabel("end-to-end latency per inference (µs, real_time)")
    ax.set_xticks(ns)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper left")
    for n, cv, ov, pv, sv in zip(ns, c, o, p, s):
        print(f"{ds:9} {n:>5} {cv or 0:>10.2f} {ov or 0:>9.2f} {pv or 0:>10.2f} "
              f"{sv or 0:>12.2f} {(pv / sv if pv and sv else 0):>19.2f}×")

fig.suptitle("Single-poller persistent kernel: end-to-end latency vs tree count "
             "(RTX 3060, release)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
suffix = "_" + "_".join(d[0] for d in datasets) if _want else ""
png = os.path.join(HERE, f"sp_latency_sweep{suffix}.png")
fig.savefig(png, dpi=130)
print(f"\nwrote {png}")
