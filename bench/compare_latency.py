#!/usr/bin/env python3
"""Summarize qleaf/external latency sweep results."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_runs(root: Path):
    for path in root.rglob("summary.json"):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        rel = path.relative_to(root)
        parts = rel.parts
        if "external_fil_modes" in parts:
            continue
        dataset = parts[0] if len(parts) >= 4 else "unknown"
        ntrees = None
        for p in parts:
            if p.startswith("n") and p[1:].isdigit():
                ntrees = int(p[1:])
                break
        for r in data.get("runs", []):
            r = dict(r)
            r["_summary_path"] = str(path)
            r["_dataset"] = dataset
            r["_ntrees_path"] = ntrees
            yield r


def family(name: str) -> str | None:
    if name.startswith("qleaf/cpu/") and "/branch/" in name:
        return "qleaf CPU traversal"
    if name.startswith("qleaf/cpu/") and "/bitmask/" in name:
        return "qleaf CPU bitmask"
    if name.startswith("qleaf/gpu/") and "/bitmask:0/" in name:
        if "/sp:1/" in name:
            if "/cached:1/" in name:
                if "/cacheallwaves:1/" in name:
                    return "qleaf GPU traversal SP cached all-waves"
                return "qleaf GPU traversal SP cached one-wave"
            return "qleaf GPU traversal SP"
        return "qleaf GPU traversal"
    if name.startswith("qleaf/gpu/") and "/bitmask:1/" in name:
        return "qleaf GPU bitmask"
    if name.startswith("xgboost/capi/"):
        return "XGBoost C API"
    if name.startswith("xgboost/"):
        return "XGBoost sklearn"
    if name.startswith("treelite/"):
        return "Treelite"
    if name.startswith("fil/"):
        return "FIL GPU"
    if name.startswith("nvforest/gpu/"):
        return "nvForest GPU"
    if name.startswith("nvforest/cpu/"):
        return "nvForest CPU"
    return None


def us(v):
    return v / 1000.0


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/latency/sweep")
    rows = []
    skips = []
    for r in load_runs(root):
        fam = family(r.get("name", ""))
        if fam is None:
            continue
        if r.get("status") != "ok":
            skips.append((r.get("_dataset"), r.get("_ntrees_path"), fam,
                          r.get("reason", "")))
            continue
        lat = r.get("latency", {})
        rows.append({
            "dataset": r.get("_dataset"),
            "ntrees": r.get("_ntrees_path") or r.get("ntrees"),
            "family": fam,
            "name": r.get("name"),
            "p50": lat.get("p50_ns", float("inf")),
            "p90": lat.get("p90_ns", float("inf")),
            "p99": lat.get("p99_ns", float("inf")),
            "mean": lat.get("mean_ns", float("inf")),
            "max": lat.get("max_ns", float("inf")),
        })

    best = {}
    for r in rows:
        key = (r["dataset"], r["ntrees"], r["family"])
        if key not in best or r["p50"] < best[key]["p50"]:
            best[key] = r

    print("# Latency Sweep Summary")
    print()
    if not best:
        print("No successful runs found.")
    else:
        print("| dataset | trees | family | p50 us | p90 us | p99 us | mean us | best config |")
        print("|---|---:|---|---:|---:|---:|---:|---|")
        for key in sorted(best, key=lambda x: (str(x[0]), int(x[1] or 0), x[2])):
            r = best[key]
            print(
                f"| {r['dataset']} | {r['ntrees']} | {r['family']} | "
                f"{us(r['p50']):.3f} | {us(r['p90']):.3f} | "
                f"{us(r['p99']):.3f} | "
                f"{us(r['mean']):.3f} | `{r['name']}` |"
            )

    successful = {(r["dataset"], r["ntrees"], r["family"]) for r in rows}
    visible_skips = [
        (ds, n, fam, reason)
        for ds, n, fam, reason in skips
        if (ds, n, fam) not in successful
    ]

    if visible_skips:
        print()
        print("## Skips")
        seen = set()
        for ds, n, fam, reason in visible_skips:
            item = (ds, n, fam, reason)
            if item in seen:
                continue
            seen.add(item)
            print(f"- {ds} n={n} {fam}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
