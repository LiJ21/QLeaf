#!/usr/bin/env python3
"""Compare two latency configurations side by side.

Each side can be either a result root containing summary.json files or a single
summary.json file. If a side contains multiple matching runs for the same
dataset/tree-count key, the script keeps the lowest value for the selected
metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


DEFAULT_METRIC = "p50_ns"
KNOWN_DATASETS = {"higgs", "epsilon"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare two latency result configurations side by side."
    )
    p.add_argument("left", type=Path, help="left result root or summary.json")
    p.add_argument("right", type=Path, help="right result root or summary.json")
    p.add_argument("--left-label", default="left", help="label for left side")
    p.add_argument("--right-label", default="right", help="label for right side")
    p.add_argument(
        "--left-filter",
        help="regex applied to left run names before selecting best rows",
    )
    p.add_argument(
        "--right-filter",
        help="regex applied to right run names before selecting best rows",
    )
    p.add_argument(
        "--metric",
        default=DEFAULT_METRIC,
        help="latency metric from run['latency']; default: p50_ns",
    )
    p.add_argument(
        "--key",
        default="dataset,ntrees",
        help="comma-separated grouping key fields; default: dataset,ntrees",
    )
    p.add_argument(
        "--csv",
        action="store_true",
        help="emit CSV instead of Markdown",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="include keys present on only one side",
    )
    p.add_argument(
        "--no-config",
        action="store_true",
        help="omit selected config-name columns",
    )
    return p.parse_args()


def summary_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("summary.json"))


def infer_dataset(summary_path: Path, root: Path, run: dict[str, Any],
                  meta: dict[str, Any]) -> str:
    for source in (run, meta):
        value = source.get("dataset")
        if value:
            return str(value)

    parts = summary_path.parts
    for part in parts:
        lower = part.lower()
        if lower in KNOWN_DATASETS:
            return lower

    try:
        rel = summary_path.relative_to(root if root.is_dir() else root.parent)
        if len(rel.parts) >= 4:
            return rel.parts[0]
    except ValueError:
        pass

    features = run.get("features", meta.get("features"))
    if features is not None:
        return f"features:{features}"
    return "unknown"


def infer_ntrees(summary_path: Path, run: dict[str, Any],
                 meta: dict[str, Any]) -> int | str:
    for source in (run, meta):
        value = source.get("ntrees")
        if value is not None:
            return value

    for part in summary_path.parts:
        if re.fullmatch(r"n\d+", part):
            return int(part[1:])
    return "unknown"


def load_runs(root: Path, pattern: str | None) -> list[dict[str, Any]]:
    regex = re.compile(pattern) if pattern else None
    rows: list[dict[str, Any]] = []
    for path in summary_paths(root):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}", file=sys.stderr)
            continue

        meta = data.get("meta", {})
        for run in data.get("runs", []):
            name = str(run.get("name", ""))
            if regex and not regex.search(name):
                continue
            if run.get("status") != "ok":
                continue
            lat = run.get("latency", {})
            rows.append({
                "dataset": infer_dataset(path, root, run, meta),
                "ntrees": infer_ntrees(path, run, meta),
                "name": name,
                "summary_path": str(path),
                "latency": lat,
            })
    return rows


def key_tuple(row: dict[str, Any], key_fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field, "unknown") for field in key_fields)


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get("latency", {}).get(metric)
    if value is None:
        return math.inf
    return float(value)


def best_by_key(rows: list[dict[str, Any]], key_fields: list[str],
                metric: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = key_tuple(row, key_fields)
        if key not in best or metric_value(row, metric) < metric_value(best[key], metric):
            best[key] = row
    return best


def sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    ret: list[Any] = []
    for item in key:
        if isinstance(item, int):
            ret.append(item)
        elif isinstance(item, str) and item.isdigit():
            ret.append(int(item))
        else:
            ret.append(str(item))
    return tuple(ret)


def ns_to_us(value: float) -> float:
    return value / 1000.0


def fmt_us(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{ns_to_us(value):.3f}"


def fmt_ratio(left_ns: float | None, right_ns: float | None) -> str:
    if not left_ns or not right_ns or not math.isfinite(left_ns) or not math.isfinite(right_ns):
        return ""
    return f"{right_ns / left_ns:.3f}"


def fmt_pct(left_ns: float | None, right_ns: float | None) -> str:
    if not left_ns or not right_ns or not math.isfinite(left_ns) or not math.isfinite(right_ns):
        return ""
    return f"{(right_ns - left_ns) / left_ns * 100.0:+.1f}%"


def winner(left_ns: float | None, right_ns: float | None,
           left_label: str, right_label: str) -> str:
    if left_ns is None or right_ns is None:
        return ""
    if not math.isfinite(left_ns) or not math.isfinite(right_ns):
        return ""
    if left_ns == right_ns:
        return "tie"
    return left_label if left_ns < right_ns else right_label


def build_rows(args: argparse.Namespace) -> list[list[str]]:
    key_fields = [field.strip() for field in args.key.split(",") if field.strip()]
    if not key_fields:
        raise SystemExit("--key must contain at least one field")

    left_rows = load_runs(args.left, args.left_filter)
    right_rows = load_runs(args.right, args.right_filter)
    if not left_rows:
        raise SystemExit("no matching successful runs found on left side")
    if not right_rows:
        raise SystemExit("no matching successful runs found on right side")

    left_best = best_by_key(left_rows, key_fields, args.metric)
    right_best = best_by_key(right_rows, key_fields, args.metric)
    if args.all:
        keys = set(left_best) | set(right_best)
    else:
        keys = set(left_best) & set(right_best)
    if not keys:
        raise SystemExit("no common comparison keys found")

    table: list[list[str]] = []
    for key in sorted(keys, key=sort_key):
        left = left_best.get(key)
        right = right_best.get(key)
        left_ns = metric_value(left, args.metric) if left else None
        right_ns = metric_value(right, args.metric) if right else None
        delta = None
        if left_ns is not None and right_ns is not None:
            delta = right_ns - left_ns

        row = [str(item) for item in key]
        row.extend([
            fmt_us(left_ns),
            fmt_us(right_ns),
            fmt_us(delta),
            fmt_pct(left_ns, right_ns),
            fmt_ratio(left_ns, right_ns),
            winner(left_ns, right_ns, args.left_label, args.right_label),
        ])
        if not args.no_config:
            row.extend([
                left.get("name", "") if left else "",
                right.get("name", "") if right else "",
            ])
        table.append(row)
    return table


def emit_markdown(args: argparse.Namespace, rows: list[list[str]]) -> None:
    key_fields = [field.strip() for field in args.key.split(",") if field.strip()]
    metric_label = args.metric.removesuffix("_ns")
    headers = [
        *key_fields,
        f"{args.left_label} {metric_label} us",
        f"{args.right_label} {metric_label} us",
        "delta us",
        "delta %",
        "right/left",
        "winner",
    ]
    if not args.no_config:
        headers.extend([f"{args.left_label} config", f"{args.right_label} config"])

    print(f"# Config Comparison ({args.metric})")
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        rendered = []
        for i, value in enumerate(row):
            if not args.no_config and i >= len(headers) - 2 and value:
                rendered.append(f"`{value}`")
            else:
                rendered.append(value)
        print("| " + " | ".join(rendered) + " |")


def emit_csv(args: argparse.Namespace, rows: list[list[str]]) -> None:
    key_fields = [field.strip() for field in args.key.split(",") if field.strip()]
    metric_label = args.metric.removesuffix("_ns")
    headers = [
        *key_fields,
        f"{args.left_label}_{metric_label}_us",
        f"{args.right_label}_{metric_label}_us",
        "delta_us",
        "delta_pct",
        "right_over_left",
        "winner",
    ]
    if not args.no_config:
        headers.extend([f"{args.left_label}_config", f"{args.right_label}_config"])
    writer = csv.writer(sys.stdout)
    writer.writerow(headers)
    writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    if args.csv:
        emit_csv(args, rows)
    else:
        emit_markdown(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
