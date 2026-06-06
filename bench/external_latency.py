#!/usr/bin/env python3
"""Batch-1 latency histograms for external tree predictors.

This runner is intentionally optional-dependency friendly: missing XGBoost,
Treelite, RAPIDS FIL, CuPy, or hdrhistogram produce skipped records in the same
summary schema as qleaf_latency instead of making the whole sweep unusable.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np


def parse_csv(path: str, features: int, samples: int) -> np.ndarray:
    rows: list[list[float]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            vals = [float(x) for x in row[:features] if x != ""]
            if vals:
                if len(vals) < features:
                    raise RuntimeError("CSV row has fewer columns than --features")
                rows.append(vals)
            if len(rows) >= samples:
                break
    if not rows:
        raise RuntimeError("CSV file has no feature rows")
    return np.ascontiguousarray(rows, dtype=np.float32)


def random_features(features: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.random((samples, features)), dtype=np.float32)


def import_optional(name: str):
    try:
        return importlib.import_module(name)
    except Exception as e:  # import-time CUDA/library errors matter too
        return e


def configure_cuda_python_env() -> None:
    os.environ.setdefault("CUPY_CACHE_DIR", "/tmp/qleaf-cupy-cache")
    if "CUDA_PATH" in os.environ:
        return
    for entry in sys.path:
        cuda_root = Path(entry) / "nvidia" / "cuda_runtime"
        if (cuda_root / "include" / "cuda.h").exists():
            os.environ["CUDA_PATH"] = str(cuda_root)
            return


class Hdr:
    def __init__(self, highest_ns: int, sigfig: int):
        mod = import_optional("hdrh.histogram")
        if isinstance(mod, Exception):
            raise RuntimeError(f"missing hdrhistogram package: {mod}")
        self.impl = mod.HdrHistogram(1, highest_ns, sigfig)
        self.dropped = 0

    def record(self, value: int) -> None:
        value = max(1, int(value))
        try:
            self.impl.record_value(value)
        except Exception:
            self.dropped += 1

    def value_at_percentile(self, p: float) -> int:
        return int(self.impl.get_value_at_percentile(p))

    def summary(self) -> dict:
        return {
            "count": int(self.impl.get_total_count()),
            "dropped": self.dropped,
            "min_ns": int(self.impl.get_min_value()),
            "max_ns": int(self.impl.get_max_value()),
            "mean_ns": float(self.impl.get_mean_value()),
            "stddev_ns": float(self.impl.get_stddev()),
            "p50_ns": self.value_at_percentile(50.0),
            "p90_ns": self.value_at_percentile(90.0),
            "p95_ns": self.value_at_percentile(95.0),
            "p99_ns": self.value_at_percentile(99.0),
            "p99_90_ns": self.value_at_percentile(99.9),
            "p99_99_ns": self.value_at_percentile(99.99),
        }

    def write_percentiles(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write("percentile,value_ns\n")
            for p in [0, 50, 75, 90, 95, 99, 99.9, 99.99, 100]:
                f.write(f"{p},{self.value_at_percentile(p)}\n")


def module_version(name: str) -> str | None:
    mod = import_optional(name)
    if isinstance(mod, Exception):
        return None
    return getattr(mod, "__version__", "unknown")


def base_record(args, name: str, status: str) -> dict:
    return {
        "name": name,
        "status": status,
        "clock": "time.perf_counter_ns",
        "histogram": "hdrhistogram",
        "warmup": args.warmup,
        "iters": args.iters,
        "samples": args.samples,
        "features": args.features,
        "ntrees": args.ntrees,
    }


def skip(args, results: list[dict], name: str, reason: str) -> None:
    if args.filter and not re.search(args.filter, name):
        return
    rec = base_record(args, name, "skipped")
    rec["reason"] = reason
    results.append(rec)
    print(f"[skip] {name}: {reason}", file=sys.stderr)


def measure(args, results: list[dict], name: str, predict: Callable[[int], float],
            extra: dict | None = None) -> None:
    if args.filter and not re.search(args.filter, name):
        return
    print(f"[run] {name}", file=sys.stderr)
    row = 0
    last = 0.0
    for _ in range(args.warmup):
        last = float(predict(row % args.samples))
        row += 1
    hist = Hdr(args.highest_ns, args.sigfig)
    begin = time.perf_counter_ns()
    for _ in range(args.iters):
        r = row % args.samples
        row += 1
        t0 = time.perf_counter_ns()
        last = float(predict(r))
        t1 = time.perf_counter_ns()
        hist.record(t1 - t0)
    elapsed = time.perf_counter_ns() - begin
    rec = base_record(args, name, "ok")
    rec["last_result"] = last
    rec["elapsed_ns"] = elapsed
    rec["latency"] = hist.summary()
    if extra:
        rec.update(extra)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    pct = Path(args.out_dir) / f"{stem}.percentiles.csv"
    hist.write_percentiles(pct)
    rec["percentiles_csv"] = str(pct)
    results.append(rec)


def xgboost_prefix_model(args, xgb, tmpdir: Path) -> str:
    if args.ntrees is None:
        return args.xgb_model
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)
    try:
        prefix = booster[: args.ntrees]
    except Exception as e:
        raise RuntimeError(f"XGBoost model slicing failed for --ntrees: {e}") from e
    out = tmpdir / f"prefix_n{args.ntrees}.ubj"
    prefix.save_model(str(out))
    return str(out)


def run_xgboost(args, results: list[dict], X: np.ndarray) -> None:
    name = f"xgboost/sklearn/threads:{args.threads}/output:{args.output}"
    xgb = import_optional("xgboost")
    if isinstance(xgb, Exception):
        skip(args, results, name, f"xgboost unavailable: {xgb}")
        return
    if not args.xgb_model:
        skip(args, results, name, "--xgb-model is required")
        return
    try:
        if args.xgb_kind == "regressor":
            model = xgb.XGBRegressor()
        else:
            model = xgb.XGBClassifier()
        model.load_model(args.xgb_model)
        model.set_params(n_jobs=args.threads)
    except Exception as e:
        skip(args, results, name, f"failed to load sklearn model: {e}")
        return

    iteration_range = (0, 0)
    if args.ntrees is not None:
        iteration_range = (0, args.ntrees)

    def predict(row: int) -> float:
        one = X[row : row + 1]
        if args.output == "margin":
            y = model.predict(one, output_margin=True,
                              iteration_range=iteration_range)
        elif args.output == "proba":
            y = model.predict_proba(one, iteration_range=iteration_range)
        else:
            y = model.predict(one, iteration_range=iteration_range)
        return float(np.asarray(y).ravel()[-1])

    measure(args, results, name, predict,
            {"engine": "xgboost_sklearn", "xgboost_version": xgb.__version__})


def load_treelite_model(treelite, xgb, xgb_model: str):
    frontend = getattr(treelite, "frontend", None)
    if frontend is not None and hasattr(frontend, "from_xgboost"):
        booster = xgb.Booster()
        booster.load_model(xgb_model)
        return frontend.from_xgboost(booster)
    for attr in ["load_xgboost_model", "load_xgboost_model_legacy_binary"]:
        fn = getattr(treelite, attr, None)
        if fn is not None:
            try:
                return fn(xgb_model)
            except Exception:
                pass
    frontend = getattr(treelite, "frontend", None)
    if frontend is not None and hasattr(frontend, "load_xgboost_model"):
        return frontend.load_xgboost_model(xgb_model)
    raise RuntimeError("could not find a compatible Treelite XGBoost loader")


def export_tl2cgen_lib(tl2cgen, model, libpath: Path) -> None:
    nthread = os.cpu_count() or 1
    tl2cgen.export_lib(model, toolchain="gcc", libpath=libpath,
                       params={"parallel_comp": nthread},
                       nthread=nthread, verbose=False)


def run_treelite(args, results: list[dict], X: np.ndarray, tmpdir: Path) -> None:
    name = (
        f"treelite/tl2cgen-lib/threads:{args.threads}"
        if args.treelite_lib or args.treelite_lib_out
        else f"treelite/tl2cgen/threads:{args.threads}"
    )
    treelite = import_optional("treelite")
    if isinstance(treelite, Exception):
        skip(args, results, name, f"treelite unavailable: {treelite}")
        return
    if not args.xgb_model and not args.treelite_lib:
        skip(args, results, name, "--xgb-model or --treelite-lib is required")
        return
    try:
        tl2cgen = import_optional("tl2cgen")
        if isinstance(tl2cgen, Exception):
            raise RuntimeError(f"tl2cgen unavailable: {tl2cgen}")
        libpath = Path(args.treelite_lib or args.treelite_lib_out or
                       (tmpdir / f"tl2cgen_n{args.ntrees or 'all'}.so"))
        compiled_now = False
        if args.treelite_lib and not libpath.exists():
            raise RuntimeError(f"--treelite-lib does not exist: {libpath}")
        if not args.treelite_lib and not libpath.exists():
            libpath.parent.mkdir(parents=True, exist_ok=True)
            xgb = import_optional("xgboost")
            if isinstance(xgb, Exception):
                raise RuntimeError(f"xgboost unavailable for prefixing: {xgb}")
            xgb_path = xgboost_prefix_model(args, xgb, tmpdir)
            tl_model = load_treelite_model(treelite, xgb, xgb_path)
            export_tl2cgen_lib(tl2cgen, tl_model, libpath)
            compiled_now = True
        predictor = tl2cgen.Predictor(libpath, nthread=args.threads,
                                      verbose=False)
        rows = [tl2cgen.DMatrix(X[i : i + 1]) for i in range(len(X))]

        def predict(row: int) -> float:
            y = predictor.predict(rows[row],
                                  pred_margin=(args.output == "margin"))
            return float(np.asarray(y).ravel()[-1])

        extra = {
            "engine": "treelite_tl2cgen",
            "treelite_version": module_version("treelite"),
            "tl2cgen_version": module_version("tl2cgen"),
            "compiled_lib": str(libpath),
            "compiled_now": compiled_now,
        }
        if args.treelite_compile_only:
            print(f"[compile] treelite TL2cgen library ready: {libpath}",
                  file=sys.stderr)
            return
    except Exception as e:
        skip(args, results, name, f"failed to prepare Treelite: {e}")
        return
    measure(args, results, name, predict, extra)


def run_fil(args, results: list[dict], X: np.ndarray, tmpdir: Path) -> None:
    cuml_fil = import_optional("cuml.fil")
    if isinstance(cuml_fil, Exception):
        skip(args, results, "fil/gpu", f"RAPIDS cuml.fil unavailable: {cuml_fil}")
        return
    if not args.xgb_model:
        skip(args, results, "fil/gpu", "--xgb-model is required")
        return
    try:
        xgb_model = args.xgb_model
        xgb = import_optional("xgboost")
        if args.ntrees is not None:
            if isinstance(xgb, Exception):
                raise RuntimeError(f"xgboost unavailable for prefixing: {xgb}")
            xgb_model = xgboost_prefix_model(args, xgb, tmpdir)
        ForestInference = cuml_fil.ForestInference
    except Exception as e:
        skip(args, results, "fil/gpu", f"failed to prepare FIL: {e}")
        return

    fil_cache = {}
    tl_model_cache = {"value": None}

    def load_fil(output_type: str):
        if output_type in fil_cache:
            return fil_cache[output_type]
        model_source = "file"
        try:
            model = ForestInference.load(
                xgb_model,
                is_classifier=(args.xgb_kind != "regressor"),
                layout=args.fil_layout,
                output_type=output_type,
                precision="single",
            )
        except Exception as direct_error:
            if isinstance(xgb, Exception):
                raise RuntimeError(
                    f"FIL direct load failed ({direct_error}); "
                    f"xgboost unavailable for Treelite fallback: {xgb}"
                )
            treelite = import_optional("treelite")
            if isinstance(treelite, Exception):
                raise RuntimeError(
                    f"FIL direct load failed ({direct_error}); "
                    f"treelite unavailable for fallback: {treelite}"
                )
            if tl_model_cache["value"] is None:
                booster = xgb.Booster()
                booster.load_model(xgb_model)
                frontend = getattr(treelite, "frontend", None)
                if frontend is not None and hasattr(frontend, "from_xgboost"):
                    tl_model_cache["value"] = frontend.from_xgboost(booster)
                else:
                    tl_model_cache["value"] = load_treelite_model(
                        treelite, xgb, xgb_model)
            model = ForestInference(
                treelite_model=tl_model_cache["value"],
                is_classifier=(args.xgb_kind != "regressor"),
                layout=args.fil_layout,
                output_type=output_type,
                precision="single",
            )
            model_source = "treelite"
        if args.fil_optimize:
            model.optimize(batch_size=1)
        fil_cache[output_type] = (model, model_source)
        return fil_cache[output_type]

    modes = [m.strip() for m in args.fil_modes.split(",") if m.strip()]
    if args.fil_input:
        modes = ["device_host" if args.fil_input == "device" else "host_host"]

    host_rows = [X[i : i + 1] for i in range(len(X))]
    cupy = None
    device_rows = None
    for mode in modes:
        if mode not in {"host_host", "device_host", "device_device_sync"}:
            skip(args, results, f"fil/gpu/mode:{mode}",
                 "unknown FIL mode")
            continue
        if mode != "host_host" and cupy is None:
            cupy = import_optional("cupy")
            if isinstance(cupy, Exception):
                skip(args, results, f"fil/gpu/mode:{mode}",
                     f"cupy unavailable: {cupy}")
                continue
            X_dev = cupy.asarray(X, dtype=cupy.float32)
            device_rows = [X_dev[i : i + 1] for i in range(len(X))]

        output_type = "numpy" if mode == "host_host" else "cupy"
        try:
            fil, model_source = load_fil(output_type)
        except Exception as e:
            skip(args, results, f"fil/gpu/mode:{mode}",
                 f"failed to prepare FIL: {e}")
            continue

        kwargs = {}
        if args.fil_chunk:
            kwargs["chunk_size"] = args.fil_chunk

        if mode == "host_host":
            name = "fil/gpu/input:host/output:host"
            rows = host_rows

            def predict(row: int) -> float:
                if args.output == "proba" and hasattr(fil, "predict_proba"):
                    y = fil.predict_proba(rows[row], **kwargs)
                else:
                    y = fil.predict(rows[row], **kwargs)
                return float(np.asarray(y).ravel()[-1])
        elif mode == "device_host":
            name = "fil/gpu/input:device/output:host"
            rows = device_rows

            def predict(row: int) -> float:
                if args.output == "proba" and hasattr(fil, "predict_proba"):
                    y = fil.predict_proba(rows[row], **kwargs)
                else:
                    y = fil.predict(rows[row], **kwargs)
                return float(cupy.asnumpy(y).ravel()[-1])
        else:
            name = "fil/gpu/input:device/output:device_sync"
            rows = device_rows

            def predict(row: int) -> float:
                if args.output == "proba" and hasattr(fil, "predict_proba"):
                    fil.predict_proba(rows[row], **kwargs)
                else:
                    fil.predict(rows[row], **kwargs)
                cupy.cuda.get_current_stream().synchronize()
                return 0.0

        measure(args, results, name, predict,
                {"engine": "fil", "cuml_version": module_version("cuml"),
                 "model_source": model_source, "fil_mode": mode,
                 "fil_layout": args.fil_layout,
                 "fil_chunk": args.fil_chunk,
                 "fil_optimize": args.fil_optimize})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=int, required=True)
    p.add_argument("--features-csv")
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out-dir", default="bench/results/latency_external")
    p.add_argument("--engines", default="xgboost,treelite,fil")
    p.add_argument("--filter", default="")
    p.add_argument("--highest-ns", type=int, default=60_000_000_000)
    p.add_argument("--sigfig", type=int, default=3)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--ntrees", type=int)
    p.add_argument("--xgb-model")
    p.add_argument("--xgb-kind", choices=["classifier", "regressor"],
                   default="classifier")
    p.add_argument("--output", choices=["margin", "value", "proba"],
                   default="margin")
    p.add_argument("--treelite-lib")
    p.add_argument("--treelite-lib-out")
    p.add_argument("--treelite-compile-only", action="store_true")
    p.add_argument("--fil-layout",
                   choices=["depth_first", "breadth_first", "layered"],
                   default="depth_first")
    p.add_argument("--fil-chunk", type=int)
    p.add_argument("--fil-modes",
                   default="host_host,device_host,device_device_sync",
                   help=("comma list: host_host, device_host, "
                         "device_device_sync"))
    p.add_argument("--fil-input", choices=["host", "device"],
                   help="legacy shortcut: host -> host_host, device -> device_host")
    p.add_argument("--fil-optimize", action="store_true")
    return p.parse_args()


def write_summary(args, results: list[dict]) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = {
        "meta": {
            "python": sys.version,
            "platform": platform.platform(),
            "clock": "time.perf_counter_ns",
            "histogram": "hdrhistogram",
            "xgboost": module_version("xgboost"),
            "treelite": module_version("treelite"),
            "tl2cgen": module_version("tl2cgen"),
            "cuml": module_version("cuml"),
            "cupy": module_version("cupy"),
            "features": args.features,
            "samples": args.samples,
            "warmup": args.warmup,
            "iters": args.iters,
            "ntrees": args.ntrees,
        },
        "runs": results,
    }
    with open(out / "summary.json", "w") as f:
        json.dump(root, f, indent=2)
        f.write("\n")
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "count", "p50_ns", "p90_ns", "p99_ns",
                    "p99_9_ns", "mean_ns", "max_ns", "reason"])
        for r in results:
            lat = r.get("latency", {})
            w.writerow([
                r.get("name", ""),
                r.get("status", ""),
                lat.get("count", 0),
                lat.get("p50_ns", 0),
                lat.get("p90_ns", 0),
                lat.get("p99_ns", 0),
                lat.get("p99_90_ns", 0),
                lat.get("mean_ns", 0),
                lat.get("max_ns", 0),
                r.get("reason", ""),
            ])


def main() -> int:
    args = parse_args()
    configure_cuda_python_env()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.features_csv:
        X = parse_csv(args.features_csv, args.features, args.samples)
    else:
        X = random_features(args.features, args.samples, args.seed)
    args.samples = len(X)
    results: list[dict] = []
    engines = {x.strip() for x in args.engines.split(",") if x.strip()}
    hdr_check = import_optional("hdrh.histogram")
    if isinstance(hdr_check, Exception):
        for engine in sorted(engines):
            skip(args, results, engine,
                 f"missing hdrhistogram package: {hdr_check}")
        write_summary(args, results)
        return 0
    with tempfile.TemporaryDirectory(prefix="qleaf_external_latency_") as td:
        tmpdir = Path(td)
        if "xgboost" in engines:
            run_xgboost(args, results, X)
        if "treelite" in engines:
            run_treelite(args, results, X, tmpdir)
        if "fil" in engines:
            run_fil(args, results, X, tmpdir)
    write_summary(args, results)
    print(f"[done] wrote {Path(args.out_dir) / 'summary.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
