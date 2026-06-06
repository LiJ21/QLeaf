#!/usr/bin/env bash
set -euo pipefail

# Batch-1 latency histogram sweep. It reuses the same qleaf/XGBoost artifacts as
# BENCHMARK.md and records per-config HdrHistogram outputs under OUT.
#
# Usage:
#   bench/sweep_latency.sh [qleaf_latency_bin] [out_dir]
#
# Environment overrides:
#   ITERS=20000 WARMUP=2000 BT=256,512,1024 THREADS=1,2,4,8,16
#   EXT_THREADS=1,2,4,8,16 DATASETS=higgs,epsilon ENGINES=xgboost,treelite,nvforest
#   CPU=1 GPU=1 RUN_EXTERNAL=1 GPU_SUITE=best QLEAF_OUT=qleaf
#   PIN_CORES=dispatch,worker0,... passes explicit qleaf CPU pinning.
#   EXTERNAL_AFFINITY=0-15 pins Python/C++ external-library processes with taskset.
#   FIL_OPTIMIZE=0 passes --fil-optimize to RAPIDS FIL when set to 1
#   EXTERNAL_CPP_BIN=build-rel-bt256/external_latency_cpp for TL2cgen C API timing
#   NVFOREST_LAYOUTS=depth_first,layered,breadth_first
#   NVFOREST_ROWS_PER_BLOCK_ITER=1,2,4,8,16,32
#   NVFOREST_ALIGN_BYTES=0,128
#   NVFOREST_MODES=host_host,device_host,device_device_sync
#   PYTHON=python3 selects the Python used by external_latency.py.
#
# Build qleaf_latency with -DQLEAF_LATENCY_DECOMP to emit extra host-side phase
# HDR files (for qleaf GPU: mapped input load and predictor call).

BIN="${1:-build-rel-bt256/qleaf_latency}"
OUT="${2:-bench/results/latency/sweep}"
ITERS="${ITERS:-20000}"
WARMUP="${WARMUP:-2000}"
BT="${BT:-256,512,1024}"
TPT="${TPT:-1,2,4,8,16,32}"
THREADS="${THREADS:-1,2,4,8,16}"
EXT_THREADS="${EXT_THREADS:-$THREADS}"
DATASETS="${DATASETS:-higgs,epsilon}"
ENGINES="${ENGINES:-xgboost,treelite,nvforest}"
SAMPLES="${SAMPLES:-1024}"
CPU="${CPU:-1}"
GPU="${GPU:-1}"
RUN_EXTERNAL="${RUN_EXTERNAL:-1}"
GPU_SUITE="${GPU_SUITE:-best}"
QLEAF_OUT="${QLEAF_OUT:-qleaf}"
FIL_OPTIMIZE="${FIL_OPTIMIZE:-0}"
NVFOREST_LAYOUTS="${NVFOREST_LAYOUTS:-depth_first,layered,breadth_first}"
NVFOREST_ROWS_PER_BLOCK_ITER="${NVFOREST_ROWS_PER_BLOCK_ITER:-1,2,4,8,16,32}"
NVFOREST_ALIGN_BYTES="${NVFOREST_ALIGN_BYTES:-0,128}"
NVFOREST_MODES="${NVFOREST_MODES:-host_host,device_host,device_device_sync}"
EXTERNAL_CPP_BIN="${EXTERNAL_CPP_BIN:-$(dirname "$BIN")/external_latency_cpp}"
PIN_CORES="${PIN_CORES:-}"
EXTERNAL_AFFINITY="${EXTERNAL_AFFINITY:-}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT"

want_engine() {
  local needle="$1"
  local engine
  IFS=',' read -r -a _engines <<< "$ENGINES"
  for engine in "${_engines[@]}"; do
    if [[ "$engine" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

python_external_engines() {
  local use_cpp_treelite="${1:-0}"
  local engines=()
  if want_engine xgboost; then
    engines+=(xgboost)
  fi
  if want_engine treelite && [[ "$use_cpp_treelite" != "1" ]]; then
    engines+=(treelite)
  fi
  if want_engine fil; then
    engines+=(fil)
  fi
  local IFS=,
  echo "${engines[*]}"
}

run_external_cmd() {
  if [[ -n "$EXTERNAL_AFFINITY" ]]; then
    taskset -c "$EXTERNAL_AFFINITY" "$@"
  else
    "$@"
  fi
}

run_dataset() {
  local ds="$1"
  local features trees feats xgb
  case "$ds" in
    higgs)
      features=28
      trees="bench/data/higgs_d6_n5000.qleaf.json"
      feats="bench/data/higgs_d6_n5000.feats.csv"
      xgb="bench/data/higgs_d6_n5000.xgb.json"
      ;;
    epsilon)
      features=2000
      trees="py/epsilon_d6_n5000.qleaf.json"
      feats="py/epsilon_d6_n5000.feats.csv"
      xgb="py/epsilon_d6_n5000.xgb.json"
      ;;
    *)
      echo "unknown dataset '$ds'" >&2
      exit 1
      ;;
  esac

  if [[ ! -f "$trees" || ! -f "$feats" ]]; then
    echo "[skip] $ds: missing $trees or $feats" >&2
    return
  fi

  for n in 500 1000 2000 5000; do
	  local qout="$OUT/$ds/n$n/$QLEAF_OUT"
	  local pin_args=()
	  if [[ -n "$PIN_CORES" ]]; then
	    pin_args=(--pin-cores "$PIN_CORES")
	  fi
	  "$BIN" \
	    --trees "$trees" \
      --features "$features" \
      --features-csv "$feats" \
      --samples "$SAMPLES" \
      --ntrees "$n" \
      --warmup "$WARMUP" \
      --iters "$ITERS" \
      --threads "$THREADS" \
      --buffers compact \
      --bt "$BT" \
      --tpt "$TPT" \
	    --cpu "$CPU" \
	    --gpu "$GPU" \
	    --gpu-suite "$GPU_SUITE" \
	    "${pin_args[@]}" \
	    --out-dir "$qout"

    local eout="$OUT/$ds/n$n/external"
    if [[ "$RUN_EXTERNAL" == "0" ]]; then
      continue
    fi
    if [[ -f "$xgb" ]]; then
      IFS=',' read -r -a ext_threads <<< "$EXT_THREADS"
      local use_cpp_treelite=0
      local tl2cgen_lib="$eout/tl2cgen_n$n.so"
      if want_engine treelite && [[ -x "$EXTERNAL_CPP_BIN" ]]; then
        use_cpp_treelite=1
        run_external_cmd "$PYTHON" bench/external_latency.py \
          --features "$features" \
          --features-csv "$feats" \
          --samples "$SAMPLES" \
          --ntrees "$n" \
          --warmup 1 \
          --iters 1 \
          --threads 1 \
          --engines treelite \
          --xgb-model "$xgb" \
          --treelite-lib-out "$tl2cgen_lib" \
          --treelite-compile-only \
          --out-dir "$eout/tl2cgen_compile"
      fi
      if want_engine nvforest && [[ -x "$EXTERNAL_CPP_BIN" ]]; then
        run_external_cmd "$EXTERNAL_CPP_BIN" \
          --engine nvforest \
          --features "$features" \
          --features-csv "$feats" \
          --samples "$SAMPLES" \
          --ntrees "$n" \
          --warmup "$WARMUP" \
          --iters "$ITERS" \
          --xgb-model "$xgb" \
          --nvforest-layouts "$NVFOREST_LAYOUTS" \
          --nvforest-rows-per-block-iter "$NVFOREST_ROWS_PER_BLOCK_ITER" \
          --nvforest-align-bytes "$NVFOREST_ALIGN_BYTES" \
          --nvforest-modes "$NVFOREST_MODES" \
          --out-dir "$eout/nvforest"
      elif want_engine nvforest; then
        echo "[skip] $ds n$n nvforest: EXTERNAL_CPP_BIN is not executable: $EXTERNAL_CPP_BIN" >&2
      fi
      local py_engines
      py_engines="$(python_external_engines "$use_cpp_treelite")"
      for th in "${ext_threads[@]}"; do
        fil_opt=()
        if [[ "$FIL_OPTIMIZE" == "1" ]]; then
          fil_opt=(--fil-optimize)
        fi
        if [[ -n "$py_engines" ]]; then
          run_external_cmd "$PYTHON" bench/external_latency.py \
            --features "$features" \
            --features-csv "$feats" \
            --samples "$SAMPLES" \
            --ntrees "$n" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --threads "$th" \
            --engines "$py_engines" \
            --xgb-model "$xgb" \
            --treelite-lib-out "$tl2cgen_lib" \
            "${fil_opt[@]}" \
            --out-dir "$eout/threads_$th"
        fi
        if [[ "$use_cpp_treelite" == "1" ]]; then
          run_external_cmd "$EXTERNAL_CPP_BIN" \
            --features "$features" \
            --features-csv "$feats" \
            --samples "$SAMPLES" \
            --ntrees "$n" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --threads "$th" \
            --tl2cgen-lib "$tl2cgen_lib" \
            --out-dir "$eout/threads_$th/tl2cgen_capi"
        fi
      done
    else
      mkdir -p "$eout"
      fil_opt=()
      if [[ "$FIL_OPTIMIZE" == "1" ]]; then
        fil_opt=(--fil-optimize)
      fi
      run_external_cmd "$PYTHON" bench/external_latency.py \
        --features "$features" \
        --features-csv "$feats" \
        --samples "$SAMPLES" \
        --ntrees "$n" \
        --warmup 1 \
        --iters 1 \
        --engines "$ENGINES" \
        --treelite-lib-out "$eout/tl2cgen_n$n.so" \
        "${fil_opt[@]}" \
        --out-dir "$eout"
    fi
  done
}

IFS=',' read -r -a datasets <<< "$DATASETS"
for ds in "${datasets[@]}"; do
  run_dataset "$ds"
done

"$PYTHON" bench/compare_latency.py "$OUT" | tee "$OUT/compare.txt"
