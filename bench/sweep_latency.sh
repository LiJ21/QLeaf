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
#   EXTERNAL_AFFINITY=0-15 pins C++ external-library processes with taskset.
#   EXTERNAL_CPP_BIN=build-rel-bt256/external_latency_cpp for C++ external timing
#   TL2CGEN_RUNTIME=.venv/.../libtl2cgen.so overrides C++ TL2cgen runtime
#   TL2CGEN_LIB_ROOT=root containing $dataset/n$n/external/tl2cgen_n$n.so
#   XGBOOST_RUNTIME=.venv/.../libxgboost.so overrides C++ XGBoost runtime
#   NVFOREST_LAYOUTS=depth_first,layered,breadth_first
#   NVFOREST_DEVICES=gpu,cpu
#   NVFOREST_ROWS_PER_BLOCK_ITER=1,2,4,8,16,32
#   NVFOREST_CPU_CHUNK_SIZES=64
#   NVFOREST_ALIGN_BYTES=0,128
#   NVFOREST_MODES=host_host,device_host,device_device_sync
#   PYTHON=python3 selects the Python used by compare_latency.py.
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
NVFOREST_LAYOUTS="${NVFOREST_LAYOUTS:-depth_first,layered,breadth_first}"
NVFOREST_DEVICES="${NVFOREST_DEVICES:-gpu,cpu}"
NVFOREST_ROWS_PER_BLOCK_ITER="${NVFOREST_ROWS_PER_BLOCK_ITER:-1,2,4,8,16,32}"
NVFOREST_CPU_CHUNK_SIZES="${NVFOREST_CPU_CHUNK_SIZES:-64}"
NVFOREST_ALIGN_BYTES="${NVFOREST_ALIGN_BYTES:-0,128}"
NVFOREST_MODES="${NVFOREST_MODES:-host_host,device_host,device_device_sync}"
EXTERNAL_CPP_BIN="${EXTERNAL_CPP_BIN:-$(dirname "$BIN")/external_latency_cpp}"
TL2CGEN_RUNTIME="${TL2CGEN_RUNTIME:-.venv/lib/python3.12/site-packages/tl2cgen/lib/libtl2cgen.so}"
TL2CGEN_LIB_ROOT="${TL2CGEN_LIB_ROOT:-}"
XGBOOST_RUNTIME="${XGBOOST_RUNTIME:-.venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.so}"
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

want_nvforest_device() {
  local needle="$1"
  local device
  IFS=',' read -r -a _nvforest_devices <<< "$NVFOREST_DEVICES"
  for device in "${_nvforest_devices[@]}"; do
    if [[ "$device" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

run_external_cmd() {
  if [[ -n "$EXTERNAL_AFFINITY" ]]; then
    taskset -c "$EXTERNAL_AFFINITY" "$@"
  else
    "$@"
  fi
}

tl2cgen_lib_for() {
  local ds="$1"
  local n="$2"
  local eout="$3"
  local name="tl2cgen_n$n.so"
  local candidates=()
  if [[ -n "$TL2CGEN_LIB_ROOT" ]]; then
    candidates+=(
      "$TL2CGEN_LIB_ROOT/$ds/n$n/external/$name"
      "$TL2CGEN_LIB_ROOT/$ds/n$n/$name"
      "$TL2CGEN_LIB_ROOT/$name"
    )
  fi
  candidates+=("$eout/$name")
  local path
  for path in "${candidates[@]}"; do
    if [[ -f "$path" ]]; then
      echo "$path"
      return 0
    fi
  done
  return 1
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
	      trees="bench/data/epsilon_d6_n5000.qleaf.json"
	      feats="bench/data/epsilon_d6_n5000.feats.csv"
	      xgb="bench/data/epsilon_d6_n5000.xgb.json"
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
      local use_cpp_xgboost=0
      local use_cpp_treelite=0
      local tl2cgen_lib="$eout/tl2cgen_n$n.so"
      if want_engine treelite && [[ -x "$EXTERNAL_CPP_BIN" ]]; then
        if tl2cgen_lib="$(tl2cgen_lib_for "$ds" "$n" "$eout")"; then
          use_cpp_treelite=1
        else
          echo "[skip] $ds n$n treelite: no TL2cgen library found; set TL2CGEN_LIB_ROOT" >&2
        fi
      fi
      if want_engine treelite && [[ ! -x "$EXTERNAL_CPP_BIN" ]]; then
        echo "[skip] $ds n$n treelite: EXTERNAL_CPP_BIN is not executable: $EXTERNAL_CPP_BIN" >&2
      fi
      if want_engine xgboost && [[ -x "$EXTERNAL_CPP_BIN" && -f "$XGBOOST_RUNTIME" ]]; then
        use_cpp_xgboost=1
      fi
      if want_engine xgboost && [[ "$use_cpp_xgboost" != "1" ]]; then
        echo "[skip] $ds n$n xgboost: need executable EXTERNAL_CPP_BIN and XGBOOST_RUNTIME=$XGBOOST_RUNTIME" >&2
      fi
      if want_engine nvforest && [[ ! -x "$EXTERNAL_CPP_BIN" ]]; then
        echo "[skip] $ds n$n nvforest: EXTERNAL_CPP_BIN is not executable: $EXTERNAL_CPP_BIN" >&2
      fi
      if want_engine nvforest && want_nvforest_device gpu && [[ -x "$EXTERNAL_CPP_BIN" ]]; then
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
          --nvforest-devices gpu \
          --nvforest-rows-per-block-iter "$NVFOREST_ROWS_PER_BLOCK_ITER" \
          --nvforest-cpu-chunk-sizes "$NVFOREST_CPU_CHUNK_SIZES" \
          --nvforest-align-bytes "$NVFOREST_ALIGN_BYTES" \
          --nvforest-modes "$NVFOREST_MODES" \
          --out-dir "$eout/nvforest"
      fi
      for th in "${ext_threads[@]}"; do
        if [[ "$use_cpp_xgboost" == "1" ]]; then
          run_external_cmd "$EXTERNAL_CPP_BIN" \
            --engine xgboost \
            --features "$features" \
            --features-csv "$feats" \
            --samples "$SAMPLES" \
            --ntrees "$n" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --threads "$th" \
            --xgb-model "$xgb" \
            --xgboost-runtime "$XGBOOST_RUNTIME" \
            --out-dir "$eout/threads_$th/xgboost_capi"
        fi
        if want_engine nvforest && want_nvforest_device cpu && [[ -x "$EXTERNAL_CPP_BIN" ]]; then
          run_external_cmd "$EXTERNAL_CPP_BIN" \
            --engine nvforest \
            --features "$features" \
            --features-csv "$feats" \
            --samples "$SAMPLES" \
            --ntrees "$n" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --threads "$th" \
            --xgb-model "$xgb" \
            --nvforest-layouts "$NVFOREST_LAYOUTS" \
            --nvforest-devices cpu \
            --nvforest-rows-per-block-iter "$NVFOREST_ROWS_PER_BLOCK_ITER" \
            --nvforest-cpu-chunk-sizes "$NVFOREST_CPU_CHUNK_SIZES" \
            --nvforest-align-bytes "$NVFOREST_ALIGN_BYTES" \
            --nvforest-modes "$NVFOREST_MODES" \
            --out-dir "$eout/threads_$th/nvforest_cpu"
        fi
        if [[ "$use_cpp_treelite" == "1" ]]; then
          local tl2cgen_runtime_args=()
          if [[ -n "$TL2CGEN_RUNTIME" && -f "$TL2CGEN_RUNTIME" ]]; then
            tl2cgen_runtime_args=(--tl2cgen-runtime "$TL2CGEN_RUNTIME")
          fi
          run_external_cmd "$EXTERNAL_CPP_BIN" \
            --features "$features" \
            --features-csv "$feats" \
            --samples "$SAMPLES" \
            --ntrees "$n" \
            --warmup "$WARMUP" \
            --iters "$ITERS" \
            --threads "$th" \
            --tl2cgen-lib "$tl2cgen_lib" \
            "${tl2cgen_runtime_args[@]}" \
            --out-dir "$eout/threads_$th/tl2cgen_capi"
        fi
      done
    else
      mkdir -p "$eout"
      echo "[skip] $ds n$n external: missing XGBoost model $xgb" >&2
    fi
  done
}

IFS=',' read -r -a datasets <<< "$DATASETS"
for ds in "${datasets[@]}"; do
  run_dataset "$ds"
done

"$PYTHON" bench/compare_latency.py "$OUT" | tee "$OUT/compare.txt"
