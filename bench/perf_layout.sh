#!/usr/bin/env bash
# Measures cache traffic per inference for compact (SoA) vs tree (AoS) node
# buffers, branch worker, single-thread inline (nothread) path. Pinned to one
# P-core; counts cpu_core PMU events (cache miss COUNTS are frequency-proof).
set -euo pipefail

BIN=./build-release/worker_bench
CORE=2  # a P-core (HT pair 2-3)
EVENTS=cpu_core/instructions/,cpu_core/cycles/,cpu_core/L1-dcache-loads/,cpu_core/L1-dcache-load-misses/,cpu_core/LLC-loads/,cpu_core/LLC-load-misses/

run() {  # name  forest  csv  iters  buffer
  local name=$1 forest=$2 csv=$3 iters=$4 buf=$5
  echo "===== $name  buffer=$buf  iters=$iters ====="
  taskset -c $CORE perf stat -e "$EVENTS" \
    "$BIN" --trees "$forest" --features 28 --features-csv "$csv" \
    --samples 1024 --threads 1 --buffers "$buf" --iters "$iters" \
    --benchmark_filter="branch/$buf/nothread" 2>&1 \
    | grep -E "instructions|cycles|L1-dcache|LLC|elapsed|branch/" \
    | grep -v "items_per_second" || true
  echo
}

F_BIG=py/higgs_d6_n5000.qleaf.json;  C_BIG=py/higgs_d6_n5000.feats.csv
F_SM=higgs_d6_n500.qleaf.json;       C_SM=higgs_d6_n500.feats.csv

run "BIG n5000 (L3-resident ~7-10MB)" "$F_BIG" "$C_BIG" 30000 compact
run "BIG n5000 (L3-resident ~7-10MB)" "$F_BIG" "$C_BIG" 30000 tree
run "SMALL n500 (L2-resident ~0.7-1MB)" "$F_SM" "$C_SM" 400000 compact
run "SMALL n500 (L2-resident ~0.7-1MB)" "$F_SM" "$C_SM" 400000 tree
