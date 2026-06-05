#!/usr/bin/env bash
# Single-poller persistent-kernel sweep: config-0/1/2 x n, per dataset.
# Each run is its own process (the host poll_wait accumulators in
# drive_persistent_ are process-global statics). Emits a tidy CSV.
#
# Usage: bench/sweep_single_poller.sh [worker_bench] [out.csv]
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${1:-build-prof-bt256/worker_bench}"
OUT="${2:-bench/results/sweep/single_poller_sweep.csv}"
NS=(500 1000 2000 5000)

# dataset:features:trees_prefix:feats_prefix:iters
DATASETS=(
  "higgs:28:bench/data/higgs_d6_n:bench/data/higgs_d6_n:50000"
  "epsilon:2000:py/epsilon_d6_n:py/epsilon_d6_n:30000"
)
# label:filter
CONFIGS=(
  "config0:crossreduce:0/features"
  "config1:sp:1/stage:0"
  "config2:sp:1/stage:1"
)

mkdir -p "$(dirname "$OUT")"
echo "dataset,features,config,n,blocks,first_ns,rest_ns,poll_ns" > "$OUT"

for ds in "${DATASETS[@]}"; do
  IFS=: read -r name feats tpre fpre iters <<<"$ds"
  for cfg in "${CONFIGS[@]}"; do
    label="${cfg%%:*}"; filt="${cfg#*:}"
    for n in "${NS[@]}"; do
      blocks=$(( (n + 255) / 256 ))
      line=$("$BIN" --trees "${tpre}${n}.qleaf.json" --features "$feats" \
        --features-csv "${fpre}${n}.feats.csv" --buffers compact \
        --iters "$iters" --benchmark_filter="$filt" 2>&1 \
        | grep '\[host\]' | tail -1)
      first=$(sed -E 's/.*first=([0-9]+)ns.*/\1/' <<<"$line")
      rest=$(sed -E 's/.*rest=([0-9]+)ns.*/\1/' <<<"$line")
      poll=$(sed -E 's/.*poll=([0-9]+)ns.*/\1/' <<<"$line")
      echo "${name},${feats},${label},${n},${blocks},${first},${rest},${poll}"
      echo "${name},${feats},${label},${n},${blocks},${first},${rest},${poll}" >> "$OUT"
    done
  done
done
echo "wrote $OUT"
