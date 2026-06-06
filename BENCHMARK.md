# qleaf Batch-1 Benchmark Summary

This file is the compact benchmark narrative. The complete pinned HDR tables are
in [LATENCY_BENCHMARK.md](LATENCY_BENCHMARK.md), and the raw summaries are under
`bench/results/latency/sweep_pinned`.

The current headline numbers come from the HDR latency harness, not the older
google-benchmark `worker_bench` tables. That matters: the old document mixed
unpinned CPU runs, older GPU policy sweeps, and external-library timings from a
different harness. The pinned sweep is the fairer current comparison.

- Hardware: RTX 3060, 20 logical host CPUs, CUDA toolkit 12.0, Release builds.
- Forests: depth-6, `n in {500,1000,2000,5000}` trees.
- Datasets: `higgs` with 28 features, `epsilon` with 2000 features.
- Metric: batch-1 end-to-end latency in microseconds, HDR p50/p90/p99.
- Input stream: feature rows rotate every request, including warmup.
- Pinning: qleaf CPU used dispatcher CPU 16 and workers 0-15; qleaf GPU host
  timing used CPU 16; external processes used affinity 0-15.

## Headline n=5000

| config | higgs p50/p90/p99 us | epsilon p50/p90/p99 us |
|---|---:|---:|
| qleaf CPU traversal, 16 threads | 5.431 / 5.767 / 6.883 | 6.103 / 6.375 / 7.187 |
| qleaf CPU bitmask, 16 threads | 96.959 / 99.583 / 110.015 | 98.239 / 99.775 / 101.631 |
| qleaf GPU traversal, best persistent SP | 6.703 / 6.891 / 8.495 | 10.063 / 10.639 / 12.887 |
| qleaf GPU bitmask, best persistent SP | 11.351 / 11.823 / 14.183 | 15.063 / 16.087 / 20.607 |
| TL2cgen C API | 313.855 / 374.527 / 517.631 | 155.903 / 166.911 / 182.015 |
| XGBoost sklearn | 763.391 / 1545.215 / 3174.399 | 513.791 / 646.655 / 1340.415 |
| FIL GPU, device input/device sync | 1374.207 / 1771.519 / 2096.127 | 330.239 / 396.799 / 833.535 |

Bottom line: with explicit pinning, qleaf CPU traversal is the fastest batch-1
path in this sweep. The optimized persistent GPU traversal remains very close on
Higgs and is still the best non-CPU-qleaf path, but it no longer beats the
pinned 16-thread CPU traversal at n=5000.

## Fastest Family By Tree Count

| dataset | trees | winner | p50/p90/p99 us |
|---|---:|---|---:|
| higgs | 500 | qleaf CPU traversal | 1.908 / 2.181 / 2.395 |
| higgs | 1000 | qleaf CPU traversal | 1.769 / 1.977 / 2.151 |
| higgs | 2000 | qleaf CPU traversal | 2.513 / 2.727 / 3.255 |
| higgs | 5000 | qleaf CPU traversal | 5.431 / 5.767 / 6.883 |
| epsilon | 500 | qleaf CPU traversal | 1.358 / 1.609 / 1.764 |
| epsilon | 1000 | qleaf CPU traversal | 1.701 / 1.985 / 2.317 |
| epsilon | 2000 | qleaf CPU traversal | 2.757 / 3.035 / 3.395 |
| epsilon | 5000 | qleaf CPU traversal | 6.103 / 6.375 / 7.187 |

One Higgs n=2000 CPU traversal point in the first pinned run was an outlier
around 7.4 us p50. A targeted rerun of the same pinned config measured
2.513/2.727/3.255 us and is the value used here.

## GPU Tuning Takeaways

The persistent GPU path is still coordination-bound once the request-poll storm
is removed. The kernel launches once and then serves requests through a
host/device handshake, so the remaining floor is wait + compute + host fan-in,
not launch overhead.

For traversal, Higgs prefers smaller blocks at small and medium tree counts and
`BT=512` at n=5000:

| dataset | trees | best GPU traversal config | p50/p90/p99 us |
|---|---:|---|---:|
| higgs | 500 | `BT=256, transposed=1, seqflag=0` | 3.867 / 4.163 / 5.747 |
| higgs | 1000 | `BT=256, transposed=1, seqflag=0` | 3.879 / 4.199 / 5.239 |
| higgs | 2000 | `BT=256, transposed=1, seqflag=0` | 4.555 / 4.735 / 5.847 |
| higgs | 5000 | `BT=512, transposed=1, seqflag=0` | 6.703 / 6.891 / 8.495 |
| epsilon | 500 | `BT=1024, transposed=1, seqflag=0` | 7.799 / 8.575 / 11.775 |
| epsilon | 1000 | `BT=1024, transposed=1, seqflag=0` | 9.511 / 10.135 / 13.311 |
| epsilon | 2000 | `BT=512, transposed=1, seqflag=0` | 9.567 / 10.847 / 14.943 |
| epsilon | 5000 | `BT=1024, transposed=1, seqflag=0` | 10.063 / 10.639 / 12.887 |

For bitmask, intra-tree parallelism helps but does not overturn traversal. The
best n=5000 configs are `BT=512,TPT=4` for both datasets. The reason is the same
coordination vise seen in the Nsight runs: higher TPT reduces per-tree walk
time, but it grows the persistent grid and fan-in cost.

| dataset | trees | best GPU bitmask config | p50/p90/p99 us |
|---|---:|---|---:|
| higgs | 500 | `BT=1024, transposed=0, TPT=16` | 4.089 / 4.527 / 5.195 |
| higgs | 1000 | `BT=256, transposed=1, TPT=8` | 4.683 / 4.871 / 7.171 |
| higgs | 2000 | `BT=512, transposed=0, TPT=8` | 5.295 / 5.775 / 8.059 |
| higgs | 5000 | `BT=512, transposed=1, TPT=4` | 11.351 / 11.823 / 14.183 |
| epsilon | 500 | `BT=1024, transposed=0, TPT=16` | 7.187 / 9.639 / 13.239 |
| epsilon | 1000 | `BT=1024, transposed=0, TPT=16` | 7.411 / 9.511 / 12.567 |
| epsilon | 2000 | `BT=1024, transposed=0, TPT=8` | 8.535 / 10.287 / 12.247 |
| epsilon | 5000 | `BT=512, transposed=1, TPT=4` | 15.063 / 16.087 / 20.607 |

## CPU Takeaways

Path traversal is the right CPU algorithm. It follows one depth-6 path per tree;
bitmask evaluates all 63 internal nodes per tree. That is why the n=5000 CPU
bitmask p50 is about 97-98 us while traversal is about 5-6 us.

The pinned n=5000 p50 thread shape:

| dataset | worker | nothread | threads 1 | threads 2 | threads 4 | threads 8 | threads 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| higgs | traversal | 149.503 | 89.663 | 83.071 | 41.439 | 13.375 | 5.431 |
| higgs | bitmask | 1611.775 | 1192.959 | 723.967 | 365.823 | 189.567 | 96.959 |
| epsilon | traversal | 161.791 | 91.135 | 83.903 | 42.239 | 14.679 | 6.103 |
| epsilon | bitmask | 678.911 | 459.263 | 395.263 | 283.647 | 180.351 | 98.239 |

The `nothread` row in this pinned sweep ran inline on dispatcher CPU 16. It is a
control for this exact pinned command, not a best single-thread P-core baseline.

## External Libraries

TL2cgen C API is the strongest external CPU baseline, and for batch-1 inference
it generally prefers one thread. XGBoost sklearn has much larger p99 tails.

FIL was measured in three modes. Device-resident input and avoiding the final
host copy both help, but the current RAPIDS/cuML environment is especially slow
on Higgs at larger tree counts. Treat those FIL rows as current-environment
measurements rather than stable cross-version claims.

## Methodology Notes

The benchmark rotates feature rows every request. A constant input is
unrealistically favorable to persistent GPU paths because it can hide the
per-request feature write. With rotation, the mapped feature buffer is rewritten
for every request, matching the CPU and external serving pattern.

The qleaf CPU and GPU latency harnesses use `CLOCK_MONOTONIC_RAW`. XGBoost and
FIL use Python `time.perf_counter_ns`. All reported p90/p99 values come from
the same best-p50 config for the row, not independently selected tail-optimal
configs.

## Reproduce

Build:

```sh
cmake --build build-rel-bt256 --target worker_bench qleaf_latency
cmake --build build-rel-bt512 --target qleaf_latency
cmake --build build-rel-bt1024 --target qleaf_latency
cmake --build build-rel-bt256 --target external_latency_cpp
```

Pinned qleaf CPU:

```sh
ITERS=20000 WARMUP=2000 THREADS=1,2,4,8,16 DATASETS=higgs,epsilon \
CPU=1 GPU=0 RUN_EXTERNAL=0 \
PIN_CORES=16,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
QLEAF_OUT=qleaf_cpu_pinned \
bench/sweep_latency.sh build-rel-bt256/qleaf_latency bench/results/latency/sweep_pinned
```

Pinned qleaf GPU:

```sh
ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon CPU=0 GPU=1 RUN_EXTERNAL=0 \
GPU_SUITE=best BT=256 PIN_CORES=16 QLEAF_OUT=qleaf_gpu_bt256 \
bench/sweep_latency.sh build-rel-bt256/qleaf_latency bench/results/latency/sweep_pinned

ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon CPU=0 GPU=1 RUN_EXTERNAL=0 \
GPU_SUITE=best BT=512 PIN_CORES=16 QLEAF_OUT=qleaf_gpu_bt512 \
bench/sweep_latency.sh build-rel-bt512/qleaf_latency bench/results/latency/sweep_pinned

ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon CPU=0 GPU=1 RUN_EXTERNAL=0 \
GPU_SUITE=best BT=1024 PIN_CORES=16 QLEAF_OUT=qleaf_gpu_bt1024 \
bench/sweep_latency.sh build-rel-bt1024/qleaf_latency bench/results/latency/sweep_pinned
```

External CPU libraries:

```sh
CUPY_CACHE_DIR=/tmp/cupy-cache NUMBA_CACHE_DIR=/tmp/numba-cache \
CUDA_CACHE_PATH=/tmp/cuda-cache PYTHON=.venv/bin/python \
ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon EXT_THREADS=1,2,4,8,16 \
ENGINES=xgboost,treelite CPU=0 GPU=0 RUN_EXTERNAL=1 EXTERNAL_AFFINITY=0-15 \
QLEAF_OUT=qleaf_none \
bench/sweep_latency.sh build-rel-bt256/qleaf_latency bench/results/latency/sweep_pinned
```

FIL modes were run separately into `external_fil_modes`, then the aggregate table
was refreshed with:

```sh
.venv/bin/python bench/compare_latency.py bench/results/latency/sweep_pinned \
  > bench/results/latency/sweep_pinned/compare.txt
```
