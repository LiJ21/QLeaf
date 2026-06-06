# Batch-1 Latency Benchmark

This report summarizes the pinned HDR-histogram batch-1 latency sweep in
`bench/results/latency/sweep_pinned`. The benchmark rotates feature rows on
every request, including warmup, so the measured stream is not a constant-input
cache artifact.

## Scope

Datasets:

- `higgs`: depth-6, 28 features.
- `epsilon`: depth-6, 2000 features.

Tree counts: `500`, `1000`, `2000`, `5000`, selected from each 5000-tree model.

Families compared:

- qleaf CPU traversal and bitmask, compact node buffer, threaded sweep.
- qleaf GPU persistent traversal and bitmask over `QLEAF_BT in {256,512,1024}`,
  plus cached persistent traversal over `QLEAF_BT in {64,96}` and a
  reduced-grid `BT=32` CacheAllWaves diagnostic.
- XGBoost sklearn predictor, `output_margin=True`.
- Treelite/TL2cgen compiled C predictor, timed through the C API harness.
- nvForest GPU, timed through the C++ harness in three input/output modes,
  with `layout in {depth_first, layered, breadth_first}`,
  `rows_per_block_iter in {1,2,4,8,16,32}`, and
  `align_bytes in {0,128}`.

The qleaf CPU benchmark now uses `qleaf::Inferrer` for the threaded and inline
paths. The CPU topology has eight SMT P-core pairs
`0/1,2/3,4/5,6/7,8/9,10/11,12/13,14/15` plus four single-thread E-cores
`16,17,18,19`. qleaf CPU rows in this report use a physical-core-first worker
order:
`--pin-cores 19,0,2,4,6,8,10,12,14,16,17,18,1,3,5,7,9`, meaning the dispatch
thread used CPU 19, workers first filled one logical CPU per available worker
core, and only then used SMT siblings. Older sibling-first CPU result
directories remain under `sweep_pinned` but are not used for the CPU rows
below. qleaf GPU host timing used `--pin-cores 16`. External Python/C++
processes used `taskset -c 0-15`.

The current machine was an RTX 3060 host with CUDA toolkit 12.0. The qleaf
result tree was produced from `build-rel-bt32`, `build-rel-bt64`,
`build-rel-bt96`, `build-rel-bt256`, `build-rel-bt512`, and
`build-rel-bt1024` Release builds. The nvForest C++ harness was built in
`build-rel-bt256-gcc14` because the RAPIDS nvForest wheel requires a GCC 14 C++
ABI-compatible build.

## Methodology

Per-sample latency is recorded into HDR histograms. The qleaf, TL2cgen C++,
and nvForest C++ runners use `CLOCK_MONOTONIC_RAW`; the external Python runner
for XGBoost uses `time.perf_counter_ns`.

Each reported histogram uses 20,000 measured repetitions after 2,000 warmup
requests, with 1,024 feature rows rotated by request index.

Feature rotation is the same serving pattern as `worker_bench`: row `i` uses
`features[i % samples]`. Warmup consumes rows from the same counter, and
measured requests continue from there. For persistent mapped GPU policies, the
resident kernel is bound to one mapped host row, so the benchmark rewrites that
mapped row with the next feature vector before every request.

Model truncation is handled per backend:

- qleaf truncates the qleaf JSON forest before constructing the inferrer.
- XGBoost sklearn uses `iteration_range=(0, ntrees)`.
- Treelite/TL2cgen and nvForest use an XGBoost prefix model for the selected
  tree count. The nvForest C++ harness reconstructs a numeric Treelite model
  from that JSON prefix before importing it into nvForest.

Selection rule: each table row is the best p50 configuration within that family
for the dataset and tree count. The p90, p99, and mean columns are from that
same configuration.

## Fastest Family By p50

| dataset | trees | fastest p50 family | p50 us | p90 us | p99 us | config |
|---|---:|---|---:|---:|---:|---|
| higgs | 500 | qleaf CPU traversal | 2.509 | 2.731 | 2.881 | `qleaf/cpu/compact/branch/threads:16` |
| higgs | 1000 | qleaf CPU traversal | 1.555 | 1.697 | 2.565 | `qleaf/cpu/compact/branch/threads:8` |
| higgs | 2000 | qleaf CPU traversal | 2.593 | 3.193 | 4.775 | `qleaf/cpu/compact/branch/threads:8` |
| higgs | 5000 | qleaf CPU traversal | 5.719 | 6.139 | 10.831 | `qleaf/cpu/compact/branch/threads:8` |
| epsilon | 500 | qleaf CPU traversal | 1.235 | 1.456 | 1.814 | `qleaf/cpu/compact/branch/threads:8` |
| epsilon | 1000 | qleaf CPU traversal | 1.767 | 1.994 | 2.467 | `qleaf/cpu/compact/branch/threads:8` |
| epsilon | 2000 | qleaf CPU traversal | 2.791 | 4.859 | 6.691 | `qleaf/cpu/compact/branch/threads:8` |
| epsilon | 5000 | qleaf CPU traversal | 7.207 | 11.959 | 18.239 | `qleaf/cpu/compact/branch/threads:8` |

## Higgs

| trees | family | p50 us | p90 us | p99 us | mean us | best config |
|---:|---|---:|---:|---:|---:|---|
| 500 | qleaf CPU traversal | 2.509 | 2.731 | 2.881 | 2.528 | `qleaf/cpu/compact/branch/threads:16` |
| 500 | qleaf CPU bitmask | 11.279 | 11.759 | 13.455 | 11.728 | `qleaf/cpu/compact/bitmask/threads:16` |
| 500 | qleaf GPU traversal SP | 3.867 | 4.163 | 5.747 | 4.256 | `qleaf/gpu/persistent/compact/bt:256/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU traversal SP cached one-wave | 3.107 | 3.411 | 4.523 | 3.209 | `qleaf/gpu/persistent/compact/bt:96/transposed:1/bitmask:0/cached:1/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU traversal SP cached all-waves | 3.203 | 3.459 | 4.251 | 3.290 | `qleaf/gpu/persistent/compact/bt:96/transposed:0/bitmask:0/cached:1/mapped:1/cacheallwaves:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU bitmask | 4.089 | 4.527 | 5.195 | 4.603 | `qleaf/gpu/persistent/compact/bt:1024/transposed:0/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:16` |
| 500 | XGBoost sklearn | 146.175 | 154.623 | 260.991 | 150.128 | `xgboost/sklearn/threads:8/output:margin` |
| 500 | Treelite | 10.039 | 12.575 | 16.527 | 10.386 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 500 | nvForest GPU | 10.735 | 11.527 | 15.903 | 11.401 | `nvforest/gpu/input:device/output:device_sync/layout:breadth_first/rows_per_block_iter:1/align_bytes:128` |
| 1000 | qleaf CPU traversal | 1.555 | 1.697 | 2.565 | 1.603 | `qleaf/cpu/compact/branch/threads:8` |
| 1000 | qleaf CPU bitmask | 22.607 | 23.439 | 25.551 | 22.313 | `qleaf/cpu/compact/bitmask/threads:16` |
| 1000 | qleaf GPU traversal SP | 3.879 | 4.199 | 5.239 | 4.132 | `qleaf/gpu/persistent/compact/bt:256/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU traversal SP cached one-wave | 3.499 | 3.631 | 5.279 | 3.672 | `qleaf/gpu/persistent/compact/bt:96/transposed:1/bitmask:0/cached:1/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU traversal SP cached all-waves | 3.515 | 3.617 | 5.211 | 3.733 | `qleaf/gpu/persistent/compact/bt:96/transposed:0/bitmask:0/cached:1/mapped:1/cacheallwaves:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU bitmask | 4.683 | 4.871 | 7.171 | 4.972 | `qleaf/gpu/persistent/compact/bt:256/transposed:1/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:8` |
| 1000 | XGBoost sklearn | 186.239 | 202.239 | 314.111 | 192.662 | `xgboost/sklearn/threads:8/output:margin` |
| 1000 | Treelite | 32.287 | 39.135 | 49.023 | 33.056 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 1000 | nvForest GPU | 14.719 | 16.039 | 20.639 | 15.667 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:128` |
| 2000 | qleaf CPU traversal | 2.593 | 3.193 | 4.775 | 2.743 | `qleaf/cpu/compact/branch/threads:8` |
| 2000 | qleaf CPU bitmask | 44.703 | 46.495 | 65.535 | 49.093 | `qleaf/cpu/compact/bitmask/threads:16` |
| 2000 | qleaf GPU traversal SP | 4.555 | 4.735 | 5.847 | 4.718 | `qleaf/gpu/persistent/compact/bt:256/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 2000 | qleaf GPU traversal SP cached one-wave | 3.693 | 3.895 | 5.487 | 3.938 | `qleaf/gpu/persistent/compact/bt:96/transposed:1/bitmask:0/cached:1/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 2000 | qleaf GPU traversal SP cached all-waves | 3.721 | 3.889 | 5.519 | 3.947 | `qleaf/gpu/persistent/compact/bt:96/transposed:0/bitmask:0/cached:1/mapped:1/cacheallwaves:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 2000 | qleaf GPU bitmask | 5.295 | 5.775 | 8.059 | 5.970 | `qleaf/gpu/persistent/compact/bt:512/transposed:0/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:8` |
| 2000 | XGBoost sklearn | 261.375 | 281.599 | 388.863 | 267.144 | `xgboost/sklearn/threads:8/output:margin` |
| 2000 | Treelite | 98.367 | 112.511 | 131.327 | 99.255 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 2000 | nvForest GPU | 21.743 | 23.951 | 32.319 | 23.145 | `nvforest/gpu/input:device/output:device_sync/layout:breadth_first/rows_per_block_iter:1/align_bytes:128` |
| 5000 | qleaf CPU traversal | 5.719 | 6.139 | 10.831 | 6.054 | `qleaf/cpu/compact/branch/threads:8` |
| 5000 | qleaf CPU bitmask | 110.079 | 114.815 | 119.039 | 108.288 | `qleaf/cpu/compact/bitmask/threads:16` |
| 5000 | qleaf GPU traversal SP | 6.703 | 6.891 | 8.495 | 6.957 | `qleaf/gpu/persistent/compact/bt:512/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 5000 | qleaf GPU traversal SP cached one-wave | skipped | skipped | skipped | skipped | persistent grid not co-resident |
| 5000 | qleaf GPU traversal SP cached all-waves | skipped | skipped | skipped | skipped | persistent grid not co-resident |
| 5000 | qleaf GPU bitmask | 11.351 | 11.823 | 14.183 | 11.961 | `qleaf/gpu/persistent/compact/bt:512/transposed:1/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:4` |
| 5000 | XGBoost sklearn | 763.391 | 1545.215 | 3174.399 | 992.046 | `xgboost/sklearn/threads:8/output:margin` |
| 5000 | Treelite | 313.855 | 374.527 | 517.631 | 323.230 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 5000 | nvForest GPU | 66.815 | 68.543 | 88.703 | 69.234 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:128` |

## Epsilon

| trees | family | p50 us | p90 us | p99 us | mean us | best config |
|---:|---|---:|---:|---:|---:|---|
| 500 | qleaf CPU traversal | 1.235 | 1.456 | 1.814 | 1.251 | `qleaf/cpu/compact/branch/threads:8` |
| 500 | qleaf CPU bitmask | 11.951 | 12.215 | 13.855 | 11.997 | `qleaf/cpu/compact/bitmask/threads:16` |
| 500 | qleaf GPU traversal SP | 7.799 | 8.575 | 11.775 | 8.509 | `qleaf/gpu/persistent/compact/bt:1024/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU traversal SP cached one-wave | 44.575 | 48.031 | 258.175 | 50.284 | `qleaf/gpu/persistent/compact/bt:64/transposed:1/bitmask:0/cached:1/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU traversal SP cached all-waves | 44.575 | 48.255 | 258.303 | 50.314 | `qleaf/gpu/persistent/compact/bt:64/transposed:0/bitmask:0/cached:1/mapped:1/cacheallwaves:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 500 | qleaf GPU bitmask | 7.187 | 9.639 | 13.239 | 8.757 | `qleaf/gpu/persistent/compact/bt:1024/transposed:0/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:16` |
| 500 | XGBoost sklearn | 149.119 | 159.999 | 264.191 | 153.580 | `xgboost/sklearn/threads:8/output:margin` |
| 500 | Treelite | 23.055 | 25.503 | 38.687 | 23.506 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 500 | nvForest GPU | 13.095 | 13.703 | 17.551 | 13.730 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:0` |
| 1000 | qleaf CPU traversal | 1.767 | 1.994 | 2.467 | 1.778 | `qleaf/cpu/compact/branch/threads:8` |
| 1000 | qleaf CPU bitmask | 22.847 | 23.343 | 24.751 | 22.892 | `qleaf/cpu/compact/bitmask/threads:16` |
| 1000 | qleaf GPU traversal SP | 9.511 | 10.135 | 13.311 | 10.626 | `qleaf/gpu/persistent/compact/bt:1024/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU traversal SP cached one-wave | 45.023 | 48.063 | 259.967 | 50.401 | `qleaf/gpu/persistent/compact/bt:64/transposed:1/bitmask:0/cached:1/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU traversal SP cached all-waves | 45.055 | 48.415 | 259.967 | 50.531 | `qleaf/gpu/persistent/compact/bt:64/transposed:0/bitmask:0/cached:1/mapped:1/cacheallwaves:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 1000 | qleaf GPU bitmask | 7.411 | 9.511 | 12.567 | 8.439 | `qleaf/gpu/persistent/compact/bt:1024/transposed:0/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:16` |
| 1000 | XGBoost sklearn | 186.367 | 196.479 | 303.359 | 190.236 | `xgboost/sklearn/threads:8/output:margin` |
| 1000 | Treelite | 50.495 | 55.103 | 66.687 | 51.276 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 1000 | nvForest GPU | 16.479 | 17.247 | 22.031 | 17.374 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:0` |
| 2000 | qleaf CPU traversal | 2.791 | 4.859 | 6.691 | 3.247 | `qleaf/cpu/compact/branch/threads:8` |
| 2000 | qleaf CPU bitmask | 40.927 | 41.695 | 43.391 | 41.012 | `qleaf/cpu/compact/bitmask/threads:16` |
| 2000 | qleaf GPU traversal SP | 9.567 | 10.847 | 14.943 | 10.926 | `qleaf/gpu/persistent/compact/bt:512/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 2000 | qleaf GPU traversal SP cached one-wave | skipped | skipped | skipped | skipped | BT64 persistent grid not co-resident; BT96 dynamic shared memory too large |
| 2000 | qleaf GPU traversal SP cached all-waves | skipped | skipped | skipped | skipped | BT64 persistent grid not co-resident; BT96 dynamic shared memory too large |
| 2000 | qleaf GPU bitmask | 8.535 | 10.287 | 12.247 | 9.477 | `qleaf/gpu/persistent/compact/bt:1024/transposed:0/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:8` |
| 2000 | XGBoost sklearn | 262.399 | 276.735 | 389.119 | 267.056 | `xgboost/sklearn/threads:8/output:margin` |
| 2000 | Treelite | 96.191 | 102.975 | 119.871 | 96.811 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 2000 | nvForest GPU | 23.327 | 24.271 | 34.335 | 24.620 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:0` |
| 5000 | qleaf CPU traversal | 7.207 | 11.959 | 18.239 | 8.850 | `qleaf/cpu/compact/branch/threads:8` |
| 5000 | qleaf CPU bitmask | 101.503 | 103.103 | 105.535 | 101.797 | `qleaf/cpu/compact/bitmask/threads:16` |
| 5000 | qleaf GPU traversal SP | 10.063 | 10.639 | 12.887 | 10.627 | `qleaf/gpu/persistent/compact/bt:1024/transposed:1/bitmask:0/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0` |
| 5000 | qleaf GPU traversal SP cached one-wave | skipped | skipped | skipped | skipped | BT64 persistent grid not co-resident; BT96 dynamic shared memory too large |
| 5000 | qleaf GPU traversal SP cached all-waves | skipped | skipped | skipped | skipped | BT64 persistent grid not co-resident; BT96 dynamic shared memory too large |
| 5000 | qleaf GPU bitmask | 15.063 | 16.087 | 20.607 | 17.021 | `qleaf/gpu/persistent/compact/bt:512/transposed:1/bitmask:1/cached:0/mapped:1/crossreduce:0/sp:1/stage:1/seqflag:0/tpt:4` |
| 5000 | XGBoost sklearn | 513.791 | 646.655 | 1340.415 | 568.885 | `xgboost/sklearn/threads:8/output:margin` |
| 5000 | Treelite | 155.903 | 166.911 | 182.015 | 156.624 | `treelite/tl2cgen-capi/threads:1/output:margin` |
| 5000 | nvForest GPU | 39.359 | 41.247 | 55.487 | 41.474 | `nvforest/gpu/input:device/output:device_sync/layout:depth_first/rows_per_block_iter:1/align_bytes:0` |

## Reduced-grid CacheAllWaves Diagnostic

The main cached qleaf GPU rows above use the full persistent grid, so at Higgs
n=2000 with `BT=96` both cached modes have one grid-stride wave per block. The
diagnostic below forces a smaller resident grid to exercise `CacheAllWaves`
with multiple waves: Higgs n=2000, `BT=32`, `persistent_grid=28`, so
`full_grid=63`, `max_waves_per_block=3`, and each block stages 96 cached trees.

| dataset | trees | config | p50 us | p90 us | p99 us | mean us | status |
|---|---:|---|---:|---:|---:|---:|---|
| higgs | 2000 | CPU traversal, physical-first best | 2.593 | 3.193 | 4.775 | 2.743 | ok |
| higgs | 2000 | cached one-wave, BT96/full-grid | 3.693 | 3.895 | 5.487 | 3.938 | ok |
| higgs | 2000 | cached all-waves, BT96/full-grid | 3.721 | 3.889 | 5.519 | 3.947 | ok |
| higgs | 2000 | uncached SP, BT32/grid28 | 6.703 | 7.411 | 10.047 | 7.578 | ok |
| higgs | 2000 | fixed cache, BT32/grid28 | skipped | skipped | skipped | skipped | fixed cache does not cover grid-stride waves |
| higgs | 2000 | cache-all-waves, BT32/grid28 | 4.375 | 4.663 | 7.087 | 4.982 | ok |

## nvForest Mode Split

nvForest was measured three ways to separate host-input staging and final
output copy:

- `nvforest/gpu/input:host/output:host`: host row into nvForest, host-visible
  result.
- `nvforest/gpu/input:device/output:host`: preloaded device row into nvForest,
  then copy the result back to host.
- `nvforest/gpu/input:device/output:device_sync`: preloaded device row into
  nvForest, then synchronize only, without copying the result back to host.

Each row below is the best p50 configuration within that mode after sweeping
`layout`, `rows_per_block_iter`, and `align_bytes`. The device-output mode is
diagnostic rather than directly comparable to qleaf's host-visible result.

### Higgs nvForest Modes

| trees | mode | best layout | rpb | align | p50 us | p90 us | p99 us | mean us |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 500 | `input:host/output:host` | `layered` | 1 | 128 | 22.639 | 23.599 | 36.735 | 24.109 |
| 500 | `input:device/output:host` | `layered` | 1 | 0 | 19.375 | 19.743 | 29.839 | 20.224 |
| 500 | `input:device/output:device_sync` | `breadth_first` | 1 | 128 | 10.735 | 11.527 | 15.903 | 11.401 |
| 1000 | `input:host/output:host` | `depth_first` | 1 | 128 | 30.735 | 31.711 | 44.191 | 32.360 |
| 1000 | `input:device/output:host` | `depth_first` | 1 | 128 | 27.503 | 27.823 | 38.943 | 28.560 |
| 1000 | `input:device/output:device_sync` | `depth_first` | 1 | 128 | 14.719 | 16.039 | 20.639 | 15.667 |
| 2000 | `input:host/output:host` | `breadth_first` | 1 | 0 | 38.943 | 39.935 | 52.415 | 40.375 |
| 2000 | `input:device/output:host` | `breadth_first` | 1 | 128 | 35.743 | 36.223 | 48.159 | 34.681 |
| 2000 | `input:device/output:device_sync` | `breadth_first` | 1 | 128 | 21.743 | 23.951 | 32.319 | 23.145 |
| 5000 | `input:host/output:host` | `depth_first` | 1 | 128 | 80.831 | 82.111 | 264.703 | 83.841 |
| 5000 | `input:device/output:host` | `depth_first` | 1 | 0 | 77.183 | 78.655 | 265.471 | 80.246 |
| 5000 | `input:device/output:device_sync` | `depth_first` | 1 | 128 | 66.815 | 68.543 | 88.703 | 69.234 |

### Epsilon nvForest Modes

| trees | mode | best layout | rpb | align | p50 us | p90 us | p99 us | mean us |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 500 | `input:host/output:host` | `depth_first` | 1 | 128 | 23.471 | 24.111 | 35.999 | 24.562 |
| 500 | `input:device/output:host` | `breadth_first` | 1 | 0 | 19.503 | 27.535 | 34.303 | 22.090 |
| 500 | `input:device/output:device_sync` | `depth_first` | 1 | 0 | 13.095 | 13.703 | 17.551 | 13.730 |
| 1000 | `input:host/output:host` | `depth_first` | 1 | 0 | 31.695 | 32.495 | 45.407 | 33.110 |
| 1000 | `input:device/output:host` | `depth_first` | 1 | 0 | 27.423 | 27.839 | 40.863 | 28.410 |
| 1000 | `input:device/output:device_sync` | `depth_first` | 1 | 0 | 16.479 | 17.247 | 22.031 | 17.374 |
| 2000 | `input:host/output:host` | `depth_first` | 1 | 0 | 39.839 | 40.895 | 57.663 | 40.266 |
| 2000 | `input:device/output:host` | `depth_first` | 1 | 0 | 35.743 | 36.159 | 49.247 | 37.088 |
| 2000 | `input:device/output:device_sync` | `depth_first` | 1 | 0 | 23.327 | 24.271 | 34.335 | 24.620 |
| 5000 | `input:host/output:host` | `depth_first` | 1 | 0 | 56.159 | 57.055 | 77.759 | 55.816 |
| 5000 | `input:device/output:host` | `depth_first` | 1 | 0 | 52.159 | 52.639 | 69.375 | 54.245 |
| 5000 | `input:device/output:device_sync` | `depth_first` | 1 | 0 | 39.359 | 41.247 | 55.487 | 41.474 |

## Thread Shapes

### qleaf CPU n=5000 p50

The inline `nothread` control in this pinned sweep ran on dispatcher CPU 19, so
use it as an inline-control row for this exact command, not as a best P-core
single-thread baseline.

| dataset | worker | nothread | threads 1 | threads 2 | threads 4 | threads 8 | threads 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| higgs | traversal | 148.863 | 89.535 | 44.127 | 14.559 | 5.719 | 7.427 |
| higgs | bitmask | 1616.895 | 1192.959 | 598.527 | 306.943 | 161.279 | 110.079 |
| epsilon | traversal | 156.159 | 187.007 | 91.327 | 16.023 | 7.207 | 9.063 |
| epsilon | bitmask | 673.791 | 593.919 | 330.495 | 280.063 | 156.543 | 101.503 |

### TL2cgen C API p50

| dataset | trees | threads 1 | threads 2 | threads 4 | threads 8 | threads 16 |
|---|---:|---:|---:|---:|---:|---:|
| higgs | 500 | 10.039 | 10.455 | 10.807 | 11.775 | 13.271 |
| higgs | 1000 | 32.287 | 33.759 | 34.367 | 37.567 | 40.095 |
| higgs | 2000 | 98.367 | 102.399 | 104.255 | 108.607 | 128.831 |
| higgs | 5000 | 313.855 | 331.007 | 338.943 | 351.999 | 421.119 |
| epsilon | 500 | 23.055 | 23.231 | 23.135 | 24.463 | 27.263 |
| epsilon | 1000 | 50.495 | 53.151 | 54.175 | 55.647 | 62.623 |
| epsilon | 2000 | 96.191 | 96.255 | 99.199 | 102.079 | 114.367 |
| epsilon | 5000 | 155.903 | 158.847 | 165.247 | 167.167 | 192.127 |

## Interpretation

The pinned batch-1 winner is qleaf CPU traversal across both datasets and all
tree counts in this sweep. The refreshed CPU run uses physical-core-first
worker placement, so early worker counts spread across separate cores before
using SMT siblings. With that order, traversal usually peaks at `threads:8`
for p50, while bitmask still benefits from `threads:16`.

The qleaf GPU persistent traversal remains the best GPU path and the best
non-CPU-qleaf path. At n=5000 it is 6.703/6.891/8.495 us on Higgs and
10.063/10.639/12.887 us on Epsilon. The gap between Higgs and Epsilon is still
mostly the wide feature buffer and host/device coordination: Epsilon needs a
larger staged feature transfer and favors `BT=1024`, while Higgs is happier at
`BT=256` or `BT=512`.

The cached persistent traversal variants improve Higgs for the tree counts
that fit in resident shared memory: one-wave cache is 3.107/3.411/4.523 us at
n=500, 3.499/3.631/5.279 us at n=1000, and 3.693/3.895/5.487 us at n=2000.
That is faster than the old uncached GPU traversal rows, but still slower than
the physical-core-first CPU traversal at the same tree counts. Higgs n=5000 skips
because the full resident persistent grid cannot fit the tree payload in shared
memory on this RTX 3060. Epsilon cached traversal is not competitive: the
2000-feature staged row consumes enough shared memory that only BT64 fits at
n=500 and n=1000, and those rows are much slower than the uncached GPU path.

The reduced-grid Higgs n=2000 diagnostic confirms that CacheAllWaves covers
multiple grid-stride waves correctly and improves the matching BT32 uncached
shape by about 35% at p50, but the reduced-grid shape is still slower than the
simpler BT96 full-grid cached run.

The qleaf GPU bitmask path is competitive at small tree counts but still loses
to traversal at n=5000. Its best n=5000 settings use `TPT=4` and `BT=512`; the
extra intra-tree parallelism reduces walk time but increases grid/fan-in
coordination. On CPU, bitmask remains around an order of magnitude slower than
traversal because it evaluates all internal nodes instead of following one
depth-6 path.

Treelite/TL2cgen is the strongest external CPU baseline. It prefers one thread
for this batch-1 shape; extra threads add overhead and do not improve p50.
XGBoost sklearn has much larger tails, especially at 5000 trees.

nvForest is now the strongest external GPU baseline in this report. The best
configuration always uses device-resident input and device-output sync; for
host-visible output, the best p50 is about 19-77 us on Higgs and 19-52 us on
Epsilon across this tree-count sweep. `rows_per_block_iter=1` won every best
row, while the best layout and alignment varied by dataset and tree count.

## Reproduce

Build:

```sh
cmake --build build-rel-bt32 --target qleaf_latency
cmake --build build-rel-bt64 --target qleaf_latency
cmake --build build-rel-bt96 --target qleaf_latency
cmake --build build-rel-bt256 --target worker_bench qleaf_latency
cmake --build build-rel-bt512 --target qleaf_latency
cmake --build build-rel-bt1024 --target qleaf_latency
cmake -S . -B build-rel-bt256-gcc14 \
  -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14
cmake --build build-rel-bt256-gcc14 --target external_latency_cpp
```

Pinned qleaf CPU:

```sh
ITERS=20000 WARMUP=2000 THREADS=1,2,4,8,16 DATASETS=higgs,epsilon \
CPU=1 GPU=0 RUN_EXTERNAL=0 \
PIN_CORES=19,0,2,4,6,8,10,12,14,16,17,18,1,3,5,7,9 \
QLEAF_OUT=qleaf_cpu_physical_first \
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

ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon CPU=0 GPU=1 RUN_EXTERNAL=0 \
GPU_SUITE=sp BT=64 PIN_CORES=16 QLEAF_OUT=qleaf_gpu_cached_bt64 \
bench/sweep_latency.sh build-rel-bt64/qleaf_latency bench/results/latency/sweep_pinned

ITERS=20000 WARMUP=2000 DATASETS=higgs,epsilon CPU=0 GPU=1 RUN_EXTERNAL=0 \
GPU_SUITE=sp BT=96 PIN_CORES=16 QLEAF_OUT=qleaf_gpu_cached_bt96 \
bench/sweep_latency.sh build-rel-bt96/qleaf_latency bench/results/latency/sweep_pinned

build-rel-bt32/qleaf_latency \
  --trees bench/data/higgs_d6_n2000.qleaf.json \
  --features 28 \
  --features-csv bench/data/higgs_d6_n2000.feats.csv \
  --samples 1024 \
  --ntrees 2000 \
  --warmup 2000 \
  --iters 20000 \
  --threads 1 \
  --buffers compact \
  --bt 32 \
  --persistent-grid 28 \
  --pin-cores 16 \
  --cpu 0 \
  --gpu 1 \
  --gpu-suite sp \
  --filter sp:1/stage:1/seqflag:0 \
  --out-dir bench/results/latency/higgs_n2000_bt32_grid28_cacheall_pinned
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

nvForest GPU modes:

```sh
ITERS=20000 WARMUP=2000 SAMPLES=1024 DATASETS=higgs,epsilon \
ENGINES=nvforest CPU=0 GPU=0 RUN_EXTERNAL=1 EXTERNAL_AFFINITY=0-15 \
EXTERNAL_CPP_BIN=build-rel-bt256-gcc14/external_latency_cpp \
PYTHON=.venv/bin/python QLEAF_OUT=qleaf_none \
bench/sweep_latency.sh build-rel-bt256/qleaf_latency bench/results/latency/sweep_pinned
```

Aggregate:

The aggregate helper selects the best p50 across all `summary.json` files under
the root. If older CPU result directories are still present, use the tables
above for the physical-first CPU comparison.

```sh
.venv/bin/python bench/compare_latency.py bench/results/latency/sweep_pinned \
  > bench/results/latency/sweep_pinned/compare.txt
```
