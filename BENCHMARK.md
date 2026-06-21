# Batch-1 Latency: Data Presentation

This file expands the benchmark summary in [README.md](README.md).

CPU and external-CPU benchmark data can be found in:

`bench/results/latency/full_regen_no_turbo_100k_2026_06_19`

Locked-clock GPU benchmark data can be found in:

`bench/results/latency/gpu_locked_no_turbo_newpath_100k_2026_06_19`

The persistent switch-ablation data can be found in:

`bench/results/latency/ablation_hdr_gpu_locked_no_turbo_newpath_100k_2026_06_19`

All main latency numbers are microseconds from the fixed-iteration HDR harness (`iters=100000`, `warmup=10000`, batch size 1, `CLOCK_MONOTONIC_RAW`). Tables labeled p50 use HDR medians; tail tables report p90 / p99 / p99.9 from the same HDR summaries. The Google Benchmark data is retained only for the separate `HDR_V_GB.md` comparison.

## Benchmark Machine

The benchmark machine is a single-socket desktop system. The CPU is a 12th Gen Intel Core i7-12700F with 20 logical CPUs: 8 performance cores with SMT (`CPU 0-15`) plus 4 efficiency cores (`CPU 16-19`). The CPU and external-CPU measurements use the `performance` governor on every CPU with Intel pstate active and turbo disabled (`no_turbo=1`); `cpupower` reported a 1.60 GHz upper hardware limit for the sampled CPU. The cache hierarchy reported by `lscpu` is 512 KiB L1d, 512 KiB L1i, 12 MiB L2, and 25 MiB L3.

The GPU is an NVIDIA GeForce RTX 3060 with 12 GiB of memory. The GPU measurements were launched from a quiet non-graphical session with application clocks requested at 1800 MHz SM and 7501 MHz memory. The GPU and CPU are on one NUMA node, and the GPU's CPU affinity covers `0-19`.

## Benchmark Data Sets

The benchmark uses two binary tree-ensemble workloads chosen mainly to separate feature-vector size from tree-count size. HIGGS is the low-dimensional case with 28 input features. epsilon keeps the same tree-count scale but uses 2000 input features, making the input footprint and host/device feature movement much more visible.

All retained headline sweeps use depth-6 ensembles and evaluate batch-1 margin-output inference over 1024 CSV feature rows. The reported tree counts are `n=500,1000,2000,5000`. The retained sweeps use the 5000-tree artifacts and benchmark prefixes through `--ntrees`. In QLeaf's compact model format, a depth-6 tree occupies 127 node slots, with unused slots serving as padding for branches that terminate earlier.

| dataset | features | depth | tree counts | feature rows | QLeaf model | external model |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 28 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/higgs_d6_n5000.qleaf.json` | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/higgs_d6_n5000.xgb.json` |
| epsilon | 2000 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/epsilon_d6_n5000.qleaf.json` | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/epsilon_d6_n5000.xgb.json` |

This contrast is useful because an optimization that improves one part of the latency budget can be neutral or harmful elsewhere. HIGGS mostly stresses tree traversal, worker coordination, launch cost, and synchronization around a small feature row; epsilon adds a much wider feature vector while preserving the same batch-1 serving contract.

## CPU and External Baselines

The CPU baseline is strong because a batch-1 request stays entirely on CPU: no host/device transfer, no GPU launch, no device synchronization. The trade-off is that low latency is bought by spending several CPU cores on one request stream.

CPU affinity matters on this hybrid processor. Both affinity sweeps reserve `CPU 19` for the dispatch thread. The physical-first policy assigns one logical CPU per physical core first:

`0,2,4,6,8,10,12,14,16,17,18,1,3,5,7,9`

The sibling-first policy assigns both SMT siblings on the P-cores before using the E-cores:

`0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`

In the controlled CPU sweep, physical-first wins 18 of the 24 comparisons at 2, 4, and 8 worker threads. At 16 workers, sibling-first wins 7 of 8 rows, which is consistent with avoiding E-core spill for the largest CPU worker pools.

| worker threads | physical-first wins | sibling-first wins | note |
| --- | --- | --- | --- |
| 2 | 5 / 8 | 3 / 8 | distinct P-cores dominate |
| 4 | 7 / 8 | 1 / 8 | physical-first usually better |
| 8 | 6 / 8 | 2 / 8 | physical-first usually better |
| 16 | 1 / 8 | 7 / 8 | sibling-first often avoids E-core spill |

For the largest forests, the effect is visible directly in the affinity sweeps:

| dataset | threads | physical-first p50 us | sibling-first p50 us |
| --- | --- | --- | --- |
| HIGGS n=5000 | 2 | 93.631 | 178.047 |
| HIGGS n=5000 | 4 | 30.575 | 87.295 |
| HIGGS n=5000 | 8 | 12.351 | 28.751 |
| HIGGS n=5000 | 16 | 24.191 | 11.959 |
| epsilon n=5000 | 2 | 196.479 | 179.327 |
| epsilon n=5000 | 4 | 88.703 | 89.151 |
| epsilon n=5000 | 8 | 43.647 | 31.343 |
| epsilon n=5000 | 16 | 31.679 | 12.559 |

The table below compares best QLeaf CPU branch traversal against external CPU libraries, nvForest CPU, and nvForest's fastest batch-1 GPU mode. Treelite is measured through the TL2cgen C API. XGBoost is measured through the native C API dense predictor (`XGBoosterPredictFromDense`) in the C++ harness. nvForest CPU is measured through the same C++ nvForest harness with host input/output, `threads=1,2,4,8,16`, layouts `depth_first,layered,breadth_first`, `align_bytes=0,64`, and `chunk_size=64`. nvForest GPU uses `rows_per_block_iter=1`; the value shown is its fastest `device_device_sync` configuration, so it is a favorable device-resident external GPU lower bound rather than the same host-output contract.

| dataset | trees | QLeaf CPU best | CPU config | Treelite TL2cgen | XGBoost C API | nvForest CPU | nvForest GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.177 | physical `threads:8` | 22.943 | 130.239 | 24.687 | 11.399 |
| HIGGS | 1000 | 3.111 | physical `threads:8` | 75.967 | 211.071 | 35.807 | 14.951 |
| HIGGS | 2000 | 10.719 | sibling `threads:16` | 234.367 | 377.087 | 49.503 | 22.159 |
| HIGGS | 5000 | 11.959 | sibling `threads:16` | 639.487 | 1020.927 | 77.887 | 70.591 |
| epsilon | 500 | 2.427 | physical `threads:8` | 50.911 | 137.215 | 28.383 | 13.655 |
| epsilon | 1000 | 3.715 | sibling `threads:16` | 119.679 | 218.239 | 38.495 | 17.359 |
| epsilon | 2000 | 5.931 | physical `threads:8` | 214.399 | 376.319 | 49.695 | 24.607 |
| epsilon | 5000 | 12.559 | sibling `threads:16` | 357.119 | 768.511 | 74.943 | 42.143 |

QLeaf CPU traversal remains far ahead of the external CPU libraries in this single-row setting. Native XGBoost C API is slower than TL2cgen or nvForest CPU, and nvForest CPU remains well behind QLeaf CPU. The nvForest GPU column should be read as an external GPU lower bound, not as proof about host-visible nvForest latency under the same contract.

The TL2cgen timings above come from the current external C++ harness. The compiled TL2cgen model libraries are staged under `bench/results/latency/full_regen_no_turbo_100k_2026_06_19/external_models/`; provenance is recorded there because fresh TL2cgen compilation of the 5000-tree generated C source was pathologically slow on this machine.

Turning turbo on substantially reduces CPU p50 latency on this desktop. In a turbo-enabled reference measurement on the same machine, QLeaf CPU best p50 was `0.992 us` for HIGGS n=500, `5.483 us` for HIGGS n=5000, and `6.031 us` for epsilon n=5000, compared with `2.177 us`, `11.959 us`, and `12.559 us` in the controlled tables above. The main tables keep turbo disabled for more predictable frequencies and cleaner benchmark hygiene. Absolute latencies, and sometimes CPU-vs-GPU rankings, remain hardware- and policy-dependent; the benchmark should therefore be read primarily as a guide to the effects of the optimization techniques.

## One-Shot GPU

The first GPU implementation is one-shot: each request launches a kernel, synchronizes, copies block partials back, and completes the reduction on the host. This is simple and gives a fair request/response shape, but it pays launch and synchronization costs on every request.

The uncached one-shot run compares transposed (node-major) layout against row-major layout at fixed `BT=256`, `TPT=4`, and `nodestride=4`. The point is to isolate the transposed-vs-row-major layout choice under the same one-shot launch shape, not to do a new BT/TPT/nodestride sweep.

The main algorithmic comparison is traversal versus bitmask. Traversal follows one root-to-leaf path per tree. Bitmask evaluation is QuickScorer-like: each node stores a reachable-leaf bitmask and the final result is determined by collecting masks and applying bitwise AND reduction. Bitmask exposes more intra-tree parallelism, but it does more work per tree and needs careful TPT tuning.

| dataset | trees | QLeaf transposed | QLeaf row-major | nvForest host_host | nvForest device_host | nvForest device_device_sync |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | **15.903** | 16.039 | 29.615 | 22.463 | 11.399 |
| HIGGS | 1000 | 15.999 | **15.959** | 29.647 | 22.543 | 14.951 |
| HIGGS | 2000 | **16.639** | 17.679 | 37.631 | 30.639 | 22.159 |
| HIGGS | 5000 | **24.383** | 29.455 | 87.231 | 80.127 | 70.591 |
| epsilon | 500 | **16.559** | 16.751 | 30.351 | 22.527 | 13.655 |
| epsilon | 1000 | **16.911** | 17.167 | 37.951 | 30.655 | 17.359 |
| epsilon | 2000 | **17.871** | 19.775 | 38.847 | 38.879 | 24.607 |
| epsilon | 5000 | **25.167** | 36.863 | 62.431 | 55.231 | 42.143 |

One-shot transposed wins 7 of 8 rows; row-major wins: HIGGS n=1000.

Against nvForest's host-host mode, QLeaf one-shot wins every row. Against nvForest's fastest device-output/sync mode, QLeaf loses on the smallest rows but becomes competitive or faster as tree count grows. So one-shot is not hopeless; the main remaining cost is that every request still pays a launch and completion path.

## Naive Persistent Kernel

The next step is to remove launch overhead by keeping a persistent kernel resident. The first persistent design is `!Mapped, !SinglePoller, SeqFlag, CrossReduce`: the host copies the feature row to device memory, increments a sequence flag, many blocks poll for work, and the last finishing block performs cross-block completion.

The tables below use fixed-iteration HDR latency from `bench/results/latency/ablation_hdr_gpu_locked_no_turbo_newpath_100k_2026_06_19/`. The one-shot reference is the matched-BT mapped one-shot HDR number for that row, not the BT256 one-shot headline above. HIGGS is held at `BT=512` and epsilon at `BT=1024` for the controlled switch path.

| dataset | kernel | BT | one-shot reference | naive persistent |
| --- | --- | --- | --- | --- |
| HIGGS n=5000 | traversal | 512 | 11.247 | 26.735 |
| HIGGS n=5000 | bitmask TPT1 | 512 | 14.607 | 38.975 |
| epsilon n=5000 | traversal | 1024 | 14.879 | 16.639 |
| epsilon n=5000 | bitmask TPT1 | 1024 | 21.775 | 38.591 |

| dataset | kernel | BT | one-shot p90 / p99 / p99.9 | naive p90 / p99 / p99.9 |
| --- | --- | --- | --- | --- |
| HIGGS n=5000 | traversal | 512 | 11.679 / 12.239 / 14.167 | 28.559 / 30.703 / 32.447 |
| HIGGS n=5000 | bitmask TPT1 | 512 | 15.039 / 15.487 / 17.503 | 43.263 / 46.175 / 48.703 |
| epsilon n=5000 | traversal | 1024 | 15.327 / 15.743 / 17.631 | 18.863 / 20.719 / 22.159 |
| epsilon n=5000 | bitmask TPT1 | 1024 | 22.287 / 22.815 / 24.559 | 39.871 / 41.151 / 42.271 |

Naive persistence is worse than the matched one-shot reference in all four largest-forest comparisons. The p99.9 tails are tight in this controlled run, but removing kernel launch is still not enough by itself: cross-block coordination and request signaling are exposed as first-order latency costs.

## Naive Persistent Decomposition

The focused runs below live in `gpu_naive_decomp_profile/` and use the same fixed-iteration HDR harness as the other latency tables. `total` is the full measured request iteration, while `predict` is the host call that dispatches the request and waits for completion. The profiler cycle-counter logs are retained in the result folder, but this decomposition is reported as HDR latency rather than profiler running averages.

| case | BT | total p50 | predict p50 | total mean | predict mean |
| --- | --- | --- | --- | --- | --- |
| HIGGS n=5000 traversal | 512 | 21.583 | 21.535 | 21.478 | 21.423 |
| HIGGS n=5000 bitmask TPT1 | 512 | 26.735 | 26.687 | 26.854 | 26.799 |
| epsilon n=5000 traversal | 1024 | 15.087 | 15.031 | 15.126 | 15.071 |
| epsilon n=5000 bitmask TPT1 | 1024 | 36.831 | 36.799 | 36.898 | 36.842 |

| case | BT | total p90 / p99 / p99.9 | predict p90 / p99 / p99.9 |
| --- | --- | --- | --- |
| HIGGS n=5000 traversal | 512 | 23.135 / 24.735 / 26.847 | 23.071 / 24.671 / 26.799 |
| HIGGS n=5000 bitmask TPT1 | 512 | 27.487 / 30.383 / 33.631 | 27.423 / 30.335 / 33.599 |
| epsilon n=5000 traversal | 1024 | 16.767 / 18.527 / 20.479 | 16.719 / 18.463 / 20.383 |
| epsilon n=5000 bitmask TPT1 | 1024 | 37.663 / 38.911 / 41.055 | 37.631 / 38.847 / 40.991 |

The decomposition shows the same shape as the larger ablation path: naive persistence removes kernel launch but exposes request signaling and cross-block completion as first-order latency costs. The naive path remains slower than the matched one-shot reference even with tight tails.

## Persistent Optimization Path: HIGGS

The low-dimensional HIGGS case gives the cleanest persistent-kernel story. The controlled path below holds `BT=512` and keeps bitmask at `TPT=1` until the final bitmask-specific tuning step. Each latency cell is `p50 / p90 / p99` in microseconds.

### Step 0: Naive Persistent

The baseline persistent kernel is `!Mapped, !SinglePoller, SeqFlag, CrossReduce`: the host copies the feature row to device memory, increments a sequence flag, many blocks poll for work, and the last finishing block performs cross-block completion. Removing launch overhead by itself is not enough; this path exposes request polling and cross-block completion as first-order costs.

| trees | traversal | bitmask TPT1 |
| --- | --- | --- |
| 500 | 7.819 / 8.047 / 8.487 | 12.735 / 13.087 / 13.407 |
| 1000 | 8.407 / 8.719 / 9.311 | 12.871 / 13.287 / 13.719 |
| 2000 | 11.303 / 11.991 / 12.783 | 15.999 / 16.703 / 17.631 |
| 5000 | 26.735 / 28.559 / 30.703 | 38.975 / 43.263 / 46.175 |

### Step 1: `!CrossReduce`

The first fix removes device-side cross-block reduction from the completion path. Instead of making all blocks coordinate through one final device-side reduction, each block writes its own result slot and the host detects completion. This is the first large win, especially at larger tree counts.

| trees | traversal cross-reduce | traversal `!CrossReduce` | bitmask cross-reduce | bitmask `!CrossReduce` |
| --- | --- | --- | --- | --- |
| 500 | 7.819 / 8.047 / 8.487 | 6.419 / 6.599 / 7.087 | 12.735 / 13.087 / 13.407 | 11.023 / 11.263 / 11.743 |
| 1000 | 8.407 / 8.719 / 9.311 | 6.487 / 6.907 / 7.167 | 12.871 / 13.287 / 13.719 | 11.111 / 11.495 / 11.935 |
| 2000 | 11.303 / 11.991 / 12.783 | 8.067 / 8.623 / 9.415 | 15.999 / 16.703 / 17.631 | 12.687 / 13.463 / 14.407 |
| 5000 | 26.735 / 28.559 / 30.703 | 17.487 / 18.959 / 20.671 | 38.975 / 43.263 / 46.175 | 28.879 / 32.623 / 35.327 |

The result matches the decomposition: the final-arriver path is not just bookkeeping. Removing cross-block completion cuts both median and tail latency for traversal and bitmask on every HIGGS row.

### Step 2: Single Poller

After removing cross-block completion, the next bottleneck is request detection. The single-poller variant lets one block observe the host-visible request state and broadcast work to the resident blocks, instead of having every block poll. This slightly hurts the small HIGGS rows, but it is a large win once enough blocks are resident.

| trees | traversal `!SinglePoller` | traversal single-poller | bitmask `!SinglePoller` | bitmask single-poller |
| --- | --- | --- | --- | --- |
| 500 | 6.419 / 6.599 / 7.087 | 6.647 / 6.843 / 7.315 | 11.023 / 11.263 / 11.743 | 11.407 / 11.607 / 12.063 |
| 1000 | 6.487 / 6.907 / 7.167 | 6.775 / 7.011 / 7.443 | 11.111 / 11.495 / 11.935 | 11.375 / 11.631 / 12.095 |
| 2000 | 8.067 / 8.623 / 9.415 | 6.827 / 7.123 / 7.475 | 12.687 / 13.463 / 14.407 | 11.607 / 11.855 / 12.271 |
| 5000 | 17.487 / 18.959 / 20.671 | 8.631 / 8.911 / 9.375 | 28.879 / 32.623 / 35.327 | 19.983 / 20.367 / 20.863 |

This is not a universal local improvement, and the table should show that. The value of single-poller grows with model size because it removes the poll storm and lets the other blocks stay on a device-side broadcast path.

### Step 3: Mapped Input

With one poller in charge, mapped input becomes useful for HIGGS. This step switches from `!Mapped` to `Mapped` so the host-visible request buffer itself can be used by the persistent worker instead of copying the row into device memory before each request.

| trees | traversal `!Mapped` | traversal mapped | bitmask `!Mapped` | bitmask mapped |
| --- | --- | --- | --- | --- |
| 500 | 6.647 / 6.843 / 7.315 | 4.959 / 5.223 / 5.427 | 11.407 / 11.607 / 12.063 | 9.711 / 9.903 / 10.111 |
| 1000 | 6.775 / 7.011 / 7.443 | 5.059 / 5.367 / 5.551 | 11.375 / 11.631 / 12.095 | 9.719 / 9.879 / 10.127 |
| 2000 | 6.827 / 7.123 / 7.475 | 5.163 / 5.567 / 5.959 | 11.607 / 11.855 / 12.271 | 9.823 / 10.079 / 10.495 |
| 5000 | 8.631 / 8.911 / 9.375 | 7.079 / 7.383 / 7.771 | 19.983 / 20.367 / 20.863 | 18.399 / 18.799 / 19.279 |

The important point is the ordering: mapped input helps after the poll storm has been removed. Used together with a single poller, it avoids explicit feature copies without letting every resident block repeatedly observe host-visible state.

### Step 4: Staged Sentinel Signal

The last protocol step removes the separate ordered sequence flag and folds staging into the handoff. In the `!SeqFlag` path, the mapped feature buffer itself carries readiness: the host writes finite feature values, the poller waits until they are visible, stages them, broadcasts work, and then resets the buffer to sentinels for the next request. The comparison below goes directly from the mapped SeqFlag path to the staged sentinel path.

| trees | traversal SeqFlag/mapped | traversal staged sentinel | bitmask SeqFlag/mapped | bitmask staged sentinel |
| --- | --- | --- | --- | --- |
| 500 | 4.959 / 5.223 / 5.427 | 4.515 / 4.623 / 4.995 | 9.711 / 9.903 / 10.111 | 10.135 / 10.263 / 10.647 |
| 1000 | 5.059 / 5.367 / 5.551 | 4.551 / 4.899 / 5.295 | 9.719 / 9.879 / 10.127 | 10.143 / 10.271 / 10.631 |
| 2000 | 5.163 / 5.567 / 5.959 | 5.319 / 5.471 / 5.855 | 9.823 / 10.079 / 10.495 | 9.975 / 10.095 / 10.447 |
| 5000 | 7.079 / 7.383 / 7.771 | 6.891 / 7.055 / 7.375 | 18.399 / 18.799 / 19.279 | 18.479 / 18.895 / 19.391 |

The staged sentinel path improves traversal p90/p99 on every HIGGS row and improves traversal p50 on all but n=2000. At `TPT=1`, bitmask does not benefit from the protocol change by itself; it needs one more knob because too much intra-tree parallelism is still unused.

### Step 5: Bitmask TPT

Traversal stops at the fixed-`BT=512` sentinel path in this controlled switch table. Bitmask then tunes threads-per-tree at the same `BT=512`; this is where the bitmask worker becomes competitive for HIGGS.

| trees | bitmask TPT1 | tuned bitmask | tuned cfg |
| --- | --- | --- | --- |
| 500 | 10.135 / 10.263 / 10.647 | 4.731 / 4.823 / 5.199 | BT512/TPT8 |
| 1000 | 10.143 / 10.271 / 10.631 | 5.291 / 5.523 / 5.843 | BT512/TPT4 |
| 2000 | 9.975 / 10.095 / 10.447 | 5.763 / 6.179 / 6.667 | BT512/TPT8 |
| 5000 | 18.479 / 18.895 / 19.391 | 12.311 / 12.831 / 13.311 | BT512/TPT4 |

For HIGGS, the main persistent lesson is that launch removal is only the opening move. The large wins come from removing cross-block completion, limiting request polling to one block, and then using mapped/sentinel signaling carefully. Bitmask only works as a latency competitor after its TPT policy is tuned.

## More is different: when 2000 features are present

epsilon uses the same persistent path at `BT=1024`, but the 2000-feature row changes the trade-off. Coordination still helps, but mapped/staged/sentinel signaling is not a universal win: for traversal it is slower than the copied-feature coordination path. This is largely expected because a 2000-float row is about 7.8 KiB, and repeatedly polling or loading the host-visible mapped buffer from the device becomes much more expensive than it is for HIGGS.

Bitmask only becomes competitive after release tuning (see next section), especially when row-major layout wins on the smaller epsilon cases.

### epsilon Traversal

| trees | naive persistent | coordination | signaling | release sweep |
| --- | --- | --- | --- | --- |
| 500 | 8.391 | 7.027 | 7.775 | 7.791 |
| 1000 | 10.359 | 8.927 | 9.759 | 9.847 |
| 2000 | 11.543 | 9.295 | 9.919 | 9.887 |
| 5000 | 16.639 | 9.871 | 10.479 | 10.495 |

### epsilon Bitmask

| trees | naive persistent | coordination | signaling | release tuned | release cfg |
| --- | --- | --- | --- | --- | --- |
| 500 | 18.159 | 16.559 | 17.135 | 7.171 | BT1024/TPT16/row |
| 1000 | 29.183 | 27.551 | 28.271 | 7.935 | BT1024/TPT16/row |
| 2000 | 30.543 | 28.735 | 29.487 | 8.879 | BT1024/TPT8/row |
| 5000 | 38.591 | 31.855 | 32.415 | 13.903 | BT1024/TPT4 |

The feature-width result is the important negative finding: the HIGGS signaling path should not be treated as a universal GPU policy. For epsilon traversal, copied features with no cross-block reduction remain best; for epsilon bitmask, BT/TPT/layout tuning is what changes the outcome.

## Best Results: HIGGS and epsilon

The final fixed-iteration table chooses the best uncached persistent GPU result observed across the locked controlled path and the locked `gpu_best/` release sweep. HIGGS cached-tree values are shown in brackets when they beat the uncached traversal row, but the leading value remains the uncached result.

| dataset | trees | QLeaf CPU best | GPU traversal best | traversal cfg | GPU bitmask best | bitmask cfg | QLeaf GPU best | best cfg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.177 | 3.803 [cached: 3.333][^cached-higgs] | BT256 | 4.731 | BT512/TPT8 | 3.803 [cached: 3.333][^cached-higgs] | traversal BT256 |
| HIGGS | 1000 | 3.111 | 4.511 [cached: 3.693][^cached-higgs] | BT256 | 5.291 | BT512/TPT4 | 4.511 [cached: 3.693][^cached-higgs] | traversal BT256 |
| HIGGS | 2000 | 10.719 | 4.691 [cached: 4.179][^cached-higgs] | BT256 | 5.763 | BT512/TPT8 | 4.691 [cached: 4.179][^cached-higgs] | traversal BT256 |
| HIGGS | 5000 | 11.959 | 6.891 | BT512 | 12.311 | BT512/TPT4 | 6.891 | traversal BT512 |
| epsilon | 500 | 2.427 | 7.027 | BT1024/!Mapped/!SP/!CrossReduce | 7.171 | BT1024/TPT16/row | 7.027 | traversal BT1024/!Mapped/!SP/!CrossReduce |
| epsilon | 1000 | 3.715 | 8.927 | BT1024/!Mapped/!SP/!CrossReduce | 7.935 | BT1024/TPT16/row | 7.935 | bitmask BT1024/TPT16/row |
| epsilon | 2000 | 5.931 | 9.295 | BT1024/!Mapped/SP/!CrossReduce | 8.879 | BT1024/TPT8/row | 8.879 | bitmask BT1024/TPT8/row |
| epsilon | 5000 | 12.559 | 9.871 | BT1024/!Mapped/SP/!CrossReduce | 13.903 | BT1024/TPT4 | 9.871 | traversal BT1024/!Mapped/SP/!CrossReduce |

[^cached-higgs]: Bracketed HIGGS values are cached-tree persistent traversal results from `gpu_cached/`; the leading value remains the uncached final persistent GPU result. HIGGS n=5000 and all epsilon rows keep the uncached final GPU result.

Under the controlled CPU settings used for the main tables, the persistent GPU path beats the best CPU result for HIGGS n=2000, HIGGS n=5000, and epsilon n=5000. It still does not universally beat the CPU path, especially on smaller forests, and the best GPU policy is row-dependent rather than a single universal configuration.

## Cached Trees Bonus

Tree caching is conditional and is not part of the main one-shot story. In the controlled GPU root, the strongest cached-tree result is persistent traversal. It helps the small HIGGS cases when the cached tree set fits:

All rows in this table are traversal kernels (`bitmask:0`). The global uncached column matches the best-result table above. The matched uncached column is the uncached row from the same BT64/BT96 cached sweep, which is useful for isolating the effect of tree caching without mixing in the separate BT256 release sweep. The cached candidates use mapped memory, single-poller, no seq flag, and no cross-block reduction. The cached trees are stored in row-major order in shared memory, and the threads from neighboring trees are arranged to access different banks.

| case | global uncached traversal | matched uncached | cached one-wave | cached all-waves |
| --- | --- | --- | --- | --- |
| HIGGS n=500 | 3.803 | 4.191 | 3.333 | 3.407 |
| HIGGS n=1000 | 4.511 | 4.639 | 3.693 | 3.767 |
| HIGGS n=2000 | 4.691 | 5.159 | 4.179 | 4.255 |

For HIGGS n=5000 the cached persistent traversal variants were skipped because the persistent grid was not co-resident. For epsilon, the cached variants that can run do not beat the best uncached final GPU result; the larger feature vector makes the shared-memory/cache trade-off less favorable.

## Discussion

The experiments above show that batch-1 tree-ensemble inference is a different optimization problem from throughput-oriented serving. For a single host-visible request, the dominant costs are often not the tree comparisons themselves, but the surrounding systems costs: thread placement, CPU frequency policy, kernel launch, host-device signaling, feature movement, cross-block coordination, and final result visibility.

The controlled CPU-frequency policy makes one machine-specific point clearer: the CPU path is highly sensitive to frequency policy. QLeaf CPU traversal is still extremely strong and remains the best path for smaller forests, while the persistent GPU wins several larger cases under the predictable-frequency policy used here. That does not make the GPU universally better; it means the CPU/GPU conclusion depends on both workload shape and platform policy.

The GPU path exposes a different trade-off. A one-shot CUDA kernel has enough tree-level parallelism, but every request pays launch and completion overhead. A persistent kernel removes the launch cost, but the naive version is not automatically faster. If many blocks poll host-visible memory or every request requires device-wide completion coordination, the synchronization overhead can dominate the actual tree walk.

The optimization path therefore removes coordination before relying on mapped memory: no cross-block reduction, then single-poller request detection, then mapped input, staged features, and sentinel signaling. Mapped memory helps most when it removes copy overhead without letting every resident block repeatedly observe host-visible state. Sentinel-based signaling further reduces the need for a separate ordered flag, which is valid here because the benchmark uses a single in-flight request and the host waits for completion before issuing the next row.

The checked multi-poller sentinel experiment is the clearest negative result: making every block participate in the handshake is slower than letting one poller stage and broadcast the request. Tree caching provides an additional conditional improvement on small HIGGS models, but it is not a general win because the shared-memory footprint can reduce residency or fail co-residency entirely.

Overall, QLeaf is best understood as a focused exploration of a real low-latency niche. Public tree-inference libraries are usually designed for broader model support, portability, and throughput. QLeaf specializes in the opposite direction: one row, one model, one immediate result.

## Evidence Check

The CPU and external-CPU full root has the data needed for those baselines:

`bench/results/latency/full_regen_no_turbo_100k_2026_06_19`

The locked-clock GPU root has the final GPU, one-shot GPU, nvForest GPU, decomposition, cached-tree, multi-poller, and correctness data:

`bench/results/latency/gpu_locked_no_turbo_newpath_100k_2026_06_19`

The locked-clock HDR ablation root has the controlled persistent optimization path:

`bench/results/latency/ablation_hdr_gpu_locked_no_turbo_newpath_100k_2026_06_19`

Generated table bundles are retained at:

`bench/results/latency/full_regen_no_turbo_100k_2026_06_19/manifest/README_TABLES.md`

`bench/results/latency/gpu_locked_no_turbo_newpath_100k_2026_06_19/manifest/README_TABLES.md`

`bench/results/latency/ablation_hdr_gpu_locked_no_turbo_newpath_100k_2026_06_19/manifest/README_ABLATION_TABLES.md`

- CPU versus external CPU baselines: `cpu_physical_first/`, `cpu_sibling_first/`, `external_cpu/`, `external_xgboost_capi/`, and `external_nvforest_cpu/` under the CPU root.
- External GPU baseline: `external_nvforest_rpb1/` under the locked-clock GPU root.
- One-shot transposed versus row-major: `gpu_oneshot_uncached/`.
- Persistent switch ablation: `gpu_ablation_path/` and `gpu_ablation_tpt/` under the locked-clock ablation root.
- HDR decomposition: `gpu_naive_decomp_profile/`.
- Checked multi-poller, non-mapped `!SeqFlag` sentinel sweep: `gpu_mp_sentinel_correct/`.
- Final fixed-iteration GPU candidates: `gpu_best/` plus the controlled ablation/TPT rows when they are lower.
- Cached-tree bonus: `gpu_cached/`, `gpu_cached_bitmask_higgs/`, and `gpu_cached_bitmask_epsilon/`.

Correctness smoke completed across both datasets and all tree counts: 88 rows reported `ok`, 16 cached rows were skipped because the configuration was not co-resident or required too much dynamic shared memory, and no row failed. The persistent HDR ablation completed with 222 `ok` rows, 18 skipped rows, and no failed rows. The locked GPU run recorded all P0 SM-clock samples at 1800 MHz; 7 of 389 P0 memory-clock samples reported 7301 MHz instead of 7501 MHz.
