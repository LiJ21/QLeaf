## Preview
| dataset | trees | QLeaf CPU best | QLeaf GPU best | External CPU best | External CPU engine | External GPU |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.177 | 3.803 [cached: 3.333][^cached-higgs] | 22.943 | Treelite TL2cgen | 11.399 |
| HIGGS | 1000 | 3.111 | 4.511 [cached: 3.693][^cached-higgs] | 35.807 | nvForest CPU | 14.951 |
| HIGGS | 2000 | 10.719 | 4.691 [cached: 4.179][^cached-higgs] | 49.503 | nvForest CPU | 22.159 |
| HIGGS | 5000 | 11.959 | 6.891 | 77.887 | nvForest CPU | 70.591 |
| epsilon | 500 | 2.427 | 7.027 | 28.383 | nvForest CPU | 13.655 |
| epsilon | 1000 | 3.715 | 7.935 | 38.495 | nvForest CPU | 17.359 |
| epsilon | 2000 | 5.931 | 8.879 | 49.695 | nvForest CPU | 24.607 |
| epsilon | 5000 | 12.559 | 9.871 | 74.943 | nvForest CPU | 42.143 |

[^cached-higgs]: Bracketed HIGGS values are cached-tree persistent traversal results from `gpu_cached/`; the leading value remains the uncached final persistent GPU result.

## Input Artifacts
| dataset | features | depth | tree counts | feature rows | QLeaf model | external model |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 28 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/higgs_d6_n5000.qleaf.json` | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/higgs_d6_n5000.xgb.json` |
| epsilon | 2000 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/epsilon_d6_n5000.qleaf.json` | `bench/results/latency/ablation_hdr_2026_06_19/inputs/bench/data/epsilon_d6_n5000.xgb.json` |

## CPU Affinity Wins
| worker threads | physical-first wins | sibling-first wins | note |
| --- | --- | --- | --- |
| 2 | 5 / 8 | 3 / 8 | distinct P-cores dominate |
| 4 | 7 / 8 | 1 / 8 | physical-first usually better |
| 8 | 6 / 8 | 2 / 8 | physical-first usually better |
| 16 | 1 / 8 | 7 / 8 | sibling-first often avoids E-core spill |

## Largest-Forest Affinity
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

## External Baselines
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

## One-Shot
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

## Naive Decomposition
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

## Multi-Poller Sentinel
| dataset | trees | multi-poller traversal | final traversal best | multi / final | multi-poller bitmask best | final bitmask best | multi / final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 10.815 | 3.803 | 2.84x | 13.751 | 4.731 | 2.91x |
| HIGGS | 1000 | 11.967 | 4.511 | 2.65x | 17.391 | 5.291 | 3.29x |
| HIGGS | 2000 | 13.519 | 4.691 | 2.88x | 18.687 | 5.763 | 3.24x |
| HIGGS | 5000 | 24.975 | 6.891 | 3.62x | 34.719 | 12.311 | 2.82x |
| epsilon | 500 | 14.799 | 7.027 | 2.11x | 17.887 | 7.171 | 2.49x |
| epsilon | 1000 | 17.007 | 8.927 | 1.90x | 20.831 | 7.935 | 2.63x |
| epsilon | 2000 | 17.583 | 9.295 | 1.89x | 27.903 | 8.879 | 3.14x |
| epsilon | 5000 | 21.295 | 9.871 | 2.16x | 36.639 | 13.903 | 2.64x |

## Best Results: HIGGS and epsilon
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

## Cached Trees
| case | uncached traversal | cached one-wave | cached all-waves |
| --- | --- | --- | --- |
| HIGGS n=500 | 4.191 | 3.333 | 3.407 |
| HIGGS n=1000 | 4.639 | 3.693 | 3.767 |
| HIGGS n=2000 | 5.159 | 4.179 | 4.255 |
