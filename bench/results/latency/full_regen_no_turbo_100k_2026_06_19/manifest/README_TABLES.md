## Preview
| dataset | trees | QLeaf CPU best | QLeaf GPU best | External CPU best | External CPU engine | External GPU |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.177 | 4.147 | 22.943 | Treelite TL2cgen | 10.591 |
| HIGGS | 1000 | 3.111 | 4.315 | 35.807 | nvForest CPU | 14.135 |
| HIGGS | 2000 | 10.719 | 4.739 | 49.503 | nvForest CPU | 21.263 |
| HIGGS | 5000 | 11.959 | 6.939 | 77.887 | nvForest CPU | 68.095 |
| epsilon | 500 | 2.427 | 9.159 | 28.383 | nvForest CPU | 13.039 |
| epsilon | 1000 | 3.715 | 9.407 | 38.495 | nvForest CPU | 16.527 |
| epsilon | 2000 | 5.931 | 9.487 | 49.695 | nvForest CPU | 23.295 |
| epsilon | 5000 | 12.559 | 10.975 | 74.943 | nvForest CPU | 39.295 |

## Input Artifacts
| dataset | features | depth | tree counts | feature rows | QLeaf model | external model |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 28 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/final_2026_06/inputs/bench/data/higgs_d6_n5000.qleaf.json` | `bench/results/latency/final_2026_06/inputs/bench/data/higgs_d6_n5000.xgb.json` |
| epsilon | 2000 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/final_2026_06/inputs/bench/data/epsilon_d6_n5000.qleaf.json` | `bench/results/latency/final_2026_06/inputs/bench/data/epsilon_d6_n5000.xgb.json` |

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
| HIGGS | 500 | 2.177 | physical `threads:8` | 22.943 | 130.239 | 24.687 | 10.591 |
| HIGGS | 1000 | 3.111 | physical `threads:8` | 75.967 | 211.071 | 35.807 | 14.135 |
| HIGGS | 2000 | 10.719 | sibling `threads:16` | 234.367 | 377.087 | 49.503 | 21.263 |
| HIGGS | 5000 | 11.959 | sibling `threads:16` | 639.487 | 1020.927 | 77.887 | 68.095 |
| epsilon | 500 | 2.427 | physical `threads:8` | 50.911 | 137.215 | 28.383 | 13.039 |
| epsilon | 1000 | 3.715 | sibling `threads:16` | 119.679 | 218.239 | 38.495 | 16.527 |
| epsilon | 2000 | 5.931 | physical `threads:8` | 214.399 | 376.319 | 49.695 | 23.295 |
| epsilon | 5000 | 12.559 | sibling `threads:16` | 357.119 | 768.511 | 74.943 | 39.295 |

## One-Shot
| dataset | trees | QLeaf transposed | QLeaf row-major | nvForest host_host | nvForest device_host | nvForest device_device_sync |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | **26.751** | 27.263 | 29.567 | 22.447 | 10.591 |
| HIGGS | 1000 | 27.151 | **27.103** | 29.119 | 22.447 | 14.135 |
| HIGGS | 2000 | **27.455** | 28.255 | 37.727 | 30.639 | 21.263 |
| HIGGS | 5000 | **35.423** | 40.415 | 87.103 | 80.063 | 68.095 |
| epsilon | 500 | **29.503** | 29.583 | 30.047 | 22.479 | 13.039 |
| epsilon | 1000 | **29.759** | 29.999 | 31.343 | 30.671 | 16.527 |
| epsilon | 2000 | **30.863** | 32.639 | 37.951 | 31.231 | 23.295 |
| epsilon | 5000 | **38.687** | 49.407 | 55.071 | 47.135 | 39.295 |

One-shot transposed wins 7 of 8 rows; row-major wins: HIGGS n=1000.

## Naive Persistent
| dataset | kernel | BT | one-shot reference | naive persistent |
| --- | --- | --- | --- | --- |
| HIGGS n=5000 | traversal | 512 | 14.409 | 29.500 |
| HIGGS n=5000 | bitmask TPT1 | 512 | 19.932 | 39.705 |
| epsilon n=5000 | traversal | 1024 | 17.445 | 20.478 |
| epsilon n=5000 | bitmask TPT1 | 1024 | 24.461 | 41.396 |

## Step 0
| dataset | trees | traversal | bitmask TPT1 |
| --- | --- | --- | --- |
| HIGGS | 500 | 9.684 | 14.088 |
| HIGGS | 1000 | 10.410 | 14.593 |
| HIGGS | 2000 | 13.677 | 17.803 |
| HIGGS | 5000 | 29.500 | 39.705 |
| epsilon | 500 | 12.847 | 21.715 |
| epsilon | 1000 | 14.507 | 31.695 |
| epsilon | 2000 | 15.412 | 32.993 |
| epsilon | 5000 | 20.478 | 41.396 |

## Step 1
| dataset | trees | traversal before | traversal +mapped | bitmask before | bitmask +mapped |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 9.684 | 6.094 | 14.088 | 10.527 |
| HIGGS | 1000 | 10.410 | 7.108 | 14.593 | 11.282 |
| HIGGS | 2000 | 13.677 | 9.730 | 17.803 | 14.036 |
| HIGGS | 5000 | 29.500 | 27.421 | 39.705 | 37.230 |
| epsilon | 500 | 12.847 | 10.764 | 21.715 | 19.682 |
| epsilon | 1000 | 14.507 | 12.957 | 31.695 | 30.280 |
| epsilon | 2000 | 15.412 | 13.345 | 32.993 | 31.441 |
| epsilon | 5000 | 20.478 | 19.010 | 41.396 | 40.255 |

## Step 2
| dataset | trees | traversal cross-reduce | traversal no-cross | bitmask cross-reduce | bitmask no-cross |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 6.094 | 4.707 | 10.527 | 9.034 |
| HIGGS | 1000 | 7.108 | 5.130 | 11.282 | 9.579 |
| HIGGS | 2000 | 9.730 | 6.613 | 14.036 | 11.280 |
| HIGGS | 5000 | 27.421 | 18.201 | 37.230 | 27.300 |
| epsilon | 500 | 10.764 | 9.197 | 19.682 | 18.185 |
| epsilon | 1000 | 12.957 | 11.639 | 30.280 | 28.391 |
| epsilon | 2000 | 13.345 | 11.900 | 31.441 | 29.553 |
| epsilon | 5000 | 19.010 | 14.830 | 40.255 | 35.237 |

## Step 3
| dataset | trees | traversal no-cross | traversal single-poller | bitmask no-cross | bitmask single-poller |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 4.707 | 4.993 | 9.034 | 9.395 |
| HIGGS | 1000 | 5.130 | 5.623 | 9.579 | 9.723 |
| HIGGS | 2000 | 6.613 | 5.624 | 11.280 | 10.114 |
| HIGGS | 5000 | 18.201 | 7.506 | 27.300 | 18.768 |
| epsilon | 500 | 9.197 | 10.133 | 18.185 | 18.989 |
| epsilon | 1000 | 11.639 | 11.936 | 28.391 | 29.106 |
| epsilon | 2000 | 11.900 | 12.153 | 29.553 | 29.843 |
| epsilon | 5000 | 14.830 | 12.811 | 35.237 | 33.983 |

## Step 4
| dataset | trees | traversal seqflag | traversal sentinel | bitmask seqflag | bitmask sentinel |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 4.993 | 4.375 | 9.395 | 9.615 |
| HIGGS | 1000 | 5.623 | 4.517 | 9.723 | 9.589 |
| HIGGS | 2000 | 5.624 | 5.179 | 10.114 | 9.565 |
| HIGGS | 5000 | 7.506 | 7.037 | 18.768 | 18.381 |
| epsilon | 500 | 10.133 | 7.831 | 18.989 | 16.842 |
| epsilon | 1000 | 11.936 | 9.520 | 29.106 | 26.893 |
| epsilon | 2000 | 12.153 | 10.814 | 29.843 | 27.819 |
| epsilon | 5000 | 12.811 | 10.901 | 33.983 | 32.818 |

## Step 5
| dataset | trees | bitmask TPT1 | tuned bitmask | tuned cfg |
| --- | --- | --- | --- | --- |
| HIGGS | 500 | 9.615 | 4.787 | BT512/TPT8 |
| HIGGS | 1000 | 9.589 | 5.440 | BT512/TPT4 |
| HIGGS | 2000 | 9.565 | 6.342 | BT512/TPT8 |
| HIGGS | 5000 | 18.381 | 13.112 | BT512/TPT4 |
| epsilon | 500 | 16.842 | 9.661 | BT1024/TPT16 |
| epsilon | 1000 | 26.893 | 10.066 | BT1024/TPT8 |
| epsilon | 2000 | 27.819 | 11.581 | BT1024/TPT8 |
| epsilon | 5000 | 32.818 | 16.043 | BT1024/TPT2 |

## Naive Decomposition Host
| case | BT | total p50 | predict p50 | host poll avg | host gap avg |
| --- | --- | --- | --- | --- | --- |
| HIGGS n=5000 traversal | 512 | 24.303 | 24.207 | 18.374 | 5.800 |
| HIGGS n=5000 bitmask TPT1 | 512 | 28.911 | 28.831 | 23.366 | 5.722 |
| epsilon n=5000 traversal | 1024 | 19.791 | 19.695 | 11.929 | 8.225 |
| epsilon n=5000 bitmask TPT1 | 1024 | 40.607 | 40.543 | 32.629 | 8.178 |

## Naive Decomposition Device
| case | wait cyc | feat | walk | reduce | publish |
| --- | --- | --- | --- | --- | --- |
| HIGGS traversal | 38,212 | 564 | 6,410 | 17,988 | 17,070 |
| HIGGS bitmask | 18,741 | 1,459 | 27,738 | 5,797 | 4,002 |
| epsilon traversal | 27,097 | 950 | 7,338 | 7,252 | 6,033 |
| epsilon bitmask | 21,824 | 2,018 | 47,731 | 6,990 | 4,188 |

## Multi-Poller Sentinel
| dataset | trees | multi-poller traversal | final traversal best | multi / final | multi-poller bitmask best | final bitmask best | multi / final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 14.079 | 4.147 | 3.39x | 16.511 | 4.735 | 3.49x |
| HIGGS | 1000 | 14.927 | 4.315 | 3.46x | 19.919 | 5.483 | 3.63x |
| HIGGS | 2000 | 16.607 | 4.739 | 3.50x | 21.359 | 6.347 | 3.37x |
| HIGGS | 5000 | 22.863 | 6.939 | 3.29x | 34.847 | 12.663 | 2.75x |
| epsilon | 500 | 21.503 | 9.271 | 2.32x | 24.367 | 9.159 | 2.66x |
| epsilon | 1000 | 23.279 | 9.407 | 2.47x | 27.407 | 9.687 | 2.83x |
| epsilon | 2000 | 24.079 | 9.487 | 2.54x | 34.783 | 10.983 | 3.17x |
| epsilon | 5000 | 27.823 | 10.975 | 2.54x | 41.983 | 16.279 | 2.58x |

## Final GPU vs CPU
| dataset | trees | CPU 4-thread | CPU 8-thread | CPU 16-thread | GPU traversal best | traversal cfg | GPU bitmask best | bitmask cfg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.987 | 2.177 | 2.269 | 4.147 | BT256 | 4.735 | BT512/TPT8 |
| HIGGS | 1000 | 4.931 | 3.111 | 3.379 | 4.315 | BT256 | 5.483 | BT512/TPT4 |
| HIGGS | 2000 | 32.047 | 16.431 | 10.719 | 4.739 | BT256 | 6.347 | BT512/TPT4 |
| HIGGS | 5000 | 30.575 | 12.351 | 11.959 | 6.939 | BT512 | 12.663 | BT1024/TPT4 |
| epsilon | 500 | 3.339 | 2.427 | 2.507 | 9.271 | BT1024 | 9.159 | BT512/TPT8/row |
| epsilon | 1000 | 9.383 | 5.511 | 3.715 | 9.407 | BT1024 | 9.687 | BT512/TPT4 |
| epsilon | 2000 | 10.207 | 5.931 | 6.163 | 9.487 | BT512 | 10.983 | BT1024/TPT8/row |
| epsilon | 5000 | 88.703 | 31.343 | 12.559 | 10.975 | BT512 | 16.279 | BT1024/TPT2 |

## Cached Trees
| case | uncached traversal | cached one-wave | cached all-waves |
| --- | --- | --- | --- |
| HIGGS n=500 | 4.119 | 3.247 | 3.311 |
| HIGGS n=1000 | 4.715 | 3.849 | 3.913 |
| HIGGS n=2000 | 5.311 | 4.367 | 4.435 |

## Correctness
Correctness smoke rows: 88 ok, 16 skipped, 0 failed.
