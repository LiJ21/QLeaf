## Preview
| dataset | trees | QLeaf CPU best | QLeaf GPU best | External CPU best | External CPU engine | External GPU |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 0.863 | 3.637 | 10.111 | Treelite TL2cgen | 10.031 |
| HIGGS | 1000 | 1.521 | 4.251 | 17.199 | nvForest CPU | 13.959 |
| HIGGS | 2000 | 2.549 | 4.451 | 22.927 | nvForest CPU | 21.631 |
| HIGGS | 5000 | 5.679 | 6.575 | 35.871 | nvForest CPU | 67.007 |
| epsilon | 500 | 1.259 | 6.663 | 13.199 | nvForest CPU | 12.375 |
| epsilon | 1000 | 1.768 | 7.767 | 18.319 | nvForest CPU | 16.087 |
| epsilon | 2000 | 2.757 | 8.759 | 23.055 | nvForest CPU | 23.327 |
| epsilon | 5000 | 6.191 | 10.247 | 35.647 | nvForest CPU | 40.255 |

## Input Artifacts
| dataset | features | depth | tree counts | feature rows | QLeaf model | external model |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 28 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/final_2026_06/inputs/bench/data/higgs_d6_n5000.qleaf.json` | `bench/results/latency/final_2026_06/inputs/bench/data/higgs_d6_n5000.xgb.json` |
| epsilon | 2000 | 6 | 500, 1000, 2000, 5000 | 1024 | `bench/results/latency/final_2026_06/inputs/bench/data/epsilon_d6_n5000.qleaf.json` | `bench/results/latency/final_2026_06/inputs/bench/data/epsilon_d6_n5000.xgb.json` |

## CPU Affinity Wins
| worker threads | physical-first wins | sibling-first wins | note |
| --- | --- | --- | --- |
| 2 | 7 / 8 | 1 / 8 | distinct P-cores dominate |
| 4 | 7 / 8 | 1 / 8 | physical-first usually better |
| 8 | 6 / 8 | 2 / 8 | physical-first usually better |
| 16 | 4 / 8 | 4 / 8 | sibling-first often avoids E-core spill |

## Largest-Forest Affinity
| dataset | threads | physical-first p50 us | sibling-first p50 us |
| --- | --- | --- | --- |
| HIGGS n=5000 | 2 | 85.695 | 83.135 |
| HIGGS n=5000 | 4 | 37.055 | 41.087 |
| HIGGS n=5000 | 8 | 15.551 | 13.303 |
| HIGGS n=5000 | 16 | 11.143 | 5.679 |
| epsilon n=5000 | 2 | 44.319 | 83.903 |
| epsilon n=5000 | 4 | 15.399 | 83.583 |
| epsilon n=5000 | 8 | 6.191 | 40.575 |
| epsilon n=5000 | 16 | 13.383 | 19.567 |

## External Baselines
| dataset | trees | QLeaf CPU best | CPU config | Treelite TL2cgen | XGBoost C API | nvForest CPU | nvForest GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 0.863 | physical `threads:8` | 10.111 | 64.767 | 11.967 | 10.031 |
| HIGGS | 1000 | 1.521 | physical `threads:8` | 31.519 | 99.007 | 17.199 | 13.959 |
| HIGGS | 2000 | 2.549 | physical `threads:8` | 98.431 | 171.903 | 22.927 | 21.631 |
| HIGGS | 5000 | 5.679 | sibling `threads:16` | 299.263 | 568.319 | 35.871 | 67.007 |
| epsilon | 500 | 1.259 | physical `threads:8` | 21.567 | 65.503 | 13.199 | 12.375 |
| epsilon | 1000 | 1.768 | physical `threads:8` | 49.727 | 102.975 | 18.319 | 16.087 |
| epsilon | 2000 | 2.757 | sibling `threads:16` | 88.895 | 172.927 | 23.055 | 23.327 |
| epsilon | 5000 | 6.191 | physical `threads:8` | 149.503 | 374.271 | 35.647 | 40.255 |

## One-Shot
| dataset | trees | QLeaf transposed | QLeaf row-major | nvForest host_host | nvForest device_host | nvForest device_device_sync |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | **16.687** | 16.767 | 19.391 | 16.343 | 10.031 |
| HIGGS | 1000 | **16.687** | 16.767 | 27.615 | 24.543 | 13.959 |
| HIGGS | 2000 | **17.455** | 18.383 | 35.775 | 32.735 | 21.631 |
| HIGGS | 5000 | **24.991** | 30.159 | 76.863 | 73.791 | 67.007 |
| epsilon | 500 | 18.015 | **17.983** | 19.455 | 16.735 | 12.375 |
| epsilon | 1000 | **18.223** | 18.351 | 27.647 | 24.543 | 16.087 |
| epsilon | 2000 | **19.087** | 20.911 | 35.839 | 32.735 | 23.327 |
| epsilon | 5000 | **26.079** | 37.599 | 52.287 | 49.119 | 40.255 |

One-shot transposed wins 7 of 8 rows; row-major wins: epsilon n=500.

## Naive Persistent
| dataset | kernel | BT | one-shot reference | naive persistent |
| --- | --- | --- | --- | --- |
| HIGGS n=5000 | traversal | 512 | 11.989 | 36.706 |
| HIGGS n=5000 | bitmask TPT1 | 512 | 17.822 | 53.149 |
| epsilon n=5000 | traversal | 1024 | 15.920 | 23.753 |
| epsilon n=5000 | bitmask TPT1 | 1024 | 23.589 | 53.179 |

## Step 0
| dataset | trees | traversal | bitmask TPT1 |
| --- | --- | --- | --- |
| HIGGS | 500 | 9.977 | 16.349 |
| HIGGS | 1000 | 11.433 | 17.182 |
| HIGGS | 2000 | 15.271 | 21.349 |
| HIGGS | 5000 | 36.706 | 53.149 |
| epsilon | 500 | 12.309 | 24.900 |
| epsilon | 1000 | 15.057 | 38.853 |
| epsilon | 2000 | 15.878 | 40.414 |
| epsilon | 5000 | 23.753 | 53.179 |

## Step 1
| dataset | trees | traversal before | traversal +mapped | bitmask before | bitmask +mapped |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 9.977 | 6.243 | 16.349 | 10.932 |
| HIGGS | 1000 | 11.433 | 6.636 | 17.182 | 11.428 |
| HIGGS | 2000 | 15.271 | 9.843 | 21.349 | 14.631 |
| HIGGS | 5000 | 36.706 | 27.711 | 53.149 | 38.160 |
| epsilon | 500 | 12.309 | 9.145 | 24.900 | 18.738 |
| epsilon | 1000 | 15.057 | 11.557 | 38.853 | 29.310 |
| epsilon | 2000 | 15.878 | 12.122 | 40.414 | 30.927 |
| epsilon | 5000 | 23.753 | 19.035 | 53.179 | 40.201 |

## Step 2
| dataset | trees | traversal cross-reduce | traversal no-cross | bitmask cross-reduce | bitmask no-cross |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 6.243 | 4.696 | 10.932 | 9.294 |
| HIGGS | 1000 | 6.636 | 5.125 | 11.428 | 9.582 |
| HIGGS | 2000 | 9.843 | 6.573 | 14.631 | 11.329 |
| HIGGS | 5000 | 27.711 | 17.628 | 38.160 | 28.029 |
| epsilon | 500 | 9.145 | 7.498 | 18.738 | 16.841 |
| epsilon | 1000 | 11.557 | 10.141 | 29.310 | 27.878 |
| epsilon | 2000 | 12.122 | 10.377 | 30.927 | 28.853 |
| epsilon | 5000 | 19.035 | 13.886 | 40.201 | 35.088 |

## Step 3
| dataset | trees | traversal no-cross | traversal single-poller | bitmask no-cross | bitmask single-poller |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 4.696 | 5.104 | 9.294 | 9.647 |
| HIGGS | 1000 | 5.125 | 5.176 | 9.582 | 10.008 |
| HIGGS | 2000 | 6.573 | 5.529 | 11.329 | 10.049 |
| HIGGS | 5000 | 17.628 | 7.135 | 28.029 | 19.067 |
| epsilon | 500 | 7.498 | 8.188 | 16.841 | 17.742 |
| epsilon | 1000 | 10.141 | 10.195 | 27.878 | 28.466 |
| epsilon | 2000 | 10.377 | 10.610 | 28.853 | 30.136 |
| epsilon | 5000 | 13.886 | 11.156 | 35.088 | 33.199 |

## Step 4
| dataset | trees | traversal seqflag | traversal sentinel | bitmask seqflag | bitmask sentinel |
| --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 5.104 | 4.433 | 9.647 | 9.940 |
| HIGGS | 1000 | 5.176 | 4.525 | 10.008 | 10.066 |
| HIGGS | 2000 | 5.529 | 4.662 | 10.049 | 9.922 |
| HIGGS | 5000 | 7.135 | 6.529 | 19.067 | 18.914 |
| epsilon | 500 | 8.188 | 7.991 | 17.742 | 17.590 |
| epsilon | 1000 | 10.195 | 10.054 | 28.466 | 28.157 |
| epsilon | 2000 | 10.610 | 10.078 | 30.136 | 29.121 |
| epsilon | 5000 | 11.156 | 10.666 | 33.199 | 32.984 |

## Step 5
| dataset | trees | bitmask TPT1 | tuned bitmask | tuned cfg |
| --- | --- | --- | --- | --- |
| HIGGS | 500 | 9.940 | 4.584 | BT512/TPT8 |
| HIGGS | 1000 | 10.066 | 5.085 | BT512/TPT8 |
| HIGGS | 2000 | 9.922 | 5.616 | BT512/TPT8 |
| HIGGS | 5000 | 18.914 | 12.167 | BT512/TPT8 |
| epsilon | 500 | 17.590 | 7.748 | BT1024/TPT16 |
| epsilon | 1000 | 28.157 | 9.794 | BT1024/TPT8 |
| epsilon | 2000 | 29.121 | 9.706 | BT1024/TPT8 |
| epsilon | 5000 | 32.984 | 14.843 | BT1024/TPT4 |

## Naive Decomposition Host
| case | BT | total p50 | predict p50 | host poll avg | host gap avg |
| --- | --- | --- | --- | --- | --- |
| HIGGS n=5000 traversal | 512 | 20.687 | 20.655 | 18.207 | 2.700 |
| HIGGS n=5000 bitmask TPT1 | 512 | 25.775 | 25.743 | 23.680 | 2.716 |
| epsilon n=5000 traversal | 1024 | 14.455 | 14.415 | 11.142 | 3.882 |
| epsilon n=5000 bitmask TPT1 | 1024 | 35.839 | 35.807 | 32.539 | 3.818 |

## Naive Decomposition Device
| case | wait cyc | feat | walk | reduce | publish |
| --- | --- | --- | --- | --- | --- |
| HIGGS traversal | 46,456 | 564 | 6,167 | 16,907 | 16,012 |
| HIGGS bitmask | 22,750 | 1,441 | 26,804 | 7,920 | 6,076 |
| epsilon traversal | 24,837 | 1,595 | 7,763 | 5,384 | 4,511 |
| epsilon bitmask | 39,372 | 2,212 | 47,123 | 6,753 | 4,109 |

## Multi-Poller Sentinel
| dataset | trees | multi-poller traversal | final traversal best | multi / final | multi-poller bitmask best | final bitmask best | multi / final |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 10.703 | 3.637 | 2.94x | 12.935 | 4.179 | 3.10x |
| HIGGS | 1000 | 11.511 | 4.251 | 2.71x | 16.375 | 4.355 | 3.76x |
| HIGGS | 2000 | 13.263 | 4.451 | 2.98x | 17.999 | 5.019 | 3.59x |
| HIGGS | 5000 | 23.199 | 6.575 | 3.53x | 32.079 | 11.495 | 2.79x |
| epsilon | 500 | 15.471 | 7.795 | 1.98x | 18.239 | 6.663 | 2.74x |
| epsilon | 1000 | 17.327 | 9.311 | 1.86x | 19.967 | 7.767 | 2.57x |
| epsilon | 2000 | 18.127 | 9.399 | 1.93x | 25.327 | 8.759 | 2.89x |
| epsilon | 5000 | 23.151 | 10.247 | 2.26x | 39.359 | 14.407 | 2.73x |

## Final GPU vs CPU
| dataset | trees | CPU 4-thread | CPU 8-thread | CPU 16-thread | GPU traversal best | traversal cfg | GPU bitmask best | bitmask cfg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 1.312 | 0.863 | 1.205 | 3.637 | BT256 | 4.179 | BT512/TPT32/row/NS16 |
| HIGGS | 1000 | 2.263 | 1.521 | 1.656 | 4.251 | BT256 | 4.355 | BT256/TPT8 |
| HIGGS | 2000 | 4.191 | 2.549 | 2.689 | 4.451 | BT256 | 5.019 | BT256/TPT4 |
| HIGGS | 5000 | 37.055 | 13.303 | 5.679 | 6.575 | BT512 | 11.495 | BT512/TPT8 |
| epsilon | 500 | 1.591 | 1.259 | 1.779 | 7.795 | BT1024 | 6.663 | BT1024/TPT16/row |
| epsilon | 1000 | 2.579 | 1.768 | 2.223 | 9.311 | BT1024 | 7.767 | BT1024/TPT8 |
| epsilon | 2000 | 9.111 | 4.539 | 2.757 | 9.399 | BT512 | 8.759 | BT1024/TPT8 |
| epsilon | 5000 | 15.399 | 6.191 | 13.383 | 10.247 | BT1024 | 14.407 | BT1024/TPT2 |

## Cached Trees
| case | uncached traversal | cached one-wave | cached all-waves |
| --- | --- | --- | --- |
| HIGGS n=500 | 3.943 | 2.475 | 2.545 |
| HIGGS n=1000 | 4.065 | 3.247 | 3.319 |
| HIGGS n=2000 | 4.507 | 3.543 | 3.615 |

## Correctness
Correctness smoke rows: 88 ok, 16 skipped, 0 failed.
