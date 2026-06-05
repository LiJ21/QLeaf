# QLeaf: Low-Latency C++ Inference for Large Ensembles of Shallow Trees

QLeaf targets a narrow but real deployment regime: batch-1, host-visible inference for large ensembles of shallow decision trees. In this setting, a model may contain hundreds or thousands of depth-limited trees, but each request consists of a single feature row and must be answered immediately. The objective is not maximum throughput over a large batch, but minimum end-to-end latency for one request.

This regime appears when tree models sit inside latency-sensitive systems: online decisioning, event-driven prediction, real-time ranking, risk checks, simulation loops, scientific control systems, or trading-style pipelines. In such settings, batching may be unavailable or undesirable because waiting for more rows would itself add latency.

## Why QLeaf?

QLeaf explores batch-1 low-latency inference for tree ensembles, where kernel launch, synchronization, memory layout, and host-device coordination can dominate arithmetic. The project compares CPU thread-pinned inference, CUDA one-shot and persistent-kernel designs, and production libraries under a fixed-iteration latency harness.

Tree ensembles remain a strong and widely used model family for tabular data. Gradient-boosted and bagged trees are often competitive, robust, and easier to deploy than larger neural models. However, most mature public tree-inference systems are designed primarily around broad model support, portability, and throughput. That is usually the right engineering target, but it leaves an important corner case: true batch-1 inference with a host-visible result and a microsecond-scale latency budget.

The project is not intended as a general replacement for XGBoost, LightGBM, Treelite, FIL, nvForest, or other mature inference libraries. Those systems solve broader problems. QLeaf instead focuses on a narrower systems question: what design choices help or hurt when the only objective is low-latency single-row inference? The answer depends on workload, hardware, and the trade-off between cost and performance. Rather than hard-coding one universal policy, QLeaf exposes a toolkit of optimization modules that can be enabled according to benchmark results.

At the moment QLeaf supports and benchmarks only a single scalar raw margin per row. This covers regression-style additive tree outputs and binary-classification margins, but full classification support, such as probability/label transforms and multiclass outputs, would require additional logic.

This makes the project both practical and pedagogical. It is practical because the target regime is real. It is pedagogical because the repo keeps the policy choices explicit: CPU traversal versus bitmask evaluation, worker threading, CPU affinity, one-shot CUDA kernels, persistent CUDA workers, mapped memory, transposed tree layouts, polling strategies, feature staging, cross-block coordination, and shared-memory caching.

The central question is not simply whether GPUs have more parallelism. They do. The harder question is whether that parallelism can be exposed without losing the latency budget to kernel launch overhead, host-device synchronization, feature movement, polling, reduction, and cross-block coordination.

## Benchmark: CPU QLeaf v. GPU QLeaf v. Externals

The table below is the destination, not the starting point. The detailed
benchmark report in [BENCHMARK.md](BENCHMARK.md) explains how the CPU numbers
depend on threading and affinity, why the first GPU versions are not enough, and
which persistent-kernel changes are needed to reach the final GPU column. Simply put,
for shallow tree ensembles at batch size 1, GPU arithmetic parallelism is often dominated by host-device coordination. Persistent kernels reduce launch overhead but introduce their own synchronization and staging costs. CPU implementations remain highly competitive when traversal is cache-friendly and threads are pinned.

All values are p50 microseconds from the fixed-iteration HDR harness. Bracketed
values in the CPU column are turbo-enabled reference measurements on the same
machine.[^turbo-cpu]

`QLeaf GPU best` is the best final persistent GPU result for that row. `External
CPU best` is the best of TL2cgen, native XGBoost C API, and nvForest CPU.
`External GPU` is nvForest's fastest batch-1 `device_device_sync` result, so it
should be read as a favorable device-resident external GPU lower bound rather
than an identical host-output contract.

| dataset | trees | QLeaf CPU best | QLeaf GPU best | External CPU best | External CPU engine | External GPU |
| --- | --- | --- | --- | --- | --- | --- |
| HIGGS | 500 | 2.177 [turbo: 0.863] | 3.803 [cached: 3.333][^cached-higgs] | 22.943 | Treelite TL2cgen | 11.399 |
| HIGGS | 1000 | 3.111 [turbo: 1.521] | 4.511 [cached: 3.693][^cached-higgs] | 35.807 | nvForest CPU | 14.951 |
| HIGGS | 2000 | 10.719 [turbo: 2.549] | 4.691 [cached: 4.179][^cached-higgs] | 49.503 | nvForest CPU | 22.159 |
| HIGGS | 5000 | 11.959 [turbo: 5.679] | 6.891 | 77.887 | nvForest CPU | 70.591 |
| epsilon | 500 | 2.427 [turbo: 1.259] | 7.027 | 28.383 | nvForest CPU | 13.655 |
| epsilon | 1000 | 3.715 [turbo: 1.768] | 7.935 | 38.495 | nvForest CPU | 17.359 |
| epsilon | 2000 | 5.931 [turbo: 2.757] | 8.879 | 49.695 | nvForest CPU | 24.607 |
| epsilon | 5000 | 12.559 [turbo: 6.191] | 9.871 | 74.943 | nvForest CPU | 42.143 |

[^turbo-cpu]: Bracketed CPU values come from the turbo-enabled reference run in `bench/results/latency/full_regen_2026_06_18/`; its manifest records Intel pstate `no_turbo=0`. They show the lower latency available when CPU boost is allowed, while the leading CPU values keep turbo disabled for the controlled benchmark policy.

[^cached-higgs]: Bracketed HIGGS values are cached-tree persistent traversal results from `gpu_cached/`; the leading value remains the uncached final persistent GPU result. HIGGS n=5000 and all epsilon rows keep the uncached final GPU result.

The short version is: CPU remains very strong under the controlled benchmark
policy; external libraries are not optimized for this narrow batch-1 shape; and
the GPU only becomes competitive after removing launch, polling, and
coordination overheads one by one.

## Overall Design

The main interface is `qleaf::Inferrer`. It owns the model representation, manages backend workers, and dispatches tree partitions to either CPU or CUDA execution paths.

On CPU, QLeaf uses compact tree layouts and depth-limited path traversal. For shallow trees, path traversal is hard to beat: each tree follows only one root-to-leaf path, the feature row is already in host memory, and there is no device synchronization. Multi-threaded CPU workers can reduce latency to the single-digit microsecond range for large forests, but the cost is that several physical cores may be dedicated to a continuously waiting inference path.

On GPU, the trade-offs are different. A one-shot CUDA kernel exposes tree-level parallelism but pays launch overhead for every request. A persistent CUDA kernel removes launch overhead, but introduces a new problem: the host and device must communicate each request, and many GPU blocks must coordinate to produce one result. Naive persistent designs can therefore be worse than one-shot kernels if many blocks poll host memory or if results are combined through expensive device-wide reduction.

QLeaf studies these trade-offs through a sequence of increasingly specialized GPU policies: transposed node-major layouts for memory coalescing, mapped host/device communication, single-poller request detection, feature staging, avoiding cross-block reduction, sentinel-based signaling, and shared-memory tree caching where the model and feature shape allow it.

The goal of the benchmark section is not only to report final latency numbers, but to explain why each design behaves the way it does. In this regime, raw compute is rarely the only bottleneck. The harder problems are data movement, synchronization, and coordination.

QLeaf is therefore best read as a focused exploration of a real low-latency inference niche: how much can be gained by specializing for batch-1 tree-ensemble serving, and where do CPU and GPU designs fundamentally differ?

## Minimal Inferrer Examples

`Inferrer` is configured in two layers. The C++ type selects the execution
policy, node layout, load balancer, and reducer. The runtime JSON supplies the
forest and one `worker` entry per tree partition. `qleaf::Config` references the
JSON object, so keep the JSON alive while constructing the inferrer.
The `num_feature` field is required by CUDA workers and ignored by CPU workers.

```cpp
#include <vector>

#include "qleaf.h"

int main() {
  nlohmann::json model{
      {"depth", 2},
      {"trees",
       {{{"indices", {0, 1, 1, 0, 0, 0, 0}},
         {"splits", {0.5f, 0.5f, 0.5f, 10.0f, 20.0f, 30.0f, 40.0f}}}}},
      {"worker", {{{"has_equal", false}, {"num_feature", 2}}}},
  };
  qleaf::Config config{model};

  using CpuInferrer = qleaf::Inferrer<
      float, qleaf::BranchRegressionWorker, qleaf::CompactNodeBuffer,
      qleaf::detail::FairBalancer, qleaf::RegressionReducer>;

  CpuInferrer cpu{config};
  float margin = cpu.predict(std::vector<float>{0.3f, 0.7f});
}
```

For CPU inference, `BranchRegressionWorker` follows one root-to-leaf path per
tree and is the default choice for shallow low-latency traversal.
`BitmaskRegressionWorker` evaluates the tree with a QuickScorer-style leaf mask,
which can expose more regular work at the cost of doing more comparisons per
tree. `ThreadedWorker<...>` wraps either CPU worker in a persistent host thread;
each `worker` entry may include a `core` field for Linux CPU affinity.
`CompactNodeBuffer` stores split values and feature indices separately and is
the layout used by the main benchmarks; `TreeNodeBuffer` is the simpler direct
node layout. `FairBalancer` partitions trees evenly across workers, and
`RegressionReducer` sums their raw margins.

The CUDA worker has one extra template layer because the CUDA traversal policy
is itself a type. Compile CUDA inferrers with a CUDA-enabled target and include
`gworkers.cuh`.

```cpp
#include <vector>

#include "gworkers.cuh"
#include "qleaf.h"

template <typename TPolicy>
struct CudaWorkerBinding {
  template <typename TValue, typename TSpan>
  using Worker = qleaf::CudaWorker<TPolicy, TValue, TSpan>;
};

using GpuPolicy = qleaf::CudaTraversePersistentSP<
    512, true, false, false, false, false, true, false, true>;

using GpuInferrer = qleaf::Inferrer<
    float, CudaWorkerBinding<GpuPolicy>::template Worker,
    qleaf::CompactNodeBuffer, qleaf::detail::FairBalancer,
    qleaf::RegressionReducer>;

int main() {
  nlohmann::json model{
      {"depth", 2},
      {"trees",
       {{{"indices", {0, 1, 1, 0, 0, 0, 0}},
         {"splits", {0.5f, 0.5f, 0.5f, 10.0f, 20.0f, 30.0f, 40.0f}}}}},
      {"worker", {{{"has_equal", false}, {"num_feature", 2}}}},
  };
  qleaf::Config config{model};

  GpuInferrer gpu{config};
  float margin = gpu.predict(std::vector<float>{0.3f, 0.7f});
}
```

CUDA policies are named for the launch strategy. `CudaTraverseOneShot` launches
a kernel per request, `CudaTraversePersistent` keeps a resident kernel alive,
and `CudaTraversePersistentSP` adds the single-poller persistent path used by
the fastest low-latency variants. The example policy expands as follows:

```cpp
qleaf::CudaTraversePersistentSP<
    kBT,
    kTransposed,
    kBitmask,
    kCached,
    kMapped,
    kCrossReduce,
    kSinglePoller,
    kStageFeatures,
    kSeqFlag,
    kTPT,
    kCacheAllWaves,
    kCacheNodeMajor,
    kNodeStride>
```

| parameter | example value | meaning |
| --- | --- | --- |
| `kBT` | `512` | CUDA threads per block. |
| `kTransposed` | `true` | Store tree nodes in transposed/node-major order for more coalesced GPU access. |
| `kBitmask` | `false` | Select traversal when false, or QuickScorer-style bitmask evaluation when true. |
| `kCached` | `false` | Stage tree nodes in shared memory when true. |
| `kMapped` | `false` | When true, use mapped host/device memory for the request and result path. |
| `kCrossReduce` | `false` | Reduce block partials on the device when true; publish per-block result slots for host collection when false. |
| `kSinglePoller` | `true` | Let one resident block poll for a new request and broadcast it to follower blocks. |
| `kStageFeatures` | `false` | When true, the single poller copies the active feature buffer into a device-side staging buffer before follower blocks read it. |
| `kSeqFlag` | `true` | Use an explicit request sequence flag when true; use the feature-buffer sentinel protocol when false. |
| `kTPT` | default `1` | Threads per tree for bitmask evaluation. Values above 1 require `kBitmask=true`. |
| `kCacheAllWaves` | default `false` | Cache every resident wave of trees when true; requires caching and no cross-reduce. |
| `kCacheNodeMajor` | default `false` | Use node-major shared-memory caching when true; only valid for cached bitmask kernels. |
| `kNodeStride` | default `1` | Permute bitmask node visitation order for `kTPT>1` bitmask kernels. |

So the minimal GPU example uses `BT=512`, transposed traversal, uncached
non-mapped persistent execution, no cross-block reduction, single-poller request
detection, no feature staging, and explicit sequence-flag signaling. A GPU
`worker` entry must include `num_feature`; persistent runs may also set
`persistent_grid` to override the resident grid size.

## AI Assistance Statement

### Scope

AI was used for writing benchmarks and unit tests as well as scaffolding,
refactoring, and iteration, but not for design decisions. OpenAI Codex was used
to assist with benchmark running and data bookkeeping.

### Review

All such code has been:
- reviewed,
- revised,
- and explicitly approved by a human before inclusion.

### Responsibility

Correctness, design decisions, and performance claims are the responsibility of the human author(s).
AI is used as a drafting tool, not as a source of truth.
