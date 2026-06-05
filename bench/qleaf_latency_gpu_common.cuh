#pragma once

#define JSON_HAS_RANGES 0
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"
#include "gworkers.cuh"
#include "qleaf_latency_common.h"

namespace qleaf_latency_gpu {

#ifndef QLEAF_BT
#define QLEAF_BT 256
#endif
constexpr int kCompiledBT = QLEAF_BT;

inline int64_t now_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    throw std::runtime_error(std::string{"clock_gettime failed: "} +
                             std::strerror(errno));
  }
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
}

inline void cuda_throw(cudaError_t e, std::string_view op) {
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string{op} +
                           " failed: " + cudaGetErrorString(e));
}

// Mapped memory feature buffer
class MappedFeatureRow {
 public:
  MappedFeatureRow() = default;
  explicit MappedFeatureRow(const std::vector<float> &values)
      : size_(values.size()) {
    if (size_ == 0) return;
    cuda_throw(
        cudaHostAlloc(&data_, size_ * sizeof(float), cudaHostAllocMapped),
        "cudaHostAllocMapped");
    std::copy(values.begin(), values.end(), data_);
  }
  ~MappedFeatureRow() { reset(); }
  MappedFeatureRow(const MappedFeatureRow &) = delete;
  MappedFeatureRow &operator=(const MappedFeatureRow &) = delete;
  MappedFeatureRow(MappedFeatureRow &&other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}
  size_t size() const { return size_; }
  const float *data() const { return data_; }
  void load(const std::vector<float> &v) const {
    std::copy(v.begin(), v.end(), data_);
  }

 private:
  void reset() {
    if (data_) cudaFreeHost(data_);
    data_ = nullptr;
    size_ = 0;
  }
  float *data_ = nullptr;
  size_t size_ = 0;
};

struct GpuInput {
  const nlohmann::json &forest;
  const std::vector<std::vector<float>> &features;
  std::vector<MappedFeatureRow> mapped_features;
  bool cuda_enabled = false;
};

inline bool cuda_runtime_available() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (count == 0) return false;
  e = cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleAuto);
  if (e != cudaSuccess && e != cudaErrorSetOnActiveProcess) {
    cudaGetLastError();
    return false;
  }
  if (e == cudaErrorSetOnActiveProcess) cudaGetLastError();
  int can_map = 0;
  e = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0);
  if (e != cudaSuccess || !can_map) {
    if (e != cudaSuccess) cudaGetLastError();
    return false;
  }
  return true;
}

inline bool prepare_cuda_features(GpuInput &input) {
  if (!cuda_runtime_available()) return false;
  input.mapped_features.reserve(input.features.size());
  for (const auto &row : input.features)
    input.mapped_features.emplace_back(row);
  return true;
}

inline nlohmann::json config_for_gpu(const nlohmann::json &forest,
                                     size_t features, int persistent_grid) {
  nlohmann::json config = forest;
  config["worker"] = nlohmann::json::array();
  nlohmann::json worker{{"has_equal", false}, {"num_feature", features}};
  if (persistent_grid > 0) worker["persistent_grid"] = persistent_grid;
  config["worker"].push_back(worker);
  return config;
}

template <typename TPolicy>
struct CudaWorkerBinding {
  template <typename TValue, typename TSpan>
  using Worker = qleaf::CudaWorker<TPolicy, TValue, TSpan>;
};

template <typename TPolicy, template <typename> typename TNodeBuffer>
using CudaInferrer =
    qleaf::Inferrer<float, CudaWorkerBinding<TPolicy>::template Worker,
                    TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

template <typename TPolicy>
std::string cuda_name(std::string_view buffer) {
  std::ostringstream os;
  os << "qleaf/gpu/" << (TPolicy::Persistent ? "persistent" : "oneshot") << "/"
     << buffer << "/bt:" << TPolicy::BT << "/transposed:" << TPolicy::Transposed
     << "/bitmask:" << TPolicy::Bitmask << "/cached:" << TPolicy::Cached
     << "/mapped:" << TPolicy::Mapped;
  if constexpr (TPolicy::Cached)
    os << "/cachenodemajor:" << TPolicy::CacheNodeMajor;
  if constexpr (TPolicy::CacheAllWaves) os << "/cacheallwaves:1";
  if constexpr (TPolicy::Persistent) {
    os << "/crossreduce:" << TPolicy::CrossReduce;
    if constexpr (TPolicy::SinglePoller)
      os << "/sp:1/stage:" << TPolicy::StageFeatures
         << "/seqflag:" << TPolicy::SeqFlag;
    else if constexpr (!TPolicy::SeqFlag)
      os << "/sp:0/stage:" << TPolicy::StageFeatures
         << "/seqflag:" << TPolicy::SeqFlag;
  }
  if constexpr (TPolicy::Bitmask)
    os << "/tpt:" << TPolicy::TPT << "/nodestride:" << TPolicy::NodeStride;
  return os.str();
}

inline bool want(const GpuLatencyRequest &req, std::string_view name) {
  return req.filter.empty() ||
         std::string{name}.find(req.filter) != std::string::npos;
}

inline GpuLatencyRun skip_run(std::string name, std::string reason) {
  GpuLatencyRun run;
  run.name = std::move(name);
  run.status = "skipped";
  run.reason = std::move(reason);
  return run;
}

template <typename TPolicy, int Depth>
bool cuda_persistent_policy_supported_for_depth(size_t smem, int grid,
                                                std::string &reason) {
  auto *kernel = tree_kernels::traverse_persistent<
      TPolicy::BT, Depth, TPolicy::Transposed, TPolicy::Bitmask,
      TPolicy::Cached, TPolicy::CacheAllWaves, TPolicy::CrossReduce,
      TPolicy::SinglePoller, TPolicy::StageFeatures, TPolicy::SeqFlag,
      TPolicy::TPT, TPolicy::CacheNodeMajor, TPolicy::NodeStride>;
  if (smem > 48u * 1024u &&
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)) != cudaSuccess) {
    reason = "dynamic shared memory opt-in failed";
    cudaGetLastError();
    return false;
  }
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_sm, kernel, TPolicy::BT, smem) != cudaSuccess) {
    reason = "occupancy query failed";
    cudaGetLastError();
    return false;
  }
  int nsm = 0;
  if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) !=
      cudaSuccess) {
    reason = "SM count query failed";
    cudaGetLastError();
    return false;
  }
  if (per_sm <= 0 || grid > per_sm * nsm) {
    reason = "persistent grid not co-resident";
    return false;
  }
  return true;
}

template <typename TPolicy>
bool cuda_policy_supported(const GpuInput &input, int persistent_grid,
                           std::string &reason) {
  if (!input.cuda_enabled) {
    reason = "CUDA unavailable";
    return false;
  }
  if constexpr (TPolicy::Mapped) {
    if (input.mapped_features.empty()) {
      reason = "mapped host features unavailable";
      return false;
    }
  }
  const size_t depth = input.forest.at("depth").get<size_t>();
  if (depth != 4 && depth != 6) {
    reason = "CUDA kernels are specialized for depth 4 and depth 6";
    return false;
  }
  if constexpr (TPolicy::Bitmask) {
    if (depth == 0 || depth > tree_kernels::QS_MAX_DEPTH) {
      reason = "bitmask requires depth in [1, " +
               std::to_string(tree_kernels::QS_MAX_DEPTH) + "]";
      return false;
    }
  }
  const size_t tree_size = (size_t{1} << (depth + 1)) - 1;
  const size_t n_trees = input.forest.at("trees").size();
  const size_t n_features = input.features.front().size();
  constexpr int trees_per_block = TPolicy::BT / TPolicy::TPT;
  auto ceil_div = [](size_t n, size_t d) { return (n + d - 1) / d; };
  const int full_grid = static_cast<int>(ceil_div(n_trees, trees_per_block));
  const int grid =
      TPolicy::Persistent && persistent_grid > 0 ? persistent_grid : full_grid;
  if (grid <= 0 || grid > full_grid) {
    reason = "persistent grid outside covered tree grid";
    return false;
  }
  const int max_waves_per_block =
      static_cast<int>(ceil_div((size_t)full_grid, (size_t)grid));
  if constexpr (TPolicy::Cached && !TPolicy::CacheAllWaves) {
    if (grid != full_grid) {
      reason = "fixed cache does not cover grid-stride waves";
      return false;
    }
  }
  const size_t cached_trees_per_block =
      TPolicy::Cached ? (TPolicy::CacheAllWaves
                             ? (size_t)max_waves_per_block * trees_per_block
                             : (size_t)trees_per_block)
                      : 0;
  const size_t cache_tree_stride =
      TPolicy::CacheNodeMajor ? tree_kernels::node_major_cache_tree_stride(
                                    cached_trees_per_block, TPolicy::TPT)
                              : cached_trees_per_block;
  const size_t smem = cache_tree_stride * tree_size *
                          (sizeof(float) + sizeof(tree_kernels::fi)) +
                      n_features * sizeof(float);
  int maxopt = 0;
  if (cudaDeviceGetAttribute(&maxopt, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             0) != cudaSuccess) {
    reason = "cudaDeviceGetAttribute failed";
    cudaGetLastError();
    return false;
  }
  if (smem > (size_t)maxopt) {
    reason = "dynamic shared memory too large";
    return false;
  }
  if constexpr (TPolicy::Persistent) {
    if (depth == 4)
      return cuda_persistent_policy_supported_for_depth<TPolicy, 4>(smem, grid,
                                                                    reason);
    return cuda_persistent_policy_supported_for_depth<TPolicy, 6>(smem, grid,
                                                                  reason);
  }
  return true;
}

template <typename TPolicy, template <typename> typename TNodeBuffer>
void run_gpu_policy(const GpuLatencyRequest &req, GpuInput &input,
                    std::vector<GpuLatencyRun> &out) {
  const std::string name = cuda_name<TPolicy>(req.buffer);
  if (!want(req, name)) return;
  std::string reason;
  if (!cuda_policy_supported<TPolicy>(input, req.persistent_grid, reason)) {
    out.push_back(skip_run(name, reason));
    return;
  }
  auto config_json =
      config_for_gpu(input.forest, req.n_features, req.persistent_grid);
  qleaf::Config config{config_json};
  CudaInferrer<TPolicy, TNodeBuffer> inferrer{config};
  GpuLatencyRun run;
  run.name = name;
  run.extra = {{"engine", "qleaf_gpu"}, {"bt", TPolicy::BT}};
  size_t row = 0;
  for (size_t i = 0; i < req.warmup; ++i) {
    const size_t r = row++ % input.features.size();
    if constexpr (TPolicy::Mapped) {
      if constexpr (TPolicy::Persistent) {
        input.mapped_features[0].load(input.features[r]);
        run.last_result = inferrer.predict(input.mapped_features[0]);
      } else {
        run.last_result = inferrer.predict(input.mapped_features[r]);
      }
    } else {
      run.last_result = inferrer.predict(input.features[r]);
    }
  }
  run.latency_ns.reserve(req.iters);
#ifdef QLEAF_LATENCY_DECOMP
  run.phases["input_load_ns"].reserve(req.iters);
  run.phases["predict_ns"].reserve(req.iters);
#endif
  for (size_t i = 0; i < req.iters; ++i) {
    const size_t r = row++ % input.features.size();
    const int64_t t0 = now_ns();
#ifdef QLEAF_LATENCY_DECOMP
    int64_t load_ns = 0;
    int64_t predict_ns = 0;
#endif
    if constexpr (TPolicy::Mapped) {
      if constexpr (TPolicy::Persistent) {
#ifdef QLEAF_LATENCY_DECOMP
        const int64_t l0 = now_ns();
#endif
        input.mapped_features[0].load(input.features[r]);
#ifdef QLEAF_LATENCY_DECOMP
        const int64_t l1 = now_ns();
        load_ns = l1 - l0;
#endif
        const int64_t p0 = now_ns();
        run.last_result = inferrer.predict(input.mapped_features[0]);
#ifdef QLEAF_LATENCY_DECOMP
        const int64_t p1 = now_ns();
        predict_ns = p1 - p0;
#endif
      } else {
        const int64_t p0 = now_ns();
        run.last_result = inferrer.predict(input.mapped_features[r]);
#ifdef QLEAF_LATENCY_DECOMP
        const int64_t p1 = now_ns();
        predict_ns = p1 - p0;
#endif
      }
    } else {
      const int64_t p0 = now_ns();
      run.last_result = inferrer.predict(input.features[r]);
#ifdef QLEAF_LATENCY_DECOMP
      const int64_t p1 = now_ns();
      predict_ns = p1 - p0;
#endif
    }
    const int64_t t1 = now_ns();
    run.latency_ns.push_back(t1 - t0);
#ifdef QLEAF_LATENCY_DECOMP
    run.phases["input_load_ns"].push_back(load_ns);
    run.phases["predict_ns"].push_back(predict_ns);
#endif
  }
  out.push_back(std::move(run));
}

template <int BT, bool Mapped, bool Stage, bool Seq,
          template <typename> typename TNodeBuffer>
void run_sp_path(const GpuLatencyRequest &req, GpuInput &input,
                 std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraversePersistentSP<BT, true, false, false, Mapped,
                                            false, true, Stage, Seq>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, bool Mapped, bool Stage, bool Seq, int TPT,
          template <typename> typename TNodeBuffer>
void run_sp_bitmask_path(const GpuLatencyRequest &req, GpuInput &input,
                         std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraversePersistentSP<BT, true, true, false, Mapped,
                                            false, true, Stage, Seq, TPT>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, bool Bitmask, int TPT,
          template <typename> typename TNodeBuffer>
void run_mp_sentinel_path(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraversePersistentSP<BT, true, Bitmask, false, false,
                                            false, false, false, false, TPT>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, template <typename> typename TNodeBuffer>
void run_mp_sentinel_tpt(const GpuLatencyRequest &req, GpuInput &input,
                         std::vector<GpuLatencyRun> &out, int tpt) {
  switch (tpt) {
    case 1:
      run_mp_sentinel_path<BT, true, 1, TNodeBuffer>(req, input, out);
      break;
    case 2:
      run_mp_sentinel_path<BT, true, 2, TNodeBuffer>(req, input, out);
      break;
    case 4:
      run_mp_sentinel_path<BT, true, 4, TNodeBuffer>(req, input, out);
      break;
    case 8:
      run_mp_sentinel_path<BT, true, 8, TNodeBuffer>(req, input, out);
      break;
    case 16:
      run_mp_sentinel_path<BT, true, 16, TNodeBuffer>(req, input, out);
      break;
    case 32:
      run_mp_sentinel_path<BT, true, 32, TNodeBuffer>(req, input, out);
      break;
  }
}

template <int BT, bool CacheAllWaves, template <typename> typename TNodeBuffer>
void run_sp_cached_path(const GpuLatencyRequest &req, GpuInput &input,
                        std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraversePersistentSP<BT, !CacheAllWaves, false, true,
                                            true, false, true, true, false, 1,
                                            CacheAllWaves>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, int TPT, bool Seq, bool Transposed, int NodeStride,
          template <typename> typename TNodeBuffer>
void run_bitmask_tpt(const GpuLatencyRequest &req, GpuInput &input,
                     std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraversePersistentSP<BT, Transposed, true, false, true,
                                            false, true, true, Seq, TPT, false,
                                            false, NodeStride>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, int TPT, bool Seq, bool Transposed,
          template <typename> typename TNodeBuffer>
void run_bitmask_tpt_layout(const GpuLatencyRequest &req, GpuInput &input,
                            std::vector<GpuLatencyRun> &out) {
  run_bitmask_tpt<BT, TPT, Seq, Transposed, 1, TNodeBuffer>(req, input, out);
  if constexpr (!Transposed && TPT > 1) {
    run_bitmask_tpt<BT, TPT, Seq, Transposed, 8, TNodeBuffer>(req, input, out);
    run_bitmask_tpt<BT, TPT, Seq, Transposed, 16, TNodeBuffer>(req, input, out);
  }
}

template <int BT, int TPT, bool Transposed, bool CacheAllWaves,
          bool CacheNodeMajor, int NodeStride,
          template <typename> typename TNodeBuffer>
void run_bitmask_cached_policy(const GpuLatencyRequest &req, GpuInput &input,
                               std::vector<GpuLatencyRun> &out) {
  using P =
      qleaf::CudaTraversePersistentSP<BT, Transposed, true, true, true, false,
                                      true, true, false, TPT, CacheAllWaves,
                                      CacheNodeMajor, NodeStride>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, int TPT, template <typename> typename TNodeBuffer>
void run_bitmask_cached_tpt(const GpuLatencyRequest &req, GpuInput &input,
                            std::vector<GpuLatencyRun> &out) {
  run_bitmask_cached_policy<BT, TPT, false, false, false, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, false, false, true, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, true, false, false, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, true, false, true, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, false, true, false, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, false, true, true, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, true, true, false, 1, TNodeBuffer>(
      req, input, out);
  run_bitmask_cached_policy<BT, TPT, true, true, true, 1, TNodeBuffer>(
      req, input, out);
  if constexpr (TPT > 1) {
    run_bitmask_cached_policy<BT, TPT, false, false, false, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, true, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, false, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, true, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, false, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, true, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, false, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, true, 2, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, false, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, true, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, false, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, true, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, false, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, true, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, false, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, true, 4, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, false, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, true, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, false, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, true, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, false, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, true, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, false, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, true, 8, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, false, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, false, true, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, false, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, false, true, true, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, false, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, false, true, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, false, 16, TNodeBuffer>(
        req, input, out);
    run_bitmask_cached_policy<BT, TPT, true, true, true, 16, TNodeBuffer>(
        req, input, out);
  }
}

template <int BT, template <typename> typename TNodeBuffer>
void run_tpt_value(const GpuLatencyRequest &req, GpuInput &input,
                   std::vector<GpuLatencyRun> &out, int tpt) {
  switch (tpt) {
    case 1:
      run_bitmask_tpt_layout<BT, 1, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 1, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 1, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 1, false, false, TNodeBuffer>(req, input, out);
      break;
    case 2:
      run_bitmask_tpt_layout<BT, 2, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 2, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 2, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 2, false, false, TNodeBuffer>(req, input, out);
      break;
    case 4:
      run_bitmask_tpt_layout<BT, 4, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 4, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 4, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 4, false, false, TNodeBuffer>(req, input, out);
      break;
    case 8:
      run_bitmask_tpt_layout<BT, 8, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 8, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 8, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 8, false, false, TNodeBuffer>(req, input, out);
      break;
    case 16:
      run_bitmask_tpt_layout<BT, 16, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 16, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 16, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 16, false, false, TNodeBuffer>(req, input,
                                                                out);
      break;
    case 32:
      run_bitmask_tpt_layout<BT, 32, true, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 32, false, true, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 32, true, false, TNodeBuffer>(req, input, out);
      run_bitmask_tpt_layout<BT, 32, false, false, TNodeBuffer>(req, input,
                                                                out);
      break;
  }
}

template <int BT, template <typename> typename TNodeBuffer>
void run_cached_tpt_value(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out, int tpt) {
  switch (tpt) {
    case 1:
      run_bitmask_cached_tpt<BT, 1, TNodeBuffer>(req, input, out);
      break;
    case 2:
      run_bitmask_cached_tpt<BT, 2, TNodeBuffer>(req, input, out);
      break;
    case 4:
      run_bitmask_cached_tpt<BT, 4, TNodeBuffer>(req, input, out);
      break;
    case 8:
      run_bitmask_cached_tpt<BT, 8, TNodeBuffer>(req, input, out);
      break;
    case 16:
      run_bitmask_cached_tpt<BT, 16, TNodeBuffer>(req, input, out);
      break;
    case 32:
      run_bitmask_cached_tpt<BT, 32, TNodeBuffer>(req, input, out);
      break;
  }
}

template <int BT, int TPT, bool Transposed, bool Cached, int NodeStride,
          template <typename> typename TNodeBuffer>
void run_oneshot_bitmask(const GpuLatencyRequest &req, GpuInput &input,
                         std::vector<GpuLatencyRun> &out) {
  using P = qleaf::CudaTraverseOneShot<BT, Transposed, true, Cached, false, TPT,
                                       false, NodeStride>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, bool Bitmask, template <typename> typename TNodeBuffer>
void run_naive_persistent(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  using P =
      qleaf::CudaTraversePersistent<BT, true, Bitmask, false, false, true>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu
