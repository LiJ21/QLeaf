#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "config.h"
#include "gworkers.cuh"

namespace {

enum class BufferKind { Compact, Tree };

struct Args {
  std::string trees_path;
  std::string features_csv;
  std::vector<size_t> threads{1};
  std::vector<BufferKind> buffers{BufferKind::Compact, BufferKind::Tree};
  size_t features = 0;
  size_t samples = 1024;
  uint32_t seed = 12345;
  std::optional<int64_t> iterations;
  std::optional<size_t> ntrees;
  std::vector<int> pin_cores;
  int persistent_grid = 0;  // >0 forces the persistent grid (e.g. 1); 0=auto
};

void cuda_throw(cudaError_t e, std::string_view op) {
  if (e == cudaSuccess) return;
  throw std::runtime_error(std::string{op} + " failed: " +
                           cudaGetErrorString(e));
}

class MappedFeatureRow {
 public:
  MappedFeatureRow() = default;

  explicit MappedFeatureRow(const std::vector<float> &values)
      : size_(values.size()) {
    if (size_ == 0) return;
    cuda_throw(cudaHostAlloc(&data_, size_ * sizeof(float),
                             cudaHostAllocMapped),
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
  // Overwrite the mapped buffer in place (same address). Lets the persistent
  // kernel -- bound to a fixed feature pointer -- process a rotating input
  // stream like the CPU/oneshot paths. const: the object identity
  // (data_/size_) is unchanged; only the pointed-to mapped memory is rewritten.
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

[[noreturn]] void usage(std::string_view message = {}) {
  if (!message.empty()) std::cerr << "error: " << message << "\n\n";
  std::cerr
      << "usage: worker_latency_bench --trees forest.json --features N "
         "[options] [-- <google benchmark options>]\n\n"
      << "options:\n"
      << "  --threads 1,2,4       thread counts to benchmark (default: 1)\n"
      << "  --buffers compact,tree  node buffers to benchmark (default: "
         "compact,tree)\n"
      << "  --features-csv file   read feature rows from CSV\n"
      << "  --samples N           generated/loaded feature rows (default: "
         "1024)\n"
      << "  --seed N              random seed (default: 12345)\n"
      << "  --ntrees N            use only the first N trees of the forest "
         "(prefix)\n"
      << "  --iters N             force exactly N measured iterations\n"
      << "  --pin-cores C0,C1,... pin dispatcher to C0 and workers to C1..CN\n"
      << "  --json                alias for --benchmark_format=json\n\n"
      << "Google Benchmark owns timing/output flags such as "
         "--benchmark_min_time, --benchmark_repetitions, --benchmark_filter, "
         "and --benchmark_out_format.\n";
  std::exit(message.empty() ? 0 : 1);
}

size_t parse_size(std::string_view text, std::string_view name) {
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument("invalid integer for " + std::string{name});
  }

  size_t pos = 0;
  std::string value{text};
  auto ret = std::stoull(value, &pos);
  if (pos != value.size()) {
    throw std::invalid_argument("invalid integer for " + std::string{name});
  }
  return ret;
}

int64_t parse_i64(std::string_view text, std::string_view name) {
  size_t pos = 0;
  std::string value{text};
  auto ret = std::stoll(value, &pos);
  if (pos != value.size()) {
    throw std::invalid_argument("invalid integer for " + std::string{name});
  }
  return ret;
}

std::vector<size_t> parse_threads(std::string_view text) {
  std::vector<size_t> ret;
  std::stringstream ss{std::string{text}};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty thread count");
    auto n = parse_size(item, "--threads");
    if (n == 0) throw std::invalid_argument("--threads values must be > 0");
    ret.push_back(n);
  }
  if (ret.empty()) throw std::invalid_argument("--threads must not be empty");
  return ret;
}

std::vector<int> parse_pin_cores(std::string_view text) {
  std::vector<int> ret;
  std::stringstream ss{std::string{text}};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty core id");
    auto core = parse_i64(item, "--pin-cores");
    if (core < 0 || core > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("--pin-cores values must be non-negative int");
    }
    ret.push_back(static_cast<int>(core));
  }
  if (ret.empty()) throw std::invalid_argument("--pin-cores must not be empty");
  return ret;
}

std::vector<BufferKind> parse_buffers(std::string_view text) {
  std::vector<BufferKind> ret;
  std::stringstream ss{std::string{text}};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item == "compact") {
      ret.push_back(BufferKind::Compact);
    } else if (item == "tree") {
      ret.push_back(BufferKind::Tree);
    } else {
      throw std::invalid_argument("unknown buffer '" + item +
                                  "' (expected compact or tree)");
    }
  }
  if (ret.empty()) throw std::invalid_argument("--buffers must not be empty");
  return ret;
}

Args parse_args(int argc, char **argv,
                std::vector<std::string> &benchmark_args_storage) {
  Args args;
  benchmark_args_storage.push_back(argv[0]);

  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    auto need_value = [&](std::string_view name) -> std::string_view {
      if (i + 1 >= argc) usage(std::string{name} + " requires a value");
      return argv[++i];
    };

    if (arg == "--") {
      while (++i < argc) benchmark_args_storage.emplace_back(argv[i]);
      break;
    }

    if (arg == "--help" || arg == "-h") {
      usage();
    } else if (arg == "--trees") {
      args.trees_path = need_value(arg);
    } else if (arg == "--threads") {
      args.threads = parse_threads(need_value(arg));
    } else if (arg == "--buffers") {
      args.buffers = parse_buffers(need_value(arg));
    } else if (arg == "--features") {
      args.features = parse_size(need_value(arg), arg);
    } else if (arg == "--features-csv") {
      args.features_csv = need_value(arg);
    } else if (arg == "--samples") {
      args.samples = parse_size(need_value(arg), arg);
    } else if (arg == "--seed") {
      args.seed = static_cast<uint32_t>(parse_size(need_value(arg), arg));
    } else if (arg == "--ntrees") {
      args.ntrees = parse_size(need_value(arg), arg);
    } else if (arg == "--iters") {
      args.iterations = parse_i64(need_value(arg), arg);
    } else if (arg == "--pin-cores") {
      args.pin_cores = parse_pin_cores(need_value(arg));
    } else if (arg == "--pin") {
      usage("--pin was removed; use --pin-cores dispatch,worker0,...");
    } else if (arg == "--persistent-grid") {
      args.persistent_grid = static_cast<int>(parse_size(need_value(arg), arg));
    } else if (arg == "--json") {
      benchmark_args_storage.emplace_back("--benchmark_format=json");
    } else {
      benchmark_args_storage.emplace_back(arg);
    }
  }

  if (args.trees_path.empty()) usage("--trees is required");
  if (args.features == 0) usage("--features must be > 0");
  if (args.samples == 0) usage("--samples must be > 0");
  if (args.iterations && *args.iterations <= 0) usage("--iters must be > 0");
  if (args.ntrees && *args.ntrees == 0) usage("--ntrees must be > 0");
  if (!args.pin_cores.empty()) {
    auto max_threads =
        *std::max_element(args.threads.begin(), args.threads.end());
    if (args.pin_cores.size() < max_threads + 1) {
      usage("--pin-cores needs dispatcher plus one worker core per worker in "
            "the largest --threads value");
    }
    unsigned hw = std::thread::hardware_concurrency();
    if (hw != 0) {
      for (int core : args.pin_cores) {
        if (static_cast<unsigned>(core) >= hw) {
          usage("--pin-cores contains core " + std::to_string(core) +
                ", but this process sees only " + std::to_string(hw) +
                " logical CPUs");
        }
      }
    }
  }
  return args;
}

nlohmann::json load_json(const std::string &path) {
  std::ifstream in{path};
  if (!in) throw std::runtime_error("failed to open JSON file: " + path);
  nlohmann::json ret;
  in >> ret;
  return ret;
}

std::vector<float> parse_csv_row(const std::string &line, size_t features) {
  std::vector<float> row;
  row.reserve(features);
  std::stringstream ss{line};
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    if (cell.empty()) continue;
    if (row.size() < features) row.push_back(std::stof(cell));
  }
  if (!row.empty() && row.size() < features) {
    throw std::runtime_error("CSV row has fewer columns than --features");
  }
  return row;
}

std::vector<std::vector<float>> load_features_csv(const std::string &path,
                                                  size_t features,
                                                  size_t samples) {
  std::ifstream in{path};
  if (!in) throw std::runtime_error("failed to open CSV file: " + path);

  std::vector<std::vector<float>> ret;
  std::string line;
  while (ret.size() < samples && std::getline(in, line)) {
    if (line.empty()) continue;
    auto row = parse_csv_row(line, features);
    if (!row.empty()) ret.push_back(std::move(row));
  }
  if (ret.empty()) throw std::runtime_error("CSV file has no feature rows");
  return ret;
}

std::vector<std::vector<float>> generate_features(size_t features,
                                                  size_t samples,
                                                  uint32_t seed) {
  std::mt19937 gen{seed};
  std::uniform_real_distribution<float> dist{0.0f, 1.0f};
  std::vector<std::vector<float>> ret(samples, std::vector<float>(features));
  for (auto &row : ret) {
    for (auto &v : row) v = dist(gen);
  }
  return ret;
}

size_t max_feature_index(const nlohmann::json &forest) {
  auto depth = forest.at("depth").get<size_t>();
  auto internal_nodes = (size_t{1} << depth) - 1;
  size_t max_idx = 0;
  for (const auto &tree : forest.at("trees")) {
    const auto &indices = tree.at("indices");
    if (indices.size() < internal_nodes) {
      throw std::runtime_error("tree has fewer indices than internal nodes");
    }
    for (size_t i = 0; i < internal_nodes; ++i) {
      max_idx = std::max(max_idx, indices.at(i).get<size_t>());
    }
  }
  return max_idx;
}

// Keep only the first `ntrees` trees. Gradient boosting is additive and
// sequential (tree i fits the residual of trees 0..i-1), so a prefix of an
// N-tree forest is exactly a K-tree model -- one big forest can stand in for
// every smaller size. The base margin lives in tree 0's leaves, so the prefix
// keeps it. Sizing the node buffer off this truncated count is what makes the
// benchmark's cache footprint match a real K-tree forest.
void truncate_forest(nlohmann::json &forest, size_t ntrees) {
  auto &trees = forest.at("trees");
  if (ntrees > trees.size()) {
    throw std::runtime_error("--ntrees " + std::to_string(ntrees) +
                             " exceeds forest size " +
                             std::to_string(trees.size()));
  }
  trees.erase(std::next(trees.begin(), static_cast<std::ptrdiff_t>(ntrees)),
              trees.end());
}

void validate_forest(const nlohmann::json &forest, size_t features) {
  auto depth = forest.at("depth").get<size_t>();
  if (depth == 0 || depth > qleaf::kDefaultMaxDepth) {
    throw std::runtime_error("depth must be in [1, " +
                             std::to_string(qleaf::kDefaultMaxDepth) + "]");
  }

  auto tree_size = (size_t{1} << (depth + 1)) - 1;
  const auto &trees = forest.at("trees");
  if (trees.empty()) throw std::runtime_error("forest has no trees");

  for (const auto &tree : trees) {
    if (tree.at("indices").size() != tree_size ||
        tree.at("splits").size() != tree_size) {
      throw std::runtime_error(
          "each tree must have exactly 2^(depth + 1) - 1 indices and splits");
    }
  }

  auto max_idx = max_feature_index(forest);
  if (max_idx >= features) {
    throw std::runtime_error("tree references feature index " +
                             std::to_string(max_idx) + ", but --features is " +
                             std::to_string(features));
  }
}

nlohmann::json config_for_threads(const nlohmann::json &forest, size_t threads,
                                  const std::vector<int> &pin_cores,
                                  size_t features) {
  nlohmann::json config = forest;
  config["worker"] = nlohmann::json::array();
  for (size_t i = 0; i < threads; ++i) {
    nlohmann::json worker{{"has_equal", false}, {"num_feature", features}};
    if (!pin_cores.empty()) worker["core"] = pin_cores.at(i + 1);
    config["worker"].push_back(worker);
  }
  return config;
}

template <template <typename, typename> typename TWorker>
struct ThreadedWorkerBinding {
  template <typename TValue, typename TSpan>
  using Worker = qleaf::ThreadedWorker<TWorker<TValue, TSpan>>;
};

template <template <typename, typename> typename TWorker,
          template <typename> typename TNodeBuffer>
using DirectBenchmarkInferrer =
    qleaf::Inferrer<float, TWorker, TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

template <template <typename, typename> typename TWorker,
          template <typename> typename TNodeBuffer>
using ThreadedBenchmarkInferrer =
    qleaf::Inferrer<float, ThreadedWorkerBinding<TWorker>::template Worker,
                    TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

struct BenchmarkInput {
  nlohmann::json forest;
  std::vector<std::vector<float>> features;
  std::vector<MappedFeatureRow> mapped_features;
  std::optional<int64_t> iterations;
  std::vector<int> pin_cores;
  bool cuda_enabled = false;
  int persistent_grid = 0;  // >0 forces the persistent grid via config key
};

bool cuda_runtime_available() {
  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) {
    std::cerr << "warning: CUDA benchmarks disabled: " << cudaGetErrorString(e)
              << "\n";
    cudaGetLastError();
    return false;
  }
  if (count == 0) {
    std::cerr << "warning: CUDA benchmarks disabled: no CUDA devices found\n";
    return false;
  }

  // How cudaStreamSynchronize waits for the GPU. Spin busy-waits for minimal
  // wakeup latency; auto/yield/blocking free the CPU at a latency cost. Default
  // auto (CUDA's default -- it already spins when CPUs >> contexts, so spin was
  // measured to be a no-op here); override with
  // QLEAF_SCHED={spin,auto,yield,blocking} to A/B the policy.
  unsigned sched = cudaDeviceScheduleAuto;
  if (const char *s = std::getenv("QLEAF_SCHED")) {
    std::string_view v{s};
    if (v == "spin") sched = cudaDeviceScheduleSpin;
    else if (v == "auto") sched = cudaDeviceScheduleAuto;
    else if (v == "yield") sched = cudaDeviceScheduleYield;
    else if (v == "blocking") sched = cudaDeviceScheduleBlockingSync;
    else usage(std::string{"unknown QLEAF_SCHED '"} + s +
               "' (expected spin|auto|yield|blocking)");
  }
  e = cudaSetDeviceFlags(cudaDeviceMapHost | sched);
  if (e != cudaSuccess && e != cudaErrorSetOnActiveProcess) {
    std::cerr << "warning: CUDA benchmarks disabled: cudaSetDeviceFlags("
                 "cudaDeviceMapHost | sched) failed: "
              << cudaGetErrorString(e) << "\n";
    cudaGetLastError();
    return false;
  }
  if (e == cudaErrorSetOnActiveProcess) cudaGetLastError();

  int can_map = 0;
  e = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0);
  if (e != cudaSuccess || !can_map) {
    std::cerr << "warning: CUDA benchmarks disabled: device cannot map host "
                 "memory\n";
    if (e != cudaSuccess) cudaGetLastError();
    return false;
  }
  return true;
}

bool prepare_cuda_features(BenchmarkInput &input) {
  if (!cuda_runtime_available()) return false;
  try {
    input.mapped_features.reserve(input.features.size());
    for (const auto &row : input.features) {
      input.mapped_features.emplace_back(row);
    }
    return true;
  } catch (const std::exception &e) {
    std::cerr << "warning: CUDA benchmarks disabled: " << e.what() << "\n";
    input.mapped_features.clear();
    return false;
  }
}

template <template <template <typename, typename> typename,
                    template <typename> typename> typename TInferrer,
          template <typename> typename TNodeBuffer>
void check_match(const BenchmarkInput &input, size_t threads) {
  auto config_json =
      config_for_threads(input.forest, threads, input.pin_cores,
                         input.features[0].size());
  qleaf::Config config{config_json};

  TInferrer<qleaf::BranchRegressionWorker, TNodeBuffer> branch{config};
  TInferrer<qleaf::BitmaskRegressionWorker, TNodeBuffer> bitmask{config};

  auto branch_result = branch.predict(input.features.front());
  auto bitmask_result = bitmask.predict(input.features.front());
  auto scale = std::max(1.0f, std::abs(branch_result));
  if (std::abs(branch_result - bitmask_result) > 1e-4f * scale) {
    std::ostringstream os;
    os << "branch and bitmask disagree before benchmarking: branch="
       << branch_result << " bitmask=" << bitmask_result;
    throw std::runtime_error(os.str());
  }
}

std::string name_for(std::string_view worker, std::string_view buffer,
                     size_t threads, size_t features) {
  std::ostringstream os;
  os << worker << "/" << buffer << "/threads:" << threads
     << "/features:" << features;
  return os.str();
}

std::string nothread_name(std::string_view worker, std::string_view buffer,
                          size_t features) {
  std::ostringstream os;
  os << worker << "/" << buffer << "/nothread/features:" << features;
  return os.str();
}

template <typename TInferrer>
void run_worker(benchmark::State &state, const BenchmarkInput *input,
                size_t threads) {
  auto config_json = config_for_threads(input->forest, threads, input->pin_cores,
                                        input->features[0].size());
  qleaf::Config config{config_json};
  TInferrer inferrer{config};

  size_t row = 0;
  for (auto _ : state) {
    const auto &fts = input->features[row++ % input->features.size()];
    auto result = inferrer.predict(fts);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations());
}

template <typename TInferrer>
void register_one(std::string name, const BenchmarkInput *input,
                  size_t threads) {
  auto *bench = benchmark::RegisterBenchmark(
                    name.c_str(), run_worker<TInferrer>, input, threads)
                    ->Unit(benchmark::kNanosecond)
                    ->UseRealTime();
  if (input->iterations) bench->Iterations(*input->iterations);
}

#ifndef QLEAF_BT
#define QLEAF_BT 256
#endif
constexpr int kCudaBT = QLEAF_BT;  // override at configure: -DCMAKE_CUDA_FLAGS=-DQLEAF_BT=1024

#ifndef QLEAF_WORKER_CUDA_ONESHOT_COMBOS
#define QLEAF_WORKER_CUDA_ONESHOT_COMBOS 1
#endif
#ifndef QLEAF_WORKER_CUDA_PERSISTENT_COMBOS
#define QLEAF_WORKER_CUDA_PERSISTENT_COMBOS 1
#endif
#ifndef QLEAF_WORKER_CUDA_SP
#define QLEAF_WORKER_CUDA_SP 1
#endif
#ifndef QLEAF_WORKER_CUDA_SP_CACHE
#define QLEAF_WORKER_CUDA_SP_CACHE 1
#endif
#ifndef QLEAF_WORKER_CUDA_BITMASK_TPT
#define QLEAF_WORKER_CUDA_BITMASK_TPT 1
#endif
#ifndef QLEAF_WORKER_CUDA_BITMASK_ONESHOT
#define QLEAF_WORKER_CUDA_BITMASK_ONESHOT 1
#endif

template <typename TPolicy>
struct CudaWorkerBinding {
  template <typename TValue, typename TSpan>
  using Worker = qleaf::CudaWorker<TPolicy, TValue, TSpan>;
};

template <typename TPolicy, template <typename> typename TNodeBuffer>
using CudaBenchmarkInferrer =
    qleaf::Inferrer<float, CudaWorkerBinding<TPolicy>::template Worker,
                    TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

template <typename TPolicy>
std::string cuda_name(std::string_view buffer, size_t features) {
  std::ostringstream os;
  os << "cuda/" << (TPolicy::Persistent ? "persistent" : "oneshot") << "/"
     << buffer << "/bt:" << TPolicy::BT
     << "/transposed:" << TPolicy::Transposed
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
  }
  if constexpr (TPolicy::Bitmask)
    os << "/tpt:" << TPolicy::TPT << "/nodestride:" << TPolicy::NodeStride;
  os << "/features:" << features;
  return os.str();
}

template <typename TPolicy, int Depth>
bool cuda_policy_kernel_supported_for_depth(size_t smem, int grid) {
  if constexpr (TPolicy::Persistent) {
    auto *kernel = tree_kernels::traverse_persistent<
        TPolicy::BT, Depth, TPolicy::Transposed, TPolicy::Bitmask,
        TPolicy::Cached, TPolicy::CacheAllWaves, TPolicy::CrossReduce,
        TPolicy::SinglePoller, TPolicy::StageFeatures, TPolicy::SeqFlag,
        TPolicy::TPT, TPolicy::CacheNodeMajor, TPolicy::NodeStride>;
    if (smem > 48u * 1024u &&
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, kernel, TPolicy::BT, smem) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    int nsm = 0;
    if (cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0) !=
        cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return per_sm > 0 && grid <= per_sm * nsm;
  } else {
    auto *kernel = tree_kernels::traverse_oneshot<
        TPolicy::BT, Depth, TPolicy::Transposed, TPolicy::Bitmask,
        TPolicy::Cached, TPolicy::TPT, TPolicy::CacheNodeMajor,
        TPolicy::NodeStride>;
    if (smem > 48u * 1024u &&
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    return true;
  }
}

template <typename TPolicy>
bool cuda_policy_supported(const BenchmarkInput &input) {
  if (!input.cuda_enabled) return false;
  if constexpr (TPolicy::Mapped) {
    if (input.mapped_features.empty()) return false;
  }

  const size_t depth = input.forest.at("depth").get<size_t>();
  if (depth != 4 && depth != 6) return false;
  if constexpr (TPolicy::Bitmask) {
    if (depth == 0 || depth > tree_kernels::QS_MAX_DEPTH) return false;
  }

  const size_t tree_size = (size_t{1} << (depth + 1)) - 1;
  const size_t n_trees = input.forest.at("trees").size();
  const size_t n_features = input.features.front().size();
  constexpr int trees_per_block = TPolicy::BT / TPolicy::TPT;
  auto ceil_div = [](size_t n, size_t d) { return (n + d - 1) / d; };
  const int full_grid = static_cast<int>(ceil_div(n_trees, trees_per_block));
  const int grid =
      TPolicy::Persistent && input.persistent_grid > 0 ? input.persistent_grid
                                                       : full_grid;
  if (grid <= 0 || grid > full_grid) return false;
  const int max_waves_per_block =
      static_cast<int>(ceil_div((size_t)full_grid, (size_t)grid));
  if constexpr (TPolicy::Cached && !TPolicy::CacheAllWaves)
    if (grid != full_grid) return false;
  const size_t cached_trees_per_block =
      TPolicy::Cached ? (TPolicy::CacheAllWaves
                             ? (size_t)max_waves_per_block * trees_per_block
                             : (size_t)trees_per_block)
                      : 0;
  const size_t cache_tree_stride =
      TPolicy::CacheNodeMajor
          ? tree_kernels::node_major_cache_tree_stride(cached_trees_per_block,
                                                       TPolicy::TPT)
          : cached_trees_per_block;
  const size_t smem = cache_tree_stride * tree_size *
                          (sizeof(float) + sizeof(tree_kernels::fi)) +
                      n_features * sizeof(float);

  int maxopt = 0;
  if (cudaDeviceGetAttribute(&maxopt, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             0) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  if (smem > (size_t)maxopt) return false;

  if (depth == 4)
    return cuda_policy_kernel_supported_for_depth<TPolicy, 4>(smem, grid);
  return cuda_policy_kernel_supported_for_depth<TPolicy, 6>(smem, grid);
}

template <typename TPolicy, template <typename> typename TNodeBuffer>
void run_cuda_worker(benchmark::State &state, const BenchmarkInput *input) {
  auto config_json =
      config_for_threads(input->forest, 1, std::vector<int>{},
                         input->features.front().size());
  if (input->persistent_grid > 0)
    config_json["worker"][0]["persistent_grid"] = input->persistent_grid;
  qleaf::Config config{config_json};
  CudaBenchmarkInferrer<TPolicy, TNodeBuffer> inferrer{config};

  size_t row = 0;
  float last_result = 0.0f;
  for (auto _ : state) {
    float result;
    if constexpr (TPolicy::Mapped) {
      if constexpr (TPolicy::Persistent) {
        // Rotate inputs like the CPU/oneshot paths: the resident kernel is
        // bound to one mapped buffer, so rewrite that buffer each request
        // instead of re-feeding the same row.
        input->mapped_features[0].load(
            input->features[row++ % input->features.size()]);
        result = inferrer.predict(input->mapped_features[0]);
      } else {
        const size_t idx = row++ % input->mapped_features.size();
        result = inferrer.predict(input->mapped_features[idx]);
      }
    } else {
      const auto &fts = input->features[row++ % input->features.size()];
      result = inferrer.predict(fts);
    }
    benchmark::DoNotOptimize(result);
    last_result = result;
  }

  // Cross-policy correctness aid: with QLEAF_PRINT_RESULT set, emit the last
  // prediction so SP configs can be diffed against the baseline.
  if (std::getenv("QLEAF_PRINT_RESULT"))
    std::fprintf(stderr, "[result] %s = %.8g\n", state.name().c_str(),
                 last_result);

  state.SetItemsProcessed(state.iterations());
}

template <typename TPolicy, template <typename> typename TNodeBuffer>
void register_cuda_one(const Args &args, const BenchmarkInput *input,
                       std::string_view buffer) {
  if (!cuda_policy_supported<TPolicy>(*input)) return;
  auto name = cuda_name<TPolicy>(buffer, args.features);
  auto *bench =
      benchmark::RegisterBenchmark(name.c_str(),
                                   run_cuda_worker<TPolicy, TNodeBuffer>, input)
          ->Unit(benchmark::kNanosecond)
          ->UseRealTime();
  if (input->iterations) bench->Iterations(*input->iterations);
}

template <int Bits, template <typename> typename TNodeBuffer>
void register_cuda_oneshot_combo(const Args &args, const BenchmarkInput *input,
                                 std::string_view buffer) {
  using Policy = qleaf::CudaTraverseOneShot<
      kCudaBT, static_cast<bool>(Bits & 1), static_cast<bool>(Bits & 2),
      static_cast<bool>(Bits & 4), static_cast<bool>(Bits & 8)>;
  register_cuda_one<Policy, TNodeBuffer>(args, input, buffer);
}

template <template <typename> typename TNodeBuffer, int... Bits>
void register_cuda_oneshot_combos(const Args &args, const BenchmarkInput *input,
                                  std::string_view buffer,
                                  std::integer_sequence<int, Bits...>) {
  (register_cuda_oneshot_combo<Bits, TNodeBuffer>(args, input, buffer), ...);
}

template <int Bits, template <typename> typename TNodeBuffer>
void register_cuda_persistent_combo(const Args &args,
                                    const BenchmarkInput *input,
                                    std::string_view buffer) {
  using Policy = qleaf::CudaTraversePersistent<
      kCudaBT, static_cast<bool>(Bits & 1), static_cast<bool>(Bits & 2),
      static_cast<bool>(Bits & 4), static_cast<bool>(Bits & 8),
      static_cast<bool>(Bits & 16)>;
  register_cuda_one<Policy, TNodeBuffer>(args, input, buffer);
}

template <template <typename> typename TNodeBuffer, int... Bits>
void register_cuda_persistent_combos(const Args &args,
                                     const BenchmarkInput *input,
                                     std::string_view buffer,
                                     std::integer_sequence<int, Bits...>) {
  (register_cuda_persistent_combo<Bits, TNodeBuffer>(args, input, buffer), ...);
}

// Run the resident kernel once and return the prediction (warmed).
template <typename TPolicy, template <typename> typename TNodeBuffer>
float cuda_predict_once(const BenchmarkInput &input) {
  auto config_json =
      config_for_threads(input.forest, 1, std::vector<int>{},
                         input.features.front().size());
  if (input.persistent_grid > 0)
    config_json["worker"][0]["persistent_grid"] = input.persistent_grid;
  qleaf::Config config{config_json};
  CudaBenchmarkInferrer<TPolicy, TNodeBuffer> inferrer{config};
  if constexpr (TPolicy::Mapped) {
    // Refresh the mapped buffer before predicting: the sentinel handshake
    // (!SeqFlag) resets it to +sentinel after consuming, so a later config in
    // the gate would otherwise read sentinels instead of real features.
    input.mapped_features[0].load(input.features[0]);
    return inferrer.predict(input.mapped_features[0]);
  }
  return inferrer.predict(input.features[0]);
}

// §8.1 acceptance gate: the single-poller configs must match the baseline
// persistent kernel within rel-tol 1e-4 (grid-stride reorders the sum, so not
// bitwise). Aborts the run on divergence, like check_match for the CPU path.
template <template <typename> typename TNodeBuffer>
void cuda_verify(const BenchmarkInput &input) {
  if (!input.cuda_enabled || input.mapped_features.empty()) return;
  using Base =
      qleaf::CudaTraversePersistent<kCudaBT, true, false, false, true, false>;
  using SP0 = qleaf::CudaTraversePersistentSP<kCudaBT, true, false, false, true,
                                              false, true, false>;
  using SP1 = qleaf::CudaTraversePersistentSP<kCudaBT, true, false, false, true,
                                              false, true, true>;
  using SP2 =  // sentinel signal (SeqFlag=false) + stage
      qleaf::CudaTraversePersistentSP<kCudaBT, true, false, false, true, false,
                                      true, true, false>;
  using CR =  // cross-reduce: device-side sum -> one done_flag
      qleaf::CudaTraversePersistent<kCudaBT, true, false, false, true, true>;
  if (!cuda_policy_supported<Base>(input) ||
      !cuda_policy_supported<SP0>(input) || !cuda_policy_supported<SP1>(input))
    return;
  const float base = cuda_predict_once<Base, TNodeBuffer>(input);
  const float sp0 = cuda_predict_once<SP0, TNodeBuffer>(input);
  const float sp1 = cuda_predict_once<SP1, TNodeBuffer>(input);
  const float sp2 = cuda_policy_supported<SP2>(input)
                        ? cuda_predict_once<SP2, TNodeBuffer>(input)
                        : base;
  const float cr =
      cuda_policy_supported<CR>(input) ? cuda_predict_once<CR, TNodeBuffer>(input)
                                       : base;
  const float tol = 1e-4f * std::max(1.0f, std::abs(base));
  // NaN/Inf must fail (abs(nan-x) > tol is false, so check finiteness too).
  if (!std::isfinite(base) || !std::isfinite(sp0) || !std::isfinite(sp1) ||
      !std::isfinite(sp2) || !std::isfinite(cr) || std::abs(sp0 - base) > tol ||
      std::abs(sp1 - base) > tol || std::abs(sp2 - base) > tol ||
      std::abs(cr - base) > tol) {
    std::ostringstream os;
    os << "persistent variant diverges from baseline: base=" << base
       << " config1=" << sp0 << " config2=" << sp1 << " config2-sentinel="
       << sp2 << " crossreduce=" << cr << " (tol=" << tol << ")";
    throw std::runtime_error(os.str());
  }
  std::fprintf(stderr,
               "[cuda-verify] base=%.8g config1=%.8g config2=%.8g "
               "config2-sentinel=%.8g crossreduce=%.8g OK\n",
               base, sp0, sp1, sp2, cr);
}

// Single-poller variants are registered as an explicit short list of the
// configs that actually appear in the experiment matrix (the measured-best
// policy: Transposed, !Bitmask, !Cached, Mapped, !CrossReduce) rather than
// widening the bit-sweep to 0..127 and rejecting most of it.
template <bool StageFeatures, bool SeqFlag, template <typename> typename TNodeBuffer>
void register_cuda_sp(const Args &args, const BenchmarkInput *input,
                      std::string_view buffer) {
  using Policy = qleaf::CudaTraversePersistentSP<
      kCudaBT, /*Transposed=*/true, /*Bitmask=*/false, /*Cached=*/false,
      /*Mapped=*/true, /*CrossReduce=*/false, /*SinglePoller=*/true,
      StageFeatures, SeqFlag>;
  register_cuda_one<Policy, TNodeBuffer>(args, input, buffer);
}

template <bool CacheAllWaves, template <typename> typename TNodeBuffer>
void register_cuda_sp_cached(const Args &args, const BenchmarkInput *input,
                             std::string_view buffer) {
  using Policy = qleaf::CudaTraversePersistentSP<
      kCudaBT, /*Transposed=*/!CacheAllWaves, /*Bitmask=*/false, /*Cached=*/true,
      /*Mapped=*/true, /*CrossReduce=*/false, /*SinglePoller=*/true,
      /*StageFeatures=*/true, /*SeqFlag=*/false, 1, CacheAllWaves>;
  register_cuda_one<Policy, TNodeBuffer>(args, input, buffer);
}

// The bitmask (QuickScorer) single-poller + stage policy with intra-tree
// parallelism: TPT lanes co-walk one tree, splitting the 63 nodes. TPT==1 is
// the original one-thread-per-tree bitmask. SeqFlag selects the request signal
// (request_seq word vs feature-buffer sentinel) -- swept so bitmask is compared
// against the path-chase configs under the *same* signaling, isolating the walk
// algorithm from the host<->device coordination lever. cuda_policy_supported()
// reports the large-TPT x large-n cells that exceed co-residency as unsupported
// (skipped), which is the informative boundary the sweep documents.
template <int TPT, bool SeqFlag, bool Transposed,
          template <typename> typename TNodeBuffer>
using BitmaskTptPolicy = qleaf::CudaTraversePersistentSP<
    kCudaBT, Transposed, /*Bitmask=*/true, /*Cached=*/false,
    /*Mapped=*/true, /*CrossReduce=*/false, /*SinglePoller=*/true,
    /*StageFeatures=*/true, SeqFlag, TPT>;

template <int TPT, bool SeqFlag, bool Transposed,
          template <typename> typename TNodeBuffer>
void register_cuda_bitmask_tpt(const Args &args, const BenchmarkInput *input,
                               std::string_view buffer) {
  register_cuda_one<BitmaskTptPolicy<TPT, SeqFlag, Transposed, TNodeBuffer>,
                    TNodeBuffer>(args, input, buffer);
}

template <bool SeqFlag, bool Transposed, template <typename> typename TNodeBuffer,
          int... TPTs>
void register_cuda_bitmask_tpt_sweep(const Args &args,
                                     const BenchmarkInput *input,
                                     std::string_view buffer,
                                     std::integer_sequence<int, TPTs...>) {
  (register_cuda_bitmask_tpt<1 << TPTs, SeqFlag, Transposed, TNodeBuffer>(
       args, input, buffer),
   ...);
}

// Oneshot bitmask with intra-tree TPT: a clean single-launch kernel (no
// persistent host<->device handshake) that exercises the same shared-mask /
// strided-walk / WarpReduce code -- the right target for kernel profilers
// (ncu/nsys), which cannot replay the resident persistent kernel. !Mapped so
// the feature vector is device-resident (no zero-copy PCIe noise in the walk).
template <int TPT, bool Transposed, template <typename> typename TNodeBuffer>
using BitmaskOneshotTptPolicy =
    qleaf::CudaTraverseOneShot<kCudaBT, Transposed, /*Bitmask=*/true,
                               /*Cached=*/false, /*Mapped=*/false, TPT>;

template <int TPT, bool Transposed, template <typename> typename TNodeBuffer>
void register_cuda_bitmask_oneshot_tpt(const Args &args,
                                       const BenchmarkInput *input,
                                       std::string_view buffer) {
  register_cuda_one<BitmaskOneshotTptPolicy<TPT, Transposed, TNodeBuffer>,
                    TNodeBuffer>(args, input, buffer);
}

template <bool Transposed, template <typename> typename TNodeBuffer, int... TPTs>
void register_cuda_bitmask_oneshot_tpt_sweep(
    const Args &args, const BenchmarkInput *input, std::string_view buffer,
    std::integer_sequence<int, TPTs...>) {
  (register_cuda_bitmask_oneshot_tpt<1 << TPTs, Transposed, TNodeBuffer>(
       args, input, buffer),
   ...);
}

// §9 acceptance: every TPT>1 bitmask result must match the TPT=1 bitmask within
// rel-tol 1e-4 (same AND reduction, same leaf -- only the work partition
// differs; neither SeqFlag nor Transposed changes the math). Co-residency-
// skipped TPTs are not checked (they won't be benchmarked either). Aborts on
// divergence.
template <int TPT, bool SeqFlag, bool Transposed,
          template <typename> typename TNodeBuffer>
void verify_one_bitmask_tpt(const BenchmarkInput &input, float ref, float tol) {
  using P = BitmaskTptPolicy<TPT, SeqFlag, Transposed, TNodeBuffer>;
  if (!cuda_policy_supported<P>(input)) return;  // exceeds co-residency: skip
  const float v = cuda_predict_once<P, TNodeBuffer>(input);
  if (!std::isfinite(v) || std::abs(v - ref) > tol) {
    std::ostringstream os;
    os << "bitmask TPT=" << TPT << " (seqflag=" << SeqFlag
       << " transposed=" << Transposed << ") diverges from TPT=1: " << v
       << " vs " << ref << " (tol=" << tol << ")";
    throw std::runtime_error(os.str());
  }
}

template <bool SeqFlag, bool Transposed, template <typename> typename TNodeBuffer,
          int... TPTs>
void cuda_verify_bitmask_tpt(const BenchmarkInput &input,
                             std::integer_sequence<int, TPTs...>) {
  if (!input.cuda_enabled || input.mapped_features.empty()) return;
  using P1 = BitmaskTptPolicy<1, SeqFlag, Transposed, TNodeBuffer>;
  if (!cuda_policy_supported<P1>(input)) return;  // bitmask unsupported (depth)
  const float ref = cuda_predict_once<P1, TNodeBuffer>(input);
  const float tol = 1e-4f * std::max(1.0f, std::abs(ref));
  (verify_one_bitmask_tpt<1 << TPTs, SeqFlag, Transposed, TNodeBuffer>(input, ref,
                                                                       tol),
   ...);
  std::fprintf(stderr,
               "[cuda-verify-tpt] bitmask TPT 1..%d (seqflag=%d transposed=%d) "
               "match TPT=1 (ref=%.8g)\n",
               1 << (sizeof...(TPTs)), (int)SeqFlag, (int)Transposed, ref);
}

template <template <typename> typename TNodeBuffer>
void register_cuda_buffer(const Args &args, const BenchmarkInput *input,
                          std::string_view buffer) {
  using Sweep = std::make_integer_sequence<int, 6>;
#if QLEAF_WORKER_CUDA_SP
  cuda_verify<TNodeBuffer>(*input);  // §8.1 gate before benchmarking
#endif
#if QLEAF_WORKER_CUDA_BITMASK_TPT
  // §9 gate: bitmask TPT 2..32 must match TPT=1 (exponents 1..5). The result is
  // layout/signal-independent, so verify under the fair signal (seqflag=0) for
  // both node layouts -- this also catches a rowmajor-vs-transposed indexing bug.
  using Tpts = std::integer_sequence<int, 1, 2, 3, 4, 5>;
  cuda_verify_bitmask_tpt</*SeqFlag=*/false, /*Transposed=*/true, TNodeBuffer>(
      *input, Tpts{});
  cuda_verify_bitmask_tpt</*SeqFlag=*/false, /*Transposed=*/false, TNodeBuffer>(
      *input, Tpts{});
#endif
#if QLEAF_WORKER_CUDA_ONESHOT_COMBOS
  register_cuda_oneshot_combos<TNodeBuffer>(
      args, input, buffer, std::make_integer_sequence<int, 16>{});
#endif
#if QLEAF_WORKER_CUDA_PERSISTENT_COMBOS
  register_cuda_persistent_combos<TNodeBuffer>(
      args, input, buffer, std::make_integer_sequence<int, 32>{});
#endif
#if QLEAF_WORKER_CUDA_SP
  // SeqFlag (request_seq) variants: config-1 (no stage) and config-2 (stage).
  register_cuda_sp</*StageFeatures=*/false, /*SeqFlag=*/true, TNodeBuffer>(
      args, input, buffer);
  register_cuda_sp</*StageFeatures=*/true, /*SeqFlag=*/true, TNodeBuffer>(
      args, input, buffer);
  // Sentinel signal (!SeqFlag, requires StageFeatures): config-2-sentinel.
  register_cuda_sp</*StageFeatures=*/true, /*SeqFlag=*/false, TNodeBuffer>(
      args, input, buffer);
#endif
#if QLEAF_WORKER_CUDA_SP_CACHE
  register_cuda_sp_cached</*CacheAllWaves=*/false, TNodeBuffer>(args, input,
                                                                buffer);
  register_cuda_sp_cached</*CacheAllWaves=*/true, TNodeBuffer>(args, input,
                                                               buffer);
#endif
#if QLEAF_WORKER_CUDA_BITMASK_TPT
  // Bitmask (QuickScorer) single-poller + stage, swept over intra-tree
  // parallelism TPT in {1,2,4,8,16,32} (exponents 0..5). Registered for both
  // signals AND both node layouts: transposed coalesces across trees (good for
  // TPT=1), rowmajor coalesces the TPT lanes' consecutive nodes within one tree
  // (the natural layout for TPT>1). Co-residency-exceeding cells self-skip.
  register_cuda_bitmask_tpt_sweep</*SeqFlag=*/true, /*Transposed=*/true,
                                  TNodeBuffer>(args, input, buffer, Sweep{});
  register_cuda_bitmask_tpt_sweep</*SeqFlag=*/false, /*Transposed=*/true,
                                  TNodeBuffer>(args, input, buffer, Sweep{});
  register_cuda_bitmask_tpt_sweep</*SeqFlag=*/true, /*Transposed=*/false,
                                  TNodeBuffer>(args, input, buffer, Sweep{});
  register_cuda_bitmask_tpt_sweep</*SeqFlag=*/false, /*Transposed=*/false,
                                  TNodeBuffer>(args, input, buffer, Sweep{});
#endif
#if QLEAF_WORKER_CUDA_BITMASK_ONESHOT
  // Oneshot bitmask TPT (both layouts): single-launch kernels for ncu/nsys.
  register_cuda_bitmask_oneshot_tpt_sweep</*Transposed=*/true, TNodeBuffer>(
      args, input, buffer, Sweep{});
  register_cuda_bitmask_oneshot_tpt_sweep</*Transposed=*/false, TNodeBuffer>(
      args, input, buffer, Sweep{});
#endif
}

template <template <typename> typename TNodeBuffer>
void register_buffer(const Args &args, const BenchmarkInput *input,
                     std::string_view buffer) {
  // No-thread baseline: worker runs inline in the calling thread.
  check_match<DirectBenchmarkInferrer, TNodeBuffer>(*input, 1);
  register_one<DirectBenchmarkInferrer<qleaf::BranchRegressionWorker,
                                       TNodeBuffer>>(
      nothread_name("branch", buffer, args.features), input, 1);
  register_one<DirectBenchmarkInferrer<qleaf::BitmaskRegressionWorker,
                                       TNodeBuffer>>(
      nothread_name("bitmask", buffer, args.features), input, 1);

  for (auto threads : args.threads) {
    check_match<ThreadedBenchmarkInferrer, TNodeBuffer>(*input, threads);
    register_one<
        ThreadedBenchmarkInferrer<qleaf::BranchRegressionWorker, TNodeBuffer>>(
        name_for("branch", buffer, threads, args.features), input, threads);
    register_one<
        ThreadedBenchmarkInferrer<qleaf::BitmaskRegressionWorker, TNodeBuffer>>(
        name_for("bitmask", buffer, threads, args.features), input, threads);
  }

  register_cuda_buffer<TNodeBuffer>(args, input, buffer);
}

void register_benchmarks(const Args &args, const BenchmarkInput *input) {
  for (auto kind : args.buffers) {
    switch (kind) {
      case BufferKind::Compact:
        register_buffer<qleaf::CompactNodeBuffer>(args, input, "compact");
        break;
      case BufferKind::Tree:
        register_buffer<qleaf::TreeNodeBuffer>(args, input, "tree");
        break;
    }
  }
}

std::vector<char *> argv_view(std::vector<std::string> &args) {
  std::vector<char *> ret;
  ret.reserve(args.size());
  for (auto &arg : args) ret.push_back(arg.data());
  return ret;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    std::vector<std::string> benchmark_args_storage;
    auto args = parse_args(argc, argv, benchmark_args_storage);

    if (!args.pin_cores.empty()) qleaf::detail::pin_to_core(args.pin_cores[0]);

    BenchmarkInput input{
        .forest = load_json(args.trees_path),
        .features =
            args.features_csv.empty()
                ? generate_features(args.features, args.samples, args.seed)
                : load_features_csv(args.features_csv, args.features,
                                    args.samples),
        .iterations = args.iterations,
        .pin_cores = args.pin_cores,
    };
    input.persistent_grid = args.persistent_grid;
    if (args.ntrees) truncate_forest(input.forest, *args.ntrees);
    validate_forest(input.forest, args.features);
    input.cuda_enabled = prepare_cuda_features(input);

    register_benchmarks(args, &input);

    auto benchmark_argv = argv_view(benchmark_args_storage);
    int benchmark_argc = static_cast<int>(benchmark_argv.size());
    benchmark::Initialize(&benchmark_argc, benchmark_argv.data());
    if (benchmark::ReportUnrecognizedArguments(benchmark_argc,
                                               benchmark_argv.data())) {
      return 1;
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
