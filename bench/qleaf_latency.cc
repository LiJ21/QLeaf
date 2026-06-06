#include <hdr/hdr_histogram.h>
#include <hdr/hdr_histogram_log.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
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
#include <sys/stat.h>
#include <sys/types.h>

#include "config.h"
#include "qleaf.h"
#include "qleaf_latency_common.h"

namespace {

#ifndef QLEAF_BT
#define QLEAF_BT 256
#endif
constexpr int kCompiledBT = QLEAF_BT;

enum class BufferKind { Compact, Tree };

struct Args {
  std::string trees_path;
  std::string features_csv;
  std::string out_dir = "bench/results/latency";
  std::vector<size_t> threads{1};
  std::vector<BufferKind> buffers{BufferKind::Compact};
  std::vector<int> bts{kCompiledBT};
  std::vector<int> tpts{1, 2, 4, 8, 16, 32};
  size_t features = 0;
  size_t samples = 1024;
  size_t warmup = 1000;
  size_t iters = 10000;
  uint32_t seed = 12345;
  std::optional<size_t> ntrees;
  std::vector<int> pin_cores;
  bool cpu = true;
  bool gpu = true;
  std::string gpu_suite = "best";
  std::string filter;
  int persistent_grid = 0;
  int64_t highest_ns = 60LL * 1000 * 1000 * 1000;
  int sigfig = 3;
};

[[noreturn]] void usage(std::string_view message = {}) {
  if (!message.empty()) std::cerr << "error: " << message << "\n\n";
  std::cerr
      << "usage: qleaf_latency --trees forest.qleaf.json --features N "
         "[options]\n\n"
      << "  --features-csv file       CSV feature rows; otherwise random rows\n"
      << "  --samples N               feature rows to load/generate (default 1024)\n"
      << "  --warmup N                warmup requests/config (default 1000)\n"
      << "  --iters N                 measured requests/config (default 10000)\n"
      << "  --ntrees N                use the first N trees of the model\n"
      << "  --threads 1,2,4           CPU thread counts (default 1)\n"
      << "  --buffers compact,tree     qleaf node buffers (default compact)\n"
      << "  --bt N                    expected compiled BT (default QLEAF_BT)\n"
      << "  --tpt 1,2,4,8,16,32       GPU bitmask TPT sweep\n"
      << "  --cpu 0|1                 enable qleaf CPU configs (default 1)\n"
      << "  --gpu 0|1                 enable qleaf GPU configs (default 1)\n"
      << "  --gpu-suite best|sp|bitmask-tpt|bitmask-cache|all|oneshot\n"
      << "  --filter text             run only names containing text\n"
      << "  --out-dir dir             output directory\n"
      << "  --persistent-grid N       force persistent grid width\n"
      << "  --pin-cores C0,C1,...     pin dispatcher to C0 and workers to "
         "C1..CN\n";
  std::exit(message.empty() ? 0 : 1);
}

int64_t now_ns() {
  timespec ts{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {
    throw std::runtime_error(std::string{"clock_gettime failed: "} +
                             std::strerror(errno));
  }
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
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

int parse_int(std::string_view text, std::string_view name) {
  size_t pos = 0;
  std::string value{text};
  auto ret = std::stoi(value, &pos);
  if (pos != value.size()) {
    throw std::invalid_argument("invalid integer for " + std::string{name});
  }
  return ret;
}

bool parse_bool(std::string_view text, std::string_view name) {
  if (text == "1" || text == "true" || text == "yes") return true;
  if (text == "0" || text == "false" || text == "no") return false;
  throw std::invalid_argument("invalid boolean for " + std::string{name});
}

template <typename T, typename ParseOne>
std::vector<T> parse_list(std::string_view text, std::string_view name,
                          ParseOne parse_one) {
  std::vector<T> ret;
  std::stringstream ss{std::string{text}};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty list item");
    ret.push_back(parse_one(item));
  }
  if (ret.empty()) throw std::invalid_argument("empty list for " +
                                               std::string{name});
  return ret;
}

std::vector<BufferKind> parse_buffers(std::string_view text) {
  return parse_list<BufferKind>(text, "--buffers", [](std::string_view item) {
    if (item == "compact") return BufferKind::Compact;
    if (item == "tree") return BufferKind::Tree;
    throw std::invalid_argument("unknown buffer '" + std::string{item} + "'");
  });
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    auto need_value = [&](std::string_view name) -> std::string_view {
      if (i + 1 >= argc) usage(std::string{name} + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") usage();
    else if (arg == "--trees") args.trees_path = need_value(arg);
    else if (arg == "--features-csv") args.features_csv = need_value(arg);
    else if (arg == "--features") args.features = parse_size(need_value(arg), arg);
    else if (arg == "--samples") args.samples = parse_size(need_value(arg), arg);
    else if (arg == "--warmup") args.warmup = parse_size(need_value(arg), arg);
    else if (arg == "--iters") args.iters = parse_size(need_value(arg), arg);
    else if (arg == "--ntrees") args.ntrees = parse_size(need_value(arg), arg);
    else if (arg == "--threads") {
      args.threads = parse_list<size_t>(need_value(arg), arg,
                                        [&](std::string_view v) {
                                          return parse_size(v, arg);
                                        });
    } else if (arg == "--buffers") {
      args.buffers = parse_buffers(need_value(arg));
    } else if (arg == "--bt") {
      args.bts = parse_list<int>(need_value(arg), arg,
                                 [&](std::string_view v) {
                                   return parse_int(v, arg);
                                 });
    } else if (arg == "--tpt") {
      args.tpts = parse_list<int>(need_value(arg), arg,
                                  [&](std::string_view v) {
                                    return parse_int(v, arg);
                                  });
    } else if (arg == "--cpu") args.cpu = parse_bool(need_value(arg), arg);
    else if (arg == "--gpu") args.gpu = parse_bool(need_value(arg), arg);
    else if (arg == "--gpu-suite") args.gpu_suite = need_value(arg);
    else if (arg == "--filter") args.filter = need_value(arg);
    else if (arg == "--out-dir") args.out_dir = need_value(arg);
    else if (arg == "--persistent-grid") args.persistent_grid = parse_int(need_value(arg), arg);
    else if (arg == "--highest-ns") args.highest_ns = static_cast<int64_t>(parse_size(need_value(arg), arg));
    else if (arg == "--sigfig") args.sigfig = parse_int(need_value(arg), arg);
    else if (arg == "--seed") args.seed = static_cast<uint32_t>(parse_size(need_value(arg), arg));
    else if (arg == "--pin-cores") {
      args.pin_cores = parse_list<int>(need_value(arg), arg,
                                       [&](std::string_view v) {
                                         auto core = parse_int(v, arg);
                                         if (core < 0)
                                           throw std::invalid_argument(
                                               "--pin-cores values must be >= 0");
                                         return core;
                                       });
    } else if (arg == "--pin") {
      usage("--pin was removed; use --pin-cores dispatch,worker0,...");
    }
    else usage("unknown option " + std::string{arg});
  }
  if (args.trees_path.empty()) usage("--trees is required");
  if (args.features == 0) usage("--features must be > 0");
  if (args.samples == 0) usage("--samples must be > 0");
  if (args.iters == 0) usage("--iters must be > 0");
  if (args.sigfig < 1 || args.sigfig > 5) usage("--sigfig must be in [1,5]");
  for (int tpt : args.tpts) {
    if (tpt <= 0 || tpt > 32 || (tpt & (tpt - 1)) != 0)
      usage("--tpt values must be powers of two in [1,32]");
  }
  if (args.gpu_suite != "best" && args.gpu_suite != "sp" &&
      args.gpu_suite != "all" && args.gpu_suite != "bitmask-tpt" &&
      args.gpu_suite != "bitmask-cache" && args.gpu_suite != "oneshot") {
    usage("--gpu-suite must be best, sp, all, bitmask-tpt, bitmask-cache, or "
          "oneshot");
  }
  if (!args.pin_cores.empty()) {
    size_t needed = 1;
    if (args.cpu) {
      needed += *std::max_element(args.threads.begin(), args.threads.end());
    }
    if (args.pin_cores.size() < needed) {
      usage("--pin-cores needs dispatcher plus one worker core per CPU worker "
            "in the largest --threads value");
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
    for (size_t i = 0; i < internal_nodes; ++i)
      max_idx = std::max(max_idx, indices.at(i).get<size_t>());
  }
  return max_idx;
}

void truncate_forest(nlohmann::json &forest, size_t ntrees) {
  auto &trees = forest.at("trees");
  if (ntrees > trees.size()) throw std::runtime_error("--ntrees exceeds forest size");
  trees.erase(std::next(trees.begin(), static_cast<std::ptrdiff_t>(ntrees)),
              trees.end());
}

void validate_forest(const nlohmann::json &forest, size_t features) {
  auto depth = forest.at("depth").get<size_t>();
  if (depth == 0 || depth > qleaf::kDefaultMaxDepth) {
    throw std::runtime_error("depth out of qleaf range");
  }
  auto tree_size = (size_t{1} << (depth + 1)) - 1;
  const auto &trees = forest.at("trees");
  if (trees.empty()) throw std::runtime_error("forest has no trees");
  for (const auto &tree : trees) {
    if (tree.at("indices").size() != tree_size ||
        tree.at("splits").size() != tree_size) {
      throw std::runtime_error("each tree must be perfect and fixed-depth");
    }
  }
  auto max_idx = max_feature_index(forest);
  if (max_idx >= features) throw std::runtime_error("forest references missing feature");
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

struct BenchmarkInput {
  nlohmann::json forest;
  std::vector<std::vector<float>> features;
};

template <template <typename, typename> typename TWorker>
struct ThreadedWorkerBinding {
  template <typename TValue, typename TSpan>
  using Worker = qleaf::ThreadedWorker<TWorker<TValue, TSpan>>;
};

template <template <typename, typename> typename TWorker,
          template <typename> typename TNodeBuffer>
using DirectInferrer =
    qleaf::Inferrer<float, TWorker, TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

template <template <typename, typename> typename TWorker,
          template <typename> typename TNodeBuffer>
using ThreadedInferrer =
    qleaf::Inferrer<float, ThreadedWorkerBinding<TWorker>::template Worker,
                    TNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

class Histogram {
 public:
  Histogram(int64_t highest_ns, int sigfig) {
    if (hdr_init(1, highest_ns, sigfig, &hist_) != 0)
      throw std::runtime_error("hdr_init failed");
  }
  ~Histogram() {
    if (hist_) hdr_close(hist_);
  }
  void record(int64_t ns) {
    if (ns < 1) ns = 1;
    if (!hdr_record_value(hist_, ns)) dropped_++;
  }
  nlohmann::json summary() const {
    return {
        {"count", hist_->total_count},
        {"dropped", dropped_},
        {"min_ns", hdr_min(hist_)},
        {"max_ns", hdr_max(hist_)},
        {"mean_ns", hdr_mean(hist_)},
        {"stddev_ns", hdr_stddev(hist_)},
        {"p50_ns", hdr_value_at_percentile(hist_, 50.0)},
        {"p90_ns", hdr_value_at_percentile(hist_, 90.0)},
        {"p95_ns", hdr_value_at_percentile(hist_, 95.0)},
        {"p99_ns", hdr_value_at_percentile(hist_, 99.0)},
        {"p99_90_ns", hdr_value_at_percentile(hist_, 99.9)},
        {"p99_99_ns", hdr_value_at_percentile(hist_, 99.99)},
    };
  }
  void write_log(const std::string &path, const std::string &name,
                 int64_t elapsed_ns) const {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("failed to open " + path);
    hdr_log_writer writer{};
    hdr_log_writer_init(&writer);
    hdr_timespec start{};
    hdr_timespec_from_double(&start, 0.0);
    std::string prefix = "# " + name + "\n";
    hdr_log_write_header(&writer, f, prefix.c_str(), &start);
    hdr_timespec begin{};
    hdr_timespec end{};
    hdr_timespec_from_double(&begin, 0.0);
    hdr_timespec_from_double(&end, static_cast<double>(elapsed_ns) / 1e9);
    hdr_log_write(&writer, f, &begin, &end, hist_);
    std::fclose(f);
  }
  void write_percentiles(const std::string &path) const {
    FILE *f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("failed to open " + path);
    hdr_percentiles_print(hist_, f, 5, 1.0, CSV);
    std::fclose(f);
  }

 private:
  hdr_histogram *hist_ = nullptr;
  int64_t dropped_ = 0;
};

void ensure_dir(const std::string &path) {
  if (path.empty()) return;
  std::string cur;
  for (size_t i = 0; i < path.size(); ++i) {
    cur.push_back(path[i]);
    if (path[i] != '/' && i + 1 != path.size()) continue;
    if (cur.empty() || cur == "/") continue;
    if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST)
      throw std::runtime_error("mkdir failed for " + cur);
  }
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) return b;
  return a.back() == '/' ? a + b : a + "/" + b;
}

std::string safe_name(std::string name) {
  for (char &c : name) {
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' ||
          c == '_')) c = '_';
  }
  return name;
}

struct RunContext {
  Args args;
  BenchmarkInput input;
  std::vector<nlohmann::json> results;
  bool want(std::string_view name) const {
    return args.filter.empty() ||
           std::string{name}.find(args.filter) != std::string::npos;
  }
};

nlohmann::json base_meta(const RunContext &ctx, std::string_view name,
                         std::string_view status = "ok") {
  return {
      {"name", name},
      {"status", status},
      {"clock", "CLOCK_MONOTONIC_RAW"},
      {"histogram", "HdrHistogram_c"},
      {"warmup", ctx.args.warmup},
      {"iters", ctx.args.iters},
      {"samples", ctx.input.features.size()},
      {"features", ctx.args.features},
      {"ntrees", ctx.input.forest.at("trees").size()},
      {"depth", ctx.input.forest.at("depth").get<size_t>()},
  };
}

void record_run(RunContext &ctx, const std::string &name,
                const std::vector<int64_t> &latencies, float last_result,
                nlohmann::json extra = {},
                const std::map<std::string, std::vector<int64_t>> &phases = {}) {
  if (!ctx.want(name)) return;
  Histogram hist{ctx.args.highest_ns, ctx.args.sigfig};
  for (auto ns : latencies) hist.record(ns);
  auto j = base_meta(ctx, name);
  j["last_result"] = last_result;
  j["latency"] = hist.summary();
  j.update(extra);
  ensure_dir(ctx.args.out_dir);
  const auto stem = safe_name(name);
  const auto hdr_path = join_path(ctx.args.out_dir, stem + ".hdr");
  const auto csv_path = join_path(ctx.args.out_dir, stem + ".percentiles.csv");
  int64_t elapsed = 0;
  for (auto ns : latencies) elapsed += ns;
  hist.write_log(hdr_path, name, elapsed);
  hist.write_percentiles(csv_path);
  j["hdr_log"] = hdr_path;
  j["percentiles_csv"] = csv_path;
  if (!phases.empty()) {
    nlohmann::json phase_json = nlohmann::json::object();
    for (const auto &[phase_name, values] : phases) {
      Histogram phase_hist{ctx.args.highest_ns, ctx.args.sigfig};
      for (auto ns : values) phase_hist.record(ns);
      const auto phase_stem = stem + "." + safe_name(phase_name);
      const auto phase_hdr = join_path(ctx.args.out_dir, phase_stem + ".hdr");
      const auto phase_csv =
          join_path(ctx.args.out_dir, phase_stem + ".percentiles.csv");
      int64_t phase_elapsed = 0;
      for (auto ns : values) phase_elapsed += ns;
      phase_hist.write_log(phase_hdr, name + "/" + phase_name, phase_elapsed);
      phase_hist.write_percentiles(phase_csv);
      phase_json[phase_name] = {
          {"latency", phase_hist.summary()},
          {"hdr_log", phase_hdr},
          {"percentiles_csv", phase_csv},
      };
    }
    j["phases"] = std::move(phase_json);
  }
  ctx.results.push_back(std::move(j));
}

void record_skip(RunContext &ctx, const std::string &name, std::string reason) {
  if (!ctx.want(name)) return;
  auto j = base_meta(ctx, name, "skipped");
  j["reason"] = std::move(reason);
  ctx.results.push_back(std::move(j));
}

template <typename TInferrer>
void run_cpu_one(RunContext &ctx, std::string name, size_t threads) {
  if (!ctx.want(name)) return;
  std::cerr << "[run] " << name << "\n";
  auto config_json =
      config_for_threads(ctx.input.forest, threads, ctx.args.pin_cores,
                         ctx.args.features);
  qleaf::Config config{config_json};
  TInferrer inferrer{config};
  size_t row = 0;
  float last = 0.0f;
  for (size_t i = 0; i < ctx.args.warmup; ++i)
    last = inferrer.predict(ctx.input.features[row++ % ctx.input.features.size()]);
  std::vector<int64_t> lat;
  lat.reserve(ctx.args.iters);
#ifdef QLEAF_LATENCY_DECOMP
  std::map<std::string, std::vector<int64_t>> phases;
  phases["predict_ns"].reserve(ctx.args.iters);
#endif
  for (size_t i = 0; i < ctx.args.iters; ++i) {
    const auto r = row++ % ctx.input.features.size();
    const auto t0 = now_ns();
    last = inferrer.predict(ctx.input.features[r]);
    const auto t1 = now_ns();
    lat.push_back(t1 - t0);
#ifdef QLEAF_LATENCY_DECOMP
    phases["predict_ns"].push_back(t1 - t0);
#endif
  }
#ifdef QLEAF_LATENCY_DECOMP
  record_run(ctx, name, lat, last,
             {{"engine", "qleaf_cpu"}, {"threads", threads}}, phases);
#else
  record_run(ctx, name, lat, last,
             {{"engine", "qleaf_cpu"}, {"threads", threads}});
#endif
}

template <template <typename> typename TNodeBuffer>
void run_cpu_buffer(RunContext &ctx, std::string_view buffer) {
  using BranchST =
      DirectInferrer<qleaf::BranchRegressionWorker, TNodeBuffer>;
  using BitmaskST =
      DirectInferrer<qleaf::BitmaskRegressionWorker, TNodeBuffer>;
  run_cpu_one<BranchST>(ctx, "qleaf/cpu/" + std::string{buffer} +
                                 "/branch/nothread",
                        1);
  run_cpu_one<BitmaskST>(ctx, "qleaf/cpu/" + std::string{buffer} +
                                  "/bitmask/nothread",
                         1);
  for (auto threads : ctx.args.threads) {
    using BranchTH =
        ThreadedInferrer<qleaf::BranchRegressionWorker, TNodeBuffer>;
    using BitmaskTH =
        ThreadedInferrer<qleaf::BitmaskRegressionWorker, TNodeBuffer>;
    run_cpu_one<BranchTH>(ctx, "qleaf/cpu/" + std::string{buffer} +
                                   "/branch/threads:" +
                                   std::to_string(threads),
                          threads);
    run_cpu_one<BitmaskTH>(ctx, "qleaf/cpu/" + std::string{buffer} +
                                    "/bitmask/threads:" +
                                    std::to_string(threads),
                           threads);
  }
}

void run_gpu(RunContext &ctx, std::string_view buffer) {
  for (int bt : ctx.args.bts) {
    if (bt != kCompiledBT) {
      record_skip(ctx, "qleaf/gpu/bt:" + std::to_string(bt) + "/" +
                           std::string{buffer},
                  "this binary was compiled with QLEAF_BT=" +
                      std::to_string(kCompiledBT));
      continue;
    }
    GpuLatencyRequest req;
    req.forest = &ctx.input.forest;
    req.features = &ctx.input.features;
    req.n_features = ctx.args.features;
    req.warmup = ctx.args.warmup;
    req.iters = ctx.args.iters;
    req.persistent_grid = ctx.args.persistent_grid;
    req.buffer = std::string{buffer};
    req.suite = ctx.args.gpu_suite;
    req.tpts = ctx.args.tpts;
    req.filter = ctx.args.filter;
    for (auto &r : run_qleaf_gpu_latency(req)) {
      if (r.status == "ok") {
        record_run(ctx, r.name, r.latency_ns, r.last_result, r.extra, r.phases);
      } else {
        record_skip(ctx, r.name, r.reason);
      }
    }
  }
}

void write_summary(RunContext &ctx) {
  ensure_dir(ctx.args.out_dir);
  nlohmann::json root;
  root["runs"] = ctx.results;
  root["meta"] = {
      {"clock", "CLOCK_MONOTONIC_RAW"},
      {"histogram", "HdrHistogram_c"},
      {"trees", ctx.args.trees_path},
      {"features_csv", ctx.args.features_csv},
      {"features", ctx.args.features},
      {"samples", ctx.input.features.size()},
      {"warmup", ctx.args.warmup},
      {"iters", ctx.args.iters},
      {"ntrees", ctx.input.forest.at("trees").size()},
      {"depth", ctx.input.forest.at("depth").get<size_t>()},
      {"gpu_suite", ctx.args.gpu_suite},
      {"compiled_bt", kCompiledBT},
      {"pin_cores", ctx.args.pin_cores},
  };
  std::ofstream out{join_path(ctx.args.out_dir, "summary.json")};
  out << std::setw(2) << root << "\n";
  std::ofstream csv{join_path(ctx.args.out_dir, "summary.csv")};
  csv << "name,status,count,p50_ns,p90_ns,p99_ns,p99_9_ns,mean_ns,max_ns,reason\n";
  for (const auto &r : ctx.results) {
    nlohmann::json lat =
        (r.contains("latency") && r["latency"].is_object())
            ? r["latency"]
            : nlohmann::json::object();
    csv << std::quoted(r.value("name", "")) << ',' << r.value("status", "")
        << ',' << lat.value("count", 0LL) << ',' << lat.value("p50_ns", 0LL)
        << ',' << lat.value("p90_ns", 0LL) << ','
        << lat.value("p99_ns", 0LL) << ',' << lat.value("p99_90_ns", 0LL)
        << ',' << lat.value("mean_ns", 0.0) << ',' << lat.value("max_ns", 0LL)
        << ',' << std::quoted(r.value("reason", "")) << "\n";
  }
}

void run_buffer(RunContext &ctx, BufferKind kind) {
  switch (kind) {
    case BufferKind::Compact:
      if (ctx.args.cpu) run_cpu_buffer<qleaf::CompactNodeBuffer>(ctx, "compact");
      if (ctx.args.gpu) run_gpu(ctx, "compact");
      break;
    case BufferKind::Tree:
      if (ctx.args.cpu) run_cpu_buffer<qleaf::TreeNodeBuffer>(ctx, "tree");
      if (ctx.args.gpu) run_gpu(ctx, "tree");
      break;
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Args args = parse_args(argc, argv);
    if (!args.pin_cores.empty()) qleaf::detail::pin_to_core(args.pin_cores[0]);
    BenchmarkInput input{
        .forest = load_json(args.trees_path),
        .features =
            args.features_csv.empty()
                ? generate_features(args.features, args.samples, args.seed)
                : load_features_csv(args.features_csv, args.features,
                                    args.samples),
    };
    if (args.ntrees) truncate_forest(input.forest, *args.ntrees);
    validate_forest(input.forest, args.features);
    RunContext ctx{.args = std::move(args), .input = std::move(input)};
    for (auto kind : ctx.args.buffers) run_buffer(ctx, kind);
    write_summary(ctx);
    std::cerr << "[done] wrote " << join_path(ctx.args.out_dir, "summary.json")
              << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
