#include <hdr/hdr_histogram.h>
#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <nlohmann/json.hpp>

#ifdef QLEAF_HAVE_NVFOREST
#include <cuda_runtime_api.h>
#include <nvforest/detail/raft_proto/device_type.hpp>
#include <nvforest/detail/raft_proto/handle.hpp>
#include <nvforest/forest_model.hpp>
#include <nvforest/infer_kind.hpp>
#include <nvforest/treelite_importer.hpp>
#include <raft/core/handle.hpp>
#include <treelite/c_api.h>
#endif

#ifdef QLEAF_HAVE_NVFOREST
namespace treelite {

std::unique_ptr<Model> ConcatenateModelObjects(
    std::vector<Model const*> const& objs) {
  if (objs.empty() || objs.front() == nullptr) {
    throw std::runtime_error("ConcatenateModelObjects requires a model");
  }
  auto const& first = *objs.front();
  auto result = Model::Create(first.GetThresholdType(), first.GetLeafOutputType());

  result->num_feature = first.num_feature;
  result->task_type = first.task_type;
  result->average_tree_output = first.average_tree_output;
  result->num_target = first.num_target;
  result->num_class = first.num_class.Clone();
  result->leaf_vector_shape = first.leaf_vector_shape.Clone();
  result->postprocessor = first.postprocessor;
  result->sigmoid_alpha = first.sigmoid_alpha;
  result->ratio_c = first.ratio_c;
  result->base_scores = first.base_scores.Clone();
  result->attributes = first.attributes;

  for (auto const* model : objs) {
    if (model == nullptr) {
      throw std::runtime_error("ConcatenateModelObjects got a null model");
    }
    if (model->GetThresholdType() != first.GetThresholdType() ||
        model->GetLeafOutputType() != first.GetLeafOutputType()) {
      throw std::runtime_error("ConcatenateModelObjects type mismatch");
    }
    std::visit(
        [](auto& dst, auto const& src) {
          using Dst = std::decay_t<decltype(dst)>;
          using Src = std::decay_t<decltype(src)>;
          if constexpr (std::is_same_v<Dst, Src>) {
            dst.trees.reserve(dst.trees.size() + src.trees.size());
            for (auto const& tree : src.trees) {
              dst.trees.push_back(tree.Clone());
            }
          }
        },
        result->variant_, model->variant_);
    result->target_id.Extend(model->target_id);
    result->class_id.Extend(model->class_id);
  }
  return result;
}

}  // namespace treelite
#endif

namespace {

struct Args {
  std::vector<std::string> engines{"treelite"};
  std::string features_csv;
  std::string xgb_model;
  std::string out_dir = "bench/results/latency_external_cpp";
  std::string tl2cgen_lib;
  std::string tl2cgen_runtime =
      "/tmp/qleaf-latency-rapids-py/tl2cgen/lib/libtl2cgen.so";
  std::string output = "margin";
  std::vector<std::string> nvforest_layouts{"depth_first", "layered",
                                            "breadth_first"};
  std::vector<size_t> nvforest_rows_per_block_iter{1, 2, 4, 8, 16, 32};
  std::vector<size_t> nvforest_align_bytes{0, 128};
  std::vector<std::string> nvforest_modes{"host_host", "device_host",
                                          "device_device_sync"};
  size_t features = 0;
  size_t samples = 1024;
  size_t warmup = 1000;
  size_t iters = 10000;
  size_t ntrees = 0;
  int threads = 1;
  int64_t highest_ns = 60LL * 1000 * 1000 * 1000;
  int sigfig = 3;
};

[[noreturn]] void usage(std::string_view message = {}) {
  if (!message.empty()) std::cerr << "error: " << message << "\n\n";
  std::cerr
      << "usage: external_latency_cpp --features N --features-csv file "
         "[options]\n\n"
      << "  --engine treelite|nvforest|all\n"
      << "  --engines a,b               comma-separated engine list\n"
      << "  --samples N                 rows to load (default 1024)\n"
      << "  --warmup N                  warmup requests (default 1000)\n"
      << "  --iters N                   measured requests (default 10000)\n"
      << "  --threads N                 TL2cgen predictor threads (default 1)\n"
      << "  --ntrees N                  metadata only; lib should already match\n"
      << "  --output margin|value       pred_margin flag (default margin)\n"
      << "  --tl2cgen-runtime path      libtl2cgen.so path\n"
      << "  --tl2cgen-lib path          TL2cgen compiled model for treelite engine\n"
      << "  --xgb-model path            XGBoost JSON model for nvforest engine\n"
      << "  --nvforest-layouts a,b      depth_first,layered,breadth_first\n"
      << "  --nvforest-rows-per-block-iter 1,2,4,8,16,32\n"
      << "  --nvforest-align-bytes 0,128\n"
      << "  --nvforest-modes host_host,device_host,device_device_sync\n"
      << "  --out-dir dir               output directory\n";
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

template <typename T, typename ParseOne>
std::vector<T> parse_list(std::string_view text, std::string_view name,
                          ParseOne parse_one) {
  std::vector<T> ret;
  std::stringstream ss{std::string{text}};
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty list item for " +
                                                  std::string{name});
    ret.push_back(parse_one(item));
  }
  if (ret.empty()) throw std::invalid_argument("empty list for " +
                                               std::string{name});
  return ret;
}

std::vector<std::string> parse_string_list(std::string_view text,
                                           std::string_view name) {
  return parse_list<std::string>(text, name, [](std::string_view item) {
    return std::string{item};
  });
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg{argv[i]};
    auto need_value = [&](std::string_view name) -> std::string_view {
      if (i + 1 >= argc) usage(std::string{name} + " requires a value");
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") usage();
    else if (arg == "--engine") {
      const auto value = std::string{need_value(arg)};
      if (value == "all") {
        args.engines = {"treelite", "nvforest"};
      } else {
        args.engines = {value};
      }
    }
    else if (arg == "--engines") args.engines = parse_string_list(need_value(arg), arg);
    else if (arg == "--features-csv") args.features_csv = need_value(arg);
    else if (arg == "--xgb-model") args.xgb_model = need_value(arg);
    else if (arg == "--features") args.features = parse_size(need_value(arg), arg);
    else if (arg == "--samples") args.samples = parse_size(need_value(arg), arg);
    else if (arg == "--warmup") args.warmup = parse_size(need_value(arg), arg);
    else if (arg == "--iters") args.iters = parse_size(need_value(arg), arg);
    else if (arg == "--threads") args.threads = parse_int(need_value(arg), arg);
    else if (arg == "--ntrees") args.ntrees = parse_size(need_value(arg), arg);
    else if (arg == "--output") args.output = need_value(arg);
    else if (arg == "--tl2cgen-lib") args.tl2cgen_lib = need_value(arg);
    else if (arg == "--tl2cgen-runtime") args.tl2cgen_runtime = need_value(arg);
    else if (arg == "--out-dir") args.out_dir = need_value(arg);
    else if (arg == "--nvforest-layouts") args.nvforest_layouts = parse_string_list(need_value(arg), arg);
    else if (arg == "--nvforest-modes") args.nvforest_modes = parse_string_list(need_value(arg), arg);
    else if (arg == "--nvforest-rows-per-block-iter") {
      args.nvforest_rows_per_block_iter =
          parse_list<size_t>(need_value(arg), arg, [&](std::string_view v) {
            return parse_size(v, arg);
          });
    }
    else if (arg == "--nvforest-align-bytes") {
      args.nvforest_align_bytes =
          parse_list<size_t>(need_value(arg), arg, [&](std::string_view v) {
            return parse_size(v, arg);
          });
    }
    else if (arg == "--highest-ns") args.highest_ns = static_cast<int64_t>(parse_size(need_value(arg), arg));
    else if (arg == "--sigfig") args.sigfig = parse_int(need_value(arg), arg);
    else usage("unknown option " + std::string{arg});
  }
  if (args.features == 0) usage("--features must be > 0");
  if (args.features_csv.empty()) usage("--features-csv is required");
  if (args.samples == 0) usage("--samples must be > 0");
  if (args.iters == 0) usage("--iters must be > 0");
  if (args.threads == 0) usage("--threads must be nonzero");
  if (args.output != "margin" && args.output != "value") {
    usage("--output must be margin or value");
  }
  for (const auto& engine : args.engines) {
    if (engine != "treelite" && engine != "nvforest") {
      usage("--engine/--engines supports treelite and nvforest");
    }
  }
  const auto wants = [&](std::string_view engine) {
    return std::find(args.engines.begin(), args.engines.end(), engine) !=
           args.engines.end();
  };
  if (wants("treelite") && args.tl2cgen_lib.empty()) {
    usage("--tl2cgen-lib is required for --engine treelite");
  }
  if (wants("nvforest") && args.xgb_model.empty()) {
    usage("--xgb-model is required for --engine nvforest");
  }
  for (const auto& layout : args.nvforest_layouts) {
    if (layout != "depth_first" && layout != "layered" &&
        layout != "breadth_first") {
      usage("--nvforest-layouts values must be depth_first, layered, or "
            "breadth_first");
    }
  }
  for (const auto& mode : args.nvforest_modes) {
    if (mode != "host_host" && mode != "device_host" &&
        mode != "device_device_sync") {
      usage("--nvforest-modes values must be host_host, device_host, or "
            "device_device_sync");
    }
  }
  for (size_t rpb : args.nvforest_rows_per_block_iter) {
    if (rpb == 0 || rpb > 32 || (rpb & (rpb - 1)) != 0) {
      usage("--nvforest-rows-per-block-iter values must be powers of two in "
            "[1,32]");
    }
  }
  return args;
}

void mkdir_p(const std::string& path) {
  if (path.empty()) return;
  std::stringstream ss{path};
  std::string cur;
  std::string part;
  if (path.front() == '/') cur = "/";
  while (std::getline(ss, part, '/')) {
    if (part.empty()) continue;
    if (!cur.empty() && cur.back() != '/') cur += "/";
    cur += part;
    if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) {
      throw std::runtime_error("failed to create directory " + cur + ": " +
                               std::strerror(errno));
    }
  }
}

std::vector<float> parse_csv_row(const std::string& line, size_t features) {
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

std::vector<float> load_features_csv(const Args& args) {
  std::ifstream in{args.features_csv};
  if (!in) throw std::runtime_error("failed to open CSV file: " + args.features_csv);
  std::vector<float> ret;
  ret.reserve(args.samples * args.features);
  std::string line;
  while (ret.size() < args.samples * args.features && std::getline(in, line)) {
    if (line.empty()) continue;
    auto row = parse_csv_row(line, args.features);
    if (row.empty()) continue;
    ret.insert(ret.end(), row.begin(), row.end());
  }
  if (ret.empty()) throw std::runtime_error("CSV file has no feature rows");
  return ret;
}

std::string safe_stem(std::string text) {
  for (char& c : text) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
    if (!ok) c = '_';
  }
  return text;
}

struct DlCloser {
  void operator()(void* handle) const {
    if (handle != nullptr) dlclose(handle);
  }
};

class Tl2cgenApi {
 public:
  using Handle = void*;

  explicit Tl2cgenApi(const std::string& path)
      : lib_{dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)} {
    if (!lib_) throw std::runtime_error(std::string{"dlopen failed: "} + dlerror());
    load("TL2cgenGetLastError", get_last_error_);
    load("TL2cgenPredictorLoad", predictor_load_);
    load("TL2cgenPredictorFree", predictor_free_);
    load("TL2cgenPredictorGetNumFeature", predictor_get_num_feature_);
    load("TL2cgenPredictorGetNumTarget", predictor_get_num_target_);
    load("TL2cgenPredictorGetNumClass", predictor_get_num_class_);
    load("TL2cgenPredictorGetLeafOutputType", predictor_get_leaf_output_type_);
    load("TL2cgenPredictorGetOutputShape", predictor_get_output_shape_);
    load("TL2cgenPredictorPredictBatch", predictor_predict_batch_);
    load("TL2cgenDMatrixCreateFromMat", dmatrix_create_from_mat_);
    load("TL2cgenDMatrixFree", dmatrix_free_);
  }

  const char* last_error() const {
    const char* msg = get_last_error_ ? get_last_error_() : nullptr;
    return msg ? msg : "unknown TL2cgen error";
  }

  void check(int code, const char* what) const {
    if (code != 0) {
      throw std::runtime_error(std::string{what} + ": " + last_error());
    }
  }

  Handle predictor_load(const std::string& path, int threads) const {
    Handle h = nullptr;
    check(predictor_load_(path.c_str(), threads, &h), "TL2cgenPredictorLoad");
    return h;
  }

  void predictor_free(Handle h) const { check(predictor_free_(h), "TL2cgenPredictorFree"); }

  int32_t predictor_num_feature(Handle h) const {
    int32_t out = 0;
    check(predictor_get_num_feature_(h, &out), "TL2cgenPredictorGetNumFeature");
    return out;
  }

  int32_t predictor_num_target(Handle h) const {
    int32_t out = 0;
    check(predictor_get_num_target_(h, &out), "TL2cgenPredictorGetNumTarget");
    return out;
  }

  std::vector<int32_t> predictor_num_class(Handle h, int32_t n) const {
    std::vector<int32_t> out(static_cast<size_t>(n));
    check(predictor_get_num_class_(h, out.data()), "TL2cgenPredictorGetNumClass");
    return out;
  }

  std::string predictor_leaf_output_type(Handle h) const {
    const char* out = nullptr;
    check(predictor_get_leaf_output_type_(h, &out), "TL2cgenPredictorGetLeafOutputType");
    return out ? out : "";
  }

  std::vector<uint64_t> output_shape(Handle predictor, Handle dmat) const {
    uint64_t* shape = nullptr;
    uint64_t ndim = 0;
    check(predictor_get_output_shape_(predictor, dmat, &shape, &ndim),
          "TL2cgenPredictorGetOutputShape");
    return std::vector<uint64_t>(shape, shape + ndim);
  }

  void predict_batch(Handle predictor, Handle dmat, bool pred_margin, void* output) const {
    check(predictor_predict_batch_(predictor, dmat, 0, pred_margin ? 1 : 0, output),
          "TL2cgenPredictorPredictBatch");
  }

  Handle dmatrix_create_from_mat(const float* data, size_t nrow, size_t ncol) const {
    Handle h = nullptr;
    float missing = std::numeric_limits<float>::quiet_NaN();
    check(dmatrix_create_from_mat_(data, "float32", nrow, ncol, &missing, &h),
          "TL2cgenDMatrixCreateFromMat");
    return h;
  }

  void dmatrix_free(Handle h) const { check(dmatrix_free_(h), "TL2cgenDMatrixFree"); }

 private:
  template <typename T>
  void load(const char* name, T& fn) {
    dlerror();
    void* sym = dlsym(lib_.get(), name);
    const char* err = dlerror();
    if (err != nullptr || sym == nullptr) {
      throw std::runtime_error(std::string{"dlsym failed for "} + name + ": " +
                               (err ? err : "null symbol"));
    }
    fn = reinterpret_cast<T>(sym);
  }

  std::unique_ptr<void, DlCloser> lib_;
  const char* (*get_last_error_)() = nullptr;
  int (*predictor_load_)(const char*, int, Handle*) = nullptr;
  int (*predictor_free_)(Handle) = nullptr;
  int (*predictor_get_num_feature_)(Handle, int32_t*) = nullptr;
  int (*predictor_get_num_target_)(Handle, int32_t*) = nullptr;
  int (*predictor_get_num_class_)(Handle, int32_t*) = nullptr;
  int (*predictor_get_leaf_output_type_)(Handle, const char**) = nullptr;
  int (*predictor_get_output_shape_)(Handle, Handle, uint64_t**, uint64_t*) = nullptr;
  int (*predictor_predict_batch_)(Handle, Handle, int, int, void*) = nullptr;
  int (*dmatrix_create_from_mat_)(const void*, const char*, size_t, size_t, const void*, Handle*) =
      nullptr;
  int (*dmatrix_free_)(Handle) = nullptr;
};

template <auto FreeFn>
struct HandleGuard {
  using Fn = decltype(FreeFn);
};

struct Histogram {
  hdr_histogram* h = nullptr;
  int dropped = 0;

  Histogram(int64_t highest_ns, int sigfig) {
    if (hdr_init(1, highest_ns, sigfig, &h) != 0 || h == nullptr) {
      throw std::runtime_error("hdr_init failed");
    }
  }
  ~Histogram() {
    if (h) hdr_close(h);
  }
  void record(int64_t value) {
    value = std::max<int64_t>(1, value);
    if (!hdr_record_value(h, value)) ++dropped;
  }
  int64_t pct(double p) const { return hdr_value_at_percentile(h, p); }
  nlohmann::json summary() const {
    return {
        {"count", h->total_count},
        {"dropped", dropped},
        {"min_ns", hdr_min(h)},
        {"max_ns", hdr_max(h)},
        {"mean_ns", hdr_mean(h)},
        {"stddev_ns", hdr_stddev(h)},
        {"p50_ns", pct(50.0)},
        {"p90_ns", pct(90.0)},
        {"p95_ns", pct(95.0)},
        {"p99_ns", pct(99.0)},
        {"p99_90_ns", pct(99.9)},
        {"p99_99_ns", pct(99.99)},
    };
  }
};

struct PredictorGuard {
  const Tl2cgenApi* api = nullptr;
  Tl2cgenApi::Handle handle = nullptr;

  PredictorGuard(const Tl2cgenApi& api_, Tl2cgenApi::Handle handle_)
      : api{&api_}, handle{handle_} {}
  PredictorGuard(const PredictorGuard&) = delete;
  PredictorGuard& operator=(const PredictorGuard&) = delete;
  ~PredictorGuard() {
    if (api != nullptr && handle != nullptr) {
      try {
        api->predictor_free(handle);
      } catch (...) {
      }
    }
  }
};

struct DMatrixGuard {
  const Tl2cgenApi* api = nullptr;
  std::vector<Tl2cgenApi::Handle>* handles = nullptr;

  DMatrixGuard(const Tl2cgenApi& api_, std::vector<Tl2cgenApi::Handle>& handles_)
      : api{&api_}, handles{&handles_} {}
  DMatrixGuard(const DMatrixGuard&) = delete;
  DMatrixGuard& operator=(const DMatrixGuard&) = delete;
  ~DMatrixGuard() {
    if (api == nullptr || handles == nullptr) return;
    for (void* h : *handles) {
      try {
        api->dmatrix_free(h);
      } catch (...) {
      }
    }
  }
};

void write_percentiles(const Histogram& hist, const std::string& path) {
  std::ofstream out{path};
  if (!out) throw std::runtime_error("failed to open percentile output: " + path);
  out << "percentile,value_ns\n";
  for (double p : {0.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9, 99.99, 100.0}) {
    out << p << "," << hist.pct(p) << "\n";
  }
}

void write_summary_csv(const std::string& path, const std::vector<nlohmann::json>& runs) {
  std::ofstream out{path};
  if (!out) throw std::runtime_error("failed to open summary CSV: " + path);
  out << "name,status,count,p50_ns,p90_ns,p99_ns,p99_9_ns,mean_ns,max_ns,reason\n";
  for (const auto& r : runs) {
    const auto lat = r.value("latency", nlohmann::json::object());
    out << r.value("name", "") << "," << r.value("status", "") << ","
        << lat.value("count", 0) << "," << lat.value("p50_ns", 0) << ","
        << lat.value("p90_ns", 0) << "," << lat.value("p99_ns", 0) << ","
        << lat.value("p99_90_ns", 0) << "," << lat.value("mean_ns", 0.0) << ","
        << lat.value("max_ns", 0) << "," << r.value("reason", "") << "\n";
  }
}

bool want_engine(const Args& args, std::string_view engine) {
  return std::find(args.engines.begin(), args.engines.end(), engine) !=
         args.engines.end();
}

nlohmann::json common_fields(const Args& args, size_t samples) {
  return {
      {"clock", "CLOCK_MONOTONIC_RAW"},
      {"histogram", "HdrHistogram_c"},
      {"warmup", args.warmup},
      {"iters", args.iters},
      {"samples", samples},
      {"features", args.features},
      {"ntrees", args.ntrees == 0 ? nullptr : nlohmann::json(args.ntrees)},
  };
}

nlohmann::json skip_run(const Args& args, size_t samples, const std::string& name,
                        const std::string& reason, nlohmann::json extra = {}) {
  nlohmann::json run = common_fields(args, samples);
  run["name"] = name;
  run["status"] = "skipped";
  run["reason"] = reason;
  for (auto& item : extra.items()) run[item.key()] = item.value();
  return run;
}

nlohmann::json run_tl2cgen(const Args& args, const std::vector<float>& features,
                           size_t samples) {
  Tl2cgenApi api{args.tl2cgen_runtime};
  auto predictor = api.predictor_load(args.tl2cgen_lib, args.threads);
  PredictorGuard predictor_guard{api, predictor};

  const int32_t num_feature = api.predictor_num_feature(predictor);
  if (num_feature != static_cast<int32_t>(args.features)) {
    throw std::runtime_error("TL2cgen model num_feature mismatch: model=" +
                             std::to_string(num_feature) + " args=" +
                             std::to_string(args.features));
  }

  std::vector<Tl2cgenApi::Handle> dmats;
  dmats.reserve(samples);
  for (size_t i = 0; i < samples; ++i) {
    dmats.push_back(api.dmatrix_create_from_mat(
        features.data() + i * args.features, 1, args.features));
  }
  DMatrixGuard dmats_guard{api, dmats};

  const auto shape = api.output_shape(predictor, dmats.front());
  const size_t output_size =
      std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>{});
  const auto leaf_type = api.predictor_leaf_output_type(predictor);
  const bool output_double = leaf_type == "float64";
  std::vector<float> output_f(output_double ? 0 : output_size);
  std::vector<double> output_d(output_double ? output_size : 0);

  auto predict = [&](size_t row) -> double {
    void* output = output_double ? static_cast<void*>(output_d.data())
                                 : static_cast<void*>(output_f.data());
    api.predict_batch(predictor, dmats[row], args.output == "margin", output);
    return output_double ? output_d.back() : output_f.back();
  };

  const std::string name =
      "treelite/tl2cgen-capi/threads:" + std::to_string(args.threads) +
      "/output:" + args.output;
  size_t row = 0;
  double last = 0.0;
  for (size_t i = 0; i < args.warmup; ++i) {
    last = predict(row % samples);
    ++row;
  }

  Histogram hist{args.highest_ns, args.sigfig};
  const int64_t begin = now_ns();
  for (size_t i = 0; i < args.iters; ++i) {
    const size_t r = row % samples;
    ++row;
    const int64_t t0 = now_ns();
    last = predict(r);
    const int64_t t1 = now_ns();
    hist.record(t1 - t0);
  }
  const int64_t elapsed = now_ns() - begin;

  const std::string pct_path = args.out_dir + "/" + safe_stem(name) + ".percentiles.csv";
  write_percentiles(hist, pct_path);

  const int32_t num_target = api.predictor_num_target(predictor);
  const auto num_class = api.predictor_num_class(predictor, num_target);
  nlohmann::json run = common_fields(args, samples);
  run.update({
      {"name", name},
      {"status", "ok"},
      {"engine", "treelite_tl2cgen_capi"},
      {"threads", args.threads},
      {"compiled_lib", args.tl2cgen_lib},
      {"tl2cgen_runtime", args.tl2cgen_runtime},
      {"leaf_output_type", leaf_type},
      {"num_target", num_target},
      {"num_class", num_class},
      {"output_shape", shape},
      {"last_result", last},
      {"elapsed_ns", elapsed},
      {"latency", hist.summary()},
      {"percentiles_csv", pct_path},
  });
  return run;
}

std::string nvforest_mode_name(const std::string& mode) {
  if (mode == "host_host") return "input:host/output:host";
  if (mode == "device_host") return "input:device/output:host";
  return "input:device/output:device_sync";
}

std::string nvforest_name(const std::string& mode, const std::string& layout,
                          size_t rows_per_block_iter, size_t align_bytes) {
  return "nvforest/gpu/" + nvforest_mode_name(mode) + "/layout:" + layout +
         "/rows_per_block_iter:" + std::to_string(rows_per_block_iter) +
         "/align_bytes:" + std::to_string(align_bytes);
}

#ifdef QLEAF_HAVE_NVFOREST
void cuda_check(cudaError_t code, const char* what) {
  if (code != cudaSuccess) {
    throw std::runtime_error(std::string{what} + ": " +
                             cudaGetErrorString(code));
  }
}

void treelite_check(int code, const char* what) {
  if (code < 0) {
    const char* err = TreeliteGetLastError();
    throw std::runtime_error(std::string{what} + ": " + (err ? err : "unknown"));
  }
}

struct TreeliteModelGuard {
  TreeliteModelHandle handle = nullptr;
  explicit TreeliteModelGuard(TreeliteModelHandle h) : handle{h} {}
  TreeliteModelGuard(const TreeliteModelGuard&) = delete;
  TreeliteModelGuard& operator=(const TreeliteModelGuard&) = delete;
  TreeliteModelGuard(TreeliteModelGuard&& other) noexcept
      : handle{std::exchange(other.handle, nullptr)} {}
  TreeliteModelGuard& operator=(TreeliteModelGuard&& other) noexcept {
    if (this != &other) {
      if (handle != nullptr) TreeliteFreeModel(handle);
      handle = std::exchange(other.handle, nullptr);
    }
    return *this;
  }
  ~TreeliteModelGuard() {
    if (handle != nullptr) {
      try {
        TreeliteFreeModel(handle);
      } catch (...) {
      }
    }
  }
};

struct CudaFloatBuffer {
  float* ptr = nullptr;
  CudaFloatBuffer() = default;
  explicit CudaFloatBuffer(size_t count) {
    if (count != 0) cuda_check(cudaMalloc(&ptr, count * sizeof(float)), "cudaMalloc");
  }
  CudaFloatBuffer(const CudaFloatBuffer&) = delete;
  CudaFloatBuffer& operator=(const CudaFloatBuffer&) = delete;
  ~CudaFloatBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }
};

nvforest::tree_layout parse_nvforest_layout(const std::string& text) {
  if (text == "depth_first") return nvforest::tree_layout::depth_first;
  if (text == "breadth_first") return nvforest::tree_layout::breadth_first;
  return nvforest::tree_layout::layered_children_together;
}

std::string prefix_xgb_json_if_needed(const Args& args) {
  if (args.ntrees == 0) return args.xgb_model;
  auto path = std::filesystem::path{args.xgb_model};
  if (path.extension() != ".json") {
    throw std::runtime_error(
        "nvforest --ntrees prefixing currently requires an XGBoost JSON model");
  }

  std::ifstream in{args.xgb_model};
  if (!in) throw std::runtime_error("failed to open XGBoost JSON: " + args.xgb_model);
  nlohmann::json model;
  in >> model;

  auto& gbm = model.at("learner").at("gradient_booster").at("model");
  auto& trees = gbm.at("trees");
  if (!trees.is_array()) throw std::runtime_error("XGBoost JSON trees is not an array");
  if (args.ntrees >= trees.size()) return args.xgb_model;

  trees.erase(trees.begin() + static_cast<std::ptrdiff_t>(args.ntrees),
              trees.end());
  if (auto it = gbm.find("tree_info"); it != gbm.end() && it->is_array()) {
    it->erase(it->begin() + static_cast<std::ptrdiff_t>(args.ntrees),
              it->end());
  }
  if (auto it = gbm.find("iteration_indptr"); it != gbm.end() && it->is_array()) {
    nlohmann::json trimmed = nlohmann::json::array();
    for (const auto& v : *it) {
      const auto idx = v.get<size_t>();
      if (idx <= args.ntrees) trimmed.push_back(idx);
    }
    if (trimmed.empty() || trimmed.front().get<size_t>() != 0) {
      trimmed.insert(trimmed.begin(), 0);
    }
    if (trimmed.back().get<size_t>() != args.ntrees) trimmed.push_back(args.ntrees);
    *it = std::move(trimmed);
  }
  gbm.at("gbtree_model_param").at("num_trees") = std::to_string(args.ntrees);

  const auto out_path =
      std::filesystem::path{args.out_dir} /
      ("nvforest_prefix_n" + std::to_string(args.ntrees) + ".xgb.json");
  std::ofstream out{out_path};
  if (!out) throw std::runtime_error("failed to write XGBoost prefix JSON: " +
                                     out_path.string());
  out << model << "\n";
  return out_path.string();
}

std::vector<double> parse_xgb_base_scores(const std::string& path) {
  std::ifstream in{path};
  if (!in) throw std::runtime_error("failed to open XGBoost JSON: " + path);
  nlohmann::json model;
  in >> model;

  auto parse_values = [](const nlohmann::json& value) {
    std::vector<double> scores;
    if (value.is_array()) {
      for (const auto& item : value) scores.push_back(item.get<double>());
      return scores;
    }
    if (value.is_number()) return std::vector<double>{value.get<double>()};
    if (value.is_string()) {
      const auto text = value.get<std::string>();
      if (!text.empty() && text.front() == '[') {
        const auto array = nlohmann::json::parse(text);
        if (!array.is_array()) {
          throw std::runtime_error("XGBoost base_score string is not an array");
        }
        for (const auto& item : array) scores.push_back(item.get<double>());
        return scores;
      }
      return std::vector<double>{std::stod(text)};
    }
    throw std::runtime_error("unsupported XGBoost base_score type");
  };

  auto scores = parse_values(
      model.at("learner").at("learner_model_param").at("base_score"));
  const auto objective =
      model.at("learner").at("objective").value("name", std::string{});
  if (objective == "binary:logistic" && scores.size() == 1 && scores[0] > 0.0 &&
      scores[0] < 1.0) {
    scores[0] = std::log(scores[0] / (1.0 - scores[0]));
  }
  return scores;
}

size_t json_size_t(const nlohmann::json& value, const char* name) {
  if (value.is_number_unsigned()) return value.get<size_t>();
  if (value.is_number_integer()) {
    const auto ret = value.get<int64_t>();
    if (ret < 0) throw std::runtime_error(std::string{name} + " must be nonnegative");
    return static_cast<size_t>(ret);
  }
  if (value.is_string()) return parse_size(value.get<std::string>(), name);
  throw std::runtime_error(std::string{name} + " must be an integer");
}

int32_t json_int32(const nlohmann::json& value, const char* name) {
  if (value.is_number_integer()) return value.get<int32_t>();
  if (value.is_string()) return parse_int(value.get<std::string>(), name);
  throw std::runtime_error(std::string{name} + " must be an integer");
}

std::unique_ptr<treelite::Model> build_treelite_xgb_json_model(
    const std::string& path) {
  std::ifstream in{path};
  if (!in) throw std::runtime_error("failed to open XGBoost JSON: " + path);
  nlohmann::json root;
  in >> root;

  const auto& learner = root.at("learner");
  const auto& learner_param = learner.at("learner_model_param");
  const auto& gbm = learner.at("gradient_booster").at("model");
  const auto& trees = gbm.at("trees");
  if (!trees.is_array() || trees.empty()) {
    throw std::runtime_error("XGBoost JSON has no trees");
  }

  auto model = treelite::Model::Create<float, float>();
  model->num_feature = json_int32(learner_param.at("num_feature"), "num_feature");
  const auto objective = learner.at("objective").value("name", std::string{});
  const int32_t xgb_num_class =
      json_int32(learner_param.at("num_class"), "num_class");
  const int32_t num_class = xgb_num_class > 0 ? xgb_num_class : 1;
  model->task_type = (objective == "binary:logistic" ||
                      objective == "binary:logitraw")
                         ? treelite::TaskType::kBinaryClf
                         : (num_class > 1 ? treelite::TaskType::kMultiClf
                                          : treelite::TaskType::kRegressor);
  model->average_tree_output = false;
  model->num_target = json_int32(learner_param.at("num_target"), "num_target");
  model->num_class = std::vector<int32_t>{num_class};
  model->leaf_vector_shape = std::vector<int32_t>{1, 1};
  model->postprocessor = objective == "binary:logistic" ? "sigmoid" : "identity";
  model->sigmoid_alpha = 1.0f;
  model->ratio_c = 1.0f;
  model->base_scores = parse_xgb_base_scores(path);
  model->attributes.clear();

  auto& preset = std::get<treelite::ModelPreset<float, float>>(model->variant_);
  preset.trees.reserve(trees.size());
  for (const auto& tree_json : trees) {
    const auto& left = tree_json.at("left_children");
    const auto& right = tree_json.at("right_children");
    const auto& split_indices = tree_json.at("split_indices");
    const auto& split_conditions = tree_json.at("split_conditions");
    const auto& default_left = tree_json.at("default_left");
    const auto& split_type = tree_json.at("split_type");
    const size_t num_nodes =
        json_size_t(tree_json.at("tree_param").at("num_nodes"), "num_nodes");
    if (left.size() != num_nodes || right.size() != num_nodes ||
        split_indices.size() != num_nodes ||
        split_conditions.size() != num_nodes ||
        default_left.size() != num_nodes || split_type.size() != num_nodes) {
      throw std::runtime_error("XGBoost tree arrays have inconsistent lengths");
    }
    if (!tree_json.value("categories", nlohmann::json::array()).empty()) {
      throw std::runtime_error("categorical XGBoost trees are not supported");
    }

    treelite::Tree<float, float> tree;
    tree.Init();
    for (size_t i = 0; i < num_nodes; ++i) tree.AllocNode();
    for (size_t i = 0; i < num_nodes; ++i) {
      const auto l = left.at(i).get<int32_t>();
      const auto r = right.at(i).get<int32_t>();
      if (l == -1 && r == -1) {
        tree.SetLeaf(static_cast<int>(i), split_conditions.at(i).get<float>());
        continue;
      }
      if (l < 0 || r < 0 || static_cast<size_t>(l) >= num_nodes ||
          static_cast<size_t>(r) >= num_nodes) {
        throw std::runtime_error("XGBoost tree contains invalid child index");
      }
      if (split_type.at(i).get<int32_t>() != 0) {
        throw std::runtime_error("categorical XGBoost split is not supported");
      }
      tree.SetChildren(static_cast<int>(i), l, r);
      tree.SetNumericalTest(static_cast<int>(i),
                            split_indices.at(i).get<int32_t>(),
                            split_conditions.at(i).get<float>(),
                            default_left.at(i).get<int32_t>() != 0,
                            treelite::Operator::kLT);
    }
    preset.trees.push_back(std::move(tree));
  }

  std::vector<int32_t> target_id(preset.trees.size(), 0);
  std::vector<int32_t> class_id(preset.trees.size(), 0);
  if (const auto it = gbm.find("tree_info"); it != gbm.end() && it->is_array() &&
      it->size() == class_id.size()) {
    for (size_t i = 0; i < it->size(); ++i) class_id[i] = it->at(i).get<int32_t>();
  }
  model->target_id = target_id;
  model->class_id = class_id;
  return model;
}

TreeliteModelGuard load_treelite_xgb_model(const std::string& path) {
  TreeliteModelHandle h = nullptr;
  const auto ext = std::filesystem::path{path}.extension().string();
  if (ext == ".json") {
    treelite_check(TreeliteLoadXGBoostModelJSON(path.c_str(), "{}", &h),
                   "TreeliteLoadXGBoostModelJSON");
  } else if (ext == ".ubj") {
    treelite_check(TreeliteLoadXGBoostModelUBJSON(path.c_str(), "{}", &h),
                   "TreeliteLoadXGBoostModelUBJSON");
  } else {
    treelite_check(TreeliteLoadXGBoostModelLegacyBinary(path.c_str(), "{}", &h),
                   "TreeliteLoadXGBoostModelLegacyBinary");
  }
  return TreeliteModelGuard{h};
}

void set_treelite_base_scores(TreeliteModelHandle handle,
                              std::vector<double> base_scores) {
  if (base_scores.empty()) return;
  char format[] = "=d";
  TreelitePyBufferFrame frame{};
  frame.buf = base_scores.data();
  frame.format = format;
  frame.itemsize = sizeof(double);
  frame.nitem = base_scores.size();
  treelite_check(TreeliteSetHeaderField(handle, "base_scores", frame),
                 "TreeliteSetHeaderField(base_scores)");
}

nlohmann::json measure_nvforest_one(
    const Args& args, const std::vector<float>& features, size_t samples,
    const std::string& model_path, nvforest::forest_model& model,
    raft_proto::handle_t& handle, CudaFloatBuffer& device_features,
    CudaFloatBuffer& device_output, std::vector<float>& host_output,
    const std::string& layout_name, size_t rows_per_block_iter,
    size_t align_bytes, const std::string& mode) {
  const auto output_size = static_cast<size_t>(model.num_outputs());
  const bool input_device = mode != "host_host";
  const bool output_device = mode == "device_device_sync";
  const auto in_mem = input_device ? raft_proto::device_type::gpu
                                   : raft_proto::device_type::cpu;
  const auto out_mem = output_device ? raft_proto::device_type::gpu
                                     : raft_proto::device_type::cpu;
  const auto rpb = std::optional<nvforest::index_type>{
      static_cast<nvforest::index_type>(rows_per_block_iter)};

  auto predict = [&](size_t row, bool capture_device_output) -> double {
    float* in = input_device ? device_features.ptr + row * args.features
                             : const_cast<float*>(features.data() + row * args.features);
    float* out = output_device ? device_output.ptr : host_output.data();
    model.predict<float>(handle, out, in, 1, out_mem, in_mem,
                         nvforest::infer_kind::default_kind, rpb);
    handle.synchronize();
    if (output_device) {
      if (capture_device_output) {
        cuda_check(cudaMemcpy(host_output.data(), device_output.ptr,
                              output_size * sizeof(float), cudaMemcpyDeviceToHost),
                   "cudaMemcpy output D2H");
      }
      return capture_device_output ? host_output.back() : 0.0;
    }
    return host_output.back();
  };

  const std::string name =
      nvforest_name(mode, layout_name, rows_per_block_iter, align_bytes);
  size_t row = 0;
  double last = 0.0;
  for (size_t i = 0; i < args.warmup; ++i) {
    last = predict(row % samples, false);
    ++row;
  }

  Histogram hist{args.highest_ns, args.sigfig};
  const int64_t begin = now_ns();
  for (size_t i = 0; i < args.iters; ++i) {
    const size_t r = row % samples;
    ++row;
    const int64_t t0 = now_ns();
    last = predict(r, false);
    const int64_t t1 = now_ns();
    hist.record(t1 - t0);
  }
  const int64_t elapsed = now_ns() - begin;
  if (output_device) last = predict((row + samples - 1) % samples, true);

  const std::string pct_path = args.out_dir + "/" + safe_stem(name) + ".percentiles.csv";
  write_percentiles(hist, pct_path);

  nlohmann::json run = common_fields(args, samples);
  run.update({
      {"name", name},
      {"status", "ok"},
      {"engine", "nvforest"},
      {"device", "gpu"},
      {"model_file", model_path},
      {"layout", layout_name},
      {"rows_per_block_iter", rows_per_block_iter},
      {"align_bytes", align_bytes},
      {"mode", mode},
      {"input_memory", input_device ? "device" : "host"},
      {"output_memory", output_device ? "device" : "host"},
      {"output_sync", true},
      {"precision", "float32"},
      {"num_outputs", output_size},
      {"num_trees", model.num_trees()},
      {"is_double_precision", model.is_double_precision()},
      {"row_postprocessing", static_cast<int>(model.row_postprocessing())},
      {"elem_postprocessing", static_cast<int>(model.elem_postprocessing())},
      {"last_result", last},
      {"elapsed_ns", elapsed},
      {"latency", hist.summary()},
      {"percentiles_csv", pct_path},
  });
  return run;
}

std::vector<nlohmann::json> run_nvforest(const Args& args,
                                         const std::vector<float>& features,
                                         size_t samples) {
  const auto model_path = prefix_xgb_json_if_needed(args);
  std::vector<nlohmann::json> runs;
  auto skip_all = [&](const std::string& reason) {
    for (const auto& layout : args.nvforest_layouts) {
      for (size_t align_bytes : args.nvforest_align_bytes) {
        for (size_t rows_per_block_iter : args.nvforest_rows_per_block_iter) {
          for (const auto& mode : args.nvforest_modes) {
            runs.push_back(skip_run(
                args, samples,
                nvforest_name(mode, layout, rows_per_block_iter, align_bytes),
                reason,
                {{"engine", "nvforest"},
                 {"layout", layout},
                 {"rows_per_block_iter", rows_per_block_iter},
                 {"align_bytes", align_bytes},
                 {"mode", mode}}));
          }
        }
      }
    }
  };
  int device = 0;
  std::unique_ptr<treelite::Model> tl_model;
  std::optional<raft::handle_t> raft_handle;
  std::optional<raft_proto::handle_t> handle;
  std::optional<CudaFloatBuffer> device_features;
  try {
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    tl_model = build_treelite_xgb_json_model(model_path);
    raft_handle.emplace();
    handle.emplace(*raft_handle);
    device_features.emplace(samples * args.features);
    cuda_check(cudaMemcpy(device_features->ptr, features.data(),
                          features.size() * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy features H2D");
  } catch (const std::exception& e) {
    skip_all(std::string{"failed to initialize nvForest/CUDA: "} + e.what());
    return runs;
  }
  for (const auto& layout : args.nvforest_layouts) {
    for (size_t align_bytes : args.nvforest_align_bytes) {
      try {
        auto model = nvforest::import_from_treelite_model(
            *tl_model, parse_nvforest_layout(layout),
            static_cast<nvforest::index_type>(align_bytes),
            std::optional<bool>{false}, raft_proto::device_type::gpu, device,
            handle->get_next_usable_stream());
        handle->synchronize();
        const auto num_features = static_cast<size_t>(model.num_features());
        if (num_features != args.features) {
          throw std::runtime_error("nvForest model num_feature mismatch: model=" +
                                   std::to_string(num_features) + " args=" +
                                   std::to_string(args.features));
        }
        std::vector<float> host_output(static_cast<size_t>(model.num_outputs()), 0.0f);
        CudaFloatBuffer device_output(static_cast<size_t>(model.num_outputs()));
        for (size_t rows_per_block_iter : args.nvforest_rows_per_block_iter) {
          for (const auto& mode : args.nvforest_modes) {
            try {
              runs.push_back(measure_nvforest_one(
                  args, features, samples, model_path, model, *handle,
                  *device_features, device_output, host_output, layout,
                  rows_per_block_iter, align_bytes, mode));
            } catch (const std::exception& e) {
              const auto name =
                  nvforest_name(mode, layout, rows_per_block_iter, align_bytes);
              runs.push_back(skip_run(
                  args, samples, name,
                  std::string{"failed to run nvForest: "} + e.what(),
                  {{"engine", "nvforest"},
                   {"layout", layout},
                   {"rows_per_block_iter", rows_per_block_iter},
                   {"align_bytes", align_bytes},
                   {"mode", mode}}));
            }
          }
        }
      } catch (const std::exception& e) {
        for (size_t rows_per_block_iter : args.nvforest_rows_per_block_iter) {
          for (const auto& mode : args.nvforest_modes) {
            const auto name =
                nvforest_name(mode, layout, rows_per_block_iter, align_bytes);
            runs.push_back(skip_run(
                args, samples, name,
                std::string{"failed to prepare nvForest model: "} + e.what(),
                {{"engine", "nvforest"},
                 {"layout", layout},
                 {"rows_per_block_iter", rows_per_block_iter},
                 {"align_bytes", align_bytes},
                 {"mode", mode}}));
          }
        }
      }
    }
  }
  return runs;
}
#else
std::vector<nlohmann::json> run_nvforest(const Args& args,
                                         const std::vector<float>&,
                                         size_t samples) {
  std::vector<nlohmann::json> runs;
  const std::string reason =
      "external_latency_cpp was built without nvForest/Treelite development "
      "headers and libraries";
  for (const auto& layout : args.nvforest_layouts) {
    for (size_t rows_per_block_iter : args.nvforest_rows_per_block_iter) {
      for (size_t align_bytes : args.nvforest_align_bytes) {
        for (const auto& mode : args.nvforest_modes) {
          runs.push_back(skip_run(
              args, samples,
              nvforest_name(mode, layout, rows_per_block_iter, align_bytes),
              reason,
              {{"engine", "nvforest"},
               {"layout", layout},
               {"rows_per_block_iter", rows_per_block_iter},
               {"align_bytes", align_bytes},
               {"mode", mode}}));
        }
      }
    }
  }
  return runs;
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    mkdir_p(args.out_dir);

    auto features = load_features_csv(args);
    const size_t samples = features.size() / args.features;

    std::vector<nlohmann::json> runs;
    if (want_engine(args, "treelite")) {
      runs.push_back(run_tl2cgen(args, features, samples));
    }
    if (want_engine(args, "nvforest")) {
      auto nvforest_runs = run_nvforest(args, features, samples);
      runs.insert(runs.end(), nvforest_runs.begin(), nvforest_runs.end());
    }
    nlohmann::json root = {
        {"meta",
         {{"clock", "CLOCK_MONOTONIC_RAW"},
          {"histogram", "HdrHistogram_c"},
          {"engines", args.engines},
          {"features", args.features},
          {"samples", samples},
          {"warmup", args.warmup},
          {"iters", args.iters},
          {"ntrees", args.ntrees == 0 ? nullptr : nlohmann::json(args.ntrees)}}},
        {"runs", runs},
    };

    std::ofstream out{args.out_dir + "/summary.json"};
    out << std::setw(2) << root << "\n";
    write_summary_csv(args.out_dir + "/summary.csv", runs);
    std::cerr << "[done] wrote " << args.out_dir << "/summary.json\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
