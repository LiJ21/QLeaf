#include <hdr/hdr_histogram.h>
#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>

#include <nlohmann/json.hpp>

namespace {

struct Args {
  std::vector<std::string> engines{"treelite"};
  std::string features_csv;
  std::string xgb_model;
  std::string out_dir = "bench/results/latency_external_cpp";
  std::string tl2cgen_lib;
  std::string tl2cgen_runtime =
      ".venv/lib/python3.12/site-packages/tl2cgen/lib/libtl2cgen.so";
  std::string xgboost_runtime =
      ".venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.so";
  std::string output = "margin";
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
      << "  --engine xgboost|treelite|all\n"
      << "  --engines a,b               comma-separated engine list\n"
      << "  --samples N                 rows to load (default 1024)\n"
      << "  --warmup N                  warmup requests (default 1000)\n"
      << "  --iters N                   measured requests (default 10000)\n"
      << "  --threads N                 XGBoost / TL2cgen predictor threads (default 1)\n"
      << "  --ntrees N                  metadata only; lib should already match\n"
      << "  --output margin|value       pred_margin flag (default margin)\n"
      << "  --tl2cgen-runtime path      libtl2cgen.so path\n"
      << "  --tl2cgen-lib path          TL2cgen compiled model for treelite engine\n"
      << "  --xgboost-runtime path      libxgboost.so path\n"
      << "  --xgb-model path            XGBoost JSON model\n"
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
        args.engines = {"xgboost", "treelite"};
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
    else if (arg == "--xgboost-runtime") args.xgboost_runtime = need_value(arg);
    else if (arg == "--out-dir") args.out_dir = need_value(arg);
    else if (arg == "--highest-ns") args.highest_ns = static_cast<int64_t>(parse_size(need_value(arg), arg));
    else if (arg == "--sigfig") args.sigfig = parse_int(need_value(arg), arg);
    else usage("unknown option " + std::string{arg});
  }
  if (args.features == 0) usage("--features must be > 0");
  if (args.features_csv.empty()) usage("--features-csv is required");
  if (args.samples == 0) usage("--samples must be > 0");
  if (args.iters == 0) usage("--iters must be > 0");
  if (args.threads <= 0) usage("--threads must be positive");
  if (args.output != "margin" && args.output != "value") {
    usage("--output must be margin or value");
  }
  for (const auto& engine : args.engines) {
    if (engine != "xgboost" && engine != "treelite") {
      usage("--engine/--engines supports xgboost and treelite");
    }
  }
  const auto wants = [&](std::string_view engine) {
    return std::find(args.engines.begin(), args.engines.end(), engine) !=
           args.engines.end();
  };
  if (wants("treelite") && args.tl2cgen_lib.empty()) {
    usage("--tl2cgen-lib is required for --engine treelite");
  }
  if (wants("xgboost") && args.xgb_model.empty()) {
    usage("--xgb-model is required for --engine xgboost");
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

class XGBoostApi {
 public:
  using Handle = void*;
  using BstULong = uint64_t;

  explicit XGBoostApi(const std::string& path)
      : lib_{dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)} {
    if (!lib_) throw std::runtime_error(std::string{"dlopen failed: "} + dlerror());
    load("XGBGetLastError", get_last_error_);
    load("XGBoostVersion", version_);
    load("XGBoosterCreate", booster_create_);
    load("XGBoosterFree", booster_free_);
    load("XGBoosterLoadModel", booster_load_model_);
    load("XGBoosterSetParam", booster_set_param_);
    load("XGBoosterGetNumFeature", booster_get_num_feature_);
    load("XGBoosterBoostedRounds", booster_boosted_rounds_);
    load("XGBoosterPredictFromDense", booster_predict_from_dense_);
  }

  const char* last_error() const {
    const char* msg = get_last_error_ ? get_last_error_() : nullptr;
    return msg ? msg : "unknown XGBoost error";
  }

  void check(int code, const char* what) const {
    if (code != 0) {
      throw std::runtime_error(std::string{what} + ": " + last_error());
    }
  }

  std::string version_string() const {
    int major = 0;
    int minor = 0;
    int patch = 0;
    version_(&major, &minor, &patch);
    return std::to_string(major) + "." + std::to_string(minor) + "." +
           std::to_string(patch);
  }

  Handle booster_create() const {
    Handle h = nullptr;
    check(booster_create_(nullptr, 0, &h), "XGBoosterCreate");
    return h;
  }

  void booster_free(Handle h) const { check(booster_free_(h), "XGBoosterFree"); }

  void booster_load_model(Handle h, const std::string& path) const {
    check(booster_load_model_(h, path.c_str()), "XGBoosterLoadModel");
  }

  void booster_set_param(Handle h, const std::string& key,
                         const std::string& value) const {
    check(booster_set_param_(h, key.c_str(), value.c_str()), "XGBoosterSetParam");
  }

  BstULong booster_num_feature(Handle h) const {
    BstULong out = 0;
    check(booster_get_num_feature_(h, &out), "XGBoosterGetNumFeature");
    return out;
  }

  int booster_boosted_rounds(Handle h) const {
    int out = 0;
    check(booster_boosted_rounds_(h, &out), "XGBoosterBoostedRounds");
    return out;
  }

  double predict_from_dense(Handle booster, const std::string& array_interface,
                            const std::string& config) const {
    const BstULong* shape = nullptr;
    BstULong ndim = 0;
    const float* preds = nullptr;
    check(booster_predict_from_dense_(booster, array_interface.c_str(),
                                      config.c_str(), nullptr, &shape, &ndim,
                                      &preds),
          "XGBoosterPredictFromDense");
    size_t output_size = 1;
    for (BstULong i = 0; i < ndim; ++i) {
      output_size *= static_cast<size_t>(shape[i]);
    }
    if (preds == nullptr || output_size == 0) {
      throw std::runtime_error("XGBoosterPredictFromDense returned no output");
    }
    return preds[output_size - 1];
  }

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
  void (*version_)(int*, int*, int*) = nullptr;
  int (*booster_create_)(Handle*, BstULong, Handle*) = nullptr;
  int (*booster_free_)(Handle) = nullptr;
  int (*booster_load_model_)(Handle, const char*) = nullptr;
  int (*booster_set_param_)(Handle, const char*, const char*) = nullptr;
  int (*booster_get_num_feature_)(Handle, BstULong*) = nullptr;
  int (*booster_boosted_rounds_)(Handle, int*) = nullptr;
  int (*booster_predict_from_dense_)(Handle, const char*, const char*, Handle,
                                     const BstULong**, BstULong*,
                                     const float**) = nullptr;
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

struct XGBoostBoosterGuard {
  const XGBoostApi* api = nullptr;
  XGBoostApi::Handle handle = nullptr;

  XGBoostBoosterGuard(const XGBoostApi& api_, XGBoostApi::Handle handle_)
      : api{&api_}, handle{handle_} {}
  XGBoostBoosterGuard(const XGBoostBoosterGuard&) = delete;
  XGBoostBoosterGuard& operator=(const XGBoostBoosterGuard&) = delete;
  ~XGBoostBoosterGuard() {
    if (api != nullptr && handle != nullptr) {
      try {
        api->booster_free(handle);
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

nlohmann::json run_xgboost_capi(const Args& args,
                                const std::vector<float>& features,
                                size_t samples) {
  XGBoostApi api{args.xgboost_runtime};
  auto booster = api.booster_create();
  XGBoostBoosterGuard booster_guard{api, booster};
  api.booster_load_model(booster, args.xgb_model);
  api.booster_set_param(booster, "verbosity", "0");
  api.booster_set_param(booster, "device", "cpu");
  api.booster_set_param(booster, "nthread", std::to_string(args.threads));

  const auto num_feature = api.booster_num_feature(booster);
  if (num_feature != static_cast<XGBoostApi::BstULong>(args.features)) {
    throw std::runtime_error("XGBoost model num_feature mismatch: model=" +
                             std::to_string(num_feature) + " args=" +
                             std::to_string(args.features));
  }

  const int boosted_rounds = api.booster_boosted_rounds(booster);
  const size_t iteration_end = args.ntrees == 0 ? 0 : args.ntrees;
  const int predict_type = args.output == "margin" ? 1 : 0;
  const std::string predict_config =
      "{\"type\":" + std::to_string(predict_type) +
      ",\"training\":false"
      ",\"iteration_begin\":0"
      ",\"iteration_end\":" + std::to_string(iteration_end) +
      ",\"missing\":NaN"
      ",\"strict_shape\":false"
      ",\"cache_id\":0}";

  std::vector<std::string> row_interfaces;
  row_interfaces.reserve(samples);
  for (size_t row = 0; row < samples; ++row) {
    const auto ptr = reinterpret_cast<std::uintptr_t>(
        features.data() + row * args.features);
    row_interfaces.push_back(
        "{\"data\":[" + std::to_string(ptr) +
        ",false],\"shape\":[1," + std::to_string(args.features) +
        "],\"typestr\":\"<f4\",\"version\":3}");
  }

  auto predict = [&](size_t row) -> double {
    return api.predict_from_dense(booster, row_interfaces[row], predict_config);
  };

  const std::string name =
      "xgboost/capi/threads:" + std::to_string(args.threads) +
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

  const std::string pct_path =
      args.out_dir + "/" + safe_stem(name) + ".percentiles.csv";
  write_percentiles(hist, pct_path);

  nlohmann::json run = common_fields(args, samples);
  run.update({
      {"name", name},
      {"status", "ok"},
      {"engine", "xgboost_capi"},
      {"threads", args.threads},
      {"model_file", args.xgb_model},
      {"xgboost_runtime", args.xgboost_runtime},
      {"xgboost_version", api.version_string()},
      {"predict_api", "XGBoosterPredictFromDense"},
      {"output", args.output},
      {"predict_type", predict_type},
      {"num_feature", num_feature},
      {"boosted_rounds", boosted_rounds},
      {"iteration_begin", 0},
      {"iteration_end", iteration_end},
      {"last_result", last},
      {"elapsed_ns", elapsed},
      {"latency", hist.summary()},
      {"percentiles_csv", pct_path},
  });
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
    if (output_double) {
      std::fill(output_d.begin(), output_d.end(), 0.0);
    } else {
      std::fill(output_f.begin(), output_f.end(), 0.0f);
    }
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

}  // namespace

int main(int argc, char** argv) {
  try {
    Args args = parse_args(argc, argv);
    mkdir_p(args.out_dir);

    auto features = load_features_csv(args);
    const size_t samples = features.size() / args.features;

    std::vector<nlohmann::json> runs;
    if (want_engine(args, "xgboost")) {
      runs.push_back(run_xgboost_capi(args, features, samples));
    }
    if (want_engine(args, "treelite")) {
      runs.push_back(run_tl2cgen(args, features, samples));
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
          {"threads", args.threads},
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
