#include "qleaf_latency_gpu_common.cuh"

#ifndef QLEAF_GPU_HAS_SP
#define QLEAF_GPU_HAS_SP 0
#endif
#ifndef QLEAF_GPU_HAS_SP_MAP
#define QLEAF_GPU_HAS_SP_MAP 0
#endif
#ifndef QLEAF_GPU_HAS_MP_SENTINEL
#define QLEAF_GPU_HAS_MP_SENTINEL 0
#endif
#ifndef QLEAF_GPU_HAS_BITMASK_TPT
#define QLEAF_GPU_HAS_BITMASK_TPT 0
#endif
#ifndef QLEAF_GPU_HAS_BITMASK_CACHE
#define QLEAF_GPU_HAS_BITMASK_CACHE 0
#endif
#ifndef QLEAF_GPU_HAS_ONESHOT
#define QLEAF_GPU_HAS_ONESHOT 0
#endif
#ifndef QLEAF_GPU_HAS_NAIVE
#define QLEAF_GPU_HAS_NAIVE 0
#endif
#ifndef QLEAF_GPU_HAS_ABLATION
#define QLEAF_GPU_HAS_ABLATION 0
#endif

namespace qleaf_latency_gpu {

#if QLEAF_GPU_HAS_SP
void run_sp_compact(const GpuLatencyRequest &, GpuInput &,
                    std::vector<GpuLatencyRun> &);
void run_sp_tree(const GpuLatencyRequest &, GpuInput &,
                 std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_SP_MAP
void run_sp_map_compact(const GpuLatencyRequest &, GpuInput &,
                        std::vector<GpuLatencyRun> &);
void run_sp_map_tree(const GpuLatencyRequest &, GpuInput &,
                     std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_MP_SENTINEL
void run_mp_sentinel_compact(const GpuLatencyRequest &, GpuInput &,
                             std::vector<GpuLatencyRun> &);
void run_mp_sentinel_tree(const GpuLatencyRequest &, GpuInput &,
                          std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_BITMASK_TPT
void run_bitmask_tpt_compact(const GpuLatencyRequest &, GpuInput &,
                             std::vector<GpuLatencyRun> &);
void run_bitmask_tpt_tree(const GpuLatencyRequest &, GpuInput &,
                          std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_BITMASK_CACHE
void run_bitmask_cache_compact(const GpuLatencyRequest &, GpuInput &,
                               std::vector<GpuLatencyRun> &);
void run_bitmask_cache_tree(const GpuLatencyRequest &, GpuInput &,
                            std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_ONESHOT
void run_oneshot_compact(const GpuLatencyRequest &, GpuInput &,
                         std::vector<GpuLatencyRun> &);
void run_oneshot_tree(const GpuLatencyRequest &, GpuInput &,
                      std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_NAIVE
void run_naive_compact(const GpuLatencyRequest &, GpuInput &,
                       std::vector<GpuLatencyRun> &);
void run_naive_tree(const GpuLatencyRequest &, GpuInput &,
                    std::vector<GpuLatencyRun> &);
#endif
#if QLEAF_GPU_HAS_ABLATION
void run_ablation_compact(const GpuLatencyRequest &, GpuInput &,
                          std::vector<GpuLatencyRun> &);
void run_ablation_tree(const GpuLatencyRequest &, GpuInput &,
                       std::vector<GpuLatencyRun> &);
#endif

namespace {

void suite_unavailable(const GpuLatencyRequest &req,
                       std::vector<GpuLatencyRun> &out) {
  out.push_back(skip_run("qleaf/gpu/" + req.suite,
                         "GPU suite was not compiled into this binary"));
}

template <typename RunCompact, typename RunTree>
void run_for_buffer(const GpuLatencyRequest &req, GpuInput &input,
                    std::vector<GpuLatencyRun> &out, RunCompact run_compact,
                    RunTree run_tree) {
  if (req.buffer == "compact") {
    run_compact(req, input, out);
  } else if (req.buffer == "tree") {
    run_tree(req, input, out);
  } else {
    out.push_back(skip_run("qleaf/gpu/" + req.buffer, "unknown node buffer"));
  }
}

bool run_best(const GpuLatencyRequest &req, GpuInput &input,
              std::vector<GpuLatencyRun> &out) {
  bool ran = false;
#if QLEAF_GPU_HAS_SP
  run_for_buffer(req, input, out, run_sp_compact, run_sp_tree);
  ran = true;
#endif
#if QLEAF_GPU_HAS_BITMASK_TPT
  run_for_buffer(req, input, out, run_bitmask_tpt_compact,
                 run_bitmask_tpt_tree);
  ran = true;
#endif
  return ran;
}

void run_compiled_suite(const GpuLatencyRequest &req, GpuInput &input,
                        std::vector<GpuLatencyRun> &out) {
  if (req.suite == "best") {
    if (!run_best(req, input, out)) suite_unavailable(req, out);
    return;
  }
  if (req.suite == "all") {
    out.push_back(skip_run("qleaf/gpu/exhaustive/bt:" +
                               std::to_string(kCompiledBT),
                           "qleaf_latency is suite-pruned at compile time; "
                           "use worker_bench or rebuild with more suites"));
    return;
  }
  if (req.suite == "sp") {
#if QLEAF_GPU_HAS_SP
    run_for_buffer(req, input, out, run_sp_compact, run_sp_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "sp-map") {
#if QLEAF_GPU_HAS_SP_MAP
    run_for_buffer(req, input, out, run_sp_map_compact, run_sp_map_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "mp-sentinel") {
#if QLEAF_GPU_HAS_MP_SENTINEL
    run_for_buffer(req, input, out, run_mp_sentinel_compact,
                   run_mp_sentinel_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "bitmask-tpt") {
#if QLEAF_GPU_HAS_BITMASK_TPT
    run_for_buffer(req, input, out, run_bitmask_tpt_compact,
                   run_bitmask_tpt_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "bitmask-cache") {
#if QLEAF_GPU_HAS_BITMASK_CACHE
    run_for_buffer(req, input, out, run_bitmask_cache_compact,
                   run_bitmask_cache_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "oneshot") {
#if QLEAF_GPU_HAS_ONESHOT
    run_for_buffer(req, input, out, run_oneshot_compact, run_oneshot_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "naive") {
#if QLEAF_GPU_HAS_NAIVE
    run_for_buffer(req, input, out, run_naive_compact, run_naive_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  if (req.suite == "ablation") {
#if QLEAF_GPU_HAS_ABLATION
    run_for_buffer(req, input, out, run_ablation_compact, run_ablation_tree);
#else
    suite_unavailable(req, out);
#endif
    return;
  }
  out.push_back(skip_run("qleaf/gpu/" + req.suite, "unknown GPU suite"));
}

}  // namespace

}  // namespace qleaf_latency_gpu

std::vector<GpuLatencyRun> run_qleaf_gpu_latency(const GpuLatencyRequest &req) {
  using namespace qleaf_latency_gpu;
  std::vector<GpuLatencyRun> out;
  if (req.forest == nullptr || req.features == nullptr ||
      req.features->empty()) {
    out.push_back(skip_run("qleaf/gpu", "empty GPU latency request"));
    return out;
  }
  GpuInput input{.forest = *req.forest, .features = *req.features};
  try {
    input.cuda_enabled = prepare_cuda_features(input);
  } catch (const std::exception &e) {
    input.cuda_enabled = false;
    out.push_back(skip_run("qleaf/gpu", e.what()));
  }
  run_compiled_suite(req, input, out);
  return out;
}
