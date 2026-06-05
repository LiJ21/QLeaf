#include "qleaf_latency_common.h"

#include <utility>

std::vector<GpuLatencyRun> run_qleaf_gpu_latency(const GpuLatencyRequest &req) {
  GpuLatencyRun run;
  run.name = "qleaf/gpu/" + req.suite;
  run.status = "skipped";
  run.reason = "QLeaf was built without CUDA";
  return {std::move(run)};
}
