#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <template <typename> typename TNodeBuffer>
void run_bitmask_tpt_buffer(const GpuLatencyRequest &req, GpuInput &input,
                            std::vector<GpuLatencyRun> &out) {
  for (int tpt : req.tpts) {
    run_tpt_value<kCompiledBT, TNodeBuffer>(req, input, out, tpt);
  }
}

}  // namespace

void run_bitmask_tpt_compact(const GpuLatencyRequest &req, GpuInput &input,
                             std::vector<GpuLatencyRun> &out) {
  run_bitmask_tpt_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_bitmask_tpt_tree(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  run_bitmask_tpt_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu

