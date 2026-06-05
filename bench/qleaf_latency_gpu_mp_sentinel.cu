#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <template <typename> typename TNodeBuffer>
void run_mp_sentinel_buffer(const GpuLatencyRequest &req, GpuInput &input,
                            std::vector<GpuLatencyRun> &out) {
  run_mp_sentinel_path<kCompiledBT, false, 1, TNodeBuffer>(req, input, out);
  for (int tpt : req.tpts) {
    run_mp_sentinel_tpt<kCompiledBT, TNodeBuffer>(req, input, out, tpt);
  }
}

}  // namespace

void run_mp_sentinel_compact(const GpuLatencyRequest &req, GpuInput &input,
                             std::vector<GpuLatencyRun> &out) {
  run_mp_sentinel_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_mp_sentinel_tree(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  run_mp_sentinel_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu

