#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <template <typename> typename TNodeBuffer>
void run_oneshot_buffer(const GpuLatencyRequest &req, GpuInput &input,
                        std::vector<GpuLatencyRun> &out) {
  run_oneshot_bitmask<kCompiledBT, 1, true, true, 1, TNodeBuffer>(req, input,
                                                                  out);
  run_oneshot_bitmask<kCompiledBT, 4, true, true, 4, TNodeBuffer>(req, input,
                                                                  out);
  run_oneshot_bitmask<kCompiledBT, 4, true, false, 4, TNodeBuffer>(req, input,
                                                                   out);
  run_oneshot_bitmask<kCompiledBT, 4, false, false, 4, TNodeBuffer>(req, input,
                                                                    out);
}

}  // namespace

void run_oneshot_compact(const GpuLatencyRequest &req, GpuInput &input,
                         std::vector<GpuLatencyRun> &out) {
  run_oneshot_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_oneshot_tree(const GpuLatencyRequest &req, GpuInput &input,
                      std::vector<GpuLatencyRun> &out) {
  run_oneshot_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu

