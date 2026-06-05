#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <template <typename> typename TNodeBuffer>
void run_naive_buffer(const GpuLatencyRequest &req, GpuInput &input,
                      std::vector<GpuLatencyRun> &out) {
  run_naive_persistent<kCompiledBT, false, TNodeBuffer>(req, input, out);
  run_naive_persistent<kCompiledBT, true, TNodeBuffer>(req, input, out);
}

}  // namespace

void run_naive_compact(const GpuLatencyRequest &req, GpuInput &input,
                       std::vector<GpuLatencyRun> &out) {
  run_naive_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_naive_tree(const GpuLatencyRequest &req, GpuInput &input,
                    std::vector<GpuLatencyRun> &out) {
  run_naive_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu

