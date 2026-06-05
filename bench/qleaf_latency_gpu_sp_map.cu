#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <template <typename> typename TNodeBuffer>
void run_sp_map_buffer(const GpuLatencyRequest &req, GpuInput &input,
                       std::vector<GpuLatencyRun> &out) {
  run_sp_path<kCompiledBT, true, false, true, TNodeBuffer>(req, input, out);
  run_sp_path<kCompiledBT, false, false, true, TNodeBuffer>(req, input, out);
  run_sp_path<kCompiledBT, true, true, true, TNodeBuffer>(req, input, out);
  run_sp_path<kCompiledBT, false, true, true, TNodeBuffer>(req, input, out);
}

}  // namespace

void run_sp_map_compact(const GpuLatencyRequest &req, GpuInput &input,
                        std::vector<GpuLatencyRun> &out) {
  run_sp_map_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_sp_map_tree(const GpuLatencyRequest &req, GpuInput &input,
                     std::vector<GpuLatencyRun> &out) {
  run_sp_map_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu

