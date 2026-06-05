#include "qleaf_latency_gpu_common.cuh"

namespace qleaf_latency_gpu {
namespace {

template <int BT, bool Mapped, bool Bitmask,
          template <typename> typename TNodeBuffer>
void run_oneshot_ablation(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  using P =
      qleaf::CudaTraverseOneShot<BT, true, Bitmask, false, Mapped, 1>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <int BT, bool Mapped, bool CrossReduce, bool Bitmask,
          template <typename> typename TNodeBuffer>
void run_persistent_ablation(const GpuLatencyRequest &req, GpuInput &input,
                             std::vector<GpuLatencyRun> &out) {
  using P =
      qleaf::CudaTraversePersistent<BT, true, Bitmask, false, Mapped,
                                    CrossReduce>;
  run_gpu_policy<P, TNodeBuffer>(req, input, out);
}

template <template <typename> typename TNodeBuffer>
void run_ablation_buffer(const GpuLatencyRequest &req, GpuInput &input,
                         std::vector<GpuLatencyRun> &out) {
  run_oneshot_ablation<kCompiledBT, true, false, TNodeBuffer>(req, input, out);
  run_oneshot_ablation<kCompiledBT, true, true, TNodeBuffer>(req, input, out);

  // Main persistent path:
  // naive -> !CrossReduce -> SinglePoller -> mapped staged input -> sentinel.
  run_persistent_ablation<kCompiledBT, false, true, false, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, false, true, true, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, false, false, false, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, false, false, true, TNodeBuffer>(
      req, input, out);
  run_sp_path<kCompiledBT, false, false, true, TNodeBuffer>(req, input, out);
  run_sp_bitmask_path<kCompiledBT, false, false, true, 1, TNodeBuffer>(
      req, input, out);
  run_sp_path<kCompiledBT, true, false, true, TNodeBuffer>(req, input, out);
  run_sp_bitmask_path<kCompiledBT, true, false, true, 1, TNodeBuffer>(
      req, input, out);
  run_sp_path<kCompiledBT, true, true, true, TNodeBuffer>(req, input, out);
  run_bitmask_tpt<kCompiledBT, 1, true, true, 1, TNodeBuffer>(req, input, out);
  run_sp_path<kCompiledBT, true, true, false, TNodeBuffer>(req, input, out);
  run_bitmask_tpt<kCompiledBT, 1, false, true, 1, TNodeBuffer>(req, input,
                                                               out);

  // Side controls retained for the old mapped-first framing.
  run_persistent_ablation<kCompiledBT, true, true, false, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, true, true, true, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, true, false, false, TNodeBuffer>(
      req, input, out);
  run_persistent_ablation<kCompiledBT, true, false, true, TNodeBuffer>(
      req, input, out);
}

}  // namespace

void run_ablation_compact(const GpuLatencyRequest &req, GpuInput &input,
                          std::vector<GpuLatencyRun> &out) {
  run_ablation_buffer<qleaf::CompactNodeBuffer>(req, input, out);
}

void run_ablation_tree(const GpuLatencyRequest &req, GpuInput &input,
                       std::vector<GpuLatencyRun> &out) {
  run_ablation_buffer<qleaf::TreeNodeBuffer>(req, input, out);
}

}  // namespace qleaf_latency_gpu
