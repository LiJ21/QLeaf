#ifndef GWORKERS
#define GWORKERS
#include <qleaf.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/atomic>
#ifndef __CUDACC__
#include <source_location>
#endif

namespace tree_kernels {

#ifndef __CUDACC__
inline void cuda_check(cudaError_t e, const std::source_location loc =
                                          std::source_location::current()) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA err %s:%u in %s: %s\n", loc.file_name(),
                 static_cast<unsigned>(loc.line()), loc.function_name(),
                 cudaGetErrorString(e));
    std::abort();
  }
}
#else
inline void cuda_check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA err: %s\n", cudaGetErrorString(e));
    std::abort();
  }
}
#endif

constexpr int QS_DEPTH = 6;                       // forest depth (<= 6 here)
constexpr int QS_LEAVES = 1 << QS_DEPTH;          // 64
constexpr int QS_INTERNAL = (1 << QS_DEPTH) - 1;  // 63
constexpr int QS_WORDS = (QS_LEAVES + 63) / 64;   // 64-bit words per mask (=1)

// node masks, level/heap order: c_node_masks[n*QS_WORDS + w]
__constant__ uint64_t c_node_masks[QS_INTERNAL * QS_WORDS];
using ni = uint32_t;
using fi = uint32_t;

// ---- one templated walk: {Transposed} x {Bitmask} ------------------------
// Addresses tree data through (spl, idx) with a layout selected at compile
// time:
//   Transposed=false (rowmajor):   elem(node) = base[tree*stride + node]
//                                  (stride = tree_size; also used for the smem
//                                  cache)
//   Transposed=true  (transposed): elem(node) = base[node*stride + tree]
//                                  (stride = ntrees)
// Bitmask=false: dependent path-chase, `depth` nodes. Bitmask=true: QuickScorer
// -- evaluate all QS_INTERNAL nodes branchlessly (needs depth==QS_DEPTH, one
// mask word) and pick the leftmost surviving leaf.
template <bool Transposed, bool Bitmask>
__device__ inline float walk(const float *__restrict__ spl,
                             const tree_kernels::fi *__restrict__ idx,
                             tree_kernels::fi stride, tree_kernels::fi tree,
                             ni depth, const float *features_, float eps) {
  auto off = [&](ni node) -> size_t {
    if constexpr (Transposed)
      return (size_t)node * stride + tree;
    else
      return (size_t)tree * stride + node;
  };
  if constexpr (!Bitmask) {
    ni node = 0;
    for (ni d = 0; d < depth; ++d) {
      const size_t o = off(node);
      node = ((node + 1u) << 1) - (ni)(features_[idx[o]] < spl[o] + eps);
    }
    return spl[off(node)];
  } else {
    uint64_t cand = ~0ull;
#pragma unroll
    for (int n = 0; n < QS_INTERNAL; ++n) {
      const size_t o = off((ni)n);
      if (!(features_[idx[o]] < spl[o] + eps))  // false => prune left subtree
        cand &= c_node_masks[n];
    }
    int leaf =
        __ffsll((unsigned long long)cand) - 1;  // leftmost surviving leaf
    return spl[off((ni)(QS_INTERNAL + leaf))];
  }
}

// Stage a block's trees from global into smem as rowmajor [localtree][node].
// Transposed selects the GLOBAL read pattern (coalesced flat vs strided
// gather); the in-smem layout is always rowmajor (stride tree_size, coprime
// with 32 -> bank-conflict-free walk).
template <int BT, bool Transposed>
__device__ inline void stage(float *s_spl, tree_kernels::fi *s_idx,
                             const float *__restrict__ gspl,
                             const tree_kernels::fi *__restrict__ gidx,
                             tree_kernels::fi ntrees,
                             tree_kernels::fi first_tree,
                             tree_kernels::fi local_ntrees, ni tree_size,
                             tree_kernels::fi ti) {
  const size_t total = (size_t)local_ntrees * tree_size;
  if constexpr (!Transposed) {
    const size_t base =
        (size_t)first_tree * tree_size;  // contiguous -> coalesced
    for (size_t i = ti; i < total; i += BT) {
      s_spl[i] = gspl[base + i];
      s_idx[i] = gidx[base + i];
    }
  } else {
    for (size_t i = ti; i < total;
         i += BT) {  // strided gather from [node][tree]
      tree_kernels::fi lt = (tree_kernels::fi)(i / tree_size);
      ni nd = (ni)(i % tree_size);
      size_t g = (size_t)nd * ntrees + first_tree + lt;
      s_spl[i] = gspl[g];
      s_idx[i] = gidx[g];
    }
  }
}

// Partition smem into [cached splits][cached indices][features]. When !Cached,
// only the feature vector lives in smem.
template <int BT, bool Cached>
__device__ inline float *smem_split(unsigned char *smem, ni tree_size,
                                    float *&s_spl, tree_kernels::fi *&s_idx) {
  if constexpr (Cached) {
    s_spl = reinterpret_cast<float *>(smem);
    s_idx = reinterpret_cast<tree_kernels::fi *>(smem + (size_t)BT * tree_size *
                                                            sizeof(float));
    return reinterpret_cast<float *>(
        smem +
        (size_t)BT * tree_size * (sizeof(float) + sizeof(tree_kernels::fi)));
  } else {
    s_spl = nullptr;
    s_idx = nullptr;
    return reinterpret_cast<float *>(smem);
  }
}

// ---- one-shot kernel ------------------------------------------------------
// *** BEST COMBINATION (measured): one-shot, Transposed, !Bitmask, !Cached,
// driven with MAPPED zero-copy I/O (features_pin in, partials_pin out -- no
// cudaMemcpy). Best BT ~256-512 (more trees -> larger BT). See bench_trees.cu.
template <int BT, bool Transposed, bool Bitmask, bool Cached>
__global__ __launch_bounds__(BT) void traverse_oneshot(
    const float *__restrict__ splits,
    const tree_kernels::fi *__restrict__ indices, tree_kernels::fi ntrees,
    ni depth, const float *__restrict__ features, tree_kernels::fi n_features,
    float eps, float *__restrict__ res_block) {
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;

  const tree_kernels::fi ti = threadIdx.x;
  const ni tree_size = (1u << (depth + 1)) - 1u;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  tree_kernels::fi *s_idx;
  float *features_ = smem_split<BT, Cached>(smem, tree_size, s_spl, s_idx);

  for (tree_kernels::fi i = ti; i < n_features; i += BT)
    features_[i] = features[i];

  tree_kernels::fi local_ntrees = 0;
  if constexpr (Cached) {
    const tree_kernels::fi first_tree = blockIdx.x * BT;
    const tree_kernels::fi remaining =
        first_tree < ntrees ? ntrees - first_tree : 0;
    local_ntrees =
        remaining < (tree_kernels::fi)BT ? remaining : (tree_kernels::fi)BT;
    stage<BT, Transposed>(s_spl, s_idx, splits, indices, ntrees, first_tree,
                          local_ntrees, tree_size, ti);
  }
  __syncthreads();

  float result = 0.0f;
  if constexpr (Cached) {
    if (ti < local_ntrees)
      result = walk<false, Bitmask>(s_spl, s_idx, tree_size, ti, depth,
                                    features_, eps);
  } else {
    const tree_kernels::fi tree = blockIdx.x * BT + ti;
    if (tree < ntrees)
      result = walk<Transposed, Bitmask>(splits, indices,
                                         Transposed ? ntrees : tree_size, tree,
                                         depth, features_, eps);
  }

  float block_sum = BlockReduce(reduce_temp).Sum(result);
  if (ti == 0) res_block[blockIdx.x] = block_sum;
}

// ---- persistent kernel (resident request/done handshake) ------------------
// Renamed from traverse_impl. CrossReduce=false: each block publishes its
// partial to res_block[blockIdx.x] + a per-block done flag (host gathers/sums).
// CrossReduce=true: blocks atomicAdd into res_block[0]; the last block
// (detected via done_counter) publishes one scalar to result_out + one
// done_flag.
template <int BT, bool Transposed, bool Bitmask, bool Cached, bool CrossReduce>
__global__ __launch_bounds__(BT) void traverse_persistent(
    const float *__restrict__ splits,
    const tree_kernels::fi *__restrict__ indices, tree_kernels::fi ntrees,
    ni depth, const float *__restrict__ features, tree_kernels::fi n_features,
    float eps, float *res_block, int *done_out,
    cuda::atomic<int, cuda::thread_scope_system> *request_seq,
    cuda::atomic<int, cuda::thread_scope_device> *done_counter,
    cuda::atomic<int, cuda::thread_scope_system> *done_flag,
    float *result_out) {
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;
  __shared__ bool exit_flag;

  const tree_kernels::fi ti = threadIdx.x;
  const ni tree_size = (1u << (depth + 1)) - 1u;
  const int total_blocks = gridDim.x;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  tree_kernels::fi *s_idx;
  float *features_ = smem_split<BT, Cached>(smem, tree_size, s_spl, s_idx);

  tree_kernels::fi local_ntrees = 0;
  if constexpr (Cached) {  // tree data is static across requests -> stage once
    const tree_kernels::fi first_tree = blockIdx.x * BT;
    const tree_kernels::fi remaining =
        first_tree < ntrees ? ntrees - first_tree : 0;
    local_ntrees =
        remaining < (tree_kernels::fi)BT ? remaining : (tree_kernels::fi)BT;
    stage<BT, Transposed>(s_spl, s_idx, splits, indices, ntrees, first_tree,
                          local_ntrees, tree_size, ti);
    __syncthreads();
  }

  int seq_local = 0;
  while (true) {
    if (ti == 0) {
      int seq;
      do {
        seq = request_seq->load(cuda::memory_order_acquire);
      } while (seq == seq_local);
      seq_local = seq;
      exit_flag = (seq < 0);
    }
    __syncthreads();
    if (exit_flag) break;

    for (tree_kernels::fi i = ti; i < n_features; i += BT)
      features_[i] = features[i];
    __syncthreads();

    float result = 0.0f;
    if constexpr (Cached) {
      if (ti < local_ntrees)
        result = walk<false, Bitmask>(s_spl, s_idx, tree_size, ti, depth,
                                      features_, eps);
    } else {
      const tree_kernels::fi tree = blockIdx.x * BT + ti;
      if (tree < ntrees)
        result = walk<Transposed, Bitmask>(splits, indices,
                                           Transposed ? ntrees : tree_size,
                                           tree, depth, features_, eps);
    }

    float block_sum = BlockReduce(reduce_temp).Sum(result);
    __syncthreads();

    if (ti == 0) {
      if constexpr (CrossReduce) {
        atomicAdd(&res_block[0], block_sum);
        __threadfence();
        int finished =
            done_counter->fetch_add(1, cuda::memory_order_acq_rel) + 1;
        if (finished == total_blocks) {  // last block: publish + reset
          *result_out = res_block[0];
          res_block[0] = 0.0f;
          done_counter->store(0, cuda::memory_order_release);
          __threadfence_system();
          done_flag->store(seq_local, cuda::memory_order_release);
        }
      } else {
        res_block[blockIdx.x] = block_sum;
        __threadfence_system();
        ((volatile int *)done_out)[blockIdx.x] = seq_local;
      }
    }
  }
}

}  // namespace tree_kernels

namespace qleaf {
template <int kBT, bool kTransposed, bool kBitmask, bool kCached, bool kMapped>
struct CudaTraverseOneShot {
  static constexpr int BT = kBT;
  static constexpr bool Transposed = kTransposed;
  static constexpr bool Bitmask = kBitmask;
  static constexpr bool Cached = kCached;
  static constexpr bool Mapped = kMapped;
  static constexpr bool Persistent = false;
  static constexpr bool CrossReduce = false;
};
template <int kBT, bool kTransposed, bool kBitmask, bool kCached, bool kMapped,
          bool kCrossReduce>
struct CudaTraversePersistent
    : CudaTraverseOneShot<kBT, kTransposed, kBitmask, kCached, kMapped> {
  static constexpr bool Persistent = true;
  static constexpr bool CrossReduce = kCrossReduce;
};

template <typename TTraversePolicy, typename TValue, typename TSpan>
class CudaWorker {
 public:
  using TraversePolicy = TTraversePolicy;
  using Value = TValue;
  using Span = TSpan;
  // using P = TraversePolicy;
  constexpr static Value kEps{1e-10};

  CudaWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes),
        depth_(depth),
        tree_size_(((size_t)1 << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        n_features_(config.template get<int>("num_feature")),
        eps_(config.template get<bool>("has_equal") ? kEps : Value{0}) {
    assert(n_trees_ * tree_size_ == nodes.size());
    setup_();
  }
  ~CudaWorker() { teardown_(); }

  CudaWorker(CudaWorker &&other) noexcept
      : nodes_(other.nodes_),
        depth_(other.depth_),
        tree_size_(other.tree_size_),
        n_trees_(other.n_trees_),
        n_features_(other.n_features_),
        eps_(other.eps_),
        result_(other.result_) {
    d_splits_ = std::exchange(other.d_splits_, nullptr);
    d_indices_ = std::exchange(other.d_indices_, nullptr);
    d_splits_T_ = std::exchange(other.d_splits_T_, nullptr);
    d_indices_T_ = std::exchange(other.d_indices_T_, nullptr);
    d_features_ = std::exchange(other.d_features_, nullptr);
    d_res_ = std::exchange(other.d_res_, nullptr);
    partials_pin_ = std::exchange(other.partials_pin_, nullptr);
    result_pin_ = std::exchange(other.result_pin_, nullptr);
    done_slots_pin_ = std::exchange(other.done_slots_pin_, nullptr);
    request_seq_ = std::exchange(other.request_seq_, nullptr);
    done_flag_ = std::exchange(other.done_flag_, nullptr);
    done_counter_ = std::exchange(other.done_counter_, nullptr);
    stream_ = std::exchange(other.stream_, cudaStream_t{});
    grid_ = std::exchange(other.grid_, 0);
    smem_ = std::exchange(other.smem_, 0);
    p_seq_ = std::exchange(other.p_seq_, 0);
    ok_ = std::exchange(other.ok_, false);
    launched_ = std::exchange(other.launched_, false);
    feat_ptr_ = std::exchange(other.feat_ptr_, nullptr);
  }
  CudaWorker(const CudaWorker &) = delete;
  CudaWorker &operator=(const CudaWorker &) = delete;

  bool ok() const { return ok_; }
  int grid() const { return grid_; }

  // Feed one feature vector (size == num_feature). Mapped policies require fts
  // to be a device-accessible (pinned/mapped) host buffer -- asserted -- and,
  // for the persistent kernel, the *same* buffer every call (the resident
  // kernel reads a tree_kernels::fixed address). !Mapped copies fts to a device
  // buffer each call.
  void predict(const auto &fts) {
    assert((size_t)fts.size() == n_features_);
    if (!ok_) return;
    const float *feat;
    if constexpr (TraversePolicy::Mapped) {
      assert_mapped_(fts.data());
      feat = fts.data();
    } else {
      tree_kernels::cuda_check(cudaMemcpy(d_features_, fts.data(),
                                          n_features_ * sizeof(float),
                                          cudaMemcpyHostToDevice));
      feat = d_features_;
    }
    if constexpr (TraversePolicy::Persistent) {
      if (!launched_) {
        feat_ptr_ = feat;
        launch_persistent_(feat);
        launched_ = true;
      } else if constexpr (TraversePolicy::Mapped)
        assert(feat == feat_ptr_);  // resident kernel bound to a
                                    // tree_kernels::fixed address
      result_ = drive_persistent_();
    } else {
      result_ = run_oneshot_(feat);
    }
  }

  Value get() const { return result_; }

 private:
  const Span nodes_;
  const size_t depth_;
  const size_t tree_size_;
  const size_t n_trees_;
  const size_t n_features_;
  const Value eps_{};
  Value result_{};

  // device/host resources
  float *d_splits_ = nullptr;
  tree_kernels::fi *d_indices_ = nullptr;
  float *d_splits_T_ = nullptr;
  tree_kernels::fi *d_indices_T_ = nullptr;
  float *d_features_ = nullptr;  // !Mapped copy target
  float *d_res_ = nullptr;       // per-block partials / accumulator (device)
  float *partials_pin_ = nullptr;
  float *result_pin_ = nullptr;
  int *done_slots_pin_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_system> *request_seq_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_system> *done_flag_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_device> *done_counter_ = nullptr;
  cudaStream_t stream_{};
  int grid_ = 0;
  size_t smem_ = 0;
  int p_seq_ = 0;
  bool ok_ = false;
  bool launched_ = false;
  const float *feat_ptr_ = nullptr;

  static constexpr auto oneshot_kernel() {
    return tree_kernels::traverse_oneshot<
        TraversePolicy::BT, TraversePolicy::Transposed, TraversePolicy::Bitmask,
        TraversePolicy::Cached>;
  }
  static constexpr auto persistent_kernel() {
    return tree_kernels::traverse_persistent<
        TraversePolicy::BT, TraversePolicy::Transposed, TraversePolicy::Bitmask,
        TraversePolicy::Cached, TraversePolicy::CrossReduce>;
  }

  void assert_mapped_(const float *p) {
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    assert(e == cudaSuccess && attr.type == cudaMemoryTypeHost &&
           "Mapped policy requires a pinned/mapped host feature buffer");
    (void)e;
  }

  // Opt into >48KB dynamic smem; false if it exceeds the device limit.
  static bool enable_smem_(const void *kernel, size_t bytes) {
    int maxopt = 0;
    tree_kernels::cuda_check(cudaDeviceGetAttribute(
        &maxopt, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    if (bytes > (size_t)maxopt) return false;
    if (bytes > 48u * 1024u)
      tree_kernels::cuda_check(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    return true;
  }
  // A persistent grid must be fully co-resident or the host<->device handshake
  // deadlocks; check occupancy x SM count >= grid.
  bool persistent_fits_() {
    int per_sm = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, persistent_kernel(), TraversePolicy::BT, smem_) !=
        cudaSuccess)
      return false;
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    return per_sm > 0 && grid_ <= per_sm * nsm;
  }

  static void upload_bitmasks_() {
    static_assert(tree_kernels::QS_WORDS == 1);
    std::array<uint64_t, tree_kernels::QS_INTERNAL * tree_kernels::QS_WORDS>
        masks{};
    size_t idx = 0;
    size_t num_per_level = 1;
    for (size_t d = 0; d < tree_kernels::QS_DEPTH; ++d, num_per_level <<= 1) {
      for (size_t i = 0; i < num_per_level; ++i) {
        const size_t sub_size = size_t{1} << (tree_kernels::QS_DEPTH - d);
        const size_t len = sub_size >> 1;
        const size_t start = sub_size * i;
        const uint64_t left_subtree = ((uint64_t{1} << len) - 1) << start;
        masks[idx++] = ~left_subtree;
      }
    }
    tree_kernels::cuda_check(
        cudaMemcpyToSymbol(tree_kernels::c_node_masks, masks.data(),
                           masks.size() * sizeof(uint64_t)));
  }

  void setup_() {
    if constexpr (TraversePolicy::Bitmask) {
      if (depth_ != tree_kernels::QS_DEPTH) {
        ok_ = false;
        return;
      }
      upload_bitmasks_();
    }

    grid_ = (int)((n_trees_ + TraversePolicy::BT - 1) / TraversePolicy::BT);
    // host arrays from the node span ([tree][node]); upload both layouts
    // needed.
    const size_t N = nodes_.size();
    std::vector<float> hs(N);
    std::vector<tree_kernels::fi> hi(N);
    for (size_t i = 0; i < N; ++i) {
      hs[i] = nodes_.split(i);
      hi[i] = (tree_kernels::fi)nodes_.idx(i);
    }
    tree_kernels::cuda_check(cudaMalloc(&d_splits_, N * sizeof(float)));
    tree_kernels::cuda_check(
        cudaMalloc(&d_indices_, N * sizeof(tree_kernels::fi)));
    tree_kernels::cuda_check(cudaMemcpy(d_splits_, hs.data(), N * sizeof(float),
                                        cudaMemcpyHostToDevice));
    tree_kernels::cuda_check(cudaMemcpy(d_indices_, hi.data(),
                                        N * sizeof(tree_kernels::fi),
                                        cudaMemcpyHostToDevice));
    if constexpr (TraversePolicy::Transposed) {
      std::vector<float> hsT(N);
      std::vector<tree_kernels::fi> hiT(N);
      for (size_t t = 0; t < n_trees_; ++t)
        for (size_t n = 0; n < tree_size_; ++n) {
          hsT[n * n_trees_ + t] = hs[t * tree_size_ + n];
          hiT[n * n_trees_ + t] = hi[t * tree_size_ + n];
        }
      tree_kernels::cuda_check(cudaMalloc(&d_splits_T_, N * sizeof(float)));
      tree_kernels::cuda_check(
          cudaMalloc(&d_indices_T_, N * sizeof(tree_kernels::fi)));
      tree_kernels::cuda_check(cudaMemcpy(
          d_splits_T_, hsT.data(), N * sizeof(float), cudaMemcpyHostToDevice));
      tree_kernels::cuda_check(cudaMemcpy(d_indices_T_, hiT.data(),
                                          N * sizeof(tree_kernels::fi),
                                          cudaMemcpyHostToDevice));
    }
    tree_kernels::cuda_check(cudaMalloc(&d_res_, n_trees_ * sizeof(float)));
    tree_kernels::cuda_check(
        cudaMallocHost(&partials_pin_, n_trees_ * sizeof(float)));
    tree_kernels::cuda_check(cudaMallocHost(&result_pin_, sizeof(float)));
    if constexpr (!TraversePolicy::Mapped)
      tree_kernels::cuda_check(
          cudaMalloc(&d_features_, n_features_ * sizeof(float)));
    if constexpr (TraversePolicy::Persistent) {
      tree_kernels::cuda_check(
          cudaMallocHost(&done_slots_pin_, n_trees_ * sizeof(int)));
      tree_kernels::cuda_check(
          cudaMallocHost(&request_seq_, sizeof(*request_seq_)));
      tree_kernels::cuda_check(
          cudaMallocHost(&done_flag_, sizeof(*done_flag_)));
      tree_kernels::cuda_check(
          cudaMalloc(&done_counter_, sizeof(*done_counter_)));
      new (request_seq_) cuda::atomic<int, cuda::thread_scope_system>(0);
      new (done_flag_) cuda::atomic<int, cuda::thread_scope_system>(0);
    }
    tree_kernels::cuda_check(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    smem_ = TraversePolicy::Cached
                ? (size_t)TraversePolicy::BT * tree_size_ *
                          (sizeof(float) + sizeof(tree_kernels::fi)) +
                      n_features_ * sizeof(float)
                : n_features_ * sizeof(float);
    const void *kern = TraversePolicy::Persistent
                           ? (const void *)persistent_kernel()
                           : (const void *)oneshot_kernel();
    ok_ = enable_smem_(kern, smem_);
    if constexpr (TraversePolicy::Persistent)
      if (ok_) ok_ = persistent_fits_();
    // Uploads ran on the null stream; predict() launches on stream_
    // (non-blocking, which does not order against the null stream). Settle
    // everything before any predict so the kernel can't race ahead of the
    // tree-data upload.
    tree_kernels::cuda_check(cudaDeviceSynchronize());
  }

  void launch_kernel_(const float *feat, float *res) {
    const float *spl = TraversePolicy::Transposed ? d_splits_T_ : d_splits_;
    const tree_kernels::fi *idx =
        TraversePolicy::Transposed ? d_indices_T_ : d_indices_;
    if constexpr (!TraversePolicy::Persistent)
      tree_kernels::traverse_oneshot<
          TraversePolicy::BT, TraversePolicy::Transposed,
          TraversePolicy::Bitmask, TraversePolicy::Cached>
          <<<grid_, TraversePolicy::BT, smem_, stream_>>>(
              spl, idx, (tree_kernels::fi)n_trees_, (tree_kernels::ni)depth_,
              feat, (tree_kernels::fi)n_features_, (float)eps_, res);
    else
      tree_kernels::traverse_persistent<
          TraversePolicy::BT, TraversePolicy::Transposed,
          TraversePolicy::Bitmask, TraversePolicy::Cached,
          TraversePolicy::CrossReduce>
          <<<grid_, TraversePolicy::BT, smem_, stream_>>>(
              spl, idx, (tree_kernels::fi)n_trees_, (tree_kernels::ni)depth_,
              feat, (tree_kernels::fi)n_features_, (float)eps_, res,
              done_slots_pin_, request_seq_, done_counter_, done_flag_,
              result_pin_);
  }

  Value run_oneshot_(const float *feat) {
    float *out = TraversePolicy::Mapped ? partials_pin_ : d_res_;
    launch_kernel_(feat, out);
    tree_kernels::cuda_check(cudaGetLastError());
    tree_kernels::cuda_check(cudaStreamSynchronize(stream_));
    if constexpr (TraversePolicy::Mapped) {
      float s = 0.0f;
      for (int b = 0; b < grid_; ++b) s += partials_pin_[b];
      return s;
    } else {
      std::vector<float> p(grid_);
      tree_kernels::cuda_check(cudaMemcpy(
          p.data(), d_res_, grid_ * sizeof(float), cudaMemcpyDeviceToHost));
      float s = 0.0f;
      for (float x : p) s += x;
      return s;
    }
  }

  void launch_persistent_(const float *feat) {
    request_seq_->store(0, cuda::memory_order_relaxed);
    done_flag_->store(0, cuda::memory_order_relaxed);
    for (int b = 0; b < grid_; ++b) {
      done_slots_pin_[b] = 0;
      partials_pin_[b] = 0.0f;
    }
    tree_kernels::cuda_check(
        cudaMemset(done_counter_, 0, sizeof(*done_counter_)));
    tree_kernels::cuda_check(cudaMemset(d_res_, 0, n_trees_ * sizeof(float)));
    p_seq_ = 0;
    float *res = TraversePolicy::CrossReduce ? d_res_ : partials_pin_;
    launch_kernel_(feat, res);
    tree_kernels::cuda_check(cudaGetLastError());
  }

  Value drive_persistent_() {
    request_seq_->store(++p_seq_, cuda::memory_order_release);
    if constexpr (TraversePolicy::CrossReduce) {
      while (done_flag_->load(cuda::memory_order_acquire) != p_seq_) {
      }
      return *result_pin_;
    } else {
      volatile int *vdone = done_slots_pin_;
      for (int b = 0; b < grid_; ++b)
        while (vdone[b] != p_seq_) {
        }
      cuda::atomic_thread_fence(cuda::memory_order_acquire,
                                cuda::thread_scope_system);
      float s = 0.0f;
      for (int b = 0; b < grid_; ++b) s += partials_pin_[b];
      return s;
    }
  }

  void teardown_() {
    if constexpr (TraversePolicy::Persistent)
      if (launched_) {
        request_seq_->store(-1, cuda::memory_order_release);
        cudaStreamSynchronize(stream_);
      }
    if (stream_) cudaStreamDestroy(stream_);
    cudaFree(d_splits_);
    cudaFree(d_indices_);
    cudaFree(d_splits_T_);
    cudaFree(d_indices_T_);
    cudaFree(d_features_);
    cudaFree(d_res_);
    cudaFreeHost(partials_pin_);
    cudaFreeHost(result_pin_);
    if constexpr (TraversePolicy::Persistent) {
      cudaFreeHost(done_slots_pin_);
      cudaFreeHost(request_seq_);
      cudaFreeHost(done_flag_);
      cudaFree(done_counter_);
    }
  }
};
// void dispatch_traverse() {}

}  // namespace qleaf
#endif
