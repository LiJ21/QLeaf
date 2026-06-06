#ifndef GWORKERS
#define GWORKERS
#include <qleaf.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <limits>
#include <utility>
#ifndef __CUDACC__
#include <source_location>
#endif

namespace tree_kernels {
// Mapped-buffer sentinel for !SeqFlag: +max = pending, -max = exit.
template <typename T>
__host__ __device__ constexpr auto sentinel() {
  return std::numeric_limits<T>::max();
}

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

// QuickScorer (QS) mask table for all depths up to QS_MAX_DEPTH.
constexpr int QS_MAX_DEPTH = (int)qleaf::kDefaultMaxDepth;
static_assert(QS_MAX_DEPTH > 0 && QS_MAX_DEPTH <= 6,
              "CUDA QS masks currently use one 64-bit word");
constexpr int QS_MAX_LEAVES = 1 << QS_MAX_DEPTH;
constexpr int QS_MAX_INTERNAL = (1 << QS_MAX_DEPTH) - 1;
constexpr int QS_MAX_WORDS = (QS_MAX_LEAVES + 63) / 64;

// QS node masks in level/heap order.
__constant__ uint64_t c_node_masks[QS_MAX_INTERNAL * QS_MAX_WORDS];
using ni = uint32_t;
using fi = uint32_t;

__host__ __device__ constexpr int gcd_constexpr(int a, int b) {
  while (b != 0) {
    const int t = a % b;
    a = b;
    b = t;
  }
  return a < 0 ? -a : a;
}

__host__ __device__ constexpr bool qs_node_stride_valid(int stride) {
  if (stride <= 0) return false;
  for (int depth = 1; depth <= QS_MAX_DEPTH; ++depth)
    if (gcd_constexpr(stride, (1 << depth) - 1) != 1) return false;
  return true;
}

// AND-reduce partial QS survivor masks across TPT lanes.
struct BitwiseAnd {
  __device__ __forceinline__ uint64_t operator()(uint64_t a, uint64_t b) const {
    return a & b;
  }
};

__host__ __device__ inline size_t
node_major_cache_tree_stride(size_t cached_trees_per_block,
                             int threads_per_tree) {
  constexpr size_t shared_memory_banks = 32;
  const size_t target_remainder =
      threads_per_tree == 1 ? 0 : shared_memory_banks / threads_per_tree;
  size_t stride = cached_trees_per_block;
  while (stride % shared_memory_banks != target_remainder) ++stride;
  return stride;
}

// QS mask source: constant memory if requested, otherwise shared copy.
#ifdef QLEAF_MASK_CONST
#define QS_NODE_MASK(n) c_node_masks[n]
#else
#define QS_NODE_MASK(n) node_masks[n]
#endif

// Tree walk, specialized by layout and traversal strategy:
//   Transposed=false (rowmajor):   elem(node) = base[tree*stride + node]
//   Transposed=true  (transposed): elem(node) = base[node*stride + tree]
// Bitmask=true uses QS; TPT>1 splits nodes across lanes and returns on lane 0.
// TPT is only used for bitmask, as traversal of a single tree cannot be
// parallelized
template <bool Transposed, bool Bitmask, int TPT = 1, int NodeStride = 1>
__device__ inline float walk(const float *__restrict__ spl,
                             const tree_kernels::fi *__restrict__ idx,
                             tree_kernels::fi stride, tree_kernels::fi tree,
                             ni depth, const float *features_, float eps,
                             const uint64_t *node_masks = nullptr, int lane = 0,
                             void *warp_temp = nullptr) {
  static_assert(NodeStride == 1 || (Bitmask && TPT > 1),
                "NodeStride only applies to TPT>1 bitmask traversal");
  static_assert(NodeStride == 1 || qs_node_stride_valid(NodeStride),
                "NodeStride must be coprime with every supported QS node count");

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
    const ni internal_nodes = ((ni)1 << depth) - 1u;
    const int leaf_shift = QS_MAX_DEPTH - (int)depth;
    uint64_t cand = ~0ull;
    if (depth == QS_MAX_DEPTH) {
#pragma unroll
      for (int k = lane; k < QS_MAX_INTERNAL; k += TPT) {
        int n = k;
        if constexpr (NodeStride != 1)
          n = (k * NodeStride) % QS_MAX_INTERNAL;
        const size_t o = off((ni)n);
        auto go_right = !(features_[idx[o]] < spl[o] + eps);
        cand &= go_right ? QS_NODE_MASK(n) : ~0ull;
      }
    } else {
      // Smaller depths use the prefix of the max-depth mask table.
      for (int k = lane; k < (int)internal_nodes; k += TPT) {
        int n = k;
        if constexpr (NodeStride != 1)
          n = (k * NodeStride) % (int)internal_nodes;
        const size_t o = off((ni)n);
        auto go_right = !(features_[idx[o]] < spl[o] + eps);
        cand &= go_right ? QS_NODE_MASK(n) : ~0ull;
      }
    }
    if constexpr (TPT == 1) {
      const int bit =
          __ffsll((unsigned long long)cand) - 1;  // leftmost survivor
      const int leaf = bit >> leaf_shift;
      return spl[off((ni)(internal_nodes + leaf))];
    } else {
      // when trees are distributed, reduce the masks from relevant threads
      using WarpReduceT = cub::WarpReduce<uint64_t, TPT>;
      auto &temp =
          *reinterpret_cast<typename WarpReduceT::TempStorage *>(warp_temp);
      cand = WarpReduceT(temp).Reduce(cand, BitwiseAnd{});  // lane 0 result
      float tree_result = 0.0f;
      if (lane == 0) {
        const int bit = __ffsll((unsigned long long)cand) - 1;
        const int leaf = bit >> leaf_shift;
        tree_result = spl[off((ni)(internal_nodes + leaf))];
      }
      return tree_result;
    }
  }
}

// Get the location of a tree in global memory given block, wave (when a block
// has to deal with more than one parallel wave), and slot
template <int TPB>
__device__ __forceinline__ tree_kernels::fi global_tree_for(
    int block, int grid_stride_wave, int tree_slot) {
  return (tree_kernels::fi)((block + grid_stride_wave * gridDim.x) * TPB +
                            tree_slot);
}

// Location in shared memory
template <int TPB>
__device__ __forceinline__ int cached_tree_for(int grid_stride_wave,
                                               int tree_slot) {
  return grid_stride_wave * TPB + tree_slot;
}

template <bool Transposed>
__device__ __forceinline__ size_t global_tree_offset(tree_kernels::fi tree,
                                                     ni node,
                                                     tree_kernels::fi ntrees,
                                                     ni tree_size) {
  if constexpr (Transposed)
    return (size_t)node * ntrees + tree;
  else
    return (size_t)tree * tree_size + node;
}

template <int BT, int TPB, bool TransposedGlobal, bool CacheNodeMajor>
__device__ inline void cache_trees_grid_stride_waves(
    float *s_spl, tree_kernels::fi *s_idx, const float *__restrict__ gspl,
    const tree_kernels::fi *__restrict__ gidx, tree_kernels::fi ntrees,
    ni tree_size, size_t cached_trees_per_block, size_t cache_tree_stride,
    tree_kernels::fi ti) {
  const size_t total = cached_trees_per_block * tree_size;
  for (size_t k = ti; k < total; k += BT) {
    int cached_tree_slot;
    ni node;
    size_t s;
    if constexpr (CacheNodeMajor) {
      node = (ni)(k / cached_trees_per_block);
      cached_tree_slot = (int)(k % cached_trees_per_block);
      s = (size_t)node * cache_tree_stride + cached_tree_slot;
    } else {
      cached_tree_slot = (int)(k / tree_size);
      node = (ni)(k % tree_size);
      s = k;
    }
    const int grid_stride_wave = cached_tree_slot / TPB;
    const int tree_slot = cached_tree_slot % TPB;
    const tree_kernels::fi global_tree =
        global_tree_for<TPB>(blockIdx.x, grid_stride_wave, tree_slot);
    if (global_tree < ntrees) {
      const size_t g =
          global_tree_offset<TransposedGlobal>(global_tree, node, ntrees,
                                               tree_size);
      s_spl[s] = gspl[g];
      s_idx[s] = gidx[g];
    }
  }
}

// Split dynamic smem into optional tree cache plus feature cache.
template <bool Cached>
__device__ inline float *smem_split(unsigned char *smem,
                                    size_t cached_trees_per_block, ni tree_size,
                                    float *&s_spl, tree_kernels::fi *&s_idx) {
  if constexpr (Cached) {
    s_spl = reinterpret_cast<float *>(smem);       // splits
    s_idx = reinterpret_cast<tree_kernels::fi *>(  // indices
        smem + cached_trees_per_block * tree_size * sizeof(float));
    return reinterpret_cast<float *>(
        smem + cached_trees_per_block * tree_size *
                   (sizeof(float) + sizeof(tree_kernels::fi)));
  } else {
    s_spl = nullptr;
    s_idx = nullptr;
    return reinterpret_cast<float *>(smem);
  }
}

// Copy QS masks to shared memory unless constant-memory mode is selected.
template <int BT, bool Bitmask>
__device__ __forceinline__ void load_node_masks(uint64_t *s_node_masks,
                                                tree_kernels::fi ti) {
#ifndef QLEAF_MASK_CONST
  if constexpr (Bitmask)
    for (int i = (int)ti; i < QS_MAX_INTERNAL; i += BT)
      s_node_masks[i] = c_node_masks[i];
#else
  (void)s_node_masks;
  (void)ti;
#endif
}

// One-shot traversal kernel.
template <int BT, bool Transposed, bool Bitmask, bool Cached, int TPT = 1,
          bool CacheNodeMajor = false, int NodeStride = 1>
__global__ __launch_bounds__(BT) void traverse_oneshot(
    const float *__restrict__ splits,
    const tree_kernels::fi *__restrict__ indices, tree_kernels::fi ntrees,
    ni depth, const float *__restrict__ features, tree_kernels::fi n_features,
    float eps, float *__restrict__ res_block) {
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  static_assert(TPT > 0 && TPT <= 32 && (TPT & (TPT - 1)) == 0,
                "TPT must be a power of 2 in [1,32]");
  static_assert(TPT == 1 || Bitmask,
                "intra-tree parallelism only applies to the bitmask path");
  static_assert(BT % TPT == 0, "block size must be divisible by TPT");
  static_assert(!CacheNodeMajor || Cached,
                "node-major caching requires Cached");
  static_assert(!CacheNodeMajor || Bitmask,
                "node-major caching is only supported for Bitmask");
  static_assert(NodeStride == 1 || (Bitmask && TPT > 1),
                "NodeStride only applies to TPT>1 bitmask traversal");
  constexpr int TPB = BT / TPT;
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;
  // QS masks; dummy-sized when !Bitmask.
  __shared__ uint64_t s_node_masks[Bitmask ? QS_MAX_INTERNAL : 1];
  // WarpReduce storage for TPT>1.
  using WarpReduceT = cub::WarpReduce<uint64_t, (TPT > 1 ? TPT : 2)>;
  __shared__
      typename WarpReduceT::TempStorage warp_reduce_temp[TPT > 1 ? TPB : 1];

  const tree_kernels::fi ti = threadIdx.x;
  const int tree_slot = (int)ti / TPT;
  const int lane = (int)ti % TPT;
  void *wtemp = nullptr;
  if constexpr (TPT > 1) wtemp = &warp_reduce_temp[tree_slot];
  const ni tree_size = (1u << (depth + 1)) - 1u;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  tree_kernels::fi *s_idx;
  const size_t cached_trees_per_block = (size_t)TPB;
  const size_t cache_tree_stride =
      CacheNodeMajor ? node_major_cache_tree_stride(cached_trees_per_block, TPT)
                     : cached_trees_per_block;
  float *features_ =
      smem_split<Cached>(smem, cache_tree_stride, tree_size, s_spl, s_idx);

  for (tree_kernels::fi i = ti; i < n_features; i += BT)
    features_[i] = features[i];
  load_node_masks<BT, Bitmask>(s_node_masks, ti);

  // Caching trees does not really make sense for one-shot kernels; only used
  // for benchmarking/profiling experiments
  if constexpr (Cached)
    cache_trees_grid_stride_waves<BT, TPB, Transposed, CacheNodeMajor>(
        s_spl, s_idx, splits, indices, ntrees, tree_size,
        cached_trees_per_block, cache_tree_stride, ti);
  __syncthreads();

  float result = 0.0f;
  if constexpr (Cached) {
    const tree_kernels::fi tree = global_tree_for<TPB>(blockIdx.x, 0, tree_slot);
    if (tree < ntrees) {
      if constexpr (CacheNodeMajor)
        result = walk<true, Bitmask, TPT, NodeStride>(
            s_spl, s_idx, cache_tree_stride, tree_slot, depth, features_, eps,
            s_node_masks, lane, wtemp);
      else
        result = walk<false, Bitmask, TPT, NodeStride>(
            s_spl, s_idx, tree_size, tree_slot, depth, features_, eps,
            s_node_masks, lane, wtemp);
    }
  } else {
    const tree_kernels::fi tree =
        global_tree_for<TPB>(blockIdx.x, 0, tree_slot);
    // Transposed == true leads to "column-major" layout of nodes (first roots
    // of all trees and so on)
    if (tree < ntrees)
      result = walk<Transposed, Bitmask, TPT, NodeStride>(
          splits, indices, Transposed ? ntrees : tree_size, tree, depth,
          features_, eps, s_node_masks, lane, wtemp);
  }

  float block_sum = BlockReduce(reduce_temp).Sum(result);
  if (ti == 0) res_block[blockIdx.x] = block_sum;
}

// Persistent traversal kernel.
//  When CrossReduce == true the kernel returns the sum of results from
//    different blocks; otherwise it returns blockwise results and the host is
//    responsible to collect and sum them up.
//  When BT * grid_ < ntrees, the kernel computes the results in multiple
//    "waves".
template <int BT, bool Transposed, bool Bitmask, bool Cached,
          bool CacheAllWaves, bool CrossReduce, bool SinglePoller = false,
          bool StageFeatures = false, bool SeqFlag = true, int TPT = 1,
          bool CacheNodeMajor = false, int NodeStride = 1>
__global__ __launch_bounds__(BT) void traverse_persistent(
    const float *__restrict__ splits,
    const tree_kernels::fi *__restrict__ indices, tree_kernels::fi ntrees,
    ni depth, const float *__restrict__ features, tree_kernels::fi n_features,
    float eps, float *res_block, int *done_out,
    cuda::atomic<int, cuda::thread_scope_system> *request_seq,
    cuda::atomic<int, cuda::thread_scope_device> *done_counter,
    cuda::atomic<int, cuda::thread_scope_system> *done_flag, float *result_out,
    cuda::atomic<int, cuda::thread_scope_device> *g_request_bcast = nullptr,
    float *d_feat_stage = nullptr, int max_waves_per_block = 1) {
  static_assert(!(CacheAllWaves && !Cached));
  static_assert(!(CacheAllWaves && CrossReduce));
  static_assert(!(SinglePoller && CrossReduce),
                "SinglePoller path implements the !CrossReduce publish only");
  static_assert(SeqFlag || (SinglePoller && StageFeatures),
                "the sentinel signal (!SeqFlag) requires SinglePoller and "
                "StageFeatures: the leader resets the mapped buffer, so the "
                "followers must read their own staged copy");
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  static_assert(TPT > 0 && TPT <= 32 && (TPT & (TPT - 1)) == 0,
                "TPT must be a power of 2 in [1,32]");
  static_assert(TPT == 1 || Bitmask,
                "intra-tree parallelism only applies to the bitmask path");
  static_assert(BT % TPT == 0, "block size must be divisible by TPT");
  static_assert(TPT == 1 || !CrossReduce,
                "TPT>1 is benchmarked on the !CrossReduce publish only");
  static_assert(!CacheNodeMajor || Cached,
                "node-major caching requires Cached");
  static_assert(!CacheNodeMajor || Bitmask,
                "node-major caching is only supported for Bitmask");
  static_assert(NodeStride == 1 || (Bitmask && TPT > 1),
                "NodeStride only applies to TPT>1 bitmask traversal");
  constexpr int TPB = BT / TPT;
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;
  __shared__ bool exit_flag;
  __shared__ uint64_t s_node_masks[Bitmask ? QS_MAX_INTERNAL : 1];
  using WarpReduceT = cub::WarpReduce<uint64_t, (TPT > 1 ? TPT : 2)>;
  __shared__
      typename WarpReduceT::TempStorage warp_reduce_temp[TPT > 1 ? TPB : 1];

  const tree_kernels::fi ti = threadIdx.x;
  const int tree_slot = (int)ti / TPT;
  const int lane = (int)ti % TPT;
  void *wtemp = nullptr;
  if constexpr (TPT > 1) wtemp = &warp_reduce_temp[tree_slot];
  const ni tree_size = (1u << (depth + 1)) - 1u;
  const int total_blocks = gridDim.x;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  tree_kernels::fi *s_idx;
  const int cached_waves = CacheAllWaves ? max_waves_per_block : 1;
  const size_t cached_trees_per_block =
      (size_t)cached_waves * TPB;
  const size_t cache_tree_stride =
      CacheNodeMajor
          ? node_major_cache_tree_stride(cached_trees_per_block, TPT)
          : cached_trees_per_block;
  float *features_ =
      smem_split<Cached>(smem, cache_tree_stride, tree_size, s_spl, s_idx);

  load_node_masks<BT, Bitmask>(s_node_masks, ti);

  if constexpr (Cached)
    cache_trees_grid_stride_waves<BT, TPB, Transposed, CacheNodeMajor>(
        s_spl, s_idx, splits, indices, ntrees, tree_size,
        cached_trees_per_block, cache_tree_stride, ti);
  if constexpr (Bitmask || Cached) __syncthreads();

  int seq_local = 0;
#ifdef QLEAF_PROF_PERSIST
  // Per-request phase timing; compiled out unless QLEAF_PROF_PERSIST is set.
  unsigned long long prof_wait = 0, prof_comp = 0, prof_pub = 0, prof_n = 0;
  // Compute breakdown: feature load, tree walk, block reduce.
  unsigned long long prof_feat = 0, prof_walk = 0, prof_reduce = 0;
  long long pt1 = 0, pt_f = 0, pt_w = 0;
#endif
  // main loop; wait for features and/or signal to compute
  while (true) {
    if constexpr (SinglePoller) {
      // Block 0 polls the host; followers poll the device-scope broadcast.
      __shared__ int s_seq;
#ifdef QLEAF_PROF_PERSIST
      long long pt0 = 0;
      if (ti == 0) pt0 = clock64();
#endif
      if (blockIdx.x == 0) {  // block 0 is the leader
        if constexpr (SeqFlag) {
          // Host increments request_seq and stores (Mapped) or copies (!Mapped)
          // the features.
          if (ti == 0) {
            int seq;
            do {
              seq = request_seq->load(cuda::memory_order_acquire);
            } while (seq == seq_local);
            s_seq = seq;
            exit_flag = (seq < 0);
          }
          __syncthreads();
        } else {
          // !SeqFlag: feature buffer itself is the signal
          // job starts when all features are not sentinels; after completion
          // they're set to sentinels
          auto features_volatile =
              reinterpret_cast<const volatile float *>(features);
          bool all_present = false, any_exit = false;
          do {
            bool present = true, exit_requested = false;
            for (tree_kernels::fi i = ti; i < n_features; i += BT) {
              float value = features_volatile[i];
              d_feat_stage[i] = value;
              present &= (value != sentinel<float>());
              exit_requested = exit_requested || (value == -sentinel<float>());
            }
            any_exit = __syncthreads_or(exit_requested);
            all_present = __syncthreads_and(present);
          } while (!all_present && !any_exit);
          if (ti == 0) {
            s_seq = any_exit ? -1 : seq_local + 1;
            exit_flag = any_exit;
          }
          __syncthreads();
        }
        // Publish features and request sequence.
        if (!exit_flag) {
          if constexpr (!SeqFlag) {
            // Reset input; staged copy feeds this request.
            for (tree_kernels::fi i = ti; i < n_features; i += BT)
              const_cast<float *>(features)[i] = sentinel<float>();
            __syncthreads();
          } else if constexpr (StageFeatures) {
            // Stage features once before broadcast (for SeqFlag case).
            for (tree_kernels::fi i = ti; i < n_features; i += BT)
              d_feat_stage[i] = features[i];
            __syncthreads();
          }
          if (ti == 0) {
            g_request_bcast->store(s_seq, cuda::memory_order_release);
          }
        } else if (ti == 0) {
          g_request_bcast->store(s_seq, cuda::memory_order_release);
        }
      } else {
        // Follower: check broadcast seq number from leader
        if (ti == 0) {
          int seq;
          do {
            seq = g_request_bcast->load(cuda::memory_order_acquire);
          } while (seq == seq_local);
          s_seq = seq;
          exit_flag = (seq < 0);
        }
        __syncthreads();
      }
      seq_local = s_seq;
#ifdef QLEAF_PROF_PERSIST
      if (ti == 0) {
        pt1 = clock64();
        prof_wait += (unsigned long long)(pt1 - pt0);
      }
#endif
      if (exit_flag) break;
    } else {
      // All blocks poll
      if (ti == 0) {
#ifdef QLEAF_PROF_PERSIST
        long long pt0 = clock64();
#endif
        int seq;
        do {
          seq = request_seq->load(cuda::memory_order_acquire);
        } while (seq == seq_local);
        seq_local = seq;
        exit_flag = (seq < 0);
#ifdef QLEAF_PROF_PERSIST
        pt1 = clock64();
        prof_wait += (unsigned long long)(pt1 - pt0);
#endif
      }
      __syncthreads();
      if (exit_flag) break;
    }

    // By default, read from the feature buffer
    const float *feat_src = features;
    // Read from the staged feature only if SinglePoller and StageFeatures
    if constexpr (SinglePoller && StageFeatures) feat_src = d_feat_stage;
    for (tree_kernels::fi i = ti; i < n_features; i += BT)
      features_[i] = feat_src[i];
    __syncthreads();
#ifdef QLEAF_PROF_PERSIST
    if (ti == 0) pt_f = clock64();
#endif

    float result = 0.0f;
    if constexpr (!Cached) {
      for (tree_kernels::fi tree =
               global_tree_for<TPB>(blockIdx.x, 0, tree_slot);
           tree < ntrees; tree += (tree_kernels::fi)gridDim.x * TPB)
        result += walk<Transposed, Bitmask, TPT, NodeStride>(
            splits, indices, Transposed ? ntrees : tree_size, tree, depth,
            features_, eps, s_node_masks, lane, wtemp);
    } else {
      for (int grid_stride_wave = 0; grid_stride_wave < cached_waves;
           ++grid_stride_wave) {
        const tree_kernels::fi tree =
            global_tree_for<TPB>(blockIdx.x, grid_stride_wave, tree_slot);
        if (tree >= ntrees) continue;
        const int cached_tree_slot =
            cached_tree_for<TPB>(grid_stride_wave, tree_slot);
        if constexpr (CacheNodeMajor)
          result += walk<true, Bitmask, TPT, NodeStride>(
              s_spl, s_idx, cache_tree_stride, cached_tree_slot, depth,
              features_, eps, s_node_masks, lane, wtemp);
        else
          result += walk<false, Bitmask, TPT, NodeStride>(
              s_spl, s_idx, tree_size, cached_tree_slot, depth, features_, eps,
              s_node_masks, lane, wtemp);
      }
    }

#ifdef QLEAF_PROF_PERSIST
    if (ti == 0) pt_w = clock64();
#endif
    float block_sum = BlockReduce(reduce_temp).Sum(result);
    __syncthreads();

    if (ti == 0) {
#ifdef QLEAF_PROF_PERSIST
      long long pt3 = clock64();
      prof_comp += (unsigned long long)(pt3 - pt1);
      prof_feat += (unsigned long long)(pt_f - pt1);
      prof_walk += (unsigned long long)(pt_w - pt_f);
      prof_reduce += (unsigned long long)(pt3 - pt_w);
#endif
      if constexpr (CrossReduce) {
        atomicAdd(&res_block[0], block_sum);
        int finished =
            done_counter->fetch_add(1, cuda::memory_order_acq_rel) + 1;
        if (finished == total_blocks) {
          *result_out = res_block[0];
          res_block[0] = 0.0f;
          done_counter->store(0, cuda::memory_order_relaxed);
          done_flag->store(seq_local, cuda::memory_order_release);
        }
      } else {
        auto b = blockIdx.x;
        res_block[b] = block_sum;
        // cuda::atomic_ref<int, cuda::thread_scope_system>(done_out[b])
        //     .store(seq_local, cuda::memory_order_release);
      }

#ifdef QLEAF_PROF_PERSIST
      prof_pub += (unsigned long long)(clock64() - pt3);
      prof_n++;
#endif
    }
  }
#ifdef QLEAF_PROF_PERSIST
  // Report first/last block timing.
  if (ti == 0 && (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1))
    printf(
        "[persist-prof blk %d] reqs=%llu  avg cyc/req: wait=%llu compute=%llu "
        "(feat=%llu walk=%llu reduce=%llu) publish=%llu\n",
        blockIdx.x, prof_n, prof_n ? prof_wait / prof_n : 0ull,
        prof_n ? prof_comp / prof_n : 0ull, prof_n ? prof_feat / prof_n : 0ull,
        prof_n ? prof_walk / prof_n : 0ull,
        prof_n ? prof_reduce / prof_n : 0ull,
        prof_n ? prof_pub / prof_n : 0ull);
#endif
}

}  // namespace tree_kernels

// qleaf wrapper
namespace qleaf {

// Dispatch policies containing parameters
template <int kBT, bool kTransposed, bool kBitmask, bool kCached, bool kMapped,
          int kTPT = 1, bool kCacheNodeMajor = false, int kNodeStride = 1>
struct CudaTraverseOneShot {
  static constexpr int BT = kBT;
  static constexpr bool Transposed = kTransposed;
  static constexpr bool Bitmask = kBitmask;
  static constexpr bool Cached = kCached;
  static constexpr bool Mapped = kMapped;
  static constexpr bool Persistent = false;
  static constexpr bool CrossReduce = false;
  static constexpr bool CacheAllWaves = false;
  static constexpr bool CacheNodeMajor = kCacheNodeMajor;
  static constexpr int NodeStride = kNodeStride;
  // SinglePoller uses a device broadcast; StageFeatures stages mapped input.
  static constexpr bool SinglePoller = false;
  static constexpr bool StageFeatures = false;
  // SeqFlag: true = request_seq, false = feature-buffer sentinel.
  static constexpr bool SeqFlag = true;
  // Threads per tree for bitmask traversal.
  static constexpr int TPT = kTPT;
  static_assert(!kCacheNodeMajor || kCached,
                "node-major caching requires Cached");
  static_assert(!kCacheNodeMajor || kBitmask,
                "node-major caching is only supported for Bitmask");
  static_assert(kNodeStride == 1 || (kBitmask && kTPT > 1),
                "NodeStride only applies to TPT>1 bitmask traversal");
  static_assert(kNodeStride == 1 ||
                    tree_kernels::qs_node_stride_valid(kNodeStride),
                "NodeStride must be coprime with every supported QS node count");
};
template <int kBT, bool kTransposed, bool kBitmask, bool kCached, bool kMapped,
          bool kCrossReduce, int kTPT = 1, bool kCacheAllWaves = false,
          bool kCacheNodeMajor = false, int kNodeStride = 1>
struct CudaTraversePersistent
    : CudaTraverseOneShot<kBT, kTransposed, kBitmask, kCached, kMapped, kTPT,
                          kCacheNodeMajor, kNodeStride> {
  static constexpr bool Persistent = true;
  static constexpr bool CrossReduce = kCrossReduce;
  static constexpr bool CacheAllWaves = kCacheAllWaves;
  static_assert(!(kCacheAllWaves && !kCached));
  static_assert(!(kCacheAllWaves && kCrossReduce));
};
template <int kBT, bool kTransposed, bool kBitmask, bool kCached, bool kMapped,
          bool kCrossReduce, bool kSinglePoller, bool kStageFeatures,
          bool kSeqFlag = true, int kTPT = 1, bool kCacheAllWaves = false,
          bool kCacheNodeMajor = false, int kNodeStride = 1>
struct CudaTraversePersistentSP
    : CudaTraversePersistent<kBT, kTransposed, kBitmask, kCached, kMapped,
                             kCrossReduce, kTPT, kCacheAllWaves,
                             kCacheNodeMajor, kNodeStride> {
  static constexpr bool SinglePoller = kSinglePoller;
  static constexpr bool StageFeatures = kStageFeatures;
  static constexpr bool SeqFlag = kSeqFlag;
  static_assert(!(kSinglePoller && kCrossReduce),
                "SinglePoller path implements the !CrossReduce publish only");
  static_assert(kSeqFlag || (kSinglePoller && kStageFeatures),
                "sentinel signal (!SeqFlag) requires SinglePoller + "
                "StageFeatures");
  static_assert(kTPT == 1 || kBitmask,
                "intra-tree parallelism (TPT>1) requires Bitmask");
};

template <typename TTraversePolicy, typename TValue, typename TSpan>
class CudaWorker {
 public:
  using TraversePolicy = TTraversePolicy;
  using Value = TValue;
  using Span = TSpan;
  constexpr static Value kEps{1e-10};

  CudaWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes),
        depth_(depth),
        tree_size_(((size_t)1 << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        n_features_(config.template get<int>("num_feature")),
        eps_(config.template get<bool>("has_equal") ? kEps : Value{0}) {
    assert(n_trees_ * tree_size_ == nodes.size());
    if constexpr (TraversePolicy::Persistent)
      if (config.contains("persistent_grid"))
        grid_override_ = config.template get<int>("persistent_grid");
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
    g_request_bcast_ = std::exchange(other.g_request_bcast_, nullptr);
    d_feat_stage_ = std::exchange(other.d_feat_stage_, nullptr);
    grid_override_ = std::exchange(other.grid_override_, 0);
    stream_ = std::exchange(other.stream_, cudaStream_t{});
    full_grid_ = std::exchange(other.full_grid_, 0);
    grid_ = std::exchange(other.grid_, 0);
    max_waves_per_block_ = std::exchange(other.max_waves_per_block_, 1);
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

  // Feed one feature vector; mapped persistent policies require a stable
  // buffer.
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
      } else if constexpr (TraversePolicy::Mapped) {
        assert(feat == feat_ptr_);  // resident kernel uses fixed address
      }
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

  // Device/host resources.
  float *d_splits_ = nullptr;
  tree_kernels::fi *d_indices_ = nullptr;
  float *d_splits_T_ = nullptr;
  tree_kernels::fi *d_indices_T_ = nullptr;
  float *d_features_ = nullptr;    // !Mapped copy target
  float *d_res_ = nullptr;         // device partials / accumulator
  float *partials_pin_ = nullptr;  // host partials
  float *result_pin_ = nullptr;    // host result (summed by device threads)
  int *done_slots_pin_ = nullptr;  // signal of completion
  cuda::atomic<int, cuda::thread_scope_system> *request_seq_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_system> *done_flag_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_device> *done_counter_ = nullptr;
  cuda::atomic<int, cuda::thread_scope_device> *g_request_bcast_ =
      nullptr;                     // device request broadcast
  float *d_feat_stage_ = nullptr;  // staged features
  int grid_override_ = 0;          // persistent grid override
  cudaStream_t stream_{};
  int full_grid_ = 0;
  int grid_ = 0;
  int max_waves_per_block_ = 1;
  size_t smem_ = 0;
  int p_seq_ = 0;
  bool ok_ = false;
  bool launched_ = false;
  const float *feat_ptr_ = nullptr;

  static constexpr auto oneshot_kernel() {
    return tree_kernels::traverse_oneshot<
        TraversePolicy::BT, TraversePolicy::Transposed, TraversePolicy::Bitmask,
        TraversePolicy::Cached, TraversePolicy::TPT,
        TraversePolicy::CacheNodeMajor, TraversePolicy::NodeStride>;
  }
  static constexpr auto persistent_kernel() {
    return tree_kernels::traverse_persistent<
        TraversePolicy::BT, TraversePolicy::Transposed, TraversePolicy::Bitmask,
        TraversePolicy::Cached, TraversePolicy::CacheAllWaves,
        TraversePolicy::CrossReduce, TraversePolicy::SinglePoller,
        TraversePolicy::StageFeatures, TraversePolicy::SeqFlag,
        TraversePolicy::TPT, TraversePolicy::CacheNodeMajor,
        TraversePolicy::NodeStride>;
  }

  // make sure the memory is mapped
  void assert_mapped_(const float *p) {
    cudaPointerAttributes attr{};
    cudaError_t e = cudaPointerGetAttributes(&attr, p);
    assert(e == cudaSuccess && attr.type == cudaMemoryTypeHost &&
           "Mapped policy requires a pinned/mapped host feature buffer");
    (void)e;
  }

  // Opt into large dynamic shared memory.
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
  // Persistent grids must be fully co-resident.
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
    static_assert(tree_kernels::QS_MAX_WORDS == 1);
    std::array<uint64_t,
               tree_kernels::QS_MAX_INTERNAL * tree_kernels::QS_MAX_WORDS>
        masks{};
    size_t idx = 0;
    size_t num_per_level = 1;
    for (size_t d = 0; d < tree_kernels::QS_MAX_DEPTH;
         ++d, num_per_level <<= 1) {
      for (size_t i = 0; i < num_per_level; ++i) {
        const size_t sub_size = size_t{1} << (tree_kernels::QS_MAX_DEPTH - d);
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
      if (depth_ == 0 || depth_ > tree_kernels::QS_MAX_DEPTH) {
        ok_ = false;
        return;
      }
      upload_bitmasks_();
    }

    // One block owns BT/TPT trees.
    constexpr int kTreesPerBlock = TraversePolicy::BT / TraversePolicy::TPT;
    auto ceil_div = [](size_t n, size_t d) { return (n + d - 1) / d; };
    full_grid_ = (int)ceil_div(n_trees_, (size_t)kTreesPerBlock);
    grid_ = grid_override_ > 0 ? grid_override_ : full_grid_;
    if (grid_ <= 0 || grid_ > full_grid_) {
      ok_ = false;
      return;
    }
    max_waves_per_block_ = (int)ceil_div((size_t)full_grid_, (size_t)grid_);
    if constexpr (TraversePolicy::Cached && !TraversePolicy::CacheAllWaves) {
      if (grid_ != full_grid_) {
        ok_ = false;
        return;
      }
    }
    // Upload row-major data, and transposed data when requested.
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
      if constexpr (TraversePolicy::SinglePoller) {
        tree_kernels::cuda_check(
            cudaMalloc(&g_request_bcast_, sizeof(*g_request_bcast_)));
        tree_kernels::cuda_check(
            cudaMemset(g_request_bcast_, 0, sizeof(*g_request_bcast_)));
        if constexpr (TraversePolicy::StageFeatures)
          tree_kernels::cuda_check(
              cudaMalloc(&d_feat_stage_, n_features_ * sizeof(float)));
      }
    }
    tree_kernels::cuda_check(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    const size_t cached_trees_per_block =
        TraversePolicy::Cached
            ? (TraversePolicy::CacheAllWaves
                   ? (size_t)max_waves_per_block_ * kTreesPerBlock
                   : (size_t)kTreesPerBlock)
            : 0;
    const size_t cache_tree_stride =
        TraversePolicy::CacheNodeMajor
            ? tree_kernels::node_major_cache_tree_stride(
                  cached_trees_per_block, TraversePolicy::TPT)
            : cached_trees_per_block;
    smem_ = cache_tree_stride * tree_size_ *
                (sizeof(float) + sizeof(tree_kernels::fi)) +
            n_features_ * sizeof(float);
    const void *kern = TraversePolicy::Persistent
                           ? (const void *)persistent_kernel()
                           : (const void *)oneshot_kernel();
    ok_ = enable_smem_(kern, smem_);
    if constexpr (TraversePolicy::Persistent)
      if (ok_) ok_ = persistent_fits_();
    // Null-stream uploads must finish before non-blocking launches.
    tree_kernels::cuda_check(cudaDeviceSynchronize());
  }

  void launch_kernel_(const float *feat, float *res) {
    const float *spl = TraversePolicy::Transposed ? d_splits_T_ : d_splits_;
    const tree_kernels::fi *idx =
        TraversePolicy::Transposed ? d_indices_T_ : d_indices_;
    if constexpr (!TraversePolicy::Persistent)
      tree_kernels::traverse_oneshot<
          TraversePolicy::BT, TraversePolicy::Transposed,
          TraversePolicy::Bitmask, TraversePolicy::Cached, TraversePolicy::TPT,
          TraversePolicy::CacheNodeMajor, TraversePolicy::NodeStride>
          <<<grid_, TraversePolicy::BT, smem_, stream_>>>(
              spl, idx, (tree_kernels::fi)n_trees_, (tree_kernels::ni)depth_,
              feat, (tree_kernels::fi)n_features_, (float)eps_, res);
    else
      tree_kernels::traverse_persistent<
          TraversePolicy::BT, TraversePolicy::Transposed,
          TraversePolicy::Bitmask, TraversePolicy::Cached,
          TraversePolicy::CacheAllWaves, TraversePolicy::CrossReduce,
          TraversePolicy::SinglePoller, TraversePolicy::StageFeatures,
          TraversePolicy::SeqFlag, TraversePolicy::TPT,
          TraversePolicy::CacheNodeMajor, TraversePolicy::NodeStride>
          <<<grid_, TraversePolicy::BT, smem_, stream_>>>(
              spl, idx, (tree_kernels::fi)n_trees_, (tree_kernels::ni)depth_,
              feat, (tree_kernels::fi)n_features_, (float)eps_, res,
              done_slots_pin_, request_seq_, done_counter_, done_flag_,
              result_pin_, g_request_bcast_, d_feat_stage_,
              max_waves_per_block_);
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
      partials_pin_[b] = std::numeric_limits<double>::quiet_NaN();
    }
    tree_kernels::cuda_check(
        cudaMemset(done_counter_, 0, sizeof(*done_counter_)));
    if constexpr (TraversePolicy::SinglePoller)
      // Clear stale broadcast state before relaunch.
      tree_kernels::cuda_check(
          cudaMemset(g_request_bcast_, 0, sizeof(*g_request_bcast_)));
    tree_kernels::cuda_check(cudaMemset(d_res_, 0, n_trees_ * sizeof(float)));
    p_seq_ = 0;
    float *res = TraversePolicy::CrossReduce ? d_res_ : partials_pin_;
    launch_kernel_(feat, res);
    tree_kernels::cuda_check(cudaGetLastError());
  }

  Value drive_persistent_() {
#ifdef QLEAF_PROF_PERSIST
    // Host-side timing split; skip gaps between benchmark sub-runs.
    static std::chrono::steady_clock::time_point prof_prev{};
    static unsigned long long prof_host = 0, prof_hn = 0, prof_wait = 0,
                              prof_first = 0, prof_rest = 0, prof_n = 0;
    auto prof_ts = std::chrono::steady_clock::now();
    if (prof_n > 0) {
      auto dt = (unsigned long long)
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        prof_ts - prof_prev)
                        .count();
      if (dt < 100000) {
        prof_host += dt;
        prof_hn++;
      }
    }
#endif
    // !SeqFlag uses the feature buffer itself as the request signal.
    if constexpr (TraversePolicy::SeqFlag)
      request_seq_->store(++p_seq_, cuda::memory_order_release);
    if constexpr (TraversePolicy::CrossReduce) {
      while (done_flag_->load(cuda::memory_order_acquire) != p_seq_) {
      }
      return *result_pin_;
    } else {
#ifdef QLEAF_PROF_PERSIST
      // Split polling into first-result latency and remaining gather time.
      bool prof_seen_first = false;
      std::chrono::steady_clock::time_point prof_tf;
#endif
      bool total_done = false;
      volatile float *vpartials = partials_pin_;
      while (!total_done) {
        total_done = true;
#ifdef QLEAF_PROF_PERSIST
        int prof_cnt = 0;
        for (int b = 0; b < grid_; ++b) {
          bool d = !std::isnan(vpartials[b]);  // partial is done flag
          total_done &= d;
          prof_cnt += d;
        }
        if (!prof_seen_first && prof_cnt > 0) {
          prof_tf = std::chrono::steady_clock::now();
          prof_seen_first = true;
        }
#else
        for (int b = 0; b < grid_; ++b) {
          bool done = !std::isnan(vpartials[b]);
          total_done &= done;
        }
#endif
      }
#ifdef QLEAF_PROF_PERSIST
      auto prof_td = std::chrono::steady_clock::now();
      auto pns = [](auto a, auto b) {
        return (unsigned long long)
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                .count();
      };
      prof_wait += pns(prof_ts, prof_td);
      prof_first += pns(prof_ts, prof_tf);
      prof_rest += pns(prof_tf, prof_td);
      prof_prev = prof_td;
      if ((++prof_n & (prof_n - 1)) == 0 || prof_n % 5000 == 0)
        printf(
            "[host] reqs=%llu host_oh=%lluns first=%lluns rest=%lluns "
            "poll=%lluns\n",
            prof_n, prof_hn ? prof_host / prof_hn : 0ull, prof_first / prof_n,
            prof_rest / prof_n, prof_wait / prof_n);
#endif
      float s = 0.0f;
      for (int b = 0; b < grid_; ++b) {
        s += vpartials[b];
        vpartials[b] = std::numeric_limits<double>::quiet_NaN();
      }
      return s;
    }
  }

  void teardown_() {
    if constexpr (TraversePolicy::Persistent)
      if (launched_) {
        if constexpr (TraversePolicy::SinglePoller &&
                      !TraversePolicy::SeqFlag) {
          // !SeqFlag exits via the mapped feature buffer.
          if (feat_ptr_)
            const_cast<float *>(feat_ptr_)[0] =
                -tree_kernels::sentinel<float>();
        } else {
          request_seq_->store(-1, cuda::memory_order_release);
        }
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
      if constexpr (TraversePolicy::SinglePoller) {
        cudaFree(g_request_bcast_);
        if constexpr (TraversePolicy::StageFeatures) cudaFree(d_feat_stage_);
      }
    }
  }
};
}  // namespace qleaf
#endif
