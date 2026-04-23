#ifndef GWORKERS
#define GWORKERS
#include <qleaf.h>

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda/atomic>

namespace tree_kernels {

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
                             const fi *__restrict__ idx, fi stride, fi tree,
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
__device__ inline void stage(float *s_spl, fi *s_idx,
                             const float *__restrict__ gspl,
                             const fi *__restrict__ gidx, fi ntrees,
                             fi first_tree, fi local_ntrees, ni tree_size,
                             fi ti) {
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
      fi lt = (fi)(i / tree_size);
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
                                    float *&s_spl, fi *&s_idx) {
  if constexpr (Cached) {
    s_spl = reinterpret_cast<float *>(smem);
    s_idx =
        reinterpret_cast<fi *>(smem + (size_t)BT * tree_size * sizeof(float));
    return reinterpret_cast<float *>(smem + (size_t)BT * tree_size *
                                                (sizeof(float) + sizeof(fi)));
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
    const float *__restrict__ splits, const fi *__restrict__ indices, fi ntrees,
    ni depth, const float *__restrict__ features, fi n_features, float eps,
    float *__restrict__ res_block) {
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;

  const fi ti = threadIdx.x;
  const ni tree_size = (1u << (depth + 1)) - 1u;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  fi *s_idx;
  float *features_ = smem_split<BT, Cached>(smem, tree_size, s_spl, s_idx);

  for (fi i = ti; i < n_features; i += BT) features_[i] = features[i];

  fi local_ntrees = 0;
  if constexpr (Cached) {
    const fi first_tree = blockIdx.x * BT;
    const fi remaining = first_tree < ntrees ? ntrees - first_tree : 0;
    local_ntrees = remaining < (fi)BT ? remaining : (fi)BT;
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
    const fi tree = blockIdx.x * BT + ti;
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
    const float *__restrict__ splits, const fi *__restrict__ indices, fi ntrees,
    ni depth, const float *__restrict__ features, fi n_features, float eps,
    float *res_block, int *done_out,
    cuda::atomic<int, cuda::thread_scope_system> *request_seq,
    cuda::atomic<int, cuda::thread_scope_device> *done_counter,
    cuda::atomic<int, cuda::thread_scope_system> *done_flag,
    float *result_out) {
  static_assert(BT > 0 && BT <= 1024 && BT % 32 == 0);
  using BlockReduce = cub::BlockReduce<float, BT>;
  __shared__ typename BlockReduce::TempStorage reduce_temp;
  __shared__ bool exit_flag;

  const fi ti = threadIdx.x;
  const ni tree_size = (1u << (depth + 1)) - 1u;
  const int total_blocks = gridDim.x;

  extern __shared__ unsigned char smem[];
  float *s_spl;
  fi *s_idx;
  float *features_ = smem_split<BT, Cached>(smem, tree_size, s_spl, s_idx);

  fi local_ntrees = 0;
  if constexpr (Cached) {  // tree data is static across requests -> stage once
    const fi first_tree = blockIdx.x * BT;
    const fi remaining = first_tree < ntrees ? ntrees - first_tree : 0;
    local_ntrees = remaining < (fi)BT ? remaining : (fi)BT;
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

    for (fi i = ti; i < n_features; i += BT) features_[i] = features[i];
    __syncthreads();

    float result = 0.0f;
    if constexpr (Cached) {
      if (ti < local_ntrees)
        result = walk<false, Bitmask>(s_spl, s_idx, tree_size, ti, depth,
                                      features_, eps);
    } else {
      const fi tree = blockIdx.x * BT + ti;
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

namespace qleaf {}
#endif