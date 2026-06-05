#ifndef INCLUDE_QLEAF
#define INCLUDE_QLEAF
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <bitset>
#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <ranges>
#include <span>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>

#include "config.h"
#ifdef __GNUC__
#include <sys/cdefs.h>
#include <sys/types.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <vector>

// Fast forest inference framework for shallow and perfect trees.
namespace qleaf {

namespace detail {
// Pin the calling thread to a single logical core. No-op off Linux.
inline void pin_to_core(int core) {
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)core;
#endif
}

struct DummyCompare {};
struct FeatureDictCompare {
  bool operator()(const auto &t1, const auto &t2) const {
    return sorted_features(t1) < sorted_features(t2);
  }

 private:
  static std::vector<size_t> sorted_features(const auto &tree) {
    auto indices = tree.get("indices");
    std::vector<size_t> features;
    features.reserve(indices.size());
    for (const auto &index : indices) {
      features.push_back(index.template get<size_t>());
    }
    std::sort(features.begin(), features.end());
    return features;
  }
};

// Sort trees by the features they use, then distribute them evenly to workers.
template <typename TCompare>
class SortedFairBalancer {
 public:
  using Compare = TCompare;
  SortedFairBalancer(const auto &trees_config, size_t num_workers)
      : trees_config_{} {
    trees_config_.reserve(trees_config.size());
    for (const auto &tree_config : trees_config) {
      trees_config_.push_back(tree_config);
    }
    if constexpr (!std::is_same_v<Compare, DummyCompare>) {
      std::sort(trees_config_.begin(), trees_config_.end(), Compare{});
    }
    auto num_trees = trees_config_.size();
    q_ = num_trees / num_workers;
    r_ = num_trees % num_workers;
  }
  size_t start(size_t index) {
    return index < r_ ? index * (q_ + 1) : r_ * (q_ + 1) + (index - r_) * q_;
  }
  size_t len(size_t index) { return index < r_ ? q_ + 1 : q_; }

  const auto &trees() const { return trees_config_; }

 private:
  size_t q_{}, r_{};
  std::vector<Config> trees_config_;
};

using FairBalancer = SortedFairBalancer<DummyCompare>;

// Leaf mask tools.
#if defined(__CUDACC__)
#define QLEAF_MASK_EVAL
#else
#define QLEAF_MASK_EVAL consteval
#endif

// Get the mask with all bits set.
template <typename TMask>
QLEAF_MASK_EVAL auto get_mask_full() {
  TMask ret;
  ret.set();
  return ret;
}

// Get the QuickScorer (QS) mask for a perfect tree.
template <size_t tMaxDepth>
QLEAF_MASK_EVAL auto get_mask() {
  constexpr size_t kMaxDepth{tMaxDepth};
  constexpr size_t kMaxTreeSize{(1 << (tMaxDepth + 1)) - 1};
  constexpr size_t kMaxLeaves{1 << kMaxDepth};
  using Mask = std::bitset<kMaxLeaves>;
  std::array<Mask, kMaxTreeSize - kMaxLeaves> mask;
  auto mask_full = get_mask_full<Mask>();
  size_t num_per_level = size_t{1};
  size_t idx = 0;
  for (size_t d = 0; d < kMaxDepth; ++d, num_per_level <<= 1) {
    for (size_t i = 0; i < num_per_level; ++i) {
      auto sub_size = size_t{1} << (kMaxDepth - d);
      auto len = sub_size >> 1;
      auto start = sub_size * i;
      mask[idx++] = ~((mask_full >> (kMaxLeaves - len)) << start);
    }
  }
  return mask;
}

#undef QLEAF_MASK_EVAL

// Convert the final QS mask, which has a single bit set, to the leaf index.
template <size_t tMaxDepth>
size_t leaf_mask_to_index(auto &&mask, auto depth) {
  constexpr size_t kMaxDepth{tMaxDepth};
  size_t bit;
  if constexpr (kMaxDepth <= 6) {
    bit = std::countr_zero(mask.to_ullong());
  } else {
    bit = mask._Find_first();
  }
  auto leaf = bit >> (kMaxDepth - depth);
  return leaf;
}
}  // namespace detail

// Concept for node buffers.
template <typename TNodeBuffer, typename TValue>
concept CNodeBuffer =
    requires(TNodeBuffer nodes, typename TNodeBuffer::Span span, size_t start,
             size_t len, TValue split) {
      { nodes.span(start, len) } -> std::same_as<typename TNodeBuffer::Span>;
      { nodes.split(start) } -> std::same_as<TValue>;
      { nodes.idx(start) } -> std::convertible_to<size_t>;
      { span.split(start) } -> std::same_as<TValue>;
      { span.idx(start) } -> std::convertible_to<size_t>;
      { nodes.emplace_back(start, split) } -> std::same_as<void>;
      { nodes.reserve(len) } -> std::same_as<void>;
    };

// Array-of-structs node buffer.
template <typename TValue>
struct TreeNode {
  using Value = TValue;
  TreeNode(size_t node_idx, Value split_value)
      : idx(node_idx), split(split_value) {}
  size_t idx;
  Value split;
};

template <typename TValue>
class TreeNodeBuffer {
 public:
  using Value = TValue;
  using NodeT = TreeNode<Value>;
  Value split(size_t i) const { return buffer_[i].split; }
  size_t idx(size_t i) const { return buffer_[i].idx; }
  class Span {
   public:
    Span(const NodeT *buffer, size_t len) : buffer_(buffer), size_(len) {}
    Value split(size_t i) const { return buffer_[i].split; }
    size_t idx(size_t i) const { return buffer_[i].idx; }
    size_t size() const { return size_; }

   private:
    const NodeT *
#ifdef __GNUC__
        __restrict__
#endif
        buffer_;
    size_t size_;
  };

  Span span(size_t start, size_t len) const {
    return Span(buffer_.data() + start, len);
  }

  void emplace_back(size_t idx, Value split) {
    buffer_.emplace_back(idx, split);
  }

  void reserve(size_t n) { buffer_.reserve(n); }

 private:
  std::vector<NodeT> buffer_;
};

// Struct-of-arrays node buffer.
template <typename TValue>
class CompactNodeBuffer {
 public:
  using Value = TValue;
  Value split(size_t i) const { return splits_[i]; }
  size_t idx(size_t i) const { return indices_[i]; }
  class Span {
   public:
    Span(const Value *splits, const size_t *indices, size_t len)
        : splits_(splits), indices_(indices), size_(len) {}
    Value split(size_t i) const { return splits_[i]; }
    size_t idx(size_t i) const { return indices_[i]; }
    size_t size() const { return size_; }

   private:
    const Value *
#ifdef __GNUC__
        __restrict__
#endif
        splits_;
    const size_t *
#ifdef __GNUC__
        __restrict__
#endif
        indices_;
    size_t size_;
  };

  Span span(size_t start, size_t len) const {
    return Span(splits_.data() + start, indices_.data() + start, len);
  }

  void emplace_back(size_t idx, Value split) {
    indices_.emplace_back(idx);
    splits_.emplace_back(split);
  }

  void reserve(size_t n) {
    splits_.reserve(n);
    indices_.reserve(n);
  }

 private:
  std::vector<Value> splits_;
  std::vector<size_t> indices_;
};

// Main inferrer interface, with policy-based design for flexibility.
template <typename TValue, template <typename, typename> typename TWorker,
          template <typename> typename TNodeBuffer, typename TLoadBalancer,
          typename TReducer>
  requires CNodeBuffer<TNodeBuffer<TValue>, TValue>
class Inferrer {
 public:
  using Value = TValue;
  using LoadBalancer = TLoadBalancer;
  using NodeBuffer = TNodeBuffer<Value>;
  using Worker = TWorker<Value, typename NodeBuffer::Span>;
  using Reducer = TReducer;
  Inferrer(auto &&config) {
    depth_ = config.template get<size_t>("depth");
    size_t tree_size = (size_t{1} << (depth_ + 1)) - 1;
    n_trees_ = config.get("trees").size();
    nodes_.reserve(n_trees_ * tree_size);

    auto config_workers = config.get("worker");
    auto n_workers = config_workers.size();

    LoadBalancer load_balancer(config.get("trees"), n_workers);
    for (const auto &tree_config : load_balancer.trees()) {
      const auto &splits = tree_config.get("splits");
      const auto &indices = tree_config.get("indices");
      for (size_t i = 0; i < tree_size; ++i) {
        nodes_.emplace_back(indices[i].template get<size_t>(),
                            splits[i].template get<Value>());
      }
    }

    workers_.reserve(n_workers);

    for (auto i : std::views::iota(size_t{0}, n_workers)) {
      workers_.push_back(std::make_unique<Worker>(
          config_workers[i], depth_,
          nodes_.span(load_balancer.start(i) * tree_size,
                      load_balancer.len(i) * tree_size)));
    }
    results_ = std::vector<Value>(n_workers, {});
  }

  auto predict(const auto &features) {
    for (auto &w : workers_) {
      w->predict(features);
    }
    for (auto i : std::views::iota(size_t{0}, workers_.size())) {
      results_[i] = workers_[i]->get();
    }
    return Reducer{}(std::span<Value>(results_));
  }

 private:
  NodeBuffer nodes_;
  // Pointers avoid std::vector's movable-value requirement.
  std::vector<std::unique_ptr<Worker>> workers_;
  // This may add small overhead on the CPU dispatch path, but branch prediction
  // should optimize it away in practice.
  std::vector<Value> results_;
  size_t depth_;
  size_t n_trees_;
};

// Threaded wrapper for a general worker.
template <typename TWorker>
class ThreadedWorker {
 public:
  using Worker = TWorker;
  using Value = typename Worker::Value;

  ThreadedWorker(auto &&config, size_t depth, auto &&nodes)
      : worker_(std::forward<decltype(config)>(config), depth,
                std::forward<decltype(nodes)>(nodes)) {
    features_.store(nullptr, std::memory_order_relaxed);
    size_.store(0, std::memory_order_relaxed);
    ready_.store(false, std::memory_order_relaxed);

    int core = config.contains("core") ? config.template get<int>("core") : -1;

    // do_work captures this, making move/copy construction difficult.
    auto do_work = [this, core](std::stop_token st) {
      if (core >= 0) detail::pin_to_core(core);
      while (!st.stop_requested()) {
        const Value *features{};
        if ((features = this->features_.load(std::memory_order_acquire))) {
          size_t size = this->size_.load(std::memory_order_acquire);
          // features_ pointer is used as the trigger, resetting it to nullptr
          // unarms the request (assuming the client waits for the request to
          // finish before sending the next one).
          features_.store(nullptr, std::memory_order_release);
          worker_.predict(std::span(features, size));
          ready_.store(true, std::memory_order_release);
        }
      }
    };

    thread_ = std::jthread(do_work);
  }

  void predict(const auto &fts) {
    size_.store(fts.size(), std::memory_order_relaxed);
    features_.store(fts.data(), std::memory_order_release);
  }

  Value get() {
    while (!ready_.load(std::memory_order_acquire)) {
#ifdef PAUSE_SPIN
#if defined(__x86_64__) || defined(_M_X64)
      _mm_pause();
#elif defined(__aarch64__)
      __asm__ __volatile__("yield" ::: "memory");
#endif
#endif
    }
    Value ret = worker_.get();
    ready_.store(false, std::memory_order_release);
    return ret;
  }

 private:
  std::atomic<const Value *> features_;
  std::atomic<size_t> size_;
  std::atomic<bool> ready_;
  Worker worker_;
  std::jthread thread_;
};

// CPU path-chasing worker.
template <typename TValue, typename TSpan>
class BranchRegressionWorker {
 public:
  using Value = TValue;
  using Span = TSpan;
  constexpr static Value kEps{1e-10};
  BranchRegressionWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes),
        depth_(depth),
        tree_size_((size_t{1} << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        eps_(config.template get<bool>("has_equal") ? kEps : 0) {
    assert(n_trees_ * tree_size_ == nodes.size());
  }

  void predict(const auto &fts) {
    result_ = 0;
    // Declare no aliasing; important for compiler optimization (SIMD/AVX).
    const Value *
#ifdef __GNUC__
        __restrict__
#endif
        features = fts.data();

    const size_t nnodes = nodes_.size();

    // Branchless traversal; generally compiled to SIMD/AVX instructions.
    for (size_t base = 0; base < nnodes; base += tree_size_) {
      size_t offset = 0;
      for (size_t d = 0; d < depth_; ++d) {
        auto idx = base + offset;
        size_t to_left = features[nodes_.idx(idx)] < nodes_.split(idx) + eps_;
        offset = 2 * (offset + 1) - to_left;
      }
      result_ += nodes_.split(base + offset);
    }
  }

  Value get() const { return result_; }

 private:
  const Span nodes_;
  const size_t depth_;
  const size_t tree_size_;
  const size_t n_trees_;
  const Value eps_{};
  Value result_{};
};

// Simple cross-worker reducer for regression.
struct RegressionReducer {
  template <typename TValue>
  TValue operator()(std::span<TValue> results) {
    TValue ret{};
    for (auto value : results) ret += value;
    return ret;
  }
};

// CPU bitmask (QuickScorer, QS) worker.
constexpr static size_t kDefaultMaxDepth{6};
template <typename TValue, typename TSpan, size_t tMaxDepth = kDefaultMaxDepth>
class BitmaskRegressionWorker {
 public:
  using Value = TValue;
  using Span = TSpan;
  constexpr static size_t kMaxDepth{tMaxDepth};
  constexpr static size_t kMaxTreeSize{(1 << (tMaxDepth + 1)) - 1};
  constexpr static size_t kMaxLeaves{1 << tMaxDepth};
  using Mask = std::bitset<kMaxLeaves>;
  constexpr static Value kEps{1e-10};
  // Masks can be precomputed at compile time for perfect trees; this requires
  // C++23.
#if !defined(__CUDACC__)
  constexpr static auto kMasks = detail::get_mask<kMaxDepth>();
#endif
  BitmaskRegressionWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes),
        depth_(depth),
        tree_size_((size_t{1} << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        eps_(config.template get<bool>("has_equal") ? kEps : 0) {
    assert(n_trees_ * tree_size_ == nodes.size());
  }

  void predict(const auto &fts) {
    result_ = 0;
    // Declare no aliasing; important for compiler optimization (SIMD/AVX).
    const Value *
#ifdef __GNUC__
        __restrict__
#endif
        features = fts.data();

    const size_t nnodes = nodes_.size();

    auto internal_num = tree_size_ - (1 << depth_);
    auto ones = detail::get_mask_full<Mask>();
    const auto &masks = get_masks();
    for (size_t base = 0; base < nnodes; base += tree_size_) {
      auto mask = ones;
      for (size_t i = 0; i < internal_num; ++i) {
        size_t idx = base + i;
        mask &= features[nodes_.idx(idx)] < nodes_.split(idx) + eps_ ? ones
                                                                     : masks[i];
      }
      auto leaf_idx = detail::leaf_mask_to_index<kMaxDepth>(mask, depth_);
      result_ += nodes_.split(base + internal_num + leaf_idx);
    }
  }

  Value get() const { return result_; }

 private:
  const Span nodes_;
  const size_t depth_;
  const size_t tree_size_;
  const size_t n_trees_;
  const Value eps_{};
  Value result_{};

  static const auto &get_masks() {
#if defined(__CUDACC__)
    static const auto masks = detail::get_mask<kMaxDepth>();
    return masks;
#else
    return kMasks;
#endif
  }
};
}  // namespace qleaf
#endif
