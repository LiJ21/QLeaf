#ifndef INCLUDE_QLEAF
#define INCLUDE_QLEAF
#include <algorithm>
#include <atomic>
#include <bitset>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <format>
#include <ranges>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#ifdef __GNUC__
#include <sys/cdefs.h>
#include <sys/types.h>
#endif
#include <iostream>
#include <vector>

// a fast forest inference framework for shallow and perfect trees
namespace qleaf {

namespace detail {
class FairBalancer {
public:
  void init(size_t num_trees, size_t num_workers) {
    q = num_trees / num_workers;
    r = num_trees % num_workers;
  }
  size_t start(size_t index) {
    return index < r ? index * (q + 1) : r * (q + 1) + (index - r) * q;
  }
  size_t len(size_t index) { return index < r ? q + 1 : q; }

private:
  size_t q{}, r{};
};

// branching tool
template <typename T> inline T to_left(T idx) { return 2 * (idx + 1) - 1; }
template <typename T> inline T to_right(T idx) { return 2 * (idx + 1); }

// leaf mask tool
template <typename TMask> consteval auto get_mask_full() {
  TMask ret;
  ret.set();
  return ret;
}

template <size_t tMaxDepth> consteval auto get_mask() {
  constexpr size_t kMaxDepth{tMaxDepth};
  constexpr size_t kMaxTreeSize{(1 << (tMaxDepth + 1)) - 1};
  constexpr size_t kMaxLeaves{1 << kMaxDepth};
  using Mask = std::bitset<kMaxLeaves>;
  std::array<Mask, kMaxTreeSize - kMaxLeaves> mask;
  auto mask_full = get_mask_full<Mask>();
  size_t num_per_level = 1uz;
  size_t idx = 0;
  for (size_t d = 0; d < kMaxDepth; ++d, num_per_level <<= 1) {
    for (size_t i = 0; i < num_per_level; ++i) {
      auto sub_size = 1uz << (kMaxDepth - d);
      auto len = sub_size >> 1;
      auto start = sub_size * i;
      mask[idx++] = ~((mask_full >> (kMaxLeaves - len)) << start);
    }
  }
  return mask;
}

template <size_t tMaxDepth> size_t leaf_mask_to_index(auto &&mask, auto depth) {
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
} // namespace detail

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

template <typename TValue> struct TreeNode {
  using Value = TValue;
  size_t idx;
  Value split;
};

template <typename TValue> class TreeNodeBuffer {
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

template <typename TValue> class CompactNodeBuffer {
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
    size_t tree_size = (1uz << (depth_ + 1)) - 1;
    n_trees_ = config.get("trees").size();
    nodes_.reserve(n_trees_ * tree_size);
    for (const auto &tree_config : config.get("trees")) {
      const auto &splits = tree_config.get("splits");
      const auto &indices = tree_config.get("indices");
      for (size_t i = 0; i < tree_size; ++i) {
        nodes_.emplace_back(indices[i].template get<size_t>(),
                            splits[i].template get<Value>());
      }
    }
    LoadBalancer load_balancer;
    auto config_workers = config.get("worker");
    auto n_workers = config_workers.size();
    load_balancer.init(n_trees_, n_workers);

    for (auto i : std::views::iota(0uz, n_workers)) {
      workers_.emplace_back(config_workers[i], depth_,
                            nodes_.span(load_balancer.start(i) * tree_size,
                                        load_balancer.len(i) * tree_size));
    }
    results_ = std::vector<Value>(n_workers, {});
  }

  auto predict(const auto &features) {
    for (auto &w : workers_) {
      w.predict(features);
    }
    for (auto i : std::views::iota(0uz, workers_.size())) {
      results_[i] = workers_[i].get();
    }
    return Reducer{}(std::span<Value>(results_));
  }

private:
  std::vector<Worker> workers_;
  NodeBuffer nodes_;
  std::vector<Value> results_;
  size_t depth_;
  size_t n_trees_;
};

template <typename TWorker> class ThreadedWorker {
public:
  using Worker = TWorker;
  using Value = typename Worker::Value;

  ThreadedWorker(auto &&config, size_t depth, auto &&nodes)
      : worker_(std::forward<decltype(config)>(config), depth,
                std::forward<decltype(nodes)>(nodes)) {
    features_.store(nullptr, std::memory_order_relaxed);
    size_.store(0, std::memory_order_relaxed);
    ready_.store(false, std::memory_order_relaxed);

    auto do_work = [this](std::stop_token st) {
      while (!st.stop_requested()) {
        const Value *features{};
        if ((features = this->features_.load(std::memory_order_acquire))) {
          size_t size = this->size_.load(std::memory_order_acquire);
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

template <typename TValue, typename TSpan> class BranchRegressionWorker {
public:
  using Value = TValue;
  using Span = TSpan;
  constexpr static Value kEps{1e-10};
  BranchRegressionWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes), depth_(depth), tree_size_((1uz << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        eps_(config.template get<bool>("has_equal") ? kEps : 0) {
    assert(n_trees_ * tree_size_ == nodes.size());
  }

  void predict(const auto &fts) {
    result_ = 0;
    const Value *
#ifdef __GNUC__
        __restrict__
#endif
        features = fts.data();

    const size_t nnodes = nodes_.size();

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

struct RegressionReducer {
  template <typename TValue> TValue operator()(std::span<TValue> results) {
    return std::ranges::fold_left(results, TValue{}, std::plus<>());
  }
};

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
  // mask can be precomputed in compile time for perfect trees
  constexpr static auto kMasks = detail::get_mask<kMaxDepth>();
  BitmaskRegressionWorker(auto &&config, size_t depth, Span nodes)
      : nodes_(nodes), depth_(depth), tree_size_((1uz << (depth_ + 1)) - 1),
        n_trees_(nodes.size() / tree_size_),
        eps_(config.template get<bool>("has_equal") ? kEps : 0) {
    assert(n_trees_ * tree_size_ == nodes.size());
  }

  void predict(const auto &fts) {
    result_ = 0;
    const Value *
#ifdef __GNUC__
        __restrict__
#endif
        features = fts.data();

    const size_t nnodes = nodes_.size();

    auto internal_num = tree_size_ - (1 << depth_);
    auto ones = detail::get_mask_full<Mask>();
    for (size_t base = 0; base < nnodes; base += tree_size_) {
      auto mask = ones;
      for (size_t i = 0; i < internal_num; ++i) {
        size_t idx = base + i;
        mask &= features[nodes_.idx(idx)] < nodes_.split(idx) + kEps
                    ? ones
                    : kMasks[i];
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
};
} // namespace qleaf
#endif