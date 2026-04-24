#ifndef INCLUDE_QLEAF
#define INCLUDE_QLEAF
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <ranges>
#include <stop_token>
#include <thread>
#ifdef __GNUC__
#include <sys/cdefs.h>
#include <sys/types.h>
#endif
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
} // namespace detail

template <typename TValue> struct Node {
  using Value = TValue;
  std::uint16_t idx;
  Value split;
};

template <typename TValue, template <typename> typename TWorker,
          typename TLoadBalancer, typename TReducer>
class Inferrer {
public:
  using Value = TValue;
  using NodeT = Node<Value>;
  using Worker = TWorker<Value>;
  using LoadBalancer = TLoadBalancer;
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
        nodes_.emplace_back(indices[i].template get<uint16_t>(),
                            splits[i].template get<Value>());
      }
    }
    LoadBalancer load_balancer;
    auto config_workers = config.get("worker");
    auto n_workers = config_workers.size();
    load_balancer.init(n_trees_, n_workers);

    for (auto i : std::views::iota(0uz, n_workers)) {
      workers_.emplace_back(
          config_workers[i], depth_,
          std::span(nodes_.data() + load_balancer.start(i) * tree_size,
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
  std::vector<NodeT> nodes_;
  std::vector<Value> results_;
  size_t depth_;
  size_t n_trees_;
};

template <typename TWorker> class ThreadedWorker {
public:
  using Worker = TWorker;
  using Value = typename Worker::Value;

  ThreadedWorker(auto &&config, size_t depth, auto &&nodes)
      : worker_(config, depth, nodes) {
    features_.store(nullptr, std::memory_order_relaxed);
    size_.store(0, std::memory_order_relaxed);
    ready_.store(false, std::memory_order_relaxed);

    auto do_work = [this](std::stop_token st) {
      while (!st.stop_requested()) {
        const Value *features{};
        if ((features = this->features_.load(std::memory_order_acquire))) {
          size_t size = this->size_.load(std::memory_order_acquire);
          features_.store(nullptr, std::memory_order_release);
          worker_.predict(
              std::span(features, size));
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

template <typename TValue> class BranchRegressionWorker {
public:
  using Value = TValue;
  using NodeT = Node<Value>;
  constexpr static Value kEps{1e-10};
  BranchRegressionWorker(auto &&config, size_t depth, auto &&nodes)
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
    const NodeT *
#ifdef __GNUC__
        __restrict__
#endif
        nodes = nodes_.data();

    const size_t nnodes = nodes_.size();

    for (size_t base = 0; base < nnodes; base += tree_size_) {
      size_t offset = 0;
      for (size_t d = 0; d < depth_; ++d) {
        auto &n = nodes[base + offset];
        if (features[n.idx] < n.split + eps_) {
          offset = detail::to_left(offset);
        } else {
          offset = detail::to_right(offset);
        }
      }
      result_ += nodes[base + offset].split;
    }
  }

  Value get() const { return result_; }

private:
  const std::span<NodeT> nodes_;
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
} // namespace qleaf
#endif