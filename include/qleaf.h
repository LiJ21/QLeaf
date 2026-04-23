#ifndef INCLUDE_QLEAF
#define INCLUDE_QLEAF
#include <cstdint>
#include <ranges>
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

inline size_t to_left(size_t idx) { return 2 * (idx + 1) - 1; }
inline size_t to_right(size_t idx) { return 2 * (idx + 1); }
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
    depth_ = config.get("depth");
    size_t tree_size = (1uz << (depth_ + 1)) - 1;
    size_t n_trees = config.get("trees").size();
    nodes_.reserve(n_trees * tree_size);
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
    load_balancer.init(n_trees, n_workers);

    for (auto i : std::views::iota(0uz, n_workers)) {
      workers_.emplace_back(
          config_workers[i], depth_,
          std::span(nodes_.data() + load_balancer.start(i) * tree_size,
                    load_balancer.len(i) * tree_size));
    }
    results_ = std::vector<Value>(n_workers, {});
  }

  auto predict(auto &&features) {
    for (auto [i, w] : std::views::enumerate(workers_)) {
      w.predict(std::forward<decltype(features)>(features), &results_[i]);
    }
    return Reducer{}(results_);
  }

private:
  std::vector<Worker> workers_;
  std::vector<NodeT> nodes_;
  std::vector<Value> results_;
  size_t depth_;
};
} // namespace qleaf
#endif