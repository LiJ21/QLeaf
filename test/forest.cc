// tests/test_inferrer.cpp
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "config.h"
#include "qleaf.h"

// Tree layout (depth=2, tree_size=7):
//
//          0
//        /   \
//       1     2
//      / \   / \
//     3   4 5   6   <- leaves (split field = prediction value)
//
// to_left(i)  = 2*(i+1)-1
// to_right(i) = 2*(i+1)

static nlohmann::json make_forest() {
  // Two identical trees, each splitting on feature 0 then feature 1
  // Internal nodes: idx=feature, split=threshold
  // Leaf nodes:     split=prediction value (idx ignored)
  auto make_tree = [](std::vector<uint16_t> indices,
                      std::vector<float> splits) {
    return nlohmann::json{{"indices", indices}, {"splits", splits}};
  };

  //        f0<0.5?          f1<0.5?          f1<0.5?
  //   idx: 0  0  1  1    (leaves 3,4,5,6 have dummy idx=0)
  // Tree 1: leaves = 1, 2, 3, 4
  auto t1 =
      make_tree({0, 1, 1, 0, 0, 0, 0}, {0.5f, 0.5f, 0.5f, 1.f, 2.f, 3.f, 4.f});
  // Tree 2: same structure, leaves = 10, 20, 30, 40
  auto t2 = make_tree({0, 1, 1, 0, 0, 0, 0},
                      {0.5f, 0.5f, 0.5f, 10.f, 20.f, 30.f, 40.f});

  return nlohmann::json{{"depth", 2},
                        {"trees", {t1, t2}},
                        {"worker",
                         {
                             {{"has_equal", false}},  // worker 0 gets tree 0
                             {{"has_equal", false}}   // worker 1 gets tree 1
                         }}};
}

using Inferrer =
    qleaf::Inferrer<float, qleaf::BranchRegressionWorker, qleaf::TreeNodeBuffer,
                    qleaf::detail::FairBalancer, qleaf::RegressionReducer>;

class ForestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest();
    config_ = std::make_unique<qleaf::Config>(json_);
    inf_ = std::make_unique<Inferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<Inferrer> inf_;
};

// feature [0.3, 0.3]: f0<0.5 → left → f1<0.5 → left → leaf 3
// tree1=1, tree2=10 → sum=11
TEST_F(ForestTest, BothLeft) {
  std::vector<float> fts = {0.3f, 0.3f};
  EXPECT_FLOAT_EQ(inf_->predict(fts), 11.f);
}

// feature [0.3, 0.7]: f0<0.5 → left → f1>=0.5 → right → leaf 4
// tree1=2, tree2=20 → sum=22
TEST_F(ForestTest, LeftThenRight) {
  std::vector<float> fts = {0.3f, 0.7f};
  EXPECT_FLOAT_EQ(inf_->predict(fts), 22.f);
}

// feature [0.7, 0.3]: f0>=0.5 → right → f1<0.5 → left → leaf 5
// tree1=3, tree2=30 → sum=33
TEST_F(ForestTest, RightThenLeft) {
  std::vector<float> fts = {0.7f, 0.3f};
  EXPECT_FLOAT_EQ(inf_->predict(fts), 33.f);
}

// feature [0.7, 0.7]: f0>=0.5 → right → f1>=0.5 → right → leaf 6
// tree1=4, tree2=40 → sum=44
TEST_F(ForestTest, BothRight) {
  std::vector<float> fts = {0.7f, 0.7f};
  EXPECT_FLOAT_EQ(inf_->predict(fts), 44.f);
}

class ThreadedForestTest : public ::testing::Test {
 protected:
  using Worker =
      qleaf::BranchRegressionWorker<float, qleaf::TreeNodeBuffer<float>::Span>;
  using Threaded = qleaf::ThreadedWorker<Worker>;

  void SetUp() override {
    json_ = make_forest();
    config_ = std::make_unique<qleaf::Config>(json_);

    size_t depth = config_->get<size_t>("depth");
    size_t tree_size = (1uz << (depth + 1)) - 1;
    size_t n_trees = config_->get("trees").size();
    size_t n_workers = 2;

    qleaf::detail::FairBalancer balancer(config_->get("trees"), n_workers);
    for (const auto &tree_config : balancer.trees()) {
      const auto &splits = tree_config.get("splits");
      const auto &indices = tree_config.get("indices");
      for (size_t i = 0; i < tree_size; ++i) {
        nodes_.emplace_back(indices[i].get<size_t>(), splits[i].get<float>());
      }
    }

    auto worker_configs = config_->get("worker");
    for (auto i : std::views::iota(0uz, n_workers)) {
      workers_.push_back(
          std::make_unique<Threaded>(worker_configs[i], depth,
                                     nodes_.span(balancer.start(i) * tree_size,
                                                 balancer.len(i) * tree_size)));
    }
  }

  float predict(std::vector<float> fts) {
    // phase 1: submit to all workers
    for (auto &w : workers_) w->predict(fts);

    // phase 2: collect after all finish
    float sum = 0;
    for (auto &w : workers_) sum += w->get();

    return sum;
  }

  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  qleaf::TreeNodeBuffer<float> nodes_;
  std::vector<std::unique_ptr<Threaded>> workers_;
};

// Same four paths as non-threaded tests — verify identical results

TEST_F(ThreadedForestTest, BothLeft) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
}

TEST_F(ThreadedForestTest, LeftThenRight) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.7f}), 22.f);
}

TEST_F(ThreadedForestTest, RightThenLeft) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.3f}), 33.f);
}

TEST_F(ThreadedForestTest, BothRight) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 44.f);
}

// Sequential predictions — verify state resets correctly between calls
TEST_F(ThreadedForestTest, SequentialPredictions) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 44.f);
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
}

// Stress test — many sequential predictions to surface races
TEST_F(ThreadedForestTest, StressManyPredictions) {
  for (int i = 0; i < 10000; ++i) {
    float result = predict({0.3f, 0.3f});
    ASSERT_FLOAT_EQ(result, 11.f) << "Failed at iteration " << i;
  }
}

// All four paths in rapid succession
TEST_F(ThreadedForestTest, StressAllPaths) {
  const std::vector<std::pair<std::vector<float>, float>> cases = {
      {{0.3f, 0.3f}, 11.f},
      {{0.3f, 0.7f}, 22.f},
      {{0.7f, 0.3f}, 33.f},
      {{0.7f, 0.7f}, 44.f},
  };
  for (int i = 0; i < 10000; ++i) {
    auto &[fts, expected] = cases[i % cases.size()];
    ASSERT_FLOAT_EQ(predict(fts), expected) << "Failed at iteration " << i;
  }
}

// ─── Mask generation tests ───

TEST(MaskTest, Depth1) {
  // Tree:    node 0
  //         /      \
  //       L0       L1
  // Going right at node 0 clears left subtree {L0}
  constexpr auto masks = qleaf::detail::get_mask<1>();
  ASSERT_EQ(masks.size(), 1);
  // mask[0]: bit 0 cleared (L0), bit 1 set (L1) = 0b10
  EXPECT_FALSE(masks[0].test(0));
  EXPECT_TRUE(masks[0].test(1));
}

TEST(MaskTest, Depth2) {
  //            node 0
  //           /      \
  //       node 1    node 2
  //       /   \     /   \
  //     L0   L1   L2   L3
  constexpr auto masks = qleaf::detail::get_mask<2>();
  ASSERT_EQ(masks.size(), 3);

  // mask[0] (node 0): clears left {L0, L1} = 0b1100
  EXPECT_FALSE(masks[0].test(0));
  EXPECT_FALSE(masks[0].test(1));
  EXPECT_TRUE(masks[0].test(2));
  EXPECT_TRUE(masks[0].test(3));

  // mask[1] (node 1): clears left {L0} = 0b1110
  EXPECT_FALSE(masks[1].test(0));
  EXPECT_TRUE(masks[1].test(1));
  EXPECT_TRUE(masks[1].test(2));
  EXPECT_TRUE(masks[1].test(3));

  // mask[2] (node 2): clears left {L2} = 0b1011
  EXPECT_TRUE(masks[2].test(0));
  EXPECT_TRUE(masks[2].test(1));
  EXPECT_FALSE(masks[2].test(2));
  EXPECT_TRUE(masks[2].test(3));
}

TEST(MaskTest, Depth3AllBitsAccountedFor) {
  constexpr auto masks = qleaf::detail::get_mask<3>();
  ASSERT_EQ(masks.size(), 7);  // 2^3 - 1 internal nodes

  // AND all masks together: should clear all bits except the last leaf
  std::bitset<8> result;
  result.set();
  for (size_t i = 0; i < 7; ++i) result &= masks[i];

  // Only the rightmost leaf (L7) survives all masks
  EXPECT_EQ(result.count(), 1);
  EXPECT_TRUE(result.test(7));
}

// 3 trees, 2 workers: FairBalancer gives worker 0 trees 0+1 and worker 1 tree
// 2, so worker 0's predict loop runs with base=0 and base=tree_size.
static nlohmann::json make_forest3() {
  auto make_tree = [](std::vector<uint16_t> indices,
                      std::vector<float> splits) {
    return nlohmann::json{{"indices", indices}, {"splits", splits}};
  };
  auto t1 =
      make_tree({0, 1, 1, 0, 0, 0, 0}, {0.5f, 0.5f, 0.5f, 1.f, 2.f, 3.f, 4.f});
  auto t2 = make_tree({0, 1, 1, 0, 0, 0, 0},
                      {0.5f, 0.5f, 0.5f, 10.f, 20.f, 30.f, 40.f});
  auto t3 = make_tree({0, 1, 1, 0, 0, 0, 0},
                      {0.5f, 0.5f, 0.5f, 100.f, 200.f, 300.f, 400.f});
  return nlohmann::json{
      {"depth", 2},
      {"trees", {t1, t2, t3}},
      {"worker", {{{"has_equal", false}}, {{"has_equal", false}}}}};
}

// BothLeft=111, LeftThenRight=222, RightThenLeft=333, BothRight=444

class ForestTest3 : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    inf_ = std::make_unique<Inferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<Inferrer> inf_;
};

TEST_F(ForestTest3, BothLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.3f}), 111.f);
}
TEST_F(ForestTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.7f}), 222.f);
}
TEST_F(ForestTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.3f}), 333.f);
}
TEST_F(ForestTest3, BothRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.7f}), 444.f);
}

class ThreadedForestTest3 : public ::testing::Test {
 protected:
  using Worker =
      qleaf::BranchRegressionWorker<float, qleaf::TreeNodeBuffer<float>::Span>;
  using Threaded = qleaf::ThreadedWorker<Worker>;

  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    size_t depth = config_->get<size_t>("depth");
    size_t tree_size = (1uz << (depth + 1)) - 1;
    size_t n_trees = config_->get("trees").size();
    nodes_.reserve(n_trees * tree_size);
    size_t n_workers = 2;
    qleaf::detail::FairBalancer balancer(config_->get("trees"), n_workers);
    for (const auto &tree_config : balancer.trees()) {
      const auto &splits = tree_config.get("splits");
      const auto &indices = tree_config.get("indices");
      for (size_t i = 0; i < tree_size; ++i)
        nodes_.emplace_back(indices[i].get<size_t>(), splits[i].get<float>());
    }
    auto worker_configs = config_->get("worker");
    for (auto i : std::views::iota(0uz, n_workers))
      workers_.push_back(
          std::make_unique<Threaded>(worker_configs[i], depth,
                                     nodes_.span(balancer.start(i) * tree_size,
                                                 balancer.len(i) * tree_size)));
  }

  float predict(std::vector<float> fts) {
    for (auto &w : workers_) w->predict(fts);
    float sum = 0;
    for (auto &w : workers_) sum += w->get();
    return sum;
  }

  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  qleaf::TreeNodeBuffer<float> nodes_;
  std::vector<std::unique_ptr<Threaded>> workers_;
};

TEST_F(ThreadedForestTest3, BothLeft) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 111.f);
}
TEST_F(ThreadedForestTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.7f}), 222.f);
}
TEST_F(ThreadedForestTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.3f}), 333.f);
}
TEST_F(ThreadedForestTest3, BothRight) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 444.f);
}

using BranchInferrer = Inferrer;

using QSInferrer =
    qleaf::Inferrer<float, qleaf::BitmaskRegressionWorker,
                    qleaf::TreeNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

class QuickScorerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest();
    config_ = std::make_unique<qleaf::Config>(json_);
    qs_ = std::make_unique<QSInferrer>(*config_);
    branch_ = std::make_unique<BranchInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<QSInferrer> qs_;
  std::unique_ptr<BranchInferrer> branch_;
};

// Same four paths, verify QuickScorer matches expected values

TEST_F(QuickScorerTest, BothLeft) {
  std::vector<float> fts = {0.3f, 0.3f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), 11.f);
}

TEST_F(QuickScorerTest, LeftThenRight) {
  std::vector<float> fts = {0.3f, 0.7f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), 22.f);
}

TEST_F(QuickScorerTest, RightThenLeft) {
  std::vector<float> fts = {0.7f, 0.3f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), 33.f);
}

TEST_F(QuickScorerTest, BothRight) {
  std::vector<float> fts = {0.7f, 0.7f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), 44.f);
}

// Cross-check: QuickScorer matches Branching for all paths
TEST_F(QuickScorerTest, MatchesBranching) {
  std::vector<std::vector<float>> inputs = {
      {0.3f, 0.3f},
      {0.3f, 0.7f},
      {0.7f, 0.3f},
      {0.7f, 0.7f},
  };
  for (const auto &fts : inputs) {
    EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts))
        << "Mismatch at features [" << fts[0] << ", " << fts[1] << "]";
  }
}

// Boundary: feature exactly at split
TEST_F(QuickScorerTest, ExactSplit) {
  std::vector<float> fts = {0.5f, 0.5f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts));
}

// Sequential predictions to verify state reset
TEST_F(QuickScorerTest, SequentialPredictions) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.7f}), 44.f);
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
}

// ─── 3-tree QuickScorer: exercises base > 0 in bitmask predict loop ───

class QuickScorerTest3 : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    qs_ = std::make_unique<QSInferrer>(*config_);
    branch_ = std::make_unique<BranchInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<QSInferrer> qs_;
  std::unique_ptr<BranchInferrer> branch_;
};

TEST_F(QuickScorerTest3, BothLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 111.f);
}
TEST_F(QuickScorerTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.7f}), 222.f);
}
TEST_F(QuickScorerTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.3f}), 333.f);
}
TEST_F(QuickScorerTest3, BothRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.7f}), 444.f);
}

TEST_F(QuickScorerTest3, MatchesBranching) {
  for (const auto &fts : std::vector<std::vector<float>>{
           {0.3f, 0.3f}, {0.3f, 0.7f}, {0.7f, 0.3f}, {0.7f, 0.7f}}) {
    EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts))
        << "Mismatch at [" << fts[0] << ", " << fts[1] << "]";
  }
}

TEST_F(QuickScorerTest3, ExactSplit) {
  std::vector<float> fts = {0.5f, 0.5f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts));
}