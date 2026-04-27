// tests/compact_forest.cc — same coverage as forest.cc using CompactNodeBuffer
#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include "config.h"
#include "qleaf.h"

static nlohmann::json make_forest() {
  auto make_tree = [](std::vector<uint16_t> indices,
                      std::vector<float> splits) {
    return nlohmann::json{{"indices", indices}, {"splits", splits}};
  };
  auto t1 =
      make_tree({0, 1, 1, 0, 0, 0, 0}, {0.5f, 0.5f, 0.5f, 1.f, 2.f, 3.f, 4.f});
  auto t2 = make_tree({0, 1, 1, 0, 0, 0, 0},
                      {0.5f, 0.5f, 0.5f, 10.f, 20.f, 30.f, 40.f});
  return nlohmann::json{{"depth", 2},
                        {"trees", {t1, t2}},
                        {"worker",
                         {
                             {{"has_equal", false}},
                             {{"has_equal", false}},
                         }}};
}

// 3 trees, 2 workers: FairBalancer gives worker 0 trees 0+1 and worker 1 tree 2
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

// ─── Branch regression via CompactNodeBuffer ───

using CompactInferrer =
    qleaf::Inferrer<float, qleaf::BranchRegressionWorker,
                    qleaf::CompactNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

class CompactForestTest : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest();
    config_ = std::make_unique<qleaf::Config>(json_);
    inf_ = std::make_unique<CompactInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<CompactInferrer> inf_;
};

TEST_F(CompactForestTest, BothLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
}

TEST_F(CompactForestTest, LeftThenRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.7f}), 22.f);
}

TEST_F(CompactForestTest, RightThenLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.3f}), 33.f);
}

TEST_F(CompactForestTest, BothRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.7f}), 44.f);
}

// ─── Threaded branch regression via CompactNodeBuffer ───

class CompactThreadedForestTest : public ::testing::Test {
 protected:
  using Worker =
      qleaf::BranchRegressionWorker<float,
                                    qleaf::CompactNodeBuffer<float>::Span>;
  using Threaded = qleaf::ThreadedWorker<Worker>;

  void SetUp() override {
    json_ = make_forest();
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
    for (auto &w : workers_) w->predict(fts);
    float sum = 0;
    for (auto &w : workers_) sum += w->get();
    return sum;
  }

  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  qleaf::CompactNodeBuffer<float> nodes_;
  std::vector<std::unique_ptr<Threaded>> workers_;
};

TEST_F(CompactThreadedForestTest, BothLeft) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
}

TEST_F(CompactThreadedForestTest, LeftThenRight) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.7f}), 22.f);
}

TEST_F(CompactThreadedForestTest, RightThenLeft) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.3f}), 33.f);
}

TEST_F(CompactThreadedForestTest, BothRight) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 44.f);
}

TEST_F(CompactThreadedForestTest, SequentialPredictions) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 44.f);
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 11.f);
}

TEST_F(CompactThreadedForestTest, StressManyPredictions) {
  for (int i = 0; i < 10000; ++i) {
    float result = predict({0.3f, 0.3f});
    ASSERT_FLOAT_EQ(result, 11.f) << "Failed at iteration " << i;
  }
}

TEST_F(CompactThreadedForestTest, StressAllPaths) {
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

// ─── Bitmask (QuickScorer) regression via CompactNodeBuffer ───

using CompactBranchInferrer = CompactInferrer;

using CompactQSInferrer =
    qleaf::Inferrer<float, qleaf::BitmaskRegressionWorker,
                    qleaf::CompactNodeBuffer, qleaf::detail::FairBalancer,
                    qleaf::RegressionReducer>;

class CompactQuickScorerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest();
    config_ = std::make_unique<qleaf::Config>(json_);
    qs_ = std::make_unique<CompactQSInferrer>(*config_);
    branch_ = std::make_unique<CompactBranchInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<CompactQSInferrer> qs_;
  std::unique_ptr<CompactBranchInferrer> branch_;
};

TEST_F(CompactQuickScorerTest, BothLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
}

TEST_F(CompactQuickScorerTest, LeftThenRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.7f}), 22.f);
}

TEST_F(CompactQuickScorerTest, RightThenLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.3f}), 33.f);
}

TEST_F(CompactQuickScorerTest, BothRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.7f}), 44.f);
}

TEST_F(CompactQuickScorerTest, MatchesBranching) {
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

TEST_F(CompactQuickScorerTest, ExactSplit) {
  std::vector<float> fts = {0.5f, 0.5f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts));
}

TEST_F(CompactQuickScorerTest, SequentialPredictions) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.7f}), 44.f);
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 11.f);
}

// ─── 3-tree variants: exercises base > 0 in predict loops ───

// BothLeft=111, LeftThenRight=222, RightThenLeft=333, BothRight=444

class CompactForestTest3 : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    inf_ = std::make_unique<CompactInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<CompactInferrer> inf_;
};

TEST_F(CompactForestTest3, BothLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.3f}), 111.f);
}
TEST_F(CompactForestTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.3f, 0.7f}), 222.f);
}
TEST_F(CompactForestTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.3f}), 333.f);
}
TEST_F(CompactForestTest3, BothRight) {
  EXPECT_FLOAT_EQ(inf_->predict(std::vector<float>{0.7f, 0.7f}), 444.f);
}

class CompactThreadedForestTest3 : public ::testing::Test {
 protected:
  using Worker =
      qleaf::BranchRegressionWorker<float,
                                    qleaf::CompactNodeBuffer<float>::Span>;
  using Threaded = qleaf::ThreadedWorker<Worker>;

  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    size_t depth = config_->get<size_t>("depth");
    size_t tree_size = (1uz << (depth + 1)) - 1;
    size_t n_workers = 2;
    size_t n_trees = config_->get("trees").size();
    nodes_.reserve(n_trees * tree_size);
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
  qleaf::CompactNodeBuffer<float> nodes_;
  std::vector<std::unique_ptr<Threaded>> workers_;
};

TEST_F(CompactThreadedForestTest3, BothLeft) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.3f}), 111.f);
}
TEST_F(CompactThreadedForestTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(predict({0.3f, 0.7f}), 222.f);
}
TEST_F(CompactThreadedForestTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.3f}), 333.f);
}
TEST_F(CompactThreadedForestTest3, BothRight) {
  EXPECT_FLOAT_EQ(predict({0.7f, 0.7f}), 444.f);
}

class CompactQuickScorerTest3 : public ::testing::Test {
 protected:
  void SetUp() override {
    json_ = make_forest3();
    config_ = std::make_unique<qleaf::Config>(json_);
    qs_ = std::make_unique<CompactQSInferrer>(*config_);
    branch_ = std::make_unique<CompactBranchInferrer>(*config_);
  }
  nlohmann::json json_;
  std::unique_ptr<qleaf::Config> config_;
  std::unique_ptr<CompactQSInferrer> qs_;
  std::unique_ptr<CompactBranchInferrer> branch_;
};

TEST_F(CompactQuickScorerTest3, BothLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.3f}), 111.f);
}
TEST_F(CompactQuickScorerTest3, LeftThenRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.3f, 0.7f}), 222.f);
}
TEST_F(CompactQuickScorerTest3, RightThenLeft) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.3f}), 333.f);
}
TEST_F(CompactQuickScorerTest3, BothRight) {
  EXPECT_FLOAT_EQ(qs_->predict(std::vector<float>{0.7f, 0.7f}), 444.f);
}

TEST_F(CompactQuickScorerTest3, MatchesBranching) {
  for (const auto &fts : std::vector<std::vector<float>>{
           {0.3f, 0.3f}, {0.3f, 0.7f}, {0.7f, 0.3f}, {0.7f, 0.7f}}) {
    EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts))
        << "Mismatch at [" << fts[0] << ", " << fts[1] << "]";
  }
}

TEST_F(CompactQuickScorerTest3, ExactSplit) {
  std::vector<float> fts = {0.5f, 0.5f};
  EXPECT_FLOAT_EQ(qs_->predict(fts), branch_->predict(fts));
}
