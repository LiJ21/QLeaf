// tests/test_inferrer.cpp
#include "config.h"
#include "qleaf.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

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
                             {{"has_equal", false}}, // worker 0 gets tree 0
                             {{"has_equal", false}}  // worker 1 gets tree 1
                         }}};
}

using Inferrer =
    qleaf::Inferrer<float, qleaf::BranchRegressionWorker,
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