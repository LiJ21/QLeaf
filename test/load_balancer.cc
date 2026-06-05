#include <gtest/gtest.h>

#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "qleaf.h"

namespace {

nlohmann::json make_feature_sorted_forest() {
  auto make_tree = [](std::string name, std::vector<uint16_t> indices) {
    return nlohmann::json{
        {"name", std::move(name)},
        {"indices", std::move(indices)},
        {"splits", {0.5f, 0.5f, 0.5f, 1.f, 2.f, 3.f, 4.f}},
    };
  };

  return nlohmann::json{
      {"depth", 2},
      {"trees",
       {
           make_tree("f2_f3", {2, 3, 3, 0, 0, 0, 0}),
           make_tree("f0_f1", {0, 1, 1, 0, 0, 0, 0}),
           make_tree("f7", {7, 7, 7, 0, 0, 0, 0}),
           make_tree("f0_f2", {0, 2, 2, 0, 0, 0, 0}),
       }},
      {"worker",
       {
           {{"has_equal", false}},
           {{"has_equal", false}},
           {{"has_equal", false}},
       }},
  };
}

}  // namespace

TEST(SortedFairBalancerTest, FeatureDictCompareSortsAndSplitsTrees) {
  auto forest = make_feature_sorted_forest();
  qleaf::Config config{forest};

  using Balancer =
      qleaf::detail::SortedFairBalancer<qleaf::detail::FeatureDictCompare>;
  Balancer balancer{config.get("trees"), 3};

  ASSERT_EQ(balancer.trees().size(), 4u);
  EXPECT_EQ(balancer.trees()[0].get<std::string>("name"), "f0_f1");
  EXPECT_EQ(balancer.trees()[1].get<std::string>("name"), "f0_f2");
  EXPECT_EQ(balancer.trees()[2].get<std::string>("name"), "f2_f3");
  EXPECT_EQ(balancer.trees()[3].get<std::string>("name"), "f7");

  std::vector<size_t> lengths;
  for (size_t worker = 0; worker < 3; ++worker) {
    lengths.push_back(balancer.len(worker));
  }

  EXPECT_EQ(lengths, (std::vector<size_t>{2, 1, 1}));
  EXPECT_EQ(std::accumulate(lengths.begin(), lengths.end(), size_t{}), 4u);
  EXPECT_EQ(balancer.start(0), 0u);
  EXPECT_EQ(balancer.start(1), 2u);
  EXPECT_EQ(balancer.start(2), 3u);
}

TEST(SortedFairBalancerTest, DictCompareInferrerConstructs) {
  auto forest = make_feature_sorted_forest();
  qleaf::Config config{forest};

  using DictInferrer = qleaf::Inferrer<
      float, qleaf::BranchRegressionWorker, qleaf::CompactNodeBuffer,
      qleaf::detail::SortedFairBalancer<qleaf::detail::FeatureDictCompare>,
      qleaf::RegressionReducer>;

  DictInferrer inferrer{config};
  EXPECT_FLOAT_EQ(inferrer.predict(std::vector<float>{0.f, 0.f, 0.f, 0.f, 0.f,
                                                      0.f, 0.f, 0.f}),
                  4.f);
}
