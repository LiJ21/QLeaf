#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

struct GpuLatencyRequest {
  const nlohmann::json *forest = nullptr;
  const std::vector<std::vector<float>> *features = nullptr;
  size_t n_features = 0;
  size_t warmup = 0;
  size_t iters = 0;
  int persistent_grid = 0;
  std::string buffer = "compact";
  std::string suite = "best";
  std::vector<int> tpts;
  std::string filter;
};

struct GpuLatencyRun {
  std::string name;
  std::string status = "ok";
  std::string reason;
  std::vector<int64_t> latency_ns;
  float last_result = 0.0f;
  nlohmann::json extra = nlohmann::json::object();
  std::map<std::string, std::vector<int64_t>> phases;
};

std::vector<GpuLatencyRun> run_qleaf_gpu_latency(const GpuLatencyRequest &req);
