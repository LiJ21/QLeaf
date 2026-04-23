#ifndef INCLUDE_CONFIG
#define INCLUDE_CONFIG
#include <nlohmann/json.hpp>
#include <span>
#include <string_view>

namespace qleaf {

class Config {
public:
  explicit Config(const nlohmann::json &node) : node_(node) {}

  template <typename T> T get(std::string_view key) const {
    return node_.at(key).get<T>();
  }

  Config get(std::string_view key) const { return Config{node_.at(key)}; }
  template <typename T> T get() const { return node_.template get<T>(); }
  Config operator[](size_t i) const { return Config{node_.at(i)}; }

  struct Iterator {
    nlohmann::json::const_iterator it;
    Config operator*() const { return Config{*it}; }
    Iterator &operator++() {
      ++it;
      return *this;
    }
    bool operator!=(const Iterator &o) const { return it != o.it; }
  };

  Iterator begin() const { return {node_.begin()}; }
  Iterator end() const { return {node_.end()}; }

  size_t size() const { return node_.size(); }

private:
  const nlohmann::json &node_;
};

} // namespace qleaf
#endif