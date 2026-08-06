#pragma once

#include <optional>

#include <nlohmann/json.hpp>

// GCC 16 implements C++26's optional range support. Without an explicit
// serializer, nlohmann can therefore treat std::optional<T> as a JSON array:
// an absent value becomes [] and a present value becomes [value]. TrainVM
// protocols define optionals as null/scalar values, never ranges. Keep that
// rule in the one header through which native code may use nlohmann JSON.
namespace nlohmann {

template <typename T>
struct adl_serializer<std::optional<T>> {
  static void to_json(json& output, const std::optional<T>& input) {
    if (input.has_value()) {
      output = *input;
    } else {
      output = nullptr;
    }
  }

  static void from_json(const json& input, std::optional<T>& output) {
    if (input.is_null()) {
      output.reset();
    } else {
      output = input.template get<T>();
    }
  }
};

}  // namespace nlohmann
