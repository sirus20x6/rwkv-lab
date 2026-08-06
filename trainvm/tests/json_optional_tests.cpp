#include "trainvm/json.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

}  // namespace

int main() {
  const std::optional<std::int64_t> absent;
  const std::optional<std::int64_t> present = 17;
  const std::optional<std::string> text = "worker";

  const nlohmann::json document{{"absent", absent},
                                {"present", present},
                                {"text", text}};
  require(document.at("absent").is_null(),
          "an absent optional did not encode as null");
  require(document.at("present").is_number_integer() &&
              document.at("present") == 17,
          "a present optional integer did not encode as a scalar");
  require(document.at("text").is_string() && document.at("text") == "worker",
          "a present optional string did not encode as a scalar");

  const std::vector<std::optional<std::int64_t>> nullable_values{
      std::nullopt, 3, std::nullopt, 5};
  const nlohmann::json encoded_values = nullable_values;
  require(encoded_values == nlohmann::json::array({nullptr, 3, nullptr, 5}),
          "optionals nested in an array acquired range shape");

  const std::optional<std::vector<std::int64_t>> present_array =
      std::vector<std::int64_t>{2, 4};
  const std::optional<std::vector<std::int64_t>> absent_array;
  require(nlohmann::json(present_array) == nlohmann::json::array({2, 4}),
          "an optional array did not preserve its underlying value shape");
  require(nlohmann::json(absent_array).is_null(),
          "an absent optional array did not encode as null");

  require(nlohmann::json(nullptr).get<std::optional<std::int64_t>>() ==
              std::nullopt,
          "JSON null did not decode to an absent optional");
  require(nlohmann::json(29).get<std::optional<std::int64_t>>() == 29,
          "a JSON scalar did not decode to a present optional");

  return 0;
}
