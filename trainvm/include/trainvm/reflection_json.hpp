#pragma once

#include <meta>

#include <concepts>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

struct Diagnostic {
  enum class Severity { info, warning, error };

  Severity severity{Severity::error};
  std::string code;
  std::string path;
  std::string message;
};

template <typename T>
struct is_optional : std::false_type {};
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {
  using value_type = T;
};
template <typename T>
inline constexpr bool is_optional_v = is_optional<T>::value;

template <typename T>
struct is_vector : std::false_type {};
template <typename T, typename Allocator>
struct is_vector<std::vector<T, Allocator>> : std::true_type {
  using value_type = T;
};

template <typename T>
struct is_string_map : std::false_type {};
template <typename T, typename Compare, typename Allocator>
struct is_string_map<std::map<std::string, T, Compare, Allocator>> : std::true_type {
  using mapped_type = T;
};

inline std::string child_path(const std::string& parent, std::string_view child) {
  if (parent.empty()) {
    return "/" + std::string(child);
  }
  return parent + "/" + std::string(child);
}

consteval std::string_view json_field_name(std::meta::info member) {
  const std::string_view name = std::meta::identifier_of(member);
  // C++ keywords cannot be member identifiers. Keep the exception centralized
  // and compile-time-visible until reflection annotations carry schema aliases.
  if (name == "default_value") {
    return "default";
  }
  return name;
}

template <typename Enum>
  requires std::is_enum_v<Enum>
std::optional<Enum> enum_from_string(std::string_view value) {
  static constexpr auto enumerators =
      std::define_static_array(std::meta::enumerators_of(^^Enum));
  template for (constexpr auto enumerator : enumerators) {
    std::string_view name = std::meta::identifier_of(enumerator);
    if constexpr (std::same_as<Enum, ControlType>) {
      if (name == "enumeration") {
        name = "enum";
      }
    }
    if (name == value) {
      return [:enumerator:];
    }
  }
  return std::nullopt;
}

template <typename Enum>
  requires std::is_enum_v<Enum>
std::string enum_to_string(Enum value) {
  static constexpr auto enumerators =
      std::define_static_array(std::meta::enumerators_of(^^Enum));
  template for (constexpr auto enumerator : enumerators) {
    if (value == [:enumerator:]) {
      std::string name(std::meta::identifier_of(enumerator));
      if constexpr (std::same_as<Enum, ControlType>) {
        if (name == "enumeration") {
          return "enum";
        }
      }
      return name;
    }
  }
  return "<invalid>";
}

template <typename T>
bool decode_json(const nlohmann::json& input, T& output, const std::string& path,
                 std::vector<Diagnostic>& diagnostics);

template <typename T>
bool decode_reflected_object(const nlohmann::json& input, T& output, const std::string& path,
                             std::vector<Diagnostic>& diagnostics) {
  if (!input.is_object()) {
    diagnostics.push_back({Diagnostic::Severity::error, "type.object", path,
                           "expected an object"});
    return false;
  }

  bool ok = true;
  constexpr auto context = std::meta::access_context::unchecked();
  static constexpr auto members = std::define_static_array(
      std::meta::nonstatic_data_members_of(^^T, context));

  for (auto iterator = input.begin(); iterator != input.end(); ++iterator) {
    bool known = false;
    template for (constexpr auto member : members) {
      if (iterator.key() == json_field_name(member)) {
        known = true;
      }
    }
    if (!known) {
      diagnostics.push_back({Diagnostic::Severity::error, "field.unknown",
                             child_path(path, iterator.key()), "unknown field"});
      ok = false;
    }
  }

  template for (constexpr auto member : members) {
    constexpr std::string_view name = json_field_name(member);
    using Member = std::remove_cvref_t<decltype(output.[:member:])>;
    const auto iterator = input.find(name);
    if (iterator == input.end()) {
      // std::optional is the ONLY thing that makes a member omissible. A
      // default member initializer does not: `bool enabled{}` is required.
      // The message says so because the failure is otherwise misread — the
      // document passed JSON Schema, which may well have declared the field
      // optional, and every previously valid document starts failing at once
      // when a non-optional member is added.
      if constexpr (!is_optional_v<Member>) {
        diagnostics.push_back(
            {Diagnostic::Severity::error, "field.required", child_path(path, name),
             "required field is missing (a reflected member is omissible only "
             "if it is std::optional; a default initializer does not make it "
             "optional)"});
        ok = false;
      }
    } else if (!decode_json(*iterator, output.[:member:], child_path(path, name), diagnostics)) {
      ok = false;
    }
  }
  return ok;
}

template <typename T>
bool decode_json(const nlohmann::json& input, T& output, const std::string& path,
                 std::vector<Diagnostic>& diagnostics) {
  if constexpr (std::same_as<T, nlohmann::json>) {
    output = input;
    return true;
  } else if constexpr (std::same_as<T, std::string>) {
    if (!input.is_string()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.string", path,
                             "expected a string"});
      return false;
    }
    output = input.get<std::string>();
    return true;
  } else if constexpr (std::same_as<T, bool>) {
    if (!input.is_boolean()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.boolean", path,
                             "expected a boolean"});
      return false;
    }
    output = input.get<bool>();
    return true;
  } else if constexpr (std::integral<T> && !std::same_as<T, bool>) {
    if (!input.is_number_integer()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.integer", path,
                             "expected an integer"});
      return false;
    }
    // nlohmann narrows silently on get<T>() rather than throwing, so a wider
    // wire value is otherwise truncated into a valid-looking one: uid
    // 4294967296 would decode to 0. Range-check against T before assigning.
    const bool representable =
        input.is_number_unsigned()
            ? std::in_range<T>(input.get<std::uint64_t>())
            : std::in_range<T>(input.get<std::int64_t>());
    if (!representable) {
      diagnostics.push_back({Diagnostic::Severity::error, "number.range", path,
                             "integer is outside the supported range"});
      return false;
    }
    output = input.get<T>();
    return true;
  } else if constexpr (std::floating_point<T>) {
    if (!input.is_number()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.number", path,
                             "expected a number"});
      return false;
    }
    output = input.get<T>();
    return true;
  } else if constexpr (std::is_enum_v<T>) {
    if (!input.is_string()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.enum", path,
                             "expected an enum string"});
      return false;
    }
    auto value = enum_from_string<T>(input.get<std::string>());
    if (!value.has_value()) {
      diagnostics.push_back({Diagnostic::Severity::error, "enum.unknown", path,
                             "unknown enum value: " + input.get<std::string>()});
      return false;
    }
    output = *value;
    return true;
  } else if constexpr (is_optional_v<T>) {
    using Value = typename is_optional<T>::value_type;
    Value value{};
    if (!decode_json(input, value, path, diagnostics)) {
      return false;
    }
    output = std::move(value);
    return true;
  } else if constexpr (is_vector<T>::value) {
    if (!input.is_array()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.array", path,
                             "expected an array"});
      return false;
    }
    using Value = typename is_vector<T>::value_type;
    bool ok = true;
    output.clear();
    output.reserve(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
      Value value{};
      if (!decode_json(input[index], value, child_path(path, std::to_string(index)), diagnostics)) {
        ok = false;
      }
      output.push_back(std::move(value));
    }
    return ok;
  } else if constexpr (is_string_map<T>::value) {
    if (!input.is_object()) {
      diagnostics.push_back({Diagnostic::Severity::error, "type.map", path,
                             "expected an object map"});
      return false;
    }
    using Value = typename is_string_map<T>::mapped_type;
    bool ok = true;
    output.clear();
    for (auto iterator = input.begin(); iterator != input.end(); ++iterator) {
      Value value{};
      if (!decode_json(iterator.value(), value, child_path(path, iterator.key()), diagnostics)) {
        ok = false;
      }
      output.emplace(iterator.key(), std::move(value));
    }
    return ok;
  } else if constexpr (std::is_class_v<T>) {
    return decode_reflected_object(input, output, path, diagnostics);
  } else {
    static_assert(std::is_same_v<T, void>, "unsupported reflected JSON field type");
  }
}

template <typename T>
nlohmann::json encode_json(const T& input);

template <typename T>
nlohmann::json encode_reflected_object(const T& input) {
  nlohmann::json output = nlohmann::json::object();
  constexpr auto context = std::meta::access_context::unchecked();
  static constexpr auto members = std::define_static_array(
      std::meta::nonstatic_data_members_of(^^T, context));
  template for (constexpr auto member : members) {
    using Member = std::remove_cvref_t<decltype(input.[:member:])>;
    if constexpr (is_optional_v<Member>) {
      if (input.[:member:].has_value()) {
        output[std::string(json_field_name(member))] = encode_json(*input.[:member:]);
      }
    } else {
      output[std::string(json_field_name(member))] = encode_json(input.[:member:]);
    }
  }
  return output;
}

template <typename T>
nlohmann::json encode_json(const T& input) {
  if constexpr (std::same_as<T, nlohmann::json> || std::same_as<T, std::string> ||
                std::same_as<T, bool> || std::integral<T> || std::floating_point<T>) {
    return nlohmann::json(input);
  } else if constexpr (std::is_enum_v<T>) {
    return nlohmann::json(enum_to_string(input));
  } else if constexpr (is_optional_v<T>) {
    if (!input.has_value()) {
      return nullptr;
    }
    return encode_json(*input);
  } else if constexpr (is_vector<T>::value) {
    nlohmann::json output = nlohmann::json::array();
    for (const auto& value : input) {
      output.push_back(encode_json(value));
    }
    return output;
  } else if constexpr (is_string_map<T>::value) {
    nlohmann::json output = nlohmann::json::object();
    for (const auto& [key, value] : input) {
      output[key] = encode_json(value);
    }
    return output;
  } else if constexpr (std::is_class_v<T>) {
    return encode_reflected_object(input);
  } else {
    static_assert(std::is_same_v<T, void>, "unsupported reflected JSON field type");
  }
}

template <typename T>
std::vector<std::string> reflected_field_names() {
  std::vector<std::string> names;
  constexpr auto context = std::meta::access_context::unchecked();
  static constexpr auto members = std::define_static_array(
      std::meta::nonstatic_data_members_of(^^T, context));
  template for (constexpr auto member : members) {
    names.emplace_back(json_field_name(member));
  }
  return names;
}

}  // namespace trainvm
