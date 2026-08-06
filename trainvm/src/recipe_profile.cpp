#include "trainvm/recipe_profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <ranges>
#include <regex>
#include <set>
#include <tuple>
#include <utility>

#include "trainvm/authority_document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

using Json = nlohmann::json;

constexpr std::size_t kMaximumRegistryBytes = 16U << 20U;
constexpr std::size_t kMaximumProfiles = 256U;
constexpr std::size_t kMaximumOverrides = 256U;
constexpr std::size_t kMaximumRules = 128U;
constexpr std::size_t kMaximumTuples = 512U;
constexpr std::size_t kMaximumIdentityBytes = 192U;
constexpr std::size_t kMaximumStringBytes = 4096U;
const std::regex kRunIdentity(
    "^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$");

[[noreturn]] void reject(std::string message) {
  throw RecipeProfileError(std::move(message));
}

bool symbolic_identity(std::string_view value, bool allow_leading_digit = false) {
  if (value.empty() || value.size() > kMaximumIdentityBytes) return false;
  if (!((value.front() >= 'a' && value.front() <= 'z') ||
        (value.front() >= 'A' && value.front() <= 'Z') ||
        (allow_leading_digit && value.front() >= '0' &&
         value.front() <= '9'))) {
    return false;
  }
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.' || character == ':';
  });
}

bool absolute_normalized_path(std::string_view value) {
  if (value.empty() || value.size() > kMaximumStringBytes ||
      value.starts_with("//") ||
      (value.size() > 1U && value.ends_with('/'))) {
    return false;
  }
  const std::filesystem::path path(value);
  return path.is_absolute() && path == path.lexically_normal();
}

bool path_within(const std::filesystem::path& child,
                 const std::filesystem::path& root) {
  const auto normalized_child = child.lexically_normal();
  const auto normalized_root = root.lexically_normal();
  auto child_iterator = normalized_child.begin();
  for (auto root_iterator = normalized_root.begin();
       root_iterator != normalized_root.end();
       ++root_iterator, ++child_iterator) {
    if (child_iterator == normalized_child.end() ||
        *child_iterator != *root_iterator) {
      return false;
    }
  }
  return true;
}

bool executable_authoring_identity(std::string_view name,
                                   std::string_view target) {
  static const std::set<std::string, std::less<>> forbidden{
      "argument", "arguments", "argv", "command", "entrypoint", "env",
      "environment", "executable", "expression", "import", "jinja",
      "module", "python", "script", "shell", "template",
  };
  std::string joined(name);
  joined.push_back('/');
  joined.append(target);
  std::string token;
  for (const char character : joined) {
    if ((character >= 'a' && character <= 'z') ||
        (character >= 'A' && character <= 'Z') ||
        (character >= '0' && character <= '9')) {
      token.push_back(static_cast<char>(std::tolower(
          static_cast<unsigned char>(character))));
    } else {
      if (forbidden.contains(token)) return true;
      token.clear();
    }
  }
  return forbidden.contains(token);
}

bool value_has_type(RecipeValueType type, const Json& value) {
  switch (type) {
    case RecipeValueType::boolean:
      return value.is_boolean();
    case RecipeValueType::integer:
      return value.is_number_integer();
    case RecipeValueType::number:
      return value.is_number() &&
             (!value.is_number_float() || std::isfinite(value.get<double>()));
    case RecipeValueType::string:
      return value.is_string() &&
             value.get_ref<const std::string&>().size() <= kMaximumStringBytes;
    case RecipeValueType::path:
      return value.is_string() &&
             absolute_normalized_path(value.get_ref<const std::string&>());
    case RecipeValueType::enumeration:
      return value.is_string() &&
             value.get_ref<const std::string&>().size() <= kMaximumIdentityBytes;
  }
  return false;
}

bool parameter_value_target(std::string_view target) {
  static const std::regex pattern("^/spec/parameters/[^/~]+/value$");
  return std::regex_match(target.begin(), target.end(), pattern);
}

bool control_default_target(std::string_view target) {
  static const std::regex pattern(
      "^/spec/controls/catalog/[^/~]+/default$");
  return std::regex_match(target.begin(), target.end(), pattern);
}

bool training_configuration_target(std::string_view target) {
  static const std::regex pattern(
      "^/spec/workflow/nodes/[^/~]+/invoke/training/components/[^/~]+/"
      "configuration/[^/~]+$");
  return std::regex_match(target.begin(), target.end(), pattern);
}

bool resource_target(std::string_view target) {
  static const std::set<std::string, std::less<>> allowed{
      "/spec/resources/accelerators/count",
      "/spec/resources/accelerators/exclusive",
      "/spec/resources/accelerators/minimum_memory_gib",
      "/spec/resources/accelerators/vendor",
      "/spec/resources/minimum_host_memory_gib",
      "/spec/resources/cpu_threads",
      "/spec/resources/lease_timeout_seconds",
      "/spec/resources/cpu_io_policy/cpu_weight",
      "/spec/resources/cpu_io_policy/io_weight",
      "/spec/resources/cpu_io_policy/omp_threads",
      "/spec/resources/cpu_io_policy/preprocessing_workers",
      "/spec/resources/cpu_io_policy/nice",
  };
  return allowed.contains(target);
}

bool authorized_target(RecipeOverrideDomain domain, std::string_view target) {
  switch (domain) {
    case RecipeOverrideDomain::model:
      return parameter_value_target(target) ||
             training_configuration_target(target);
    case RecipeOverrideDomain::data:
      return parameter_value_target(target);
    case RecipeOverrideDomain::trainability:
    case RecipeOverrideDomain::hyperparameters:
      return parameter_value_target(target) ||
             training_configuration_target(target) ||
             control_default_target(target);
    case RecipeOverrideDomain::evaluation:
    case RecipeOverrideDomain::checkpointing:
      return parameter_value_target(target) || control_default_target(target);
    case RecipeOverrideDomain::resources:
      return resource_target(target);
    case RecipeOverrideDomain::controls:
      return control_default_target(target);
  }
  return false;
}

bool authorized_type(RecipeOverrideDomain domain, RecipeValueType type) {
  switch (domain) {
    case RecipeOverrideDomain::model:
    case RecipeOverrideDomain::data:
      return type == RecipeValueType::string || type == RecipeValueType::path ||
             type == RecipeValueType::enumeration;
    case RecipeOverrideDomain::checkpointing:
      return type != RecipeValueType::string;
    case RecipeOverrideDomain::trainability:
    case RecipeOverrideDomain::hyperparameters:
    case RecipeOverrideDomain::evaluation:
    case RecipeOverrideDomain::resources:
    case RecipeOverrideDomain::controls:
      return type != RecipeValueType::string && type != RecipeValueType::path;
  }
  return false;
}

bool pointer_ancestor(std::string_view left, std::string_view right) {
  return right.size() > left.size() && right.starts_with(left) &&
         right[left.size()] == '/';
}

void validate_value(const RecipeOverrideField& field, const Json& value,
                    std::string_view context) {
  if (!value_has_type(field.type, value))
    reject(std::string(context) + " has the wrong type for override " +
           field.name);
  if (value.is_number()) {
    const double numeric = value.get<double>();
    if ((field.minimum && numeric < *field.minimum) ||
        (field.maximum && numeric > *field.maximum)) {
      reject(std::string(context) + " violates the numeric bound for override " +
             field.name);
    }
  }
  if (field.values &&
      std::ranges::find(*field.values, value) == field.values->end()) {
    reject(std::string(context) + " violates the allow-list for override " +
           field.name);
  }
}

void validate_path_authority(const RecipeOverrideField& field,
                             const Json& value,
                             const Json& template_document) {
  if (field.type != RecipeValueType::path) return;
  const auto workspace = template_document.find("spec");
  if (workspace == template_document.end() || !workspace->is_object() ||
      !workspace->contains("workspace") ||
      !workspace->at("workspace").is_object()) {
    reject("recipe path override has no workspace authority");
  }
  const Json& authority = workspace->at("workspace");
  const auto roots = authority.find("allowed_read_roots");
  if (roots == authority.end() || !roots->is_array() || roots->empty())
    reject("recipe path override requires declared allowed_read_roots");
  const std::filesystem::path candidate(value.get_ref<const std::string&>());
  if (!std::ranges::any_of(*roots, [&](const Json& root) {
        return root.is_string() &&
               path_within(candidate,
                           std::filesystem::path(
                               root.get_ref<const std::string&>()));
      })) {
    reject("recipe path override is outside allowed_read_roots: " +
           candidate.string());
  }
}

void validate_field(RecipeOverrideField& field, const Json& template_document) {
  if (!symbolic_identity(field.name) || field.target.empty() ||
      field.target.size() > kMaximumStringBytes ||
      !field.target.starts_with('/') ||
      (field.description && field.description->size() > 2048U)) {
    reject("recipe override identity, target, or description is malformed");
  }
  if (!authorized_target(field.domain, field.target)) {
    reject("recipe override target is outside the authority-safe surface: " +
           field.target);
  }
  if (!authorized_type(field.domain, field.type) ||
      executable_authoring_identity(field.name, field.target)) {
    reject("recipe override exposes executable or unbounded authoring material: " +
           field.name);
  }
  if (field.minimum &&
      (!std::isfinite(*field.minimum) ||
       (field.maximum && *field.minimum > *field.maximum))) {
    reject("recipe override numeric bounds are invalid");
  }
  if (field.maximum && !std::isfinite(*field.maximum))
    reject("recipe override numeric bounds are invalid");
  if ((field.minimum || field.maximum) &&
      field.type != RecipeValueType::integer &&
      field.type != RecipeValueType::number) {
    reject("only numeric recipe overrides may declare numeric bounds");
  }
  if (field.values) {
    if (field.type != RecipeValueType::enumeration || field.values->empty() ||
        field.values->size() > kMaximumTuples ||
        std::ranges::any_of(*field.values, [&](const Json& value) {
          return !value_has_type(field.type, value);
        })) {
      reject("recipe override allow-list is invalid");
    }
    std::ranges::sort(*field.values);
    if (std::ranges::adjacent_find(*field.values) != field.values->end())
      reject("recipe override allow-list contains duplicate values");
  } else if (field.type == RecipeValueType::enumeration) {
    reject("enumeration recipe override requires an allow-list");
  }

  try {
    const Json::json_pointer pointer(field.target);
    if (!template_document.contains(pointer))
      reject("recipe override target does not exist in its template: " +
             field.target);
    const Json& base = template_document.at(pointer);
    if (base.is_structured())
      reject("recipe overrides may target scalar values only: " + field.target);
    validate_value(field, base, "recipe template value");
    validate_path_authority(field, base, template_document);
  } catch (const Json::exception& error) {
    reject("recipe override target is not a valid JSON pointer: " +
           field.target + ": " + error.what());
  }
}

std::string tuple_sort_key(const std::vector<Json>& tuple) {
  return Json(tuple).dump();
}

void validate_compatibility(
    std::vector<RecipeCompatibilityRule>& rules,
    const std::map<std::string, const RecipeOverrideField*, std::less<>>& fields) {
  if (rules.size() > kMaximumRules)
    reject("recipe compatibility rule count exceeds its bound");
  for (RecipeCompatibilityRule& rule : rules) {
    if (rule.fields.size() < 2U || rule.fields.size() > 8U ||
        rule.allowed.empty() || rule.allowed.size() > kMaximumTuples ||
        (rule.description && rule.description->size() > 2048U)) {
      reject("recipe compatibility rule shape is invalid");
    }
    std::set<std::string> unique_fields;
    for (const std::string& name : rule.fields) {
      if (!unique_fields.insert(name).second || !fields.contains(name))
        reject("recipe compatibility rule names an unknown or duplicate field");
    }
    for (const std::vector<Json>& tuple : rule.allowed) {
      if (tuple.size() != rule.fields.size())
        reject("recipe compatibility tuple has the wrong arity");
      for (std::size_t index = 0; index < tuple.size(); ++index)
        validate_value(*fields.at(rule.fields[index]), tuple[index],
                       "recipe compatibility tuple value");
    }
    std::ranges::sort(rule.allowed, {}, tuple_sort_key);
    if (std::ranges::adjacent_find(rule.allowed) != rule.allowed.end())
      reject("recipe compatibility rule contains a duplicate tuple");
  }
  std::ranges::sort(rules, [](const auto& left, const auto& right) {
    return std::tie(left.fields, left.allowed) <
           std::tie(right.fields, right.allowed);
  });
  if (std::ranges::adjacent_find(
          rules, [](const auto& left, const auto& right) {
            return left.fields == right.fields;
          }) != rules.end()) {
    reject("recipe compatibility fields have ambiguous ownership");
  }
}

void canonicalize_profile(RecipeProfile& profile) {
  if (!symbolic_identity(profile.key.name) ||
      !symbolic_identity(profile.key.version, true) ||
      (profile.description && profile.description->size() > 4096U) ||
      profile.overrides.size() > kMaximumOverrides) {
    reject("recipe profile identity or size is invalid");
  }
  CompileResult compiled = compile_document(profile.template_document);
  if (!compiled.valid()) {
    reject("recipe template is not a valid Experiment: " +
           diagnostics_json(compiled.diagnostics).dump());
  }
  profile.template_document = std::move(compiled.plan->canonical_plan);
  const Json& workspace = profile.template_document.at("spec").at("workspace");
  const std::string run_directory =
      workspace.at("run_directory").get<std::string>();
  const auto write_roots = workspace.find("allowed_write_roots");
  if (write_roots == workspace.end() || !write_roots->is_array() ||
      std::ranges::count(*write_roots, Json(run_directory)) != 1) {
    reject("recipe template must grant its run_directory exactly once in "
           "allowed_write_roots so run identity can be bound atomically");
  }

  for (RecipeOverrideField& field : profile.overrides)
    validate_field(field, profile.template_document);
  std::ranges::sort(profile.overrides, {}, &RecipeOverrideField::name);
  if (std::ranges::adjacent_find(
          profile.overrides, {}, &RecipeOverrideField::name) !=
      profile.overrides.end()) {
    reject("recipe override fields must have unique names");
  }
  for (std::size_t left = 0; left < profile.overrides.size(); ++left) {
    for (std::size_t right = left + 1U; right < profile.overrides.size();
         ++right) {
      const std::string& first = profile.overrides[left].target;
      const std::string& second = profile.overrides[right].target;
      if (first == second || pointer_ancestor(first, second) ||
          pointer_ancestor(second, first)) {
        reject("recipe override targets have ambiguous ownership");
      }
    }
  }
  std::map<std::string, const RecipeOverrideField*, std::less<>> fields;
  for (const RecipeOverrideField& field : profile.overrides)
    fields.emplace(field.name, &field);
  if (profile.compatibility)
    validate_compatibility(*profile.compatibility, fields);
}

Json parse_exact_json(std::string_view document, std::string_view label,
                      std::size_t maximum_bytes) {
  if (document.empty() || document.size() > maximum_bytes)
    reject(std::string(label) + " document size is invalid");
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    const Json::parser_callback_t reject_duplicates =
        [&](int depth, Json::parse_event_t event, Json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == Json::parse_event_t::object_start) {
            if (object_keys.size() <= index + 1U)
              object_keys.resize(index + 2U);
            object_keys[index + 1U].clear();
          } else if (event == Json::parse_event_t::key) {
            if (object_keys.size() <= index) object_keys.resize(index + 1U);
            if (!object_keys[index].insert(parsed.get<std::string>()).second)
              duplicate_key = true;
          } else if (event == Json::parse_event_t::object_end &&
                     object_keys.size() > index + 1U) {
            object_keys[index + 1U].clear();
          }
          return true;
        };
    Json result = Json::parse(document, reject_duplicates);
    if (duplicate_key) reject(std::string(label) + " contains a duplicate key");
    return result;
  } catch (const RecipeProfileError&) {
    throw;
  } catch (const Json::exception& error) {
    reject(std::string(label) + " is not valid JSON: " + error.what());
  }
}

std::string escape_pointer_token(std::string_view token) {
  std::string result;
  result.reserve(token.size());
  for (const char character : token) {
    if (character == '~') {
      result += "~0";
    } else if (character == '/') {
      result += "~1";
    } else {
      result += character;
    }
  }
  return result;
}

void add_template_provenance(
    const Json& value, const std::string& path, const RecipeKey& key,
    std::map<std::string, RecipeValueSource>& provenance) {
  if (value.is_object()) {
    for (const auto& [name, child] : value.items()) {
      add_template_provenance(
          child, path + "/" + escape_pointer_token(name), key, provenance);
    }
    return;
  }
  if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      add_template_provenance(value[index], path + "/" + std::to_string(index),
                              key, provenance);
    }
    return;
  }
  provenance.emplace(
      path, RecipeValueSource{
                .kind = "recipe_template",
                .reference = key.name + "@" + key.version + "#template" + path,
            });
}

void set_derived_run_value(
    Json& document, const std::string& path, Json value,
    const RecipeKey& key, std::string_view run_identity,
    std::map<std::string, RecipeValueSource>& provenance) {
  document[Json::json_pointer(path)] = std::move(value);
  provenance[path] = {
      .kind = "instance_run_identity",
      .reference = key.name + "@" + key.version + "#run_identity/" +
                   std::string(run_identity),
  };
}

void bind_run_identity(
    Json& document, const RecipeKey& key, std::string_view run_identity,
    std::map<std::string, RecipeValueSource>& provenance) {
  if (run_identity.empty() || run_identity.size() > 128U ||
      !std::regex_match(run_identity.begin(), run_identity.end(),
                        kRunIdentity)) {
    reject("recipe run_identity must be a bounded lowercase symbolic identity");
  }
  Json& workspace = document.at("spec").at("workspace");
  const std::string base_run_directory =
      workspace.at("run_directory").get<std::string>();
  const std::string base_concurrency_key =
      workspace.at("concurrency_key").get<std::string>();
  const std::filesystem::path base_path(base_run_directory);
  if (!absolute_normalized_path(base_run_directory) ||
      base_path.parent_path().empty()) {
    reject("recipe template run_directory cannot derive run authority");
  }
  const std::filesystem::path derived_path =
      (base_path.parent_path() / std::string(run_identity)).lexically_normal();
  if (!path_within(derived_path, base_path.parent_path()))
    reject("derived recipe run_directory escaped its authority root");
  const std::string derived_concurrency =
      base_concurrency_key + "." + std::string(run_identity);
  if (derived_concurrency.size() > 128U)
    reject("derived recipe concurrency_key exceeds its bound");

  set_derived_run_value(document, "/metadata/name", run_identity, key,
                        run_identity, provenance);
  set_derived_run_value(document, "/spec/workspace/run_directory",
                        derived_path.string(), key, run_identity, provenance);
  set_derived_run_value(document, "/spec/workspace/concurrency_key",
                        derived_concurrency, key, run_identity, provenance);

  Json& write_roots = workspace.at("allowed_write_roots");
  std::size_t replaced_roots = 0U;
  for (std::size_t index = 0; index < write_roots.size(); ++index) {
    if (write_roots[index] == base_run_directory) {
      const std::string path =
          "/spec/workspace/allowed_write_roots/" + std::to_string(index);
      set_derived_run_value(document, path, derived_path.string(), key,
                            run_identity, provenance);
      ++replaced_roots;
    }
  }
  if (replaced_roots != 1U) {
    reject("recipe template must grant its run_directory exactly once in "
           "allowed_write_roots");
  }

  Json& nodes = document.at("spec").at("workflow").at("nodes");
  const Json& components = document.at("spec").at("components");
  std::size_t acquire_literals = 0U;
  std::size_t release_literals = 0U;
  for (auto& [node_name, node] : nodes.items()) {
    Json& invocation = node.at("invoke");
    const std::string component = invocation.at("component").get<std::string>();
    const auto descriptor = components.find(component);
    if (descriptor == components.end() ||
        descriptor->at("adapter") != "trainvm.core" ||
        descriptor->at("version") != "1.0.0" ||
        descriptor->at("runtime") != "builtin") {
      continue;
    }
    const std::string operation = invocation.at("operation").get<std::string>();
    if (operation != "acquire_resources" && operation != "release_resources")
      continue;
    Json& literal = invocation.at("inputs").at("concurrency_key").at("literal");
    if (literal != base_concurrency_key)
      reject("recipe resource literal disagrees with its concurrency_key");
    const std::string path =
        "/spec/workflow/nodes/" + escape_pointer_token(node_name) +
        "/invoke/inputs/concurrency_key/literal";
    set_derived_run_value(document, path, derived_concurrency, key,
                          run_identity, provenance);
    if (operation == "acquire_resources") {
      ++acquire_literals;
    } else {
      ++release_literals;
    }
  }
  if ((acquire_literals == 0U) != (release_literals == 0U) ||
      acquire_literals > 1U)
    reject("recipe resource lifecycle must bind acquire and release together");
}

Json canonical_instance_json(const RecipeInstance& instance) {
  return encode_json(instance);
}

RecipeInstance decode_instance(const Json& source) {
  RecipeInstance instance;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, instance, "", diagnostics))
    reject("recipe instance schema validation failed: " +
           diagnostics_json(diagnostics).dump());
  if (instance.api_version != "trainvm.recipe-instance/v1")
    reject("recipe instance api_version is unsupported");
  return instance;
}

void check_compatibility(
    const RecipeProfile& profile,
    const std::map<std::string, Json>& effective_overrides) {
  if (!profile.compatibility) return;
  for (const RecipeCompatibilityRule& rule : *profile.compatibility) {
    std::vector<Json> tuple;
    tuple.reserve(rule.fields.size());
    for (const std::string& name : rule.fields)
      tuple.push_back(effective_overrides.at(name));
    if (std::ranges::find(rule.allowed, tuple) == rule.allowed.end()) {
      reject("recipe override combination is incompatible for fields: " +
             Json(rule.fields).dump());
    }
  }
}

}  // namespace

RecipeProfileRegistry::RecipeProfileRegistry(
    std::vector<RecipeProfile> profiles) {
  if (profiles.empty() || profiles.size() > kMaximumProfiles)
    reject("recipe profile registry profile count is invalid");
  for (RecipeProfile& profile : profiles) {
    canonicalize_profile(profile);
    const RecipeKey key = profile.key;
    const std::string digest =
        "sha256:" + sha256_hex(encode_json(profile).dump());
    if (!profiles_.emplace(key, std::move(profile)).second)
      reject("recipe profile registry contains a duplicate exact key");
    profile_digests_.emplace(key, digest);
  }
  const std::string canonical = document_json().dump();
  if (canonical.size() > kMaximumRegistryBytes)
    reject("canonical recipe profile registry exceeds its byte bound");
  registry_digest_ = "sha256:" + sha256_hex(canonical);
}

RecipeProfileRegistry RecipeProfileRegistry::from_json(
    std::string_view document) {
  const Json source =
      parse_exact_json(document, "recipe profile registry", kMaximumRegistryBytes);
  RecipeProfileRegistryDocument decoded;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, decoded, "", diagnostics))
    reject("recipe profile registry schema validation failed: " +
           diagnostics_json(diagnostics).dump());
  if (decoded.api_version != "trainvm.recipe-profiles/v1")
    reject("recipe profile registry api_version is unsupported");
  return RecipeProfileRegistry(std::move(decoded.recipes));
}

RecipeProfileRegistry RecipeProfileRegistry::load_file(
    const std::filesystem::path& path) {
  return from_json(read_authority_document(
      path, "recipe profile registry", kMaximumRegistryBytes));
}

const RecipeProfile& RecipeProfileRegistry::profile(
    const RecipeKey& key) const {
  const auto found = profiles_.find(key);
  if (found == profiles_.end())
    reject("no recipe profile matches the exact requested key");
  return found->second;
}

std::string RecipeProfileRegistry::profile_digest(
    const RecipeKey& key) const {
  const auto found = profile_digests_.find(key);
  if (found == profile_digests_.end())
    reject("no recipe profile matches the exact requested key");
  return found->second;
}

ExpandedRecipe RecipeProfileRegistry::expand(
    const RecipeInstance& instance) const {
  if (instance.api_version != "trainvm.recipe-instance/v1")
    reject("recipe instance api_version is unsupported");
  const RecipeProfile& selected = profile(instance.recipe);
  std::map<std::string, const RecipeOverrideField*, std::less<>> fields;
  for (const RecipeOverrideField& field : selected.overrides)
    fields.emplace(field.name, &field);
  for (const auto& [name, unused] : instance.overrides) {
    (void)unused;
    if (!fields.contains(name))
      reject("recipe instance contains an unknown override: " + name);
  }

  Json expanded_document = selected.template_document;
  std::map<std::string, Json> effective;
  std::map<std::string, RecipeValueSource> provenance;
  add_template_provenance(expanded_document, "", selected.key, provenance);
  bind_run_identity(expanded_document, selected.key, instance.run_identity,
                    provenance);
  for (const RecipeOverrideField& field : selected.overrides) {
    const auto supplied = instance.overrides.find(field.name);
    if (field.required && supplied == instance.overrides.end())
      reject("recipe instance is missing required override: " + field.name);
    const Json::json_pointer target(field.target);
    if (supplied != instance.overrides.end()) {
      validate_value(field, supplied->second, "recipe instance value");
      validate_path_authority(field, supplied->second,
                              selected.template_document);
      expanded_document[target] = supplied->second;
      provenance[field.target] = {
          .kind = "instance_override",
          .reference = instance.recipe.name + "@" + instance.recipe.version +
                       "#override/" + field.name,
      };
    }
    effective.emplace(field.name, expanded_document.at(target));
  }
  check_compatibility(selected, effective);

  CompileResult compiled = compile_document(expanded_document);
  if (!compiled.valid()) {
    reject("expanded recipe is not a valid Experiment: " +
           diagnostics_json(compiled.diagnostics).dump());
  }
  const Json canonical_instance = canonical_instance_json(instance);
  return {
      .recipe = selected.key,
      .run_identity = instance.run_identity,
      .registry_digest = registry_digest_,
      .profile_digest = profile_digest(selected.key),
      .instance_digest = "sha256:" + sha256_hex(canonical_instance.dump()),
      .expanded_plan_digest = "sha256:" + compiled.plan->plan_hash,
      .effective_overrides = std::move(effective),
      .provenance = std::move(provenance),
      .plan = std::move(*compiled.plan),
  };
}

ExpandedRecipe RecipeProfileRegistry::expand_json(const Json& instance) const {
  return expand(decode_instance(instance));
}

const std::string& RecipeProfileRegistry::registry_digest() const noexcept {
  return registry_digest_;
}

Json RecipeProfileRegistry::document_json() const {
  std::vector<RecipeProfile> recipes;
  recipes.reserve(profiles_.size());
  for (const auto& [unused, profile] : profiles_) {
    (void)unused;
    recipes.push_back(profile);
  }
  return encode_json(RecipeProfileRegistryDocument{
      .api_version = "trainvm.recipe-profiles/v1",
      .recipes = std::move(recipes),
  });
}

std::vector<RecipePlanDifference> diff_recipe_plans(
    const ExpandedRecipe& left, const ExpandedRecipe& right) {
  std::set<std::string> paths;
  for (const auto& [path, unused] : left.provenance) {
    (void)unused;
    paths.insert(path);
  }
  for (const auto& [path, unused] : right.provenance) {
    (void)unused;
    paths.insert(path);
  }
  std::vector<RecipePlanDifference> differences;
  for (const std::string& path : paths) {
    const Json::json_pointer pointer(path);
    const bool has_left = left.plan.canonical_plan.contains(pointer);
    const bool has_right = right.plan.canonical_plan.contains(pointer);
    const Json left_value = has_left ? left.plan.canonical_plan.at(pointer) : Json{};
    const Json right_value = has_right ? right.plan.canonical_plan.at(pointer) : Json{};
    if (has_left == has_right && left_value == right_value) continue;
    const auto left_source = left.provenance.find(path);
    const auto right_source = right.provenance.find(path);
    differences.push_back({
        .path = path,
        .left = left_value,
        .right = right_value,
        .left_source = left_source == left.provenance.end()
                           ? std::nullopt
                           : std::optional<RecipeValueSource>{left_source->second},
        .right_source = right_source == right.provenance.end()
                            ? std::nullopt
                            : std::optional<RecipeValueSource>{right_source->second},
    });
  }
  return differences;
}

Json expanded_recipe_json(const ExpandedRecipe& expanded,
                          bool include_canonical_plan) {
  Json result{
      {"api_version", "trainvm.expanded-recipe/v1"},
      {"recipe", encode_json(expanded.recipe)},
      {"run_identity", expanded.run_identity},
      {"registry_digest", expanded.registry_digest},
      {"profile_digest", expanded.profile_digest},
      {"instance_digest", expanded.instance_digest},
      {"expanded_plan_digest", expanded.expanded_plan_digest},
      {"plan_hash", expanded.plan.plan_hash},
      {"effective_overrides", expanded.effective_overrides},
      {"provenance", encode_json(expanded.provenance)},
  };
  if (include_canonical_plan)
    result["canonical_plan"] = expanded.plan.canonical_plan;
  return result;
}

Json recipe_plan_diff_json(
    const std::vector<RecipePlanDifference>& differences) {
  Json result = Json::array();
  for (const RecipePlanDifference& difference : differences)
    result.push_back(encode_json(difference));
  return result;
}

}  // namespace trainvm
