#pragma once

#include <compare>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/document.hpp"

namespace trainvm {

enum class RecipeOverrideDomain {
  model,
  data,
  trainability,
  hyperparameters,
  evaluation,
  checkpointing,
  resources,
  controls,
};

enum class RecipeValueType {
  boolean,
  integer,
  number,
  string,
  path,
  enumeration,
};

struct RecipeKey final {
  std::string name;
  std::string version;

  auto operator<=>(const RecipeKey&) const = default;
};

// One scalar authoring affordance. The target is an RFC 6901 JSON pointer into
// the authority-owned canonical Experiment template. It cannot create fields
// or alter topology, adapter identity, operation identity, workspace authority,
// artifact authority, recovery policy, or executable material.
struct RecipeOverrideField final {
  std::string name;
  RecipeOverrideDomain domain{};
  RecipeValueType type{};
  std::string target;
  bool required{};
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<std::vector<nlohmann::json>> values;
  std::optional<std::string> description;

  bool operator==(const RecipeOverrideField&) const = default;
};

// A finite tuple allow-list is intentionally less expressive than predicates.
// It represents compatibility without introducing expressions, imports, or an
// embedded authoring language. Every named field must have an effective value.
struct RecipeCompatibilityRule final {
  std::vector<std::string> fields;
  std::vector<std::vector<nlohmann::json>> allowed;
  std::optional<std::string> description;

  bool operator==(const RecipeCompatibilityRule&) const = default;
};

// Authority-derived immutable content identity. Both targets are RFC 6901
// pointers into the closed canonical template; operators provide the path
// once and never copy a digest into the instance.
struct RecipeContentBinding final {
  std::string path_target;
  std::string fingerprint_target;

  bool operator==(const RecipeContentBinding&) const = default;
};

struct RecipeProfile final {
  RecipeKey key;
  std::optional<std::string> description;
  nlohmann::json template_document;
  std::vector<RecipeOverrideField> overrides;
  std::optional<std::vector<RecipeContentBinding>> content_bindings;
  std::optional<std::vector<RecipeCompatibilityRule>> compatibility;

  bool operator==(const RecipeProfile&) const = default;
};

struct RecipeProfileRegistryDocument final {
  std::string api_version;
  std::vector<RecipeProfile> recipes;
};

struct RecipeInstance final {
  std::string api_version;
  RecipeKey recipe;
  // One submission identity atomically derives every coupled run/output field;
  // callers never override workspace authority or resource literals directly.
  std::string run_identity;
  std::map<std::string, nlohmann::json> overrides;
};

struct RecipeValueSource final {
  std::string kind;
  std::string reference;

  bool operator==(const RecipeValueSource&) const = default;
};

struct ExpandedRecipe final {
  RecipeKey recipe;
  std::string run_identity;
  std::string registry_digest;
  std::string profile_digest;
  std::string instance_digest;
  std::string expanded_plan_digest;
  std::map<std::string, nlohmann::json> effective_overrides;
  std::map<std::string, RecipeValueSource> provenance;
  CompiledPlan plan;
};

struct RecipePlanDifference final {
  std::string path;
  nlohmann::json left;
  nlohmann::json right;
  std::optional<RecipeValueSource> left_source;
  std::optional<RecipeValueSource> right_source;
};

class RecipeProfileError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class RecipeProfileRegistry final {
 public:
  explicit RecipeProfileRegistry(std::vector<RecipeProfile> profiles);

  static RecipeProfileRegistry from_json(std::string_view document);
  static RecipeProfileRegistry load_file(const std::filesystem::path& path);

  [[nodiscard]] const RecipeProfile& profile(const RecipeKey& key) const;
  [[nodiscard]] std::string profile_digest(const RecipeKey& key) const;
  [[nodiscard]] ExpandedRecipe expand(const RecipeInstance& instance) const;
  [[nodiscard]] ExpandedRecipe expand_json(
      const nlohmann::json& instance) const;
  [[nodiscard]] const std::string& registry_digest() const noexcept;
  [[nodiscard]] nlohmann::json document_json() const;

 private:
  std::map<RecipeKey, RecipeProfile> profiles_;
  std::map<RecipeKey, std::string> profile_digests_;
  std::string registry_digest_;
};

[[nodiscard]] std::vector<RecipePlanDifference> diff_recipe_plans(
    const ExpandedRecipe& left, const ExpandedRecipe& right);
[[nodiscard]] nlohmann::json expanded_recipe_json(
    const ExpandedRecipe& expanded, bool include_canonical_plan = true);
[[nodiscard]] nlohmann::json recipe_plan_diff_json(
    const std::vector<RecipePlanDifference>& differences);

}  // namespace trainvm
