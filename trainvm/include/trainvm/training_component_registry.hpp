#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

struct CompiledPlan;

enum class TrainingComponentBackend {
  python,
  native,
  cuda_extension,
  runtime_builtin,
};

enum class TrainingValueType {
  boolean,
  integer,
  number,
  string,
  path,
  string_list,
  enumeration,
};

// Lists must declare whether their order is part of experiment identity. Sets
// are sorted and deduplicated by the authority; ordered lists retain author
// order. Requiring this declaration prevents a target-selector set from
// acquiring a different digest merely because a YAML author reordered it.
enum class TrainingCollectionSemantics {
  ordered,
  set,
};

enum class TrainingStringFormat {
  parameter_selector,
  sha256_digest,
};

enum class TrainingStateGrade {
  stateless,
  compatible,
  exact,
};

// A deliberately flat reflected field contract. Nested trainer-specific
// objects stay behind an adapter input contract; common training knobs remain
// independently inspectable, comparable, and editable by the dashboard.
struct TrainingComponentField final {
  std::string name;
  TrainingValueType type{};
  bool required{};
  std::optional<nlohmann::json> default_value;
  std::optional<double> minimum;
  std::optional<double> maximum;
  std::optional<std::vector<nlohmann::json>> values;
  std::optional<TrainingCollectionSemantics> collection_semantics = std::nullopt;
  std::optional<TrainingStringFormat> string_format = std::nullopt;
  std::optional<std::string> unit;
  std::optional<std::string> description;

  bool operator==(const TrainingComponentField&) const = default;
};

struct TrainingComponentDescriptor final {
  TrainingComponentKey key;
  TrainingComponentBackend backend{};
  // Authority-owned symbolic implementation identity. It is not argv, a
  // filesystem path, or an experiment-supplied import string.
  std::string implementation;
  std::vector<std::string> model_families;
  std::vector<std::string> required_capabilities;
  std::vector<TrainingComponentField> configuration;
  std::vector<TrainingComponentField> state;
  std::optional<StepDomain> step_domain;
  TrainingStateGrade state_grade{};
  bool reference_implementation{};

  bool operator==(const TrainingComponentDescriptor&) const = default;
};

struct TrainingComponentRegistryDocument final {
  std::string api_version;
  std::vector<TrainingComponentDescriptor> components;
};

struct TrainingComponentRequest final {
  TrainingComponentKey key;
  std::string model_family;
  nlohmann::json configuration = nlohmann::json::object();
};

struct ResolvedTrainingComponent final {
  TrainingComponentDescriptor descriptor;
  nlohmann::json configuration = nlohmann::json::object();
  std::string descriptor_digest;

  bool operator==(const ResolvedTrainingComponent&) const = default;
};

struct ResolvedTrainingComposition final {
  std::string model_family;
  std::map<std::string, ResolvedTrainingComponent> components;
  // The lowered research-topology block, null when the composition selects
  // none. Part of the composition digest, so a topology change is a different
  // composition rather than a silent substitution.
  nlohmann::json topologies = nullptr;
  // The lowered post-training arm, null when the composition declares none.
  // Also part of the composition digest: two runs with identical components
  // but different bounds, seeds or reproducibility claims are different
  // experiments, and a digest that could not tell them apart would let one be
  // substituted for the other.
  nlohmann::json post_training = nullptr;
  std::string registry_digest;
  std::string composition_digest;

  bool operator==(const ResolvedTrainingComposition&) const = default;
};

[[nodiscard]] nlohmann::json resolved_training_composition_json(
    const ResolvedTrainingComposition& composition);

class TrainingComponentResolutionError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TrainingComponentRegistry final {
 public:
  explicit TrainingComponentRegistry(
      std::vector<TrainingComponentDescriptor> descriptors);

  static TrainingComponentRegistry from_json(std::string_view document);
  static TrainingComponentRegistry load_file(
      const std::filesystem::path& path);

  [[nodiscard]] const TrainingComponentDescriptor& descriptor(
      const TrainingComponentKey& key) const;
  [[nodiscard]] ResolvedTrainingComponent resolve(
      const TrainingComponentRequest& request) const;
  [[nodiscard]] ResolvedTrainingComposition resolve_composition(
      const TrainingComposition& composition) const;
  // Validate a worker checkpoint's per-slot component state before any tensor
  // state is restored. The object must contain exactly the stateful slots and
  // exactly the fields declared by their canonical descriptors.
  void validate_resume_state(const ResolvedTrainingComposition& composition,
                             const nlohmann::json& state) const;
  [[nodiscard]] WorkerLaunchRequest augment_worker_launch_request(
      WorkerLaunchRequest request,
      const std::optional<TrainingComposition>& composition) const;
  [[nodiscard]] bool plan_uses_components(const CompiledPlan& plan) const;
  void validate_plan(const CompiledPlan& plan) const;
  [[nodiscard]] std::string plan_lock_manifest(
      const CompiledPlan& plan) const;
  [[nodiscard]] std::string plan_lock_digest(const CompiledPlan& plan) const;
  void validate_submission_lock(const CompiledPlan& plan,
                                const nlohmann::json& submission) const;
  [[nodiscard]] const std::string& registry_digest() const noexcept;
  [[nodiscard]] std::string descriptor_digest(
      const TrainingComponentKey& key) const;
  [[nodiscard]] nlohmann::json document_json() const;
  [[nodiscard]] nlohmann::json descriptors_json() const;

 private:
  std::map<TrainingComponentKey, TrainingComponentDescriptor> descriptors_;
  std::string registry_digest_;
};

}  // namespace trainvm
