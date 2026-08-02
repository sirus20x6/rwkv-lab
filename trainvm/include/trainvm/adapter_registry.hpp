#pragma once

#include <compare>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/model.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

struct AdapterKey {
  std::string adapter;
  std::string version;
  ComponentRuntime runtime{};
  std::string operation;
  std::string contract;

  auto operator<=>(const AdapterKey&) const = default;
};

enum class ResumeGrade {
  none,
  restart_only,
  terminal_checkpoint,
  compatible,
  exact,
};

// A deliberately small, closed value vocabulary for operation ports. It is
// descriptive authoring authority, not an invitation to embed arbitrary JSON
// Schema in the adapter registry. Artifact ports may further narrow the
// logical artifact kind and schema below.
enum class OperationPortType {
  string,
  integer,
  number,
  boolean,
  object,
  artifact,
};

struct OperationPortDescriptor {
  OperationPortType type{};
  bool required{};
  std::optional<ArtifactType> artifact_type;
  std::optional<std::string> artifact_schema;
  std::optional<std::string> description;

  bool operator==(const OperationPortDescriptor&) const = default;
};

// Every registered operation must deliberately publish this declaration,
// even when one of the maps is empty. Keeping it optional on AdapterProfile
// lets registry validation distinguish an intentionally empty declaration
// from metadata that was omitted by a producer.
struct OperationAuthoringDeclaration {
  std::map<std::string, OperationPortDescriptor> inputs;
  std::map<std::string, OperationPortDescriptor> outputs;

  bool operator==(const OperationAuthoringDeclaration&) const = default;
};

// A closed, authority-owned declaration of the lifecycle protocol implemented
// by one exact adapter operation. These booleans are intentionally bounded:
// experiment documents can select capabilities, but cannot add arbitrary
// lifecycle verbs or infer them from signals, files, or worker output.
struct OperationLifecycleCapabilities {
  bool stateful{};
  bool graceful_stop{};
  bool checkpoint_now{};
  bool pause_keep_resources{};
  bool pause_release_resources{};
  bool compile{};
  bool warmup{};
  bool qualify{};
  bool profile{};
  ResumeGrade resume_grade{};

  bool operator==(const OperationLifecycleCapabilities&) const = default;
};

// The exact component surface consumed by one adapter operation. The adapter
// owns slot semantics; experiment documents may select implementations for
// these slots, but cannot invent new slots or attach components to an adapter
// that does not declare a composition contract.
struct TrainingCompositionContract {
  std::string model_family;
  std::map<std::string, TrainingComponentCategory> slots;

  bool operator==(const TrainingCompositionContract&) const = default;
};

struct AdapterProfile {
  AdapterKey key;
  Effect effect{};
  Idempotency idempotency{};
  std::string code_fingerprint;
  std::vector<std::string> required_capabilities;
  OperationLifecycleCapabilities lifecycle{};
  std::optional<TrainingCompositionContract> training_composition{};
  std::optional<OperationAuthoringDeclaration> authoring{};
};

struct OperationDescriptorDocument {
  std::string api_version;
  std::vector<AdapterProfile> operations;
};

struct AdapterRegistryDocument {
  std::string api_version;
  std::vector<AdapterProfile> profiles;
};

class AdapterResolutionError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Authority-owned registry. Experiment documents may select an exact key, but
// they never supply code identity or worker capabilities.
class AdapterRegistry {
 public:
  explicit AdapterRegistry(std::vector<AdapterProfile> profiles);

  static AdapterRegistry load_file(const std::filesystem::path& path);

  [[nodiscard]] const AdapterProfile& resolve(
      const Component& component, std::string_view operation) const;
  [[nodiscard]] const AdapterProfile& resolve(const AdapterKey& key) const;
  [[nodiscard]] std::string profile_digest(const AdapterKey& key) const;
  void validate_plan(const CompiledPlan& plan) const;
  [[nodiscard]] WorkerLaunchRequest worker_launch_request(
      const Component& component, std::string_view operation) const;
  [[nodiscard]] const std::string& registry_digest() const;
  [[nodiscard]] nlohmann::json operation_descriptors_json() const;
  [[nodiscard]] const std::string& operation_descriptors_digest() const;
  [[nodiscard]] std::string plan_lock_manifest(const CompiledPlan& plan) const;
  [[nodiscard]] std::string plan_lock_digest(const CompiledPlan& plan) const;
  void validate_submission_lock(const CompiledPlan& plan,
                                const nlohmann::json& submission) const;

 private:
  std::map<AdapterKey, AdapterProfile> profiles_;
  std::string registry_digest_;
  nlohmann::json operation_descriptors_json_;
  std::string operation_descriptors_digest_;
};

}  // namespace trainvm
