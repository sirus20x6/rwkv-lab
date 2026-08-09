#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/host_resources.hpp"

namespace trainvm {

struct ExpandedRecipe;

inline constexpr std::string_view kTrainingPreflightEnvironmentApiVersion =
    "trainvm.training-preflight-environment/v1";
inline constexpr std::string_view kTrainingPreflightReceiptApiVersion =
    "trainvm.training-preflight-receipt/v1";

// These are semantic obligations, not implementation names. A family probe
// must account for every obligation for each training node. A genuinely
// irrelevant obligation is recorded as not_applicable with a reason; silence
// never turns into success.
enum class TrainingPreflightCheckKind {
  model_configuration,
  tokenizer,
  processor,
  dataset_schema,
  dataset_sample_decode,
  parameter_selection,
  kernel_runtime,
  checkpoint_compatibility,
  step_zero_evaluator,
  dashboard_artifacts,
};

enum class TrainingPreflightCheckDisposition {
  passed,
  not_applicable,
  failed,
};

struct TrainingPreflightCheckEvidence final {
  TrainingPreflightCheckKind kind{};
  TrainingPreflightCheckDisposition disposition{};
  std::string evidence_digest;
  std::optional<std::string> detail;

  bool operator==(const TrainingPreflightCheckEvidence &) const = default;
};

struct TrainingNodePreflightEvidence final {
  std::string node_id;
  // Binds family-specific CPU probing to the exact immutable invocation.
  std::string node_input_digest;
  std::vector<TrainingPreflightCheckEvidence> checks;
  // This is deliberately distinct from Resources::minimum_memory_gib, which
  // is a total-capacity selector. It is the trainer's passive free-memory
  // policy extracted by the registered family probe.
  std::optional<double> minimum_free_memory_gib;
  // Exact registered adapter/launch/runtime profile used by the passive
  // probe; submission rechecks the same authority lock before launch.
  std::string runtime_profile_digest;
  std::vector<std::string> required_capabilities;
  std::vector<std::string> provided_capabilities;

  bool operator==(const TrainingNodePreflightEvidence &) const = default;
};

struct PassiveAcceleratorMemoryEvidence final {
  AcceleratorVendor vendor{};
  std::string stable_id;
  std::uint64_t total_memory_bytes{};
  std::uint64_t free_memory_bytes{};
  std::map<std::string, std::string> selector_labels;
  std::string observation_digest;
  // As observed by the host authority. Preflight applies
  // resource_disposition_permits to it rather than requiring
  // `audited_eligible`, so a plan declaring cooperative access can select a
  // device that is occupied only because it is driving a display.
  ResourceObservationDisposition disposition{};

  bool operator==(const PassiveAcceleratorMemoryEvidence &) const = default;
};

struct BoundedGpuQualificationEvidence final {
  std::uint64_t maximum_duration_milliseconds{};
  bool passed{};
  std::string receipt_digest;

  bool operator==(const BoundedGpuQualificationEvidence &) const = default;
};

struct TrainingPreflightRecipeProvenance final {
  std::string registry_digest;
  std::string profile_digest;
  std::string instance_digest;
  std::string expanded_plan_digest;

  bool operator==(const TrainingPreflightRecipeProvenance &) const = default;
};

// Projects the four immutable identities emitted by recipe expansion into the
// passive environment receipt. The preflight continues to consume only the
// ordinary CompiledPlan and verifies that expanded_plan_digest names it.
[[nodiscard]] TrainingPreflightRecipeProvenance
training_preflight_recipe_provenance(const ExpandedRecipe &expanded);

struct TrainingPreflightEnvironment final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  // Identity of the passive host/runtime snapshot. A changed snapshot changes
  // the receipt cache key even when the experiment is unchanged.
  std::string snapshot_digest;
  std::uint64_t snapshot_observed_monotonic_ns{};
  std::uint64_t snapshot_valid_until_monotonic_ns{};
  std::uint64_t evaluation_monotonic_ns{};
  std::uint32_t worker_uid{};
  std::uint32_t worker_gid{};
  // Empty is an explicit attestation that hostd clears supplementary groups.
  // Otherwise this is the complete sorted unique effective set.
  std::vector<std::uint32_t> supplementary_gids;
  // Digest of the exact hostd worker principal/profile, not merely a reusable
  // numeric uid/gid pair.
  std::string worker_principal_digest;
  std::uint64_t total_host_memory_bytes{};
  std::uint64_t available_host_memory_bytes{};
  std::uint32_t logical_cpu_count{};
  std::vector<PassiveAcceleratorMemoryEvidence> accelerators;
  std::vector<TrainingNodePreflightEvidence> training_nodes;
  std::optional<BoundedGpuQualificationEvidence> gpu_qualification;
  std::optional<TrainingPreflightRecipeProvenance> recipe_provenance;

  bool operator==(const TrainingPreflightEnvironment &) const = default;
};

struct TrainingPreflightDiagnostic final {
  Diagnostic::Severity severity{Diagnostic::Severity::error};
  std::string code;
  std::string path;
  std::string message;
  std::string help;

  bool operator==(const TrainingPreflightDiagnostic &) const = default;
};

struct TrainingPreflightReceipt final {
  std::string api_version;
  bool passed{};
  bool accelerator_passive{};
  bool cacheable{};
  std::string plan_hash;
  std::string input_identity_digest;
  std::string environment_digest;
  std::string cache_key;
  std::uint64_t valid_until_monotonic_ns{};
  std::vector<TrainingPreflightDiagnostic> diagnostics;
  std::string receipt_digest;

  bool operator==(const TrainingPreflightReceipt &) const = default;
};

class TrainingPreflightError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] std::string
training_preflight_node_input_digest(const CompiledPlan &plan,
                                     std::string_view node_id);

// Bounded, read-only, and GPU-passive unless a bounded qualification receipt
// is explicitly supplied. This function never creates a directory, journal
// row, lease, process, device context, or dashboard run.
[[nodiscard]] TrainingPreflightReceipt
run_training_preflight(const CompiledPlan &plan,
                       const TrainingPreflightEnvironment &environment);

[[nodiscard]] TrainingPreflightEnvironment
load_training_preflight_environment(const std::filesystem::path &path);

} // namespace trainvm
