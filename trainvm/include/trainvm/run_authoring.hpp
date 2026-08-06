#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/adapter_registry.hpp"
#include "trainvm/input_content_authority.hpp"
#include "trainvm/recipe_profile.hpp"
#include "trainvm/training_preflight.hpp"

namespace trainvm {

inline constexpr std::string_view kAuthorRunApiVersion =
    "trainvm.author-run/v1";
inline constexpr std::string_view kAuthoringClientApiVersion =
    "trainvm.authoring-client/v1";
inline constexpr std::string_view kInstalledAuthoringClientPath =
    "/etc/trainvm/authoring-client.json";
inline constexpr std::string_view kInstalledRecipeProfilePath =
    "/etc/trainvm/recipe-profiles.json";
inline constexpr std::string_view kSealedStructuralPreflightCapability =
    "trainvm.passive-preflight.structural@1";

struct AuthorRunRecipeSource final {
  std::string registry_path;
  RecipeInstance instance;

  bool operator==(const AuthorRunRecipeSource &) const = default;
};

struct AuthorRunSource final {
  std::optional<nlohmann::json> experiment;
  std::optional<AuthorRunRecipeSource> recipe;

  bool operator==(const AuthorRunSource &) const = default;
};

// This is the complete operator-authored surface. It deliberately contains no
// socket, environment, argv, import, executable, probe result, preflight
// receipt, hash, or run ID field. Those identities are authority-owned.
struct AuthorRunDocument final {
  std::string api_version;
  AuthorRunSource source;
  std::optional<InputContentRootSet> input_content;
  std::string author;
  std::string reason;

  bool operator==(const AuthorRunDocument &) const = default;
};

// Installed by deployment, never selected by an author-run document.
struct AuthoringClientConfiguration final {
  std::string api_version;
  std::string controller_target;
  std::string dashboard_base_url;

  bool operator==(const AuthoringClientConfiguration &) const = default;
};

enum class AuthorRunStage {
  validating,
  resolving,
  locking_inputs,
  preflight,
  provisioning,
  submitting,
  complete,
  failed,
};

struct AuthorRunStageUpdate final {
  AuthorRunStage stage{};
  std::string detail;
  std::optional<std::string> plan_hash;
  std::vector<TrainingPreflightDiagnostic> diagnostics;

  bool operator==(const AuthorRunStageUpdate &) const = default;
};

struct ResolvedAuthorRun final {
  CompiledPlan plan;
  std::optional<TrainingPreflightRecipeProvenance> recipe_provenance;
  std::optional<nlohmann::json> recipe_expansion;
  std::string request_digest;
  bool content_lock_reused{};
  std::vector<InputContentMeasurementStats> content_measurements;
};

struct TrainingPreflightEvidenceResult final {
  std::optional<TrainingPreflightEnvironment> environment;
  std::vector<TrainingPreflightDiagnostic> diagnostics;
};

class ITrainingPreflightEvidenceProvider {
public:
  virtual ~ITrainingPreflightEvidenceProvider() = default;
  [[nodiscard]] virtual TrainingPreflightEvidenceResult collect(
      const CompiledPlan &plan,
      const std::optional<TrainingPreflightRecipeProvenance> &recipe) = 0;
};

using TrainingNodeProbe = std::function<TrainingNodePreflightEvidence(
    const CompiledPlan &, std::string_view, const AdapterProfile &)>;

struct RegisteredTrainingNodeProbe final {
  AdapterKey key;
  TrainingNodeProbe probe;
};

using PassiveHostSnapshotSource =
    std::function<TrainingPreflightEnvironment(const CompiledPlan &)>;
using PassiveAcceleratorSnapshotSource = std::function<
    std::vector<PassiveAcceleratorMemoryEvidence>(const CompiledPlan &)>;

[[nodiscard]] PassiveHostSnapshotSource make_local_passive_host_snapshot_source(
    std::string host_id, std::string boot_id,
    std::function<std::uint64_t()> monotonic_now_ns,
    PassiveAcceleratorSnapshotSource accelerators = {});
[[nodiscard]] TrainingNodeProbe make_sealed_structural_training_node_probe();
struct PassiveRuntimeProfileEvidence final {
  std::string profile_digest;
  std::vector<std::string> provided_capabilities;
};
using PassiveRuntimeProfileSource =
    std::function<PassiveRuntimeProfileEvidence(const AdapterProfile &)>;
[[nodiscard]] TrainingNodeProbe make_hf_multimodal_sft_training_node_probe(
    PassiveRuntimeProfileSource runtime_profile);
[[nodiscard]] bool hf_lora_selectors_match_parameter_index(
    const std::vector<std::string> &selectors,
    const std::vector<std::string> &parameter_keys);

// Composes one bounded authority-owned host snapshot with exact AdapterKey
// probes. Missing selected adapters fail `new_implementation_required`; the
// provider never guesses model-family semantics from native code.
class RegisteredTrainingPreflightEvidenceProvider final
    : public ITrainingPreflightEvidenceProvider {
public:
  RegisteredTrainingPreflightEvidenceProvider(
      const AdapterRegistry &adapters, PassiveHostSnapshotSource host_snapshot,
      std::vector<RegisteredTrainingNodeProbe> probes,
      TrainingNodeProbe sealed_structural_probe = {});

  [[nodiscard]] TrainingPreflightEvidenceResult collect(
      const CompiledPlan &plan,
      const std::optional<TrainingPreflightRecipeProvenance> &recipe) override;

private:
  const AdapterRegistry &adapters_;
  PassiveHostSnapshotSource host_snapshot_;
  std::map<AdapterKey, TrainingNodeProbe> probes_;
  TrainingNodeProbe sealed_structural_probe_;
};

class RunAuthoringError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] AuthorRunDocument decode_author_run_document(
    std::string_view source, std::string_view source_format);
[[nodiscard]] AuthoringClientConfiguration load_authoring_client_configuration(
    const std::filesystem::path &path =
        std::filesystem::path(kInstalledAuthoringClientPath));

// Pure authoring preparation: decode has already completed; this resolves an
// ordinary experiment or recipe, reuses an exact existing content lock when
// possible, otherwise measures the declared root set, then recompiles.
[[nodiscard]] ResolvedAuthorRun
resolve_and_lock_author_run(const AuthorRunDocument &document);

// The authority creates only the exact final run-directory component after a
// passing passive receipt. Existing exact directories make retry idempotent;
// symlinks, foreign ownership, unsafe modes, or nonempty conflicts reject.
class AuthorizedRunDirectoryProvision final {
public:
  AuthorizedRunDirectoryProvision(std::filesystem::path path,
                                  std::string marker, bool created);
  ~AuthorizedRunDirectoryProvision();
  AuthorizedRunDirectoryProvision(AuthorizedRunDirectoryProvision &&) noexcept;
  AuthorizedRunDirectoryProvision &
  operator=(AuthorizedRunDirectoryProvision &&) noexcept;
  AuthorizedRunDirectoryProvision(const AuthorizedRunDirectoryProvision &) =
      delete;
  AuthorizedRunDirectoryProvision &
  operator=(const AuthorizedRunDirectoryProvision &) = delete;

  void mark_durable() noexcept;
  [[nodiscard]] bool created() const noexcept;

private:
  std::filesystem::path path_;
  std::string marker_;
  bool created_{};
  bool durable_{};
};

[[nodiscard]] AuthorizedRunDirectoryProvision
provision_authorized_run_directory(
    const CompiledPlan &plan,
    const TrainingPreflightEnvironment &environment,
    std::string_view request_digest,
    // Deterministic test seam after the exact mkdir is pinned and before the
    // authority marker is published. Production callers always omit it.
    std::function<void()> before_marker_test_seam = {});

[[nodiscard]] std::string author_run_stage_name(AuthorRunStage stage);

} // namespace trainvm
