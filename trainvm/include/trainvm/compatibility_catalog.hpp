#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace trainvm {

// This catalog describes legacy compatibility evidence. It is deliberately
// disjoint from AdapterRegistry and the host execution registry: none of these
// types carries an adapter key, executable identity, capability, or execution
// method.
enum class CompatibilityAuthority {
  compatibility_evidence_only,
};

enum class WorkflowFamily {
  rwkv,
  transformer,
  vision_multimodal,
  mageflow_diffusion,
  conversion_distillation,
  rwkv_posttraining,
  rwkv_rlvr,
  external_trainer,
  data_cache,
  evaluation_profile_export,
  control_plane,
};

enum class ObservedInvocationKind {
  python_module,
  console_script,
  host_script,
  http_control_handler,
  // Historical dashboard mutation surface retained as a migration obligation
  // after its route and execution authority have been removed.
  retired_legacy,
  library_only,
  design_only,
};

// Catalog-local evidence vocabulary. In particular, exact_candidate is only a
// legacy implementation observation; it can never equal AdapterRegistry's
// sealed exact-resume authority.
enum class CompatibilityResumeEvidence {
  none,
  restart_only,
  terminal_checkpoint,
  compatible,
  exact_candidate,
};

enum class CompatibilityOperationRole {
  training,
  campaign,
  supervisor,
  acquisition,
  inventory,
  review,
  scoring,
  deduplication,
  cache_handoff,
  preprocess,
  cache_build,
  conversion,
  distillation,
  evaluation,
  profiling,
  export_artifact,
  qualification,
  library_oracle,
  design_spec,
};

struct CompatibilityWorkflowEntry {
  std::string stable_id;
  WorkflowFamily family{};
  std::vector<std::string> source_paths;
  ObservedInvocationKind observed_invocation{};
  bool stateful{};
  CompatibilityResumeEvidence resume_evidence{};
  CompatibilityOperationRole operation_role{};
  std::string notes;
  std::optional<std::string> legacy_invocation_display;
};

struct CompatibilityCatalogDocument {
  std::string api_version;
  CompatibilityAuthority authority{};
  std::string source_tree_digest;
  std::vector<CompatibilityWorkflowEntry> entries;
};

class CompatibilityCatalog {
 public:
  static CompatibilityCatalog load_file(
      const std::filesystem::path& catalog_path,
      const std::filesystem::path& repository_root);

  [[nodiscard]] const std::vector<CompatibilityWorkflowEntry>& entries() const;
  [[nodiscard]] const std::string& catalog_digest() const;
  [[nodiscard]] const std::string& source_tree_digest() const;
  [[nodiscard]] const std::string& repository_root_identity_display() const;
  [[nodiscard]] CompatibilityAuthority authority() const;
  [[nodiscard]] static std::string_view reviewed_catalog_digest();

 private:
  explicit CompatibilityCatalog(CompatibilityCatalogDocument document,
                                const std::filesystem::path& repository_root);

  CompatibilityAuthority authority_{};
  std::vector<CompatibilityWorkflowEntry> entries_;
  std::string catalog_digest_;
  std::string source_tree_digest_;
  std::string repository_root_identity_display_;
};

}  // namespace trainvm
