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

// One referenced source and the exact bytes the catalog was reviewed against.
// The byte binding used to be a single whole-tree digest, which meant every
// edit to any of the 155 referenced sources rewrote the same line -- so two
// pull requests touching two unrelated files conflicted on it whether or not
// their work overlapped. Per-path storage keeps the binding (each file is still
// pinned to its exact bytes, and a mismatch now names the file) and lets git
// merge two independent refreshes, because they are two different objects.
struct CompatibilitySourceDigest {
  std::string source_path;
  std::string source_sha256;
};

struct CompatibilityCatalogDocument {
  std::string api_version;
  CompatibilityAuthority authority{};
  // Exactly the union of every entry's source_paths, sorted, no duplicates and
  // no extras. The whole-tree digest is derived from this at load time rather
  // than stored: it is a pure function of these pairs, so a stored copy carried
  // no information and cost every concurrent change a guaranteed conflict.
  std::vector<CompatibilitySourceDigest> source_digests;
  // Binds only the part of each source that could change how the entry beside
  // it is classified: its entrypoint, its argument surface, and its checkpoint
  // and resume call sites. This, not source_tree_digest, is what the compiled
  // reviewed digest covers, so editing a comment or an internal computation no
  // longer demands a workflow re-review while editing how the thing is invoked
  // or resumes still fails closed.
  std::string classification_surface_digest;
  std::vector<CompatibilityWorkflowEntry> entries;
};

// What `trainvm print-catalog-digests` reports. The two pinned digests were
// maintained entirely by hand -- nothing in the repository computed them --
// which is a large part of why re-pinning felt like a chore rather than a
// review. Refreshing the byte digest is now mechanical, so the only thing left
// that demands human judgement is a real classification change.
struct CompatibilityCatalogComputedDigests {
  // Reported for continuity -- `validate-catalog` and the loader still expose a
  // whole-tree digest -- but it is derived from source_digests below, and no
  // catalog stores it.
  std::string source_tree_digest;
  // What actually gets written back into the catalog, one object per referenced
  // path. `print-catalog-digests --write` splices exactly these.
  std::vector<CompatibilitySourceDigest> source_digests;
  std::string classification_surface_digest;
  // Reported so a vacuous extraction is visible rather than inferred: a Python
  // file listed here exposes no entrypoint or argument surface at all.
  std::vector<std::string> paths_with_empty_classification_surface;
};

class CompatibilityCatalog {
 public:
  static CompatibilityCatalog load_file(
      const std::filesystem::path& catalog_path,
      const std::filesystem::path& repository_root);

  // Computes both digests from the referenced sources WITHOUT checking them
  // against the pinned values, so it still reports when they disagree. That is
  // the point: it is what you run to find out what to pin.
  static CompatibilityCatalogComputedDigests compute_digests(
      const std::filesystem::path& catalog_path,
      const std::filesystem::path& repository_root);

  [[nodiscard]] const std::vector<CompatibilityWorkflowEntry>& entries() const;
  [[nodiscard]] const std::string& catalog_digest() const;
  [[nodiscard]] const std::string& source_tree_digest() const;
  [[nodiscard]] const std::string& classification_surface_digest() const;
  [[nodiscard]] const std::string& repository_root_identity_display() const;
  [[nodiscard]] CompatibilityAuthority authority() const;
  [[nodiscard]] static std::string_view reviewed_catalog_digest();

  // Exposed for the extractor's own tests. A surface that silently came back
  // empty on an unfamiliar file would make every entry trivially unchanged,
  // which is the failure this whole mechanism exists to avoid, so the
  // extraction has to be directly testable rather than only observable
  // through a digest.
  [[nodiscard]] static std::string classification_surface_for_testing(
      std::string_view relative_path, std::string_view bytes);

  // Exposed for the same reason. While the whole-tree digest was stored in the
  // catalog, that stored constant incidentally pinned this fold: change a
  // separator or a domain-separation prefix and the shipped catalog stopped
  // loading. Deriving the digest removes that accident, so the fold needs a
  // test that pins it directly -- over fixed pairs, so the pin does not move
  // when a referenced source is edited.
  [[nodiscard]] static std::string source_tree_digest_for_testing(
      const std::vector<CompatibilitySourceDigest>& source_digests);

 private:
  explicit CompatibilityCatalog(CompatibilityCatalogDocument document,
                                const std::filesystem::path& repository_root);

  CompatibilityAuthority authority_{};
  std::vector<CompatibilityWorkflowEntry> entries_;
  std::string catalog_digest_;
  std::string source_tree_digest_;
  std::string classification_surface_digest_;
  std::string repository_root_identity_display_;
};

}  // namespace trainvm
