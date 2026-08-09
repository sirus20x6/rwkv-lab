#pragma once

#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace trainvm {

// Evidence-only inventory vocabulary. These values describe legacy source;
// they do not authorize the service to execute it.
enum class SourceDispositionClass {
  executable_operation,
  wrapper_alias,
  internal_utility_data_tool,
  supervisor_orchestrator,
  install_bootstrap,
  test_benchmark,
  explicit_exclusion,
  internal_model_layer_kernel_optimizer_library,
  data_eval_export_tool,
  research_oracle_poc,
  test_fixture,
};

enum class SourceResumeRelevance {
  none,
  restart_only,
  terminal_checkpoint,
  compatible,
  exact_candidate,
  consumer_owned,
  none_or_self_test_only,
};

struct SourceDispositionScope {
  std::string prefix;
  bool recursive{};
  std::vector<std::string> extensions;
};

struct SourceDispositionEntry {
  std::string source_path;
  SourceDispositionClass disposition_class{};
  std::string canonical_entry_point;
  std::vector<std::string> effects;
  SourceResumeRelevance resume_relevance{};
  std::optional<std::string> compatibility_workflow_id;
  std::string source_sha256;
  std::optional<std::string> family;
  std::optional<std::string> coverage;
  std::vector<std::string> consumers;
  std::vector<std::string> compatibility_workflow_ids;
  std::optional<std::string> language;
};

struct SourceDispositionDocument {
  std::string api_version;
  std::string authority;
  std::string source_repository;
  std::string source_revision;
  SourceDispositionScope source_scope;
  // Derived from entries by source_tree_digest(), never read from the document.
  // The catalog used to carry this as a stored field, which made it the single
  // line every concurrent change to the scope had to rewrite -- four pull
  // requests conflicted on it in one evening while touching disjoint work.
  // A document that still declares one is rejected rather than ignored.
  std::string source_tree_digest;
  std::vector<SourceDispositionEntry> entries;
};

// SHA-256 over the ASCII domain "trainvm.source-disposition-tree/v1" followed,
// in entry order, by NUL, source_path, NUL, and the entry's leaf digest.
//
// Exposed because scripts/print_disposition_digests.py maintains a second,
// hand-written implementation of this fold. Nothing compares the two unless
// something drives both over the same entries, which is what
// source_disposition_catalog_tests.cpp does with this declaration.
[[nodiscard]] std::string source_tree_digest(
    const std::vector<SourceDispositionEntry>& entries);

class SourceDispositionCatalog {
 public:
  // repository_root is optional so checked-in catalogs can be inspected and
  // tested hermetically. When present, exact scope membership and every source
  // digest are validated. known_workflow_ids, when nonempty, closes all
  // non-null compatibility links against the reviewed compatibility catalog.
  static SourceDispositionCatalog load_file(
      const std::filesystem::path& catalog_path,
      const std::optional<std::filesystem::path>& repository_root = std::nullopt,
      const std::set<std::string>& known_workflow_ids = {});

  [[nodiscard]] const SourceDispositionDocument& document() const;
  [[nodiscard]] const std::vector<SourceDispositionEntry>& entries() const;
  // SHA-256 over the document with every per-file source_sha256 removed: the
  // classification itself -- paths, classes, entry points, effects, resume
  // grades, workflow links -- and nothing that moves when a classified file's
  // bytes change. That separation is the point. The digest this replaced
  // covered the whole document, so editing any classified source moved it, and
  // the constant pinning it in the native tests became a second per-scope
  // serialization point in a second file. Source bytes are pinned per entry
  // and re-checked against disk; only the review needs a whole-document pin.
  [[nodiscard]] const std::string& reviewed_classification_digest() const;
  [[nodiscard]] const std::optional<std::string>&
  repository_root_identity_display() const;

 private:
  SourceDispositionDocument document_;
  std::string reviewed_classification_digest_;
  std::optional<std::string> repository_root_identity_display_;
};

}  // namespace trainvm
