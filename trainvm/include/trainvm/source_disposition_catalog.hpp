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
  std::string source_tree_digest;
  std::vector<SourceDispositionEntry> entries;
};

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
  [[nodiscard]] const std::string& catalog_digest() const;
  [[nodiscard]] const std::optional<std::string>&
  repository_root_identity_display() const;

 private:
  SourceDispositionDocument document_;
  std::string catalog_digest_;
  std::optional<std::string> repository_root_identity_display_;
};

}  // namespace trainvm
