#include "trainvm/source_disposition_catalog.hpp"

#include <openssl/evp.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/json.hpp"

namespace {

int failures = 0;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    std::array<char, 64> pattern{};
    constexpr std::string_view prefix = "/tmp/trainvm-source-dispositions-XXXXXX";
    std::ranges::copy(prefix, pattern.begin());
    const auto created = ::mkdtemp(pattern.data());
    if (created == nullptr) throw std::runtime_error("could not create temporary directory");
    path_ = created;
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }
 private:
  std::filesystem::path path_;
};

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

template <typename Function>
bool throws_invalid_argument(Function&& function) {
  try { function(); } catch (const std::invalid_argument&) { return true; }
  return false;
}

std::string sha256(std::string_view bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("SHA-256 initialization failed");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 || length != 32U) {
    throw std::runtime_error("SHA-256 finalization failed");
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(hex[digest[index] >> 4U]);
    result.push_back(hex[digest[index] & 15U]);
  }
  return result;
}

void write_text(const std::filesystem::path& path, std::string_view text) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output) throw std::runtime_error("could not write fixture");
  output << text;
}

nlohmann::json one_file_document(std::string_view source_bytes) {
  const std::string leaf = "sha256:" + sha256(source_bytes);
  std::string tree = "trainvm.source-disposition-tree/v1";
  tree.append("\0scripts/a.py\0", 14U);
  tree.append(leaf);
  return {{"api_version", "trainvm.source-dispositions/v1"},
          {"authority", "compatibility_evidence_only"},
          {"source_repository", "fixture"},
          {"source_revision", "git-sha1:0000000000000000000000000000000000000000"},
          {"source_scope", {{"prefix", "scripts"}, {"recursive", false},
                            {"extensions", {".py", ".sh"}}}},
          {"source_tree_digest", "sha256:" + sha256(tree)},
          {"entries", {{{"source_path", "scripts/a.py"},
                         {"class", "executable_operation"},
                         {"canonical_entry_point", "python scripts/a.py"},
                         {"effects", {"read_source"}},
                         {"resume_relevance", "none"},
                         {"compatibility_workflow_id", nullptr},
                         {"source_sha256", leaf}}}}};
}

void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
  write_text(path, value.dump(2) + "\n");
}

std::set<std::string> compatibility_ids(const std::filesystem::path& path) {
  std::ifstream input(path);
  nlohmann::json value;
  input >> value;
  std::set<std::string> result;
  for (const auto& entry : value.at("entries")) {
    result.insert(entry.at("stable_id").get<std::string>());
  }
  return result;
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dashboard_root = fs::canonical(TRAINVM_SOURCE_ROOT);
  const auto known_ids = compatibility_ids(
      dashboard_root / "docs/experiment-vm/compatibility-workflows.v1.json");
  const fs::path scripts_checked =
      dashboard_root / "docs/experiment-vm/source-dispositions.scripts.v1.json";
  const fs::path rwkv_checked =
      dashboard_root / "docs/experiment-vm/source-dispositions.rwkv-lab.v1.json";
  const auto scripts_catalog = trainvm::SourceDispositionCatalog::load_file(
      scripts_checked, std::nullopt, known_ids);
  check(scripts_catalog.entries().size() == 128U,
        "script disposition catalog covers all 128 reviewed scripts");
  check(scripts_catalog.catalog_digest() == "sha256:f4c86a147116f44bf2d5057c39f6d0ee1b5dcc0be48492a238634ce6bdcc3a64",
        std::string("script catalog pins the exact reviewed canonical mapping") +
            " (computed " + scripts_catalog.catalog_digest() + ")");
  std::map<trainvm::SourceDispositionClass, std::size_t> classes;
  std::set<std::string> linked_workflow_ids;
  std::size_t linked_script_sources = 0;
  std::size_t compatibility_references = 0;
  for (const auto& entry : scripts_catalog.entries()) {
    ++classes[entry.disposition_class];
    if (entry.compatibility_workflow_id) {
      ++linked_script_sources;
      linked_workflow_ids.insert(*entry.compatibility_workflow_id);
    }
    compatibility_references += entry.compatibility_workflow_ids.size();
  }
  check(classes[trainvm::SourceDispositionClass::executable_operation] == 31U,
        "script catalog pins executable operation count");
  check(classes[trainvm::SourceDispositionClass::wrapper_alias] == 14U,
        "script catalog pins wrapper count");
  check(classes[trainvm::SourceDispositionClass::internal_utility_data_tool] == 38U,
        "script catalog pins internal utility count");
  check(classes[trainvm::SourceDispositionClass::supervisor_orchestrator] == 38U,
        "script catalog pins supervisor count");
  check(classes[trainvm::SourceDispositionClass::install_bootstrap] == 3U,
        "script catalog pins bootstrap count");
  check(classes[trainvm::SourceDispositionClass::test_benchmark] == 3U,
        "script catalog pins test/benchmark count");
  check(classes[trainvm::SourceDispositionClass::explicit_exclusion] == 1U,
        "script catalog pins explicit exclusion count");
  check(linked_script_sources == 65U && linked_workflow_ids.size() == 48U &&
            scripts_catalog.entries().size() - linked_script_sources == 63U,
        "script catalog pins 65 directly linked sources, 48 primary workflows, and 63 gaps");
  std::set<std::string> all_workflow_ids;
  for (const auto& entry : scripts_catalog.entries()) {
    all_workflow_ids.insert(entry.compatibility_workflow_ids.begin(),
                            entry.compatibility_workflow_ids.end());
  }
  check(compatibility_references == 67U && all_workflow_ids.size() == 50U,
        "script catalog preserves all 67 references across 50 workflows");

  // The dashboard tree is the third reviewed source root. Its scope is
  // recursive, so a new Python file anywhere under dashboard/ must be
  // classified rather than silently escaping review by living in a
  // subdirectory the other two catalogs do not reach.
  const fs::path dashboard_checked =
      dashboard_root / "docs/experiment-vm/source-dispositions.dashboard.v1.json";
  const auto dashboard_catalog = trainvm::SourceDispositionCatalog::load_file(
      dashboard_checked, dashboard_root, known_ids);
  check(dashboard_catalog.entries().size() == 3U,
        "dashboard disposition catalog covers every reviewed dashboard module");
  std::map<trainvm::SourceDispositionClass, std::size_t> dashboard_classes;
  for (const auto& entry : dashboard_catalog.entries()) {
    ++dashboard_classes[entry.disposition_class];
  }
  check(dashboard_classes[trainvm::SourceDispositionClass::explicit_exclusion] == 1U &&
            dashboard_classes[trainvm::SourceDispositionClass::
                                  internal_utility_data_tool] == 1U &&
            dashboard_classes[trainvm::SourceDispositionClass::test_fixture] == 1U,
        "the stale instrumented trainer fork stays an explicit exclusion, not an operation");

  const auto rwkv_catalog = trainvm::SourceDispositionCatalog::load_file(
      rwkv_checked, std::nullopt, known_ids);
  check(rwkv_catalog.entries().size() == 165U,
        "RWKV disposition catalog covers all 165 reviewed modules");
  check(rwkv_catalog.catalog_digest() == "sha256:d9a35833a46f9dda16860b3a6d0521f44fda6ffc867ee0a0071ab700daf6dfab",
        std::string("RWKV catalog pins the exact reviewed canonical mapping") +
            " (computed " + rwkv_catalog.catalog_digest() + ")");
  classes.clear();
  std::map<std::string, std::size_t> coverage;
  bool complete_rwkv_metadata = true;
  for (const auto& entry : rwkv_catalog.entries()) {
    ++classes[entry.disposition_class];
    complete_rwkv_metadata = complete_rwkv_metadata && entry.language.has_value() &&
        entry.family.has_value() && entry.coverage.has_value();
    if (entry.coverage) ++coverage[*entry.coverage];
  }
  check(classes[trainvm::SourceDispositionClass::executable_operation] == 24U &&
            classes[trainvm::SourceDispositionClass::data_eval_export_tool] == 28U &&
            classes[trainvm::SourceDispositionClass::
                        internal_model_layer_kernel_optimizer_library] == 60U &&
            classes[trainvm::SourceDispositionClass::research_oracle_poc] == 52U &&
            classes[trainvm::SourceDispositionClass::explicit_exclusion] == 1U,
        "RWKV catalog pins the audited 24/28/60/52/1 role split");
  check(complete_rwkv_metadata,
        "every RWKV disposition retains language, family, and coverage metadata");
  check(coverage["direct"] == 68U && coverage["transitive"] == 60U &&
            coverage["uncovered"] == 36U && coverage["direct_gap"] == 1U,
        "RWKV catalog pins the audited 68/60/36/1 coverage split");

  TemporaryDirectory temporary;
  const fs::path repository = temporary.path() / "repository";
  const fs::path fixture_catalog = temporary.path() / "catalog.json";
  write_text(repository / "scripts/a.py", "print('fixture')\n");
  write_json(fixture_catalog, one_file_document("print('fixture')\n"));
  check(trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository)
            .entries().size() == 1U,
        "hermetic source-root fixture validates");

  write_text(repository / "scripts/a.py", "print('drift')\n");
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }), "source byte drift is rejected");
  write_text(repository / "scripts/a.py", "print('fixture')\n");
  write_text(repository / "scripts/stale.sh", "#!/bin/sh\n");
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }), "unreviewed source paths are rejected");
  fs::remove(repository / "scripts/stale.sh");
  fs::remove(repository / "scripts/a.py");
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }), "missing reviewed source paths are rejected");

  auto malformed = one_file_document("print('fixture')\n");
  malformed["entries"][0]["class"] = "unknown";
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "unknown classes are rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["entries"][0]["resume_relevance"] = "unknown";
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "unknown resume grades are rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["entries"][0]["effects"] = {"read_source", "unknown_effect"};
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "unknown effects are rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["entries"][0]["coverage"] = "unknown";
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "unknown coverage values are rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["entries"][0]["unknown_optional_metadata"] = "not allowed";
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "unknown optional metadata is rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["source_revision"] = "0000000000000000000000000000000000000000";
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "untyped source revisions are rejected");
  malformed = one_file_document("print('fixture')\n");
  malformed["entries"].push_back(malformed["entries"][0]);
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "duplicate source paths are rejected");

  if (const char* source_root = std::getenv("TRAINVM_LEGACY_SOURCE_ROOT");
      source_root != nullptr && *source_root != '\0') {
    const auto canonical_root = fs::canonical(source_root);
    const auto live_scripts = trainvm::SourceDispositionCatalog::load_file(
        scripts_checked, canonical_root, known_ids);
    const auto live_rwkv = trainvm::SourceDispositionCatalog::load_file(
        rwkv_checked, canonical_root, known_ids);
    check(live_scripts.repository_root_identity_display().has_value() &&
              live_rwkv.repository_root_identity_display().has_value(),
          "explicit live source validation reports both root identities");
  }

  if (failures != 0) {
    std::cerr << failures << " source disposition catalog test(s) failed\n";
    return 1;
  }
  std::cout << "source disposition catalog tests passed\n";
  return 0;
}
