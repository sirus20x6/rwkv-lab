#include "trainvm/source_disposition_catalog.hpp"

#include <openssl/evp.h>

#include <array>
#include <cstdio>
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
#include <utility>
#include <vector>

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
  return {{"api_version", "trainvm.source-dispositions/v1"},
          {"authority", "compatibility_evidence_only"},
          {"source_repository", "fixture"},
          {"source_revision", "git-sha1:0000000000000000000000000000000000000000"},
          {"source_scope", {{"prefix", "scripts"}, {"recursive", false},
                            {"extensions", {".py", ".sh"}}}},
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

// Run the Python mirror over exactly these entries and return its answer.
//
// This is the whole point of the file. scripts/print_disposition_digests.py
// carries a second, hand-written implementation of the tree-digest fold, and
// until now nothing compared the two directly: both were compared against a
// digest stored in each catalog, so they were cross-checked only transitively,
// and only at the moment somebody regenerated that stored value. The stored
// value is gone, so the comparison has to be made here, with both
// implementations driven over the same entries and nothing in between.
//
// It throws rather than returning an error string. A cross-language agreement
// check that degrades to a skip when the interpreter is missing is a check
// that reports success for the wrong reason.
std::string python_tree_digest(const std::filesystem::path& repository_root,
                               const std::filesystem::path& scratch,
                               const std::vector<trainvm::SourceDispositionEntry>& entries) {
  static int sequence = 0;
  nlohmann::json payload = nlohmann::json::array();
  for (const auto& entry : entries) {
    payload.push_back({{"source_path", entry.source_path},
                       {"source_sha256", entry.source_sha256}});
  }
  const auto request = scratch / ("entries-" + std::to_string(++sequence) + ".json");
  write_text(request, payload.dump());
  const std::string command =
      "python3 '" + (repository_root / "scripts/print_disposition_digests.py").string() +
      "' '" + request.string() + "' --tree-digest";
  struct PipeCloser {
    void operator()(FILE* stream) const { if (stream != nullptr) ::pclose(stream); }
  };
  std::unique_ptr<FILE, PipeCloser> pipe(::popen(command.c_str(), "r"));
  if (!pipe) throw std::runtime_error("could not run the Python digest mirror");
  std::string output;
  std::array<char, 256> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    output.append(buffer.data());
  }
  const int status = ::pclose(pipe.release());
  if (status != 0) {
    throw std::runtime_error("the Python digest mirror exited " + std::to_string(status) +
                             " for " + request.string());
  }
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) output.pop_back();
  if (output.size() != 71U || !output.starts_with("sha256:")) {
    throw std::runtime_error("the Python digest mirror printed no digest: " + output);
  }
  return output;
}

// Run the Python checker over one fixture catalog and return its exit status.
//
// The fold agreement above is about a computed value. This is about the other
// half of the contract: which documents each side REFUSES. It matters more than
// it looks, because the loader's source-byte verification only runs when
// load_file is handed a repository root, and the live-root check below is
// guarded by TRAINVM_LEGACY_SOURCE_ROOT, unset in CI. For the scripts and RWKV
// catalogs' 294 sources, `print_disposition_digests.py --check` is the only
// instrument in a hosted run, so its refusal conditions are the effective
// contract and have to be the ones written in C++.
int python_check_status(const std::filesystem::path& repository_root,
                        const std::filesystem::path& catalog,
                        const std::filesystem::path& source_root) {
  const std::string command =
      "python3 '" + (repository_root / "scripts/print_disposition_digests.py").string() +
      "' '" + catalog.string() + "' --root '" + source_root.string() +
      "' --check > /dev/null 2>&1";
  const int status = std::system(command.c_str());
  if (status < 0) throw std::runtime_error("could not run the Python pin checker");
  return status;
}

trainvm::SourceDispositionEntry synthetic(std::string path, std::string leaf) {
  trainvm::SourceDispositionEntry entry;
  entry.source_path = std::move(path);
  entry.source_sha256 = std::move(leaf);
  return entry;
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
  check(scripts_catalog.entries().size() == 129U,
        "script disposition catalog covers all 129 reviewed scripts");
  check(scripts_catalog.reviewed_classification_digest() == "sha256:931285aaaa688f5a07211d5b114885a315b719b8ce94ba04678c45e0758536c4",
        std::string("script catalog pins the exact reviewed classification") +
            " (computed " + scripts_catalog.reviewed_classification_digest() + ")");
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
  check(classes[trainvm::SourceDispositionClass::test_benchmark] == 4U,
        "script catalog pins test/benchmark count");
  check(classes[trainvm::SourceDispositionClass::explicit_exclusion] == 1U,
        "script catalog pins explicit exclusion count");
  check(linked_script_sources == 65U && linked_workflow_ids.size() == 48U &&
            scripts_catalog.entries().size() - linked_script_sources == 64U,
        "script catalog pins 65 directly linked sources, 48 primary workflows, and 64 gaps");
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
  check(dashboard_catalog.entries().size() == 4U,
        "dashboard disposition catalog covers every reviewed dashboard module");
  std::map<trainvm::SourceDispositionClass, std::size_t> dashboard_classes;
  for (const auto& entry : dashboard_catalog.entries()) {
    ++dashboard_classes[entry.disposition_class];
  }
  check(dashboard_classes[trainvm::SourceDispositionClass::explicit_exclusion] == 1U &&
            dashboard_classes[trainvm::SourceDispositionClass::
                                  internal_utility_data_tool] == 1U &&
            dashboard_classes[trainvm::SourceDispositionClass::test_fixture] == 2U,
        "the stale instrumented trainer fork stays an explicit exclusion, not an operation");

  const auto rwkv_catalog = trainvm::SourceDispositionCatalog::load_file(
      rwkv_checked, std::nullopt, known_ids);
  check(rwkv_catalog.entries().size() == 165U,
        "RWKV disposition catalog covers all 165 reviewed modules");
  check(rwkv_catalog.reviewed_classification_digest() == "sha256:1bb5314b848e75af740d537584d05acd32c967deaa3ba15198a58750cbf37b71",
        std::string("RWKV catalog pins the exact reviewed classification") +
            " (computed " + rwkv_catalog.reviewed_classification_digest() + ")");
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

  // The two implementations of the tree-digest fold, driven over the same
  // entries, with no stored value between them.
  //
  // The real catalogs come first because they are the input that matters: 129
  // and 165 entries of genuine paths and digests. The synthetic vectors after
  // them exist because agreeing on well-behaved input is weak evidence -- they
  // are the cases where two plausible implementations disagree. The collision
  // pair is the important one: ("ab", "c") and ("a", "bc") concatenate to the
  // same bytes and must not produce the same digest, which is what the NUL
  // framing buys and what a length-prefixed or unseparated fold would lose.
  const std::vector<std::pair<std::string, std::vector<trainvm::SourceDispositionEntry>>>
      agreement_vectors{
          {"the checked-in scripts catalog", scripts_catalog.entries()},
          {"the checked-in rwkv-lab catalog", rwkv_catalog.entries()},
          {"the checked-in dashboard catalog", dashboard_catalog.entries()},
          {"no entries at all", {}},
          {"a single entry", {synthetic("scripts/a.py", "sha256:" + sha256("a"))}},
          {"separator collision, split one way",
           {synthetic("ab", "c"), synthetic("d", "e")}},
          {"separator collision, split the other",
           {synthetic("a", "bc"), synthetic("d", "e")}},
          {"the same entries in the other order",
           {synthetic("d", "e"), synthetic("a", "bc")}},
          {"paths that exercise JSON escaping",
           {synthetic("scripts/\"quoted\".py", "sha256:" + sha256("q")),
            synthetic("scripts/back\\slash.py", "sha256:" + sha256("b")),
            synthetic("scripts/\xc3\xbcnicode.py", "sha256:" + sha256("u")),
            synthetic(std::string("scripts/embedded\0nul.py", 23U), "sha256:" + sha256("n"))}},
      };
  std::set<std::string> distinct_agreement_digests;
  for (const auto& [description, entries] : agreement_vectors) {
    const auto native = trainvm::source_tree_digest(entries);
    const auto mirrored = python_tree_digest(dashboard_root, temporary.path(), entries);
    distinct_agreement_digests.insert(native);
    check(native == mirrored,
          "the Python mirror agrees with the C++ fold over " + description +
              " (C++ " + native + ", Python " + mirrored + ")");
  }
  // Guards the loop above against agreeing vacuously: if the fold ever
  // collapsed to a constant, every vector would still "agree".
  check(distinct_agreement_digests.size() == agreement_vectors.size(),
        "every agreement vector folds to a distinct digest (" +
            std::to_string(distinct_agreement_digests.size()) + " of " +
            std::to_string(agreement_vectors.size()) + ")");

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

  // Both instruments, one fixture. Each state below is checked from the C++
  // side and from the Python side, and they must agree -- a state where the
  // loader refuses and the checker is green is exactly the failure this pair
  // exists to prevent.
  write_text(repository / "scripts/a.py", "print('fixture')\n");
  write_json(fixture_catalog, one_file_document("print('fixture')\n"));
  check(trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository)
            .entries().size() == 1U &&
            python_check_status(dashboard_root, fixture_catalog, repository) == 0,
        "both instruments accept the fixture before it is broken");

  // A source replaced by a symlink to byte-identical content. Every digest
  // still matches; what has changed is that the reviewed file is gone. The C++
  // refuses it twice over -- enumerate_scope counts regular files only, and
  // read_bounded_regular_file opens O_NOFOLLOW -- and a checker that merely
  // hashes the path cannot see it at all.
  write_text(repository / "identical-twin.py", "print('fixture')\n");
  fs::remove(repository / "scripts/a.py");
  fs::create_symlink(repository / "identical-twin.py", repository / "scripts/a.py");
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }) && python_check_status(dashboard_root, fixture_catalog, repository) != 0,
        "both instruments refuse a source replaced by a symlink to identical bytes");
  fs::remove(repository / "scripts/a.py");
  fs::remove(repository / "identical-twin.py");
  write_text(repository / "scripts/a.py", "print('fixture')\n");

  // An unnormalized source_path. validate_relative_path refuses it before
  // anything is opened; Python would otherwise resolve the `..` itself and hash
  // whatever it landed on.
  auto unnormalized = one_file_document("print('fixture')\n");
  unnormalized["entries"][0]["source_path"] = "scripts/../scripts/a.py";
  write_json(fixture_catalog, unnormalized);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }) && python_check_status(dashboard_root, fixture_catalog, repository) != 0,
        "both instruments refuse an unnormalized source_path");
  // Deliberately a spelling that still lands on the real file. An absolute or
  // escaping path is refused by Python for a second reason -- it resolves to
  // nothing -- so it would keep this check green with the spelling rule
  // removed, which is the trap of asserting that something failed rather than
  // why. A redundant "." component resolves to exactly the pinned bytes.
  unnormalized["entries"][0]["source_path"] = "scripts/./a.py";
  write_json(fixture_catalog, unnormalized);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog, repository);
        }) && python_check_status(dashboard_root, fixture_catalog, repository) != 0,
        "both instruments refuse a source_path with a redundant '.' component");
  write_json(fixture_catalog, one_file_document("print('fixture')\n"));

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
  // A catalog that still carries the retired stored pin must be refused, not
  // tolerated. Ignoring it would let a value that agrees with nothing sit in a
  // document looking authoritative, which is the failure mode this repository
  // keeps paying for.
  malformed = one_file_document("print('fixture')\n");
  malformed["source_tree_digest"] =
      trainvm::source_tree_digest({synthetic("scripts/a.py",
                                             "sha256:" + sha256("print('fixture')\n"))});
  write_json(fixture_catalog, malformed);
  check(throws_invalid_argument([&] {
          (void)trainvm::SourceDispositionCatalog::load_file(fixture_catalog);
        }), "a stored source_tree_digest is rejected, even a correct one");
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
