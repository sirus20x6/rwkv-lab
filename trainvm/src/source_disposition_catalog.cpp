#include "trainvm/source_disposition_catalog.hpp"

#include <fcntl.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

#include "trainvm/json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumCatalogBytes = 4U << 20U;
constexpr std::size_t kMaximumSourceBytes = 256U << 20U;
constexpr std::size_t kMaximumEntries = 4096U;
constexpr std::size_t kMaximumTextBytes = 2048U;

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() { if (value_ >= 0) ::close(value_); }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  [[nodiscard]] int get() const { return value_; }
 private:
  int value_;
};

std::string sha256_bytes(std::string_view bytes) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("could not initialize SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1 ||
      length != 32U) {
    throw std::runtime_error("could not finalize SHA-256");
  }
  constexpr char hexadecimal[] = "0123456789abcdef";
  std::string result;
  result.reserve(64U);
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(hexadecimal[digest[index] >> 4U]);
    result.push_back(hexadecimal[digest[index] & 0x0fU]);
  }
  return result;
}

bool lowercase_sha256(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
      std::ranges::all_of(value.substr(7), [](char character) {
        return std::isdigit(static_cast<unsigned char>(character)) ||
               (character >= 'a' && character <= 'f');
      });
}

bool git_sha1_revision(std::string_view value) {
  return value.size() == 49U && value.starts_with("git-sha1:") &&
      std::ranges::all_of(value.substr(9), [](char character) {
        return std::isdigit(static_cast<unsigned char>(character)) ||
               (character >= 'a' && character <= 'f');
      });
}

bool stable_identifier(std::string_view value) {
  return !value.empty() && value.size() <= 128U &&
      std::isalnum(static_cast<unsigned char>(value.front())) &&
      std::ranges::all_of(value, [](char character) {
        return std::isalnum(static_cast<unsigned char>(character)) ||
               character == '.' || character == '_' || character == '-';
      });
}

void validate_relative_path(std::string_view value, std::string_view field) {
  if (value.empty() || value.size() > 1024U) {
    throw std::invalid_argument(std::string(field) + " must be bounded and nonempty");
  }
  const std::filesystem::path path(value);
  if (path.is_absolute() || path.has_root_path() ||
      path.lexically_normal() != path ||
      std::ranges::any_of(path, [](const auto& component) {
        return component.empty() || component == "." || component == "..";
      })) {
    throw std::invalid_argument(std::string(field) +
                                " must be a normalized repository-relative path");
  }
}

std::string read_bounded_regular_file(const std::filesystem::path& path,
                                      std::size_t maximum,
                                      std::string_view description) {
  FileDescriptor descriptor(::open(path.c_str(), O_RDONLY | O_CLOEXEC |
                                                 O_NOFOLLOW | O_NONBLOCK));
  if (descriptor.get() < 0) {
    throw std::invalid_argument("could not open " + std::string(description) +
                                ": " + std::strerror(errno));
  }
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size < 0 || static_cast<std::uintmax_t>(before.st_size) > maximum) {
    throw std::invalid_argument(std::string(description) +
                                " must be a bounded regular file");
  }
  std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = ::read(descriptor.get(), bytes.data() + offset,
                              bytes.size() - offset);
    if (count <= 0) {
      throw std::invalid_argument("could not read " + std::string(description));
    }
    offset += static_cast<std::size_t>(count);
  }
  char extra{};
  struct stat after {};
  if (::read(descriptor.get(), &extra, 1U) != 0 ||
      ::fstat(descriptor.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec) {
    throw std::invalid_argument(std::string(description) +
                                " changed while being read");
  }
  return bytes;
}

nlohmann::json parse_json_strict(std::string_view text) {
  bool duplicate = false;
  std::vector<std::set<std::string>> keys;
  try {
    nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys.size() <= index + 1U) keys.resize(index + 2U);
            keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (keys.size() <= index) keys.resize(index + 1U);
            duplicate = duplicate || !keys[index].insert(parsed.get<std::string>()).second;
          }
          return true;
        };
    auto value = nlohmann::json::parse(text, callback);
    if (duplicate) {
      throw std::invalid_argument("source disposition catalog contains a duplicate JSON key");
    }
    return value;
  } catch (const nlohmann::json::exception& exception) {
    throw std::invalid_argument("source disposition catalog is not valid JSON: " +
                                std::string(exception.what()));
  }
}

void require_keys(const nlohmann::json& value,
                  const std::set<std::string>& expected,
                  std::string_view where) {
  if (!value.is_object()) {
    throw std::invalid_argument(std::string(where) + " must be an object");
  }
  std::set<std::string> actual;
  for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
    actual.insert(iterator.key());
  }
  if (actual != expected) {
    throw std::invalid_argument(std::string(where) +
                                " has missing or unknown fields");
  }
}

void require_keys_with_optional(const nlohmann::json& value,
                                const std::set<std::string>& required,
                                const std::set<std::string>& optional,
                                std::string_view where) {
  if (!value.is_object()) {
    throw std::invalid_argument(std::string(where) + " must be an object");
  }
  std::set<std::string> actual;
  for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
    if (!required.contains(iterator.key()) && !optional.contains(iterator.key())) {
      throw std::invalid_argument(std::string(where) + " has an unknown field: " +
                                  iterator.key());
    }
    actual.insert(iterator.key());
  }
  if (!std::ranges::includes(actual, required)) {
    throw std::invalid_argument(std::string(where) + " has missing fields");
  }
}

template <typename Enum>
Enum parse_enum(std::string_view value, const std::vector<std::pair<std::string_view, Enum>>& values,
                std::string_view field) {
  for (const auto& [name, result] : values) if (name == value) return result;
  throw std::invalid_argument("unknown " + std::string(field) + ": " + std::string(value));
}

SourceDispositionClass parse_class(std::string_view value) {
  using E = SourceDispositionClass;
  return parse_enum<E>(value, {{"executable_operation", E::executable_operation},
      {"wrapper_alias", E::wrapper_alias},
      {"internal_utility_data_tool", E::internal_utility_data_tool},
      {"supervisor_orchestrator", E::supervisor_orchestrator},
      {"install_bootstrap", E::install_bootstrap},
      {"test_benchmark", E::test_benchmark},
      {"explicit_exclusion", E::explicit_exclusion},
      {"internal_model_layer_kernel_optimizer_library",
       E::internal_model_layer_kernel_optimizer_library},
      {"data_eval_export_tool", E::data_eval_export_tool},
      {"research_oracle_poc", E::research_oracle_poc},
      {"test_fixture", E::test_fixture}}, "class");
}

SourceResumeRelevance parse_resume(std::string_view value) {
  using E = SourceResumeRelevance;
  return parse_enum<E>(value, {{"none", E::none}, {"restart_only", E::restart_only},
      {"terminal_checkpoint", E::terminal_checkpoint}, {"compatible", E::compatible},
      {"exact_candidate", E::exact_candidate}, {"consumer_owned", E::consumer_owned},
      {"none_or_self_test_only", E::none_or_self_test_only}}, "resume_relevance");
}

std::string require_string(const nlohmann::json& value, std::string_view field) {
  const auto& item = value.at(field);
  if (!item.is_string()) throw std::invalid_argument(std::string(field) + " must be a string");
  return item.get<std::string>();
}

std::vector<std::string> require_string_array(const nlohmann::json& value,
                                              std::string_view field) {
  const auto& item = value.at(field);
  if (!item.is_array() || item.empty()) {
    throw std::invalid_argument(std::string(field) + " must be a nonempty array");
  }
  std::vector<std::string> result;
  std::set<std::string> unique;
  for (const auto& element : item) {
    if (!element.is_string() || !stable_identifier(element.get<std::string>()) ||
        !unique.insert(element.get<std::string>()).second) {
      throw std::invalid_argument(std::string(field) +
                                  " must contain unique stable identifiers");
    }
    result.push_back(element.get<std::string>());
  }
  return result;
}

std::vector<std::string> require_bounded_string_array(
    const nlohmann::json& value, std::string_view field) {
  const auto& item = value.at(field);
  if (!item.is_array() || item.size() > kMaximumEntries) {
    throw std::invalid_argument(std::string(field) + " must be a bounded array");
  }
  std::vector<std::string> result;
  std::set<std::string> unique;
  for (const auto& element : item) {
    if (!element.is_string() || element.get<std::string>().empty() ||
        element.get<std::string>().size() > kMaximumTextBytes ||
        !unique.insert(element.get<std::string>()).second) {
      throw std::invalid_argument(std::string(field) +
                                  " must contain unique bounded strings");
    }
    result.push_back(element.get<std::string>());
  }
  return result;
}

bool known_effect(std::string_view value) {
  static constexpr std::array<std::string_view, 10> effects = {
      "gpu_compute",       "human_review",      "install_environment",
      "mutate_source",     "network",           "process_control",
      "read_source",       "serve_http",        "subprocess",
      "write_artifact",
  };
  return std::ranges::contains(effects, value);
}

bool known_coverage(std::string_view value) {
  static constexpr std::array<std::string_view, 4> values = {
      "direct", "transitive", "uncovered", "direct_gap"};
  return std::ranges::contains(values, value);
}

bool path_in_scope(const std::filesystem::path& path,
                   const SourceDispositionScope& scope) {
  if (path.extension().empty() ||
      !std::ranges::contains(scope.extensions, path.extension().string())) return false;
  const auto relative = path.lexically_relative(scope.prefix);
  if (relative.empty() || relative.native().starts_with("..")) return false;
  return scope.recursive || std::distance(relative.begin(), relative.end()) == 1;
}

std::set<std::string> enumerate_scope(const std::filesystem::path& root,
                                      const SourceDispositionScope& scope) {
  const auto base = root / scope.prefix;
  std::error_code error;
  if (!std::filesystem::is_directory(base, error) || error) {
    throw std::invalid_argument("source disposition scope prefix is not a directory");
  }
  std::set<std::string> result;
  const auto inspect = [&](const auto& item) {
    const auto status = item.symlink_status();
    if (!std::filesystem::is_regular_file(status)) return;
    const auto relative = item.path().lexically_relative(root);
    if (path_in_scope(relative, scope)) result.insert(relative.generic_string());
  };
  if (scope.recursive) {
    for (const auto& item : std::filesystem::recursive_directory_iterator(base)) inspect(item);
  } else {
    for (const auto& item : std::filesystem::directory_iterator(base)) inspect(item);
  }
  return result;
}

}  // namespace

std::string source_tree_digest(const std::vector<SourceDispositionEntry>& entries) {
  std::string material = "trainvm.source-disposition-tree/v1";
  for (const auto& entry : entries) {
    material.push_back('\0');
    material.append(entry.source_path);
    material.push_back('\0');
    material.append(entry.source_sha256);
  }
  return "sha256:" + sha256_bytes(material);
}

SourceDispositionCatalog SourceDispositionCatalog::load_file(
    const std::filesystem::path& catalog_path,
    const std::optional<std::filesystem::path>& repository_root,
    const std::set<std::string>& known_workflow_ids) {
  if (catalog_path.empty() || !catalog_path.is_absolute()) {
    throw std::invalid_argument("source disposition catalog path must be absolute");
  }
  const auto bytes = read_bounded_regular_file(catalog_path, kMaximumCatalogBytes,
                                               "source disposition catalog");
  const auto root = parse_json_strict(bytes);
  // source_tree_digest is deliberately absent from this set. It is a pure
  // function of entries, so storing it added no information and cost every
  // concurrent change to the scope a guaranteed conflict on one line. Because
  // require_keys compares the key set exactly, a document that still declares
  // one is refused here rather than silently ignored -- a stale stored pin
  // must not be able to sit in a catalog looking authoritative.
  require_keys(root, {"api_version", "authority", "source_repository",
                      "source_revision", "source_scope", "entries"},
               "source disposition catalog");
  SourceDispositionCatalog catalog;
  auto& document = catalog.document_;
  document.api_version = require_string(root, "api_version");
  document.authority = require_string(root, "authority");
  document.source_repository = require_string(root, "source_repository");
  document.source_revision = require_string(root, "source_revision");
  if (document.api_version != "trainvm.source-dispositions/v1" ||
      document.authority != "compatibility_evidence_only") {
    throw std::invalid_argument("unsupported or authority-bearing source disposition catalog");
  }
  if (!stable_identifier(document.source_repository) ||
      !git_sha1_revision(document.source_revision)) {
    throw std::invalid_argument("invalid source disposition provenance");
  }

  const auto& scope = root.at("source_scope");
  require_keys(scope, {"prefix", "recursive", "extensions"}, "source_scope");
  document.source_scope.prefix = require_string(scope, "prefix");
  validate_relative_path(document.source_scope.prefix, "source_scope.prefix");
  if (!scope.at("recursive").is_boolean()) {
    throw std::invalid_argument("source_scope.recursive must be boolean");
  }
  document.source_scope.recursive = scope.at("recursive").get<bool>();
  document.source_scope.extensions =
      require_bounded_string_array(scope, "extensions");
  if (document.source_scope.extensions.empty()) {
    throw std::invalid_argument("source_scope.extensions must be nonempty");
  }
  for (const auto& extension : document.source_scope.extensions) {
    if (!extension.starts_with('.') || extension.contains('/')) {
      throw std::invalid_argument("source_scope extensions must be dotted suffixes");
    }
  }

  const auto& entries = root.at("entries");
  if (!entries.is_array() || entries.empty() || entries.size() > kMaximumEntries) {
    throw std::invalid_argument("source disposition entries must be nonempty and bounded");
  }
  std::set<std::string> paths;
  for (const auto& value : entries) {
    require_keys_with_optional(
        value,
        {"source_path", "class", "canonical_entry_point", "effects",
         "resume_relevance", "compatibility_workflow_id", "source_sha256"},
        {"family", "coverage", "consumers", "compatibility_workflow_ids", "language"},
        "source disposition entry");
    SourceDispositionEntry entry;
    entry.source_path = require_string(value, "source_path");
    validate_relative_path(entry.source_path, "source_path");
    if (!path_in_scope(entry.source_path, document.source_scope) ||
        !paths.insert(entry.source_path).second) {
      throw std::invalid_argument("source disposition paths must be unique and in scope");
    }
    entry.disposition_class = parse_class(require_string(value, "class"));
    entry.canonical_entry_point = require_string(value, "canonical_entry_point");
    if (entry.canonical_entry_point.empty() ||
        entry.canonical_entry_point.size() > kMaximumTextBytes) {
      throw std::invalid_argument("canonical_entry_point must be bounded and nonempty");
    }
    entry.effects = require_string_array(value, "effects");
    if (!std::ranges::all_of(entry.effects, known_effect)) {
      throw std::invalid_argument(
          "effects must use the closed source-disposition vocabulary");
    }
    entry.resume_relevance = parse_resume(require_string(value, "resume_relevance"));
    const auto& workflow = value.at("compatibility_workflow_id");
    if (!workflow.is_null()) {
      if (!workflow.is_string() || !stable_identifier(workflow.get<std::string>())) {
        throw std::invalid_argument("compatibility_workflow_id must be null or stable ID");
      }
      entry.compatibility_workflow_id = workflow.get<std::string>();
      if (!known_workflow_ids.empty() &&
          !known_workflow_ids.contains(*entry.compatibility_workflow_id)) {
        throw std::invalid_argument("unknown compatibility_workflow_id: " +
                                    *entry.compatibility_workflow_id);
      }
    }
    entry.source_sha256 = require_string(value, "source_sha256");
    if (!lowercase_sha256(entry.source_sha256)) {
      throw std::invalid_argument("source_sha256 must be lowercase SHA-256");
    }
    for (const auto field : {"family", "coverage"}) {
      if (value.contains(field)) {
        const auto rich_value = require_string(value, field);
        if (!stable_identifier(rich_value)) {
          throw std::invalid_argument(std::string(field) + " must be a stable identifier");
        }
        if (std::string_view(field) == "family") {
          entry.family = rich_value;
        } else {
          if (!known_coverage(rich_value)) {
            throw std::invalid_argument(
                "coverage must use the closed source-disposition vocabulary");
          }
          entry.coverage = rich_value;
        }
      }
    }
    if (value.contains("language")) {
      const auto language = require_string(value, "language");
      if (!stable_identifier(language)) {
        throw std::invalid_argument("language must be a stable identifier");
      }
      entry.language = language;
    }
    if (value.contains("consumers")) {
      entry.consumers = require_bounded_string_array(value, "consumers");
    }
    if (value.contains("compatibility_workflow_ids")) {
      entry.compatibility_workflow_ids =
          require_string_array(value, "compatibility_workflow_ids");
      for (const auto& identifier : entry.compatibility_workflow_ids) {
        if (!known_workflow_ids.empty() && !known_workflow_ids.contains(identifier)) {
          throw std::invalid_argument("unknown compatibility_workflow_ids member: " +
                                      identifier);
        }
      }
      if (entry.compatibility_workflow_id &&
          !std::ranges::contains(entry.compatibility_workflow_ids,
                                 *entry.compatibility_workflow_id)) {
        throw std::invalid_argument(
            "compatibility_workflow_id must occur in compatibility_workflow_ids");
      }
    }
    document.entries.push_back(std::move(entry));
  }
  if (!std::ranges::is_sorted(document.entries, {}, &SourceDispositionEntry::source_path)) {
    throw std::invalid_argument("source disposition entries must be sorted by source_path");
  }
  document.source_tree_digest = source_tree_digest(document.entries);

  if (repository_root) {
    if (!repository_root->is_absolute()) {
      throw std::invalid_argument("source repository root must be absolute");
    }
    const auto actual_paths = enumerate_scope(*repository_root, document.source_scope);
    if (actual_paths != paths) {
      throw std::invalid_argument("source disposition scope has missing or stale paths");
    }
    for (const auto& entry : document.entries) {
      const auto source = read_bounded_regular_file(*repository_root / entry.source_path,
                                                    kMaximumSourceBytes,
                                                    "source disposition input");
      if ("sha256:" + sha256_bytes(source) != entry.source_sha256) {
        throw std::invalid_argument("source digest drift: " + entry.source_path);
      }
    }
    struct stat status {};
    FileDescriptor descriptor(::open(repository_root->c_str(), O_RDONLY | O_DIRECTORY |
                                                               O_CLOEXEC | O_NOFOLLOW));
    if (descriptor.get() < 0 || ::fstat(descriptor.get(), &status) != 0) {
      throw std::invalid_argument("could not identify source repository root");
    }
    catalog.repository_root_identity_display_ = repository_root->string() + " [dev=" +
        std::to_string(static_cast<std::uintmax_t>(status.st_dev)) + ",ino=" +
        std::to_string(static_cast<std::uintmax_t>(status.st_ino)) + "]";
  }
  // Digest the classification, not the file bytes it classifies. Erasing the
  // per-entry source_sha256 leaves a value that moves when a review decision
  // moves and stays still when a classified source is merely edited, so the
  // constant pinning it stops being a line every concurrent change rewrites.
  // Nothing is lost: source bytes are pinned by those same per-entry digests,
  // which merge cleanly because two changes touch two different entries.
  auto reviewed = root;
  for (auto& entry : reviewed.at("entries")) {
    entry.erase("source_sha256");
  }
  catalog.reviewed_classification_digest_ = "sha256:" + sha256_bytes(reviewed.dump());
  return catalog;
}

const SourceDispositionDocument& SourceDispositionCatalog::document() const {
  return document_;
}
const std::vector<SourceDispositionEntry>& SourceDispositionCatalog::entries() const {
  return document_.entries;
}
const std::string& SourceDispositionCatalog::reviewed_classification_digest() const {
  return reviewed_classification_digest_;
}
const std::optional<std::string>&
SourceDispositionCatalog::repository_root_identity_display() const {
  return repository_root_identity_display_;
}

}  // namespace trainvm
