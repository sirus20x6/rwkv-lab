#include "trainvm/run_authoring.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <fnmatch.h>
#include <limits>
#include <new>
#include <ranges>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include <linux/openat2.h>
#include <yaml-cpp/yaml.h>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumAuthorRunBytes = 2U * 1024U * 1024U;
constexpr std::size_t kMaximumClientConfigurationBytes = 64U * 1024U;
constexpr std::string_view kMarkerName = ".trainvm-authoring.json";

class ImplementationRequired final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class Descriptor final {
public:
  explicit Descriptor(int value = -1) : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  Descriptor(Descriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  [[nodiscard]] int get() const { return value_; }

private:
  int value_;
};

[[noreturn]] void reject(std::string message) {
  throw RunAuthoringError(std::move(message));
}

bool canonical_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

nlohmann::json yaml_to_json(const YAML::Node &node) {
  if (!node || node.IsNull())
    return nullptr;
  if (node.IsSequence()) {
    nlohmann::json result = nlohmann::json::array();
    for (const auto &child : node)
      result.push_back(yaml_to_json(child));
    return result;
  }
  if (node.IsMap()) {
    nlohmann::json result = nlohmann::json::object();
    std::set<std::string> keys;
    for (const auto &entry : node) {
      if (!entry.first.IsScalar())
        reject("author-run YAML mapping keys must be scalar strings");
      const std::string key = entry.first.Scalar();
      if (!keys.insert(key).second)
        reject("author-run document contains duplicate mapping key: " + key);
      result[key] = yaml_to_json(entry.second);
    }
    return result;
  }
  if (!node.IsScalar())
    reject("author-run YAML contains an unsupported node kind");
  const std::string value = node.Scalar();
  if (node.Tag() == "!")
    return value;
  if (value == "null" || value == "Null" || value == "NULL" || value == "~")
    return nullptr;
  if (value == "true" || value == "True" || value == "TRUE")
    return true;
  if (value == "false" || value == "False" || value == "FALSE")
    return false;
  std::int64_t integer{};
  const auto parsed_integer =
      std::from_chars(value.data(), value.data() + value.size(), integer);
  if (parsed_integer.ec == std::errc{} &&
      parsed_integer.ptr == value.data() + value.size())
    return integer;
  double number{};
  const auto parsed_number =
      std::from_chars(value.data(), value.data() + value.size(), number);
  if (parsed_number.ec == std::errc{} &&
      parsed_number.ptr == value.data() + value.size())
    return number;
  return value;
}

nlohmann::json parse_document(std::string_view source,
                              std::string_view format) {
  if (source.empty() || source.size() > kMaximumAuthorRunBytes)
    reject("author-run document is empty or exceeds 2 MiB");
  try {
    if (format == "json") {
      // nlohmann intentionally accepts duplicate keys with last-write-wins.
      // YAML is only used here as a duplicate-aware structural pre-parse;
      // nlohmann still owns strict JSON syntax and scalar semantics below.
      (void)yaml_to_json(YAML::Load(std::string(source)));
      return nlohmann::json::parse(source);
    }
    if (format == "yaml")
      return yaml_to_json(YAML::Load(std::string(source)));
  } catch (const std::exception &error) {
    reject("author-run document could not be parsed: " +
           std::string(error.what()));
  }
  reject("author-run source_format must be json or yaml");
}

template <typename T>
T decode_closed(const nlohmann::json &document, std::string_view description) {
  T result;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(document, result, "", diagnostics) ||
      !diagnostics.empty() || encode_json(result) != document) {
    std::ostringstream message;
    message << description << " does not match its closed reflected schema";
    if (!diagnostics.empty())
      message << ": " << diagnostics.front().path << " "
              << diagnostics.front().message;
    reject(message.str());
  }
  return result;
}

bool bounded_text(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum &&
         std::ranges::none_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return byte < 0x20U && character != '\n' && character != '\t';
         });
}

bool canonical_absolute(const std::filesystem::path &path) {
  return path.is_absolute() && !path.empty() && path.lexically_normal() == path &&
         path.native().size() <= 4'096U;
}

bool has_training_node(const CompiledPlan &plan) {
  return std::ranges::any_of(
      plan.experiment.spec.workflow.nodes,
      [](const auto &entry) { return entry.second.invoke.training.has_value(); });
}

std::vector<std::string> locked_paths(const CompiledPlan &plan) {
  std::vector<std::string> result;
  if (!plan.experiment.spec.workspace.input_content_roots)
    return result;
  for (const auto &root :
       *plan.experiment.spec.workspace.input_content_roots)
    result.push_back(root.path);
  return result;
}

std::vector<std::string> normalized_root_set_paths(
    const InputContentRootSet &root_set) {
  if (root_set.api_version != kInputContentRootSetApiVersion ||
      root_set.paths.empty() || root_set.paths.size() > 256U)
    reject("input_content must name between 1 and 256 roots using "
           "trainvm.input-content-root-set/v1");
  std::vector<std::string> result = root_set.paths;
  std::ranges::sort(result);
  if (std::ranges::adjacent_find(result) != result.end())
    reject("input_content roots must be unique");
  for (const auto &value : result) {
    if (!canonical_absolute(std::filesystem::path(value)))
      reject("input_content contains a noncanonical absolute path");
  }
  return result;
}

bool path_within_root(const std::filesystem::path &child,
                      const std::filesystem::path &root);

void require_authorized_content_paths(const CompiledPlan &plan,
                                      const InputContentRootSet &root_set) {
  const auto paths = normalized_root_set_paths(root_set);
  const auto &allowed = plan.experiment.spec.workspace.allowed_read_roots;
  if (!allowed)
    reject("input_content cannot be measured without allowed_read_roots");
  for (const auto &path : paths) {
    if (!std::ranges::any_of(*allowed, [&](const std::string &root) {
          return path_within_root(path, root);
        })) {
      reject("input_content path is outside every allowed_read_root");
    }
  }
}

TrainingPreflightDiagnostic diagnostic(std::string code, std::string path,
                                       std::string message,
                                       std::string help) {
  return {.severity = Diagnostic::Severity::error,
          .code = std::move(code),
          .path = std::move(path),
          .message = std::move(message),
          .help = std::move(help)};
}

struct PassiveFile final {
  std::string bytes;
  std::string digest;
};

PassiveFile read_passive_file(const std::filesystem::path &path,
                              std::size_t maximum_bytes) {
  Descriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat status {};
  if (file.get() < 0 || ::fstat(file.get(), &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_size <= 0 ||
      static_cast<std::uint64_t>(status.st_size) > maximum_bytes)
    throw std::runtime_error("passive preflight file is absent, unsafe, or "
                             "outside its bound: " + path.string());
  std::string bytes(static_cast<std::size_t>(status.st_size), '\0');
  std::size_t offset{};
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(file.get(), bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("passive preflight file changed while read: " +
                               path.string());
    offset += static_cast<std::size_t>(count);
  }
  const std::string digest = "sha256:" + sha256_hex(bytes);
  return {.bytes = std::move(bytes), .digest = digest};
}

nlohmann::json read_passive_json_object(const std::filesystem::path &path,
                                        std::size_t maximum_bytes,
                                        std::string &digest) {
  PassiveFile file = read_passive_file(path, maximum_bytes);
  auto value = nlohmann::json::parse(file.bytes);
  if (!value.is_object())
    throw std::runtime_error("passive preflight JSON is not an object: " +
                             path.string());
  digest = std::move(file.digest);
  return value;
}

bool path_within_root(const std::filesystem::path &child,
                      const std::filesystem::path &root) {
  const auto normalized_child = child.lexically_normal();
  const auto normalized_root = root.lexically_normal();
  auto child_iterator = normalized_child.begin();
  for (auto root_iterator = normalized_root.begin();
       root_iterator != normalized_root.end();
       ++root_iterator, ++child_iterator) {
    if (child_iterator == normalized_child.end() ||
        *child_iterator != *root_iterator)
      return false;
  }
  return true;
}

struct PassiveImageHeader final {
  std::string format;
  std::uint32_t width{};
  std::uint32_t height{};
};

PassiveImageHeader inspect_passive_image_header(
    const std::filesystem::path &path) {
  Descriptor file(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  struct stat status {};
  constexpr std::size_t maximum_header_bytes = 1U << 20U;
  if (file.get() < 0 || ::fstat(file.get(), &status) != 0 ||
      !S_ISREG(status.st_mode) || status.st_size < 24 ||
      status.st_size > (1LL << 34U))
    throw std::runtime_error(
        "manifest image sample is absent, unsafe, or unbounded");
  const std::size_t size = std::min(
      maximum_header_bytes, static_cast<std::size_t>(status.st_size));
  std::string bytes(size, '\0');
  std::size_t offset{};
  while (offset < bytes.size()) {
    const ssize_t count =
        ::read(file.get(), bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      throw std::runtime_error("manifest image header changed while read");
    offset += static_cast<std::size_t>(count);
  }
  const auto byte = [&](std::size_t index) {
    return static_cast<std::uint8_t>(
        static_cast<unsigned char>(bytes.at(index)));
  };
  if (bytes.size() >= 24U &&
      bytes.compare(0U, 8U, "\x89PNG\r\n\x1a\n", 8U) == 0) {
    const auto be32 = [&](std::size_t index) {
      return (static_cast<std::uint32_t>(byte(index)) << 24U) |
             (static_cast<std::uint32_t>(byte(index + 1U)) << 16U) |
             (static_cast<std::uint32_t>(byte(index + 2U)) << 8U) |
             static_cast<std::uint32_t>(byte(index + 3U));
    };
    const auto width = be32(16U);
    const auto height = be32(20U);
    if (width == 0U || height == 0U)
      throw std::runtime_error("PNG sample has zero dimensions");
    return {.format = "png", .width = width, .height = height};
  }
  if (bytes.size() >= 4U && byte(0U) == 0xffU && byte(1U) == 0xd8U) {
    std::size_t cursor = 2U;
    while (cursor + 8U < bytes.size()) {
      while (cursor < bytes.size() && byte(cursor) == 0xffU)
        ++cursor;
      if (cursor >= bytes.size())
        break;
      const std::uint8_t marker = byte(cursor++);
      if (marker == 0xd8U || marker == 0x01U)
        continue;
      if (marker == 0xd9U || marker == 0xdaU || cursor + 1U >= bytes.size())
        break;
      const std::size_t length =
          (static_cast<std::size_t>(byte(cursor)) << 8U) |
          static_cast<std::size_t>(byte(cursor + 1U));
      const bool start_of_frame =
          (marker >= 0xc0U && marker <= 0xc3U) ||
          (marker >= 0xc5U && marker <= 0xc7U) ||
          (marker >= 0xc9U && marker <= 0xcbU) ||
          (marker >= 0xcdU && marker <= 0xcfU);
      if (start_of_frame && length >= 7U && cursor + 6U < bytes.size()) {
        const auto height = static_cast<std::uint32_t>(
            (static_cast<std::uint32_t>(byte(cursor + 3U)) << 8U) |
            byte(cursor + 4U));
        const auto width = static_cast<std::uint32_t>(
            (static_cast<std::uint32_t>(byte(cursor + 5U)) << 8U) |
            byte(cursor + 6U));
        if (width == 0U || height == 0U)
          throw std::runtime_error("JPEG sample has zero dimensions");
        return {.format = "jpeg", .width = width, .height = height};
      }
      if (length < 2U || cursor + length > bytes.size())
        break;
      cursor += length;
    }
  }
  throw std::runtime_error(
      "manifest sample is not a bounded supported PNG/JPEG image");
}

bool wildcard_match(std::string_view pattern, std::string_view value) {
  return ::fnmatch(std::string(pattern).c_str(), std::string(value).c_str(),
                   0) == 0;
}

int open_directory_no_symlinks(const std::filesystem::path &path) {
  struct open_how how {};
  how.flags = O_RDONLY | O_DIRECTORY | O_CLOEXEC;
  how.resolve = RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
  return static_cast<int>(
      ::syscall(SYS_openat2, AT_FDCWD, path.c_str(), &how, sizeof(how)));
}

std::string read_marker(int directory) {
  Descriptor file(::openat(directory, kMarkerName.data(),
                           O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0)
    return {};
  struct stat status {};
  if (::fstat(file.get(), &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size <= 0 || status.st_size > 64 * 1024)
    reject("existing run-directory marker is malformed");
  std::string result(static_cast<std::size_t>(status.st_size), '\0');
  std::size_t offset{};
  while (offset < result.size()) {
    const ssize_t count =
        ::read(file.get(), result.data() + offset, result.size() - offset);
    if (count <= 0)
      reject("existing run-directory marker could not be read");
    offset += static_cast<std::size_t>(count);
  }
  return result;
}

bool directory_empty(int directory) {
  const int duplicate = ::dup(directory);
  if (duplicate < 0)
    reject("could not inspect the provisioned run directory");
  DIR *stream = ::fdopendir(duplicate);
  if (stream == nullptr) {
    (void)::close(duplicate);
    reject("could not inspect the provisioned run directory");
  }
  bool empty = true;
  errno = 0;
  while (const dirent *entry = ::readdir(stream)) {
    const std::string_view name(entry->d_name);
    if (name != "." && name != "..") {
      empty = false;
      break;
    }
  }
  const int read_error = errno;
  (void)::closedir(stream);
  if (read_error != 0)
    reject("could not enumerate the provisioned run directory");
  return empty;
}

void write_marker(int directory, std::string_view content) {
  Descriptor file(::openat(directory, kMarkerName.data(),
                           O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                           0640));
  if (file.get() < 0)
    reject("could not create the authority run-directory marker: " +
           std::string(std::strerror(errno)));
  std::size_t offset{};
  while (offset < content.size()) {
    const ssize_t count =
        ::write(file.get(), content.data() + offset, content.size() - offset);
    if (count <= 0)
      reject("could not write the authority run-directory marker");
    offset += static_cast<std::size_t>(count);
  }
  if (::fsync(file.get()) != 0 || ::fsync(directory) != 0)
    reject("could not durably publish the authority run-directory marker");
}

} // namespace

AuthorRunDocument decode_author_run_document(std::string_view source,
                                             std::string_view source_format) {
  const nlohmann::json document = parse_document(source, source_format);
  AuthorRunDocument result =
      decode_closed<AuthorRunDocument>(document, "author-run document");
  if (result.api_version != kAuthorRunApiVersion)
    reject("author-run api_version is unsupported");
  if (result.source.experiment.has_value() ==
      result.source.recipe.has_value())
    reject("author-run source must contain exactly one of experiment or recipe");
  if (result.source.experiment && !result.source.experiment->is_object())
    reject("author-run experiment source must be an object");
  if (result.source.recipe) {
    const std::filesystem::path registry(result.source.recipe->registry_path);
    if (!canonical_absolute(registry))
      reject("recipe registry_path must be canonical and absolute");
  }
  if (!bounded_text(result.author, 192U) ||
      !bounded_text(result.reason, 2'048U))
    reject("author-run author/reason is empty, unbounded, or contains control bytes");
  if (result.input_content)
    (void)normalized_root_set_paths(*result.input_content);
  return result;
}

AuthoringClientConfiguration load_authoring_client_configuration(
    const std::filesystem::path &path) {
  if (!canonical_absolute(path))
    reject("installed authoring-client path is not canonical and absolute");
  struct stat status {};
  if (::lstat(path.c_str(), &status) != 0 || !S_ISREG(status.st_mode) ||
      S_ISLNK(status.st_mode) || status.st_uid != 0U ||
      (status.st_mode & 0022U) != 0U || status.st_size <= 0 ||
      static_cast<std::uint64_t>(status.st_size) >
          kMaximumClientConfigurationBytes)
    reject("installed authoring-client configuration must be a bounded "
           "root-owned non-writable regular file");
  std::ifstream input(path);
  std::ostringstream contents;
  contents << input.rdbuf();
  if (!input && !input.eof())
    reject("installed authoring-client configuration could not be read");
  const auto document = nlohmann::json::parse(contents.str());
  auto result = decode_closed<AuthoringClientConfiguration>(
      document, "authoring-client configuration");
  const std::filesystem::path socket =
      result.controller_target.starts_with("unix:")
          ? std::filesystem::path(result.controller_target.substr(5U))
          : std::filesystem::path{};
  if (result.api_version != kAuthoringClientApiVersion ||
      !result.controller_target.starts_with("unix:/") ||
      !canonical_absolute(socket) || result.controller_target.size() > 4'096U ||
      result.dashboard_base_url.size() > 4'096U ||
      !(result.dashboard_base_url.starts_with("http://") ||
        result.dashboard_base_url.starts_with("https://")) ||
      result.dashboard_base_url.ends_with('/'))
    reject("installed authoring-client configuration contains an invalid "
           "authority target or dashboard base URL");
  return result;
}

ResolvedAuthorRun resolve_and_lock_author_run(
    const AuthorRunDocument &document,
    InputContentMeasurementCache *content_cache) {
  CompiledPlan plan;
  std::optional<TrainingPreflightRecipeProvenance> recipe_provenance;
  std::optional<nlohmann::json> base_recipe_expansion;
  std::vector<RecipeContentBinding> recipe_content_bindings;
  if (document.source.recipe) {
    const auto registry = RecipeProfileRegistry::load_file(
        document.source.recipe->registry_path);
    const auto &profile =
        registry.profile(document.source.recipe->instance.recipe);
    if (profile.content_bindings)
      recipe_content_bindings = *profile.content_bindings;
    ExpandedRecipe expanded = registry.expand(document.source.recipe->instance);
    recipe_provenance = training_preflight_recipe_provenance(expanded);
    base_recipe_expansion = expanded_recipe_json(expanded);
    plan = std::move(expanded.plan);
  } else {
    const CompileResult compiled = compile_document(*document.source.experiment);
    if (!compiled.valid() || !compiled.plan) {
      const std::string detail = compiled.diagnostics.empty()
                                     ? "unknown compiler failure"
                                     : compiled.diagnostics.front().path + " " +
                                           compiled.diagnostics.front().message;
      reject("author-run experiment validation failed: " + detail);
    }
    plan = *compiled.plan;
  }

  bool reused = false;
  std::vector<InputContentMeasurementStats> content_measurements;
  std::vector<InputContentRootIdentity> measured_identities;
  std::optional<InputContentMeasurementTransaction> cache_transaction;
  const auto measure = [&](const InputContentRootSet &roots) {
    require_authorized_content_paths(plan, roots);
    if (content_cache != nullptr && !cache_transaction)
      cache_transaction.emplace(content_cache->begin_transaction());
    return measure_input_content_root_set(
        roots, &content_measurements,
        cache_transaction ? &*cache_transaction : nullptr);
  };
  nlohmann::json locked_document = plan.canonical_plan;
  const std::vector<std::string> existing_paths = locked_paths(plan);
  nlohmann::json derived_content = nlohmann::json::array();
  if (document.source.recipe && document.input_content)
    reject("recipe author-runs derive input content from their authority "
           "profile and cannot carry duplicate input_content paths");
  if (!recipe_content_bindings.empty()) {
    InputContentRootSet roots{
        .api_version = std::string(kInputContentRootSetApiVersion), .paths = {}};
    for (const auto &binding : recipe_content_bindings) {
      const nlohmann::json::json_pointer path_pointer(binding.path_target);
      if (!locked_document.contains(path_pointer) ||
          !locked_document.at(path_pointer).is_string())
        reject("recipe content binding path vanished after expansion");
      roots.paths.push_back(
          locked_document.at(path_pointer).get<std::string>());
    }
    roots.paths = normalized_root_set_paths(roots);
    if (roots.paths.size() != recipe_content_bindings.size())
      reject("recipe content bindings resolve to ambiguous duplicate roots");
    measured_identities = measure(roots);
    locked_document["spec"]["workspace"]["input_content_roots"] =
        encode_json(measured_identities);
    for (const auto &binding : recipe_content_bindings) {
      const nlohmann::json::json_pointer path_pointer(binding.path_target);
      const std::string path =
          locked_document.at(path_pointer).get<std::string>();
      const auto identity = std::ranges::find(
          measured_identities, path, &InputContentRootIdentity::path);
      if (identity == measured_identities.end())
        reject("recipe content binding has no exact measured root identity");
      locked_document[nlohmann::json::json_pointer(
          binding.fingerprint_target)] = identity->tree_sha256;
      derived_content.push_back({
          {"path_target", binding.path_target},
          {"path", identity->path},
          {"fingerprint_target", binding.fingerprint_target},
          {"tree_sha256", identity->tree_sha256},
          {"provenance", "authority_measured"},
      });
    }
  } else if (document.input_content) {
    measured_identities = measure(*document.input_content);
    locked_document["spec"]["workspace"]["input_content_roots"] =
        encode_json(measured_identities);
  } else if (!existing_paths.empty()) {
    if (document.source.recipe) {
      InputContentRootSet roots{
          .api_version = std::string(kInputContentRootSetApiVersion),
          .paths = existing_paths,
      };
      measured_identities = measure(roots);
      const auto &declared =
          *plan.experiment.spec.workspace.input_content_roots;
      reused = measured_identities == declared;
      locked_document["spec"]["workspace"]["input_content_roots"] =
          encode_json(measured_identities);
    } else if (has_training_node(plan))
      reject("direct training author-runs must remeasure input_content; "
             "document-supplied workspace locks are not authority provenance");
    else
      reused = true;
  } else if (has_training_node(plan)) {
    reject("training author-run requires input_content paths or an exact "
           "existing trusted content lock");
  }

  const CompileResult locked = compile_document(locked_document);
  if (!locked.valid() || !locked.plan)
    reject("content locking produced an invalid compiled experiment");
  plan = *locked.plan;
  std::optional<InputContentMeasurementCacheCommitStats> cache_commit;
  if (cache_transaction)
    cache_commit = cache_transaction->commit();
  if (recipe_provenance)
    recipe_provenance->expanded_plan_digest = "sha256:" + plan.plan_hash;
  std::optional<nlohmann::json> recipe_expansion;
  if (base_recipe_expansion) {
    recipe_expansion = nlohmann::json{
        {"api_version", "trainvm.author-run-recipe-expansion/v1"},
        {"base_recipe_expansion", std::move(*base_recipe_expansion)},
        {"final_plan_hash", plan.plan_hash},
        {"final_plan_digest", "sha256:" + plan.plan_hash},
        {"final_canonical_plan", plan.canonical_plan},
        {"content_lock_reused", reused},
        {"derived_content_bindings", std::move(derived_content)},
    };
  }
  const std::string request_digest =
      "sha256:" + sha256_hex(nlohmann::json{
                                 {"api_version", kAuthorRunApiVersion},
                                 {"document", encode_json(document)},
                                 {"plan_hash", plan.plan_hash},
                             }
                                 .dump());
  std::optional<InputContentMeasurementReceipt> content_receipt;
  if (cache_commit) {
    if (measured_identities.size() != content_measurements.size())
      throw std::logic_error(
          "content measurement identities and telemetry diverged");
    std::vector<InputContentRootMeasurementReceipt> roots;
    roots.reserve(measured_identities.size());
    for (std::size_t index = 0U; index < measured_identities.size(); ++index) {
      const auto &identity = measured_identities[index];
      const auto &stats = content_measurements[index];
      roots.push_back({
          .path = identity.path,
          .tree_sha256 = identity.tree_sha256,
          .file_count = identity.file_count,
          .total_bytes = identity.total_bytes,
          .cache_hits = stats.cache_hits,
          .cache_misses = stats.cache_misses,
          .cache_bypasses = stats.cache_bypasses,
          .staging_saturations = stats.staging_saturations,
          .bytes_hashed = stats.bytes_hashed,
          .elapsed_nanoseconds = stats.elapsed_nanoseconds,
      });
    }
    content_receipt = InputContentMeasurementReceipt{
        .api_version = std::string(kInputContentMeasurementReceiptApiVersion),
        .cache_api_version =
            std::string(kInputContentMeasurementCacheApiVersion),
        .cache_policy_digest = content_cache->policy_digest(),
        .request_digest = request_digest,
        .plan_hash = plan.plan_hash,
        .roots = std::move(roots),
        .cache_commit = *cache_commit,
        .receipt_digest = {},
    };
    nlohmann::json principal = encode_json(*content_receipt);
    principal.erase("receipt_digest");
    content_receipt->receipt_digest = "sha256:" + sha256_hex(principal.dump());
  }
  return {.plan = std::move(plan),
          .recipe_provenance = std::move(recipe_provenance),
          .recipe_expansion = std::move(recipe_expansion),
          .request_digest = request_digest,
          .content_lock_reused = reused,
          .content_measurements = std::move(content_measurements),
          .content_measurement_receipt = std::move(content_receipt)};
}

PassiveHostSnapshotSource make_local_passive_host_snapshot_source(
    std::string host_id, std::string boot_id,
    std::function<std::uint64_t()> monotonic_now_ns,
    PassiveAcceleratorSnapshotSource accelerators) {
  if (host_id.empty() || boot_id.empty() || !monotonic_now_ns)
    throw std::invalid_argument(
        "local passive host snapshot requires host/boot identities and clock");
  return [host_id = std::move(host_id), boot_id = std::move(boot_id),
          monotonic_now_ns = std::move(monotonic_now_ns),
          accelerators = std::move(accelerators)](
             const CompiledPlan &plan) -> TrainingPreflightEnvironment {
    std::ifstream meminfo("/proc/meminfo");
    std::uint64_t total_kib{};
    std::uint64_t available_kib{};
    std::string key;
    std::uint64_t value{};
    std::string unit;
    while (meminfo >> key >> value >> unit) {
      if (key == "MemTotal:")
        total_kib = value;
      else if (key == "MemAvailable:")
        available_kib = value;
    }
    const long logical_cpus = ::sysconf(_SC_NPROCESSORS_ONLN);
    if (total_kib == 0U || available_kib > total_kib || logical_cpus <= 0 ||
        logical_cpus > static_cast<long>(
                           std::numeric_limits<std::uint32_t>::max()))
      throw std::runtime_error(
          "Linux passive CPU/memory snapshot is unavailable");
    std::vector<std::uint32_t> groups;
    const int group_count = ::getgroups(0, nullptr);
    if (group_count < 0 || group_count > 256)
      throw std::runtime_error(
          "effective supplementary group snapshot is unavailable or unbounded");
    std::vector<gid_t> native_groups(static_cast<std::size_t>(group_count));
    if (group_count > 0 &&
        ::getgroups(group_count, native_groups.data()) != group_count)
      throw std::runtime_error(
          "effective supplementary group snapshot changed during capture");
    const auto primary_gid = static_cast<std::uint32_t>(::getegid());
    for (const gid_t group : native_groups) {
      const auto converted = static_cast<std::uint32_t>(group);
      if (converted != primary_gid)
        groups.push_back(converted);
    }
    std::ranges::sort(groups);
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());

    std::vector<PassiveAcceleratorMemoryEvidence> observed_accelerators;
    if (accelerators &&
        plan.experiment.spec.resources.accelerators.count > 0)
      observed_accelerators = accelerators(plan);
    else if (plan.experiment.spec.resources.accelerators.count > 0)
      throw ImplementationRequired(
          "selected plan requires an authority-owned passive accelerator "
          "memory sampler");

    const std::uint64_t observed = monotonic_now_ns();
    constexpr std::uint64_t lifetime = 30'000'000'000ULL;
    if (observed > std::numeric_limits<std::uint64_t>::max() - lifetime)
      throw std::runtime_error("passive snapshot validity overflowed");
    const nlohmann::json principal{
        {"uid", static_cast<std::uint32_t>(::geteuid())},
        {"gid", primary_gid},
        {"supplementary_gids", groups},
    };
    const std::string principal_digest =
        "sha256:" + sha256_hex(principal.dump());
    const nlohmann::json snapshot{
        {"host_id", host_id},
        {"boot_id", boot_id},
        {"observed_monotonic_ns", observed},
        {"total_host_memory_bytes", total_kib * 1'024U},
        {"available_host_memory_bytes", available_kib * 1'024U},
        {"logical_cpu_count", logical_cpus},
        {"worker_principal_digest", principal_digest},
        {"accelerators", encode_json(observed_accelerators)},
    };
    return {
        .api_version = std::string(kTrainingPreflightEnvironmentApiVersion),
        .host_id = host_id,
        .boot_id = boot_id,
        .snapshot_digest = "sha256:" + sha256_hex(snapshot.dump()),
        .snapshot_observed_monotonic_ns = observed,
        .snapshot_valid_until_monotonic_ns = observed + lifetime,
        .evaluation_monotonic_ns = observed,
        .worker_uid = static_cast<std::uint32_t>(::geteuid()),
        .worker_gid = primary_gid,
        .supplementary_gids = std::move(groups),
        .worker_principal_digest = principal_digest,
        .total_host_memory_bytes = total_kib * 1'024U,
        .available_host_memory_bytes = available_kib * 1'024U,
        .logical_cpu_count = static_cast<std::uint32_t>(logical_cpus),
        .accelerators = std::move(observed_accelerators),
        .training_nodes = {},
        .gpu_qualification = std::nullopt,
        .recipe_provenance = std::nullopt,
    };
  };
}

TrainingNodeProbe make_sealed_structural_training_node_probe() {
  return [](const CompiledPlan &plan, std::string_view node_id,
            const AdapterProfile &profile) {
    const auto &node = plan.canonical_plan.at("spec")
                           .at("workflow")
                           .at("nodes")
                           .at(std::string(node_id));
    const auto &invoke = node.at("invoke");
    if (!invoke.contains("training") || !invoke.at("training").is_object())
      throw std::runtime_error(
          "sealed structural probe requires a closed training invocation");
    const auto &inputs = invoke.at("inputs");
    if (!inputs.contains("manifest") ||
        !inputs.at("manifest").contains("parameter"))
      throw std::runtime_error(
          "sealed structural probe requires a manifest parameter input");
    const std::string parameter =
        inputs.at("manifest").at("parameter").get<std::string>();
    const auto &parameter_value =
        plan.canonical_plan.at("spec").at("parameters").at(parameter);
    if (parameter_value.at("type") != "path")
      throw std::runtime_error("manifest parameter is not a path");
    const std::filesystem::path manifest(
        parameter_value.at("value").get<std::string>());
    const auto status = std::filesystem::symlink_status(manifest);
    if (!std::filesystem::is_regular_file(status) ||
        std::filesystem::is_symlink(status) ||
        std::filesystem::file_size(manifest) > (4ULL << 30U))
      throw std::runtime_error(
          "manifest must be a bounded regular non-symlink file");
    std::ifstream stream(manifest, std::ios::binary);
    std::string sample;
    while (std::getline(stream, sample) && sample.empty()) {
    }
    if (sample.empty() || sample.size() > (1U << 20U) ||
        !nlohmann::json::parse(sample).is_object())
      throw std::runtime_error(
          "manifest has no bounded decodable JSON object sample");
    const auto &components = invoke.at("training").at("components");
    if (!components.is_object() || components.empty())
      throw std::runtime_error(
          "training invocation has no closed parameter-selection components");
    bool has_step_zero = false;
    for (const auto &[name, binding] : inputs.items()) {
      if (!name.starts_with("step_zero") || !binding.contains("parameter"))
        continue;
      const auto &value = plan.canonical_plan.at("spec")
                              .at("parameters")
                              .at(binding.at("parameter").get<std::string>())
                              .at("value");
      has_step_zero = value.is_number_integer() && value.get<std::int64_t>() > 0;
    }
    if (!has_step_zero)
      throw std::runtime_error(
          "sealed structural probe requires a positive step-zero example input");
    if (!node.contains("publishes") || !node.at("publishes").is_object() ||
        node.at("publishes").empty())
      throw std::runtime_error(
          "sealed structural probe requires dashboard example publication");

    const std::string input_digest =
        training_preflight_node_input_digest(plan, node_id);
    const auto check = [&](TrainingPreflightCheckKind kind,
                           TrainingPreflightCheckDisposition disposition,
                           std::optional<std::string> detail = std::nullopt) {
      return TrainingPreflightCheckEvidence{
          .kind = kind,
          .disposition = disposition,
          .evidence_digest =
              "sha256:" +
              sha256_hex(nlohmann::json{{"input_digest", input_digest},
                                        {"kind", enum_to_string(kind)},
                                        {"disposition",
                                         enum_to_string(disposition)},
                                        {"detail", detail}}
                             .dump()),
          .detail = std::move(detail),
      };
    };
    std::vector<TrainingPreflightCheckEvidence> checks{
        check(TrainingPreflightCheckKind::model_configuration,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::tokenizer,
              TrainingPreflightCheckDisposition::not_applicable,
              "exact adapter opted into worker-owned tokenizer validation"),
        check(TrainingPreflightCheckKind::processor,
              TrainingPreflightCheckDisposition::not_applicable,
              "exact adapter opted into worker-owned processor validation"),
        check(TrainingPreflightCheckKind::dataset_schema,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::dataset_sample_decode,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::parameter_selection,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::kernel_runtime,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::checkpoint_compatibility,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::step_zero_evaluator,
              TrainingPreflightCheckDisposition::passed),
        check(TrainingPreflightCheckKind::dashboard_artifacts,
              TrainingPreflightCheckDisposition::passed),
    };
    return TrainingNodePreflightEvidence{
        .node_id = std::string(node_id),
        .node_input_digest = input_digest,
        .checks = std::move(checks),
        .minimum_free_memory_gib =
            plan.experiment.spec.resources.accelerators.count > 0
                ? std::optional<double>(
                      plan.experiment.spec.resources.accelerators
                          .minimum_memory_gib.value_or(0.0))
                : std::optional<double>(0.0),
        .runtime_profile_digest =
            "sha256:" + sha256_hex(encode_json(profile).dump()),
        .required_capabilities = profile.required_capabilities,
        .provided_capabilities = profile.required_capabilities,
    };
  };
}

bool hf_lora_selectors_match_parameter_index(
    const std::vector<std::string> &selectors,
    const std::vector<std::string> &parameter_keys) {
  if (selectors.empty() || parameter_keys.empty())
    return false;
  std::set<std::string> module_names;
  for (const std::string &parameter : parameter_keys) {
    for (const std::string_view suffix : {".weight", ".bias"}) {
      if (parameter.ends_with(suffix) && parameter.size() > suffix.size())
        module_names.insert(
            parameter.substr(0U, parameter.size() - suffix.size()));
    }
  }
  return !module_names.empty() &&
         std::ranges::all_of(selectors, [&](const std::string &selector) {
           return std::ranges::any_of(
               module_names, [&](const std::string &module) {
                 return wildcard_match(selector, module);
               });
         });
}

TrainingNodeProbe make_hf_multimodal_sft_training_node_probe(
    PassiveRuntimeProfileSource runtime_profile) {
  if (!runtime_profile)
    throw std::invalid_argument(
        "HF multimodal passive probe requires installed runtime evidence");
  return [runtime_profile = std::move(runtime_profile)](
             const CompiledPlan &plan, std::string_view node_id,
             const AdapterProfile &profile) {
    const AdapterKey expected{
        .adapter = "rwkv-lab.hf-multimodal-sft",
        .version = "1.0.0",
        .runtime = ComponentRuntime::python_worker,
        .operation = "train",
        .contract = "rwkv_lab.hf_multimodal_sft.v1.Train",
    };
    if (profile.key != expected)
      throw std::runtime_error(
          "HF multimodal passive probe received another AdapterKey");
    const PassiveRuntimeProfileEvidence runtime = runtime_profile(profile);
    if (!canonical_digest(runtime.profile_digest) ||
        !std::ranges::all_of(profile.required_capabilities,
                            [&](const auto &required) {
                              return std::ranges::find(
                                         runtime.provided_capabilities,
                                         required) !=
                                     runtime.provided_capabilities.end();
                            }))
      throw std::runtime_error(
          "installed HF worker runtime does not provide the exact adapter "
          "capability set");
    const auto &node = plan.canonical_plan.at("spec")
                           .at("workflow")
                           .at("nodes")
                           .at(std::string(node_id));
    const auto &invoke = node.at("invoke");
    const auto &components = invoke.at("training").at("components");
    const auto component = [&](std::string_view slot,
                               std::string_view expected_category,
                               std::string_view expected_name)
        -> const nlohmann::json & {
      const auto found = components.find(std::string(slot));
      if (found == components.end() || !found->is_object() ||
          !found->contains("key") || !found->at("key").is_object() ||
          found->at("key").value("category", "") != expected_category ||
          found->at("key").value("name", "") != expected_name ||
          found->at("key").value("version", "") != "1.0.0" ||
          !found->contains("configuration") ||
          !found->at("configuration").is_object())
        throw std::runtime_error("HF multimodal passive probe requires exact " +
                                 std::string(slot) + "=" +
                                 std::string(expected_name));
      return *found;
    };

    const auto &loader =
        component("model_loader", "model_loader", "hf_multimodal");
    const auto &loader_config = loader.at("configuration");
    const std::filesystem::path model_path(
        loader_config.at("model_path").get<std::string>());
    const auto &locked_roots = plan.canonical_plan.at("spec")
                                   .at("workspace")
                                   .at("input_content_roots");
    const auto require_locked_fingerprint =
        [&](const std::filesystem::path &path,
            const nlohmann::json &configuration,
            std::string_view field) {
          const nlohmann::json *found = nullptr;
          for (const nlohmann::json &root : locked_roots) {
            if (root.at("path").get<std::string>() == path.string()) {
              found = &root;
              break;
            }
          }
          if (found == nullptr ||
              configuration.at(std::string(field)).get<std::string>() !=
                  found->at("tree_sha256").get<std::string>())
            throw std::runtime_error(
                "HF content fingerprint is not authority-derived from its "
                "exact locked root");
        };
    if (!model_path.is_absolute() ||
        !loader_config.value("local_files_only", false) ||
        loader_config.value("trust_remote_code", true) ||
        !loader_config.value("exact_checkpoint", false))
      throw std::runtime_error(
          "HF model loader must be absolute, local-only, exact, and must not "
          "trust remote code");
    require_locked_fingerprint(model_path, loader_config,
                               "checkpoint_fingerprint");
    std::string model_config_digest;
    std::string tokenizer_config_digest;
    std::string processor_config_digest;
    const auto model_config = read_passive_json_object(
        model_path / "config.json", 16U * 1024U * 1024U,
        model_config_digest);
    const auto tokenizer_config = read_passive_json_object(
        model_path / "tokenizer_config.json", 16U * 1024U * 1024U,
        tokenizer_config_digest);
    (void)model_config;
    (void)tokenizer_config;
    std::filesystem::path processor_path = model_path / "processor_config.json";
    struct stat processor_status {};
    if (::lstat(processor_path.c_str(), &processor_status) != 0)
      processor_path = model_path / "preprocessor_config.json";
    const auto processor_config = read_passive_json_object(
        processor_path, 16U * 1024U * 1024U, processor_config_digest);
    (void)processor_config;
    std::optional<PassiveFile> tokenizer_payload;
    std::string tokenizer_payload_name;
    for (const std::string_view name : {"tokenizer.json", "tokenizer.model",
                                        "vocab.json"}) {
      const auto candidate = model_path / name;
      struct stat status {};
      if (::lstat(candidate.c_str(), &status) != 0)
        continue;
      tokenizer_payload =
          read_passive_file(candidate, 128U * 1024U * 1024U);
      tokenizer_payload_name = name;
      break;
    }
    if (!tokenizer_payload)
      throw std::runtime_error(
          "HF local model has no bounded tokenizer payload identity");

    const auto &data =
        component("data", "data_source", "jsonl_image_caption");
    const auto &data_config = data.at("configuration");
    const std::filesystem::path manifest(
        data_config.at("manifest_path").get<std::string>());
    if (invoke.contains("inputs") && invoke.at("inputs").contains("manifest")) {
      const auto &binding = invoke.at("inputs").at("manifest");
      if (!binding.is_object() || !binding.contains("parameter") ||
          !binding.at("parameter").is_string())
        throw std::runtime_error(
            "HF manifest invocation binding is not a closed parameter");
      const auto &parameters = plan.canonical_plan.at("spec").at("parameters");
      const std::string parameter =
          binding.at("parameter").get<std::string>();
      if (!parameters.contains(parameter) ||
          parameters.at(parameter).at("value").get<std::string>() !=
              manifest.string())
        throw std::runtime_error(
            "HF invocation parameter and data component name different "
            "manifests");
    }
    const std::filesystem::path image_root(
        data_config.at("image_root").get<std::string>());
    if (!path_within_root(manifest, image_root))
      throw std::runtime_error(
          "HF manifest is outside its authority-bound image root");
    require_locked_fingerprint(image_root, data_config,
                               "content_fingerprint");
    std::ifstream manifest_stream(manifest, std::ios::binary);
    std::string sample_line;
    while (std::getline(manifest_stream, sample_line) && sample_line.empty()) {
    }
    if (sample_line.empty() || sample_line.size() > (1U << 20U))
      throw std::runtime_error("HF manifest has no bounded JSONL sample");
    const auto sample = nlohmann::json::parse(sample_line);
    const std::string image_column =
        data_config.at("image_column").get<std::string>();
    const auto caption_columns =
        data_config.at("caption_columns").get<std::vector<std::string>>();
    if (!sample.is_object() || image_column.empty() ||
        !sample.contains(image_column) ||
        !sample.at(image_column).is_string() ||
        sample.at(image_column).get<std::string>().empty() ||
        caption_columns.empty() ||
        !std::ranges::all_of(caption_columns, [&](const std::string &column) {
          return sample.contains(column) && sample.at(column).is_string() &&
                 !sample.at(column).get<std::string>().empty();
        }))
      throw std::runtime_error(
          "HF manifest sample does not satisfy its image/caption schema");
    std::filesystem::path sample_image(
        sample.at(image_column).get<std::string>());
    if (!sample_image.is_absolute())
      sample_image = image_root / sample_image;
    sample_image = sample_image.lexically_normal();
    if (!path_within_root(sample_image, image_root))
      throw std::runtime_error("HF manifest sample escapes its image root");
    const PassiveImageHeader image_header =
        inspect_passive_image_header(sample_image);

    const auto &processor =
        component("processor", "sample_processor", "image_caption");
    if (processor.at("configuration").at("image_column") != image_column ||
        processor.at("configuration").at("caption_columns") !=
            caption_columns)
      throw std::runtime_error(
          "HF processor schema does not match the data component");
    (void)component("sample_mapping", "sample_mapper", "assistant_only");

    const auto &trainability =
        component("trainability", "trainability", "lora");
    const auto &trainability_config = trainability.at("configuration");
    const auto target_selectors =
        trainability_config.at("target_selectors")
            .get<std::vector<std::string>>();
    if (trainability_config.at("rank").get<std::int64_t>() <= 0 ||
        target_selectors.empty() || target_selectors.size() > 256U ||
        !std::ranges::all_of(target_selectors, [](const auto &selector) {
          return !selector.empty() && selector.size() <= 512U;
        }))
      throw std::runtime_error(
          "HF LoRA trainability target selection is empty or unbounded");
    std::string tensor_index_digest;
    const auto tensor_index = read_passive_json_object(
        model_path / "model.safetensors.index.json",
        64U * 1024U * 1024U, tensor_index_digest);
    std::vector<std::string> parameter_keys;
    if (tensor_index.contains("weight_map") &&
        tensor_index.at("weight_map").is_object()) {
      for (const auto &entry : tensor_index.at("weight_map").items())
        parameter_keys.push_back(entry.key());
    }
    if (!tensor_index.contains("weight_map") ||
        !tensor_index.at("weight_map").is_object() ||
        tensor_index.at("weight_map").empty() ||
        !hf_lora_selectors_match_parameter_index(target_selectors,
                                                  parameter_keys))
      throw std::runtime_error(
          "HF LoRA selector does not match the passive safetensors key index");

    const auto &schedule =
        component("evaluation_schedule", "evaluation_schedule",
                  "launch_gate_periodic");
    if (!schedule.at("configuration").value("full_step_zero", false) ||
        schedule.at("configuration").value("launch_gate_examples", 0) <= 0)
      throw std::runtime_error(
          "HF evaluation schedule does not require full step-zero evidence");
    (void)component("qualitative_samples", "qualitative_sample",
                    "fixed_held_out");
    const auto &renderer =
        component("artifact_renderer", "artifact_renderer",
                  "caption_triplet");
    if (renderer.at("configuration").value("schema", "") !=
        "trainvm.caption-triplet.v1")
      throw std::runtime_error(
          "HF step-zero renderer does not publish caption triplets");
    const auto &checkpoint =
        component("checkpoint_policy", "checkpoint_policy",
                  "atomic_retained");
    if (trainability_config.value("merge_on_completion", true) ||
        !checkpoint.at("configuration").value("publish_final", false) ||
        checkpoint.at("configuration").value("resume_grade", "") != "exact" ||
        profile.lifecycle.resume_grade != ResumeGrade::exact)
      throw std::runtime_error(
          "HF checkpoint policy is incompatible with exact unmerged LoRA "
          "resume");
    if (!node.at("publishes").contains("eval_gallery") ||
        node.at("publishes").at("eval_gallery") != "eval_gallery" ||
        !node.at("publishes").contains("checkpoint") ||
        node.at("publishes").at("checkpoint") != "checkpoint")
      throw std::runtime_error(
          "HF training node omits checkpoint or eval-gallery publication");
    const auto &artifacts = plan.canonical_plan.at("spec").at("artifacts");
    if (artifacts.at("checkpoint").at("schema") !=
            "hf.multimodal-sft.v1" ||
        artifacts.at("eval_gallery").at("schema") !=
            "rwkv-lab.eval-gallery.v2" ||
        !artifacts.at("eval_gallery").value("required", false))
      throw std::runtime_error(
          "HF output artifact schemas are not the admitted exact pair");

    const std::string input_digest =
        training_preflight_node_input_digest(plan, node_id);
    const nlohmann::json exact_identity{
        {"node_input_digest", input_digest},
        {"model_config_digest", model_config_digest},
        {"tokenizer_config_digest", tokenizer_config_digest},
        {"tokenizer_payload_name", tokenizer_payload_name},
        {"tokenizer_payload_digest", tokenizer_payload->digest},
        {"processor_config_digest", processor_config_digest},
        {"manifest_sample_digest", "sha256:" + sha256_hex(sample_line)},
        {"sample_image",
         {{"path", sample_image.string()},
          {"format", image_header.format},
          {"width", image_header.width},
          {"height", image_header.height}}},
        {"trainability", trainability_config},
        {"tensor_index_digest", tensor_index_digest},
        {"evaluation_schedule", schedule.at("configuration")},
        {"renderer", renderer.at("configuration")},
        {"checkpoint", checkpoint.at("configuration")},
        {"runtime_profile_digest", runtime.profile_digest},
        {"runtime_capabilities", runtime.provided_capabilities},
    };
    std::vector<TrainingPreflightCheckEvidence> checks;
    for (const auto kind : {
             TrainingPreflightCheckKind::model_configuration,
             TrainingPreflightCheckKind::tokenizer,
             TrainingPreflightCheckKind::processor,
             TrainingPreflightCheckKind::dataset_schema,
             TrainingPreflightCheckKind::dataset_sample_decode,
             TrainingPreflightCheckKind::parameter_selection,
             TrainingPreflightCheckKind::kernel_runtime,
             TrainingPreflightCheckKind::checkpoint_compatibility,
             TrainingPreflightCheckKind::step_zero_evaluator,
             TrainingPreflightCheckKind::dashboard_artifacts,
         }) {
      checks.push_back({
          .kind = kind,
          .disposition = TrainingPreflightCheckDisposition::passed,
          .evidence_digest =
              "sha256:" + sha256_hex(nlohmann::json{
                  {"identity", exact_identity},
                  {"check", enum_to_string(kind)},
              }.dump()),
          .detail =
              "exact rwkv-lab.hf-multimodal-sft@1.0.0 passive check",
      });
    }
    return TrainingNodePreflightEvidence{
        .node_id = std::string(node_id),
        .node_input_digest = input_digest,
        .checks = std::move(checks),
        .minimum_free_memory_gib =
            plan.experiment.spec.resources.accelerators.count > 0
                ? std::optional<double>(
                      plan.experiment.spec.resources.accelerators
                          .minimum_memory_gib.value_or(0.0))
                : std::optional<double>(0.0),
        .runtime_profile_digest = runtime.profile_digest,
        .required_capabilities = profile.required_capabilities,
        .provided_capabilities = runtime.provided_capabilities,
    };
  };
}

RegisteredTrainingPreflightEvidenceProvider::
    RegisteredTrainingPreflightEvidenceProvider(
        const AdapterRegistry &adapters,
        PassiveHostSnapshotSource host_snapshot,
        std::vector<RegisteredTrainingNodeProbe> probes,
        TrainingNodeProbe sealed_structural_probe)
    : adapters_(adapters), host_snapshot_(std::move(host_snapshot)),
      sealed_structural_probe_(std::move(sealed_structural_probe)) {
  for (auto &registered : probes) {
    if (!registered.probe ||
        !probes_.emplace(std::move(registered.key),
                         std::move(registered.probe))
             .second)
      throw std::invalid_argument(
          "training preflight probes must have unique exact AdapterKeys");
  }
}

TrainingPreflightEvidenceResult
RegisteredTrainingPreflightEvidenceProvider::collect(
    const CompiledPlan &plan,
    const std::optional<TrainingPreflightRecipeProvenance> &recipe) {
  if (!host_snapshot_) {
    return {.environment = std::nullopt,
            .diagnostics = {diagnostic(
                "new_implementation_required", "/host",
                "no bounded passive host snapshot source is registered",
                "Install an authority-owned passive host sampler before "
                "using canonical author-run submission.")}};
  }
  struct SelectedProbe final {
    std::string node_id;
    const AdapterProfile *profile{};
    const TrainingNodeProbe *probe{};
  };
  std::vector<SelectedProbe> selected;
  for (const auto &[node_id, node] :
       plan.experiment.spec.workflow.nodes) {
    if (!node.invoke.training)
      continue;
    const auto component =
        plan.experiment.spec.components.find(node.invoke.component);
    if (component == plan.experiment.spec.components.end())
      return {.environment = std::nullopt,
              .diagnostics = {diagnostic(
                  "preflight.component_missing",
                  "/spec/workflow/nodes/" + node_id + "/invoke/component",
                  "training node references an unavailable component",
                  "Correct and validate the experiment before preflight.")}};
    const AdapterProfile *profile{};
    try {
      profile = &adapters_.resolve(component->second, node.invoke.operation);
    } catch (const std::exception &error) {
      return {.environment = std::nullopt,
              .diagnostics = {diagnostic(
                  "preflight.adapter_unresolved",
                  "/spec/workflow/nodes/" + node_id, error.what(),
                  "Install the exact adapter profile selected by the recipe.")}};
    }
    const auto probe = probes_.find(profile->key);
    const bool structural_opt_in =
        sealed_structural_probe_ &&
        std::ranges::find(profile->required_capabilities,
                          kSealedStructuralPreflightCapability) !=
            profile->required_capabilities.end();
    if (probe == probes_.end() && !structural_opt_in) {
      return {.environment = std::nullopt,
              .diagnostics = {diagnostic(
                  "new_implementation_required",
                  "/spec/workflow/nodes/" + node_id,
                  "selected adapter has no registered stateless passive "
                  "preflight probe: " +
                      profile->key.adapter + "@" + profile->key.version,
                  "Implement and register the exact AdapterKey probe; native "
                  "model-family inference is forbidden.")}};
    }
    selected.push_back({.node_id = node_id,
                        .profile = profile,
                        .probe = probe == probes_.end()
                                     ? &sealed_structural_probe_
                                     : &probe->second});
  }
  TrainingPreflightEnvironment environment;
  try {
    environment = host_snapshot_(plan);
  } catch (const ImplementationRequired &error) {
    return {.environment = std::nullopt,
            .diagnostics = {diagnostic(
                "new_implementation_required", "/host/accelerators",
                error.what(),
                "Register the bounded passive host sampler; no run was "
                "created.")}};
  } catch (const std::exception &error) {
    return {.environment = std::nullopt,
            .diagnostics = {diagnostic(
                "preflight.host_probe_failed", "/host", error.what(),
                "Repair the passive authority sampler; no run was created.")}};
  }
  environment.recipe_provenance = recipe;
  environment.training_nodes.clear();
  for (const auto &selected_probe : selected) {
    try {
      TrainingNodePreflightEvidence evidence =
          (*selected_probe.probe)(plan, selected_probe.node_id,
                                  *selected_probe.profile);
      if (evidence.node_id != selected_probe.node_id ||
          evidence.node_input_digest !=
              training_preflight_node_input_digest(plan,
                                                   selected_probe.node_id)) {
        return {.environment = std::nullopt,
                .diagnostics = {diagnostic(
                    "preflight.probe_identity",
                    "/spec/workflow/nodes/" + selected_probe.node_id,
                    "registered probe did not bind the exact node input digest",
                    "Fix the AdapterKey probe; the authority will not rewrite "
                    "or infer its evidence.")}};
      }
      environment.training_nodes.push_back(std::move(evidence));
    } catch (const std::exception &error) {
      return {.environment = std::nullopt,
              .diagnostics = {diagnostic(
                  "preflight.family_probe_failed",
                  "/spec/workflow/nodes/" + selected_probe.node_id,
                  error.what(),
                  "Correct the model/data/runtime issue reported by the exact "
                  "registered family probe.")}};
    }
  }
  return {.environment = std::move(environment), .diagnostics = {}};
}

AuthorizedRunDirectoryProvision::AuthorizedRunDirectoryProvision(
    std::filesystem::path path, std::string marker, bool created)
    : path_(std::move(path)), marker_(std::move(marker)), created_(created) {}

AuthorizedRunDirectoryProvision::~AuthorizedRunDirectoryProvision() {
  if (!created_ || durable_ || path_.empty())
    return;
  const std::filesystem::path parent = path_.parent_path();
  Descriptor parent_directory(open_directory_no_symlinks(parent));
  if (parent_directory.get() < 0)
    return;
  Descriptor directory(::openat(parent_directory.get(), path_.filename().c_str(),
                                O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                    O_NOFOLLOW));
  if (directory.get() < 0)
    return;
  try {
    if (read_marker(directory.get()) != marker_)
      return;
  } catch (...) {
    return;
  }
  // The parent is expected to be authority-controlled. Still close the
  // practical substitution window as far as pathname APIs allow: the exact
  // name must resolve to the already-pinned directory immediately before any
  // unlink. A hostile same-UID writer in the parent remains outside this
  // cooperative rollback boundary and is why production parents are sealed.
  struct stat pinned {};
  struct stat named {};
  if (::fstat(directory.get(), &pinned) != 0 ||
      ::fstatat(parent_directory.get(), path_.filename().c_str(), &named,
                AT_SYMLINK_NOFOLLOW) != 0 ||
      !S_ISDIR(named.st_mode) || pinned.st_dev != named.st_dev ||
      pinned.st_ino != named.st_ino)
    return;
  (void)::unlinkat(directory.get(), kMarkerName.data(), 0);
  struct stat still_named {};
  if (::fstatat(parent_directory.get(), path_.filename().c_str(), &still_named,
                AT_SYMLINK_NOFOLLOW) != 0 ||
      !S_ISDIR(still_named.st_mode) || pinned.st_dev != still_named.st_dev ||
      pinned.st_ino != still_named.st_ino)
    return;
  (void)::unlinkat(parent_directory.get(), path_.filename().c_str(),
                   AT_REMOVEDIR);
}

AuthorizedRunDirectoryProvision::AuthorizedRunDirectoryProvision(
    AuthorizedRunDirectoryProvision &&other) noexcept
    : path_(std::move(other.path_)), marker_(std::move(other.marker_)),
      created_(std::exchange(other.created_, false)),
      durable_(std::exchange(other.durable_, true)) {}

AuthorizedRunDirectoryProvision &
AuthorizedRunDirectoryProvision::operator=(
    AuthorizedRunDirectoryProvision &&other) noexcept {
  if (this != &other) {
    this->~AuthorizedRunDirectoryProvision();
    new (this) AuthorizedRunDirectoryProvision(std::move(other));
  }
  return *this;
}

void AuthorizedRunDirectoryProvision::mark_durable() noexcept {
  durable_ = true;
}

bool AuthorizedRunDirectoryProvision::created() const noexcept {
  return created_;
}

AuthorizedRunDirectoryProvision provision_authorized_run_directory(
    const CompiledPlan &plan,
    const TrainingPreflightEnvironment &environment,
    std::string_view request_digest,
    std::function<void()> before_marker_test_seam) {
  if (!canonical_digest(request_digest))
    reject("run-directory provisioning requires the exact author-run digest");
  const std::filesystem::path run(
      plan.experiment.spec.workspace.run_directory);
  if (!canonical_absolute(run) || run.filename().empty() ||
      run.filename() == "." || run.filename() == "..")
    reject("compiled run directory is not a canonical provisionable path");
  const std::filesystem::path parent = run.parent_path();
  Descriptor parent_directory(open_directory_no_symlinks(parent));
  if (parent_directory.get() < 0)
    reject("authority could not securely open the run-directory parent: " +
           std::string(std::strerror(errno)));
  bool created = false;
  if (::mkdirat(parent_directory.get(), run.filename().c_str(), 0770) == 0) {
    created = true;
  } else if (errno != EEXIST) {
    reject("authority could not provision the run directory: " +
           std::string(std::strerror(errno)));
  }
  Descriptor directory(::openat(parent_directory.get(), run.filename().c_str(),
                                O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (directory.get() < 0)
    reject("authority could not pin the provisioned run directory");
  const std::string marker =
      nlohmann::json{{"api_version", "trainvm.author-run-directory/v1"},
                     {"plan_hash", plan.plan_hash},
                     {"request_digest", request_digest},
                     {"worker_principal_digest",
                      environment.worker_principal_digest}}
          .dump();
  try {
    if (created && before_marker_test_seam)
      before_marker_test_seam();
    if (created) {
      if (::fchown(directory.get(), static_cast<uid_t>(environment.worker_uid),
                   static_cast<gid_t>(environment.worker_gid)) != 0 ||
          ::fchmod(directory.get(), 0770) != 0) {
        reject("authority could not install the attested worker ownership/mode "
               "on the run directory");
      }
    }
    struct stat status {};
    if (::fstat(directory.get(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        status.st_uid != static_cast<uid_t>(environment.worker_uid) ||
        status.st_gid != static_cast<gid_t>(environment.worker_gid) ||
        (status.st_mode & 07777U) != 0770U)
      reject("existing run directory conflicts with the attested worker "
             "principal or required mode 0770");

    const std::string existing = read_marker(directory.get());
    if (existing.empty()) {
      if (!directory_empty(directory.get()))
        reject("existing run directory is nonempty and has no matching "
               "authority marker");
      write_marker(directory.get(), marker);
    } else if (existing != marker) {
      reject("existing run directory belongs to a different author-run "
             "identity");
    }
  } catch (...) {
    if (created) {
      // This call owns the newly-created name. Remove only its marker and the
      // now-empty exact directory through the already-pinned descriptors;
      // never clean a pre-existing retry target.
      struct stat pinned {};
      struct stat named {};
      if (::fstat(directory.get(), &pinned) == 0 &&
          ::fstatat(parent_directory.get(), run.filename().c_str(), &named,
                    AT_SYMLINK_NOFOLLOW) == 0 &&
          S_ISDIR(named.st_mode) && pinned.st_dev == named.st_dev &&
          pinned.st_ino == named.st_ino) {
        (void)::unlinkat(directory.get(), kMarkerName.data(), 0);
        struct stat still_named {};
        if (::fstatat(parent_directory.get(), run.filename().c_str(),
                      &still_named, AT_SYMLINK_NOFOLLOW) == 0 &&
            S_ISDIR(still_named.st_mode) &&
            pinned.st_dev == still_named.st_dev &&
            pinned.st_ino == still_named.st_ino)
          (void)::unlinkat(parent_directory.get(), run.filename().c_str(),
                           AT_REMOVEDIR);
      }
    }
    throw;
  }
  return AuthorizedRunDirectoryProvision(run, marker, created);
}

std::string author_run_stage_name(AuthorRunStage stage) {
  switch (stage) {
  case AuthorRunStage::validating:
    return "validating";
  case AuthorRunStage::resolving:
    return "resolving";
  case AuthorRunStage::locking_inputs:
    return "locking_inputs";
  case AuthorRunStage::preflight:
    return "preflight";
  case AuthorRunStage::provisioning:
    return "provisioning";
  case AuthorRunStage::submitting:
    return "submitting";
  case AuthorRunStage::complete:
    return "complete";
  case AuthorRunStage::failed:
    return "failed";
  }
  return "failed";
}

} // namespace trainvm
