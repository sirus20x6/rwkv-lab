#include "trainvm/compatibility_catalog.hpp"

#include <fcntl.h>
#include <linux/openat2.h>
#include <openssl/evp.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumCatalogBytes = 1U << 20U;
constexpr std::size_t kMaximumSourceBytes = 64U << 20U;
constexpr std::size_t kMaximumEntries = 512U;
constexpr std::size_t kMaximumSourcePaths = 32U;
constexpr std::size_t kMaximumIdentifierBytes = 96U;
constexpr std::size_t kMaximumPathBytes = 512U;
constexpr std::size_t kMaximumLegacyDisplayBytes = 1024U;
constexpr std::size_t kMaximumNotesBytes = 2048U;
constexpr std::string_view kReviewedCatalogDigest =
    "sha256:2182f180dac210b83c773e69d8f1d4fc754aa66b342cdeb8437a8be9d797eb15";

constexpr std::array<std::string_view, 140> kReviewedWorkflowIds = {
    "acquisition.civitai-anima",
    "acquisition.civitai-balanced",
    "acquisition.i1-direct-archives",
    "acquisition.kimi-teacher",
    "cache.vision-features",
    "cache.vision-finalization",
    "cache.vision-fusion-features",
    "cache.vision-overlay-assembly",
    "cache.vision-radio1d",
    "cache.vision-sam-dense",
    "cache.vision-ten-percent-handoff",
    "cache.vision-v4h",
    "control.experiment-launch",
    "control.gpu-launch-queue",
    "control.manual-training-launch",
    "control.posttraining-launch",
    "control.qualification-launch",
    "control.rlvr-launch",
    "control.sample-launch",
    "conversion.assemble-looped",
    "conversion.attention-l3-poc",
    "conversion.drive-isolation",
    "conversion.engram-patch",
    "conversion.gate-ab-campaign",
    "conversion.gate-ab2-campaign",
    "conversion.gdn-sweep-supervisor",
    "conversion.memory-target-cache",
    "conversion.per-layer-rwkv-train",
    "conversion.rel-sweep-supervisor",
    "conversion.stack-consolidation",
    "conversion.v4h-dino-compactor",
    "data.ao3-pack",
    "data.ao3-rewrite-eos",
    "data.ao3-source-preparation",
    "data.ao3-tokenize",
    "data.engram-allocation",
    "data.engram-frequency",
    "data.gelbooru-trainer-snapshot",
    "data.mageflow-manifest-preparation",
    "data.midjourney-v6-caption-routing",
    "data.midjourney-v6-continuation",
    "data.midjourney-v6-expert-stage",
    "data.posttraining-validation",
    "data.qwen35-conversion-corpus",
    "data.reddit-trainer-snapshot",
    "data.rwkv-corpus-preparation",
    "dedup.materialize-exact-links",
    "evaluation.conversion-baseline",
    "evaluation.loop-probe",
    "evaluation.radio1d-batch-caption",
    "evaluation.rwkv-generation",
    "evaluation.vision-caption",
    "export.frozen-vision-compressor",
    "export.legacy-mutable-bundle",
    "export.megakernel-aot",
    "export.production-kernels-aot",
    "external.ltx23-lora",
    "external.ltx23-plan",
    "external.ltx23-prepare",
    "external.ltx23-run",
    "inventory.unlabeled-images",
    "mageflow.adaptation-benchmark-spec",
    "mageflow.adaptation-domain-audit",
    "mageflow.adaptation-domain-prepare",
    "mageflow.cache-finish-resume",
    "mageflow.cache-resume-supervisor",
    "mageflow.expert-encoder-cache",
    "mageflow.expert-plan",
    "mageflow.full-backbone-plan",
    "mageflow.full-backbone-pretrain",
    "mageflow.high-resolution-design",
    "mageflow.routed-expert-train",
    "mageflow.terminal-cache-span-plan",
    "mageflow.terminal-cache-span-prepare",
    "mageflow.terminal-domain-window-plan",
    "mageflow.terminal-encoder-cache",
    "mageflow.terminal-expert-migration",
    "mageflow.terminal-preparation",
    "mageflow.terminal-tread-train",
    "mageflow.tread-loop-conversion",
    "oracle.adapter-consolidation",
    "oracle.decoding-evaluation",
    "oracle.diffusion-rwkv-head",
    "oracle.distillation-merge",
    "oracle.engram-lmb",
    "oracle.looped-rwkv-integrated",
    "oracle.lossless-gdn-map",
    "oracle.mla-training-components",
    "oracle.reasoning-cache",
    "oracle.recurrent-serving",
    "oracle.rosa-retrieval",
    "oracle.rosa-sam",
    "oracle.rosa-soft-layer",
    "oracle.smt-dmt-losses",
    "oracle.test-time-training",
    "posttraining.adapter-recursive",
    "posttraining.rwkv-adapter",
    "posttraining.rwkv-campaign",
    "profiling.mageflow-runtime",
    "profiling.qwen-prompt-hints",
    "profiling.vision-loop-telemetry",
    "qualification.converted-forward",
    "qualification.engram-integration",
    "qualification.lossless-gdn-map",
    "qualification.native-g1g",
    "qualification.posttraining-kernels",
    "qualification.production-kernels",
    "qualification.vision-run-evidence",
    "review.dedupe-cutoff",
    "review.i1-quality-viewer",
    "rlvr.recursive-improve",
    "rlvr.rwkv-campaign",
    "rlvr.rwkv-train",
    "rwkv.config-campaign",
    "rwkv.legacy-sweep-campaigns",
    "rwkv.optimizer-finetune",
    "rwkv.scratch-pretrain",
    "rwkv.synthetic-campaign",
    "scoring.i1-anime-aesthetic",
    "scoring.i1-deepghs-classification",
    "transformer.engram-prefill",
    "transformer.engram-staged-supervisor",
    "transformer.mla-engram-train",
    "transformer.mla-train",
    "transformer.qwen-ao3-audit",
    "transformer.qwen-ao3-cpt",
    "transformer.qwen-ao3-plan",
    "vision.continuation-watchdog",
    "vision.frozen-adapter-train",
    "vision.moonvit-continuation-launch-profiles",
    "vision.native-head-launch-profile",
    "vision.native-head-train",
    "vision.radio1d-launch-profiles",
    "vision.raw-pixel-student",
    "vision.raw-pixel-student-launch-profile",
    "vision.representation-ab",
    "vision.teacher-compressor",
    "vision.teacher-compressor-launch-profile",
    "vision.teacher-student-supervisor",
    "vision.v4h-launch-profiles",
};

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor& operator=(FileDescriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0) (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  [[nodiscard]] int get() const { return value_; }

 private:
  int value_;
};

struct EvpContextDeleter {
  void operator()(EVP_MD_CTX* context) const { EVP_MD_CTX_free(context); }
};

using EvpContext = std::unique_ptr<EVP_MD_CTX, EvpContextDeleter>;

std::string system_error_message(std::string_view operation) {
  return std::string(operation) + ": " + std::strerror(errno);
}

bool bounded_nonempty(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum;
}

bool bounded_text(std::string_view value, std::size_t maximum) {
  return bounded_nonempty(value, maximum) &&
         std::ranges::any_of(value, [](char character) {
           return !std::isspace(static_cast<unsigned char>(character));
         }) &&
         std::ranges::none_of(value, [](char character) {
           return std::iscntrl(static_cast<unsigned char>(character));
         });
}

bool stable_identifier(std::string_view value) {
  if (!bounded_nonempty(value, kMaximumIdentifierBytes) ||
      !std::isalnum(static_cast<unsigned char>(value.front()))) {
    return false;
  }
  return std::ranges::all_of(value, [](char character) {
    const auto byte = static_cast<unsigned char>(character);
    return std::islower(byte) || std::isdigit(byte) || character == '.' ||
           character == '_' || character == '-';
  });
}

bool stable_stat(const struct stat& before, const struct stat& after) {
  return before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
         before.st_mode == after.st_mode && before.st_size == after.st_size &&
         before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
         before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
         before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
         before.st_ctim.tv_nsec == after.st_ctim.tv_nsec;
}

std::string hex_digest(const unsigned char* bytes, unsigned int length) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(static_cast<std::size_t>(length) * 2U);
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(digits[bytes[index] >> 4U]);
    result.push_back(digits[bytes[index] & 0x0fU]);
  }
  return result;
}

std::string sha256_bytes(std::string_view bytes) {
  EvpContext context(EVP_MD_CTX_new());
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("could not initialize SHA-256 digest");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) {
    throw std::runtime_error("could not finalize SHA-256 digest");
  }
  return hex_digest(digest.data(), length);
}

std::string read_regular_file_stably(int descriptor, std::size_t maximum,
                                     std::string_view description) {
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > maximum) {
    throw std::invalid_argument(std::string(description) +
                                " must be a bounded nonempty regular file");
  }
  std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = ::read(descriptor, bytes.data() + offset,
                              bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw std::invalid_argument(std::string(description) +
                                  " changed or failed while being read");
    }
    offset += static_cast<std::size_t>(count);
  }
  char extra = '\0';
  const auto extra_count = ::read(descriptor, &extra, 1U);
  struct stat after {};
  if (extra_count != 0 || ::fstat(descriptor, &after) != 0 ||
      !stable_stat(before, after)) {
    throw std::invalid_argument(std::string(description) +
                                " changed while being read");
  }
  return bytes;
}

std::string read_catalog(const std::filesystem::path& path) {
  if (path.empty() || !path.is_absolute()) {
    throw std::invalid_argument(
        "compatibility catalog path must be absolute and nonempty");
  }
  FileDescriptor descriptor(
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  if (descriptor.get() < 0) {
    throw std::invalid_argument(system_error_message(
        "could not securely open compatibility catalog"));
  }
  return read_regular_file_stably(descriptor.get(), kMaximumCatalogBytes,
                                  "compatibility catalog");
}

nlohmann::json parse_catalog(std::string_view text) {
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    const nlohmann::json::parser_callback_t reject_duplicates =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (object_keys.size() <= index + 1U) {
              object_keys.resize(index + 2U);
            }
            object_keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (object_keys.size() <= index) object_keys.resize(index + 1U);
            if (!object_keys[index].insert(parsed.get<std::string>()).second) {
              duplicate_key = true;
            }
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     object_keys.size() > index + 1U) {
            object_keys[index + 1U].clear();
          }
          return true;
        };
    auto source = nlohmann::json::parse(text, reject_duplicates);
    if (duplicate_key) {
      throw std::invalid_argument(
          "compatibility catalog JSON contains a duplicate object key");
    }
    return source;
  } catch (const nlohmann::json::exception& exception) {
    throw std::invalid_argument("compatibility catalog is not valid JSON: " +
                                std::string(exception.what()));
  }
}

FileDescriptor open_repository_root(const std::filesystem::path& root) {
  if (root.empty() || !root.is_absolute()) {
    throw std::invalid_argument(
        "compatibility repository root must be absolute and nonempty");
  }
  FileDescriptor descriptor(
      ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw std::invalid_argument(system_error_message(
        "could not securely open compatibility repository root"));
  }
  struct stat status {};
  if (::fstat(descriptor.get(), &status) != 0 || !S_ISDIR(status.st_mode)) {
    throw std::invalid_argument(
        "compatibility repository root must be a directory");
  }
  return descriptor;
}

void validate_source_path_spelling(const std::string& value) {
  if (!bounded_nonempty(value, kMaximumPathBytes)) {
    throw std::invalid_argument(
        "compatibility source paths must be bounded and nonempty");
  }
  const std::filesystem::path relative(value);
  if (relative.is_absolute() || relative.has_root_path() ||
      relative.lexically_normal() != relative ||
      std::ranges::any_of(relative, [](const auto& component) {
        return component == "." || component == ".." || component.empty();
      })) {
    throw std::invalid_argument(
        "compatibility source paths must be normalized repo-relative paths");
  }
}

FileDescriptor open_source_beneath(int root_descriptor,
                                   const std::string& relative_path) {
  struct open_how how {};
  how.flags = O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK;
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS;
  const auto result = ::syscall(SYS_openat2, root_descriptor,
                                relative_path.c_str(), &how, sizeof(how));
  if (result < 0) {
    throw std::invalid_argument(
        system_error_message("could not securely open compatibility source " +
                             relative_path));
  }
  return FileDescriptor(static_cast<int>(result));
}

std::string compute_source_tree_digest(
    int root_descriptor, const std::set<std::string>& source_paths) {
  std::string tree_material = "trainvm.compatibility-source-tree/v1";
  for (const auto& path : source_paths) {
    auto descriptor = open_source_beneath(root_descriptor, path);
    const auto bytes = read_regular_file_stably(
        descriptor.get(), kMaximumSourceBytes,
        "compatibility source " + path);
    std::string leaf_material = "trainvm.compatibility-source-leaf/v1";
    leaf_material.push_back('\0');
    leaf_material.append(path);
    leaf_material.push_back('\0');
    leaf_material.append(sha256_bytes(bytes));
    tree_material.push_back('\0');
    tree_material.append(sha256_bytes(leaf_material));
  }
  return "sha256:" + sha256_bytes(tree_material);
}

std::string repository_identity(int descriptor) {
  struct stat status {};
  if (::fstat(descriptor, &status) != 0) {
    throw std::invalid_argument("could not identify compatibility repository root");
  }
  const std::string proc_path = "/proc/self/fd/" + std::to_string(descriptor);
  std::array<char, 4096> resolved{};
  const auto length = ::readlink(proc_path.c_str(), resolved.data(),
                                 resolved.size() - 1U);
  const std::string display = length > 0
      ? std::string(resolved.data(), static_cast<std::size_t>(length))
      : std::string("<unresolved>");
  return display + " [dev=" +
         std::to_string(static_cast<std::uintmax_t>(status.st_dev)) +
         ",ino=" + std::to_string(static_cast<std::uintmax_t>(status.st_ino)) +
         "]";
}

bool is_noninvocable(ObservedInvocationKind invocation) {
  return invocation == ObservedInvocationKind::library_only ||
         invocation == ObservedInvocationKind::design_only;
}

void validate_role(const CompatibilityWorkflowEntry& entry) {
  if (entry.observed_invocation == ObservedInvocationKind::library_only &&
      entry.operation_role != CompatibilityOperationRole::library_oracle) {
    throw std::invalid_argument(
        "library-only compatibility entries must have the library_oracle role");
  }
  if (entry.observed_invocation == ObservedInvocationKind::design_only &&
      entry.operation_role != CompatibilityOperationRole::design_spec) {
    throw std::invalid_argument(
        "design-only compatibility entries must have the design_spec role");
  }
  if (!is_noninvocable(entry.observed_invocation) &&
      (entry.operation_role == CompatibilityOperationRole::library_oracle ||
       entry.operation_role == CompatibilityOperationRole::design_spec)) {
    throw std::invalid_argument(
        "observed invocations cannot claim library/design-only roles");
  }
}

void validate_resume_claim(const CompatibilityWorkflowEntry& entry) {
  if (is_noninvocable(entry.observed_invocation) &&
      (entry.stateful ||
       entry.resume_evidence != CompatibilityResumeEvidence::none)) {
    throw std::invalid_argument(
        "library/design-only evidence cannot claim stateful resume");
  }
  if (!entry.stateful &&
      entry.resume_evidence != CompatibilityResumeEvidence::none) {
    throw std::invalid_argument(
        "stateless compatibility evidence must use resume evidence none");
  }
  if (entry.stateful &&
      entry.resume_evidence == CompatibilityResumeEvidence::none) {
    throw std::invalid_argument(
        "stateful compatibility evidence must declare observed resume evidence");
  }
}

std::string diagnostic_summary(const std::vector<Diagnostic>& diagnostics) {
  nlohmann::json output = nlohmann::json::array();
  for (const auto& diagnostic : diagnostics) {
    output.push_back({{"code", diagnostic.code},
                      {"path", diagnostic.path},
                      {"message", diagnostic.message}});
  }
  return output.dump();
}

}  // namespace

CompatibilityCatalog::CompatibilityCatalog(
    CompatibilityCatalogDocument document,
    const std::filesystem::path& repository_root) {
  if (document.api_version != "trainvm.compatibility-workflows/v1") {
    throw std::invalid_argument(
        "compatibility catalog api_version must be trainvm.compatibility-workflows/v1");
  }
  if (document.authority !=
      CompatibilityAuthority::compatibility_evidence_only) {
    throw std::invalid_argument(
        "compatibility catalog cannot grant adapter or host execution authority");
  }
  if (document.entries.empty() || document.entries.size() > kMaximumEntries) {
    throw std::invalid_argument(
        "compatibility catalog entries must be nonempty and bounded");
  }
  if (document.source_tree_digest.size() != 71U ||
      !document.source_tree_digest.starts_with("sha256:") ||
      !std::ranges::all_of(document.source_tree_digest.substr(7),
                           [](char character) {
        return std::isdigit(static_cast<unsigned char>(character)) ||
               (character >= 'a' && character <= 'f');
      })) {
    throw std::invalid_argument(
        "compatibility source_tree_digest must be lowercase sha256");
  }

  auto root_descriptor = open_repository_root(repository_root);
  repository_root_identity_display_ = repository_identity(root_descriptor.get());

  std::set<std::string> ids;
  std::set<std::string> all_source_paths;
  std::set<WorkflowFamily> families;
  for (auto& entry : document.entries) {
    if (!stable_identifier(entry.stable_id) ||
        !ids.insert(entry.stable_id).second) {
      throw std::invalid_argument(
          "compatibility workflow IDs must be unique stable identifiers");
    }
    if (entry.source_paths.empty() ||
        entry.source_paths.size() > kMaximumSourcePaths) {
      throw std::invalid_argument(
          "compatibility source_paths must be nonempty and bounded");
    }
    std::set<std::string> entry_source_paths;
    for (const auto& source_path : entry.source_paths) {
      validate_source_path_spelling(source_path);
      if (!entry_source_paths.insert(source_path).second) {
        throw std::invalid_argument(
            "compatibility source_paths must be unique within an entry");
      }
      all_source_paths.insert(source_path);
    }
    if (!bounded_text(entry.notes, kMaximumNotesBytes)) {
      throw std::invalid_argument(
          "compatibility notes must be bounded and nonempty");
    }
    if (entry.legacy_invocation_display &&
        !bounded_text(*entry.legacy_invocation_display,
                      kMaximumLegacyDisplayBytes)) {
      throw std::invalid_argument(
          "legacy invocation displays must be bounded nonempty text");
    }
    if (is_noninvocable(entry.observed_invocation) &&
        entry.legacy_invocation_display) {
      throw std::invalid_argument(
          "library/design-only evidence cannot declare a legacy invocation display");
    }
    validate_role(entry);
    validate_resume_claim(entry);
    families.insert(entry.family);
    std::ranges::sort(entry.source_paths);
  }

  const std::set<std::string> expected(kReviewedWorkflowIds.begin(),
                                       kReviewedWorkflowIds.end());
  if (ids != expected) {
    throw std::invalid_argument(
        "compatibility catalog differs from the compiled reviewed v1 workflow inventory");
  }

  static constexpr WorkflowFamily required_families[] = {
      WorkflowFamily::rwkv,
      WorkflowFamily::transformer,
      WorkflowFamily::vision_multimodal,
      WorkflowFamily::mageflow_diffusion,
      WorkflowFamily::conversion_distillation,
      WorkflowFamily::rwkv_posttraining,
      WorkflowFamily::rwkv_rlvr,
      WorkflowFamily::external_trainer,
      WorkflowFamily::data_cache,
      WorkflowFamily::evaluation_profile_export,
      WorkflowFamily::control_plane,
  };
  for (const auto family : required_families) {
    if (!families.contains(family)) {
      throw std::invalid_argument(
          "compatibility catalog is missing a canonical workflow family");
    }
  }

  const auto computed_source_digest =
      compute_source_tree_digest(root_descriptor.get(), all_source_paths);
  if (computed_source_digest != document.source_tree_digest) {
    throw std::invalid_argument(
        "compatibility source tree digest does not match referenced file bytes");
  }

  std::ranges::sort(document.entries, {},
                    &CompatibilityWorkflowEntry::stable_id);
  authority_ = document.authority;
  source_tree_digest_ = document.source_tree_digest;
  entries_ = std::move(document.entries);
  const nlohmann::json canonical = {
      {"api_version", "trainvm.compatibility-workflows/v1"},
      {"authority", enum_to_string(authority_)},
      {"source_tree_digest", source_tree_digest_},
      {"entries", encode_json(entries_)},
  };
  catalog_digest_ = "sha256:" + sha256_bytes(canonical.dump());
  if (catalog_digest_ != kReviewedCatalogDigest) {
    throw std::invalid_argument(
        "compatibility catalog digest " + catalog_digest_ +
        " differs from compiled reviewed v1 mapping " +
        std::string(kReviewedCatalogDigest));
  }
}

CompatibilityCatalog CompatibilityCatalog::load_file(
    const std::filesystem::path& catalog_path,
    const std::filesystem::path& repository_root) {
  CompatibilityCatalogDocument document;
  std::vector<Diagnostic> diagnostics;
  const auto source = parse_catalog(read_catalog(catalog_path));
  if (!decode_json(source, document, "", diagnostics)) {
    throw std::invalid_argument(
        "compatibility catalog schema validation failed: " +
        diagnostic_summary(diagnostics));
  }
  return CompatibilityCatalog(std::move(document), repository_root);
}

const std::vector<CompatibilityWorkflowEntry>&
CompatibilityCatalog::entries() const {
  return entries_;
}

const std::string& CompatibilityCatalog::catalog_digest() const {
  return catalog_digest_;
}

const std::string& CompatibilityCatalog::source_tree_digest() const {
  return source_tree_digest_;
}

const std::string& CompatibilityCatalog::repository_root_identity_display()
    const {
  return repository_root_identity_display_;
}

CompatibilityAuthority CompatibilityCatalog::authority() const {
  return authority_;
}

std::string_view CompatibilityCatalog::reviewed_catalog_digest() {
  return kReviewedCatalogDigest;
}

}  // namespace trainvm
