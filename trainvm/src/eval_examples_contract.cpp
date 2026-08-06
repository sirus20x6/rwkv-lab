#include "trainvm/eval_examples_contract.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <linux/openat2.h>
#include <memory>
#include <openssl/evp.h>
#include <set>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumExamples = 512U;
constexpr std::size_t kMaximumPartsPerField = 32U;
constexpr std::size_t kMaximumTextBytes = 16U * 1024U;
constexpr std::size_t kMaximumStructuredBytes = 16U * 1024U;

bool bounded_text(const std::string &value, std::size_t maximum = 1024U) {
  return !value.empty() && value.size() <= maximum &&
         std::ranges::none_of(value, [](unsigned char character) {
           return character == 0U || character == '\n' || character == '\r';
         });
}

bool digest(const std::string &value) {
  if (value.size() != 71U || !value.starts_with("sha256:"))
    return false;
  return std::ranges::all_of(value.substr(7), [](unsigned char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

void validate_parts(const std::vector<EvalExamplesPart> &parts,
                    std::string_view label) {
  if (parts.empty() || parts.size() > kMaximumPartsPerField)
    throw std::invalid_argument("eval example " + std::string(label) +
                                " parts are empty or exceed their bound");
  for (const EvalExamplesPart &part : parts) {
    if (part.kind == "text") {
      if (!part.text || part.text->empty() ||
          part.text->size() > kMaximumTextBytes || part.path ||
          part.media_type || part.sha256 || part.size_bytes || part.schema ||
          part.value)
        throw std::invalid_argument("eval text part is not canonical");
      continue;
    }
    if (part.kind == "image" || part.kind == "video" || part.kind == "audio") {
      if (part.text || !part.path || !bounded_text(*part.path, 4096U) ||
          std::filesystem::path(*part.path).is_absolute() ||
          std::ranges::any_of(std::filesystem::path(*part.path),
                              [](const auto &element) {
                                return element == "." || element == "..";
                              }) ||
          !part.media_type || !bounded_text(*part.media_type, 256U) ||
          !part.sha256 || !digest(*part.sha256) || !part.size_bytes ||
          *part.size_bytes == 0U || part.schema || part.value)
        throw std::invalid_argument("eval media part is not canonical");
      continue;
    }
    if (part.kind == "structured") {
      if (part.text || part.path || part.media_type || part.sha256 ||
          part.size_bytes || !part.schema ||
          !bounded_text(*part.schema, 256U) || !part.value ||
          !(part.value->is_object() || part.value->is_array()) ||
          part.value->empty() ||
          part.value->dump().size() > kMaximumStructuredBytes)
        throw std::invalid_argument("eval structured part is not canonical");
      continue;
    }
    throw std::invalid_argument("eval example part kind is unsupported");
  }
}

class Descriptor final {
public:
  explicit Descriptor(int value = -1) : value_(value) {}
  Descriptor(const Descriptor &) = delete;
  Descriptor &operator=(const Descriptor &) = delete;
  Descriptor(Descriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  ~Descriptor() {
    if (value_ >= 0)
      ::close(value_);
  }
  int get() const { return value_; }
  explicit operator bool() const { return value_ >= 0; }

private:
  int value_;
};

Descriptor open_beneath(int root, std::string_view relative, int flags) {
  const std::string owned(relative);
  open_how how{};
  how.flags = static_cast<std::uint64_t>(flags | O_CLOEXEC | O_NOFOLLOW);
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS;
  const long result =
      ::syscall(SYS_openat2, root, owned.c_str(), &how, sizeof(how));
  if (result < 0)
    throw std::invalid_argument("eval-examples payload path cannot be pinned");
  return Descriptor(static_cast<int>(result));
}

std::string descriptor_sha256(int descriptor, std::uint64_t expected_size) {
  if (::lseek(descriptor, 0, SEEK_SET) < 0)
    throw std::invalid_argument("eval-examples payload cannot be rewound");
  EVP_MD_CTX *raw = EVP_MD_CTX_new();
  if (raw == nullptr)
    throw std::runtime_error("eval-examples SHA-256 allocation failed");
  const auto release = [](EVP_MD_CTX *value) { EVP_MD_CTX_free(value); };
  std::unique_ptr<EVP_MD_CTX, decltype(release)> context(raw, release);
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("eval-examples SHA-256 initialization failed");
  std::array<unsigned char, 1024U * 1024U> buffer{};
  std::uint64_t total = 0U;
  while (true) {
    const ssize_t count = ::read(descriptor, buffer.data(), buffer.size());
    if (count < 0) {
      if (errno == EINTR)
        continue;
      throw std::invalid_argument("eval-examples payload read failed");
    }
    if (count == 0)
      break;
    total += static_cast<std::uint64_t>(count);
    if (total > expected_size ||
        EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1)
      throw std::invalid_argument("eval-examples payload size or hash failed");
  }
  if (total != expected_size)
    throw std::invalid_argument("eval-examples payload size disagrees");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest_value{};
  unsigned int length = 0U;
  if (EVP_DigestFinal_ex(context.get(), digest_value.data(), &length) != 1 ||
      length != 32U)
    throw std::runtime_error("eval-examples SHA-256 finalization failed");
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result("sha256:");
  result.reserve(71U);
  for (unsigned int index = 0U; index < length; ++index) {
    result.push_back(alphabet[digest_value[index] >> 4U]);
    result.push_back(alphabet[digest_value[index] & 0x0fU]);
  }
  return result;
}

} // namespace

EvalExamplesManifest
validate_eval_examples_manifest(const nlohmann::json &document) {
  EvalExamplesManifest manifest;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(document, manifest, "/", diagnostics) ||
      !diagnostics.empty())
    throw std::invalid_argument(
        "eval-examples manifest has invalid reflected shape");
  if (manifest.api_version != kEvalExamplesSchema ||
      !bounded_text(manifest.run_id) || !bounded_text(manifest.node_id) ||
      !bounded_text(manifest.attempt_id) ||
      manifest.step_domain != "optimizer_step" ||
      !bounded_text(manifest.series_id) ||
      !bounded_text(manifest.heldout.identity_field, 256U) ||
      !digest(manifest.heldout.identities_digest) ||
      !digest(manifest.heldout.selector_digest) ||
      !digest(manifest.evaluator.component_digest) ||
      manifest.evaluator.metric_names.empty() ||
      manifest.evaluator.metric_names.size() > 64U ||
      !bounded_text(manifest.checkpoint.artifact_id) ||
      !digest(manifest.checkpoint.manifest_digest) ||
      !digest(manifest.policy_digest) ||
      !digest(manifest.canonical_manifest_digest) ||
      manifest.examples.empty() || manifest.examples.size() > kMaximumExamples)
    throw std::invalid_argument("eval-examples manifest fields are invalid");

  std::set<std::string> metrics;
  for (const std::string &metric : manifest.evaluator.metric_names) {
    if (!bounded_text(metric, 256U) || !metrics.insert(metric).second)
      throw std::invalid_argument(
          "eval-examples metric identities are invalid");
  }
  std::set<std::string> example_ids;
  std::set<std::string> heldout_ids;
  for (const EvalExample &example : manifest.examples) {
    if (!bounded_text(example.example_id) ||
        !example_ids.insert(example.example_id).second ||
        !bounded_text(example.heldout_item_id, 4096U) ||
        !heldout_ids.insert(example.heldout_item_id).second ||
        !digest(example.heldout_item_digest))
      throw std::invalid_argument("eval example identities are invalid");
    validate_parts(example.input, "input");
    validate_parts(example.target, "target");
    validate_parts(example.prediction, "prediction");
  }
  nlohmann::json body = document;
  body.erase("canonical_manifest_digest");
  if (manifest.canonical_manifest_digest != "sha256:" + sha256_hex(body.dump()))
    throw std::invalid_argument(
        "eval-examples canonical manifest digest disagrees");
  if (encode_json(manifest) != document)
    throw std::invalid_argument("eval-examples manifest is not canonical");
  return manifest;
}

void validate_eval_examples_payload(const EvalExamplesManifest &manifest,
                                    std::string_view manifest_uri,
                                    std::string_view canonical_manifest_bytes,
                                    std::string_view manifest_fingerprint,
                                    std::string_view run_directory,
                                    std::string_view artifact_id) {
  constexpr std::string_view prefix = "file://";
  if (!manifest_uri.starts_with(prefix) ||
      manifest_uri.find('%') != std::string_view::npos)
    throw std::invalid_argument(
        "eval-examples manifest URI must be an unescaped local file URI");
  const std::filesystem::path manifest_path(
      std::string(manifest_uri.substr(prefix.size())));
  if (!manifest_path.is_absolute() ||
      manifest_path.filename() != "manifest.json")
    throw std::invalid_argument("eval-examples manifest URI is not canonical");
  const std::filesystem::path run_root(run_directory);
  const std::filesystem::path artifact_component(artifact_id);
  if (!run_root.is_absolute() || run_root.lexically_normal() != run_root ||
      !bounded_text(std::string(artifact_id)) ||
      artifact_component != artifact_component.filename() ||
      artifact_component == "." || artifact_component == "..")
    throw std::invalid_argument(
        "eval-examples immutable revision authority is invalid");
  const std::filesystem::path expected_revision =
      run_root / "trainvm_artifacts" / "eval_examples" / "revisions" /
      artifact_component;
  if (manifest_path.lexically_normal() != manifest_path ||
      manifest_path.parent_path() != expected_revision)
    throw std::invalid_argument(
        "eval-examples manifest is outside its immutable run revision");
  Descriptor filesystem_root(
      ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (!filesystem_root)
    throw std::runtime_error("eval-examples filesystem root cannot be pinned");
  const std::string relative_manifest =
      manifest_path.lexically_relative("/").generic_string();
  Descriptor manifest_file =
      open_beneath(filesystem_root.get(), relative_manifest, O_RDONLY);
  struct stat manifest_status{};
  if (::fstat(manifest_file.get(), &manifest_status) != 0 ||
      !S_ISREG(manifest_status.st_mode) || manifest_status.st_size < 0 ||
      static_cast<std::uint64_t>(manifest_status.st_size) !=
          canonical_manifest_bytes.size() ||
      descriptor_sha256(manifest_file.get(), canonical_manifest_bytes.size()) !=
          manifest_fingerprint)
    throw std::invalid_argument("eval-examples manifest payload disagrees");
  if (::lseek(manifest_file.get(), 0, SEEK_SET) < 0)
    throw std::invalid_argument("eval-examples manifest cannot be rewound");
  std::string observed(canonical_manifest_bytes.size(), '\0');
  std::size_t offset = 0U;
  while (offset < observed.size()) {
    const ssize_t count = ::read(manifest_file.get(), observed.data() + offset,
                                 observed.size() - offset);
    if (count < 0) {
      if (errno == EINTR)
        continue;
      throw std::invalid_argument("eval-examples manifest read failed");
    }
    if (count == 0)
      throw std::invalid_argument("eval-examples manifest was truncated");
    offset += static_cast<std::size_t>(count);
  }
  if (observed != canonical_manifest_bytes)
    throw std::invalid_argument("eval-examples manifest bytes disagree");

  const std::string revision_relative =
      manifest_path.parent_path().lexically_relative("/").generic_string();
  Descriptor revision = open_beneath(filesystem_root.get(), revision_relative,
                                     O_PATH | O_DIRECTORY);
  for (const EvalExample &example : manifest.examples) {
    const auto validate = [&](const std::vector<EvalExamplesPart> &parts) {
      for (const EvalExamplesPart &part : parts) {
        if (!part.path)
          continue;
        Descriptor media = open_beneath(revision.get(), *part.path, O_RDONLY);
        struct stat status{};
        if (::fstat(media.get(), &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_size < 0 || !part.size_bytes || !part.sha256 ||
            static_cast<std::uint64_t>(status.st_size) != *part.size_bytes ||
            descriptor_sha256(media.get(), *part.size_bytes) != *part.sha256)
          throw std::invalid_argument(
              "eval-examples media payload disagrees with manifest");
      }
    };
    validate(example.input);
    validate(example.target);
    validate(example.prediction);
  }
}

void validate_eval_examples_gate_provenance(
    const EvalExamplesManifest &manifest,
    const nlohmann::json &resolved_training,
    const std::vector<Event> &prior_events) {
  if (!resolved_training.is_object() ||
      !resolved_training.contains("components") ||
      !resolved_training.at("components").is_object())
    throw std::invalid_argument(
        "eval-examples has no resolved training component authority");
  const nlohmann::json *evaluator = nullptr;
  for (const auto &[slot, component] :
       resolved_training.at("components").items()) {
    (void)slot;
    if (!component.is_object() || !component.contains("descriptor") ||
        !component.at("descriptor").is_object() ||
        !component.at("descriptor").contains("key") ||
        !component.at("descriptor").at("key").is_object() ||
        component.at("descriptor").at("key").value("category", std::string{}) !=
            "evaluator")
      continue;
    if (evaluator != nullptr)
      throw std::invalid_argument(
          "eval-examples resolved training has multiple evaluators");
    evaluator = &component;
  }
  if (evaluator == nullptr || !evaluator->contains("configuration") ||
      !evaluator->at("configuration").is_object() ||
      !evaluator->at("configuration").contains("metrics") ||
      !evaluator->at("configuration").at("metrics").is_array() ||
      std::ranges::any_of(
          evaluator->at("configuration").at("metrics"),
          [](const nlohmann::json &metric) { return !metric.is_string(); }) ||
      evaluator->value("descriptor_digest", std::string{}) !=
          manifest.evaluator.component_digest)
    throw std::invalid_argument(
        "eval-examples evaluator provenance disagrees with resolved training");
  const std::vector<std::string> declared_metrics =
      evaluator->at("configuration")
          .at("metrics")
          .get<std::vector<std::string>>();
  std::set<std::string> declared(declared_metrics.begin(),
                                 declared_metrics.end());
  std::set<std::string> manifested(manifest.evaluator.metric_names.begin(),
                                   manifest.evaluator.metric_names.end());
  if (declared.empty() || declared != manifested)
    throw std::invalid_argument(
        "eval-examples scalar metrics disagree with resolved evaluator");

  bool checkpoint = false;
  bool scalar = false;
  for (const Event &event : prior_events) {
    if (event.run_id != manifest.run_id || event.node_id != manifest.node_id)
      continue;
    if (event.event_type == "artifact.published" && event.payload.is_object() &&
        event.payload.value("artifact_id", std::string{}) ==
            manifest.checkpoint.artifact_id &&
        event.payload.value("kind", std::string{}) == "checkpoint" &&
        event.optimizer_step == 0U &&
        event.payload.value("complete", false) &&
        event.payload.value("fingerprint_algorithm", std::string{}) ==
            "manifest_sha256" &&
        event.payload.value("fingerprint", std::string{}) ==
            manifest.checkpoint.manifest_digest)
      checkpoint = true;
    if (event.event_type == "metric.sampled" && event.optimizer_step == 0U &&
        event.payload.is_object() &&
        event.payload.value("step_domain", std::string{}) == "optimizer_step" &&
        declared.contains(event.payload.value("name", std::string{})))
      scalar = true;
  }
  if (!checkpoint)
    throw std::invalid_argument("eval-examples checkpoint is not one prior "
                                "durable matching checkpoint artifact");
  if (!scalar)
    throw std::invalid_argument("eval-examples has no prior durable declared "
                                "evaluator scalar at step zero");
}

bool invocation_requires_step_zero_eval_gate(const nlohmann::json &publishes) {
  if (!publishes.is_object())
    return false;
  return std::ranges::any_of(publishes.items(), [](const auto &item) {
    const nlohmann::json &publication = item.value();
    const auto declaration = publication.find("declaration");
    return declaration != publication.end() && declaration->is_object() &&
           declaration->value("required", false) &&
           declaration->value("type", std::string{}) == "eval_examples" &&
           declaration->value("schema", std::string{}) == kEvalExamplesSchema;
  });
}

bool durable_step_zero_eval_gate_satisfied(const std::vector<Event> &events,
                                           std::string_view run_id,
                                           std::string_view node_id) {
  std::set<std::string> step_zero_metrics;
  for (const Event &event : events) {
    if (event.run_id != run_id || event.node_id != node_id)
      continue;
    if (event.event_type == "metric.sampled" && event.optimizer_step == 0U &&
        event.payload.is_object() &&
        event.payload.value("step_domain", std::string{}) == "optimizer_step") {
      const std::string name = event.payload.value("name", std::string{});
      if (!name.empty())
        step_zero_metrics.insert(name);
      continue;
    }
    if (event.event_type != "artifact.published" ||
        event.optimizer_step != 0U || !event.payload.is_object() ||
        event.payload.value("kind", std::string{}) != "eval_examples" ||
        event.payload.value("schema", std::string{}) != kEvalExamplesSchema ||
        !event.payload.value("complete", false) ||
        !event.payload.contains("eval_examples_manifest") ||
        !event.payload.at("eval_examples_manifest").is_object())
      continue;
    try {
      const EvalExamplesManifest manifest = validate_eval_examples_manifest(
          event.payload.at("eval_examples_manifest"));
      if (manifest.run_id != run_id || manifest.node_id != node_id ||
          manifest.optimizer_step != 0U)
        continue;
      if (std::ranges::any_of(manifest.evaluator.metric_names,
                              [&](const std::string &name) {
                                return step_zero_metrics.contains(name);
                              }))
        return true;
    } catch (const std::invalid_argument &) {
    }
  }
  return false;
}

} // namespace trainvm
