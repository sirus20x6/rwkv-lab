#include "trainvm/linux_cache_evidence.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/fs.h>
#include <linux/openat2.h>
#include <openssl/rand.h>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumPathBytes = 4096U;

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

[[noreturn]] void fail(std::string_view message) {
  throw CacheArtifactAuthorityError(std::string(message));
}

[[noreturn]] void fail_errno(std::string_view message) {
  throw CacheArtifactAuthorityError(std::string(message) + ": " +
                                    std::strerror(errno));
}

bool lower_hex(std::string_view value) {
  return std::ranges::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool valid_sha256(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         lower_hex(value.substr(7U));
}

std::string receipt_name(std::string_view domain,
                         const nlohmann::json& identity) {
  return sha256_hex(nlohmann::json{{"domain", domain}, {"identity", identity}}
                        .dump()) +
         ".json";
}

std::string temporary_name() {
  std::array<unsigned char, 16U> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    fail("cache evidence temporary-name entropy failed");
  }
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result = ".publish-";
  for (unsigned char byte : bytes) {
    result.push_back(alphabet[byte >> 4U]);
    result.push_back(alphabet[byte & 0x0fU]);
  }
  return result;
}

Descriptor open_beneath(int root, std::string_view relative, int flags) {
  struct open_how how{};
  how.flags = static_cast<std::uint64_t>(flags | O_CLOEXEC | O_NOFOLLOW);
  how.resolve = RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
                RESOLVE_NO_XDEV;
  const std::string owned(relative);
  const long result =
      ::syscall(SYS_openat2, root, owned.c_str(), &how, sizeof(how));
  if (result < 0)
    fail_errno("cache evidence path resolution failed");
  return Descriptor(static_cast<int>(result));
}

void validate_directory(int descriptor, uid_t authority_uid, dev_t device,
                        std::string_view message) {
  struct stat metadata{};
  if (::fstat(descriptor, &metadata) != 0 || !S_ISDIR(metadata.st_mode) ||
      metadata.st_uid != authority_uid || metadata.st_dev != device ||
      (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    fail(message);
  }
}

class ImmutableReceiptDirectory final {
 public:
  ImmutableReceiptDirectory(LinuxCacheEvidenceConfig config,
                            std::string_view category,
                            bool require_owner_write = false)
      : config_(std::move(config)) {
    if (config_.receipt_root.empty() || !config_.receipt_root.is_absolute() ||
        config_.receipt_root.lexically_normal() != config_.receipt_root ||
        config_.receipt_root.native().size() > kMaximumPathBytes ||
        config_.authority_uid != ::geteuid() ||
        config_.maximum_receipt_bytes == 0U ||
        config_.maximum_receipt_bytes > (16U << 20U)) {
      fail("cache evidence configuration is malformed or exceeds bounds");
    }
    root_ = Descriptor(::open(config_.receipt_root.c_str(),
                              O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (root_.get() < 0)
      fail_errno("cache evidence root could not be opened");
    struct stat root_metadata{};
    if (::fstat(root_.get(), &root_metadata) != 0 ||
        !S_ISDIR(root_metadata.st_mode)) {
      fail("cache evidence root is not a directory");
    }
    device_ = root_metadata.st_dev;
    validate_directory(root_.get(), config_.authority_uid, device_,
                       "cache evidence root has unsafe metadata");
    category_ = open_beneath(root_.get(), category, O_RDONLY | O_DIRECTORY);
    validate_directory(category_.get(), config_.authority_uid, device_,
                       "cache evidence category has unsafe metadata");
    struct stat category_metadata {};
    if (require_owner_write &&
        (::fstat(category_.get(), &category_metadata) != 0 ||
         (category_metadata.st_mode & S_IWUSR) == 0)) {
      fail("cache evidence publisher category is not owner-writable");
    }
  }

  [[nodiscard]] nlohmann::json read(std::string_view name) const {
    if (name.size() != 69U || !lower_hex(name.substr(0U, 64U)) ||
        name.substr(64U) != ".json") {
      fail("cache evidence receipt name is invalid");
    }
    Descriptor file =
        open_beneath(category_.get(), name, O_RDONLY | O_NONBLOCK);
    struct stat before{};
    if (::fstat(file.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
        before.st_uid != config_.authority_uid || before.st_dev != device_ ||
        before.st_nlink != 1 || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) >
            config_.maximum_receipt_bytes ||
        (before.st_mode & 0222) != 0) {
      fail("cache evidence receipt metadata is unsafe or exceeds bounds");
    }
    std::string bytes(static_cast<std::size_t>(before.st_size), '\0');
    std::size_t offset = 0U;
    while (offset < bytes.size()) {
      const ssize_t count =
          ::read(file.get(), bytes.data() + offset, bytes.size() - offset);
      if (count < 0 && errno == EINTR)
        continue;
      if (count <= 0)
        fail_errno("cache evidence receipt read failed");
      offset += static_cast<std::size_t>(count);
    }
    char trailing{};
    ssize_t trailing_count{};
    do {
      trailing_count = ::read(file.get(), &trailing, 1U);
    } while (trailing_count < 0 && errno == EINTR);
    struct stat after{};
    if (trailing_count != 0 || ::fstat(file.get(), &after) != 0 ||
        before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_nlink != after.st_nlink || before.st_uid != after.st_uid ||
        before.st_mode != after.st_mode || before.st_size != after.st_size ||
        before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
        before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
        before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
        before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
      fail("cache evidence receipt changed while it was read");
    }
    nlohmann::json parsed;
    try {
      parsed = nlohmann::json::parse(bytes);
    } catch (const nlohmann::json::exception&) {
      fail("cache evidence receipt is malformed JSON");
    }
    if (!parsed.is_object() || parsed.dump() != bytes) {
      fail("cache evidence receipt is not canonical JSON");
    }
    return parsed;
  }

  void publish(std::string_view name, const nlohmann::json& document) const {
    const std::string bytes = document.dump();
    if (!document.is_object() || bytes.empty() ||
        bytes.size() > config_.maximum_receipt_bytes) {
      fail("cache evidence publication document is invalid or oversized");
    }
    const std::string final_name(name);
    const std::string temporary = temporary_name();
    Descriptor file(::openat(category_.get(), temporary.c_str(),
                             O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                 O_NOFOLLOW,
                             0600));
    if (file.get() < 0)
      fail_errno("cache evidence temporary receipt creation failed");
    try {
      std::size_t offset = 0U;
      while (offset < bytes.size()) {
        const ssize_t count =
            ::write(file.get(), bytes.data() + offset, bytes.size() - offset);
        if (count < 0 && errno == EINTR)
          continue;
        if (count <= 0)
          fail_errno("cache evidence receipt write failed");
        offset += static_cast<std::size_t>(count);
      }
      if (::fchmod(file.get(), 0440) != 0 || ::fsync(file.get()) != 0) {
        fail("cache evidence receipt mode or data sync failed");
      }
      if (::syscall(SYS_renameat2, category_.get(), temporary.c_str(),
                    category_.get(), final_name.c_str(),
                    RENAME_NOREPLACE) != 0) {
        if (errno != EEXIST)
          fail_errno("cache evidence atomic promotion failed");
        if (::unlinkat(category_.get(), temporary.c_str(), 0) != 0)
          fail_errno("cache evidence replay cleanup failed");
        if (::fsync(category_.get()) != 0)
          fail("cache evidence replay cleanup was not durable");
        if (read(final_name) != document) {
          fail("cache evidence receipt identity has conflicting content");
        }
        return;
      }
      if (::fsync(category_.get()) != 0)
        fail("cache evidence directory sync failed");
      if (read(final_name) != document)
        fail("published cache evidence differs from requested bytes");
    } catch (...) {
      (void)::unlinkat(category_.get(), temporary.c_str(), 0);
      throw;
    }
  }

 private:
  LinuxCacheEvidenceConfig config_;
  Descriptor root_;
  Descriptor category_;
  dev_t device_{};
};

template <typename Value>
Value decode_exact(const nlohmann::json& input, std::string_view label) {
  Value result{};
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(input, result, "", diagnostics) || !diagnostics.empty() ||
      encode_json(result) != input) {
    fail(std::string(label) + " has an invalid reflected schema");
  }
  return result;
}

void validate_runtime_context_identity(
    const CacheRuntimeProbeContext& context) {
  if (!valid_sha256(context.host.host_id) || context.host.boot_id.empty() ||
      context.host.boot_id.size() > 256U ||
      !valid_sha256(context.launch_spec_digest) ||
      !valid_sha256(context.inventory_receipt_digest) ||
      !valid_sha256(context.resource_binding_digest) ||
      context.selected_resources.size() > 256U) {
    fail("cache runtime probe context identity is malformed");
  }
}

void validate_qualification_identity(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact) {
  (void)cache_namespace_authority_receipt_json(authority);
  if (artifact.api_version != "trainvm.immutable-cache-tree/v1" ||
      !valid_sha256(artifact.namespace_digest) ||
      !valid_sha256(artifact.artifact_tree_digest) ||
      !valid_sha256(artifact.store_receipt_digest) ||
      artifact.namespace_digest != authority.cache_namespace.namespace_digest ||
      artifact.file_count == 0U || artifact.total_bytes == 0U ||
      !artifact.immutable) {
    fail("cache qualification receipt identity is malformed");
  }
}

}  // namespace

std::string
cache_runtime_probe_receipt_name(const CacheRuntimeProbeContext& context) {
  validate_runtime_context_identity(context);
  return receipt_name("trainvm.cache-runtime-probe-receipt-name/v1",
                      encode_json(context));
}

nlohmann::json
cache_runtime_probe_receipt_json(const CacheRuntimeProbeContext& context,
                                 const CacheRuntimeProbeSnapshot& snapshot) {
  const std::string name = cache_runtime_probe_receipt_name(context);
  return {{"api_version", "trainvm.sealed-cache-runtime-probe/v1"},
          {"receipt_name", name},
          {"context", encode_json(context)},
          {"snapshot", encode_json(snapshot)}};
}

std::string cache_qualification_evidence_receipt_name(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact) {
  validate_qualification_identity(authority, artifact);
  return receipt_name(
      "trainvm.cache-qualification-evidence-receipt-name/v1",
      {{"authority_receipt_digest", authority.receipt_digest},
       {"artifact_store_receipt_digest", artifact.store_receipt_digest}});
}

nlohmann::json cache_qualification_evidence_receipt_json(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact,
    const CacheQualificationEvidence& evidence) {
  const std::string name =
      cache_qualification_evidence_receipt_name(authority, artifact);
  return {{"api_version", "trainvm.immutable-cache-qualification-evidence/v1"},
          {"receipt_name", name},
          {"authority_receipt_digest", authority.receipt_digest},
          {"artifact_store_receipt_digest", artifact.store_receipt_digest},
          {"evidence", encode_json(evidence)}};
}

struct LinuxSealedCacheRuntimeProbe::Implementation {
  explicit Implementation(LinuxCacheEvidenceConfig config)
      : receipts(std::move(config), "runtime") {}
  ImmutableReceiptDirectory receipts;
};

LinuxSealedCacheRuntimeProbe::LinuxSealedCacheRuntimeProbe(
    LinuxCacheEvidenceConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

LinuxSealedCacheRuntimeProbe::~LinuxSealedCacheRuntimeProbe() = default;
LinuxSealedCacheRuntimeProbe::LinuxSealedCacheRuntimeProbe(
    LinuxSealedCacheRuntimeProbe&&) noexcept = default;
LinuxSealedCacheRuntimeProbe& LinuxSealedCacheRuntimeProbe::operator=(
    LinuxSealedCacheRuntimeProbe&&) noexcept = default;

CacheRuntimeProbeSnapshot
LinuxSealedCacheRuntimeProbe::capture(const CacheRuntimeProbeContext& context) {
  const std::string name = cache_runtime_probe_receipt_name(context);
  const nlohmann::json receipt = implementation_->receipts.read(name);
  if (receipt.size() != 4U ||
      receipt.value("api_version", std::string{}) !=
          "trainvm.sealed-cache-runtime-probe/v1" ||
      receipt.value("receipt_name", std::string{}) != name ||
      !receipt.contains("context") || !receipt.contains("snapshot")) {
    fail("sealed cache runtime receipt envelope is invalid");
  }
  const CacheRuntimeProbeContext observed_context =
      decode_exact<CacheRuntimeProbeContext>(receipt.at("context"),
                                             "runtime probe context");
  const CacheRuntimeProbeSnapshot snapshot =
      decode_exact<CacheRuntimeProbeSnapshot>(receipt.at("snapshot"),
                                              "runtime probe snapshot");
  if (observed_context != context ||
      receipt != cache_runtime_probe_receipt_json(context, snapshot)) {
    fail("sealed cache runtime receipt is bound to a different context");
  }
  return snapshot;
}

struct LinuxImmutableCacheQualificationSource::Implementation {
  explicit Implementation(LinuxCacheEvidenceConfig config)
      : receipts(std::move(config), "qualification") {}

  CacheQualificationEvidence
  load(const CacheNamespaceAuthorityReceipt& authority,
       const ImmutableCacheTreeReceipt& artifact) const {
    const std::string name =
        cache_qualification_evidence_receipt_name(authority, artifact);
    const nlohmann::json receipt = receipts.read(name);
    if (receipt.size() != 5U ||
        receipt.value("api_version", std::string{}) !=
            "trainvm.immutable-cache-qualification-evidence/v1" ||
        receipt.value("receipt_name", std::string{}) != name ||
        receipt.value("authority_receipt_digest", std::string{}) !=
            authority.receipt_digest ||
        receipt.value("artifact_store_receipt_digest", std::string{}) !=
            artifact.store_receipt_digest ||
        !receipt.contains("evidence")) {
      fail("immutable cache qualification evidence envelope is invalid");
    }
    CacheQualificationEvidence evidence =
        decode_exact<CacheQualificationEvidence>(receipt.at("evidence"),
                                                 "qualification evidence");
    if (evidence.authority_receipt_digest != authority.receipt_digest ||
        evidence.namespace_digest != artifact.namespace_digest ||
        evidence.artifact_tree_digest != artifact.artifact_tree_digest ||
        receipt != cache_qualification_evidence_receipt_json(
                       authority, artifact, evidence)) {
      fail("immutable cache qualification evidence has different identity");
    }
    (void)qualify_cache_artifact(evidence);
    return evidence;
  }

  ImmutableReceiptDirectory receipts;
};

LinuxImmutableCacheQualificationSource::LinuxImmutableCacheQualificationSource(
    LinuxCacheEvidenceConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

LinuxImmutableCacheQualificationSource::
    ~LinuxImmutableCacheQualificationSource() = default;
LinuxImmutableCacheQualificationSource::LinuxImmutableCacheQualificationSource(
    LinuxImmutableCacheQualificationSource&&) noexcept = default;
LinuxImmutableCacheQualificationSource&
LinuxImmutableCacheQualificationSource::operator=(
    LinuxImmutableCacheQualificationSource&&) noexcept = default;

CacheQualificationEvidence LinuxImmutableCacheQualificationSource::capture(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact) {
  return implementation_->load(authority, artifact);
}

void LinuxImmutableCacheQualificationSource::require_trusted(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact,
    const CacheQualificationReceipt& qualification) {
  const CacheQualificationEvidence evidence =
      implementation_->load(authority, artifact);
  if (qualification != qualify_cache_artifact(evidence)) {
    fail("cache qualification receipt is not backed by immutable evidence");
  }
}

struct LinuxCacheEvidencePublisher::Implementation {
  explicit Implementation(LinuxCacheEvidenceConfig config)
      : runtime(config, "runtime", true),
        qualification(std::move(config), "qualification", true) {}

  ImmutableReceiptDirectory runtime;
  ImmutableReceiptDirectory qualification;
};

LinuxCacheEvidencePublisher::LinuxCacheEvidencePublisher(
    LinuxCacheEvidenceConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}

LinuxCacheEvidencePublisher::~LinuxCacheEvidencePublisher() = default;
LinuxCacheEvidencePublisher::LinuxCacheEvidencePublisher(
    LinuxCacheEvidencePublisher&&) noexcept = default;
LinuxCacheEvidencePublisher& LinuxCacheEvidencePublisher::operator=(
    LinuxCacheEvidencePublisher&&) noexcept = default;

std::string LinuxCacheEvidencePublisher::publish_runtime(
    const CacheRuntimeProbeContext& context,
    const CacheRuntimeProbeSnapshot& snapshot) {
  validate_cache_runtime_probe_snapshot(snapshot, context);
  const std::string name = cache_runtime_probe_receipt_name(context);
  implementation_->runtime.publish(
      name, cache_runtime_probe_receipt_json(context, snapshot));
  return name;
}

std::string LinuxCacheEvidencePublisher::publish_qualification(
    const CacheNamespaceAuthorityReceipt& authority,
    const ImmutableCacheTreeReceipt& artifact,
    const CacheQualificationEvidence& evidence) {
  validate_qualification_identity(authority, artifact);
  if (evidence.authority_receipt_digest != authority.receipt_digest ||
      evidence.namespace_digest != artifact.namespace_digest ||
      evidence.artifact_tree_digest != artifact.artifact_tree_digest) {
    fail("cache qualification publication has different evidence identity");
  }
  (void)qualify_cache_artifact(evidence);
  const std::string name =
      cache_qualification_evidence_receipt_name(authority, artifact);
  implementation_->qualification.publish(
      name, cache_qualification_evidence_receipt_json(authority, artifact,
                                                      evidence));
  return name;
}

}  // namespace trainvm
