#include "trainvm/host_launch.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <linux/memfd.h>
#include <linux/openat2.h>
#include <openssl/evp.h>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::uint64_t kMaximumLaunchArtifactBytes = 256ULL << 20U;

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) : value_(value) {}
  Descriptor(Descriptor&& other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  Descriptor& operator=(Descriptor&& other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  ~Descriptor() { reset(); }

  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;

  [[nodiscard]] int get() const { return value_; }
  [[nodiscard]] int release() { return std::exchange(value_, -1); }

 private:
  void reset() noexcept {
    if (value_ >= 0) (void)::close(value_);
    value_ = -1;
  }

  int value_;
};

bool path_within(const std::filesystem::path& child,
                 const std::filesystem::path& root) {
  const auto normalized_child = child.lexically_normal();
  const auto normalized_root = root.lexically_normal();
  auto child_iterator = normalized_child.begin();
  for (auto root_iterator = normalized_root.begin();
       root_iterator != normalized_root.end();
       ++root_iterator, ++child_iterator) {
    if (child_iterator == normalized_child.end() ||
        *child_iterator != *root_iterator) {
      return false;
    }
  }
  return true;
}

int openat2_checked(int directory, const char* path, int flags,
                    std::uint64_t resolve) {
  struct open_how how {};
  how.flags = static_cast<std::uint64_t>(flags);
  how.resolve = resolve;
  const long result = ::syscall(SYS_openat2, directory, path, &how,
                                sizeof(how));
  if (result < 0) {
    throw HostLaunchResolutionError(
        "secure host path resolution failed: " +
        std::string(std::strerror(errno)));
  }
  return static_cast<int>(result);
}

struct OpenedPath {
  Descriptor descriptor;
  struct stat metadata {};
};

OpenedPath open_beneath(const std::vector<std::string>& trusted_roots,
                        const std::string& requested_path, int flags,
                        bool require_directory) {
  const std::filesystem::path requested =
      std::filesystem::path(requested_path).lexically_normal();
  const std::string* selected_root = nullptr;
  for (const std::string& root : trusted_roots) {
    if (path_within(requested, root) &&
        (selected_root == nullptr || root.size() > selected_root->size())) {
      selected_root = &root;
    }
  }
  if (selected_root == nullptr) {
    throw HostLaunchResolutionError(
        "host launch path is outside every trusted root");
  }
  Descriptor root(openat2_checked(
      AT_FDCWD, selected_root->c_str(), O_PATH | O_DIRECTORY | O_CLOEXEC,
      RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS));
  struct stat root_metadata {};
  if (::fstat(root.get(), &root_metadata) != 0 ||
      !S_ISDIR(root_metadata.st_mode) ||
      (root_metadata.st_uid != 0U &&
       root_metadata.st_uid != ::geteuid()) ||
      (root_metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw HostLaunchResolutionError(
        "trusted host launch root has unsafe ownership or mode");
  }
  const auto relative = requested.lexically_relative(*selected_root);
  if (relative.empty() || relative == ".") {
    if (!require_directory) {
      throw HostLaunchResolutionError(
          "launch artifact cannot be the trusted root directory");
    }
    const int duplicate = ::fcntl(root.get(), F_DUPFD_CLOEXEC, 3);
    if (duplicate < 0) {
      throw HostLaunchResolutionError(
          "could not retain trusted working directory");
    }
    return {.descriptor = Descriptor(duplicate),
            .metadata = root_metadata};
  }
  int open_flags = flags | O_CLOEXEC | O_NOFOLLOW;
  if (require_directory) open_flags |= O_DIRECTORY;
  Descriptor target(openat2_checked(
      root.get(), relative.c_str(), open_flags,
      RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS |
          RESOLVE_NO_XDEV));
  struct stat metadata {};
  if (::fstat(target.get(), &metadata) != 0 ||
      (require_directory ? !S_ISDIR(metadata.st_mode)
                         : !S_ISREG(metadata.st_mode)) ||
      (metadata.st_uid != 0U && metadata.st_uid != ::geteuid()) ||
      (metadata.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw HostLaunchResolutionError(
        "resolved host launch path has unsafe type, ownership, or mode");
  }
  if (!require_directory &&
      ((metadata.st_mode & S_IWGRP) != 0 || metadata.st_size <= 0 ||
       static_cast<std::uint64_t>(metadata.st_size) >
           kMaximumLaunchArtifactBytes)) {
    throw HostLaunchResolutionError(
        "launch artifact is writable, empty, or exceeds 256 MiB");
  }
  return {.descriptor = std::move(target), .metadata = metadata};
}

std::string digest_and_copy(int source, int destination,
                            std::uint64_t expected_size) {
  EVP_MD_CTX* raw_context = EVP_MD_CTX_new();
  if (raw_context == nullptr) {
    throw std::runtime_error("could not allocate SHA-256 context");
  }
  const auto free_context = [](EVP_MD_CTX* context) {
    EVP_MD_CTX_free(context);
  };
  std::unique_ptr<EVP_MD_CTX, decltype(free_context)> context(
      raw_context, free_context);
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("could not initialize SHA-256 context");
  }
  std::array<unsigned char, 1U << 16U> buffer{};
  std::uint64_t remaining = expected_size;
  while (remaining != 0U) {
    const std::size_t wanted = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, buffer.size()));
    const ssize_t count = ::read(source, buffer.data(), wanted);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      throw HostLaunchResolutionError(
          "could not read launch artifact: " +
          std::string(std::strerror(errno)));
    }
    if (count == 0) {
      throw HostLaunchResolutionError(
          "launch artifact ended before its validated size");
    }
    if (EVP_DigestUpdate(context.get(), buffer.data(),
                         static_cast<std::size_t>(count)) != 1) {
      throw std::runtime_error("could not hash launch artifact");
    }
    ssize_t written = 0;
    while (written < count) {
      const ssize_t result =
          ::write(destination, buffer.data() + written,
                  static_cast<std::size_t>(count - written));
      if (result < 0 && errno == EINTR) continue;
      if (result <= 0) {
        throw HostLaunchResolutionError(
            "could not seal launch artifact: " +
            std::string(std::strerror(errno)));
      }
      written += result;
    }
    remaining -= static_cast<std::uint64_t>(count);
  }
  unsigned char extra = 0U;
  ssize_t trailing = 0;
  do {
    trailing = ::read(source, &extra, 1U);
  } while (trailing < 0 && errno == EINTR);
  if (trailing != 0) {
    throw HostLaunchResolutionError(
        "launch artifact grew while it was being sealed");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1 ||
      digest_size != 32U) {
    throw std::runtime_error("could not finalize launch artifact hash");
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string output(64U, '0');
  for (std::size_t index = 0; index < digest_size; ++index) {
    output[index * 2U] = hex[digest[index] >> 4U];
    output[index * 2U + 1U] = hex[digest[index] & 0x0fU];
  }
  return "sha256:" + output;
}

struct SealedArtifact {
  Descriptor descriptor;
  VerifiedLaunchArtifact identity;
};

SealedArtifact resolve_artifact(
    const std::vector<std::string>& roots, const std::string& path,
    const std::string& expected_fingerprint, bool executable) {
  OpenedPath source = open_beneath(roots, path, O_RDONLY, false);
  if (executable && (source.metadata.st_mode & 0111) == 0) {
    throw HostLaunchResolutionError(
        "resolved worker executable has no execute bit");
  }
  if (executable) {
    std::array<unsigned char, 4U> magic{};
    const ssize_t count =
        ::pread(source.descriptor.get(), magic.data(), magic.size(), 0);
    if (count != static_cast<ssize_t>(magic.size()) ||
        magic != std::array<unsigned char, 4U>{0x7fU, 'E', 'L', 'F'}) {
      throw HostLaunchResolutionError(
          "resolved worker executable is not an ELF payload");
    }
  }
  const int raw_memfd = static_cast<int>(::syscall(
      SYS_memfd_create, executable ? "trainvm-executable" : "trainvm-code",
      MFD_CLOEXEC | MFD_ALLOW_SEALING |
          (executable ? MFD_EXEC : MFD_NOEXEC_SEAL)));
  if (raw_memfd < 0) {
    throw HostLaunchResolutionError(
        "could not create sealed launch artifact: " +
        std::string(std::strerror(errno)));
  }
  Descriptor sealed(raw_memfd);
  const std::string digest =
      digest_and_copy(source.descriptor.get(), sealed.get(),
                      static_cast<std::uint64_t>(source.metadata.st_size));
  struct stat after {};
  if (::fstat(source.descriptor.get(), &after) != 0 ||
      source.metadata.st_dev != after.st_dev ||
      source.metadata.st_ino != after.st_ino ||
      source.metadata.st_size != after.st_size ||
      source.metadata.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      source.metadata.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      source.metadata.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      source.metadata.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw HostLaunchResolutionError(
        "launch artifact changed while it was being sealed");
  }
  if (digest != expected_fingerprint) {
    throw HostLaunchResolutionError(
        "launch artifact bytes disagree with the trusted fingerprint");
  }
  const int seals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK |
                    (executable ? F_SEAL_EXEC : 0) | F_SEAL_SEAL;
  if (::fchmod(sealed.get(), executable ? 0500 : 0400) != 0 ||
      ::fcntl(sealed.get(), F_ADD_SEALS, seals) != 0 ||
      ::lseek(sealed.get(), 0, SEEK_SET) != 0) {
    throw HostLaunchResolutionError(
        "could not make launch artifact immutable");
  }
  return {
      .descriptor = std::move(sealed),
      .identity = {.source_path = path,
                   .source_device = static_cast<std::uint64_t>(source.metadata.st_dev),
                   .source_inode = static_cast<std::uint64_t>(source.metadata.st_ino),
                   .source_size = static_cast<std::uint64_t>(source.metadata.st_size),
                   .source_mode = static_cast<std::uint32_t>(source.metadata.st_mode),
                   .source_uid = static_cast<std::uint32_t>(source.metadata.st_uid),
                   .source_gid = static_cast<std::uint32_t>(source.metadata.st_gid),
                   .sealed_sha256 = digest},
  };
}

std::string bounded_system_identity(const char* path, std::size_t maximum) {
  Descriptor descriptor(::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (descriptor.get() < 0) {
    throw HostLaunchResolutionError("could not open host identity " +
                                    std::string(path));
  }
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 ||
      !S_ISREG(before.st_mode) || before.st_uid != 0U ||
      (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw HostLaunchResolutionError("host identity has unsafe metadata");
  }
  std::string value;
  value.reserve(maximum + 1U);
  std::array<char, 128U> buffer{};
  while (value.size() <= maximum) {
    const std::size_t wanted =
        std::min(buffer.size(), maximum + 1U - value.size());
    const ssize_t count = ::read(descriptor.get(), buffer.data(), wanted);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      throw HostLaunchResolutionError("could not read host identity");
    }
    if (count == 0) break;
    value.append(buffer.data(), static_cast<std::size_t>(count));
  }
  struct stat after {};
  if (value.size() > maximum ||
      ::fstat(descriptor.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw HostLaunchResolutionError(
        "host identity is unbounded or changed while being read");
  }
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  if (value.empty()) {
    throw HostLaunchResolutionError("host identity is empty");
  }
  return value;
}

nlohmann::json file_identity_json(const VerifiedLaunchArtifact& identity) {
  return {{"source_path", identity.source_path},
          {"source_device", identity.source_device},
          {"source_inode", identity.source_inode},
          {"source_size", identity.source_size},
          {"source_mode", identity.source_mode},
          {"source_uid", identity.source_uid},
          {"source_gid", identity.source_gid},
          {"sealed_sha256", identity.sealed_sha256}};
}

nlohmann::json directory_identity_json(
    const OpenedDirectoryIdentity& identity) {
  return {{"source_path", identity.source_path},
          {"device", identity.device},
          {"inode", identity.inode},
          {"mode", identity.mode},
          {"uid", identity.uid},
          {"gid", identity.gid}};
}

bool canonical_absolute_path(const std::string& value) {
  if (value.empty() || value.size() > 4096U) return false;
  const std::filesystem::path path(value);
  return path.is_absolute() && path.lexically_normal() == path;
}

bool sha256_digest(std::string_view value) {
  if (!value.starts_with("sha256:") || value.size() != 71U) return false;
  return std::ranges::all_of(value.substr(7U), [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

bool canonical_lower_hex(std::string_view value, std::size_t size) {
  return value.size() == size &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool canonical_uuid(std::string_view value) {
  if (value.size() != 36U || value[8U] != '-' || value[13U] != '-' ||
      value[18U] != '-' || value[23U] != '-') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == 8U || index == 13U || index == 18U || index == 23U) continue;
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

void validate_resolved_identity(const ResolvedLaunchIdentity& identity) {
  const AdapterKey& key = identity.adapter_key;
  if (identity.api_version != "trainvm.resolved-launch/v1" ||
      identity.run_id.empty() || identity.run_id.size() > 512U ||
      identity.node_id.empty() || identity.node_id.size() > 128U ||
      identity.attempt_id.empty() || identity.attempt_id.size() > 512U ||
      identity.launch_nonce.empty() || identity.launch_nonce.size() > 512U ||
      identity.launch_event_id != identity.run_id + ":worker-launch:" +
                                      identity.node_id + ":" +
                                      identity.attempt_id ||
      key.adapter.empty() || key.adapter.size() > 256U || key.version.empty() ||
      key.version.size() > 128U || key.operation.empty() ||
      key.operation.size() > 128U || key.contract.empty() ||
      key.contract.size() > 512U || key.runtime == ComponentRuntime::builtin ||
      key.runtime == ComponentRuntime::external_worker ||
      !sha256_digest(identity.code_fingerprint) ||
      !sha256_digest(identity.host_registry_digest) ||
      !sha256_digest(identity.host_profile_digest) ||
      !sha256_digest(identity.host.host_id) ||
      !canonical_uuid(identity.host.boot_id) ||
      identity.concurrency_key.empty() ||
      identity.concurrency_key.size() > 512U || identity.lease_id.empty() ||
      identity.lease_id.size() > 512U || identity.fencing_token == 0U) {
    throw std::invalid_argument(
        "resolved launch identity has malformed authority fields");
  }
  if (identity.required_capabilities.size() > 256U ||
      !std::ranges::is_sorted(identity.required_capabilities) ||
      std::ranges::adjacent_find(identity.required_capabilities) !=
          identity.required_capabilities.end() ||
      std::ranges::any_of(identity.required_capabilities,
                          [](const std::string& capability) {
                            return capability.empty() ||
                                   capability.size() > 256U;
                          })) {
    throw std::invalid_argument(
        "resolved launch capabilities are not canonical and bounded");
  }
  const auto validate_artifact = [](const VerifiedLaunchArtifact& artifact,
                                    bool executable) {
    if (!canonical_absolute_path(artifact.source_path) ||
        artifact.source_device == 0U || artifact.source_inode == 0U ||
        artifact.source_size == 0U ||
        artifact.source_size > kMaximumLaunchArtifactBytes ||
        (artifact.source_mode & S_IFMT) != S_IFREG ||
        (artifact.source_mode & (S_IWGRP | S_IWOTH)) != 0 ||
        (executable && (artifact.source_mode & 0111U) == 0U) ||
        !sha256_digest(artifact.sealed_sha256)) {
      throw std::invalid_argument(
          "resolved launch artifact identity is malformed");
    }
  };
  validate_artifact(identity.executable, true);
  if (key.runtime == ComponentRuntime::python_worker) {
    if (!identity.code) {
      throw std::invalid_argument(
          "resolved Python launch has no immutable code artifact");
    }
    validate_artifact(*identity.code, false);
    if (identity.code->sealed_sha256 != identity.code_fingerprint) {
      throw std::invalid_argument(
          "resolved Python code bytes disagree with the adapter fingerprint");
    }
  } else if (identity.code ||
             identity.executable.sealed_sha256 !=
                 identity.code_fingerprint) {
    throw std::invalid_argument(
        "resolved native code bytes disagree with the adapter fingerprint");
  }
  if (!canonical_absolute_path(identity.working_directory.source_path) ||
      identity.working_directory.device == 0U ||
      identity.working_directory.inode == 0U ||
      (identity.working_directory.mode & S_IFMT) != S_IFDIR ||
      (identity.working_directory.mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::invalid_argument(
        "resolved launch working-directory identity is malformed");
  }
  std::size_t argument_bytes = 0U;
  for (const std::string& argument : identity.public_arguments) {
    if (argument.empty() || argument.size() > 4096U ||
        argument.size() > 65'536U - argument_bytes ||
        argument.find('\0') != std::string::npos ||
        argument.contains("secret://") || argument.contains("${") ||
        argument.contains("{{")) {
      throw std::invalid_argument(
          "resolved launch arguments are not fixed public literals");
    }
    argument_bytes += argument.size();
  }
  if (identity.public_arguments.size() > 256U ||
      argument_bytes > 65'536U) {
    throw std::invalid_argument(
        "resolved launch arguments exceed their public manifest bounds");
  }
}

}  // namespace

nlohmann::json resolved_launch_identity_json(
    const ResolvedLaunchIdentity& identity) {
  nlohmann::json output{
      {"api_version", identity.api_version},
      {"launch_event_id", identity.launch_event_id},
      {"run_id", identity.run_id},
      {"node_id", identity.node_id},
      {"attempt_id", identity.attempt_id},
      {"launch_nonce", identity.launch_nonce},
      {"adapter_key",
       {{"adapter", identity.adapter_key.adapter},
        {"version", identity.adapter_key.version},
        {"runtime", enum_to_string(identity.adapter_key.runtime)},
        {"operation", identity.adapter_key.operation},
        {"contract", identity.adapter_key.contract}}},
      {"code_fingerprint", identity.code_fingerprint},
      {"required_capabilities", identity.required_capabilities},
      {"host_registry_digest", identity.host_registry_digest},
      {"host_profile_digest", identity.host_profile_digest},
      {"concurrency_key", identity.concurrency_key},
      {"lease_id", identity.lease_id},
      {"fencing_token", identity.fencing_token},
      {"host", {{"host_id", identity.host.host_id},
                {"boot_id", identity.host.boot_id}}},
      {"executable", file_identity_json(identity.executable)},
      {"public_arguments", identity.public_arguments},
      {"working_directory",
       directory_identity_json(identity.working_directory)},
  };
  if (identity.code) output["code"] = file_identity_json(*identity.code);
  return output;
}

nlohmann::json resolved_launch_spec_json(const ResolvedLaunchSpec& spec) {
  return {{"identity", resolved_launch_identity_json(spec.identity)},
          {"spec_digest", spec.spec_digest}};
}

ResolvedLaunchSpec resolved_launch_spec_from_json(
    const nlohmann::json& source) {
  ResolvedLaunchSpec spec;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, spec, "", diagnostics)) {
    throw std::invalid_argument(
        "resolved launch specification is malformed: " +
        diagnostics_json(diagnostics).dump());
  }
  validate_resolved_identity(spec.identity);
  const std::string expected =
      "sha256:" +
      sha256_hex(resolved_launch_identity_json(spec.identity).dump());
  if (spec.spec_digest != expected ||
      source != resolved_launch_spec_json(spec)) {
    throw std::invalid_argument(
        "resolved launch specification is not canonical or content-addressed");
  }
  return spec;
}

ResolvedLaunch::ResolvedLaunch(ResolvedLaunchSpec spec, int executable_fd,
                               std::optional<int> code_fd,
                               int working_directory_fd) noexcept
    : spec_(std::move(spec)), executable_fd_(executable_fd),
      code_fd_(code_fd), working_directory_fd_(working_directory_fd) {}

ResolvedLaunch::ResolvedLaunch(ResolvedLaunch&& other) noexcept
    : spec_(std::move(other.spec_)),
      executable_fd_(std::exchange(other.executable_fd_, -1)),
      code_fd_(std::exchange(other.code_fd_, std::nullopt)),
      working_directory_fd_(
          std::exchange(other.working_directory_fd_, -1)) {}

ResolvedLaunch& ResolvedLaunch::operator=(ResolvedLaunch&& other) noexcept {
  if (this != &other) {
    close_descriptors();
    spec_ = std::move(other.spec_);
    executable_fd_ = std::exchange(other.executable_fd_, -1);
    code_fd_ = std::exchange(other.code_fd_, std::nullopt);
    working_directory_fd_ =
        std::exchange(other.working_directory_fd_, -1);
  }
  return *this;
}

ResolvedLaunch::~ResolvedLaunch() {
  close_descriptors();
}

void ResolvedLaunch::close_descriptors() noexcept {
  if (executable_fd_ >= 0) (void)::close(executable_fd_);
  if (code_fd_ && *code_fd_ >= 0) (void)::close(*code_fd_);
  if (working_directory_fd_ >= 0) (void)::close(working_directory_fd_);
  executable_fd_ = -1;
  code_fd_.reset();
  working_directory_fd_ = -1;
}

const ResolvedLaunchSpec& ResolvedLaunch::spec() const { return spec_; }
int ResolvedLaunch::duplicate_executable_fd() const {
  const int duplicate = ::fcntl(executable_fd_, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0) {
    throw HostLaunchResolutionError(
        "could not duplicate sealed executable descriptor");
  }
  return duplicate;
}
std::optional<int> ResolvedLaunch::duplicate_code_fd() const {
  if (!code_fd_) return std::nullopt;
  const int duplicate = ::fcntl(*code_fd_, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0) {
    throw HostLaunchResolutionError(
        "could not duplicate sealed code descriptor");
  }
  return duplicate;
}
int ResolvedLaunch::duplicate_working_directory_fd() const {
  const int duplicate =
      ::fcntl(working_directory_fd_, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0) {
    throw HostLaunchResolutionError(
        "could not duplicate working-directory descriptor");
  }
  return duplicate;
}
HostLaunchResolver::HostLaunchResolver(const HostLaunchRegistry& registry,
                                       HostIdentity host)
    : registry_(registry), host_(std::move(host)) {
  if (!sha256_digest(host_.host_id) || !canonical_uuid(host_.boot_id)) {
    throw std::invalid_argument("host launch identity is malformed");
  }
}

HostIdentity HostLaunchResolver::local_host_identity() {
  const std::string machine_id =
      bounded_system_identity("/etc/machine-id", 128U);
  const std::string boot_id = bounded_system_identity(
      "/proc/sys/kernel/random/boot_id", 128U);
  if (!canonical_lower_hex(machine_id, 32U) || !canonical_uuid(boot_id)) {
    throw HostLaunchResolutionError(
        "Linux machine or boot identity is not canonical");
  }
  return {.host_id = "sha256:" + sha256_hex(machine_id),
          .boot_id = boot_id};
}

ResolvedLaunch HostLaunchResolver::resolve(
    const WorkerLaunchTicket& ticket, const AdapterKey& key) const {
  if (ticket.run_id.empty() || ticket.node_id.empty() ||
      ticket.attempt_id.empty() || ticket.launch_nonce.empty() ||
      ticket.adapter != key.adapter ||
      ticket.adapter_version != key.version || ticket.fencing_token == 0U) {
    throw HostLaunchResolutionError(
        "worker launch ticket disagrees with its host adapter key");
  }
  const HostLaunchProfile& profile =
      registry_.resolve(key, ticket.code_fingerprint);
  SealedArtifact executable = resolve_artifact(
      registry_.trusted_roots(), profile.executable_path,
      profile.executable_fingerprint, true);
  std::optional<SealedArtifact> code;
  if (profile.code_path) {
    code = resolve_artifact(registry_.trusted_roots(), *profile.code_path,
                            profile.code_fingerprint, false);
  }
  OpenedPath working_directory = open_beneath(
      registry_.trusted_roots(), profile.working_directory, O_PATH, true);
  ResolvedLaunchIdentity identity{
      .api_version = "trainvm.resolved-launch/v1",
      .launch_event_id = ticket.run_id + ":worker-launch:" + ticket.node_id +
                         ":" + ticket.attempt_id,
      .run_id = ticket.run_id,
      .node_id = ticket.node_id,
      .attempt_id = ticket.attempt_id,
      .launch_nonce = ticket.launch_nonce,
      .adapter_key = key,
      .code_fingerprint = ticket.code_fingerprint,
      .required_capabilities = ticket.required_capabilities,
      .host_registry_digest = registry_.registry_digest(),
      .host_profile_digest =
          registry_.profile_digest(key, ticket.code_fingerprint),
      .concurrency_key = ticket.concurrency_key,
      .lease_id = ticket.lease_id,
      .fencing_token = ticket.fencing_token,
      .host = host_,
      .executable = executable.identity,
      .code = code ? std::optional<VerifiedLaunchArtifact>(code->identity)
                   : std::nullopt,
      .public_arguments = profile.public_arguments,
      .working_directory =
          {.source_path = profile.working_directory,
           .device = static_cast<std::uint64_t>(
               working_directory.metadata.st_dev),
           .inode = static_cast<std::uint64_t>(
               working_directory.metadata.st_ino),
           .mode = static_cast<std::uint32_t>(
               working_directory.metadata.st_mode),
           .uid = static_cast<std::uint32_t>(
               working_directory.metadata.st_uid),
           .gid = static_cast<std::uint32_t>(
               working_directory.metadata.st_gid)},
  };
  const std::string spec_digest =
      "sha256:" + sha256_hex(resolved_launch_identity_json(identity).dump());
  ResolvedLaunchSpec spec{.identity = std::move(identity),
                          .spec_digest = spec_digest};
  const int executable_fd = executable.descriptor.release();
  const std::optional<int> code_fd =
      code ? std::optional<int>(code->descriptor.release()) : std::nullopt;
  const int working_directory_fd = working_directory.descriptor.release();
  return ResolvedLaunch(std::move(spec), executable_fd, code_fd,
                        working_directory_fd);
}

}  // namespace trainvm
