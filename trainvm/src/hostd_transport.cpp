#include "trainvm/hostd_transport.hpp"
#include "trainvm/document.hpp"

#include <fcntl.h>
#include <linux/openat2.h>
#include <openssl/sha.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <mutex>
#include <ranges>
#include <set>
#include <span>
#include <utility>

#include "trainvm/json.hpp"

namespace trainvm {
namespace {

constexpr std::array<std::byte, 4U> kWireMagic{
    std::byte{'T'}, std::byte{'V'}, std::byte{'H'}, std::byte{'D'}};
constexpr std::uint16_t kStatusRequestOpcode = 1U;
constexpr std::uint16_t kStatusResponseOpcode = 2U;
constexpr std::uint16_t kErrorResponseOpcode = 3U;
constexpr std::uint16_t kMutationOpenOpcode = 4U;
constexpr std::uint16_t kMutationChallengeOpcode = 5U;
constexpr std::uint16_t kMutationCommandOpcode = 6U;
constexpr std::uint16_t kMutationReplyOpcode = 7U;
constexpr std::size_t kMaximumSocketPathBytes = 107U;
constexpr std::size_t kMaximumSocketBasenameBytes = 96U;
constexpr std::size_t kMaximumControlFds = 8U;
constexpr std::size_t kMaximumIdentifierBytes = 192U;
constexpr std::size_t kMaximumPoisonReasonBytes = 512U;
constexpr std::size_t kMaximumReportedSessions = 65536U;

class FileDescriptor final {
public:
  explicit FileDescriptor(int value = -1) noexcept : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0)
      (void)::close(value_);
  }
  FileDescriptor(FileDescriptor &&other) noexcept
      : value_(std::exchange(other.value_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other) {
      if (value_ >= 0)
        (void)::close(value_);
      value_ = std::exchange(other.value_, -1);
    }
    return *this;
  }
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }
  [[nodiscard]] int release() noexcept { return std::exchange(value_, -1); }

private:
  int value_;
};

[[noreturn]] void throw_errno(std::string_view action) {
  const int error = errno;
  throw HostdTransportError(std::string(action) + ": " +
                            std::strerror(error));
}

bool safe_basename(std::string_view value) {
  if (value.empty() || value.size() > kMaximumSocketBasenameBytes ||
      value == "." || value == "..")
    return false;
  return std::ranges::all_of(value, [](char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '.' || character == '_' ||
           character == '-';
  });
}

FileDescriptor duplicate_cloexec(int descriptor) {
  const int duplicate = ::fcntl(descriptor, F_DUPFD_CLOEXEC, 3);
  if (duplicate < 0)
    throw_errno("could not duplicate hostd descriptor");
  return FileDescriptor(duplicate);
}

FileDescriptor open_parent_from_root(const std::filesystem::path &path) {
  FileDescriptor current(
      ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
  if (current.get() < 0)
    throw_errno("could not open filesystem root for hostd socket");
  for (const auto &part_path : path.relative_path()) {
    const std::string part = part_path.string();
    if (!safe_basename(part))
      throw HostdTransportError("hostd socket ancestry is noncanonical");
    struct open_how how{};
    how.flags = O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW;
    how.resolve =
        RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS;
    const int opened = static_cast<int>(::syscall(
        SYS_openat2, current.get(), part.c_str(), &how, sizeof(how)));
    if (opened < 0)
      throw_errno("could not securely open hostd socket ancestry");
    current = FileDescriptor(opened);
  }
  return current;
}

struct stat inspect_fd(int descriptor, std::string_view description) {
  struct stat status {};
  if (::fstat(descriptor, &status) != 0)
    throw_errno(std::string("could not inspect ") + std::string(description));
  return status;
}

int fstat_nointr(int descriptor, struct stat *status) noexcept {
  int result = -1;
  do {
    result = ::fstat(descriptor, status);
  } while (result != 0 && errno == EINTR);
  return result;
}

int fstatat_nointr(int parent, const char *basename, struct stat *status,
                   int flags) noexcept {
  int result = -1;
  do {
    result = ::fstatat(parent, basename, status, flags);
  } while (result != 0 && errno == EINTR);
  return result;
}

void validate_parent(const struct stat &status,
                     const HostdSocketAuthorityConfig &config) {
  const auto mode = static_cast<std::uint32_t>(status.st_mode) & 07777U;
  if (!S_ISDIR(status.st_mode) ||
      static_cast<std::uint32_t>(status.st_uid) != config.expected_owner_uid ||
      static_cast<std::uint32_t>(status.st_gid) != config.expected_owner_gid ||
      mode != config.expected_parent_mode || (mode & 0022U) != 0U) {
    throw HostdTransportError(
        "hostd socket parent is not the pinned trusted directory");
  }
}

HostdSocketIdentity socket_identity(const struct stat &parent,
                                    const struct stat &path) {
  return {.parent_device = static_cast<std::uint64_t>(parent.st_dev),
          .parent_inode = static_cast<std::uint64_t>(parent.st_ino),
          .parent_mode = static_cast<std::uint32_t>(parent.st_mode) & 07777U,
          .parent_owner_uid = static_cast<std::uint32_t>(parent.st_uid),
          .parent_owner_gid = static_cast<std::uint32_t>(parent.st_gid),
          .path_device = static_cast<std::uint64_t>(path.st_dev),
          .path_inode = static_cast<std::uint64_t>(path.st_ino),
          .path_mode = static_cast<std::uint32_t>(path.st_mode) & 07777U,
          .owner_uid = static_cast<std::uint32_t>(path.st_uid),
          .owner_gid = static_cast<std::uint32_t>(path.st_gid),
          .link_count = static_cast<std::uint64_t>(path.st_nlink)};
}

void validate_socket_path(const struct stat &status,
                          const HostdSocketAuthorityConfig &config) {
  const auto mode = static_cast<std::uint32_t>(status.st_mode) & 07777U;
  if (!S_ISSOCK(status.st_mode) || status.st_nlink != 1 ||
      static_cast<std::uint32_t>(status.st_uid) != config.expected_owner_uid ||
      static_cast<std::uint32_t>(status.st_gid) != config.expected_owner_gid ||
      mode != config.expected_socket_mode || (mode & 0007U) != 0U ||
      (mode & 0111U) != 0U) {
    throw HostdTransportError("hostd socket pathname policy is unsafe");
  }
}

template <typename ClientConfig>
HostdSocketIdentity inspect_client_endpoint(const ClientConfig &config) {
  if (!config.socket_path.is_absolute() || config.socket_path.empty() ||
      config.socket_path.lexically_normal() != config.socket_path ||
      !safe_basename(config.socket_path.filename().string()))
    throw HostdTransportError("hostd client endpoint path is noncanonical");
  auto parent = open_parent_from_root(config.socket_path.parent_path());
  const struct stat parent_status =
      inspect_fd(parent.get(), "hostd client endpoint parent");
  const std::string basename = config.socket_path.filename().string();
  const int opened = ::openat(parent.get(), basename.c_str(),
                              O_PATH | O_CLOEXEC | O_NOFOLLOW);
  if (opened < 0)
    throw_errno("could not pin hostd client endpoint");
  FileDescriptor path(opened);
  const struct stat path_status =
      inspect_fd(path.get(), "hostd client endpoint path");
  if (!S_ISSOCK(path_status.st_mode) || path_status.st_nlink != 1)
    throw HostdTransportError("hostd client endpoint is not a singleton socket");
  return socket_identity(parent_status, path_status);
}

sockaddr_un socket_address(const std::filesystem::path &path,
                           socklen_t &length) {
  const std::string native = path.string();
  if (native.size() > kMaximumSocketPathBytes || native.find('\0') !=
                                                        std::string::npos)
    throw HostdTransportError("hostd socket path does not fit sockaddr_un");
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, native.data(), native.size());
  address.sun_path[native.size()] = '\0';
  length = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                                  native.size() + 1U);
  return address;
}

void validate_listener_fd(int descriptor,
                          const HostdSocketAuthorityConfig &config) {
  const int descriptor_flags = ::fcntl(descriptor, F_GETFD);
  const int status_flags = ::fcntl(descriptor, F_GETFL);
  if (descriptor_flags < 0 || status_flags < 0)
    throw_errno("could not inspect hostd listener flags");
  if ((descriptor_flags & FD_CLOEXEC) == 0 ||
      (status_flags & O_NONBLOCK) == 0)
    throw HostdTransportError("hostd listener must be CLOEXEC and nonblocking");
  int domain = 0;
  int type = 0;
  int accepting = 0;
  socklen_t option_length = sizeof(int);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_DOMAIN, &domain,
                   &option_length) != 0 ||
      option_length != sizeof(int))
    throw_errno("could not inspect hostd listener domain");
  option_length = sizeof(int);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &type, &option_length) !=
          0 ||
      option_length != sizeof(int))
    throw_errno("could not inspect hostd listener type");
  option_length = sizeof(int);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_ACCEPTCONN, &accepting,
                   &option_length) != 0 ||
      option_length != sizeof(int))
    throw_errno("could not inspect hostd listener state");
  if (domain != AF_UNIX || type != SOCK_SEQPACKET || accepting != 1)
    throw HostdTransportError(
        "hostd listener has the wrong domain/type/state");

  sockaddr_un actual{};
  socklen_t actual_length = sizeof(actual);
  if (::getsockname(descriptor, reinterpret_cast<sockaddr *>(&actual),
                    &actual_length) != 0)
    throw_errno("could not inspect hostd listener address");
  socklen_t expected_length = 0U;
  const sockaddr_un expected =
      socket_address(config.socket_path.filename(), expected_length);
  if (actual_length != expected_length || actual.sun_family != AF_UNIX ||
      std::memcmp(actual.sun_path, expected.sun_path,
                  static_cast<std::size_t>(expected_length) -
                      offsetof(sockaddr_un, sun_path)) != 0)
    throw HostdTransportError("hostd listener address is inexact");
}

void put_u16(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint16_t value) {
  bytes.at(offset) = std::byte((value >> 8U) & 0xffU);
  bytes.at(offset + 1U) = std::byte(value & 0xffU);
}

void put_u32(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint32_t value) {
  for (std::size_t index = 0U; index < 4U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (24U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

void put_u64(std::vector<std::byte> &bytes, std::size_t offset,
             std::uint64_t value) {
  for (std::size_t index = 0U; index < 8U; ++index)
    bytes.at(offset + index) =
        std::byte((value >> (56U - static_cast<unsigned int>(index) * 8U)) &
                  0xffU);
}

std::uint16_t get_u16(const std::vector<std::byte> &bytes,
                      std::size_t offset) {
  return static_cast<std::uint16_t>(
      (std::to_integer<std::uint16_t>(bytes.at(offset)) << 8U) |
      std::to_integer<std::uint16_t>(bytes.at(offset + 1U)));
}

std::uint32_t get_u32(const std::vector<std::byte> &bytes,
                      std::size_t offset) {
  std::uint32_t result = 0U;
  for (std::size_t index = 0U; index < 4U; ++index)
    result = (result << 8U) |
             std::to_integer<std::uint32_t>(bytes.at(offset + index));
  return result;
}

std::uint64_t get_u64(const std::vector<std::byte> &bytes,
                      std::size_t offset) {
  std::uint64_t result = 0U;
  for (std::size_t index = 0U; index < 8U; ++index)
    result = (result << 8U) |
             std::to_integer<std::uint64_t>(bytes.at(offset + index));
  return result;
}

std::array<std::byte, SHA256_DIGEST_LENGTH>
digest(std::string_view payload) {
  std::array<std::byte, SHA256_DIGEST_LENGTH> result{};
  (void)::SHA256(reinterpret_cast<const unsigned char *>(payload.data()),
                 payload.size(),
                 reinterpret_cast<unsigned char *>(result.data()));
  return result;
}

struct WirePacket final {
  std::uint16_t opcode{};
  std::uint64_t correlation_id{};
  std::string payload;
};

std::vector<std::byte> encode_packet(std::uint16_t opcode,
                                     std::uint64_t correlation_id,
                                     const nlohmann::json &payload) {
  if (correlation_id == 0U)
    throw HostdTransportError("hostd correlation ID must be nonzero");
  const std::string canonical = payload.dump();
  if (canonical.size() > kHostdStatusMaximumPayloadBytes)
    throw HostdTransportError("hostd status payload is too large");
  std::vector<std::byte> result(kHostdStatusWireHeaderBytes + canonical.size());
  std::ranges::copy(kWireMagic, result.begin());
  put_u16(result, 4U, kHostdStatusWireVersion);
  put_u16(result, 6U,
          static_cast<std::uint16_t>(kHostdStatusWireHeaderBytes));
  put_u16(result, 8U, opcode);
  put_u16(result, 10U, 0U);
  put_u32(result, 12U, static_cast<std::uint32_t>(canonical.size()));
  put_u64(result, 16U, correlation_id);
  const auto payload_digest = digest(canonical);
  std::ranges::copy(payload_digest, result.begin() + 24);
  std::memcpy(result.data() + kHostdStatusWireHeaderBytes, canonical.data(),
              canonical.size());
  return result;
}

std::vector<std::byte> encode_canonical_packet(std::uint16_t opcode,
                                               std::uint64_t correlation_id,
                                               std::string_view payload) {
  try {
    const auto parsed = nlohmann::json::parse(payload);
    if (parsed.dump() != payload)
      throw HostdTransportError("hostd packet payload is not canonical JSON");
    return encode_packet(opcode, correlation_id, parsed);
  } catch (const HostdTransportError &) {
    throw;
  } catch (...) {
    throw HostdTransportError("hostd canonical packet encoding failed");
  }
}

WirePacket decode_packet(const std::vector<std::byte> &bytes,
                         std::size_t maximum_payload) {
  if (bytes.size() < kHostdStatusWireHeaderBytes)
    throw HostdTransportError("hostd packet header is truncated");
  if (!std::ranges::equal(kWireMagic, bytes | std::views::take(4)))
    throw HostdTransportError("hostd packet magic is invalid");
  if (get_u16(bytes, 4U) != kHostdStatusWireVersion)
    throw HostdTransportError("hostd packet version is unsupported");
  if (get_u16(bytes, 6U) != kHostdStatusWireHeaderBytes)
    throw HostdTransportError("hostd packet header length is invalid");
  if (get_u16(bytes, 10U) != 0U)
    throw HostdTransportError("hostd packet flags are unsupported");
  const std::size_t payload_size = get_u32(bytes, 12U);
  if (payload_size > maximum_payload ||
      bytes.size() != kHostdStatusWireHeaderBytes + payload_size)
    throw HostdTransportError("hostd packet payload length is inexact");
  const std::uint64_t correlation = get_u64(bytes, 16U);
  if (correlation == 0U)
    throw HostdTransportError("hostd packet correlation is invalid");
  const std::string payload(
      reinterpret_cast<const char *>(bytes.data() +
                                     kHostdStatusWireHeaderBytes),
      payload_size);
  const auto expected = digest(payload);
  if (!std::ranges::equal(
          expected, bytes | std::views::drop(24) |
                        std::views::take(SHA256_DIGEST_LENGTH)))
    throw HostdTransportError("hostd packet payload digest is invalid");
  return {.opcode = get_u16(bytes, 8U),
          .correlation_id = correlation,
          .payload = payload};
}

nlohmann::json parse_canonical_json(const std::string &payload) {
  try {
    const nlohmann::json value = nlohmann::json::parse(payload);
    if (value.dump() != payload)
      throw HostdTransportError("hostd JSON payload is not canonical");
    return value;
  } catch (const HostdTransportError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError("hostd JSON payload is malformed");
  }
}

void require_fields(const nlohmann::json &value,
                    std::initializer_list<std::string_view> expected) {
  if (!value.is_object() || value.size() != expected.size())
    throw HostdTransportError("hostd JSON fields are inexact");
  for (const std::string_view field : expected)
    if (!value.contains(std::string(field)))
      throw HostdTransportError("hostd JSON field is missing");
}

bool valid_identifier(std::string_view value) {
  if (value.empty() || value.size() > kMaximumIdentifierBytes)
    return false;
  return std::ranges::all_of(value, [](char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '.' || character == '_' ||
           character == '-' || character == ':' || character == '/' ||
           character == '@';
  });
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_bounded_text(std::string_view value, std::size_t maximum_bytes,
                        bool empty_allowed) {
  if ((!empty_allowed && value.empty()) || value.size() > maximum_bytes)
    return false;
  return std::ranges::all_of(value, [](unsigned char character) {
    return character == '\t' || character == '\n' ||
           (character >= 0x20U && character <= 0x7eU);
  });
}

std::string lifecycle_name(HostdLifecycle lifecycle) {
  switch (lifecycle) {
  case HostdLifecycle::sealed:
    return "sealed";
  case HostdLifecycle::startup_auditing:
    return "startup_auditing";
  case HostdLifecycle::startup_blocked:
    return "startup_blocked";
  case HostdLifecycle::admitting:
    return "admitting";
  case HostdLifecycle::poisoned:
    return "poisoned";
  }
  throw HostdTransportError("hostd lifecycle is invalid");
}

HostdLifecycle parse_lifecycle(const std::string &value) {
  if (value == "sealed")
    return HostdLifecycle::sealed;
  if (value == "startup_auditing")
    return HostdLifecycle::startup_auditing;
  if (value == "startup_blocked")
    return HostdLifecycle::startup_blocked;
  if (value == "admitting")
    return HostdLifecycle::admitting;
  if (value == "poisoned")
    return HostdLifecycle::poisoned;
  throw HostdTransportError("hostd lifecycle value is invalid");
}

nlohmann::json chain_head_json(const HostLedgerChainHead &head) {
  return {{"chain_hash", head.chain_hash},
          {"ledger_sequence", head.ledger_sequence}};
}

nlohmann::json audit_json(const HostStartupAuditReceipt &audit) {
  return {{"api_version", audit.api_version},
          {"audit_id", audit.audit_id},
          {"boot_id", audit.boot_id},
          {"broker_epoch", audit.broker_epoch},
          {"broker_instance_id", audit.broker_instance_id},
          {"commit_record_digest", audit.commit_record_digest},
          {"committed_boottime_ns", audit.committed_boottime_ns},
          {"committed_ledger_head",
           chain_head_json(audit.committed_ledger_head)},
          {"committed_wall_time_ns", audit.committed_wall_time_ns},
          {"disposition",
           audit.disposition == HostStartupAuditDisposition::passed
               ? "passed"
               : audit.disposition == HostStartupAuditDisposition::failed
                     ? "failed"
                     : throw HostdTransportError(
                           "startup audit disposition is invalid")},
          {"findings_digest", audit.findings_digest},
          {"host_id", audit.host_id},
          {"inventory_digest", audit.inventory_digest},
          {"ledger_head_before", chain_head_json(audit.ledger_head_before)},
          {"policy_digest", audit.policy_digest},
          {"post_occupancy_digest", audit.post_occupancy_digest},
          {"pre_occupancy_digest", audit.pre_occupancy_digest},
          {"receipt_digest", audit.receipt_digest},
          {"report_digest", audit.report_digest},
          {"topology_digest", audit.topology_digest}};
}

nlohmann::json status_json(const HostdCoordinatorStatus &status) {
  return {{"admission_counts_are_cached_evidence",
           status.admission_counts_are_cached_evidence},
          {"admission_sessions", status.admission_sessions},
          {"api_version", status.api_version},
          {"boot_id", status.boot_id},
          {"broker_epoch", status.broker_epoch},
          {"host_id", status.host_id},
          {"inventory_digest", status.inventory_digest},
          {"lifecycle", lifecycle_name(status.lifecycle)},
          {"live_sessions", status.live_sessions},
          {"poison_reason", status.poison_reason},
          {"release_only_sessions", status.release_only_sessions},
          {"stale_admission_sessions", status.stale_admission_sessions},
          {"startup_audit", status.startup_audit
                                ? audit_json(*status.startup_audit)
                                : nlohmann::json(nullptr)}};
}

std::string startup_phase_name(HostdStartupPhase phase) {
  switch (phase) {
  case HostdStartupPhase::reconciling:
    return "reconciling";
  case HostdStartupPhase::auditing:
    return "auditing";
  case HostdStartupPhase::admitting:
    return "admitting";
  case HostdStartupPhase::exhausted:
    return "exhausted";
  case HostdStartupPhase::failed:
    return "failed";
  }
  throw HostdTransportError("hostd startup phase is invalid");
}

std::string process_phase_name(HostdProcessAuthorityPhase phase) {
  switch (phase) {
  case HostdProcessAuthorityPhase::launch_intent:
    return "launch_intent";
  case HostdProcessAuthorityPhase::spawned:
    return "spawned";
  case HostdProcessAuthorityPhase::terminal_pending_release:
    return "terminal_pending_release";
  }
  throw HostdTransportError("hostd process authority phase is invalid");
}

std::string resource_kind_name(HostResourceKind kind) {
  switch (kind) {
  case HostResourceKind::accelerator:
    return "accelerator";
  case HostResourceKind::accelerator_partition:
    return "accelerator_partition";
  case HostResourceKind::host_mutex:
    return "host_mutex";
  }
  throw HostdTransportError("hostd resource kind is invalid");
}

std::optional<std::string> vendor_name(
    const std::optional<HostAcceleratorVendor> &vendor) {
  if (!vendor)
    return std::nullopt;
  switch (*vendor) {
  case HostAcceleratorVendor::nvidia:
    return "nvidia";
  case HostAcceleratorVendor::amd:
    return "amd";
  case HostAcceleratorVendor::intel:
    return "intel";
  case HostAcceleratorVendor::other:
    return "other";
  }
  throw HostdTransportError("hostd resource vendor is invalid");
}

nlohmann::json nullable_string_json(
    const std::optional<std::string> &value) {
  return value ? nlohmann::json(*value) : nlohmann::json(nullptr);
}

nlohmann::json nullable_vendor_json(
    const std::optional<HostAcceleratorVendor> &vendor) {
  return nullable_string_json(vendor_name(vendor));
}

nlohmann::json fence_status_json(const ResourceFence &fence) {
  return {
      {"generation", fence.generation},
      {"inventory_digest", fence.inventory_digest},
      {"resource",
       {{"kind", resource_kind_name(fence.resource.kind)},
        {"parent_id", nullable_string_json(fence.resource.parent_id)},
        {"stable_id", fence.resource.stable_id},
        {"vendor", nullable_vendor_json(fence.resource.vendor)}}},
      {"topology_digest", fence.topology_digest},
  };
}

nlohmann::json process_status_json(
    const HostdProcessAuthorityStatus &status) {
  return {
      {"accelerator_contexts_empty", status.accelerator_contexts_empty},
      {"allocation_id", status.allocation_id},
      {"cgroup_empty", status.cgroup_empty},
      {"cgroup_path", status.cgroup_path},
      {"context_audit_digest", status.context_audit_digest},
      {"device_policy_digest", status.device_policy_digest},
      {"device_policy_installation_digest",
       status.device_policy_installation_digest},
      {"device_policy_installed", status.device_policy_installed},
      {"device_policy_intended", status.device_policy_intended},
      {"host_pid", status.host_pid},
      {"journal_id", status.journal_id},
      {"launch_id", status.launch_id},
      {"logical_fencing_token", status.logical_fencing_token},
      {"logical_lease_id", status.logical_lease_id},
      {"phase", process_phase_name(status.phase)},
      {"process_policy_digest", status.process_policy_digest},
      {"process_policy_installation_digest",
       status.process_policy_installation_digest},
      {"process_policy_installed", status.process_policy_installed},
      {"process_policy_intended", status.process_policy_intended},
      {"process_starttime_ticks", status.process_starttime_ticks},
      {"run_id", status.run_id},
      {"terminal_receipt_digest", status.terminal_receipt_digest},
  };
}

nlohmann::json passive_memory_identity_json(
    const HostdAuthorityStatus &status) {
  nlohmann::json rows = nlohmann::json::array();
  for (const auto &memory : status.passive_accelerator_memory) {
    rows.push_back({
        {"resource_kind", resource_kind_name(memory.resource_kind)},
        {"vendor", nullable_vendor_json(memory.vendor)},
        {"stable_id", memory.stable_id},
        {"parent_id", nullable_string_json(memory.parent_id)},
        {"audited_eligible", memory.audited_eligible},
        {"total_memory_bytes", memory.total_memory_bytes},
        {"free_memory_bytes", memory.free_memory_bytes},
        {"selector_labels", memory.selector_labels},
    });
  }
  return {{"api_version", "trainvm.hostd-passive-memory/v1"},
          {"host_id", status.passive_memory_host_id},
          {"boot_id", status.passive_memory_boot_id},
          {"inventory_digest", status.passive_memory_inventory_digest},
          {"inventory_receipt_digest",
           status.passive_memory_inventory_receipt_digest},
          {"observed_monotonic_ns",
           status.passive_memory_observed_monotonic_ns},
          {"accelerator_count", status.passive_accelerator_memory_count},
          {"accelerators_truncated",
           status.passive_accelerator_memory_truncated},
          {"accelerators", std::move(rows)}};
}

std::string plain_sha256(std::string_view bytes) {
  return "sha256:" + sha256_hex(bytes);
}

nlohmann::json authority_status_json(const HostdAuthorityStatus &status) {
  nlohmann::json fences = nlohmann::json::array();
  for (const ResourceFence &fence : status.active_fences)
    fences.push_back(fence_status_json(fence));
  nlohmann::json processes = nlohmann::json::array();
  for (const HostdProcessAuthorityStatus &process : status.active_processes)
    processes.push_back(process_status_json(process));
  nlohmann::json passive_memory = nlohmann::json::array();
  for (const auto &memory : status.passive_accelerator_memory) {
    passive_memory.push_back({
        {"resource_kind", resource_kind_name(memory.resource_kind)},
        {"vendor", nullable_vendor_json(memory.vendor)},
        {"stable_id", memory.stable_id},
        {"parent_id", nullable_string_json(memory.parent_id)},
        {"audited_eligible", memory.audited_eligible},
        {"total_memory_bytes", memory.total_memory_bytes},
        {"free_memory_bytes", memory.free_memory_bytes},
        {"selector_labels", memory.selector_labels},
    });
  }
  return {
      {"active_fence_count", status.active_fence_count},
      {"active_fences", std::move(fences)},
      {"active_fences_truncated", status.active_fences_truncated},
      {"active_process_count", status.active_process_count},
      {"active_processes", std::move(processes)},
      {"active_processes_truncated", status.active_processes_truncated},
      {"api_version", status.api_version},
      {"ledger_chain_head", chain_head_json(status.ledger_chain_head)},
      {"ledger_record_count", status.ledger_record_count},
      {"ledger_verification_reason", status.ledger_verification_reason},
      {"ledger_verified", status.ledger_verified},
      {"mutation_disabled_reason", status.mutation_disabled_reason},
      {"mutation_enabled", status.mutation_enabled},
      {"current_inventory_digest", status.current_inventory_digest},
      {"current_inventory_receipt_digest",
       status.current_inventory_receipt_digest},
      {"passive_memory_host_id", status.passive_memory_host_id},
      {"passive_memory_boot_id", status.passive_memory_boot_id},
      {"passive_memory_inventory_digest",
       status.passive_memory_inventory_digest},
      {"passive_memory_inventory_receipt_digest",
       status.passive_memory_inventory_receipt_digest},
      {"passive_memory_observed_monotonic_ns",
       status.passive_memory_observed_monotonic_ns},
      {"passive_memory_observation_digest",
       status.passive_memory_observation_digest},
      {"passive_accelerator_memory_count",
       status.passive_accelerator_memory_count},
      {"passive_accelerator_memory", std::move(passive_memory)},
      {"passive_accelerator_memory_truncated",
       status.passive_accelerator_memory_truncated},
      {"degraded_resource_count", status.degraded_resource_count},
      {"occupancy_digest", status.occupancy_digest},
      {"occupancy_ledger_sequence", status.occupancy_ledger_sequence},
      {"process_launch_enabled", status.process_launch_enabled},
      {"resource_degradation_reason", status.resource_degradation_reason},
      {"resource_inventory_observation_age_ns",
       status.resource_inventory_observation_age_ns},
      {"resource_inventory_observed", status.resource_inventory_observed},
      {"remaining_terminal_release_records",
       status.remaining_terminal_release_records},
      {"remaining_unclosed_process_records",
       status.remaining_unclosed_process_records},
      {"startup_phase", startup_phase_name(status.startup_phase)},
      {"startup_recovery_steps", status.startup_recovery_steps},
  };
}

HostLedgerChainHead parse_chain_head(const nlohmann::json &value) {
  require_fields(value, {"chain_hash", "ledger_sequence"});
  const auto &sequence = value.at("ledger_sequence");
  if (!sequence.is_number_unsigned())
    throw HostdTransportError("startup audit ledger sequence is not unsigned");
  return {.ledger_sequence = sequence.get<std::uint64_t>(),
          .chain_hash = value.at("chain_hash").get<std::string>()};
}

std::int64_t parse_signed_integer(const nlohmann::json &value,
                                  std::string_view description) {
  if (!value.is_number_integer())
    throw HostdTransportError(std::string(description) +
                              " is not an exact integer");
  try {
    if (value.is_number_unsigned() &&
        value.get<std::uint64_t>() >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int64_t>::max()))
      throw HostdTransportError(std::string(description) +
                                " is outside the signed integer range");
    return value.get<std::int64_t>();
  } catch (const HostdTransportError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError(std::string(description) +
                              " is outside the signed integer range");
  }
}

std::size_t parse_size(const nlohmann::json &value,
                       std::string_view description) {
  if (!value.is_number_unsigned())
    throw HostdTransportError(std::string(description) +
                              " is not an exact unsigned integer");
  try {
    const std::uint64_t parsed = value.get<std::uint64_t>();
    if (parsed > static_cast<std::uint64_t>(
                     std::numeric_limits<std::size_t>::max()))
      throw HostdTransportError(std::string(description) +
                                " is outside the platform size range");
    return static_cast<std::size_t>(parsed);
  } catch (const HostdTransportError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError(std::string(description) +
                              " is outside the platform size range");
  }
}

std::uint64_t parse_unsigned_integer(const nlohmann::json &value,
                                     std::string_view description) {
  if (!value.is_number_unsigned())
    throw HostdTransportError(std::string(description) +
                              " is not an exact unsigned integer");
  try {
    return value.get<std::uint64_t>();
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError(std::string(description) +
                              " is outside the unsigned integer range");
  }
}

std::optional<std::string> parse_optional_string(
    const nlohmann::json &value, std::string_view description) {
  if (value.is_null())
    return std::nullopt;
  if (!value.is_string())
    throw HostdTransportError(std::string(description) +
                              " is not a string or null");
  return value.get<std::string>();
}

std::optional<bool> parse_optional_bool(const nlohmann::json &value,
                                        std::string_view description) {
  if (value.is_null())
    return std::nullopt;
  if (!value.is_boolean())
    throw HostdTransportError(std::string(description) +
                              " is not a boolean or null");
  return value.get<bool>();
}

std::optional<std::int64_t> parse_optional_signed(
    const nlohmann::json &value, std::string_view description) {
  if (value.is_null())
    return std::nullopt;
  return parse_signed_integer(value, description);
}

std::optional<std::uint64_t> parse_optional_unsigned(
    const nlohmann::json &value, std::string_view description) {
  if (value.is_null())
    return std::nullopt;
  return parse_unsigned_integer(value, description);
}

HostdStartupPhase parse_startup_phase(const std::string &value) {
  if (value == "reconciling")
    return HostdStartupPhase::reconciling;
  if (value == "auditing")
    return HostdStartupPhase::auditing;
  if (value == "admitting")
    return HostdStartupPhase::admitting;
  if (value == "exhausted")
    return HostdStartupPhase::exhausted;
  if (value == "failed")
    return HostdStartupPhase::failed;
  throw HostdTransportError("hostd startup phase is invalid");
}

HostdProcessAuthorityPhase parse_process_phase(const std::string &value) {
  if (value == "launch_intent")
    return HostdProcessAuthorityPhase::launch_intent;
  if (value == "spawned")
    return HostdProcessAuthorityPhase::spawned;
  if (value == "terminal_pending_release")
    return HostdProcessAuthorityPhase::terminal_pending_release;
  throw HostdTransportError("hostd process authority phase is invalid");
}

HostResourceKind parse_resource_kind(const std::string &value) {
  if (value == "accelerator")
    return HostResourceKind::accelerator;
  if (value == "accelerator_partition")
    return HostResourceKind::accelerator_partition;
  if (value == "host_mutex")
    return HostResourceKind::host_mutex;
  throw HostdTransportError("hostd resource kind is invalid");
}

std::optional<HostAcceleratorVendor>
parse_resource_vendor(const nlohmann::json &value) {
  const auto name = parse_optional_string(value, "resource vendor");
  if (!name)
    return std::nullopt;
  if (*name == "nvidia")
    return HostAcceleratorVendor::nvidia;
  if (*name == "amd")
    return HostAcceleratorVendor::amd;
  if (*name == "intel")
    return HostAcceleratorVendor::intel;
  if (*name == "other")
    return HostAcceleratorVendor::other;
  throw HostdTransportError("hostd resource vendor is invalid");
}

ResourceFence parse_fence_status(const nlohmann::json &value) {
  require_fields(value,
                 {"generation", "inventory_digest", "resource",
                  "topology_digest"});
  const auto &resource = value.at("resource");
  require_fields(resource, {"kind", "parent_id", "stable_id", "vendor"});
  return {
      .resource =
          {.kind = parse_resource_kind(
               resource.at("kind").get<std::string>()),
           .vendor = parse_resource_vendor(resource.at("vendor")),
           .stable_id = resource.at("stable_id").get<std::string>(),
           .parent_id =
               parse_optional_string(resource.at("parent_id"), "parent_id")},
      .generation =
          parse_unsigned_integer(value.at("generation"), "fence generation"),
      .inventory_digest =
          value.at("inventory_digest").get<std::string>(),
      .topology_digest = value.at("topology_digest").get<std::string>(),
  };
}

HostdProcessAuthorityStatus parse_process_status(
    const nlohmann::json &value) {
  require_fields(
      value,
      {"accelerator_contexts_empty", "allocation_id", "cgroup_empty",
       "cgroup_path", "context_audit_digest", "device_policy_digest",
       "device_policy_installation_digest", "device_policy_installed",
       "device_policy_intended", "host_pid", "journal_id", "launch_id",
       "logical_fencing_token", "logical_lease_id", "phase",
       "process_policy_digest", "process_policy_installation_digest",
       "process_policy_installed", "process_policy_intended",
       "process_starttime_ticks", "run_id", "terminal_receipt_digest"});
  return {
      .allocation_id = value.at("allocation_id").get<std::string>(),
      .journal_id = value.at("journal_id").get<std::string>(),
      .run_id = value.at("run_id").get<std::string>(),
      .logical_lease_id = value.at("logical_lease_id").get<std::string>(),
      .logical_fencing_token = parse_unsigned_integer(
          value.at("logical_fencing_token"), "logical fencing token"),
      .launch_id = value.at("launch_id").get<std::string>(),
      .phase = parse_process_phase(value.at("phase").get<std::string>()),
      .cgroup_path = value.at("cgroup_path").get<std::string>(),
      .host_pid = parse_optional_signed(value.at("host_pid"), "host_pid"),
      .process_starttime_ticks = parse_optional_unsigned(
          value.at("process_starttime_ticks"), "process_starttime_ticks"),
      .device_policy_intended =
          value.at("device_policy_intended").get<bool>(),
      .device_policy_installed =
          value.at("device_policy_installed").get<bool>(),
      .device_policy_digest =
          value.at("device_policy_digest").get<std::string>(),
      .device_policy_installation_digest =
          value.at("device_policy_installation_digest").get<std::string>(),
      .process_policy_intended =
          value.at("process_policy_intended").get<bool>(),
      .process_policy_installed =
          value.at("process_policy_installed").get<bool>(),
      .process_policy_digest =
          value.at("process_policy_digest").get<std::string>(),
      .process_policy_installation_digest =
          value.at("process_policy_installation_digest").get<std::string>(),
      .cgroup_empty =
          parse_optional_bool(value.at("cgroup_empty"), "cgroup_empty"),
      .accelerator_contexts_empty = parse_optional_bool(
          value.at("accelerator_contexts_empty"),
          "accelerator_contexts_empty"),
      .context_audit_digest =
          value.at("context_audit_digest").get<std::string>(),
      .terminal_receipt_digest =
          value.at("terminal_receipt_digest").get<std::string>(),
  };
}

HostdAuthorityStatus parse_authority_status(const nlohmann::json &value) {
  require_fields(
      value,
      {"active_fence_count", "active_fences", "active_fences_truncated",
       "active_process_count", "active_processes",
       "active_processes_truncated", "api_version", "ledger_chain_head",
       "ledger_record_count", "ledger_verification_reason",
       "ledger_verified", "mutation_disabled_reason", "mutation_enabled",
       "current_inventory_digest", "current_inventory_receipt_digest",
       "passive_memory_host_id", "passive_memory_boot_id",
       "passive_memory_inventory_digest",
       "passive_memory_inventory_receipt_digest",
       "passive_memory_observed_monotonic_ns",
       "passive_memory_observation_digest",
       "passive_accelerator_memory_count",
       "passive_accelerator_memory",
       "passive_accelerator_memory_truncated",
       "degraded_resource_count",
       "occupancy_digest", "occupancy_ledger_sequence",
       "process_launch_enabled", "resource_degradation_reason",
       "resource_inventory_observation_age_ns",
       "resource_inventory_observed", "remaining_terminal_release_records",
       "remaining_unclosed_process_records", "startup_phase",
       "startup_recovery_steps"});
  if (!value.at("active_fences").is_array() ||
      !value.at("active_processes").is_array() ||
      !value.at("passive_accelerator_memory").is_array())
    throw HostdTransportError("hostd authority rows are not arrays");
  HostdAuthorityStatus status{
      .api_version = value.at("api_version").get<std::string>(),
      .startup_phase =
          parse_startup_phase(value.at("startup_phase").get<std::string>()),
      .startup_recovery_steps = parse_size(
          value.at("startup_recovery_steps"), "startup_recovery_steps"),
      .remaining_unclosed_process_records = parse_size(
          value.at("remaining_unclosed_process_records"),
          "remaining_unclosed_process_records"),
      .remaining_terminal_release_records = parse_size(
          value.at("remaining_terminal_release_records"),
          "remaining_terminal_release_records"),
      .ledger_verified = value.at("ledger_verified").get<bool>(),
      .ledger_verification_reason =
          value.at("ledger_verification_reason").get<std::string>(),
      .ledger_chain_head = parse_chain_head(value.at("ledger_chain_head")),
      .ledger_record_count = parse_unsigned_integer(
          value.at("ledger_record_count"), "ledger_record_count"),
      .occupancy_ledger_sequence = parse_unsigned_integer(
          value.at("occupancy_ledger_sequence"),
          "occupancy_ledger_sequence"),
      .occupancy_digest = value.at("occupancy_digest").get<std::string>(),
      .resource_inventory_observed =
          value.at("resource_inventory_observed").get<bool>(),
      .resource_inventory_observation_age_ns = parse_unsigned_integer(
          value.at("resource_inventory_observation_age_ns"),
          "resource_inventory_observation_age_ns"),
      .current_inventory_digest =
          value.at("current_inventory_digest").get<std::string>(),
      .current_inventory_receipt_digest =
          value.at("current_inventory_receipt_digest").get<std::string>(),
      .passive_memory_host_id =
          value.at("passive_memory_host_id").get<std::string>(),
      .passive_memory_boot_id =
          value.at("passive_memory_boot_id").get<std::string>(),
      .passive_memory_inventory_digest =
          value.at("passive_memory_inventory_digest").get<std::string>(),
      .passive_memory_inventory_receipt_digest =
          value.at("passive_memory_inventory_receipt_digest")
              .get<std::string>(),
      .passive_memory_observed_monotonic_ns = parse_unsigned_integer(
          value.at("passive_memory_observed_monotonic_ns"),
          "passive_memory_observed_monotonic_ns"),
      .passive_memory_observation_digest =
          value.at("passive_memory_observation_digest").get<std::string>(),
      .passive_accelerator_memory_count = parse_size(
          value.at("passive_accelerator_memory_count"),
          "passive_accelerator_memory_count"),
      .passive_accelerator_memory = {},
      .passive_accelerator_memory_truncated =
          value.at("passive_accelerator_memory_truncated").get<bool>(),
      .degraded_resource_count = parse_size(
          value.at("degraded_resource_count"), "degraded_resource_count"),
      .resource_degradation_reason =
          value.at("resource_degradation_reason").get<std::string>(),
      .active_fence_count =
          parse_size(value.at("active_fence_count"), "active_fence_count"),
      .active_fences = {},
      .active_fences_truncated =
          value.at("active_fences_truncated").get<bool>(),
      .active_process_count = parse_size(value.at("active_process_count"),
                                         "active_process_count"),
      .active_processes = {},
      .active_processes_truncated =
          value.at("active_processes_truncated").get<bool>(),
      .process_launch_enabled =
          value.at("process_launch_enabled").get<bool>(),
      .mutation_enabled = value.at("mutation_enabled").get<bool>(),
      .mutation_disabled_reason =
          value.at("mutation_disabled_reason").get<std::string>(),
  };
  for (const auto &fence : value.at("active_fences"))
    status.active_fences.push_back(parse_fence_status(fence));
  for (const auto &process : value.at("active_processes"))
    status.active_processes.push_back(parse_process_status(process));
  for (const auto &memory : value.at("passive_accelerator_memory")) {
    require_fields(memory, {"audited_eligible", "free_memory_bytes",
                            "parent_id", "resource_kind", "selector_labels",
                            "stable_id", "total_memory_bytes", "vendor"});
    const auto vendor = parse_resource_vendor(memory.at("vendor"));
    if (!vendor)
      throw HostdTransportError(
          "passive accelerator memory vendor is absent");
    status.passive_accelerator_memory.push_back({
        .resource_kind = parse_resource_kind(
            memory.at("resource_kind").get<std::string>()),
        .vendor = *vendor,
        .stable_id = memory.at("stable_id").get<std::string>(),
        .parent_id = parse_optional_string(memory.at("parent_id"),
                                           "passive memory parent_id"),
        .audited_eligible = memory.at("audited_eligible").get<bool>(),
        .total_memory_bytes = parse_unsigned_integer(
            memory.at("total_memory_bytes"), "total_memory_bytes"),
        .free_memory_bytes = parse_unsigned_integer(
            memory.at("free_memory_bytes"), "free_memory_bytes"),
        .selector_labels =
            memory.at("selector_labels")
                .get<std::map<std::string, std::string>>(),
    });
  }
  return status;
}

HostStartupAuditDisposition parse_audit_disposition(const std::string &value) {
  if (value == "passed")
    return HostStartupAuditDisposition::passed;
  if (value == "failed")
    return HostStartupAuditDisposition::failed;
  throw HostdTransportError("startup audit disposition is invalid");
}

HostStartupAuditReceipt parse_audit(const nlohmann::json &value) {
  require_fields(value,
                 {"api_version", "audit_id", "boot_id", "broker_epoch",
                  "broker_instance_id", "commit_record_digest",
                  "committed_boottime_ns", "committed_ledger_head",
                  "committed_wall_time_ns", "disposition", "findings_digest",
                  "host_id", "inventory_digest", "ledger_head_before",
                  "policy_digest", "post_occupancy_digest",
                  "pre_occupancy_digest", "receipt_digest", "report_digest",
                  "topology_digest"});
  return {.api_version = value.at("api_version").get<std::string>(),
          .audit_id = value.at("audit_id").get<std::string>(),
          .report_digest = value.at("report_digest").get<std::string>(),
          .host_id = value.at("host_id").get<std::string>(),
          .boot_id = value.at("boot_id").get<std::string>(),
          .broker_epoch = value.at("broker_epoch").get<std::string>(),
          .broker_instance_id =
              value.at("broker_instance_id").get<std::string>(),
          .inventory_digest =
              value.at("inventory_digest").get<std::string>(),
          .topology_digest = value.at("topology_digest").get<std::string>(),
          .pre_occupancy_digest =
              value.at("pre_occupancy_digest").get<std::string>(),
          .post_occupancy_digest =
              value.at("post_occupancy_digest").get<std::string>(),
          .policy_digest = value.at("policy_digest").get<std::string>(),
          .findings_digest = value.at("findings_digest").get<std::string>(),
          .disposition = parse_audit_disposition(
              value.at("disposition").get<std::string>()),
          .ledger_head_before = parse_chain_head(value.at("ledger_head_before")),
          .committed_ledger_head =
              parse_chain_head(value.at("committed_ledger_head")),
          .commit_record_digest =
              value.at("commit_record_digest").get<std::string>(),
          .committed_boottime_ns = parse_signed_integer(
              value.at("committed_boottime_ns"), "committed_boottime_ns"),
          .committed_wall_time_ns = parse_signed_integer(
              value.at("committed_wall_time_ns"), "committed_wall_time_ns"),
          .receipt_digest = value.at("receipt_digest").get<std::string>()};
}

HostdCoordinatorStatus parse_status(const nlohmann::json &value) {
  require_fields(value,
                 {"admission_counts_are_cached_evidence",
                  "admission_sessions", "api_version", "boot_id",
                  "broker_epoch", "host_id", "inventory_digest",
                  "lifecycle", "live_sessions", "poison_reason",
                  "release_only_sessions", "stale_admission_sessions",
                  "startup_audit"});
  std::optional<HostStartupAuditReceipt> audit;
  if (!value.at("startup_audit").is_null())
    audit = parse_audit(value.at("startup_audit"));
  return {
      .api_version = value.at("api_version").get<std::string>(),
      .lifecycle =
          parse_lifecycle(value.at("lifecycle").get<std::string>()),
      .host_id = value.at("host_id").get<std::string>(),
      .boot_id = value.at("boot_id").get<std::string>(),
      .broker_epoch = value.at("broker_epoch").get<std::string>(),
      .inventory_digest = value.at("inventory_digest").get<std::string>(),
      .live_sessions = parse_size(value.at("live_sessions"), "live_sessions"),
      .admission_sessions =
          parse_size(value.at("admission_sessions"), "admission_sessions"),
      .stale_admission_sessions = parse_size(
          value.at("stale_admission_sessions"), "stale_admission_sessions"),
      .release_only_sessions = parse_size(value.at("release_only_sessions"),
                                          "release_only_sessions"),
      .admission_counts_are_cached_evidence =
          value.at("admission_counts_are_cached_evidence").get<bool>(),
      .startup_audit = std::move(audit),
      .poison_reason = value.at("poison_reason").get<std::string>()};
}

bool valid_audit_identifier(std::string_view value) {
  return !value.empty() &&
         value.size() <= HostStartupAuditBounds::maximum_identifier_bytes &&
         std::ranges::all_of(value, [](unsigned char character) {
           return character >= 0x20U && character <= 0x7eU;
         });
}

void validate_chain_head(const HostLedgerChainHead &head) {
  if (head.ledger_sequence >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      !valid_digest(head.chain_hash))
    throw HostdTransportError("startup audit ledger head is invalid");
}

std::string startup_audit_receipt_digest(
    const HostStartupAuditReceipt &audit) {
  nlohmann::json value = audit_json(audit);
  value.erase("receipt_digest");
  std::string bytes = "trainvm.host-startup-audit-receipt/v2";
  bytes.push_back('\0');
  bytes.append(value.dump());
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  (void)::SHA256(reinterpret_cast<const unsigned char *>(bytes.data()),
                 bytes.size(), digest.data());
  static constexpr char digits[] = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(7U + digest.size() * 2U);
  for (const unsigned char byte : digest) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

void validate_audit_semantics(const HostStartupAuditReceipt &audit,
                              const HostdCoordinatorStatus &status) {
  validate_chain_head(audit.ledger_head_before);
  validate_chain_head(audit.committed_ledger_head);
  const bool valid_disposition =
      audit.disposition == HostStartupAuditDisposition::passed ||
      audit.disposition == HostStartupAuditDisposition::failed;
  if (audit.api_version != kHostStartupAuditReceiptApiVersion ||
      !valid_audit_identifier(audit.audit_id) ||
      !valid_audit_identifier(audit.host_id) ||
      !valid_audit_identifier(audit.boot_id) ||
      !valid_audit_identifier(audit.broker_epoch) ||
      !valid_audit_identifier(audit.broker_instance_id) ||
      !valid_digest(audit.report_digest) ||
      !valid_digest(audit.inventory_digest) ||
      !valid_digest(audit.topology_digest) ||
      !valid_digest(audit.pre_occupancy_digest) ||
      !valid_digest(audit.post_occupancy_digest) ||
      !valid_digest(audit.policy_digest) ||
      !valid_digest(audit.findings_digest) ||
      !valid_digest(audit.commit_record_digest) ||
      !valid_digest(audit.receipt_digest) || !valid_disposition ||
      audit.host_id != status.host_id || audit.boot_id != status.boot_id ||
      audit.broker_epoch != status.broker_epoch ||
      audit.inventory_digest != status.inventory_digest ||
      audit.ledger_head_before.ledger_sequence ==
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      audit.committed_ledger_head.ledger_sequence !=
          audit.ledger_head_before.ledger_sequence + 1U ||
      audit.committed_boottime_ns < 0 || audit.committed_wall_time_ns < 0 ||
      audit.receipt_digest != startup_audit_receipt_digest(audit))
    throw HostdTransportError("hostd startup audit semantics are invalid");
}

void validate_status_semantics(const HostdCoordinatorStatus &status) {
  if (status.api_version != kHostdCoordinatorApiVersion ||
      !valid_identifier(status.host_id) || !valid_identifier(status.boot_id) ||
      !valid_identifier(status.broker_epoch) ||
      !valid_digest(status.inventory_digest) ||
      status.live_sessions > kMaximumReportedSessions ||
      status.admission_sessions > status.live_sessions ||
      status.stale_admission_sessions > status.live_sessions ||
      status.release_only_sessions > status.live_sessions ||
      status.admission_sessions >
          status.live_sessions - status.stale_admission_sessions ||
      status.release_only_sessions >
          status.live_sessions - status.admission_sessions -
              status.stale_admission_sessions ||
      !status.admission_counts_are_cached_evidence ||
      !valid_bounded_text(status.poison_reason, kMaximumPoisonReasonBytes,
                          true))
    throw HostdTransportError("hostd coordinator status semantics are invalid");
  if (status.startup_audit)
    validate_audit_semantics(*status.startup_audit, status);
  switch (status.lifecycle) {
  case HostdLifecycle::sealed:
  case HostdLifecycle::startup_auditing:
    if (status.startup_audit || !status.poison_reason.empty())
      throw HostdTransportError("hostd pre-audit lifecycle is contradictory");
    return;
  case HostdLifecycle::admitting:
    if (!status.startup_audit || !status.poison_reason.empty() ||
        status.startup_audit->disposition !=
            HostStartupAuditDisposition::passed)
      throw HostdTransportError("hostd admitting lifecycle is contradictory");
    return;
  case HostdLifecycle::startup_blocked:
    if (!status.startup_audit || status.poison_reason.empty())
      throw HostdTransportError(
          "hostd startup-blocked lifecycle is contradictory");
    return;
  case HostdLifecycle::poisoned:
    if (status.poison_reason.empty())
      throw HostdTransportError("hostd poisoned lifecycle lacks a reason");
    return;
  }
  throw HostdTransportError("hostd coordinator lifecycle is invalid");
}

void validate_fence_status(const ResourceFence &fence) {
  const HostResourceId &resource = fence.resource;
  const bool accelerator = resource.kind == HostResourceKind::accelerator;
  const bool partition =
      resource.kind == HostResourceKind::accelerator_partition;
  const bool mutex = resource.kind == HostResourceKind::host_mutex;
  if ((!accelerator && !partition && !mutex) ||
      !valid_identifier(resource.stable_id) || fence.generation == 0U ||
      !valid_digest(fence.inventory_digest) ||
      !valid_digest(fence.topology_digest) ||
      (accelerator && (!resource.vendor || resource.parent_id)) ||
      (partition && (!resource.vendor || !resource.parent_id ||
                     !valid_identifier(*resource.parent_id) ||
                     *resource.parent_id == resource.stable_id)) ||
      (mutex && (resource.vendor || resource.parent_id ||
                 !resource.stable_id.starts_with("host-mutex:")))) {
    throw HostdTransportError("hostd authority resource fence is invalid");
  }
}

bool valid_status_path(std::string_view path) {
  return !path.empty() && path.size() <= 4'096U && path.front() == '/' &&
         path.find('\0') == std::string_view::npos &&
         valid_bounded_text(path, 4'096U, false);
}

void validate_process_status(const HostdProcessAuthorityStatus &status) {
  const bool has_process = status.host_pid.has_value() &&
                           status.process_starttime_ticks.has_value();
  const bool terminal = status.phase ==
                        HostdProcessAuthorityPhase::terminal_pending_release;
  const bool spawned = status.phase == HostdProcessAuthorityPhase::spawned;
  const bool intent = status.phase == HostdProcessAuthorityPhase::launch_intent;
  if ((!intent && !spawned && !terminal) ||
      !valid_identifier(status.allocation_id) ||
      !valid_identifier(status.journal_id) || !valid_identifier(status.run_id) ||
      !valid_identifier(status.logical_lease_id) ||
      status.logical_fencing_token == 0U ||
      !valid_identifier(status.launch_id) ||
      !valid_status_path(status.cgroup_path) ||
      (status.host_pid.has_value() !=
       status.process_starttime_ticks.has_value()) ||
      (has_process && (*status.host_pid <= 0 ||
                       *status.process_starttime_ticks == 0U)) ||
      (intent && has_process) || (!intent && !has_process) ||
      (status.device_policy_intended !=
       !status.device_policy_digest.empty()) ||
      (status.device_policy_installed !=
       !status.device_policy_installation_digest.empty()) ||
      (status.process_policy_intended !=
       !status.process_policy_digest.empty()) ||
      (status.process_policy_installed !=
       !status.process_policy_installation_digest.empty()) ||
      (status.device_policy_installed &&
       !status.device_policy_intended) ||
      (status.process_policy_installed &&
       !status.process_policy_intended) ||
      (!status.device_policy_digest.empty() &&
       !valid_digest(status.device_policy_digest)) ||
      (!status.device_policy_installation_digest.empty() &&
       !valid_digest(status.device_policy_installation_digest)) ||
      (!status.process_policy_digest.empty() &&
       !valid_digest(status.process_policy_digest)) ||
      (!status.process_policy_installation_digest.empty() &&
       !valid_digest(status.process_policy_installation_digest)) ||
      (intent &&
       (status.device_policy_installed || status.process_policy_installed)) ||
      (terminal != status.cgroup_empty.has_value()) ||
      (terminal != status.accelerator_contexts_empty.has_value()) ||
      (terminal != !status.context_audit_digest.empty()) ||
      (terminal != !status.terminal_receipt_digest.empty()) ||
      (terminal && (!valid_digest(status.context_audit_digest) ||
                    !valid_digest(status.terminal_receipt_digest)))) {
    throw HostdTransportError("hostd process authority status is invalid");
  }
}

void validate_authority_status_semantics(
    const HostdAuthorityStatus &status,
    const HostdCoordinatorStatus &coordinator) {
  validate_chain_head(status.ledger_chain_head);
  if (status.api_version != kHostdAuthorityStatusApiVersion ||
      status.startup_recovery_steps > 1'000'000U ||
      status.remaining_unclosed_process_records >
          HostResourceBounds::maximum_active_fences ||
      status.remaining_terminal_release_records >
          HostResourceBounds::maximum_active_fences ||
      status.active_fence_count >
          HostResourceBounds::maximum_active_fences ||
      status.active_process_count >
          HostResourceBounds::maximum_active_fences ||
      status.active_fences.size() >
          HostdAuthorityStatus::maximum_reported_rows ||
      status.active_processes.size() >
          HostdAuthorityStatus::maximum_reported_rows ||
      status.passive_accelerator_memory_count >
          HostResourceBounds::maximum_resources ||
      status.passive_accelerator_memory.size() >
          HostdAuthorityStatus::maximum_passive_memory_rows ||
      status.passive_accelerator_memory.size() >
          status.passive_accelerator_memory_count ||
      status.passive_accelerator_memory_truncated !=
          (status.passive_accelerator_memory.size() <
           status.passive_accelerator_memory_count) ||
      (!status.passive_accelerator_memory.empty() &&
       passive_memory_identity_json(status).dump().size() >
           HostdAuthorityStatus::maximum_passive_memory_identity_bytes) ||
      status.active_fences.size() > status.active_fence_count ||
      status.active_processes.size() > status.active_process_count ||
      status.active_fences_truncated !=
          (status.active_fences.size() < status.active_fence_count) ||
      status.active_processes_truncated !=
          (status.active_processes.size() < status.active_process_count) ||
      status.active_process_count !=
          status.remaining_unclosed_process_records +
              status.remaining_terminal_release_records ||
      status.occupancy_ledger_sequence >
          status.ledger_chain_head.ledger_sequence ||
      !valid_digest(status.occupancy_digest) ||
      status.degraded_resource_count > status.active_fence_count ||
      (status.resource_inventory_observed !=
       (!status.current_inventory_digest.empty() &&
        !status.current_inventory_receipt_digest.empty())) ||
      (!status.current_inventory_digest.empty() &&
       !valid_digest(status.current_inventory_digest)) ||
      (!status.current_inventory_receipt_digest.empty() &&
       !valid_digest(status.current_inventory_receipt_digest)) ||
      (status.resource_degradation_reason.empty() !=
       (status.resource_inventory_observed &&
        status.degraded_resource_count == 0U)) ||
      !valid_bounded_text(status.resource_degradation_reason,
                          kMaximumPoisonReasonBytes, true) ||
      !valid_bounded_text(status.ledger_verification_reason,
                          kMaximumPoisonReasonBytes, true) ||
      !valid_bounded_text(status.mutation_disabled_reason,
                          kMaximumPoisonReasonBytes, true) ||
      (status.ledger_verified !=
       status.ledger_verification_reason.empty()) ||
      (status.mutation_enabled ==
       !status.mutation_disabled_reason.empty()) ||
      (status.mutation_enabled &&
       (!status.ledger_verified || !status.process_launch_enabled ||
        status.startup_phase != HostdStartupPhase::admitting ||
        coordinator.lifecycle != HostdLifecycle::admitting))) {
    throw HostdTransportError("hostd authority status semantics are invalid");
  }
  std::set<std::string> passive_ids;
  for (const auto &memory : status.passive_accelerator_memory) {
    if ((memory.resource_kind != HostResourceKind::accelerator &&
         memory.resource_kind != HostResourceKind::accelerator_partition) ||
        !valid_identifier(memory.stable_id) ||
        (memory.parent_id && !valid_identifier(*memory.parent_id)) ||
        memory.total_memory_bytes == 0U ||
        memory.free_memory_bytes > memory.total_memory_bytes ||
        memory.selector_labels.size() >
            HostResourceBounds::maximum_labels_per_resource ||
        !std::ranges::all_of(memory.selector_labels, [](const auto &label) {
          return valid_identifier(label.first) &&
                 valid_bounded_text(label.second, 512U, false);
        }) ||
        !passive_ids.insert(memory.stable_id).second ||
        !status.resource_inventory_observed)
      throw HostdTransportError(
          "hostd passive accelerator memory semantics are invalid");
  }
  for (const auto &memory : status.passive_accelerator_memory) {
    if (!memory.audited_eligible || !memory.parent_id)
      continue;
    const auto parent = std::ranges::find(
        status.passive_accelerator_memory, *memory.parent_id,
        &HostdPassiveAcceleratorMemory::stable_id);
    if (parent != status.passive_accelerator_memory.end() &&
        parent->audited_eligible)
      throw HostdTransportError(
          "hostd passive memory cannot make parent and child both eligible");
  }
  const bool has_passive =
      !status.passive_memory_observation_digest.empty() ||
      status.passive_accelerator_memory_count != 0U ||
      !status.passive_accelerator_memory.empty() ||
      status.passive_accelerator_memory_truncated ||
      status.passive_memory_observed_monotonic_ns != 0U;
  if (has_passive &&
      (status.passive_memory_host_id != coordinator.host_id ||
       status.passive_memory_boot_id != coordinator.boot_id ||
       status.passive_memory_inventory_digest !=
           status.current_inventory_digest ||
       status.passive_memory_inventory_receipt_digest !=
           status.current_inventory_receipt_digest))
    throw HostdTransportError(
        "hostd passive memory is not bound to host/boot/inventory identity");
  if (has_passive &&
      (status.passive_memory_observed_monotonic_ns == 0U ||
       status.passive_accelerator_memory_count == 0U ||
       (status.passive_accelerator_memory.empty() &&
        !status.passive_accelerator_memory_truncated)))
    throw HostdTransportError(
        "hostd passive memory observation is structurally incomplete");
  if (has_passive) {
    const std::string expected_passive_digest =
        plain_sha256(passive_memory_identity_json(status).dump());
    if (status.passive_memory_observation_digest != expected_passive_digest)
      throw HostdTransportError(
          "hostd passive memory observation digest does not match its "
          "evidence");
  }
  if (!has_passive &&
      (!status.passive_memory_host_id.empty() ||
       !status.passive_memory_boot_id.empty() ||
       !status.passive_memory_inventory_digest.empty() ||
       !status.passive_memory_inventory_receipt_digest.empty()))
    throw HostdTransportError(
        "hostd empty passive memory observation carries partial identity");
  std::set<std::string> resource_ids;
  for (const ResourceFence &fence : status.active_fences) {
    validate_fence_status(fence);
    if (!resource_ids.insert(fence.resource.stable_id).second)
      throw HostdTransportError(
          "hostd authority status repeats a resource fence");
  }
  std::set<std::string> launch_ids;
  for (const HostdProcessAuthorityStatus &process : status.active_processes) {
    validate_process_status(process);
    if (!launch_ids.insert(process.launch_id).second)
      throw HostdTransportError(
          "hostd authority status repeats a process launch");
  }
}

HostdCoordinatorStatus parse_validated_status(const nlohmann::json &value) {
  try {
    HostdCoordinatorStatus status = parse_status(value);
    validate_status_semantics(status);
    return status;
  } catch (const HostdTransportError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError("hostd status JSON types are invalid");
  }
}

HostdAuthorityStatus parse_validated_authority_status(
    const nlohmann::json &value, const HostdCoordinatorStatus &coordinator) {
  try {
    HostdAuthorityStatus status = parse_authority_status(value);
    validate_authority_status_semantics(status, coordinator);
    return status;
  } catch (const HostdTransportError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    throw HostdTransportError("hostd authority status JSON types are invalid");
  }
}

bool wait_ready(int descriptor, short events,
                std::int64_t absolute_deadline_ns) {
  for (;;) {
    const std::int64_t now = hostd_monotonic_now_ns();
    if (absolute_deadline_ns <= now)
      return false;
    const std::int64_t remaining = absolute_deadline_ns - now;
    timespec timeout{.tv_sec = remaining / 1'000'000'000LL,
                     .tv_nsec = remaining % 1'000'000'000LL};
    pollfd poll_descriptor{.fd = descriptor, .events = events, .revents = 0};
    const int result = ::ppoll(&poll_descriptor, 1U, &timeout, nullptr);
    if (result > 0) {
      // A final seqpacket remains readable when the peer sends and immediately
      // closes, so requested readiness wins over co-reported POLLHUP.
      if ((poll_descriptor.revents & events) != 0)
        return true;
      if ((poll_descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0)
        return false;
      continue;
    }
    if (result == 0)
      return false;
    if (errno != EINTR)
      throw_errno("hostd transport poll failed");
  }
}

struct ReceivedPacket final {
  std::vector<std::byte> bytes;
  ucred credentials{};
  std::vector<FileDescriptor> descriptors;
};

struct alignas(cmsghdr) ReceiveControlBuffer final {
  std::array<std::byte, CMSG_SPACE(sizeof(ucred)) +
                            CMSG_SPACE(sizeof(int) * kMaximumControlFds)>
      bytes{};
};

struct alignas(cmsghdr) SendControlBuffer final {
  std::array<std::byte, CMSG_SPACE(sizeof(ucred)) +
                            CMSG_SPACE(sizeof(int) * kMaximumControlFds)>
      bytes{};
};

ReceivedPacket receive_packet(int descriptor, const ucred &expected,
                              std::size_t maximum_payload,
                              std::int64_t deadline,
                              std::size_t maximum_rights = 0U) {
  for (;;) {
    if (!wait_ready(descriptor, POLLIN, deadline))
      throw HostdTransportError("hostd receive deadline expired");
    std::vector<std::byte> bytes(kHostdStatusWireHeaderBytes + maximum_payload);
    ReceiveControlBuffer control;
    iovec vector{.iov_base = bytes.data(), .iov_len = bytes.size()};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1U;
    message.msg_control = control.bytes.data();
    message.msg_controllen = control.bytes.size();
    const ssize_t received =
        ::recvmsg(descriptor, &message,
                  MSG_DONTWAIT | MSG_TRUNC | MSG_CMSG_CLOEXEC);
    if (received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        continue;
      throw_errno("hostd recvmsg failed");
    }
  if (received == 0)
    throw HostdTransportError("hostd peer closed before sending a packet");
  bool credentials_seen = false;
  ucred credentials{};
  bool ancillary_rejected = false;
  std::vector<FileDescriptor> delegated;
  for (cmsghdr *header = CMSG_FIRSTHDR(&message); header != nullptr;
       header = CMSG_NXTHDR(&message, header)) {
    if (header->cmsg_len < CMSG_LEN(0U)) {
      ancillary_rejected = true;
      continue;
    }
    if (header->cmsg_level == SOL_SOCKET &&
        header->cmsg_type == SCM_CREDENTIALS &&
        header->cmsg_len == CMSG_LEN(sizeof(ucred)) && !credentials_seen) {
      std::memcpy(&credentials, CMSG_DATA(header), sizeof(credentials));
      credentials_seen = true;
      continue;
    }
    if (header->cmsg_level == SOL_SOCKET &&
        header->cmsg_type == SCM_RIGHTS &&
        header->cmsg_len >= CMSG_LEN(0U)) {
      const std::size_t payload = header->cmsg_len - CMSG_LEN(0U);
      if (payload % sizeof(int) != 0U) ancillary_rejected = true;
      const std::size_t count = payload / sizeof(int);
      const int *descriptors =
          reinterpret_cast<const int *>(CMSG_DATA(header));
      for (std::size_t index = 0U; index < count; ++index) {
        if (descriptors[index] < 0) {
          ancillary_rejected = true;
        } else if (delegated.size() < maximum_rights) {
          delegated.emplace_back(descriptors[index]);
        } else {
          (void)::close(descriptors[index]);
          ancillary_rejected = true;
        }
      }
      continue;
    }
    ancillary_rejected = true;
  }
  if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0 ||
      static_cast<std::uint64_t>(received) > bytes.size())
    throw HostdTransportError("hostd packet or ancillary data was truncated");
  if (ancillary_rejected)
    throw HostdTransportError("hostd packet carried forbidden ancillary data");
  if (!credentials_seen || credentials.pid != expected.pid ||
      credentials.uid != expected.uid || credentials.gid != expected.gid)
    throw HostdTransportError("hostd per-packet credentials are inexact");
  bytes.resize(static_cast<std::size_t>(received));
  return {.bytes = std::move(bytes),
          .credentials = credentials,
          .descriptors = std::move(delegated)};
  }
}

void send_packet(int descriptor, const std::vector<std::byte> &packet,
                 std::int64_t deadline,
                 std::span<const int> delegated = {}) {
  if (delegated.size() > kMaximumControlFds ||
      std::ranges::any_of(delegated, [](int value) { return value < 0; }))
    throw HostdTransportError("hostd delegated descriptor set is invalid");
  for (;;) {
    if (!wait_ready(descriptor, POLLOUT, deadline))
      throw HostdTransportError("hostd send deadline expired");
    SendControlBuffer control;
    iovec vector{.iov_base = const_cast<std::byte *>(packet.data()),
                 .iov_len = packet.size()};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1U;
    message.msg_control = control.bytes.data();
    message.msg_controllen = CMSG_SPACE(sizeof(ucred)) +
                             (delegated.empty()
                                  ? 0U
                                  : CMSG_SPACE(sizeof(int) * delegated.size()));
    cmsghdr *header = CMSG_FIRSTHDR(&message);
    if (header == nullptr)
      throw HostdTransportError(
          "could not construct hostd credential message");
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_CREDENTIALS;
    header->cmsg_len = CMSG_LEN(sizeof(ucred));
    const ucred credentials{.pid = ::getpid(),
                            .uid = ::geteuid(),
                            .gid = ::getegid()};
    std::memcpy(CMSG_DATA(header), &credentials, sizeof(credentials));
    if (!delegated.empty()) {
      cmsghdr *rights = CMSG_NXTHDR(&message, header);
      if (rights == nullptr)
        throw HostdTransportError(
            "could not construct hostd descriptor message");
      rights->cmsg_level = SOL_SOCKET;
      rights->cmsg_type = SCM_RIGHTS;
      rights->cmsg_len = CMSG_LEN(sizeof(int) * delegated.size());
      std::memcpy(CMSG_DATA(rights), delegated.data(),
                  sizeof(int) * delegated.size());
    }
    const ssize_t sent =
        ::sendmsg(descriptor, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        continue;
      throw_errno("hostd packet send failed");
    }
    if (static_cast<std::size_t>(sent) != packet.size())
      throw HostdTransportError("hostd packet send was partial");
    return;
  }
}

ucred peer_credentials(int descriptor) {
  ucred credentials{};
  socklen_t length = sizeof(credentials);
  if (::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credentials,
                   &length) != 0 ||
      length != sizeof(credentials) || credentials.pid <= 0)
    throw HostdTransportError("hostd peer credentials are unavailable");
  return credentials;
}

void enable_passcred(int descriptor) {
  int enabled = 1;
  if (::setsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &enabled,
                   sizeof(enabled)) != 0)
    throw_errno("could not enable hostd per-packet credentials");
}

std::vector<std::byte> error_packet(std::uint64_t correlation,
                                    std::string code,
                                    std::string message,
                                    std::string_view api_version =
                                        kHostdStatusTransportApiVersion) {
  return encode_packet(kErrorResponseOpcode, correlation,
                       {{"api_version", api_version},
                        {"code", std::move(code)},
                        {"message", std::move(message)}});
}

bool payload_fits(const std::vector<std::byte> &packet,
                  std::size_t maximum_payload_bytes) {
  return packet.size() >= kHostdStatusWireHeaderBytes &&
         packet.size() - kHostdStatusWireHeaderBytes <= maximum_payload_bytes;
}

std::uint64_t tentative_correlation(const std::vector<std::byte> &bytes) {
  if (bytes.size() < 24U)
    return 0U;
  try {
    return get_u64(bytes, 16U);
  } catch (...) {
    return 0U;
  }
}

class SocketBoundMutationPeerSource final : public IHostdPeerEvidenceSource {
public:
  SocketBoundMutationPeerSource(
      HostdLinuxBoundSocketPeer peer, HostdSessionChallengeEvidence evidence,
      std::shared_ptr<IHostdMutationServiceIdentityAuthority> authority,
      HostdMutationTransportEnforcementGrade transport_grade)
      : peer_(std::move(peer)), evidence_(std::move(evidence)),
        authority_(std::move(authority)), transport_grade_(transport_grade) {}

  HostdPeerEvidence observe() override {
    const HostdSocketPeerInstance observed = peer_.reobserve();
    if (observed != evidence_.peer)
      throw HostdUnauthorized(
          "socket peer changed after mutation challenge verification");
    if (transport_grade_ ==
            HostdMutationTransportEnforcementGrade::strict_service_identity &&
        peer_.enforcement_grade() !=
            HostdLinuxSessionEnforcementGrade::
                strict_host_namespaces_and_socket_pidfd) {
      throw HostdUnauthorized(
          "strict mutation transport lacks strict socket peer evidence");
    }
    HostdMutationServiceAuthorization authorization =
        authority_->authorize(observed);
    if (!authorization.service_identity_enforced ||
        authorization.access == HostdSessionAccess::denied ||
        authorization.access == HostdSessionAccess::read_only ||
        !valid_identifier(authorization.service_identity)) {
      throw HostdUnauthorized(
          "host service identity does not authorize mutation access");
    }
    if (baseline_set_ && baseline_ != authorization)
      throw HostdUnauthorized(
          "host service authorization changed during mutation session");
    baseline_ = authorization;
    baseline_set_ = true;
    return {
        .api_version = std::string(kHostdPeerEvidenceApiVersion),
        .peer_uid = observed.uid,
        .peer_gid = observed.gid,
        .peer_pid = observed.pid,
        .service_identity = authorization.service_identity,
        .enforcement_grade =
            HostdPeerEnforcementGrade::service_identity_enforced,
        .access = authorization.access,
        .evidence_digest = evidence_.evidence_digest,
    };
  }

private:
  HostdLinuxBoundSocketPeer peer_;
  HostdSessionChallengeEvidence evidence_;
  std::shared_ptr<IHostdMutationServiceIdentityAuthority> authority_;
  HostdMutationTransportEnforcementGrade transport_grade_;
  HostdMutationServiceAuthorization baseline_;
  bool baseline_set_{};
};

[[gnu::noinline]] std::shared_ptr<IHostdPeerEvidenceSource>
make_socket_bound_mutation_peer_source(
    HostdLinuxBoundSocketPeer peer, HostdSessionChallengeEvidence evidence,
    std::shared_ptr<IHostdMutationServiceIdentityAuthority> authority,
    HostdMutationTransportEnforcementGrade transport_grade) {
  return std::shared_ptr<IHostdPeerEvidenceSource>(
      new SocketBoundMutationPeerSource(
          std::move(peer), std::move(evidence), std::move(authority),
          transport_grade));
}

class CoordinatorSessionGuard final {
public:
  CoordinatorSessionGuard(std::shared_ptr<HostGrantCoordinator> coordinator,
                          std::string session_id)
      : coordinator_(std::move(coordinator)), session_id_(std::move(session_id)) {}
  ~CoordinatorSessionGuard() {
    if (!coordinator_ || session_id_.empty())
      return;
    try {
      coordinator_->disconnect(session_id_);
    } catch (...) {
      // The transport must close the socket even if an already-failed
      // coordinator cannot complete best-effort in-memory session cleanup.
    }
  }
  CoordinatorSessionGuard(const CoordinatorSessionGuard &) = delete;
  CoordinatorSessionGuard &operator=(const CoordinatorSessionGuard &) = delete;

private:
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  std::string session_id_;
};

class ChallengeDiscardGuard final {
public:
  explicit ChallengeDiscardGuard(
      std::shared_ptr<HostdSessionChallengeVerifier> verifier)
      : verifier_(std::move(verifier)) {}
  ~ChallengeDiscardGuard() {
    if (!challenge_)
      return;
    try {
      (void)verifier_->discard(challenge_->challenge_id, challenge_->peer);
    } catch (...) {
    }
  }
  void arm(const HostdSessionChallenge &challenge) { challenge_ = challenge; }
  void consumed() noexcept { challenge_.reset(); }
  ChallengeDiscardGuard(const ChallengeDiscardGuard &) = delete;
  ChallengeDiscardGuard &operator=(const ChallengeDiscardGuard &) = delete;

private:
  std::shared_ptr<HostdSessionChallengeVerifier> verifier_;
  std::optional<HostdSessionChallenge> challenge_;
};

std::int64_t checked_clock_ns(clockid_t clock, std::string_view name) {
  timespec value{};
  if (::clock_gettime(clock, &value) != 0)
    throw_errno(std::string("could not read hostd ") + std::string(name));
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1'000'000'000L ||
      value.tv_sec >
          std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL)
    throw HostdTransportError(std::string("hostd ") + std::string(name) +
                              " clock is out of range");
  return value.tv_sec * 1'000'000'000LL + value.tv_nsec;
}

void mutation_checkpoint(
    const std::shared_ptr<IHostdMutationTransportFaultInjector> &injector,
    HostdMutationTransportCheckpoint checkpoint) {
  if (!injector)
    return;
  try {
    injector->checkpoint(checkpoint);
  } catch (const HostdTransportError &) {
    throw;
  } catch (...) {
    throw HostdTransportError("hostd mutation fault checkpoint interrupted");
  }
}

} // namespace

struct HostdSocketAuthority::Implementation final {
  HostdSocketAuthorityConfig config;
  FileDescriptor parent;
  FileDescriptor listener;
  FileDescriptor pinned_path;
  std::shared_ptr<IHostdSingletonToken> singleton;
  HostdSocketIdentity identity;
  bool owns_path{};
  bool poisoned{};
  std::string poison_reason;
  mutable std::mutex mutex;

  void cleanup_owned_path() noexcept {
    if (!owns_path)
      return;
    const std::string basename = config.socket_path.filename().string();
    for (;;) {
      struct stat path{};
      struct stat parent_status{};
      if (fstat_nointr(parent.get(), &parent_status) != 0 ||
          fstatat_nointr(parent.get(), basename.c_str(), &path,
                         AT_SYMLINK_NOFOLLOW) != 0 ||
          socket_identity(parent_status, path) != identity)
        break;
      if (::unlinkat(parent.get(), basename.c_str(), 0) == 0 ||
          errno != EINTR)
        break;
      // An interrupted unlink may have completed. Re-attest the full parent
      // and pathname identity before every retry; never carry authority over
      // to a same-name replacement.
    }
    owns_path = false;
  }

  ~Implementation() { cleanup_owned_path(); }

  HostdSocketIdentity attest() {
    if (poisoned)
      throw HostdTransportError(poison_reason);
    try {
      const struct stat held_parent = inspect_fd(parent.get(), "socket parent");
      validate_parent(held_parent, config);
      auto reopened_parent = open_parent_from_root(config.socket_path.parent_path());
      const struct stat reopened_parent_status =
          inspect_fd(reopened_parent.get(), "reopened socket parent");
      if (held_parent.st_dev != reopened_parent_status.st_dev ||
          held_parent.st_ino != reopened_parent_status.st_ino)
        throw HostdTransportError("hostd socket parent path was replaced");
      const std::string basename = config.socket_path.filename().string();
      const int opened = ::openat(parent.get(), basename.c_str(),
                                  O_PATH | O_CLOEXEC | O_NOFOLLOW);
      if (opened < 0)
        throw_errno("could not reopen hostd socket pathname");
      FileDescriptor reopened(opened);
      const struct stat held_path = inspect_fd(pinned_path.get(), "pinned socket");
      const struct stat path = inspect_fd(reopened.get(), "socket pathname");
      validate_socket_path(path, config);
      const HostdSocketIdentity current = socket_identity(held_parent, path);
      if (current != identity || held_path.st_dev != path.st_dev ||
          held_path.st_ino != path.st_ino)
        throw HostdTransportError("hostd socket pathname identity changed");
      validate_listener_fd(listener.get(), config);
      if (!singleton || !singleton->attest_held())
        throw HostdTransportError("hostd singleton token is no longer held");
      return current;
    } catch (const std::exception &error) {
      poisoned = true;
      poison_reason = std::string("hostd socket authority poisoned: ") +
                      error.what();
      throw HostdTransportError(poison_reason);
    }
  }
};

namespace {

void bind_checkpoint(
    const std::shared_ptr<IHostdSocketBindFaultInjector> &injector,
    HostdSocketBindCheckpoint checkpoint) {
  if (!injector)
    return;
  try {
    injector->checkpoint(checkpoint);
  } catch (const HostdTransportError &) {
    throw;
  } catch (const std::exception &) {
    throw HostdTransportError("hostd bind fault checkpoint failed");
  } catch (...) {
    throw HostdTransportError("hostd bind fault checkpoint failed");
  }
}

void validate_authority_config(const HostdSocketAuthorityConfig &config,
                               int parent_fd) {
  const std::string native = config.socket_path.string();
  if (config.api_version != kHostdStatusTransportApiVersion ||
      !config.socket_path.is_absolute() || config.socket_path.empty() ||
      config.socket_path.lexically_normal() != config.socket_path ||
      native.size() > kMaximumSocketPathBytes ||
      (native.size() > 1U && native.starts_with("//")) ||
      !safe_basename(config.socket_path.filename().string()) ||
      config.listen_backlog == 0U || config.listen_backlog > 4096U ||
      config.enforcement_grade !=
          HostdSocketEnforcementGrade::cooperative_test ||
      (config.expected_parent_mode != 0700U &&
       config.expected_parent_mode != 0750U) ||
      (config.expected_socket_mode != 0600U &&
       config.expected_socket_mode != 0660U))
    throw HostdTransportError("hostd socket authority config is invalid");
  const struct stat supplied_parent = inspect_fd(parent_fd, "supplied parent");
  validate_parent(supplied_parent, config);
  auto path_parent = open_parent_from_root(config.socket_path.parent_path());
  const struct stat path_parent_status =
      inspect_fd(path_parent.get(), "path parent");
  if (supplied_parent.st_dev != path_parent_status.st_dev ||
      supplied_parent.st_ino != path_parent_status.st_ino)
    throw HostdTransportError(
        "supplied parent descriptor does not match socket path");
}

std::unique_ptr<HostdSocketAuthority::Implementation>
finish_authority(HostdSocketAuthorityConfig config, int parent_fd,
                 FileDescriptor listener,
                 std::shared_ptr<IHostdSingletonToken> singleton,
                 bool owns_path) {
  validate_listener_fd(listener.get(), config);
  enable_passcred(listener.get());
  FileDescriptor parent = duplicate_cloexec(parent_fd);
  const std::string basename = config.socket_path.filename().string();
  const int path_fd = ::openat(parent.get(), basename.c_str(),
                               O_PATH | O_CLOEXEC | O_NOFOLLOW);
  if (path_fd < 0)
    throw_errno("could not pin hostd socket pathname");
  FileDescriptor pinned_path(path_fd);
  const struct stat parent_status = inspect_fd(parent.get(), "socket parent");
  const struct stat path_status = inspect_fd(pinned_path.get(), "socket path");
  validate_parent(parent_status, config);
  validate_socket_path(path_status, config);
  auto implementation = std::make_unique<HostdSocketAuthority::Implementation>();
  implementation->config = std::move(config);
  implementation->parent = std::move(parent);
  implementation->listener = std::move(listener);
  implementation->pinned_path = std::move(pinned_path);
  implementation->singleton = std::move(singleton);
  implementation->identity = socket_identity(parent_status, path_status);
  implementation->owns_path = owns_path;
  return implementation;
}

class BoundPathRollback final {
public:
  BoundPathRollback(int pinned_parent_fd, std::string basename)
      : parent_(duplicate_cloexec(pinned_parent_fd)),
        basename_(std::move(basename)) {}

  void capture_created(
      const std::shared_ptr<IHostdSocketBindFaultInjector> &fault_injector) {
    try {
      bind_checkpoint(fault_injector,
                      HostdSocketBindCheckpoint::before_identity_capture);
    } catch (...) {
      std::terminate();
    }
    struct stat created_status{};
    while (::fstatat(parent_.get(), basename_.c_str(), &created_status,
                     AT_SYMLINK_NOFOLLOW) != 0) {
      if (errno == EINTR)
        continue;
      // Once bind succeeds, proceeding without the inode would make exact
      // rollback impossible. ENOENT proves there is no pathname to clean;
      // every other failure is fail-stop rather than risking a stale live
      // process or blindly unlinking a same-name replacement.
      if (errno != ENOENT)
        std::terminate();
      throw_errno("could not capture newly bound hostd socket pathname");
    }
    if (!S_ISSOCK(created_status.st_mode) || created_status.st_nlink != 1)
      throw HostdTransportError("newly bound hostd path is not a socket");
    created_device_ = created_status.st_dev;
    created_inode_ = created_status.st_ino;
    captured_ = true;
  }

  void prove_absolute_parent(const std::filesystem::path &absolute_parent) {
    auto resolved = open_parent_from_root(absolute_parent);
    const struct stat resolved_status =
        inspect_fd(resolved.get(), "resolved bind parent");
    const struct stat pinned_status =
        inspect_fd(parent_.get(), "pinned bind parent");
    if (resolved_status.st_dev != pinned_status.st_dev ||
        resolved_status.st_ino != pinned_status.st_ino)
      throw HostdTransportError(
          "configured AF_UNIX parent changed during anchored bind");
  }

  ~BoundPathRollback() {
    if (!captured_ || released_)
      return;
    for (;;) {
      struct stat current{};
      if (fstatat_nointr(parent_.get(), basename_.c_str(), &current,
                         AT_SYMLINK_NOFOLLOW) != 0 ||
          current.st_dev != created_device_ ||
          current.st_ino != created_inode_)
        break;
      if (::unlinkat(parent_.get(), basename_.c_str(), 0) == 0 ||
          errno != EINTR)
        break;
      // Re-attest the exact captured inode before an EINTR retry.
    }
  }

  void release() noexcept { released_ = true; }

private:
  FileDescriptor parent_;
  std::string basename_;
  dev_t created_device_{};
  ino_t created_inode_{};
  bool captured_{};
  bool released_{};
};

class StartupCwdGuard final {
public:
  StartupCwdGuard(
      int pinned_parent_fd,
      std::shared_ptr<IHostdSocketBindFaultInjector> fault_injector)
      : fault_injector_(std::move(fault_injector)) {
    sigset_t all{};
    if (::sigfillset(&all) != 0 ||
        ::pthread_sigmask(SIG_SETMASK, &all, &previous_mask_) != 0)
      throw HostdTransportError("could not block signals for anchored bind");
    signals_blocked_ = true;
    try {
      bind_checkpoint(fault_injector_,
                      HostdSocketBindCheckpoint::signals_blocked);
      std::size_t tasks = 0U;
      for (const auto &entry :
           std::filesystem::directory_iterator("/proc/self/task")) {
        (void)entry;
        ++tasks;
      }
      if (tasks != 1U)
        throw HostdTransportError(
            "hostd self-bind requires single-threaded startup");
    } catch (const HostdTransportError &) {
      restore_noexcept();
      throw;
    } catch (const std::exception &) {
      restore_noexcept();
      throw HostdTransportError("could not enumerate hostd startup threads");
    } catch (...) {
      restore_noexcept();
      throw HostdTransportError("could not enumerate hostd startup threads");
    }
    cwd_ = FileDescriptor(
        ::open(".", O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (cwd_.get() < 0 || ::fchdir(pinned_parent_fd) != 0) {
      restore_noexcept();
      throw_errno("could not enter pinned hostd socket parent");
    }
    changed_directory_ = true;
  }

  ~StartupCwdGuard() { restore_noexcept(); }

  void restore() {
    try {
      bind_checkpoint(fault_injector_,
                      HostdSocketBindCheckpoint::before_cwd_restore);
    } catch (...) {
      std::terminate();
    }
    if (changed_directory_ && ::fchdir(cwd_.get()) != 0)
      std::terminate();
    changed_directory_ = false;
    if (signals_blocked_ &&
        ::pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr) != 0)
      std::terminate();
    signals_blocked_ = false;
  }

private:
  void restore_noexcept() noexcept {
    if (changed_directory_ && ::fchdir(cwd_.get()) != 0)
      std::terminate();
    changed_directory_ = false;
    if (signals_blocked_ &&
        ::pthread_sigmask(SIG_SETMASK, &previous_mask_, nullptr) != 0)
      std::terminate();
    signals_blocked_ = false;
  }

  FileDescriptor cwd_;
  std::shared_ptr<IHostdSocketBindFaultInjector> fault_injector_;
  sigset_t previous_mask_{};
  bool signals_blocked_{};
  bool changed_directory_{};
};

} // namespace

HostdSocketAuthority HostdSocketAuthority::self_bind(
    HostdSocketAuthorityConfig config, int pinned_parent_fd,
    std::shared_ptr<IHostdSingletonToken> singleton) {
  validate_authority_config(config, pinned_parent_fd);
  if (!singleton || !singleton->attest_held())
    throw HostdTransportError(
        "self-bound hostd socket requires an already-held singleton token");
  const std::string basename = config.socket_path.filename().string();
  struct stat existing{};
  if (::fstatat(pinned_parent_fd, basename.c_str(), &existing,
                AT_SYMLINK_NOFOLLOW) == 0)
    throw HostdTransportError(
        "hostd socket pathname already exists and is never blindly removed");
  if (errno != ENOENT)
    throw_errno("could not inspect hostd socket pathname before bind");
  FileDescriptor listener(
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (listener.get() < 0)
    throw_errno("could not create hostd seqpacket listener");
  BoundPathRollback rollback(pinned_parent_fd, basename);
  {
    StartupCwdGuard cwd(pinned_parent_fd, config.fault_injector);
    socklen_t address_length = 0U;
    const sockaddr_un address =
        socket_address(std::filesystem::path(basename), address_length);
    if (::bind(listener.get(), reinterpret_cast<const sockaddr *>(&address),
               address_length) != 0)
      throw_errno("could not bind hostd filesystem socket");
    rollback.capture_created(config.fault_injector);
    bind_checkpoint(config.fault_injector,
                    HostdSocketBindCheckpoint::identity_captured);
    rollback.prove_absolute_parent(config.socket_path.parent_path());
    cwd.restore();
  }
  bind_checkpoint(config.fault_injector,
                  HostdSocketBindCheckpoint::before_socket_protection);
  if (::fchownat(pinned_parent_fd, basename.c_str(),
                 config.expected_owner_uid, config.expected_owner_gid,
                 AT_SYMLINK_NOFOLLOW) != 0 ||
      ::fchmodat(pinned_parent_fd, basename.c_str(),
                 static_cast<mode_t>(config.expected_socket_mode), 0) != 0)
    throw_errno("could not protect hostd socket pathname");
  bind_checkpoint(config.fault_injector,
                  HostdSocketBindCheckpoint::before_listen);
  if (::listen(listener.get(), static_cast<int>(config.listen_backlog)) != 0)
    throw_errno("could not listen on hostd socket");
  config.fault_injector.reset();
  auto implementation = finish_authority(
      std::move(config), pinned_parent_fd, std::move(listener),
      std::move(singleton), true);
  rollback.release();
  HostdSocketAuthority result(std::move(implementation));
  (void)result.reattest();
  return result;
}

HostdSocketAuthority::HostdSocketAuthority(
    std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {}

HostdSocketAuthority::~HostdSocketAuthority() {
}

HostdSocketAuthority::HostdSocketAuthority(HostdSocketAuthority &&) noexcept =
    default;
HostdSocketAuthority &
HostdSocketAuthority::operator=(HostdSocketAuthority &&other) noexcept {
  if (this != &other)
    implementation_ = std::move(other.implementation_);
  return *this;
}

HostdSocketIdentity HostdSocketAuthority::reattest() {
  if (!implementation_)
    throw HostdTransportError("hostd socket authority was moved from");
  std::scoped_lock lock(implementation_->mutex);
  return implementation_->attest();
}

bool HostdSocketAuthority::poisoned() const {
  if (!implementation_)
    return true;
  std::scoped_lock lock(implementation_->mutex);
  return implementation_->poisoned;
}

std::string HostdSocketAuthority::poison_reason() const {
  if (!implementation_)
    return "hostd socket authority was moved from";
  std::scoped_lock lock(implementation_->mutex);
  return implementation_->poison_reason;
}

const std::filesystem::path &HostdSocketAuthority::socket_path() const {
  if (!implementation_)
    throw HostdTransportError("hostd socket authority was moved from");
  return implementation_->config.socket_path;
}

int HostdSocketAuthority::listener_fd() const noexcept {
  return implementation_ ? implementation_->listener.get() : -1;
}

HostdStatusServer::HostdStatusServer(
    std::shared_ptr<HostdSocketAuthority> authority,
    std::shared_ptr<HostGrantCoordinator> coordinator,
    HostdStatusPeerPolicy peer_policy, HostdStatusTransportLimits limits,
    std::shared_ptr<IHostdAuthorityStatusSource> authority_status_source)
    : authority_(std::move(authority)), coordinator_(std::move(coordinator)),
      authority_status_source_(std::move(authority_status_source)),
      peer_policy_(peer_policy), limits_(limits) {
  if (!authority_ || !coordinator_ || limits_.maximum_payload_bytes == 0U ||
      limits_.maximum_payload_bytes > kHostdStatusMaximumPayloadBytes ||
      limits_.per_session_timeout_ns < 1'000'000LL ||
      limits_.per_session_timeout_ns > 3'600'000'000'000LL)
    throw HostdTransportError("hostd status transport limits are invalid");
}

HostdServeResult
HostdStatusServer::serve_one(std::int64_t absolute_monotonic_deadline_ns) {
  try {
    (void)authority_->reattest();
    if (!wait_ready(authority_->listener_fd(), POLLIN,
                    absolute_monotonic_deadline_ns))
      return HostdServeResult::timed_out;
    for (;;) {
      const int accepted =
          ::accept4(authority_->listener_fd(), nullptr, nullptr,
                    SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (accepted >= 0)
        return serve_accepted(accepted, absolute_monotonic_deadline_ns);
      if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        throw_errno("hostd accept4 failed");
      if (!wait_ready(authority_->listener_fd(), POLLIN,
                      absolute_monotonic_deadline_ns))
        return HostdServeResult::timed_out;
    }
  } catch (const std::exception &error) {
    if (limits_.rejection_observer) {
      try {
        limits_.rejection_observer(error.what());
      } catch (...) {
      }
    }
    return HostdServeResult::rejected;
  } catch (...) {
    if (limits_.rejection_observer) {
      try {
        limits_.rejection_observer("unknown status listener rejection");
      } catch (...) {
      }
    }
    return HostdServeResult::rejected;
  }
}

HostdServeResult HostdStatusServer::serve_accepted(
    int connection_fd, std::int64_t absolute_monotonic_deadline_ns) {
  FileDescriptor connection(connection_fd);
  if (connection.get() < 0) return HostdServeResult::rejected;
  try {
    (void)authority_->reattest();
    enable_passcred(connection.get());
    const ucred peer = peer_credentials(connection.get());
    if (peer.uid != peer_policy_.allowed_uid ||
        peer.gid != peer_policy_.allowed_gid) {
      if (limits_.rejection_observer) {
        try {
          limits_.rejection_observer(
              "peer credential mismatch: observed uid=" +
              std::to_string(peer.uid) + " gid=" + std::to_string(peer.gid) +
              ", allowed uid=" + std::to_string(peer_policy_.allowed_uid) +
              " gid=" + std::to_string(peer_policy_.allowed_gid));
        } catch (...) {
        }
      }
      return HostdServeResult::rejected;
    }
    const std::int64_t now = hostd_monotonic_now_ns();
    const std::int64_t relative_deadline =
        limits_.per_session_timeout_ns >
                std::numeric_limits<std::int64_t>::max() - now
            ? std::numeric_limits<std::int64_t>::max()
            : now + limits_.per_session_timeout_ns;
    const std::int64_t session_deadline =
        std::min(absolute_monotonic_deadline_ns, relative_deadline);
    ReceivedPacket received = receive_packet(
        connection.get(), peer, limits_.maximum_payload_bytes,
        session_deadline);
    const std::uint64_t tentative = tentative_correlation(received.bytes);
    WirePacket request;
    try {
      request = decode_packet(received.bytes, limits_.maximum_payload_bytes);
      if (request.opcode != kStatusRequestOpcode)
        throw HostdTransportError("hostd status opcode is unsupported");
      const nlohmann::json payload = parse_canonical_json(request.payload);
      require_fields(payload, {"api_version"});
      if (payload.at("api_version").get<std::string>() !=
          kHostdStatusTransportApiVersion)
        throw HostdTransportError("hostd status request API is unsupported");
    } catch (const std::exception &) {
      if (tentative != 0U) {
        const auto error = error_packet(
            tentative, "malformed_request",
            "request failed strict transport validation");
        if (payload_fits(error, limits_.maximum_payload_bytes))
          send_packet(connection.get(), error, session_deadline);
      }
      return HostdServeResult::rejected;
    }
    const HostdCoordinatorStatus status = coordinator_->status();
    validate_status_semantics(status);
    std::optional<HostdAuthorityStatus> authority_status;
    if (authority_status_source_) {
      authority_status = authority_status_source_->snapshot();
      validate_authority_status_semantics(*authority_status, status);
    }
    const auto response = encode_packet(
        kStatusResponseOpcode, request.correlation_id,
        {{"api_version", kHostdStatusTransportApiVersion},
         {"authority_status", authority_status
                                  ? authority_status_json(*authority_status)
                                  : nlohmann::json(nullptr)},
         {"status", status_json(status)}});
    if (response.size() - kHostdStatusWireHeaderBytes >
        limits_.maximum_payload_bytes) {
      const auto error = error_packet(
          request.correlation_id, "status_too_large",
          "status exceeds negotiated transport bound");
      if (payload_fits(error, limits_.maximum_payload_bytes))
        send_packet(connection.get(), error, session_deadline);
      return HostdServeResult::rejected;
    }
    send_packet(connection.get(), response, session_deadline);
    (void)::shutdown(connection.get(), SHUT_RDWR);
    return HostdServeResult::served;
  } catch (const std::exception &error) {
    if (limits_.rejection_observer) {
      try {
        limits_.rejection_observer(error.what());
      } catch (...) {
      }
    }
    return HostdServeResult::rejected;
  } catch (...) {
    if (limits_.rejection_observer) {
      try {
        limits_.rejection_observer("unknown accepted status rejection");
      } catch (...) {
      }
    }
    return HostdServeResult::rejected;
  }
}

HostdMutationServer::HostdMutationServer(
    std::shared_ptr<HostdSocketAuthority> authority,
    std::shared_ptr<HostGrantCoordinator> coordinator,
    std::shared_ptr<HostdSessionChallengeVerifier> challenge_verifier,
    std::shared_ptr<IHostdLinuxSessionKernel> session_kernel,
    std::shared_ptr<IHostdMutationServiceIdentityAuthority>
        service_identity_authority,
    std::shared_ptr<IHostdLedgerTimeSource> ledger_time_source,
    HostdMutationTransportConfig config,
    std::shared_ptr<IHostdProcessSupervisor> process_supervisor)
    : authority_(std::move(authority)), coordinator_(std::move(coordinator)),
      challenge_verifier_(std::move(challenge_verifier)),
      session_kernel_(std::move(session_kernel)),
      service_identity_authority_(std::move(service_identity_authority)),
      ledger_time_source_(std::move(ledger_time_source)),
      process_supervisor_(std::move(process_supervisor)),
      config_(std::move(config)) {
  const bool strict =
      config_.enforcement_grade ==
      HostdMutationTransportEnforcementGrade::strict_service_identity;
  if (!authority_ || !coordinator_ || !challenge_verifier_ ||
      !session_kernel_ || !service_identity_authority_ ||
      !ledger_time_source_ ||
      config_.api_version != kHostdMutationTransportApiVersion ||
      config_.maximum_payload_bytes == 0U ||
      config_.maximum_payload_bytes > kHostdStatusMaximumPayloadBytes ||
      config_.per_session_timeout_ns < 1'000'000LL ||
      config_.per_session_timeout_ns > 3'600'000'000'000LL ||
      (strict && config_.socket_peer_grade !=
                     HostdLinuxSessionEnforcementGrade::
                         strict_host_namespaces_and_socket_pidfd) ||
      (strict && session_kernel_->enforcement_grade() !=
                     HostdLinuxSessionEnforcementGrade::
                         strict_host_namespaces_and_socket_pidfd)) {
    throw HostdTransportError("hostd mutation transport config is invalid");
  }
}

HostdServeResult HostdMutationServer::serve_one(
    std::int64_t absolute_monotonic_deadline_ns) {
  try {
    (void)authority_->reattest();
    if (!wait_ready(authority_->listener_fd(), POLLIN,
                    absolute_monotonic_deadline_ns))
      return HostdServeResult::timed_out;
    for (;;) {
      const int accepted =
          ::accept4(authority_->listener_fd(), nullptr, nullptr,
                    SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (accepted >= 0)
        return serve_accepted(accepted, absolute_monotonic_deadline_ns);
      if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        throw_errno("hostd mutation accept4 failed");
      if (!wait_ready(authority_->listener_fd(), POLLIN,
                      absolute_monotonic_deadline_ns))
        return HostdServeResult::timed_out;
    }
  } catch (...) {
    return HostdServeResult::rejected;
  }
}

HostdServeResult HostdMutationServer::serve_accepted(
    int connection_fd, std::int64_t absolute_monotonic_deadline_ns) {
  FileDescriptor connection(connection_fd);
  if (connection.get() < 0) return HostdServeResult::rejected;
  try {
    (void)authority_->reattest();
    enable_passcred(connection.get());
    const ucred credentials = peer_credentials(connection.get());
    if (credentials.uid != config_.allowed_uid ||
        credentials.gid != config_.allowed_gid)
      return HostdServeResult::rejected;
    const std::int64_t accepted_at = hostd_monotonic_now_ns();
    const std::int64_t relative_deadline =
        config_.per_session_timeout_ns >
                std::numeric_limits<std::int64_t>::max() - accepted_at
            ? std::numeric_limits<std::int64_t>::max()
            : accepted_at + config_.per_session_timeout_ns;
    const std::int64_t session_deadline =
        std::min(absolute_monotonic_deadline_ns, relative_deadline);

    HostdLinuxBoundSocketPeer bound_peer = make_hostd_linux_bound_socket_peer(
        connection.get(), session_kernel_, config_.socket_peer_grade);
    const ReceivedPacket open_packet = receive_packet(
        connection.get(), credentials, config_.maximum_payload_bytes,
        session_deadline);
    const std::uint64_t tentative = tentative_correlation(open_packet.bytes);
    std::uint64_t correlation = tentative;
    const auto send_error = [&](std::string code, std::string message) {
      if (correlation == 0U)
        return;
      try {
        const auto packet = error_packet(
            correlation, std::move(code), std::move(message),
            kHostdMutationTransportApiVersion);
        if (payload_fits(packet, config_.maximum_payload_bytes))
          send_packet(connection.get(), packet, session_deadline);
      } catch (...) {
      }
    };
    ChallengeDiscardGuard challenge_guard(challenge_verifier_);

    try {
      const WirePacket decoded_open =
          decode_packet(open_packet.bytes, config_.maximum_payload_bytes);
      correlation = decoded_open.correlation_id;
      if (decoded_open.opcode != kMutationOpenOpcode)
        throw HostdTransportError("hostd mutation open opcode is unsupported");
      const HostdMutationOpen open =
          hostd_mutation_open_from_canonical_json(decoded_open.payload);
      const HostdSessionChallenge challenge = challenge_verifier_->issue(
          bound_peer.instance(), open.claim);
      challenge_guard.arm(challenge);
      send_packet(connection.get(),
                  encode_canonical_packet(
                      kMutationChallengeOpcode, correlation,
                      hostd_session_challenge_canonical_json(challenge)),
                  session_deadline);
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::
                              after_challenge_sent);

      const ReceivedPacket command_packet = receive_packet(
          connection.get(), credentials, config_.maximum_payload_bytes,
          session_deadline, 6U);
      const WirePacket decoded_command =
          decode_packet(command_packet.bytes, config_.maximum_payload_bytes);
      if (decoded_command.correlation_id != correlation ||
          decoded_command.opcode != kMutationCommandOpcode)
        throw HostdTransportError(
            "hostd mutation command framing is inexact");
      const HostdMutationCommand command =
          hostd_mutation_command_from_canonical_json(decoded_command.payload);
      const std::size_t expected_descriptors =
          command.process_prepare
              ? command.process_prepare->descriptor_roles.size()
              : 0U;
      if (command_packet.descriptors.size() != expected_descriptors)
        throw HostdTransportError(
            "hostd process descriptor count is inexact");
      validate_hostd_mutation_exchange(open, challenge, command);
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::
                              after_command_received);

      const HostdSocketPeerInstance observed_peer = bound_peer.reobserve();
      // verify() consumes a matching challenge before any callback or failure.
      challenge_guard.consumed();
      HostdSessionChallengeEvidence challenge_evidence =
          challenge_verifier_->verify(command.challenge_response,
                                      observed_peer);
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::
                              after_challenge_verified);
      auto peer_source = make_socket_bound_mutation_peer_source(
          std::move(bound_peer), std::move(challenge_evidence),
          service_identity_authority_, config_.enforcement_grade);
      const auto &journal = command.challenge_response.claim.journal;
      const auto &controller = command.challenge_response.claim.controller;
      HostdConnectRequest connect{
          .attribution = HostdSessionAttribution{
              .journal_id = journal.journal_id,
              .run_id = controller.run_id,
              .concurrency_key = controller.concurrency_key,
              .logical_lease_id = controller.logical_lease_id,
              .logical_fencing_token = controller.logical_fencing_token,
          }};
      const HostdConnectedSession session =
          coordinator_->connect(std::move(connect), std::move(peer_source));
      CoordinatorSessionGuard session_guard(coordinator_, session.session_id);
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::
                              after_coordinator_connected);

      HostdMutationReply reply{
          .api_version = std::string(kHostdMutationProtocolApiVersion),
          .kind = HostdMutationReplyKind::reconciliation_missing,
          .challenge_id = command.challenge_response.challenge_id,
          .command_digest = command.command_digest,
          .bundle_result = std::nullopt,
          .release_result = std::nullopt,
          .process_prepared = std::nullopt,
          .process_committed = std::nullopt,
          .process_exit = std::nullopt,
      };
      switch (command.mutation) {
      case HostdMutationKind::request_bundle: {
        const HostLedgerTime now = ledger_time_source_->now();
        if (now.boottime_ns < 0 || now.wall_time_ns < 0)
          throw HostdTransportError("hostd ledger time is invalid");
        reply.kind = HostdMutationReplyKind::bundle_outcome;
        reply.bundle_result = coordinator_->request_bundle(
            session.session_id, *command.bundle_request, now);
        break;
      }
      case HostdMutationKind::reconcile_bundle_outcome: {
        reply.bundle_result = coordinator_->reconcile_bundle_outcome(
            session.session_id, *command.bundle_request);
        if (reply.bundle_result)
          reply.kind = HostdMutationReplyKind::bundle_outcome;
        break;
      }
      case HostdMutationKind::release_bundle: {
        const HostLedgerTime now = ledger_time_source_->now();
        if (now.boottime_ns < 0 || now.wall_time_ns < 0)
          throw HostdTransportError("hostd ledger time is invalid");
        reply.kind = HostdMutationReplyKind::release_outcome;
        reply.release_result = coordinator_->release_bundle(
            session.session_id, *command.release_request, now);
        break;
      }
      case HostdMutationKind::prepare_process: {
        if (session.effective_access != HostdSessionAccess::grant_release ||
            !process_supervisor_ || command_packet.descriptors.size() < 2U)
          throw HostdTransportError(
              "hostd process prepare authority is unavailable");
        const auto& descriptors = command_packet.descriptors;
        const bool has_code = command.process_prepare->launch.identity.code.has_value();
        const bool has_profiler =
            command.process_prepare->launch.identity.profiler.has_value();
        const std::size_t working_index = has_code ? 2U : 1U;
        const std::size_t bootstrap_index = has_code ? 3U : 2U;
        const std::size_t profiler_index = has_code ? 4U : 3U;
        reply.kind = HostdMutationReplyKind::process_prepared;
        reply.process_prepared = process_supervisor_->prepare(
            *command.process_prepare, descriptors[0].get(),
            has_code ? std::optional<int>(descriptors[1].get()) : std::nullopt,
            descriptors[working_index].get(), descriptors[bootstrap_index].get(),
            has_profiler
                ? std::optional<int>(descriptors[profiler_index].get())
                : std::nullopt,
            has_profiler
                ? std::optional<int>(descriptors[profiler_index + 1U].get())
                : std::nullopt);
        break;
      }
      case HostdMutationKind::commit_process: {
        if (session.effective_access != HostdSessionAccess::grant_release ||
            !process_supervisor_)
          throw HostdTransportError(
              "hostd process commit authority is unavailable");
        reply.kind = HostdMutationReplyKind::process_committed;
        reply.process_committed =
            process_supervisor_->commit(*command.process_commit);
        break;
      }
      case HostdMutationKind::finalize_process: {
        if (session.effective_access != HostdSessionAccess::grant_release ||
            !process_supervisor_)
          throw HostdTransportError(
              "hostd process exit authority is unavailable");
        reply.kind = HostdMutationReplyKind::process_exited;
        reply.process_exit =
            process_supervisor_->finalize(*command.process_exit);
        break;
      }
      }
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::
                              after_dispatch_committed);
      validate_hostd_mutation_reply(command, reply);
      const auto reply_packet = encode_canonical_packet(
          kMutationReplyOpcode, correlation,
          hostd_mutation_reply_canonical_json(reply));
      if (!payload_fits(reply_packet, config_.maximum_payload_bytes))
        throw HostdTransportError("hostd mutation reply exceeds payload bound");
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::before_reply_send);
      send_packet(connection.get(), reply_packet, session_deadline);
      mutation_checkpoint(config_.fault_injector,
                          HostdMutationTransportCheckpoint::after_reply_send);
      (void)::shutdown(connection.get(), SHUT_RDWR);
      return HostdServeResult::served;
    } catch (const HostdSessionChallengeRejected &) {
      send_error("challenge_rejected",
                 "mutation challenge verification failed");
      return HostdServeResult::rejected;
    } catch (const HostdUnauthorized &) {
      send_error("unauthorized", "mutation authorization failed");
      return HostdServeResult::rejected;
    } catch (const HostdStateError &) {
      send_error("state_rejected", "hostd state rejected the mutation");
      return HostdServeResult::rejected;
    } catch (const std::exception &) {
      send_error("mutation_rejected",
                 "mutation failed strict transport validation");
      return HostdServeResult::rejected;
    } catch (...) {
      send_error("mutation_rejected", "mutation failed closed");
      return HostdServeResult::rejected;
    }
  } catch (...) {
    return HostdServeResult::rejected;
  }
}

HostdUnifiedServer::HostdUnifiedServer(
    std::shared_ptr<HostdSocketAuthority> authority,
    HostdStatusServer& status, HostdMutationServer& mutation)
    : authority_(std::move(authority)), status_(status), mutation_(mutation) {
  if (!authority_)
    throw HostdTransportError("hostd unified server requires socket authority");
}

HostdServeResult HostdUnifiedServer::serve_one(
    std::int64_t absolute_monotonic_deadline_ns) {
  try {
    (void)authority_->reattest();
    if (!wait_ready(authority_->listener_fd(), POLLIN,
                    absolute_monotonic_deadline_ns))
      return HostdServeResult::timed_out;
    FileDescriptor connection;
    for (;;) {
      const int accepted =
          ::accept4(authority_->listener_fd(), nullptr, nullptr,
                    SOCK_CLOEXEC | SOCK_NONBLOCK);
      if (accepted >= 0) {
        connection = FileDescriptor(accepted);
        break;
      }
      if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        throw_errno("hostd unified accept4 failed");
      if (!wait_ready(authority_->listener_fd(), POLLIN,
                      absolute_monotonic_deadline_ns))
        return HostdServeResult::timed_out;
    }
    (void)authority_->reattest();
    std::array<std::byte, 12U> prefix{};
    for (;;) {
      if (!wait_ready(connection.get(), POLLIN,
                      absolute_monotonic_deadline_ns))
        return HostdServeResult::rejected;
      const ssize_t received =
          ::recv(connection.get(), prefix.data(), prefix.size(),
                 MSG_DONTWAIT | MSG_PEEK);
      if (received < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        throw_errno("hostd unified protocol peek failed");
      }
      if (received < static_cast<ssize_t>(prefix.size()))
        return HostdServeResult::rejected;
      break;
    }
    const std::vector<std::byte> wire_prefix(prefix.begin(), prefix.end());
    if (!std::ranges::equal(kWireMagic,
                            wire_prefix | std::views::take(4)) ||
        get_u16(wire_prefix, 4U) != kHostdStatusWireVersion ||
        get_u16(wire_prefix, 6U) != kHostdStatusWireHeaderBytes ||
        get_u16(wire_prefix, 10U) != 0U)
      return HostdServeResult::rejected;
    const std::uint16_t opcode = get_u16(wire_prefix, 8U);
    if (opcode == kStatusRequestOpcode)
      return status_.serve_accepted(connection.release(),
                                    absolute_monotonic_deadline_ns);
    if (opcode == kMutationOpenOpcode)
      return mutation_.serve_accepted(connection.release(),
                                      absolute_monotonic_deadline_ns);
    return HostdServeResult::rejected;
  } catch (...) {
    return HostdServeResult::rejected;
  }
}

std::int64_t hostd_monotonic_now_ns() {
  timespec now{};
  if (::clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    throw_errno("could not read hostd monotonic clock");
  if (now.tv_sec >
      std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL)
    throw HostdTransportError("hostd monotonic clock overflowed");
  return now.tv_sec * 1'000'000'000LL + now.tv_nsec;
}

HostLedgerTime HostdLinuxLedgerTimeSource::now() {
  // Sample boottime after realtime so the durable operation is never stamped
  // earlier than the beginning of the host-side time observation.
  const std::int64_t wall = checked_clock_ns(CLOCK_REALTIME, "wall");
  const std::int64_t boot = checked_clock_ns(CLOCK_BOOTTIME, "boottime");
  return {.boottime_ns = boot, .wall_time_ns = wall};
}

std::vector<std::byte>
hostd_encode_status_request(std::uint64_t correlation_id) {
  try {
    return encode_packet(kStatusRequestOpcode, correlation_id,
                         {{"api_version", kHostdStatusTransportApiVersion}});
  } catch (const HostdTransportError &) {
    throw;
  } catch (const std::exception &) {
    throw HostdTransportError("hostd request encoding failed");
  } catch (...) {
    throw HostdTransportError("hostd request encoding failed");
  }
}

namespace {

void connect_until(int descriptor, const sockaddr_un &address,
                   socklen_t address_length,
                   std::int64_t absolute_monotonic_deadline_ns) {
  for (;;) {
    if (absolute_monotonic_deadline_ns <= hostd_monotonic_now_ns())
      throw HostdTransportError("hostd status connect deadline expired");
    if (::connect(descriptor, reinterpret_cast<const sockaddr *>(&address),
                  address_length) == 0 || errno == EISCONN)
      return;
    const int connect_error = errno;
    if (connect_error == EINTR)
      continue;
    if (connect_error == EAGAIN || connect_error == EWOULDBLOCK) {
      if (!wait_ready(descriptor, POLLOUT,
                      absolute_monotonic_deadline_ns))
        throw HostdTransportError("hostd status connect deadline expired");
      continue;
    }
    if (connect_error != EINPROGRESS && connect_error != EALREADY) {
      errno = connect_error;
      throw_errno("could not connect to hostd status socket");
    }
    if (!wait_ready(descriptor, POLLOUT, absolute_monotonic_deadline_ns))
      throw HostdTransportError("hostd status connect deadline expired");
    int socket_error = 0;
    socklen_t error_length = sizeof(socket_error);
    for (;;) {
      if (::getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &socket_error,
                       &error_length) == 0)
        break;
      if (errno != EINTR)
        throw_errno("could not inspect hostd connect outcome");
      if (absolute_monotonic_deadline_ns <= hostd_monotonic_now_ns())
        throw HostdTransportError("hostd status connect deadline expired");
    }
    if (error_length != sizeof(socket_error))
      throw HostdTransportError("hostd connect outcome is malformed");
    if (socket_error == 0)
      return;
    if (socket_error == EINPROGRESS || socket_error == EALREADY ||
        socket_error == EAGAIN || socket_error == EWOULDBLOCK)
      continue;
    errno = socket_error;
    throw_errno("hostd status connect failed");
  }
}

HostdStatusReply
request_status_impl(const HostdStatusClientConfig &config,
                    std::uint64_t correlation_id,
                    std::int64_t absolute_monotonic_deadline_ns) {
  if (config.maximum_payload_bytes == 0U ||
      config.maximum_payload_bytes > kHostdStatusMaximumPayloadBytes)
    throw HostdTransportError("hostd client payload limit is invalid");
  if (inspect_client_endpoint(config) != config.expected_endpoint)
    throw HostdTransportError("hostd client endpoint identity is inexact");
  FileDescriptor connection(
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (connection.get() < 0)
    throw_errno("could not create hostd status client socket");
  enable_passcred(connection.get());
  socklen_t address_length = 0U;
  const sockaddr_un address =
      socket_address(config.socket_path, address_length);
  connect_until(connection.get(), address, address_length,
                absolute_monotonic_deadline_ns);
  const ucred peer = peer_credentials(connection.get());
  if (peer.uid != config.expected_server_uid ||
      peer.gid != config.expected_server_gid)
    throw HostdTransportError("hostd server credentials are inexact");
  if (inspect_client_endpoint(config) != config.expected_endpoint)
    throw HostdTransportError(
        "hostd client endpoint changed during connection");
  send_packet(connection.get(), hostd_encode_status_request(correlation_id),
              absolute_monotonic_deadline_ns);
  const ReceivedPacket received = receive_packet(
      connection.get(), peer, config.maximum_payload_bytes,
      absolute_monotonic_deadline_ns);
  const WirePacket response =
      decode_packet(received.bytes, config.maximum_payload_bytes);
  if (response.correlation_id != correlation_id)
    throw HostdTransportError("hostd response correlation is inexact");
  const nlohmann::json payload = parse_canonical_json(response.payload);
  if (response.opcode == kStatusResponseOpcode) {
    require_fields(payload, {"api_version", "authority_status", "status"});
    if (payload.at("api_version").get<std::string>() !=
        kHostdStatusTransportApiVersion)
      throw HostdTransportError("hostd status response API is unsupported");
    HostdCoordinatorStatus status =
        parse_validated_status(payload.at("status"));
    std::optional<HostdAuthorityStatus> authority_status;
    if (!payload.at("authority_status").is_null())
      authority_status = parse_validated_authority_status(
          payload.at("authority_status"), status);
    return {.kind = HostdStatusReplyKind::status,
            .correlation_id = response.correlation_id,
            .status = std::move(status),
            .authority_status = std::move(authority_status),
            .error = std::nullopt};
  }
  if (response.opcode == kErrorResponseOpcode) {
    require_fields(payload, {"api_version", "code", "message"});
    if (payload.at("api_version").get<std::string>() !=
        kHostdStatusTransportApiVersion)
      throw HostdTransportError("hostd error response API is unsupported");
    const std::string code = payload.at("code").get<std::string>();
    const std::string message = payload.at("message").get<std::string>();
    if (!valid_identifier(code) ||
        !valid_bounded_text(message, kMaximumPoisonReasonBytes, false))
      throw HostdTransportError("hostd typed error semantics are invalid");
    return {.kind = HostdStatusReplyKind::error,
            .correlation_id = response.correlation_id,
            .status = std::nullopt,
            .authority_status = std::nullopt,
            .error = HostdTypedError{.code = code, .message = message}};
  }
  throw HostdTransportError("hostd response opcode is unsupported");
}

[[noreturn]] void reject_mutation_error(const WirePacket &packet) {
  const nlohmann::json payload = parse_canonical_json(packet.payload);
  require_fields(payload, {"api_version", "code", "message"});
  if (payload.at("api_version").get<std::string>() !=
      kHostdMutationTransportApiVersion)
    throw HostdTransportError("hostd mutation error API is unsupported");
  const std::string code = payload.at("code").get<std::string>();
  const std::string message = payload.at("message").get<std::string>();
  if (!valid_identifier(code) ||
      !valid_bounded_text(message, kMaximumPoisonReasonBytes, false))
    throw HostdTransportError("hostd mutation error semantics are invalid");
  throw HostdTransportError("hostd mutation rejected: " + code + ": " +
                            message);
}

HostdMutationReply request_mutation_impl(
    const HostdMutationClientConfig &config,
    const HostdMutationRequest &request, std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns) {
  if (config.maximum_payload_bytes == 0U ||
      config.maximum_payload_bytes > kHostdStatusMaximumPayloadBytes)
    throw HostdTransportError("hostd mutation client payload limit is invalid");
  if (inspect_client_endpoint(config) != config.expected_endpoint)
    throw HostdTransportError(
        "hostd mutation client endpoint identity is inexact");
  FileDescriptor connection(
      ::socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (connection.get() < 0)
    throw_errno("could not create hostd mutation client socket");
  enable_passcred(connection.get());
  socklen_t address_length = 0U;
  const sockaddr_un address = socket_address(config.socket_path, address_length);
  connect_until(connection.get(), address, address_length,
                absolute_monotonic_deadline_ns);
  const ucred peer = peer_credentials(connection.get());
  if (peer.uid != config.expected_server_uid ||
      peer.gid != config.expected_server_gid)
    throw HostdTransportError("hostd mutation server credentials are inexact");
  if (inspect_client_endpoint(config) != config.expected_endpoint)
    throw HostdTransportError(
        "hostd mutation endpoint changed during connection");

  send_packet(connection.get(),
              encode_canonical_packet(
                  kMutationOpenOpcode, correlation_id,
                  hostd_mutation_open_canonical_json(request.open)),
              absolute_monotonic_deadline_ns);
  const ReceivedPacket challenge_packet = receive_packet(
      connection.get(), peer, config.maximum_payload_bytes,
      absolute_monotonic_deadline_ns);
  const WirePacket decoded_challenge =
      decode_packet(challenge_packet.bytes, config.maximum_payload_bytes);
  if (decoded_challenge.correlation_id != correlation_id)
    throw HostdTransportError("hostd mutation challenge correlation is inexact");
  if (decoded_challenge.opcode == kErrorResponseOpcode)
    reject_mutation_error(decoded_challenge);
  if (decoded_challenge.opcode != kMutationChallengeOpcode)
    throw HostdTransportError("hostd mutation challenge opcode is unsupported");
  const HostdSessionChallenge challenge =
      hostd_session_challenge_from_canonical_json(decoded_challenge.payload);
  HostdMutationCommand command = seal_hostd_mutation_command({
      .api_version = std::string(kHostdMutationProtocolApiVersion),
      .challenge_response = hostd_session_challenge_response(challenge),
      .mutation = request.mutation,
      .bundle_request = request.bundle_request,
      .release_request = request.release_request,
      .process_prepare = request.process_prepare,
      .process_commit = request.process_commit,
      .process_exit = request.process_exit,
      .command_digest = {},
  });
  validate_hostd_mutation_exchange(request.open, challenge, command);
  std::array<int, 6U> descriptor_storage{};
  std::span<const int> descriptors;
  if (request.mutation == HostdMutationKind::prepare_process) {
    if (!request.process_prepare || !request.delegated_launch)
      throw HostdTransportError(
          "hostd process prepare descriptors are missing");
    const auto& delegated = *request.delegated_launch;
    const bool has_code = delegated.code_fd.has_value();
    const bool has_profiler = delegated.profiler_executable_fd.has_value();
    if (has_profiler != delegated.profiler_authority_fd.has_value() ||
        has_profiler !=
            request.process_prepare->launch.identity.profiler.has_value()) {
      throw HostdTransportError(
          "hostd profiler descriptor presence is inexact");
    }
    const std::size_t base_count = has_code ? 4U : 3U;
    const std::size_t count = base_count + (has_profiler ? 2U : 0U);
    descriptor_storage[0] = delegated.executable_fd;
    const std::size_t working_index = has_code ? 2U : 1U;
    const std::size_t bootstrap_index = has_code ? 3U : 2U;
    descriptor_storage[working_index] =
        delegated.working_directory_fd;
    descriptor_storage[bootstrap_index] = delegated.worker_bootstrap_fd;
    if (has_code) descriptor_storage[1] = *delegated.code_fd;
    if (has_profiler) {
      descriptor_storage[base_count] = *delegated.profiler_executable_fd;
      descriptor_storage[base_count + 1U] = *delegated.profiler_authority_fd;
    }
    descriptors = std::span<const int>(descriptor_storage.data(), count);
    if (request.process_prepare->descriptor_roles.size() != count)
      throw HostdTransportError(
          "hostd process prepare descriptor roles are inexact");
  } else if (request.delegated_launch) {
    throw HostdTransportError(
        "hostd non-prepare mutation carried delegated descriptors");
  }
  send_packet(connection.get(),
              encode_canonical_packet(
                  kMutationCommandOpcode, correlation_id,
                  hostd_mutation_command_canonical_json(command)),
              absolute_monotonic_deadline_ns, descriptors);

  const ReceivedPacket reply_packet = receive_packet(
      connection.get(), peer, config.maximum_payload_bytes,
      absolute_monotonic_deadline_ns);
  const WirePacket decoded_reply =
      decode_packet(reply_packet.bytes, config.maximum_payload_bytes);
  if (decoded_reply.correlation_id != correlation_id)
    throw HostdTransportError("hostd mutation reply correlation is inexact");
  if (decoded_reply.opcode == kErrorResponseOpcode)
    reject_mutation_error(decoded_reply);
  if (decoded_reply.opcode != kMutationReplyOpcode)
    throw HostdTransportError("hostd mutation reply opcode is unsupported");
  HostdMutationReply reply =
      hostd_mutation_reply_from_canonical_json(decoded_reply.payload);
  validate_hostd_mutation_reply(command, reply);
  return reply;
}

} // namespace

HostdStatusReply
hostd_request_status(const HostdStatusClientConfig &config,
                     std::uint64_t correlation_id,
                     std::int64_t absolute_monotonic_deadline_ns) {
  try {
    return request_status_impl(config, correlation_id,
                               absolute_monotonic_deadline_ns);
  } catch (const HostdTransportError &) {
    throw;
  } catch (const std::exception &) {
    throw HostdTransportError("hostd client rejected an internal exception");
  } catch (...) {
    throw HostdTransportError("hostd client rejected an unknown exception");
  }
}

HostdMutationReply hostd_request_mutation(
    const HostdMutationClientConfig &config,
    const HostdMutationRequest &request, std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns) {
  try {
    return request_mutation_impl(config, request, correlation_id,
                                 absolute_monotonic_deadline_ns);
  } catch (const HostdTransportError &) {
    throw;
  } catch (const std::exception &) {
    throw HostdTransportError(
        "hostd mutation client rejected an internal exception");
  } catch (...) {
    throw HostdTransportError(
        "hostd mutation client rejected an unknown exception");
  }
}

} // namespace trainvm
