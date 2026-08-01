#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

#include "trainvm/hostd.hpp"
#include "trainvm/hostd_linux_session_authority.hpp"
#include "trainvm/hostd_mutation_protocol.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdStatusTransportApiVersion =
    "trainvm.hostd-status-transport/v2";
inline constexpr std::string_view kHostdMutationTransportApiVersion =
    "trainvm.hostd-mutation-transport/v1";
inline constexpr std::uint16_t kHostdStatusWireVersion = 2U;
inline constexpr std::size_t kHostdStatusWireHeaderBytes = 56U;
inline constexpr std::size_t kHostdStatusMaximumPayloadBytes = 64U * 1024U;

class HostdTransportError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class IHostdSingletonToken {
public:
  virtual ~IHostdSingletonToken() = default;
  // This is an explicit integration seam, not a socket-path singleton claim.
  // Production passes the already-held durable host-ledger authority token.
  [[nodiscard]] virtual bool attest_held() const = 0;
};

enum class HostdSocketBindCheckpoint {
  signals_blocked,
  before_identity_capture,
  identity_captured,
  before_cwd_restore,
  before_socket_protection,
  before_listen,
};

// Deterministic startup-only fault seam. Production leaves this null. Throws
// after identity capture are normalized to HostdTransportError and rolled
// back exactly. Throws before identity capture or cwd restoration are
// deliberately fail-stop because safe recovery cannot be proven.
class IHostdSocketBindFaultInjector {
public:
  virtual ~IHostdSocketBindFaultInjector() = default;
  virtual void checkpoint(HostdSocketBindCheckpoint checkpoint) = 0;
};

enum class HostdSocketEnforcementGrade { cooperative_test };

struct HostdSocketAuthorityConfig final {
  std::string api_version;
  std::filesystem::path socket_path;
  uid_t expected_owner_uid{};
  gid_t expected_owner_gid{};
  std::uint32_t expected_parent_mode{};
  std::uint32_t expected_socket_mode{};
  std::size_t listen_backlog{64U};
  HostdSocketEnforcementGrade enforcement_grade{
      HostdSocketEnforcementGrade::cooperative_test};
  std::shared_ptr<IHostdSocketBindFaultInjector> fault_injector;

  bool operator==(const HostdSocketAuthorityConfig &) const = default;
};

struct HostdSocketIdentity final {
  std::uint64_t parent_device{};
  std::uint64_t parent_inode{};
  std::uint32_t parent_mode{};
  std::uint32_t parent_owner_uid{};
  std::uint32_t parent_owner_gid{};
  std::uint64_t path_device{};
  std::uint64_t path_inode{};
  std::uint32_t path_mode{};
  std::uint32_t owner_uid{};
  std::uint32_t owner_gid{};
  std::uint64_t link_count{};

  bool operator==(const HostdSocketIdentity &) const = default;
};

// Owns a validated listener and pins the trusted parent and socket pathname.
// This slice is cooperative_test only. Because AF_UNIX has no bindat(),
// self_bind is a single-threaded-startup operation: it blocks signals, changes
// cwd to the pinned parent, binds only the canonical basename, restores cwd,
// and re-proves the configured absolute parent. It retains a separately held
// singleton token and never removes a pre-existing pathname.
// Trusted systemd activation and descriptor-to-path attestation are future
// work and deliberately absent. Move operations require external exclusion;
// ordinary observations are internally synchronized.
class HostdSocketAuthority final {
public:
  struct Implementation;
  static HostdSocketAuthority
  self_bind(HostdSocketAuthorityConfig config, int pinned_parent_fd,
            std::shared_ptr<IHostdSingletonToken> singleton);

  ~HostdSocketAuthority();
  HostdSocketAuthority(HostdSocketAuthority &&) noexcept;
  HostdSocketAuthority &operator=(HostdSocketAuthority &&) noexcept;
  HostdSocketAuthority(const HostdSocketAuthority &) = delete;
  HostdSocketAuthority &operator=(const HostdSocketAuthority &) = delete;

  [[nodiscard]] HostdSocketIdentity reattest();
  [[nodiscard]] bool poisoned() const;
  [[nodiscard]] std::string poison_reason() const;
  [[nodiscard]] const std::filesystem::path &socket_path() const;
  [[nodiscard]] int listener_fd() const noexcept;

private:
  explicit HostdSocketAuthority(
      std::unique_ptr<Implementation> implementation) noexcept;
  std::unique_ptr<Implementation> implementation_;
};

enum class HostdStatusReplyKind { status, error };

struct HostdTypedError final {
  std::string code;
  std::string message;
  bool operator==(const HostdTypedError &) const = default;
};

struct HostdStatusReply final {
  HostdStatusReplyKind kind{};
  std::uint64_t correlation_id{};
  std::optional<HostdCoordinatorStatus> status;
  std::optional<HostdTypedError> error;
  bool operator==(const HostdStatusReply &) const = default;
};

struct HostdStatusTransportLimits final {
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
  std::int64_t per_session_timeout_ns{1'000'000'000LL};
};

struct HostdStatusPeerPolicy final {
  uid_t allowed_uid{};
  gid_t allowed_gid{};
};

enum class HostdServeResult { served, rejected, timed_out };

class HostdStatusServer final {
public:
  // Status-only cooperative boundary. UID/GID and SCM_CREDENTIALS prevent
  // accidental cross-identity/delegated-fd use, not hostile same-UID access.
  // cgroups, pidfds, systemd activation, journal challenge, mutation RPCs,
  // stale cleanup, singleton continuity, and launch remain out of scope.
  HostdStatusServer(std::shared_ptr<HostdSocketAuthority> authority,
                    std::shared_ptr<HostGrantCoordinator> coordinator,
                    HostdStatusPeerPolicy peer_policy,
                    HostdStatusTransportLimits limits = {});
  [[nodiscard]] HostdServeResult
  serve_one(std::int64_t absolute_monotonic_deadline_ns);

private:
  std::shared_ptr<HostdSocketAuthority> authority_;
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  HostdStatusPeerPolicy peer_policy_;
  HostdStatusTransportLimits limits_;
};

struct HostdStatusClientConfig final {
  std::filesystem::path socket_path;
  HostdSocketIdentity expected_endpoint;
  uid_t expected_server_uid{};
  gid_t expected_server_gid{};
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
};

[[nodiscard]] std::int64_t hostd_monotonic_now_ns();
[[nodiscard]] std::vector<std::byte>
hostd_encode_status_request(std::uint64_t correlation_id);
[[nodiscard]] HostdStatusReply
hostd_request_status(const HostdStatusClientConfig &config,
                     std::uint64_t correlation_id,
                     std::int64_t absolute_monotonic_deadline_ns);

// Authorization returned by a host-owned service identity authority after it
// has inspected the socket-bound process instance. The request payload never
// names its own service identity or access. Production implementations must
// prove service/cgroup membership outside the caller's namespace; tests may
// use a cooperative implementation only when the transport config says so.
struct HostdMutationServiceAuthorization final {
  std::string service_identity;
  HostdSessionAccess access{};
  bool service_identity_enforced{};

  bool operator==(const HostdMutationServiceAuthorization &) const = default;
};

class IHostdMutationServiceIdentityAuthority {
public:
  virtual ~IHostdMutationServiceIdentityAuthority() = default;
  [[nodiscard]] virtual HostdMutationServiceAuthorization
  authorize(const HostdSocketPeerInstance &peer) = 0;
};

class IHostdLedgerTimeSource {
public:
  virtual ~IHostdLedgerTimeSource() = default;
  // Must return host-observed CLOCK_BOOTTIME and wall clock values. No client
  // timestamp participates in a grant or release.
  [[nodiscard]] virtual HostLedgerTime now() = 0;
};

class HostdLinuxLedgerTimeSource final : public IHostdLedgerTimeSource {
public:
  [[nodiscard]] HostLedgerTime now() override;
};

enum class HostdMutationTransportEnforcementGrade {
  cooperative_test,
  strict_service_identity,
};

struct HostdMutationTransportConfig final {
  std::string api_version{std::string(kHostdMutationTransportApiVersion)};
  uid_t allowed_uid{};
  gid_t allowed_gid{};
  HostdLinuxSessionEnforcementGrade socket_peer_grade{
      HostdLinuxSessionEnforcementGrade::cooperative_namespace_observation};
  HostdMutationTransportEnforcementGrade enforcement_grade{
      HostdMutationTransportEnforcementGrade::cooperative_test};
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
  std::int64_t per_session_timeout_ns{5'000'000'000LL};

  bool operator==(const HostdMutationTransportConfig &) const = default;
};

// Multi-packet, one-command connection:
//   mutation-open -> challenge -> sealed-command -> bound reply.
// A command is dispatched only after the accepted socket peer is reobserved,
// its challenge is consumed successfully, and an external service authority
// authorizes that same process instance. Every coordinator session is scoped
// to the connection and disconnected on every exit path.
class HostdMutationServer final {
public:
  HostdMutationServer(
      std::shared_ptr<HostdSocketAuthority> authority,
      std::shared_ptr<HostGrantCoordinator> coordinator,
      std::shared_ptr<HostdSessionChallengeVerifier> challenge_verifier,
      std::shared_ptr<IHostdLinuxSessionKernel> session_kernel,
      std::shared_ptr<IHostdMutationServiceIdentityAuthority>
          service_identity_authority,
      std::shared_ptr<IHostdLedgerTimeSource> ledger_time_source,
      HostdMutationTransportConfig config);
  [[nodiscard]] HostdServeResult
  serve_one(std::int64_t absolute_monotonic_deadline_ns);

private:
  std::shared_ptr<HostdSocketAuthority> authority_;
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  std::shared_ptr<HostdSessionChallengeVerifier> challenge_verifier_;
  std::shared_ptr<IHostdLinuxSessionKernel> session_kernel_;
  std::shared_ptr<IHostdMutationServiceIdentityAuthority>
      service_identity_authority_;
  std::shared_ptr<IHostdLedgerTimeSource> ledger_time_source_;
  HostdMutationTransportConfig config_;
};

struct HostdMutationClientConfig final {
  std::filesystem::path socket_path;
  HostdSocketIdentity expected_endpoint;
  uid_t expected_server_uid{};
  gid_t expected_server_gid{};
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
};

struct HostdMutationRequest final {
  HostdMutationOpen open;
  HostdMutationKind mutation{};
  std::optional<ResourceBundleRequest> bundle_request;
  std::optional<ResourceReleaseRequest> release_request;

  bool operator==(const HostdMutationRequest &) const = default;
};

[[nodiscard]] HostdMutationReply hostd_request_mutation(
    const HostdMutationClientConfig &config,
    const HostdMutationRequest &request, std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns);

} // namespace trainvm
