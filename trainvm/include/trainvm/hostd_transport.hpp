#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/types.h>

#include "trainvm/hostd.hpp"
#include "trainvm/hostd_linux_session_authority.hpp"
#include "trainvm/hostd_linux_process_authority.hpp"
#include "trainvm/hostd_mutation_protocol.hpp"
#include "trainvm/hostd_startup_controller.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdStatusTransportApiVersion =
    "trainvm.hostd-status-transport/v3";
inline constexpr std::string_view kHostdAuthorityStatusApiVersion =
    "trainvm.hostd-authority-status/v1";
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

// Bounded inspection views copied from hostd's durable ledger and startup
// controller. They intentionally contain no session identifier, bearer
// capability, pidfd, or mutation token. Counts and digests always describe the
// complete authority state; rows are a deterministic bounded prefix and say
// explicitly when they were truncated.
enum class HostdProcessAuthorityPhase {
  launch_intent,
  spawned,
  terminal_pending_release,
};

struct HostdProcessAuthorityStatus final {
  std::string allocation_id;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string launch_id;
  HostdProcessAuthorityPhase phase{};
  std::string cgroup_path;
  std::optional<std::int64_t> host_pid;
  std::optional<std::uint64_t> process_starttime_ticks;
  bool device_policy_intended{};
  bool device_policy_installed{};
  std::string device_policy_digest;
  std::string device_policy_installation_digest;
  bool process_policy_intended{};
  bool process_policy_installed{};
  std::string process_policy_digest;
  std::string process_policy_installation_digest;
  std::optional<bool> cgroup_empty;
  std::optional<bool> accelerator_contexts_empty;
  std::string context_audit_digest;
  std::string terminal_receipt_digest;

  bool operator==(const HostdProcessAuthorityStatus &) const = default;
};

struct HostdPassiveAcceleratorMemory final {
  HostResourceKind resource_kind{};
  HostAcceleratorVendor vendor{};
  std::string stable_id;
  std::optional<std::string> parent_id;
  bool audited_eligible{};
  std::uint64_t total_memory_bytes{};
  std::uint64_t free_memory_bytes{};
  std::map<std::string, std::string> selector_labels;

  bool operator==(const HostdPassiveAcceleratorMemory &) const = default;
};

struct HostdAuthorityStatus final {
  // Eight worst-case rows (4 KiB cgroup paths plus bounded identifiers and
  // digests) still fit the 64 KiB status packet. Complete counts and state
  // digests remain present when the human-readable prefix is truncated.
  static constexpr std::size_t maximum_reported_rows = 8U;
  // Exact identity and memory fields only. This bounded prefix remains safe
  // beside worst-case process/fence prefixes in the 64 KiB status packet.
  static constexpr std::size_t maximum_passive_memory_rows = 32U;
  static constexpr std::size_t maximum_passive_memory_identity_bytes =
      12U * 1024U;

  std::string api_version;
  HostdStartupPhase startup_phase{};
  std::size_t startup_recovery_steps{};
  std::size_t remaining_unclosed_process_records{};
  std::size_t remaining_terminal_release_records{};
  bool ledger_verified{};
  std::string ledger_verification_reason;
  HostLedgerChainHead ledger_chain_head;
  std::uint64_t ledger_record_count{};
  std::uint64_t occupancy_ledger_sequence{};
  std::string occupancy_digest;
  bool resource_inventory_observed{};
  std::uint64_t resource_inventory_observation_age_ns{};
  std::string current_inventory_digest;
  std::string current_inventory_receipt_digest;
  // Separate volatile observation: never part of inventory/topology identity.
  std::string passive_memory_host_id;
  std::string passive_memory_boot_id;
  std::string passive_memory_inventory_digest;
  std::string passive_memory_inventory_receipt_digest;
  std::uint64_t passive_memory_observed_monotonic_ns{};
  std::string passive_memory_observation_digest;
  std::size_t passive_accelerator_memory_count{};
  std::vector<HostdPassiveAcceleratorMemory> passive_accelerator_memory;
  bool passive_accelerator_memory_truncated{};
  std::size_t degraded_resource_count{};
  std::string resource_degradation_reason;
  std::size_t active_fence_count{};
  std::vector<ResourceFence> active_fences;
  bool active_fences_truncated{};
  std::size_t active_process_count{};
  std::vector<HostdProcessAuthorityStatus> active_processes;
  bool active_processes_truncated{};
  bool process_launch_enabled{};
  bool mutation_enabled{};
  std::string mutation_disabled_reason;

  bool operator==(const HostdAuthorityStatus &) const = default;
};

// The passive-memory observation digest, and the size of the evidence it is
// taken over. Both are exported so a status producer and the transport that
// validates it agree by construction: they were previously computed from two
// separately written JSON encodings, which silently disagreed — the validating
// side included an api_version field and hand-built each accelerator row, the
// producing side did neither — so every status carrying passive memory failed
// its own digest check and the connection was closed with no reply.
[[nodiscard]] std::string hostd_passive_memory_observation_digest(
    const HostdAuthorityStatus &status);
[[nodiscard]] std::size_t hostd_passive_memory_identity_bytes(
    const HostdAuthorityStatus &status);

class IHostdAuthorityStatusSource {
public:
  virtual ~IHostdAuthorityStatusSource() = default;
  [[nodiscard]] virtual HostdAuthorityStatus snapshot() const = 0;
};

struct HostdTypedError final {
  std::string code;
  std::string message;
  bool operator==(const HostdTypedError &) const = default;
};

struct HostdStatusReply final {
  HostdStatusReplyKind kind{};
  std::uint64_t correlation_id{};
  std::optional<HostdCoordinatorStatus> status;
  std::optional<HostdAuthorityStatus> authority_status;
  std::optional<HostdTypedError> error;
  bool operator==(const HostdStatusReply &) const = default;
};

struct HostdStatusTransportLimits final {
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
  std::int64_t per_session_timeout_ns{1'000'000'000LL};
  // Optional diagnostic seam for tests/daemon logging. It observes a rejected
  // session after authority checks; it cannot change the fail-closed result.
  std::function<void(std::string_view)> rejection_observer{};
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
                    HostdStatusTransportLimits limits = {},
                    std::shared_ptr<IHostdAuthorityStatusSource>
                        authority_status_source = nullptr);
  [[nodiscard]] HostdServeResult
  serve_one(std::int64_t absolute_monotonic_deadline_ns);
  // Takes ownership of one accepted SOCK_SEQPACKET descriptor. This exists so
  // the unified listener can route the first opcode without duplicating the
  // status protocol implementation.
  [[nodiscard]] HostdServeResult
  serve_accepted(int connection_fd,
                 std::int64_t absolute_monotonic_deadline_ns);

private:
  std::shared_ptr<HostdSocketAuthority> authority_;
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  std::shared_ptr<IHostdAuthorityStatusSource> authority_status_source_;
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

enum class HostdMutationTransportCheckpoint {
  after_challenge_sent,
  after_command_received,
  after_challenge_verified,
  after_coordinator_connected,
  after_dispatch_committed,
  before_reply_send,
  after_reply_send,
};

class IHostdMutationTransportFaultInjector {
public:
  virtual ~IHostdMutationTransportFaultInjector() = default;
  virtual void checkpoint(HostdMutationTransportCheckpoint checkpoint) = 0;
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
  // Test-only deterministic crash-window seam. Production leaves this null.
  std::shared_ptr<IHostdMutationTransportFaultInjector> fault_injector;

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
      HostdMutationTransportConfig config,
      std::shared_ptr<IHostdProcessSupervisor> process_supervisor = nullptr);
  [[nodiscard]] HostdServeResult
  serve_one(std::int64_t absolute_monotonic_deadline_ns);
  // Takes ownership of one accepted SOCK_SEQPACKET descriptor. The first
  // packet remains unread and is still subject to the full mutation protocol.
  [[nodiscard]] HostdServeResult
  serve_accepted(int connection_fd,
                 std::int64_t absolute_monotonic_deadline_ns);

private:
  std::shared_ptr<HostdSocketAuthority> authority_;
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  std::shared_ptr<HostdSessionChallengeVerifier> challenge_verifier_;
  std::shared_ptr<IHostdLinuxSessionKernel> session_kernel_;
  std::shared_ptr<IHostdMutationServiceIdentityAuthority>
      service_identity_authority_;
  std::shared_ptr<IHostdLedgerTimeSource> ledger_time_source_;
  std::shared_ptr<IHostdProcessSupervisor> process_supervisor_;
  HostdMutationTransportConfig config_;
};

// One listener serves both the status handshake and the authenticated
// mutation exchange. It peeks only the fixed wire prefix, consumes no payload
// or ancillary descriptors, and transfers the accepted descriptor to exactly
// one protocol server. This removes accept-order races between independent
// status and mutation loops on the shared configured endpoint.
class HostdUnifiedServer final {
 public:
  HostdUnifiedServer(std::shared_ptr<HostdSocketAuthority> authority,
                     HostdStatusServer& status,
                     HostdMutationServer& mutation);
  [[nodiscard]] HostdServeResult
  serve_one(std::int64_t absolute_monotonic_deadline_ns);

 private:
  std::shared_ptr<HostdSocketAuthority> authority_;
  HostdStatusServer& status_;
  HostdMutationServer& mutation_;
};

struct HostdMutationClientConfig final {
  std::filesystem::path socket_path;
  HostdSocketIdentity expected_endpoint;
  uid_t expected_server_uid{};
  gid_t expected_server_gid{};
  std::size_t maximum_payload_bytes{kHostdStatusMaximumPayloadBytes};
};

struct HostdMutationRequest final {
  struct DelegatedLaunchDescriptors final {
    int executable_fd{-1};
    std::optional<int> code_fd;
    int working_directory_fd{-1};
    int worker_bootstrap_fd{-1};
    std::optional<int> profiler_executable_fd = std::nullopt;
    std::optional<int> profiler_authority_fd = std::nullopt;

    bool operator==(const DelegatedLaunchDescriptors &) const = default;
  };

  HostdMutationOpen open;
  HostdMutationKind mutation{};
  std::optional<ResourceBundleRequest> bundle_request{};
  std::optional<ResourceReleaseRequest> release_request{};
  std::optional<HostdProcessPrepareRequest> process_prepare{};
  std::optional<HostdProcessCommitRequest> process_commit{};
  std::optional<HostdProcessExitCommand> process_exit{};
  // Borrowed only until hostd_request_mutation() returns.
  std::optional<DelegatedLaunchDescriptors> delegated_launch{};

  bool operator==(const HostdMutationRequest &) const = default;
};

[[nodiscard]] HostdMutationReply hostd_request_mutation(
    const HostdMutationClientConfig &config,
    const HostdMutationRequest &request, std::uint64_t correlation_id,
    std::int64_t absolute_monotonic_deadline_ns);

} // namespace trainvm
