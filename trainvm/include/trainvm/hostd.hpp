#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <sys/types.h>

#include "trainvm/host_ledger.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdCoordinatorApiVersion =
    "trainvm.hostd-coordinator/v1";
inline constexpr std::string_view kHostdPeerEvidenceApiVersion =
    "trainvm.hostd-peer-evidence/v1";
inline constexpr std::string_view kHostdLogicalFenceEvidenceApiVersion =
    "trainvm.hostd-logical-fence-evidence/v1";

enum class HostdLifecycle {
  sealed,
  startup_auditing,
  startup_blocked,
  admitting,
  poisoned,
};

enum class HostdSessionAccess {
  denied,
  read_only,
  release_only,
  grant_release,
};

enum class HostdPeerEnforcementGrade {
  observed_only,
  service_identity_enforced,
};

struct HostdCoordinatorConfig final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::size_t maximum_live_sessions{1024U};
  std::size_t maximum_logical_scopes{4096U};

  bool operator==(const HostdCoordinatorConfig &) const = default;
};

struct HostdSessionAttribution final {
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};

  bool operator==(const HostdSessionAttribution &) const = default;
};

struct HostdConnectRequest final {
  std::optional<HostdSessionAttribution> attribution;

  bool operator==(const HostdConnectRequest &) const = default;
};

struct HostdLogicalFenceEvidence final {
  std::string api_version;
  HostdSessionAttribution attribution;
  bool live{};
  bool cleanup_authorized{};
  std::string cleanup_allocation_id;
  std::string cleanup_grant_digest;
  std::string cleanup_release_request_digest;
  std::string evidence_digest;

  bool operator==(const HostdLogicalFenceEvidence &) const = default;
};

// This slice requires a fresh, externally owned logical-lease attestation for
// every new grant-capable session and every grant. An in-memory coordinator
// high-water mark is never treated as durable lease authority. The production
// implementation will be backed by the journal authority; tests use a fake.
class IHostdLogicalFenceEvidenceSource {
public:
  virtual ~IHostdLogicalFenceEvidenceSource() = default;
  [[nodiscard]] virtual HostdLogicalFenceEvidence
  attest(const HostdSessionAttribution &attribution) = 0;
};

struct HostdPeerEvidence final {
  std::string api_version;
  uid_t peer_uid{};
  gid_t peer_gid{};
  pid_t peer_pid{};
  std::string service_identity;
  HostdPeerEnforcementGrade enforcement_grade{};
  HostdSessionAccess access{};
  std::string evidence_digest;

  bool operator==(const HostdPeerEvidence &) const = default;
};

class IHostdPeerEvidenceSource {
public:
  virtual ~IHostdPeerEvidenceSource() = default;
  [[nodiscard]] virtual HostdPeerEvidence observe() = 0;
};

struct HostdConnectedSession final {
  std::string session_id;
  HostdPeerEvidence peer;
  HostdSessionAccess effective_access{};
  std::optional<HostdSessionAttribution> attribution;

  bool operator==(const HostdConnectedSession &) const = default;
};

struct HostdCoordinatorStatus final {
  std::string api_version;
  HostdLifecycle lifecycle{};
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string inventory_digest;
  // Retained fake-session records, not a transport-liveness assertion.
  std::size_t live_sessions{};
  std::size_t admission_sessions{};
  std::size_t stale_admission_sessions{};
  std::size_t release_only_sessions{};
  // Session counts use the last admitted/in-operation fence evidence and do
  // not synchronously call external authorities while holding the status
  // mutex. Every actual grant performs a fresh attestation.
  bool admission_counts_are_cached_evidence{true};
  // Inspection-only ledger receipt; never a bearer admission capability.
  std::optional<HostStartupAuditReceipt> startup_audit;
  std::string poison_reason;

  bool operator==(const HostdCoordinatorStatus &) const = default;
};

class HostdError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class HostdStateError final : public HostdError {
public:
  using HostdError::HostdError;
};

class HostdUnauthorized final : public HostdError {
public:
  using HostdError::HostdError;
};

// Narrow callback boundary used by hostd so all ledger verification and
// mutation calls can run outside the coordinator mutex. Production wraps the
// SQLite ledger; deterministic/adversarial tests may provide an exact fake.
class IHostdLedgerBoundary {
public:
  virtual ~IHostdLedgerBoundary() = default;
  [[nodiscard]] virtual bool verify() const = 0;
  [[nodiscard]] virtual HostInventoryReceipt inventory() const = 0;
  [[nodiscard]] virtual HostLedgerChainHead chain_head() const = 0;
  [[nodiscard]] virtual ResourceOccupancySnapshot occupancy() const = 0;
  [[nodiscard]] virtual HostStartupAuditLedgerCommitResult
  commit_startup_audit(const HostStartupAuditReport &report,
                       const HostLedgerTime &now) = 0;
  [[nodiscard]] virtual HostLedgerAdmissionFinalizeResult
  finalize_startup_admission(const HostStartupAuditReport &report,
                             const HostStartupAuditReceipt &receipt,
                             const HostLedgerTime &now) = 0;
  [[nodiscard]] virtual BundleRequestResult
  request_bundle(const ResourceBundleRequest &request,
                 const HostLedgerTime &now,
                 const HostLedgerAdmissionEpoch &admission_epoch) = 0;
  [[nodiscard]] virtual std::optional<BundleRequestResult>
  reconcile_bundle_outcome(const ResourceBundleRequest &request) const = 0;
  [[nodiscard]] virtual BundleReleaseResult
  release_bundle(const ResourceReleaseRequest &request,
                 const HostLedgerTime &now) = 0;
};

// This coordinator is a typed admission layer. The evidence interfaces are
// deliberately abstract/fake in this slice: they do not claim the future
// SOCK_SEQPACKET/SO_PEERCRED plus service-cgroup peer enforcement, a durable
// journal lease reader, or a real device/process-launch enforcement boundary.
// Session IDs remain bounded test capabilities rather than transport-bound
// credentials. The retained peer source is re-observed on every use so this
// abstraction fails closed when its simulated connection/peer disappears.
class HostGrantCoordinator final {
public:
  HostGrantCoordinator(HostdCoordinatorConfig config,
                       std::shared_ptr<SQLiteHostLedger> ledger,
                       std::shared_ptr<IHostdLogicalFenceEvidenceSource>
                           logical_fence_evidence = nullptr);
  HostGrantCoordinator(HostdCoordinatorConfig config,
                       std::shared_ptr<IHostdLedgerBoundary> ledger,
                       std::shared_ptr<IHostdLogicalFenceEvidenceSource>
                           logical_fence_evidence = nullptr);
  ~HostGrantCoordinator();

  HostGrantCoordinator(const HostGrantCoordinator &) = delete;
  HostGrantCoordinator &operator=(const HostGrantCoordinator &) = delete;
  HostGrantCoordinator(HostGrantCoordinator &&) = delete;
  HostGrantCoordinator &operator=(HostGrantCoordinator &&) = delete;

  [[nodiscard]] HostStartupAuditReceipt
  run_startup_audit(IConfiguredHostStartupAuditorV2 &auditor,
                    const HostLedgerTime &now);

  [[nodiscard]] HostdConnectedSession
  connect(HostdConnectRequest request,
          std::shared_ptr<IHostdPeerEvidenceSource> peer_source);
  void disconnect(std::string_view session_id);
  // Observes at most max_observations retained peer sources without holding
  // the coordinator mutex, then conditionally reaps only unchanged, idle dead
  // records. Returns the number of reclaimed session slots.
  [[nodiscard]] std::size_t
  reap_dead_sessions(std::size_t max_observations = 64U);

  [[nodiscard]] BundleRequestResult
  request_bundle(std::string_view session_id,
                 const ResourceBundleRequest &request,
                 const HostLedgerTime &now);
  // Recovers only an already committed exact request outcome. This path is
  // available while sealed or startup-blocked, but requires a freshly
  // observed service-identity session, exact attribution, and well-formed
  // durable logical-fence evidence. A stale exact fence remains recoverable so
  // a lost grant reply can be turned into cleanup; it is never grant authority.
  [[nodiscard]] std::optional<BundleRequestResult>
  reconcile_bundle_outcome(std::string_view session_id,
                           const ResourceBundleRequest &request);
  [[nodiscard]] BundleReleaseResult
  release_bundle(std::string_view session_id,
                 const ResourceReleaseRequest &request,
                 const HostLedgerTime &now);

  // Local status and session-scoped status are both observational. In this
  // fake slice the latter only re-observes the retained peer-evidence source;
  // it does not claim socket ownership or make the string ID a credential.
  [[nodiscard]] HostdCoordinatorStatus status() const;
  [[nodiscard]] HostdCoordinatorStatus
  status(std::string_view session_id) const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

} // namespace trainvm
