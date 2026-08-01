#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/host_ledger_authority.hpp"
#include "trainvm/host_resources.hpp"
#include "trainvm/host_startup_audit.hpp"

namespace trainvm {

inline constexpr std::string_view kHostLedgerGrantApiVersion =
    "trainvm.host-resource-grant/v1";
inline constexpr std::string_view kHostLedgerReleaseRequestApiVersion =
    "trainvm.host-resource-release-request/v1";
inline constexpr std::string_view kHostLedgerReleaseApiVersion =
    "trainvm.host-resource-release/v1";
inline constexpr std::string_view kHostLedgerAdmissionEpochApiVersion =
    "trainvm.host-ledger-admission-epoch/v1";
inline constexpr std::string_view kHostProcessLaunchRequestApiVersion =
    "trainvm.host-process-launch-request/v1";
inline constexpr std::string_view kHostProcessLaunchRequestApiVersionV2 =
    "trainvm.host-process-launch-request/v2";
inline constexpr std::string_view kHostProcessLaunchIntentApiVersion =
    "trainvm.host-process-launch-intent/v1";
inline constexpr std::string_view kHostProcessLaunchIntentApiVersionV2 =
    "trainvm.host-process-launch-intent/v2";
inline constexpr std::string_view kHostProcessSpawnRequestApiVersion =
    "trainvm.host-process-spawn-request/v1";
inline constexpr std::string_view kHostProcessSpawnRequestApiVersionV2 =
    "trainvm.host-process-spawn-request/v2";
inline constexpr std::string_view kHostProcessSpawnReceiptApiVersion =
    "trainvm.host-process-spawn-receipt/v1";
inline constexpr std::string_view kHostProcessSpawnReceiptApiVersionV2 =
    "trainvm.host-process-spawn-receipt/v2";
inline constexpr std::string_view kHostProcessExitRequestApiVersion =
    "trainvm.host-process-exit-request/v1";
inline constexpr std::string_view kHostProcessExitReceiptApiVersion =
    "trainvm.host-process-exit-receipt/v1";
inline constexpr std::string_view kHostProcessRecoveryExitRequestApiVersion =
    "trainvm.host-process-recovery-exit-request/v1";
inline constexpr std::string_view kHostProcessRecoveryExitReceiptApiVersion =
    "trainvm.host-process-recovery-exit-receipt/v1";

struct HostLedgerTime final {
  std::int64_t boottime_ns{};
  std::int64_t wall_time_ns{};

  bool operator==(const HostLedgerTime&) const = default;
};

struct ResourceBundleGrant final {
  std::string api_version;
  std::string allocation_id;
  std::string request_id;
  std::string request_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::vector<ResourceFence> fences;
  std::int64_t granted_boottime_ns{};
  std::int64_t granted_wall_time_ns{};
  std::string previous_receipt_digest;
  std::string receipt_digest;

  bool operator==(const ResourceBundleGrant&) const = default;
};

enum class BundleRequestStatus { granted, busy };

struct BundleRequestResult final {
  BundleRequestStatus status{};
  std::optional<ResourceBundleGrant> grant;
  std::string outcome_digest;
  bool replayed{};

  bool operator==(const BundleRequestResult&) const = default;
};

struct ResourceReleaseRequest final {
  std::string api_version;
  std::string release_request_id;
  std::string allocation_id;
  std::string grant_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string canonical_request_digest;

  bool operator==(const ResourceReleaseRequest&) const = default;
};

struct ResourceReleaseReceipt final {
  std::string api_version;
  std::string release_request_id;
  std::string release_request_digest;
  std::string allocation_id;
  std::string grant_digest;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t released_boottime_ns{};
  std::int64_t released_wall_time_ns{};
  std::string previous_receipt_digest;
  std::string receipt_digest;

  bool operator==(const ResourceReleaseReceipt&) const = default;
};

struct BundleReleaseResult final {
  ResourceReleaseReceipt receipt;
  bool replayed{};

  bool operator==(const BundleReleaseResult&) const = default;
};

// Opaque in-process authority returned only by an exact ledger finalize or its
// exact lost-reply replay. It is intentionally not serializable and is never
// exposed through hostd status. Its digest names persisted inspection state;
// constructing equivalent bytes outside SQLiteHostLedger does not mint this
// C++ capability.
class HostLedgerAdmissionEpoch final {
 public:
  HostLedgerAdmissionEpoch(const HostLedgerAdmissionEpoch&) = default;
  HostLedgerAdmissionEpoch& operator=(const HostLedgerAdmissionEpoch&) = default;
  bool operator==(const HostLedgerAdmissionEpoch&) const = default;

 private:
  friend class SQLiteHostLedger;
  explicit HostLedgerAdmissionEpoch(std::string epoch_digest)
      : epoch_digest_(std::move(epoch_digest)) {}
  std::string epoch_digest_;
};

struct HostLedgerAdmissionFinalizeResult final {
  HostLedgerAdmissionEpoch epoch;
  bool replayed{};

  bool operator==(const HostLedgerAdmissionFinalizeResult&) const = default;
};

// The resolved launch digest commits to the complete launcher input and may be
// a typed compound digest that also binds sealed per-attempt bootstrap data.
// The ledger
// separately persists the security-sensitive executable and cgroup identity so
// a later spawn receipt cannot be attached to a different launch boundary.
struct HostDevicePolicyIntentBinding final {
  std::string policy_digest;
  std::string image_digest;
  std::string program_name;

  bool operator==(const HostDevicePolicyIntentBinding&) const = default;
};

// Exact kernel observation copied into the stopped-child receipt. The
// installation digest uses the same canonical bytes as the Linux device-policy
// installer and binds allocation, launch, cgroup, compiler output, and kernel
// program identity without making the host ledger depend on Linux headers.
struct HostDevicePolicyInstallationBinding final {
  std::string policy_digest;
  std::string image_digest;
  std::uint32_t program_id{};
  std::uint32_t program_type{};
  std::string program_tag;
  std::string program_name;
  std::uint32_t attach_flags{};
  std::string installation_digest;

  bool operator==(const HostDevicePolicyInstallationBinding&) const = default;
};

struct HostProcessLaunchRequest final {
  std::string api_version;
  std::string launch_id;
  std::string allocation_id;
  std::string grant_digest;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::string resolved_launch_digest;
  std::string executable_path;
  std::string executable_digest;
  std::string cgroup_path;
  std::uint64_t cgroup_device{};
  std::uint64_t cgroup_inode{};
  std::optional<HostDevicePolicyIntentBinding> device_policy;
  std::string canonical_request_digest;

  bool operator==(const HostProcessLaunchRequest&) const = default;
};

struct HostProcessLaunchIntent final {
  std::string api_version;
  HostProcessLaunchRequest request;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t intended_boottime_ns{};
  std::int64_t intended_wall_time_ns{};
  std::string previous_process_receipt_digest;
  std::string receipt_digest;

  bool operator==(const HostProcessLaunchIntent&) const = default;
};

struct HostProcessLaunchResult final {
  HostProcessLaunchIntent intent;
  bool replayed{};

  bool operator==(const HostProcessLaunchResult&) const = default;
};

struct HostProcessSpawnRequest final {
  std::string api_version;
  std::string launch_id;
  std::string launch_intent_digest;
  std::int64_t host_pid{};
  std::uint64_t process_starttime_ticks{};
  std::string boot_id;
  std::string cgroup_path;
  std::uint64_t cgroup_device{};
  std::uint64_t cgroup_inode{};
  std::string executable_digest;
  std::optional<HostDevicePolicyInstallationBinding> device_policy;
  std::string canonical_request_digest;

  bool operator==(const HostProcessSpawnRequest&) const = default;
};

struct HostProcessSpawnReceipt final {
  std::string api_version;
  HostProcessSpawnRequest request;
  std::string host_id;
  std::string broker_epoch;
  std::int64_t observed_boottime_ns{};
  std::int64_t observed_wall_time_ns{};
  std::string previous_process_receipt_digest;
  std::string receipt_digest;

  bool operator==(const HostProcessSpawnReceipt&) const = default;
};

struct HostProcessSpawnResult final {
  HostProcessSpawnReceipt receipt;
  bool replayed{};

  bool operator==(const HostProcessSpawnResult&) const = default;
};

struct HostProcessExitRequest final {
  std::string api_version;
  std::string exit_request_id;
  std::string launch_id;
  std::string spawn_receipt_digest;
  std::int64_t host_pid{};
  std::uint64_t process_starttime_ticks{};
  std::int32_t wait_code{};
  std::int32_t wait_status{};
  std::string cgroup_path;
  std::uint64_t cgroup_device{};
  std::uint64_t cgroup_inode{};
  bool cgroup_empty{};
  bool accelerator_contexts_empty{};
  std::string context_audit_digest;
  std::string canonical_request_digest;

  bool operator==(const HostProcessExitRequest&) const = default;
};

struct HostProcessExitReceipt final {
  std::string api_version;
  HostProcessExitRequest request;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t observed_boottime_ns{};
  std::int64_t observed_wall_time_ns{};
  std::string previous_process_receipt_digest;
  std::string receipt_digest;

  bool operator==(const HostProcessExitReceipt&) const = default;
};

struct HostProcessExitResult final {
  HostProcessExitReceipt receipt;
  bool replayed{};

  bool operator==(const HostProcessExitResult&) const = default;
};

// Terminal observations available to a restarted daemon that is no longer the
// worker's parent. None of these claims a waitpid/waitid status.
enum class HostProcessRecoveryExitObservation {
  pidfd_terminal,
  pid_absent,
  identity_superseded,
};

struct HostProcessRecoveryExitRequest final {
  std::string api_version;
  std::string recovery_exit_request_id;
  std::string launch_id;
  std::string spawn_receipt_digest;
  std::int64_t host_pid{};
  std::uint64_t process_starttime_ticks{};
  HostProcessRecoveryExitObservation observation{};
  std::string observation_digest;
  std::string cgroup_path;
  std::uint64_t cgroup_device{};
  std::uint64_t cgroup_inode{};
  bool cgroup_empty{};
  bool accelerator_contexts_empty{};
  std::string context_audit_digest;
  std::string canonical_request_digest;

  bool operator==(const HostProcessRecoveryExitRequest&) const = default;
};

struct HostProcessRecoveryExitReceipt final {
  std::string api_version;
  HostProcessRecoveryExitRequest request;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t observed_boottime_ns{};
  std::int64_t observed_wall_time_ns{};
  std::string previous_process_receipt_digest;
  std::string receipt_digest;

  bool operator==(const HostProcessRecoveryExitReceipt&) const = default;
};

struct HostProcessRecoveryExitResult final {
  HostProcessRecoveryExitReceipt receipt;
  bool replayed{};

  bool operator==(const HostProcessRecoveryExitResult&) const = default;
};

// Read-only, exact persisted evidence for a process boundary that has no
// terminal exit receipt. A missing spawn means the durable intent committed
// before any stopped-child receipt; it is cleanup evidence, not a live PID.
struct HostProcessRecoveryRecord final {
  ResourceBundleGrant grant;
  HostProcessLaunchIntent intent;
  std::optional<HostProcessSpawnReceipt> spawn;

  bool operator==(const HostProcessRecoveryRecord&) const = default;
};

// Exact active-allocation evidence for a process whose terminal receipt is
// durable but whose cgroup cleanup and bundle release may have been interrupted.
// Exactly one terminal receipt is present.
struct HostProcessTerminalReleaseRecord final {
  ResourceBundleGrant grant;
  HostProcessLaunchIntent intent;
  HostProcessSpawnReceipt spawn;
  std::optional<HostProcessExitReceipt> child_exit;
  std::optional<HostProcessRecoveryExitReceipt> recovery_exit;

  bool operator==(const HostProcessTerminalReleaseRecord&) const = default;
};

enum class HostLedgerFaultPoint {
  after_startup_audit_migration_schema,
  after_startup_audit_record,
  after_startup_audit_projection,
  after_startup_audit_commit,
  after_admission_finalize_commit,
  after_request_record,
  after_generation_update,
  after_grant_projection,
  before_commit,
  after_release_record,
  after_process_intent_record,
  after_process_intent_projection,
  after_process_intent_commit,
  after_process_spawn_record,
  after_process_spawn_projection,
  after_process_spawn_commit,
  after_process_exit_record,
  after_process_exit_projection,
  after_process_exit_commit,
  after_process_recovery_exit_record,
  after_process_recovery_exit_projection,
  after_process_recovery_exit_commit,
};

class IHostLedgerFaultInjector {
 public:
  virtual ~IHostLedgerFaultInjector() = default;
  virtual void hit(HostLedgerFaultPoint point) = 0;
};

class HostLedgerError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class HostLedgerConflict final : public HostLedgerError {
 public:
  using HostLedgerError::HostLedgerError;
};

class HostLedgerBusy final : public HostLedgerError {
 public:
  using HostLedgerError::HostLedgerError;
};

class SQLiteHostLedger final {
 public:
  explicit SQLiteHostLedger(
      std::shared_ptr<HostLedgerFilesystemAuthority> authority,
      HostInventoryReceipt inventory,
      IHostLedgerFaultInjector* fault_injector = nullptr,
      std::optional<HostStartupAuditPolicy> trusted_startup_audit_policy =
          std::nullopt);
  ~SQLiteHostLedger();

  SQLiteHostLedger(const SQLiteHostLedger&) = delete;
  SQLiteHostLedger& operator=(const SQLiteHostLedger&) = delete;
  SQLiteHostLedger(SQLiteHostLedger&&) = delete;
  SQLiteHostLedger& operator=(SQLiteHostLedger&&) = delete;

  [[nodiscard]] BundleRequestResult request_bundle(
      const ResourceBundleRequest& request, const HostLedgerTime& now);
  [[nodiscard]] BundleRequestResult request_bundle(
      const ResourceBundleRequest& request, const HostLedgerTime& now,
      const HostLedgerAdmissionEpoch& admission_epoch);
  // Inspection-only lost-reply recovery. This never consults or advances the
  // active admission epoch and never creates request evidence, occupancy, or
  // generations. A missing outcome is therefore distinct from a busy outcome.
  [[nodiscard]] std::optional<BundleRequestResult> reconcile_bundle_outcome(
      const ResourceBundleRequest& request) const;
  [[nodiscard]] BundleReleaseResult release_bundle(
      const ResourceReleaseRequest& request, const HostLedgerTime& now);
  // This operation exists only when construction retained a trusted policy.
  // Its return value and every decoded report/receipt remain inspection data;
  // this API does not mint an admission capability.
  [[nodiscard]] HostStartupAuditLedgerCommitResult commit_startup_audit(
      const HostStartupAuditReport& report, const HostLedgerTime& now);
  [[nodiscard]] HostLedgerAdmissionFinalizeResult finalize_startup_admission(
      const HostStartupAuditReport& report,
      const HostStartupAuditReceipt& receipt, const HostLedgerTime& now);
  [[nodiscard]] HostProcessLaunchResult commit_process_launch_intent(
      const HostProcessLaunchRequest& request, const HostLedgerTime& now);
  [[nodiscard]] HostProcessSpawnResult commit_process_spawn(
      const HostProcessSpawnRequest& request, const HostLedgerTime& now);
  [[nodiscard]] HostProcessExitResult commit_process_exit(
      const HostProcessExitRequest& request, const HostLedgerTime& now);
  [[nodiscard]] HostProcessRecoveryExitResult commit_process_recovery_exit(
      const HostProcessRecoveryExitRequest& request,
      const HostLedgerTime& now);
  [[nodiscard]] std::vector<HostProcessRecoveryRecord>
  active_process_recovery_records(
      std::size_t maximum_records =
          HostResourceBounds::maximum_active_fences) const;
  [[nodiscard]] std::vector<HostProcessTerminalReleaseRecord>
  active_terminal_process_release_records(
      std::size_t maximum_records =
          HostResourceBounds::maximum_active_fences) const;
  [[nodiscard]] HostLedgerChainHead chain_head() const;
  [[nodiscard]] ResourceOccupancySnapshot occupancy() const;
  [[nodiscard]] std::uint64_t generation(
      const HostResourceId& resource) const;
  [[nodiscard]] std::uint64_t record_count() const;
  [[nodiscard]] bool verify(std::string* reason = nullptr) const;
  [[nodiscard]] HostInventoryReceipt inventory() const;

 private:
  [[nodiscard]] BundleRequestResult request_bundle_authorized(
      const ResourceBundleRequest& request, const HostLedgerTime& now,
      const HostLedgerAdmissionEpoch* admission_epoch);
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

[[nodiscard]] ResourceReleaseRequest seal_resource_release_request(
    ResourceReleaseRequest request);
[[nodiscard]] nlohmann::json resource_release_request_json(
    const ResourceReleaseRequest& request);
[[nodiscard]] ResourceReleaseRequest resource_release_request_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json bundle_request_result_json(
    const BundleRequestResult& result);
[[nodiscard]] BundleRequestResult bundle_request_result_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json resource_bundle_grant_json(
    const ResourceBundleGrant& grant);
[[nodiscard]] ResourceBundleGrant resource_bundle_grant_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json resource_release_receipt_json(
    const ResourceReleaseReceipt& receipt);
[[nodiscard]] ResourceReleaseReceipt resource_release_receipt_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json bundle_release_result_json(
    const BundleReleaseResult& result);
[[nodiscard]] BundleReleaseResult bundle_release_result_from_json(
    const nlohmann::json& source);
[[nodiscard]] HostProcessLaunchRequest seal_host_process_launch_request(
    HostProcessLaunchRequest request);
[[nodiscard]] nlohmann::json host_process_launch_request_json(
    const HostProcessLaunchRequest& request);
[[nodiscard]] HostProcessLaunchRequest host_process_launch_request_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json host_process_launch_intent_json(
    const HostProcessLaunchIntent& intent);
[[nodiscard]] HostProcessLaunchIntent host_process_launch_intent_from_json(
    const nlohmann::json& source);
[[nodiscard]] HostProcessSpawnRequest seal_host_process_spawn_request(
    HostProcessSpawnRequest request);
[[nodiscard]] nlohmann::json host_process_spawn_request_json(
    const HostProcessSpawnRequest& request);
[[nodiscard]] HostProcessSpawnRequest host_process_spawn_request_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json host_process_spawn_receipt_json(
    const HostProcessSpawnReceipt& receipt);
[[nodiscard]] HostProcessSpawnReceipt host_process_spawn_receipt_from_json(
    const nlohmann::json& source);
[[nodiscard]] std::string host_device_policy_installation_digest(
    std::string_view allocation_id, std::string_view launch_id,
    std::string_view cgroup_path, std::uint64_t cgroup_device,
    std::uint64_t cgroup_inode,
    const HostDevicePolicyInstallationBinding& installation);
[[nodiscard]] HostProcessExitRequest seal_host_process_exit_request(
    HostProcessExitRequest request);
[[nodiscard]] nlohmann::json host_process_exit_request_json(
    const HostProcessExitRequest& request);
[[nodiscard]] HostProcessExitRequest host_process_exit_request_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json host_process_exit_receipt_json(
    const HostProcessExitReceipt& receipt);
[[nodiscard]] HostProcessExitReceipt host_process_exit_receipt_from_json(
    const nlohmann::json& source);
[[nodiscard]] HostProcessRecoveryExitRequest
seal_host_process_recovery_exit_request(HostProcessRecoveryExitRequest request);
[[nodiscard]] nlohmann::json host_process_recovery_exit_request_json(
    const HostProcessRecoveryExitRequest& request);
[[nodiscard]] HostProcessRecoveryExitRequest
host_process_recovery_exit_request_from_json(const nlohmann::json& source);
[[nodiscard]] nlohmann::json host_process_recovery_exit_receipt_json(
    const HostProcessRecoveryExitReceipt& receipt);
[[nodiscard]] HostProcessRecoveryExitReceipt
host_process_recovery_exit_receipt_from_json(const nlohmann::json& source);

}  // namespace trainvm
