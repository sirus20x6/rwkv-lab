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
inline constexpr std::string_view kHostProcessLaunchIntentApiVersion =
    "trainvm.host-process-launch-intent/v1";
inline constexpr std::string_view kHostProcessSpawnRequestApiVersion =
    "trainvm.host-process-spawn-request/v1";
inline constexpr std::string_view kHostProcessSpawnReceiptApiVersion =
    "trainvm.host-process-spawn-receipt/v1";

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

// The resolved launch digest commits to the complete launcher input (including
// argv/environment material that may be inappropriate to persist). The ledger
// separately persists the security-sensitive executable and cgroup identity so
// a later spawn receipt cannot be attached to a different launch boundary.
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

}  // namespace trainvm
