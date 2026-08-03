#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/sqlite_filesystem_authority.hpp"
#include "trainvm/hostd_journal_fence_attestor.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_service_identity.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/hostd_restart_process_recovery.hpp"
#include "trainvm/hostd_startup_auditor.hpp"
#include "trainvm/hostd_startup_controller.hpp"
#include "trainvm/linux_nvidia_inventory.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdDaemonConfigurationApiVersion =
    "trainvm.hostd-daemon/v1";

class HostdDaemonConfigurationError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct HostdDaemonInventoryDocument final {
  std::size_t maximum_devices{};
  std::size_t maximum_partitions_per_device{};
  std::size_t maximum_processes_per_device{};
  std::uint64_t maximum_capture_duration_ns{};
  std::uint64_t maximum_snapshot_age_ns{};
  bool trusted_host_namespace{};
  bool trusted_nvml_loader{};

  bool operator==(const HostdDaemonInventoryDocument &) const = default;
};

struct HostdDaemonGpuFaultGuardDocument final {
  std::string state_path;
  std::uint64_t maximum_state_age_ns{};

  bool operator==(const HostdDaemonGpuFaultGuardDocument &) const = default;
};

struct HostdDaemonCgroupDocument final {
  std::string root_path;
  std::string root_unified_path;

  bool operator==(const HostdDaemonCgroupDocument &) const = default;
};

struct HostdDaemonWorkerIdentityDocument final {
  std::uint32_t uid{};
  std::uint32_t gid{};
  bool no_new_privileges{};

  bool operator==(const HostdDaemonWorkerIdentityDocument &) const = default;
};

struct HostdDaemonSocketDocument final {
  std::string path;
  std::uint32_t parent_mode{};
  std::uint32_t socket_mode{};
  std::size_t listen_backlog{};

  bool operator==(const HostdDaemonSocketDocument &) const = default;
};

struct HostdDaemonTransportDocument final {
  std::uint32_t allowed_uid{};
  std::uint32_t allowed_gid{};
  std::size_t maximum_payload_bytes{};
  std::int64_t status_session_timeout_ns{};
  std::int64_t mutation_session_timeout_ns{};
  std::int64_t serve_wake_interval_ns{};

  bool operator==(const HostdDaemonTransportDocument &) const = default;
};

struct HostdDaemonChallengeDocument final {
  std::int64_t ttl_ns{};
  std::size_t maximum_outstanding{};
  std::size_t maximum_outstanding_per_peer{};

  bool operator==(const HostdDaemonChallengeDocument &) const = default;
};

struct HostdDaemonStartupDocument final {
  bool require_stable_occupancy{};
  bool fail_on_blocking_findings{};
  std::uint32_t maximum_findings{};
  std::string exact_live_policy;
  bool reconcile_observed_nonlive{};
  std::size_t maximum_recovery_steps{};

  bool operator==(const HostdDaemonStartupDocument &) const = default;
};

struct HostdDaemonServiceRoleDocument final {
  std::string cgroup_path;
  std::string service_identity;
  std::uint32_t expected_uid{};
  std::uint32_t expected_gid{};
  std::string access;

  bool operator==(const HostdDaemonServiceRoleDocument &) const = default;
};

struct HostdDaemonConfigurationDocument final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string broker_instance_id;
  std::uint32_t authority_uid{};
  std::uint32_t authority_gid{};
  HostdDaemonWorkerIdentityDocument worker_identity;
  std::string ledger_path;
  std::string journal_path;
  JournalFileIdentity journal_identity;
  HostdDaemonInventoryDocument inventory;
  std::optional<HostdDaemonGpuFaultGuardDocument> gpu_fault_guard;
  HostdDaemonCgroupDocument cgroup;
  HostdDaemonSocketDocument socket;
  HostdLinuxHostNamespacePolicy host_namespaces;
  std::vector<HostdDaemonServiceRoleDocument> service_roles;
  std::size_t maximum_cgroup_file_bytes{};
  std::size_t maximum_live_sessions{};
  std::size_t maximum_logical_scopes{};
  HostdDaemonTransportDocument transport;
  HostdDaemonChallengeDocument challenge;
  HostdDaemonStartupDocument startup;

  bool operator==(const HostdDaemonConfigurationDocument &) const = default;
};

class HostdDaemonConfiguration final {
public:
  explicit HostdDaemonConfiguration(HostdDaemonConfigurationDocument document);
  static HostdDaemonConfiguration load_file(const std::filesystem::path &path);

  [[nodiscard]] const HostdDaemonConfigurationDocument &
  document() const noexcept;
  [[nodiscard]] SqliteAuthorityConfig ledger_authority() const;
  [[nodiscard]] LinuxNvidiaInventoryConfig inventory() const;
  [[nodiscard]] LinuxCgroupAuthorityConfig cgroup() const;
  [[nodiscard]] LinuxWorkerCredentialSpec worker_credentials() const;
  [[nodiscard]] HostdCoordinatorConfig coordinator() const;
  [[nodiscard]] HostdConfiguredStartupAuditorConfig startup_auditor() const;
  [[nodiscard]] HostdRestartProcessRecoveryConfig restart_recovery() const;
  [[nodiscard]] HostdStartupControllerConfig startup_controller() const;
  [[nodiscard]] HostdSocketAuthorityConfig socket() const;
  [[nodiscard]] HostdLinuxSessionKernelConfig session_kernel() const;
  [[nodiscard]] HostdLinuxServiceIdentityConfig service_identity() const;
  [[nodiscard]] HostdSessionChallengeVerifierConfig challenge() const;
  [[nodiscard]] HostdStatusPeerPolicy status_peer() const;
  [[nodiscard]] HostdStatusTransportLimits status_transport() const;
  [[nodiscard]] HostdMutationTransportConfig mutation_transport() const;
  [[nodiscard]] HostdDynamicJournalFenceAttestorConfig journal_attestor() const;
  [[nodiscard]] HostIdentity journal_host() const;
  [[nodiscard]] std::filesystem::path journal_path() const;
  [[nodiscard]] std::int64_t serve_wake_interval_ns() const noexcept;

private:
  HostdDaemonConfigurationDocument document_;
  HostStartupAuditPolicy startup_policy_;
  HostdExactRecoveredProcessPolicy exact_live_policy_{};
  std::vector<HostdLinuxServiceRole> service_roles_;
};

} // namespace trainvm
