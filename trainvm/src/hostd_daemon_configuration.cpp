#include "trainvm/hostd_daemon_configuration.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <ranges>
#include <set>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/authority_document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::uintmax_t kMaximumDocumentBytes = 256U << 10U;

[[noreturn]] void reject(std::string message) {
  throw HostdDaemonConfigurationError(std::move(message));
}

bool identifier(std::string_view value) {
  return !value.empty() && value.size() <= 192U &&
         std::ranges::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return std::isalnum(byte) != 0 || character == '.' ||
                  character == '_' || character == ':' || character == '/' ||
                  character == '-' || character == '@';
         });
}

bool boot_uuid(std::string_view value) {
  if (value.size() != 36U || value[8U] != '-' || value[13U] != '-' ||
      value[18U] != '-' || value[23U] != '-')
    return false;
  for (std::size_t index = 0U; index < value.size(); ++index) {
    if (index == 8U || index == 13U || index == 18U || index == 23U)
      continue;
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f')))
      return false;
  }
  return true;
}

bool canonical_absolute(std::string_view value) {
  if (value.empty() || value.find('\0') != value.npos)
    return false;
  const std::filesystem::path path(value);
  return path.is_absolute() && path.lexically_normal() == path;
}

bool safe_basename(std::string_view value, std::size_t maximum_bytes) {
  if (value.empty() || value.size() > maximum_bytes || value == "." ||
      value == ".." || value.find('\0') != value.npos ||
      value.find('/') != value.npos)
    return false;
  return std::filesystem::path(value).filename() == value;
}

bool namespace_identity(HostdLinuxNamespaceIdentity value) {
  return value.device != 0U && value.inode != 0U;
}

HostdSessionAccess access(std::string_view value) {
  if (value == "release_only")
    return HostdSessionAccess::release_only;
  if (value == "grant_release")
    return HostdSessionAccess::grant_release;
  reject("hostd service role has an unsupported access policy");
}

nlohmann::json strict_json(std::string text) {
  bool duplicate = false;
  std::vector<std::set<std::string>> keys;
  try {
    const nlohmann::json::parser_callback_t callback =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json &parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (keys.size() <= index + 1U)
              keys.resize(index + 2U);
            keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (keys.size() <= index)
              keys.resize(index + 1U);
            if (!keys[index].insert(parsed.get<std::string>()).second)
              duplicate = true;
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     keys.size() > index + 1U) {
            keys[index + 1U].clear();
          }
          return true;
        };
    nlohmann::json result = nlohmann::json::parse(text, callback);
    if (duplicate)
      reject("hostd daemon document contains a duplicate key");
    return result;
  } catch (const HostdDaemonConfigurationError &) {
    throw;
  } catch (const nlohmann::json::exception &) {
    reject("hostd daemon document is not valid JSON");
  }
}

} // namespace

HostdDaemonConfiguration::HostdDaemonConfiguration(
    HostdDaemonConfigurationDocument document)
    : document_(std::move(document)) {
  if (document_.api_version != kHostdDaemonConfigurationApiVersion ||
      !identifier(document_.host_id) || !boot_uuid(document_.boot_id) ||
      !identifier(document_.broker_epoch) ||
      !identifier(document_.broker_instance_id) ||
      document_.authority_uid >
          static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max()) ||
      document_.authority_gid >
          static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max()) ||
      document_.worker_identity.uid == 0U ||
      document_.worker_identity.gid == 0U ||
      document_.worker_identity.uid == document_.authority_uid ||
      document_.worker_identity.gid == document_.authority_gid ||
      document_.worker_identity.uid >
          static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max()) ||
      document_.worker_identity.gid >
          static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max()) ||
      !document_.worker_identity.no_new_privileges ||
      !canonical_absolute(document_.ledger_path) ||
      !canonical_absolute(document_.journal_path) ||
      document_.ledger_path == document_.journal_path ||
      !canonical_absolute(document_.journal_identity.directory_path) ||
      std::filesystem::path(document_.journal_path).parent_path() !=
          document_.journal_identity.directory_path ||
      std::filesystem::path(document_.journal_path).filename() !=
          document_.journal_identity.journal_name ||
      !safe_basename(document_.journal_identity.journal_name, 128U) ||
      !safe_basename(document_.journal_identity.authority_name, 128U) ||
      document_.journal_identity.directory_device == 0U ||
      document_.journal_identity.directory_inode == 0U ||
      document_.journal_identity.device == 0U ||
      document_.journal_identity.inode == 0U ||
      document_.journal_identity.authority_device == 0U ||
      document_.journal_identity.authority_inode == 0U ||
      document_.journal_identity.owner_uid != document_.authority_uid) {
    reject("hostd daemon authority identity or path is invalid");
  }
  if (document_.inventory.maximum_devices == 0U ||
      document_.inventory.maximum_devices >
          HostResourceBounds::maximum_resources ||
      document_.inventory.maximum_partitions_per_device == 0U ||
      document_.inventory.maximum_partitions_per_device > 64U ||
      document_.inventory.maximum_processes_per_device == 0U ||
      document_.inventory.maximum_processes_per_device > 65536U ||
      document_.inventory.maximum_capture_duration_ns == 0U ||
      document_.inventory.maximum_capture_duration_ns > 60'000'000'000ULL ||
      document_.inventory.maximum_snapshot_age_ns == 0U ||
      document_.inventory.maximum_snapshot_age_ns > 60'000'000'000ULL ||
      !document_.inventory.trusted_host_namespace ||
      !document_.inventory.trusted_nvml_loader) {
    reject("hostd daemon inventory trust or bounds are invalid");
  }
  if (!canonical_absolute(document_.cgroup.root_path) ||
      document_.cgroup.root_unified_path.empty() ||
      document_.cgroup.root_unified_path.front() != '/' ||
      document_.cgroup.root_unified_path.find("..") != std::string::npos ||
      !canonical_absolute(document_.socket.path) ||
      !safe_basename(
          std::filesystem::path(document_.socket.path).filename().string(),
          96U) ||
      document_.socket.path.size() > 107U ||
      (document_.socket.parent_mode != 0700U &&
       document_.socket.parent_mode != 0750U) ||
      (document_.socket.socket_mode != 0600U &&
       document_.socket.socket_mode != 0660U) ||
      document_.socket.listen_backlog == 0U ||
      document_.socket.listen_backlog > 4096U) {
    reject("hostd daemon cgroup or socket policy is invalid");
  }
  const auto &namespaces = document_.host_namespaces;
  if (!namespace_identity(namespaces.mount_namespace) ||
      !namespace_identity(namespaces.pid_namespace) ||
      !namespace_identity(namespaces.cgroup_namespace) ||
      !namespace_identity(namespaces.time_namespace) ||
      !namespace_identity(namespaces.time_for_children_namespace)) {
    reject("hostd daemon host namespace policy is incomplete");
  }
  if (document_.service_roles.empty() || document_.service_roles.size() > 64U ||
      document_.maximum_cgroup_file_bytes == 0U ||
      document_.maximum_cgroup_file_bytes > (1U << 20U) ||
      document_.maximum_live_sessions == 0U ||
      document_.maximum_live_sessions > 65536U ||
      document_.maximum_logical_scopes == 0U ||
      document_.maximum_logical_scopes > 262144U) {
    reject("hostd daemon service/session bounds are invalid");
  }
  std::set<std::string> cgroups;
  std::set<std::string> identities;
  for (const HostdDaemonServiceRoleDocument &role : document_.service_roles) {
    if (!canonical_absolute(role.cgroup_path) ||
        !identifier(role.service_identity) ||
        role.expected_uid >
            static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max()) ||
        role.expected_gid >
            static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max()) ||
        role.expected_uid != document_.transport.allowed_uid ||
        role.expected_gid != document_.transport.allowed_gid ||
        !cgroups.insert(role.cgroup_path).second ||
        !identities.insert(role.service_identity).second) {
      reject("hostd daemon service role is invalid or duplicate");
    }
    service_roles_.push_back(
        {.cgroup_path = role.cgroup_path,
         .service_identity = role.service_identity,
         .expected_uid = static_cast<uid_t>(role.expected_uid),
         .expected_gid = static_cast<gid_t>(role.expected_gid),
         .access = access(role.access)});
  }
  if (document_.transport.allowed_uid >
          static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max()) ||
      document_.transport.allowed_gid >
          static_cast<std::uint64_t>(std::numeric_limits<gid_t>::max()) ||
      document_.transport.maximum_payload_bytes == 0U ||
      document_.transport.maximum_payload_bytes >
          kHostdStatusMaximumPayloadBytes ||
      document_.transport.status_session_timeout_ns < 1'000'000LL ||
      document_.transport.status_session_timeout_ns > 30'000'000'000LL ||
      document_.transport.mutation_session_timeout_ns < 1'000'000LL ||
      document_.transport.mutation_session_timeout_ns > 30'000'000'000LL ||
      document_.transport.serve_wake_interval_ns < 1'000'000LL ||
      document_.transport.serve_wake_interval_ns > 1'000'000'000LL ||
      document_.challenge.ttl_ns < 1'000'000LL ||
      document_.challenge.ttl_ns > 60'000'000'000LL ||
      document_.challenge.maximum_outstanding == 0U ||
      document_.challenge.maximum_outstanding > 65536U ||
      document_.challenge.maximum_outstanding_per_peer == 0U ||
      document_.challenge.maximum_outstanding_per_peer >
          document_.challenge.maximum_outstanding ||
      document_.startup.maximum_findings == 0U ||
      document_.startup.maximum_findings >
          HostStartupAuditBounds::maximum_findings) {
    reject("hostd daemon transport or challenge bounds are invalid");
  }
  startup_policy_ = canonicalize_host_startup_audit_policy({
      .api_version = std::string(kHostStartupAuditPolicyApiVersion),
      .require_stable_occupancy = document_.startup.require_stable_occupancy,
      .fail_on_blocking_findings = document_.startup.fail_on_blocking_findings,
      .maximum_findings = document_.startup.maximum_findings,
      .policy_digest = {},
  });
  if (document_.startup.exact_live_policy == "leave_and_block") {
    exact_live_policy_ = HostdExactRecoveredProcessPolicy::leave_and_block;
  } else if (document_.startup.exact_live_policy == "terminate_and_reconcile") {
    exact_live_policy_ =
        HostdExactRecoveredProcessPolicy::terminate_and_reconcile;
  } else {
    reject("hostd daemon restart recovery policy is invalid");
  }
  if (document_.startup.maximum_recovery_steps == 0U ||
      document_.startup.maximum_recovery_steps > 1'000'000U)
    reject("hostd daemon recovery step bound is invalid");
}

HostdDaemonConfiguration
HostdDaemonConfiguration::load_file(const std::filesystem::path &path) {
  HostdDaemonConfigurationDocument document;
  std::vector<Diagnostic> diagnostics;
  const std::string text = read_authority_document(
      path, "hostd daemon configuration", kMaximumDocumentBytes);
  if (!decode_json(strict_json(text), document, "", diagnostics)) {
    reject("hostd daemon schema validation failed: " +
           diagnostics_json(diagnostics).dump());
  }
  return HostdDaemonConfiguration(std::move(document));
}

const HostdDaemonConfigurationDocument &
HostdDaemonConfiguration::document() const noexcept {
  return document_;
}

SqliteAuthorityConfig HostdDaemonConfiguration::ledger_authority() const {
  return {.api_version = std::string(kSqliteAuthorityApiVersion),
          .ledger_path = document_.ledger_path,
          .expected_owner_uid = static_cast<uid_t>(document_.authority_uid),
          .expected_owner_gid = static_cast<gid_t>(document_.authority_gid),
          .enforcement_grade = SqliteAuthorityEnforcementGrade::strict_filesystem};
}

LinuxNvidiaInventoryConfig HostdDaemonConfiguration::inventory() const {
  return {.api_version = std::string(kLinuxNvidiaInventoryApiVersion),
          .broker_epoch = document_.broker_epoch,
          .maximum_devices = document_.inventory.maximum_devices,
          .maximum_partitions_per_device =
              document_.inventory.maximum_partitions_per_device,
          .maximum_processes_per_device =
              document_.inventory.maximum_processes_per_device,
          .maximum_capture_duration_ns =
              document_.inventory.maximum_capture_duration_ns,
          .maximum_snapshot_age_ns =
              document_.inventory.maximum_snapshot_age_ns,
          .trusted_host_namespace = true,
          .trusted_nvml_loader = true};
}

LinuxCgroupAuthorityConfig HostdDaemonConfiguration::cgroup() const {
  return {.root_path = document_.cgroup.root_path,
          .root_unified_path = document_.cgroup.root_unified_path,
          .expected_owner_uid = static_cast<uid_t>(document_.authority_uid),
          .expected_owner_gid = static_cast<gid_t>(document_.authority_gid)};
}

LinuxWorkerCredentialSpec HostdDaemonConfiguration::worker_credentials() const {
  return {
      .uid = static_cast<uid_t>(document_.worker_identity.uid),
      .gid = static_cast<gid_t>(document_.worker_identity.gid),
      .no_new_privileges = true,
  };
}

HostdCoordinatorConfig HostdDaemonConfiguration::coordinator() const {
  return {.api_version = std::string(kHostdCoordinatorApiVersion),
          .host_id = document_.host_id,
          .boot_id = document_.boot_id,
          .broker_epoch = document_.broker_epoch,
          .maximum_live_sessions = document_.maximum_live_sessions,
          .maximum_logical_scopes = document_.maximum_logical_scopes};
}

HostdConfiguredStartupAuditorConfig
HostdDaemonConfiguration::startup_auditor() const {
  return {.api_version = std::string(kHostdConfiguredStartupAuditorApiVersion),
          .broker_instance_id = document_.broker_instance_id,
          .policy = startup_policy_};
}

HostdRestartProcessRecoveryConfig
HostdDaemonConfiguration::restart_recovery() const {
  return {.exact_live_policy = exact_live_policy_,
          .reconcile_observed_nonlive =
              document_.startup.reconcile_observed_nonlive};
}

HostdStartupControllerConfig
HostdDaemonConfiguration::startup_controller() const {
  return {.api_version = std::string(kHostdStartupControllerApiVersion),
          .maximum_recovery_steps = document_.startup.maximum_recovery_steps};
}

HostdSocketAuthorityConfig HostdDaemonConfiguration::socket() const {
  return {.api_version = std::string(kHostdStatusTransportApiVersion),
          .socket_path = document_.socket.path,
          .expected_owner_uid = static_cast<uid_t>(document_.authority_uid),
          .expected_owner_gid = static_cast<gid_t>(document_.authority_gid),
          .expected_parent_mode = document_.socket.parent_mode,
          .expected_socket_mode = document_.socket.socket_mode,
          .listen_backlog = document_.socket.listen_backlog,
          .enforcement_grade = HostdSocketEnforcementGrade::cooperative_test,
          .fault_injector = nullptr};
}

HostdLinuxSessionKernelConfig HostdDaemonConfiguration::session_kernel() const {
  return {.api_version = std::string(kHostdLinuxSessionAuthorityApiVersion),
          .enforcement_grade = HostdLinuxSessionEnforcementGrade::
              strict_host_namespaces_and_socket_pidfd,
          .expected_host_namespaces = document_.host_namespaces};
}

HostdLinuxServiceIdentityConfig
HostdDaemonConfiguration::service_identity() const {
  return {.api_version = std::string(kHostdLinuxServiceIdentityApiVersion),
          .roles = service_roles_,
          .maximum_cgroup_file_bytes = document_.maximum_cgroup_file_bytes};
}

HostdSessionChallengeVerifierConfig
HostdDaemonConfiguration::challenge() const {
  return {.api_version = std::string(kHostdSessionChallengeApiVersion),
          .host_id = document_.host_id,
          .boot_id = document_.boot_id,
          .broker_epoch = document_.broker_epoch,
          .challenge_ttl_ns = document_.challenge.ttl_ns,
          .maximum_outstanding_challenges =
              document_.challenge.maximum_outstanding,
          .maximum_outstanding_challenges_per_peer =
              document_.challenge.maximum_outstanding_per_peer};
}

HostdStatusPeerPolicy HostdDaemonConfiguration::status_peer() const {
  return {.allowed_uid = static_cast<uid_t>(document_.transport.allowed_uid),
          .allowed_gid = static_cast<gid_t>(document_.transport.allowed_gid)};
}

HostdStatusTransportLimits HostdDaemonConfiguration::status_transport() const {
  return {.maximum_payload_bytes = document_.transport.maximum_payload_bytes,
          .per_session_timeout_ns =
              document_.transport.status_session_timeout_ns};
}

HostdMutationTransportConfig
HostdDaemonConfiguration::mutation_transport() const {
  return {.api_version = std::string(kHostdMutationTransportApiVersion),
          .allowed_uid = static_cast<uid_t>(document_.transport.allowed_uid),
          .allowed_gid = static_cast<gid_t>(document_.transport.allowed_gid),
          .socket_peer_grade = HostdLinuxSessionEnforcementGrade::
              strict_host_namespaces_and_socket_pidfd,
          .enforcement_grade =
              HostdMutationTransportEnforcementGrade::strict_service_identity,
          .maximum_payload_bytes = document_.transport.maximum_payload_bytes,
          .per_session_timeout_ns =
              document_.transport.mutation_session_timeout_ns,
          .fault_injector = nullptr};
}

HostdDynamicJournalFenceAttestorConfig
HostdDaemonConfiguration::journal_attestor() const {
  return {.api_version = std::string(kHostdJournalFenceAttestorApiVersion),
          .host_id = document_.host_id,
          .boot_id = document_.boot_id,
          .broker_epoch = document_.broker_epoch};
}

HostIdentity HostdDaemonConfiguration::journal_host() const {
  return {.host_id = document_.host_id, .boot_id = document_.boot_id};
}

std::filesystem::path HostdDaemonConfiguration::journal_path() const {
  return document_.journal_path;
}

std::int64_t HostdDaemonConfiguration::serve_wake_interval_ns() const noexcept {
  return document_.transport.serve_wake_interval_ns;
}

} // namespace trainvm
