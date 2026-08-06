#include "trainvm/hostd_daemon_configuration.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

HostdDaemonConfigurationDocument document() {
  const std::uint32_t uid = static_cast<std::uint32_t>(::geteuid());
  const std::uint32_t gid = static_cast<std::uint32_t>(::getegid());
  return {
      .api_version = std::string(kHostdDaemonConfigurationApiVersion),
      .host_id = "host-daemon",
      .boot_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      .broker_epoch = "broker-daemon",
      .broker_instance_id = "hostd-instance-daemon",
      .authority_uid = 0U,
      .authority_gid = gid,
      .worker_identity = {.uid = uid == 1000U ? 1001U : 1000U,
                          .gid = gid == 1000U ? 1001U : 1000U,
                          .no_new_privileges = true},
      .ledger_path = "/var/lib/trainvm-hostd/host-ledger.sqlite3",
      .journal_path = "/var/lib/trainvm/journal.sqlite3",
      .journal_identity = {.directory_path = "/var/lib/trainvm",
                           .journal_name = "journal.sqlite3",
                           .authority_name = "journal.sqlite3.authority.lock",
                           .directory_device = 11U,
                           .directory_inode = 12U,
                           .device = 13U,
                           .inode = 14U,
                           .authority_device = 15U,
                           .authority_inode = 16U,
                           .owner_uid = uid},
      .inventory = {.maximum_devices = 16U,
                    .maximum_partitions_per_device = 16U,
                    .maximum_processes_per_device = 4096U,
                    .maximum_capture_duration_ns = 5'000'000'000ULL,
                    .maximum_snapshot_age_ns = 1'000'000'000ULL,
                    .trusted_host_namespace = true,
                    .trusted_nvml_loader = true},
      .gpu_fault_guard = std::nullopt,
      .cgroup = {.root_path = "/sys/fs/cgroup/trainvm-workers.slice",
                 .root_unified_path = "/trainvm-workers.slice"},
      .socket = {.path = "/run/trainvm-hostd/hostd.sock",
                 .parent_mode = 0750U,
                 .socket_mode = 0660U,
                 .listen_backlog = 128U},
      .host_namespaces = {.mount_namespace = {.device = 21U, .inode = 22U},
                          .pid_namespace = {.device = 23U, .inode = 24U},
                          .cgroup_namespace = {.device = 25U, .inode = 26U},
                          .time_namespace = {.device = 27U, .inode = 28U},
                          .time_for_children_namespace = {.device = 29U,
                                                          .inode = 30U}},
      .service_roles = {{.cgroup_path = "/system.slice/trainvm.service",
                         .service_identity = "trainvm-dashboard",
                         .expected_uid = uid,
                         .expected_gid = gid,
                         .access = "grant_release"}},
      .maximum_cgroup_file_bytes = 4096U,
      .maximum_live_sessions = 1024U,
      .maximum_logical_scopes = 4096U,
      .transport = {.allowed_uid = uid,
                    .allowed_gid = gid,
                    .maximum_payload_bytes = 64U * 1024U,
                    .status_session_timeout_ns = 1'000'000'000LL,
                    .mutation_session_timeout_ns = 5'000'000'000LL,
                    .serve_wake_interval_ns = 100'000'000LL},
      .challenge = {.ttl_ns = 5'000'000'000LL,
                    .maximum_outstanding = 256U,
                    .maximum_outstanding_per_peer = 8U},
      .startup = {.require_stable_occupancy = true,
                  .fail_on_blocking_findings = true,
                  .maximum_findings = 128U,
                  .exact_live_policy = "terminate_and_reconcile",
                  .reconcile_observed_nonlive = true,
                  .maximum_recovery_steps = 64U},
  };
}

class TemporaryDocument final {
public:
  explicit TemporaryDocument(std::string text) {
    std::array<char, 64U> pattern{};
    const std::string source = "/tmp/trainvm-hostd-config-XXXXXX";
    std::copy(source.begin(), source.end(), pattern.begin());
    const int descriptor = ::mkstemp(pattern.data());
    if (descriptor < 0)
      throw std::runtime_error("could not create config");
    path_ = pattern.data();
    std::size_t offset = 0U;
    while (offset < text.size()) {
      const ssize_t count =
          ::write(descriptor, text.data() + offset, text.size() - offset);
      if (count <= 0) {
        (void)::close(descriptor);
        throw std::runtime_error("could not write config");
      }
      offset += static_cast<std::size_t>(count);
    }
    if (::fchmod(descriptor, 0600) != 0 || ::close(descriptor) != 0)
      throw std::runtime_error("could not seal config file");
  }

  ~TemporaryDocument() { (void)::unlink(path_.c_str()); }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void valid_document_compiles_every_authority_policy() {
  auto source = document();
  source.gpu_fault_guard = HostdDaemonGpuFaultGuardDocument{
      .state_path = "/run/trainvm-gpu-fault/state.json",
      .maximum_state_age_ns = 10'000'000'000ULL};
  const HostdDaemonConfiguration config(std::move(source));
  require(
      config.ledger_authority().enforcement_grade ==
              HostLedgerEnforcementGrade::strict_privileged_filesystem &&
          config.inventory().trusted_host_namespace &&
          config.inventory().trusted_nvml_loader &&
          config.document().gpu_fault_guard.has_value() &&
          config.document().gpu_fault_guard->maximum_state_age_ns ==
              10'000'000'000ULL &&
          config.cgroup().root_unified_path == "/trainvm-workers.slice" &&
          config.worker_credentials().uid ==
              config.document().worker_identity.uid &&
          config.worker_credentials().gid ==
              config.document().worker_identity.gid &&
          config.worker_credentials().no_new_privileges &&
          config.coordinator().host_id == "host-daemon" &&
          config.startup_auditor().policy.policy_digest.starts_with(
              "sha256:") &&
          config.restart_recovery().exact_live_policy ==
              HostdExactRecoveredProcessPolicy::terminate_and_reconcile &&
          config.restart_recovery().reconcile_observed_nonlive &&
          config.startup_controller().maximum_recovery_steps == 64U &&
          config.socket().socket_path == "/run/trainvm-hostd/hostd.sock" &&
          config.session_kernel().enforcement_grade ==
              HostdLinuxSessionEnforcementGrade::
                  strict_host_namespaces_and_socket_pidfd &&
          config.service_identity().roles.size() == 1U &&
          config.service_identity().roles.front().access ==
              HostdSessionAccess::grant_release &&
          config.mutation_transport().enforcement_grade ==
              HostdMutationTransportEnforcementGrade::strict_service_identity &&
          config.challenge().maximum_outstanding_challenges == 256U &&
          config.journal_attestor().broker_epoch == "broker-daemon" &&
          config.journal_host().boot_id ==
              "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa" &&
          config.serve_wake_interval_ns() == 100'000'000LL,
      "one daemon document compiles all strict authority sub-configs");
}

void root_hostd_accepts_a_separate_journal_service_owner() {
  auto separated = document();
  separated.authority_uid = 0U;
  separated.authority_gid =
      separated.worker_identity.gid == 1002U ? 1003U : 1002U;
  separated.journal_identity.owner_uid = separated.transport.allowed_uid;
  const HostdDaemonConfiguration config(std::move(separated));
  require(config.document().authority_uid == 0U &&
              config.document().journal_identity.owner_uid ==
                  config.document().transport.allowed_uid,
          "root hostd pins the non-root TrainVM journal service identity");

  auto mismatched = document();
  mismatched.journal_identity.owner_uid =
      mismatched.transport.allowed_uid == 1002U ? 1003U : 1002U;
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(mismatched)); },
      "hostd refuses a journal owner unrelated to its admitted TrainVM peer");
}

void unsafe_paths_trust_roles_and_bounds_are_rejected() {
  auto relative = document();
  relative.ledger_path = "host-ledger.sqlite3";
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(relative)); },
      "relative authority paths are forbidden");

  auto untrusted = document();
  untrusted.inventory.trusted_nvml_loader = false;
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(untrusted)); },
      "daemon inventory cannot silently downgrade to observation-only");

  auto role = document();
  role.service_roles.front().access = "denied";
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(role)); },
      "unknown or denied service access is not a configured authority role");

  auto bound = document();
  bound.startup.maximum_recovery_steps = 0U;
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(bound)); },
      "startup recovery must have an explicit nonzero bound");

  auto authority_name = document();
  authority_name.journal_identity.authority_name = "../authority.lock";
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(authority_name)); },
      "journal authority names must be safe basenames");

  auto root_worker = document();
  root_worker.worker_identity.uid = 0U;
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(root_worker)); },
      "training workers can never inherit root identity");

  auto same_worker = document();
  same_worker.worker_identity.uid = same_worker.authority_uid;
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(same_worker)); },
      "training workers must differ from the host authority identity");

  auto socket_name = document();
  socket_name.socket.path =
      "/run/trainvm-hostd/"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.sock";
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(socket_name)); },
      "socket basenames have an explicit portable bound");

  auto relative_fault_state = document();
  relative_fault_state.gpu_fault_guard = HostdDaemonGpuFaultGuardDocument{
      .state_path = "state.json", .maximum_state_age_ns = 10'000'000'000ULL};
  require_throws<HostdDaemonConfigurationError>(
      [&] {
        HostdDaemonConfiguration invalid(std::move(relative_fault_state));
      },
      "GPU fault evidence cannot be redirected through a relative path");

  auto stale_fault_policy = document();
  stale_fault_policy.gpu_fault_guard = HostdDaemonGpuFaultGuardDocument{
      .state_path = "/run/trainvm-gpu-fault/state.json",
      .maximum_state_age_ns = 0U};
  require_throws<HostdDaemonConfigurationError>(
      [&] { HostdDaemonConfiguration invalid(std::move(stale_fault_policy)); },
      "GPU fault evidence must have a bounded freshness policy");
}

void file_loading_is_strict_and_reflection_closed() {
  const auto expected = document();
  TemporaryDocument valid(encode_json(expected).dump());
  const HostdDaemonConfiguration loaded =
      HostdDaemonConfiguration::load_file(valid.path());
  require(loaded.document() == expected,
          "authority-owned JSON round-trips through the reflected schema");

  auto unknown = encode_json(expected);
  unknown["surprise"] = true;
  TemporaryDocument extra(unknown.dump());
  require_throws<HostdDaemonConfigurationError>(
      [&] { (void)HostdDaemonConfiguration::load_file(extra.path()); },
      "unknown daemon configuration keys are rejected");

  TemporaryDocument duplicate("{\"api_version\":\"trainvm.hostd-daemon/v1\","
                              "\"api_version\":\"trainvm.hostd-daemon/v1\"}");
  require_throws<HostdDaemonConfigurationError>(
      [&] { (void)HostdDaemonConfiguration::load_file(duplicate.path()); },
      "duplicate daemon configuration keys are rejected before decoding");
}

} // namespace

int main() {
  try {
    valid_document_compiles_every_authority_policy();
    root_hostd_accepts_a_separate_journal_service_owner();
    unsafe_paths_trust_roles_and_bounds_are_rejected();
    file_loading_is_strict_and_reflection_closed();
    std::cout << "hostd daemon configuration tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd daemon configuration test failure: " << error.what()
              << '\n';
    return 1;
  }
}
