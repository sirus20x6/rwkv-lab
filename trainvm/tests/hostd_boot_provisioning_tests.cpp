#include "trainvm/hostd_boot_provisioning.hpp"
#include "trainvm/hostd_gpu_authorization.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "trainvm/authority_document.hpp"

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

HostdLinuxHostNamespacePolicy namespaces(std::uint64_t base) {
  return {
      .mount_namespace = {.device = base + 1U, .inode = base + 2U},
      .pid_namespace = {.device = base + 3U, .inode = base + 4U},
      .cgroup_namespace = {.device = base + 5U, .inode = base + 6U},
      .time_namespace = {.device = base + 7U, .inode = base + 8U},
      .time_for_children_namespace = {.device = base + 9U,
                                      .inode = base + 10U},
  };
}

// The simulated authority identity. These documents describe a host that is not
// the one running the test — every device, inode and owner in them is invented —
// so they must not be derived from the ambient euid: an unprivileged authority
// is now a validity rule, and deriving from the ambient identity made the whole
// fixture invalid whenever the suite ran as root, as CI does.
constexpr std::uint32_t kAuthorityUid = 1000U;
constexpr std::uint32_t kAuthorityGid = 1000U;
constexpr std::uint32_t kWorkerUid = 1001U;
constexpr std::uint32_t kWorkerGid = 1001U;

HostdDaemonConfigurationDocument daemon_document() {
  constexpr std::uint32_t uid = kAuthorityUid;
  constexpr std::uint32_t gid = kAuthorityGid;
  return {
      .api_version = std::string(kHostdDaemonConfigurationApiVersion),
      .host_id = "host-boot-provisioning",
      .boot_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
      .broker_epoch = "broker-boot-provisioning",
      .broker_instance_id = "hostd-boot-provisioning",
      .authority_uid = uid,
      .authority_gid = gid,
      .worker_identity = {.uid = kWorkerUid,
                          .gid = kWorkerGid,
                          .no_new_privileges = true,
                          .inherit_authority_supplementary_groups =
                              std::nullopt},
      .ledger_path = "/var/lib/trainvm-hostd/boot-ledger.sqlite3",
      .journal_path = "/var/lib/trainvm/boot-journal.sqlite3",
      .journal_identity = {.directory_path = "/var/lib/trainvm",
                           .journal_name = "boot-journal.sqlite3",
                           .authority_name =
                               "boot-journal.sqlite3.authority.lock",
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
      .host_namespaces = namespaces(20U),
      .service_roles = {{.cgroup_path = "/system.slice/trainvm.service",
                         .service_identity = "trainvm-authority",
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

HostdSocketIdentity endpoint(std::uint64_t path_inode) {
  return {
      .parent_device = 41U,
      .parent_inode = 42U,
      .parent_mode = 0750U,
      // The socket is owned by the unprivileged authority that bound it, which
      // is what the published client document pins as the expected server.
      .parent_owner_uid = kAuthorityUid,
      .parent_owner_gid = kAuthorityGid,
      .path_device = 41U,
      .path_inode = path_inode,
      .path_mode = 0660U,
      .owner_uid = kAuthorityUid,
      .owner_gid = kAuthorityGid,
      .link_count = 1U,
  };
}

void simulated_reboot_changes_only_boot_authority() {
  const HostdDaemonConfiguration original(daemon_document());
  const HostdLinuxBootAuthoritySnapshot first{
      .boot_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
      .host_namespaces = namespaces(100U),
  };
  const HostdLinuxBootAuthoritySnapshot second{
      .boot_id = "cccccccc-cccc-cccc-cccc-cccccccccccc",
      .host_namespaces = namespaces(200U),
  };
  const HostdDaemonConfiguration boot_one =
      materialize_hostd_daemon_boot(original, first);
  const HostdDaemonConfiguration boot_two =
      materialize_hostd_daemon_boot(original, second);

  auto expected_one = original.document();
  expected_one.boot_id = first.boot_id;
  expected_one.host_namespaces = first.host_namespaces;
  auto expected_two = original.document();
  expected_two.boot_id = second.boot_id;
  expected_two.host_namespaces = second.host_namespaces;
  require(boot_one.document() == expected_one &&
              boot_two.document() == expected_two,
          "boot materialization may replace only boot and namespace identity");
  require(boot_one.document().journal_identity ==
                  boot_two.document().journal_identity &&
              boot_one.document().service_roles ==
                  boot_two.document().service_roles &&
              boot_one.document().startup == boot_two.document().startup &&
              boot_one.document().socket == boot_two.document().socket,
          "reboot cannot weaken journal, peer, recovery, or socket policy");

  auto malformed = first;
  malformed.boot_id = "not-a-boot-id";
  require_throws<HostdDaemonConfigurationError>(
      [&] { (void)materialize_hostd_daemon_boot(original, malformed); },
      "malformed observed boot identity must fail closed");
  malformed = first;
  malformed.host_namespaces.mount_namespace = {};
  require_throws<HostdDaemonConfigurationError>(
      [&] { (void)materialize_hostd_daemon_boot(original, malformed); },
      "incomplete observed namespace identity must fail closed");
}

void socket_replacement_requires_a_new_client_document() {
  const HostdDaemonConfiguration daemon(daemon_document());
  const HostdClientConfiguration old_client =
      make_hostd_client_configuration(daemon, endpoint(43U),
                                      5'000'000'000LL);
  const HostdClientConfiguration new_client =
      make_hostd_client_configuration(daemon, endpoint(44U),
                                      5'000'000'000LL);
  require(old_client.document().expected_endpoint != endpoint(44U) &&
              new_client.document().expected_endpoint == endpoint(44U) &&
              old_client.document() != new_client.document(),
          "a reboot socket inode replacement must invalidate the old controller policy");
  require(old_client.document().socket_path ==
                  new_client.document().socket_path &&
              old_client.document().expected_server_uid ==
                  daemon.document().authority_uid &&
              old_client.document().expected_server_gid ==
                  daemon.document().authority_gid,
          "endpoint replacement preserves server identity and canonical path");
}

class TemporaryDirectory final {
 public:
  TemporaryDirectory() {
    std::string pattern = "/tmp/trainvm-hostd-boot-publication-XXXXXX";
    if (::mkdtemp(pattern.data()) == nullptr)
      throw std::runtime_error("could not create temporary directory");
    path_ = pattern;
    if (::chmod(path_.c_str(), 0750) != 0)
      throw std::runtime_error("could not protect temporary directory");
  }
  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void gpu_authorization_is_explicit_boot_scoped_and_strict() {
  constexpr std::string_view kGpuA =
      "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
  constexpr std::string_view kGpuB =
      "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
  const HostdDaemonConfiguration daemon(daemon_document());
  const HostdLinuxBootAuthoritySnapshot first{
      .boot_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
      .host_namespaces = namespaces(100U),
  };
  const HostdLinuxBootAuthoritySnapshot second{
      .boot_id = "cccccccc-cccc-cccc-cccc-cccccccccccc",
      .host_namespaces = namespaces(200U),
  };
  const HostdDaemonConfiguration first_runtime =
      materialize_hostd_daemon_boot(daemon, first);
  const HostdDaemonConfiguration second_runtime =
      materialize_hostd_daemon_boot(daemon, second);

  const HostdGpuAuthorization deny = make_hostd_gpu_authorization(
      daemon, first, HostdDisplayGpuPolicy::deny);
  deny.require_matches(first_runtime);
  require(deny.display_policy() == HostdDisplayGpuPolicy::deny &&
              deny.allowed_display_gpu_ids().empty() &&
              deny.document().driver_probe_authorized,
          "explicit deny policy authorizes only the boot-scoped read-only probe");
  require_throws<HostdGpuAuthorizationError>(
      [&] { deny.require_matches(second_runtime); },
      "a reboot must invalidate the prior GPU authorization");

  const HostdGpuAuthorization cooperative = make_hostd_gpu_authorization(
      daemon, first, HostdDisplayGpuPolicy::cooperative_allowlist,
      {std::string(kGpuB), std::string(kGpuA)});
  cooperative.require_matches(first_runtime);
  require(cooperative.allowed_display_gpu_ids() ==
              std::vector<std::string>({std::string(kGpuA),
                                        std::string(kGpuB)}),
          "display authorization must canonicalize its exact UUID allowlist");

  auto tampered = cooperative.document();
  tampered.host_id = "different-host";
  require_throws<HostdGpuAuthorizationError>(
      [&] { (void)HostdGpuAuthorization(tampered); },
      "authorization identity mutation must invalidate its digest");
  require_throws<HostdGpuAuthorizationError>(
      [&] {
        (void)make_hostd_gpu_authorization(
            daemon, first, HostdDisplayGpuPolicy::cooperative_allowlist,
            {std::string(kGpuA), std::string(kGpuA)});
      },
      "duplicate display UUID authority must fail closed");
  require_throws<HostdGpuAuthorizationError>(
      [&] {
        (void)make_hostd_gpu_authorization(
            daemon, first, HostdDisplayGpuPolicy::cooperative_allowlist,
            {"GPU-not-a-canonical-uuid"});
      },
      "malformed display UUID authority must fail closed");
  auto incomplete_boot = first;
  incomplete_boot.host_namespaces.mount_namespace = {};
  require_throws<HostdDaemonConfigurationError>(
      [&] {
        (void)make_hostd_gpu_authorization(
            daemon, incomplete_boot, HostdDisplayGpuPolicy::deny);
      },
      "GPU authorization cannot be minted from incomplete boot evidence");

  TemporaryDirectory directory;
  const AuthorityDocumentPublicationPolicy policy{
      .owner_uid = ::geteuid(),
      .owner_gid = ::getegid(),
      .file_mode = 0600,
      .parent_owner_uid = ::geteuid(),
      .parent_owner_gid = ::getegid(),
  };
  const auto path = directory.path() / "gpu-authorization.json";
  publish_authority_document(
      path, "test GPU authorization",
      hostd_gpu_authorization_json(cooperative), policy,
      kHostdGpuAuthorizationMaximumBytes);
  require(HostdGpuAuthorization::load_file(path).document() ==
              cooperative.document(),
          "published GPU authorization must round-trip exactly");

  std::string duplicate_key = hostd_gpu_authorization_json(cooperative);
  duplicate_key.insert(duplicate_key.find('{') + 1U,
                       "\n  \"api_version\": \"duplicate\",");
  publish_authority_document(path, "test GPU authorization", duplicate_key,
                             policy, kHostdGpuAuthorizationMaximumBytes);
  require_throws<HostdGpuAuthorizationError>(
      [&] { (void)HostdGpuAuthorization::load_file(path); },
      "duplicate JSON fields must fail closed before schema decoding");
}

void atomic_publication_is_exact_and_rejects_unsafe_ancestry() {
  TemporaryDirectory directory;
  const AuthorityDocumentPublicationPolicy policy{
      .owner_uid = ::geteuid(),
      .owner_gid = ::getegid(),
      .file_mode = 0640,
      .parent_owner_uid = ::geteuid(),
      .parent_owner_gid = ::getegid(),
  };
  const auto path = directory.path() / "client.json";
  publish_authority_document(path, "test client document", "first\n",
                             policy, 1024U);
  publish_authority_document(path, "test client document", "second\n",
                             policy, 1024U);
  require(read_authority_document(path, "test client document", 1024U) ==
              "second\n",
          "atomic publication must replace the complete prior document");
  struct stat status {};
  require(::lstat(path.c_str(), &status) == 0 && S_ISREG(status.st_mode) &&
              status.st_nlink == 1 && (status.st_mode & 07777) == 0640 &&
              status.st_uid == ::geteuid() && status.st_gid == ::getegid(),
          "published authority document identity must be exact");

  const HostdDaemonConfiguration daemon(daemon_document());
  const HostdClientConfiguration client =
      make_hostd_client_configuration(daemon, endpoint(43U),
                                      5'000'000'000LL);
  publish_authority_document(path, "test client document",
                             hostd_client_configuration_json(client), policy,
                             kHostdClientConfigurationMaximumBytes);
  require(HostdClientConfiguration::load_file(path).document() ==
              client.document(),
          "published controller document must round-trip exactly");

  const HostdDaemonConfiguration runtime = materialize_hostd_daemon_boot(
      daemon,
      {.boot_id = "dddddddd-dddd-dddd-dddd-dddddddddddd",
       .host_namespaces = namespaces(300U)});
  const auto daemon_path = directory.path() / "hostd.json";
  publish_authority_document(daemon_path, "test daemon document",
                             hostd_daemon_configuration_json(runtime), policy,
                             kHostdDaemonConfigurationMaximumBytes);
  require(HostdDaemonConfiguration::load_file(daemon_path).document() ==
              runtime.document(),
          "published boot runtime document must round-trip exactly");

  std::size_t entries = 0U;
  for (const auto& ignored : std::filesystem::directory_iterator(directory.path())) {
    (void)ignored;
    ++entries;
  }
  require(entries == 2U, "atomic publication cannot leak temporary files");

  require(::chmod(directory.path().c_str(), 0770) == 0,
          "test must make parent unsafe");
  require_throws<std::invalid_argument>(
      [&] {
        publish_authority_document(directory.path() / "unsafe.json",
                                   "unsafe test document", "unsafe\n",
                                   policy, 1024U);
      },
      "group-writable authority parent must be rejected");
  require(::chmod(directory.path().c_str(), 0750) == 0,
          "test must restore protected parent");

  const auto real = directory.path() / "real";
  const auto link = directory.path() / "link";
  std::filesystem::create_directory(real);
  require(::chmod(real.c_str(), 0750) == 0 &&
              ::symlink(real.c_str(), link.c_str()) == 0,
          "test must create symlink ancestry");
  require_throws<std::invalid_argument>(
      [&] {
        publish_authority_document(link / "symlink.json",
                                   "symlink test document", "unsafe\n",
                                   policy, 1024U);
      },
      "symlinked authority ancestry must be rejected");
}

void live_boot_observation_is_accepted_without_gpu_access() {
  const HostdLinuxBootAuthoritySnapshot observed =
      observe_hostd_linux_boot_authority();
  const HostdDaemonConfiguration materialized =
      materialize_hostd_daemon_boot(HostdDaemonConfiguration(daemon_document()),
                                    observed);
  require(materialized.document().boot_id == observed.boot_id &&
              materialized.document().host_namespaces ==
                  observed.host_namespaces,
          "pinned procfs boot evidence must materialize a strict daemon document");
}

void deployment_uses_boot_materialization_and_stable_peer_authority() {
#ifndef TRAINVM_SOURCE_ROOT
#error "TRAINVM_SOURCE_ROOT is required for deployment contract tests"
#endif
  const std::filesystem::path root(TRAINVM_SOURCE_ROOT);
  const HostdDaemonConfiguration deployed =
      HostdDaemonConfiguration::load_file(root / "deploy/trainvm-hostd.json");
  require(deployed.document().service_roles.size() == 1U &&
              deployed.document().service_roles.front().cgroup_path ==
                  "/system.slice/trainvm-controller.service",
          "deployed peer authority must use a reboot-stable service cgroup");
  std::ifstream input(root / "deploy/trainvm-hostd.service");
  require(input.is_open(), "deployment unit must be readable");
  const std::string unit((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  // The authority owns and writes its own configuration and GPU authorization,
  // so both live in its state directory rather than under /etc, which it has no
  // authority to write.
  const std::string state_dir = "/home/sirus/.local/state/trainvm-hostd";
  require(unit.find("--materialize-config " + state_dir +
                    "/hostd.template.json /run/trainvm-hostd/hostd.json") !=
                  std::string::npos &&
              unit.find("--check-gpu-authorization " + state_dir +
                        "/hostd.template.json " + state_dir +
                        "/hostd-gpu-authorization.json") != std::string::npos &&
              unit.find("--gpu-authorization " + state_dir +
                        "/hostd-gpu-authorization.json") !=
                  std::string::npos &&
              unit.find("--publish-client-config /run/trainvm-hostd/client.json") !=
                  std::string::npos &&
              unit.find("LoadCredential=") == std::string::npos &&
              unit.find("StartLimitBurst=3") != std::string::npos,
          "enabled hostd unit must require explicit boot GPU authority, materialize each boot, publish the endpoint, and bound failed starts");

  // The daemon must be unprivileged, and must be so by construction rather than
  // by the configuration it happens to be given.
  require(unit.find("User=root") == std::string::npos &&
              unit.find("User=sirus") != std::string::npos &&
              unit.find("CapabilityBoundingSet=\n") != std::string::npos &&
              unit.find("AmbientCapabilities=\n") != std::string::npos &&
              unit.find("NoNewPrivileges=yes") != std::string::npos &&
              unit.find("Delegate=yes") != std::string::npos,
          "hostd must run unprivileged with no capabilities and a delegated cgroup");
  const auto authorization_check = unit.find("--check-gpu-authorization");
  const auto first_start_pre = unit.find("ExecStartPre=");
  require(authorization_check != std::string::npos &&
              first_start_pre != std::string::npos &&
              authorization_check < first_start_pre,
          "GPU authorization must be checked before startup work or driver probing");

  std::ifstream sudoers_input(root / "deploy/install-hostd-sudoers.sh");
  require(sudoers_input.is_open(), "sudoers installer must be readable");
  const std::string sudoers((std::istreambuf_iterator<char>(sudoers_input)),
                            std::istreambuf_iterator<char>());
  // GPU authority is now published by the unprivileged authority itself, under
  // an identity check that a sudo invocation would fail. No privileged rule may
  // grant, install, or stand in for it.
  require(sudoers.find("--authorize-gpu-start") == std::string::npos &&
              sudoers.find("hostd.template.json") == std::string::npos &&
              sudoers.find("hostd-gpu-authorization") == std::string::npos &&
              sudoers.find("deploy/trainvm-hostd-gpu-authorization") ==
                  std::string::npos,
          "no privileged rule may install or stand in for GPU authority");
}

}  // namespace

int main() {
  try {
    simulated_reboot_changes_only_boot_authority();
    socket_replacement_requires_a_new_client_document();
    gpu_authorization_is_explicit_boot_scoped_and_strict();
    atomic_publication_is_exact_and_rejects_unsafe_ancestry();
    live_boot_observation_is_accepted_without_gpu_access();
    deployment_uses_boot_materialization_and_stable_peer_authority();
    std::cout << "hostd boot provisioning tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd boot provisioning test failure: " << error.what()
              << '\n';
    return 1;
  }
}
