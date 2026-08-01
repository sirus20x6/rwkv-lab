#include "trainvm/hostd_daemon_runtime.hpp"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/host_ledger_authority.hpp"
#include "trainvm/hostd_journal_fence_attestor.hpp"
#include "trainvm/hostd_journal_logical_fence.hpp"
#include "trainvm/hostd_ledger_singleton_token.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_device_kernel.hpp"
#include "trainvm/hostd_linux_inventory_context_auditor.hpp"
#include "trainvm/hostd_linux_process_authority.hpp"
#include "trainvm/hostd_linux_process_policy_kernel.hpp"
#include "trainvm/hostd_linux_service_identity.hpp"
#include "trainvm/hostd_linux_session_authority.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/hostd_restart_process_recovery.hpp"
#include "trainvm/hostd_session_challenge.hpp"
#include "trainvm/hostd_startup_auditor.hpp"
#include "trainvm/hostd_terminal_release_recovery.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/linux_nvidia_inventory.hpp"

namespace trainvm {
namespace {

pid_t current_tid() noexcept {
  const long value = ::syscall(SYS_gettid);
  if (value <= 0 ||
      static_cast<unsigned long>(value) >
          static_cast<unsigned long>(std::numeric_limits<pid_t>::max()))
    return -1;
  return static_cast<pid_t>(value);
}

void require_inventory_identity(
    const HostdDaemonConfigurationDocument &document,
    const HostInventoryReceipt &inventory, const AuthorityTimeSample &time) {
  if (inventory.host_id != document.host_id ||
      inventory.boot_id != document.boot_id ||
      inventory.broker_epoch != document.broker_epoch ||
      time.boot_id != document.boot_id) {
    throw HostdDaemonRuntimeError(
        "configured host, boot, or broker identity does not match live "
        "authority evidence");
  }
}

int open_socket_parent(const std::filesystem::path &path) {
  const int descriptor = ::open(path.parent_path().c_str(),
                                O_PATH | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0)
    throw HostdDaemonRuntimeError(
        "could not open configured hostd socket parent");
  return descriptor;
}

class CloseDescriptor final {
public:
  explicit CloseDescriptor(int descriptor) noexcept : descriptor_(descriptor) {}
  ~CloseDescriptor() {
    if (descriptor_ >= 0)
      (void)::close(descriptor_);
  }
  [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
  int descriptor_;
};

} // namespace

struct HostdDaemonRuntime::Implementation final {
  explicit Implementation(HostdDaemonConfiguration input)
      : configuration(std::move(input)), owner_pid(::getpid()),
        owner_tid(current_tid()),
        launch_capable(configuration.document().authority_uid == 0U &&
                       ::geteuid() == 0U &&
                       configuration.worker_credentials().uid != 0U &&
                       configuration.worker_credentials().gid != 0U &&
                       configuration.worker_credentials().no_new_privileges) {
    if (owner_tid <= 0)
      throw HostdDaemonRuntimeError("could not bind hostd owner thread");

    clock = std::make_unique<AuthorityClock>();
    session_kernel =
        make_hostd_linux_session_kernel(configuration.session_kernel());
    challenge_time = std::make_shared<HostdLinuxBoottimeSource>(
        configuration.document().boot_id, session_kernel);
    (void)challenge_time->now();
    service_identity = std::make_shared<HostdLinuxServiceIdentityAuthority>(
        configuration.service_identity());
    cgroups = std::make_unique<LinuxCgroupAuthority>(configuration.cgroup());
    inventory_kernel = std::make_unique<LinuxNvidiaInventoryCollector>(
        configuration.inventory());
    inventory = capture_host_inventory(*inventory_kernel);
    require_inventory_identity(configuration.document(), inventory,
                               clock->sample());

    journal = std::make_unique<Journal>(
        configuration.journal_path(), configuration.document().journal_identity,
        HostGrantEnforcement::required, configuration.journal_host());

    ledger_authority = std::make_shared<HostLedgerFilesystemAuthority>(
        HostLedgerFilesystemAuthority::acquire(
            configuration.ledger_authority()));
    singleton = std::make_shared<HostdLedgerSingletonToken>(ledger_authority);
    ledger = std::make_shared<SQLiteHostLedger>(
        ledger_authority, inventory, nullptr,
        configuration.startup_auditor().policy);
    logical_fence = std::make_shared<JournalHostdLogicalFenceEvidenceSource>(
        *journal, *clock);
    coordinator = std::make_shared<HostGrantCoordinator>(
        configuration.coordinator(), ledger, logical_fence);

    launcher = std::make_unique<LinuxStoppedLauncherKernel>();
    device_kernel = std::make_unique<LinuxCgroupDeviceKernel>();
    device_installer =
        std::make_unique<LinuxDevicePolicyInstaller>(*device_kernel);
    process_policy_kernel =
        std::make_unique<LinuxCgroupProcessPolicyKernel>();
    process_policy_installer =
        std::make_unique<LinuxProcessPolicyInstaller>(*process_policy_kernel);
    context_auditor = std::make_unique<LinuxInventoryProcessContextAuditor>(
        *inventory_kernel);
    process_authority = std::make_unique<LinuxProcessAuthority>(
        *ledger, *clock, *cgroups, *device_installer,
        *process_policy_installer, *launcher,
        configuration.worker_credentials(), *context_auditor);
    process_supervisor =
        std::make_shared<HostdLinuxProcessSupervisor>(*process_authority);
    release_authority =
        std::make_unique<SQLiteHostdTerminalReleaseAuthority>(*ledger, *clock);
    cgroup_cleaner =
        std::make_unique<LinuxHostdTerminalCgroupCleaner>(*cgroups);
    terminal_recovery = std::make_unique<HostdTerminalReleaseRecovery>(
        *release_authority, *cgroup_cleaner);
    restart_recovery = std::make_unique<HostdRestartProcessRecovery>(
        *release_authority, *terminal_recovery, *process_supervisor,
        configuration.restart_recovery());
    startup_auditor = std::make_unique<HostdConfiguredStartupAuditor>(
        *ledger, *clock, configuration.startup_auditor());
    startup_admission =
        std::make_unique<HostdCoordinatorStartupAdmission>(*coordinator);
    startup = std::make_unique<HostdStartupController>(
        *restart_recovery, *startup_admission, *startup_auditor, *clock,
        configuration.startup_controller());

    nonce_source =
        std::make_shared<HostdLinuxCSPRNGNonceSource>(session_kernel);
    journal_attestor = std::make_shared<HostdDynamicJournalFenceAttestor>(
        *journal, configuration.journal_attestor(), challenge_time);
    challenge_verifier = std::make_shared<HostdSessionChallengeVerifier>(
        configuration.challenge(), nonce_source, challenge_time,
        journal_attestor);
    ledger_time = std::make_shared<HostdLinuxLedgerTimeSource>();
  }

  void attest_owner() const {
    if (::getpid() != owner_pid || current_tid() != owner_tid)
      throw HostdDaemonRuntimeError(
          "hostd runtime used outside its authority owner thread");
  }

  void bind_transport() {
    if (unified_server)
      return;
    if (socket || status_server || mutation_server)
      throw HostdDaemonRuntimeError(
          "hostd transport assembly is partially initialized");
    CloseDescriptor parent(
        open_socket_parent(configuration.document().socket.path));
    auto bound = HostdSocketAuthority::self_bind(configuration.socket(),
                                                 parent.get(), singleton);
    auto candidate_socket =
        std::make_shared<HostdSocketAuthority>(std::move(bound));
    auto candidate_status = std::make_unique<HostdStatusServer>(
        candidate_socket, coordinator, configuration.status_peer(),
        configuration.status_transport());
    auto candidate_mutation = std::make_unique<HostdMutationServer>(
        candidate_socket, coordinator, challenge_verifier, session_kernel,
        service_identity, ledger_time, configuration.mutation_transport(),
        launch_capable ? process_supervisor : nullptr);
    auto candidate_unified = std::make_unique<HostdUnifiedServer>(
        candidate_socket, *candidate_status, *candidate_mutation);
    socket = std::move(candidate_socket);
    status_server = std::move(candidate_status);
    mutation_server = std::move(candidate_mutation);
    unified_server = std::move(candidate_unified);
  }

  HostdDaemonConfiguration configuration;
  pid_t owner_pid{};
  pid_t owner_tid{};
  bool launch_capable{};
  std::unique_ptr<AuthorityClock> clock;
  std::unique_ptr<LinuxNvidiaInventoryCollector> inventory_kernel;
  HostInventoryReceipt inventory;
  std::shared_ptr<HostLedgerFilesystemAuthority> ledger_authority;
  std::shared_ptr<HostdLedgerSingletonToken> singleton;
  std::shared_ptr<SQLiteHostLedger> ledger;
  std::unique_ptr<Journal> journal;
  std::shared_ptr<JournalHostdLogicalFenceEvidenceSource> logical_fence;
  std::shared_ptr<HostGrantCoordinator> coordinator;
  std::unique_ptr<LinuxCgroupAuthority> cgroups;
  std::unique_ptr<LinuxStoppedLauncherKernel> launcher;
  std::unique_ptr<LinuxCgroupDeviceKernel> device_kernel;
  std::unique_ptr<LinuxDevicePolicyInstaller> device_installer;
  std::unique_ptr<LinuxCgroupProcessPolicyKernel> process_policy_kernel;
  std::unique_ptr<LinuxProcessPolicyInstaller> process_policy_installer;
  std::unique_ptr<LinuxInventoryProcessContextAuditor> context_auditor;
  std::unique_ptr<LinuxProcessAuthority> process_authority;
  std::shared_ptr<HostdLinuxProcessSupervisor> process_supervisor;
  std::unique_ptr<SQLiteHostdTerminalReleaseAuthority> release_authority;
  std::unique_ptr<LinuxHostdTerminalCgroupCleaner> cgroup_cleaner;
  std::unique_ptr<HostdTerminalReleaseRecovery> terminal_recovery;
  std::unique_ptr<HostdRestartProcessRecovery> restart_recovery;
  std::unique_ptr<HostdConfiguredStartupAuditor> startup_auditor;
  std::unique_ptr<HostdCoordinatorStartupAdmission> startup_admission;
  std::unique_ptr<HostdStartupController> startup;
  std::shared_ptr<IHostdLinuxSessionKernel> session_kernel;
  std::shared_ptr<HostdLinuxCSPRNGNonceSource> nonce_source;
  std::shared_ptr<HostdLinuxBoottimeSource> challenge_time;
  std::shared_ptr<HostdDynamicJournalFenceAttestor> journal_attestor;
  std::shared_ptr<HostdSessionChallengeVerifier> challenge_verifier;
  std::shared_ptr<HostdLinuxServiceIdentityAuthority> service_identity;
  std::shared_ptr<HostdLinuxLedgerTimeSource> ledger_time;
  std::shared_ptr<HostdSocketAuthority> socket;
  std::unique_ptr<HostdStatusServer> status_server;
  std::unique_ptr<HostdMutationServer> mutation_server;
  std::unique_ptr<HostdUnifiedServer> unified_server;
};

HostdDaemonRuntime::HostdDaemonRuntime(HostdDaemonConfiguration configuration)
    : implementation_(
          std::make_unique<Implementation>(std::move(configuration))) {}

HostdDaemonRuntime::~HostdDaemonRuntime() = default;

HostdStartupControllerStatus HostdDaemonRuntime::advance_startup() {
  implementation_->attest_owner();
  HostdStartupControllerStatus status = implementation_->startup->advance();
  if (status.phase == HostdStartupPhase::admitting)
    implementation_->bind_transport();
  return status;
}

HostdStartupControllerStatus HostdDaemonRuntime::startup_status() const {
  implementation_->attest_owner();
  return implementation_->startup->status();
}

bool HostdDaemonRuntime::ready() const {
  implementation_->attest_owner();
  return implementation_->unified_server != nullptr &&
         implementation_->startup->status().phase ==
             HostdStartupPhase::admitting;
}

bool HostdDaemonRuntime::process_launch_enabled() const noexcept {
  return implementation_->launch_capable;
}

HostdCoordinatorStatus HostdDaemonRuntime::coordinator_status() const {
  implementation_->attest_owner();
  return implementation_->coordinator->status();
}

HostdSocketIdentity HostdDaemonRuntime::socket_identity() {
  implementation_->attest_owner();
  if (!ready())
    throw HostdDaemonRuntimeError(
        "hostd socket is unavailable before admission");
  return implementation_->socket->reattest();
}

HostdServeResult HostdDaemonRuntime::serve_one() {
  implementation_->attest_owner();
  if (!ready())
    throw HostdDaemonRuntimeError(
        "hostd cannot serve before startup admission");
  const std::int64_t now = hostd_monotonic_now_ns();
  const std::int64_t wake =
      implementation_->configuration.serve_wake_interval_ns();
  if (wake > std::numeric_limits<std::int64_t>::max() - now)
    throw HostdDaemonRuntimeError("hostd serve deadline overflow");
  const HostdServeResult result =
      implementation_->unified_server->serve_one(now + wake);
  if (implementation_->socket->poisoned())
    throw HostdDaemonRuntimeError(implementation_->socket->poison_reason());
  return result;
}

} // namespace trainvm
