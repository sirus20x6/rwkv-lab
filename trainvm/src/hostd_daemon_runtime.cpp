#include "trainvm/hostd_daemon_runtime.hpp"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cerrno>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#include "trainvm/authority_time.hpp"
#include "trainvm/document.hpp"
#include "trainvm/gpu_fault_observer.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/sqlite_filesystem_authority.hpp"
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

// Captured once, before NVML and before any fork, while the daemon is still
// single-threaded. An unprivileged authority cannot drop its supplementary
// groups, so the worker inherits them; sealing the observed set here — rather
// than declaring it in configuration, where it would rot the first time
// /etc/group changed — lets the launcher attest the worker against an exact
// set instead of loosening the check to whatever the child happens to carry.
std::vector<gid_t> seal_authority_supplementary_groups() {
  const int count = ::getgroups(0, nullptr);
  if (count < 0 || static_cast<std::size_t>(count) > kMaximumSupplementaryGroups)
    throw HostdDaemonRuntimeError("hostd supplementary group set is unbounded");
  std::vector<gid_t> groups(static_cast<std::size_t>(count));
  if (count > 0 && ::getgroups(count, groups.data()) != count)
    throw HostdDaemonRuntimeError(
        "hostd supplementary group set changed while sealing");
  std::ranges::sort(groups);
  if (std::ranges::adjacent_find(groups) != groups.end() ||
      std::ranges::find(groups, 0U) != groups.end()) {
    throw HostdDaemonRuntimeError(
        "hostd supplementary group set is not a canonical non-root set");
  }
  return groups;
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

std::string bounded_status_reason(std::string_view reason) {
  constexpr std::size_t maximum = 512U;
  return std::string(reason.substr(0U, maximum));
}

void apply_spawn_status(HostdProcessAuthorityStatus &status,
                        const HostProcessSpawnReceipt &spawn) {
  status.phase = HostdProcessAuthorityPhase::spawned;
  status.host_pid = spawn.request.host_pid;
  status.process_starttime_ticks = spawn.request.process_starttime_ticks;
  if (spawn.request.device_policy) {
    status.device_policy_installed = true;
    status.device_policy_installation_digest =
        spawn.request.device_policy->installation_digest;
  }
  if (spawn.request.process_policy) {
    status.process_policy_installed = true;
    status.process_policy_installation_digest =
        spawn.request.process_policy->installation_digest;
  }
}

HostdProcessAuthorityStatus process_status(
    const ResourceBundleGrant &grant, const HostProcessLaunchIntent &intent,
    const std::optional<HostProcessSpawnReceipt> &spawn) {
  const HostProcessLaunchRequest &request = intent.request;
  HostdProcessAuthorityStatus status;
  status.allocation_id = grant.allocation_id;
  status.journal_id = grant.journal_id;
  status.run_id = grant.run_id;
  status.logical_lease_id = grant.logical_lease_id;
  status.logical_fencing_token = grant.logical_fencing_token;
  status.launch_id = request.launch_id;
  status.phase = HostdProcessAuthorityPhase::launch_intent;
  status.cgroup_path = request.cgroup_path;
  status.device_policy_intended = request.device_policy.has_value();
  status.device_policy_digest = request.device_policy
                                    ? request.device_policy->policy_digest
                                    : std::string{};
  status.process_policy_intended = request.process_policy.has_value();
  status.process_policy_digest = request.process_policy
                                     ? request.process_policy->policy_digest
                                     : std::string{};
  if (spawn)
    apply_spawn_status(status, *spawn);
  return status;
}

HostdProcessAuthorityStatus process_status(
    const HostProcessTerminalReleaseRecord &record) {
  HostdProcessAuthorityStatus status =
      process_status(record.grant, record.intent, record.spawn);
  status.phase = HostdProcessAuthorityPhase::terminal_pending_release;
  if (record.child_exit) {
    status.cgroup_empty = record.child_exit->request.cgroup_empty;
    status.accelerator_contexts_empty =
        record.child_exit->request.accelerator_contexts_empty;
    status.context_audit_digest =
        record.child_exit->request.context_audit_digest;
    status.terminal_receipt_digest = record.child_exit->receipt_digest;
  } else if (record.recovery_exit) {
    status.cgroup_empty = record.recovery_exit->request.cgroup_empty;
    status.accelerator_contexts_empty =
        record.recovery_exit->request.accelerator_contexts_empty;
    status.context_audit_digest =
        record.recovery_exit->request.context_audit_digest;
    status.terminal_receipt_digest = record.recovery_exit->receipt_digest;
  }
  return status;
}

class RuntimeAuthorityStatusSource final
    : public IHostdAuthorityStatusSource {
public:
  static constexpr std::int64_t inventory_refresh_interval_ns =
      30'000'000'000LL;

  RuntimeAuthorityStatusSource(
      std::shared_ptr<SQLiteHostLedger> ledger,
      std::shared_ptr<HostGrantCoordinator> coordinator,
      HostdStartupController &startup, IHostKernel &inventory_kernel,
      bool process_launch_enabled)
      : ledger_(std::move(ledger)), coordinator_(std::move(coordinator)),
        startup_(startup), inventory_kernel_(inventory_kernel),
        process_launch_enabled_(process_launch_enabled) {
    if (!ledger_ || !coordinator_)
      throw HostdDaemonRuntimeError(
          "hostd status source requires ledger and coordinator authority");
  }

  [[nodiscard]] HostdAuthorityStatus snapshot() const override {
    const HostdStartupControllerStatus startup = startup_.status();
    const HostdCoordinatorStatus coordinator = coordinator_->status();
    std::string verification_reason;
    const bool verified = ledger_->verify(&verification_reason);
    const HostLedgerChainHead chain_head = ledger_->chain_head();
    const ResourceOccupancySnapshot occupancy = ledger_->occupancy();
    const auto unclosed = ledger_->active_process_recovery_records();
    const auto terminal = ledger_->active_terminal_process_release_records();

    HostdAuthorityStatus status;
    status.api_version = std::string(kHostdAuthorityStatusApiVersion);
    status.startup_phase = startup.phase;
    status.startup_recovery_steps = startup.recovery_steps;
    status.remaining_unclosed_process_records = unclosed.size();
    status.remaining_terminal_release_records = terminal.size();
    status.ledger_verified = verified;
    status.ledger_verification_reason =
        verified ? std::string{}
                 : bounded_status_reason(
                       verification_reason.empty()
                           ? std::string_view("host ledger verification failed")
                           : std::string_view(verification_reason));
    status.ledger_chain_head = chain_head;
    status.ledger_record_count = ledger_->record_count();
    status.occupancy_ledger_sequence = occupancy.ledger_sequence;
    status.occupancy_digest = occupancy.occupancy_digest;
    const InventoryObservation observation = inventory_observation();
    status.resource_inventory_observation_age_ns = observation.age_ns;
    if (observation.receipt) {
      const HostInventoryReceipt &current = *observation.receipt;
      status.resource_inventory_observed = true;
      status.current_inventory_digest = current.inventory_digest;
      status.current_inventory_receipt_digest = current.receipt_digest;
      if (observation.passive_memory &&
          observation.passive_memory->host_id == current.host_id &&
          observation.passive_memory->boot_id == current.boot_id) {
        const auto &passive = *observation.passive_memory;
        status.passive_memory_host_id = passive.host_id;
        status.passive_memory_boot_id = passive.boot_id;
        status.passive_memory_inventory_digest = current.inventory_digest;
        status.passive_memory_inventory_receipt_digest = current.receipt_digest;
        status.passive_memory_observed_monotonic_ns =
            passive.observed_monotonic_ns;
        for (const auto &memory : passive.accelerators) {
          const auto resource = std::ranges::find_if(
              current.resources, [&](const auto &candidate) {
                return candidate.id.stable_id == memory.stable_id &&
                       candidate.id.vendor == memory.vendor &&
                       candidate.total_memory_bytes ==
                           memory.total_memory_bytes;
              });
          if (resource == current.resources.end())
            continue;
          status.passive_accelerator_memory.push_back({
              .resource_kind = resource->id.kind,
              .vendor = memory.vendor,
              .stable_id = memory.stable_id,
              .parent_id = resource->id.parent_id,
              .audited_eligible =
                  resource->disposition ==
                  ResourceObservationDisposition::audited_eligible,
              .total_memory_bytes = memory.total_memory_bytes,
              .free_memory_bytes = memory.free_memory_bytes,
              .selector_labels = resource->labels,
          });
        }
        std::ranges::sort(status.passive_accelerator_memory, {},
                          &HostdPassiveAcceleratorMemory::stable_id);
        status.passive_accelerator_memory_count =
            status.passive_accelerator_memory.size();
        if (status.passive_accelerator_memory.size() >
            HostdAuthorityStatus::maximum_passive_memory_rows) {
          status.passive_accelerator_memory.resize(
              HostdAuthorityStatus::maximum_passive_memory_rows);
          status.passive_accelerator_memory_truncated = true;
        }
        const auto identity = [&] {
          return nlohmann::json{
              {"api_version", "trainvm.hostd-passive-memory/v1"},
              {"host_id", status.passive_memory_host_id},
              {"boot_id", status.passive_memory_boot_id},
              {"inventory_digest", status.passive_memory_inventory_digest},
              {"inventory_receipt_digest",
               status.passive_memory_inventory_receipt_digest},
              {"observed_monotonic_ns",
               status.passive_memory_observed_monotonic_ns},
              {"accelerator_count", status.passive_accelerator_memory_count},
              {"accelerators_truncated",
               status.passive_accelerator_memory_truncated},
              {"accelerators", encode_json(status.passive_accelerator_memory)},
          };
        };
        while (!status.passive_accelerator_memory.empty() &&
               identity().dump().size() >
                   HostdAuthorityStatus::
                       maximum_passive_memory_identity_bytes) {
          status.passive_accelerator_memory.pop_back();
          status.passive_accelerator_memory_truncated = true;
        }
        status.passive_memory_observation_digest =
            "sha256:" + sha256_hex(identity().dump());
      }
      for (const ResourceFence &fence : occupancy.active_fences) {
        const auto observed = std::find_if(
            current.resources.begin(), current.resources.end(),
            [&](const ObservedHostResource &resource) {
              return resource.id.stable_id == fence.resource.stable_id;
            });
        if (current.host_id != coordinator.host_id ||
            current.boot_id != coordinator.boot_id ||
            current.topology_digest != fence.topology_digest ||
            observed == current.resources.end() ||
            observed->id != fence.resource) {
          ++status.degraded_resource_count;
        }
      }
      if (status.degraded_resource_count != 0U) {
        status.resource_degradation_reason =
            std::to_string(status.degraded_resource_count) +
            " active resource fence(s) no longer match the current "
            "inventory receipt";
      }
    } else {
      status.resource_degradation_reason = observation.failure_reason;
    }
    status.active_fence_count = occupancy.active_fences.size();
    status.active_fences_truncated =
        occupancy.active_fences.size() >
        HostdAuthorityStatus::maximum_reported_rows;
    status.active_process_count = unclosed.size() + terminal.size();
    status.active_processes_truncated =
        status.active_process_count > HostdAuthorityStatus::maximum_reported_rows;
    status.process_launch_enabled = process_launch_enabled_;

    const std::size_t fence_count = std::min(
        occupancy.active_fences.size(),
        HostdAuthorityStatus::maximum_reported_rows);
    status.active_fences.assign(occupancy.active_fences.begin(),
                                occupancy.active_fences.begin() +
                                    static_cast<std::ptrdiff_t>(fence_count));
    status.active_processes.reserve(std::min(
        status.active_process_count,
        HostdAuthorityStatus::maximum_reported_rows));
    for (const HostProcessRecoveryRecord &record : unclosed) {
      if (status.active_processes.size() ==
          HostdAuthorityStatus::maximum_reported_rows)
        break;
      status.active_processes.push_back(
          process_status(record.grant, record.intent, record.spawn));
    }
    for (const HostProcessTerminalReleaseRecord &record : terminal) {
      if (status.active_processes.size() ==
          HostdAuthorityStatus::maximum_reported_rows)
        break;
      status.active_processes.push_back(process_status(record));
    }

    if (!verified) {
      status.mutation_disabled_reason = verification_reason.empty()
                                            ? "host ledger verification failed"
                                            : bounded_status_reason(
                                                  verification_reason);
    } else if (startup.phase != HostdStartupPhase::admitting) {
      status.mutation_disabled_reason =
          "hostd startup recovery has not reached admission";
    } else if (coordinator.lifecycle != HostdLifecycle::admitting) {
      status.mutation_disabled_reason = coordinator.poison_reason.empty()
                                            ? "hostd coordinator is not admitting"
                                            : bounded_status_reason(
                                                  coordinator.poison_reason);
    } else if (!process_launch_enabled_) {
      status.mutation_disabled_reason =
          "strict process-launch enforcement is unavailable";
    } else {
      status.mutation_enabled = true;
    }
    return status;
  }

private:
  struct InventoryObservation final {
    std::optional<HostInventoryReceipt> receipt;
    std::optional<PassiveHostMemorySnapshot> passive_memory;
    std::string failure_reason;
    std::uint64_t age_ns{};
  };

  [[nodiscard]] InventoryObservation inventory_observation() const {
    std::scoped_lock lock(inventory_mutex_);
    const std::int64_t now = hostd_monotonic_now_ns();
    if (!inventory_observed_at_ns_ || now < *inventory_observed_at_ns_ ||
        now - *inventory_observed_at_ns_ >= inventory_refresh_interval_ns) {
      try {
        cached_inventory_ = capture_host_inventory(inventory_kernel_);
        cached_passive_memory_ = inventory_kernel_.passive_memory_snapshot();
        cached_inventory_failure_.clear();
      } catch (const std::exception &exception) {
        cached_inventory_.reset();
        cached_passive_memory_.reset();
        cached_inventory_failure_ = bounded_status_reason(
            std::string("current inventory observation failed: ") +
            exception.what());
      } catch (...) {
        cached_inventory_.reset();
        cached_passive_memory_.reset();
        cached_inventory_failure_ =
            "current inventory observation failed with a non-standard exception";
      }
      inventory_observed_at_ns_ = hostd_monotonic_now_ns();
    }
    const std::int64_t observed_at = *inventory_observed_at_ns_;
    const std::int64_t sampled_now = hostd_monotonic_now_ns();
    const std::uint64_t age = sampled_now > observed_at
                                  ? static_cast<std::uint64_t>(sampled_now - observed_at)
                                  : 0U;
    return {.receipt = cached_inventory_,
            .passive_memory = cached_passive_memory_,
            .failure_reason = cached_inventory_failure_,
            .age_ns = age};
  }

  std::shared_ptr<SQLiteHostLedger> ledger_;
  std::shared_ptr<HostGrantCoordinator> coordinator_;
  HostdStartupController &startup_;
  IHostKernel &inventory_kernel_;
  mutable std::mutex inventory_mutex_;
  mutable std::optional<HostInventoryReceipt> cached_inventory_;
  mutable std::optional<PassiveHostMemorySnapshot> cached_passive_memory_;
  mutable std::string cached_inventory_failure_;
  mutable std::optional<std::int64_t> inventory_observed_at_ns_;
  bool process_launch_enabled_{};
};

} // namespace

struct HostdDaemonRuntime::Implementation final {
  Implementation(HostdDaemonConfiguration input,
                 HostdGpuAuthorization input_gpu_authorization)
      : configuration(std::move(input)),
        gpu_authorization(std::move(input_gpu_authorization)),
        owner_pid(::getpid()),
        owner_tid(current_tid()),
        launch_capable(configuration.document().authority_uid != 0U &&
                       ::geteuid() ==
                           static_cast<uid_t>(
                               configuration.document().authority_uid) &&
                       configuration.worker_credentials().uid != 0U &&
                       configuration.worker_credentials().gid != 0U &&
                       configuration.worker_credentials().no_new_privileges) {
    if (owner_tid <= 0)
      throw HostdDaemonRuntimeError("could not bind hostd owner thread");
    // hostd_main enforces this before publishing, but the runtime is also
    // constructed directly, and every ownership expectation below is derived
    // from the configured authority rather than from the running identity.
    if (::geteuid() !=
            static_cast<uid_t>(configuration.document().authority_uid) ||
        ::getegid() !=
            static_cast<gid_t>(configuration.document().authority_gid)) {
      throw HostdDaemonRuntimeError(
          "hostd runtime identity is not its configured authority");
    }
    gpu_authorization.require_matches(configuration);

    clock = std::make_unique<AuthorityClock>();
    session_kernel =
        make_hostd_linux_session_kernel(configuration.session_kernel());
    challenge_time = std::make_shared<HostdLinuxBoottimeSource>(
        configuration.document().boot_id, session_kernel);
    (void)challenge_time->now();
    service_identity = std::make_shared<HostdLinuxServiceIdentityAuthority>(
        configuration.service_identity());
    cgroups = std::make_unique<LinuxCgroupAuthority>(configuration.cgroup());
    // hostd attests the established controller journal; it must never become a
    // second writer, so it opens read-only and without exclusive WAL ownership.
    journal = std::make_unique<Journal>(
        configuration.journal_path(), configuration.document().journal_identity,
        HostGrantEnforcement::required, configuration.journal_host(), nullptr,
        false, JournalAccessMode::read_only);

    ledger_authority = std::make_shared<SqliteFilesystemAuthority>(
        SqliteFilesystemAuthority::acquire(
            configuration.ledger_authority()));
    singleton = std::make_shared<HostdLedgerSingletonToken>(ledger_authority);
    // Socket self-bind temporarily changes the process working directory to a
    // pinned descriptor, so it must happen before NVML loads the CUDA driver
    // and creates its helper thread.  The retained singleton token is already
    // sufficient to authorize the bind; request serving is assembled only
    // after the inventory, ledger, and recovery authorities are admitted.
    CloseDescriptor parent(
        open_socket_parent(configuration.document().socket.path));
    socket = std::make_shared<HostdSocketAuthority>(
        HostdSocketAuthority::self_bind(configuration.socket(), parent.get(),
                                        singleton));

    LinuxNvidiaInventoryConfig inventory_configuration =
        configuration.inventory();
    inventory_configuration.authorized_display_gpu_ids =
        gpu_authorization.allowed_display_gpu_ids();
    inventory_kernel = std::make_unique<LinuxNvidiaInventoryCollector>(
        std::move(inventory_configuration));
    inventory = capture_host_inventory(*inventory_kernel);
    require_inventory_identity(configuration.document(), inventory,
                               clock->sample());
    ledger = std::make_shared<SQLiteHostLedger>(
        ledger_authority, inventory, nullptr,
        configuration.startup_auditor().policy);
    logical_fence = std::make_shared<JournalHostdLogicalFenceEvidenceSource>(
        *journal, *clock);
    if (configuration.document().gpu_fault_guard) {
      const auto& guard = *configuration.document().gpu_fault_guard;
      gpu_fault_guard = std::make_shared<LinuxGpuFaultAdmissionGuard>(
          guard.state_path, configuration.document().boot_id,
          guard.maximum_state_age_ns, configuration.document().authority_uid);
    }
    coordinator = std::make_shared<HostGrantCoordinator>(
        configuration.coordinator(), ledger, logical_fence, gpu_fault_guard);

    launcher = std::make_unique<LinuxStoppedLauncherKernel>();
    // No device policy: installing one needs CAP_BPF, which an unprivileged
    // authority does not hold. The BPF layer stays built and unit-tested; it is
    // simply not wired into any production launch.
    process_policy_kernel =
        std::make_unique<LinuxCgroupProcessPolicyKernel>();
    process_policy_installer =
        std::make_unique<LinuxProcessPolicyInstaller>(*process_policy_kernel);
    context_auditor = std::make_unique<LinuxInventoryProcessContextAuditor>(
        *inventory_kernel);
    LinuxWorkerCredentialSpec worker_credentials =
        configuration.worker_credentials();
    if (configuration.document()
            .worker_identity.inherit_authority_supplementary_groups
            .value_or(false)) {
      worker_credentials.supplementary_gids =
          seal_authority_supplementary_groups();
    }
    process_authority = std::make_unique<LinuxProcessAuthority>(
        *ledger, *clock, *cgroups, nullptr, *process_policy_installer,
        *launcher, worker_credentials, *context_auditor);
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
    authority_status_source = std::make_shared<RuntimeAuthorityStatusSource>(
        ledger, coordinator, *startup, *inventory_kernel, launch_capable);

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

  // The authority socket is already self-bound by the constructor, which must
  // happen before NVML loads the CUDA driver.  This step only assembles
  // request serving over that bound socket, which the constructor deliberately
  // deferred until the inventory, ledger, and recovery authorities were
  // admitted.
  void bind_transport() {
    if (unified_server)
      return;
    if (status_server || mutation_server)
      throw HostdDaemonRuntimeError(
          "hostd transport assembly is partially initialized");
    if (!socket)
      throw HostdDaemonRuntimeError(
          "hostd transport assembly requires the bound authority socket");
    // Every status rejection is otherwise silent: the connection is closed with
    // no reply and nothing is recorded, so a controller that cannot complete a
    // status exchange reports only that the peer closed on it. Surface the
    // reason the transport already computes.
    HostdStatusTransportLimits status_limits = configuration.status_transport();
    status_limits.rejection_observer = [](std::string_view reason) {
      std::cerr << "trainvm-hostd: status rejected: " << reason << '\n';
    };
    auto candidate_status = std::make_unique<HostdStatusServer>(
        socket, coordinator, configuration.status_peer(), status_limits,
        authority_status_source);
    auto candidate_mutation = std::make_unique<HostdMutationServer>(
        socket, coordinator, challenge_verifier, session_kernel,
        service_identity, ledger_time, configuration.mutation_transport(),
        launch_capable ? process_supervisor : nullptr);
    auto candidate_unified = std::make_unique<HostdUnifiedServer>(
        socket, *candidate_status, *candidate_mutation);
    status_server = std::move(candidate_status);
    mutation_server = std::move(candidate_mutation);
    unified_server = std::move(candidate_unified);
  }

  HostdDaemonConfiguration configuration;
  HostdGpuAuthorization gpu_authorization;
  pid_t owner_pid{};
  pid_t owner_tid{};
  bool launch_capable{};
  std::unique_ptr<AuthorityClock> clock;
  std::unique_ptr<LinuxNvidiaInventoryCollector> inventory_kernel;
  HostInventoryReceipt inventory;
  std::shared_ptr<SqliteFilesystemAuthority> ledger_authority;
  std::shared_ptr<HostdLedgerSingletonToken> singleton;
  std::shared_ptr<SQLiteHostLedger> ledger;
  std::unique_ptr<Journal> journal;
  std::shared_ptr<JournalHostdLogicalFenceEvidenceSource> logical_fence;
  std::shared_ptr<LinuxGpuFaultAdmissionGuard> gpu_fault_guard;
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
  std::shared_ptr<IHostdAuthorityStatusSource> authority_status_source;
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

HostdDaemonRuntime::HostdDaemonRuntime(
    HostdDaemonConfiguration configuration,
    HostdGpuAuthorization gpu_authorization)
    : implementation_(std::make_unique<Implementation>(
          std::move(configuration), std::move(gpu_authorization))) {}

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
