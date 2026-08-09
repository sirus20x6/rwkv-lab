#include "trainvm/hostd_linux_process_authority.hpp"

#include <fcntl.h>
#include <optional>
#include <ranges>
#include <string>
#include <unistd.h>
#include <utility>

#include "trainvm/worker_bootstrap.hpp"

namespace trainvm {
namespace {

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const { return value_; }

 private:
  int value_;
};

[[noreturn]] void reject(std::string message) {
  throw LinuxStoppedLauncherError(std::move(message));
}

HostLedgerTime ledger_time(const AuthorityTimeSample& sample,
                           std::string_view expected_boot_id) {
  if (sample.boot_id != expected_boot_id) {
    reject("process launch authority crossed a boot identity boundary");
  }
  return {.boottime_ns = sample.boot.nanoseconds,
          .wall_time_ns = sample.wall.nanoseconds};
}

void validate_binding(const ResolvedLaunchSpec& spec,
                      const ResourceBundleGrant& grant,
                      const HostInventoryReceipt& inventory) {
  const ResolvedLaunchIdentity& identity = spec.identity;
  if (!identity.host_grant ||
      identity.host_grant->request_id != grant.request_id ||
      identity.host_grant->grant_digest != grant.receipt_digest ||
      identity.host_grant->fences != grant.fences ||
      identity.run_id != grant.run_id ||
      identity.lease_id != grant.logical_lease_id ||
      identity.fencing_token != grant.logical_fencing_token ||
      identity.host.host_id != grant.host_id ||
      identity.host.boot_id != grant.boot_id ||
      grant.host_id != inventory.host_id ||
      grant.boot_id != inventory.boot_id ||
      grant.broker_epoch != inventory.broker_epoch) {
    reject("resolved launch does not bind the exact active resource grant");
  }
}

}  // namespace

LinuxPreparedLaunch::LinuxPreparedLaunch(
    HostProcessLaunchIntent intent, HostProcessSpawnReceipt spawn_receipt,
    LinuxProcessPolicy process_policy,
    LinuxProcessPolicyInstallation process_policy_installation,
    LinuxAllocationCgroup cgroup, LinuxStoppedChild child) noexcept
    : intent_(std::move(intent)), spawn_receipt_(std::move(spawn_receipt)),
      process_policy_(std::move(process_policy)),
      process_policy_installation_(std::move(process_policy_installation)),
      cgroup_(std::move(cgroup)),
      child_(std::move(child)) {}

const HostProcessLaunchIntent& LinuxPreparedLaunch::intent() const {
  return intent_;
}

const HostProcessSpawnReceipt& LinuxPreparedLaunch::spawn_receipt() const {
  return spawn_receipt_;
}

const LinuxStoppedChildIdentity& LinuxPreparedLaunch::child_identity() const {
  return child_.identity();
}

const std::optional<HostProcessExitReceipt>& LinuxPreparedLaunch::exit_receipt()
    const {
  return exit_receipt_;
}

LinuxRecoveredLaunch::LinuxRecoveredLaunch(
    HostProcessRecoveryRecord record, LinuxRecoveredProcess process,
    LinuxAllocationCgroup cgroup,
    std::optional<LinuxDevicePolicyInstallation> device_policy,
    std::optional<LinuxProcessPolicy> process_policy,
    std::optional<LinuxProcessPolicyInstallation>
        process_policy_installation) noexcept
    : record_(std::move(record)), process_(std::move(process)),
      cgroup_(std::move(cgroup)), device_policy_(std::move(device_policy)),
      process_policy_(std::move(process_policy)),
      process_policy_installation_(std::move(process_policy_installation)) {}

const HostProcessRecoveryRecord& LinuxRecoveredLaunch::record() const noexcept {
  return record_;
}

const LinuxRecoveredProcess& LinuxRecoveredLaunch::process() const noexcept {
  return process_;
}

const std::optional<HostProcessRecoveryExitReceipt>&
LinuxRecoveredLaunch::exit_receipt() const noexcept {
  return exit_receipt_;
}

LinuxProcessAuthority::LinuxProcessAuthority(
    SQLiteHostLedger& ledger, AuthorityClock& clock,
    LinuxCgroupAuthority& cgroups,
    LinuxDevicePolicyInstaller* device_policies,
    LinuxProcessPolicyInstaller& process_policies,
    LinuxStoppedLauncherKernel& launcher,
    LinuxWorkerCredentialSpec worker_credentials,
    ILinuxProcessContextAuditor& context_auditor)
    : ledger_(ledger), clock_(clock), cgroups_(cgroups),
      device_policies_(device_policies), process_policies_(process_policies),
      launcher_(launcher),
      worker_credentials_(worker_credentials), context_auditor_(context_auditor) {
  if (worker_credentials_.uid == 0U || worker_credentials_.gid == 0U ||
      !worker_credentials_.no_new_privileges) {
    reject("process authority requires a non-root worker credential boundary");
  }
  // A sealed supplementary set is only meaningful for a worker that shares this
  // authority's identity; anything else means the seal was taken from the wrong
  // process.
  if (!worker_credentials_.supplementary_gids.empty() &&
      worker_credentials_.uid != ::geteuid()) {
    reject(
        "inherited supplementary groups require the worker to share the "
        "authority identity");
  }
}

LinuxPreparedLaunch LinuxProcessAuthority::prepare(
    const ResolvedLaunch& resolved, const ResourceBundleGrant& grant,
    int worker_bootstrap_fd, std::string_view worker_bootstrap_digest,
    const LinuxProcessPolicy& process_policy) {
  const ResolvedLaunchSpec spec = resolved_launch_spec_from_json(
      resolved_launch_spec_json(resolved.spec()));
  const HostInventoryReceipt inventory = ledger_.inventory();
  validate_binding(spec, grant, inventory);
  const WorkerBootstrapSpec bootstrap = worker_bootstrap_from_sealed_fd(
      worker_bootstrap_fd, worker_bootstrap_digest);
  const auto& identity = spec.identity;
  if (bootstrap.run_id != identity.run_id ||
      bootstrap.node_id != identity.node_id ||
      bootstrap.attempt_id != identity.attempt_id ||
      bootstrap.launch_nonce != identity.launch_nonce ||
      bootstrap.adapter != identity.adapter_key.adapter ||
      bootstrap.adapter_version != identity.adapter_key.version ||
      bootstrap.code_fingerprint != identity.code_fingerprint ||
      bootstrap.capabilities != identity.provided_capabilities ||
      bootstrap.concurrency_key != identity.concurrency_key ||
      bootstrap.lease_id != identity.lease_id ||
      bootstrap.fencing_token != identity.fencing_token) {
    reject("worker bootstrap disagrees with resolved launch authority");
  }
  LinuxAllocationCgroup cgroup = cgroups_.open_or_create(
      grant.allocation_id, spec.identity.launch_event_id);
  const auto& cgroup_identity = cgroup.identity();
  const HostProcessLaunchRequest request =
      seal_host_process_launch_request({
          .api_version =
              std::string(kHostProcessLaunchRequestApiVersionV4),
          .launch_id = spec.identity.launch_event_id,
          .allocation_id = grant.allocation_id,
          .grant_digest = grant.receipt_digest,
          .journal_id = grant.journal_id,
          .run_id = grant.run_id,
          .logical_lease_id = grant.logical_lease_id,
          .logical_fencing_token = grant.logical_fencing_token,
          .resolved_launch_digest = hostd_bound_process_launch_digest(
              spec, worker_bootstrap_digest, process_policy),
          .executable_path = spec.identity.executable.source_path,
          .executable_digest = spec.identity.executable.sealed_sha256,
          .cgroup_path = cgroup_identity.unified_path,
          .cgroup_device = cgroup_identity.device,
          .cgroup_inode = cgroup_identity.inode,
          .worker_credentials = HostWorkerCredentialBinding{
              .uid = static_cast<std::uint32_t>(worker_credentials_.uid),
              .gid = static_cast<std::uint32_t>(worker_credentials_.gid),
              .no_new_privileges = true},
          .device_policy = std::nullopt,
          .process_policy =
              host_process_policy_intent_binding(process_policy),
          .canonical_request_digest = {},
      });
  const auto intended = ledger_.commit_process_launch_intent(
      request, ledger_time(clock_.sample(), inventory.boot_id));
  cgroup.retain_for_durable_intent();
  LinuxProcessPolicyInstallation installed_process_policy =
      process_policies_.install(process_policy, grant.allocation_id,
                                spec.identity.launch_event_id, cgroup);
  Descriptor cgroup_fd(cgroup.duplicate_fd());
  Descriptor executable_fd(resolved.duplicate_executable_fd());
  const std::optional<int> code = resolved.duplicate_code_fd();
  Descriptor code_fd(code.value_or(-1));
  Descriptor working_directory_fd(
      resolved.duplicate_working_directory_fd());
  const std::optional<int> profiler_executable =
      resolved.duplicate_profiler_executable_fd();
  const std::optional<int> profiler_authority =
      resolved.duplicate_profiler_authority_fd();
  Descriptor profiler_executable_fd(profiler_executable.value_or(-1));
  Descriptor profiler_authority_fd(profiler_authority.value_or(-1));
  std::vector<std::string> arguments = spec.identity.public_arguments;
  if (code) {
    if (spec.identity.code_argument_index >= arguments.size())
      reject("Python launch has no in-range fixed code argument slot");
    arguments.at(spec.identity.code_argument_index) =
        "/proc/self/fd/" + std::to_string(kLinuxWorkerCodeDescriptor);
  }
  arguments.push_back("--trainvm-bootstrap-fd=" +
                      std::to_string(kLinuxWorkerBootstrapDescriptor));
  LinuxStoppedChild child = launcher_.spawn_stopped({
      .launch_id = request.launch_id,
      .cgroup_fd = cgroup_fd.get(),
      .expected_cgroup_path = cgroup_identity.unified_path,
      .expected_cgroup_device = cgroup_identity.device,
      .expected_cgroup_inode = cgroup_identity.inode,
      .executable_fd = executable_fd.get(),
      .code_fd = code ? std::optional<int>{code_fd.get()} : std::nullopt,
      .worker_bootstrap_fd = worker_bootstrap_fd,
      .executable_name = spec.identity.executable.source_path,
      .executable_digest = spec.identity.executable.sealed_sha256,
      .working_directory_fd = working_directory_fd.get(),
      .credentials = worker_credentials_,
      .nice = process_policy.nice
                  ? std::optional<std::int32_t>{
                        static_cast<std::int32_t>(*process_policy.nice)}
                  : std::nullopt,
      .code_argument_index = spec.identity.code_argument_index,
      .arguments = std::move(arguments),
      .profiler = identity.profiler
          ? std::optional<LinuxExternalProfilerLaunchSpec>{
                LinuxExternalProfilerLaunchSpec{
                    .executable_fd = profiler_executable_fd.get(),
                    .authority_fd = profiler_authority_fd.get(),
                    .executable_name =
                        identity.profiler->executable.source_path,
                    .executable_digest =
                        identity.profiler->executable.sealed_sha256,
                    .execute_from_source =
                        identity.profiler->execute_from_source,
                    .source_device =
                        identity.profiler->executable.source_device,
                    .source_inode =
                        identity.profiler->executable.source_inode,
                    .source_size =
                        identity.profiler->executable.source_size,
                    .source_mode =
                        identity.profiler->executable.source_mode,
                    .source_uid = identity.profiler->executable.source_uid,
                    .source_gid = identity.profiler->executable.source_gid,
                    .arguments = identity.profiler->public_arguments,
                }}
          : std::nullopt,
  });
  const LinuxStoppedChildIdentity& observed = child.identity();
  if (observed.uid != worker_credentials_.uid ||
      observed.gid != worker_credentials_.gid ||
      !observed.no_new_privileges ||
      (process_policy.nice && observed.nice != *process_policy.nice)) {
    reject("stopped child did not retain its sealed worker credentials");
  }
  installed_process_policy = process_policies_.bind_process_identity(
      process_policy, std::move(installed_process_policy), observed.nice);
  const HostProcessSpawnRequest spawn_request =
      seal_host_process_spawn_request({
          .api_version =
              std::string(kHostProcessSpawnRequestApiVersionV4),
          .launch_id = request.launch_id,
          .launch_intent_digest = intended.intent.receipt_digest,
          .host_pid = observed.host_pid,
          .process_starttime_ticks = observed.process_starttime_ticks,
          .boot_id = inventory.boot_id,
          .cgroup_path = observed.cgroup_path,
          .cgroup_device = observed.cgroup_device,
          .cgroup_inode = observed.cgroup_inode,
          .executable_digest = observed.executable_digest,
          .worker_credentials = request.worker_credentials,
          .device_policy = std::nullopt,
          .process_policy = host_process_policy_installation_binding(
              installed_process_policy),
          .canonical_request_digest = {},
      });
  const auto spawned = ledger_.commit_process_spawn(
      spawn_request, ledger_time(clock_.sample(), inventory.boot_id));
  return LinuxPreparedLaunch(intended.intent, spawned.receipt, process_policy,
                             installed_process_policy, std::move(cgroup),
                             std::move(child));
}

void LinuxProcessAuthority::release_to_exec(LinuxPreparedLaunch& launch) {
  if (launch.exit_receipt_) {
    reject("terminal process cannot regain executable authority");
  }
  process_policies_.verify(launch.process_policy_,
                           launch.process_policy_installation_,
                           launch.cgroup_, launch.child_.identity().nice);
  launch.child_.release_to_exec();
}

HostProcessExitResult LinuxProcessAuthority::finalize_exit(
    LinuxPreparedLaunch& launch, const ResourceBundleGrant& grant,
    std::string exit_request_id, bool request_termination) {
  if (launch.intent_.request.allocation_id != grant.allocation_id ||
      launch.intent_.request.grant_digest != grant.receipt_digest ||
      launch.spawn_receipt_.request.launch_id !=
          launch.intent_.request.launch_id) {
    reject("terminal process request does not match the prepared grant");
  }
  if (launch.exit_receipt_) {
    if (!launch.cgroup_removed_) {
      launch.cgroup_.remove_if_empty();
      launch.cgroup_removed_ = true;
    }
    return {.receipt = *launch.exit_receipt_, .replayed = true};
  }
  const LinuxChildExitObservation exited =
      request_termination ? launch.child_.terminate_and_observe()
                          : launch.child_.wait_and_reap();
  if (!launch.cgroup_.empty()) {
    reject("terminal launcher still has descendants in its allocation cgroup");
  }
  const LinuxProcessContextAudit context =
      context_auditor_.audit(grant, launch.spawn_receipt_);
  if (!context.complete || !context.accelerator_contexts_empty) {
    reject("terminal launcher still has accelerator context authority");
  }
  const auto& spawn = launch.spawn_receipt_;
  const auto& cgroup = launch.cgroup_.identity();
  const HostProcessExitRequest request = seal_host_process_exit_request({
      .api_version = std::string(kHostProcessExitRequestApiVersion),
      .exit_request_id = std::move(exit_request_id),
      .launch_id = spawn.request.launch_id,
      .spawn_receipt_digest = spawn.receipt_digest,
      .host_pid = spawn.request.host_pid,
      .process_starttime_ticks = spawn.request.process_starttime_ticks,
      .wait_code = exited.wait_code,
      .wait_status = exited.wait_status,
      .cgroup_path = cgroup.unified_path,
      .cgroup_device = cgroup.device,
      .cgroup_inode = cgroup.inode,
      .cgroup_empty = true,
      .accelerator_contexts_empty = true,
      .context_audit_digest = context.evidence_digest,
      .canonical_request_digest = {},
  });
  const auto terminal = ledger_.commit_process_exit(
      request, ledger_time(clock_.sample(), spawn.request.boot_id));
  launch.exit_receipt_ = terminal.receipt;
  launch.cgroup_.remove_if_empty();
  launch.cgroup_removed_ = true;
  return terminal;
}

LinuxRecoveredLaunch LinuxProcessAuthority::adopt_recovered(
    HostProcessRecoveryRecord record, LinuxRecoveredProcess process) {
  if (!record.spawn || process.identity() != record.spawn->request ||
      record.intent.request.launch_id != record.spawn->request.launch_id ||
      record.intent.request.allocation_id != record.grant.allocation_id ||
      record.intent.request.grant_digest != record.grant.receipt_digest ||
      process.state() == LinuxPidfdState::observation_failed) {
    reject("recovered process does not bind its durable ledger authority");
  }
  const HostInventoryReceipt inventory = ledger_.inventory();
  if (record.grant.host_id != inventory.host_id ||
      record.grant.boot_id != inventory.boot_id ||
      record.grant.broker_epoch != inventory.broker_epoch ||
      record.spawn->request.boot_id != inventory.boot_id) {
    reject("recovered process crossed its durable host or boot authority");
  }
  const std::vector<HostProcessRecoveryRecord> active =
      ledger_.active_process_recovery_records();
  const auto found = std::ranges::find_if(
      active, [&record](const HostProcessRecoveryRecord& candidate) {
        return candidate.intent.request.launch_id ==
               record.intent.request.launch_id;
      });
  if (found == active.end() || *found != record) {
    reject("recovered process evidence is no longer the active ledger record");
  }
  const auto& spawn = record.spawn->request;
  LinuxAllocationCgroup cgroup = cgroups_.open_existing_for_recovery(
      record.grant.allocation_id, record.intent.request.launch_id,
      {.unified_path = spawn.cgroup_path,
       .device = spawn.cgroup_device,
       .inode = spawn.cgroup_inode});
  std::optional<LinuxDevicePolicyInstallation> device_policy;
  if (record.intent.api_version == kHostProcessLaunchIntentApiVersionV2 ||
      record.intent.api_version == kHostProcessLaunchIntentApiVersionV3 ||
      record.spawn->api_version == kHostProcessSpawnReceiptApiVersionV2 ||
      record.spawn->api_version == kHostProcessSpawnReceiptApiVersionV3) {
    // A record sealed by a privileged authority. Re-verifying it means querying
    // the attached BPF program, which needs CAP_BPF. Refuse it outright rather
    // than adopt a process whose device confinement we cannot confirm.
    if (device_policies_ == nullptr) {
      reject(
          "recovered process carries device-policy evidence this authority "
          "cannot verify");
    }
    device_policy = linux_device_policy_installation_from_process(
        record.intent, *record.spawn);
    (void)device_policies_->verify(*device_policy, cgroup);
  }
  std::optional<LinuxProcessPolicy> process_policy;
  std::optional<LinuxProcessPolicyInstallation> process_policy_installation;
  if (record.intent.api_version == kHostProcessLaunchIntentApiVersionV3 ||
      record.intent.api_version == kHostProcessLaunchIntentApiVersionV4 ||
      record.spawn->api_version == kHostProcessSpawnReceiptApiVersionV3 ||
      record.spawn->api_version == kHostProcessSpawnReceiptApiVersionV4) {
    process_policy =
        linux_process_policy_from_process(record.intent, *record.spawn);
    process_policy_installation =
        linux_process_policy_installation_from_process(record.intent,
                                                       *record.spawn);
    process_policies_.verify(*process_policy, *process_policy_installation,
                             cgroup,
                             static_cast<std::int32_t>(
                                 process_policy_installation->nice.value_or(
                                     0)));
  }
  return LinuxRecoveredLaunch(std::move(record), std::move(process),
                              std::move(cgroup), std::move(device_policy),
                              std::move(process_policy),
                              std::move(process_policy_installation));
}

HostProcessRecoveryExitResult LinuxProcessAuthority::finalize_recovered_exit(
    LinuxRecoveredLaunch& launch, const ResourceBundleGrant& grant,
    std::string recovery_exit_request_id, bool request_termination) {
  if (!launch.record_.spawn || launch.record_.grant != grant ||
      launch.process_.identity() != launch.record_.spawn->request) {
    reject("recovered terminal request does not match the durable grant");
  }
  if (launch.exit_receipt_) {
    if (!launch.cgroup_removed_) {
      launch.cgroup_.remove_if_empty();
      launch.cgroup_removed_ = true;
    }
    return {.receipt = *launch.exit_receipt_, .replayed = true};
  }
  LinuxPidfdState state = launch.process_.state();
  if (state == LinuxPidfdState::live) {
    if (!request_termination) {
      reject("recovered process remains live and termination was not requested");
    }
    if (!launch.termination_requested_) {
      const LinuxRecoveredTerminationResult termination =
          launch.process_.request_termination();
      if (termination.disposition ==
          LinuxRecoveredTerminationDisposition::observation_failed) {
        reject("recovered process termination could not be delivered");
      }
      launch.termination_requested_ = true;
    }
    state = launch.process_.state();
    if (state == LinuxPidfdState::live) {
      throw LinuxRecoveredProcessPending(
          "recovered process termination is pending terminal observation");
    }
  }
  if (state != LinuxPidfdState::terminal) {
    reject("recovered process terminal state could not be observed");
  }
  const std::optional<std::string> observation_digest =
      launch.process_.terminal_observation_digest();
  if (!observation_digest || !launch.cgroup_.empty()) {
    reject("recovered process still has live allocation authority");
  }
  const LinuxProcessContextAudit context =
      context_auditor_.audit(grant, *launch.record_.spawn);
  if (!context.complete || !context.accelerator_contexts_empty) {
    reject("recovered process still has accelerator context authority");
  }
  const auto& spawn = *launch.record_.spawn;
  const auto& cgroup = launch.cgroup_.identity();
  const HostProcessRecoveryExitRequest request =
      seal_host_process_recovery_exit_request({
          .api_version =
              std::string(kHostProcessRecoveryExitRequestApiVersion),
          .recovery_exit_request_id = std::move(recovery_exit_request_id),
          .launch_id = spawn.request.launch_id,
          .spawn_receipt_digest = spawn.receipt_digest,
          .host_pid = spawn.request.host_pid,
          .process_starttime_ticks = spawn.request.process_starttime_ticks,
          .observation =
              HostProcessRecoveryExitObservation::pidfd_terminal,
          .observation_digest = *observation_digest,
          .cgroup_path = cgroup.unified_path,
          .cgroup_device = cgroup.device,
          .cgroup_inode = cgroup.inode,
          .cgroup_empty = true,
          .accelerator_contexts_empty = true,
          .context_audit_digest = context.evidence_digest,
          .canonical_request_digest = {},
      });
  const auto terminal = ledger_.commit_process_recovery_exit(
      request, ledger_time(clock_.sample(), spawn.request.boot_id));
  launch.exit_receipt_ = terminal.receipt;
  launch.cgroup_.remove_if_empty();
  launch.cgroup_removed_ = true;
  return terminal;
}

HostProcessRecoveryExitResult
LinuxProcessAuthority::finalize_nonlive_recovered_exit(
    const HostProcessRecoveryRecord& record,
    LinuxProcessRecoveryDisposition disposition,
    std::string observation_digest) {
  if (!record.spawn ||
      (disposition != LinuxProcessRecoveryDisposition::already_gone &&
       disposition != LinuxProcessRecoveryDisposition::identity_mismatch)) {
    reject("nonlive recovery requires a conclusive spawned observation");
  }
  const HostInventoryReceipt inventory = ledger_.inventory();
  const auto& spawn = *record.spawn;
  if (record.intent.request.allocation_id != record.grant.allocation_id ||
      record.intent.request.grant_digest != record.grant.receipt_digest ||
      spawn.request.launch_id != record.intent.request.launch_id ||
      spawn.request.boot_id != inventory.boot_id ||
      record.grant.host_id != inventory.host_id ||
      record.grant.boot_id != inventory.boot_id ||
      record.grant.broker_epoch != inventory.broker_epoch) {
    reject("nonlive recovery crossed its durable host or grant authority");
  }
  const std::vector<HostProcessRecoveryRecord> active =
      ledger_.active_process_recovery_records();
  const auto found = std::ranges::find_if(
      active, [&record](const HostProcessRecoveryRecord& candidate) {
        return candidate.intent.request.launch_id ==
               record.intent.request.launch_id;
      });
  if (found == active.end() || *found != record) {
    reject("nonlive recovery evidence is no longer active in the ledger");
  }
  const auto& identity = spawn.request;
  (void)cgroups_.cleanup_terminal_or_confirm_absent(
      record.grant.allocation_id, record.intent.request.launch_id,
      {.unified_path = identity.cgroup_path,
       .device = identity.cgroup_device,
       .inode = identity.cgroup_inode});
  const LinuxProcessContextAudit context =
      context_auditor_.audit(record.grant, spawn);
  if (!context.complete || !context.accelerator_contexts_empty) {
    reject("nonlive recovered process retains accelerator context authority");
  }
  const HostProcessRecoveryExitRequest request =
      seal_host_process_recovery_exit_request({
          .api_version =
              std::string(kHostProcessRecoveryExitRequestApiVersion),
          .recovery_exit_request_id =
              "hostd-observed-recovery-exit:" + identity.launch_id,
          .launch_id = identity.launch_id,
          .spawn_receipt_digest = spawn.receipt_digest,
          .host_pid = identity.host_pid,
          .process_starttime_ticks = identity.process_starttime_ticks,
          .observation =
              disposition == LinuxProcessRecoveryDisposition::already_gone
                  ? HostProcessRecoveryExitObservation::pid_absent
                  : HostProcessRecoveryExitObservation::identity_superseded,
          .observation_digest = std::move(observation_digest),
          .cgroup_path = identity.cgroup_path,
          .cgroup_device = identity.cgroup_device,
          .cgroup_inode = identity.cgroup_inode,
          .cgroup_empty = true,
          .accelerator_contexts_empty = true,
          .context_audit_digest = context.evidence_digest,
          .canonical_request_digest = {},
      });
  return ledger_.commit_process_recovery_exit(
      request, ledger_time(clock_.sample(), identity.boot_id));
}

struct HostdLinuxProcessSupervisor::Entry final {
  HostdProcessPrepareRequest request;
  LinuxPreparedLaunch launch;
  bool released_to_exec{};
  std::optional<HostdProcessExitCommand> exit_command;
  std::optional<HostProcessExitResult> exit_result;
};

struct HostdLinuxProcessSupervisor::RecoveredEntry final {
  LinuxRecoveredLaunch launch;
};

void HostdLinuxProcessSupervisor::require_process_binding(
    const HostdProcessCommitRequest& request, const Entry& entry) {
  const auto& grant = entry.request.grant;
  const auto& spawn = entry.launch.spawn_receipt();
  if (request.launch_id != entry.request.launch.identity.launch_event_id ||
      request.allocation_id != grant.allocation_id ||
      request.grant_digest != grant.receipt_digest ||
      request.journal_id != grant.journal_id ||
      request.run_id != grant.run_id ||
      request.logical_lease_id != grant.logical_lease_id ||
      request.logical_fencing_token != grant.logical_fencing_token ||
      request.spawn_receipt_digest != spawn.receipt_digest) {
    reject("process command does not match the retained prepared launch");
  }
}

namespace {
HostdProcessCommitRequest commit_identity(
    const HostdProcessExitCommand& request) {
  return {
      .api_version = std::string(kHostdProcessCommitApiVersion),
      .launch_id = request.launch_id,
      .allocation_id = request.allocation_id,
      .grant_digest = request.grant_digest,
      .journal_id = request.journal_id,
      .run_id = request.run_id,
      .logical_lease_id = request.logical_lease_id,
      .logical_fencing_token = request.logical_fencing_token,
      .spawn_receipt_digest = request.spawn_receipt_digest,
  };
}

}  // namespace

HostdLinuxProcessSupervisor::HostdLinuxProcessSupervisor(
    LinuxProcessAuthority& authority)
    : authority_(authority) {}

HostdLinuxProcessSupervisor::~HostdLinuxProcessSupervisor() = default;

HostdProcessPreparedResult HostdLinuxProcessSupervisor::prepare(
    const HostdProcessPrepareRequest& request, int executable_fd,
    std::optional<int> code_fd, int working_directory_fd,
    int worker_bootstrap_fd,
    std::optional<int> profiler_executable_fd,
    std::optional<int> profiler_authority_fd) {
  // Reattestation happens on retries too, so a successful replay never turns
  // descriptor role/count validation into a confused-deputy bypass.
  ResolvedLaunch resolved = ResolvedLaunch::adopt_delegated(
      request.launch, executable_fd, code_fd, working_directory_fd,
      profiler_executable_fd, profiler_authority_fd);
  std::scoped_lock lock(mutex_);
  const auto existing = entries_.find(request.launch.identity.launch_event_id);
  if (existing != entries_.end()) {
    if (existing->second->request != request)
      reject("process prepare replay changed its canonical request");
    return {
        .api_version = std::string(kHostdProcessPreparedApiVersion),
        .intent = existing->second->launch.intent(),
        .spawn = existing->second->launch.spawn_receipt(),
        .replayed = true,
    };
  }
  LinuxPreparedLaunch launch = authority_.prepare(
      resolved, request.grant, worker_bootstrap_fd,
      request.worker_bootstrap_digest, request.process_policy);
  auto entry = std::make_unique<Entry>(Entry{
      .request = request,
      .launch = std::move(launch),
      .released_to_exec = false,
      .exit_command = std::nullopt,
      .exit_result = std::nullopt,
  });
  HostdProcessPreparedResult result{
      .api_version = std::string(kHostdProcessPreparedApiVersion),
      .intent = entry->launch.intent(),
      .spawn = entry->launch.spawn_receipt(),
      .replayed = false,
  };
  entries_.emplace(request.launch.identity.launch_event_id, std::move(entry));
  return result;
}

HostdProcessCommittedResult HostdLinuxProcessSupervisor::commit(
    const HostdProcessCommitRequest& request) {
  std::scoped_lock lock(mutex_);
  const auto found = entries_.find(request.launch_id);
  if (found == entries_.end())
    reject("process commit has no retained prepared launch");
  Entry& entry = *found->second;
  require_process_binding(request, entry);
  if (entry.exit_result)
    reject("terminal process cannot be released to exec");
  const bool replayed = entry.released_to_exec;
  if (!entry.released_to_exec) {
    authority_.release_to_exec(entry.launch);
    entry.released_to_exec = true;
  }
  return {
      .api_version = std::string(kHostdProcessCommittedApiVersion),
      .launch_id = request.launch_id,
      .spawn_receipt_digest = request.spawn_receipt_digest,
      .released_to_exec = true,
      .replayed = replayed,
  };
}

HostProcessExitResult HostdLinuxProcessSupervisor::finalize(
    const HostdProcessExitCommand& request) {
  std::scoped_lock lock(mutex_);
  const auto found = entries_.find(request.launch_id);
  if (found == entries_.end())
    reject("process exit has no retained prepared launch");
  Entry& entry = *found->second;
  require_process_binding(commit_identity(request), entry);
  if (entry.exit_command) {
    if (*entry.exit_command != request || !entry.exit_result)
      reject("process exit replay changed its canonical request");
    HostProcessExitResult replay = *entry.exit_result;
    replay.replayed = true;
    return replay;
  }
  HostProcessExitResult result = authority_.finalize_exit(
      entry.launch, entry.request.grant, request.exit_request_id,
      request.request_termination);
  entry.exit_command = request;
  entry.exit_result = result;
  return result;
}

std::size_t HostdLinuxProcessSupervisor::adopt_exact_recovered_processes(
    LinuxProcessRecoverySet& recovery) {
  std::scoped_lock lock(mutex_);
  std::size_t adopted = 0U;
  for (const LinuxProcessRecoveryEntry& entry : recovery.entries()) {
    if (entry.disposition !=
            LinuxProcessRecoveryDisposition::exact_live_process ||
        !entry.process) {
      continue;
    }
    const std::string& launch_id = entry.record.intent.request.launch_id;
    if (recovered_entries_.contains(launch_id)) continue;
    if (entries_.contains(launch_id)) {
      reject("recovered launch identity is already retained");
    }
    std::optional<LinuxRecoveredProcess> process =
        recovery.take_exact_live_process_for_adoption(launch_id);
    if (!process) reject("exact recovered pidfd transfer was lost");
    LinuxRecoveredLaunch launch = authority_.adopt_recovered(
        entry.record, std::move(*process));
    recovered_entries_.emplace(
        launch_id, std::make_unique<RecoveredEntry>(
                       RecoveredEntry{.launch = std::move(launch)}));
    ++adopted;
  }
  return adopted;
}

std::size_t HostdLinuxProcessSupervisor::finalize_observed_nonlive_processes(
    const LinuxProcessRecoverySet& recovery) {
  std::scoped_lock lock(mutex_);
  std::size_t finalized = 0U;
  for (const LinuxProcessRecoveryEntry& entry : recovery.entries()) {
    if (entry.disposition != LinuxProcessRecoveryDisposition::already_gone &&
        entry.disposition !=
            LinuxProcessRecoveryDisposition::identity_mismatch) {
      continue;
    }
    (void)authority_.finalize_nonlive_recovered_exit(
        entry.record, entry.disposition, entry.evidence_digest);
    ++finalized;
  }
  return finalized;
}

HostdRecoveredProcessProgress
HostdLinuxProcessSupervisor::progress_recovered_terminations() {
  std::scoped_lock lock(mutex_);
  HostdRecoveredProcessProgress progress{
      .retained_before = recovered_entries_.size(),
  };
  auto entry = recovered_entries_.begin();
  while (entry != recovered_entries_.end()) {
    LinuxRecoveredLaunch& launch = entry->second->launch;
    try {
      (void)authority_.finalize_recovered_exit(
          launch, launch.record().grant,
          "hostd-recovered-exit:" + entry->first, true);
      entry = recovered_entries_.erase(entry);
      ++progress.finalized;
    } catch (const LinuxRecoveredProcessPending&) {
      ++progress.pending;
      ++entry;
    }
  }
  progress.retained_after = recovered_entries_.size();
  return progress;
}

}  // namespace trainvm
