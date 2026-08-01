#include "trainvm/hostd_linux_process_authority.hpp"

#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>

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
    LinuxAllocationCgroup cgroup, LinuxStoppedChild child) noexcept
    : intent_(std::move(intent)), spawn_receipt_(std::move(spawn_receipt)),
      cgroup_(std::move(cgroup)), child_(std::move(child)) {}

const HostProcessLaunchIntent& LinuxPreparedLaunch::intent() const {
  return intent_;
}

const HostProcessSpawnReceipt& LinuxPreparedLaunch::spawn_receipt() const {
  return spawn_receipt_;
}

const LinuxStoppedChildIdentity& LinuxPreparedLaunch::child_identity() const {
  return child_.identity();
}

void LinuxPreparedLaunch::release_to_exec() { child_.release_to_exec(); }

LinuxProcessAuthority::LinuxProcessAuthority(
    SQLiteHostLedger& ledger, AuthorityClock& clock,
    LinuxCgroupAuthority& cgroups, LinuxStoppedLauncherKernel& launcher)
    : ledger_(ledger), clock_(clock), cgroups_(cgroups), launcher_(launcher) {}

LinuxPreparedLaunch LinuxProcessAuthority::prepare(
    const ResolvedLaunch& resolved, const ResourceBundleGrant& grant) {
  const ResolvedLaunchSpec spec = resolved_launch_spec_from_json(
      resolved_launch_spec_json(resolved.spec()));
  const HostInventoryReceipt inventory = ledger_.inventory();
  validate_binding(spec, grant, inventory);
  const std::optional<int> code = resolved.duplicate_code_fd();
  if (code) {
    (void)::close(*code);
    reject("Python code-fd argv binding is not yet process-authority enabled");
  }
  LinuxAllocationCgroup cgroup = cgroups_.open_or_create(
      grant.allocation_id, spec.identity.launch_event_id);
  const auto& cgroup_identity = cgroup.identity();
  const HostProcessLaunchRequest request =
      seal_host_process_launch_request({
          .api_version = std::string(kHostProcessLaunchRequestApiVersion),
          .launch_id = spec.identity.launch_event_id,
          .allocation_id = grant.allocation_id,
          .grant_digest = grant.receipt_digest,
          .journal_id = grant.journal_id,
          .run_id = grant.run_id,
          .logical_lease_id = grant.logical_lease_id,
          .logical_fencing_token = grant.logical_fencing_token,
          .resolved_launch_digest = spec.spec_digest,
          .executable_path = spec.identity.executable.source_path,
          .executable_digest = spec.identity.executable.sealed_sha256,
          .cgroup_path = cgroup_identity.unified_path,
          .cgroup_device = cgroup_identity.device,
          .cgroup_inode = cgroup_identity.inode,
          .canonical_request_digest = {},
      });
  const auto intended = ledger_.commit_process_launch_intent(
      request, ledger_time(clock_.sample(), inventory.boot_id));
  cgroup.retain_for_durable_intent();
  Descriptor cgroup_fd(cgroup.duplicate_fd());
  Descriptor executable_fd(resolved.duplicate_executable_fd());
  Descriptor working_directory_fd(
      resolved.duplicate_working_directory_fd());
  LinuxStoppedChild child = launcher_.spawn_stopped({
      .launch_id = request.launch_id,
      .cgroup_fd = cgroup_fd.get(),
      .expected_cgroup_path = cgroup_identity.unified_path,
      .expected_cgroup_device = cgroup_identity.device,
      .expected_cgroup_inode = cgroup_identity.inode,
      .executable_fd = executable_fd.get(),
      .executable_name = spec.identity.executable.source_path,
      .executable_digest = spec.identity.executable.sealed_sha256,
      .working_directory_fd = working_directory_fd.get(),
      .arguments = spec.identity.public_arguments,
  });
  const LinuxStoppedChildIdentity& observed = child.identity();
  const HostProcessSpawnRequest spawn_request =
      seal_host_process_spawn_request({
          .api_version = std::string(kHostProcessSpawnRequestApiVersion),
          .launch_id = request.launch_id,
          .launch_intent_digest = intended.intent.receipt_digest,
          .host_pid = observed.host_pid,
          .process_starttime_ticks = observed.process_starttime_ticks,
          .boot_id = inventory.boot_id,
          .cgroup_path = observed.cgroup_path,
          .cgroup_device = observed.cgroup_device,
          .cgroup_inode = observed.cgroup_inode,
          .executable_digest = observed.executable_digest,
          .canonical_request_digest = {},
      });
  const auto spawned = ledger_.commit_process_spawn(
      spawn_request, ledger_time(clock_.sample(), inventory.boot_id));
  return LinuxPreparedLaunch(intended.intent, spawned.receipt,
                             std::move(cgroup), std::move(child));
}

}  // namespace trainvm
