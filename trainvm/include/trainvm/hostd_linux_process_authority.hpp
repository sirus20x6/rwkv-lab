#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_device_kernel.hpp"
#include "trainvm/hostd_linux_process_recovery.hpp"
#include "trainvm/hostd_linux_process_policy_kernel.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/hostd_process_protocol.hpp"

namespace trainvm {

class LinuxRecoveredProcessPending final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct LinuxProcessContextAudit final {
  bool complete{};
  bool accelerator_contexts_empty{};
  std::string evidence_digest;

  bool operator==(const LinuxProcessContextAudit&) const = default;
};

class ILinuxProcessContextAuditor {
 public:
  virtual ~ILinuxProcessContextAuditor() = default;
  [[nodiscard]] virtual LinuxProcessContextAudit audit(
      const ResourceBundleGrant& grant,
      const HostProcessSpawnReceipt& spawn) = 0;
};

class LinuxPreparedLaunch final {
 public:
  LinuxPreparedLaunch(LinuxPreparedLaunch&&) noexcept = default;
  LinuxPreparedLaunch& operator=(LinuxPreparedLaunch&&) noexcept = default;

  LinuxPreparedLaunch(const LinuxPreparedLaunch&) = delete;
  LinuxPreparedLaunch& operator=(const LinuxPreparedLaunch&) = delete;

  [[nodiscard]] const HostProcessLaunchIntent& intent() const;
  [[nodiscard]] const HostProcessSpawnReceipt& spawn_receipt() const;
  [[nodiscard]] const LinuxStoppedChildIdentity& child_identity() const;
  [[nodiscard]] const std::optional<HostProcessExitReceipt>& exit_receipt() const;

 private:
  friend class LinuxProcessAuthority;
  LinuxPreparedLaunch(HostProcessLaunchIntent intent,
                      HostProcessSpawnReceipt spawn_receipt,
                      LinuxProcessPolicy process_policy,
                      LinuxProcessPolicyInstallation process_policy_installation,
                      LinuxAllocationCgroup cgroup,
                      LinuxStoppedChild child) noexcept;

  HostProcessLaunchIntent intent_;
  HostProcessSpawnReceipt spawn_receipt_;
  LinuxProcessPolicy process_policy_;
  LinuxProcessPolicyInstallation process_policy_installation_;
  // Destruction is reverse declaration order: child is killed/reaped before
  // the retained cgroup descriptor is closed.
  LinuxAllocationCgroup cgroup_;
  LinuxStoppedChild child_;
  std::optional<HostProcessExitReceipt> exit_receipt_;
  bool cgroup_removed_{};
};

class LinuxRecoveredLaunch final {
 public:
  LinuxRecoveredLaunch(LinuxRecoveredLaunch&&) noexcept = default;
  LinuxRecoveredLaunch& operator=(LinuxRecoveredLaunch&&) noexcept = default;

  LinuxRecoveredLaunch(const LinuxRecoveredLaunch&) = delete;
  LinuxRecoveredLaunch& operator=(const LinuxRecoveredLaunch&) = delete;

  [[nodiscard]] const HostProcessRecoveryRecord& record() const noexcept;
  [[nodiscard]] const LinuxRecoveredProcess& process() const noexcept;
  [[nodiscard]] const std::optional<HostProcessRecoveryExitReceipt>&
  exit_receipt() const noexcept;

 private:
  friend class LinuxProcessAuthority;
  LinuxRecoveredLaunch(HostProcessRecoveryRecord record,
                       LinuxRecoveredProcess process,
                       LinuxAllocationCgroup cgroup,
                       std::optional<LinuxDevicePolicyInstallation>
                           device_policy,
                       std::optional<LinuxProcessPolicy> process_policy,
                       std::optional<LinuxProcessPolicyInstallation>
                           process_policy_installation) noexcept;

  HostProcessRecoveryRecord record_;
  LinuxRecoveredProcess process_;
  LinuxAllocationCgroup cgroup_;
  std::optional<LinuxDevicePolicyInstallation> device_policy_;
  std::optional<LinuxProcessPolicy> process_policy_;
  std::optional<LinuxProcessPolicyInstallation>
      process_policy_installation_;
  bool termination_requested_{};
  std::optional<HostProcessRecoveryExitReceipt> exit_receipt_;
  bool cgroup_removed_{};
};

class LinuxProcessAuthority final {
 public:
  LinuxProcessAuthority(SQLiteHostLedger& ledger, AuthorityClock& clock,
                        LinuxCgroupAuthority& cgroups,
                        LinuxDevicePolicyInstaller* device_policies,
                        LinuxProcessPolicyInstaller& process_policies,
                        LinuxStoppedLauncherKernel& launcher,
                        LinuxWorkerCredentialSpec worker_credentials,
                        ILinuxProcessContextAuditor& context_auditor);

  // Returns only after both intent and stopped-child identity are durable.
  // The returned gate remains closed until the caller has durably copied the
  // spawn receipt into its journal and calls release_to_exec().
  [[nodiscard]] LinuxPreparedLaunch prepare(
      const ResolvedLaunch& resolved, const ResourceBundleGrant& grant,
      int worker_bootstrap_fd, std::string_view worker_bootstrap_digest,
      const LinuxProcessPolicy& process_policy);
  [[nodiscard]] HostProcessExitResult finalize_exit(
      LinuxPreparedLaunch& launch, const ResourceBundleGrant& grant,
      std::string exit_request_id, bool request_termination);
  void release_to_exec(LinuxPreparedLaunch& launch);
  [[nodiscard]] LinuxRecoveredLaunch adopt_recovered(
      HostProcessRecoveryRecord record, LinuxRecoveredProcess process);
  // If SIGKILL was just delivered and the pidfd remains live, this fails
  // closed with a retryable authority error; no terminal receipt is written
  // until a later call observes the pidfd terminal.
  [[nodiscard]] HostProcessRecoveryExitResult finalize_recovered_exit(
      LinuxRecoveredLaunch& launch, const ResourceBundleGrant& grant,
      std::string recovery_exit_request_id, bool request_termination);
  [[nodiscard]] HostProcessRecoveryExitResult
  finalize_nonlive_recovered_exit(
      const HostProcessRecoveryRecord& record,
      LinuxProcessRecoveryDisposition disposition,
      std::string observation_digest);

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
  LinuxCgroupAuthority& cgroups_;
  // Null when the authority is unprivileged: installing a cgroup device policy
  // needs CAP_BPF. Recovered records that carry device evidence are refused
  // rather than silently accepted without it.
  LinuxDevicePolicyInstaller* device_policies_;
  LinuxProcessPolicyInstaller& process_policies_;
  LinuxStoppedLauncherKernel& launcher_;
  LinuxWorkerCredentialSpec worker_credentials_;
  ILinuxProcessContextAuditor& context_auditor_;
};

class IHostdProcessSupervisor {
 public:
  virtual ~IHostdProcessSupervisor() = default;
  // Descriptor arguments are borrowed for the duration of prepare().
  [[nodiscard]] virtual HostdProcessPreparedResult prepare(
      const HostdProcessPrepareRequest& request, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd,
      int worker_bootstrap_fd,
      std::optional<int> profiler_executable_fd,
      std::optional<int> profiler_authority_fd) = 0;
  [[nodiscard]] virtual HostdProcessCommittedResult commit(
      const HostdProcessCommitRequest& request) = 0;
  [[nodiscard]] virtual HostProcessExitResult finalize(
      const HostdProcessExitCommand& request) = 0;
};

struct HostdRecoveredProcessProgress final {
  std::size_t retained_before{};
  std::size_t finalized{};
  std::size_t pending{};
  std::size_t retained_after{};

  bool operator==(const HostdRecoveredProcessProgress&) const = default;
};

class IHostdRecoveredProcessSupervisor {
 public:
  virtual ~IHostdRecoveredProcessSupervisor() = default;
  [[nodiscard]] virtual std::size_t adopt_exact_recovered_processes(
      LinuxProcessRecoverySet& recovery) = 0;
  [[nodiscard]] virtual std::size_t finalize_observed_nonlive_processes(
      const LinuxProcessRecoverySet& recovery) = 0;
  [[nodiscard]] virtual HostdRecoveredProcessProgress
  progress_recovered_terminations() = 0;
};

// Retains current stopped launches and explicitly adopted restart pidfds,
// cgroup descriptors, and grants until terminal evidence is durable. Ordinary
// command retries remain scoped to one daemon lifetime; restart adoption is a
// separate explicit method and never happens from a numeric PID alone.
class HostdLinuxProcessSupervisor final : public IHostdProcessSupervisor,
                                          public IHostdRecoveredProcessSupervisor {
 public:
  explicit HostdLinuxProcessSupervisor(LinuxProcessAuthority& authority);
  ~HostdLinuxProcessSupervisor() override;

  [[nodiscard]] HostdProcessPreparedResult prepare(
      const HostdProcessPrepareRequest& request, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd,
      int worker_bootstrap_fd,
      std::optional<int> profiler_executable_fd,
      std::optional<int> profiler_authority_fd) override;
  [[nodiscard]] HostdProcessCommittedResult commit(
      const HostdProcessCommitRequest& request) override;
  [[nodiscard]] HostProcessExitResult finalize(
      const HostdProcessExitCommand& request) override;
  [[nodiscard]] std::size_t adopt_exact_recovered_processes(
      LinuxProcessRecoverySet& recovery) override;
  [[nodiscard]] std::size_t finalize_observed_nonlive_processes(
      const LinuxProcessRecoverySet& recovery) override;
  [[nodiscard]] HostdRecoveredProcessProgress
  progress_recovered_terminations() override;

 private:
  struct Entry;
  struct RecoveredEntry;
  static void require_process_binding(
      const HostdProcessCommitRequest& request, const Entry& entry);
  LinuxProcessAuthority& authority_;
  std::mutex mutex_;
  std::map<std::string, std::unique_ptr<Entry>, std::less<>> entries_;
  std::map<std::string, std::unique_ptr<RecoveredEntry>, std::less<>>
      recovered_entries_;
};

}  // namespace trainvm
