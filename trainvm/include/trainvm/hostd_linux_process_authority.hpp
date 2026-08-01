#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_stopped_launcher.hpp"
#include "trainvm/hostd_process_protocol.hpp"

namespace trainvm {

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
  void release_to_exec();

 private:
  friend class LinuxProcessAuthority;
  LinuxPreparedLaunch(HostProcessLaunchIntent intent,
                      HostProcessSpawnReceipt spawn_receipt,
                      LinuxAllocationCgroup cgroup,
                      LinuxStoppedChild child) noexcept;

  HostProcessLaunchIntent intent_;
  HostProcessSpawnReceipt spawn_receipt_;
  // Destruction is reverse declaration order: child is killed/reaped before
  // the retained cgroup descriptor is closed.
  LinuxAllocationCgroup cgroup_;
  LinuxStoppedChild child_;
  std::optional<HostProcessExitReceipt> exit_receipt_;
  bool cgroup_removed_{};
};

class LinuxProcessAuthority final {
 public:
  LinuxProcessAuthority(SQLiteHostLedger& ledger, AuthorityClock& clock,
                        LinuxCgroupAuthority& cgroups,
                        LinuxStoppedLauncherKernel& launcher,
                        ILinuxProcessContextAuditor& context_auditor);

  // Returns only after both intent and stopped-child identity are durable.
  // The returned gate remains closed until the caller has durably copied the
  // spawn receipt into its journal and calls release_to_exec().
  [[nodiscard]] LinuxPreparedLaunch prepare(
      const ResolvedLaunch& resolved, const ResourceBundleGrant& grant,
      int worker_bootstrap_fd, std::string_view worker_bootstrap_digest);
  [[nodiscard]] HostProcessExitResult finalize_exit(
      LinuxPreparedLaunch& launch, const ResourceBundleGrant& grant,
      std::string exit_request_id, bool request_termination);

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
  LinuxCgroupAuthority& cgroups_;
  LinuxStoppedLauncherKernel& launcher_;
  ILinuxProcessContextAuditor& context_auditor_;
};

class IHostdProcessSupervisor {
 public:
  virtual ~IHostdProcessSupervisor() = default;
  // Descriptor arguments are borrowed for the duration of prepare().
  [[nodiscard]] virtual HostdProcessPreparedResult prepare(
      const HostdProcessPrepareRequest& request, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd,
      int worker_bootstrap_fd) = 0;
  [[nodiscard]] virtual HostdProcessCommittedResult commit(
      const HostdProcessCommitRequest& request) = 0;
  [[nodiscard]] virtual HostProcessExitResult finalize(
      const HostdProcessExitCommand& request) = 0;
};

// Retains the stopped child, pidfd, cgroup descriptor, and grant until a
// terminal exit is durably recorded. Exact retries are replay-safe for the
// lifetime of one hostd process. Durable daemon-restart adoption remains a
// separate recovery boundary and is not claimed by this class.
class HostdLinuxProcessSupervisor final : public IHostdProcessSupervisor {
 public:
  explicit HostdLinuxProcessSupervisor(LinuxProcessAuthority& authority);
  ~HostdLinuxProcessSupervisor() override;

  [[nodiscard]] HostdProcessPreparedResult prepare(
      const HostdProcessPrepareRequest& request, int executable_fd,
      std::optional<int> code_fd, int working_directory_fd,
      int worker_bootstrap_fd) override;
  [[nodiscard]] HostdProcessCommittedResult commit(
      const HostdProcessCommitRequest& request) override;
  [[nodiscard]] HostProcessExitResult finalize(
      const HostdProcessExitCommand& request) override;

 private:
  struct Entry;
  static void require_process_binding(
      const HostdProcessCommitRequest& request, const Entry& entry);
  LinuxProcessAuthority& authority_;
  std::mutex mutex_;
  std::map<std::string, std::unique_ptr<Entry>, std::less<>> entries_;
};

}  // namespace trainvm
