#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/hostd_process_protocol.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/worker_bootstrap.hpp"

namespace trainvm {

enum class HostProcessSagaFaultPoint {
  before_prepare_host,
  after_prepare_host,
  after_prepare_journal,
  before_commit_host,
  after_commit_host,
  after_commit_journal,
  before_exit_host,
  after_exit_host,
  after_exit_journal,
};

class IHostProcessSagaFaultInjector {
 public:
  virtual ~IHostProcessSagaFaultInjector() = default;
  virtual void hit(HostProcessSagaFaultPoint point) = 0;
};

// Synchronous process mutation seam. Implementations may duplicate the
// borrowed descriptors during prepare, but must not retain the C++ objects.
class IHostProcessClient {
 public:
  virtual ~IHostProcessClient() = default;
  [[nodiscard]] virtual HostdProcessPreparedResult prepare_process(
      const HostdProcessPrepareRequest& request,
      const ResolvedLaunch& resolved,
      const SealedWorkerBootstrap& bootstrap) = 0;
  [[nodiscard]] virtual HostdProcessCommittedResult commit_process(
      const HostdProcessCommitRequest& request) = 0;
  [[nodiscard]] virtual HostProcessExitResult finalize_process(
      const HostdProcessExitCommand& request) = 0;
};

// Executes the stopped-child prepare -> durable journal copy -> exec commit
// transaction. The caller must serialize this object with the same authority
// mutex used by reconciliation and WorkerControl for the entire call.
class HostProcessSagaReconciler final {
 public:
  HostProcessSagaReconciler(
      Journal& journal, IHostProcessClient& host,
      IHostProcessSagaFaultInjector* faults = nullptr);

  [[nodiscard]] HostProcessSagaSnapshot reconcile(
      const ResolvedLaunch& resolved, const ResourceBundleGrant& grant,
      const LinuxProcessPolicy& process_policy, std::string controller_target,
      const AuthorityTimeSample& now);
  [[nodiscard]] HostProcessSagaSnapshot reconcile_exit(
      const std::string& launch_id, bool request_termination,
      const AuthorityTimeSample& now);

 private:
  void fault(HostProcessSagaFaultPoint point) const;

  Journal& journal_;
  IHostProcessClient& host_;
  IHostProcessSagaFaultInjector* faults_{};
};

}  // namespace trainvm
