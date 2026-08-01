#include "trainvm/host_process_saga.hpp"

#include <utility>
#include <vector>

namespace trainvm {

HostProcessSagaReconciler::HostProcessSagaReconciler(
    Journal& journal, IHostProcessClient& host,
    IHostProcessSagaFaultInjector* faults)
    : journal_(journal), host_(host), faults_(faults) {}

void HostProcessSagaReconciler::fault(
    HostProcessSagaFaultPoint point) const {
  if (faults_ != nullptr) faults_->hit(point);
}

HostProcessSagaSnapshot HostProcessSagaReconciler::reconcile(
    const ResolvedLaunch& resolved, const ResourceBundleGrant& grant,
    std::string controller_target, const AuthorityTimeSample& now) {
  const auto& identity = resolved.spec().identity;
  SealedWorkerBootstrap bootstrap = create_sealed_worker_bootstrap({
      .api_version = std::string(kWorkerBootstrapApiVersion),
      .controller_target = std::move(controller_target),
      .run_id = identity.run_id,
      .node_id = identity.node_id,
      .attempt_id = identity.attempt_id,
      .launch_nonce = identity.launch_nonce,
      .adapter = identity.adapter_key.adapter,
      .adapter_version = identity.adapter_key.version,
      .code_fingerprint = identity.code_fingerprint,
      .capabilities = identity.required_capabilities,
      .last_acked_controller_sequence = 0U,
      .concurrency_key = identity.concurrency_key,
      .lease_id = identity.lease_id,
      .fencing_token = identity.fencing_token,
      .bootstrap_digest = {},
  });
  const HostdProcessPrepareRequest prepare{
      .api_version = std::string(kHostdProcessPrepareApiVersion),
      .launch = resolved.spec(),
      .grant = grant,
      .worker_bootstrap_digest = bootstrap.spec().bootstrap_digest,
      .descriptor_roles =
          identity.code
              ? std::vector<std::string>{"executable", "code",
                                         "working_directory",
                                         "worker_bootstrap"}
              : std::vector<std::string>{"executable", "working_directory",
                                         "worker_bootstrap"},
  };
  (void)hostd_process_prepare_canonical_json(prepare);

  std::optional<HostProcessSagaSnapshot> saga =
      journal_.host_process_saga(identity.launch_event_id);
  if (saga && saga->prepare != prepare) {
    throw OperationPreconditionError(
        "host process retry changed the sealed prepare request");
  }
  if (!saga) {
    fault(HostProcessSagaFaultPoint::before_prepare_host);
    const HostdProcessPreparedResult prepared =
        host_.prepare_process(prepare, resolved, bootstrap);
    fault(HostProcessSagaFaultPoint::after_prepare_host);
    saga = journal_.record_host_process_prepared(prepare, prepared, now);
    fault(HostProcessSagaFaultPoint::after_prepare_journal);
  }
  if (saga->committed) return *saga;

  const HostdProcessCommitRequest commit{
      .api_version = std::string(kHostdProcessCommitApiVersion),
      .launch_id = identity.launch_event_id,
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .spawn_receipt_digest = saga->prepared.spawn.receipt_digest,
  };
  (void)hostd_process_commit_canonical_json(commit);
  fault(HostProcessSagaFaultPoint::before_commit_host);
  const HostdProcessCommittedResult committed = host_.commit_process(commit);
  fault(HostProcessSagaFaultPoint::after_commit_host);
  HostProcessSagaSnapshot complete =
      journal_.record_host_process_committed(commit, committed, now);
  fault(HostProcessSagaFaultPoint::after_commit_journal);
  return complete;
}

}  // namespace trainvm
