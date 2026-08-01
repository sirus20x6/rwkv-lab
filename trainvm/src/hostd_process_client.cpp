#include "trainvm/hostd_process_client.hpp"

#include <optional>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace trainvm {
namespace {

class Descriptor final {
 public:
  explicit Descriptor(int value = -1) noexcept : value_(value) {}
  ~Descriptor() {
    if (value_ >= 0) (void)::close(value_);
  }
  Descriptor(const Descriptor&) = delete;
  Descriptor& operator=(const Descriptor&) = delete;
  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_;
};

void require_bootstrap_binding(const HostdProcessPrepareRequest& request,
                               const WorkerBootstrapSpec& bootstrap) {
  const auto& identity = request.launch.identity;
  if (bootstrap.bootstrap_digest != request.worker_bootstrap_digest ||
      bootstrap.run_id != identity.run_id ||
      bootstrap.node_id != identity.node_id ||
      bootstrap.attempt_id != identity.attempt_id ||
      bootstrap.launch_nonce != identity.launch_nonce ||
      bootstrap.adapter != identity.adapter_key.adapter ||
      bootstrap.adapter_version != identity.adapter_key.version ||
      bootstrap.code_fingerprint != identity.code_fingerprint ||
      bootstrap.capabilities != identity.required_capabilities ||
      bootstrap.concurrency_key != identity.concurrency_key ||
      bootstrap.lease_id != identity.lease_id ||
      bootstrap.fencing_token != identity.fencing_token) {
    throw OperationPreconditionError(
        "hostd client bootstrap disagrees with resolved launch authority");
  }
}

}  // namespace

HostdProcessClient::HostdProcessClient(HostdProcessClientOptions options)
    : open_for_launch_(std::move(options.open_for_launch)),
      channel_(std::move(options.channel)) {
  if (!open_for_launch_) {
    throw std::invalid_argument(
        "hostd process client has no mutation-open authority source");
  }
}

HostdMutationOpen HostdProcessClient::open_for(
    std::string_view launch_id, std::string_view journal_id,
    std::string_view run_id, std::string_view logical_lease_id,
    std::uint64_t logical_fencing_token,
    std::string_view concurrency_key) const {
  HostdMutationOpen open = open_for_launch_(launch_id);
  (void)hostd_mutation_open_canonical_json(open);
  const auto& claim = open.claim;
  if (claim.journal.journal_id != journal_id ||
      claim.controller.run_id != run_id ||
      claim.controller.logical_lease_id != logical_lease_id ||
      claim.controller.logical_fencing_token != logical_fencing_token ||
      (!concurrency_key.empty() &&
       claim.controller.concurrency_key != concurrency_key)) {
    throw OperationPreconditionError(
        "hostd mutation claim disagrees with process attribution");
  }
  return open;
}

HostdProcessPreparedResult HostdProcessClient::prepare_process(
    const HostdProcessPrepareRequest& request,
    const ResolvedLaunch& resolved,
    const SealedWorkerBootstrap& bootstrap) {
  (void)hostd_process_prepare_canonical_json(request);
  if (resolved.spec() != request.launch) {
    throw OperationPreconditionError(
        "hostd client descriptors disagree with process prepare identity");
  }
  require_bootstrap_binding(request, bootstrap.spec());
  const auto& identity = request.launch.identity;
  HostdMutationOpen open = open_for(
      identity.launch_event_id, request.grant.journal_id,
      request.grant.run_id, request.grant.logical_lease_id,
      request.grant.logical_fencing_token, identity.concurrency_key);

  Descriptor executable(resolved.duplicate_executable_fd());
  const std::optional<int> code_value = resolved.duplicate_code_fd();
  Descriptor code(code_value.value_or(-1));
  Descriptor working_directory(resolved.duplicate_working_directory_fd());
  Descriptor bootstrap_descriptor(bootstrap.duplicate_fd());
  HostdMutationRequest mutation{
      .open = std::move(open),
      .mutation = HostdMutationKind::prepare_process,
      .process_prepare = request,
      .delegated_launch =
          HostdMutationRequest::DelegatedLaunchDescriptors{
              .executable_fd = executable.get(),
              .code_fd = code_value ? std::optional<int>{code.get()}
                                    : std::nullopt,
              .working_directory_fd = working_directory.get(),
              .worker_bootstrap_fd = bootstrap_descriptor.get()},
  };
  const HostdMutationReply reply = channel_.request(std::move(mutation));
  if (reply.kind != HostdMutationReplyKind::process_prepared ||
      !reply.process_prepared) {
    throw HostdTransportError(
        "hostd process prepare returned the wrong reply kind");
  }
  const auto& result = *reply.process_prepared;
  (void)hostd_process_prepared_canonical_json(result);
  if (result.intent.request.launch_id != identity.launch_event_id ||
      result.intent.request.allocation_id != request.grant.allocation_id ||
      result.intent.request.grant_digest != request.grant.receipt_digest ||
      result.intent.request.resolved_launch_digest !=
          hostd_bound_process_launch_digest(
              request.launch, request.worker_bootstrap_digest) ||
      result.spawn.request.launch_id != identity.launch_event_id) {
    throw HostdTransportError(
        "hostd process prepare reply diverges from its request");
  }
  return result;
}

HostdProcessCommittedResult HostdProcessClient::commit_process(
    const HostdProcessCommitRequest& request) {
  (void)hostd_process_commit_canonical_json(request);
  HostdMutationRequest mutation{
      .open = open_for(request.launch_id, request.journal_id, request.run_id,
                       request.logical_lease_id,
                       request.logical_fencing_token),
      .mutation = HostdMutationKind::commit_process,
      .process_commit = request,
  };
  const HostdMutationReply reply = channel_.request(std::move(mutation));
  if (reply.kind != HostdMutationReplyKind::process_committed ||
      !reply.process_committed) {
    throw HostdTransportError(
        "hostd process commit returned the wrong reply kind");
  }
  const auto& result = *reply.process_committed;
  (void)hostd_process_committed_canonical_json(result);
  if (result.launch_id != request.launch_id ||
      result.spawn_receipt_digest != request.spawn_receipt_digest ||
      !result.released_to_exec) {
    throw HostdTransportError(
        "hostd process commit reply diverges from its request");
  }
  return result;
}

HostProcessExitResult HostdProcessClient::finalize_process(
    const HostdProcessExitCommand& request) {
  (void)hostd_process_exit_canonical_json(request);
  HostdMutationRequest mutation{
      .open = open_for(request.launch_id, request.journal_id, request.run_id,
                       request.logical_lease_id,
                       request.logical_fencing_token),
      .mutation = HostdMutationKind::finalize_process,
      .process_exit = request,
  };
  const HostdMutationReply reply = channel_.request(std::move(mutation));
  if (reply.kind != HostdMutationReplyKind::process_exited ||
      !reply.process_exit) {
    throw HostdTransportError(
        "hostd process finalize returned the wrong reply kind");
  }
  const auto& result = *reply.process_exit;
  (void)host_process_exit_receipt_json(result.receipt);
  if (result.receipt.request.exit_request_id != request.exit_request_id ||
      result.receipt.request.launch_id != request.launch_id ||
      result.receipt.request.spawn_receipt_digest !=
          request.spawn_receipt_digest) {
    throw HostdTransportError(
        "hostd process finalize reply diverges from its request");
  }
  return result;
}

}  // namespace trainvm
