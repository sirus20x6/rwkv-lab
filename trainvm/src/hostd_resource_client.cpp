#include "trainvm/hostd_resource_client.hpp"

#include <stdexcept>
#include <utility>

namespace trainvm {

HostdResourceClient::HostdResourceClient(HostdResourceClientOptions options)
    : open_for_request_(std::move(options.open_for_request)),
      channel_(std::move(options.channel)) {
  if (!open_for_request_) {
    throw std::invalid_argument(
        "hostd resource client has no mutation-open authority source");
  }
}

HostdMutationOpen HostdResourceClient::open_for(
    std::string_view request_id, std::string_view journal_id,
    std::string_view run_id, std::string_view logical_lease_id,
    std::uint64_t logical_fencing_token) const {
  HostdMutationOpen open = open_for_request_(request_id);
  (void)hostd_mutation_open_canonical_json(open);
  const auto& claim = open.claim;
  if (claim.journal.journal_id != journal_id ||
      claim.controller.run_id != run_id ||
      claim.controller.logical_lease_id != logical_lease_id ||
      claim.controller.logical_fencing_token != logical_fencing_token) {
    throw OperationPreconditionError(
        "hostd mutation claim disagrees with resource attribution");
  }
  return open;
}

BundleRequestResult HostdResourceClient::request_bundle(
    const ResourceBundleRequest& request) {
  validate_resource_request(request);
  HostdMutationRequest mutation{
      .open = open_for(request.request_id, request.journal_id, request.run_id,
                       request.logical_lease_id,
                       request.logical_fencing_token),
      .mutation = HostdMutationKind::request_bundle,
      .bundle_request = request,
  };
  const HostdMutationReply reply = channel_.request(std::move(mutation));
  if (reply.kind != HostdMutationReplyKind::bundle_outcome ||
      !reply.bundle_result) {
    throw HostdTransportError(
        "hostd resource request returned the wrong reply kind");
  }
  const BundleRequestResult& result = *reply.bundle_result;
  (void)bundle_request_result_json(result);
  if ((result.status == BundleRequestStatus::granted &&
       (!result.grant || result.grant->request_id != request.request_id ||
        result.grant->request_digest != request.canonical_request_digest ||
        result.grant->journal_id != request.journal_id ||
        result.grant->run_id != request.run_id ||
        result.grant->logical_lease_id != request.logical_lease_id ||
        result.grant->logical_fencing_token !=
            request.logical_fencing_token ||
        result.outcome_digest != result.grant->receipt_digest)) ||
      (result.status == BundleRequestStatus::busy && result.grant)) {
    throw HostdTransportError(
        "hostd resource outcome diverges from its request");
  }
  return result;
}

BundleReleaseResult HostdResourceClient::release_bundle(
    const ResourceReleaseRequest& request) {
  (void)resource_release_request_json(request);
  HostdMutationRequest mutation{
      .open = open_for(request.release_request_id, request.journal_id,
                       request.run_id, request.logical_lease_id,
                       request.logical_fencing_token),
      .mutation = HostdMutationKind::release_bundle,
      .release_request = request,
  };
  const HostdMutationReply reply = channel_.request(std::move(mutation));
  if (reply.kind != HostdMutationReplyKind::release_outcome ||
      !reply.release_result) {
    throw HostdTransportError(
        "hostd resource release returned the wrong reply kind");
  }
  const BundleReleaseResult& result = *reply.release_result;
  (void)bundle_release_result_json(result);
  if (result.receipt.release_request_id != request.release_request_id ||
      result.receipt.release_request_digest !=
          request.canonical_request_digest ||
      result.receipt.allocation_id != request.allocation_id ||
      result.receipt.grant_digest != request.grant_digest) {
    throw HostdTransportError(
        "hostd resource release receipt diverges from its request");
  }
  return result;
}

}  // namespace trainvm
