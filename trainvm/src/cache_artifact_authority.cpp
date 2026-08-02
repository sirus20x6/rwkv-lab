#include "trainvm/cache_artifact_authority.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <ranges>
#include <string_view>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::uint64_t kMaximumCacheFiles = 1'000'000U;
constexpr std::uint64_t kMaximumCacheBytes = 1ULL << 50U;

bool valid_sha256(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool bounded_text(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum &&
         value.find('\0') == std::string_view::npos &&
         value.find('\n') == std::string_view::npos &&
         value.find('\r') == std::string_view::npos;
}

std::string digest(std::string_view domain, const nlohmann::json& value) {
  return "sha256:" +
         sha256_hex(
             nlohmann::json{{"domain", domain}, {"value", value}}.dump());
}

void validate_lease(const ResourceLease& lease) {
  if (!bounded_text(lease.concurrency_key, 1024U) ||
      !bounded_text(lease.owner_run_id, 1024U) ||
      !bounded_text(lease.lease_id, 1024U) || lease.fencing_token == 0U ||
      lease.clock_domain != ResourceLease::kBootTimeDomain ||
      !bounded_text(lease.boot_id, 256U) || lease.acquired_boottime_ns < 0 ||
      lease.expires_boottime_ns <= lease.acquired_boottime_ns ||
      lease.acquired_wall_time_ns < 0 ||
      lease.expires_wall_time_ns < lease.acquired_wall_time_ns) {
    throw CacheArtifactAuthorityError("cache artifact lease is malformed");
  }
}

nlohmann::json qualification_body(const CacheQualificationReceipt& receipt) {
  return {{"api_version", receipt.api_version},
          {"evidence", encode_json(receipt.evidence)},
          {"qualified", receipt.qualified},
          {"rejection_reasons", receipt.rejection_reasons}};
}

std::string qualification_digest(const CacheQualificationReceipt& receipt) {
  return digest("trainvm.cache-qualification-receipt/v1",
                qualification_body(receipt));
}

std::vector<std::string>
qualification_rejection_reasons(const CacheQualificationEvidence& evidence) {
  if (evidence.api_version != "trainvm.cache-qualification-evidence/v1" ||
      !valid_sha256(evidence.authority_receipt_digest) ||
      !valid_sha256(evidence.namespace_digest) ||
      !valid_sha256(evidence.artifact_tree_digest) ||
      !valid_sha256(evidence.baseline_run_digest) ||
      !valid_sha256(evidence.candidate_run_digest) ||
      !valid_sha256(evidence.shape_coverage_digest) ||
      (evidence.workload_class != CacheWorkloadClass::training &&
       evidence.workload_class != CacheWorkloadClass::serving &&
       evidence.workload_class != CacheWorkloadClass::preprocessing) ||
      !std::isfinite(evidence.baseline_throughput) ||
      !std::isfinite(evidence.candidate_throughput) ||
      evidence.baseline_throughput <= 0.0 ||
      evidence.candidate_throughput <= 0.0 ||
      evidence.baseline_peak_memory_bytes == 0U ||
      evidence.candidate_peak_memory_bytes == 0U ||
      !std::isfinite(evidence.minimum_throughput_gain_ratio) ||
      evidence.minimum_throughput_gain_ratio < 0.0 ||
      evidence.minimum_throughput_gain_ratio > 10.0 ||
      !std::isfinite(evidence.maximum_memory_regression_ratio) ||
      evidence.maximum_memory_regression_ratio < 0.0 ||
      evidence.maximum_memory_regression_ratio > 10.0) {
    throw CacheArtifactAuthorityError(
        "cache qualification evidence is malformed or unbounded");
  }

  std::vector<std::string> reasons;
  const auto reject = [&](bool condition, std::string reason) {
    if (condition)
      reasons.push_back(std::move(reason));
  };
  reject(evidence.baseline_instrumented || evidence.candidate_instrumented,
         "instrumented_timing");
  reject(!evidence.transition_coverage, "missing_transition_coverage");
  reject(!evidence.output_parity, "output_parity_failed");
  reject(!evidence.determinism_parity, "determinism_parity_failed");
  if (evidence.workload_class == CacheWorkloadClass::training) {
    reject(!evidence.gradient_parity, "gradient_parity_failed");
    reject(!evidence.optimizer_update_parity, "optimizer_update_parity_failed");
    reject(!evidence.state_parity, "state_parity_failed");
    reject(!evidence.resumed_trajectory_parity,
           "resumed_trajectory_parity_failed");
    reject(!evidence.model_quality_pass, "model_quality_failed");
  } else if (evidence.workload_class == CacheWorkloadClass::serving) {
    reject(!evidence.state_parity, "state_parity_failed");
    reject(!evidence.model_quality_pass, "model_quality_failed");
  } else {
    reject(!evidence.content_parity, "content_parity_failed");
    reject(!evidence.ordering_parity, "ordering_parity_failed");
    reject(!evidence.manifest_parity, "manifest_parity_failed");
  }
  const double gain =
      evidence.candidate_throughput / evidence.baseline_throughput - 1.0;
  const double memory_regression =
      static_cast<double>(evidence.candidate_peak_memory_bytes) /
          static_cast<double>(evidence.baseline_peak_memory_bytes) -
      1.0;
  reject(gain < evidence.minimum_throughput_gain_ratio,
         "throughput_gate_failed");
  reject(memory_regression > evidence.maximum_memory_regression_ratio,
         "memory_gate_failed");
  std::ranges::sort(reasons);
  reasons.erase(std::ranges::unique(reasons).begin(), reasons.end());
  return reasons;
}

void validate_qualification_receipt(const CacheQualificationReceipt& receipt) {
  const std::vector<std::string> expected_reasons =
      qualification_rejection_reasons(receipt.evidence);
  if (receipt.api_version != "trainvm.cache-qualification/v1" ||
      !valid_sha256(receipt.receipt_digest) ||
      receipt.receipt_digest != qualification_digest(receipt) ||
      receipt.qualified != receipt.rejection_reasons.empty() ||
      receipt.rejection_reasons != expected_reasons ||
      receipt.rejection_reasons.size() > 32U ||
      !std::ranges::is_sorted(receipt.rejection_reasons) ||
      std::ranges::adjacent_find(receipt.rejection_reasons) !=
          receipt.rejection_reasons.end()) {
    throw CacheArtifactAuthorityError(
        "cache qualification receipt is self-inconsistent");
  }
}

void validate_tree_receipt(const ImmutableCacheTreeReceipt& receipt) {
  if (receipt.api_version != "trainvm.immutable-cache-tree/v1" ||
      !valid_sha256(receipt.namespace_digest) ||
      !valid_sha256(receipt.artifact_tree_digest) ||
      !valid_sha256(receipt.manifest_digest) ||
      !valid_sha256(receipt.store_receipt_digest) ||
      !bounded_text(receipt.content_address, 4096U) ||
      receipt.file_count == 0U || receipt.file_count > kMaximumCacheFiles ||
      receipt.total_bytes == 0U || receipt.total_bytes > kMaximumCacheBytes ||
      !receipt.immutable) {
    throw CacheArtifactAuthorityError(
        "immutable cache store returned an invalid receipt");
  }
}

nlohmann::json
publication_body(const CacheArtifactPublicationReceipt& receipt) {
  return {
      {"api_version", receipt.api_version},
      {"authority", cache_namespace_authority_receipt_json(receipt.authority)},
      {"qualification",
       cache_qualification_receipt_json(receipt.qualification)},
      {"publisher_lease", encode_json(receipt.publisher_lease)},
      {"artifact", encode_json(receipt.artifact)},
  };
}

std::string publication_digest(const CacheArtifactPublicationReceipt& receipt) {
  return digest("trainvm.cache-artifact-publication/v1",
                publication_body(receipt));
}

void validate_publication(const CacheArtifactPublicationReceipt& receipt) {
  (void)cache_namespace_authority_receipt_json(receipt.authority);
  validate_qualification_receipt(receipt.qualification);
  validate_lease(receipt.publisher_lease);
  validate_tree_receipt(receipt.artifact);
  const auto& namespace_digest =
      receipt.authority.cache_namespace.namespace_digest;
  if (receipt.api_version != "trainvm.cache-artifact-publication/v1" ||
      !receipt.qualification.qualified ||
      receipt.qualification.evidence.authority_receipt_digest !=
          receipt.authority.receipt_digest ||
      receipt.qualification.evidence.namespace_digest != namespace_digest ||
      receipt.qualification.evidence.artifact_tree_digest !=
          receipt.artifact.artifact_tree_digest ||
      receipt.artifact.namespace_digest != namespace_digest ||
      receipt.publisher_lease.owner_run_id != receipt.authority.run_id ||
      receipt.publisher_lease.concurrency_key !=
          receipt.authority.concurrency_key ||
      receipt.publisher_lease.lease_id != receipt.authority.lease_id ||
      receipt.publisher_lease.fencing_token !=
          receipt.authority.fencing_token ||
      receipt.publication_digest != publication_digest(receipt)) {
    throw CacheArtifactAuthorityError(
        "cache artifact publication receipt is self-inconsistent");
  }
}

nlohmann::json adoption_body(const CacheArtifactAdoptionGrant& grant) {
  return {{"api_version", grant.api_version},
          {"namespace_digest", grant.namespace_digest},
          {"content_address", grant.content_address},
          {"artifact_tree_digest", grant.artifact_tree_digest},
          {"publication_digest", grant.publication_digest},
          {"current_authority_receipt_digest",
           grant.current_authority_receipt_digest},
          {"adopter_lease", encode_json(grant.adopter_lease)}};
}

std::string adoption_digest(const CacheArtifactAdoptionGrant& grant) {
  return digest("trainvm.cache-artifact-adoption-grant/v1",
                adoption_body(grant));
}

void validate_adoption(const CacheArtifactAdoptionGrant& grant) {
  validate_lease(grant.adopter_lease);
  if (grant.api_version != "trainvm.cache-artifact-adoption/v1" ||
      !valid_sha256(grant.namespace_digest) ||
      !valid_sha256(grant.artifact_tree_digest) ||
      !valid_sha256(grant.publication_digest) ||
      !valid_sha256(grant.current_authority_receipt_digest) ||
      !bounded_text(grant.content_address, 4096U) ||
      grant.grant_digest != adoption_digest(grant)) {
    throw CacheArtifactAuthorityError(
        "cache artifact adoption grant is self-inconsistent");
  }
}

} // namespace

CacheQualificationReceipt
qualify_cache_artifact(CacheQualificationEvidence evidence) {
  std::vector<std::string> reasons = qualification_rejection_reasons(evidence);
  CacheQualificationReceipt receipt{
      .api_version = "trainvm.cache-qualification/v1",
      .evidence = std::move(evidence),
      .qualified = reasons.empty(),
      .rejection_reasons = std::move(reasons),
      .receipt_digest = {},
  };
  receipt.receipt_digest = qualification_digest(receipt);
  validate_qualification_receipt(receipt);
  return receipt;
}

nlohmann::json
cache_qualification_receipt_json(const CacheQualificationReceipt& receipt) {
  validate_qualification_receipt(receipt);
  nlohmann::json result = qualification_body(receipt);
  result["receipt_digest"] = receipt.receipt_digest;
  return result;
}

CacheArtifactAuthority::CacheArtifactAuthority(
    ICacheLeaseAuthority& leases, ICacheArtifactStore& store,
    ICacheQualificationEvidenceSource& qualification)
    : leases_(leases), store_(store), qualification_(qualification) {}

CacheArtifactPublicationReceipt
CacheArtifactAuthority::publish(CacheArtifactPublicationRequest request) const {
  (void)cache_namespace_authority_receipt_json(request.authority);
  validate_lease(request.publisher_lease);
  if (request.publisher_lease.owner_run_id != request.authority.run_id ||
      request.publisher_lease.concurrency_key !=
          request.authority.concurrency_key ||
      request.publisher_lease.lease_id != request.authority.lease_id ||
      request.publisher_lease.fencing_token !=
          request.authority.fencing_token ||
      !std::filesystem::path(request.candidate.source_directory)
           .is_absolute() ||
      request.candidate.maximum_file_count == 0U ||
      request.candidate.maximum_file_count > kMaximumCacheFiles ||
      request.candidate.maximum_total_bytes == 0U ||
      request.candidate.maximum_total_bytes > kMaximumCacheBytes) {
    throw CacheArtifactAuthorityError(
        "cache publication request is unbound or exceeds policy bounds");
  }
  leases_.require_current(request.publisher_lease);
  ImmutableCacheTreeReceipt tree =
      store_.publish(request.authority, request.candidate);
  validate_tree_receipt(tree);
  if (tree.namespace_digest !=
          request.authority.cache_namespace.namespace_digest ||
      tree.file_count > request.candidate.maximum_file_count ||
      tree.total_bytes > request.candidate.maximum_total_bytes) {
    throw CacheArtifactAuthorityError(
        "immutable cache tree disagrees with namespace or publication bounds");
  }
  CacheQualificationEvidence evidence =
      qualification_.capture(request.authority, tree);
  evidence.authority_receipt_digest = request.authority.receipt_digest;
  evidence.namespace_digest = tree.namespace_digest;
  evidence.artifact_tree_digest = tree.artifact_tree_digest;
  CacheQualificationReceipt qualification =
      qualify_cache_artifact(std::move(evidence));
  if (!qualification.qualified) {
    throw CacheArtifactAuthorityError(
        "cache candidate failed qualification and cannot be published");
  }
  // Close the potentially long store/qualification window against lease loss.
  leases_.require_current(request.publisher_lease);
  CacheArtifactPublicationReceipt receipt{
      .api_version = "trainvm.cache-artifact-publication/v1",
      .authority = std::move(request.authority),
      .qualification = std::move(qualification),
      .publisher_lease = std::move(request.publisher_lease),
      .artifact = std::move(tree),
      .publication_digest = {},
  };
  receipt.publication_digest = publication_digest(receipt);
  validate_publication(receipt);
  return receipt;
}

CacheArtifactAdoptionGrant
CacheArtifactAuthority::adopt(CacheArtifactAdoptionRequest request) const {
  (void)cache_namespace_authority_receipt_json(request.current_authority);
  validate_publication(request.publication);
  validate_lease(request.adopter_lease);
  if (request.adopter_lease.owner_run_id != request.current_authority.run_id ||
      request.adopter_lease.concurrency_key !=
          request.current_authority.concurrency_key ||
      request.adopter_lease.lease_id != request.current_authority.lease_id ||
      request.adopter_lease.fencing_token !=
          request.current_authority.fencing_token ||
      request.current_authority.cache_namespace.namespace_digest !=
          request.publication.authority.cache_namespace.namespace_digest) {
    throw CacheArtifactAuthorityError(
        "cache adoption does not bind the current owner or exact namespace");
  }
  leases_.require_current(request.adopter_lease);
  qualification_.require_trusted(request.publication.authority,
                                 request.publication.artifact,
                                 request.publication.qualification);
  const ImmutableCacheTreeReceipt verified =
      store_.verify(request.publication.artifact.content_address);
  validate_tree_receipt(verified);
  if (verified != request.publication.artifact) {
    throw CacheArtifactAuthorityError(
        "cache artifact changed after immutable publication");
  }
  leases_.require_current(request.adopter_lease);
  CacheArtifactAdoptionGrant grant{
      .api_version = "trainvm.cache-artifact-adoption/v1",
      .namespace_digest = verified.namespace_digest,
      .content_address = verified.content_address,
      .artifact_tree_digest = verified.artifact_tree_digest,
      .publication_digest = request.publication.publication_digest,
      .current_authority_receipt_digest =
          request.current_authority.receipt_digest,
      .adopter_lease = std::move(request.adopter_lease),
      .grant_digest = {},
  };
  grant.grant_digest = adoption_digest(grant);
  validate_adoption(grant);
  return grant;
}

nlohmann::json cache_artifact_publication_receipt_json(
    const CacheArtifactPublicationReceipt& receipt) {
  validate_publication(receipt);
  nlohmann::json result = publication_body(receipt);
  result["publication_digest"] = receipt.publication_digest;
  return result;
}

nlohmann::json
cache_artifact_adoption_grant_json(const CacheArtifactAdoptionGrant& grant) {
  validate_adoption(grant);
  nlohmann::json result = adoption_body(grant);
  result["grant_digest"] = grant.grant_digest;
  return result;
}

} // namespace trainvm
