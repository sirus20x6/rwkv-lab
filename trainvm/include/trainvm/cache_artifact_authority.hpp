#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/cache_namespace_authority.hpp"
#include "trainvm/lease.hpp"

namespace trainvm {

enum class CacheWorkloadClass {
  training,
  serving,
  preprocessing,
};

// Unprofiled baseline/candidate evidence. Raw profiler timing is deliberately
// excluded because instrumentation changes latency. The booleans are explicit
// rather than inferred from a family name or a conveniently present metric.
struct CacheQualificationEvidence {
  std::string api_version;
  std::string authority_receipt_digest;
  std::string namespace_digest;
  std::string artifact_tree_digest;
  CacheWorkloadClass workload_class{};
  std::string baseline_run_digest;
  std::string candidate_run_digest;
  std::string shape_coverage_digest;
  bool transition_coverage{};
  bool baseline_instrumented{};
  bool candidate_instrumented{};
  bool output_parity{};
  bool gradient_parity{};
  bool optimizer_update_parity{};
  bool state_parity{};
  bool resumed_trajectory_parity{};
  bool determinism_parity{};
  bool content_parity{};
  bool ordering_parity{};
  bool manifest_parity{};
  bool model_quality_pass{};
  double baseline_throughput{};
  double candidate_throughput{};
  std::uint64_t baseline_peak_memory_bytes{};
  std::uint64_t candidate_peak_memory_bytes{};
  double minimum_throughput_gain_ratio{};
  double maximum_memory_regression_ratio{};

  bool operator==(const CacheQualificationEvidence&) const = default;
};

struct CacheQualificationReceipt {
  std::string api_version;
  CacheQualificationEvidence evidence;
  bool qualified{};
  std::vector<std::string> rejection_reasons;
  std::string receipt_digest;

  bool operator==(const CacheQualificationReceipt&) const = default;
};

struct CacheArtifactCandidate {
  std::string source_directory;
  std::uint64_t maximum_file_count{};
  std::uint64_t maximum_total_bytes{};
};

// Store-owned result. A production store must obtain this by descriptor-rooted,
// symlink-safe traversal, per-file hashing, fsync, and atomic content-addressed
// promotion. It is never accepted directly from an experiment document.
struct ImmutableCacheTreeReceipt {
  std::string api_version;
  std::string namespace_digest;
  std::string artifact_tree_digest;
  std::string manifest_digest;
  std::string content_address;
  std::uint64_t file_count{};
  std::uint64_t total_bytes{};
  bool immutable{};
  std::string store_receipt_digest;

  bool operator==(const ImmutableCacheTreeReceipt&) const = default;
};

class ICacheArtifactStore {
 public:
  virtual ~ICacheArtifactStore() = default;
  [[nodiscard]] virtual ImmutableCacheTreeReceipt
  publish(const CacheNamespaceAuthorityReceipt& authority,
          const CacheArtifactCandidate& candidate) = 0;
  [[nodiscard]] virtual ImmutableCacheTreeReceipt
  verify(const std::string& content_address) = 0;
};

// A narrow authority seam. Production binds this to the journal's live,
// boot-scoped lease projection; tests inject an exact fake.
class ICacheLeaseAuthority {
 public:
  virtual ~ICacheLeaseAuthority() = default;
  virtual void require_current(const ResourceLease& lease) = 0;
};

// Production resolves this from immutable baseline/candidate qualification
// artifacts and their journaled node receipts. A caller-supplied experiment
// document is never a qualification authority.
class ICacheQualificationEvidenceSource {
 public:
  virtual ~ICacheQualificationEvidenceSource() = default;
  [[nodiscard]] virtual CacheQualificationEvidence
  capture(const CacheNamespaceAuthorityReceipt& authority,
          const ImmutableCacheTreeReceipt& artifact) = 0;
  virtual void
  require_trusted(const CacheNamespaceAuthorityReceipt& authority,
                  const ImmutableCacheTreeReceipt& artifact,
                  const CacheQualificationReceipt& qualification) = 0;
};

struct CacheArtifactPublicationRequest {
  CacheNamespaceAuthorityReceipt authority;
  ResourceLease publisher_lease;
  CacheArtifactCandidate candidate;
};

struct CacheArtifactPublicationReceipt {
  std::string api_version;
  CacheNamespaceAuthorityReceipt authority;
  CacheQualificationReceipt qualification;
  ResourceLease publisher_lease;
  ImmutableCacheTreeReceipt artifact;
  std::string publication_digest;

  bool operator==(const CacheArtifactPublicationReceipt&) const = default;
};

struct CacheArtifactAdoptionRequest {
  CacheNamespaceAuthorityReceipt current_authority;
  ResourceLease adopter_lease;
  CacheArtifactPublicationReceipt publication;
};

struct CacheArtifactAdoptionGrant {
  std::string api_version;
  std::string namespace_digest;
  std::string content_address;
  std::string artifact_tree_digest;
  std::string publication_digest;
  std::string current_authority_receipt_digest;
  ResourceLease adopter_lease;
  std::string grant_digest;

  bool operator==(const CacheArtifactAdoptionGrant&) const = default;
};

class CacheArtifactAuthorityError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] CacheQualificationReceipt
qualify_cache_artifact(CacheQualificationEvidence evidence);
[[nodiscard]] nlohmann::json
cache_qualification_receipt_json(const CacheQualificationReceipt& receipt);

class CacheArtifactAuthority final {
 public:
  CacheArtifactAuthority(ICacheLeaseAuthority& leases,
                         ICacheArtifactStore& store,
                         ICacheQualificationEvidenceSource& qualification);

  [[nodiscard]] CacheArtifactPublicationReceipt
  publish(CacheArtifactPublicationRequest request) const;
  [[nodiscard]] CacheArtifactAdoptionGrant
  adopt(CacheArtifactAdoptionRequest request) const;

 private:
  ICacheLeaseAuthority& leases_;
  ICacheArtifactStore& store_;
  ICacheQualificationEvidenceSource& qualification_;
};

[[nodiscard]] nlohmann::json cache_artifact_publication_receipt_json(
    const CacheArtifactPublicationReceipt& receipt);
[[nodiscard]] nlohmann::json
cache_artifact_adoption_grant_json(const CacheArtifactAdoptionGrant& grant);

} // namespace trainvm
