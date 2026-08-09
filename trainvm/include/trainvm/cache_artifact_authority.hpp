#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/cache_namespace_authority.hpp"
#include "trainvm/lease.hpp"
#include "trainvm/trajectory_parity.hpp"

namespace trainvm {

enum class CacheWorkloadClass {
  training,
  serving,
  preprocessing,
};

// torch's fused AdamW keeps `step` as a CUDA tensor on the parameter's device;
// the foreach reference keeps it on the host. Same key, same value, different
// device. The round trip was measured and is ACCEPTED rather than rejected,
// but only under a stated condition, because it works for a reason a naive
// state-dict comparison hides:
//
// `Optimizer.load_state_dict` normalizes `step` to the parameter's device when
// the receiving implementation is fused or capturable, and otherwise leaves it
// where the load put it. So a fused-written checkpoint resumed by a foreach
// optimizer keeps a device-resident `step`, which is numerically correct and
// costs a host synchronization on every step; the reverse direction is
// normalized for you. Both are fine; neither is bit-comparable as raw state.
//
// The gate therefore requires the round trip to be graded after that
// normalization, and requires the candidate to say which it is, so
// `state_parity` cannot be satisfied by an accidental device match nor failed
// by a difference that is not one.
enum class OptimizerStateDevicePolicy {
  // No optimizer state to round-trip. Serving and preprocessing only.
  not_applicable,
  // The round trip was compared after device normalization, and agreed.
  normalized_on_load,
  // State placement is load-bearing and is not normalized. Rejected for
  // training: it makes a resume depend on which implementation wrote it.
  device_bound,
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
  OptimizerStateDevicePolicy optimizer_state_device_policy{};
  // Not a boolean. See trainvm/trajectory_parity.hpp for why a boolean cannot
  // express the only honest answer a non-bit-identical kernel can give.
  TrajectoryParityEvidence resumed_trajectory_parity;
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
  // Present exactly when the workload has a trajectory to judge. It carries
  // the statistics the verdict was derived from, so a later reader can tell
  // why a candidate was admitted rather than only that it was.
  std::optional<TrajectoryParityAssessment> trajectory_assessment;
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
