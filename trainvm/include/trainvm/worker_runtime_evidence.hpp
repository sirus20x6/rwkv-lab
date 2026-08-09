#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/cache_namespace_authority.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_resources.hpp"
#include "trainvm/linux_cache_evidence.hpp"

namespace trainvm {

// What a worker measured about the runtime it is actually running in, and the
// attempt identity it claims while saying so.
//
// The split matters more than the field list. Everything here is either
//
//   (a) a fact only the worker can observe -- it holds the interpreter, the
//       loaded ELF graph and the compute context, and the authority does not;
//       or
//   (b) an identity claim that the authority already knows independently, and
//       checks for equality rather than believes.
//
// Nothing here names a receipt, selects a context, or carries a digest the
// authority derives for itself: no host_id, no boot_id, no launch_spec_digest,
// no inventory_receipt_digest, no resource_binding_digest, no receipt name.
// That absence is the reason a worker cannot author a receipt that authorizes
// reuse, so it is asserted by a test rather than left to review.
struct WorkerRuntimeEvidenceReport {
  std::string api_version;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string compute_device_vendor;
  std::string compute_architecture;
  std::optional<std::string> compute_device_uuid;
  std::optional<std::string> compute_device_pci_address;
  std::string driver_version;
  std::vector<RuntimeVersionIdentity> runtime_versions;
  std::string runtime_closure_fingerprint;
  std::string host_abi_digest;
  std::string compute_compatibility_digest;

  bool operator==(const WorkerRuntimeEvidenceReport&) const = default;
};

// The authority-owned state a report is admitted against. The caller supplies
// none of it.
struct WorkerRuntimeEvidenceBinding {
  HostIdentity host;
  ResolvedLaunchSpec launch;
  HostInventoryReceipt inventory;
  bool placement_specific{};
};

// An admitted report, rebuilt as an authority-owned observation. The context
// is derived, not received; the snapshot's identity fields are copied from
// that context and the worker's corresponding claims are only ever compared.
struct AdmittedWorkerRuntimeEvidence {
  CacheRuntimeProbeContext context;
  CacheRuntimeProbeSnapshot snapshot;
};

class WorkerRuntimeEvidenceError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] WorkerRuntimeEvidenceReport worker_runtime_evidence_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json worker_runtime_evidence_json(
    const WorkerRuntimeEvidenceReport& report);

// Bind one worker report to the active attempt, fence, sealed launch and
// selected devices, and convert it into the observation the authority would
// have had to make for itself. Throws WorkerRuntimeEvidenceError on any
// disagreement; it never repairs a report to make it fit.
[[nodiscard]] AdmittedWorkerRuntimeEvidence admit_worker_runtime_evidence(
    const WorkerRuntimeEvidenceReport& report,
    const WorkerRuntimeEvidenceBinding& binding);

// Admit and then publish, through the authority's immutable publisher, in one
// step -- so no caller can reach the publisher with an unadmitted snapshot.
// Returns the immutable receipt name the sealed probe will later read.
[[nodiscard]] std::string publish_worker_runtime_evidence(
    LinuxCacheEvidencePublisher& publisher,
    const WorkerRuntimeEvidenceReport& report,
    const WorkerRuntimeEvidenceBinding& binding);

// The deployment-owned authority a worker's evidence is published through.
//
// The service holds at most a pointer to one. It supplies only what it knows
// for itself -- the host identity it is running as and the sealed launch it
// resolved for the attempt -- and this seam supplies the rest of the binding
// from deployment configuration: the immutable receipt root the publisher
// writes under and the host inventory receipt the launch was fenced against.
// Where a deployment configures no receipt root there is no authority, and the
// service refuses the message rather than accepting one it cannot publish.
class IWorkerRuntimeEvidenceAuthority {
 public:
  virtual ~IWorkerRuntimeEvidenceAuthority() = default;
  // Returns the immutable receipt name. Throws WorkerRuntimeEvidenceError (or
  // CacheNamespaceAuthorityError) when the report does not belong to this
  // launch; it never repairs a report and never publishes an unadmitted one.
  [[nodiscard]] virtual std::string publish(
      const WorkerRuntimeEvidenceReport& report, const HostIdentity& host,
      const ResolvedLaunchSpec& launch) = 0;
};

// The production authority. Constructed from deployment configuration: the
// LinuxCacheEvidencePublisher's own configuration validates the receipt root,
// so a misconfigured root fails here rather than at the first worker message.
class LinuxWorkerRuntimeEvidenceAuthority final
    : public IWorkerRuntimeEvidenceAuthority {
 public:
  // `inventory` is the host inventory receipt this authority publishes
  // against. It is a callable rather than a value because the receipt is
  // re-observed as the host changes, and a stale one would name a receipt the
  // namespace derivation would never look for.
  LinuxWorkerRuntimeEvidenceAuthority(
      LinuxCacheEvidenceConfig config,
      std::function<HostInventoryReceipt()> inventory);

  [[nodiscard]] std::string publish(const WorkerRuntimeEvidenceReport& report,
                                    const HostIdentity& host,
                                    const ResolvedLaunchSpec& launch) override;

 private:
  LinuxCacheEvidencePublisher publisher_;
  std::function<HostInventoryReceipt()> inventory_;
};

}  // namespace trainvm
