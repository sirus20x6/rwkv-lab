#include "trainvm/worker_runtime_evidence.hpp"

#include <algorithm>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::string_view kReportApiVersion =
    "trainvm.worker-runtime-evidence/v1";
constexpr std::string_view kSnapshotApiVersion =
    "trainvm.cache-runtime-probe/v1";
constexpr std::size_t kMaximumRuntimeVersions = 64U;

[[noreturn]] void fail(std::string_view message) {
  throw WorkerRuntimeEvidenceError(std::string(message));
}

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
         std::ranges::none_of(value, [](char character) {
           return character == '\0' || character == '\n' || character == '\r';
         });
}

// Shape only. Nothing here decides whether the report belongs to this attempt;
// that is admission's job, and keeping the two apart stops a well-formed
// report from reading as an authorized one.
void validate_report_shape(const WorkerRuntimeEvidenceReport& report) {
  const bool identities =
      report.api_version == kReportApiVersion &&
      bounded_text(report.run_id, 1024U) &&
      bounded_text(report.node_id, 1024U) &&
      bounded_text(report.attempt_id, 1024U) &&
      bounded_text(report.launch_nonce, 1024U) &&
      bounded_text(report.concurrency_key, 1024U) &&
      bounded_text(report.lease_id, 1024U) && report.fencing_token != 0U;
  const bool measurements =
      bounded_text(report.compute_device_vendor, 64U) &&
      bounded_text(report.compute_architecture, 256U) &&
      bounded_text(report.driver_version, 256U) &&
      valid_sha256(report.runtime_closure_fingerprint) &&
      valid_sha256(report.host_abi_digest) &&
      valid_sha256(report.compute_compatibility_digest) &&
      (!report.compute_device_uuid ||
       bounded_text(*report.compute_device_uuid, 256U)) &&
      (!report.compute_device_pci_address ||
       bounded_text(*report.compute_device_pci_address, 256U));
  // Canonical ordering is part of the shape: the namespace digest closes over
  // these entries, so two workers reporting the same runtime in a different
  // order must not reach two namespaces.
  const bool versions =
      report.runtime_versions.size() <= kMaximumRuntimeVersions &&
      std::ranges::all_of(report.runtime_versions,
                          [](const RuntimeVersionIdentity& version) {
                            return bounded_text(version.name, 128U) &&
                                   bounded_text(version.version, 128U);
                          }) &&
      std::ranges::is_sorted(report.runtime_versions,
                             [](const RuntimeVersionIdentity& left,
                                const RuntimeVersionIdentity& right) {
                               return left.name < right.name;
                             }) &&
      std::ranges::adjacent_find(report.runtime_versions,
                                 [](const RuntimeVersionIdentity& left,
                                    const RuntimeVersionIdentity& right) {
                                   return left.name == right.name;
                                 }) == report.runtime_versions.end();
  if (!identities || !measurements || !versions) {
    fail("worker runtime evidence report is malformed or noncanonical");
  }
}

}  // namespace

WorkerRuntimeEvidenceReport worker_runtime_evidence_from_json(
    const nlohmann::json& source) {
  WorkerRuntimeEvidenceReport report{};
  std::vector<Diagnostic> diagnostics;
  // Exact schema: an unknown member, a missing member, or a member the encoder
  // would have written differently all fail here. A transport that silently
  // ignored an extra key would be a transport a worker could extend.
  if (!source.is_object() ||
      !decode_json(source, report, "", diagnostics) || !diagnostics.empty() ||
      encode_json(report) != source) {
    fail("worker runtime evidence report has an invalid reflected schema");
  }
  validate_report_shape(report);
  return report;
}

nlohmann::json worker_runtime_evidence_json(
    const WorkerRuntimeEvidenceReport& report) {
  validate_report_shape(report);
  return encode_json(report);
}

AdmittedWorkerRuntimeEvidence admit_worker_runtime_evidence(
    const WorkerRuntimeEvidenceReport& report,
    const WorkerRuntimeEvidenceBinding& binding) {
  validate_report_shape(report);
  const ResolvedLaunchIdentity& identity = binding.launch.identity;
  // The attempt and its fence. A report from a superseded attempt, a released
  // lease, or a relaunch of the same attempt id is refused here rather than
  // deposited where the current attempt's derivation would read it.
  if (report.run_id != identity.run_id || report.node_id != identity.node_id ||
      report.attempt_id != identity.attempt_id ||
      report.launch_nonce != identity.launch_nonce ||
      report.concurrency_key != identity.concurrency_key ||
      report.lease_id != identity.lease_id ||
      report.fencing_token != identity.fencing_token) {
    fail("worker runtime evidence is not bound to the active attempt fence");
  }
  if (identity.host != binding.host ||
      binding.inventory.host_id != binding.host.host_id ||
      binding.inventory.boot_id != binding.host.boot_id) {
    fail("worker runtime evidence binding host authority disagrees");
  }
  // The sealed launch decided which runtime this worker was authorized to be.
  // A worker reporting a different closure is not lying about a detail: it is
  // not the launch that was authorized, whether it drifted or chose to say so.
  if (report.runtime_closure_fingerprint !=
      identity.bootstrap_runtime_closure_fingerprint) {
    fail("worker runtime closure differs from the sealed launch authority");
  }

  CacheRuntimeProbeContext context = cache_runtime_probe_context(
      binding.host, binding.launch, binding.inventory,
      binding.placement_specific);
  // Identity comes from the context, measurements come from the report. This
  // assignment, not a comparison, is what makes a caller-authored receipt
  // impossible: the fields that name the receipt are never read off the wire.
  CacheRuntimeProbeSnapshot snapshot{
      .api_version = std::string(kSnapshotApiVersion),
      .host_id = context.host.host_id,
      .boot_id = context.host.boot_id,
      .launch_spec_digest = context.launch_spec_digest,
      .inventory_receipt_digest = context.inventory_receipt_digest,
      .resource_binding_digest = context.resource_binding_digest,
      .compute_device_vendor = report.compute_device_vendor,
      .compute_architecture = report.compute_architecture,
      .compute_device_uuid = report.compute_device_uuid,
      .compute_device_pci_address = report.compute_device_pci_address,
      .driver_version = report.driver_version,
      .runtime_versions = report.runtime_versions,
      .runtime_closure_fingerprint = report.runtime_closure_fingerprint,
      .host_abi_digest = report.host_abi_digest,
      .compute_compatibility_digest = report.compute_compatibility_digest,
  };
  // Binds the measured device identity to the devices this launch was fenced
  // to: vendor against the selection, and under placement specificity the
  // exact UUID and PCI address of the one device it holds.
  try {
    validate_cache_runtime_probe_snapshot(snapshot, context);
  } catch (const CacheNamespaceAuthorityError& error) {
    fail(std::string("worker runtime evidence rejected by authority: ") +
         error.what());
  }
  return {.context = std::move(context), .snapshot = std::move(snapshot)};
}

std::string publish_worker_runtime_evidence(
    LinuxCacheEvidencePublisher& publisher,
    const WorkerRuntimeEvidenceReport& report,
    const WorkerRuntimeEvidenceBinding& binding) {
  const AdmittedWorkerRuntimeEvidence admitted =
      admit_worker_runtime_evidence(report, binding);
  return publisher.publish_runtime(admitted.context, admitted.snapshot);
}

LinuxWorkerRuntimeEvidenceAuthority::LinuxWorkerRuntimeEvidenceAuthority(
    LinuxCacheEvidenceConfig config, InventoryLookup inventory)
    : publisher_(std::move(config)), inventory_(std::move(inventory)) {
  if (!inventory_) {
    fail("worker runtime evidence authority requires an inventory source");
  }
}

std::string LinuxWorkerRuntimeEvidenceAuthority::publish(
    const WorkerRuntimeEvidenceReport& report, const HostIdentity& host,
    const ResolvedLaunchSpec& launch) {
  const std::optional<GrantInventoryProjection> granted = inventory_(launch);
  if (!granted) {
    // Not `fail`: this one is a deployment gap rather than a worker fault, and
    // the caller has to answer it with a different status code. See the type's
    // declaration.
    throw WorkerRuntimeEvidenceUnavailableError(
        "authority holds no grant-time host inventory projection for this "
        "launch to publish worker runtime evidence against");
  }
  const GrantInventoryProjection& inventory = *granted;
  // Placement specificity is derived, never configured and never claimed. A
  // launch fenced to exactly one device is the only shape whose measured UUID
  // and PCI address are admissible at all -- the shared validator refuses a
  // placement-specific probe against any other count -- and a launch holding
  // no device fence has no placement to be specific about. Deriving it here
  // keeps one answer to the question; a second one would name a receipt the
  // namespace derivation never looks for.
  const CacheResourceBinding devices = cache_resource_binding(launch, inventory);
  const WorkerRuntimeEvidenceBinding binding{
      .host = host,
      .launch = launch,
      .inventory = inventory,
      .placement_specific = devices.devices.size() == 1U,
  };
  return publish_worker_runtime_evidence(publisher_, report, binding);
}

}  // namespace trainvm
