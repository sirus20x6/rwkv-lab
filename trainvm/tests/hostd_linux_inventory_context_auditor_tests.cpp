#include "trainvm/hostd_linux_inventory_context_auditor.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace trainvm;

constexpr std::string_view kGpu =
    "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

HostResourceId gpu_id(HostAcceleratorVendor vendor =
                          HostAcceleratorVendor::nvidia) {
  return {.kind = HostResourceKind::accelerator,
          .vendor = vendor,
          .stable_id = std::string(kGpu),
          .parent_id = std::nullopt};
}

HostKernelSnapshot snapshot(ResourceContextDisposition compute,
                            ResourceContextDisposition graphics,
                            ProbeDisposition probe = ProbeDisposition::complete,
                            bool context_details_complete = true) {
  return {
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-context-audit",
      .boot_id = "boot-context-audit",
      .broker_epoch = "broker-context-audit",
      .begin_revision = "context-audit-revision",
      .end_revision = "context-audit-revision",
      .probes = {{.vendor = HostAcceleratorVendor::nvidia,
                  .disposition = probe,
                  .context_details_complete = context_details_complete,
                  .detail = "test-context-evidence"}},
      .resources = {{.id = gpu_id(),
                     .disposition =
                         compute == ResourceContextDisposition::present ||
                                 graphics == ResourceContextDisposition::present
                             ? ResourceObservationDisposition::occupied
                             : (probe == ProbeDisposition::complete &&
                                        context_details_complete
                                    ? ResourceObservationDisposition::
                                          audited_eligible
                                    : ResourceObservationDisposition::
                                          probe_unknown),
                     .compute_contexts = compute,
                     .graphics_contexts = graphics,
                     .pci_bdf = "0000:01:00.0",
                     .device_major = 195U,
                     .device_minor = 0U,
                     .device_nodes = {},
                     .numa_node = 0,
                     .pcie_root_id = std::nullopt,
                     .fabric_clique_id = std::nullopt,
                     .total_memory_bytes = 24ULL << 30U,
                     .labels = {{"backend", "fake"}}}},
  };
}

ResourceBundleGrant grant(HostResourceId resource = gpu_id()) {
  ResourceBundleGrant result;
  result.receipt_digest = "sha256:" + std::string(64U, 'a');
  result.host_id = "host-context-audit";
  result.boot_id = "boot-context-audit";
  result.broker_epoch = "broker-context-audit";
  result.fences.push_back({.resource = std::move(resource),
                           .generation = 1U,
                           .inventory_digest =
                               "sha256:" + std::string(64U, 'b'),
                           .topology_digest =
                               "sha256:" + std::string(64U, 'c')});
  return result;
}

HostProcessSpawnReceipt spawn() {
  HostProcessSpawnReceipt result;
  result.receipt_digest = "sha256:" + std::string(64U, 'd');
  result.host_id = "host-context-audit";
  result.broker_epoch = "broker-context-audit";
  result.request.boot_id = "boot-context-audit";
  return result;
}

LinuxProcessContextAudit audit(HostKernelSnapshot observed,
                               ResourceBundleGrant durable_grant = grant(),
                               HostProcessSpawnReceipt durable_spawn = spawn()) {
  FakeHostKernel kernel({FakeHostKernelStep{
      .snapshot = std::move(observed), .failure = std::nullopt}});
  LinuxInventoryProcessContextAuditor auditor(kernel);
  const auto result = auditor.audit(durable_grant, durable_spawn);
  require(kernel.calls() == 1U && kernel.remaining() == 0U,
          "context audit consumes exactly one inventory capture");
  return result;
}

void absent_contexts_are_complete_terminal_evidence() {
  const auto result =
      audit(snapshot(ResourceContextDisposition::absent,
                     ResourceContextDisposition::absent));
  require(result.complete && result.accelerator_contexts_empty &&
              result.evidence_digest.starts_with("sha256:") &&
              result.evidence_digest.size() == 71U,
          "fresh absent compute and graphics contexts close the exact grant");
}

void occupied_or_unknown_contexts_fail_closed() {
  const auto occupied =
      audit(snapshot(ResourceContextDisposition::present,
                     ResourceContextDisposition::absent));
  require(occupied.complete && !occupied.accelerator_contexts_empty,
          "complete occupied evidence is distinct from incomplete evidence");

  const auto unknown =
      audit(snapshot(ResourceContextDisposition::unknown,
                     ResourceContextDisposition::unknown,
                     ProbeDisposition::partial, false));
  require(!unknown.complete && !unknown.accelerator_contexts_empty &&
              unknown.evidence_digest != occupied.evidence_digest,
          "partial or unknown context evidence cannot release a grant");
}

void identity_drift_and_unsupported_vendors_fail_closed() {
  auto wrong_host = snapshot(ResourceContextDisposition::absent,
                             ResourceContextDisposition::absent);
  wrong_host.host_id = "other-host";
  const auto drifted = audit(std::move(wrong_host));
  require(!drifted.complete && !drifted.accelerator_contexts_empty,
          "host identity drift invalidates otherwise empty evidence");

  const auto unsupported =
      audit(snapshot(ResourceContextDisposition::absent,
                     ResourceContextDisposition::absent),
            grant(gpu_id(HostAcceleratorVendor::amd)));
  require(!unsupported.complete && !unsupported.accelerator_contexts_empty,
          "an unsupported accelerator backend is never assumed empty");
}

void capture_failure_and_mutex_only_work_are_distinct() {
  FakeHostKernel failing(
      {FakeHostKernelStep{.snapshot = std::nullopt, .failure = "failed"}});
  LinuxInventoryProcessContextAuditor auditor(failing);
  const auto failed = auditor.audit(grant(), spawn());
  require(!failed.complete && !failed.accelerator_contexts_empty &&
              failed.evidence_digest.starts_with("sha256:"),
          "capture failure returns sealed incomplete evidence");

  HostResourceId mutex{.kind = HostResourceKind::host_mutex,
                       .vendor = std::nullopt,
                       .stable_id = "host-mutex:test",
                       .parent_id = std::nullopt};
  const auto mutex_only =
      audit(snapshot(ResourceContextDisposition::absent,
                     ResourceContextDisposition::absent),
            grant(std::move(mutex)));
  require(mutex_only.complete && mutex_only.accelerator_contexts_empty,
          "a mutex-only grant has no accelerator context to close");
}

}  // namespace

int main() {
  try {
    absent_contexts_are_complete_terminal_evidence();
    occupied_or_unknown_contexts_fail_closed();
    identity_drift_and_unsupported_vendors_fail_closed();
    capture_failure_and_mutex_only_work_are_distinct();
    std::cout << "hostd Linux inventory context auditor tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd Linux inventory context auditor test failure: "
              << error.what() << '\n';
    return 1;
  }
}
