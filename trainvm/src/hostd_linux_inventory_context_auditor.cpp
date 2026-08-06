#include "trainvm/hostd_linux_inventory_context_auditor.hpp"

#include <algorithm>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

std::string disposition_name(ResourceContextDisposition disposition) {
  switch (disposition) {
    case ResourceContextDisposition::absent:
      return "absent";
    case ResourceContextDisposition::present:
      return "present";
    case ResourceContextDisposition::unknown:
      return "unknown";
  }
  return "invalid";
}

std::string evidence_digest(const ResourceBundleGrant& grant,
                            const HostProcessSpawnReceipt& spawn,
                            std::string_view inventory_digest,
                            bool complete, bool contexts_empty,
                            const nlohmann::json& observations) {
  const nlohmann::json evidence{
      {"api_version", kLinuxInventoryProcessContextAuditApiVersion},
      {"grant_digest", grant.receipt_digest},
      {"spawn_receipt_digest", spawn.receipt_digest},
      {"inventory_digest", inventory_digest},
      {"complete", complete},
      {"accelerator_contexts_empty", contexts_empty},
      {"observations", observations},
  };
  return "sha256:" + sha256_hex(evidence.dump());
}

LinuxProcessContextAudit failed_capture(const ResourceBundleGrant& grant,
                                        const HostProcessSpawnReceipt& spawn) {
  const nlohmann::json observations = nlohmann::json::array(
      {{{"status", "capture_failed"}}});
  return {.complete = false,
          .accelerator_contexts_empty = false,
          .evidence_digest = evidence_digest(grant, spawn, {}, false, false,
                                             observations)};
}

}  // namespace

LinuxInventoryProcessContextAuditor::LinuxInventoryProcessContextAuditor(
    IHostKernel& kernel)
    : kernel_(kernel) {}

LinuxProcessContextAudit LinuxInventoryProcessContextAuditor::audit(
    const ResourceBundleGrant& grant,
    const HostProcessSpawnReceipt& spawn) {
  HostInventoryReceipt inventory;
  try {
    inventory = capture_host_inventory(kernel_);
  } catch (...) {
    return failed_capture(grant, spawn);
  }

  const auto nvidia_probe = std::ranges::find_if(
      inventory.probes, [](const HostProbeResult& probe) {
        return probe.vendor == HostAcceleratorVendor::nvidia;
      });
  bool complete = inventory.host_id == grant.host_id &&
                  inventory.boot_id == grant.boot_id &&
                  inventory.broker_epoch == grant.broker_epoch &&
                  spawn.host_id == grant.host_id &&
                  spawn.request.boot_id == grant.boot_id &&
                  spawn.broker_epoch == grant.broker_epoch;
  bool contexts_empty = true;
  nlohmann::json observations = nlohmann::json::array();
  for (const ResourceFence& fence : grant.fences) {
    if (fence.resource.kind == HostResourceKind::host_mutex) continue;
    nlohmann::json observation{
        {"resource", canonical_resource_key(fence.resource)},
    };
    if (!fence.resource.vendor ||
        *fence.resource.vendor != HostAcceleratorVendor::nvidia) {
      observation["status"] = "unsupported_vendor";
      complete = false;
      contexts_empty = false;
      observations.push_back(std::move(observation));
      continue;
    }
    if (nvidia_probe == inventory.probes.end() ||
        nvidia_probe->disposition != ProbeDisposition::complete ||
        !nvidia_probe->context_details_complete) {
      observation["status"] = "probe_incomplete";
      complete = false;
      contexts_empty = false;
      observations.push_back(std::move(observation));
      continue;
    }
    const auto found = std::ranges::find_if(
        inventory.resources, [&fence](const ObservedHostResource& resource) {
          return resource.id == fence.resource;
        });
    if (found == inventory.resources.end()) {
      observation["status"] = "missing";
      complete = false;
      contexts_empty = false;
      observations.push_back(std::move(observation));
      continue;
    }
    observation["status"] = "observed";
    observation["compute"] = disposition_name(found->compute_contexts);
    observation["graphics"] = disposition_name(found->graphics_contexts);
    if (found->compute_contexts == ResourceContextDisposition::unknown ||
        found->graphics_contexts == ResourceContextDisposition::unknown) {
      complete = false;
      contexts_empty = false;
    } else if (found->compute_contexts == ResourceContextDisposition::present ||
               found->graphics_contexts ==
                   ResourceContextDisposition::present) {
      contexts_empty = false;
    }
    observations.push_back(std::move(observation));
  }
  if (!complete) contexts_empty = false;
  return {.complete = complete,
          .accelerator_contexts_empty = contexts_empty,
          .evidence_digest = evidence_digest(
              grant, spawn, inventory.inventory_digest, complete,
              contexts_empty, observations)};
}

}  // namespace trainvm
