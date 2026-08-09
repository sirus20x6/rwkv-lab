#include "trainvm/host_resources.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

constexpr std::string_view kGpuA = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
constexpr std::string_view kGpuB = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";
constexpr std::string_view kGpuC = "GPU-cccccccc-cccc-cccc-cccc-cccccccccccc";
constexpr std::string_view kGpuP = "GPU-dddddddd-dddd-dddd-dddd-dddddddddddd";
constexpr std::string_view kGpuQ = "GPU-eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee";
constexpr std::string_view kMig0 = "MIG-00000000-0000-0000-0000-000000000000";
constexpr std::string_view kMig1 = "MIG-11111111-1111-1111-1111-111111111111";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void require_rejected(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const HostResourceError&) {
    return;
  }
  throw std::runtime_error(message);
}

HostResourceId gpu_id(std::string_view stable_id) {
  HostResourceId id{};
  id.kind = HostResourceKind::accelerator;
  id.vendor = HostAcceleratorVendor::nvidia;
  id.stable_id = stable_id;
  return id;
}

HostResourceId mig_id(std::string_view stable_id,
                      std::string_view parent_id) {
  HostResourceId id{};
  id.kind = HostResourceKind::accelerator_partition;
  id.vendor = HostAcceleratorVendor::nvidia;
  id.stable_id = stable_id;
  id.parent_id = parent_id;
  return id;
}

ObservedHostResource accelerator(HostResourceId id, std::int32_t numa,
                                 std::string pcie_root,
                                 std::string fabric_clique,
                                 std::uint64_t memory = 80ULL << 30U) {
  ObservedHostResource resource{};
  resource.id = std::move(id);
  resource.disposition = ResourceObservationDisposition::audited_eligible;
  resource.compute_contexts = ResourceContextDisposition::absent;
  resource.graphics_contexts = ResourceContextDisposition::absent;
  if (resource.id.kind == HostResourceKind::accelerator_partition) {
    resource.pci_bdf = "0000:04:00.0";
  } else {
    const std::uint32_t ordinal = resource.id.stable_id == kGpuA ? 1U :
                                  resource.id.stable_id == kGpuB ? 2U :
                                  resource.id.stable_id == kGpuC ? 3U :
                                  resource.id.stable_id == kGpuP ? 4U : 5U;
    resource.pci_bdf = "0000:0" + std::to_string(ordinal) + ":00.0";
    resource.device_major = 195U;
    resource.device_minor = ordinal;
  }
  resource.numa_node = numa;
  resource.pcie_root_id = std::move(pcie_root);
  resource.fabric_clique_id = std::move(fabric_clique);
  resource.total_memory_bytes = memory;
  resource.labels = {{"class", "training"}, {"case", "ExactValue"}};
  return resource;
}

HostKernelSnapshot base_snapshot(std::string revision = "revision-1") {
  HostKernelSnapshot snapshot{};
  snapshot.api_version = std::string(kHostInventoryApiVersion);
  snapshot.host_id = "host-001";
  snapshot.boot_id = "boot-001";
  snapshot.broker_epoch = "broker-001";
  snapshot.begin_revision = revision;
  snapshot.end_revision = revision;
  HostProbeResult probe{};
  probe.vendor = HostAcceleratorVendor::nvidia;
  probe.disposition = ProbeDisposition::complete;
  probe.context_details_complete = true;
  probe.detail = "fake-complete";
  snapshot.probes.push_back(probe);
  snapshot.resources = {
      accelerator(gpu_id(kGpuC), 1, "PCIE-1", "FABRIC-0"),
      accelerator(gpu_id(kGpuA), 0, "PCIE-0", "FABRIC-0"),
      accelerator(gpu_id(kGpuP), 2, "PCIE-2", "FABRIC-0"),
      accelerator(gpu_id(kGpuB), 0, "PCIE-0", "FABRIC-0"),
      accelerator(mig_id(kMig1, kGpuP), 2, "PCIE-2", "FABRIC-0",
                  20ULL << 30U),
      accelerator(mig_id(kMig0, kGpuP), 2, "PCIE-2", "FABRIC-0",
                  20ULL << 30U),
  };
  ObservedHostResource mutex{};
  mutex.id.kind = HostResourceKind::host_mutex;
  mutex.id.stable_id = "host-mutex:dataset-cache";
  mutex.disposition = ResourceObservationDisposition::audited_eligible;
  mutex.compute_contexts = ResourceContextDisposition::absent;
  mutex.graphics_contexts = ResourceContextDisposition::absent;
  mutex.labels = {{"scope", "dataset"}};
  snapshot.resources.push_back(std::move(mutex));
  return snapshot;
}

HostInventoryReceipt capture(HostKernelSnapshot snapshot) {
  FakeHostKernel kernel({FakeHostKernelStep{.snapshot = std::move(snapshot),
                                            .failure = std::nullopt}});
  auto receipt = capture_host_inventory(kernel);
  require(kernel.calls() == 1U && kernel.remaining() == 0U,
          "fake kernel consumes exactly one immutable snapshot");
  return receipt;
}

ResourceBundleRequest request(ResourceAccessMode mode, TopologyPolicy topology,
                              std::uint32_t count) {
  ResourceBundleRequest value{};
  value.api_version = std::string(kHostResourceRequestApiVersion);
  value.request_id = "request-001";
  value.journal_id = "journal-001";
  value.run_id = "run-001";
  value.logical_lease_id = "lease-001";
  value.logical_fencing_token = 7U;
  value.count = count;
  value.access_mode = mode;
  value.topology = topology;
  return value;
}

ResourceOccupancySnapshot occupancy(
    const HostInventoryReceipt& inventory,
    std::vector<ResourceFence> active_fences = {}) {
  ResourceOccupancySnapshot value{};
  value.api_version = std::string(kHostResourceOccupancyApiVersion);
  value.host_id = inventory.host_id;
  value.boot_id = inventory.boot_id;
  value.inventory_digest = inventory.inventory_digest;
  value.ledger_sequence = 11U;
  value.active_fences = std::move(active_fences);
  return seal_resource_occupancy(inventory, std::move(value));
}

ResourceFence fence(const HostInventoryReceipt& inventory,
                    HostResourceId resource, std::uint64_t generation = 1U) {
  ResourceFence value{};
  value.resource = std::move(resource);
  value.generation = generation;
  value.inventory_digest = inventory.inventory_digest;
  value.topology_digest = inventory.topology_digest;
  return value;
}

std::vector<std::string> selected_ids(
    const std::optional<ResourceBundleSelection>& selection) {
  require(selection.has_value(), "expected a resource selection");
  std::vector<std::string> ids;
  for (const auto& resource : selection->resources) {
    ids.push_back(resource.id.stable_id);
  }
  return ids;
}

}  // namespace

int main() {
  try {
    const auto inventory = capture(base_snapshot());
    const auto free = occupancy(inventory);

    const auto mutex_request = seal_resource_request(request(
        ResourceAccessMode::mutex_exclusive, TopologyPolicy::any, 1U));
    require(selected_ids(select_host_resources(inventory, mutex_request, free)) ==
                std::vector<std::string>({"host-mutex:dataset-cache"}),
            "host mutexes have stable IDs and explicit exclusive selection");
    auto invalid_mutex = request(ResourceAccessMode::mutex_exclusive,
                                 TopologyPolicy::any, 1U);
    invalid_mutex.selector.vendor = HostAcceleratorVendor::nvidia;
    require_rejected([&] { (void)seal_resource_request(invalid_mutex); },
                     "host mutex selection rejects accelerator filters");

    auto permuted_snapshot = base_snapshot();
    std::ranges::reverse(permuted_snapshot.resources);
    const auto permuted = capture(std::move(permuted_snapshot));
    require(inventory.receipt_digest == permuted.receipt_digest &&
                inventory.topology_digest == permuted.topology_digest,
            "inventory order cannot change canonical receipt identity");

    for (const auto& [policy, count, expected] :
         std::vector<std::tuple<TopologyPolicy, std::uint32_t,
                                std::vector<std::string>>>{
             {TopologyPolicy::any, 2U, {std::string(kGpuA), std::string(kGpuB)}},
             {TopologyPolicy::same_numa_node, 2U, {std::string(kGpuA), std::string(kGpuB)}},
             {TopologyPolicy::same_pcie_root, 2U, {std::string(kGpuA), std::string(kGpuB)}},
             {TopologyPolicy::same_fabric_clique,
              3U,
              {std::string(kGpuA), std::string(kGpuB), std::string(kGpuC)}},
         }) {
      const auto sealed =
          seal_resource_request(request(ResourceAccessMode::exclusive_device,
                                        policy, count));
      const auto first = select_host_resources(inventory, sealed, free);
      const auto second = select_host_resources(permuted, sealed,
                                                occupancy(permuted));
      require(selected_ids(first) == expected && selected_ids(second) == expected,
              "topology selection is deterministic under inventory permutation");
      require(first->selection_digest == second->selection_digest,
              "semantic inventory permutations preserve selection digest");
    }

    auto opposed_groups = base_snapshot("revision-opposed-groups");
    for (auto& resource : opposed_groups.resources) {
      if (resource.id.stable_id == kGpuA || resource.id.stable_id == kGpuB) {
        resource.numa_node = 9;
      } else if (resource.id.kind != HostResourceKind::host_mutex) {
        resource.numa_node = 0;
      }
    }
    const auto opposed_inventory = capture(std::move(opposed_groups));
    const auto same_numa = seal_resource_request(request(
        ResourceAccessMode::exclusive_device,
        TopologyPolicy::same_numa_node, 2U));
    require(selected_ids(select_host_resources(
                opposed_inventory, same_numa, occupancy(opposed_inventory))) ==
                std::vector<std::string>({std::string(kGpuA), std::string(kGpuB)}),
            "bundle stable-ID order wins over topology group-key order");

    for (const auto policy : {TopologyPolicy::same_numa_node,
                              TopologyPolicy::same_pcie_root,
                              TopologyPolicy::same_fabric_clique}) {
      auto missing_topology = base_snapshot("revision-missing-topology");
      for (auto& resource : missing_topology.resources) {
        if (resource.id.kind == HostResourceKind::host_mutex) continue;
        if (policy == TopologyPolicy::same_numa_node) resource.numa_node.reset();
        if (policy == TopologyPolicy::same_pcie_root) resource.pcie_root_id.reset();
        if (policy == TopologyPolicy::same_fabric_clique) {
          resource.fabric_clique_id.reset();
        }
      }
      const auto missing_inventory = capture(std::move(missing_topology));
      const auto constrained = seal_resource_request(request(
          ResourceAccessMode::exclusive_device, policy, 1U));
      require(!select_host_resources(missing_inventory, constrained,
                                     occupancy(missing_inventory)),
              "missing constrained topology evidence is ineligible");
    }

    auto exact = request(ResourceAccessMode::exclusive_device,
                         TopologyPolicy::exact_resources, 2U);
    exact.selector.vendor = HostAcceleratorVendor::nvidia;
    exact.selector.exact_labels = {{"case", "ExactValue"}};
    exact.selector.exact_resources = {gpu_id(kGpuC), gpu_id(kGpuA)};
    exact = seal_resource_request(std::move(exact));
    require(selected_ids(select_host_resources(inventory, exact, free)) ==
                std::vector<std::string>({std::string(kGpuA), std::string(kGpuC)}),
            "exact selection canonicalizes IDs without substitution");
    auto impossible_exact = exact;
    impossible_exact.selector.minimum_memory_bytes = 81ULL << 30U;
    impossible_exact = seal_resource_request(std::move(impossible_exact));
    require(!select_host_resources(inventory, impossible_exact, free),
            "exact resources must still satisfy every selector filter");
    auto exact_count_mismatch = request(ResourceAccessMode::exclusive_device,
                                        TopologyPolicy::exact_resources, 2U);
    exact_count_mismatch.selector.exact_resources = {gpu_id(kGpuA)};
    require_rejected(
        [&] { (void)seal_resource_request(exact_count_mismatch); },
        "exact resource list length must equal bundle count");
    auto exact_on_any = request(ResourceAccessMode::exclusive_device,
                                TopologyPolicy::any, 1U);
    exact_on_any.selector.exact_resources = {gpu_id(kGpuA)};
    require_rejected([&] { (void)seal_resource_request(exact_on_any); },
                     "exact IDs cannot accompany a non-exact policy");
    auto zero_memory = request(ResourceAccessMode::exclusive_device,
                               TopologyPolicy::any, 1U);
    zero_memory.selector.minimum_memory_bytes = 0U;
    require_rejected([&] { (void)seal_resource_request(zero_memory); },
                     "zero minimum memory has no duplicate encoding");

    const auto siblings = seal_resource_request(request(
        ResourceAccessMode::partition_exclusive, TopologyPolicy::any, 2U));
    require(selected_ids(select_host_resources(inventory, siblings, free)) ==
                std::vector<std::string>({std::string(kMig0), std::string(kMig1)}),
            "nonoverlapping sibling partitions may coexist");
    const auto child_busy = occupancy(
        inventory, {fence(inventory, mig_id(kMig0, kGpuP))});
    const auto one_partition = seal_resource_request(request(
        ResourceAccessMode::partition_exclusive, TopologyPolicy::any, 1U));
    require(selected_ids(select_host_resources(inventory, one_partition,
                                               child_busy)) ==
                std::vector<std::string>({std::string(kMig1)}),
            "active child blocks itself and its parent but not its sibling");
    const auto parent_busy =
        occupancy(inventory, {fence(inventory, gpu_id(kGpuP))});
    require(!select_host_resources(inventory, one_partition, parent_busy),
            "active full device blocks all child partitions");
    auto exact_parent = request(ResourceAccessMode::exclusive_device,
                                TopologyPolicy::exact_resources, 1U);
    exact_parent.selector.exact_resources = {gpu_id(kGpuP)};
    exact_parent = seal_resource_request(std::move(exact_parent));
    require(!select_host_resources(inventory, exact_parent, child_busy),
            "active partition blocks an exact parent request");

    auto child_observed_busy = base_snapshot("revision-child-observed");
    for (auto& resource : child_observed_busy.resources) {
      if (resource.id.stable_id == kMig0) {
        resource.disposition = ResourceObservationDisposition::occupied;
        resource.compute_contexts = ResourceContextDisposition::present;
      }
    }
    const auto child_observed_inventory =
        capture(std::move(child_observed_busy));
    require(!select_host_resources(child_observed_inventory, exact_parent,
                                   occupancy(child_observed_inventory)),
            "occupied child observation blocks its full parent");
    require(selected_ids(select_host_resources(
                child_observed_inventory, one_partition,
                occupancy(child_observed_inventory))) ==
                std::vector<std::string>({std::string(kMig1)}),
            "occupied child does not block a disjoint sibling partition");

    auto parent_observed_busy = base_snapshot("revision-parent-observed");
    for (auto& resource : parent_observed_busy.resources) {
      if (resource.id.stable_id == kGpuP) {
        resource.disposition = ResourceObservationDisposition::reserved;
      }
    }
    const auto parent_observed_inventory =
        capture(std::move(parent_observed_busy));
    require(!select_host_resources(parent_observed_inventory, one_partition,
                                   occupancy(parent_observed_inventory)),
            "reserved parent observation blocks every child partition");

    auto child_unknown = base_snapshot("revision-child-unknown");
    for (auto& resource : child_unknown.resources) {
      if (resource.id.stable_id == kMig0) {
        resource.disposition = ResourceObservationDisposition::probe_unknown;
        resource.compute_contexts = ResourceContextDisposition::unknown;
      }
    }
    const auto child_unknown_inventory = capture(std::move(child_unknown));
    require(!select_host_resources(child_unknown_inventory, exact_parent,
                                   occupancy(child_unknown_inventory)),
            "unknown child context blocks its full parent");

    auto graphics_snapshot = base_snapshot("revision-graphics");
    for (auto& resource : graphics_snapshot.resources) {
      if (resource.id.stable_id == kGpuA) {
        resource.graphics_contexts = ResourceContextDisposition::present;
      }
    }
    const auto graphics_inventory = capture(std::move(graphics_snapshot));
    auto graphics_exact = request(ResourceAccessMode::exclusive_compute,
                                  TopologyPolicy::exact_resources, 1U);
    graphics_exact.selector.exact_resources = {gpu_id(kGpuA)};
    graphics_exact = seal_resource_request(std::move(graphics_exact));
    require(select_host_resources(graphics_inventory, graphics_exact,
                                  occupancy(graphics_inventory)).has_value(),
            "exclusive compute may coexist with classified host graphics");
    graphics_exact.access_mode = ResourceAccessMode::exclusive_device;
    graphics_exact = seal_resource_request(std::move(graphics_exact));
    require(!select_host_resources(graphics_inventory, graphics_exact,
                                   occupancy(graphics_inventory)),
            "exclusive device rejects an observed graphics context");

    auto workstation_snapshot = base_snapshot("revision-workstation");
    for (auto& resource : workstation_snapshot.resources) {
      if (resource.id.stable_id == kGpuA) {
        resource.disposition = ResourceObservationDisposition::occupied;
        resource.compute_contexts = ResourceContextDisposition::present;
        resource.graphics_contexts = ResourceContextDisposition::present;
        resource.labels["display"] = "active";
      }
    }
    auto authorized_workstation_snapshot = workstation_snapshot;
    const auto workstation_inventory = capture(std::move(workstation_snapshot));
    auto cooperative = request(ResourceAccessMode::cooperative_compute,
                               TopologyPolicy::exact_resources, 1U);
    cooperative.selector.exact_resources = {gpu_id(kGpuA)};
    cooperative = seal_resource_request(std::move(cooperative));
    require(!select_host_resources(workstation_inventory, cooperative,
                                   occupancy(workstation_inventory)),
            "display occupancy is denied without explicit operator authority");
    for (auto& resource : authorized_workstation_snapshot.resources) {
      if (resource.id.stable_id == kGpuA) {
        resource.labels["display-sharing"] =
            "operator-authorized-cooperative";
      }
    }
    const auto authorized_workstation_inventory =
        capture(std::move(authorized_workstation_snapshot));
    require(select_host_resources(authorized_workstation_inventory, cooperative,
                                  occupancy(authorized_workstation_inventory))
                .has_value(),
            "cooperative display sharing requires the exact operator label");

    for (const auto disposition : {ProbeDisposition::unavailable,
                                   ProbeDisposition::partial,
                                   ProbeDisposition::denied,
                                   ProbeDisposition::timeout}) {
      auto unknown_snapshot = base_snapshot("revision-unknown");
      unknown_snapshot.probes.front().disposition = disposition;
      unknown_snapshot.probes.front().context_details_complete = false;
      for (auto& resource : unknown_snapshot.resources) {
        if (resource.id.kind == HostResourceKind::host_mutex) continue;
        resource.disposition = ResourceObservationDisposition::probe_unknown;
        resource.compute_contexts = ResourceContextDisposition::unknown;
        resource.graphics_contexts = ResourceContextDisposition::unknown;
      }
      const auto unknown_inventory = capture(std::move(unknown_snapshot));
      const auto any = seal_resource_request(request(
          ResourceAccessMode::exclusive_device, TopologyPolicy::any, 1U));
      require(!select_host_resources(unknown_inventory, any,
                                     occupancy(unknown_inventory)),
              "unavailable/partial/denied/timeout probe is never eligible");
    }
    auto dishonest_partial = base_snapshot("revision-dishonest");
    dishonest_partial.probes.front().disposition = ProbeDisposition::partial;
    dishonest_partial.probes.front().context_details_complete = false;
    require_rejected([&] { (void)capture(dishonest_partial); },
                     "partial probe cannot retain eligible resources");

    auto complete_empty = base_snapshot("revision-empty");
    complete_empty.resources.clear();
    const auto empty_inventory = capture(std::move(complete_empty));
    const auto any_one = seal_resource_request(request(
        ResourceAccessMode::exclusive_device, TopologyPolicy::any, 1U));
    require(!select_host_resources(empty_inventory, any_one,
                                   occupancy(empty_inventory)),
            "successful empty inventory is known empty, not fabricated free");

    require(host_inventory_from_json(host_inventory_json(inventory)) == inventory &&
                resource_request_from_json(resource_request_json(exact)) == exact &&
                resource_occupancy_from_json(
                    inventory, resource_occupancy_json(free)) == free,
            "strict reflected inventory/request/occupancy codecs round trip");
    const auto selection =
        *select_host_resources(inventory, exact, free);
    require(
        inventory.receipt_digest ==
            "sha256:26e3f78b79eef7c4411261449fbb11bd4d046c10abd7dd3e92bf1e567bfa9f70" &&
        exact.canonical_request_digest ==
            "sha256:846e1019a1034ec1447a55048e54f3dc69a007afbc7b5b448e1dd5bd691a3e7e" &&
        free.occupancy_digest ==
            "sha256:9a55192945c5c60d49c28f33ab6a29c60c17f9a2a3edff59660ea6c7063fbfe8" &&
        selection.selection_digest ==
            "sha256:3354094910119e43fbd85817cd01e47a565abea5b574feb7574fcb3c5f4ff939",
        "versioned inventory/request/occupancy/selection digests are pinned");
    require(resource_selection_from_json(resource_selection_json(selection)) ==
                selection,
            "strict reflected selection codec round trips");
    auto unknown_json = resource_occupancy_json(free);
    unknown_json["legacy_free_guess"] = true;
    require_rejected(
        [&] { (void)resource_occupancy_from_json(inventory, unknown_json); },
        "unknown occupancy fields cannot become scheduling authority");

    auto forged_occupancy = free;
    forged_occupancy.host_id = "host-forged";
    require_rejected([&] { (void)resource_occupancy_json(forged_occupancy); },
                     "occupancy host mutation invalidates its digest");
    forged_occupancy = free;
    forged_occupancy.boot_id = "boot-forged";
    require_rejected([&] { (void)resource_occupancy_json(forged_occupancy); },
                     "occupancy boot mutation invalidates its digest");
    forged_occupancy = free;
    forged_occupancy.ledger_sequence += 1U;
    require_rejected([&] { (void)resource_occupancy_json(forged_occupancy); },
                     "occupancy sequence mutation invalidates its digest");
    const auto two_fences = occupancy(
        inventory,
        {fence(inventory, mig_id(kMig1, kGpuP)),
         fence(inventory, mig_id(kMig0, kGpuP))});
    forged_occupancy = two_fences;
    std::ranges::reverse(forged_occupancy.active_fences);
    require_rejected([&] { (void)resource_occupancy_json(forged_occupancy); },
                     "occupancy fence reordering cannot serialize stale authority");

    auto label_only = base_snapshot("revision-label");
    for (auto& resource : label_only.resources) {
      resource.labels["telemetry"] = "changed";
    }
    const auto label_inventory = capture(std::move(label_only));
    require(detect_bundle_degradation(selection, label_inventory).health ==
                BundleHealth::intact,
            "non-topology label drift does not degrade stable resources");
    auto vanished = base_snapshot("revision-vanished");
    std::erase_if(vanished.resources, [](const auto& resource) {
      return resource.id.stable_id == kGpuA;
    });
    require(detect_bundle_degradation(selection, capture(std::move(vanished)))
                .health == BundleHealth::degraded,
            "vanished selected UUID degrades the allocation");
    auto topology_changed = base_snapshot("revision-topology");
    topology_changed.resources.front().total_memory_bytes += 1U;
    require(detect_bundle_degradation(
                selection, capture(std::move(topology_changed)))
                .topology_changed,
            "capacity/topology digest change degrades the allocation");

    auto bdf_drift = base_snapshot("revision-bdf-drift");
    for (auto& resource : bdf_drift.resources) {
      if (resource.id.stable_id == kGpuA) {
        resource.pci_bdf = "0000:06:00.0";
      }
    }
    require(detect_bundle_degradation(selection, capture(std::move(bdf_drift)))
                .health == BundleHealth::degraded,
            "selected UUID PCI BDF drift degrades the allocation");
    auto device_node_drift = base_snapshot("revision-device-node-drift");
    for (auto& resource : device_node_drift.resources) {
      if (resource.id.stable_id == kGpuA) {
        resource.device_minor = 6U;
      }
    }
    require(detect_bundle_degradation(
                selection, capture(std::move(device_node_drift)))
                .health == BundleHealth::degraded,
            "selected UUID device-node drift degrades the allocation");

    auto exact_child_request = request(
        ResourceAccessMode::partition_exclusive,
        TopologyPolicy::exact_resources, 1U);
    exact_child_request.selector.exact_resources = {
        mig_id(kMig0, kGpuP)};
    exact_child_request = seal_resource_request(std::move(exact_child_request));
    const auto exact_child_selection = *select_host_resources(
        inventory, exact_child_request, free);
    auto changed_parent = base_snapshot("revision-changed-parent");
    changed_parent.resources.push_back(
        accelerator(gpu_id(kGpuQ), 2, "PCIE-2", "FABRIC-0"));
    for (auto& resource : changed_parent.resources) {
      if (resource.id.stable_id == kMig0) {
        resource.id.parent_id = kGpuQ;
        resource.pci_bdf = "0000:05:00.0";
      }
    }
    const auto changed_parent_report = detect_bundle_degradation(
        exact_child_selection, capture(std::move(changed_parent)));
    require(changed_parent_report.health == BundleHealth::degraded &&
                changed_parent_report.changed_parent_resources ==
                    std::vector<std::string>({std::string(kMig0)}),
            "changed partition parent identity degrades the allocation");

    auto duplicate = base_snapshot("revision-duplicate");
    duplicate.resources.push_back(duplicate.resources.front());
    require_rejected([&] { (void)capture(duplicate); },
                     "duplicate stable IDs fail closed");
    auto duplicate_bdf = base_snapshot("revision-duplicate-bdf");
    for (auto& resource : duplicate_bdf.resources) {
      if (resource.id.stable_id == kGpuB) resource.pci_bdf = "0000:01:00.0";
    }
    require_rejected([&] { (void)capture(duplicate_bdf); },
                     "distinct full UUIDs cannot share one PCI BDF");
    auto invalid_function = base_snapshot("revision-bdf-function");
    invalid_function.resources.front().pci_bdf = "0000:03:00.8";
    require_rejected([&] { (void)capture(invalid_function); },
                     "PCI function 8 is outside canonical bounds");
    invalid_function = base_snapshot("revision-bdf-function-f");
    invalid_function.resources.front().pci_bdf = "0000:03:00.f";
    require_rejected([&] { (void)capture(invalid_function); },
                     "PCI function f is outside canonical bounds");
    auto invalid_slot = base_snapshot("revision-bdf-slot");
    invalid_slot.resources.front().pci_bdf = "0000:03:20.0";
    require_rejected([&] { (void)capture(invalid_slot); },
                     "PCI slot 20 hex is outside canonical bounds");
    auto excess_partition_capacity = base_snapshot("revision-capacity");
    for (auto& resource : excess_partition_capacity.resources) {
      if (resource.id.kind == HostResourceKind::accelerator_partition) {
        resource.total_memory_bytes = 50ULL << 30U;
      }
    }
    require_rejected([&] { (void)capture(excess_partition_capacity); },
                     "aggregate sibling capacity cannot exceed its parent");
    auto missing_parent = base_snapshot("revision-parent");
    std::erase_if(missing_parent.resources, [](const auto& resource) {
      return resource.id.stable_id == kGpuP;
    });
    require_rejected([&] { (void)capture(missing_parent); },
                     "partition without same-vendor full parent fails closed");
    auto negative_numa = base_snapshot("revision-numa");
    negative_numa.resources.front().numa_node = -1;
    require_rejected([&] { (void)capture(negative_numa); },
                     "unknown NUMA has one canonical null encoding");

    auto capability_snapshot = base_snapshot("revision-capability-nodes");
    auto& capability_gpu = *std::ranges::find_if(
        capability_snapshot.resources, [](const auto& resource) {
          return resource.id.stable_id == kGpuA;
        });
    capability_gpu.device_nodes = {
        {.type = HostDeviceNodeType::character,
         .purpose = HostDeviceNodePurpose::assigned_accelerator,
         .major = 195U,
         .minor = 1U,
         .read = true,
         .write = true},
        {.type = HostDeviceNodeType::character,
         .purpose = HostDeviceNodePurpose::shared_driver_control,
         .major = 195U,
         .minor = 255U,
         .read = true,
         .write = true}};
    const auto capability_inventory = capture(capability_snapshot);
    const auto& captured_capability = *std::ranges::find_if(
        capability_inventory.resources, [](const auto& resource) {
          return resource.id.stable_id == kGpuA;
        });
    require(captured_capability.device_nodes == capability_gpu.device_nodes,
            "exact assigned and shared device capabilities enter inventory identity");
    auto mismatched_capability = capability_snapshot;
    auto& mismatched_gpu = *std::ranges::find_if(
        mismatched_capability.resources, [](const auto& resource) {
          return resource.id.stable_id == kGpuA;
        });
    mismatched_gpu.device_nodes.front().minor = 9U;
    require_rejected([&] { (void)capture(mismatched_capability); },
                     "assigned capability must match the observed GPU node");
    auto partition_capability = base_snapshot("revision-partition-capability");
    auto& partition_node = *std::ranges::find_if(
        partition_capability.resources, [](const auto& resource) {
          return resource.id.stable_id == kMig0;
        });
    partition_node.device_nodes = capability_gpu.device_nodes;
    require_rejected([&] { (void)capture(partition_capability); },
                     "a parent GPU capability map cannot authorize a partition");

    auto nvidia_alias = request(ResourceAccessMode::exclusive_device,
                                TopologyPolicy::exact_resources, 1U);
    HostResourceId alias{};
    alias.kind = HostResourceKind::accelerator;
    alias.vendor = HostAcceleratorVendor::nvidia;
    alias.stable_id = "GPU-A";
    nvidia_alias.selector.exact_resources = {alias};
    require_rejected([&] { (void)seal_resource_request(nvidia_alias); },
                     "NVIDIA aliases cannot replace canonical lowercase UUIDs");

    auto oversized_request = request(ResourceAccessMode::exclusive_device,
                                     TopologyPolicy::any, 17U);
    require_rejected(
        [&] { (void)seal_resource_request(oversized_request); },
        "bundle count max+1 is rejected");
    auto oversized_labels = request(ResourceAccessMode::exclusive_device,
                                    TopologyPolicy::any, 1U);
    for (std::size_t index = 0;
         index <= HostResourceBounds::maximum_selector_labels; ++index) {
      oversized_labels.selector.exact_labels.emplace(
          "key-" + std::to_string(index), "value");
    }
    require_rejected([&] { (void)seal_resource_request(oversized_labels); },
                     "selector label max+1 is rejected");
    auto oversized_inventory = base_snapshot("revision-oversized");
    oversized_inventory.resources.clear();
    for (std::size_t index = 0;
         index <= HostResourceBounds::maximum_resources; ++index) {
      ObservedHostResource mutex{};
      mutex.id.kind = HostResourceKind::host_mutex;
      mutex.id.stable_id = "host-mutex:item-" + std::to_string(index);
      mutex.disposition =
          ResourceObservationDisposition::audited_eligible;
      mutex.compute_contexts = ResourceContextDisposition::absent;
      mutex.graphics_contexts = ResourceContextDisposition::absent;
      oversized_inventory.resources.push_back(std::move(mutex));
    }
    require_rejected([&] { (void)capture(oversized_inventory); },
                     "inventory resource max+1 is rejected");
    auto unknown_enum = resource_request_json(exact);
    unknown_enum["access_mode"] = "shared_guess";
    require_rejected([&] { (void)resource_request_from_json(unknown_enum); },
                     "unknown reflected enum values fail closed");
    auto hostile_json = resource_request_json(exact);
    hostile_json["run_id"] = std::string(4097U, 'x');
    require_rejected([&] { (void)resource_request_from_json(hostile_json); },
                     "raw JSON strings are bounded before reflected decode");

    auto torn = base_snapshot("revision-torn");
    torn.end_revision = "revision-other";
    FakeHostKernel scripted({
        FakeHostKernelStep{.snapshot = std::nullopt,
                           .failure = "probe permission denied"},
        FakeHostKernelStep{.snapshot = torn, .failure = std::nullopt},
    });
    require_rejected([&] { (void)capture_host_inventory(scripted); },
                     "scripted kernel probe failure fails closed");
    require_rejected([&] { (void)capture_host_inventory(scripted); },
                     "mid-capture revision change is rejected as torn");
    require_rejected([&] { (void)capture_host_inventory(scripted); },
                     "unexpected fake-kernel call fails closed");
    require(scripted.calls() == 2U,
            "unexpected-call failure does not invent another observation");

    auto unknown_fence = fence(
        inventory, gpu_id("GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"));
    require_rejected(
        [&] { (void)occupancy(inventory, {unknown_fence}); },
        "unknown active fence resource fails closed");
    require_rejected(
        [&] {
          (void)occupancy(
              inventory,
              {fence(inventory, gpu_id(kGpuP)),
               fence(inventory, mig_id(kMig0, kGpuP))});
        },
        "conflicting active fence rows fail closed");

    std::cout << "host resource P0.1 tests passed\n";
  } catch (const std::exception& exception) {
    std::cerr << "host_resources_tests: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
