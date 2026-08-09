#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

namespace trainvm {

inline constexpr std::string_view kHostInventoryApiVersion =
    "trainvm.host-inventory/v2";
inline constexpr std::string_view kHostResourceRequestApiVersion =
    "trainvm.host-resource-request/v1";
inline constexpr std::string_view kHostResourceSelectionApiVersion =
    "trainvm.host-resource-selection/v2";
inline constexpr std::string_view kHostResourceOccupancyApiVersion =
    "trainvm.host-resource-occupancy/v1";

struct HostResourceBounds final {
  static constexpr std::size_t maximum_resources = 256U;
  static constexpr std::size_t maximum_probes = 16U;
  static constexpr std::size_t maximum_bundle_count = 16U;
  static constexpr std::size_t maximum_active_fences = 512U;
  static constexpr std::size_t maximum_labels_per_resource = 32U;
  static constexpr std::size_t maximum_selector_labels = 32U;
  static constexpr std::size_t maximum_identifier_bytes = 128U;
  static constexpr std::size_t maximum_label_key_bytes = 64U;
  static constexpr std::size_t maximum_label_value_bytes = 256U;
  static constexpr std::size_t maximum_probe_detail_bytes = 512U;
  // A grant-scoped inventory projection rides inside the grant on the hostd
  // mutation reply, whose canonical JSON is capped at 64 KiB in two
  // independent places (kHostdStatusMaximumPayloadBytes and
  // kMaximumCanonicalBytes). The grant itself already spends a few KiB on its
  // identity and its fences, so the projection gets a bounded share rather
  // than whatever is left: a bound that depends on the rest of the document
  // fails somewhere other than where it is declared.
  //
  // maximum_bundle_count rows of the shapes a real host kernel produces come
  // to well under a kilobyte each; the worst case the *type* permits (32
  // labels of 64+256 bytes) does not fit, and is why the producer omits an
  // oversized projection instead of failing the grant that carries it.
  static constexpr std::size_t maximum_grant_inventory_projection_bytes =
      24U * 1024U;
};

enum class HostResourceKind {
  accelerator,
  accelerator_partition,
  host_mutex,
};

enum class HostAcceleratorVendor {
  nvidia,
  amd,
  intel,
  other,
};

enum class ProbeDisposition {
  complete,
  unavailable,
  partial,
  denied,
  timeout,
};

enum class ResourceObservationDisposition {
  audited_eligible,
  occupied,
  reserved,
  probe_unknown,
};

enum class ResourceContextDisposition {
  absent,
  present,
  unknown,
};

enum class ResourceAccessMode {
  cooperative_compute,
  exclusive_compute,
  exclusive_device,
  partition_exclusive,
  mutex_exclusive,
};

// Whether an observed disposition permits selecting a resource under `mode`.
//
// A device that also drives a display is observed `occupied` for as long as it
// is driving that display — no authorization changes the observation, because
// the observation is true. What an authorization changes is whether an occupied
// device may still be selected cooperatively, which is the question this answers
// and the reason `exclusive: false` exists in a plan.
//
// Exported because two call sites need the identical answer: resource
// acquisition, which holds a full ObservedHostResource, and passive preflight,
// which holds only what the host authority put on the wire. Those two had drifted
// into one place applying this rule and the other applying `disposition ==
// audited_eligible`, so a cooperative plan was rejected before preflight ever
// saw the device.
[[nodiscard]] bool resource_disposition_permits(
    ResourceObservationDisposition disposition,
    const std::map<std::string, std::string>& labels, ResourceAccessMode mode);

enum class TopologyPolicy {
  any,
  same_numa_node,
  same_pcie_root,
  same_fabric_clique,
  exact_resources,
};

struct HostResourceId final {
  // NVIDIA IDs use canonical lowercase GPU-<UUID> and modern R470+
  // MIG-<UUID>. A kernel backend seeing legacy MIG-GPU-... syntax reports a
  // non-complete probe; it never normalizes an authority identity. Other
  // vendors must supply their vendor-canonical persistent ID unchanged.
  HostResourceKind kind{};
  std::optional<HostAcceleratorVendor> vendor;
  std::string stable_id;
  std::optional<std::string> parent_id;

  bool operator==(const HostResourceId&) const = default;
};

struct HostProbeResult final {
  HostAcceleratorVendor vendor{};
  ProbeDisposition disposition{};
  bool context_details_complete{};
  std::string detail;

  bool operator==(const HostProbeResult&) const = default;
};

enum class HostDeviceNodeType {
  character,
  block,
};

enum class HostDeviceNodePurpose {
  assigned_accelerator,
  shared_driver_control,
};

struct HostDeviceNodeCapability final {
  HostDeviceNodeType type{};
  HostDeviceNodePurpose purpose{};
  std::uint32_t major{};
  std::uint32_t minor{};
  bool read{};
  bool write{};

  bool operator==(const HostDeviceNodeCapability&) const = default;
};

struct ObservedHostResource final {
  HostResourceId id;
  // Disposition and context observations are owned by this exact stable ID.
  // A full-device row must not aggregate child-partition contexts: otherwise
  // one occupied child would incorrectly make every sibling unavailable.
  ResourceObservationDisposition disposition{};
  ResourceContextDisposition compute_contexts{};
  ResourceContextDisposition graphics_contexts{};
  std::optional<std::string> pci_bdf;
  // P0.1 records a single device node only for full accelerators. Hardware
  // partitions require a future capability-node set and therefore leave this
  // pair absent; it must never be interpreted as an isolation allowlist.
  std::optional<std::uint32_t> device_major;
  std::optional<std::uint32_t> device_minor;
  // Exact open capabilities used by the cgroup-device policy. An empty set is
  // valid scheduling evidence but cannot authorize process launch. Partition
  // resources remain launch-ineligible until their complete capability-node
  // map is observed; a parent GPU node is never substituted for that map.
  std::vector<HostDeviceNodeCapability> device_nodes;
  std::optional<std::int32_t> numa_node;
  std::optional<std::string> pcie_root_id;
  std::optional<std::string> fabric_clique_id;
  std::uint64_t total_memory_bytes{};
  std::map<std::string, std::string> labels;

  bool operator==(const ObservedHostResource&) const = default;
};

struct HostKernelSnapshot final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string begin_revision;
  std::string end_revision;
  std::vector<HostProbeResult> probes;
  std::vector<ObservedHostResource> resources;

  bool operator==(const HostKernelSnapshot&) const = default;
};

// Volatile point-in-time evidence kept outside HostInventoryReceipt and all
// topology/resource digests. It is observation-only and cannot authorize an
// allocation, lease, device policy, or process.
struct PassiveHostAcceleratorMemory final {
  HostAcceleratorVendor vendor{};
  std::string stable_id;
  std::uint64_t total_memory_bytes{};
  std::uint64_t free_memory_bytes{};

  bool operator==(const PassiveHostAcceleratorMemory &) const = default;
};

struct PassiveHostMemorySnapshot final {
  std::string host_id;
  std::string boot_id;
  std::uint64_t observed_monotonic_ns{};
  std::vector<PassiveHostAcceleratorMemory> accelerators;

  bool operator==(const PassiveHostMemorySnapshot &) const = default;
};

struct HostInventoryReceipt final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string snapshot_revision;
  std::vector<HostProbeResult> probes;
  std::vector<ObservedHostResource> resources;
  std::string topology_digest;
  std::string inventory_digest;
  std::string receipt_digest;

  bool operator==(const HostInventoryReceipt&) const = default;
};

struct ResourceSelector final {
  std::optional<HostAcceleratorVendor> vendor;
  std::optional<std::uint64_t> minimum_memory_bytes;
  std::map<std::string, std::string> exact_labels;
  std::vector<HostResourceId> exact_resources;

  bool operator==(const ResourceSelector&) const = default;
};

struct ResourceBundleRequest final {
  std::string api_version;
  std::string request_id;
  std::string journal_id;
  std::string run_id;
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};
  std::uint32_t count{};
  ResourceAccessMode access_mode{};
  TopologyPolicy topology{};
  ResourceSelector selector;
  std::string canonical_request_digest;

  bool operator==(const ResourceBundleRequest&) const = default;
};

struct ResourceFence final {
  HostResourceId resource;
  std::uint64_t generation{};
  std::string inventory_digest;
  std::string topology_digest;

  bool operator==(const ResourceFence&) const = default;
};

inline constexpr std::string_view kGrantInventoryProjectionApiVersion =
    "trainvm.grant-inventory-projection/v1";

// The grant-time projection of a host inventory receipt onto exactly the
// resources one grant fenced.
//
// Deliberately NOT a HostInventoryReceipt, and not convertible to one. A
// receipt names a whole host and recomputes its own three digests from its own
// contents -- validate_host_inventory rebuilds all three -- so a receipt
// carrying a subset of its rows would be a document that fails the system's
// own check, and would fail it only where someone happened to run that check.
// This type carries the same inventory *identity* the grant fenced against
// over a bounded subset of rows, and says so in its name.
//
// It exists because the whole receipt does not fit the hostd packet budget:
// maximum_resources is 256 and a row may carry 32 labels. What consumes it is
// cache_resource_binding, which reads only the rows its own launch fenced, so
// projecting changes no derived digest -- the binding digest taken over a
// projection equals the one taken over the receipt it came from. That is
// asserted by a test rather than argued here.
struct GrantInventoryProjection final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  // Exactly the fenced rows, in canonical resource-key order.
  std::vector<ObservedHostResource> resources;
  std::string topology_digest;
  std::string inventory_digest;
  std::string receipt_digest;

  bool operator==(const GrantInventoryProjection&) const = default;
};

struct ResourceOccupancySnapshot final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string inventory_digest;
  std::uint64_t ledger_sequence{};
  std::vector<ResourceFence> active_fences;
  std::string occupancy_digest;

  bool operator==(const ResourceOccupancySnapshot&) const = default;
};

struct ResourceBundleSelection final {
  std::string api_version;
  std::string request_digest;
  std::string host_id;
  std::string boot_id;
  std::string snapshot_revision;
  std::string inventory_digest;
  std::string topology_digest;
  std::string occupancy_digest;
  ResourceAccessMode access_mode{};
  std::vector<ObservedHostResource> resources;
  std::string selection_digest;

  bool operator==(const ResourceBundleSelection&) const = default;
};

enum class BundleHealth {
  intact,
  degraded,
};

struct BundleDegradationReport final {
  BundleHealth health{};
  bool host_or_boot_changed{};
  bool topology_changed{};
  std::vector<std::string> vanished_resources;
  std::vector<std::string> changed_parent_resources;

  bool operator==(const BundleDegradationReport&) const = default;
};

class HostResourceError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class IHostKernel {
 public:
  virtual ~IHostKernel() = default;
  [[nodiscard]] virtual HostKernelSnapshot capture_inventory() = 0;
  [[nodiscard]] virtual std::optional<PassiveHostMemorySnapshot>
  passive_memory_snapshot() const {
    return std::nullopt;
  }
};

struct FakeHostKernelStep final {
  std::optional<HostKernelSnapshot> snapshot;
  std::optional<std::string> failure;
};

class FakeHostKernel final : public IHostKernel {
 public:
  explicit FakeHostKernel(std::vector<FakeHostKernelStep> script);
  [[nodiscard]] HostKernelSnapshot capture_inventory() override;
  [[nodiscard]] std::size_t calls() const;
  [[nodiscard]] std::size_t remaining() const;

 private:
  std::vector<FakeHostKernelStep> script_;
  std::size_t cursor_{};
};

[[nodiscard]] std::string canonical_resource_key(const HostResourceId& id);
[[nodiscard]] bool host_resources_conflict(const HostResourceId& left,
                                           const HostResourceId& right);

void validate_host_inventory(const HostInventoryReceipt& receipt);
void validate_resource_request(const ResourceBundleRequest& request);
void validate_resource_fence_shape(
    const std::vector<ResourceFence>& fences,
    std::size_t maximum_count = HostResourceBounds::maximum_active_fences);
void validate_resource_fences(const HostInventoryReceipt& inventory,
                              const std::vector<ResourceFence>& active_fences);
void validate_resource_occupancy(
    const HostInventoryReceipt& inventory,
    const ResourceOccupancySnapshot& occupancy);
void validate_resource_selection(const ResourceBundleSelection& selection);
// Shape, bounds and internal agreement only. It cannot decide whether the
// projection is the one a given grant fenced -- that is
// validate_grant_inventory_projection_against_fences, which the grant's own
// codec and the controller's journal copy both run.
void validate_grant_inventory_projection(
    const GrantInventoryProjection& projection);
void validate_grant_inventory_projection_against_fences(
    const GrantInventoryProjection& projection,
    const std::vector<ResourceFence>& fences);

[[nodiscard]] HostInventoryReceipt capture_host_inventory(IHostKernel& kernel);
[[nodiscard]] ResourceBundleRequest seal_resource_request(
    ResourceBundleRequest request);
[[nodiscard]] ResourceOccupancySnapshot seal_resource_occupancy(
    const HostInventoryReceipt& inventory,
    ResourceOccupancySnapshot occupancy);
[[nodiscard]] std::optional<ResourceBundleSelection> select_host_resources(
    const HostInventoryReceipt& inventory,
    const ResourceBundleRequest& request,
    const ResourceOccupancySnapshot& occupancy);
// Project a receipt onto the rows one grant fenced.
//
// Throws if a fence names a resource the receipt does not hold, or if the
// projection it built disagrees with the fences it was built from. Returns
// nullopt for exactly one fault -- exceeding
// maximum_grant_inventory_projection_bytes -- because that is the only one the
// grant path may legitimately absorb: a grant that cannot be delivered stops a
// run, while a missing projection only refuses a cache evidence publication.
// Answering it by return type rather than by an exception message keeps the
// absorbing caller from having to recognise one throw among several.
[[nodiscard]] std::optional<GrantInventoryProjection>
grant_inventory_projection(const HostInventoryReceipt& receipt,
                           const std::vector<ResourceFence>& fences);
[[nodiscard]] BundleDegradationReport detect_bundle_degradation(
    const ResourceBundleSelection& selection,
    const HostInventoryReceipt& current_inventory);

[[nodiscard]] nlohmann::json host_inventory_json(
    const HostInventoryReceipt& receipt);
[[nodiscard]] HostInventoryReceipt host_inventory_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json resource_request_json(
    const ResourceBundleRequest& request);
[[nodiscard]] ResourceBundleRequest resource_request_from_json(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json resource_occupancy_json(
    const ResourceOccupancySnapshot& occupancy);
[[nodiscard]] ResourceOccupancySnapshot resource_occupancy_from_json(
    const HostInventoryReceipt& inventory, const nlohmann::json& source);
[[nodiscard]] nlohmann::json resource_selection_json(
    const ResourceBundleSelection& selection);
[[nodiscard]] ResourceBundleSelection resource_selection_from_json(
    const nlohmann::json& source);

}  // namespace trainvm
