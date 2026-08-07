#include "trainvm/host_resources.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <concepts>
#include <memory>
#include <ranges>
#include <set>
#include <tuple>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr bool ascii_digit(char character) {
  return character >= '0' && character <= '9';
}

constexpr bool ascii_lower_hex(char character) {
  return ascii_digit(character) ||
         (character >= 'a' && character <= 'f');
}

constexpr bool ascii_alnum(char character) {
  return ascii_digit(character) ||
         (character >= 'a' && character <= 'z') ||
         (character >= 'A' && character <= 'Z');
}

constexpr bool ascii_printable(char character) {
  const auto byte = static_cast<unsigned char>(character);
  return byte >= 0x20U && byte <= 0x7eU;
}

struct EvpDeleter {
  void operator()(EVP_MD_CTX* context) const { EVP_MD_CTX_free(context); }
};

using EvpContext = std::unique_ptr<EVP_MD_CTX, EvpDeleter>;

std::string hex_digest(const unsigned char* bytes, unsigned int length) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(static_cast<std::size_t>(length) * 2U);
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(digits[bytes[index] >> 4U]);
    result.push_back(digits[bytes[index] & 0x0fU]);
  }
  return result;
}

std::string sha256(std::string_view domain, const nlohmann::json& value) {
  EvpContext context(EVP_MD_CTX_new());
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), domain.data(), domain.size()) != 1) {
    throw HostResourceError("could not initialize host-resource SHA-256");
  }
  const char separator = '\0';
  const auto canonical = value.dump();
  if (EVP_DigestUpdate(context.get(), &separator, 1U) != 1 ||
      EVP_DigestUpdate(context.get(), canonical.data(), canonical.size()) != 1) {
    throw HostResourceError("could not update host-resource SHA-256");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) {
    throw HostResourceError("could not finalize host-resource SHA-256");
  }
  return "sha256:" + hex_digest(digest.data(), length);
}

bool digest_valid(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7), [](char character) {
           return ascii_lower_hex(character);
         });
}

bool bounded_identifier(std::string_view value) {
  return !value.empty() &&
         value.size() <= HostResourceBounds::maximum_identifier_bytes &&
         std::ranges::all_of(value, [](char character) {
           return ascii_alnum(character) || character == '.' || character == '_' ||
                  character == ':' || character == '/' || character == '-';
         });
}

bool canonical_pci_bdf(std::string_view value) {
  if (value.size() != 12U || value[4] != ':' || value[7] != ':' ||
      value[10] != '.') {
    return false;
  }
  for (const auto index : {0U, 1U, 2U, 3U, 5U, 6U, 8U, 9U, 11U}) {
    const char character = value[index];
    if (!ascii_lower_hex(character)) {
      return false;
    }
  }
  const auto slot_value = [](char high, char low) {
    const auto digit = [](char nibble) -> unsigned int {
      return ascii_digit(nibble) ? static_cast<unsigned int>(nibble - '0')
                                 : static_cast<unsigned int>(nibble - 'a' + 10);
    };
    return digit(high) * 16U + digit(low);
  };
  return slot_value(value[8], value[9]) <= 0x1fU && value[11] <= '7';
}

bool canonical_uuid_id(std::string_view value, std::string_view prefix) {
  if (!value.starts_with(prefix) || value.size() != prefix.size() + 36U) {
    return false;
  }
  const auto suffix = value.substr(prefix.size());
  for (std::size_t index = 0; index < suffix.size(); ++index) {
    const bool hyphen = index == 8U || index == 13U || index == 18U ||
                        index == 23U;
    if (hyphen ? suffix[index] != '-' : !ascii_lower_hex(suffix[index])) {
      return false;
    }
  }
  return true;
}

void require_identifier(std::string_view value, std::string_view field) {
  if (!bounded_identifier(value)) {
    throw HostResourceError(std::string(field) +
                            " must be a bounded stable identifier");
  }
}

void validate_labels(const std::map<std::string, std::string>& labels,
                     std::size_t maximum) {
  if (labels.size() > maximum) {
    throw HostResourceError("host-resource label count exceeds its bound");
  }
  for (const auto& [key, value] : labels) {
    if (key.empty() || key.size() > HostResourceBounds::maximum_label_key_bytes ||
        value.size() > HostResourceBounds::maximum_label_value_bytes ||
        !std::ranges::all_of(key, [](char character) {
          return ascii_alnum(character) || character == '.' || character == '_' ||
                 character == '-';
        }) ||
        !std::ranges::all_of(value, ascii_printable)) {
      throw HostResourceError("host-resource labels are not canonical bounded text");
    }
  }
}

void validate_resource_id(const HostResourceId& id) {
  require_identifier(id.stable_id, "resource stable_id");
  if (id.kind == HostResourceKind::accelerator) {
    if (!id.vendor || id.parent_id) {
      throw HostResourceError(
          "full accelerators require a vendor and forbid a parent");
    }
    if (id.vendor == HostAcceleratorVendor::nvidia &&
        !canonical_uuid_id(id.stable_id, "GPU-")) {
      throw HostResourceError("NVIDIA full devices require canonical GPU UUIDs");
    }
  } else if (id.kind == HostResourceKind::accelerator_partition) {
    if (!id.vendor || !id.parent_id ||
        !bounded_identifier(*id.parent_id) || *id.parent_id == id.stable_id) {
      throw HostResourceError(
          "accelerator partitions require a distinct stable parent and vendor");
    }
    if (id.vendor == HostAcceleratorVendor::nvidia &&
        (!canonical_uuid_id(id.stable_id, "MIG-") ||
         !canonical_uuid_id(*id.parent_id, "GPU-"))) {
      throw HostResourceError(
          "NVIDIA partitions require canonical MIG and parent GPU UUIDs");
    }
  } else if (id.vendor || id.parent_id ||
             !id.stable_id.starts_with("host-mutex:") ||
             id.stable_id.size() == std::string_view("host-mutex:").size()) {
    throw HostResourceError(
        "host mutexes require a canonical host-mutex ID and forbid vendor/parent");
  }
}

bool id_less(const HostResourceId& left, const HostResourceId& right) {
  return canonical_resource_key(left) < canonical_resource_key(right);
}

bool resource_less(const ObservedHostResource& left,
                   const ObservedHostResource& right) {
  return id_less(left.id, right.id);
}

bool device_node_less(const HostDeviceNodeCapability& left,
                      const HostDeviceNodeCapability& right) {
  return std::tie(left.type, left.purpose, left.major, left.minor, left.read,
                  left.write) <
         std::tie(right.type, right.purpose, right.major, right.minor,
                  right.read, right.write);
}

void validate_device_nodes(const ObservedHostResource& resource) {
  if (resource.device_nodes.size() > 32U ||
      !std::ranges::is_sorted(resource.device_nodes, device_node_less)) {
    throw HostResourceError(
        "device-node capabilities are unbounded or noncanonical");
  }
  std::set<std::tuple<HostDeviceNodeType, std::uint32_t, std::uint32_t>> nodes;
  std::size_t assigned = 0U;
  for (const auto& node : resource.device_nodes) {
    if ((!node.read && !node.write) || node.major > 0xfffU ||
        node.minor > 0xfffffU ||
        !nodes.emplace(node.type, node.major, node.minor).second) {
      throw HostResourceError(
          "device-node capability is unusable or duplicate");
    }
    if (node.purpose == HostDeviceNodePurpose::assigned_accelerator) {
      ++assigned;
      if (!resource.device_major || !resource.device_minor ||
          node.type != HostDeviceNodeType::character ||
          node.major != *resource.device_major ||
          node.minor != *resource.device_minor) {
        throw HostResourceError(
            "assigned device-node capability disagrees with topology");
      }
    }
  }
  if (!resource.device_nodes.empty() && assigned != 1U) {
    throw HostResourceError(
        "nonempty accelerator capability map requires one assigned node");
  }
}

void validate_observed_resource(const ObservedHostResource& resource) {
  validate_resource_id(resource.id);
  if (resource.labels.size() > HostResourceBounds::maximum_labels_per_resource) {
    throw HostResourceError("resource label count exceeds its bound");
  }
  validate_labels(resource.labels,
                  HostResourceBounds::maximum_labels_per_resource);
  if (resource.device_major.has_value() != resource.device_minor.has_value()) {
    throw HostResourceError("device major and minor must be observed together");
  }
  validate_device_nodes(resource);
  if (resource.pci_bdf && !canonical_pci_bdf(*resource.pci_bdf)) {
    throw HostResourceError("PCI BDF must use canonical lowercase dddd:bb:ss.f");
  }
  if (resource.numa_node && *resource.numa_node < 0) {
    throw HostResourceError("NUMA identity must be nonnegative or absent");
  }
  for (const auto* identity : {&resource.pci_bdf, &resource.pcie_root_id,
                               &resource.fabric_clique_id}) {
    if (*identity && !bounded_identifier(**identity)) {
      throw HostResourceError("topology identities must be bounded stable text");
    }
  }
  if (resource.id.kind == HostResourceKind::host_mutex) {
    if (resource.total_memory_bytes != 0U || resource.pci_bdf ||
        resource.device_major || resource.device_minor ||
        !resource.device_nodes.empty() || resource.numa_node ||
        resource.pcie_root_id ||
        resource.fabric_clique_id) {
      throw HostResourceError("host mutexes cannot carry accelerator topology");
    }
    if (resource.compute_contexts != ResourceContextDisposition::absent ||
        resource.graphics_contexts != ResourceContextDisposition::absent) {
      throw HostResourceError("host mutexes cannot carry accelerator contexts");
    }
  } else if (resource.total_memory_bytes == 0U) {
    throw HostResourceError("accelerator resources require nonzero capacity");
  }
  if (resource.id.kind == HostResourceKind::accelerator_partition &&
      (resource.device_major || resource.device_minor ||
       !resource.device_nodes.empty())) {
    throw HostResourceError(
        "partition single-node evidence is not an isolation mapping");
  }
  if (resource.disposition ==
          ResourceObservationDisposition::audited_eligible &&
      (resource.compute_contexts != ResourceContextDisposition::absent ||
       resource.graphics_contexts == ResourceContextDisposition::unknown)) {
    throw HostResourceError(
        "audited eligible resources require absent compute and known graphics contexts");
  }
}

nlohmann::json topology_json(
    const std::vector<ObservedHostResource>& resources) {
  nlohmann::json values = nlohmann::json::array();
  for (const auto& resource : resources) {
    values.push_back({
        {"id", encode_json(resource.id)},
        {"pci_bdf", resource.pci_bdf},
        {"device_major", resource.device_major},
        {"device_minor", resource.device_minor},
        {"device_nodes", encode_json(resource.device_nodes)},
        {"numa_node", resource.numa_node},
        {"pcie_root_id", resource.pcie_root_id},
        {"fabric_clique_id", resource.fabric_clique_id},
        {"total_memory_bytes", resource.total_memory_bytes},
    });
  }
  return values;
}

nlohmann::json inventory_digest_json(const HostInventoryReceipt& receipt) {
  return {{"api_version", receipt.api_version},
          {"host_id", receipt.host_id},
          {"boot_id", receipt.boot_id},
          {"broker_epoch", receipt.broker_epoch},
          {"snapshot_revision", receipt.snapshot_revision},
          {"probes", encode_json(receipt.probes)},
          {"resources", encode_json(receipt.resources)},
          {"topology_digest", receipt.topology_digest}};
}

nlohmann::json inventory_receipt_digest_json(
    const HostInventoryReceipt& receipt) {
  auto value = inventory_digest_json(receipt);
  value["inventory_digest"] = receipt.inventory_digest;
  return value;
}

nlohmann::json request_digest_json(const ResourceBundleRequest& request) {
  return {{"api_version", request.api_version},
          {"request_id", request.request_id},
          {"journal_id", request.journal_id},
          {"run_id", request.run_id},
          {"logical_lease_id", request.logical_lease_id},
          {"logical_fencing_token", request.logical_fencing_token},
          {"count", request.count},
          {"access_mode", enum_to_string(request.access_mode)},
          {"topology", enum_to_string(request.topology)},
          {"selector", encode_json(request.selector)}};
}

nlohmann::json selection_digest_json(
    const ResourceBundleSelection& selection) {
  return {{"api_version", selection.api_version},
          {"request_digest", selection.request_digest},
          {"host_id", selection.host_id},
          {"boot_id", selection.boot_id},
          {"snapshot_revision", selection.snapshot_revision},
          {"inventory_digest", selection.inventory_digest},
          {"topology_digest", selection.topology_digest},
          {"occupancy_digest", selection.occupancy_digest},
          {"access_mode", enum_to_string(selection.access_mode)},
          {"resources", encode_json(selection.resources)}};
}

nlohmann::json occupancy_digest_json(
    const ResourceOccupancySnapshot& occupancy) {
  return {{"api_version", occupancy.api_version},
          {"host_id", occupancy.host_id},
          {"boot_id", occupancy.boot_id},
          {"inventory_digest", occupancy.inventory_digest},
          {"ledger_sequence", occupancy.ledger_sequence},
          {"active_fences", encode_json(occupancy.active_fences)}};
}

void validate_occupancy_intrinsic(
    const ResourceOccupancySnapshot& occupancy) {
  if (occupancy.api_version != kHostResourceOccupancyApiVersion) {
    throw HostResourceError("unsupported resource occupancy api_version");
  }
  require_identifier(occupancy.host_id, "occupancy host_id");
  require_identifier(occupancy.boot_id, "occupancy boot_id");
  if (!digest_valid(occupancy.inventory_digest) ||
      occupancy.active_fences.size() >
          HostResourceBounds::maximum_active_fences ||
      !std::ranges::is_sorted(occupancy.active_fences,
                              [](const auto& left, const auto& right) {
        return id_less(left.resource, right.resource);
      })) {
    throw HostResourceError("resource occupancy is malformed or noncanonical");
  }
  std::set<std::string> ids;
  for (std::size_t left = 0; left < occupancy.active_fences.size(); ++left) {
    const auto& fence = occupancy.active_fences[left];
    validate_resource_id(fence.resource);
    if (fence.generation == 0U || !digest_valid(fence.inventory_digest) ||
        !digest_valid(fence.topology_digest) ||
        !ids.insert(fence.resource.stable_id).second) {
      throw HostResourceError("resource occupancy fence is invalid or duplicate");
    }
    for (std::size_t right = left + 1U;
         right < occupancy.active_fences.size(); ++right) {
      if (host_resources_conflict(fence.resource,
                                  occupancy.active_fences[right].resource)) {
        throw HostResourceError("resource occupancy fences conflict");
      }
    }
  }
  const auto expected = sha256("trainvm.host-resource-occupancy/v1",
                               occupancy_digest_json(occupancy));
  if (occupancy.occupancy_digest != expected) {
    throw HostResourceError("resource occupancy digest validation failed");
  }
}

template <typename T>
T strict_decode(const nlohmann::json& source, std::string_view description) {
  std::size_t nodes = 0U;
  const auto preflight = [&]<typename Self>(this Self&& self,
                                            const nlohmann::json& value,
                                            std::size_t depth) -> void {
    if (depth > 32U || ++nodes > 8192U ||
        (value.is_array() && value.size() > 512U) ||
        (value.is_object() && value.size() > 64U) ||
        (value.is_string() && value.get_ref<const std::string&>().size() >
                                  4096U)) {
      throw HostResourceError("host-resource JSON exceeds structural bounds");
    }
    if (value.is_object()) {
      for (const auto& [key, child] : value.items()) {
        if (key.size() > HostResourceBounds::maximum_identifier_bytes ||
            !std::ranges::all_of(key, ascii_printable)) {
          throw HostResourceError("host-resource JSON key is not bounded ASCII");
        }
        self(child, depth + 1U);
      }
    } else if (value.is_array()) {
      for (const auto& child : value) self(child, depth + 1U);
    }
  };
  preflight(source, 0U);
  try {
    if (source.dump(-1, ' ', false,
                    nlohmann::json::error_handler_t::strict).size() >
        (2U << 20U)) {
      throw HostResourceError("host-resource JSON exceeds its byte bound");
    }
  } catch (const nlohmann::json::exception&) {
    throw HostResourceError("host-resource JSON is not canonical UTF-8");
  }
  T value;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, value, "", diagnostics)) {
    throw HostResourceError(std::string(description) +
                            " failed strict reflected decoding");
  }
  return value;
}

void preflight_kernel_snapshot(const HostKernelSnapshot& snapshot) {
  if (snapshot.api_version != kHostInventoryApiVersion ||
      snapshot.probes.size() > HostResourceBounds::maximum_probes ||
      snapshot.resources.size() > HostResourceBounds::maximum_resources) {
    throw HostResourceError("kernel snapshot exceeds collection bounds");
  }
  require_identifier(snapshot.host_id, "snapshot host_id");
  require_identifier(snapshot.boot_id, "snapshot boot_id");
  require_identifier(snapshot.broker_epoch, "snapshot broker_epoch");
  require_identifier(snapshot.begin_revision, "snapshot begin_revision");
  require_identifier(snapshot.end_revision, "snapshot end_revision");
  for (const auto& probe : snapshot.probes) {
    if (probe.detail.size() > HostResourceBounds::maximum_probe_detail_bytes ||
        !std::ranges::all_of(probe.detail, ascii_printable)) {
      throw HostResourceError("kernel probe detail exceeds its bound");
    }
  }
  for (const auto& resource : snapshot.resources) {
    validate_observed_resource(resource);
  }
}

void preflight_request(const ResourceBundleRequest& request) {
  if (request.api_version != kHostResourceRequestApiVersion) {
    throw HostResourceError("unsupported resource request api_version");
  }
  require_identifier(request.request_id, "request_id");
  require_identifier(request.journal_id, "journal_id");
  require_identifier(request.run_id, "run_id");
  require_identifier(request.logical_lease_id, "logical_lease_id");
  if (request.logical_fencing_token == 0U || request.count == 0U ||
      request.count > HostResourceBounds::maximum_bundle_count ||
      request.selector.exact_resources.size() >
          HostResourceBounds::maximum_bundle_count) {
    throw HostResourceError("resource request exceeds collection bounds");
  }
  validate_labels(request.selector.exact_labels,
                  HostResourceBounds::maximum_selector_labels);
  for (const auto& resource : request.selector.exact_resources) {
    validate_resource_id(resource);
  }
}

void preflight_occupancy(const ResourceOccupancySnapshot& occupancy) {
  if (occupancy.api_version != kHostResourceOccupancyApiVersion ||
      !digest_valid(occupancy.inventory_digest)) {
    throw HostResourceError("resource occupancy preflight failed");
  }
  require_identifier(occupancy.host_id, "occupancy host_id");
  require_identifier(occupancy.boot_id, "occupancy boot_id");
  if (occupancy.active_fences.size() >
      HostResourceBounds::maximum_active_fences) {
    throw HostResourceError("resource occupancy exceeds its fence bound");
  }
  for (const auto& fence : occupancy.active_fences) {
    validate_resource_id(fence.resource);
    if (fence.generation == 0U || !digest_valid(fence.inventory_digest) ||
        !digest_valid(fence.topology_digest)) {
      throw HostResourceError("resource occupancy fence preflight failed");
    }
  }
}

const HostProbeResult* probe_for(const HostInventoryReceipt& inventory,
                                 HostAcceleratorVendor vendor) {
  const auto found = std::ranges::find(inventory.probes, vendor,
                                       &HostProbeResult::vendor);
  return found == inventory.probes.end() ? nullptr : &*found;
}

bool probe_allows_selection(const HostInventoryReceipt& inventory,
                            const ObservedHostResource& resource) {
  if (resource.id.kind == HostResourceKind::host_mutex) return true;
  const auto* probe = probe_for(inventory, *resource.id.vendor);
  return probe != nullptr && probe->disposition == ProbeDisposition::complete &&
         probe->context_details_complete;
}

HostResourceKind kind_for(ResourceAccessMode mode) {
  if (mode == ResourceAccessMode::partition_exclusive) {
    return HostResourceKind::accelerator_partition;
  }
  if (mode == ResourceAccessMode::mutex_exclusive) {
    return HostResourceKind::host_mutex;
  }
  return HostResourceKind::accelerator;
}

bool selector_fields_match(const ObservedHostResource& resource,
                           const ResourceSelector& selector) {
  if (selector.vendor && resource.id.vendor != selector.vendor) return false;
  if (selector.minimum_memory_bytes &&
      resource.total_memory_bytes < *selector.minimum_memory_bytes) {
    return false;
  }
  for (const auto& [key, value] : selector.exact_labels) {
    const auto found = resource.labels.find(key);
    if (found == resource.labels.end() || found->second != value) return false;
  }
  return true;
}

bool contexts_allow(const ObservedHostResource& resource,
                    ResourceAccessMode mode) {
  if (mode == ResourceAccessMode::mutex_exclusive) return true;
  if (resource.compute_contexts == ResourceContextDisposition::unknown ||
      resource.graphics_contexts == ResourceContextDisposition::unknown) {
    return false;
  }
  if (mode == ResourceAccessMode::cooperative_compute) return true;
  if (resource.compute_contexts != ResourceContextDisposition::absent) {
    return false;
  }
  if (mode == ResourceAccessMode::exclusive_compute) return true;
  return resource.graphics_contexts == ResourceContextDisposition::absent;
}

bool disposition_allows(const ObservedHostResource& resource,
                        ResourceAccessMode mode) {
  return resource_disposition_permits(resource.disposition, resource.labels,
                                      mode);
}

bool blocked_by_conflicting_observation(
    const HostInventoryReceipt& inventory,
    const ObservedHostResource& candidate, ResourceAccessMode access_mode) {
  return std::ranges::any_of(inventory.resources, [&](const auto& observed) {
    if (observed.id.stable_id == candidate.id.stable_id ||
        !host_resources_conflict(candidate.id, observed.id)) {
      return false;
    }
    return !disposition_allows(observed, access_mode) ||
           !probe_allows_selection(inventory, observed) ||
           !contexts_allow(observed, access_mode);
  });
}

bool blocked_by_fence(const HostResourceId& resource,
                      const std::vector<ResourceFence>& active_fences) {
  return std::ranges::any_of(active_fences, [&](const ResourceFence& fence) {
    return host_resources_conflict(resource, fence.resource);
  });
}

std::optional<std::string> topology_group_key(
    const ObservedHostResource& resource, TopologyPolicy policy) {
  if (policy == TopologyPolicy::same_numa_node) {
    if (!resource.numa_node) return std::nullopt;
    return "numa:" + std::to_string(*resource.numa_node);
  }
  if (policy == TopologyPolicy::same_pcie_root) {
    return resource.pcie_root_id;
  }
  if (policy == TopologyPolicy::same_fabric_clique) {
    return resource.fabric_clique_id;
  }
  return std::string("any");
}

bool topology_facts_equal(const ObservedHostResource& left,
                          const ObservedHostResource& right) {
  return left.id == right.id && left.pci_bdf == right.pci_bdf &&
         left.device_major == right.device_major &&
         left.device_minor == right.device_minor &&
         left.device_nodes == right.device_nodes &&
         left.numa_node == right.numa_node &&
         left.pcie_root_id == right.pcie_root_id &&
         left.fabric_clique_id == right.fabric_clique_id &&
         left.total_memory_bytes == right.total_memory_bytes;
}

}  // namespace

bool resource_disposition_permits(
    ResourceObservationDisposition disposition,
    const std::map<std::string, std::string>& labels, ResourceAccessMode mode) {
  if (disposition == ResourceObservationDisposition::audited_eligible)
    return true;
  if (mode != ResourceAccessMode::cooperative_compute ||
      disposition != ResourceObservationDisposition::occupied) {
    return false;
  }
  const bool display_device = labels.contains("display");
  if (!display_device) return true;
  const auto authorization = labels.find("display-sharing");
  return authorization != labels.end() &&
         authorization->second == "operator-authorized-cooperative";
}

std::string canonical_resource_key(const HostResourceId& id) {
  return id.stable_id + "\x1f" + enum_to_string(id.kind) + "\x1f" +
         (id.vendor ? enum_to_string(*id.vendor) : "-") + "\x1f" +
         (id.parent_id ? *id.parent_id : "-");
}

bool host_resources_conflict(const HostResourceId& left,
                             const HostResourceId& right) {
  if (left.stable_id == right.stable_id) return true;
  if (left.kind == HostResourceKind::accelerator && right.parent_id &&
      *right.parent_id == left.stable_id) {
    return true;
  }
  if (right.kind == HostResourceKind::accelerator && left.parent_id &&
      *left.parent_id == right.stable_id) {
    return true;
  }
  return false;
}

void validate_host_inventory(const HostInventoryReceipt& receipt) {
  if (receipt.api_version != kHostInventoryApiVersion) {
    throw HostResourceError("unsupported host inventory api_version");
  }
  require_identifier(receipt.host_id, "host_id");
  require_identifier(receipt.boot_id, "boot_id");
  require_identifier(receipt.broker_epoch, "broker_epoch");
  require_identifier(receipt.snapshot_revision, "snapshot_revision");
  if (receipt.probes.size() > HostResourceBounds::maximum_probes ||
      receipt.resources.size() > HostResourceBounds::maximum_resources) {
    throw HostResourceError("host inventory exceeds a collection bound");
  }
  if (!std::ranges::is_sorted(receipt.probes, {}, &HostProbeResult::vendor) ||
      !std::ranges::is_sorted(receipt.resources, resource_less)) {
    throw HostResourceError("host inventory collections are not canonical");
  }
  std::set<HostAcceleratorVendor> vendors;
  for (const auto& probe : receipt.probes) {
    if (!vendors.insert(probe.vendor).second ||
        probe.detail.size() > HostResourceBounds::maximum_probe_detail_bytes ||
        !std::ranges::all_of(probe.detail, ascii_printable)) {
      throw HostResourceError("host probe evidence is duplicate or unbounded");
    }
    if (probe.disposition != ProbeDisposition::complete &&
        probe.context_details_complete) {
      throw HostResourceError(
          "non-complete probes cannot claim complete context evidence");
    }
  }
  std::map<std::string, const ObservedHostResource*> resources;
  std::set<std::string> full_device_bdfs;
  std::set<std::pair<std::uint32_t, std::uint32_t>> full_device_nodes;
  for (const auto& resource : receipt.resources) {
    validate_observed_resource(resource);
    if (!resources.emplace(resource.id.stable_id, &resource).second) {
      throw HostResourceError("resource stable IDs must be globally unique");
    }
    if (resource.id.kind == HostResourceKind::accelerator) {
      if (resource.pci_bdf &&
          !full_device_bdfs.insert(*resource.pci_bdf).second) {
        throw HostResourceError("full accelerator PCI BDFs must be unique");
      }
      if (resource.device_major &&
          !full_device_nodes
               .emplace(*resource.device_major, *resource.device_minor)
               .second) {
        throw HostResourceError(
            "full accelerator device-node identities must be unique");
      }
    }
    if (resource.id.kind != HostResourceKind::host_mutex) {
      const auto* probe = probe_for(receipt, *resource.id.vendor);
      if (probe == nullptr) {
        throw HostResourceError("accelerator resource has no vendor probe");
      }
      if ((probe->disposition != ProbeDisposition::complete ||
           !probe->context_details_complete) &&
          resource.disposition ==
              ResourceObservationDisposition::audited_eligible) {
        throw HostResourceError(
            "unknown vendor/context evidence can never be eligible");
      }
    }
  }
  std::map<std::string, std::uint64_t> partition_capacity;
  for (const auto& [stable_id, resource] : resources) {
    (void)stable_id;
    if (resource->id.kind != HostResourceKind::accelerator_partition) continue;
    const auto parent = resources.find(*resource->id.parent_id);
    if (parent == resources.end() ||
        parent->second->id.kind != HostResourceKind::accelerator ||
        parent->second->id.vendor != resource->id.vendor) {
      throw HostResourceError(
          "partition parent must be an observed same-vendor full accelerator");
    }
    const auto* physical_parent = parent->second;
    if (resource->pci_bdf != physical_parent->pci_bdf ||
        resource->numa_node != physical_parent->numa_node ||
        resource->pcie_root_id != physical_parent->pcie_root_id ||
        resource->fabric_clique_id != physical_parent->fabric_clique_id ||
        resource->total_memory_bytes > physical_parent->total_memory_bytes) {
      throw HostResourceError(
          "partition placement/capacity must agree with its physical parent");
    }
    auto& total = partition_capacity[*resource->id.parent_id];
    if (resource->total_memory_bytes >
        physical_parent->total_memory_bytes - total) {
      throw HostResourceError(
          "aggregate sibling partition capacity exceeds its parent");
    }
    total += resource->total_memory_bytes;
  }
  const auto expected_topology =
      sha256("trainvm.host-topology/v2", topology_json(receipt.resources));
  const auto expected_inventory = sha256(
      "trainvm.host-inventory-content/v2", inventory_digest_json(receipt));
  const auto expected_receipt = sha256(
      "trainvm.host-inventory-receipt/v2",
      inventory_receipt_digest_json(receipt));
  if (receipt.topology_digest != expected_topology ||
      receipt.inventory_digest != expected_inventory ||
      receipt.receipt_digest != expected_receipt) {
    throw HostResourceError("host inventory digest validation failed");
  }
}

void validate_resource_request(const ResourceBundleRequest& request) {
  if (request.api_version != kHostResourceRequestApiVersion) {
    throw HostResourceError("unsupported resource request api_version");
  }
  require_identifier(request.request_id, "request_id");
  require_identifier(request.journal_id, "journal_id");
  require_identifier(request.run_id, "run_id");
  require_identifier(request.logical_lease_id, "logical_lease_id");
  if (request.logical_fencing_token == 0U || request.count == 0U ||
      request.count > HostResourceBounds::maximum_bundle_count) {
    throw HostResourceError("resource request token/count is outside bounds");
  }
  validate_labels(request.selector.exact_labels,
                  HostResourceBounds::maximum_selector_labels);
  if (request.access_mode == ResourceAccessMode::mutex_exclusive &&
      (request.selector.vendor || request.selector.minimum_memory_bytes)) {
    throw HostResourceError("mutex requests cannot use accelerator filters");
  }
  if (request.selector.minimum_memory_bytes == 0U) {
    throw HostResourceError("minimum memory must be nonzero when present");
  }
  if (request.access_mode == ResourceAccessMode::mutex_exclusive &&
      request.topology != TopologyPolicy::any &&
      request.topology != TopologyPolicy::exact_resources) {
    throw HostResourceError("host mutex requests cannot use hardware topology");
  }
  if (request.topology == TopologyPolicy::exact_resources) {
    if (request.selector.exact_resources.empty() ||
        request.selector.exact_resources.size() != request.count ||
        !std::ranges::is_sorted(request.selector.exact_resources, id_less)) {
      throw HostResourceError(
          "exact topology requires a canonical exact list matching count");
    }
    std::set<std::string> exact_ids;
    for (const auto& id : request.selector.exact_resources) {
      validate_resource_id(id);
      if (id.kind != kind_for(request.access_mode) ||
          (request.selector.vendor && id.vendor != request.selector.vendor) ||
          !exact_ids.insert(id.stable_id).second) {
        throw HostResourceError(
            "exact resources must be unique and match access mode");
      }
    }
    for (std::size_t left = 0; left < request.selector.exact_resources.size();
         ++left) {
      for (std::size_t right = left + 1U;
           right < request.selector.exact_resources.size(); ++right) {
        if (host_resources_conflict(request.selector.exact_resources[left],
                                    request.selector.exact_resources[right])) {
          throw HostResourceError("exact resource bundle conflicts internally");
        }
      }
    }
  } else if (!request.selector.exact_resources.empty()) {
    throw HostResourceError(
        "exact_resources selector is valid only with exact topology");
  }
  const auto expected =
      sha256("trainvm.host-resource-request/v1", request_digest_json(request));
  if (request.canonical_request_digest != expected) {
    throw HostResourceError("resource request digest validation failed");
  }
}

void validate_resource_fences(
    const HostInventoryReceipt& inventory,
    const std::vector<ResourceFence>& active_fences) {
  validate_resource_fence_shape(active_fences,
                                HostResourceBounds::maximum_active_fences);
  for (const auto& fence : active_fences) {
    const auto found = std::ranges::find_if(
        inventory.resources, [&](const auto& resource) {
          return resource.id.stable_id == fence.resource.stable_id;
        });
    if (found == inventory.resources.end() || found->id != fence.resource) {
      throw HostResourceError(
          "active fence resource vanished or changed identity");
    }
  }
}

void validate_resource_fence_shape(
    const std::vector<ResourceFence>& active_fences,
    std::size_t maximum_count) {
  if (active_fences.size() > maximum_count) {
    throw HostResourceError("active fence count exceeds its bound");
  }
  if (!std::ranges::is_sorted(active_fences, [](const auto& left,
                                                const auto& right) {
        return id_less(left.resource, right.resource);
      })) {
    throw HostResourceError("active fences are not canonically ordered");
  }
  std::set<std::string> ids;
  for (const auto& fence : active_fences) {
    validate_resource_id(fence.resource);
    if (fence.generation == 0U || !digest_valid(fence.inventory_digest) ||
        !digest_valid(fence.topology_digest) ||
        !ids.insert(fence.resource.stable_id).second) {
      throw HostResourceError("active fence evidence is invalid or duplicate");
    }
  }
  for (std::size_t left = 0; left < active_fences.size(); ++left) {
    for (std::size_t right = left + 1U; right < active_fences.size(); ++right) {
      if (host_resources_conflict(active_fences[left].resource,
                                  active_fences[right].resource)) {
        throw HostResourceError("active fences conflict with one another");
      }
    }
  }
}

void validate_resource_selection(const ResourceBundleSelection& selection) {
  if (selection.api_version != kHostResourceSelectionApiVersion ||
      !digest_valid(selection.request_digest) ||
      !digest_valid(selection.inventory_digest) ||
      !digest_valid(selection.topology_digest) ||
      !digest_valid(selection.occupancy_digest) ||
      selection.resources.empty() ||
      selection.resources.size() > HostResourceBounds::maximum_bundle_count ||
      !std::ranges::is_sorted(selection.resources, resource_less)) {
    throw HostResourceError("resource selection is malformed or noncanonical");
  }
  require_identifier(selection.host_id, "selection host_id");
  require_identifier(selection.boot_id, "selection boot_id");
  require_identifier(selection.snapshot_revision, "selection revision");
  std::set<std::string> selected_ids;
  for (const auto& resource : selection.resources) {
    validate_observed_resource(resource);
    if (!disposition_allows(resource, selection.access_mode) ||
        resource.id.kind != kind_for(selection.access_mode) ||
        !contexts_allow(resource, selection.access_mode) ||
        !selected_ids.insert(resource.id.stable_id).second) {
      throw HostResourceError(
          "selection contains a duplicate or ineligible resource");
    }
  }
  for (std::size_t left = 0; left < selection.resources.size(); ++left) {
    for (std::size_t right = left + 1U; right < selection.resources.size();
         ++right) {
      if (host_resources_conflict(selection.resources[left].id,
                                  selection.resources[right].id)) {
        throw HostResourceError("selection contains conflicting resources");
      }
    }
  }
  const auto expected = sha256("trainvm.host-resource-selection/v2",
                               selection_digest_json(selection));
  if (selection.selection_digest != expected) {
    throw HostResourceError("resource selection digest validation failed");
  }
}

void validate_resource_occupancy(
    const HostInventoryReceipt& inventory,
    const ResourceOccupancySnapshot& occupancy) {
  validate_occupancy_intrinsic(occupancy);
  if (occupancy.host_id != inventory.host_id ||
      occupancy.boot_id != inventory.boot_id ||
      occupancy.inventory_digest != inventory.inventory_digest) {
    throw HostResourceError(
        "resource occupancy is not bound to the current host inventory");
  }
  validate_resource_fences(inventory, occupancy.active_fences);
}

HostInventoryReceipt capture_host_inventory(IHostKernel& kernel) {
  auto snapshot = kernel.capture_inventory();
  preflight_kernel_snapshot(snapshot);
  if (snapshot.api_version != kHostInventoryApiVersion ||
      snapshot.begin_revision != snapshot.end_revision) {
    throw HostResourceError(
        "kernel inventory snapshot is unsupported or torn");
  }
  require_identifier(snapshot.begin_revision, "kernel snapshot revision");
  std::ranges::sort(snapshot.probes, {}, &HostProbeResult::vendor);
  std::ranges::sort(snapshot.resources, resource_less);
  HostInventoryReceipt receipt{
      .api_version = std::move(snapshot.api_version),
      .host_id = std::move(snapshot.host_id),
      .boot_id = std::move(snapshot.boot_id),
      .broker_epoch = std::move(snapshot.broker_epoch),
      .snapshot_revision = std::move(snapshot.begin_revision),
      .probes = std::move(snapshot.probes),
      .resources = std::move(snapshot.resources),
      .topology_digest = {},
      .inventory_digest = {},
      .receipt_digest = {},
  };
  receipt.topology_digest =
      sha256("trainvm.host-topology/v2", topology_json(receipt.resources));
  receipt.inventory_digest = sha256("trainvm.host-inventory-content/v2",
                                    inventory_digest_json(receipt));
  receipt.receipt_digest = sha256("trainvm.host-inventory-receipt/v2",
                                  inventory_receipt_digest_json(receipt));
  validate_host_inventory(receipt);
  return receipt;
}

ResourceBundleRequest seal_resource_request(ResourceBundleRequest request) {
  preflight_request(request);
  if (request.topology == TopologyPolicy::exact_resources) {
    std::ranges::sort(request.selector.exact_resources, id_less);
  }
  request.canonical_request_digest =
      sha256("trainvm.host-resource-request/v1", request_digest_json(request));
  validate_resource_request(request);
  return request;
}

ResourceOccupancySnapshot seal_resource_occupancy(
    const HostInventoryReceipt& inventory,
    ResourceOccupancySnapshot occupancy) {
  validate_host_inventory(inventory);
  preflight_occupancy(occupancy);
  std::ranges::sort(occupancy.active_fences,
                    [](const auto& left, const auto& right) {
    return id_less(left.resource, right.resource);
  });
  occupancy.occupancy_digest = sha256(
      "trainvm.host-resource-occupancy/v1", occupancy_digest_json(occupancy));
  validate_resource_occupancy(inventory, occupancy);
  return occupancy;
}

std::optional<ResourceBundleSelection> select_host_resources(
    const HostInventoryReceipt& inventory,
    const ResourceBundleRequest& request,
    const ResourceOccupancySnapshot& occupancy) {
  validate_host_inventory(inventory);
  validate_resource_request(request);
  validate_resource_occupancy(inventory, occupancy);

  std::vector<ObservedHostResource> candidates;
  for (const auto& resource : inventory.resources) {
    if (resource.id.kind == kind_for(request.access_mode) &&
        disposition_allows(resource, request.access_mode) &&
        probe_allows_selection(inventory, resource) &&
        contexts_allow(resource, request.access_mode) &&
        selector_fields_match(resource, request.selector) &&
        !blocked_by_conflicting_observation(inventory, resource,
                                            request.access_mode) &&
        !blocked_by_fence(resource.id, occupancy.active_fences)) {
      candidates.push_back(resource);
    }
  }

  std::vector<ObservedHostResource> chosen;
  if (request.topology == TopologyPolicy::exact_resources) {
    for (const auto& exact : request.selector.exact_resources) {
      const auto found = std::ranges::find_if(
          candidates, [&](const auto& candidate) {
            return candidate.id.stable_id == exact.stable_id;
          });
      if (found == candidates.end() || found->id != exact) return std::nullopt;
      chosen.push_back(*found);
    }
  } else if (request.topology == TopologyPolicy::any) {
    if (candidates.size() < request.count) return std::nullopt;
    chosen.assign(candidates.begin(),
                  candidates.begin() + static_cast<std::ptrdiff_t>(request.count));
  } else {
    std::map<std::string, std::vector<ObservedHostResource>> groups;
    for (const auto& candidate : candidates) {
      const auto key = topology_group_key(candidate, request.topology);
      if (key) groups[*key].push_back(candidate);
    }
    std::optional<std::vector<ObservedHostResource>> best;
    for (const auto& [key, group] : groups) {
      (void)key;
      if (group.size() < request.count) continue;
      std::vector<ObservedHostResource> proposed(
          group.begin(),
          group.begin() + static_cast<std::ptrdiff_t>(request.count));
      if (!best || std::ranges::lexicographical_compare(
                       proposed, *best, resource_less)) {
        best = std::move(proposed);
      }
    }
    if (!best) return std::nullopt;
    chosen = std::move(*best);
  }

  std::ranges::sort(chosen, resource_less);
  ResourceBundleSelection selection{
      .api_version = std::string(kHostResourceSelectionApiVersion),
      .request_digest = request.canonical_request_digest,
      .host_id = inventory.host_id,
      .boot_id = inventory.boot_id,
      .snapshot_revision = inventory.snapshot_revision,
      .inventory_digest = inventory.inventory_digest,
      .topology_digest = inventory.topology_digest,
      .occupancy_digest = occupancy.occupancy_digest,
      .access_mode = request.access_mode,
      .resources = std::move(chosen),
      .selection_digest = {},
  };
  selection.selection_digest = sha256("trainvm.host-resource-selection/v2",
                                      selection_digest_json(selection));
  validate_resource_selection(selection);
  return selection;
}

BundleDegradationReport detect_bundle_degradation(
    const ResourceBundleSelection& selection,
    const HostInventoryReceipt& current_inventory) {
  validate_resource_selection(selection);
  validate_host_inventory(current_inventory);
  BundleDegradationReport report{
      .health = BundleHealth::intact,
      .host_or_boot_changed = selection.host_id != current_inventory.host_id ||
                              selection.boot_id != current_inventory.boot_id,
      .topology_changed =
          selection.topology_digest != current_inventory.topology_digest,
      .vanished_resources = {},
      .changed_parent_resources = {},
  };
  for (const auto& selected : selection.resources) {
    const auto current = std::ranges::find_if(
        current_inventory.resources, [&](const auto& resource) {
          return resource.id.stable_id == selected.id.stable_id;
        });
    if (current == current_inventory.resources.end()) {
      report.vanished_resources.push_back(selected.id.stable_id);
      continue;
    }
    if (current->id != selected.id) {
      report.changed_parent_resources.push_back(selected.id.stable_id);
    }
    if (!topology_facts_equal(selected, *current)) {
      report.topology_changed = true;
    }
  }
  if (report.host_or_boot_changed || report.topology_changed ||
      !report.vanished_resources.empty() ||
      !report.changed_parent_resources.empty()) {
    report.health = BundleHealth::degraded;
  }
  return report;
}

nlohmann::json host_inventory_json(const HostInventoryReceipt& receipt) {
  validate_host_inventory(receipt);
  return encode_json(receipt);
}

HostInventoryReceipt host_inventory_from_json(const nlohmann::json& source) {
  auto receipt = strict_decode<HostInventoryReceipt>(source, "host inventory");
  validate_host_inventory(receipt);
  return receipt;
}

nlohmann::json resource_request_json(const ResourceBundleRequest& request) {
  validate_resource_request(request);
  return encode_json(request);
}

ResourceBundleRequest resource_request_from_json(const nlohmann::json& source) {
  auto request = strict_decode<ResourceBundleRequest>(source, "resource request");
  validate_resource_request(request);
  return request;
}

nlohmann::json resource_occupancy_json(
    const ResourceOccupancySnapshot& occupancy) {
  validate_occupancy_intrinsic(occupancy);
  return encode_json(occupancy);
}

ResourceOccupancySnapshot resource_occupancy_from_json(
    const HostInventoryReceipt& inventory, const nlohmann::json& source) {
  auto occupancy =
      strict_decode<ResourceOccupancySnapshot>(source, "resource occupancy");
  validate_resource_occupancy(inventory, occupancy);
  return occupancy;
}

nlohmann::json resource_selection_json(
    const ResourceBundleSelection& selection) {
  validate_resource_selection(selection);
  return encode_json(selection);
}

ResourceBundleSelection resource_selection_from_json(
    const nlohmann::json& source) {
  auto selection =
      strict_decode<ResourceBundleSelection>(source, "resource selection");
  validate_resource_selection(selection);
  return selection;
}

FakeHostKernel::FakeHostKernel(std::vector<FakeHostKernelStep> script)
    : script_(std::move(script)) {
  if (script_.empty()) {
    throw std::invalid_argument("fake host kernel requires a nonempty script");
  }
  for (const auto& step : script_) {
    if (step.snapshot.has_value() == step.failure.has_value() ||
        (step.failure && step.failure->empty())) {
      throw std::invalid_argument(
          "fake host kernel steps require exactly one snapshot or failure");
    }
  }
}

HostKernelSnapshot FakeHostKernel::capture_inventory() {
  if (cursor_ >= script_.size()) {
    throw HostResourceError("unexpected fake host kernel capture");
  }
  const auto& step = script_[cursor_++];
  if (step.failure) throw HostResourceError(*step.failure);
  return *step.snapshot;
}

std::size_t FakeHostKernel::calls() const { return cursor_; }

std::size_t FakeHostKernel::remaining() const {
  return script_.size() - cursor_;
}

}  // namespace trainvm
