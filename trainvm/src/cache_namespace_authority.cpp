#include "trainvm/cache_namespace_authority.hpp"

#include <algorithm>
#include <ranges>
#include <set>
#include <string_view>
#include <utility>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

bool valid_sha256(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string digest(std::string_view domain, const nlohmann::json& value) {
  return "sha256:" +
         sha256_hex(nlohmann::json{{"domain", domain}, {"value", value}}.dump());
}

void validate_compile_inputs(const CacheCompileInputManifest& inputs,
                             const WorkerInvocationSpec& invocation) {
  if (inputs.api_version != "trainvm.cache-compile-inputs/v1" ||
      inputs.invocation_digest != invocation.invocation_digest ||
      !valid_sha256(inputs.invocation_digest) ||
      !valid_sha256(inputs.model_topology_digest) ||
      !valid_sha256(inputs.shape_set_digest) ||
      !valid_sha256(inputs.dtype_precision_digest) ||
      !valid_sha256(inputs.runtime_options_digest) ||
      !valid_sha256(inputs.embedded_constants_digest) ||
      !valid_sha256(inputs.compiler_configuration_digest) ||
      inputs.checkpoint_fingerprints.size() > 256U ||
      std::ranges::any_of(inputs.checkpoint_fingerprints,
                          [](const std::string& value) {
                            return !valid_sha256(value);
                          }) ||
      !std::ranges::is_sorted(inputs.checkpoint_fingerprints) ||
      std::ranges::adjacent_find(inputs.checkpoint_fingerprints) !=
          inputs.checkpoint_fingerprints.end()) {
    throw CacheNamespaceAuthorityError(
        "cache compile-input manifest is unbound, malformed, or noncanonical");
  }
}

std::string compile_inputs_digest(const CacheCompileInputManifest& inputs,
                                  const WorkerInvocationSpec& invocation) {
  const nlohmann::json invocation_context{
      {"adapter", encode_json(invocation.adapter)},
      {"controls", invocation.controls},
      {"execution", invocation.execution},
      {"inputs", invocation.inputs},
      {"training", invocation.training},
  };
  return digest("trainvm.cache-compile-inputs-authority/v1",
                nlohmann::json{{"adapter_manifest", encode_json(inputs)},
                               {"invocation_context", invocation_context}});
}

const ObservedHostResource* find_resource(
    const HostInventoryReceipt& inventory, const HostResourceId& id) {
  const auto found = std::ranges::find_if(
      inventory.resources,
      [&](const ObservedHostResource& resource) { return resource.id == id; });
  return found == inventory.resources.end() ? nullptr : &*found;
}

struct ResourceBinding {
  std::vector<ObservedHostResource> devices;
  std::string digest;
};

ResourceBinding bind_resources(const ResolvedLaunchSpec& launch,
                               const HostInventoryReceipt& inventory) {
  nlohmann::json fences = nlohmann::json::array();
  nlohmann::json resources = nlohmann::json::array();
  std::vector<ObservedHostResource> devices;
  std::set<std::string> identities;
  if (launch.identity.host_grant) {
    const HostLaunchGrantClaim& grant = *launch.identity.host_grant;
    if (grant.request_id.empty() || !valid_sha256(grant.grant_digest) ||
        grant.fences.empty() || grant.fences.size() > 256U) {
      throw CacheNamespaceAuthorityError(
          "cache namespace launch grant is malformed or unbounded");
    }
    for (const ResourceFence& fence : grant.fences) {
      if (fence.generation == 0U ||
          fence.inventory_digest != inventory.inventory_digest ||
          fence.topology_digest != inventory.topology_digest ||
          !identities.insert(encode_json(fence.resource).dump()).second) {
        throw CacheNamespaceAuthorityError(
            "cache namespace resource fence disagrees with inventory authority");
      }
      const ObservedHostResource* resource =
          find_resource(inventory, fence.resource);
      if (resource == nullptr) {
        throw CacheNamespaceAuthorityError(
            "cache namespace resource fence is absent from inventory");
      }
      fences.push_back(encode_json(fence));
      resources.push_back(encode_json(*resource));
      if (resource->id.kind == HostResourceKind::accelerator ||
          resource->id.kind == HostResourceKind::accelerator_partition) {
        devices.push_back(*resource);
      }
    }
  }
  return {
      .devices = std::move(devices),
      .digest = digest(
          "trainvm.cache-resource-binding/v1",
          nlohmann::json{{"fences", std::move(fences)},
                         {"inventory_receipt_digest", inventory.receipt_digest},
                         {"resources", std::move(resources)}}),
  };
}

std::string vendor_name(HostAcceleratorVendor vendor) {
  return encode_json(vendor).get<std::string>();
}

void validate_runtime_probe_snapshot_impl(
    const CacheRuntimeProbeSnapshot& probe,
    const CacheRuntimeProbeContext& context) {
  if (probe.api_version != "trainvm.cache-runtime-probe/v1" ||
      probe.host_id != context.host.host_id ||
      probe.boot_id != context.host.boot_id ||
      probe.launch_spec_digest != context.launch_spec_digest ||
      probe.inventory_receipt_digest != context.inventory_receipt_digest ||
      probe.resource_binding_digest != context.resource_binding_digest ||
      !valid_sha256(probe.runtime_closure_fingerprint) ||
      !valid_sha256(probe.host_abi_digest) ||
      !valid_sha256(probe.compute_compatibility_digest) ||
      probe.compute_device_vendor.empty() ||
      probe.compute_device_vendor.size() > 64U ||
      probe.compute_architecture.empty() ||
      probe.compute_architecture.size() > 256U || probe.driver_version.empty() ||
      probe.driver_version.size() > 256U) {
    throw CacheNamespaceAuthorityError(
        "cache runtime probe is not bound to the active authority context");
  }
  if (context.selected_resources.empty()) {
    if (probe.compute_device_vendor != "cpu") {
      throw CacheNamespaceAuthorityError(
          "cache CPU runtime probe claims an accelerator vendor");
    }
  } else {
    const auto vendor = context.selected_resources.front().id.vendor;
    if (!vendor || probe.compute_device_vendor != vendor_name(*vendor) ||
        std::ranges::any_of(
            context.selected_resources,
            [&](const ObservedHostResource& resource) {
              return resource.id.vendor != vendor;
            })) {
      throw CacheNamespaceAuthorityError(
          "cache runtime vendor disagrees with selected resources");
    }
  }
  if (context.placement_specific) {
    if (context.selected_resources.size() != 1U ||
        !probe.compute_device_uuid ||
        *probe.compute_device_uuid !=
            context.selected_resources.front().id.stable_id ||
        probe.compute_device_pci_address !=
            context.selected_resources.front().pci_bdf) {
      throw CacheNamespaceAuthorityError(
          "placement-specific cache probe disagrees with its exact device");
    }
  } else if (probe.compute_device_uuid ||
             probe.compute_device_pci_address) {
    throw CacheNamespaceAuthorityError(
        "portable cache probe retains placement-specific identity");
  }
}

nlohmann::json authority_receipt_body(
    const CacheNamespaceAuthorityReceipt& receipt) {
  return {
      {"adapter_registry_digest", receipt.adapter_registry_digest},
      {"api_version", receipt.api_version},
      {"cache_namespace", cache_namespace_claim_json(receipt.cache_namespace)},
      {"compile_inputs_digest", receipt.compile_inputs_digest},
      {"host_launch_registry_digest", receipt.host_launch_registry_digest},
      {"inventory_receipt_digest", receipt.inventory_receipt_digest},
      {"invocation_digest", receipt.invocation_digest},
      {"run_id", receipt.run_id},
      {"node_id", receipt.node_id},
      {"attempt_id", receipt.attempt_id},
      {"concurrency_key", receipt.concurrency_key},
      {"lease_id", receipt.lease_id},
      {"fencing_token", receipt.fencing_token},
      {"launch_spec_digest", receipt.launch_spec_digest},
      {"resource_binding_digest", receipt.resource_binding_digest},
      {"runtime_probe_digest", receipt.runtime_probe_digest},
  };
}

std::string authority_receipt_digest(
    const CacheNamespaceAuthorityReceipt& receipt) {
  return digest("trainvm.cache-namespace-authority-receipt/v1",
                authority_receipt_body(receipt));
}

void validate_authority_receipt(
    const CacheNamespaceAuthorityReceipt& receipt) {
  (void)cache_namespace_claim_json(receipt.cache_namespace);
  if (receipt.api_version != "trainvm.cache-namespace-authority/v1" ||
      !valid_sha256(receipt.adapter_registry_digest) ||
      !valid_sha256(receipt.host_launch_registry_digest) ||
      !valid_sha256(receipt.invocation_digest) ||
      receipt.run_id.empty() || receipt.run_id.size() > 1024U ||
      receipt.node_id.empty() || receipt.node_id.size() > 1024U ||
      receipt.attempt_id.empty() || receipt.attempt_id.size() > 1024U ||
      receipt.concurrency_key.empty() ||
      receipt.concurrency_key.size() > 1024U || receipt.lease_id.empty() ||
      receipt.lease_id.size() > 1024U || receipt.fencing_token == 0U ||
      !valid_sha256(receipt.launch_spec_digest) ||
      !valid_sha256(receipt.inventory_receipt_digest) ||
      !valid_sha256(receipt.resource_binding_digest) ||
      !valid_sha256(receipt.runtime_probe_digest) ||
      !valid_sha256(receipt.compile_inputs_digest) ||
      receipt.receipt_digest != authority_receipt_digest(receipt)) {
    throw CacheNamespaceAuthorityError(
        "cache namespace authority receipt is self-inconsistent");
  }
}

}  // namespace

void validate_cache_runtime_probe_snapshot(
    const CacheRuntimeProbeSnapshot& snapshot,
    const CacheRuntimeProbeContext& context) {
  validate_runtime_probe_snapshot_impl(snapshot, context);
}

CacheNamespaceAuthority::CacheNamespaceAuthority(
    const AdapterRegistry& adapters, const HostLaunchRegistry& launches,
    HostIdentity host, ICacheRuntimeProbe& runtime_probe)
    : adapters_(adapters),
      launches_(launches),
      host_(std::move(host)),
      runtime_probe_(runtime_probe) {
  if (!valid_sha256(host_.host_id) || host_.boot_id.empty() ||
      host_.boot_id.size() > 256U) {
    throw CacheNamespaceAuthorityError(
        "cache namespace authority host identity is invalid");
  }
}

CacheNamespaceAuthorityReceipt CacheNamespaceAuthority::derive(
    const CacheNamespaceAuthorityRequest& request) const {
  WorkerInvocationSpec invocation;
  ResolvedLaunchSpec launch;
  HostInventoryReceipt inventory;
  try {
    invocation = worker_invocation_from_canonical_json(
        worker_invocation_canonical_json(request.invocation));
    launch = resolved_launch_spec_from_json(
        resolved_launch_spec_json(request.launch));
    inventory = host_inventory_from_json(host_inventory_json(request.inventory));
  } catch (const std::exception& exception) {
    throw CacheNamespaceAuthorityError(
        "cache namespace input authority receipt failed validation: " +
        std::string(exception.what()));
  }
  const AdapterProfile& adapter = adapters_.resolve(invocation.adapter);
  if (!adapter.lifecycle.compile || adapter.code_fingerprint.empty() ||
      launch.identity.adapter_key != invocation.adapter ||
      launch.identity.code_fingerprint != adapter.code_fingerprint ||
      launch.identity.run_id != invocation.run_id ||
      launch.identity.node_id != invocation.node_id ||
      launch.identity.attempt_id != invocation.attempt_id ||
      launch.identity.host != host_ || invocation.host_id != host_.host_id ||
      inventory.host_id != host_.host_id || inventory.boot_id != host_.boot_id) {
    throw CacheNamespaceAuthorityError(
        "cache namespace invocation, launch, registry, or host binding disagrees");
  }
  const HostLaunchProfile& host_profile =
      launches_.resolve(invocation.adapter, adapter.code_fingerprint);
  const std::string host_profile_digest =
      launches_.profile_digest(invocation.adapter, adapter.code_fingerprint);
  if (launch.identity.host_registry_digest != launches_.registry_digest() ||
      launch.identity.host_profile_digest != host_profile_digest ||
      launch.identity.bootstrap_runtime_closure_fingerprint !=
          host_profile.bootstrap_runtime_closure_fingerprint ||
      launch.identity.executable.sealed_sha256 !=
          host_profile.executable_fingerprint ||
      (launch.identity.code &&
       launch.identity.code->sealed_sha256 != adapter.code_fingerprint)) {
    throw CacheNamespaceAuthorityError(
        "cache namespace sealed launch differs from active host authority");
  }
  if (!invocation.execution.is_object() ||
      !invocation.execution.contains("compile") ||
      !invocation.execution.at("compile").is_object() ||
      !invocation.execution.at("compile").value("enabled", false)) {
    throw CacheNamespaceAuthorityError(
        "cache namespace requires an authority-declared compile phase");
  }
  validate_compile_inputs(request.compile_inputs, invocation);
  const std::string compile_digest =
      compile_inputs_digest(request.compile_inputs, invocation);
  ResourceBinding resources = bind_resources(launch, inventory);
  std::size_t required_accelerators = 0U;
  std::string required_vendor;
  if (invocation.resources.is_object() &&
      invocation.resources.contains("accelerators")) {
    const auto& accelerators = invocation.resources.at("accelerators");
    if (!accelerators.is_object() || !accelerators.contains("count") ||
        !accelerators.at("count").is_number_unsigned() ||
        !accelerators.contains("vendor") ||
        !accelerators.at("vendor").is_string()) {
      throw CacheNamespaceAuthorityError(
          "cache namespace invocation accelerator authority is malformed");
    }
    required_accelerators = accelerators.at("count").get<std::size_t>();
    required_vendor = accelerators.at("vendor").get<std::string>();
  }
  if (required_accelerators != resources.devices.size() ||
      (required_accelerators != 0U && !launch.identity.host_grant) ||
      std::ranges::any_of(
          resources.devices, [&](const ObservedHostResource& resource) {
            return !resource.id.vendor ||
                   vendor_name(*resource.id.vendor) != required_vendor;
          })) {
    throw CacheNamespaceAuthorityError(
        "cache namespace selected devices disagree with invocation resources");
  }
  CacheRuntimeProbeContext probe_context{
      .host = host_,
      .launch_spec_digest = launch.spec_digest,
      .inventory_receipt_digest = inventory.receipt_digest,
      .resource_binding_digest = resources.digest,
      .selected_resources = resources.devices,
      .placement_specific = request.placement_specific,
  };
  CacheRuntimeProbeSnapshot probe = runtime_probe_.capture(probe_context);
  validate_cache_runtime_probe_snapshot(probe, probe_context);
  if (probe.runtime_closure_fingerprint !=
      launch.identity.bootstrap_runtime_closure_fingerprint) {
    throw CacheNamespaceAuthorityError(
        "cache runtime probe closure differs from the sealed launch authority");
  }
  const std::string probe_digest =
      digest("trainvm.cache-runtime-probe-receipt/v1", encode_json(probe));

  CacheNamespaceClaimEvidence evidence{
      .api_version = "trainvm.cache-evidence-claim/v1",
      .adapter_key = invocation.adapter,
      .adapter_profile_digest = adapters_.profile_digest(invocation.adapter),
      .code_fingerprint = adapter.code_fingerprint,
      .runtime_closure_fingerprint = probe.runtime_closure_fingerprint,
      .executable_fingerprint = launch.identity.executable.sealed_sha256,
      .compute_device_vendor = probe.compute_device_vendor,
      .compute_architecture = probe.compute_architecture,
      .host_abi_digest = probe.host_abi_digest,
      .compute_compatibility_digest = probe.compute_compatibility_digest,
      .placement_specific = request.placement_specific,
      .compute_device_uuid = probe.compute_device_uuid,
      .compute_device_pci_address = probe.compute_device_pci_address,
      .driver_version = probe.driver_version,
      .runtime_versions = std::move(probe.runtime_versions),
      .compile_input_manifest_digest = compile_digest,
      .compiler_configuration_digest =
          request.compile_inputs.compiler_configuration_digest,
  };
  CacheNamespaceAuthorityReceipt receipt{
      .api_version = "trainvm.cache-namespace-authority/v1",
      .cache_namespace = derive_cache_namespace_claim(std::move(evidence)),
      .adapter_registry_digest = adapters_.registry_digest(),
      .host_launch_registry_digest = launches_.registry_digest(),
      .invocation_digest = invocation.invocation_digest,
      .run_id = invocation.run_id,
      .node_id = invocation.node_id,
      .attempt_id = invocation.attempt_id,
      .concurrency_key = launch.identity.concurrency_key,
      .lease_id = launch.identity.lease_id,
      .fencing_token = launch.identity.fencing_token,
      .launch_spec_digest = launch.spec_digest,
      .inventory_receipt_digest = inventory.receipt_digest,
      .resource_binding_digest = resources.digest,
      .runtime_probe_digest = probe_digest,
      .compile_inputs_digest = compile_digest,
      .receipt_digest = {},
  };
  receipt.receipt_digest = authority_receipt_digest(receipt);
  validate_authority_receipt(receipt);
  return receipt;
}

nlohmann::json cache_namespace_authority_receipt_json(
    const CacheNamespaceAuthorityReceipt& receipt) {
  validate_authority_receipt(receipt);
  nlohmann::json result = authority_receipt_body(receipt);
  result["receipt_digest"] = receipt.receipt_digest;
  return result;
}

}  // namespace trainvm
