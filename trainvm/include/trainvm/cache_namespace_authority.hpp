#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/adapter_invocation.hpp"
#include "trainvm/cache_namespace.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_resources.hpp"

namespace trainvm {

// Adapter-owned compile facts that the control plane cannot infer from a plan
// alone. Every field is a content digest; paths, argv, environment, arbitrary
// compiler flags, and executable code are deliberately excluded.
struct CacheCompileInputManifest {
  std::string api_version;
  std::string invocation_digest;
  std::string model_topology_digest;
  std::string shape_set_digest;
  std::string dtype_precision_digest;
  std::string runtime_options_digest;
  std::string embedded_constants_digest;
  std::string compiler_configuration_digest;
  std::vector<std::string> checkpoint_fingerprints;

  bool operator==(const CacheCompileInputManifest&) const = default;
};

struct CacheRuntimeProbeContext {
  HostIdentity host;
  std::string launch_spec_digest;
  std::string inventory_receipt_digest;
  std::string resource_binding_digest;
  std::vector<ObservedHostResource> selected_resources;
  bool placement_specific{};

  bool operator==(const CacheRuntimeProbeContext&) const = default;
};

// One bounded runtime observation. Production implementations run inside the
// sealed launch/runtime boundary. Tests inject the same interface without
// turning a serialized snapshot into authority.
struct CacheRuntimeProbeSnapshot {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string launch_spec_digest;
  std::string inventory_receipt_digest;
  std::string resource_binding_digest;
  std::string compute_device_vendor;
  std::string compute_architecture;
  std::optional<std::string> compute_device_uuid;
  std::optional<std::string> compute_device_pci_address;
  std::string driver_version;
  std::vector<RuntimeVersionIdentity> runtime_versions;
  std::string runtime_closure_fingerprint;
  std::string host_abi_digest;
  std::string compute_compatibility_digest;

  bool operator==(const CacheRuntimeProbeSnapshot&) const = default;
};

class ICacheRuntimeProbe {
 public:
  virtual ~ICacheRuntimeProbe() = default;
  [[nodiscard]] virtual CacheRuntimeProbeSnapshot capture(
      const CacheRuntimeProbeContext& context) = 0;
};

// The accelerators a sealed launch actually holds, and the digest that binds
// them. Exposed so that everything which needs a probe context derives one the
// same way; a second implementation would name a different receipt and the
// resulting namespace would silently never be found.
struct CacheResourceBinding {
  std::vector<ObservedHostResource> devices;
  std::string binding_digest;

  bool operator==(const CacheResourceBinding&) const = default;
};

[[nodiscard]] CacheResourceBinding cache_resource_binding(
    const ResolvedLaunchSpec& launch, const HostInventoryReceipt& inventory);
// The same binding, taken over the grant-time projection of that receipt onto
// the rows the launch fenced. It reads only fenced rows either way, so the two
// overloads return equal bindings for the same launch -- which is what lets a
// controller that can only obtain the projection derive the same namespace an
// inventory holder would.
[[nodiscard]] CacheResourceBinding cache_resource_binding(
    const ResolvedLaunchSpec& launch,
    const GrantInventoryProjection& inventory);

// The authority's own answer to "which observation would this launch need".
// Every field is derived from host identity, the sealed launch, and the
// inventory receipt. Nothing a worker says can move it, which is what makes a
// worker-measured snapshot admissible without being authoritative.
[[nodiscard]] CacheRuntimeProbeContext cache_runtime_probe_context(
    const HostIdentity& host, const ResolvedLaunchSpec& launch,
    const HostInventoryReceipt& inventory, bool placement_specific);
[[nodiscard]] CacheRuntimeProbeContext cache_runtime_probe_context(
    const HostIdentity& host, const ResolvedLaunchSpec& launch,
    const GrantInventoryProjection& inventory, bool placement_specific);

// Shared semantic validator used both before immutable receipt publication and
// after a probe capture, so malformed authority observations cannot poison a
// no-replace receipt name.
void validate_cache_runtime_probe_snapshot(
    const CacheRuntimeProbeSnapshot& snapshot,
    const CacheRuntimeProbeContext& context);

struct CacheNamespaceAuthorityRequest {
  WorkerInvocationSpec invocation;
  ResolvedLaunchSpec launch;
  HostInventoryReceipt inventory;
  CacheCompileInputManifest compile_inputs;
  bool placement_specific{};
};

// Journalable evidence derived by the active authority. It is not itself a
// cache artifact or reuse grant: immutable publication, the live owner/fence,
// and qualification still gate adoption of bytes in this namespace.
struct CacheNamespaceAuthorityReceipt {
  std::string api_version;
  CacheNamespaceClaim cache_namespace;
  std::string adapter_registry_digest;
  std::string host_launch_registry_digest;
  std::string invocation_digest;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string launch_spec_digest;
  std::string inventory_receipt_digest;
  std::string resource_binding_digest;
  std::string runtime_probe_digest;
  std::string compile_inputs_digest;
  std::string receipt_digest;

  bool operator==(const CacheNamespaceAuthorityReceipt&) const = default;
};

class CacheNamespaceAuthorityError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class CacheNamespaceAuthority final {
 public:
  CacheNamespaceAuthority(const AdapterRegistry& adapters,
                          const HostLaunchRegistry& launches,
                          HostIdentity host, ICacheRuntimeProbe& runtime_probe);

  [[nodiscard]] CacheNamespaceAuthorityReceipt derive(
      const CacheNamespaceAuthorityRequest& request) const;

 private:
  const AdapterRegistry& adapters_;
  const HostLaunchRegistry& launches_;
  HostIdentity host_;
  ICacheRuntimeProbe& runtime_probe_;
};

[[nodiscard]] nlohmann::json cache_namespace_authority_receipt_json(
    const CacheNamespaceAuthorityReceipt& receipt);

}  // namespace trainvm
