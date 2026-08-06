#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

struct RuntimeVersionIdentity {
  std::string name;
  std::string version;

  bool operator==(const RuntimeVersionIdentity&) const = default;
};

// Public, non-secret values claiming one compiler/JIT cache compatibility
// domain. This caller-constructible structure is syntax-checked only: its
// fields are not yet bound to authority-owned probes, registries, or receipts.
// Arbitrary environment maps are deliberately absent.
struct CacheNamespaceClaimEvidence {
  std::string api_version;
  AdapterKey adapter_key;
  std::string adapter_profile_digest;
  std::string code_fingerprint;
  std::string runtime_closure_fingerprint;
  std::string executable_fingerprint;
  std::string compute_device_vendor;
  std::string compute_architecture;
  std::string host_abi_digest;
  std::string compute_compatibility_digest;
  bool placement_specific{};
  std::optional<std::string> compute_device_uuid;
  std::optional<std::string> compute_device_pci_address;
  std::string driver_version;
  std::vector<RuntimeVersionIdentity> runtime_versions;
  std::string compile_input_manifest_digest;
  std::string compiler_configuration_digest;

  bool operator==(const CacheNamespaceClaimEvidence&) const = default;
};

// A canonical, untrusted claim. This format can be persisted and compared, but
// it does not authorize cache reuse. CacheNamespaceAuthority, kept in a
// separate module, can bind it to registries, sealed launch/invocation,
// inventory, a runtime probe, and compile inputs. Immutable cache publication,
// owner/fence checks, and qualification remain separate reuse gates.
struct CacheNamespaceClaim {
  std::string api_version;
  CacheNamespaceClaimEvidence evidence;
  std::string namespace_digest;

  bool operator==(const CacheNamespaceClaim&) const = default;
};

class CacheNamespaceError final : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

[[nodiscard]] CacheNamespaceClaim derive_cache_namespace_claim(
    CacheNamespaceClaimEvidence evidence);
[[nodiscard]] nlohmann::json cache_namespace_claim_json(
    const CacheNamespaceClaim& cache_namespace);
[[nodiscard]] CacheNamespaceClaim cache_namespace_claim_from_json(
    const nlohmann::json& source);

}  // namespace trainvm
