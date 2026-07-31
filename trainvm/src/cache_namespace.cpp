#include "trainvm/cache_namespace.hpp"

#include <algorithm>
#include <ranges>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumIdentityBytes = 256U;

bool valid_sha256(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool fixed_public_identity(std::string_view value) {
  if (value.empty() || value.size() > kMaximumIdentityBytes ||
      value.find('\0') != std::string_view::npos ||
      value.contains("secret://") || value.contains("${") ||
      value.contains("{{")) {
    return false;
  }
  return std::ranges::all_of(value, [](unsigned char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-' || character == '+' ||
           character == ':';
  });
}

bool lower_hex(char value) {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
}

bool canonical_pci_address(std::string_view value) {
  if (value.size() != 12U || value[4U] != ':' || value[7U] != ':' ||
      value[10U] != '.') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == 4U || index == 7U || index == 10U) continue;
    if (!lower_hex(value[index])) return false;
  }
  return value[11U] >= '0' && value[11U] <= '7';
}

void validate_key(const AdapterKey& key) {
  if (!fixed_public_identity(key.adapter) ||
      !fixed_public_identity(key.version) ||
      !fixed_public_identity(key.operation) ||
      !fixed_public_identity(key.contract) ||
      (key.runtime != ComponentRuntime::python_worker &&
       key.runtime != ComponentRuntime::native_worker &&
       key.runtime != ComponentRuntime::external_worker)) {
    throw CacheNamespaceError(
        "cache namespace adapter key is malformed or unsupported");
  }
}

void validate_evidence(const CacheNamespaceClaimEvidence& evidence) {
  validate_key(evidence.adapter_key);
  if (evidence.runtime_versions.empty() ||
      evidence.runtime_versions.size() > 64U) {
    throw CacheNamespaceError(
        "cache namespace runtime identity closure is empty or unbounded");
  }
  std::string_view previous_name;
  for (const RuntimeVersionIdentity& identity : evidence.runtime_versions) {
    if (!fixed_public_identity(identity.name) ||
        !fixed_public_identity(identity.version) ||
        (!previous_name.empty() && identity.name <= previous_name)) {
      throw CacheNamespaceError(
          "cache namespace runtime identities must be canonical, unique, and sorted");
    }
    previous_name = identity.name;
  }
  if (evidence.api_version != "trainvm.cache-evidence-claim/v1" ||
      !valid_sha256(evidence.adapter_profile_digest) ||
      !valid_sha256(evidence.code_fingerprint) ||
      !valid_sha256(evidence.runtime_closure_fingerprint) ||
      !valid_sha256(evidence.executable_fingerprint) ||
      !valid_sha256(evidence.host_abi_digest) ||
      !valid_sha256(evidence.compute_compatibility_digest) ||
      !valid_sha256(evidence.compile_input_manifest_digest) ||
      !valid_sha256(evidence.compiler_configuration_digest) ||
      !fixed_public_identity(evidence.compute_device_vendor) ||
      !fixed_public_identity(evidence.compute_architecture) ||
      (evidence.placement_specific !=
       evidence.compute_device_uuid.has_value()) ||
      (evidence.compute_device_uuid &&
       !fixed_public_identity(*evidence.compute_device_uuid)) ||
      (!evidence.placement_specific &&
       evidence.compute_device_pci_address.has_value()) ||
      (evidence.compute_device_pci_address &&
       !canonical_pci_address(*evidence.compute_device_pci_address)) ||
      !fixed_public_identity(evidence.driver_version)) {
    throw CacheNamespaceError(
        "cache namespace evidence is noncanonical, unbounded, or incomplete");
  }
}

std::string namespace_digest(const CacheNamespaceClaimEvidence& evidence) {
  return "sha256:" +
         sha256_hex(nlohmann::json{
                        {"domain", "trainvm.cache-namespace-claim/v1"},
                        {"evidence", encode_json(evidence)},
                    }
                        .dump());
}

}  // namespace

CacheNamespaceClaim derive_cache_namespace_claim(
    CacheNamespaceClaimEvidence evidence) {
  validate_evidence(evidence);
  const std::string digest = namespace_digest(evidence);
  return {
      .api_version = "trainvm.cache-namespace-claim/v1",
      .evidence = std::move(evidence),
      .namespace_digest = digest,
  };
}

nlohmann::json cache_namespace_claim_json(
    const CacheNamespaceClaim& cache_namespace) {
  validate_evidence(cache_namespace.evidence);
  if (cache_namespace.api_version != "trainvm.cache-namespace-claim/v1" ||
      !valid_sha256(cache_namespace.namespace_digest) ||
      cache_namespace.namespace_digest !=
          namespace_digest(cache_namespace.evidence)) {
    throw CacheNamespaceError(
        "cannot serialize an invalid cache namespace claim");
  }
  return encode_json(cache_namespace);
}

CacheNamespaceClaim cache_namespace_claim_from_json(
    const nlohmann::json& source) {
  if (source.dump().size() > 65'536U) {
    throw CacheNamespaceError("cache namespace claim exceeds 65536 bytes");
  }
  CacheNamespaceClaim decoded;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, decoded, "", diagnostics)) {
    throw CacheNamespaceError(
        "cache namespace schema validation failed: " +
        diagnostics_json(diagnostics).dump());
  }
  validate_evidence(decoded.evidence);
  if (decoded.api_version != "trainvm.cache-namespace-claim/v1" ||
      !valid_sha256(decoded.namespace_digest) ||
      decoded.namespace_digest != namespace_digest(decoded.evidence)) {
    throw CacheNamespaceError(
        "cache namespace digest or API version is invalid");
  }
  return decoded;
}

}  // namespace trainvm
