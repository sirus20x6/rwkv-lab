#include "trainvm/cache_namespace.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

trainvm::CacheNamespaceClaimEvidence fixture() {
  return {
      .api_version = "trainvm.cache-evidence-claim/v1",
      .adapter_key = {
          .adapter = "rwkv-lab.mageflow",
          .version = "1.0.0",
          .runtime = trainvm::ComponentRuntime::python_worker,
          .operation = "train",
          .contract = "rwkv_lab.mageflow.v1.Train",
      },
      .adapter_profile_digest = "sha256:" + std::string(64U, '1'),
      .code_fingerprint = "sha256:" + std::string(64U, '2'),
      .runtime_closure_fingerprint = "sha256:" + std::string(64U, '3'),
      .executable_fingerprint = "sha256:" + std::string(64U, '4'),
      .compute_device_vendor = "nvidia",
      .compute_architecture = "sm_120",
      .host_abi_digest = "sha256:" + std::string(64U, '7'),
      .compute_compatibility_digest = "sha256:" + std::string(64U, '8'),
      .placement_specific = true,
      .compute_device_uuid = "GPU-d6984164-b341-f8e8-519e-2243f11edc41",
      .compute_device_pci_address = "0000:01:00.0",
      .driver_version = "610.43.03",
      .runtime_versions = {
          {.name = "cuda", .version = "13.1"},
          {.name = "fa4", .version = "0.1.0"},
          {.name = "pytorch", .version = "2.10.0+cu130"},
          {.name = "triton", .version = "3.6.0"},
      },
      .compile_input_manifest_digest = "sha256:" + std::string(64U, '6'),
      .compiler_configuration_digest = "sha256:" + std::string(64U, '5'),
  };
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callable>
void rejected(Callable&& callable, const std::string& message) {
  try {
    callable();
  } catch (const trainvm::CacheNamespaceError&) {
    return;
  }
  throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const auto first = trainvm::derive_cache_namespace_claim(fixture());
    const auto second = trainvm::derive_cache_namespace_claim(fixture());
    const auto encoded = trainvm::cache_namespace_claim_json(first);
    require(first == second && first.namespace_digest.starts_with("sha256:") &&
                trainvm::cache_namespace_claim_from_json(encoded) == first,
            "cache evidence must derive one canonical round-trippable claim");
    require(first.namespace_digest ==
                "sha256:fa43e20146c0f5ee70e97d4ddce51bd6402d916da04a59a84b10be3567cef53b",
            "canonical cache claim digest changed without an API migration");

    auto changed = fixture();
    changed.runtime_versions[1].version = "0.1.1";
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "every compiler/runtime identity field must change the cache namespace");
    changed = fixture();
    changed.compile_input_manifest_digest =
        "sha256:" + std::string(64U, '7');
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "graph, shape, precision, and embedded-constant inputs must partition cache namespaces");
    changed = fixture();
    changed.compute_device_pci_address = "0000:02:00.0";
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "GPU topology identity must partition cache namespaces");
    changed = fixture();
    changed.compute_device_vendor = "amd";
    changed.compute_device_uuid = "AMD-d6984164-b341-f8e8-519e-2243f11edc41";
    changed.compute_architecture = "gfx1201";
    changed.runtime_versions = {
        {.name = "hip", .version = "7.0"},
        {.name = "pytorch", .version = "2.10.0+rocm70"},
        {.name = "triton", .version = "3.6.0"},
    };
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "cache evidence must support and isolate non-NVIDIA accelerator stacks");
    changed = fixture();
    changed.compute_device_vendor = "amd";
    changed.placement_specific = false;
    changed.compute_device_uuid.reset();
    changed.compute_device_pci_address.reset();
    changed.compute_architecture = "znver3";
    changed.driver_version = "linux-6.18.0";
    changed.runtime_versions = {
        {.name = "gcc", .version = "16.0.1"},
        {.name = "pytorch", .version = "2.10.0"},
    };
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "cache evidence must support and isolate CPU compile stacks");
    changed = fixture();
    changed.adapter_key.runtime = trainvm::ComponentRuntime::external_worker;
    require(trainvm::derive_cache_namespace_claim(changed).namespace_digest !=
                first.namespace_digest,
            "external trainer adapters must have isolated cache claims");

    auto forged = encoded;
    forged["evidence"]["runtime_versions"][0]["version"] = "13.2";
    rejected([&] { (void)trainvm::cache_namespace_claim_from_json(forged); },
             "self-inconsistent cache manifests must fail closed");
    forged = encoded;
    forged["extra"] = true;
    rejected([&] { (void)trainvm::cache_namespace_claim_from_json(forged); },
             "unknown cache manifest fields must fail closed");
    forged = encoded;
    forged["evidence"]["adapter_key"] = nullptr;
    rejected([&] { (void)trainvm::cache_namespace_claim_from_json(forged); },
             "invalid nested field types must fail closed");
    forged = encoded;
    forged["evidence"]["driver_version"] = std::string(70'000U, 'x');
    rejected([&] { (void)trainvm::cache_namespace_claim_from_json(forged); },
             "oversized cache claims must fail before reflected decoding");
    auto invalid_claim = first;
    invalid_claim.evidence.driver_version = "610.44.00";
    rejected([&] { (void)trainvm::cache_namespace_claim_json(invalid_claim); },
             "serialization must reject a manually forged claim");

    auto invalid = fixture();
    invalid.compute_device_uuid = "secret://inventory/device#1";
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "explicit secret-reference syntax must fail in public device identity");
    invalid = fixture();
    invalid.compute_device_pci_address = "1:0:0";
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "malformed PCI identity must fail closed");
    invalid = fixture();
    invalid.placement_specific = false;
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "portable claims cannot retain placement identity");
    invalid = fixture();
    invalid.runtime_versions[1].version = "secret://vault/fa4#1";
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "explicit secret-reference syntax must fail in public runtime identity");
    invalid = fixture();
    invalid.code_fingerprint = "sha256:not-a-digest";
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "malformed fingerprint must fail closed");
    invalid = fixture();
    std::swap(invalid.runtime_versions[0], invalid.runtime_versions[1]);
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "unsorted runtime identity closure must fail closed");
    invalid = fixture();
    invalid.runtime_versions[1].name = invalid.runtime_versions[0].name;
    rejected([&] { (void)trainvm::derive_cache_namespace_claim(invalid); },
             "duplicate runtime identities must fail closed");
  } catch (const std::exception& exception) {
    std::cerr << "cache_namespace_tests: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
