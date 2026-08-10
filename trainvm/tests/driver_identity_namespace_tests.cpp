// The Python driver identity, judged by the C++ validator that decides it.
//
// `fixed_public_identity` in src/cache_namespace.cpp is the only statement of
// which characters a cache namespace claim may carry. `driver_identity()` in
// src/rwkv_lab/trainvm_runtime_guard.py has to satisfy it, and until PR #162 it
// did not -- the identity was the whole first line of
// /proc/driver/nvidia/version, which carries a space, `(`, `)` and `@`, so on
// any real NVIDIA host `validate_evidence` threw and no claim could be made.
//
// Nothing caught it because each half was tested against its own assumption
// about the other: cache_namespace_tests.cpp writes `.driver_version =
// "610.43.03"`, a value no Python code emits, and the Python tests asserted the
// producer's shape while guessing at the validator. Restating the character set
// in Python would only move the guess.
//
// So this test drives both sides over one input. It shells out to
// tests/driver_identity_fixture.py -- the same shape
// source_disposition_catalog_tests.cpp uses to reach
// scripts/print_disposition_digests.py, and for the same reason: the Python
// answer has to come from the Python code, not from a transcription of it. The
// script plants report text and returns what the real `driver_identity()` made
// of it; every one of those strings then goes through the real
// `derive_cache_namespace_claim`.
//
// Both assertions are load-bearing and they fail in opposite directions:
//
//   * every produced identity must be ACCEPTED -- narrowing the character set,
//     or changing the identity's shape to something illegal, fails here;
//   * every raw report line must be REJECTED -- widening the character set to
//     admit the space, parentheses and `@` that made the v3 identity illegal
//     fails here too, and that is the exact defect #162 found.
//
// No GPU, no NVIDIA host, no skip: the report text is planted by the fixture,
// so this says the same thing on a hosted runner as on the training host.

#include "trainvm/cache_namespace.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "trainvm/json.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

// Everything except the driver identity is a value this suite already trusts;
// only `driver_version` varies, so a failure names the identity and nothing
// else.
trainvm::CacheNamespaceClaimEvidence evidence_with(std::string driver_version) {
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
      .placement_specific = false,
      .compute_device_uuid = std::nullopt,
      .compute_device_pci_address = std::nullopt,
      .driver_version = std::move(driver_version),
      .runtime_versions = {{.name = "cuda", .version = "13.1"}},
      .compile_input_manifest_digest = "sha256:" + std::string(64U, '6'),
      .compiler_configuration_digest = "sha256:" + std::string(64U, '5'),
  };
}

bool claim_accepts(const std::string& driver_version) {
  try {
    const auto claim =
        trainvm::derive_cache_namespace_claim(evidence_with(driver_version));
    return claim.evidence.driver_version == driver_version;
  } catch (const trainvm::CacheNamespaceError&) {
    return false;
  }
}

nlohmann::json produced_identities(const std::filesystem::path& repository_root) {
  const std::string command =
      "python3 '" +
      (repository_root / "trainvm/tests/driver_identity_fixture.py").string() + "'";
  struct PipeCloser {
    void operator()(FILE* stream) const {
      if (stream != nullptr) ::pclose(stream);
    }
  };
  std::unique_ptr<FILE, PipeCloser> pipe(::popen(command.c_str(), "r"));
  if (!pipe) throw std::runtime_error("could not run the driver identity fixture");
  std::string output;
  std::array<char, 512> buffer{};
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) !=
         nullptr) {
    output.append(buffer.data());
  }
  const int status = ::pclose(pipe.release());
  if (status != 0) {
    throw std::runtime_error("the driver identity fixture exited " +
                             std::to_string(status));
  }
  return nlohmann::json::parse(output);
}

}  // namespace

int main() {
  try {
    const std::filesystem::path repository_root =
        std::filesystem::canonical(TRAINVM_SOURCE_ROOT);
    const nlohmann::json cases = produced_identities(repository_root);
    require(cases.is_array() && cases.size() >= 4U,
            "the driver identity fixture must produce every case it declares");

    // A fixture that stopped exercising the shape this host actually serves
    // would leave the agreement looking checked while checking nothing.
    bool saw_build_id = false;
    bool saw_bare_version = false;
    for (const auto& entry : cases) {
      const auto name = entry.at("case").get<std::string>();
      const auto identity = entry.at("identity").get<std::string>();
      const auto report = entry.at("report").get<std::string>();
      require(!identity.empty() && !report.empty(),
              "driver identity case " + name + " produced an empty string");
      saw_build_id = saw_build_id || identity.contains("+gnu-build-id:");
      saw_bare_version = saw_bare_version || !identity.contains("+");

      require(claim_accepts(identity),
              "the cache namespace claim rejected the driver identity Python "
              "produces for " +
                  name + ": " + identity);
      require(!claim_accepts(report),
              "the whole /proc/driver/nvidia/version line was accepted as a "
              "cache namespace identity for " +
                  name +
                  "; the character set must not admit the space, parentheses "
                  "and @ it carries: " +
                  report);
      require(identity != report,
              "the driver identity for " + name +
                  " is the raw report line, which is exactly the defect that "
                  "made the claim unmakeable");
    }
    require(saw_build_id,
            "no case exercised a version token plus a module build ID, the "
            "identity every NVIDIA host with a linked module produces");
    require(saw_bare_version,
            "no case exercised the version token alone, the identity a module "
            "linked with --build-id=none produces");

    std::cout << "driver identity namespace tests passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "driver identity namespace tests failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
