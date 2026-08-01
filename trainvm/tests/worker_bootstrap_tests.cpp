#include "trainvm/worker_bootstrap.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

trainvm::WorkerBootstrapSpec fixture() {
  return trainvm::seal_worker_bootstrap({
      .api_version = std::string(trainvm::kWorkerBootstrapApiVersion),
      .controller_target = "unix:/run/user/1000/trainvm.sock",
      .run_id = "run-1",
      .node_id = "train",
      .attempt_id = "attempt-1",
      .launch_nonce = "launch-1",
      .adapter = "rwkv-lab.mageflow",
      .adapter_version = "1.0.0",
      .code_fingerprint = "sha256:" + std::string(64U, 'a'),
      .capabilities = {"artifact.manifest.v1", "metric.scalar.v1"},
      .last_acked_controller_sequence = 7U,
      .concurrency_key = "gpu:0",
      .lease_id = "lease-1",
      .fencing_token = 4U,
      .bootstrap_digest = {},
  });
}

void round_trip_is_exact() {
  const auto expected = fixture();
  const std::string encoded =
      trainvm::worker_bootstrap_canonical_json(expected);
  check(trainvm::worker_bootstrap_from_canonical_json(encoded) == expected,
        "worker bootstrap round trip is exact");
  check(expected.bootstrap_digest ==
            "sha256:55996b8d641668a7c0b989df4f94561e93109fe45f3cdd819e8e10fe763e106b",
        "C++ and Python bootstrap canonicalization share one golden digest");
  check(encoded.starts_with("{\"adapter\":"),
        "worker bootstrap uses canonical key ordering");
}

void changes_and_noncanonical_inputs_fail_closed() {
  const std::string encoded =
      trainvm::worker_bootstrap_canonical_json(fixture());
  auto tampered = nlohmann::json::parse(encoded);
  tampered["attempt_id"] = "attempt-2";
  bool rejected_tamper = false;
  try {
    (void)trainvm::worker_bootstrap_from_canonical_json(tampered.dump());
  } catch (const std::invalid_argument&) {
    rejected_tamper = true;
  }
  check(rejected_tamper, "worker bootstrap digest rejects changed identity");

  bool rejected_spacing = false;
  try {
    (void)trainvm::worker_bootstrap_from_canonical_json(
        nlohmann::json::parse(encoded).dump(2));
  } catch (const std::invalid_argument&) {
    rejected_spacing = true;
  }
  check(rejected_spacing, "noncanonical JSON is rejected");

  auto unsorted = fixture();
  unsorted.capabilities = {"metric.scalar.v1", "artifact.manifest.v1"};
  bool rejected_capabilities = false;
  try {
    (void)trainvm::seal_worker_bootstrap(std::move(unsorted));
  } catch (const std::invalid_argument&) {
    rejected_capabilities = true;
  }
  check(rejected_capabilities,
        "capabilities must be a sorted duplicate-free set");
}

void sealed_descriptor_round_trips_and_rejects_wrong_digest() {
  const auto expected = fixture();
  auto sealed = trainvm::create_sealed_worker_bootstrap(expected);
  const int descriptor = sealed.duplicate_fd();
  check(trainvm::worker_bootstrap_from_sealed_fd(
            descriptor, expected.bootstrap_digest) == expected,
        "sealed worker bootstrap descriptor round trips exactly");
  bool rejected = false;
  try {
    (void)trainvm::worker_bootstrap_from_sealed_fd(
        descriptor, "sha256:" + std::string(64U, 'f'));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "sealed worker bootstrap rejects a mismatched request digest");
  (void)::close(descriptor);
}

}  // namespace

int main() {
  round_trip_is_exact();
  changes_and_noncanonical_inputs_fail_closed();
  sealed_descriptor_round_trips_and_rejects_wrong_digest();
  if (failures != 0) return 1;
  std::cout << "worker bootstrap tests passed\n";
  return 0;
}
