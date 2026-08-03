#include "trainvm/resource_request_builder.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using namespace trainvm;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

ResourceRequestBuildContext context() {
  return {
      .journal_id = "journal-builder",
      .plan_hash = "sha256:" + std::string(64U, 'a'),
      .run_id = "run-builder",
      .resources = {
          .accelerators = {.vendor = AcceleratorVendor::nvidia,
                           .count = 2,
                           .minimum_memory_gib = 15.5,
                           .exclusive = true,
                           .selector =
                               std::map<std::string, std::string>{{"tier",
                                                                   "training"}}},
          .minimum_host_memory_gib = std::nullopt,
          .cpu_threads = std::nullopt,
          .lease_timeout_seconds = 60,
          .cpu_io_policy = std::nullopt,
      },
      .lease = {.concurrency_key = "gpu:builder",
                .owner_run_id = "run-builder",
                .lease_id = "lease-builder",
                .fencing_token = 7U,
                .clock_domain = ResourceLease::kBootTimeDomain,
                .boot_id = "11111111-1111-1111-1111-111111111111",
                .acquired_boottime_ns = 100,
                .expires_boottime_ns = 200,
                .acquired_wall_time_ns = 1'000,
                .expires_wall_time_ns = 1'100},
  };
}

void test_deterministic_lowering() {
  const auto source = context();
  const auto first = build_resource_bundle_request(source);
  const auto second = build_resource_bundle_request(source);
  require(first == second && first.request_id.starts_with("host-request-") &&
              first.request_id.size() == 77U && first.count == 2U &&
              first.access_mode == ResourceAccessMode::exclusive_device &&
              first.topology == TopologyPolicy::any &&
              first.selector.vendor == HostAcceleratorVendor::nvidia &&
              first.selector.minimum_memory_bytes == 16'642'998'272ULL &&
              first.selector.exact_labels.at("tier") == "training" &&
              first.logical_fencing_token == 7U,
          "portable resources lower to one deterministic sealed host request");
  validate_resource_request(first);
}

void test_mode_and_bounds() {
  auto shared = context();
  shared.resources.accelerators.exclusive = false;
  require(build_resource_bundle_request(shared).access_mode ==
              ResourceAccessMode::exclusive_compute,
          "non-device-exclusive training still receives exclusive compute authority");

  auto zero = context();
  zero.resources.accelerators.vendor = AcceleratorVendor::none;
  zero.resources.accelerators.count = 0;
  auto oversized = context();
  oversized.resources.accelerators.count = 17;
  bool zero_rejected = false;
  bool oversized_rejected = false;
  try {
    (void)build_resource_bundle_request(zero);
  } catch (const ResourceRequestBuildError&) {
    zero_rejected = true;
  }
  try {
    (void)build_resource_bundle_request(oversized);
  } catch (const ResourceRequestBuildError&) {
    oversized_rejected = true;
  }
  require(zero_rejected && oversized_rejected,
          "unsupported CPU-only and oversized bundles fail closed");
}

}  // namespace

int main() {
  try {
    test_deterministic_lowering();
    test_mode_and_bounds();
    std::cout << "resource request builder tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "resource request builder test failure: " << error.what()
              << '\n';
    return 1;
  }
}
