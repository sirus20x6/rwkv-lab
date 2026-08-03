#include "trainvm/hostd_linux_process_policy.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Action>
void rejects(Action&& action, const std::string& message) {
  bool rejected = false;
  try {
    action();
  } catch (const trainvm::LinuxProcessPolicyError&) {
    rejected = true;
  }
  require(rejected, message);
}

}  // namespace

int main() {
  try {
    const trainvm::CpuIoPolicy source{
        .cpuset = std::nullopt,
        .cpus = std::vector<std::int64_t>{0, 1, 2, 5, 8, 9},
        .cpu_weight = 250,
        .io_weight = 90,
        .omp_threads = 4,
        .preprocessing_workers = 2,
        .nice = 3,
    };
    const auto compiled = trainvm::compile_linux_process_policy(source);
    require(compiled.cpuset == "0-2,5,8-9" &&
                compiled.policy_digest.starts_with("sha256:") &&
                trainvm::linux_process_policy_from_json(
                    trainvm::linux_process_policy_json(compiled)) == compiled,
            "process policy must normalize and round-trip exactly");

    const auto defaults =
        trainvm::compile_linux_process_policy(std::nullopt);
    require(!defaults.cpuset && !defaults.cpu_weight && !defaults.nice,
            "absent policy must compile to an explicit bounded default");

    rejects(
        [] {
          trainvm::CpuIoPolicy bad;
          bad.cpuset = "0-2,3";
          (void)trainvm::compile_linux_process_policy(bad);
        },
        "adjacent source ranges must be rejected before normalization");
    rejects(
        [] {
          trainvm::CpuIoPolicy bad;
          bad.cpus = std::vector<std::int64_t>{1, 0};
          (void)trainvm::compile_linux_process_policy(bad);
        },
        "structured CPU lists must be sorted");
    rejects(
        [&] {
          auto changed = trainvm::linux_process_policy_json(compiled);
          changed["cpu_weight"] = 251;
          (void)trainvm::linux_process_policy_from_json(changed);
        },
        "process policy content must be bound by its digest");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
