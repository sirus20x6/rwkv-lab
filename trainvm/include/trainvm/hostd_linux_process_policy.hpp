#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxProcessPolicyApiVersion =
    "trainvm.linux-process-policy/v1";

// Canonical cross-boundary form of the declarative CPU/I/O policy. CPU-list
// syntax is normalized once here; hostd and the worker never reinterpret the
// user's two accepted source representations independently.
struct LinuxProcessPolicy final {
  std::string api_version;
  std::optional<std::string> cpuset;
  std::optional<std::int64_t> cpu_weight;
  std::optional<std::int64_t> io_weight;
  std::optional<std::int64_t> omp_threads;
  std::optional<std::int64_t> preprocessing_workers;
  std::optional<std::int64_t> nice;
  std::string policy_digest;

  bool operator==(const LinuxProcessPolicy&) const = default;
};

class LinuxProcessPolicyError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] LinuxProcessPolicy compile_linux_process_policy(
    const std::optional<CpuIoPolicy>& policy);
void validate_linux_process_policy(const LinuxProcessPolicy& policy);
[[nodiscard]] nlohmann::json linux_process_policy_json(
    const LinuxProcessPolicy& policy);
[[nodiscard]] LinuxProcessPolicy linux_process_policy_from_json(
    const nlohmann::json& source);

}  // namespace trainvm
