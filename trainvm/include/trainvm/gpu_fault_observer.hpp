#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <sys/types.h>

#include "trainvm/hostd.hpp"

namespace trainvm {

inline constexpr std::string_view kGpuFaultObserverStateApiVersion =
    "trainvm.gpu-fault-observer-state/v1";

struct NvidiaXidEvent final {
  std::uint32_t code{};
  std::string line_digest;

  bool operator==(const NvidiaXidEvent&) const = default;
};

struct GpuFaultObserverState final {
  std::string api_version;
  std::string boot_id;
  bool blocked{};
  std::uint64_t event_count{};
  std::uint32_t last_xid{};
  std::string last_event_digest;
  std::uint64_t observed_boottime_ns{};
  std::string state_digest;

  bool operator==(const GpuFaultObserverState&) const = default;
};

[[nodiscard]] std::optional<NvidiaXidEvent>
parse_nvidia_xid_line(std::string_view line);
[[nodiscard]] GpuFaultObserverState make_gpu_fault_observer_state(
    std::string boot_id, std::uint64_t observed_boottime_ns);
[[nodiscard]] GpuFaultObserverState update_gpu_fault_observer_state(
    GpuFaultObserverState state, std::uint64_t observed_boottime_ns,
    std::optional<NvidiaXidEvent> event = std::nullopt);
[[nodiscard]] std::string gpu_fault_observer_state_json(
    const GpuFaultObserverState& state);
[[nodiscard]] GpuFaultObserverState gpu_fault_observer_state_from_json(
    std::string_view source);
[[nodiscard]] GpuFaultObserverState read_gpu_fault_observer_state(
    const std::filesystem::path& path, uid_t expected_owner);
void write_gpu_fault_observer_state(
    const std::filesystem::path& path,
    const GpuFaultObserverState& state, uid_t expected_directory_owner);
[[nodiscard]] std::uint64_t linux_boottime_now_ns();

// A configured guard is deliberately fail-closed: missing, malformed, stale,
// wrong-boot, or fault-blocked observer evidence denies only new grants.
// Inspection, reconciliation, and release paths remain available.
class LinuxGpuFaultAdmissionGuard final : public IHostdGrantAdmissionGuard {
 public:
  LinuxGpuFaultAdmissionGuard(std::filesystem::path state_path,
                              std::string expected_boot_id,
                              std::uint64_t maximum_state_age_ns,
                              uid_t expected_owner = 0U);

  void require_new_grant_allowed() override;

 private:
  std::filesystem::path state_path_;
  std::string expected_boot_id_;
  std::uint64_t maximum_state_age_ns_{};
  uid_t expected_owner_{};
};

}  // namespace trainvm
