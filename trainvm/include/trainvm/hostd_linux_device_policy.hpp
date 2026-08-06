#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/host_ledger.hpp"
#include "trainvm/host_resources.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxDevicePolicyApiVersion =
    "trainvm.linux-device-policy/v1";
inline constexpr std::string_view kLinuxDeviceProgramImageApiVersion =
    "trainvm.linux-device-program-image/v1";

enum class LinuxDevicePolicyRuleSource {
  runtime_baseline,
  inventory_capability,
};

struct LinuxDevicePolicyRule final {
  HostDeviceNodeType type{};
  std::uint32_t major{};
  std::uint32_t minor{};
  bool read{};
  bool write{};
  LinuxDevicePolicyRuleSource source{};

  bool operator==(const LinuxDevicePolicyRule &) const = default;
};

struct LinuxDevicePolicySpec final {
  std::string api_version;
  std::string allocation_id;
  std::string launch_id;
  std::string grant_digest;
  std::string inventory_digest;
  std::string topology_digest;
  LinuxAllocationCgroupIdentity cgroup;
  std::vector<LinuxDevicePolicyRule> rules;
  std::string policy_digest;

  bool operator==(const LinuxDevicePolicySpec &) const = default;
};

// Stable, kernel-header-independent representation of one eBPF instruction.
// The kernel adapter converts this to bpf_insn only at the syscall boundary.
struct LinuxBpfInstruction final {
  std::uint8_t code{};
  std::uint8_t destination_register{};
  std::uint8_t source_register{};
  std::int16_t offset{};
  std::int32_t immediate{};

  bool operator==(const LinuxBpfInstruction &) const = default;
};

struct LinuxDeviceProgramImage final {
  std::string api_version;
  std::string policy_digest;
  std::vector<LinuxBpfInstruction> instructions;
  std::string image_digest;

  bool operator==(const LinuxDeviceProgramImage &) const = default;
};

class LinuxDevicePolicyError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

[[nodiscard]] LinuxDevicePolicySpec derive_linux_device_policy(
    const HostInventoryReceipt &inventory, const ResourceBundleGrant &grant,
    const LinuxAllocationCgroupIdentity &cgroup, std::string launch_id);
[[nodiscard]] LinuxDeviceProgramImage
compile_linux_device_program(const LinuxDevicePolicySpec &policy);
void validate_linux_device_policy(const LinuxDevicePolicySpec &policy);
void validate_linux_device_program(const LinuxDeviceProgramImage &image);

[[nodiscard]] nlohmann::json
linux_device_policy_json(const LinuxDevicePolicySpec &policy);
[[nodiscard]] nlohmann::json
linux_device_program_image_json(const LinuxDeviceProgramImage &image);

} // namespace trainvm
