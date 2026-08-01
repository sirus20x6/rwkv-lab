#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_process_policy.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxProcessPolicyInstallationApiVersion =
    "trainvm.linux-process-policy-installation/v1";

struct LinuxProcessPolicyInstallation final {
  std::string api_version;
  std::string policy_digest;
  LinuxAllocationCgroupIdentity cgroup;
  std::optional<std::string> cpuset;
  std::optional<std::string> cpuset_mems;
  std::optional<std::int64_t> cpu_weight;
  std::optional<std::int64_t> io_weight;
  std::string installation_digest;

  bool operator==(const LinuxProcessPolicyInstallation&) const = default;
};

class LinuxProcessPolicyKernelError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ILinuxProcessPolicyKernel {
 public:
  virtual ~ILinuxProcessPolicyKernel() = default;
  [[nodiscard]] virtual std::string read(int cgroup_fd,
                                         std::string_view control) = 0;
  virtual void write(int cgroup_fd, std::string_view control,
                     std::string_view value) = 0;
};

class LinuxCgroupProcessPolicyKernel final
    : public ILinuxProcessPolicyKernel {
 public:
  [[nodiscard]] std::string read(int cgroup_fd,
                                 std::string_view control) override;
  void write(int cgroup_fd, std::string_view control,
             std::string_view value) override;
};

class LinuxProcessPolicyInstaller final {
 public:
  explicit LinuxProcessPolicyInstaller(ILinuxProcessPolicyKernel& kernel);

  [[nodiscard]] LinuxProcessPolicyInstallation install(
      const LinuxProcessPolicy& policy,
      const LinuxAllocationCgroup& cgroup);
  void verify(const LinuxProcessPolicy& policy,
              const LinuxProcessPolicyInstallation& installation,
              const LinuxAllocationCgroup& cgroup);

 private:
  ILinuxProcessPolicyKernel& kernel_;
};

void validate_linux_process_policy_installation(
    const LinuxProcessPolicyInstallation& installation);

}  // namespace trainvm
