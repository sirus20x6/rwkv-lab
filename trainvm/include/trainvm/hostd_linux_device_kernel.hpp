#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/hostd_linux_cgroup_authority.hpp"
#include "trainvm/hostd_linux_device_policy.hpp"

namespace trainvm {

inline constexpr std::string_view kLinuxDevicePolicyInstallationApiVersion =
    "trainvm.linux-device-policy-installation/v1";

struct LinuxDeviceKernelProgramIdentity final {
  std::uint32_t program_id{};
  std::uint32_t program_type{};
  std::string program_tag;
  std::string program_name;

  bool operator==(const LinuxDeviceKernelProgramIdentity &) const = default;
};

struct LinuxDeviceKernelQuery final {
  std::uint32_t attach_flags{};
  std::vector<LinuxDeviceKernelProgramIdentity> programs;

  bool operator==(const LinuxDeviceKernelQuery &) const = default;
};

struct LinuxDevicePolicyInstallation final {
  std::string api_version;
  std::string allocation_id;
  std::string launch_id;
  std::string policy_digest;
  std::string image_digest;
  LinuxAllocationCgroupIdentity cgroup;
  LinuxDeviceKernelProgramIdentity program;
  std::uint32_t attach_flags{};
  std::string installation_digest;

  bool operator==(const LinuxDevicePolicyInstallation &) const = default;
};

class LinuxDeviceKernelError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class LinuxLoadedDeviceProgram final {
public:
  LinuxLoadedDeviceProgram(LinuxLoadedDeviceProgram &&other) noexcept;
  LinuxLoadedDeviceProgram &
  operator=(LinuxLoadedDeviceProgram &&other) noexcept;
  ~LinuxLoadedDeviceProgram();

  LinuxLoadedDeviceProgram(const LinuxLoadedDeviceProgram &) = delete;
  LinuxLoadedDeviceProgram &
  operator=(const LinuxLoadedDeviceProgram &) = delete;

  [[nodiscard]] const LinuxDeviceKernelProgramIdentity &identity() const;

private:
  friend class ILinuxDevicePolicyKernel;
  friend class LinuxCgroupDeviceKernel;
  friend class LinuxDevicePolicyInstaller;
  LinuxLoadedDeviceProgram(int descriptor,
                           LinuxDeviceKernelProgramIdentity identity) noexcept;
  void close() noexcept;

  int descriptor_{-1};
  LinuxDeviceKernelProgramIdentity identity_;
};

class ILinuxDevicePolicyKernel {
public:
  virtual ~ILinuxDevicePolicyKernel() = default;
  [[nodiscard]] virtual LinuxLoadedDeviceProgram
  load(const LinuxDeviceProgramImage &image, std::string_view program_name) = 0;
  [[nodiscard]] virtual LinuxDeviceKernelQuery query_local(int cgroup_fd) = 0;
  virtual void attach(int cgroup_fd,
                      const LinuxLoadedDeviceProgram &program) = 0;

protected:
  [[nodiscard]] static LinuxLoadedDeviceProgram
  adopt_loaded_program(int descriptor,
                       LinuxDeviceKernelProgramIdentity identity);
};

class LinuxCgroupDeviceKernel final : public ILinuxDevicePolicyKernel {
public:
  [[nodiscard]] LinuxLoadedDeviceProgram
  load(const LinuxDeviceProgramImage &image,
       std::string_view program_name) override;
  [[nodiscard]] LinuxDeviceKernelQuery query_local(int cgroup_fd) override;
  void attach(int cgroup_fd, const LinuxLoadedDeviceProgram &program) override;
};

class LinuxDevicePolicyInstaller final {
public:
  explicit LinuxDevicePolicyInstaller(ILinuxDevicePolicyKernel &kernel);

  [[nodiscard]] LinuxDevicePolicyInstallation
  install(const LinuxDevicePolicySpec &policy,
          const LinuxDeviceProgramImage &image,
          const LinuxAllocationCgroup &cgroup);
  [[nodiscard]] LinuxDeviceKernelQuery
  verify(const LinuxDevicePolicyInstallation &expected,
         const LinuxAllocationCgroup &cgroup);

private:
  ILinuxDevicePolicyKernel &kernel_;
};

void validate_linux_device_policy_installation(
    const LinuxDevicePolicyInstallation &installation);
[[nodiscard]] nlohmann::json linux_device_policy_installation_json(
    const LinuxDevicePolicyInstallation &installation);
[[nodiscard]] HostDevicePolicyIntentBinding
host_device_policy_intent_binding(const LinuxDevicePolicySpec &policy,
                                  const LinuxDeviceProgramImage &image);
[[nodiscard]] HostDevicePolicyInstallationBinding
host_device_policy_installation_binding(
    const LinuxDevicePolicyInstallation &installation);
[[nodiscard]] LinuxDevicePolicyInstallation
linux_device_policy_installation_from_process(
    const HostProcessLaunchIntent &intent,
    const HostProcessSpawnReceipt &spawn);

namespace hostd_linux_device_kernel_test_seam {

[[nodiscard]] std::string
program_name_for_image(const LinuxDeviceProgramImage &image);

} // namespace hostd_linux_device_kernel_test_seam

} // namespace trainvm
