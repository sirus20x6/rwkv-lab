#include "trainvm/hostd_linux_process_policy_kernel.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <fcntl.h>
#include <ranges>
#include <system_error>
#include <unistd.h>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

[[noreturn]] void reject(std::string message) {
  throw LinuxProcessPolicyKernelError(std::move(message));
}

std::string system_error(std::string_view operation) {
  return std::string(operation) + ": " +
         std::error_code(errno, std::generic_category()).message();
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool supported_control(std::string_view value) {
  return value == "cpuset.cpus" || value == "cpuset.mems" ||
         value == "cpuset.mems.effective" || value == "cpu.weight" ||
         value == "io.weight";
}

std::string trim_newline(std::string value) {
  if (!value.empty() && value.back() == '\n') value.pop_back();
  if (value.contains('\n') || value.contains('\r') || value.contains('\0'))
    reject("cgroup control contains unexpected line data");
  return value;
}

std::int64_t parse_weight(std::string_view value,
                          std::string_view prefix = {}) {
  if (!prefix.empty()) {
    if (!value.starts_with(prefix)) reject("cgroup weight prefix is inexact");
    value.remove_prefix(prefix.size());
  }
  std::int64_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size() || result < 1 ||
      result > 10'000) {
    reject("cgroup weight is malformed or out of range");
  }
  return result;
}

nlohmann::json installation_body(
    const LinuxProcessPolicyInstallation& value) {
  return {
      {"api_version", value.api_version},
      {"cgroup",
       {{"device", value.cgroup.device},
        {"inode", value.cgroup.inode},
        {"unified_path", value.cgroup.unified_path}}},
      {"cpuset", value.cpuset ? nlohmann::json(*value.cpuset)
                              : nlohmann::json(nullptr)},
      {"cpuset_mems", value.cpuset_mems ? nlohmann::json(*value.cpuset_mems)
                                        : nlohmann::json(nullptr)},
      {"cpu_weight", value.cpu_weight ? nlohmann::json(*value.cpu_weight)
                                      : nlohmann::json(nullptr)},
      {"io_weight", value.io_weight ? nlohmann::json(*value.io_weight)
                                    : nlohmann::json(nullptr)},
      {"policy_digest", value.policy_digest},
  };
}

std::string installation_digest(
    const LinuxProcessPolicyInstallation& value) {
  std::string material("trainvm.linux-process-policy-installation/v1");
  material.push_back('\0');
  material += installation_body(value).dump();
  return "sha256:" + sha256_hex(material);
}

std::string read_control(ILinuxProcessPolicyKernel& kernel, int descriptor,
                         std::string_view control) {
  return trim_newline(kernel.read(descriptor, control));
}

void require_identity(const LinuxProcessPolicyInstallation& installation,
                      const LinuxAllocationCgroup& cgroup) {
  if (installation.cgroup != cgroup.identity())
    reject("process policy installation changed cgroup identity");
}

}  // namespace

std::string LinuxCgroupProcessPolicyKernel::read(
    int cgroup_fd, std::string_view control) {
  if (cgroup_fd < 0 || !supported_control(control))
    reject("cgroup process-policy read is unauthorized");
  const int descriptor = ::openat(cgroup_fd, std::string(control).c_str(),
                                  O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) reject(system_error("could not open cgroup control"));
  std::string result;
  std::array<char, 512U> buffer{};
  while (result.size() <= 4096U) {
    const std::size_t wanted =
        std::min(buffer.size(), 4097U - result.size());
    const ssize_t count = ::read(descriptor, buffer.data(), wanted);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      const int saved = errno;
      (void)::close(descriptor);
      errno = saved;
      reject(system_error("could not read cgroup control"));
    }
    if (count == 0) break;
    result.append(buffer.data(), static_cast<std::size_t>(count));
  }
  const int saved = errno;
  (void)::close(descriptor);
  errno = saved;
  if (result.size() > 4096U) reject("cgroup control exceeds its audit bound");
  return result;
}

void LinuxCgroupProcessPolicyKernel::write(
    int cgroup_fd, std::string_view control, std::string_view value) {
  if (cgroup_fd < 0 || !supported_control(control) ||
      control == "cpuset.mems.effective" || value.empty() ||
      value.size() > 4096U || value.contains('\0') || value.contains('\n') ||
      value.contains('\r')) {
    reject("cgroup process-policy write is unauthorized or malformed");
  }
  const int descriptor = ::openat(cgroup_fd, std::string(control).c_str(),
                                  O_WRONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) reject(system_error("could not open cgroup control"));
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const ssize_t count =
        ::write(descriptor, value.data() + offset, value.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      const int saved = errno;
      (void)::close(descriptor);
      errno = saved;
      reject(system_error("could not write cgroup control"));
    }
    offset += static_cast<std::size_t>(count);
  }
  if (::close(descriptor) != 0)
    reject(system_error("could not close cgroup control"));
}

LinuxProcessPolicyInstaller::LinuxProcessPolicyInstaller(
    ILinuxProcessPolicyKernel& kernel)
    : kernel_(kernel) {}

LinuxProcessPolicyInstallation LinuxProcessPolicyInstaller::install(
    const LinuxProcessPolicy& policy,
    const LinuxAllocationCgroup& cgroup) {
  validate_linux_process_policy(policy);
  const int raw = cgroup.duplicate_fd();
  struct Close final {
    int descriptor;
    ~Close() { (void)::close(descriptor); }
  } close{raw};
  std::optional<std::string> mems;
  if (policy.cpuset) {
    mems = read_control(kernel_, raw, "cpuset.mems.effective");
    if (mems->empty()) reject("effective cpuset memory nodes are empty");
    kernel_.write(raw, "cpuset.mems", *mems);
    if (read_control(kernel_, raw, "cpuset.mems") != *mems)
      reject("cpuset memory-node installation is inexact");
    kernel_.write(raw, "cpuset.cpus", *policy.cpuset);
    if (read_control(kernel_, raw, "cpuset.cpus") != *policy.cpuset)
      reject("CPU affinity installation is inexact");
  }
  if (policy.cpu_weight) {
    kernel_.write(raw, "cpu.weight", std::to_string(*policy.cpu_weight));
    if (parse_weight(read_control(kernel_, raw, "cpu.weight")) !=
        *policy.cpu_weight)
      reject("CPU weight installation is inexact");
  }
  if (policy.io_weight) {
    const std::string expected = "default " + std::to_string(*policy.io_weight);
    kernel_.write(raw, "io.weight", expected);
    if (parse_weight(read_control(kernel_, raw, "io.weight"), "default ") !=
        *policy.io_weight)
      reject("I/O weight installation is inexact");
  }
  LinuxProcessPolicyInstallation result{
      .api_version = std::string(kLinuxProcessPolicyInstallationApiVersion),
      .policy_digest = policy.policy_digest,
      .cgroup = cgroup.identity(),
      .cpuset = policy.cpuset,
      .cpuset_mems = std::move(mems),
      .cpu_weight = policy.cpu_weight,
      .io_weight = policy.io_weight,
      .installation_digest = {},
  };
  result.installation_digest = installation_digest(result);
  validate_linux_process_policy_installation(result);
  return result;
}

void LinuxProcessPolicyInstaller::verify(
    const LinuxProcessPolicy& policy,
    const LinuxProcessPolicyInstallation& installation,
    const LinuxAllocationCgroup& cgroup) {
  validate_linux_process_policy(policy);
  validate_linux_process_policy_installation(installation);
  require_identity(installation, cgroup);
  if (installation.policy_digest != policy.policy_digest ||
      installation.cpuset != policy.cpuset ||
      installation.cpu_weight != policy.cpu_weight ||
      installation.io_weight != policy.io_weight ||
      installation.cpuset_mems.has_value() != policy.cpuset.has_value()) {
    reject("process policy installation disagrees with compiled intent");
  }
  const int raw = cgroup.duplicate_fd();
  struct Close final {
    int descriptor;
    ~Close() { (void)::close(descriptor); }
  } close{raw};
  if (policy.cpuset &&
      (read_control(kernel_, raw, "cpuset.cpus") != *policy.cpuset ||
       read_control(kernel_, raw, "cpuset.mems") !=
           *installation.cpuset_mems)) {
    reject("recovered cpuset controls drifted");
  }
  if (policy.cpu_weight &&
      parse_weight(read_control(kernel_, raw, "cpu.weight")) !=
          *policy.cpu_weight)
    reject("recovered CPU weight drifted");
  if (policy.io_weight &&
      parse_weight(read_control(kernel_, raw, "io.weight"), "default ") !=
          *policy.io_weight)
    reject("recovered I/O weight drifted");
}

void validate_linux_process_policy_installation(
    const LinuxProcessPolicyInstallation& value) {
  if (value.api_version != kLinuxProcessPolicyInstallationApiVersion ||
      !valid_digest(value.policy_digest) ||
      value.cgroup.unified_path.empty() ||
      value.cgroup.unified_path.front() != '/' || value.cgroup.device == 0U ||
      value.cgroup.inode == 0U ||
      value.cpuset.has_value() != value.cpuset_mems.has_value() ||
      (value.cpuset && (value.cpuset->empty() || value.cpuset_mems->empty())) ||
      !valid_digest(value.installation_digest) ||
      value.installation_digest != installation_digest(value)) {
    reject("process policy installation evidence is malformed");
  }
}

}  // namespace trainvm
