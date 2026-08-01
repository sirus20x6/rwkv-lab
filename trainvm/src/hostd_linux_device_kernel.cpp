#include "trainvm/hostd_linux_device_kernel.hpp"

#include <linux/bpf.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <unistd.h>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumAttachedPrograms = 64U;
constexpr std::size_t kVerifierLogBytes = 1U << 20U;

[[noreturn]] void reject(std::string message) {
  throw LinuxDeviceKernelError(std::move(message));
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

bool canonical_absolute_path(std::string_view value) {
  if (value.empty() || value.front() != '/' || value.size() > 4096U ||
      value.find('\0') != std::string_view::npos ||
      value.find("//") != std::string_view::npos) {
    return false;
  }
  std::size_t begin = 1U;
  while (begin < value.size()) {
    const std::size_t end = value.find('/', begin);
    const auto component =
        value.substr(begin, end == std::string_view::npos ? value.size() - begin
                                                          : end - begin);
    if (component.empty() || component == "." || component == "..") {
      return false;
    }
    if (end == std::string_view::npos)
      break;
    begin = end + 1U;
  }
  return value == "/" || value.back() != '/';
}

bool lower_hex(std::string_view value) {
  return std::ranges::all_of(value, [](char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

std::string digest(std::string_view domain, const nlohmann::json &value) {
  std::string material(domain);
  material.push_back('\0');
  material += value.dump();
  return "sha256:" + sha256_hex(material);
}

std::string bytes_hex(const std::uint8_t *bytes, std::size_t count) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string result;
  result.reserve(count * 2U);
  for (std::size_t index = 0U; index < count; ++index) {
    result.push_back(digits[bytes[index] >> 4U]);
    result.push_back(digits[bytes[index] & 0x0fU]);
  }
  return result;
}

int bpf_call(enum bpf_cmd command, union bpf_attr &attributes) {
  return static_cast<int>(
      ::syscall(SYS_bpf, command, &attributes, sizeof(attributes)));
}

LinuxDeviceKernelProgramIdentity inspect_program_fd(int descriptor) {
  struct bpf_prog_info information{};
  union bpf_attr attributes{};
  attributes.info.bpf_fd = static_cast<std::uint32_t>(descriptor);
  attributes.info.info_len = sizeof(information);
  attributes.info.info = reinterpret_cast<std::uint64_t>(&information);
  if (bpf_call(BPF_OBJ_GET_INFO_BY_FD, attributes) != 0) {
    reject(system_error("could not inspect cgroup-device eBPF program"));
  }
  const auto name_length =
      ::strnlen(information.name, static_cast<std::size_t>(BPF_OBJ_NAME_LEN));
  return {.program_id = information.id,
          .program_type = information.type,
          .program_tag = bytes_hex(information.tag, BPF_TAG_SIZE),
          .program_name = std::string(information.name, name_length)};
}

LinuxDeviceKernelProgramIdentity inspect_program_id(std::uint32_t program_id) {
  union bpf_attr attributes{};
  attributes.prog_id = program_id;
  const int descriptor = bpf_call(BPF_PROG_GET_FD_BY_ID, attributes);
  if (descriptor < 0) {
    reject(system_error("could not reopen attached cgroup-device program"));
  }
  try {
    auto identity = inspect_program_fd(descriptor);
    if (identity.program_id != program_id) {
      reject("reopened cgroup-device program changed kernel identity");
    }
    (void)::close(descriptor);
    return identity;
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

nlohmann::json
installation_digest_json(const LinuxDevicePolicyInstallation &installation) {
  return {
      {"api_version", installation.api_version},
      {"allocation_id", installation.allocation_id},
      {"launch_id", installation.launch_id},
      {"policy_digest", installation.policy_digest},
      {"image_digest", installation.image_digest},
      {"cgroup",
       {{"unified_path", installation.cgroup.unified_path},
        {"device", installation.cgroup.device},
        {"inode", installation.cgroup.inode}}},
      {"program",
       {{"program_id", installation.program.program_id},
        {"program_type", installation.program.program_type},
        {"program_tag", installation.program.program_tag},
        {"program_name", installation.program.program_name}}},
      {"attach_flags", installation.attach_flags},
  };
}

void require_exact_image(const LinuxDevicePolicySpec &policy,
                         const LinuxDeviceProgramImage &image) {
  const auto expected = compile_linux_device_program(policy);
  if (image != expected) {
    reject("device program is not the exact compiler output for its policy");
  }
}

void require_exact_program(const LinuxDeviceKernelQuery &observed,
                           const LinuxDeviceKernelProgramIdentity &expected) {
  if (observed.attach_flags != 0U || observed.programs.size() != 1U ||
      observed.programs.front() != expected) {
    reject("allocation cgroup does not contain its one exact device program");
  }
}

} // namespace

LinuxLoadedDeviceProgram::LinuxLoadedDeviceProgram(
    int descriptor, LinuxDeviceKernelProgramIdentity identity) noexcept
    : descriptor_(descriptor), identity_(std::move(identity)) {}

LinuxLoadedDeviceProgram::LinuxLoadedDeviceProgram(
    LinuxLoadedDeviceProgram &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)),
      identity_(std::move(other.identity_)) {}

LinuxLoadedDeviceProgram &
LinuxLoadedDeviceProgram::operator=(LinuxLoadedDeviceProgram &&other) noexcept {
  if (this != &other) {
    close();
    descriptor_ = std::exchange(other.descriptor_, -1);
    identity_ = std::move(other.identity_);
  }
  return *this;
}

LinuxLoadedDeviceProgram::~LinuxLoadedDeviceProgram() { close(); }

const LinuxDeviceKernelProgramIdentity &
LinuxLoadedDeviceProgram::identity() const {
  return identity_;
}

void LinuxLoadedDeviceProgram::close() noexcept {
  if (descriptor_ >= 0) {
    (void)::close(descriptor_);
    descriptor_ = -1;
  }
}

LinuxLoadedDeviceProgram ILinuxDevicePolicyKernel::adopt_loaded_program(
    int descriptor, LinuxDeviceKernelProgramIdentity identity) {
  return LinuxLoadedDeviceProgram(descriptor, std::move(identity));
}

LinuxLoadedDeviceProgram
LinuxCgroupDeviceKernel::load(const LinuxDeviceProgramImage &image,
                              std::string_view program_name) {
  validate_linux_device_program(image);
  if (program_name.empty() || program_name.size() >= BPF_OBJ_NAME_LEN ||
      !std::ranges::all_of(program_name, [](char character) {
        return (character >= 'a' && character <= 'z') ||
               (character >= 'A' && character <= 'Z') ||
               (character >= '0' && character <= '9') || character == '_';
      })) {
    reject("cgroup-device program name is not kernel-canonical");
  }
  if (image.instructions.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    reject("cgroup-device program instruction count exceeds kernel range");
  }

  std::vector<struct bpf_insn> instructions;
  instructions.reserve(image.instructions.size());
  for (const auto &source : image.instructions) {
    if (source.destination_register > BPF_REG_10 ||
        source.source_register > BPF_REG_10) {
      reject("cgroup-device program names an invalid eBPF register");
    }
    struct bpf_insn target{};
    target.code = source.code;
    target.dst_reg = source.destination_register & 0x0fU;
    target.src_reg = source.source_register & 0x0fU;
    target.off = source.offset;
    target.imm = source.immediate;
    instructions.push_back(target);
  }

  static constexpr char license[] = "GPL";
  std::vector<char> verifier_log(kVerifierLogBytes, '\0');
  union bpf_attr attributes{};
  attributes.prog_type = BPF_PROG_TYPE_CGROUP_DEVICE;
  attributes.insn_cnt = static_cast<std::uint32_t>(instructions.size());
  attributes.insns = reinterpret_cast<std::uint64_t>(instructions.data());
  attributes.license = reinterpret_cast<std::uint64_t>(license);
  attributes.log_level = 1U;
  attributes.log_size = static_cast<std::uint32_t>(verifier_log.size());
  attributes.log_buf = reinterpret_cast<std::uint64_t>(verifier_log.data());
  attributes.expected_attach_type = BPF_CGROUP_DEVICE;
  std::ranges::copy(program_name, attributes.prog_name);
  const int descriptor = bpf_call(BPF_PROG_LOAD, attributes);
  if (descriptor < 0) {
    const std::size_t log_length = ::strnlen(verifier_log.data(), 4096U);
    reject(system_error("could not load cgroup-device eBPF program") +
           (log_length == 0U
                ? std::string{}
                : ": " + std::string(verifier_log.data(), log_length)));
  }
  try {
    auto identity = inspect_program_fd(descriptor);
    if (identity.program_id == 0U ||
        identity.program_type != BPF_PROG_TYPE_CGROUP_DEVICE ||
        identity.program_tag.size() != BPF_TAG_SIZE * 2U ||
        identity.program_name != program_name) {
      reject("loaded cgroup-device program identity is incomplete");
    }
    return adopt_loaded_program(descriptor, std::move(identity));
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

LinuxDeviceKernelQuery LinuxCgroupDeviceKernel::query_local(int cgroup_fd) {
  if (cgroup_fd < 0)
    reject("cgroup-device query received an invalid cgroup");
  std::array<std::uint32_t, kMaximumAttachedPrograms> program_ids{};
  union bpf_attr attributes{};
  attributes.query.target_fd = cgroup_fd;
  attributes.query.attach_type = BPF_CGROUP_DEVICE;
  attributes.query.query_flags = 0U;
  attributes.query.prog_cnt = static_cast<std::uint32_t>(program_ids.size());
  attributes.query.prog_ids =
      reinterpret_cast<std::uint64_t>(program_ids.data());
  if (bpf_call(BPF_PROG_QUERY, attributes) != 0) {
    reject(system_error("could not query local cgroup-device programs"));
  }
  if (attributes.query.prog_cnt > program_ids.size()) {
    reject("local cgroup-device program count exceeds its audit bound");
  }
  LinuxDeviceKernelQuery result{.attach_flags = attributes.query.attach_flags,
                                .programs = {}};
  result.programs.reserve(attributes.query.prog_cnt);
  for (std::size_t index = 0U; index < attributes.query.prog_cnt; ++index) {
    result.programs.push_back(inspect_program_id(program_ids[index]));
  }
  return result;
}

void LinuxCgroupDeviceKernel::attach(int cgroup_fd,
                                     const LinuxLoadedDeviceProgram &program) {
  if (cgroup_fd < 0 || program.descriptor_ < 0 ||
      program.identity_.program_type != BPF_PROG_TYPE_CGROUP_DEVICE) {
    reject("cgroup-device attachment received invalid authority");
  }
  union bpf_attr attributes{};
  attributes.target_fd = cgroup_fd;
  attributes.attach_bpf_fd = static_cast<std::uint32_t>(program.descriptor_);
  attributes.attach_type = BPF_CGROUP_DEVICE;
  attributes.attach_flags = 0U;
  if (bpf_call(BPF_PROG_ATTACH, attributes) != 0) {
    reject(system_error("could not attach cgroup-device eBPF program"));
  }
}

LinuxDevicePolicyInstaller::LinuxDevicePolicyInstaller(
    ILinuxDevicePolicyKernel &kernel)
    : kernel_(kernel) {}

LinuxDevicePolicyInstallation
LinuxDevicePolicyInstaller::install(const LinuxDevicePolicySpec &policy,
                                    const LinuxDeviceProgramImage &image,
                                    const LinuxAllocationCgroup &cgroup) {
  validate_linux_device_policy(policy);
  require_exact_image(policy, image);
  if (policy.cgroup != cgroup.identity()) {
    reject("device policy does not bind the open allocation cgroup");
  }
  const int descriptor = cgroup.duplicate_fd();
  if (descriptor < 0)
    reject("could not duplicate allocation cgroup authority");
  try {
    const LinuxDeviceKernelQuery before = kernel_.query_local(descriptor);
    if (before.attach_flags != 0U || !before.programs.empty()) {
      reject("allocation cgroup already has local device-program authority");
    }
    const std::string name =
        hostd_linux_device_kernel_test_seam::program_name_for_image(image);
    auto loaded = kernel_.load(image, name);
    if (loaded.identity().program_type != BPF_PROG_TYPE_CGROUP_DEVICE ||
        loaded.identity().program_name != name) {
      reject("kernel loaded a different cgroup-device program identity");
    }
    kernel_.attach(descriptor, loaded);
    const LinuxDeviceKernelQuery after = kernel_.query_local(descriptor);
    require_exact_program(after, loaded.identity());
    LinuxDevicePolicyInstallation installation{
        .api_version = std::string(kLinuxDevicePolicyInstallationApiVersion),
        .allocation_id = policy.allocation_id,
        .launch_id = policy.launch_id,
        .policy_digest = policy.policy_digest,
        .image_digest = image.image_digest,
        .cgroup = policy.cgroup,
        .program = loaded.identity(),
        .attach_flags = after.attach_flags,
        .installation_digest = {},
    };
    installation.installation_digest =
        digest("trainvm.linux-device-policy-installation/v1",
               installation_digest_json(installation));
    validate_linux_device_policy_installation(installation);
    (void)::close(descriptor);
    return installation;
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

LinuxDeviceKernelQuery LinuxDevicePolicyInstaller::verify(
    const LinuxDevicePolicyInstallation &expected,
    const LinuxAllocationCgroup &cgroup) {
  validate_linux_device_policy_installation(expected);
  if (expected.cgroup != cgroup.identity()) {
    reject("device installation does not bind the open allocation cgroup");
  }
  const int descriptor = cgroup.duplicate_fd();
  if (descriptor < 0)
    reject("could not duplicate allocation cgroup authority");
  try {
    auto observed = kernel_.query_local(descriptor);
    require_exact_program(observed, expected.program);
    (void)::close(descriptor);
    return observed;
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

void validate_linux_device_policy_installation(
    const LinuxDevicePolicyInstallation &installation) {
  if (installation.api_version != kLinuxDevicePolicyInstallationApiVersion ||
      installation.allocation_id.empty() || installation.launch_id.empty() ||
      !valid_digest(installation.policy_digest) ||
      !valid_digest(installation.image_digest) ||
      !canonical_absolute_path(installation.cgroup.unified_path) ||
      installation.cgroup.device == 0U || installation.cgroup.inode == 0U ||
      installation.program.program_id == 0U ||
      installation.program.program_type != BPF_PROG_TYPE_CGROUP_DEVICE ||
      installation.program.program_tag.size() != BPF_TAG_SIZE * 2U ||
      !lower_hex(installation.program.program_tag) ||
      installation.program.program_name.empty() ||
      installation.program.program_name.size() >= BPF_OBJ_NAME_LEN ||
      installation.program.program_name !=
          "tvmdev_" + installation.image_digest.substr(7U, 8U) ||
      installation.attach_flags != 0U ||
      installation.installation_digest !=
          digest("trainvm.linux-device-policy-installation/v1",
                 installation_digest_json(installation))) {
    reject("device policy installation receipt is invalid");
  }
}

nlohmann::json linux_device_policy_installation_json(
    const LinuxDevicePolicyInstallation &installation) {
  validate_linux_device_policy_installation(installation);
  nlohmann::json value = installation_digest_json(installation);
  value["installation_digest"] = installation.installation_digest;
  return value;
}

namespace hostd_linux_device_kernel_test_seam {

std::string program_name_for_image(const LinuxDeviceProgramImage &image) {
  validate_linux_device_program(image);
  return "tvmdev_" + image.image_digest.substr(7U, 8U);
}

} // namespace hostd_linux_device_kernel_test_seam

} // namespace trainvm
