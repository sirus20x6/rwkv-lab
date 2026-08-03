#include "trainvm/hostd_linux_device_kernel.hpp"

#include <linux/bpf.h>

#include <fcntl.h>
#include <unistd.h>

#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

constexpr std::string_view kGpu = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

std::string digest(std::string_view domain, const nlohmann::json &value) {
  std::string material(domain);
  material.push_back('\0');
  material += value.dump();
  return "sha256:" + sha256_hex(material);
}

HostResourceId gpu_id() {
  return {.kind = HostResourceKind::accelerator,
          .vendor = HostAcceleratorVendor::nvidia,
          .stable_id = std::string(kGpu),
          .parent_id = std::nullopt};
}

HostInventoryReceipt inventory() {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-device-kernel",
      .boot_id = "boot-device-kernel",
      .broker_epoch = "broker-device-kernel",
      .begin_revision = "revision-device-kernel",
      .end_revision = "revision-device-kernel",
      .probes = {{.vendor = HostAcceleratorVendor::nvidia,
                  .disposition = ProbeDisposition::complete,
                  .context_details_complete = true,
                  .detail = "test"}},
      .resources =
          {{.id = gpu_id(),
            .disposition = ResourceObservationDisposition::audited_eligible,
            .compute_contexts = ResourceContextDisposition::absent,
            .graphics_contexts = ResourceContextDisposition::absent,
            .pci_bdf = "0000:01:00.0",
            .device_major = 195U,
            .device_minor = 0U,
            .device_nodes =
                {{.type = HostDeviceNodeType::character,
                  .purpose = HostDeviceNodePurpose::assigned_accelerator,
                  .major = 195U,
                  .minor = 0U,
                  .read = true,
                  .write = true},
                 {.type = HostDeviceNodeType::character,
                  .purpose = HostDeviceNodePurpose::shared_driver_control,
                  .major = 195U,
                  .minor = 255U,
                  .read = true,
                  .write = true}},
            .numa_node = 0,
            .pcie_root_id = std::nullopt,
            .fabric_clique_id = std::nullopt,
            .total_memory_bytes = 24ULL << 30U,
            .labels = {{"backend", "test"}}}},
  };
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResourceBundleGrant grant(const HostInventoryReceipt &observed) {
  ResourceBundleGrant value{
      .api_version = std::string(kHostLedgerGrantApiVersion),
      .allocation_id = "allocation-device-kernel",
      .request_id = "request-device-kernel",
      .request_digest = "sha256:" + std::string(64U, '1'),
      .journal_id = "journal-device-kernel",
      .run_id = "run-device-kernel",
      .logical_lease_id = "lease-device-kernel",
      .logical_fencing_token = 11U,
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .fences = {{.resource = gpu_id(),
                  .generation = 1U,
                  .inventory_digest = observed.inventory_digest,
                  .topology_digest = observed.topology_digest}},
      .granted_boottime_ns = 10,
      .granted_wall_time_ns = 20,
      .previous_receipt_digest = "sha256:" + std::string(64U, '0'),
      .receipt_digest = {},
  };
  auto canonical = encode_json(value);
  canonical.erase("receipt_digest");
  value.receipt_digest = digest("trainvm.host-resource-grant/v1", canonical);
  (void)resource_bundle_grant_json(value);
  return value;
}

LinuxAllocationCgroupIdentity cgroup_identity() {
  return {.unified_path = "/trainvm/allocation-device-kernel",
          .device = 31U,
          .inode = 41U};
}

LinuxAllocationCgroup fake_cgroup() {
  const int descriptor = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    throw std::runtime_error("could not open test fd");
  return LinuxAllocationCgroup(cgroup_identity(), descriptor, -1, {}, false);
}

std::pair<LinuxDevicePolicySpec, LinuxDeviceProgramImage> policy_image() {
  const auto observed = inventory();
  auto policy = derive_linux_device_policy(
      observed, grant(observed), cgroup_identity(), "launch-device-kernel");
  auto image = compile_linux_device_program(policy);
  return {std::move(policy), std::move(image)};
}

class FakeDeviceKernel final : public ILinuxDevicePolicyKernel {
public:
  LinuxLoadedDeviceProgram load(const LinuxDeviceProgramImage &image,
                                std::string_view program_name) override {
    ++load_calls;
    loaded_image_digest = image.image_digest;
    loaded_identity = {.program_id = 73U,
                       .program_type = BPF_PROG_TYPE_CGROUP_DEVICE,
                       .program_tag = "0011223344556677",
                       .program_name = std::string(program_name)};
    return adopt_loaded_program(-1, loaded_identity);
  }

  LinuxDeviceKernelQuery query_local(int cgroup_fd) override {
    require(cgroup_fd >= 0, "installer passes a live duplicate cgroup fd");
    ++query_calls;
    if (query_calls == 1U && preexisting) {
      return {.attach_flags = 0U, .programs = {*preexisting}};
    }
    if (!attached)
      return {};
    auto identity = loaded_identity;
    if (corrupt_post_attach)
      ++identity.program_id;
    return {.attach_flags = post_attach_flags, .programs = {identity}};
  }

  void attach(int cgroup_fd, const LinuxLoadedDeviceProgram &program) override {
    require(cgroup_fd >= 0 && program.identity() == loaded_identity,
            "installer attaches only the just-loaded program");
    ++attach_calls;
    attached = true;
  }

  std::optional<LinuxDeviceKernelProgramIdentity> preexisting;
  bool corrupt_post_attach{};
  bool attached{};
  std::uint32_t post_attach_flags{};
  std::size_t load_calls{};
  std::size_t query_calls{};
  std::size_t attach_calls{};
  std::string loaded_image_digest;
  LinuxDeviceKernelProgramIdentity loaded_identity;
};

void exact_installation_is_sealed_and_restart_verifiable() {
  auto [policy, image] = policy_image();
  auto cgroup = fake_cgroup();
  FakeDeviceKernel kernel;
  LinuxDevicePolicyInstaller installer(kernel);
  const auto installed = installer.install(policy, image, cgroup);
  const auto intent_binding = host_device_policy_intent_binding(policy, image);
  const auto installation_binding =
      host_device_policy_installation_binding(installed);
  require(kernel.load_calls == 1U && kernel.attach_calls == 1U &&
              kernel.query_calls == 2U &&
              kernel.loaded_image_digest == image.image_digest &&
              installed.program == kernel.loaded_identity &&
              installed.policy_digest == policy.policy_digest &&
              installed.image_digest == image.image_digest &&
              intent_binding.policy_digest == policy.policy_digest &&
              intent_binding.image_digest == image.image_digest &&
              intent_binding.program_name == installed.program.program_name &&
              installation_binding.installation_digest ==
                  installed.installation_digest &&
              linux_device_policy_installation_json(installed).at(
                  "installation_digest") == installed.installation_digest,
          "installation binds compiler output and exact kernel identity");
  const auto verified = installer.verify(installed, cgroup);
  require(kernel.query_calls == 3U && verified.programs.size() == 1U &&
              verified.programs.front() == installed.program,
          "restart verification requires the one exact attached program");
}

void preexisting_or_changed_programs_fail_closed() {
  auto [policy, image] = policy_image();
  auto cgroup = fake_cgroup();
  FakeDeviceKernel occupied;
  occupied.preexisting = {.program_id = 19U,
                          .program_type = BPF_PROG_TYPE_CGROUP_DEVICE,
                          .program_tag = "ffeeddccbbaa9988",
                          .program_name = "unknown"};
  LinuxDevicePolicyInstaller occupied_installer(occupied);
  require_throws<LinuxDeviceKernelError>(
      [&] { (void)occupied_installer.install(policy, image, cgroup); },
      "an unknown local program cannot be replaced or composed");
  require(occupied.load_calls == 0U && occupied.attach_calls == 0U,
          "occupied cgroups are rejected before any external mutation");

  FakeDeviceKernel changed;
  changed.corrupt_post_attach = true;
  LinuxDevicePolicyInstaller changed_installer(changed);
  require_throws<LinuxDeviceKernelError>(
      [&] { (void)changed_installer.install(policy, image, cgroup); },
      "post-attach kernel identity drift fails closed");
}

void noncompiler_images_and_receipt_tampering_are_rejected() {
  auto [policy, image] = policy_image();
  auto cgroup = fake_cgroup();
  auto altered = image;
  altered.instructions.front().offset = 4;
  FakeDeviceKernel unused;
  LinuxDevicePolicyInstaller installer(unused);
  require_throws<LinuxDeviceKernelError>(
      [&] { (void)installer.install(policy, altered, cgroup); },
      "tampered image fails its seal before reaching the kernel");
  require(unused.query_calls == 0U && unused.load_calls == 0U,
          "invalid images cause no kernel observation or mutation");

  const auto installed = installer.install(policy, image, cgroup);
  auto tampered = installed;
  tampered.program.program_id += 1U;
  require_throws<LinuxDeviceKernelError>(
      [&] { validate_linux_device_policy_installation(tampered); },
      "kernel identity tampering invalidates the installation receipt");
  require(hostd_linux_device_kernel_test_seam::program_name_for_image(image) ==
              installed.program.program_name,
          "program names deterministically bind the instruction image");
}

} // namespace

int main() {
  try {
    exact_installation_is_sealed_and_restart_verifiable();
    preexisting_or_changed_programs_fail_closed();
    noncompiler_images_and_receipt_tampering_are_rejected();
    std::cout << "hostd Linux device kernel tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd Linux device kernel test failure: " << error.what()
              << '\n';
    return 1;
  }
}
