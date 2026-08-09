#include "trainvm/hostd_linux_device_policy.hpp"

#include <linux/bpf.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace {

using namespace trainvm;

constexpr std::string_view kGpuA = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";
constexpr std::string_view kGpuB = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb";

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

HostResourceId gpu_id(std::string id) {
  return {.kind = HostResourceKind::accelerator,
          .vendor = HostAcceleratorVendor::nvidia,
          .stable_id = std::move(id),
          .parent_id = std::nullopt};
}

ObservedHostResource gpu(std::string id, std::string bdf, std::uint32_t minor) {
  return {
      .id = gpu_id(std::move(id)),
      .disposition = ResourceObservationDisposition::audited_eligible,
      .compute_contexts = ResourceContextDisposition::absent,
      .graphics_contexts = ResourceContextDisposition::absent,
      .pci_bdf = std::move(bdf),
      .device_major = 195U,
      .device_minor = minor,
      .device_nodes = {{.type = HostDeviceNodeType::character,
                        .purpose = HostDeviceNodePurpose::assigned_accelerator,
                        .major = 195U,
                        .minor = minor,
                        .read = true,
                        .write = true},
                       {.type = HostDeviceNodeType::character,
                        .purpose = HostDeviceNodePurpose::shared_driver_control,
                        .major = 195U,
                        .minor = 255U,
                        .read = true,
                        .write = true},
                       {.type = HostDeviceNodeType::character,
                        .purpose = HostDeviceNodePurpose::shared_driver_control,
                        .major = 511U,
                        .minor = 0U,
                        .read = true,
                        .write = true}},
      .numa_node = 0,
      .pcie_root_id = std::nullopt,
      .fabric_clique_id = std::nullopt,
      .total_memory_bytes = 24ULL << 30U,
      .labels = {{"backend", "test"}},
  };
}

HostInventoryReceipt inventory(bool second_gpu = true) {
  HostKernelSnapshot snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = "host-device-policy",
      .boot_id = "boot-device-policy",
      .broker_epoch = "broker-device-policy",
      .begin_revision = "revision-device-policy",
      .end_revision = "revision-device-policy",
      .probes = {{.vendor = HostAcceleratorVendor::nvidia,
                  .disposition = ProbeDisposition::complete,
                  .context_details_complete = true,
                  .detail = "test"}},
      .resources = {gpu(std::string(kGpuA), "0000:01:00.0", 0U)},
  };
  if (second_gpu) {
    snapshot.resources.push_back(gpu(std::string(kGpuB), "0000:02:00.0", 1U));
  }
  FakeHostKernel kernel(
      {{.snapshot = std::move(snapshot), .failure = std::nullopt}});
  return capture_host_inventory(kernel);
}

ResourceBundleGrant grant(const HostInventoryReceipt &observed,
                          bool second_gpu = true) {
  ResourceBundleGrant value{
      .api_version = std::string(kHostLedgerGrantApiVersion),
      .allocation_id = "allocation-device-policy",
      .request_id = "request-device-policy",
      .request_digest = "sha256:" + std::string(64U, '1'),
      .journal_id = "journal-device-policy",
      .run_id = "run-device-policy",
      .logical_lease_id = "lease-device-policy",
      .logical_fencing_token = 7U,
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .fences = {{.resource = gpu_id(std::string(kGpuA)),
                  .generation = 1U,
                  .inventory_digest = observed.inventory_digest,
                  .topology_digest = observed.topology_digest}},
      // This fixture predates grant-time inventory projections and stays that
      // way on purpose: it is the older grant shape, and it still encodes and
      // digests exactly as it did.
      .inventory_projection = std::nullopt,
      .granted_boottime_ns = 10,
      .granted_wall_time_ns = 20,
      .previous_receipt_digest = "sha256:" + std::string(64U, '0'),
      .receipt_digest = {},
  };
  if (second_gpu) {
    value.fences.push_back({.resource = gpu_id(std::string(kGpuB)),
                            .generation = 1U,
                            .inventory_digest = observed.inventory_digest,
                            .topology_digest = observed.topology_digest});
  }
  auto canonical = encode_json(value);
  canonical.erase("receipt_digest");
  value.receipt_digest = digest("trainvm.host-resource-grant/v1", canonical);
  (void)resource_bundle_grant_json(value);
  return value;
}

LinuxAllocationCgroupIdentity cgroup() {
  return {.unified_path = "/trainvm/allocation-device-policy",
          .device = 31U,
          .inode = 41U};
}

std::uint64_t execute(const LinuxDeviceProgramImage &image,
                      std::uint32_t access_type, std::uint32_t major,
                      std::uint32_t minor) {
  std::array<std::uint64_t, 11U> registers{};
  const std::array<std::uint32_t, 3U> context{access_type, major, minor};
  std::size_t pc = 0U;
  while (pc < image.instructions.size()) {
    const auto &instruction = image.instructions[pc];
    const auto code = instruction.code;
    if (code == static_cast<std::uint8_t>(BPF_LDX | BPF_MEM | BPF_W)) {
      require(instruction.source_register == BPF_REG_1 &&
                  instruction.offset >= 0 && instruction.offset % 4 == 0,
              "test interpreter received an invalid context load");
      registers[instruction.destination_register] =
          context[static_cast<std::size_t>(instruction.offset / 4)];
    } else if (code == static_cast<std::uint8_t>(BPF_ALU64 | BPF_MOV | BPF_X)) {
      registers[instruction.destination_register] =
          registers[instruction.source_register];
    } else if (code == static_cast<std::uint8_t>(BPF_ALU64 | BPF_MOV | BPF_K)) {
      registers[instruction.destination_register] =
          static_cast<std::uint32_t>(instruction.immediate);
    } else if (code == static_cast<std::uint8_t>(BPF_ALU64 | BPF_AND | BPF_K)) {
      registers[instruction.destination_register] &=
          static_cast<std::uint32_t>(instruction.immediate);
    } else if (code == static_cast<std::uint8_t>(BPF_JMP | BPF_JNE | BPF_K)) {
      if (registers[instruction.destination_register] !=
          static_cast<std::uint32_t>(instruction.immediate)) {
        pc += static_cast<std::size_t>(instruction.offset) + 1U;
        continue;
      }
    } else if (code == static_cast<std::uint8_t>(BPF_JMP | BPF_EXIT)) {
      return registers[BPF_REG_0];
    } else {
      throw std::runtime_error("test interpreter received an unknown opcode");
    }
    ++pc;
  }
  throw std::runtime_error("device program fell off its instruction image");
}

std::uint32_t access(HostDeviceNodeType type, std::uint32_t permissions) {
  const auto kernel_type = type == HostDeviceNodeType::character
                               ? BPF_DEVCG_DEV_CHAR
                               : BPF_DEVCG_DEV_BLOCK;
  return kernel_type | (permissions << 16U);
}

void exact_capabilities_compile_to_default_deny_program() {
  const auto observed = inventory();
  const auto policy = derive_linux_device_policy(
      observed, grant(observed), cgroup(), "launch-device-policy");
  const auto image = compile_linux_device_program(policy);
  require(policy.rules.size() == 9U,
          "shared NVIDIA nodes are deduplicated across assigned GPUs");
  require(image.instructions.size() == 114U &&
              linux_device_policy_json(policy).at("policy_digest") ==
                  policy.policy_digest &&
              linux_device_program_image_json(image).at("image_digest") ==
                  image.image_digest,
          "policy and instruction image are bounded and canonically sealed");

  require(execute(image,
                  access(HostDeviceNodeType::character,
                         BPF_DEVCG_ACC_READ | BPF_DEVCG_ACC_WRITE),
                  195U, 0U) == 1U &&
              execute(image,
                      access(HostDeviceNodeType::character,
                             BPF_DEVCG_ACC_READ | BPF_DEVCG_ACC_WRITE),
                      195U, 1U) == 1U &&
              execute(image,
                      access(HostDeviceNodeType::character,
                             BPF_DEVCG_ACC_READ | BPF_DEVCG_ACC_WRITE),
                      195U, 255U) == 1U,
          "assigned GPUs and shared driver controls are allowed exactly");
  require(
      execute(image, access(HostDeviceNodeType::character, BPF_DEVCG_ACC_READ),
              1U, 9U) == 1U &&
          execute(image,
                  access(HostDeviceNodeType::character, BPF_DEVCG_ACC_WRITE),
                  1U, 9U) == 0U,
      "runtime entropy nodes are read-only capabilities");
  require(
      execute(image, access(HostDeviceNodeType::character, BPF_DEVCG_ACC_READ),
              195U, 2U) == 0U &&
          execute(image, access(HostDeviceNodeType::block, BPF_DEVCG_ACC_READ),
                  195U, 0U) == 0U &&
          execute(image,
                  access(HostDeviceNodeType::character, BPF_DEVCG_ACC_MKNOD),
                  195U, 0U) == 0U,
      "unknown nodes, wrong types, and mknod remain default denied");
}

void stale_or_incomplete_authority_fails_closed() {
  const auto observed = inventory(false);
  auto stale = grant(observed, false);
  stale.fences.front().inventory_digest = "sha256:" + std::string(64U, '9');
  auto canonical = encode_json(stale);
  canonical.erase("receipt_digest");
  stale.receipt_digest = digest("trainvm.host-resource-grant/v1", canonical);
  require_throws<LinuxDevicePolicyError>(
      [&] {
        (void)derive_linux_device_policy(observed, stale, cgroup(),
                                         "launch-stale");
      },
      "a stale grant inventory digest is rejected");

  HostKernelSnapshot empty_snapshot{
      .api_version = std::string(kHostInventoryApiVersion),
      .host_id = observed.host_id,
      .boot_id = observed.boot_id,
      .broker_epoch = observed.broker_epoch,
      .begin_revision = "revision-no-capability",
      .end_revision = "revision-no-capability",
      .probes = observed.probes,
      .resources = {gpu(std::string(kGpuA), "0000:01:00.0", 0U)},
  };
  empty_snapshot.resources.front().device_nodes.clear();
  FakeHostKernel kernel(
      {{.snapshot = std::move(empty_snapshot), .failure = std::nullopt}});
  const auto incomplete = capture_host_inventory(kernel);
  require_throws<LinuxDevicePolicyError>(
      [&] {
        (void)derive_linux_device_policy(incomplete, grant(incomplete, false),
                                         cgroup(), "launch-incomplete");
      },
      "an accelerator without a complete node map cannot launch");
}

void sealed_policy_and_image_reject_tampering() {
  const auto observed = inventory(false);
  auto policy = derive_linux_device_policy(observed, grant(observed, false),
                                           cgroup(), "launch-tamper");
  policy.rules.front().write = !policy.rules.front().write;
  require_throws<LinuxDevicePolicyError>(
      [&] { validate_linux_device_policy(policy); },
      "policy rule tampering invalidates the policy digest");

  const auto clean_policy = derive_linux_device_policy(
      observed, grant(observed, false), cgroup(), "launch-clean");
  auto image = compile_linux_device_program(clean_policy);
  image.instructions.front().offset = 4;
  require_throws<LinuxDevicePolicyError>(
      [&] { validate_linux_device_program(image); },
      "instruction tampering invalidates the image digest");
}

} // namespace

int main() {
  try {
    exact_capabilities_compile_to_default_deny_program();
    stale_or_incomplete_authority_fails_closed();
    sealed_policy_and_image_reject_tampering();
    std::cout << "hostd Linux device policy tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd Linux device policy test failure: " << error.what()
              << '\n';
    return 1;
  }
}
