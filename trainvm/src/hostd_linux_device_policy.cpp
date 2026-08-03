#include "trainvm/hostd_linux_device_policy.hpp"

#include <linux/bpf.h>

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <ranges>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

constexpr std::size_t kMaximumPolicyRules = 256U;
constexpr std::size_t kInstructionsPerRule = 12U;
constexpr std::size_t kFixedInstructionCount = 6U;
constexpr std::size_t kMaximumProgramInstructions = 4096U;
constexpr std::uint32_t kDeviceTypeMask = 0x0000ffffU;
constexpr std::uint32_t kDeviceAccessMask = 0xffff0000U;

[[noreturn]] void reject(std::string message) {
  throw LinuxDevicePolicyError(std::move(message));
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

std::string node_type_name(HostDeviceNodeType type) {
  switch (type) {
  case HostDeviceNodeType::character:
    return "character";
  case HostDeviceNodeType::block:
    return "block";
  }
  reject("device policy contains an unknown node type");
}

std::string source_name(LinuxDevicePolicyRuleSource source) {
  switch (source) {
  case LinuxDevicePolicyRuleSource::runtime_baseline:
    return "runtime_baseline";
  case LinuxDevicePolicyRuleSource::inventory_capability:
    return "inventory_capability";
  }
  reject("device policy contains an unknown rule source");
}

auto rule_key(const LinuxDevicePolicyRule &rule) {
  return std::tuple{rule.type, rule.major, rule.minor,
                    rule.read, rule.write, rule.source};
}

nlohmann::json policy_digest_json(const LinuxDevicePolicySpec &policy) {
  nlohmann::json rules = nlohmann::json::array();
  for (const auto &rule : policy.rules) {
    rules.push_back({{"type", node_type_name(rule.type)},
                     {"major", rule.major},
                     {"minor", rule.minor},
                     {"read", rule.read},
                     {"write", rule.write},
                     {"source", source_name(rule.source)}});
  }
  return {{"api_version", policy.api_version},
          {"allocation_id", policy.allocation_id},
          {"launch_id", policy.launch_id},
          {"grant_digest", policy.grant_digest},
          {"inventory_digest", policy.inventory_digest},
          {"topology_digest", policy.topology_digest},
          {"cgroup",
           {{"unified_path", policy.cgroup.unified_path},
            {"device", policy.cgroup.device},
            {"inode", policy.cgroup.inode}}},
          {"rules", std::move(rules)}};
}

nlohmann::json instruction_json(const LinuxBpfInstruction &instruction) {
  return {{"code", instruction.code},
          {"destination_register", instruction.destination_register},
          {"source_register", instruction.source_register},
          {"offset", instruction.offset},
          {"immediate", instruction.immediate}};
}

nlohmann::json image_digest_json(const LinuxDeviceProgramImage &image) {
  nlohmann::json instructions = nlohmann::json::array();
  for (const auto &instruction : image.instructions) {
    instructions.push_back(instruction_json(instruction));
  }
  return {{"api_version", image.api_version},
          {"policy_digest", image.policy_digest},
          {"instructions", std::move(instructions)}};
}

std::string digest(std::string_view domain, const nlohmann::json &value) {
  std::string material(domain);
  material.push_back('\0');
  material += value.dump();
  return "sha256:" + sha256_hex(material);
}

const ObservedHostResource *find_resource(const HostInventoryReceipt &inventory,
                                          const HostResourceId &id) {
  const auto found =
      std::ranges::find(inventory.resources, id, &ObservedHostResource::id);
  return found == inventory.resources.end() ? nullptr : &*found;
}

LinuxDevicePolicyRule
capability_rule(const HostDeviceNodeCapability &capability) {
  return {.type = capability.type,
          .major = capability.major,
          .minor = capability.minor,
          .read = capability.read,
          .write = capability.write,
          .source = LinuxDevicePolicyRuleSource::inventory_capability};
}

std::vector<LinuxDevicePolicyRule> baseline_rules() {
  return {
      {.type = HostDeviceNodeType::character,
       .major = 1U,
       .minor = 3U,
       .read = true,
       .write = true,
       .source = LinuxDevicePolicyRuleSource::runtime_baseline},
      {.type = HostDeviceNodeType::character,
       .major = 1U,
       .minor = 5U,
       .read = true,
       .write = true,
       .source = LinuxDevicePolicyRuleSource::runtime_baseline},
      {.type = HostDeviceNodeType::character,
       .major = 1U,
       .minor = 7U,
       .read = true,
       .write = true,
       .source = LinuxDevicePolicyRuleSource::runtime_baseline},
      {.type = HostDeviceNodeType::character,
       .major = 1U,
       .minor = 8U,
       .read = true,
       .write = false,
       .source = LinuxDevicePolicyRuleSource::runtime_baseline},
      {.type = HostDeviceNodeType::character,
       .major = 1U,
       .minor = 9U,
       .read = true,
       .write = false,
       .source = LinuxDevicePolicyRuleSource::runtime_baseline},
  };
}

LinuxBpfInstruction instruction(std::uint8_t code, std::uint8_t destination,
                                std::uint8_t source, std::int16_t offset,
                                std::int32_t immediate) {
  return {.code = code,
          .destination_register = destination,
          .source_register = source,
          .offset = offset,
          .immediate = immediate};
}

LinuxBpfInstruction load_context_word(std::uint8_t destination,
                                      std::int16_t offset) {
  return instruction(static_cast<std::uint8_t>(BPF_LDX | BPF_MEM | BPF_W),
                     destination, BPF_REG_1, offset, 0);
}

LinuxBpfInstruction move_register(std::uint8_t destination,
                                  std::uint8_t source) {
  return instruction(static_cast<std::uint8_t>(BPF_ALU64 | BPF_MOV | BPF_X),
                     destination, source, 0, 0);
}

LinuxBpfInstruction move_immediate(std::uint8_t destination,
                                   std::int32_t value) {
  return instruction(static_cast<std::uint8_t>(BPF_ALU64 | BPF_MOV | BPF_K),
                     destination, 0, 0, value);
}

LinuxBpfInstruction and_immediate(std::uint8_t destination,
                                  std::uint32_t value) {
  return instruction(static_cast<std::uint8_t>(BPF_ALU64 | BPF_AND | BPF_K),
                     destination, 0, 0, static_cast<std::int32_t>(value));
}

LinuxBpfInstruction jump_not_equal(std::uint8_t destination,
                                   std::uint32_t value, std::int16_t offset) {
  return instruction(static_cast<std::uint8_t>(BPF_JMP | BPF_JNE | BPF_K),
                     destination, 0, offset, static_cast<std::int32_t>(value));
}

LinuxBpfInstruction exit_instruction() {
  return instruction(static_cast<std::uint8_t>(BPF_JMP | BPF_EXIT), 0, 0, 0, 0);
}

std::uint32_t kernel_device_type(HostDeviceNodeType type) {
  return type == HostDeviceNodeType::character ? BPF_DEVCG_DEV_CHAR
                                               : BPF_DEVCG_DEV_BLOCK;
}

} // namespace

LinuxDevicePolicySpec derive_linux_device_policy(
    const HostInventoryReceipt &inventory, const ResourceBundleGrant &grant,
    const LinuxAllocationCgroupIdentity &cgroup, std::string launch_id) {
  validate_host_inventory(inventory);
  (void)resource_bundle_grant_json(grant);
  if (grant.host_id != inventory.host_id ||
      grant.boot_id != inventory.boot_id ||
      grant.broker_epoch != inventory.broker_epoch || launch_id.empty() ||
      !canonical_absolute_path(cgroup.unified_path) || cgroup.device == 0U ||
      cgroup.inode == 0U) {
    reject("device policy authority identities do not match");
  }

  std::vector<LinuxDevicePolicyRule> rules = baseline_rules();
  std::size_t accelerator_count = 0U;
  for (const auto &fence : grant.fences) {
    if (fence.inventory_digest != inventory.inventory_digest ||
        fence.topology_digest != inventory.topology_digest) {
      reject("device policy fence does not bind the current inventory");
    }
    const ObservedHostResource *resource =
        find_resource(inventory, fence.resource);
    if (resource == nullptr) {
      reject("device policy fence names an absent inventory resource");
    }
    if (resource->id.kind == HostResourceKind::host_mutex)
      continue;
    ++accelerator_count;
    if (resource->id.kind == HostResourceKind::accelerator_partition) {
      reject("partition device capabilities are not yet launch-authoritative");
    }
    if (resource->device_nodes.empty()) {
      reject("accelerator has no launch-authoritative device capabilities");
    }
    const bool has_assigned_node = std::ranges::any_of(
        resource->device_nodes, [](const HostDeviceNodeCapability &capability) {
          return capability.purpose ==
                 HostDeviceNodePurpose::assigned_accelerator;
        });
    if (!has_assigned_node) {
      reject("accelerator capability set omits its assigned device node");
    }
    for (const auto &capability : resource->device_nodes) {
      rules.push_back(capability_rule(capability));
    }
  }
  if (accelerator_count == 0U) {
    reject("GPU device policy requires an accelerator grant");
  }

  std::ranges::sort(rules, {}, rule_key);
  rules.erase(std::unique(rules.begin(), rules.end(),
                          [](const auto &left, const auto &right) {
                            return left.type == right.type &&
                                   left.major == right.major &&
                                   left.minor == right.minor &&
                                   left.read == right.read &&
                                   left.write == right.write;
                          }),
              rules.end());
  if (rules.size() > kMaximumPolicyRules) {
    reject("device policy exceeds its rule bound");
  }

  LinuxDevicePolicySpec policy{
      .api_version = std::string(kLinuxDevicePolicyApiVersion),
      .allocation_id = grant.allocation_id,
      .launch_id = std::move(launch_id),
      .grant_digest = grant.receipt_digest,
      .inventory_digest = inventory.inventory_digest,
      .topology_digest = inventory.topology_digest,
      .cgroup = cgroup,
      .rules = std::move(rules),
      .policy_digest = {},
  };
  policy.policy_digest =
      digest("trainvm.linux-device-policy/v1", policy_digest_json(policy));
  validate_linux_device_policy(policy);
  return policy;
}

LinuxDeviceProgramImage
compile_linux_device_program(const LinuxDevicePolicySpec &policy) {
  validate_linux_device_policy(policy);
  std::vector<LinuxBpfInstruction> instructions;
  instructions.reserve(kFixedInstructionCount +
                       kInstructionsPerRule * policy.rules.size());
  instructions.push_back(load_context_word(BPF_REG_2, 0));
  instructions.push_back(move_register(BPF_REG_3, BPF_REG_2));
  instructions.push_back(and_immediate(
      BPF_REG_3, static_cast<std::uint32_t>(BPF_DEVCG_ACC_MKNOD) << 16U));
  const std::size_t rules_instruction_count =
      policy.rules.size() * kInstructionsPerRule;
  if (rules_instruction_count >
      static_cast<std::size_t>(std::numeric_limits<std::int16_t>::max())) {
    reject("device policy jump exceeds the eBPF offset range");
  }
  instructions.push_back(jump_not_equal(
      BPF_REG_3, 0U, static_cast<std::int16_t>(rules_instruction_count)));

  for (const auto &rule : policy.rules) {
    const std::uint32_t allowed_access =
        ((rule.read ? static_cast<std::uint32_t>(BPF_DEVCG_ACC_READ) : 0U) |
         (rule.write ? static_cast<std::uint32_t>(BPF_DEVCG_ACC_WRITE) : 0U))
        << 16U;
    const std::uint32_t disallowed_access = kDeviceAccessMask & ~allowed_access;
    instructions.push_back(move_register(BPF_REG_3, BPF_REG_2));
    instructions.push_back(and_immediate(BPF_REG_3, kDeviceTypeMask));
    instructions.push_back(
        jump_not_equal(BPF_REG_3, kernel_device_type(rule.type), 9));
    instructions.push_back(move_register(BPF_REG_3, BPF_REG_2));
    instructions.push_back(and_immediate(BPF_REG_3, disallowed_access));
    instructions.push_back(jump_not_equal(BPF_REG_3, 0U, 6));
    instructions.push_back(load_context_word(BPF_REG_3, 4));
    instructions.push_back(jump_not_equal(BPF_REG_3, rule.major, 4));
    instructions.push_back(load_context_word(BPF_REG_3, 8));
    instructions.push_back(jump_not_equal(BPF_REG_3, rule.minor, 2));
    instructions.push_back(move_immediate(BPF_REG_0, 1));
    instructions.push_back(exit_instruction());
  }
  instructions.push_back(move_immediate(BPF_REG_0, 0));
  instructions.push_back(exit_instruction());
  if (instructions.size() > kMaximumProgramInstructions) {
    reject("device policy program exceeds the eBPF instruction bound");
  }

  LinuxDeviceProgramImage image{
      .api_version = std::string(kLinuxDeviceProgramImageApiVersion),
      .policy_digest = policy.policy_digest,
      .instructions = std::move(instructions),
      .image_digest = {},
  };
  image.image_digest =
      digest("trainvm.linux-device-program-image/v1", image_digest_json(image));
  validate_linux_device_program(image);
  return image;
}

void validate_linux_device_policy(const LinuxDevicePolicySpec &policy) {
  if (policy.api_version != kLinuxDevicePolicyApiVersion ||
      policy.allocation_id.empty() || policy.launch_id.empty() ||
      !valid_digest(policy.grant_digest) ||
      !valid_digest(policy.inventory_digest) ||
      !valid_digest(policy.topology_digest) ||
      !canonical_absolute_path(policy.cgroup.unified_path) ||
      policy.cgroup.device == 0U || policy.cgroup.inode == 0U ||
      policy.rules.empty() || policy.rules.size() > kMaximumPolicyRules ||
      !std::ranges::is_sorted(policy.rules, {}, rule_key)) {
    reject("device policy shape is invalid");
  }
  std::set<std::tuple<HostDeviceNodeType, std::uint32_t, std::uint32_t>> nodes;
  for (const auto &rule : policy.rules) {
    if ((!rule.read && !rule.write) ||
        !nodes.emplace(rule.type, rule.major, rule.minor).second) {
      reject("device policy contains an empty or duplicate node capability");
    }
  }
  if (policy.policy_digest !=
      digest("trainvm.linux-device-policy/v1", policy_digest_json(policy))) {
    reject("device policy digest validation failed");
  }
}

void validate_linux_device_program(const LinuxDeviceProgramImage &image) {
  if (image.api_version != kLinuxDeviceProgramImageApiVersion ||
      !valid_digest(image.policy_digest) || image.instructions.empty() ||
      image.instructions.size() > kMaximumProgramInstructions ||
      image.image_digest != digest("trainvm.linux-device-program-image/v1",
                                   image_digest_json(image))) {
    reject("device program image is invalid");
  }
}

nlohmann::json linux_device_policy_json(const LinuxDevicePolicySpec &policy) {
  validate_linux_device_policy(policy);
  nlohmann::json value = policy_digest_json(policy);
  value["policy_digest"] = policy.policy_digest;
  return value;
}

nlohmann::json
linux_device_program_image_json(const LinuxDeviceProgramImage &image) {
  validate_linux_device_program(image);
  nlohmann::json value = image_digest_json(image);
  value["image_digest"] = image.image_digest;
  return value;
}

} // namespace trainvm
