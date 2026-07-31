#include "trainvm/adapter_registry.hpp"

#include <algorithm>
#include <ranges>
#include <set>
#include <tuple>
#include <utility>

namespace trainvm {
namespace {

bool valid_sha256_fingerprint(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::string> canonical_capabilities(
    std::vector<std::string> capabilities) {
  if (capabilities.size() > 256U ||
      std::ranges::any_of(capabilities, [](const std::string& capability) {
        return capability.empty() || capability.size() > 256U;
      })) {
    throw std::invalid_argument(
        "adapter profile capabilities must be bounded and nonempty");
  }
  std::ranges::sort(capabilities);
  if (std::ranges::adjacent_find(capabilities) != capabilities.end()) {
    throw std::invalid_argument(
        "adapter profile capabilities must be unique");
  }
  return capabilities;
}

void validate_profile(AdapterProfile& profile) {
  if (profile.key.adapter.empty() || profile.key.version.empty() ||
      profile.key.operation.empty() || profile.key.contract.empty()) {
    throw std::invalid_argument("adapter profile key fields must not be empty");
  }
  const bool builtin = profile.key.runtime == ComponentRuntime::builtin;
  if (builtin) {
    if (!profile.code_fingerprint.empty() ||
        !profile.required_capabilities.empty()) {
      throw std::invalid_argument(
          "builtin adapter profiles cannot carry worker launch authority");
    }
    const bool acquire =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "acquire_resources" &&
        profile.key.contract == "trainvm.v1.AcquireResources" &&
        profile.effect == Effect::resource &&
        profile.idempotency == Idempotency::receipt_required;
    const bool validate =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "validate_artifact" &&
        profile.key.contract == "trainvm.v1.ValidateArtifact" &&
        profile.effect == Effect::read_only &&
        profile.idempotency == Idempotency::replay_safe;
    const bool release =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "release_resources" &&
        profile.key.contract == "trainvm.v1.ReleaseResources" &&
        profile.effect == Effect::resource &&
        profile.idempotency == Idempotency::replay_safe;
    if (!acquire && !validate && !release) {
      throw std::invalid_argument(
          "builtin adapter profile has no exact typed trainvm.core executor");
    }
    return;
  }
  if (!valid_sha256_fingerprint(profile.code_fingerprint)) {
    throw std::invalid_argument(
        "worker adapter code fingerprint must be canonical sha256 hex");
  }
  profile.required_capabilities =
      canonical_capabilities(std::move(profile.required_capabilities));
}

}  // namespace

AdapterRegistry::AdapterRegistry(std::vector<AdapterProfile> profiles) {
  std::set<std::tuple<std::string, std::string, ComponentRuntime, std::string>>
      selectors;
  for (AdapterProfile& profile : profiles) {
    validate_profile(profile);
    const auto selector = std::tuple{
        profile.key.adapter, profile.key.version, profile.key.runtime,
        profile.key.operation};
    if (!selectors.insert(selector).second) {
      throw std::invalid_argument(
          "adapter registry contains a duplicate operation selector");
    }
    const AdapterKey key = profile.key;
    if (!profiles_.emplace(key, std::move(profile)).second) {
      throw std::invalid_argument("adapter registry contains a duplicate exact key");
    }
  }
}

const AdapterProfile& AdapterRegistry::resolve(
    const Component& component, std::string_view operation) const {
  const auto declared = component.operations.find(std::string(operation));
  if (declared == component.operations.end()) {
    throw AdapterResolutionError(
        "component does not declare the requested adapter operation");
  }
  const AdapterKey key{
      .adapter = component.adapter,
      .version = component.version,
      .runtime = component.runtime,
      .operation = std::string(operation),
      .contract = declared->second.contract,
  };
  const auto profile = profiles_.find(key);
  if (profile == profiles_.end()) {
    throw AdapterResolutionError(
        "no authority-owned adapter profile matches the exact component key");
  }
  return profile->second;
}

void AdapterRegistry::validate_plan(const CompiledPlan& plan) const {
  for (const auto& [name, component] : plan.experiment.spec.components) {
    (void)name;
    for (const auto& [operation, declaration] : component.operations) {
      (void)declaration;
      (void)resolve(component, operation);
    }
  }
  for (const auto& [name, node] :
       plan.experiment.spec.workflow.nodes) {
    const Component& component =
        plan.experiment.spec.components.at(node.invoke.component);
    const AdapterProfile& profile =
        resolve(component, node.invoke.operation);
    if (profile.effect != node.effect ||
        profile.idempotency != node.idempotency) {
      throw AdapterResolutionError(
          "workflow node " + name +
          " disagrees with its authority-owned effect or idempotency class");
    }
  }
}

WorkerLaunchRequest AdapterRegistry::worker_launch_request(
    const Component& component, std::string_view operation) const {
  const AdapterProfile& profile = resolve(component, operation);
  if (profile.key.runtime == ComponentRuntime::builtin) {
    throw AdapterResolutionError(
        "builtin adapter profiles cannot authorize a worker launch");
  }
  return WorkerLaunchRequest{
      .code_fingerprint = profile.code_fingerprint,
      .required_capabilities = profile.required_capabilities,
  };
}

}  // namespace trainvm
