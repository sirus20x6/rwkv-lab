#include "trainvm/rwkv_lab_worker_contract.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "trainvm/training_component_registry.hpp"

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

const trainvm::AdapterProfile& find_profile(
    const trainvm::RwkvLabWorkerContract& contract,
    std::string_view adapter) {
  const auto profile = std::ranges::find_if(
      contract.adapter_registry.profiles,
      [&](const trainvm::AdapterProfile& candidate) {
        return candidate.key.adapter == adapter;
      });
  if (profile == contract.adapter_registry.profiles.end()) {
    throw std::runtime_error("expected rwkv_lab adapter profile is absent");
  }
  return *profile;
}

}  // namespace

int main() {
  try {
    const std::string fingerprint = "sha256:" + std::string(64U, 'a');
    const std::string runtime_closure =
        "sha256:" + std::string(64U, 'c');
    const trainvm::RwkvLabWorkerContract contract =
        trainvm::rwkv_lab_worker_contract(fingerprint);
    require(contract.adapter_registry.api_version == "trainvm.adapters/v2" &&
                contract.adapter_registry.profiles.size() == 4U,
            "rwkv_lab catalog must expose four exact adapter profiles");
    require(std::ranges::is_sorted(contract.provided_capabilities) &&
                std::ranges::adjacent_find(contract.provided_capabilities) ==
                    contract.provided_capabilities.end(),
            "rwkv_lab provided capabilities must be canonical");

    const auto runtime_requirements =
        trainvm::rwkv_lab_worker_runtime_requirements();
    require(runtime_requirements.api_version ==
                    "trainvm.rwkv-lab-worker-runtime-requirements/v1" &&
                runtime_requirements.profiles.size() == 4U &&
                runtime_requirements.shared_root_distributions ==
                    std::vector<std::string>(
                        {"grpcio", "pillow", "protobuf", "torch"}),
            "native runtime requirements must expose the shared worker closure");
    for (std::size_t index = 0;
         index < runtime_requirements.profiles.size(); ++index) {
      const auto& requirements = runtime_requirements.profiles.at(index);
      require(requirements.adapter ==
                      contract.adapter_registry.profiles.at(index).key.adapter &&
                  std::ranges::is_sorted(requirements.root_distributions) &&
                  std::ranges::adjacent_find(
                      requirements.root_distributions) ==
                      requirements.root_distributions.end() &&
                  std::ranges::includes(requirements.root_distributions,
                                        runtime_requirements
                                            .shared_root_distributions),
              "each runtime profile must exactly cover one registered adapter");
    }

    const auto& appearance = find_profile(
        contract, "rwkv-lab.mageflow-appearance-expert");
    const auto& terminal = find_profile(
        contract, "rwkv-lab.mageflow-terminal-expert");
    const auto& qwen = find_profile(contract, "rwkv-lab.qwen-ao3");
    const auto& rwkv = find_profile(contract, "rwkv-lab.rwkv-scratch");
    require(appearance.training_composition &&
                appearance.training_composition->model_family == "mageflow" &&
                appearance.training_composition->slots.size() == 5U &&
                terminal.training_composition &&
                terminal.training_composition->model_family == "mageflow" &&
                terminal.training_composition->slots.size() == 6U &&
                qwen.training_composition &&
                qwen.training_composition->model_family == "transformer" &&
                qwen.training_composition->slots.size() == 4U &&
                rwkv.training_composition &&
                rwkv.training_composition->model_family == "rwkv" &&
                rwkv.training_composition->slots.size() == 10U,
            "real trainer profiles must expose exact family-specific slot surfaces");
    require(appearance.lifecycle.resume_grade ==
                trainvm::ResumeGrade::compatible &&
                appearance.lifecycle.checkpoint_now &&
                terminal.lifecycle.resume_grade ==
                    trainvm::ResumeGrade::compatible &&
                qwen.lifecycle.resume_grade ==
                    trainvm::ResumeGrade::compatible &&
                rwkv.lifecycle.resume_grade ==
                    trainvm::ResumeGrade::terminal_checkpoint &&
                !rwkv.lifecycle.checkpoint_now &&
                !rwkv.lifecycle.pause_keep_resources &&
                !rwkv.lifecycle.pause_release_resources,
            "real trainer lifecycle grades must not overclaim checkpoint support");

    const auto component_path =
        std::filesystem::path(TRAINVM_SOURCE_ROOT) /
        "docs/experiment-vm/examples/training-components.v1.json";
    const trainvm::TrainingComponentRegistry components =
        trainvm::TrainingComponentRegistry::load_file(
            std::filesystem::absolute(component_path));
    std::set<std::string> component_capabilities;
    for (const nlohmann::json& descriptor : components.descriptors_json()) {
      const auto required =
          descriptor.at("required_capabilities").get<std::vector<std::string>>();
      component_capabilities.insert(required.begin(), required.end());
    }
    require(std::ranges::includes(contract.provided_capabilities,
                                  component_capabilities),
            "sealed rwkv_lab worker contract must cover the checked-in component catalog");

    std::vector<trainvm::RwkvLabWorkerRuntimeDeploymentSpec> runtimes;
    for (const trainvm::AdapterProfile& profile :
         contract.adapter_registry.profiles) {
      runtimes.push_back({
          .adapter = profile.key.adapter,
          .code_path = "/opt/trainvm/workers/" + profile.key.adapter + ".pyz",
          .code_fingerprint = fingerprint,
          .bootstrap_runtime_closure_fingerprint = runtime_closure,
          .executable_path = "/opt/trainvm/python/bin/python3",
          .executable_fingerprint = "sha256:" + std::string(64U, 'b'),
          .working_directory = "/srv/trainvm/work",
      });
    }
    runtimes.at(2).code_fingerprint =
        "sha256:" + std::string(64U, 'd');
    runtimes.at(2).bootstrap_runtime_closure_fingerprint =
        "sha256:" + std::string(64U, 'e');
    const auto deployment = trainvm::rwkv_lab_worker_deployment({
        .api_version = "trainvm.rwkv-lab-worker-runtimes/v1",
        .runtimes = runtimes,
        .trusted_roots = {"/srv/trainvm", "/opt/trainvm"},
    });
    require(deployment.adapter_registry.profiles.size() ==
                    contract.adapter_registry.profiles.size() &&
                deployment.provided_capabilities ==
                    contract.provided_capabilities &&
                deployment.host_launch_registry.api_version ==
                    "trainvm.host-launches/v4" &&
                deployment.host_launch_registry.profiles.size() == 4U,
            "deployment lowering must retain the complete reflected worker catalog");
    for (const trainvm::HostLaunchProfile& launch :
         deployment.host_launch_registry.profiles) {
      const auto expected = std::ranges::find_if(
          runtimes, [&](const auto& runtime) {
            return runtime.adapter == launch.key.adapter;
          });
      require(expected != runtimes.end() &&
                  launch.code_fingerprint == expected->code_fingerprint &&
                  launch.bootstrap_runtime_closure_fingerprint ==
                      expected->bootstrap_runtime_closure_fingerprint &&
                  launch.provided_capabilities ==
                      contract.provided_capabilities &&
                  launch.code_argument_index == 1U &&
                  launch.public_arguments ==
                      std::vector<std::string>({"-I", "rwkv-lab-worker.pyz"}),
              "each deployment profile must bind isolation before its sealed code slot");
    }
    const auto qwen_profile = std::ranges::find_if(
        deployment.adapter_registry.profiles, [](const auto& profile) {
          return profile.key.adapter == "rwkv-lab.qwen-ao3";
        });
    require(
        qwen_profile != deployment.adapter_registry.profiles.end() &&
            qwen_profile->code_fingerprint == runtimes.at(2).code_fingerprint,
            "adapter registry and host profile must share each adapter-specific code identity");

    auto missing_runtime = runtimes;
    missing_runtime.pop_back();
    bool missing_runtime_rejected = false;
    try {
      (void)trainvm::rwkv_lab_worker_deployment({
          .api_version = "trainvm.rwkv-lab-worker-runtimes/v1",
          .runtimes = std::move(missing_runtime),
          .trusted_roots = {"/srv/trainvm", "/opt/trainvm"},
      });
    } catch (const std::invalid_argument&) {
      missing_runtime_rejected = true;
    }
    require(missing_runtime_rejected,
            "deployment lowering must reject a missing adapter runtime");

    bool invalid_fingerprint_rejected = false;
    try {
      (void)trainvm::rwkv_lab_worker_contract("sha256:not-a-digest");
    } catch (const std::invalid_argument&) {
      invalid_fingerprint_rejected = true;
    }
    require(invalid_fingerprint_rejected,
            "worker contract must reject an unsealed code identity");
    std::cout << "rwkv_lab worker contract tests passed\n";
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "FAIL: " << exception.what() << '\n';
    return 1;
  }
}
