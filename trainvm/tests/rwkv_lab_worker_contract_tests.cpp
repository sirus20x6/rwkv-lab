#include "trainvm/rwkv_lab_worker_contract.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

#include "trainvm/reflection_json.hpp"
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

nlohmann::json load_mageflow_fixture() {
  const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                    "docs/experiment-vm/examples/mageflow-cache-resume.json";
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open MageFlow fixture");
  nlohmann::json source;
  input >> source;
  return source;
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
                contract.adapter_registry.profiles.size() == 13U,
            "rwkv_lab catalog must expose thirteen exact adapter profiles");
    require(std::ranges::is_sorted(contract.provided_capabilities) &&
                std::ranges::adjacent_find(contract.provided_capabilities) ==
                    contract.provided_capabilities.end(),
            "rwkv_lab provided capabilities must be canonical");

    const auto runtime_requirements =
        trainvm::rwkv_lab_worker_runtime_requirements();
    require(runtime_requirements.api_version ==
                    "trainvm.rwkv-lab-worker-runtime-requirements/v1" &&
                runtime_requirements.profiles.size() == 13U &&
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
    const auto& posttraining =
        find_profile(contract, "rwkv-lab.rwkv-posttraining");
    const auto& rwkv = find_profile(contract, "rwkv-lab.rwkv-scratch");
    const std::vector<std::string> transformer_adapters{
        "rwkv-lab.transformer-mla",
        "rwkv-lab.transformer-mla-engram",
        "rwkv-lab.transformer-mla-fsp",
        "rwkv-lab.transformer-mla-full-backbone",
        "rwkv-lab.transformer-mla-mtp",
        "rwkv-lab.transformer-mla-mutor",
        "rwkv-lab.transformer-mla-parallel",
        "rwkv-lab.transformer-mla-rwkv8",
    };
    const bool transformer_contracts_exact =
        std::ranges::all_of(transformer_adapters, [&](const auto& adapter) {
          const auto& transformer = find_profile(contract, adapter);
          return transformer.training_composition &&
                 transformer.training_composition->model_family ==
                     "transformer" &&
                 transformer.training_composition->slots.size() ==
                     (adapter == "rwkv-lab.transformer-mla-engram" ? 8U : 7U) &&
                 (adapter != "rwkv-lab.transformer-mla-engram" ||
                  (transformer.training_composition->slots.at(
                       "host_optimizer") ==
                       trainvm::TrainingComponentCategory::optimizer &&
                   transformer.training_composition->allowed_components->at(
                       "host_optimizer") ==
                       std::vector<trainvm::TrainingComponentKey>{{
                           trainvm::TrainingComponentCategory::optimizer,
                           "torch_sparse_adam", "1.0.0"}})) &&
                 transformer.lifecycle.resume_grade ==
                     trainvm::ResumeGrade::compatible &&
                 transformer.lifecycle.checkpoint_now &&
                 transformer.training_composition->allowed_components->at(
                     "optimizer").size() == 2U &&
                 !std::ranges::contains(
                     transformer.training_composition->allowed_components->at(
                         "optimizer"),
                     trainvm::TrainingComponentKey{
                         trainvm::TrainingComponentCategory::optimizer,
                         "torch_sparse_adam", "1.0.0"}) &&
                 transformer.key.contract.starts_with(
                     "rwkv_lab.transformer_mla");
        });
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
                rwkv.training_composition->slots.size() == 10U &&
                posttraining.training_composition &&
                posttraining.training_composition->model_family == "rwkv" &&
                posttraining.training_composition->slots.size() == 4U &&
                posttraining.training_composition->allowed_components->at(
                    "optimizer").size() == 1U &&
                posttraining.training_composition->allowed_components->at(
                    "optimizer").front().category ==
                    trainvm::TrainingComponentCategory::optimizer &&
                posttraining.training_composition->allowed_components->at(
                    "optimizer").front().name == "torch_adamw_no_decay" &&
                posttraining.training_composition->allowed_components->at(
                    "optimizer").front().version == "2.0.0" &&
                transformer_contracts_exact,
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
                posttraining.lifecycle.resume_grade ==
                    trainvm::ResumeGrade::restart_only &&
                !posttraining.lifecycle.checkpoint_now &&
                !posttraining.lifecycle.pause_keep_resources &&
                !posttraining.lifecycle.pause_release_resources &&
                !rwkv.lifecycle.checkpoint_now &&
                !rwkv.lifecycle.pause_keep_resources &&
                !rwkv.lifecycle.pause_release_resources,
            "real trainer lifecycle grades must not overclaim checkpoint support");

    require(
        trainvm::reflected_field_names<trainvm::OperationPortDescriptor>() ==
                std::vector<std::string>({"type", "required", "artifact_type",
                                          "artifact_schema", "description"}) &&
            trainvm::reflected_field_names<
                trainvm::OperationAuthoringDeclaration>() ==
                std::vector<std::string>({"inputs", "outputs"}) &&
            trainvm::reflected_field_names<
                trainvm::OperationDescriptorDocument>() ==
                std::vector<std::string>({"api_version", "operations"}),
        "operation descriptor authority must remain reflection-derived");

    const trainvm::AdapterRegistry operation_registry(
        contract.adapter_registry.profiles);
    const nlohmann::json operation_document =
        operation_registry.operation_descriptors_json();
    require(operation_document.at("api_version") ==
                    "trainvm.operations/v1" &&
                operation_document.at("operations").size() == 13U &&
                operation_registry.operation_descriptors_digest() ==
                    "sha256:" +
                        trainvm::sha256_hex(operation_document.dump()),
            "operation descriptor document must exactly enumerate and hash the registered profiles");
    const auto& operations = operation_document.at("operations");
    require(operations.at(0).at("key").at("adapter") ==
                    "rwkv-lab.mageflow-appearance-expert" &&
                operations.at(1).at("key").at("adapter") ==
                    "rwkv-lab.mageflow-terminal-expert" &&
                operations.at(2).at("key").at("adapter") ==
                    "rwkv-lab.qwen-ao3" &&
                operations.at(3).at("key").at("adapter") ==
                    "rwkv-lab.rwkv-posttraining" &&
                operations.at(4).at("key").at("adapter") ==
                    "rwkv-lab.rwkv-scratch" &&
                operations.at(5).at("key").at("adapter") ==
                    "rwkv-lab.transformer-mla" &&
                operations.at(11).at("key").at("adapter") ==
                    "rwkv-lab.transformer-mla-parallel" &&
                operations.at(12).at("key").at("adapter") ==
                    "rwkv-lab.transformer-mla-rwkv8",
            "operation descriptors must use canonical exact-key ordering");
    for (const nlohmann::json& operation : operations) {
      const bool is_posttraining =
          operation.at("key").at("adapter") ==
          "rwkv-lab.rwkv-posttraining";
      require(operation.at("authoring").at("inputs").at("config").at(
                  "type") == "object" &&
                  operation.at("authoring").at("inputs").at("config").at(
                      "required") == true &&
                  (is_posttraining
                       ? operation.at("authoring")
                                 .at("outputs")
                                 .at("adapter")
                                 .at("type") == "artifact" &&
                             operation.at("authoring")
                                 .at("outputs")
                                 .at("adapter")
                                 .at("required") == true &&
                             operation.at("authoring")
                                 .at("outputs")
                                 .at("adapter")
                                 .at("artifact_type") == "opaque" &&
                             operation.at("authoring")
                                 .at("outputs")
                                 .at("adapter")
                                 .at("artifact_schema") ==
                                 "rwkv-lab.posttraining-output.v1"
                       : operation.at("authoring")
                                     .at("outputs")
                                     .at("checkpoint")
                                     .at("type") == "artifact" &&
                             operation.at("authoring")
                                     .at("outputs")
                                     .at("checkpoint")
                                     .at("required") == false &&
                             operation.at("authoring")
                                     .at("outputs")
                                     .at("checkpoint")
                                     .at("artifact_type") == "checkpoint") &&
                  operation.contains("lifecycle") &&
                  operation.contains("training_composition"),
              "each real trainer descriptor must expose honest ports, lifecycle, and slots");
    }
    require(appearance.authoring &&
                appearance.authoring->outputs.size() == 1U &&
                appearance.authoring->outputs.contains("checkpoint") &&
                !appearance.authoring->outputs.contains("eval_gallery") &&
                !appearance.authoring->outputs.contains("log") &&
                !appearance.authoring->outputs.contains("metrics"),
            "real adapter descriptors must not advertise local gallery, log, or metric files as artifact outputs until handlers protocol-publish them");

    nlohmann::json exact_source = load_mageflow_fixture();
    exact_source["spec"].erase("execution");
    exact_source["spec"]["recovery"]["exact_resume"] = false;
    exact_source["spec"]["components"]["mageflow"] = {
        {"adapter", "rwkv-lab.mageflow-appearance-expert"},
        {"version", "1.0.0"},
        {"runtime", "python_worker"},
        {"operations",
         {{"train",
           {{"contract",
             "rwkv_lab.mageflow_appearance_expert.v1.Train"}}}}},
    };
    nlohmann::json acquire = exact_source["spec"]["workflow"]["nodes"]
                                         ["acquire_gpu"];
    acquire["transitions"][0]["target"] = "train_to_boundary";
    nlohmann::json train = exact_source["spec"]["workflow"]["nodes"]
                                       ["train_to_boundary"];
    train["invoke"]["inputs"] = {
        {"config", {{"literal", nlohmann::json::object()}}},
    };
    train["invoke"]["training"] = {
        {"model_family", "mageflow"},
        {"components",
         {
             {"gradient_clipping",
              {{"key",
                {{"category", "gradient_clipping"},
                 {"name", "global_norm"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"learning_rate",
              {{"key",
                {{"category", "learning_rate_schedule"},
                 {"name", "linear_warmup_cosine"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"optimizer",
              {{"key",
                {{"category", "optimizer"},
                 {"name", "torch_adamw"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"parameter_router",
              {{"key",
                {{"category", "parameter_router"},
                 {"name", "mageflow_appearance_expert"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"weight_decay",
              {{"key",
                {{"category", "weight_decay_schedule"},
                 {"name", "constant"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
         }},
    };
    train["publishes"] = {{"checkpoint", "checkpoint"}};
    train["transitions"] = {
        {{"on", "worker.completed"}, {"target", "release_gpu"}},
        {{"on", "operation.failed"}, {"target", "$failed"}},
    };
    nlohmann::json release = exact_source["spec"]["workflow"]["nodes"]
                                         ["release_gpu"];
    exact_source["spec"]["workflow"] = {
        {"entrypoint", "acquire_gpu"},
        {"nodes",
         {{"acquire_gpu", std::move(acquire)},
          {"train_to_boundary", std::move(train)},
          {"release_gpu", std::move(release)}}},
    };
    const trainvm::CompileResult exact_plan =
        trainvm::compile_document(exact_source);
    require(exact_plan.valid(),
            "exact appearance-expert authoring fixture must compile");
    const auto registry_path =
        std::filesystem::temp_directory_path() /
        ("trainvm-exact-operation-" + std::to_string(::getpid()) + ".json");
    {
      std::ofstream output(registry_path, std::ios::binary | std::ios::trunc);
      output << trainvm::encode_json(contract.adapter_registry).dump();
    }
    std::filesystem::permissions(
        registry_path, std::filesystem::perms::owner_read |
                           std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace);
    bool exact_plan_accepted = exact_plan.valid();
    if (exact_plan.valid()) {
      try {
        trainvm::AdapterRegistry::load_file(
            std::filesystem::absolute(registry_path))
            .validate_plan(*exact_plan.plan);
      } catch (const std::exception&) {
        exact_plan_accepted = false;
      }
    }
    std::filesystem::remove(registry_path);
    require(exact_plan_accepted,
            "an executable exact-profile plan must validate against the production rwkv_lab worker registry plus core operations");

    nlohmann::json posttraining_source = exact_source;
    posttraining_source["metadata"]["name"] = "rwkv-posttraining-v1";
    posttraining_source["spec"]["components"]["mageflow"] = {
        {"adapter", "rwkv-lab.rwkv-posttraining"},
        {"version", "1.0.0"},
        {"runtime", "python_worker"},
        {"operations",
         {{"train",
           {{"contract", "rwkv_lab.rwkv_posttraining.v1.Train"}}}}},
    };
    posttraining_source["spec"]["artifacts"]["adapter_bundle"] = {
        {"type", "opaque"},
        {"schema", "rwkv-lab.posttraining-output.v1"},
        {"immutability", "immutable"},
        {"fingerprint", "manifest_sha256"},
    };
    auto& posttraining_node =
        posttraining_source["spec"]["workflow"]["nodes"]
                           ["train_to_boundary"];
    posttraining_node["invoke"]["inputs"] = {
        {"config",
         {{"literal",
           {{"checkpoint", "/thearray/git/moe-mla/fixtures/base.pt"},
            {"data", "/thearray/git/moe-mla/fixtures/sft.jsonl"},
            {"output_dir",
             "/thearray/git/moe-mla/runs/mage-flow-cache-resume"},
            {"steps", 10}}}}},
    };
    posttraining_node["invoke"]["training"] = {
        {"model_family", "rwkv"},
        {"components",
         {
             {"gradient_clipping",
              {{"key",
                {{"category", "gradient_clipping"},
                 {"name", "global_norm"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"learning_rate",
              {{"key",
                {{"category", "learning_rate_schedule"},
                 {"name", "linear_warmup_cosine"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"optimizer",
              {{"key",
                {{"category", "optimizer"},
                 {"name", "torch_adamw_no_decay"},
                 {"version", "2.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
             {"weight_decay",
              {{"key",
                {{"category", "weight_decay_schedule"},
                 {"name", "constant"},
                 {"version", "1.0.0"}}},
               {"configuration", nlohmann::json::object()}}},
         }},
    };
    posttraining_node["publishes"] = {{"adapter", "adapter_bundle"}};
    posttraining_source["spec"]["controls"]["catalog"] =
        nlohmann::json::object();
    posttraining_source["spec"]["recovery"].erase("checkpoint_artifact");
    posttraining_source["spec"]["recovery"].erase(
        "release_accelerators_when_paused");
    const auto posttraining_plan =
        trainvm::compile_document(posttraining_source);
    require(posttraining_plan.valid(),
            "descriptor-backed RWKV post-training fixture must compile");
    {
      std::ofstream output(registry_path, std::ios::binary | std::ios::trunc);
      output << trainvm::encode_json(contract.adapter_registry).dump();
    }
    std::filesystem::permissions(
        registry_path, std::filesystem::perms::owner_read |
                           std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace);
    bool posttraining_accepted = posttraining_plan.valid();
    std::string posttraining_error;
    try {
      if (posttraining_plan.valid()) {
        trainvm::AdapterRegistry::load_file(
            std::filesystem::absolute(registry_path))
            .validate_plan(*posttraining_plan.plan);
      }
    } catch (const std::exception& error) {
      posttraining_accepted = false;
      posttraining_error = error.what();
    }
    require(posttraining_accepted,
            "descriptor-backed RWKV post-training must validate against exact launch authority: " +
                posttraining_error);
    nlohmann::json missing_adapter_output = posttraining_source;
    missing_adapter_output["spec"]["workflow"]["nodes"]
                          ["train_to_boundary"]["publishes"] =
        nlohmann::json::object();
    const auto missing_adapter_plan =
        trainvm::compile_document(missing_adapter_output);
    bool missing_adapter_rejected = !missing_adapter_plan.valid();
    try {
      if (missing_adapter_plan.valid()) {
        trainvm::AdapterRegistry::load_file(
            std::filesystem::absolute(registry_path))
            .validate_plan(*missing_adapter_plan.plan);
      }
    } catch (const std::exception&) {
      missing_adapter_rejected = true;
    }
    std::filesystem::remove(registry_path);
    require(missing_adapter_rejected,
            "RWKV post-training cannot launch without its required immutable adapter output");

    std::vector<trainvm::AdapterProfile> extended_profiles =
        contract.adapter_registry.profiles;
    trainvm::AdapterProfile synthetic = qwen;
    synthetic.key.adapter = "rwkv-lab.synthetic-compatible";
    synthetic.key.contract = "rwkv_lab.synthetic_compatible.v1.Train";
    extended_profiles.push_back(synthetic);
    const trainvm::AdapterRegistry extended_registry(
        std::move(extended_profiles));
    const nlohmann::json extended_document =
        extended_registry.operation_descriptors_json();
    const auto& extended_operations =
        extended_document.at("operations");
    require(extended_operations.size() == 14U &&
                std::ranges::any_of(
                    extended_operations, [](const nlohmann::json& operation) {
                      return operation.at("key").at("adapter") ==
                                 "rwkv-lab.synthetic-compatible" &&
                             operation.at("training_composition")
                                     .at("model_family") == "transformer";
                    }),
            "a compatible newly registered profile must automatically enter the operation descriptor document");

    bool missing_authoring_rejected = false;
    try {
      synthetic.authoring = std::nullopt;
      (void)trainvm::AdapterRegistry({synthetic});
    } catch (const std::invalid_argument& error) {
      missing_authoring_rejected =
          std::string_view(error.what()).find("authoring") !=
          std::string_view::npos;
    }
    require(missing_authoring_rejected,
            "registered profiles missing authoring declarations must fail closed");

    bool primitive_output_rejected = false;
    try {
      synthetic = qwen;
      synthetic.authoring->outputs.at("checkpoint").type =
          trainvm::OperationPortType::string;
      synthetic.authoring->outputs.at("checkpoint").artifact_type =
          std::nullopt;
      (void)trainvm::AdapterRegistry({synthetic});
    } catch (const std::invalid_argument& error) {
      primitive_output_rejected =
          std::string_view(error.what()).find("artifact ports") !=
          std::string_view::npos;
    }
    bool oversized_authoring_rejected = false;
    try {
      synthetic = qwen;
      synthetic.authoring->inputs.clear();
      for (std::size_t index = 0; index < 65U; ++index) {
        synthetic.authoring->inputs.emplace(
            "input_" + std::to_string(index),
            trainvm::OperationPortDescriptor{
                .type = trainvm::OperationPortType::string,
                .required = false,
                .artifact_type = std::nullopt,
                .artifact_schema = std::nullopt,
                .description = std::nullopt,
            });
      }
      (void)trainvm::AdapterRegistry({synthetic});
    } catch (const std::invalid_argument& error) {
      oversized_authoring_rejected =
          std::string_view(error.what()).find("at most 64") !=
          std::string_view::npos;
    }
    require(primitive_output_rejected && oversized_authoring_rejected,
            "operation descriptors must reject primitive publishes and unbounded authoring surfaces");

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
    const auto qwen_runtime = std::ranges::find_if(
        runtimes, [](const auto& runtime) {
          return runtime.adapter == "rwkv-lab.qwen-ao3";
        });
    require(qwen_runtime != runtimes.end(),
            "Qwen runtime must be present before deployment lowering");
    qwen_runtime->code_fingerprint =
        "sha256:" + std::string(64U, 'd');
    qwen_runtime->bootstrap_runtime_closure_fingerprint =
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
                deployment.host_launch_registry.profiles.size() == 13U,
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
            qwen_profile->code_fingerprint == qwen_runtime->code_fingerprint,
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
