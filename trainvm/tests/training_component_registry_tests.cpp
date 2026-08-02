#include "trainvm/training_component_registry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "trainvm/reflection_json.hpp"
#include "trainvm/document.hpp"

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

trainvm::TrainingComponentField number_field(
    std::string name, bool required, double default_value, double minimum,
    double maximum) {
  return {
      .name = std::move(name),
      .type = trainvm::TrainingValueType::number,
      .required = required,
      .default_value = default_value,
      .minimum = minimum,
      .maximum = maximum,
      .values = std::nullopt,
      .unit = "ratio",
      .description = std::nullopt,
  };
}

trainvm::TrainingComponentField state_field(
    std::string name, trainvm::TrainingValueType type) {
  return {
      .name = std::move(name),
      .type = type,
      .required = true,
      .default_value = std::nullopt,
      .minimum = std::nullopt,
      .maximum = std::nullopt,
      .values = std::nullopt,
      .unit = std::nullopt,
      .description = std::nullopt,
  };
}

trainvm::TrainingComponentDescriptor adamw() {
  return {
      .key = {.category = trainvm::TrainingComponentCategory::optimizer,
              .name = "adamw",
              .version = "1.0.0"},
      .backend = trainvm::TrainingComponentBackend::python,
      .implementation = "torch.optim.AdamW",
      .model_families = {"transformer", "mageflow", "rwkv"},
      .required_capabilities = {"optimizer.state_dict"},
      .configuration = {
          number_field("weight_decay", false, 0.01, 0.0, 1.0),
          number_field("learning_rate", true, 0.0001, 0.0, 1.0),
      },
      .state = {state_field("parameter_state",
                            trainvm::TrainingValueType::string)},
      .step_domain = std::nullopt,
      .state_grade = trainvm::TrainingStateGrade::exact,
      .reference_implementation = true,
  };
}

trainvm::TrainingComponentDescriptor cosine_schedule() {
  return {
      .key = {
          .category =
              trainvm::TrainingComponentCategory::learning_rate_schedule,
          .name = "cosine_with_warmup",
          .version = "1.0.0"},
      .backend = trainvm::TrainingComponentBackend::native,
      .implementation = "trainvm.schedule.cosine_with_warmup",
      .model_families = {"*"},
      .required_capabilities = {},
      .configuration = {
          number_field("minimum_ratio", false, 0.0, 0.0, 1.0),
      },
      .state = {state_field("cursor",
                            trainvm::TrainingValueType::integer)},
      .step_domain = trainvm::StepDomain::optimizer_step,
      .state_grade = trainvm::TrainingStateGrade::exact,
      .reference_implementation = true,
  };
}

trainvm::TrainingComponentDescriptor silu() {
  return {
      .key = {.category = trainvm::TrainingComponentCategory::activation,
              .name = "silu",
              .version = "1.0.0"},
      .backend = trainvm::TrainingComponentBackend::runtime_builtin,
      .implementation = "runtime.activation.silu",
      .model_families = {"*"},
      .required_capabilities = {},
      .configuration = {},
      .state = {},
      .step_domain = std::nullopt,
      .state_grade = trainvm::TrainingStateGrade::stateless,
      .reference_implementation = true,
  };
}

template <typename Function>
bool rejects(Function&& function) {
  try {
    function();
    return false;
  } catch (const trainvm::TrainingComponentResolutionError&) {
    return true;
  }
}

nlohmann::json experiment_fixture() {
  const std::filesystem::path path =
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/mageflow-cache-resume.json";
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open experiment fixture");
  return nlohmann::json::parse(input);
}

void registry_is_canonical_and_resolves_typed_configuration() {
  trainvm::TrainingComponentRegistry registry(
      {silu(), cosine_schedule(), adamw()});
  trainvm::TrainingComponentRegistry reordered(
      {adamw(), silu(), cosine_schedule()});
  check(registry.registry_digest() == reordered.registry_digest(),
        "registry digest is independent of declaration order");
  const auto resolved = registry.resolve({
      .key = adamw().key,
      .model_family = "mageflow",
      .configuration = {{"learning_rate", 0.000002}},
  });
  check(resolved.configuration ==
            nlohmann::json{{"learning_rate", 0.000002},
                           {"weight_decay", 0.01}},
        "resolution applies typed defaults without trainer-wide switches");
  check(resolved.descriptor.configuration.front().name == "learning_rate" &&
            resolved.descriptor.configuration.back().name == "weight_decay",
        "descriptor fields are stored in canonical name order");
  check(resolved.descriptor_digest.starts_with("sha256:") &&
            resolved.descriptor_digest.size() == 71U,
        "resolved component carries a content-addressed descriptor identity");

  const nlohmann::json document{
      {"api_version", "trainvm.training-components/v1"},
      {"components", registry.descriptors_json()},
  };
  const auto decoded =
      trainvm::TrainingComponentRegistry::from_json(document.dump());
  check(decoded.registry_digest() == registry.registry_digest(),
        "reflected registry JSON round trips to the same authority identity");
  check(registry.document_json() == document &&
            "sha256:" + trainvm::sha256_hex(
                            registry.document_json().dump()) ==
                registry.registry_digest(),
        "descriptor document bytes carry the exact registry identity");

  const trainvm::TrainingComposition composition{
      .model_family = "mageflow",
      .components = {
          {"optimizer",
           {.key = adamw().key,
            .configuration = {{"learning_rate", 0.000002}}}},
          {"schedule",
           {.key = cosine_schedule().key,
            .configuration = nlohmann::json::object()}},
      },
  };
  const auto resolved_composition =
      registry.resolve_composition(composition);
  check(resolved_composition.components.size() == 2U &&
            resolved_composition.registry_digest ==
                registry.registry_digest() &&
            resolved_composition.composition_digest.starts_with("sha256:") &&
            resolved_composition.composition_digest.size() == 71U,
        "composition resolves independent slots into one locked identity");
}

void invalid_descriptors_and_requests_fail_closed() {
  check(rejects([] {
          auto schedule = cosine_schedule();
          schedule.step_domain.reset();
          trainvm::TrainingComponentRegistry invalid({std::move(schedule)});
        }),
        "schedule without an explicit step domain is rejected");
  check(rejects([] {
          auto activation = silu();
          activation.state.push_back(state_field(
              "unexpected", trainvm::TrainingValueType::integer));
          trainvm::TrainingComponentRegistry invalid({std::move(activation)});
        }),
        "stateless descriptor with checkpoint state is rejected");
  check(rejects([] {
          auto duplicate = adamw();
          trainvm::TrainingComponentRegistry invalid(
              {duplicate, std::move(duplicate)});
        }),
        "duplicate exact component keys are rejected");
  check(rejects([] {
          auto ambiguous = silu();
          ambiguous.model_families = {"*", "mageflow"};
          trainvm::TrainingComponentRegistry invalid({std::move(ambiguous)});
        }),
        "wildcard family compatibility cannot hide redundant family entries");

  const trainvm::TrainingComponentRegistry registry({adamw()});
  check(rejects([&] {
          (void)registry.resolve({
              .key = adamw().key,
              .model_family = "vision",
              .configuration = {{"learning_rate", 0.1}},
          });
        }),
        "incompatible model family is rejected");
  check(rejects([&] {
          (void)registry.resolve({
              .key = adamw().key,
              .model_family = "rwkv",
              .configuration = {{"learning_rate", 2.0}},
          });
        }),
        "out-of-range configuration is rejected");
  check(rejects([&] {
          (void)registry.resolve({
              .key = adamw().key,
              .model_family = "transformer",
              .configuration = {{"learning_rate", 0.1},
                                {"trainer_switch", true}},
          });
        }),
        "unknown trainer-wide switch is rejected");
  check(rejects([] {
          (void)trainvm::TrainingComponentRegistry::from_json(
              R"({"api_version":"trainvm.training-components/v1","api_version":"wrong","components":[]})");
        }),
        "duplicate registry JSON keys are rejected");
}

void compositions_extend_worker_authority() {
  const trainvm::TrainingComponentRegistry registry(
      {adamw(), cosine_schedule()});
  const trainvm::TrainingComposition composition{
      .model_family = "mageflow",
      .components = {
          {"optimizer",
           {.key = adamw().key,
            .configuration = {{"learning_rate", 0.000002}}}},
          {"schedule",
           {.key = cosine_schedule().key,
            .configuration = nlohmann::json::object()}},
      },
  };
  const auto augmented = registry.augment_worker_launch_request(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"adapter.base", "optimizer.state_dict"}},
      composition);
  check(augmented.required_capabilities ==
            std::vector<std::string>{"adapter.base",
                                     "optimizer.state_dict"},
        "component capabilities are merged, sorted, and deduplicated into worker authority");
  const auto unchanged = registry.augment_worker_launch_request(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"adapter.base"}},
      std::nullopt);
  check(unchanged.required_capabilities ==
            std::vector<std::string>{"adapter.base"},
        "nodes without a composition retain their adapter-only worker authority");
}

void experiment_composition_is_reflected_and_bounded() {
  nlohmann::json source = experiment_fixture();
  source["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
        ["training"] = {
      {"model_family", "mageflow"},
      {"components",
       {{"optimizer",
         {{"key",
           {{"category", "optimizer"},
            {"name", "adamw"},
            {"version", "1.0.0"}}},
          {"configuration", {{"learning_rate", 0.000002}}}}}}},
  };
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid() && compiled.plan &&
            compiled.plan->experiment.spec.workflow.nodes
                .at("train_to_boundary")
                .invoke.training &&
            compiled.plan->canonical_plan["spec"]["workflow"]["nodes"]
                                         ["train_to_boundary"]["invoke"]
                                         .contains("training"),
        "training composition participates in reflected canonical plans");
  if (compiled.valid() && compiled.plan) {
    const auto& composition = *compiled.plan->experiment.spec.workflow.nodes
                                   .at("train_to_boundary")
                                   .invoke.training;
    const trainvm::TrainingComponentRegistry registry({adamw()});
    const auto resolved = registry.resolve_composition(composition);
    const std::string lock = registry.plan_lock_manifest(*compiled.plan);
    const nlohmann::json submission{
        {"training_component_lock_digest",
         registry.plan_lock_digest(*compiled.plan)},
        {"training_component_lock", nlohmann::json::parse(lock)},
    };
    check(resolved
                  .components.at("optimizer")
                  .configuration.at("learning_rate") == 0.000002,
          "compiled composition resolves through the authority registry");
    registry.validate_submission_lock(*compiled.plan, submission);
    check(registry.plan_uses_components(*compiled.plan) &&
              nlohmann::json::parse(lock).at("nodes").contains(
                  "train_to_boundary"),
          "plan lock freezes every resolved node composition");
    check(rejects([&] {
            nlohmann::json changed = submission;
            changed["training_component_lock_digest"] =
                "sha256:" + std::string(64U, '0');
            registry.validate_submission_lock(*compiled.plan, changed);
          }),
          "training-component submission lock rejects registry drift");

    auto compatible_optimizer = adamw();
    compatible_optimizer.state_grade =
        trainvm::TrainingStateGrade::compatible;
    const trainvm::TrainingComponentRegistry compatible_registry(
        {std::move(compatible_optimizer)});
    check(rejects([&] {
            compatible_registry.validate_plan(*compiled.plan);
          }),
          "exact-resume workflows reject compatibility-grade component state");

    auto builtin_plan = *compiled.plan;
    auto& builtin_node = builtin_plan.experiment.spec.workflow.nodes.at(
        "train_to_boundary");
    builtin_node.invoke.component = "core";
    builtin_node.invoke.operation = "acquire_resources";
    builtin_node.effect = trainvm::Effect::resource;
    check(rejects([&] { registry.validate_plan(builtin_plan); }),
          "training compositions cannot be attached to builtin resource operations");
  }

  source["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
        ["training"]["components"]["optimizer"]["configuration"]
        ["learning_rate"] = nlohmann::json::array({0.1});
  check(!trainvm::compile_document(source).valid(),
        "non-scalar component configuration is rejected during compilation");
}

void checked_in_component_catalog_matches_native_authority_contract() {
  const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                    "docs/experiment-vm/examples/training-components.v1.json";
  const trainvm::TrainingComponentRegistry registry =
      trainvm::TrainingComponentRegistry::load_file(
          std::filesystem::absolute(path));
  check(registry.document_json().at("components").size() == 17U &&
            registry.registry_digest().starts_with("sha256:") &&
            registry.registry_digest().size() == 71U,
        "checked-in cross-family component catalog is a canonical native authority document");
  const auto optimizer = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::optimizer,
              .name = "fp32_master_adamw",
              .version = "1.0.0"},
      .model_family = "mageflow",
      .configuration = {{"learning_rate", 0.00001}},
  });
  check(optimizer.descriptor.implementation ==
            "rwkv_lab.optimizer.fp32_master_adamw.v1" &&
            optimizer.configuration.at("beta1") == 0.9 &&
            optimizer.configuration.at("foreach") == true &&
            optimizer.configuration.at("fused") == false &&
            optimizer.descriptor.required_capabilities ==
                std::vector<std::string>{
                    "optimizer.fp32_master_adamw.v1"},
        "native resolution and the closed PyTorch implementation share exact identities and defaults");
  check(rejects([&] {
          (void)registry.resolve({
              .key = optimizer.descriptor.key,
              .model_family = "transformer",
              .configuration = {{"learning_rate", 0.00001}},
          });
        }),
        "MageFlow-only FP32-master optimizer cannot route to a transformer family");
  const auto router = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::parameter_router,
              .name = "mageflow_terminal_expert",
              .version = "1.0.0"},
      .model_family = "mageflow",
      .configuration = nlohmann::json::object(),
  });
  check(router.descriptor.implementation ==
            "rwkv_lab.parameter_router.mageflow_terminal_expert.v1" &&
            router.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            router.configuration.at("shared_backbone_multiplier") == 0.5 &&
            router.configuration.at("repa_projection_multiplier") == 1.0,
        "terminal expert ownership and LR multipliers resolve as one stateless component");
  const auto clipping = registry.resolve({
      .key = {
          .category = trainvm::TrainingComponentCategory::gradient_clipping,
          .name = "global_norm",
          .version = "1.0.0"},
      .model_family = "transformer",
      .configuration = {{"max_norm", 1.0},
                        {"error_if_nonfinite", true}},
  });
  check(clipping.descriptor.implementation ==
            "rwkv_lab.gradient_clipping.global_norm.v1" &&
            clipping.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            clipping.configuration ==
                nlohmann::json{{"error_if_nonfinite", true},
                               {"max_norm", 1.0},
                               {"norm_type", 2.0}},
        "global-norm clipping resolves independently of optimizer and schedule policy");
  const auto accumulation = registry.resolve({
      .key = {
          .category =
              trainvm::TrainingComponentCategory::gradient_accumulation,
          .name = "fixed",
          .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = {{"microbatches_per_optimizer_step", 4}},
  });
  check(accumulation.descriptor.implementation ==
            "rwkv_lab.gradient_accumulation.fixed.v1" &&
            accumulation.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            accumulation.descriptor.step_domain ==
                trainvm::StepDomain::microbatch &&
            accumulation.configuration.at(
                "microbatches_per_optimizer_step") == 4,
        "fixed optimizer-step accumulation resolves independently of clipping and optimizer policy");
  const auto objective = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::objective,
              .name = "linear_head_cross_entropy",
              .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = nlohmann::json::object(),
  });
  check(objective.descriptor.implementation ==
            "rwkv_lab.objective.linear_head_cross_entropy.v1" &&
            objective.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            objective.configuration.at("chunk_size") == 2048 &&
            objective.configuration.at("prefer_fused") == true,
        "linear-head language objective resolves independently of model topology and optimizer policy");
  const auto precision = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::precision,
              .name = "bf16_parameters_fp32_reductions",
              .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = nlohmann::json::object(),
  });
  check(precision.descriptor.implementation ==
            "rwkv_lab.precision.bf16_parameters_fp32_reductions.v1" &&
            precision.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            precision.configuration.at("parameter_dtype") == "bfloat16" &&
            precision.configuration.at("reduction_dtype") == "float32" &&
            precision.configuration.at("gradient_scaling") == false,
        "BF16 parameter/compute and FP32 reduction policy resolves independently of optimizer state");
  const auto activation = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::activation,
              .name = "silu",
              .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = nlohmann::json::object(),
  });
  check(activation.descriptor.implementation ==
            "rwkv_lab.activation.silu.v1" &&
            activation.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            activation.configuration.empty(),
        "activation identity resolves independently of family topology installation");
  const auto normalization = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::normalization,
              .name = "layer_norm",
              .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = nlohmann::json::object(),
  });
  check(normalization.descriptor.implementation ==
            "rwkv_lab.normalization.layer_norm.v1" &&
            normalization.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            normalization.configuration.at("epsilon") == 1.0e-5,
        "normalization construction resolves independently of family topology installation");
  const auto curriculum = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::curriculum,
              .name = "context_length",
              .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = {{"maximum_sequence_length", 1024},
                        {"base_batch_size", 8}},
  });
  check(curriculum.descriptor.implementation ==
            "rwkv_lab.curriculum.context_length.v1" &&
            curriculum.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            curriculum.descriptor.step_domain ==
                trainvm::StepDomain::optimizer_step &&
            curriculum.configuration.at("stages") == "",
        "context curriculum resolves independently from optimizer and topology state");
  const auto no_decay_optimizer = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::optimizer,
              .name = "torch_adamw_no_decay",
              .version = "2.0.0"},
      .model_family = "rwkv",
      .configuration = {{"learning_rate", 0.0003}},
  });
  const auto decay = registry.resolve({
      .key = {
          .category =
              trainvm::TrainingComponentCategory::weight_decay_schedule,
          .name = "constant",
          .version = "1.0.0"},
      .model_family = "rwkv",
      .configuration = {{"weight_decay", 0.1}},
  });
  check(!no_decay_optimizer.configuration.contains("weight_decay") &&
            no_decay_optimizer.descriptor.implementation ==
                "rwkv_lab.optimizer.torch_adamw_no_decay.v2" &&
            decay.configuration == nlohmann::json{{"weight_decay", 0.1}} &&
            decay.descriptor.step_domain ==
                trainvm::StepDomain::optimizer_step,
        "optimizer mechanics and weight-decay policy resolve as independent components");
  const auto powercool = registry.resolve({
      .key = {
          .category =
              trainvm::TrainingComponentCategory::learning_rate_schedule,
          .name = "powercool",
          .version = "1.0.0"},
      .model_family = "transformer",
      .configuration = {{"warmup_steps", 100}, {"max_steps", 10'000}},
  });
  check(powercool.descriptor.implementation ==
            "rwkv_lab.schedule.powercool.v1" &&
            powercool.descriptor.step_domain ==
                trainvm::StepDomain::optimizer_step &&
            powercool.configuration.at("minimum_ratio") == 0.0 &&
            powercool.configuration.at("cooldown_fraction") == 0.2 &&
            powercool.configuration.at("power") == 2.0,
        "RWKV and transformer PowerCool schedules share one exact reflected contract");
}

}  // namespace

int main() {
  try {
    registry_is_canonical_and_resolves_typed_configuration();
    invalid_descriptors_and_requests_fail_closed();
    compositions_extend_worker_authority();
    experiment_composition_is_reflected_and_bounded();
    checked_in_component_catalog_matches_native_authority_contract();
  } catch (const std::exception& exception) {
    std::cerr << "UNCAUGHT: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " training component registry test(s) failed\n";
    return 1;
  }
  std::cout << "training component registry tests passed\n";
  return 0;
}
