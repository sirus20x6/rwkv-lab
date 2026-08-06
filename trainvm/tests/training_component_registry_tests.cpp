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
      .topologies = std::nullopt,
      .post_training = std::nullopt,
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

void a_post_training_arm_reaches_the_worker_and_binds_the_digest() {
  const trainvm::TrainingComponentRegistry registry({adamw(), cosine_schedule()});
  const auto compose = [](std::optional<trainvm::PostTrainingArmDeclaration> arm) {
    return trainvm::TrainingComposition{
        .model_family = "rwkv",
        .components =
            {
                {"optimizer",
                 {.key = adamw().key,
                  .configuration = nlohmann::json::object()}},
            },
        .topologies = std::nullopt,
        .post_training = std::move(arm),
    };
  };
  trainvm::PostTrainingArmDeclaration declared{
      .arm_id = "arm.finetune-a",
      .kind = "supervised_finetune",
      .bounds = {{.kind = "optimizer_steps", .magnitude = 10000U}},
      .reproducibility_claim = "exact",
      .seed = 7U,
      .verifier_identity = std::nullopt,
      .external_mutations = std::nullopt,
      .claims_trajectory_preserving_resume = std::nullopt,
  };

  const auto without = registry.resolve_composition(compose(std::nullopt));
  check(without.post_training.is_null(),
        "a composition declaring no arm carries none");

  const auto with = registry.resolve_composition(compose(declared));
  const nlohmann::json emitted = trainvm::resolved_training_composition_json(with);
  check(emitted.contains("post_training") &&
            emitted["post_training"]["arm_id"] == "arm.finetune-a" &&
            emitted["post_training"]["reproducibility_claim"] == "exact" &&
            emitted["post_training"]["bounds"][0]["kind"] == "optimizer_steps" &&
            emitted["post_training"]["seed"] == 7U,
        "the lowered arm travels to the worker inside the resolved composition");
  check(!emitted["post_training"].contains("verifier_identity") &&
            !emitted["post_training"].contains("external_mutations"),
        "absent optional fields are omitted rather than emitted as null");

  // The digest must separate two compositions that differ only in the arm.
  // Identical components with a different endpoint or a different claim are
  // different experiments, and a digest that could not tell them apart would
  // let one be substituted for the other.
  trainvm::PostTrainingArmDeclaration relabelled = declared;
  relabelled.reproducibility_claim = "none";
  const auto other = registry.resolve_composition(compose(relabelled));
  check(other.composition_digest != with.composition_digest,
        "the composition digest changes when only the arm's claim changes");
  check(without.composition_digest != with.composition_digest,
        "the composition digest changes when an arm is added");

  // The authority repeats compile's check, so a plan cannot reach a worker
  // carrying an arm that compilation would have refused.
  trainvm::PostTrainingArmDeclaration dishonest = declared;
  dishonest.bounds = {{.kind = "wall_clock_seconds", .magnitude = 3600U}};
  bool refused = false;
  try {
    (void)registry.resolve_composition(compose(dishonest));
  } catch (const std::exception&) {
    refused = true;
  }
  check(refused,
        "a wall-clock arm claiming exact reproducibility is refused by the "
        "registry, not only by compilation");
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
      .topologies = std::nullopt,
      .post_training = std::nullopt,
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
        ["learning_rate"] = nlohmann::json{{"nested", 0.1}};
  check(!trainvm::compile_document(source).valid(),
        "nested component configuration is rejected during compilation");
}

void checked_in_component_catalog_matches_native_authority_contract() {
  const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                    "docs/experiment-vm/examples/training-components.v1.json";
  const trainvm::TrainingComponentRegistry registry =
      trainvm::TrainingComponentRegistry::load_file(
          std::filesystem::absolute(path));
  check(registry.document_json().at("components").size() == 53U &&
            registry.registry_digest().starts_with("sha256:") &&
            registry.registry_digest().size() == 71U,
        "checked-in cross-family component catalog is a canonical native authority document");

  const std::string model_path =
      std::filesystem::absolute(TRAINVM_SOURCE_ROOT).string();
  const auto composition_for = [&](std::string policy,
                                   nlohmann::json loader_configuration,
                                   nlohmann::json policy_configuration) {
    return trainvm::TrainingComposition{
        .model_family = "transformer",
        .components =
            {{"model",
              {.key = {.category =
                           trainvm::TrainingComponentCategory::model_loader,
                       .name = "hf_multimodal",
                       .version = "1.0.0"},
               .configuration = std::move(loader_configuration)}},
             {"trainability",
              {.key = {.category =
                           trainvm::TrainingComponentCategory::trainability,
                       .name = std::move(policy),
                       .version = "1.0.0"},
               .configuration = std::move(policy_configuration)}},
             {"checkpoint_policy",
              {.key = {.category =
                           trainvm::TrainingComponentCategory::checkpoint_policy,
                       .name = "atomic_retained",
                       .version = "1.0.0"},
               .configuration = {{"every_steps", 100},
                                 {"keep_last", 2},
                                 {"resume_grade", "exact"}}}}},
        .topologies = std::nullopt,
        .post_training = std::nullopt,
    };
  };
  const nlohmann::json loader_configuration{
      {"model_path", model_path},
      {"checkpoint_fingerprint", "sha256:" + std::string(64U, 'a')}};
  const auto lora_composition = registry.resolve_composition(composition_for(
      "lora", loader_configuration,
      {{"rank", 256},
       {"alpha", 512},
       {"target_selectors",
        nlohmann::json::array(
            {"language_model.layers.*.self_attn.*_proj"})}}));
  check(lora_composition.components.at("trainability")
                .configuration.at("rank") == 256 &&
            lora_composition.components.at("trainability")
                .configuration.at("modules_to_save")
                .empty(),
        "Hugging Face multimodal loading and LoRA resolve without a workload-specific trainer");
  registry.validate_resume_state(
      lora_composition,
      {{"model",
        {{"base_checkpoint_fingerprint", "sha256:" + std::string(64U, 'a')},
         {"load_receipt_digest", "sha256:" + std::string(64U, 'b')}}},
       {"trainability",
        {{"adapter_state_manifest", "sha256:" + std::string(64U, 'c')},
         {"merged", false},
         {"trainable_parameter_manifest",
          "sha256:" + std::string(64U, 'd')}}},
       {"checkpoint_policy",
        {{"last_published_step", 0},
         {"publication_manifest", "sha256:" + std::string(64U, 'e')},
         {"retention_manifest", "sha256:" + std::string(64U, 'f')}}}});
  check(rejects([&] {
          registry.validate_resume_state(
              lora_composition,
              {{"model",
                {{"base_checkpoint_fingerprint",
                  "sha256:" + std::string(64U, 'a')}}},
               {"trainability",
                {{"adapter_state_manifest",
                  "sha256:" + std::string(64U, 'c')},
                 {"merged", false},
                 {"trainable_parameter_manifest",
                  "sha256:" + std::string(64U, 'd')}}}});
        }),
        "incomplete component resume state fails before tensor restoration");
  for (const auto& [policy, configuration] :
       std::vector<std::pair<std::string, nlohmann::json>>{
           {"full", nlohmann::json::object()},
           {"named_rules", {{"unfreeze_patterns",
                              nlohmann::json::array({"language_model.*"})}}},
           {"lora", {{"rank", 256},
                     {"alpha", 512},
                     {"target_selectors",
                      nlohmann::json::array({"language_model.*"})}}}}) {
    auto missing_checkpoint =
        composition_for(policy, loader_configuration, configuration);
    missing_checkpoint.components.erase("checkpoint_policy");
    check(rejects([&] {
            (void)registry.resolve_composition(missing_checkpoint);
          }),
          "every trainable model policy requires an explicit checkpoint policy");
  }
  auto frozen_without_checkpoint = composition_for(
      "frozen", loader_configuration, nlohmann::json::object());
  frozen_without_checkpoint.components.erase("checkpoint_policy");
  check(!rejects([&] {
          (void)registry.resolve_composition(frozen_without_checkpoint);
        }),
        "frozen evaluation-only model compositions may omit checkpoints");
  nlohmann::json quantized = loader_configuration;
  quantized["quantization"] = "4bit";
  check(rejects([&] {
          (void)registry.resolve_composition(
              composition_for("full", quantized, nlohmann::json::object()));
        }),
        "quantized loading is rejected with incompatible full trainability");
  check(rejects([&] {
          nlohmann::json missing = loader_configuration;
          missing["model_path"] =
              "/definitely/missing/trainvm-model-component-test";
          (void)registry.resolve_composition(composition_for(
              "lora", missing,
              {{"rank", 256},
               {"alpha", 512},
               {"target_selectors", nlohmann::json::array({"q_proj"})}}));
        }),
        "missing model assets fail during component resolution");
  check(rejects([&] {
          (void)registry.resolve_composition(composition_for(
              "lora", loader_configuration,
              {{"rank", 256},
               {"alpha", 512},
               {"target_selectors", nlohmann::json::array({"["})}}));
        }),
        "malformed parameter selectors fail during component resolution");
  const auto pipeline_composition = [&] {
    const auto component = [](trainvm::TrainingComponentCategory category,
                              std::string name,
                              nlohmann::json configuration) {
      return trainvm::TrainingComponentSelection{
          .key = {.category = category,
                  .name = std::move(name),
                  .version = "1.0.0"},
          .configuration = std::move(configuration)};
    };
    return trainvm::TrainingComposition{
        .model_family = "transformer",
        .components =
            {{"data",
              component(
                  trainvm::TrainingComponentCategory::data_source,
                  "jsonl_image_caption",
                  {{"manifest_path",
                    (std::filesystem::absolute(TRAINVM_SOURCE_ROOT) /
                     "README.md")
                        .string()},
                   {"image_root", model_path},
                   {"content_fingerprint",
                    "sha256:" + std::string(64U, '1')},
                   {"declared_columns",
                    nlohmann::json::array({"caption", "id", "image"})},
                   {"image_column", "image"},
                   {"caption_columns", nlohmann::json::array({"caption"})},
                   {"id_column", "id"}})},
             {"processor",
              component(
                  trainvm::TrainingComponentCategory::sample_processor,
                  "image_caption",
                  {{"image_column", "image"},
                   {"caption_columns", nlohmann::json::array({"caption"})},
                   {"maximum_pixels", 1048576},
                   {"maximum_edge", 2048}})},
             {"sample_mapping",
              component(
                  trainvm::TrainingComponentCategory::sample_mapper,
                  "assistant_only",
                  {{"fixed_prompt", "Describe the image."},
                   {"target_column", "caption"},
                   {"maximum_tokens", 1024}})},
             {"collation",
              component(trainvm::TrainingComponentCategory::collator,
                        "padded",
                        {{"pad_token_id", 0},
                         {"maximum_sequence_length", 1024}})},
             {"sampler",
              component(trainvm::TrainingComponentCategory::sampler,
                        "deterministic", {{"seed", 17}})},
             {"batching",
              component(trainvm::TrainingComponentCategory::batching,
                        "bucketed",
                        {{"bucket_by", "image_area"},
                         {"bucket_boundaries",
                          nlohmann::json::array({"262144", "1048576"})},
                         {"batch_sizes",
                          nlohmann::json::array({"8", "4", "2"})}})},
             {"split",
              component(trainvm::TrainingComponentCategory::split_selector,
                        "deterministic_holdout",
                        {{"seed", 29}, {"held_out_count", 10}})}},
        .topologies = std::nullopt,
        .post_training = std::nullopt};
  }();
  const auto resolved_pipeline =
      registry.resolve_composition(pipeline_composition);
  check(resolved_pipeline.components.size() == 7U &&
            resolved_pipeline.components.at("sampler")
                    .descriptor.state_grade ==
                trainvm::TrainingStateGrade::exact &&
            resolved_pipeline.components.at("batching")
                    .descriptor.state_grade ==
                trainvm::TrainingStateGrade::exact &&
            resolved_pipeline.components.at("split")
                    .descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless,
        "a complete declarative image-caption pipeline resolves without trainer code");
  registry.validate_resume_state(
      resolved_pipeline,
      {{"data",
        {{"content_fingerprint", "sha256:" + std::string(64U, '1')},
         {"cursor", 12}}},
       {"sampler",
        {{"cursor", 3},
         {"epoch", 0},
         {"order_digest", "sha256:" + std::string(64U, '2')},
         {"rng_state_digest", "sha256:" + std::string(64U, '3')}}},
       {"batching",
        {{"batches_emitted", 2},
         {"pending_sample_ids", nlohmann::json::array({"image-1"})},
         {"bucket_assignment_digest",
          "sha256:" + std::string(64U, '4')}}}});
  check(rejects([&] {
          auto incomplete = pipeline_composition;
          incomplete.components.erase("collation");
          (void)registry.resolve_composition(incomplete);
        }),
        "partial declarative data pipelines fail closed before worker launch");
  check(rejects([&] {
          auto incompatible = pipeline_composition;
          incompatible.components.at("batching").configuration["bucket_by"] =
              "token_length";
          incompatible.components.at("processor").key.name = "token_ids";
          incompatible.components.at("processor").configuration =
              {{"token_column", "caption"},
               {"maximum_tokens", 1024},
               {"vocabulary_size", 152064}};
          (void)registry.resolve_composition(incompatible);
        }),
        "incompatible source and processor modalities fail during native resolution");
  auto evaluation_composition = pipeline_composition;
  const auto selection = [](trainvm::TrainingComponentCategory category,
                            std::string name,
                            nlohmann::json configuration) {
    return trainvm::TrainingComponentSelection{
        .key = {.category = category,
                .name = std::move(name),
                .version = "1.0.0"},
        .configuration = std::move(configuration)};
  };
  evaluation_composition.components.emplace(
      "evaluation_split",
      selection(trainvm::TrainingComponentCategory::split_selector,
                "deterministic_holdout",
                {{"seed", 29},
                 {"held_out_count", 10},
                 {"selection", "held_out"}}));
  evaluation_composition.components.emplace(
      "evaluator",
      selection(trainvm::TrainingComponentCategory::evaluator,
                "scalar_loss", {{"metrics", nlohmann::json::array({"loss"})}}));
  evaluation_composition.components.emplace(
      "evaluation_schedule",
      selection(trainvm::TrainingComponentCategory::evaluation_schedule,
                "launch_gate_periodic",
                {{"launch_gate_examples", 10}, {"full_every_steps", 250}}));
  evaluation_composition.components.emplace(
      "qualitative_samples",
      selection(trainvm::TrainingComponentCategory::qualitative_sample,
                "fixed_held_out",
                {{"identity_field", "id"},
                 {"identities_digest", "sha256:" + std::string(64U, 'e')},
                 {"selector_digest", "sha256:" + std::string(64U, 'f')},
                 {"sample_count", 10}}));
  evaluation_composition.components.emplace(
      "artifact_renderer",
      selection(trainvm::TrainingComponentCategory::artifact_renderer,
                "caption_triplet", nlohmann::json::object()));
  evaluation_composition.components.emplace(
      "checkpoint_policy",
      selection(trainvm::TrainingComponentCategory::checkpoint_policy,
                "atomic_retained",
                {{"every_steps", 250},
                 {"keep_last", 3},
                 {"resume_grade", "exact"}}));
  const auto evaluation = registry.resolve_composition(evaluation_composition);
  check(evaluation.components.at("evaluator")
                .configuration.at("split_slot") == "evaluation_split" &&
            evaluation.components.at("evaluation_schedule")
                .configuration.at("full_step_zero") == true &&
            evaluation.components.at("checkpoint_policy")
                .descriptor.state_grade == trainvm::TrainingStateGrade::exact,
        "evaluation and checkpoint policies resolve against one deterministic held-out split");
  check(rejects([&] {
          auto incomplete = evaluation_composition;
          incomplete.components.erase("artifact_renderer");
          (void)registry.resolve_composition(incomplete);
        }),
        "partial evaluation suites fail before worker launch");
  check(rejects([&] {
          auto training_view = evaluation_composition;
          training_view.components.at("evaluation_split")
              .configuration["selection"] = "train";
          (void)registry.resolve_composition(training_view);
        }),
        "an evaluator cannot consume the training split view");
  check(rejects([&] {
          auto no_final_evidence = evaluation_composition;
          no_final_evidence.components.at("evaluation_schedule")
              .configuration["final"] = false;
          (void)registry.resolve_composition(no_final_evidence);
        }),
        "a training evaluation suite cannot disable final evidence");
  check(rejects([&] {
          auto changed_partition = evaluation_composition;
          changed_partition.components.at("evaluation_split")
              .configuration["seed"] = 30;
          (void)registry.resolve_composition(changed_partition);
        }),
        "training and evaluation views cannot name different deterministic partitions");
  check(rejects([&] {
          auto too_small = evaluation_composition;
          too_small.components.at("qualitative_samples")
              .configuration["sample_count"] = 9;
          (void)registry.resolve_composition(too_small);
        }),
        "legacy step-zero launch evidence count cannot diverge from the fixed held-out identity set");
  auto exact_plan_result = trainvm::compile_document(experiment_fixture());
  if (exact_plan_result.plan) {
    auto exact_plan = *exact_plan_result.plan;
    auto weaker = evaluation_composition;
    weaker.components.at("checkpoint_policy")
        .configuration["resume_grade"] = "compatible";
    exact_plan.experiment.spec.workflow.nodes.at("train_to_boundary")
        .invoke.training = std::move(weaker);
    check(rejects([&] { registry.validate_plan(exact_plan); }),
          "exact-resume plans reject a checkpoint policy that declares a weaker grade");
  } else {
    check(false, "exact-resume checkpoint policy fixture compiles");
  }
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
  const auto full_backbone_router = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::parameter_router,
              .name = "mageflow_full_backbone",
              .version = "1.0.0"},
      .model_family = "mageflow",
      .configuration = nlohmann::json::object(),
  });
  check(full_backbone_router.descriptor.implementation ==
                "rwkv_lab.parameter_router.mageflow_full_backbone.v1" &&
            full_backbone_router.configuration.empty() &&
            full_backbone_router.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless,
        "full-backbone ownership resolves as one closed empty configuration");
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
  const auto vision_precision = registry.resolve({
      .key = {.category = trainvm::TrainingComponentCategory::precision,
              .name = "fp32_parameters_bf16_compute",
              .version = "1.0.0"},
      .model_family = "vision",
      .configuration = nlohmann::json::object(),
  });
  check(vision_precision.descriptor.implementation ==
            "rwkv_lab.precision.fp32_parameters_bf16_compute.v1" &&
            vision_precision.configuration.at("parameter_dtype") == "float32" &&
            vision_precision.configuration.at("compute_dtype") == "bfloat16" &&
            vision_precision.configuration.at("reduction_dtype") == "float32" &&
            vision_precision.configuration.at("gradient_scaling") == false,
        "vision FP32 parameters and BF16 compute resolve as a truthful independent precision policy");
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
  const auto constant_schedule = registry.resolve({
      .key = {
          .category =
              trainvm::TrainingComponentCategory::learning_rate_schedule,
          .name = "constant",
          .version = "1.0.0"},
      .model_family = "vision",
      .configuration = nlohmann::json::object(),
  });
  check(constant_schedule.descriptor.implementation ==
            "rwkv_lab.schedule.constant.v1" &&
            constant_schedule.descriptor.state_grade ==
                trainvm::TrainingStateGrade::stateless &&
            constant_schedule.descriptor.step_domain ==
                trainvm::StepDomain::optimizer_step &&
            constant_schedule.configuration.empty(),
        "constant learning rate is an independent stateless vision-compatible schedule");
  const auto warmup_constant = registry.resolve({
      .key = {
          .category =
              trainvm::TrainingComponentCategory::learning_rate_schedule,
          .name = "linear_warmup_constant",
          .version = "1.0.0",
      },
      .model_family = "rwkv",
      .configuration = {{"warmup_steps", 10}},
  });
  check(warmup_constant.descriptor.implementation ==
                "rwkv_lab.schedule.linear_warmup_constant.v1" &&
            warmup_constant.descriptor.state_grade ==
                trainvm::TrainingStateGrade::exact &&
            warmup_constant.configuration.at("warmup_steps") == 10,
        "linear warmup to a constant rate is an exact independent RLVR schedule");
}

}  // namespace

int main() {
  try {
    registry_is_canonical_and_resolves_typed_configuration();
    invalid_descriptors_and_requests_fail_closed();
    compositions_extend_worker_authority();
  a_post_training_arm_reaches_the_worker_and_binds_the_digest();
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
