#include "trainvm/recipe_profile.hpp"
#include "trainvm/training_component_registry.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
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

trainvm::RecipeOverrideField path_field() {
  return {
      .name = "data.source_config",
      .domain = trainvm::RecipeOverrideDomain::data,
      .type = trainvm::RecipeValueType::path,
      .target = "/spec/parameters/source_config/value",
      .required = true,
      .minimum = std::nullopt,
      .maximum = std::nullopt,
      .values = std::nullopt,
      .description = "Frozen source configuration",
  };
}

trainvm::RecipeOverrideField target_step_field() {
  return {
      .name = "training.target_step",
      .domain = trainvm::RecipeOverrideDomain::hyperparameters,
      .type = trainvm::RecipeValueType::integer,
      .target = "/spec/parameters/target_step/value",
      .required = false,
      .minimum = 1.0,
      .maximum = 100000.0,
      .values = std::nullopt,
      .description = std::nullopt,
  };
}

trainvm::RecipeOverrideField precision_field() {
  return {
      .name = "training.precision",
      .domain = trainvm::RecipeOverrideDomain::controls,
      .type = trainvm::RecipeValueType::enumeration,
      .target = "/spec/controls/catalog/mixed_precision/default",
      .required = false,
      .minimum = std::nullopt,
      .maximum = std::nullopt,
      .values = std::vector<nlohmann::json>{"fp16", "bf16"},
      .description = std::nullopt,
  };
}

trainvm::RecipeOverrideField memory_field() {
  return {
      .name = "resources.memory_gib",
      .domain = trainvm::RecipeOverrideDomain::resources,
      .type = trainvm::RecipeValueType::number,
      .target = "/spec/resources/accelerators/minimum_memory_gib",
      .required = false,
      .minimum = 24.0,
      .maximum = 96.0,
      .values = std::nullopt,
      .description = std::nullopt,
  };
}

trainvm::RecipeProfile profile() {
  return {
      .key = {.name = "mageflow_cache_resume", .version = "1"},
      .description = "Test profile",
      .template_document = experiment_fixture(),
      .overrides = {
          precision_field(), memory_field(), target_step_field(), path_field()},
      .content_bindings = std::nullopt,
      .compatibility = std::vector<trainvm::RecipeCompatibilityRule>{
          {
              .fields = {"training.precision", "resources.memory_gib"},
              .allowed = {{"bf16", 48.0}, {"fp16", 96.0}},
              .description = "Fixture precision/memory qualification",
          },
      },
  };
}

trainvm::RecipeInstance instance() {
  return {
      .api_version = "trainvm.recipe-instance/v1",
      .recipe = {.name = "mageflow_cache_resume", .version = "1"},
      .run_identity = "recipe-profile-test-a",
      .overrides = {
          {"data.source_config",
           "/thearray/git/moe-mla/experiments/mageflow_terminal_repa_fixed_v2.json"},
      },
  };
}

template <typename Function>
bool rejects(Function&& function) {
  try {
    function();
    return false;
  } catch (const trainvm::RecipeProfileError&) {
    return true;
  }
}

void profiles_are_reflected_versioned_and_canonical() {
  trainvm::RecipeProfileRegistry registry({profile()});
  check(registry.registry_digest().starts_with("sha256:") &&
            registry.registry_digest().size() == 71U,
        "registry has a content digest");
  check(registry.profile_digest(profile().key).starts_with("sha256:") &&
            registry.profile_digest(profile().key).size() == 71U,
        "exact recipe version has a content digest");

  const nlohmann::json document = registry.document_json();
  const auto decoded =
      trainvm::RecipeProfileRegistry::from_json(document.dump());
  check(decoded.registry_digest() == registry.registry_digest(),
        "reflected registry schema round trips to the same identity");

  auto reordered = profile();
  std::ranges::reverse(reordered.overrides);
  trainvm::RecipeProfileRegistry reordered_registry({std::move(reordered)});
  check(reordered_registry.registry_digest() == registry.registry_digest(),
        "declaration order does not change the registry identity");
}

void expansion_preserves_ordinary_plan_identity_and_provenance() {
  const trainvm::RecipeProfileRegistry registry({profile()});
  const auto expanded = registry.expand(instance());
  check(expanded.expanded_plan_digest == "sha256:" + expanded.plan.plan_hash,
        "expanded plan has a namespaced digest");
  check(expanded.profile_digest == registry.profile_digest(expanded.recipe),
        "expansion pins the exact profile digest");
  check(expanded.provenance.size() > 40U,
        "every scalar canonical plan value receives provenance");
  check(expanded.provenance.at("/spec/parameters/source_config/value").kind ==
            "instance_override",
        "supplied value names its instance override source");
  check(expanded.provenance.at("/kind").kind == "recipe_template",
        "unchanged authority value names its exact recipe template source");

  const auto ordinary =
      trainvm::compile_document(expanded.plan.canonical_plan);
  check(ordinary.valid() && ordinary.plan->plan_hash == expanded.plan.plan_hash,
        "fully expanded documents retain the same existing plan identity");

  const auto expanded_from_json = registry.expand_json(
      nlohmann::json{{"api_version", "trainvm.recipe-instance/v1"},
                     {"recipe", {{"name", "mageflow_cache_resume"},
                                  {"version", "1"}}},
                     {"run_identity", "recipe-profile-test-a"},
                     {"overrides",
                      {{"data.source_config",
                        "/thearray/git/moe-mla/experiments/"
                        "mageflow_terminal_repa_fixed_v2.json"}}}});
  check(expanded_from_json.instance_digest == expanded.instance_digest,
        "reflected recipe instances have canonical identities");
  check(expanded.plan.canonical_plan.at("metadata").at("name") ==
            "recipe-profile-test-a" &&
            expanded.plan.canonical_plan.at("spec").at("workspace").at(
                "run_directory") ==
                "/thearray/git/moe-mla/runs/recipe-profile-test-a" &&
            expanded.plan.canonical_plan.at("spec").at("workspace").at(
                "concurrency_key") ==
                "local-gpu-training.recipe-profile-test-a",
        "one run identity atomically derives name, output, and concurrency");
  check(expanded.provenance.at("/metadata/name").kind ==
            "instance_run_identity" &&
            expanded.provenance.at(
                "/spec/workspace/allowed_write_roots/0").kind ==
                "instance_run_identity",
        "derived output authority has explicit submission provenance");
}

void invalid_and_ambiguous_authoring_fails_closed() {
  const trainvm::RecipeProfileRegistry registry({profile()});
  check(rejects([&] {
          auto value = instance();
          value.overrides.emplace("python", "evil.module");
          (void)registry.expand(value);
        }),
        "unknown overrides are rejected");
  check(rejects([&] {
          auto value = instance();
          value.run_identity = "../../other-run";
          (void)registry.expand(value);
        }),
        "run identity cannot inject or escape a run directory");
  check(rejects([&] {
          auto value = instance();
          value.overrides["training.target_step"] = "many";
          (void)registry.expand(value);
        }),
        "wrongly typed overrides are rejected");
  check(rejects([&] {
          auto value = instance();
          value.overrides["training.target_step"] = 100001;
          (void)registry.expand(value);
        }),
        "out-of-bound overrides are rejected");
  check(rejects([&] {
          auto value = instance();
          value.overrides.clear();
          (void)registry.expand(value);
        }),
        "missing required overrides are rejected");
  check(rejects([&] {
          auto value = instance();
          value.overrides["data.source_config"] = "/etc/passwd";
          (void)registry.expand(value);
        }),
        "path overrides cannot escape authority-owned read roots");
  check(rejects([&] {
          auto value = instance();
          value.overrides["training.precision"] = "fp16";
          (void)registry.expand(value);
        }),
        "incompatible finite component combinations are rejected");

  check(rejects([] {
          auto unsafe = profile();
          unsafe.overrides.front().target = "/spec/workspace/root";
          trainvm::RecipeProfileRegistry invalid({std::move(unsafe)});
        }),
        "workspace and other authority fields cannot be overridden");
  check(rejects([] {
          auto executable = profile();
          executable.overrides.front().name = "data.python_module";
          trainvm::RecipeProfileRegistry invalid({std::move(executable)});
        }),
        "recipe profiles cannot expose imports or executable material");
  check(rejects([] {
          auto ambiguous = profile();
          auto duplicate = target_step_field();
          duplicate.name = "training.other_target_step";
          ambiguous.overrides.push_back(std::move(duplicate));
          trainvm::RecipeProfileRegistry invalid({std::move(ambiguous)});
        }),
        "two override names cannot own the same target");
  check(rejects([] {
          (void)trainvm::RecipeProfileRegistry::from_json(
              R"({"api_version":"trainvm.recipe-profiles/v1","api_version":"wrong","recipes":[]})");
        }),
        "duplicate registry object keys are rejected");
  check(rejects([] {
          const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                            "docs/experiment-vm/examples/"
                            "hf-multimodal-sft.recipe-profiles.v1.json";
          std::ifstream input(path);
          auto document = nlohmann::json::parse(input);
          document["recipes"][0]["content_bindings"].push_back(
              document["recipes"][0]["content_bindings"][0]);
          (void)trainvm::RecipeProfileRegistry::from_json(document.dump());
        }),
        "two derived content bindings cannot own the same effective root");
  check(rejects([] {
          const auto path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                            "docs/experiment-vm/examples/"
                            "hf-multimodal-sft.recipe-profiles.v1.json";
          std::ifstream input(path);
          auto document = nlohmann::json::parse(input);
          document["recipes"][0]["content_bindings"][0]["path_target"] =
              "/spec/workflow/nodes/train/invoke/training/components/"
              "model_loader/configuration/missing";
          (void)trainvm::RecipeProfileRegistry::from_json(document.dump());
        }),
        "derived content binding cannot name a missing root target");
}

void expanded_instances_have_source_aware_diffs() {
  const trainvm::RecipeProfileRegistry registry({profile()});
  const auto left = registry.expand(instance());
  auto changed = instance();
  changed.overrides["training.target_step"] = 6000;
  const auto right = registry.expand(changed);
  const auto differences = trainvm::diff_recipe_plans(left, right);
  check(differences.size() == 1U,
        "diff contains exactly one effective canonical value change");
  check(differences.front().path == "/spec/parameters/target_step/value" &&
            differences.front().left == 5500 &&
            differences.front().right == 6000,
        "diff identifies the canonical path and values");
  check(differences.front().left_source->kind == "recipe_template" &&
            differences.front().right_source->kind == "instance_override",
        "diff reports both effective value sources");
  check(trainvm::recipe_plan_diff_json(differences).size() == 1U,
        "diff has a reflected JSON presentation");

  auto next_run = instance();
  next_run.run_identity = "recipe-profile-test-b";
  const auto run_differences =
      trainvm::diff_recipe_plans(left, registry.expand(next_run));
  check(std::ranges::any_of(run_differences, [](const auto& difference) {
          return difference.path == "/spec/workspace/run_directory" &&
                 difference.right ==
                     "/thearray/git/moe-mla/runs/recipe-profile-test-b";
        }) &&
            std::ranges::all_of(run_differences, [](const auto& difference) {
              return difference.right_source &&
                     difference.right_source->kind ==
                         "instance_run_identity";
            }),
        "run identity diff atomically shows every derived field and source");
}

void checked_in_qwen_example_expands_without_source_changes() {
  const std::filesystem::path root(TRAINVM_SOURCE_ROOT);
  const auto registry = trainvm::RecipeProfileRegistry::load_file(
      root / "docs/experiment-vm/examples/"
             "hf-multimodal-sft.recipe-profiles.v1.json");
  std::ifstream input(
      root / "docs/experiment-vm/examples/"
             "qwen-caption-lora-r256.recipe-instance.v1.json");
  if (!input) throw std::runtime_error("could not open recipe instance fixture");
  const auto expanded = registry.expand_json(nlohmann::json::parse(input));
  check(expanded.recipe ==
            trainvm::RecipeKey{.name = "hf_multimodal_sft", .version = "1"},
        "compact Qwen instance selects the exact versioned recipe");
  check(expanded.effective_overrides.at("trainability.lora_rank") == 256 &&
            expanded.effective_overrides.at(
                "hyperparameters.learning_rate") == 0.00002,
        "compact Qwen instance changes run data rather than source code");
  check(expanded.plan.canonical_plan.at("spec").at("workflow").at("nodes").
                size() == 3U,
        "Qwen example expands to the complete exact workflow graph");

  nlohmann::json locally_resolvable = expanded.plan.canonical_plan;
  locally_resolvable[nlohmann::json::json_pointer(
      "/spec/workflow/nodes/train/invoke/training/components/model_loader/"
      "configuration/model_path")] = std::filesystem::canonical(root).string();
  locally_resolvable[nlohmann::json::json_pointer(
      "/spec/workflow/nodes/train/invoke/training/components/data/"
      "configuration/dataset_root")] = std::filesystem::canonical(root).string();
  locally_resolvable[nlohmann::json::json_pointer(
      "/spec/workflow/nodes/train/invoke/training/components/trainability/"
      "configuration/target_manifest_path")] =
      std::filesystem::canonical(root / "README.md").string();
  const auto local_plan = trainvm::compile_document(locally_resolvable);
  if (!local_plan.valid())
    throw std::runtime_error("locally resolvable recipe plan did not compile");
  const auto components = trainvm::TrainingComponentRegistry::load_file(
      root / "docs/experiment-vm/examples/training-components.v1.json");
  components.validate_plan(*local_plan.plan);
  check(true,
        "Qwen graph resolves registered model, data, evaluation, and checkpoint slots");
}

}  // namespace

int main() {
  profiles_are_reflected_versioned_and_canonical();
  expansion_preserves_ordinary_plan_identity_and_provenance();
  invalid_and_ambiguous_authoring_fails_closed();
  expanded_instances_have_source_aware_diffs();
  checked_in_qwen_example_expands_without_source_changes();
  if (failures == 0) {
    std::cout << "recipe profile tests passed\n";
    return 0;
  }
  std::cerr << failures << " recipe profile test(s) failed\n";
  return 1;
}
