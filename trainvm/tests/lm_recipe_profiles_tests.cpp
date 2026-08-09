#include "trainvm/adapter_registry.hpp"
#include "trainvm/document.hpp"
#include "trainvm/recipe_profile.hpp"
#include "trainvm/rwkv_lab_worker_contract.hpp"
#include "trainvm/training_component_registry.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

nlohmann::json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open " + path.string());
  return nlohmann::json::parse(input);
}

// Catalog validation calls filesystem::exists on every field of type `path`.
// The checked-in instances are deployment examples: they name the repository
// and dataset directories of the host they were authored on, and the recipes
// pin allowed_read_roots to that same host, so the paths cannot be redirected
// in the instance -- expansion rejects that before validation is reached.
// Validating the expansion verbatim therefore asserted the contents of
// whichever machine ran the suite: it passed on the deployment host and
// aborted in CI with "training component configuration path is unavailable".
//
// Redirecting the already-expanded plan is the same move the checked-in Qwen
// example makes in recipe_profile_tests. The whole catalog and adapter check
// still runs, on every host; the only thing it stops asserting is whether this
// particular machine happens to have the fixture's directories.
nlohmann::json locally_resolvable(const nlohmann::json& canonical_plan,
                                  const std::filesystem::path& root) {
  nlohmann::json local = canonical_plan;
  const std::string directory = std::filesystem::canonical(root).string();
  const std::string file =
      std::filesystem::canonical(root / "README.md").string();
  const auto retarget = [&local](std::string_view pointer,
                                 const std::string& value) {
    const nlohmann::json::json_pointer at{std::string(pointer)};
    if (local.contains(at)) local[at] = value;
  };
  retarget(
      "/spec/workflow/nodes/train/invoke/training/components/model_loader/"
      "configuration/model_path",
      directory);
  retarget(
      "/spec/workflow/nodes/train/invoke/training/components/data/"
      "configuration/dataset_root",
      directory);
  retarget(
      "/spec/workflow/nodes/train/invoke/training/components/model_loader/"
      "configuration/checkpoint_path",
      file);
  return local;
}

const nlohmann::json& training_components(
    const trainvm::ExpandedRecipe& expanded) {
  return expanded.plan.canonical_plan.at("spec")
      .at("workflow")
      .at("nodes")
      .at("train")
      .at("invoke")
      .at("training")
      .at("components");
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

}  // namespace

int main() {
  const std::filesystem::path root(TRAINVM_SOURCE_ROOT);
  const auto registry = trainvm::RecipeProfileRegistry::load_file(
      root / "docs/experiment-vm/examples/lm-training.recipe-profiles.v1.json");
  const auto components = trainvm::TrainingComponentRegistry::load_file(
      root / "docs/experiment-vm/examples/training-components.v1.json");
  const auto worker = trainvm::rwkv_lab_worker_contract(
      "sha256:" + std::string(64U, '0'));
  const auto adapter_registry_path =
      std::filesystem::temp_directory_path() /
      "trainvm-lm-recipe-adapters.v2.json";
  {
    std::ofstream output(adapter_registry_path,
                         std::ios::binary | std::ios::trunc);
    output << trainvm::encode_json(worker.adapter_registry).dump();
  }
  std::filesystem::permissions(
      adapter_registry_path, std::filesystem::perms::owner_read |
                                 std::filesystem::perms::owner_write,
      std::filesystem::perm_options::replace);
  const auto adapters =
      trainvm::AdapterRegistry::load_file(adapter_registry_path);
  check(adapters
            .resolve(trainvm::AdapterKey{
                .adapter = "rwkv-lab.hf-multimodal-sft",
                .version = "1.0.0",
                .runtime = trainvm::ComponentRuntime::python_worker,
                .operation = "train",
                .contract = "rwkv_lab.hf_multimodal_sft.v1.Train",
            })
            .lifecycle.pause_release_resources,
        "HF trainer registry supports resource-releasing pause");
  const auto validate_locally = [&](const trainvm::ExpandedRecipe& expanded) {
    const auto compiled = trainvm::compile_document(
        locally_resolvable(expanded.plan.canonical_plan, root));
    if (!compiled.valid())
      throw std::runtime_error("locally resolvable LM plan did not compile");
    components.validate_plan(*compiled.plan);
    adapters.validate_plan(*compiled.plan);
  };
  const auto expand = [&](std::string_view name) {
    const auto instance = read_json(root / "docs/experiment-vm/examples" /
                                    std::string(name));
    auto result = registry.expand_json(instance);
    validate_locally(result);
    return result;
  };

  const auto full = expand("transformer-lm.recipe-instance.v1.json");
  const auto lora = expand("transformer-lm-lora.recipe-instance.v1.json");
  const auto packed = expand("transformer-lm-packed.recipe-instance.v1.json");
  for (const auto* recipe : {&full, &lora, &packed}) {
    const auto& selected = training_components(*recipe);
    check(selected.at("model_loader").at("key").at("name") == "hf_causal" &&
              selected.at("data").at("key").at("name") ==
                  "manifested_jsonl_token_splits" &&
              selected.at("processor").at("key").at("name") == "token_ids" &&
              selected.at("sample_mapping").at("key").at("name") ==
                  "causal_tokens",
          "every decoder-LM recipe uses the registered causal/token pipeline");
    check(selected.at("evaluation_schedule")
                  .at("configuration")
                  .at("full_step_zero") == true &&
              selected.at("qualitative_samples")
                      .at("configuration")
                      .at("sample_count") > 0 &&
              selected.at("artifact_renderer").at("configuration").at(
                  "modality") == "text",
          "step-zero scalar and text-example policy is mandatory");
    check(selected.at("checkpoint_policy").at("key").at("name") ==
                  "atomic_retained" &&
              recipe->plan.canonical_plan.at("spec")
                      .at("recovery")
                      .at("exact_resume") == true,
          "decoder-LM recipes use registered exact checkpoint policy");
  }

  check(training_components(full).at("trainability").at("key").at("name") ==
            "full" &&
            training_components(lora)
                    .at("trainability")
                    .at("key")
                    .at("name") == "lora" &&
            training_components(lora)
                    .at("trainability")
                    .at("configuration")
                    .at("rank") == 256,
        "ordinary full training and LoRA SFT are separate declarative profiles");
  check(training_components(packed)
                .at("collation")
                .at("key")
                .at("name") == "packed_tokens" &&
            training_components(packed)
                .at("collation")
                .at("configuration")
                .at("maximum_sequence_length") == 4096 &&
            training_components(packed)
                    .at("gradient_accumulation")
                    .at("configuration")
                    .at("microbatches_per_optimizer_step") == 4,
        "packed-token instance selects real packing and preserves its declared trajectory");
  auto invalid = read_json(root / "docs/experiment-vm/examples/"
                                  "transformer-lm.recipe-instance.v1.json");
  invalid["overrides"]["hyperparameters.learning_rate"] = "fast";
  check(rejects([&] { (void)registry.expand_json(invalid); }),
        "wrongly typed optimizer hyperparameters fail at recipe expansion");

  const auto packed_differences = trainvm::diff_recipe_plans(full, packed);
  check(!packed_differences.empty(),
        "packed-token trajectory has a source-aware plan diff");
  check(std::ranges::any_of(packed_differences, [](const auto& difference) {
          return difference.path.ends_with(
              "/collation/key/name");
        }),
        "packed-token parity diff names the collator implementation change");

  const auto rwkv_profiles_path =
      root / "docs/experiment-vm/examples/rwkv-lm.recipe-profiles.v1.json";
  const auto rwkv_registry =
      trainvm::RecipeProfileRegistry::load_file(rwkv_profiles_path);
  // A `path` override has to sit inside the read roots the recipe declares, so
  // the checkpoint below is built from one instead of naming the deployment
  // host's directories — the same reason locally_resolvable above exists.
  const std::string rwkv_read_root = read_json(rwkv_profiles_path)
                                         .at("recipes")
                                         .at(0)
                                         .at("template_document")
                                         .at("spec")
                                         .at("workspace")
                                         .at("allowed_read_roots")
                                         .at(0)
                                         .get<std::string>();
  const auto expand_rwkv = [&](std::string_view name) {
    const auto instance = read_json(root / "docs/experiment-vm/examples" /
                                    std::string(name));
    auto result = rwkv_registry.expand_json(instance);
    validate_locally(result);
    return result;
  };
  const auto rwkv_scratch =
      expand_rwkv("rwkv-lm-scratch.recipe-instance.v1.json");
  auto continuation_instance = read_json(
      root / "docs/experiment-vm/examples/rwkv-lm-scratch.recipe-instance.v1.json");
  continuation_instance["recipe"]["name"] = "rwkv_lm_continuation";
  continuation_instance["run_identity"] = "rwkv-lm-continuation-test";
  continuation_instance["overrides"]["model.checkpoint_path"] =
      rwkv_read_root + "/README.md";
  continuation_instance["overrides"]["model.activation"] = "silu@1.0.0";
  continuation_instance["overrides"]["hyperparameters.optimizer"] =
      "torch_adamw_no_decay@2.0.0";
  continuation_instance["overrides"]["hyperparameters.learning_rate_schedule"] =
      "linear_warmup_cosine@1.0.0";
  const auto rwkv_continuation =
      rwkv_registry.expand_json(continuation_instance);
  validate_locally(rwkv_continuation);
  for (const auto* recipe : {&rwkv_scratch, &rwkv_continuation}) {
    const auto& selected = training_components(*recipe);
    check(selected.at("data").at("key").at("name") ==
                  "manifested_jsonl_token_splits" &&
              selected.at("processor").at("key").at("name") == "token_ids" &&
              selected.at("sample_mapping").at("key").at("name") ==
                  "causal_tokens" &&
              selected.at("normalization").at("key").at("name") ==
                  "layer_norm",
          "every RWKV recipe uses the registered token and normalization pipeline");
    check(selected.at("evaluation_schedule")
                  .at("configuration")
                  .at("full_step_zero") == true &&
              selected.at("qualitative_samples")
                      .at("configuration")
                      .at("sample_count") > 0 &&
              selected.at("artifact_renderer").at("configuration").at(
                  "modality") == "text",
          "RWKV step-zero scalar and text-example policy is mandatory");
    check(selected.at("checkpoint_policy").at("key").at("name") ==
              "atomic_retained",
          "RWKV recipes use the registered checkpoint policy");
  }
  check(training_components(rwkv_scratch)
                    .at("model_loader")
                    .at("key")
                    .at("name") == "rwkv_scratch" &&
            training_components(rwkv_continuation)
                    .at("model_loader")
                    .at("key")
                    .at("name") == "rwkv_checkpoint",
        "scratch and continuation RWKV recipes select distinct registered model loaders");
  check(training_components(rwkv_scratch)
                    .at("optimizer")
                    .at("key")
                    .at("name") == "torch_adamw" &&
            training_components(rwkv_scratch)
                    .at("learning_rate")
                    .at("key")
                    .at("name") == "powercool" &&
            training_components(rwkv_scratch)
                    .at("activation")
                    .at("key")
                    .at("name") == "squared_relu" &&
            training_components(rwkv_continuation)
                    .at("optimizer")
                    .at("key")
                    .at("name") == "torch_adamw_no_decay" &&
            training_components(rwkv_continuation)
                    .at("learning_rate")
                    .at("key")
                    .at("name") == "linear_warmup_cosine" &&
            training_components(rwkv_continuation)
                    .at("activation")
                    .at("key")
                    .at("name") == "silu" &&
            training_components(rwkv_continuation)
                    .at("optimizer")
                    .at("key")
                    .at("version") == "2.0.0",
        "RWKV optimizer, schedule, and activation choices lower declaratively");
  auto unsafe_registry_document = read_json(
      root / "docs/experiment-vm/examples/rwkv-lm.recipe-profiles.v1.json");
  for (auto& field : unsafe_registry_document["recipes"][0]["overrides"]) {
    if (field["name"] == "hyperparameters.optimizer") {
      field["values"] = {"torch_adamw", "torch_adamw_no_decay"};
      break;
    }
  }
  check(rejects([&] {
          (void)trainvm::RecipeProfileRegistry::from_json(
              unsafe_registry_document.dump());
        }),
        "name-only component choices cannot retain a template version");
  auto invalid_rwkv = read_json(
      root / "docs/experiment-vm/examples/rwkv-lm-scratch.recipe-instance.v1.json");
  invalid_rwkv["overrides"]["hyperparameters.optimizer"] = "unregistered";
  check(rejects([&] { (void)rwkv_registry.expand_json(invalid_rwkv); }),
        "unregistered RWKV optimizer choices fail at recipe expansion");
  invalid_rwkv = read_json(
      root / "docs/experiment-vm/examples/rwkv-lm-scratch.recipe-instance.v1.json");
  invalid_rwkv["overrides"]["model.normalization"] = "unregistered";
  check(rejects([&] { (void)rwkv_registry.expand_json(invalid_rwkv); }),
        "unregistered RWKV normalization choices fail at recipe expansion");

  const auto rwkv_differences =
      trainvm::diff_recipe_plans(rwkv_scratch, rwkv_continuation);
  check(std::ranges::any_of(rwkv_differences, [](const auto& difference) {
          return difference.path.ends_with("/model_loader/key/name");
        }),
        "RWKV scratch-to-continuation parity diff names the model-loader change");

  if (failures == 0) {
    std::cout << "LM recipe profile tests passed\n";
    return 0;
  }
  std::cerr << failures << " LM recipe profile test(s) failed\n";
  return 1;
}
