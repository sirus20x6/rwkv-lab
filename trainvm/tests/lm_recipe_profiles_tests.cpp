#include "trainvm/recipe_profile.hpp"
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
  const auto expand = [&](std::string_view name) {
    const auto instance = read_json(root / "docs/experiment-vm/examples" /
                                    std::string(name));
    auto result = registry.expand_json(instance);
    components.validate_plan(result.plan);
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
                .at("configuration")
                .at("maximum_sequence_length") == 4096 &&
            training_components(packed)
                    .at("gradient_accumulation")
                    .at("configuration")
                    .at("microbatches_per_optimizer_step") == 4,
        "packed-token instance preserves its declared sequence and accumulation trajectory");
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
              "/collation/configuration/maximum_sequence_length");
        }),
        "packed-token parity diff names the context-length change");

  if (failures == 0) {
    std::cout << "LM recipe profile tests passed\n";
    return 0;
  }
  std::cerr << failures << " LM recipe profile test(s) failed\n";
  return 1;
}
