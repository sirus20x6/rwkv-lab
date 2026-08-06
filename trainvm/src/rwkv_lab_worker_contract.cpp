#include "trainvm/rwkv_lab_worker_contract.hpp"

#include <algorithm>
#include <map>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace trainvm {
namespace {

AdapterKey key(std::string adapter, std::string contract,
               std::string operation = "train") {
  return {
      .adapter = std::move(adapter),
      .version = "1.0.0",
      .runtime = ComponentRuntime::python_worker,
      .operation = std::move(operation),
      .contract = std::move(contract),
  };
}

OperationLifecycleCapabilities resumable_training_lifecycle() {
  return {
      .stateful = true,
      .graceful_stop = true,
      .checkpoint_now = true,
      .pause_keep_resources = true,
      .pause_release_resources = true,
      .compile = false,
      .warmup = false,
      .qualify = false,
      .profile = true,
      .resume_grade = ResumeGrade::compatible,
  };
}

OperationLifecycleCapabilities receipted_training_lifecycle() {
  OperationLifecycleCapabilities lifecycle = resumable_training_lifecycle();
  lifecycle.compile = true;
  lifecycle.warmup = true;
  return lifecycle;
}

OperationLifecycleCapabilities exact_training_lifecycle() {
  OperationLifecycleCapabilities lifecycle = receipted_training_lifecycle();
  lifecycle.resume_grade = ResumeGrade::exact;
  return lifecycle;
}

OperationLifecycleCapabilities restart_only_training_lifecycle() {
  return {
      .stateful = true,
      .graceful_stop = true,
      .checkpoint_now = false,
      .pause_keep_resources = false,
      .pause_release_resources = false,
      .compile = false,
      .warmup = false,
      .qualify = false,
      .profile = true,
      .resume_grade = ResumeGrade::restart_only,
  };
}

OperationAuthoringDeclaration checkpoint_authoring() {
  return {
      .inputs = {
          {"config",
           OperationPortDescriptor{
               .type = OperationPortType::object,
               .required = true,
               .artifact_type = std::nullopt,
               .artifact_schema = std::nullopt,
               .description =
                   "Typed trainer configuration object consumed by this "
                   "exact adapter contract.",
           }},
      },
      .outputs = {
          {"checkpoint",
           OperationPortDescriptor{
               .type = OperationPortType::artifact,
               .required = false,
               .artifact_type = ArtifactType::checkpoint,
               .artifact_schema = std::nullopt,
               .description =
                   "Optional authority-published trainer checkpoint.",
           }},
      },
  };
}

OperationAuthoringDeclaration mageflow_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  authoring.outputs.emplace(
      "eval_gallery",
      OperationPortDescriptor{
          .type = OperationPortType::artifact,
          .required = false,
          .artifact_type = ArtifactType::image_gallery,
          .artifact_schema = "rwkv-lab.eval-gallery.v2",
          .description =
              "Optional checkpoint-bound generated/original held-out gallery.",
      });
  return authoring;
}

OperationAuthoringDeclaration hf_multimodal_sft_authoring() {
  OperationAuthoringDeclaration authoring;
  authoring.outputs = {
      {"checkpoint",
       OperationPortDescriptor{
           .type = OperationPortType::artifact,
           .required = true,
           .artifact_type = ArtifactType::checkpoint,
           .artifact_schema = "hf.multimodal-sft.v1",
           .description =
               "Required exact-resume Hugging Face model, optimizer, data and RNG state.",
       }},
      {"eval_gallery",
       OperationPortDescriptor{
           .type = OperationPortType::artifact,
           .required = true,
           .artifact_type = ArtifactType::image_gallery,
           .artifact_schema = "rwkv-lab.eval-gallery.v2",
           .description =
               "Required checkpoint-bound step-zero and periodic held-out evidence.",
       }},
      {"test_eval",
       OperationPortDescriptor{
           .type = OperationPortType::artifact,
           .required = true,
           .artifact_type = ArtifactType::report,
           .artifact_schema =
               "rwkv-lab.hf-test-caption-evidence-bundle.v1",
           .description =
               "Required immutable baseline/final test-caption evidence bundle.",
       }},
  };
  return authoring;
}

OperationAuthoringDeclaration posttraining_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  authoring.outputs = {
      {"adapter",
       OperationPortDescriptor{
           .type = OperationPortType::artifact,
           .required = true,
           .artifact_type = ArtifactType::opaque,
           .artifact_schema = "rwkv-lab.posttraining-output.v1",
           .description =
               "Immutable adapter, reward-head, result, and metric bundle.",
       }},
  };
  return authoring;
}

OperationAuthoringDeclaration vision_compressor_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema =
      "rwkv-lab.vision-teacher-compressor-checkpoint.v1";
  checkpoint.description =
      "Required resumable multi-teacher compressor checkpoint.";
  return authoring;
}

OperationAuthoringDeclaration vision_frozen_adapter_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema =
      "rwkv-lab.vision-frozen-adapter-checkpoint.v1";
  checkpoint.description =
      "Required compatible cached MoonViT/compressor caption checkpoint.";
  authoring.outputs.emplace(
      "result",
      OperationPortDescriptor{
          .type = OperationPortType::artifact,
          .required = true,
          .artifact_type = ArtifactType::report,
          .artifact_schema = "rwkv-lab.scalar-metric-result.v1",
          .description =
              "Required immutable best-evaluation scalar metric result.",
      });
  return authoring;
}

OperationAuthoringDeclaration scalar_metric_decision_authoring() {
  const OperationPortDescriptor result{
      .type = OperationPortType::artifact,
      .required = true,
      .artifact_type = ArtifactType::report,
      .artifact_schema = "rwkv-lab.scalar-metric-result.v1",
      .description = "Immutable scalar metric candidate result.",
  };
  return {
      .inputs = {
          {"config",
           OperationPortDescriptor{
               .type = OperationPortType::object,
               .required = true,
               .artifact_type = std::nullopt,
               .artifact_schema = std::nullopt,
               .description =
                   "Typed metric, direction, subject, and tolerance policy.",
           }},
          {"left", result},
          {"right", result},
      },
      .outputs = {
          {"decision",
           OperationPortDescriptor{
               .type = OperationPortType::artifact,
               .required = true,
               .artifact_type = ArtifactType::report,
               .artifact_schema = "rwkv-lab.scalar-metric-decision.v1",
               .description =
                   "Immutable lineage-bound scalar comparison decision.",
           }},
      },
  };
}

OperationAuthoringDeclaration vision_native_head_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema = "rwkv-lab.vision-native-head-checkpoint.v1";
  checkpoint.description =
      "Required compatible native RWKV vision-head checkpoint.";
  return authoring;
}

OperationAuthoringDeclaration vision_rwkv_student_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema = "rwkv-lab.vision-rwkv-student-checkpoint.v1";
  checkpoint.description =
      "Required compatible raw-pixel Vision-RWKV student checkpoint.";
  return authoring;
}

TrainingCompositionContract vision_native_head_composition() {
  return {
      .model_family = "vision",
      .slots = {
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"learning_rate",
           TrainingComponentCategory::learning_rate_schedule},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"precision", TrainingComponentCategory::precision},
          {"weight_decay",
           TrainingComponentCategory::weight_decay_schedule},
      },
      .allowed_components =
          std::map<std::string, std::vector<TrainingComponentKey>>{
              {"learning_rate",
               {{TrainingComponentCategory::learning_rate_schedule,
                 "constant", "1.0.0"}}},
              {"optimizer",
               {{TrainingComponentCategory::optimizer,
                 "torch_adamw_no_decay", "2.0.0"}}},
              {"precision",
               {{TrainingComponentCategory::precision,
                 "fp32_parameters_bf16_compute", "1.0.0"}}},
          },
  };
}

TrainingCompositionContract vision_rwkv_student_composition() {
  TrainingCompositionContract composition = vision_native_head_composition();
  composition.allowed_components->at("precision") = {
      {TrainingComponentCategory::precision,
       "bf16_parameters_fp32_reductions", "1.0.0"}};
  return composition;
}

OperationAuthoringDeclaration rlvr_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema = "rwkv-lab.rlvr-candidate-checkpoint.v1";
  checkpoint.description =
      "Required terminal RLVR candidate checkpoint and promotion lineage.";
  return authoring;
}

TrainingCompositionContract rlvr_composition() {
  return {
      .model_family = "rwkv",
      .slots = {
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"learning_rate",
           TrainingComponentCategory::learning_rate_schedule},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"weight_decay",
           TrainingComponentCategory::weight_decay_schedule},
      },
      .allowed_components =
          std::map<std::string, std::vector<TrainingComponentKey>>{
              {"learning_rate",
               {{TrainingComponentCategory::learning_rate_schedule,
                 "linear_warmup_constant", "1.0.0"}}},
              {"optimizer",
               {{TrainingComponentCategory::optimizer,
                 "torch_adamw_no_decay", "2.0.0"}}},
          },
  };
}

TrainingCompositionContract vision_compressor_composition() {
  return {
      .model_family = "vision",
      .slots = {
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"learning_rate",
           TrainingComponentCategory::learning_rate_schedule},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"precision", TrainingComponentCategory::precision},
          {"weight_decay",
           TrainingComponentCategory::weight_decay_schedule},
      },
      .allowed_components =
          std::map<std::string, std::vector<TrainingComponentKey>>{
              {"learning_rate",
               {{TrainingComponentCategory::learning_rate_schedule,
                 "constant", "1.0.0"}}},
              {"optimizer",
               {{TrainingComponentCategory::optimizer, "torch_adamw",
                 "1.0.0"}}},
              {"precision",
               {{TrainingComponentCategory::precision,
                 "fp32_parameters_bf16_compute", "1.0.0"}}},
          },
  };
}

TrainingCompositionContract vision_frozen_adapter_composition() {
  return {
      .model_family = "vision",
      .slots = {
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"precision", TrainingComponentCategory::precision},
      },
      .allowed_components =
          std::map<std::string, std::vector<TrainingComponentKey>>{
              {"optimizer",
               {{TrainingComponentCategory::optimizer, "torch_adamw",
                 "1.0.0"}}},
              {"precision",
               {{TrainingComponentCategory::precision,
                 "fp32_parameters_bf16_compute", "1.0.0"}}},
          },
  };
}

TrainingCompositionContract mageflow_full_backbone_composition() {
  return {
      .model_family = "mageflow",
      .slots = {
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"learning_rate",
           TrainingComponentCategory::learning_rate_schedule},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"parameter_router", TrainingComponentCategory::parameter_router},
          {"weight_decay",
           TrainingComponentCategory::weight_decay_schedule},
      },
      .allowed_components =
          std::map<std::string, std::vector<TrainingComponentKey>>{
              {"optimizer",
               {{TrainingComponentCategory::optimizer,
                 "torch_adamw_no_decay", "2.0.0"}}},
              {"parameter_router",
               {{TrainingComponentCategory::parameter_router,
                 "mageflow_full_backbone", "1.0.0"}}},
          },
  };
}

TrainingCompositionContract transformer_mla_composition() {
  TrainingCompositionContract composition{
      .model_family = "transformer",
      .slots = {
          {"gradient_accumulation",
           TrainingComponentCategory::gradient_accumulation},
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"learning_rate",
           TrainingComponentCategory::learning_rate_schedule},
          {"objective", TrainingComponentCategory::objective},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"precision", TrainingComponentCategory::precision},
          {"weight_decay",
           TrainingComponentCategory::weight_decay_schedule},
      },
  };
  composition.allowed_components =
      std::map<std::string, std::vector<TrainingComponentKey>>{
          {"optimizer",
           {{TrainingComponentCategory::optimizer, "torch_adamw", "1.0.0"},
            {TrainingComponentCategory::optimizer, "torch_adamw_no_decay",
             "2.0.0"}}},
      };
  return composition;
}

TrainingCompositionContract transformer_mla_engram_composition() {
  TrainingCompositionContract composition = transformer_mla_composition();
  composition.slots.emplace("host_optimizer",
                            TrainingComponentCategory::optimizer);
  composition.allowed_components->emplace(
      "host_optimizer",
      std::vector<TrainingComponentKey>{{TrainingComponentCategory::optimizer,
                                         "torch_sparse_adam", "1.0.0"}});
  return composition;
}

TrainingCompositionContract hf_multimodal_sft_composition() {
  TrainingCompositionContract composition{
      .model_family = "transformer",
      .slots = {
          {"activation_memory", TrainingComponentCategory::activation_memory},
          {"artifact_renderer", TrainingComponentCategory::artifact_renderer},
          {"batching", TrainingComponentCategory::batching},
          {"checkpoint_policy", TrainingComponentCategory::checkpoint_policy},
          {"collation", TrainingComponentCategory::collator},
          {"data", TrainingComponentCategory::data_source},
          {"evaluation_schedule",
           TrainingComponentCategory::evaluation_schedule},
          {"evaluation_split", TrainingComponentCategory::split_selector},
          {"evaluator", TrainingComponentCategory::evaluator},
          {"gradient_accumulation",
           TrainingComponentCategory::gradient_accumulation},
          {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
          {"generation_policy", TrainingComponentCategory::generation_policy},
          {"learning_rate", TrainingComponentCategory::learning_rate_schedule},
          {"model_loader", TrainingComponentCategory::model_loader},
          {"objective", TrainingComponentCategory::objective},
          {"optimizer", TrainingComponentCategory::optimizer},
          {"precision", TrainingComponentCategory::precision},
          {"processor", TrainingComponentCategory::sample_processor},
          {"qualitative_samples",
           TrainingComponentCategory::qualitative_sample},
          {"sample_mapping", TrainingComponentCategory::sample_mapper},
          {"sampler", TrainingComponentCategory::sampler},
          {"split", TrainingComponentCategory::split_selector},
          {"test_split", TrainingComponentCategory::split_selector},
          {"trainability", TrainingComponentCategory::trainability},
          {"weight_decay", TrainingComponentCategory::weight_decay_schedule},
      },
  };
  composition.allowed_components =
      std::map<std::string, std::vector<TrainingComponentKey>>{
          {"activation_memory",
           {{TrainingComponentCategory::activation_memory,
             "hf_gradient_checkpointing", "1.0.0"}}},
          {"artifact_renderer",
           {{TrainingComponentCategory::artifact_renderer, "caption_triplet",
             "1.0.0"},
            {TrainingComponentCategory::artifact_renderer, "evidence_envelope",
             "1.0.0"}}},
          {"batching",
           {{TrainingComponentCategory::batching, "bucketed", "1.0.0"},
            {TrainingComponentCategory::batching, "fixed", "1.0.0"}}},
          {"checkpoint_policy",
           {{TrainingComponentCategory::checkpoint_policy, "atomic_retained",
             "1.0.0"}}},
          {"collation",
           {{TrainingComponentCategory::collator, "padded", "1.0.0"}}},
          {"data",
           {{TrainingComponentCategory::data_source, "jsonl_image_caption",
             "1.0.0"},
            {TrainingComponentCategory::data_source, "jsonl_token_corpus",
             "1.0.0"},
            {TrainingComponentCategory::data_source,
             "manifested_jsonl_image_splits", "1.0.0"},
            {TrainingComponentCategory::data_source,
             "manifested_jsonl_token_splits", "1.0.0"}}},
          {"evaluation_schedule",
           {{TrainingComponentCategory::evaluation_schedule,
             "launch_gate_periodic", "1.0.0"},
            {TrainingComponentCategory::evaluation_schedule,
             "launch_gate_periodic", "2.0.0"}}},
          {"evaluation_split",
           {{TrainingComponentCategory::split_selector, "deterministic_holdout",
             "1.0.0"},
            {TrainingComponentCategory::split_selector, "frozen_named",
             "1.0.0"}}},
          {"evaluator",
           {{TrainingComponentCategory::evaluator, "scalar_loss", "1.0.0"}}},
          {"gradient_accumulation",
           {{TrainingComponentCategory::gradient_accumulation, "fixed",
             "1.0.0"}}},
          {"gradient_clipping",
           {{TrainingComponentCategory::gradient_clipping, "global_norm",
             "1.0.0"}}},
          {"generation_policy",
           {{TrainingComponentCategory::generation_policy, "greedy", "1.0.0"}}},
          {"learning_rate",
           {{TrainingComponentCategory::learning_rate_schedule,
             "linear_warmup_cosine", "1.0.0"}}},
          {"model_loader",
           {{TrainingComponentCategory::model_loader, "hf_causal", "1.0.0"},
            {TrainingComponentCategory::model_loader, "hf_multimodal",
             "1.0.0"}}},
          {"objective",
           {{TrainingComponentCategory::objective, "linear_head_cross_entropy",
             "1.0.0"}}},
          {"optimizer",
           {{TrainingComponentCategory::optimizer, "torch_adamw", "1.0.0"},
            {TrainingComponentCategory::optimizer, "torch_adamw_no_decay",
             "2.0.0"}}},
          {"precision",
           {{TrainingComponentCategory::precision,
             "bf16_parameters_fp32_reductions", "1.0.0"},
            {TrainingComponentCategory::precision,
             "fp32_parameters_bf16_compute", "1.0.0"}}},
          {"processor",
           {{TrainingComponentCategory::sample_processor, "image_caption",
             "1.0.0"},
            {TrainingComponentCategory::sample_processor, "token_ids",
             "1.0.0"}}},
          {"qualitative_samples",
           {{TrainingComponentCategory::qualitative_sample, "fixed_held_out",
             "1.0.0"},
            {TrainingComponentCategory::qualitative_sample, "fixed_held_out",
             "2.0.0"}}},
          {"sample_mapping",
           {{TrainingComponentCategory::sample_mapper,
             "assistant_conversation", "2.0.0"},
            {TrainingComponentCategory::sample_mapper, "assistant_only",
             "1.0.0"},
            {TrainingComponentCategory::sample_mapper, "causal_tokens",
             "1.0.0"}}},
          {"sampler",
           {{TrainingComponentCategory::sampler, "deterministic", "1.0.0"}}},
          {"split",
           {{TrainingComponentCategory::split_selector, "deterministic_holdout",
             "1.0.0"},
            {TrainingComponentCategory::split_selector, "frozen_named",
             "1.0.0"}}},
          {"test_split",
           {{TrainingComponentCategory::split_selector, "frozen_named",
             "1.0.0"}}},
          {"trainability",
           {{TrainingComponentCategory::trainability, "full", "1.0.0"},
            {TrainingComponentCategory::trainability, "lora", "1.0.0"},
            {TrainingComponentCategory::trainability,
             "lora_target_manifest", "2.0.0"},
            {TrainingComponentCategory::trainability, "named_rules",
             "1.0.0"}}},
          {"weight_decay",
           {{TrainingComponentCategory::weight_decay_schedule, "constant",
             "1.0.0"}}},
      };
  return composition;
}

std::vector<std::string> canonical_distributions(
    std::initializer_list<std::string_view> values) {
  std::vector<std::string> result;
  result.reserve(values.size());
  for (const std::string_view value : values) {
    if (value.empty()) {
      throw std::logic_error("runtime distribution name must not be empty");
    }
    result.emplace_back(value);
  }
  std::ranges::sort(result);
  if (std::ranges::adjacent_find(result) != result.end()) {
    throw std::logic_error("runtime distribution names must be unique");
  }
  return result;
}

AdapterProfile profile(AdapterKey adapter_key, std::string code_fingerprint,
                       TrainingCompositionContract composition,
                       OperationLifecycleCapabilities lifecycle,
                       std::optional<OperationAuthoringDeclaration> authoring =
                           std::nullopt) {
  return {
      .key = std::move(adapter_key),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = std::move(code_fingerprint),
      .required_capabilities = {"worker.controls", "worker.metrics"},
      .lifecycle = lifecycle,
      .training_composition = std::move(composition),
      .authoring = authoring ? std::move(*authoring) : checkpoint_authoring(),
  };
}

}  // namespace

RwkvLabWorkerContract rwkv_lab_worker_contract(
    std::string code_fingerprint) {
  std::vector<AdapterProfile> profiles;
  profiles.push_back(profile(
      key("rwkv-lab.mageflow-appearance-expert",
          "rwkv_lab.mageflow_appearance_expert.v1.Train"),
      code_fingerprint,
      {.model_family = "mageflow",
       .slots = {
           {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
           {"learning_rate",
            TrainingComponentCategory::learning_rate_schedule},
           {"optimizer", TrainingComponentCategory::optimizer},
           {"parameter_router", TrainingComponentCategory::parameter_router},
           {"weight_decay",
            TrainingComponentCategory::weight_decay_schedule},
      }},
      receipted_training_lifecycle(), mageflow_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.mageflow-full-backbone",
          "rwkv_lab.mageflow_full_backbone.v1.Train"),
      code_fingerprint, mageflow_full_backbone_composition(),
      receipted_training_lifecycle(), mageflow_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.mageflow-terminal-expert",
          "rwkv_lab.mageflow_terminal_expert.v1.Train"),
      code_fingerprint,
      {.model_family = "mageflow",
       .slots = {
           {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
           {"learning_rate",
            TrainingComponentCategory::learning_rate_schedule},
           {"loop_gate_gradient_clipping",
            TrainingComponentCategory::gradient_clipping},
           {"optimizer", TrainingComponentCategory::optimizer},
           {"parameter_router", TrainingComponentCategory::parameter_router},
           {"weight_decay",
            TrainingComponentCategory::weight_decay_schedule},
       }},
      receipted_training_lifecycle(), mageflow_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.rwkv-posttraining",
          "rwkv_lab.rwkv_posttraining.v1.Train"),
      code_fingerprint,
      {.model_family = "rwkv",
       .slots = {
           {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
           {"learning_rate",
            TrainingComponentCategory::learning_rate_schedule},
           {"optimizer", TrainingComponentCategory::optimizer},
           {"weight_decay",
            TrainingComponentCategory::weight_decay_schedule},
       },
       .allowed_components =
           std::map<std::string, std::vector<TrainingComponentKey>>{
               {"optimizer",
                {{TrainingComponentCategory::optimizer,
                  "torch_adamw_no_decay", "2.0.0"}}},
           }},
      restart_only_training_lifecycle(), posttraining_authoring()));
  profiles.push_back({
      .key = key("rwkv-lab.scalar-metric-decision",
                 "rwkv_lab.scalar_metric_decision.v1.Decide", "decide"),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = code_fingerprint,
      .required_capabilities = {"worker.controls", "worker.metrics"},
      .lifecycle = {
          .stateful = false,
          .graceful_stop = false,
          .checkpoint_now = false,
          .pause_keep_resources = false,
          .pause_release_resources = false,
          .compile = false,
          .warmup = false,
          .qualify = false,
          .profile = false,
          .resume_grade = ResumeGrade::none,
      },
      .training_composition = std::nullopt,
      .authoring = scalar_metric_decision_authoring(),
  });
  profiles.push_back(profile(
      key("rwkv-lab.qwen-ao3", "rwkv_lab.qwen_ao3.v1.Train"),
      code_fingerprint,
      {.model_family = "transformer",
       .slots = {
           {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
           {"learning_rate",
            TrainingComponentCategory::learning_rate_schedule},
           {"optimizer", TrainingComponentCategory::optimizer},
           {"weight_decay",
            TrainingComponentCategory::weight_decay_schedule},
      }},
      resumable_training_lifecycle()));
  profiles.push_back(profile(
      key("rwkv-lab.hf-multimodal-sft",
          "rwkv_lab.hf_multimodal_sft.v1.Train"),
      code_fingerprint, hf_multimodal_sft_composition(),
      exact_training_lifecycle(), hf_multimodal_sft_authoring()));
  for (const auto& [adapter, contract] :
       std::initializer_list<std::pair<std::string, std::string>>{
           {"rwkv-lab.transformer-mla", "rwkv_lab.transformer_mla.v1.Train"},
           {"rwkv-lab.transformer-mla-mtp",
            "rwkv_lab.transformer_mla_mtp.v1.Train"},
           {"rwkv-lab.transformer-mla-mutor",
            "rwkv_lab.transformer_mla_mutor.v1.Train"},
           {"rwkv-lab.transformer-mla-fsp",
            "rwkv_lab.transformer_mla_fsp.v1.Train"},
           {"rwkv-lab.transformer-mla-parallel",
            "rwkv_lab.transformer_mla_parallel.v1.Train"},
           {"rwkv-lab.transformer-mla-rwkv8",
            "rwkv_lab.transformer_mla_rwkv8.v1.Train"},
           {"rwkv-lab.transformer-mla-engram",
            "rwkv_lab.transformer_mla_engram.v1.Train"},
           {"rwkv-lab.transformer-mla-full-backbone",
            "rwkv_lab.transformer_mla_full_backbone.v1.Train"},
       }) {
    profiles.push_back(profile(key(adapter, contract), code_fingerprint,
                               adapter == "rwkv-lab.transformer-mla-engram"
                                   ? transformer_mla_engram_composition()
                                   : transformer_mla_composition(),
                               resumable_training_lifecycle()));
  }
  profiles.push_back(profile(
      key("rwkv-lab.vision-teacher-compressor",
          "rwkv_lab.vision_teacher_compressor.v1.Train"),
      code_fingerprint, vision_compressor_composition(),
      resumable_training_lifecycle(), vision_compressor_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.vision-frozen-adapter",
          "rwkv_lab.vision_frozen_adapter.v1.Train"),
      code_fingerprint, vision_frozen_adapter_composition(),
      resumable_training_lifecycle(), vision_frozen_adapter_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.vision-native-head",
          "rwkv_lab.vision_native_head.v1.Train"),
      code_fingerprint, vision_native_head_composition(),
      resumable_training_lifecycle(), vision_native_head_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.vision-rwkv-student",
          "rwkv_lab.vision_rwkv_student.v1.Train"),
      code_fingerprint, vision_rwkv_student_composition(),
      resumable_training_lifecycle(), vision_rwkv_student_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.rwkv-rlvr", "rwkv_lab.rwkv_rlvr.v1.Train"),
      code_fingerprint, rlvr_composition(),
      {.stateful = true,
       .graceful_stop = false,
       .checkpoint_now = false,
       .pause_keep_resources = false,
       .pause_release_resources = false,
       .compile = false,
       .warmup = false,
       .qualify = false,
       .profile = true,
       .resume_grade = ResumeGrade::terminal_checkpoint},
      rlvr_authoring()));
  profiles.push_back(profile(
      key("rwkv-lab.rwkv-scratch", "rwkv_lab.rwkv_scratch.v1.Train"),
      std::move(code_fingerprint),
      {.model_family = "rwkv",
       .slots = {
           {"activation", TrainingComponentCategory::activation},
           {"curriculum", TrainingComponentCategory::curriculum},
           {"gradient_accumulation",
            TrainingComponentCategory::gradient_accumulation},
           {"gradient_clipping", TrainingComponentCategory::gradient_clipping},
           {"learning_rate",
            TrainingComponentCategory::learning_rate_schedule},
           {"normalization", TrainingComponentCategory::normalization},
           {"objective", TrainingComponentCategory::objective},
           {"optimizer", TrainingComponentCategory::optimizer},
           {"precision", TrainingComponentCategory::precision},
           {"weight_decay",
            TrainingComponentCategory::weight_decay_schedule},
       }},
      {.stateful = true,
       .graceful_stop = true,
       .checkpoint_now = false,
       .pause_keep_resources = false,
       .pause_release_resources = false,
       .compile = true,
       .warmup = true,
       .qualify = false,
       .profile = true,
       .resume_grade = ResumeGrade::terminal_checkpoint}));

  (void)AdapterRegistry(profiles);
  return {
      .adapter_registry = {
          .api_version = "trainvm.adapters/v2",
          .profiles = std::move(profiles),
      },
      .provided_capabilities = {
          "activation.silu.v1",
          "activation.squared_relu.v1",
          "activation_memory.hf_gradient_checkpointing.v1",
          "artifact_renderer.caption_triplet.v1",
          "artifact_renderer.evidence_envelope.v1",
          "batching.bucketed.v1",
          "batching.fixed.v1",
          "checkpoint_policy.atomic_retained.v1",
          "collator.padded.v1",
          "curriculum.context_length.v1",
          "data_source.jsonl_frozen_image_splits.v1",
          "data_source.jsonl_frozen_token_splits.v1",
          "data_source.jsonl_image_caption.v1",
          "data_source.jsonl_token_corpus.v1",
          "evaluation_schedule.launch_gate_periodic.v1",
          "evaluation_schedule.launch_gate_periodic.v2",
          "evaluator.scalar_loss.v1",
          "generation_policy.greedy.v1",
          "gradient_accumulation.fixed.v1",
          "gradient_clipping.global_norm.v1",
          "model_loader.hf_causal.v1",
          "model_loader.hf_multimodal.v1",
          "normalization.layer_norm.v1",
          "objective.linear_head_cross_entropy.v1",
          "optimizer.fp32_master_adamw.v1",
          "optimizer.fp32_master_adamw_no_decay.v2",
          "optimizer.torch_adamw.v1",
          "optimizer.torch_adamw_no_decay.v2",
          "optimizer.torch_sparse_adam.v1",
          "parameter_router.mageflow_appearance_expert.v1",
          "parameter_router.mageflow_full_backbone.v1",
          "parameter_router.mageflow_terminal_expert.v1",
          "precision.bf16_parameters_fp32_reductions.v1",
          "precision.fp32_parameters_bf16_compute.v1",
          "qualitative_sample.fixed_held_out.v1",
          "qualitative_sample.fixed_held_out.v2",
          "sample_mapper.assistant_conversation.v2",
          "sample_mapper.assistant_only.v1",
          "sample_mapper.causal_tokens.v1",
          "sample_processor.image_caption.v1",
          "sample_processor.token_ids.v1",
          "sampler.deterministic.v1",
          "schedule.constant.v1",
          "schedule.linear_warmup_constant.v1",
          "schedule.linear_warmup_cosine.v1",
          "schedule.powercool.v1",
          "split_selector.deterministic_holdout.v1",
          "split_selector.frozen_named.v1",
          "trainability.frozen.v1",
          "trainability.full.v1",
          "trainability.lora.v1",
          "trainability.lora_target_manifest.v2",
          "trainability.named_rules.v1",
          "weight_decay_schedule.constant.v1",
          "worker.controls",
          "worker.metrics",
      },
  };
}

RwkvLabWorkerRuntimeRequirementsContract
rwkv_lab_worker_runtime_requirements() {
  const std::vector<std::string> shared = canonical_distributions(
      {"grpcio", "pillow", "protobuf", "torch"});
  std::map<std::string, std::vector<std::string>> requirements{
      {"rwkv-lab.hf-multimodal-sft",
       canonical_distributions(
           {"accelerate", "grpcio", "numpy", "peft", "pillow", "protobuf",
            "safetensors", "torch", "transformers"})},
      {"rwkv-lab.mageflow-appearance-expert",
       canonical_distributions(
           {"accelerate", "deepspeed", "diffusers", "einops", "flash-attn",
            "grpcio", "huggingface-hub", "mage-flow", "numpy", "pillow",
            "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.mageflow-full-backbone",
       canonical_distributions(
           {"accelerate", "deepspeed", "diffusers", "einops", "flash-attn",
            "grpcio", "huggingface-hub", "mage-flow", "numpy", "pillow",
            "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.mageflow-terminal-expert",
       canonical_distributions(
           {"accelerate", "deepspeed", "diffusers", "einops", "flash-attn",
            "grpcio", "huggingface-hub", "mage-flow", "numpy", "pillow",
            "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.qwen-ao3",
       canonical_distributions(
           {"accelerate", "bitsandbytes", "causal-conv1d", "einops",
            "flash-attn", "flash-linear-attention", "grpcio",
            "huggingface-hub", "numpy", "packaging", "peft", "pillow",
            "protobuf", "psutil", "safetensors", "tokenizers", "torch",
            "transformers", "zstandard"})},
      {"rwkv-lab.transformer-mla",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-mtp",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-mutor",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-fsp",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-parallel",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-rwkv8",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "flash-linear-attention",
            "grpcio", "numpy", "pillow", "protobuf", "safetensors",
            "torch", "transformers"})},
      {"rwkv-lab.transformer-mla-engram",
       canonical_distributions(
           {"accelerate", "einops", "engram-ext", "flash-attn", "grpcio",
            "numpy", "pillow", "protobuf", "safetensors", "torch",
            "transformers"})},
      {"rwkv-lab.transformer-mla-full-backbone",
       canonical_distributions(
           {"accelerate", "einops", "flash-attn", "grpcio", "numpy",
            "pillow", "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.rwkv-scratch",
       canonical_distributions(
           {"einops", "grpcio", "numpy", "pillow", "protobuf", "torch"})},
      {"rwkv-lab.rwkv-posttraining",
       canonical_distributions(
           {"einops", "grpcio", "numpy", "pillow", "protobuf",
            "safetensors", "torch"})},
      {"rwkv-lab.rwkv-rlvr",
       canonical_distributions(
           {"einops", "grpcio", "numpy", "pillow", "protobuf", "torch"})},
      {"rwkv-lab.scalar-metric-decision",
       canonical_distributions({"grpcio", "pillow", "protobuf", "torch"})},
      {"rwkv-lab.vision-teacher-compressor",
       canonical_distributions(
           {"grpcio", "numpy", "pillow", "protobuf", "safetensors",
            "torch", "transformers"})},
      {"rwkv-lab.vision-frozen-adapter",
       canonical_distributions(
           {"einops", "flash-attn", "grpcio", "numpy", "pillow",
            "protobuf", "safetensors", "torch", "transformers"})},
      {"rwkv-lab.vision-native-head",
       canonical_distributions(
           {"einops", "grpcio", "numpy", "pillow", "protobuf",
            "safetensors", "torch", "transformers"})},
      {"rwkv-lab.vision-rwkv-student",
       canonical_distributions(
           {"einops", "grpcio", "numpy", "pillow", "protobuf",
            "safetensors", "torch", "transformers"})},
  };

  const RwkvLabWorkerContract worker =
      rwkv_lab_worker_contract("sha256:" + std::string(64U, '0'));
  std::vector<RwkvLabWorkerAdapterRuntimeRequirements> profiles;
  profiles.reserve(worker.adapter_registry.profiles.size());
  for (const AdapterProfile& profile : worker.adapter_registry.profiles) {
    const auto selected = requirements.find(profile.key.adapter);
    if (selected == requirements.end()) {
      throw std::logic_error(
          "runtime requirements omit a registered rwkv_lab adapter");
    }
    if (!std::ranges::includes(selected->second, shared)) {
      throw std::logic_error(
          "adapter runtime requirements omit a shared worker distribution");
    }
    profiles.push_back({
        .adapter = profile.key.adapter,
        .root_distributions = std::move(selected->second),
    });
    requirements.erase(selected);
  }
  if (!requirements.empty()) {
    throw std::logic_error(
        "runtime requirements name an unregistered rwkv_lab adapter");
  }
  return {
      .api_version =
          "trainvm.rwkv-lab-worker-runtime-requirements/v1",
      .shared_root_distributions = shared,
      .profiles = std::move(profiles),
  };
}

RwkvLabWorkerDeploymentContract rwkv_lab_worker_deployment(
    RwkvLabWorkerDeploymentSpec spec) {
  if (spec.api_version != "trainvm.rwkv-lab-worker-runtimes/v1") {
    throw std::invalid_argument(
        "rwkv_lab deployment runtime spec has an unsupported api_version");
  }
  std::map<std::string, RwkvLabWorkerRuntimeDeploymentSpec> runtimes;
  for (auto& runtime : spec.runtimes) {
    if (runtime.adapter.empty() ||
        !runtimes.emplace(runtime.adapter, std::move(runtime)).second) {
      throw std::invalid_argument(
          "rwkv_lab deployment requires unique nonempty adapter runtimes");
    }
  }
  RwkvLabWorkerContract worker = rwkv_lab_worker_contract(
      "sha256:" + std::string(64U, '0'));
  std::ranges::sort(spec.trusted_roots);
  std::vector<HostLaunchProfile> launches;
  launches.reserve(worker.adapter_registry.profiles.size());
  for (AdapterProfile& adapter : worker.adapter_registry.profiles) {
    const auto selected = runtimes.find(adapter.key.adapter);
    if (selected == runtimes.end()) {
      throw std::invalid_argument(
          "rwkv_lab deployment omits a registered adapter runtime");
    }
    const RwkvLabWorkerRuntimeDeploymentSpec& runtime = selected->second;
    adapter.code_fingerprint = runtime.code_fingerprint;
    launches.push_back({
        .key = adapter.key,
        .code_fingerprint = adapter.code_fingerprint,
        .bootstrap_runtime_closure_fingerprint =
            runtime.bootstrap_runtime_closure_fingerprint,
        .provided_capabilities = worker.provided_capabilities,
        .executable_path = runtime.executable_path,
        .executable_fingerprint = runtime.executable_fingerprint,
        .code_path = runtime.code_path,
        .code_argument_index = 1U,
        .public_arguments = {"-I", "rwkv-lab-worker.pyz"},
        .working_directory = runtime.working_directory,
    });
    runtimes.erase(selected);
  }
  if (!runtimes.empty()) {
    throw std::invalid_argument(
        "rwkv_lab deployment names an unregistered adapter runtime");
  }
  HostLaunchRegistryDocument host{
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = std::move(spec.trusted_roots),
      .profiles = std::move(launches),
  };
  (void)AdapterRegistry(worker.adapter_registry.profiles);
  (void)HostLaunchRegistry(host);
  return {
      .adapter_registry = std::move(worker.adapter_registry),
      .host_launch_registry = std::move(host),
      .provided_capabilities = std::move(worker.provided_capabilities),
  };
}

}  // namespace trainvm
