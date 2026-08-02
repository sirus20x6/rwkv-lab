#include "trainvm/rwkv_lab_worker_contract.hpp"

#include <utility>

namespace trainvm {
namespace {

AdapterKey key(std::string adapter, std::string contract) {
  return {
      .adapter = std::move(adapter),
      .version = "1.0.0",
      .runtime = ComponentRuntime::python_worker,
      .operation = "train",
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

AdapterProfile profile(AdapterKey adapter_key, std::string code_fingerprint,
                       TrainingCompositionContract composition,
                       OperationLifecycleCapabilities lifecycle) {
  return {
      .key = std::move(adapter_key),
      .effect = Effect::process,
      .idempotency = Idempotency::receipt_required,
      .code_fingerprint = std::move(code_fingerprint),
      .required_capabilities = {"worker.controls", "worker.metrics"},
      .lifecycle = lifecycle,
      .training_composition = std::move(composition),
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
      resumable_training_lifecycle()));
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
      resumable_training_lifecycle()));
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
       .compile = false,
       .warmup = false,
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
          "curriculum.context_length.v1",
          "gradient_accumulation.fixed.v1",
          "gradient_clipping.global_norm.v1",
          "normalization.layer_norm.v1",
          "objective.linear_head_cross_entropy.v1",
          "optimizer.fp32_master_adamw.v1",
          "optimizer.fp32_master_adamw_no_decay.v2",
          "optimizer.torch_adamw.v1",
          "optimizer.torch_adamw_no_decay.v2",
          "parameter_router.mageflow_appearance_expert.v1",
          "parameter_router.mageflow_terminal_expert.v1",
          "precision.bf16_parameters_fp32_reductions.v1",
          "schedule.linear_warmup_cosine.v1",
          "schedule.powercool.v1",
          "weight_decay_schedule.constant.v1",
          "worker.controls",
          "worker.metrics",
      },
  };
}

}  // namespace trainvm
