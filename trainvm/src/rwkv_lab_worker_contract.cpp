#include "trainvm/rwkv_lab_worker_contract.hpp"

#include <algorithm>
#include <map>
#include <ranges>
#include <stdexcept>
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

OperationAuthoringDeclaration vision_native_head_authoring() {
  OperationAuthoringDeclaration authoring = checkpoint_authoring();
  auto& checkpoint = authoring.outputs.at("checkpoint");
  checkpoint.required = true;
  checkpoint.artifact_schema = "rwkv-lab.vision-native-head-checkpoint.v1";
  checkpoint.description =
      "Required compatible native RWKV vision-head checkpoint.";
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
      key("rwkv-lab.vision-native-head",
          "rwkv_lab.vision_native_head.v1.Train"),
      code_fingerprint, vision_native_head_composition(),
      resumable_training_lifecycle(), vision_native_head_authoring()));
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
          "optimizer.torch_sparse_adam.v1",
          "parameter_router.mageflow_appearance_expert.v1",
          "parameter_router.mageflow_terminal_expert.v1",
          "precision.bf16_parameters_fp32_reductions.v1",
          "precision.fp32_parameters_bf16_compute.v1",
          "schedule.constant.v1",
          "schedule.linear_warmup_constant.v1",
          "schedule.linear_warmup_cosine.v1",
          "schedule.powercool.v1",
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
      {"rwkv-lab.mageflow-appearance-expert",
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
      {"rwkv-lab.vision-teacher-compressor",
       canonical_distributions(
           {"grpcio", "numpy", "pillow", "protobuf", "safetensors",
            "torch", "transformers"})},
      {"rwkv-lab.vision-native-head",
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
