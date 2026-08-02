#include "trainvm/adapter_invocation.hpp"

#include <array>
#include <ranges>
#include <stdexcept>

#include "trainvm/control.hpp"
#include "trainvm/document.hpp"
#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

using Json = nlohmann::json;

[[noreturn]] void reject(std::string message) {
  throw AdapterResolutionError(std::move(message));
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_plan_hash(std::string_view value) {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void exact_fields(const Json& value,
                  std::initializer_list<std::string_view> fields) {
  if (!value.is_object() || value.size() != fields.size())
    reject("worker invocation fields are inexact");
  for (const std::string_view field : fields)
    if (!value.contains(std::string(field)))
      reject("worker invocation field is missing");
}

Json invocation_body(const WorkerInvocationSpec& value) {
  const bool legacy_v1 = value.api_version == kWorkerInvocationApiVersionV1;
  if ((!legacy_v1 && value.api_version != kWorkerInvocationApiVersion) ||
      value.run_id.empty() || !valid_digest(value.host_id) ||
      !valid_plan_hash(value.plan_hash) ||
      value.plan_revision == 0U || value.node_id.empty() ||
      value.attempt_id.empty() || value.dispatch_id.empty() ||
      value.adapter.adapter.empty() || value.adapter.version.empty() ||
      value.adapter.operation.empty() || value.adapter.contract.empty() ||
      value.adapter.runtime == ComponentRuntime::builtin ||
      value.adapter.runtime == ComponentRuntime::external_worker ||
      !value.workspace.is_object() || !value.resources.is_object() ||
      !value.inputs.is_object() || !value.controls.is_object() ||
      !value.publishes.is_object() || !value.observability.is_object() ||
      (!value.execution.is_null() && !value.execution.is_object()) ||
      (!value.training.is_null() && !value.training.is_object()) ||
      (!value.resume.is_null() && !value.resume.is_object()) ||
      (legacy_v1 && !value.resume.is_null())) {
    reject("worker invocation semantics are invalid");
  }
  Json body{{"adapter", encode_json(value.adapter)},
            {"api_version", value.api_version},
            {"attempt_id", value.attempt_id},
            {"controls", value.controls},
            {"dispatch_id", value.dispatch_id},
            {"effective_control_revision",
             value.effective_control_revision},
            {"execution", value.execution},
            {"host_id", value.host_id},
            {"inputs", value.inputs},
            {"node_id", value.node_id},
            {"observability", value.observability},
            {"plan_hash", value.plan_hash},
            {"plan_revision", value.plan_revision},
            {"publishes", value.publishes},
            {"resources", value.resources},
            {"run_id", value.run_id},
            {"training", value.training},
            {"workspace", value.workspace}};
  if (!legacy_v1) body["resume"] = value.resume;
  return body;
}

Json require_artifact_manifest(std::string_view input_name,
                               const std::string& logical_name,
                               const Json& manifest, const Spec& spec) {
  const auto declaration = spec.artifacts.find(logical_name);
  if (declaration == spec.artifacts.end() || !manifest.is_object())
    reject("worker invocation has no declared artifact for input " +
           std::string(input_name));
  constexpr std::array<std::string_view, 13> fields{
      "artifact_id", "logical_name", "kind", "schema", "uri",
      "size_bytes", "fingerprint_algorithm", "fingerprint", "complete",
      "producer_node_id", "producer_attempt_id", "parent_artifact_ids",
      "published_at_ns"};
  if (manifest.size() != fields.size() ||
      !std::ranges::all_of(fields, [&](std::string_view field) {
        return manifest.contains(std::string(field));
      }))
    reject("worker invocation artifact manifest fields are inexact");
  const Json encoded_declaration = encode_json(declaration->second);
  const std::string expected_schema =
      encoded_declaration.value("schema", std::string{});
  if (manifest.value("logical_name", std::string{}) != logical_name ||
      manifest.value("kind", std::string{}) !=
          encoded_declaration.value("type", std::string{}) ||
      manifest.value("schema", std::string{}) != expected_schema ||
      manifest.value("fingerprint_algorithm", std::string{}) !=
          encoded_declaration.value("fingerprint", std::string{}) ||
      !manifest.value("complete", false) ||
      manifest.value("artifact_id", std::string{}).empty() ||
      manifest.value("uri", std::string{}).empty() ||
      manifest.value("producer_node_id", std::string{}).empty() ||
      manifest.value("producer_attempt_id", std::string{}).empty() ||
      !manifest.at("size_bytes").is_number_unsigned() ||
      !manifest.at("published_at_ns").is_number_integer() ||
      !manifest.at("parent_artifact_ids").is_array()) {
    reject("worker invocation artifact manifest disagrees with declaration");
  }
  return manifest;
}

Json require_resume_checkpoint(const Json& resume, const Spec& spec,
                               const WorkerInvocationContext& context) {
  exact_fields(resume, {"api_version", "checkpoint", "optimizer_step",
                        "pause_command_id", "resume_command_id"});
  if (resume.value("api_version", std::string{}) !=
          "trainvm.resume-checkpoint/v1" ||
      !resume.at("optimizer_step").is_number_unsigned() ||
      resume.value("pause_command_id", std::string{}).empty() ||
      resume.value("resume_command_id", std::string{}).empty()) {
    reject("worker invocation resume authority is malformed");
  }
  const Json& checkpoint = resume.at("checkpoint");
  const std::string logical_name =
      checkpoint.value("logical_name", std::string{});
  const Json validated = require_artifact_manifest(
      "resume", logical_name, checkpoint, spec);
  if (validated.value("kind", std::string{}) != "checkpoint" ||
      validated.value("producer_node_id", std::string{}) != context.node_id ||
      validated.value("producer_attempt_id", std::string{}).empty() ||
      validated.value("producer_attempt_id", std::string{}) ==
          context.attempt_id ||
      validated.value("fingerprint_algorithm", std::string{}) !=
          "manifest_sha256" ||
      !valid_digest(validated.value("fingerprint", std::string{}))) {
    reject("worker invocation resume checkpoint lineage is invalid");
  }
  return resume;
}

void seal(WorkerInvocationSpec& value) {
  const std::string body = invocation_body(value).dump();
  if (body.size() > kMaximumWorkerInvocationBytes)
    reject("worker invocation exceeds its canonical size bound");
  value.invocation_digest = "sha256:" + sha256_hex(body);
}

Json context_value(std::string_view name, const Spec& spec,
                   const WorkerInvocationContext& context) {
  if (name == "run_id") return context.run_id;
  if (name == "run_directory")
    return spec.workspace.run_directory;
  if (name == "plan_revision") return context.plan_revision;
  if (name == "attempt_id") return context.attempt_id;
  if (name == "host_id") return context.host_id;
  reject("worker invocation contains an unsupported context binding");
}

std::string artifact_name_for(const Binding& binding, const Spec& spec) {
  if (binding.artifact) return *binding.artifact;
  if (binding.node_output) {
    const Node& producer =
        spec.workflow.nodes.at(binding.node_output->node);
    return producer.publishes->at(binding.node_output->name);
  }
  return {};
}

Json resolve_inputs(const Node& node, const Spec& spec,
                    const WorkerInvocationContext& context,
                    const Json& controls) {
  Json inputs = Json::object();
  for (const auto& [name, binding] : node.invoke.inputs) {
    if (binding.literal) {
      inputs[name] = *binding.literal;
    } else if (binding.parameter) {
      inputs[name] = spec.parameters.at(*binding.parameter).value;
    } else if (binding.control) {
      inputs[name] = controls.at(*binding.control);
    } else if (binding.context) {
      inputs[name] = context_value(*binding.context, spec, context);
    } else {
      const std::string artifact_name = artifact_name_for(binding, spec);
      const auto artifact = context.artifacts.find(artifact_name);
      if (artifact_name.empty() || artifact == context.artifacts.end() ||
          !artifact->second.is_object()) {
        reject("worker invocation has no complete artifact for input " + name);
      }
      inputs[name] = require_artifact_manifest(
          name, artifact_name, artifact->second, spec);
    }
  }
  return inputs;
}

bool control_targets_operation(std::string_view name,
                               const Control& declaration,
                               const Node& node) {
  const std::string target_prefix = node.invoke.component + "." +
                                    node.invoke.operation + ".";
  if (std::ranges::any_of(declaration.targets, [&](const std::string& target) {
        return target.starts_with(target_prefix);
      }))
    return true;
  return std::ranges::any_of(node.invoke.inputs, [&](const auto& item) {
    return item.second.control && *item.second.control == name;
  });
}

}  // namespace

WorkerInvocationSpec build_worker_invocation(
    const CompiledPlan& plan, const WorkerInvocationContext& context) {
  const Spec& spec = plan.experiment.spec;
  const auto selected = spec.workflow.nodes.find(context.node_id);
  if (selected == spec.workflow.nodes.end())
    reject("worker invocation node is absent from the immutable plan");
  if (context.run_id.empty() || context.attempt_id.empty() ||
      context.dispatch_id.empty())
    reject("worker invocation identity is incomplete");
  if (context.plan_revision == 0U)
    reject("worker invocation plan revision is invalid");
  if (context.host_id.empty())
    reject("worker invocation host identity is missing");
  if (!context.effective_controls.is_object())
    reject("worker invocation effective controls are not an object");
  if (selected->second.invoke.training) {
    if (!context.resolved_training.is_object())
      reject("worker invocation has no authority-resolved training composition");
  } else if (!context.resolved_training.is_null()) {
    reject("worker invocation supplied training components to an undeclared node");
  }
  Json resume = nullptr;
  if (!context.resume.is_null()) {
    if (!context.resume.is_object())
      reject("worker invocation resume authority is not an object");
    resume = require_resume_checkpoint(context.resume, spec, context);
  }
  const Node& node = selected->second;
  const Component& component = spec.components.at(node.invoke.component);
  const Operation& operation = component.operations.at(node.invoke.operation);
  Json controls = Json::object();
  for (const auto& [name, declaration] : spec.controls.catalog) {
    if (control_targets_operation(name, declaration, node))
      controls[name] = declaration.default_value;
  }
  for (const auto& [name, value] : context.effective_controls.items()) {
    if (!spec.controls.catalog.contains(name))
      reject("effective worker control is absent from the immutable plan");
    if (controls.contains(name)) controls[name] = value;
  }
  if (!context.effective_controls.empty()) {
    const ControlPatchValidation validation = validate_control_patch(
        plan, context.effective_controls, false, false);
    if (!validation.valid())
      reject("effective worker controls violate the immutable plan");
  }
  Json publishes = Json::object();
  if (node.publishes) {
    for (const auto& [output, logical_name] : *node.publishes) {
      publishes[output] = {
          {"declaration", encode_json(spec.artifacts.at(logical_name))},
          {"logical_name", logical_name},
      };
    }
  }
  Json execution = nullptr;
  if (spec.execution && spec.execution->component == node.invoke.component &&
      spec.execution->operation == node.invoke.operation)
    execution = encode_json(*spec.execution);
  WorkerInvocationSpec result{
      .api_version = std::string(kWorkerInvocationApiVersion),
      .run_id = context.run_id,
      .host_id = context.host_id,
      .plan_hash = plan.plan_hash,
      .plan_revision = context.plan_revision,
      .node_id = context.node_id,
      .attempt_id = context.attempt_id,
      .dispatch_id = context.dispatch_id,
      .adapter = {.adapter = component.adapter,
                  .version = component.version,
                  .runtime = component.runtime,
                  .operation = node.invoke.operation,
                  .contract = operation.contract},
      .workspace = encode_json(spec.workspace),
      .resources = encode_json(spec.resources),
      .inputs = resolve_inputs(node, spec, context, controls),
      .controls = std::move(controls),
      .effective_control_revision = context.effective_control_revision,
      .publishes = std::move(publishes),
      .observability = encode_json(spec.observability),
      .execution = std::move(execution),
      .training = context.resolved_training,
      .resume = std::move(resume),
      .invocation_digest = {},
  };
  seal(result);
  return result;
}

std::string worker_invocation_canonical_json(
    const WorkerInvocationSpec& value) {
  WorkerInvocationSpec canonical = value;
  seal(canonical);
  if (canonical.invocation_digest != value.invocation_digest)
    reject("worker invocation digest is not canonical");
  Json output = invocation_body(value);
  output["invocation_digest"] = value.invocation_digest;
  const std::string encoded = output.dump();
  if (encoded.size() > kMaximumWorkerInvocationBytes)
    reject("worker invocation exceeds its wire size bound");
  return encoded;
}

WorkerInvocationSpec worker_invocation_from_canonical_json(
    std::string_view value) {
  if (value.empty() || value.size() > kMaximumWorkerInvocationBytes)
    reject("worker invocation canonical JSON size is invalid");
  try {
    const Json parsed = Json::parse(value);
    if (parsed.dump() != value)
      reject("worker invocation JSON is not canonical");
    const std::string api_version =
        parsed.value("api_version", std::string{});
    const bool legacy_v1 = api_version == kWorkerInvocationApiVersionV1;
    if (legacy_v1) {
      exact_fields(parsed,
                   {"adapter", "api_version", "attempt_id", "controls",
                    "dispatch_id", "effective_control_revision", "execution",
                    "host_id", "inputs", "invocation_digest", "node_id",
                    "observability", "plan_hash", "plan_revision", "publishes",
                    "resources", "run_id", "training", "workspace"});
    } else {
      exact_fields(parsed,
                   {"adapter", "api_version", "attempt_id", "controls",
                    "dispatch_id", "effective_control_revision", "execution",
                    "host_id", "inputs", "invocation_digest", "node_id",
                    "observability", "plan_hash", "plan_revision", "publishes",
                    "resources", "resume", "run_id", "training", "workspace"});
    }
    AdapterKey adapter;
    std::vector<Diagnostic> diagnostics;
    if (!decode_json(parsed.at("adapter"), adapter, "/adapter", diagnostics))
      reject("worker invocation adapter key is malformed");
    WorkerInvocationSpec result{
        .api_version = api_version,
        .run_id = parsed.at("run_id").get<std::string>(),
        .host_id = parsed.at("host_id").get<std::string>(),
        .plan_hash = parsed.at("plan_hash").get<std::string>(),
        .plan_revision = parsed.at("plan_revision").get<std::uint64_t>(),
        .node_id = parsed.at("node_id").get<std::string>(),
        .attempt_id = parsed.at("attempt_id").get<std::string>(),
        .dispatch_id = parsed.at("dispatch_id").get<std::string>(),
        .adapter = std::move(adapter),
        .workspace = parsed.at("workspace"),
        .resources = parsed.at("resources"),
        .inputs = parsed.at("inputs"),
        .controls = parsed.at("controls"),
        .effective_control_revision =
            parsed.at("effective_control_revision").get<std::uint64_t>(),
        .publishes = parsed.at("publishes"),
        .observability = parsed.at("observability"),
        .execution = parsed.at("execution"),
        .training = parsed.at("training"),
        .resume = legacy_v1 ? Json(nullptr) : parsed.at("resume"),
        .invocation_digest =
            parsed.at("invocation_digest").get<std::string>(),
    };
    if (!valid_digest(result.invocation_digest) ||
        worker_invocation_canonical_json(result) != value)
      reject("worker invocation is not canonical or content-addressed");
    return result;
  } catch (const AdapterResolutionError&) {
    throw;
  } catch (...) {
    reject("worker invocation decoding failed closed");
  }
}

}  // namespace trainvm
