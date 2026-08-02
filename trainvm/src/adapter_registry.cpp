#include "trainvm/adapter_registry.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <algorithm>
#include <iterator>
#include <ranges>
#include <set>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <utility>

#include <nlohmann/json.hpp>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

class FileDescriptor {
 public:
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  ~FileDescriptor() {
    if (descriptor_ >= 0) (void)::close(descriptor_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  [[nodiscard]] int get() const { return descriptor_; }

 private:
  int descriptor_;
};

bool valid_sha256_fingerprint(std::string_view value) {
  constexpr std::string_view prefix = "sha256:";
  return value.size() == prefix.size() + 64U && value.starts_with(prefix) &&
         std::ranges::all_of(value.substr(prefix.size()), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool symbolic_identity(std::string_view value) {
  constexpr std::size_t kMaximumIdentityBytes = 192U;
  if (value.empty() || value.size() > kMaximumIdentityBytes ||
      !((value.front() >= 'a' && value.front() <= 'z') ||
        (value.front() >= 'A' && value.front() <= 'Z'))) {
    return false;
  }
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.' || character == ':';
  });
}

void validate_training_contract(
    const TrainingCompositionContract& contract) {
  if (!symbolic_identity(contract.model_family) || contract.slots.empty() ||
      contract.slots.size() > 64U ||
      std::ranges::any_of(contract.slots, [](const auto& slot) {
        return !symbolic_identity(slot.first) ||
               enum_to_string(slot.second) == "<invalid>";
      })) {
    throw std::invalid_argument(
        "adapter training composition contract must declare a bounded model "
        "family and between 1 and 64 symbolic slots");
  }
}

OperationPortDescriptor port(OperationPortType type, bool required,
                             std::optional<ArtifactType> artifact_type = {}) {
  return {
      .type = type,
      .required = required,
      .artifact_type = artifact_type,
      .artifact_schema = std::nullopt,
      .description = std::nullopt,
  };
}

OperationAuthoringDeclaration core_authoring(std::string_view operation) {
  if (operation == "acquire_resources" || operation == "release_resources") {
    return {
        .inputs = {{"concurrency_key",
                    port(OperationPortType::string, true)}},
        .outputs = {},
    };
  }
  if (operation == "validate_artifact") {
    return {
        .inputs = {
            {"artifact", port(OperationPortType::artifact, true)},
            {"required_schema", port(OperationPortType::string, true)},
        },
        .outputs = {},
    };
  }
  throw std::logic_error("core authoring declaration has no typed executor");
}

void validate_port(std::string_view name, const OperationPortDescriptor& port) {
  constexpr std::size_t kMaximumDescriptionBytes = 4U << 10U;
  constexpr std::size_t kMaximumSchemaBytes = 512U;
  if (!symbolic_identity(name)) {
    throw std::invalid_argument(
        "adapter operation port names must be bounded symbolic identities");
  }
  if (enum_to_string(port.type) == "<invalid>" ||
      (port.artifact_type &&
       enum_to_string(*port.artifact_type) == "<invalid>")) {
    throw std::invalid_argument(
        "adapter operation ports must use closed value and artifact types");
  }
  if (port.type != OperationPortType::artifact &&
      (port.artifact_type || port.artifact_schema)) {
    throw std::invalid_argument(
        "only artifact operation ports may narrow artifact type or schema");
  }
  if ((port.artifact_schema &&
       (port.artifact_schema->empty() ||
        port.artifact_schema->size() > kMaximumSchemaBytes)) ||
      (port.description &&
       port.description->size() > kMaximumDescriptionBytes)) {
    throw std::invalid_argument(
        "adapter operation port schema and description must be bounded");
  }
}

void validate_authoring(const OperationAuthoringDeclaration& authoring) {
  constexpr std::size_t kMaximumPorts = 64U;
  if (authoring.inputs.size() > kMaximumPorts ||
      authoring.outputs.size() > kMaximumPorts) {
    throw std::invalid_argument(
        "adapter operation authoring declarations support at most 64 inputs "
        "and 64 outputs");
  }
  for (const auto& [name, descriptor] : authoring.inputs) {
    validate_port(name, descriptor);
  }
  for (const auto& [name, descriptor] : authoring.outputs) {
    validate_port(name, descriptor);
    if (descriptor.type != OperationPortType::artifact) {
      throw std::invalid_argument(
          "adapter operation outputs must be artifact ports because workflow "
          "publishes bind logical artifacts");
    }
  }
}

std::optional<OperationPortType> literal_type(const nlohmann::json& value) {
  if (value.is_string()) return OperationPortType::string;
  if (value.is_number_integer() || value.is_number_unsigned()) {
    return OperationPortType::integer;
  }
  if (value.is_number_float()) return OperationPortType::number;
  if (value.is_boolean()) return OperationPortType::boolean;
  if (value.is_object()) return OperationPortType::object;
  return std::nullopt;
}

OperationPortType parameter_type(ParameterType type) {
  switch (type) {
    case ParameterType::string:
    case ParameterType::path:
    case ParameterType::duration:
      return OperationPortType::string;
    case ParameterType::integer:
      return OperationPortType::integer;
    case ParameterType::number:
      return OperationPortType::number;
    case ParameterType::boolean:
      return OperationPortType::boolean;
  }
  throw AdapterResolutionError("parameter uses an invalid closed type");
}

OperationPortType control_type(ControlType type) {
  switch (type) {
    case ControlType::number:
      return OperationPortType::number;
    case ControlType::integer:
      return OperationPortType::integer;
    case ControlType::boolean:
      return OperationPortType::boolean;
    case ControlType::string:
    case ControlType::enumeration:
      return OperationPortType::string;
  }
  throw AdapterResolutionError("control uses an invalid closed type");
}

bool compatible_value_type(OperationPortType expected,
                           OperationPortType actual) {
  return expected == actual ||
         (expected == OperationPortType::number &&
          actual == OperationPortType::integer);
}

const Artifact* bound_artifact(const Binding& binding, const Spec& spec) {
  std::string logical_name;
  if (binding.artifact) {
    logical_name = *binding.artifact;
  } else if (binding.node_output) {
    const auto producer = spec.workflow.nodes.find(binding.node_output->node);
    if (producer == spec.workflow.nodes.end() || !producer->second.publishes) {
      return nullptr;
    }
    const auto published =
        producer->second.publishes->find(binding.node_output->name);
    if (published == producer->second.publishes->end()) return nullptr;
    logical_name = published->second;
  } else {
    return nullptr;
  }
  const auto artifact = spec.artifacts.find(logical_name);
  return artifact == spec.artifacts.end() ? nullptr : &artifact->second;
}

std::optional<OperationPortType> bound_value_type(const Binding& binding,
                                                  const Spec& spec) {
  if (binding.literal) return literal_type(*binding.literal);
  if (binding.parameter) {
    const auto parameter = spec.parameters.find(*binding.parameter);
    if (parameter != spec.parameters.end()) {
      return parameter_type(parameter->second.type);
    }
  }
  if (binding.control) {
    const auto control = spec.controls.catalog.find(*binding.control);
    if (control != spec.controls.catalog.end()) {
      return control_type(control->second.type);
    }
  }
  if (binding.context) {
    return *binding.context == "plan_revision" ? OperationPortType::integer
                                                : OperationPortType::string;
  }
  if (binding.artifact || binding.node_output) {
    return OperationPortType::artifact;
  }
  return std::nullopt;
}

void require_artifact_contract(const Artifact& artifact,
                               const OperationPortDescriptor& port,
                               std::string_view message_prefix) {
  if (port.artifact_type && artifact.type != *port.artifact_type) {
    throw AdapterResolutionError(std::string(message_prefix) +
                                 " has an incompatible artifact type");
  }
  if (port.artifact_schema &&
      (!artifact.schema || *artifact.schema != *port.artifact_schema)) {
    throw AdapterResolutionError(std::string(message_prefix) +
                                 " has an incompatible artifact schema");
  }
}

void validate_operation_authoring(const std::string& node_name,
                                  const Node& node,
                                  const AdapterProfile& profile,
                                  const Spec& spec) {
  if (!profile.authoring) {
    throw AdapterResolutionError(
        "resolved operation unexpectedly lacks authoring authority");
  }
  const OperationAuthoringDeclaration& authoring = *profile.authoring;
  for (const auto& [name, port] : authoring.inputs) {
    if (port.required && !node.invoke.inputs.contains(name)) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " omits required operation input " + name);
    }
  }
  for (const auto& [name, binding] : node.invoke.inputs) {
    const auto declared = authoring.inputs.find(name);
    if (declared == authoring.inputs.end()) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " supplies undeclared operation input " +
                                   name);
    }
    const auto actual = bound_value_type(binding, spec);
    if (!actual || !compatible_value_type(declared->second.type, *actual)) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " input " + name +
                                   " has an incompatible value type");
    }
    if (declared->second.type == OperationPortType::artifact) {
      const Artifact* artifact = bound_artifact(binding, spec);
      if (artifact == nullptr) {
        throw AdapterResolutionError("workflow node " + node_name +
                                     " input " + name +
                                     " has no statically known artifact contract");
      }
      require_artifact_contract(
          *artifact, declared->second,
          "workflow node " + node_name + " input " + name);
    }
  }

  const std::map<std::string, std::string> empty_publishes;
  const auto& publishes = node.publishes ? *node.publishes : empty_publishes;
  for (const auto& [name, port] : authoring.outputs) {
    if (port.required && !publishes.contains(name)) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " omits required operation output " + name);
    }
  }
  for (const auto& [name, logical_name] : publishes) {
    const auto declared = authoring.outputs.find(name);
    if (declared == authoring.outputs.end()) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " publishes undeclared operation output " +
                                   name);
    }
    const auto artifact = spec.artifacts.find(logical_name);
    if (artifact == spec.artifacts.end()) {
      throw AdapterResolutionError("workflow node " + node_name +
                                   " output " + name +
                                   " has no declared logical artifact");
    }
    require_artifact_contract(
        artifact->second, declared->second,
        "workflow node " + node_name + " output " + name);
  }
}

std::vector<AdapterProfile> core_profiles() {
  const auto key = [](std::string operation, std::string contract) {
    return AdapterKey{.adapter = "trainvm.core",
                      .version = "1.0.0",
                      .runtime = ComponentRuntime::builtin,
                      .operation = std::move(operation),
                      .contract = std::move(contract)};
  };
  return {
      {.key = key("acquire_resources", "trainvm.v1.AcquireResources"),
       .effect = Effect::resource,
       .idempotency = Idempotency::receipt_required,
       .code_fingerprint = {},
       .required_capabilities = {},
       .authoring = core_authoring("acquire_resources")},
      {.key = key("validate_artifact", "trainvm.v1.ValidateArtifact"),
       .effect = Effect::read_only,
       .idempotency = Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {},
       .authoring = core_authoring("validate_artifact")},
      {.key = key("release_resources", "trainvm.v1.ReleaseResources"),
       .effect = Effect::resource,
       .idempotency = Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {},
       .authoring = core_authoring("release_resources")},
  };
}

std::vector<std::string> canonical_capabilities(
    std::vector<std::string> capabilities) {
  if (capabilities.size() > 256U ||
      std::ranges::any_of(capabilities, [](const std::string& capability) {
        return capability.empty() || capability.size() > 256U;
      })) {
    throw std::invalid_argument(
        "adapter profile capabilities must be bounded and nonempty");
  }
  std::ranges::sort(capabilities);
  if (std::ranges::adjacent_find(capabilities) != capabilities.end()) {
    throw std::invalid_argument(
        "adapter profile capabilities must be unique");
  }
  return capabilities;
}

bool resume_from_checkpoint(ResumeGrade grade) {
  return grade == ResumeGrade::compatible || grade == ResumeGrade::exact;
}

void validate_lifecycle(const OperationLifecycleCapabilities& lifecycle) {
  if (!lifecycle.stateful) {
    if (lifecycle.checkpoint_now || lifecycle.pause_keep_resources ||
        lifecycle.pause_release_resources ||
        lifecycle.resume_grade != ResumeGrade::none) {
      throw std::invalid_argument(
          "stateless adapter operations cannot declare checkpoint, pause, or "
          "resume capabilities");
    }
    return;
  }
  if (lifecycle.resume_grade == ResumeGrade::exact &&
      !lifecycle.checkpoint_now) {
    throw std::invalid_argument(
        "exact-resume adapter operations must support checkpoint_now");
  }
  if (lifecycle.resume_grade == ResumeGrade::terminal_checkpoint &&
      lifecycle.checkpoint_now) {
    throw std::invalid_argument(
        "terminal-checkpoint adapter operations cannot support checkpoint_now");
  }
  if (lifecycle.pause_release_resources &&
      (!lifecycle.checkpoint_now ||
       !resume_from_checkpoint(lifecycle.resume_grade))) {
    throw std::invalid_argument(
        "pause_release_resources requires checkpoint_now and a resumable "
        "checkpoint grade");
  }
}

void validate_profile(AdapterProfile& profile) {
  if (profile.key.adapter.empty() || profile.key.version.empty() ||
      profile.key.operation.empty() || profile.key.contract.empty()) {
    throw std::invalid_argument("adapter profile key fields must not be empty");
  }
  const bool builtin = profile.key.runtime == ComponentRuntime::builtin;
  if (!profile.authoring) {
    throw std::invalid_argument(
        "adapter profile must publish an explicit operation authoring "
        "declaration");
  }
  validate_authoring(*profile.authoring);
  validate_lifecycle(profile.lifecycle);
  if (profile.training_composition) {
    validate_training_contract(*profile.training_composition);
    if (profile.effect != Effect::process) {
      throw std::invalid_argument(
          "adapter training composition contracts require process effect");
    }
  }
  if (profile.lifecycle.stateful && profile.effect != Effect::process) {
    throw std::invalid_argument(
        "stateful adapter operations must have process effect");
  }
  if (builtin) {
    if (!profile.code_fingerprint.empty() ||
        !profile.required_capabilities.empty() ||
        profile.lifecycle != OperationLifecycleCapabilities{} ||
        profile.training_composition) {
      throw std::invalid_argument(
          "builtin adapter profiles cannot carry worker launch, lifecycle, or "
          "training-composition authority");
    }
    const bool acquire =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "acquire_resources" &&
        profile.key.contract == "trainvm.v1.AcquireResources" &&
        profile.effect == Effect::resource &&
        profile.idempotency == Idempotency::receipt_required;
    const bool validate =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "validate_artifact" &&
        profile.key.contract == "trainvm.v1.ValidateArtifact" &&
        profile.effect == Effect::read_only &&
        profile.idempotency == Idempotency::replay_safe;
    const bool release =
        profile.key.adapter == "trainvm.core" &&
        profile.key.version == "1.0.0" &&
        profile.key.operation == "release_resources" &&
        profile.key.contract == "trainvm.v1.ReleaseResources" &&
        profile.effect == Effect::resource &&
        profile.idempotency == Idempotency::replay_safe;
    if (!acquire && !validate && !release) {
      throw std::invalid_argument(
          "builtin adapter profile has no exact typed trainvm.core executor");
    }
    return;
  }
  if (profile.key.adapter.starts_with("coverage.")) {
    throw std::invalid_argument(
        "coverage.* adapter names are reserved for compile-only fixtures");
  }
  if (!valid_sha256_fingerprint(profile.code_fingerprint)) {
    throw std::invalid_argument(
        "worker adapter code fingerprint must be canonical sha256 hex");
  }
  profile.required_capabilities =
      canonical_capabilities(std::move(profile.required_capabilities));
}

}  // namespace

AdapterRegistry::AdapterRegistry(std::vector<AdapterProfile> profiles) {
  constexpr std::size_t kMaximumProfiles = 4U << 10U;
  if (profiles.empty() || profiles.size() > kMaximumProfiles) {
    throw std::invalid_argument(
        "adapter registry must contain between 1 and 4096 operation profiles");
  }
  std::set<std::tuple<std::string, std::string, ComponentRuntime, std::string>>
      selectors;
  for (AdapterProfile& profile : profiles) {
    validate_profile(profile);
    const auto selector = std::tuple{
        profile.key.adapter, profile.key.version, profile.key.runtime,
        profile.key.operation};
    if (!selectors.insert(selector).second) {
      throw std::invalid_argument(
          "adapter registry contains a duplicate operation selector");
    }
    const AdapterKey key = profile.key;
    if (!profiles_.emplace(key, std::move(profile)).second) {
      throw std::invalid_argument("adapter registry contains a duplicate exact key");
    }
  }
  nlohmann::json canonical_profiles = nlohmann::json::array();
  for (const auto& [key, profile] : profiles_) {
    (void)key;
    canonical_profiles.push_back(encode_json(profile));
  }
  registry_digest_ = "sha256:" +
                     sha256_hex(nlohmann::json{
                                    {"api_version", "trainvm.adapters/v2"},
                                    {"profiles", std::move(canonical_profiles)}}
                                    .dump());
  OperationDescriptorDocument descriptors{
      .api_version = "trainvm.operations/v1",
      .operations = {},
  };
  descriptors.operations.reserve(profiles_.size());
  for (const auto& [key, profile] : profiles_) {
    (void)key;
    descriptors.operations.push_back(profile);
  }
  operation_descriptors_json_ = encode_json(descriptors);
  operation_descriptors_digest_ =
      "sha256:" + sha256_hex(operation_descriptors_json_.dump());
}

AdapterRegistry AdapterRegistry::load_file(
    const std::filesystem::path& path) {
  constexpr std::uintmax_t kMaximumRegistryBytes = 1U << 20U;
  if (path.empty() || !path.is_absolute()) {
    throw std::invalid_argument(
        "adapter registry path must be absolute and nonempty");
  }
  const int raw_descriptor =
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (raw_descriptor < 0) {
    throw std::invalid_argument("could not securely open adapter registry " +
                                path.string() + ": " +
                                std::strerror(errno));
  }
  FileDescriptor descriptor(raw_descriptor);
  struct stat before {};
  if (::fstat(descriptor.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > kMaximumRegistryBytes ||
      (before.st_uid != 0U && before.st_uid != ::geteuid()) ||
      (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw std::invalid_argument(
        "adapter registry must be an owner/root-owned regular file that is not "
        "group/world-writable and is no larger than 1 MiB");
  }
  std::string text(static_cast<std::size_t>(before.st_size), '\0');
  std::size_t offset = 0;
  while (offset < text.size()) {
    const ssize_t count =
        ::read(descriptor.get(), text.data() + offset, text.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) {
      throw std::invalid_argument(
          "adapter registry changed or failed while being read");
    }
    offset += static_cast<std::size_t>(count);
  }
  char extra = '\0';
  const ssize_t trailing = ::read(descriptor.get(), &extra, 1U);
  struct stat after {};
  if (trailing != 0 || ::fstat(descriptor.get(), &after) != 0 ||
      before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
      before.st_size != after.st_size ||
      before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec ||
      before.st_ctim.tv_nsec != after.st_ctim.tv_nsec) {
    throw std::invalid_argument(
        "adapter registry changed while it was being read");
  }
  nlohmann::json source;
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    const nlohmann::json::parser_callback_t reject_duplicates =
        [&](int depth, nlohmann::json::parse_event_t event,
            nlohmann::json& parsed) {
          const auto index = static_cast<std::size_t>(depth);
          if (event == nlohmann::json::parse_event_t::object_start) {
            if (object_keys.size() <= index + 1U) {
              object_keys.resize(index + 2U);
            }
            object_keys[index + 1U].clear();
          } else if (event == nlohmann::json::parse_event_t::key) {
            if (object_keys.size() <= index) object_keys.resize(index + 1U);
            if (!object_keys[index].insert(parsed.get<std::string>()).second) {
              duplicate_key = true;
            }
          } else if (event == nlohmann::json::parse_event_t::object_end &&
                     object_keys.size() > index + 1U) {
            object_keys[index + 1U].clear();
          }
          return true;
        };
    source = nlohmann::json::parse(text, reject_duplicates);
  } catch (const nlohmann::json::exception& exception) {
    throw std::invalid_argument(
        "adapter registry is not valid JSON: " + std::string(exception.what()));
  }
  if (duplicate_key) {
    throw std::invalid_argument(
        "adapter registry JSON contains a duplicate object key");
  }
  AdapterRegistryDocument document;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, document, "", diagnostics)) {
    throw std::invalid_argument(
        "adapter registry schema validation failed: " +
        diagnostics_json(diagnostics).dump());
  }
  if (document.api_version != "trainvm.adapters/v2") {
    throw std::invalid_argument(
        "adapter registry api_version must be trainvm.adapters/v2");
  }
  if (std::ranges::any_of(
          document.profiles, [](const AdapterProfile& profile) {
            return profile.key.runtime == ComponentRuntime::builtin ||
                   profile.key.adapter == "trainvm.core";
          })) {
    throw std::invalid_argument(
        "adapter registry files cannot occupy the compiled trainvm.core namespace");
  }
  std::vector<AdapterProfile> profiles = core_profiles();
  profiles.insert(profiles.end(),
                  std::make_move_iterator(document.profiles.begin()),
                  std::make_move_iterator(document.profiles.end()));
  return AdapterRegistry(std::move(profiles));
}

const AdapterProfile& AdapterRegistry::resolve(
    const Component& component, std::string_view operation) const {
  const auto declared = component.operations.find(std::string(operation));
  if (declared == component.operations.end()) {
    throw AdapterResolutionError(
        "component does not declare the requested adapter operation");
  }
  const AdapterKey key{
      .adapter = component.adapter,
      .version = component.version,
      .runtime = component.runtime,
      .operation = std::string(operation),
      .contract = declared->second.contract,
  };
  const auto profile = profiles_.find(key);
  if (profile == profiles_.end()) {
    throw AdapterResolutionError(
        "no authority-owned adapter profile matches the exact component key");
  }
  return profile->second;
}

const AdapterProfile& AdapterRegistry::resolve(const AdapterKey& key) const {
  const auto profile = profiles_.find(key);
  if (profile == profiles_.end()) {
    throw AdapterResolutionError(
        "no authority-owned adapter profile matches the exact key");
  }
  return profile->second;
}

std::string AdapterRegistry::profile_digest(const AdapterKey& key) const {
  return "sha256:" +
         sha256_hex(nlohmann::json{
                        {"api_version", "trainvm.adapter-profile/v1"},
                        {"profile", encode_json(resolve(key))},
                    }
                        .dump());
}

void AdapterRegistry::validate_plan(const CompiledPlan& plan) const {
  const Workflow& workflow = plan.experiment.spec.workflow;
  std::set<std::string> reachable;
  if (workflow.nodes.contains(workflow.entrypoint)) {
    reachable.insert(workflow.entrypoint);
    bool changed = true;
    while (changed) {
      changed = false;
      const std::vector<std::string> snapshot(reachable.begin(),
                                               reachable.end());
      for (const std::string& name : snapshot) {
        for (const Transition& transition : workflow.nodes.at(name).transitions) {
          if (!transition.target.starts_with('$') &&
              workflow.nodes.contains(transition.target)) {
            changed = reachable.insert(transition.target).second || changed;
          }
        }
      }
    }
  }
  const bool pause_required = std::ranges::any_of(
      plan.experiment.spec.controls.catalog,
      [](const auto& entry) {
        return entry.second.requires_pause.value_or(false);
      });
  for (const auto& [name, component] : plan.experiment.spec.components) {
    (void)name;
    for (const auto& [operation, declaration] : component.operations) {
      (void)declaration;
      (void)resolve(component, operation);
    }
  }
  for (const auto& [name, node] :
       plan.experiment.spec.workflow.nodes) {
    const Component& component =
        plan.experiment.spec.components.at(node.invoke.component);
    const AdapterProfile& profile =
        resolve(component, node.invoke.operation);
    validate_operation_authoring(name, node, profile,
                                 plan.experiment.spec);
    if (profile.effect != node.effect ||
        profile.idempotency != node.idempotency) {
      throw AdapterResolutionError(
          "workflow node " + name +
          " disagrees with its authority-owned effect or idempotency class");
    }
    if (profile.training_composition) {
      if (!node.invoke.training) {
        throw AdapterResolutionError(
            "workflow node " + name +
            " omits the training composition required by its authority-owned adapter profile");
      }
      const TrainingCompositionContract& contract =
          *profile.training_composition;
      const TrainingComposition& composition = *node.invoke.training;
      if (composition.model_family != contract.model_family) {
        throw AdapterResolutionError(
            "workflow node " + name +
            " training model family disagrees with its authority-owned adapter profile");
      }
      if (composition.components.size() != contract.slots.size()) {
        throw AdapterResolutionError(
            "workflow node " + name +
            " training component slots disagree with its authority-owned adapter profile");
      }
      for (const auto& [slot, category] : contract.slots) {
        const auto selected = composition.components.find(slot);
        if (selected == composition.components.end() ||
            selected->second.key.category != category) {
          throw AdapterResolutionError(
              "workflow node " + name + " training component slot " + slot +
              " disagrees with its authority-owned category contract");
        }
      }
    } else if (node.invoke.training) {
      throw AdapterResolutionError(
          "workflow node " + name +
          " attaches a training composition to an adapter profile that does not consume one");
    }
    if (reachable.contains(name)) {
      if (node.effect == Effect::process &&
          plan.experiment.spec.recovery.exact_resume &&
          node.idempotency == Idempotency::at_most_once) {
        throw AdapterResolutionError(
            "workflow node " + name +
            " requests exact resume but is an at-most-once process operation");
      }
      if (profile.lifecycle.stateful) {
        if (!profile.lifecycle.graceful_stop) {
          throw AdapterResolutionError(
              "workflow node " + name +
              " has a graceful-stop timeout but its stateful operation cannot stop gracefully");
        }
        if (plan.experiment.spec.recovery.exact_resume &&
            profile.lifecycle.resume_grade != ResumeGrade::exact) {
          throw AdapterResolutionError(
              "workflow node " + name +
              " requests exact resume but its stateful process operation is not "
              "exact-resumable");
        }
        if (plan.experiment.spec.recovery.release_accelerators_when_paused) {
          const bool release = *plan.experiment.spec.recovery
                                    .release_accelerators_when_paused;
          const bool supported =
              release ? profile.lifecycle.pause_release_resources
                      : profile.lifecycle.pause_keep_resources;
          if (!supported) {
            throw AdapterResolutionError(
                "workflow node " + name +
                (release
                     ? " cannot pause while releasing resources as requested"
                     : " cannot pause while retaining resources as requested"));
          }
        } else if (pause_required &&
                   !profile.lifecycle.pause_keep_resources &&
                   !profile.lifecycle.pause_release_resources) {
          throw AdapterResolutionError(
              "workflow node " + name +
              " is targeted by a pause-required plan without a supported pause protocol");
        }
      }
    }
  }
  if (plan.experiment.spec.execution) {
    const ExecutionPhases& execution = *plan.experiment.spec.execution;
    const Component& component =
        plan.experiment.spec.components.at(execution.component);
    const AdapterProfile& profile =
        resolve(component, execution.operation);
    const bool reachable_target = std::ranges::any_of(
        reachable, [&](const std::string& name) {
          const Invocation& invocation = workflow.nodes.at(name).invoke;
          return invocation.component == execution.component &&
                 invocation.operation == execution.operation;
        });
    if (!reachable_target) {
      throw AdapterResolutionError(
          "execution phases must target an operation invoked by a reachable workflow node");
    }
    const auto reject_unsupported = [&](bool requested, bool supported,
                                        std::string_view capability) {
      if (requested && !supported) {
        throw AdapterResolutionError(
            "execution requests " + std::string(capability) +
            " but the authority-owned operation does not support it");
      }
    };
    reject_unsupported(execution.compile && execution.compile->enabled,
                       profile.lifecycle.compile, "compile");
    reject_unsupported(execution.warmup && execution.warmup->enabled,
                       profile.lifecycle.warmup, "warmup");
    reject_unsupported(execution.qualify && execution.qualify->enabled,
                       profile.lifecycle.qualify, "qualify");
    reject_unsupported(execution.gpu_trace && execution.gpu_trace->enabled,
                       profile.lifecycle.profile, "profile");
  }
}

WorkerLaunchRequest AdapterRegistry::worker_launch_request(
    const Component& component, std::string_view operation) const {
  const AdapterProfile& profile = resolve(component, operation);
  if (profile.key.runtime == ComponentRuntime::builtin) {
    throw AdapterResolutionError(
        "builtin adapter profiles cannot authorize a worker launch");
  }
  return WorkerLaunchRequest{
      .code_fingerprint = profile.code_fingerprint,
      .required_capabilities = profile.required_capabilities,
  };
}

const std::string& AdapterRegistry::registry_digest() const {
  return registry_digest_;
}

nlohmann::json AdapterRegistry::operation_descriptors_json() const {
  return operation_descriptors_json_;
}

const std::string& AdapterRegistry::operation_descriptors_digest() const {
  return operation_descriptors_digest_;
}

std::string AdapterRegistry::plan_lock_digest(const CompiledPlan& plan) const {
  return "sha256:" + sha256_hex(plan_lock_manifest(plan));
}

std::string AdapterRegistry::plan_lock_manifest(
    const CompiledPlan& plan) const {
  validate_plan(plan);
  std::set<std::string> canonical_profiles;
  for (const auto& [name, component] : plan.experiment.spec.components) {
    (void)name;
    for (const auto& [operation, declaration] : component.operations) {
      (void)declaration;
      canonical_profiles.insert(encode_json(resolve(component, operation)).dump());
    }
  }
  nlohmann::json locked = nlohmann::json::array();
  for (const std::string& profile : canonical_profiles) {
    locked.push_back(nlohmann::json::parse(profile));
  }
  return nlohmann::json{{"api_version", "trainvm.adapter-lock/v2"},
                        {"profiles", std::move(locked)}}
      .dump();
}

void AdapterRegistry::validate_submission_lock(
    const CompiledPlan& plan, const nlohmann::json& submission) const {
  const std::string manifest = plan_lock_manifest(plan);
  const std::string digest = "sha256:" + sha256_hex(manifest);
  if (!submission.is_object() ||
      submission.value("adapter_lock_digest", std::string{}) != digest ||
      !submission.contains("adapter_lock") ||
      submission.at("adapter_lock") != nlohmann::json::parse(manifest)) {
    throw AdapterResolutionError(
        "run adapter lock differs from the active authority registry");
  }
}

}  // namespace trainvm
