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
       .required_capabilities = {}},
      {.key = key("validate_artifact", "trainvm.v1.ValidateArtifact"),
       .effect = Effect::read_only,
       .idempotency = Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {}},
      {.key = key("release_resources", "trainvm.v1.ReleaseResources"),
       .effect = Effect::resource,
       .idempotency = Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {}},
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
  validate_lifecycle(profile.lifecycle);
  if (profile.lifecycle.stateful && profile.effect != Effect::process) {
    throw std::invalid_argument(
        "stateful adapter operations must have process effect");
  }
  if (builtin) {
    if (!profile.code_fingerprint.empty() ||
        !profile.required_capabilities.empty() ||
        profile.lifecycle != OperationLifecycleCapabilities{}) {
      throw std::invalid_argument(
          "builtin adapter profiles cannot carry worker launch or lifecycle authority");
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
    if (profile.effect != node.effect ||
        profile.idempotency != node.idempotency) {
      throw AdapterResolutionError(
          "workflow node " + name +
          " disagrees with its authority-owned effect or idempotency class");
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
