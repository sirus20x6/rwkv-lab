#include "trainvm/adapter_registry.hpp"
#include "trainvm/controller.hpp"
#include "trainvm/control.hpp"
#include "trainvm/document.hpp"
#include "trainvm/fake_worker.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_launch_registry.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/model.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/reconciler.hpp"
#include "trainvm/service.hpp"
#include "trainvm/v1/trainvm.pb.h"

#include <sqlite3.h>
#include <fcntl.h>
#include <linux/memfd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace trainvm {

class JournalTestAccess {
 public:
  static std::uint64_t append(Journal& journal, const Event& event) {
    return journal.append(event);
  }

  static std::vector<std::uint64_t> append_batch(
      Journal& journal, const std::vector<Event>& events) {
    return journal.append_batch(events);
  }
};

}  // namespace trainvm

namespace {

int failures = 0;

constexpr const char* kTestBootId =
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

trainvm::AuthorityTimeSample test_time(std::int64_t nanoseconds,
                                       std::int64_t wall_nanoseconds = -1) {
  return {
      .wall = {.nanoseconds = wall_nanoseconds < 0 ? nanoseconds
                                                   : wall_nanoseconds},
      .boot = {.nanoseconds = nanoseconds},
      .boot_id = kTestBootId,
  };
}

trainvm::AuthorityTimeSample test_time_on_boot(
    std::int64_t nanoseconds, std::string boot_id,
    std::int64_t wall_nanoseconds = -1) {
  return {
      .wall = {.nanoseconds = wall_nanoseconds < 0 ? nanoseconds
                                                   : wall_nanoseconds},
      .boot = {.nanoseconds = nanoseconds},
      .boot_id = std::move(boot_id),
  };
}

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

nlohmann::json load_fixture() {
  const std::filesystem::path path = std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/mageflow-cache-resume.json";
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("could not open fixture " + path.string());
  }
  nlohmann::json value;
  input >> value;
  return value;
}

std::vector<trainvm::AdapterProfile> fixture_adapter_profiles(
    char fingerprint_digit = 'a') {
  const std::string worker_fingerprint =
      "sha256:" + std::string(64, fingerprint_digit);
  const auto key = [](std::string adapter, std::string version,
                      trainvm::ComponentRuntime runtime,
                      std::string operation, std::string contract) {
    return trainvm::AdapterKey{
        .adapter = std::move(adapter),
        .version = std::move(version),
        .runtime = runtime,
        .operation = std::move(operation),
        .contract = std::move(contract),
    };
  };
  return {
      {.key = key("trainvm.core", "1.0.0",
                  trainvm::ComponentRuntime::builtin, "acquire_resources",
                  "trainvm.v1.AcquireResources"),
       .effect = trainvm::Effect::resource,
       .idempotency = trainvm::Idempotency::receipt_required,
       .code_fingerprint = {},
       .required_capabilities = {}},
      {.key = key("trainvm.core", "1.0.0",
                  trainvm::ComponentRuntime::builtin, "validate_artifact",
                  "trainvm.v1.ValidateArtifact"),
       .effect = trainvm::Effect::read_only,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {}},
      {.key = key("trainvm.core", "1.0.0",
                  trainvm::ComponentRuntime::builtin, "release_resources",
                  "trainvm.v1.ReleaseResources"),
       .effect = trainvm::Effect::resource,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {}},
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker, "train",
                  "rwkv_lab.mageflow.v1.Train"),
       .effect = trainvm::Effect::process,
       .idempotency = trainvm::Idempotency::receipt_required,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker,
                  "prepare_cache_span",
                  "rwkv_lab.mageflow.v1.PrepareCacheSpan"),
       .effect = trainvm::Effect::workspace_write,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker, "cache_encoders",
                  "rwkv_lab.mageflow.v1.CacheEncoders"),
       .effect = trainvm::Effect::process,
       .idempotency = trainvm::Idempotency::receipt_required,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"}},
  };
}

std::vector<trainvm::AdapterProfile> fixture_external_adapter_profiles(
    char fingerprint_digit = 'a') {
  auto profiles = fixture_adapter_profiles(fingerprint_digit);
  profiles.erase(
      std::remove_if(profiles.begin(), profiles.end(),
                     [](const trainvm::AdapterProfile& profile) {
                       return profile.key.runtime ==
                              trainvm::ComponentRuntime::builtin;
                     }),
      profiles.end());
  return profiles;
}

trainvm::HostLaunchRegistry fixture_disabled_host_launch_registry() {
  return trainvm::HostLaunchRegistry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {},
      .profiles = {},
  });
}

nlohmann::json adapter_locked_submission(
    const trainvm::CompiledPlan& plan,
    const trainvm::AdapterRegistry& registry) {
  const std::string manifest = registry.plan_lock_manifest(plan);
  return {{"adapter_lock_digest", registry.plan_lock_digest(plan)},
          {"adapter_lock", nlohmann::json::parse(manifest)}};
}

nlohmann::json fixture_adapter_locked_submission(
    const trainvm::CompiledPlan& plan) {
  return adapter_locked_submission(
      plan, trainvm::AdapterRegistry(fixture_adapter_profiles()));
}

trainvm::HostIdentity fixture_test_host_identity() {
  return {
      .host_id = "sha256:" + std::string(64U, '7'),
      .boot_id = "77777777-7777-7777-7777-777777777777",
  };
}

trainvm::HostLaunchRegistry fixture_test_host_launch_registry(
    const trainvm::CompiledPlan& plan,
    const trainvm::WorkerLaunchTicket& launch) {
  const auto& node = plan.experiment.spec.workflow.nodes.at(launch.node_id);
  const auto& component =
      plan.experiment.spec.components.at(node.invoke.component);
  const auto& operation = component.operations.at(node.invoke.operation);
  const trainvm::AdapterKey key{
      .adapter = component.adapter,
      .version = component.version,
      .runtime = component.runtime,
      .operation = node.invoke.operation,
      .contract = operation.contract,
  };
  return trainvm::HostLaunchRegistry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {"/test"},
      .profiles = {{
          .key = key,
          .code_fingerprint = launch.code_fingerprint,
          .executable_path = "/test/trainvm-worker",
          .executable_fingerprint =
              "sha256:" + std::string(64U, 'e'),
          .code_path =
              component.runtime == trainvm::ComponentRuntime::python_worker
                  ? std::optional<std::string>{"/test/worker.pyz"}
                  : std::nullopt,
          .public_arguments = {"worker.pyz"},
          .working_directory = "/test/work",
      }},
  });
}

void prime_test_service_launch(trainvm::TrainVMService& service,
                               const trainvm::WorkerLaunchTicket& launch) {
  const std::string launch_id = launch.run_id + ":worker-launch:" +
                                launch.node_id + ":" + launch.attempt_id;
  const auto binding = service.journal_.launch_binding(launch_id);
  if (!binding || binding->identity.host != service.authority_host_ ||
      binding->identity.host_registry_digest !=
          service.host_launch_registry_.registry_digest()) {
    throw std::runtime_error(
        "test service launch authority disagrees with durable binding");
  }
  auto [retained, inserted] = service.resolved_launches_.emplace(
      launch_id,
      trainvm::ResolvedLaunch(
          *binding, -1,
          binding->identity.code ? std::optional<int>{-1} : std::nullopt,
          -1));
  if (!inserted || retained->second.spec() != *binding) {
    throw std::runtime_error("could not prime exact test launch bundle");
  }
}

// Controller-focused tests do not exercise secure host I/O. They still cross
// the same durable launch-bound gate with a canonical, content-addressed host
// identity before presenting WorkerHello evidence.
trainvm::ResolvedLaunchSpec bind_test_worker_launch(
    trainvm::Controller& controller,
    const trainvm::WorkerLaunchTicket& launch, std::int64_t now_ns) {
  controller.recover();
  const auto& plan = controller.plan();
  const auto& state = controller.state();
  const auto& node =
      plan.experiment.spec.workflow.nodes.at(state.current_node_id);
  const auto& component =
      plan.experiment.spec.components.at(node.invoke.component);
  const auto& operation = component.operations.at(node.invoke.operation);
  const trainvm::AdapterKey key{
      .adapter = component.adapter,
      .version = component.version,
      .runtime = component.runtime,
      .operation = node.invoke.operation,
      .contract = operation.contract,
  };
  const std::string executable_digest =
      "sha256:" + std::string(64U, 'e');
  const trainvm::HostLaunchProfile profile{
      .key = key,
      .code_fingerprint = launch.code_fingerprint,
      .executable_path = "/test/trainvm-worker",
      .executable_fingerprint = executable_digest,
      .code_path = component.runtime == trainvm::ComponentRuntime::python_worker
                       ? std::optional<std::string>{"/test/worker.pyz"}
                       : std::nullopt,
      .public_arguments = {"worker.pyz"},
      .working_directory = "/test/work",
  };
  const trainvm::HostLaunchRegistry registry =
      fixture_test_host_launch_registry(plan, launch);
  const trainvm::HostIdentity host = fixture_test_host_identity();
  const trainvm::VerifiedLaunchArtifact executable{
      .source_path = profile.executable_path,
      .source_device = 1,
      .source_inode = 1,
      .source_size = 1,
      .source_mode = static_cast<std::uint32_t>(S_IFREG | 0500),
      .source_uid = 0,
      .source_gid = 0,
      .sealed_sha256 = executable_digest,
  };
  const auto code = profile.code_path
                        ? std::optional<trainvm::VerifiedLaunchArtifact>{{
                              .source_path = *profile.code_path,
                              .source_device = 2,
                              .source_inode = 2,
                              .source_size = 1,
                              .source_mode = static_cast<std::uint32_t>(
                                  S_IFREG | 0400),
                              .source_uid = 0,
                              .source_gid = 0,
                              .sealed_sha256 = launch.code_fingerprint,
                          }}
                        : std::nullopt;
  trainvm::ResolvedLaunchIdentity identity{
      .api_version = "trainvm.resolved-launch/v1",
      .launch_event_id = launch.run_id + ":worker-launch:" + launch.node_id +
                         ":" + launch.attempt_id,
      .run_id = launch.run_id,
      .node_id = launch.node_id,
      .attempt_id = launch.attempt_id,
      .launch_nonce = launch.launch_nonce,
      .adapter_key = key,
      .code_fingerprint = launch.code_fingerprint,
      .required_capabilities = launch.required_capabilities,
      .host_registry_digest = registry.registry_digest(),
      .host_profile_digest =
          registry.profile_digest(key, launch.code_fingerprint),
      .concurrency_key = launch.concurrency_key,
      .lease_id = launch.lease_id,
      .fencing_token = launch.fencing_token,
      .host = host,
      .executable = executable,
      .code = code,
      .public_arguments = profile.public_arguments,
      .working_directory = {
          .source_path = profile.working_directory,
          .device = 3,
          .inode = 3,
          .mode = static_cast<std::uint32_t>(S_IFDIR | 0700),
          .uid = 0,
          .gid = 0,
      },
  };
  trainvm::ResolvedLaunchSpec spec{
      .identity = std::move(identity),
      .spec_digest = {},
  };
  spec.spec_digest = "sha256:" + trainvm::sha256_hex(
                                    trainvm::resolved_launch_identity_json(
                                        spec.identity)
                                        .dump());
  trainvm::ResolvedLaunch resolved(
      spec, -1, code ? std::optional<int>{-1} : std::nullopt, -1);
  return controller.bind_worker_launch(resolved, registry, host,
                                       test_time(now_ns));
}

bool has_diagnostic(const trainvm::CompileResult& result, std::string_view code) {
  return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const trainvm::Diagnostic& diagnostic) { return diagnostic.code == code; });
}

void test_reflection_and_compiler() {
  const auto metadata_fields = trainvm::reflected_field_names<trainvm::Metadata>();
  check(metadata_fields == std::vector<std::string>({"name", "description", "labels"}),
        "reflection enumerates Metadata fields in declaration order");
  const auto control_fields = trainvm::reflected_field_names<trainvm::Control>();
  check(std::find(control_fields.begin(), control_fields.end(), "default") != control_fields.end(),
        "reflection maps the C++ default_value member to the JSON default field");
  check(trainvm::reflected_field_names<trainvm::CpuIoPolicy>() ==
            std::vector<std::string>({"cpuset", "cpus", "cpu_weight",
                                      "io_weight", "omp_threads",
                                      "preprocessing_workers", "nice"}) &&
            trainvm::reflected_field_names<trainvm::ExecutionPhases>() ==
                std::vector<std::string>({"component", "operation",
                                          "compile", "warmup", "qualify",
                                          "gpu_trace"}) &&
            trainvm::reflected_field_names<trainvm::GpuTraceCapture>() ==
                std::vector<std::string>({
                    "enabled", "backend", "warmup_steps", "skip_steps",
                    "capture_steps", "output_artifact", "activities",
                    "record_shapes", "profile_memory", "with_stack"}),
        "reflection exposes the closed lifecycle, profiler, and CPU/I/O policy schema");

  auto fixture = load_fixture();
  auto result = trainvm::compile_document(fixture);
  check(result.valid(), "MageFlow fixture compiles");
  if (!result.valid()) {
    std::cerr << trainvm::diagnostics_json(result.diagnostics).dump(2) << '\n';
    return;
  }
  check(result.plan->experiment.spec.workflow.nodes.size() == 7U, "compiled plan has seven nodes");
  check(result.plan->experiment.spec.controls.catalog.size() == 4U, "compiled plan has four controls");
  check(result.plan->experiment.spec.execution &&
            result.plan->experiment.spec.execution->gpu_trace &&
            result.plan->experiment.spec.execution->gpu_trace->backend ==
                trainvm::ProfilerBackend::torch &&
            result.plan->experiment.spec.resources.cpu_io_policy &&
            result.plan->experiment.spec.resources.cpu_io_policy->cpuset ==
                std::optional<std::string>{"0-15"},
        "compiler retains model-family-neutral lifecycle and host resource declarations");
  check(result.plan->plan_hash == "d9874d50706cb8b13f3803258bde08f2175bfb4869eae860aa47994d151e901e",
        "MageFlow canonical plan matches its golden SHA-256 identity");
  check(result.plan->canonical_plan["spec"]["controls"]["catalog"]["learning_rate"].contains("default"),
        "canonical plan uses schema field aliases");
  check(!result.plan->canonical_plan["spec"]["controls"]["catalog"]["learning_rate"].contains("default_value"),
        "canonical plan does not leak C++ keyword workarounds");

  auto reordered = nlohmann::json::parse(fixture.dump());
  auto reordered_result = trainvm::compile_document(reordered);
  check(reordered_result.valid() && reordered_result.plan->plan_hash == result.plan->plan_hash,
        "equivalent JSON key order has a stable plan hash");

  auto unknown = fixture;
  unknown["metadata"]["mystery"] = true;
  auto unknown_result = trainvm::compile_document(unknown);
  check(!unknown_result.valid() && has_diagnostic(unknown_result, "field.unknown"),
        "reflected decoder rejects unknown fields");

  auto missing = fixture;
  missing["spec"]["workflow"].erase("entrypoint");
  auto missing_result = trainvm::compile_document(missing);
  check(!missing_result.valid() && has_diagnostic(missing_result, "field.required"),
        "reflected decoder rejects missing required fields");

  auto bad_reference = fixture;
  bad_reference["spec"]["workflow"]["nodes"]["release_gpu"]["transitions"][0]["target"] = "missing";
  auto bad_reference_result = trainvm::compile_document(bad_reference);
  check(!bad_reference_result.valid() && has_diagnostic(bad_reference_result, "transition.target"),
        "semantic compiler rejects unknown transition targets");

  auto unbounded = fixture;
  unbounded["spec"]["workflow"]["nodes"]["resume_training"].erase("loop_guard");
  auto unbounded_result = trainvm::compile_document(unbounded);
  check(!unbounded_result.valid() && has_diagnostic(unbounded_result, "workflow.unbounded_cycle"),
        "semantic compiler rejects an unbounded cycle");

  auto bad_parameter = fixture;
  bad_parameter["spec"]["parameters"]["final_step"]["value"] = "12228";
  auto bad_parameter_result = trainvm::compile_document(bad_parameter);
  check(!bad_parameter_result.valid() && has_diagnostic(bad_parameter_result, "parameter.value_type"),
        "semantic compiler enforces parameter value types");

  auto raw_secret = fixture;
  raw_secret["spec"]["parameters"]["credential"] = {
      {"type", "string"}, {"value", "plaintext-token"},
      {"secret_reference", true}};
  const auto raw_secret_result = trainvm::compile_document(raw_secret);
  check(!raw_secret_result.valid() &&
            has_diagnostic(raw_secret_result,
                           "parameter.secret_reference"),
        "semantic compiler refuses raw values marked as secrets");

  auto opaque_secret = fixture;
  opaque_secret["spec"]["parameters"]["credential"] = {
      {"type", "string"},
      {"value", "secret://local/mageflow-api#v1"},
      {"secret_reference", true}};
  const auto opaque_secret_result = trainvm::compile_document(opaque_secret);
  check(opaque_secret_result.valid(),
        "semantic compiler accepts only versioned opaque secret references");

  auto bad_enum = fixture;
  bad_enum["spec"]["resources"]["accelerators"]["vendor"] = "cuda-ish";
  auto bad_enum_result = trainvm::compile_document(bad_enum);
  check(!bad_enum_result.valid() && has_diagnostic(bad_enum_result, "enum.unknown"),
        "reflected enum decoder rejects unknown values");

  auto arbitrary_launch = fixture;
  arbitrary_launch["spec"]["execution"]["compile"]["command"] =
      "python train.py";
  arbitrary_launch["spec"]["execution"]["compile"]["env"] =
      {{"TOKEN", "plaintext"}};
  const auto arbitrary_launch_result =
      trainvm::compile_document(arbitrary_launch);
  check(!arbitrary_launch_result.valid() &&
            has_diagnostic(arbitrary_launch_result, "field.unknown"),
        "execution declarations reject arbitrary commands and environment maps");

  auto builtin_execution = fixture;
  builtin_execution["spec"]["execution"]["component"] = "core";
  builtin_execution["spec"]["execution"]["operation"] =
      "acquire_resources";
  const auto builtin_execution_result =
      trainvm::compile_document(builtin_execution);
  check(!builtin_execution_result.valid() &&
            has_diagnostic(builtin_execution_result, "execution.target"),
        "lifecycle policy is scoped to an exact external adapter operation");

  auto unknown_profiler = fixture;
  unknown_profiler["spec"]["execution"]["gpu_trace"]["backend"] =
      "shell-profiler";
  const auto unknown_profiler_result =
      trainvm::compile_document(unknown_profiler);
  check(!unknown_profiler_result.valid() &&
            has_diagnostic(unknown_profiler_result, "enum.unknown"),
        "profiler backend is a closed reflected enum");

  auto disabled_phase = fixture;
  disabled_phase["spec"]["execution"]["warmup"]["enabled"] = false;
  const auto disabled_phase_result =
      trainvm::compile_document(disabled_phase);
  check(!disabled_phase_result.valid() &&
            has_diagnostic(disabled_phase_result,
                           "execution.disabled_configuration"),
        "disabled lifecycle phases cannot retain active settings");

  auto oversized_trace = fixture;
  oversized_trace["spec"]["execution"]["gpu_trace"]["capture_steps"] =
      129;
  const auto oversized_trace_result =
      trainvm::compile_document(oversized_trace);
  check(!oversized_trace_result.valid() &&
            has_diagnostic(oversized_trace_result,
                           "execution.trace_steps"),
        "GPU trace capture enforces a strict small step window");

  auto incompatible_profiler = fixture;
  incompatible_profiler["spec"]["execution"]["gpu_trace"]["backend"] =
      "nsys";
  const auto incompatible_profiler_result =
      trainvm::compile_document(incompatible_profiler);
  check(!incompatible_profiler_result.valid() &&
            has_diagnostic(incompatible_profiler_result,
                           "execution.trace_backend_options"),
        "backend-specific profiler flags cannot be silently ignored");

  auto missing_trace_artifact = fixture;
  missing_trace_artifact["spec"]["execution"]["gpu_trace"]
                        ["output_artifact"] = "not_declared";
  const auto missing_trace_artifact_result =
      trainvm::compile_document(missing_trace_artifact);
  check(!missing_trace_artifact_result.valid() &&
            has_diagnostic(missing_trace_artifact_result,
                           "reference.artifact"),
        "GPU trace output must name a declared artifact contract");

  auto wrong_trace_artifact = fixture;
  wrong_trace_artifact["spec"]["execution"]["gpu_trace"]
                      ["output_artifact"] = "eval_gallery";
  const auto wrong_trace_artifact_result =
      trainvm::compile_document(wrong_trace_artifact);
  check(!wrong_trace_artifact_result.valid() &&
            has_diagnostic(wrong_trace_artifact_result,
                           "execution.trace_artifact"),
        "GPU trace output rejects an incompatible artifact type");

  auto cpuset_conflict = fixture;
  cpuset_conflict["spec"]["resources"]["cpu_io_policy"]["cpus"] =
      {0, 1, 2, 3};
  const auto cpuset_conflict_result =
      trainvm::compile_document(cpuset_conflict);
  check(!cpuset_conflict_result.valid() &&
            has_diagnostic(cpuset_conflict_result,
                           "resources.cpuset_conflict"),
        "CPU policy rejects simultaneous textual and structured CPU sets");

  auto structured_cpus = fixture;
  structured_cpus["spec"]["resources"]["cpu_io_policy"].erase("cpuset");
  structured_cpus["spec"]["resources"]["cpu_io_policy"]["cpus"] =
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const auto structured_cpus_result =
      trainvm::compile_document(structured_cpus);
  check(structured_cpus_result.valid(),
        "CPU policy accepts the typed structured CPU-list alternative");

  auto malformed_cpuset = fixture;
  malformed_cpuset["spec"]["resources"]["cpu_io_policy"]["cpuset"] =
      "0-8,8-15";
  const auto malformed_cpuset_result =
      trainvm::compile_document(malformed_cpuset);
  check(!malformed_cpuset_result.valid() &&
            has_diagnostic(malformed_cpuset_result, "resources.cpuset"),
        "textual CPU sets reject overlapping or ambiguous ranges");

  auto duplicate_cpus = fixture;
  duplicate_cpus["spec"]["resources"]["cpu_io_policy"].erase("cpuset");
  duplicate_cpus["spec"]["resources"]["cpu_io_policy"]["cpus"] =
      {0, 1, 1};
  const auto duplicate_cpus_result =
      trainvm::compile_document(duplicate_cpus);
  check(!duplicate_cpus_result.valid() &&
            has_diagnostic(duplicate_cpus_result, "resources.cpus"),
        "structured CPU sets require unique bounded indices");

  auto oversubscribed_cpu = fixture;
  oversubscribed_cpu["spec"]["resources"]["cpu_io_policy"]
                    ["omp_threads"] = 12;
  const auto oversubscribed_cpu_result =
      trainvm::compile_document(oversubscribed_cpu);
  check(!oversubscribed_cpu_result.valid() &&
            has_diagnostic(oversubscribed_cpu_result,
                           "resources.thread_conflict"),
        "CPU policy rejects declared preprocessing and OMP oversubscription");

  auto invalid_cpu_weight = fixture;
  invalid_cpu_weight["spec"]["resources"]["cpu_io_policy"]
                    ["cpu_weight"] = 0;
  const auto invalid_cpu_weight_result =
      trainvm::compile_document(invalid_cpu_weight);
  check(!invalid_cpu_weight_result.valid() &&
            has_diagnostic(invalid_cpu_weight_result, "number.range"),
        "CPU and I/O policy weights use bounded cgroup-v2 ranges");

  auto valid_ncu = fixture;
  auto& ncu_trace = valid_ncu["spec"]["execution"]["gpu_trace"];
  ncu_trace["backend"] = "ncu";
  ncu_trace["activities"] = {"accelerator"};
  ncu_trace.erase("record_shapes");
  ncu_trace.erase("profile_memory");
  ncu_trace.erase("with_stack");
  const auto valid_ncu_result = trainvm::compile_document(valid_ncu);
  check(valid_ncu_result.valid(),
        "ncu is a typed backend with its bounded accelerator-only option surface");

  auto bad_binding = fixture;
  bad_binding["spec"]["workflow"]["nodes"]["prepare_cache"]["invoke"]["inputs"]["final_step"] =
      {{"parameter", "not_declared"}};
  auto bad_binding_result = trainvm::compile_document(bad_binding);
  check(!bad_binding_result.valid() && has_diagnostic(bad_binding_result, "reference.parameter"),
        "semantic compiler resolves typed bindings");

  auto unavailable_artifact = fixture;
  unavailable_artifact["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]["inputs"]["cache"] =
      {{"artifact", "encoder_cache"}};
  auto unavailable_result = trainvm::compile_document(unavailable_artifact);
  check(!unavailable_result.valid() && has_diagnostic(unavailable_result, "artifact.not_available"),
        "semantic compiler rejects artifacts used before publication");

  const std::filesystem::path yaml_path = std::filesystem::temp_directory_path() /
      ("trainvm-fixture-" + std::to_string(static_cast<long long>(getpid())) + ".yaml");
  {
    std::ofstream output(yaml_path);
    output << fixture.dump(2) << '\n';  // JSON is a strict YAML subset.
  }
  auto yaml_result = trainvm::compile_document_file(yaml_path);
  check(yaml_result.valid() && yaml_result.plan->plan_hash == result.plan->plan_hash,
        "YAML input compiles to the same canonical plan identity");
  std::filesystem::remove(yaml_path);
}

void test_wire_contract() {
  trainvm::v1::RunSummary summary;
  summary.mutable_identity()->set_run_id("run-1");
  summary.mutable_identity()->set_revision(4);
  summary.mutable_identity()->set_plan_hash("abc123");
  summary.set_desired_state(trainvm::v1::DESIRED_STATE_PAUSED);
  summary.set_observed_state(trainvm::v1::OBSERVED_STATE_PAUSED);
  summary.set_latest_requested_control_revision(7);
  summary.set_latest_effective_control_revision(6);
  std::string wire;
  check(summary.SerializeToString(&wire), "Protobuf RunSummary serializes");
  trainvm::v1::RunSummary decoded;
  check(decoded.ParseFromString(wire), "Protobuf RunSummary parses");
  check(decoded.identity().run_id() == "run-1" && decoded.identity().revision() == 4U &&
            decoded.latest_requested_control_revision() == 7U &&
            decoded.latest_effective_control_revision() == 6U,
        "generated C++ protocol types preserve the run identity");

  trainvm::v1::RunCommandResponse response;
  response.set_disposition(trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED);
  auto* control = response.mutable_control();
  control->set_command_id("control-1");
  control->set_control_revision(7);
  control->set_apply_point(trainvm::v1::APPLY_POINT_NEXT_OPTIMIZER_STEP);
  control->set_requires_pause(false);
  control->set_status(trainvm::v1::ControlCommandResult::STATUS_REQUESTED);
  auto* assignment = control->add_assignments();
  assignment->set_key("learning_rate");
  assignment->mutable_value()->set_number_value(0.00001);
  wire.clear();
  check(response.SerializeToString(&wire), "Protobuf control command result serializes");
  trainvm::v1::RunCommandResponse decoded_response;
  check(decoded_response.ParseFromString(wire) && decoded_response.has_control() &&
            decoded_response.control().command_id() == "control-1" &&
            decoded_response.control().control_revision() == 7U &&
            decoded_response.control().status() ==
                trainvm::v1::ControlCommandResult::STATUS_REQUESTED &&
            decoded_response.control().assignments_size() == 1,
        "generated C++ protocol types preserve typed control command results");
}

trainvm::Event event_for(const trainvm::ExecutionState& state, std::string id,
                         std::string type, nlohmann::json payload = nlohmann::json::object(),
                         std::optional<std::uint64_t> step = std::nullopt) {
  return trainvm::Event{
      .event_id = std::move(id),
      .run_id = state.run_id,
      .run_revision = state.revision,
      .plan_revision = 1,
      .node_id = state.current_node_id,
      .attempt_id = state.current_attempt_id,
      .worker_sequence = 0,
      .event_type = std::move(type),
      .event_version = 1,
      .wall_time_ns = 0,
      .monotonic_time_ns = 0,
      .optimizer_step = step,
      .payload = std::move(payload),
  };
}

void test_fsm() {
  auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by FSM test compiles");
  if (!compiled.valid()) {
    return;
  }
  auto state = trainvm::start_execution(*compiled.plan, "fsm-run");
  check(state.current_node_id == "acquire_gpu" && state.current_attempt_id == "acquire_gpu@1",
        "execution starts at a deterministic first attempt");
  std::vector<trainvm::Event> events;
  const auto advance = [&](std::string id, std::string type,
                           nlohmann::json payload = nlohmann::json::object(),
                           std::optional<std::uint64_t> step = std::nullopt) {
    events.push_back(event_for(state, std::move(id), std::move(type), std::move(payload), step));
    state = trainvm::advance_execution(*compiled.plan, state, events.back()).state;
  };

  advance("fsm-1", "resource.acquired");
  check(state.current_node_id == "train_to_boundary", "resource acquisition enters training");

  auto wrong_reason = event_for(state, "wrong-reason", "worker.completed", {{"reason", "unknown"}});
  bool missing_transition_rejected = false;
  try {
    (void)trainvm::advance_execution(*compiled.plan, state, wrong_reason);
  } catch (const std::logic_error&) {
    missing_transition_rejected = true;
  }
  check(missing_transition_rejected, "an event with no matching conditional transition is rejected");

  auto ambiguous_plan = *compiled.plan;
  auto& ambiguous_transitions =
      ambiguous_plan.experiment.spec.workflow.nodes.at("train_to_boundary").transitions;
  ambiguous_transitions.push_back(ambiguous_transitions.front());
  auto ambiguous_event = event_for(state, "ambiguous", "worker.completed",
                                   {{"reason", "cache_span_complete"}}, 5500);
  bool ambiguity_rejected = false;
  try {
    (void)trainvm::advance_execution(ambiguous_plan, state, ambiguous_event);
  } catch (const std::logic_error&) {
    ambiguity_rejected = true;
  }
  check(ambiguity_rejected, "multiple matching conditional transitions are rejected");

  auto wrong_attempt = event_for(state, "wrong-attempt", "worker.completed",
                                 {{"reason", "cache_span_complete"}}, 5500);
  wrong_attempt.attempt_id = "train_to_boundary@stale";
  bool stale_attempt_rejected = false;
  try {
    (void)trainvm::advance_execution(*compiled.plan, state, wrong_attempt);
  } catch (const std::invalid_argument&) {
    stale_attempt_rejected = true;
  }
  check(stale_attempt_rejected, "events from stale node attempts are rejected");

  advance("fsm-2", "worker.completed", {{"reason", "cache_span_complete"}}, 5500);
  advance("fsm-3", "operation.completed");
  advance("fsm-4", "operation.completed");
  advance("fsm-5", "artifact.validated");
  check(state.current_node_id == "resume_training" && state.visits.at("resume_training") == 1U,
        "validated cache enters the resume node");

  advance("fsm-6", "worker.restart_requested", nlohmann::json::object(), 6000);
  check(state.current_attempt_id == "resume_training@2" &&
            state.loop_progress.at("resume_training") == 6000.0,
        "clean-process restart increments the attempt and records progress");
  auto stalled = event_for(state, "fsm-stalled", "worker.restart_requested",
                           nlohmann::json::object(), 6000);
  bool stalled_rejected = false;
  try {
    (void)trainvm::advance_execution(*compiled.plan, state, stalled);
  } catch (const std::logic_error&) {
    stalled_rejected = true;
  }
  check(stalled_rejected, "loop re-entry without monotonic progress is rejected");
  advance("fsm-7", "worker.restart_requested", nlohmann::json::object(), 6500);
  advance("fsm-8", "worker.completed", {{"reason", "training_complete"}}, 12228);
  advance("fsm-9", "resource.released");
  check(state.status == trainvm::ExecutionStatus::completed && state.current_node_id.empty(),
        "release transition reaches the completed terminal state");
  check(state.transition_count == 9U && state.revision == 10U,
        "FSM revisions count committed transitions");

  const auto replayed = trainvm::replay_execution(*compiled.plan, "fsm-run", events);
  check(replayed == state, "event replay deterministically reconstructs execution state");
  bool terminal_rejected = false;
  try {
    (void)trainvm::advance_execution(*compiled.plan, state, events.back());
  } catch (const std::logic_error&) {
    terminal_rejected = true;
  }
  check(terminal_rejected, "terminal execution cannot advance");

  trainvm::Event predicate_event = event_for(trainvm::start_execution(*compiled.plan, "predicate-run"),
                                              "predicate", "test", {{"domain", "animation"}, {"quality", 7}});
  const nlohmann::json predicate = {{"all", {{{"field", "domain"}, {"operator", "in"},
                                                {"value", {"animation", "photo"}}},
                                               {{"field", "quality"}, {"operator", "ge"}, {"value", 5}}}}};
  check(trainvm::predicate_matches(predicate, predicate_event),
        "compound predicates resolve payload fields without string evaluation");

  auto limited_plan = *compiled.plan;
  limited_plan.experiment.spec.workflow.nodes.at("resume_training").loop_guard->max_visits = 2;
  auto limited_state = trainvm::start_execution(limited_plan, "limited-run");
  const auto limited_advance = [&](std::string type, nlohmann::json payload = nlohmann::json::object(),
                                   std::optional<std::uint64_t> step = std::nullopt) {
    auto event = event_for(limited_state, "limited-" + std::to_string(limited_state.revision),
                           std::move(type), std::move(payload), step);
    limited_state = trainvm::advance_execution(limited_plan, limited_state, event).state;
  };
  limited_advance("resource.acquired");
  limited_advance("worker.completed", {{"reason", "cache_span_complete"}}, 5500);
  limited_advance("operation.completed");
  limited_advance("operation.completed");
  limited_advance("artifact.validated");
  limited_advance("worker.restart_requested", nlohmann::json::object(), 6000);
  bool visit_limit_rejected = false;
  try {
    limited_advance("worker.restart_requested", nlohmann::json::object(), 6500);
  } catch (const std::logic_error&) {
    visit_limit_rejected = true;
  }
  check(visit_limit_rejected, "loop visit limit is enforced before state advances");
}

void test_control_validation() {
  auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by control validation compiles");
  if (!compiled.valid()) {
    return;
  }
  const auto patch = trainvm::validate_control_patch(
      *compiled.plan, {{"caption_dropout", 0.25}, {"eval_every", 250}}, true, false);
  check(patch.valid() && patch.assignments.size() == 2U &&
            patch.apply_point == trainvm::ApplyPoint::next_optimizer_step &&
            !patch.requires_pause,
        "atomic control patch selects its latest required safe point");

  const auto invalid = trainvm::validate_control_patch(
      *compiled.plan, {{"caption_dropout", 2.0}, {"not_declared", 1}}, true, false);
  check(!invalid.valid() && invalid.assignments.empty() && invalid.diagnostics.size() == 2U,
        "one invalid control rejects the entire patch with complete diagnostics");

  const auto wrong_type =
      trainvm::validate_control_patch(*compiled.plan, {{"eval_every", 2.5}}, true, false);
  check(!wrong_type.valid() && wrong_type.diagnostics.front().code == "control.value_type",
        "integer controls reject fractional JSON numbers");

  const auto running_restart =
      trainvm::validate_control_patch(*compiled.plan, {{"mixed_precision", "fp16"}}, true, false);
  check(!running_restart.valid() && running_restart.requires_pause,
        "pause-required restart controls reject a running mutation");
  const auto paused_restart =
      trainvm::validate_control_patch(*compiled.plan, {{"mixed_precision", "fp16"}}, true, true);
  check(paused_restart.valid() && paused_restart.requires_pause &&
            paused_restart.apply_point == trainvm::ApplyPoint::restart,
        "paused restart control validates with the restart application point");

  auto immutable_plan = *compiled.plan;
  immutable_plan.experiment.spec.controls.catalog.at("learning_rate").mutable_after_start = false;
  const auto immutable =
      trainvm::validate_control_patch(immutable_plan, {{"learning_rate", 0.00001}}, true, false);
  check(!immutable.valid() && immutable.diagnostics.front().code == "control.immutable",
        "started runs reject controls declared immutable after start");

  auto incomparable_plan = *compiled.plan;
  incomparable_plan.experiment.spec.controls.catalog.at("learning_rate").apply =
      trainvm::ApplyPoint::next_eval;
  incomparable_plan.experiment.spec.controls.catalog.at("eval_every").apply =
      trainvm::ApplyPoint::next_checkpoint;
  const auto incomparable = trainvm::validate_control_patch(
      incomparable_plan, {{"learning_rate", 0.00001}, {"eval_every", 100}}, true, false);
  check(!incomparable.valid() && incomparable.assignments.empty() &&
            incomparable.diagnostics.back().code == "control.apply_incompatible",
        "atomic patch rejects incomparable eval and checkpoint application barriers");
}

trainvm::Event created_event(const std::string& plan_hash) {
  return trainvm::Event{
      .event_id = "event-created",
      .run_id = "run-1",
      .run_revision = 1,
      .plan_revision = 1,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "run.created",
      .event_version = 1,
      .wall_time_ns = 100,
      .monotonic_time_ns = 10,
      .optimizer_step = std::nullopt,
      .payload = {{"experiment_name", "mageflow-cache-handoff-resume"},
                  {"plan_hash", plan_hash},
                  {"desired_state", "running"},
                  {"observed_state", "acquiring"}},
  };
}

void test_journal() {
  auto compiled = trainvm::compile_document(load_fixture());
  if (!compiled.valid()) {
    check(false, "fixture required by journal test compiles");
    return;
  }
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::filesystem::path database_path = directory / "journal.db";

  trainvm::Journal journal(database_path);
  const auto created = created_event(compiled.plan->plan_hash);
  check(trainvm::JournalTestAccess::append(journal, created) == 1U,
        "first event receives journal sequence one");
  check(trainvm::JournalTestAccess::append(journal, created) == 1U,
        "identical event append is idempotent");
  check(journal.event_count() == 1U, "idempotent append does not duplicate the event");

  auto conflicting = created;
  conflicting.payload["observed_state"] = "running";
  bool conflict_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append(journal, conflicting);
  } catch (const std::invalid_argument&) {
    conflict_rejected = true;
  }
  check(conflict_rejected, "same event ID with different content is rejected");

  trainvm::Event entered{
      .event_id = "event-entered",
      .run_id = "run-1",
      .run_revision = 2,
      .plan_revision = 1,
      .node_id = "acquire_gpu",
      .attempt_id = "attempt-1",
      .worker_sequence = 1,
      .event_type = "node.entered",
      .event_version = 1,
      .wall_time_ns = 200,
      .monotonic_time_ns = 20,
      .optimizer_step = std::nullopt,
  };
  trainvm::JournalTestAccess::append(journal, entered);
  trainvm::Event heartbeat{
      .event_id = "event-heartbeat",
      .run_id = "run-1",
      .run_revision = 3,
      .plan_revision = 1,
      .node_id = "acquire_gpu",
      .attempt_id = "attempt-1",
      .worker_sequence = 2,
      .event_type = "worker.heartbeat",
      .wall_time_ns = 300,
      .monotonic_time_ns = 30,
      .optimizer_step = 12,
  };
  trainvm::JournalTestAccess::append(journal, heartbeat);
  trainvm::Event desired{
      .event_id = "event-pause",
      .run_id = "run-1",
      .run_revision = 4,
      .plan_revision = 1,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "run.desired_state_changed",
      .event_version = 1,
      .wall_time_ns = 400,
      .monotonic_time_ns = 40,
      .optimizer_step = std::nullopt,
      .payload = {{"state", "paused"}},
  };
  trainvm::JournalTestAccess::append(journal, desired);

  const auto before = journal.projection("run-1");
  check(before.has_value(), "run projection exists");
  if (before) {
    check(before->desired_state == "paused", "desired state projection advances");
    check(before->observed_state == "acquiring", "unmodified observed state is retained");
    check(before->current_node_id == "acquire_gpu" && before->current_attempt_id == "attempt-1",
          "node attempt projection advances");
    check(before->optimizer_step == 12U && before->last_heartbeat_ns == 300,
          "heartbeat projection carries step and time");
    check(before->last_event_sequence == 4U, "projection tracks journal position");
  }
  std::string reason;
  check(journal.verify_chain(&reason) && reason.empty(), "journal hash chain verifies");
  check(journal.rebuild_projections() == 4U, "replay consumes every event");
  check(journal.projection("run-1") == before, "replay deterministically rebuilds the same projection");

  trainvm::Event batch_first{
      .event_id = "batch-first",
      .run_id = "run-1",
      .run_revision = 5,
      .plan_revision = 1,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "run.observed_state_changed",
      .event_version = 1,
      .wall_time_ns = 450,
      .monotonic_time_ns = 45,
      .optimizer_step = std::nullopt,
      .payload = {{"state", "pausing"}},
  };
  auto batch_invalid = batch_first;
  batch_invalid.event_id = "batch-invalid";
  batch_invalid.run_id = "run-does-not-exist";
  bool batch_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append_batch(
        journal, {batch_first, batch_invalid});
  } catch (const std::invalid_argument&) {
    batch_rejected = true;
  }
  check(batch_rejected, "invalid event rejects its entire journal batch");
  check(journal.event_count() == 4U && journal.projection("run-1") == before,
        "failed batch rolls back both events and projection changes");

  auto batch_second = batch_first;
  batch_second.event_id = "batch-second";
  batch_second.run_revision = 6;
  batch_second.event_type = "run.observed_state_changed";
  batch_second.payload = {{"state", "paused"}};
  const auto batch_sequences = trainvm::JournalTestAccess::append_batch(
      journal, {batch_first, batch_second});
  check(batch_sequences == std::vector<std::uint64_t>({5U, 6U}),
        "valid batch receives contiguous journal sequences");
  const auto after_batch = journal.projection("run-1");
  check(after_batch && after_batch->observed_state == "paused" &&
            after_batch->run_revision == 6U && after_batch->last_event_sequence == 6U,
        "valid batch atomically advances the materialized projection");
  check(journal.verify_chain(&reason), "hash chain includes every event in a committed batch");

  auto regressed = heartbeat;
  regressed.event_id = "event-regressed";
  regressed.worker_sequence = 1;
  bool sequence_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append(journal, regressed);
  } catch (const std::invalid_argument&) {
    sequence_rejected = true;
  }
  check(sequence_rejected, "worker sequence regression is rejected");

  auto stale_revision = heartbeat;
  stale_revision.event_id = "event-stale-revision";
  stale_revision.worker_sequence = 3;
  stale_revision.run_revision = 2;
  bool revision_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append(journal, stale_revision);
  } catch (const std::invalid_argument&) {
    revision_rejected = true;
  }
  check(revision_rejected, "run revision regression is rejected");

  sqlite3* tamper_database = nullptr;
  check(sqlite3_open(database_path.c_str(), &tamper_database) == SQLITE_OK,
        "test can open journal for tamper simulation");
  if (tamper_database) {
    check(sqlite3_exec(tamper_database,
                       "UPDATE events SET payload_json='{}' WHERE event_id='event-pause'",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          "tamper simulation updates the underlying row");
    sqlite3_close(tamper_database);
  }
  check(!journal.verify_chain(&reason) && reason.find("sequence 4") != std::string::npos,
        "hash chain identifies a modified event");
  bool replay_refused = false;
  try {
    (void)journal.rebuild_projections();
  } catch (const std::runtime_error&) {
    replay_refused = true;
  }
  check(replay_refused, "projection replay refuses a corrupted journal");
  std::filesystem::remove_all(directory);
}

std::vector<trainvm::FakeOutcome> mageflow_outcomes() {
  return {
      {.expected_node_id = "acquire_gpu",
       .expected_operation = "acquire_resources",
       .event_type = "resource.acquired",
       .optimizer_step = std::nullopt},
      {.expected_node_id = "train_to_boundary",
       .expected_operation = "train",
       .event_type = "worker.completed",
       .payload = {{"reason", "cache_span_complete"}},
       .optimizer_step = 5500},
      {.expected_node_id = "prepare_cache",
       .expected_operation = "prepare_cache_span",
       .event_type = "operation.completed",
       .optimizer_step = std::nullopt},
      {.expected_node_id = "build_cache",
       .expected_operation = "cache_encoders",
       .event_type = "operation.completed",
       .optimizer_step = std::nullopt},
      {.expected_node_id = "validate_cache",
       .expected_operation = "validate_artifact",
       .event_type = "artifact.validated",
       .optimizer_step = std::nullopt},
      {.expected_node_id = "resume_training",
       .expected_operation = "train",
       .event_type = "worker.restart_requested",
       .optimizer_step = 6000},
      {.expected_node_id = "resume_training",
       .expected_operation = "train",
       .event_type = "worker.restart_requested",
       .optimizer_step = 6500},
      {.expected_node_id = "resume_training",
       .expected_operation = "train",
       .event_type = "worker.completed",
       .payload = {{"reason", "training_complete"}},
       .optimizer_step = 12228},
      {.expected_node_id = "release_gpu",
       .expected_operation = "release_resources",
       .event_type = "resource.released",
       .optimizer_step = std::nullopt},
  };
}

void test_controller_and_fake_worker() {
  auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by controller test compiles");
  if (!compiled.valid()) {
    return;
  }
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-controller-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::filesystem::path database_path = directory / "journal.db";

  {
    trainvm::Journal journal(database_path);
    trainvm::FakeWorker worker(mageflow_outcomes());
    trainvm::Controller controller(*compiled.plan, journal, "controller-run");
    check(!controller.initialized(), "controller begins without invented in-memory state");
    controller.create();
    check(controller.state().current_node_id == "acquire_gpu" && journal.event_count() == 2U,
          "run creation atomically records run and initial node entry");

    const auto first_dispatch = controller.prepare_dispatch();
    auto first_event = worker.execute(controller.plan(), controller.state(), first_dispatch);
    auto stale_event = first_event;
    stale_event.run_revision = 0;
    bool stale_worker_rejected = false;
    try {
      controller.handle_event(stale_event);
    } catch (const std::invalid_argument&) {
      stale_worker_rejected = true;
    }
    auto reserved_event = first_event;
    reserved_event.event_type = "node.entered";
    bool reserved_worker_rejected = false;
    try {
      controller.handle_event(reserved_event);
    } catch (const std::invalid_argument&) {
      reserved_worker_rejected = true;
    }
    auto desired_event = first_event;
    desired_event.event_id = "worker-forged-desire";
    desired_event.event_type = "run.desired_state_changed";
    bool desired_worker_rejected = false;
    try {
      controller.handle_event(desired_event);
    } catch (const std::invalid_argument&) {
      desired_worker_rejected = true;
    }
    check(stale_worker_rejected && reserved_worker_rejected && desired_worker_rejected &&
              journal.event_count() == 3U,
          "controller rejects stale revisions and worker use of reserved event types");

    const trainvm::ExecutionState before_receipt_crash = controller.state();
    trainvm::Controller after_receipt_crash(*compiled.plan, journal, "controller-run");
    after_receipt_crash.recover();
    const auto recovered_dispatch = after_receipt_crash.prepare_dispatch();
    const auto repeated_result =
        worker.execute(after_receipt_crash.plan(), after_receipt_crash.state(), recovered_dispatch);
    check(after_receipt_crash.state() == before_receipt_crash &&
              recovered_dispatch == first_dispatch && repeated_result.event_id == first_event.event_id &&
              worker.remaining() == 8U,
          "restart after worker effect reuses dispatch and its idempotent worker receipt");
    after_receipt_crash.handle_event(repeated_result);
    for (std::size_t index = 1; index < 5U; ++index) {
      const auto dispatch = after_receipt_crash.prepare_dispatch();
      const auto event = worker.execute(after_receipt_crash.plan(), after_receipt_crash.state(), dispatch);
      after_receipt_crash.handle_event(event);
    }
    check(after_receipt_crash.state().current_node_id == "resume_training" &&
              after_receipt_crash.state().revision == 6U && worker.remaining() == 4U,
          "scripted worker drives the first durable workflow segment");
    const trainvm::ExecutionState before_restart = after_receipt_crash.state();

    trainvm::Controller restarted(*compiled.plan, journal, "controller-run");
    restarted.recover();
    check(restarted.state() == before_restart,
          "controller restart deterministically recovers the exact FSM state");

    trainvm::Event final_cause;
    while (worker.remaining() > 0U) {
      const auto dispatch = restarted.prepare_dispatch();
      auto event = worker.execute(restarted.plan(), restarted.state(), dispatch);
      final_cause = event;
      restarted.handle_event(event);
    }
    check(restarted.state().status == trainvm::ExecutionStatus::completed &&
              restarted.state().revision == 10U && restarted.state().transition_count == 9U,
          "recovered controller resumes to terminal completion");
    check(journal.event_count() == 47U,
          "dispatch intents, receipts, causes, transitions, and state observations commit once");

    const auto ordered = journal.events_for_run("controller-run");
    check(ordered.size() == 47U && ordered.front().event_type == "run.created" &&
              ordered.back().event_type == "node.dispatch_completed",
          "journal exposes a stable run-local replay order");
    check(journal.event(final_cause.event_id).has_value(), "journal resolves exact events for retries");

    restarted.handle_event(final_cause);
    check(journal.event_count() == 47U && restarted.state().status == trainvm::ExecutionStatus::completed,
          "retrying a committed worker event is idempotent even after completion");

    const auto before_rebuild = journal.projection("controller-run");
    check(before_rebuild && before_rebuild->observed_state == "completed" &&
              before_rebuild->current_node_id.empty() && before_rebuild->current_attempt_id.empty() &&
              before_rebuild->run_revision == 10U && before_rebuild->optimizer_step == 12228U,
          "terminal projection clears stale active-node state and retains training progress");
    std::string reason;
    check(journal.verify_chain(&reason), "controller journal hash chain verifies");
    check(journal.rebuild_projections() == 47U && journal.projection("controller-run") == before_rebuild,
          "controller projection survives complete journal replay");

    auto mismatched_plan = *compiled.plan;
    mismatched_plan.plan_hash = std::string(64, '0');
    trainvm::Controller mismatch(mismatched_plan, journal, "controller-run");
    bool mismatch_rejected = false;
    try {
      mismatch.recover();
    } catch (const std::runtime_error&) {
      mismatch_rejected = true;
    }
    check(mismatch_rejected, "controller refuses recovery under a different compiled plan");
  }

  const std::filesystem::path collision_path = directory / "collision.db";
  {
    trainvm::Journal journal(collision_path);
    trainvm::Controller controller(*compiled.plan, journal, "collision-run");
    controller.create();
    trainvm::FakeWorker worker({mageflow_outcomes().front()});
    const auto dispatch = controller.prepare_dispatch();
    const auto cause = worker.execute(controller.plan(), controller.state(), dispatch);
    auto holder = created_event(compiled.plan->plan_hash);
    holder.event_id = "collision-holder-created";
    holder.run_id = "collision-holder";
    trainvm::JournalTestAccess::append(journal, holder);
    trainvm::Event collision{
        .event_id = cause.event_id + ":transition",
        .run_id = "collision-holder",
        .run_revision = 1,
        .plan_revision = 1,
        .node_id = "",
        .attempt_id = "",
        .worker_sequence = 0,
        .event_type = "diagnostic.collision",
        .event_version = 1,
        .wall_time_ns = 0,
        .monotonic_time_ns = 0,
        .optimizer_step = std::nullopt,
        .payload = nlohmann::json::object(),
    };
    trainvm::JournalTestAccess::append(journal, collision);
    const auto before = controller.state();
    bool collision_rejected = false;
    try {
      controller.handle_event(cause);
    } catch (const std::invalid_argument&) {
      collision_rejected = true;
    }
    check(collision_rejected, "derived event-ID collision rejects the controller transition batch");
    check(controller.state() == before && !journal.event(cause.event_id).has_value() &&
              journal.event_count() == 5U &&
              journal.dispatch(dispatch.dispatch_id)->status == trainvm::DispatchStatus::prepared,
          "failed completion rolls back its cause and leaves dispatch plus memory state resumable");
    trainvm::Controller recovered(*compiled.plan, journal, "collision-run");
    check(recovered.recover() == before,
          "controller recovers the prepared dispatch after an atomic completion rollback");
  }

  trainvm::FakeWorker wrong_worker({{.expected_node_id = "acquire_gpu",
                                      .expected_operation = "not_the_plan_operation",
                                      .event_type = "resource.acquired",
                                      .optimizer_step = std::nullopt}});
  bool wrong_operation_rejected = false;
  try {
    const auto state = trainvm::start_execution(*compiled.plan, "wrong-operation-run");
    const trainvm::Dispatch dispatch{
        .dispatch_id = "wrong-operation-dispatch",
        .run_id = state.run_id,
        .run_revision = state.revision,
        .plan_revision = 1,
        .node_id = state.current_node_id,
        .attempt_id = state.current_attempt_id,
        .component = "core",
        .operation = "acquire_resources",
        .status = trainvm::DispatchStatus::prepared,
        .result_event_id = std::nullopt,
    };
    (void)wrong_worker.execute(*compiled.plan, state, dispatch);
  } catch (const std::logic_error&) {
    wrong_operation_rejected = true;
  }
  check(wrong_operation_rejected && wrong_worker.remaining() == 1U,
        "fake worker validates node operation before consuming scripted work");
  std::filesystem::remove_all(directory);
}

void test_compiled_plan_persistence() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by compiled-plan persistence test compiles");
  if (!compiled.valid() || !compiled.plan) {
    return;
  }
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-plan-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::filesystem::path database_path = directory / "journal.db";

  {
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, "persisted-plan-run");
    controller.create();
    const auto stored = journal.compiled_plan(compiled.plan->plan_hash);
    check(stored && stored->plan_hash == compiled.plan->plan_hash &&
              stored->canonical_plan == compiled.plan->canonical_plan,
          "run creation atomically stores its content-addressed canonical plan");
  }
  {
    trainvm::Journal journal(database_path);
    const auto reloaded = journal.compiled_plan(compiled.plan->plan_hash);
    trainvm::Controller restarted(*compiled.plan, journal, "persisted-plan-run");
    check(reloaded && restarted.recover().current_node_id == "acquire_gpu",
          "journal restart recompiles the persisted plan and recovers its controller");
  }

  sqlite3* tamper_database = nullptr;
  check(sqlite3_open(database_path.c_str(), &tamper_database) == SQLITE_OK,
        "test can open compiled-plan store for tamper simulation");
  if (tamper_database) {
    auto tampered = compiled.plan->canonical_plan;
    tampered["metadata"]["description"] = "tampered canonical plan";
    sqlite3_stmt* update = nullptr;
    const char* sql = "UPDATE compiled_plans SET canonical_plan_json=? WHERE plan_hash=?";
    bool tampered_row = sqlite3_prepare_v2(tamper_database, sql, -1, &update, nullptr) == SQLITE_OK;
    const std::string tampered_text = tampered.dump();
    if (tampered_row) {
      tampered_row = sqlite3_bind_text(update, 1, tampered_text.c_str(),
                                       static_cast<int>(tampered_text.size()), SQLITE_TRANSIENT) ==
                         SQLITE_OK &&
                     sqlite3_bind_text(update, 2, compiled.plan->plan_hash.c_str(),
                                       static_cast<int>(compiled.plan->plan_hash.size()),
                                       SQLITE_TRANSIENT) == SQLITE_OK &&
                     sqlite3_step(update) == SQLITE_DONE;
    }
    check(tampered_row, "tamper simulation changes the stored canonical plan");
    sqlite3_finalize(update);
    sqlite3_close(tamper_database);
  }
  {
    trainvm::Journal journal(database_path);
    bool load_rejected = false;
    try {
      (void)journal.compiled_plan(compiled.plan->plan_hash);
    } catch (const std::runtime_error&) {
      load_rejected = true;
    }
    bool recovery_rejected = false;
    try {
      trainvm::Controller controller(*compiled.plan, journal, "persisted-plan-run");
      (void)controller.recover();
    } catch (const std::runtime_error&) {
      recovery_rejected = true;
    }
    check(load_rejected && recovery_rejected,
          "content-address verification rejects a tampered plan during load and recovery");
  }

  const std::filesystem::path rollback_path = directory / "rollback.db";
  auto unique_source = load_fixture();
  unique_source["metadata"]["name"] = "atomic-plan-rollback";
  const auto unique = trainvm::compile_document(unique_source);
  check(unique.valid(), "atomic rollback fixture compiles");
  if (unique.valid() && unique.plan) {
    trainvm::Journal journal(rollback_path);
    auto collision = created_event(compiled.plan->plan_hash);
    collision.event_id = "atomic-run:created";
    collision.run_id = "collision-holder";
    trainvm::JournalTestAccess::append(journal, collision);
    bool creation_rejected = false;
    try {
      trainvm::Controller controller(*unique.plan, journal, "atomic-run");
      (void)controller.create();
    } catch (const std::invalid_argument&) {
      creation_rejected = true;
    }
    check(creation_rejected && !journal.compiled_plan(unique.plan->plan_hash),
          "failed initial event batch rolls back its newly inserted compiled plan");
  }
  std::filesystem::remove_all(directory);
}

void test_resource_leases() {
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-lease-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::filesystem::path database_path = directory / "journal.db";

  {
    trainvm::Journal first(database_path);
    trainvm::Journal second(database_path);
    const auto acquired = first.acquire_lease("local-gpu", "run-a", "lease-a", test_time(100), 50);
    check(acquired.status == trainvm::LeaseAcquireStatus::acquired &&
              acquired.lease.fencing_token == 1U && acquired.lease.expires_boottime_ns == 150,
          "first resource lease acquisition receives fencing token one");

    const auto repeated = second.acquire_lease("local-gpu", "run-a", "lease-a", test_time(110), 500);
    check(repeated.status == trainvm::LeaseAcquireStatus::already_owned &&
              repeated.lease == acquired.lease,
          "same live lease acquisition is idempotent and does not silently extend expiry");
    const auto busy = second.acquire_lease("local-gpu", "run-b", "lease-b", test_time(120), 50);
    check(busy.status == trainvm::LeaseAcquireStatus::busy &&
              busy.lease.owner_run_id == "run-a" && busy.lease.fencing_token == 1U,
          "exclusive lease reports its current owner instead of overlapping");

    check(!second.renew_lease("local-gpu", "run-b", "lease-b", 1, test_time(125), 50),
          "non-owner cannot renew a resource lease");
    check(first.renew_lease("local-gpu", "run-a", "lease-a", 1, test_time(130), 100),
          "exact owner can renew a live resource lease");
    const auto renewed = second.active_lease("local-gpu", test_time(200));
    check(renewed && renewed->owner_run_id == "run-a" && renewed->expires_boottime_ns == 230,
          "renewed lease is visible across independent journal connections");
  }

  {
    trainvm::Journal restarted(database_path);
    const auto recovered = restarted.active_lease("local-gpu", test_time(220));
    check(recovered && recovered->lease_id == "lease-a" && recovered->fencing_token == 1U,
          "resource lease survives controller and database connection restart");

    const auto successor = restarted.acquire_lease("local-gpu", "run-b", "lease-b", test_time(230), 100);
    check(successor.status == trainvm::LeaseAcquireStatus::acquired &&
              successor.lease.fencing_token == 2U && successor.lease.expires_boottime_ns == 330,
          "expired lease transfers ownership with a larger fencing token");
    check(!restarted.renew_lease("local-gpu", "run-a", "lease-a", 1, test_time(240), 100) &&
              !restarted.release_lease("local-gpu", "run-a", "lease-a", 1, test_time(240)),
          "stale owner cannot renew or release a successor lease");
    check(restarted.release_lease("local-gpu", "run-b", "lease-b", 2, test_time(240)) &&
              !restarted.release_lease("local-gpu", "run-b", "lease-b", 2, test_time(241)) &&
              !restarted.active_lease("local-gpu", test_time(241)),
          "release is owner-fenced, idempotent, and immediately removes active ownership");
    sqlite3* receipt_database = nullptr;
    check(sqlite3_open(database_path.c_str(), &receipt_database) == SQLITE_OK,
          "direct lease release receipt is independently readable");
    if (receipt_database != nullptr) {
      sqlite3_stmt* statement = nullptr;
      std::string receipt;
      constexpr const char* query = R"sql(
        SELECT clock_domain || '|' || boot_id || '|' || released_wall_time_ns
        FROM resource_lease_releases
        WHERE concurrency_key='local-gpu' AND owner_run_id='run-b'
          AND lease_id='lease-b' AND fencing_token=2
      )sql";
      if (sqlite3_prepare_v2(receipt_database, query, -1, &statement, nullptr) ==
              SQLITE_OK &&
          sqlite3_step(statement) == SQLITE_ROW) {
        receipt = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
      }
      sqlite3_finalize(statement);
      sqlite3_close(receipt_database);
      check(receipt ==
                "boottime/v1|aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa|240",
            "direct lease release records an immutable boot-scoped receipt");
    }
    const auto reacquired = restarted.acquire_lease("local-gpu", "run-b", "lease-b", test_time(242), 10);
    check(reacquired.status == trainvm::LeaseAcquireStatus::acquired &&
              reacquired.lease.fencing_token == 3U,
          "reacquiring a released key advances its fencing token");

    bool invalid_timeout_rejected = false;
    try {
      (void)restarted.acquire_lease("another-gpu", "run-c", "lease-c", test_time(1), 0);
    } catch (const std::invalid_argument&) {
      invalid_timeout_rejected = true;
    }
    check(invalid_timeout_rejected, "nonpositive resource lease timeout is rejected");

    const auto before_reboot = restarted.acquire_lease(
        "reboot-gpu", "old-boot-run", "old-boot-lease",
        test_time(1'000, 50'000), 1'000'000);
    const auto other_boot = test_time_on_boot(
        5, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", 60'000);
    check(before_reboot.status == trainvm::LeaseAcquireStatus::acquired &&
              !restarted.active_lease("reboot-gpu", other_boot) &&
              !restarted.renew_lease("reboot-gpu", "old-boot-run",
                                     "old-boot-lease",
                                     before_reboot.lease.fencing_token,
                                     other_boot, 1'000) &&
              !restarted.release_lease("reboot-gpu", "old-boot-run",
                                       "old-boot-lease",
                                       before_reboot.lease.fencing_token,
                                       other_boot),
          "a lease from another boot identity cannot remain active or authorize mutation");
    const auto after_reboot = restarted.acquire_lease(
        "reboot-gpu", "new-boot-run", "new-boot-lease", other_boot, 100);
    check(after_reboot.status == trainvm::LeaseAcquireStatus::acquired &&
              after_reboot.lease.fencing_token ==
                  before_reboot.lease.fencing_token + 1U &&
              after_reboot.lease.boot_id == other_boot.boot_id,
          "a new boot supersedes an old-boot lease with a larger fence");
  }
  std::filesystem::remove_all(directory);
}

void test_authority_clock_integration() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-authority-integration-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  {
    trainvm::Journal journal(directory / "reconciler.db");
    trainvm::AdapterRegistry registry(fixture_adapter_profiles());
    std::mutex authority_mutex;
    std::size_t source_calls = 0U;
    trainvm::Reconciler reconciler(
        journal, registry, authority_mutex,
        [&] {
          ++source_calls;
          return source_calls == 1U ? test_time(100, 10'000)
                                    : test_time(99, 10'001);
        });
    bool first_reached_journal = false;
    try {
      (void)reconciler.step("missing-run");
    } catch (const std::invalid_argument&) {
      first_reached_journal = true;
    }
    bool regression_refused = false;
    bool poison_sticky = false;
    try {
      (void)reconciler.step("missing-run");
    } catch (const trainvm::AuthorityClockError&) {
      regression_refused = true;
    }
    try {
      (void)reconciler.step("missing-run");
    } catch (const trainvm::AuthorityClockError&) {
      poison_sticky = true;
    }
    check(first_reached_journal && regression_refused && poison_sticky &&
              source_calls == 2U,
          "reconciler rejects same-boot BOOTTIME regression and keeps its injected clock poisoned");
  }

  {
    std::size_t source_calls = 0U;
    trainvm::TrainVMService service(
        directory / "service-boot-flip.db",
        trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_disabled_host_launch_registry(),
        [&] {
          ++source_calls;
          return source_calls == 1U
                     ? test_time_on_boot(
                           100, "11111111-1111-1111-1111-111111111111",
                           20'000)
                     : test_time_on_boot(
                           101, "22222222-2222-2222-2222-222222222222",
                           20'001);
        });
    const auto first = service.authority_now();
    bool boot_flip_refused = false;
    bool poison_sticky = false;
    try {
      (void)service.authority_now();
    } catch (const trainvm::AuthorityClockError&) {
      boot_flip_refused = true;
    }
    try {
      (void)service.authority_now();
    } catch (const trainvm::AuthorityClockError&) {
      poison_sticky = true;
    }
    check(first.boot.nanoseconds == 100 && boot_flip_refused && poison_sticky &&
              source_calls == 2U,
          "service rejects an injected boot identity flip and keeps the authority poisoned");
  }

  {
    std::size_t source_calls = 0U;
    trainvm::TrainVMService service(
        directory / "service-malformed.db",
        trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_disabled_host_launch_registry(),
        [&] {
          ++source_calls;
          return trainvm::AuthorityTimeSample{
              .wall = {.nanoseconds = 30'000},
              .boot = {.nanoseconds = 100},
              .boot_id = "malformed-boot-id",
          };
        });
    bool malformed_refused = false;
    bool poison_sticky = false;
    try {
      (void)service.authority_now();
    } catch (const trainvm::AuthorityClockError&) {
      malformed_refused = true;
    }
    try {
      (void)service.authority_now();
    } catch (const trainvm::AuthorityClockError&) {
      poison_sticky = true;
    }
    check(malformed_refused && poison_sticky && source_calls == 1U,
          "service rejects malformed injected authority time and never resamples after poison");
  }

  std::filesystem::remove_all(directory);
}

void test_authority_lock_file_identity() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-authority-file-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  std::filesystem::permissions(
      directory, std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);

  const auto target = directory / "target.db";
  {
    trainvm::Journal journal(target);
  }
  const auto symlink = directory / "symlink.db";
  std::filesystem::create_symlink(target.filename(), symlink);
  bool symlink_refused = false;
  try {
    trainvm::TrainVMService service(
        symlink, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_disabled_host_launch_registry(),
        [] { return test_time(1); });
  } catch (const std::runtime_error&) {
    symlink_refused = true;
  }
  check(symlink_refused,
        "service authority rejects a symlink journal before SQLite initialization");

  const auto hardlink = directory / "hardlink.db";
  std::filesystem::create_hard_link(target, hardlink);
  bool hardlink_refused = false;
  try {
    trainvm::TrainVMService service(
        hardlink, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_disabled_host_launch_registry(),
        [] { return test_time(1); });
  } catch (const std::runtime_error&) {
    hardlink_refused = true;
  }
  check(hardlink_refused,
        "service authority rejects hardlink aliases instead of creating independent locks");
  std::filesystem::remove(hardlink);

  const auto raced = directory / "raced.db";
  const auto displaced = directory / "displaced.db";
  bool retarget_refused = false;
  std::uintmax_t replacement_size = std::numeric_limits<std::uintmax_t>::max();
  {
    trainvm::AuthorityLock authority(raced);
    std::filesystem::rename(raced, displaced);
    const int replacement = ::open(
        raced.c_str(), O_CREAT | O_EXCL | O_CLOEXEC | O_RDWR,
        S_IRUSR | S_IWUSR);
    check(replacement >= 0,
          "authority file race fixture creates a replacement journal inode");
    if (replacement >= 0) {
      (void)::close(replacement);
    }
    try {
      trainvm::Journal journal(authority.journal_path(),
                               authority.journal_identity());
    } catch (const std::runtime_error& exception) {
      retarget_refused =
          std::string(exception.what()).find("authority-locked inode") !=
          std::string::npos;
    }
    replacement_size = std::filesystem::file_size(raced);
  }
  check(retarget_refused && replacement_size == 0U,
        "SQLite rejects a retargeted journal inode before performing schema writes");

  const auto simultaneous = directory / "simultaneous.db";
  bool simultaneous_refused = false;
  {
    trainvm::AuthorityLock first(simultaneous);
    try {
      trainvm::AuthorityLock second(simultaneous);
    } catch (const std::runtime_error&) {
      simultaneous_refused = true;
    }
  }
  check(simultaneous_refused,
        "the authority sidecar serializes simultaneous owners of one journal namespace");

  const auto split_path = directory / "split-namespace.db";
  bool split_authority_refused = false;
  {
    trainvm::AuthorityLock first(split_path);
    trainvm::Journal journal(first.journal_path(), first.journal_identity());
    const auto displaced_main = directory / "split-namespace.main-old";
    const auto lock =
        std::filesystem::path(split_path.string() + ".authority.lock");
    const auto displaced_lock = directory / "split-namespace.lock-old";
    std::filesystem::rename(split_path, displaced_main);
    std::filesystem::rename(lock, displaced_lock);
    std::vector<std::pair<std::filesystem::path, std::filesystem::path>>
        displaced_auxiliaries;
    for (const std::string_view suffix :
         {std::string_view{"-journal"}, std::string_view{"-wal"},
          std::string_view{"-shm"}}) {
      const auto source =
          std::filesystem::path(split_path.string() + std::string(suffix));
      if (std::filesystem::exists(source)) {
        const auto displaced_auxiliary = directory /
            ("split-namespace" + std::string(suffix) + "-old");
        std::filesystem::rename(source, displaced_auxiliary);
        displaced_auxiliaries.emplace_back(source, displaced_auxiliary);
      }
    }
    try {
      trainvm::AuthorityLock second(split_path);
    } catch (const std::runtime_error&) {
      split_authority_refused = true;
    }
    for (auto iterator = displaced_auxiliaries.rbegin();
         iterator != displaced_auxiliaries.rend(); ++iterator) {
      std::filesystem::rename(iterator->second, iterator->first);
    }
    std::filesystem::rename(displaced_lock, lock);
    std::filesystem::rename(displaced_main, split_path);
  }
  check(split_authority_refused,
        "kernel namespace locking prevents split authority after the database, sidecar, and SQLite auxiliaries are renamed together");

  bool auxiliary_aliases_refused = true;
  for (const std::string_view suffix :
       {std::string_view{"-journal"}, std::string_view{"-wal"},
        std::string_view{"-shm"}}) {
    for (const bool symbolic : {false, true}) {
      const auto case_directory =
          directory / ("aux-" + std::string(suffix.substr(1)) +
                       (symbolic ? "-symlink" : "-hardlink"));
      std::filesystem::create_directories(case_directory);
      std::filesystem::permissions(
          case_directory, std::filesystem::perms::owner_all,
          std::filesystem::perm_options::replace);
      const auto victim = case_directory / "victim";
      {
        std::ofstream output(victim, std::ios::binary);
        output << "preserve-this-file";
      }
      const auto journal_path = case_directory / "journal.db";
      const auto auxiliary =
          std::filesystem::path(journal_path.string() + std::string(suffix));
      if (symbolic) {
        std::filesystem::create_symlink(victim.filename(), auxiliary);
      } else {
        std::filesystem::create_hard_link(victim, auxiliary);
      }
      bool refused = false;
      try {
        trainvm::AuthorityLock authority(journal_path);
      } catch (const std::runtime_error&) {
        refused = true;
      }
      std::ifstream input(victim, std::ios::binary);
      const std::string contents((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());
      auxiliary_aliases_refused = auxiliary_aliases_refused && refused &&
                                  contents == "preserve-this-file";
    }
  }
  check(auxiliary_aliases_refused,
        "authority acquisition rejects SQLite journal, WAL, and SHM aliases without touching their victims");

  const auto real_parent = directory / "real-parent";
  const auto linked_parent = directory / "linked-parent";
  std::filesystem::create_directories(real_parent);
  std::filesystem::permissions(real_parent,
                               std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace);
  std::filesystem::create_directory_symlink(real_parent.filename(),
                                            linked_parent);
  bool parent_symlink_refused = false;
  try {
    trainvm::AuthorityLock authority(linked_parent / "journal.db");
  } catch (const std::runtime_error&) {
    parent_symlink_refused = true;
  }
  check(parent_symlink_refused,
        "component-wise authority resolution refuses an intermediate directory symlink");

  const auto lifetime_path = directory / "lifetime-main.db";
  const auto lifetime_displaced = directory / "lifetime-main.displaced";
  bool lifetime_move_refused = false;
  bool lifetime_poison_sticky = false;
  {
    trainvm::AuthorityLock authority(lifetime_path);
    trainvm::Journal journal(authority.journal_path(),
                             authority.journal_identity());
    std::filesystem::rename(lifetime_path, lifetime_displaced);
    const int replacement = ::open(
        lifetime_path.c_str(), O_CREAT | O_EXCL | O_CLOEXEC | O_RDWR,
        S_IRUSR | S_IWUSR);
    check(replacement >= 0,
          "lifetime identity test creates a replacement database inode");
    if (replacement >= 0) (void)::close(replacement);
    try {
      (void)journal.event_count();
    } catch (const std::runtime_error&) {
      lifetime_move_refused = true;
    }
    std::filesystem::remove(lifetime_path);
    std::filesystem::rename(lifetime_displaced, lifetime_path);
    try {
      (void)journal.event_count();
    } catch (const std::runtime_error&) {
      lifetime_poison_sticky = true;
    }
  }
  check(lifetime_move_refused && lifetime_poison_sticky,
        "a post-construction main-file move poisons every later journal operation even after restoration");

  const auto sidecar_lifetime_path = directory / "lifetime-sidecar.db";
  bool sidecar_move_refused = false;
  {
    trainvm::AuthorityLock authority(sidecar_lifetime_path);
    trainvm::Journal journal(authority.journal_path(),
                             authority.journal_identity());
    const auto sidecar = std::filesystem::path(
        sidecar_lifetime_path.string() + ".authority.lock");
    const auto displaced_sidecar = directory / "lifetime-sidecar.displaced";
    std::filesystem::rename(sidecar, displaced_sidecar);
    const int replacement = ::open(
        sidecar.c_str(), O_CREAT | O_EXCL | O_CLOEXEC | O_RDWR,
        S_IRUSR | S_IWUSR);
    if (replacement >= 0) (void)::close(replacement);
    try {
      (void)journal.event_count();
    } catch (const std::runtime_error&) {
      sidecar_move_refused = true;
    }
    if (replacement >= 0) std::filesystem::remove(sidecar);
    std::filesystem::rename(displaced_sidecar, sidecar);
  }
  check(sidecar_move_refused,
        "a post-construction authority-sidecar replacement poisons the old journal before another operation");

  const auto auxiliary_lifetime_path = directory / "lifetime-aux.db";
  bool auxiliary_lifetime_refused = false;
  {
    trainvm::AuthorityLock authority(auxiliary_lifetime_path);
    trainvm::Journal journal(authority.journal_path(),
                             authority.journal_identity());
    const auto wal =
        std::filesystem::path(auxiliary_lifetime_path.string() + "-wal");
    const auto wal_victim = directory / "lifetime-wal-victim";
    const auto wal_alias = directory / "lifetime-wal-alias";
    if (std::filesystem::exists(wal)) {
      std::filesystem::create_hard_link(wal, wal_alias);
    } else {
      {
        std::ofstream output(wal_victim, std::ios::binary);
        output << "preserve-lifetime-victim";
      }
      std::filesystem::create_hard_link(wal_victim, wal);
    }
    try {
      (void)journal.event_count();
    } catch (const std::runtime_error&) {
      auxiliary_lifetime_refused = true;
    }
    if (std::filesystem::exists(wal_alias)) {
      std::filesystem::remove(wal_alias);
    } else {
      std::filesystem::remove(wal);
      std::filesystem::remove(wal_victim);
    }
  }
  check(auxiliary_lifetime_refused,
        "lifetime boundaries poison a journal when a live SQLite auxiliary gains a hardlink alias");

  const auto namespace_path = directory / "lifetime-namespace";
  const auto namespace_displaced = directory / "lifetime-namespace.displaced";
  std::filesystem::create_directories(namespace_path);
  std::filesystem::permissions(namespace_path,
                               std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace);
  bool namespace_move_refused = false;
  {
    const auto path = namespace_path / "journal.db";
    trainvm::AuthorityLock authority(path);
    trainvm::Journal journal(authority.journal_path(),
                             authority.journal_identity());
    std::filesystem::rename(namespace_path, namespace_displaced);
    std::filesystem::create_directories(namespace_path);
    std::filesystem::permissions(namespace_path,
                                 std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace);
    try {
      (void)journal.event_count();
    } catch (const std::runtime_error&) {
      namespace_move_refused = true;
    }
    std::filesystem::remove(namespace_path);
    std::filesystem::rename(namespace_displaced, namespace_path);
  }
  check(namespace_move_refused,
        "lifetime validation re-walks and rejects replacement of the configured parent directory");

  const auto cli_directory = directory / "cli-adversarial";
  std::filesystem::create_directories(cli_directory);
  std::filesystem::permissions(cli_directory,
                               std::filesystem::perms::owner_all,
                               std::filesystem::perm_options::replace);
  const auto cli_victim = cli_directory / "victim";
  {
    std::ofstream output(cli_victim, std::ios::binary);
    output << "cli-victim";
  }
  const auto cli_journal = cli_directory / "journal.db";
  std::filesystem::create_hard_link(
      cli_victim, std::filesystem::path(cli_journal.string() + "-wal"));
  std::array<char, 4096> executable_buffer{};
  const ssize_t executable_size = ::readlink(
      "/proc/self/exe", executable_buffer.data(), executable_buffer.size() - 1U);
  bool cli_refused = false;
  if (executable_size > 0) {
    executable_buffer[static_cast<std::size_t>(executable_size)] = '\0';
    const auto cli_binary =
        std::filesystem::path(executable_buffer.data()).parent_path() / "trainvm";
    const pid_t child = ::fork();
    if (child == 0) {
      ::execl(cli_binary.c_str(), cli_binary.c_str(), "journal", "init",
              cli_journal.c_str(), static_cast<char*>(nullptr));
      ::_exit(127);
    }
    if (child > 0) {
      int status = 0;
      if (::waitpid(child, &status, 0) == child) {
        cli_refused = WIFEXITED(status) && WEXITSTATUS(status) != 0;
      }
    }
  }
  std::ifstream cli_input(cli_victim, std::ios::binary);
  const std::string cli_contents((std::istreambuf_iterator<char>(cli_input)),
                                 std::istreambuf_iterator<char>());
  check(cli_refused && cli_contents == "cli-victim",
        "journal CLI uses AuthorityLock auxiliary validation and cannot overwrite an aliased WAL victim");

  std::filesystem::remove_all(directory);
}

void test_control_command_journal() {
  auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by control command journal compiles");
  if (!compiled.valid()) {
    return;
  }
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-control-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  {
    trainvm::Journal journal(directory / "journal.db");
    trainvm::Controller controller(*compiled.plan, journal, "control-run");
    controller.create();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const auto lease = journal.acquire_lease("local-gpu-training", "control-run", "lease-1",
                                             test_time(now_ns - 1000), 3'600'000'000'000LL);
    const auto acknowledgement = [&](std::uint64_t worker_sequence) {
      return trainvm::ControlAcknowledgementIdentity{
          .concurrency_key = "local-gpu-training",
          .lease_id = "lease-1",
          .fencing_token = lease.lease.fencing_token,
          .node_id = controller.state().current_node_id,
          .attempt_id = controller.state().current_attempt_id,
          .worker_sequence = worker_sequence,
      };
    };
    const auto request = controller.request_controls(
        "browser-request-1", 1, 0,
        {{"learning_rate", 0.00001}, {"eval_every", 250}}, "operator",
        "test live tuning");
    check(request.valid() && request.command, "control command test patch validates");
    if (!request.command) {
      return;
    }
    const auto submitted = *request.command;
    check(submitted.control_revision == 1U &&
              submitted.status == trainvm::ControlCommandStatus::requested &&
              journal.latest_control_revision("control-run") == 1U && journal.event_count() == 3U,
          "control request atomically receives revision and journal event");
    const auto retry = controller.request_controls(
        "browser-request-1", 1, 0,
        {{"learning_rate", 0.00001}, {"eval_every", 250}}, "operator",
        "test live tuning");
    check(retry.command == submitted && journal.event_count() == 3U,
          "identical control request is idempotent");

    bool identity_conflict = false;
    try {
      (void)controller.request_controls(
          "browser-request-1", 1, 0,
          {{"learning_rate", 0.00001}, {"eval_every", 500}}, "operator",
          "test live tuning");
    } catch (const std::invalid_argument&) {
      identity_conflict = true;
    }
    check(identity_conflict, "idempotency key rejects different control command content");

    bool stale_rejected = false;
    try {
      (void)controller.request_controls(
          "browser-request-stale", 1, 0, {{"learning_rate", 0.00002}}, "operator",
          "stale browser test");
    } catch (const std::invalid_argument&) {
      stale_rejected = true;
    }
    check(stale_rejected, "stale expected control revision loses an optimistic race");

    const auto applied = controller.acknowledge_controls(
        submitted.command_id, acknowledgement(1),
        trainvm::ControlCommandStatus::applied, 300,
        submitted.assignments, nlohmann::json::array(), test_time(now_ns));
    check(applied.status == trainvm::ControlCommandStatus::applied &&
              applied.effective_step == std::optional<std::uint64_t>{300} &&
              journal.event_count() == 4U && journal.control_command(submitted.command_id) == applied,
          "worker acknowledgement atomically records exact effective values and step");
    check(controller.acknowledge_controls(
              submitted.command_id, acknowledgement(1),
              trainvm::ControlCommandStatus::applied, 300,
              submitted.assignments, nlohmann::json::array(),
              test_time(now_ns)) == applied &&
              journal.event_count() == 4U,
          "identical worker control acknowledgement is idempotent");
    bool changed_ack_rejected = false;
    try {
      (void)controller.acknowledge_controls(
          submitted.command_id, acknowledgement(1),
          trainvm::ControlCommandStatus::applied, 301,
          submitted.assignments, nlohmann::json::array(), test_time(now_ns));
    } catch (const std::invalid_argument&) {
      changed_ack_rejected = true;
    }
    check(changed_ack_rejected, "completed control command rejects a different acknowledgement");

    const auto invalid_controller_request = controller.request_controls(
        "browser-request-invalid", 1, 1, {{"caption_dropout", 4.0}},
        "operator", "invalid range test");
    check(!invalid_controller_request.valid() && !invalid_controller_request.command &&
              journal.event_count() == 4U,
          "controller returns native control diagnostics without journaling an invalid patch");
    const auto controller_request = controller.request_controls(
        "browser-request-2", 1, 1, {{"caption_dropout", 0.2}},
        "operator", "adjust dropout");
    check(controller_request.valid() && controller_request.command &&
              controller_request.command->control_revision == 2U && journal.event_count() == 5U,
          "controller validates and durably submits a revision-checked control patch");
    const auto later_request = controller.request_controls(
        "browser-request-3", 1, 2, {{"caption_dropout", 0.3}},
        "operator", "later dropout adjustment");
    check(later_request.command && later_request.command->control_revision == 3U &&
              journal.event_count() == 6U,
          "successive control requests receive monotonic revisions");
    bool out_of_order_rejected = false;
    try {
      (void)controller.acknowledge_controls(
          later_request.command->command_id, acknowledgement(3),
          trainvm::ControlCommandStatus::applied, 400,
          later_request.command->assignments, nlohmann::json::array(),
          test_time(now_ns));
    } catch (const std::invalid_argument&) {
      out_of_order_rejected = true;
    }
    check(out_of_order_rejected && journal.event_count() == 6U,
          "worker cannot apply control revisions out of order");
    const auto second_applied = controller.acknowledge_controls(
        controller_request.command->command_id, acknowledgement(2),
        trainvm::ControlCommandStatus::applied, 350,
        controller_request.command->assignments, nlohmann::json::array(),
        test_time(now_ns));
    const auto third_rejected = controller.acknowledge_controls(
        later_request.command->command_id, acknowledgement(3),
        trainvm::ControlCommandStatus::rejected,
        std::nullopt, nlohmann::json::object(),
        nlohmann::json::array({{{"code", "worker.rejected"}}}),
        test_time(now_ns));
    check(second_applied.status == trainvm::ControlCommandStatus::applied &&
              third_rejected.status == trainvm::ControlCommandStatus::rejected &&
              journal.event_count() == 8U,
          "ordered acknowledgements advance the durable command stream");
    const auto fenced_request = controller.request_controls(
        "browser-request-4", 1, 3, {{"caption_dropout", 0.4}},
        "operator", "fencing test");
    check(journal.release_lease("local-gpu-training", "control-run", "lease-1",
                                lease.lease.fencing_token, test_time(now_ns)),
          "fencing test releases the first worker lease");
    const auto successor = journal.acquire_lease("local-gpu-training", "control-run", "lease-2",
                                                 test_time(now_ns), 3'600'000'000'000LL);
    bool stale_lease_rejected = false;
    try {
      (void)controller.acknowledge_controls(
          fenced_request.command->command_id, acknowledgement(4),
          trainvm::ControlCommandStatus::applied, 450,
          fenced_request.command->assignments, nlohmann::json::array(),
          test_time(now_ns));
    } catch (const std::invalid_argument&) {
      stale_lease_rejected = true;
    }
    const trainvm::ControlAcknowledgementIdentity successor_identity{
        .concurrency_key = "local-gpu-training",
        .lease_id = "lease-2",
        .fencing_token = successor.lease.fencing_token,
        .node_id = controller.state().current_node_id,
        .attempt_id = controller.state().current_attempt_id,
        .worker_sequence = 4,
    };
    const auto fenced_applied = controller.acknowledge_controls(
        fenced_request.command->command_id, successor_identity,
        trainvm::ControlCommandStatus::applied, 450,
        fenced_request.command->assignments, nlohmann::json::array(),
        test_time(now_ns));
    check(stale_lease_rejected && successor.lease.fencing_token == 2U &&
              fenced_applied.acknowledgement == successor_identity && journal.event_count() == 10U,
          "stale worker fencing tokens cannot acknowledge controls after lease takeover");

    trainvm::Controller restarted(*compiled.plan, journal, "control-run");
    check(restarted.recover() == controller.state(),
          "controller recovery tolerates and verifies interleaved control command events");
    trainvm::Controller other(*compiled.plan, journal, "other-control-run");
    other.create();
    bool cross_run_ack_rejected = false;
    try {
      (void)other.acknowledge_controls(
          submitted.command_id, acknowledgement(1),
          trainvm::ControlCommandStatus::applied, 300,
          submitted.assignments, nlohmann::json::array(), test_time(now_ns));
    } catch (const std::invalid_argument&) {
      cross_run_ack_rejected = true;
    }
    check(cross_run_ack_rejected,
          "a controller cannot acknowledge another run's control command");
    std::string reason;
    check(journal.verify_chain(&reason) && journal.rebuild_projections() == 12U &&
              journal.control_command(fenced_applied.command_id) == fenced_applied &&
              journal.latest_control_revision("control-run") == 4U &&
              journal.latest_effective_control_revision("control-run") == 4U,
          "control request, acknowledgement, and effective revisions rebuild from journal history");
  }
  std::filesystem::remove_all(directory);
}

void test_command_service() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by command service compiles");
  if (!compiled.valid()) {
    return;
  }
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-service-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  std::string journal_id;
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, "service-run");
    controller.create();
    (void)controller.prepare_dispatch();
    journal_id = journal.journal_id();
  }

  trainvm::TrainVMService service(
      database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
      fixture_disabled_host_launch_registry());
  auto request = [&] (std::string idempotency_key, std::uint64_t expected_control_revision,
                     double value) {
    trainvm::v1::RunCommandRequest output;
    output.set_run_id("service-run");
    output.set_expected_run_revision(1);
    output.set_idempotency_key(std::move(idempotency_key));
    output.set_author("dashboard");
    output.set_reason("service test");
    output.set_expected_journal_id(journal_id);
    output.set_expected_plan_hash(compiled.plan->plan_hash);
    auto* controls = output.mutable_controls();
    controls->set_expected_control_revision(expected_control_revision);
    auto* assignment = controls->add_assignments();
    assignment->set_key("learning_rate");
    assignment->mutable_value()->set_number_value(value);
    return output;
  };

  auto first_request = request("intent-1", 0, 0.00001);
  trainvm::v1::RunCommandResponse first_response;
  const auto first_status = service.CommandRun(nullptr, &first_request, &first_response);
  check(first_status.ok() &&
            first_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED &&
            first_response.control().control_revision() == 1U &&
            first_response.run().latest_requested_control_revision() == 1U,
        "native command service validates, persists, and returns a typed control result");

  trainvm::v1::RunCommandResponse retry_response;
  const auto retry_status = service.CommandRun(nullptr, &first_request, &retry_response);
  check(retry_status.ok() &&
            retry_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED &&
            retry_response.control().status() ==
                trainvm::v1::ControlCommandResult::STATUS_REQUESTED &&
            retry_response.control().command_id() == first_response.control().command_id(),
        "native command service reports a pending idempotent retry as accepted, not applied");

  {
    trainvm::Journal worker_journal(database_path);
    trainvm::Controller worker(*compiled.plan, worker_journal, "service-run");
    worker.recover();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    const auto lease = worker_journal.acquire_lease(
        "local-gpu-training", "service-run", "service-worker-lease", test_time(now_ns - 1000),
        3'600'000'000'000LL);
    (void)worker.acknowledge_controls(
        first_response.control().command_id(),
        trainvm::ControlAcknowledgementIdentity{
            .concurrency_key = "local-gpu-training",
            .lease_id = "service-worker-lease",
            .fencing_token = lease.lease.fencing_token,
            .node_id = worker.state().current_node_id,
            .attempt_id = worker.state().current_attempt_id,
            .worker_sequence = 1,
        },
        trainvm::ControlCommandStatus::rejected, std::nullopt, nlohmann::json::object(),
        nlohmann::json::array(
            {{{"severity", "error"}, {"code", "worker.control_rejected"},
              {"message", "worker rejected the requested value"}}}),
        test_time(now_ns));
  }
  trainvm::v1::RunCommandResponse terminal_retry_response;
  const auto terminal_retry_status =
      service.CommandRun(nullptr, &first_request, &terminal_retry_response);
  check(terminal_retry_status.ok() &&
            terminal_retry_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_REJECTED &&
            terminal_retry_response.control().status() ==
                trainvm::v1::ControlCommandResult::STATUS_REJECTED &&
            terminal_retry_response.diagnostics_size() == 1 &&
            terminal_retry_response.diagnostics(0).code() == "worker.control_rejected",
        "terminal command retries return their durable status and worker diagnostics");

  {
    trainvm::Journal runtime_journal(database_path);
    const auto projection = runtime_journal.projection("service-run");
    check(projection.has_value(), "pause test has a durable run projection");
    const auto lifecycle_event = [&](std::string id, std::uint64_t revision,
                                     std::string type, std::string state) {
      return trainvm::Event{
          .event_id = std::move(id),
          .run_id = "service-run",
          .run_revision = revision,
          .plan_revision = 1,
          .node_id = projection ? projection->current_node_id : std::string{},
          .attempt_id = projection ? projection->current_attempt_id : std::string{},
          .worker_sequence = 0,
          .event_type = std::move(type),
          .event_version = 1,
          .wall_time_ns = static_cast<std::int64_t>(revision),
          .monotonic_time_ns = revision,
          .optimizer_step = std::nullopt,
          .payload = {{"state", std::move(state)}},
      };
    };
    trainvm::JournalTestAccess::append_batch(runtime_journal, {
        lifecycle_event("service-run:pause-desired", 2, "run.desired_state_changed", "paused"),
        lifecycle_event("service-run:pausing", 3, "run.observed_state_changed", "pausing"),
        lifecycle_event("service-run:paused", 4, "run.observed_state_changed", "paused"),
    });
  }
  auto paused_request = request("paused-intent", 1, 0.0);
  paused_request.set_expected_run_revision(4);
  paused_request.mutable_controls()->clear_assignments();
  auto* paused_assignment = paused_request.mutable_controls()->add_assignments();
  paused_assignment->set_key("mixed_precision");
  paused_assignment->mutable_value()->set_string_value("fp16");
  trainvm::v1::RunCommandResponse paused_response;
  const auto paused_status = service.CommandRun(nullptr, &paused_request, &paused_response);
  check(paused_status.ok() &&
            paused_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED &&
            paused_response.control().requires_pause() &&
            paused_response.control().apply_point() == trainvm::v1::APPLY_POINT_RESTART,
        "paused runs accept controls that declare a durable pause requirement");

  {
    trainvm::Journal runtime_journal(database_path);
    const auto projection = runtime_journal.projection("service-run");
    const auto lifecycle_event = [&](std::string id, std::uint64_t revision,
                                     std::string type, std::string state) {
      return trainvm::Event{
          .event_id = std::move(id),
          .run_id = "service-run",
          .run_revision = revision,
          .plan_revision = 1,
          .node_id = projection ? projection->current_node_id : std::string{},
          .attempt_id = projection ? projection->current_attempt_id : std::string{},
          .worker_sequence = 0,
          .event_type = std::move(type),
          .event_version = 1,
          .wall_time_ns = static_cast<std::int64_t>(revision),
          .monotonic_time_ns = revision,
          .optimizer_step = std::nullopt,
          .payload = {{"state", std::move(state)}},
      };
    };
    trainvm::JournalTestAccess::append_batch(runtime_journal, {
        lifecycle_event("service-run:resume-desired", 5, "run.desired_state_changed", "running"),
        lifecycle_event("service-run:resuming", 6, "run.observed_state_changed", "resuming"),
        lifecycle_event("service-run:resumed", 7, "run.observed_state_changed", "running"),
    });
  }
  trainvm::v1::RunCommandResponse resumed_retry_response;
  const auto resumed_retry_status =
      service.CommandRun(nullptr, &paused_request, &resumed_retry_response);
  check(resumed_retry_status.ok() &&
            resumed_retry_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED &&
            resumed_retry_response.control().command_id() == paused_response.control().command_id(),
        "an exact paused command retry keeps its identity after the run resumes");
  {
    trainvm::Journal runtime_journal(database_path);
    trainvm::Controller resumed(*compiled.plan, runtime_journal, "service-run");
    resumed.recover();
    const auto redispatch = resumed.prepare_dispatch();
    check(redispatch.run_revision == 7U &&
              redispatch.dispatch_id == "service-run:dispatch:acquire_gpu:acquire_gpu@1" &&
              redispatch.status == trainvm::DispatchStatus::prepared,
          "resume reissues the prepared node attempt at the current run revision");
  }

  auto changed_request = request("intent-1", 0, 0.00002);
  trainvm::v1::RunCommandResponse changed_response;
  const auto changed_status = service.CommandRun(nullptr, &changed_request, &changed_response);
  check(changed_status.ok() &&
            changed_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_CONFLICT,
        "native command service reports changed idempotent content as a conflict");

  auto invalid_request = request("invalid-intent", 1, 2.0);
  trainvm::v1::RunCommandResponse invalid_response;
  const auto invalid_status = service.CommandRun(nullptr, &invalid_request, &invalid_response);
  check(invalid_status.ok() &&
            invalid_response.disposition() ==
                trainvm::v1::RunCommandResponse::DISPOSITION_REJECTED &&
            invalid_response.diagnostics_size() > 0,
        "native command service returns semantic diagnostics without mutating the journal");

  auto race = [&](std::string key, double value) {
    auto race_request = request(std::move(key), 2, value);
    race_request.set_expected_run_revision(7);
    trainvm::v1::RunCommandResponse response;
    const auto status = service.CommandRun(nullptr, &race_request, &response);
    return std::pair{status.ok(), response.disposition()};
  };
  auto left = std::async(std::launch::async, race, "race-left", 0.00003);
  auto right = std::async(std::launch::async, race, "race-right", 0.00004);
  const auto left_result = left.get();
  const auto right_result = right.get();
  const int accepted =
      (left_result.second == trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED ? 1 : 0) +
      (right_result.second == trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED ? 1 : 0);
  const int conflicts =
      (left_result.second == trainvm::v1::RunCommandResponse::DISPOSITION_CONFLICT ? 1 : 0) +
      (right_result.second == trainvm::v1::RunCommandResponse::DISPOSITION_CONFLICT ? 1 : 0);
  check(left_result.first && right_result.first && accepted == 1 && conflicts == 1,
        "concurrent dashboard edits at one control revision have exactly one winner");
  {
    trainvm::Journal verify(database_path);
    check(verify.event_count() == 13U && verify.latest_control_revision("service-run") == 3U,
          "service retries, rejections, and losing races leave no duplicate command events");
  }
  std::filesystem::remove_all(directory);
}

void test_submission_and_queue_boundary() {
  const auto fixture = load_fixture();
  const auto json_source = fixture.dump();
  const auto json_result = trainvm::compile_document_source(json_source, "json");
  const auto yaml_result = trainvm::compile_document_source(json_source, "yaml");
  check(json_result.valid() && yaml_result.valid() &&
            json_result.plan->plan_hash == yaml_result.plan->plan_hash,
        "in-memory JSON and YAML submission sources compile identically");
  check(!trainvm::compile_document_source(json_source, "toml").valid(),
        "submission compiler rejects unsupported source formats");

  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-submission-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  std::string journal_id;
  {
    trainvm::Journal journal(database_path);
    journal_id = journal.journal_id();
  }
  auto service = std::make_unique<trainvm::TrainVMService>(
      database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
      fixture_disabled_host_launch_registry());
  std::string adapter_lock_digest;
  const auto request = [&](bool create_run, std::string key, std::string source = {}) {
    trainvm::v1::SubmitExperimentRequest output;
    output.set_source_document(source.empty() ? json_source : std::move(source));
    output.set_source_format("json");
    output.set_create_run(create_run);
    output.set_idempotency_key(std::move(key));
    output.set_expected_journal_id(journal_id);
    output.set_expected_plan_hash(json_result.plan->plan_hash);
    if (create_run && !adapter_lock_digest.empty()) {
      output.set_expected_adapter_lock_digest(adapter_lock_digest);
    }
    output.set_author("dashboard");
    output.set_reason("submission test");
    return output;
  };

  auto validate = request(false, "");
  trainvm::v1::SubmitExperimentResponse validate_response;
  const auto validate_status = service->SubmitExperiment(nullptr, &validate, &validate_response);
  check(validate_status.ok() && !validate_response.has_run() &&
            validate_response.plan_hash() == json_result.plan->plan_hash &&
            validate_response.adapter_lock_digest().starts_with("sha256:") &&
            validate_response.adapter_lock_digest().size() == 71U &&
            validate_response.canonical_document() == validate_response.canonical_plan(),
        "validation-only submission returns canonical content without a run");
  adapter_lock_digest = validate_response.adapter_lock_digest();
  {
    trainvm::Journal journal(database_path);
    check(journal.event_count() == 0U,
          "validation-only submission leaves the authority journal unchanged");
  }

  auto missing_adapter_lock = request(true, "missing-adapter-lock");
  missing_adapter_lock.clear_expected_adapter_lock_digest();
  trainvm::v1::SubmitExperimentResponse missing_adapter_lock_response;
  const auto missing_adapter_lock_status = service->SubmitExperiment(
      nullptr, &missing_adapter_lock, &missing_adapter_lock_response);
  check(missing_adapter_lock_status.error_code() ==
            grpc::StatusCode::INVALID_ARGUMENT,
        "run submission requires the adapter lock returned by preview");

  auto create = request(true, "submission-1");
  trainvm::v1::SubmitExperimentResponse created;
  const auto create_status = service->SubmitExperiment(nullptr, &create, &created);
  check(create_status.ok() && created.has_run() && created.run().revision() == 1U &&
            created.run().plan_hash() == json_result.plan->plan_hash &&
            created.adapter_lock_digest() == adapter_lock_digest,
        "run submission creates a deterministic revision-one run");
  const std::string run_id = created.run().run_id();
  {
    trainvm::Journal journal(database_path);
    const auto projection = journal.projection(run_id);
    check(projection && projection->desired_state == "queued" &&
              projection->observed_state == "queued" && projection->current_node_id.empty() &&
              journal.event_count() == 1U,
          "new submissions are durably queued without dispatching work");
  }

  trainvm::v1::SubmitExperimentResponse replayed;
  const auto replay_status = service->SubmitExperiment(nullptr, &create, &replayed);
  check(replay_status.ok() && replayed.has_run() && replayed.run().run_id() == run_id,
        "exact submission retry returns the original run identity");
  {
    trainvm::Journal journal(database_path);
    check(journal.event_count() == 1U,
          "exact submission retry does not duplicate durable creation events");
  }

  {
    service.reset();
    auto unresolved_profiles = fixture_adapter_profiles();
    unresolved_profiles.erase(
        std::remove_if(unresolved_profiles.begin(), unresolved_profiles.end(),
                       [](const trainvm::AdapterProfile& profile) {
                         return profile.key.operation == "train";
                       }),
        unresolved_profiles.end());
    trainvm::TrainVMService unresolved_registry_service(
        database_path,
        trainvm::AdapterRegistry(std::move(unresolved_profiles)),
        fixture_disabled_host_launch_registry());
    trainvm::v1::SubmitExperimentResponse unresolved_registry_replay;
    const auto unresolved_registry_replay_status =
        unresolved_registry_service.SubmitExperiment(
            nullptr, &create, &unresolved_registry_replay);
    auto unresolved_registry_conflict = create;
    unresolved_registry_conflict.set_reason("changed retry identity");
    trainvm::v1::SubmitExperimentResponse
        unresolved_registry_conflict_response;
    const auto unresolved_registry_conflict_status =
        unresolved_registry_service.SubmitExperiment(
            nullptr, &unresolved_registry_conflict,
            &unresolved_registry_conflict_response);
    trainvm::Journal journal(database_path);
    check(unresolved_registry_replay_status.ok() &&
              unresolved_registry_replay.has_run() &&
              unresolved_registry_replay.run().run_id() == run_id &&
              unresolved_registry_replay.run().revision() == 1U &&
              unresolved_registry_replay.adapter_lock_digest() ==
                  adapter_lock_digest &&
              unresolved_registry_conflict_status.error_code() ==
                  grpc::StatusCode::ALREADY_EXISTS &&
              journal.event_count() == 1U,
          "durable submission identity replays before current registry resolution and still rejects changed retries");
  }
  service = std::make_unique<trainvm::TrainVMService>(
      database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
      fixture_disabled_host_launch_registry());

  auto concurrent_request = request(true, "concurrent-submission");
  auto concurrent_submit = [&] {
    trainvm::v1::SubmitExperimentResponse response;
    const auto status = service->SubmitExperiment(nullptr, &concurrent_request, &response);
    return std::pair{status, response.run().run_id()};
  };
  auto concurrent_left = std::async(std::launch::async, concurrent_submit);
  auto concurrent_right = std::async(std::launch::async, concurrent_submit);
  const auto left_submission = concurrent_left.get();
  const auto right_submission = concurrent_right.get();
  check(left_submission.first.ok() && right_submission.first.ok() &&
            !left_submission.second.empty() && left_submission.second == right_submission.second,
        "concurrent exact submissions converge on one deterministic run identity");

  auto changed_fixture = fixture;
  changed_fixture["metadata"]["name"] = "changed-submission";
  auto changed = request(true, "submission-1", changed_fixture.dump());
  const auto changed_compiled = trainvm::compile_document(changed_fixture);
  check(changed_compiled.valid(), "changed submission conflict fixture compiles");
  changed.set_expected_plan_hash(changed_compiled.plan->plan_hash);
  trainvm::v1::SubmitExperimentResponse changed_response;
  const auto changed_status = service->SubmitExperiment(nullptr, &changed, &changed_response);
  check(changed_status.error_code() == grpc::StatusCode::ALREADY_EXISTS,
        "same submission key with changed canonical content conflicts");
  auto changed_reason = create;
  changed_reason.set_reason("different retry content");
  trainvm::v1::SubmitExperimentResponse changed_reason_response;
  const auto changed_reason_status =
      service->SubmitExperiment(nullptr, &changed_reason, &changed_reason_response);
  check(changed_reason_status.error_code() == grpc::StatusCode::ALREADY_EXISTS,
        "same submission key with changed audit content conflicts");

  auto wrong_authority = request(false, "");
  wrong_authority.set_expected_journal_id("00000000000000000000000000000000");
  trainvm::v1::SubmitExperimentResponse wrong_authority_response;
  const auto wrong_authority_status =
      service->SubmitExperiment(nullptr, &wrong_authority, &wrong_authority_response);
  check(wrong_authority_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
        "submission refuses a mismatched journal authority identity");

  auto wrong_preview = request(true, "wrong-preview");
  wrong_preview.set_expected_plan_hash(std::string(64, '0'));
  trainvm::v1::SubmitExperimentResponse wrong_preview_response;
  const auto wrong_preview_status =
      service->SubmitExperiment(nullptr, &wrong_preview, &wrong_preview_response);
  check(wrong_preview_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
        "submission refuses compiler skew from the validated preview plan");

  auto wrong_adapter_lock = request(true, "wrong-adapter-lock");
  wrong_adapter_lock.set_expected_adapter_lock_digest(
      "sha256:" + std::string(64, '0'));
  trainvm::v1::SubmitExperimentResponse wrong_adapter_lock_response;
  const auto wrong_adapter_lock_status = service->SubmitExperiment(
      nullptr, &wrong_adapter_lock, &wrong_adapter_lock_response);
  check(wrong_adapter_lock_status.error_code() ==
            grpc::StatusCode::FAILED_PRECONDITION,
        "submission refuses adapter registry skew from the validated preview");

  sqlite3* raw_database = nullptr;
  check(sqlite3_open(database_path.c_str(), &raw_database) == SQLITE_OK,
        "submission recovery test opens journal for fault injection");
  if (raw_database != nullptr) {
    const std::string corrupt_projection =
        "UPDATE run_projection SET desired_state='running' WHERE run_id='" + run_id + "'";
    check(sqlite3_exec(raw_database, corrupt_projection.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK,
          "submission recovery test corrupts its rebuildable projection");
    sqlite3_close(raw_database);
    raw_database = nullptr;
  }
  trainvm::v1::SubmitExperimentResponse inconsistent_replay;
  const auto inconsistent_status =
      service->SubmitExperiment(nullptr, &create, &inconsistent_replay);
  check(inconsistent_status.error_code() == grpc::StatusCode::DATA_LOSS,
        "idempotent replay refuses projection and journal disagreement");
  check(sqlite3_open(database_path.c_str(), &raw_database) == SQLITE_OK,
        "submission recovery test reopens journal projection");
  if (raw_database != nullptr) {
    const std::string restore_projection =
        "UPDATE run_projection SET desired_state='queued' WHERE run_id='" + run_id + "'";
    check(sqlite3_exec(raw_database, restore_projection.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK,
          "submission recovery test restores its projection");
    sqlite3_close(raw_database);
    raw_database = nullptr;
  }

  {
    trainvm::Journal journal(database_path);
    trainvm::Controller queued(*json_result.plan, journal, run_id);
    queued.recover();
    bool dispatch_refused = false;
    try {
      (void)queued.prepare_dispatch();
    } catch (const std::logic_error&) {
      dispatch_refused = true;
    }
    check(dispatch_refused, "queued runs cannot dispatch work before they are started");
    const auto projection = journal.projection(run_id);
    check(projection && projection->observed_state == "queued",
          "submission cannot fabricate running state without lease and worker evidence");
  }

  check(sqlite3_open(database_path.c_str(), &raw_database) == SQLITE_OK,
        "submission chain test opens journal for fault injection");
  if (raw_database != nullptr) {
    const std::string corrupt_chain =
        "UPDATE events SET chain_hash='" + std::string(64, '0') +
        "' WHERE event_id='" + run_id + ":created'";
    check(sqlite3_exec(raw_database, corrupt_chain.c_str(), nullptr, nullptr, nullptr) == SQLITE_OK,
          "submission chain test tampers with the event chain");
    sqlite3_close(raw_database);
  }
  trainvm::v1::SubmitExperimentResponse corrupt_replay;
  const auto corrupt_status =
      service->SubmitExperiment(nullptr, &create, &corrupt_replay);
  check(corrupt_status.error_code() == grpc::StatusCode::DATA_LOSS,
        "idempotent replay refuses a tampered global event chain");
  std::filesystem::remove_all(directory);
}

void test_atomic_queue_acquisition_boundary() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by queue acquisition compiles");
  if (!compiled.valid())
    return;
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-acquisition-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  trainvm::Journal journal(database_path);
  trainvm::Controller controller(*compiled.plan, journal, "queued-acquisition-run");
  controller.create_queued();

  const auto blocker =
      journal.acquire_lease(compiled.plan->experiment.spec.workspace.concurrency_key,
                            "blocking-run", "blocking-lease",
                            test_time(100, 10'000), 1'000);
  check(blocker.status == trainvm::LeaseAcquireStatus::acquired,
        "queue acquisition test establishes a competing lease");
  const auto busy = controller.begin_acquisition(test_time(200, 20'000));
  const auto queued = journal.projection("queued-acquisition-run");
  check(busy.status == trainvm::LeaseAcquireStatus::busy && queued &&
            queued->desired_state == "queued" && queued->observed_state == "queued" &&
            queued->run_revision == 1U && journal.event_count() == 1U,
        "busy resource lease leaves the queued run and journal unchanged");

  check(journal.release_lease(blocker.lease.concurrency_key, blocker.lease.owner_run_id,
                              blocker.lease.lease_id, blocker.lease.fencing_token, test_time(250)),
        "queue acquisition test releases its competing lease");
  const auto acquired = controller.begin_acquisition(test_time(300, 30'000));
  const auto acquiring = journal.projection("queued-acquisition-run");
  check(acquired.status == trainvm::LeaseAcquireStatus::acquired && acquiring &&
            acquiring->desired_state == "running" && acquiring->observed_state == "acquiring" &&
            acquiring->run_revision == 4U && acquiring->current_node_id.empty() &&
            acquiring->current_attempt_id.empty() && journal.event_count() == 6U &&
            controller.state().current_node_id == "train_to_boundary",
        "lease acquisition advances builtin admission while remaining "
        "unassigned");
  const auto lease_event = journal.event("queued-acquisition-run:lease-acquired");
  const auto acquisition_events = journal.events_for_run("queued-acquisition-run");
  check(lease_event &&
            lease_event->wall_time_ns == 30'000 &&
            lease_event->payload.value("clock_domain", std::string{}) ==
                trainvm::ResourceLease::kBootTimeDomain &&
            lease_event->payload.value("boot_id", std::string{}) == kTestBootId &&
            lease_event->payload.value("acquired_boottime_ns", std::int64_t{}) == 300 &&
            lease_event->payload.value("fencing_token", std::uint64_t{}) ==
                acquired.lease.fencing_token &&
            lease_event->payload.value("owner_run_id", std::string{}) == "queued-acquisition-run" &&
            acquisition_events.size() == 6U &&
            acquisition_events[1].event_type == "run.desired_state_changed" &&
            acquisition_events[2].event_type == "resource.lease_acquired" &&
            acquisition_events[3].event_type == "run.observed_state_changed" &&
            acquisition_events[4].event_type == "resource.acquired" &&
            acquisition_events[5].event_type == "fsm.transitioned",
        "resource acquisition records its fence and builtin FSM transition");
  bool dispatch_refused = false;
  try {
    (void)controller.prepare_dispatch();
  } catch (const std::logic_error&) {
    dispatch_refused = true;
  }
  check(dispatch_refused && !journal.dispatch("queued-acquisition-run:dispatch:train_to_"
                                              "boundary:train_to_boundary@1"),
        "acquiring state cannot dispatch before verified worker readiness");

  const auto expires_at = acquired.lease.expires_boottime_ns;
  const auto replayed = controller.begin_acquisition(test_time(400, 40'000));
  check(replayed.status == trainvm::LeaseAcquireStatus::already_owned &&
            replayed.lease.fencing_token == acquired.lease.fencing_token &&
            replayed.lease.expires_boottime_ns == expires_at && journal.event_count() == 6U,
        "acquisition retry neither renews the lease nor duplicates admission "
        "events");
  bool negative_retry_rejected = false;
  try {
    (void)controller.begin_acquisition(test_time(-1));
  } catch (const std::invalid_argument&) {
    negative_retry_rejected = true;
  }
  check(negative_retry_rejected, "acquisition retry rejects a negative lease clock consistently");
  trainvm::Controller restarted(*compiled.plan, journal, "queued-acquisition-run");
  check(restarted.recover().revision == 4U &&
            restarted.state().current_node_id == "train_to_boundary",
        "controller recovery accepts acquiring as a stable process-free "
        "boundary");
  bool expired_reconcile_refused = false;
  try {
    (void)restarted.begin_acquisition(test_time(expires_at + 1));
  } catch (const std::runtime_error&) {
    expired_reconcile_refused = true;
  }
  check(expired_reconcile_refused && restarted.recover().revision == 4U,
        "expired acquiring lease remains replayable but requires a future "
        "fenced decision");
  const auto before_rebuild = journal.projection("queued-acquisition-run");
  journal.rebuild_projections();
  check(journal.projection("queued-acquisition-run") == before_rebuild,
        "projection rebuild reproduces queued-to-acquiring lifecycle state");
  auto invalid_admission_document = load_fixture();
  invalid_admission_document["spec"]["workflow"]["nodes"]["acquire_gpu"]
                            ["transitions"][0]["on"] = "resource.unexpected";
  const auto invalid_admission =
      trainvm::compile_document(invalid_admission_document);
  check(invalid_admission.valid(),
        "unsupported admission transition remains a structurally valid plan fixture");
  if (invalid_admission.valid()) {
    trainvm::Journal invalid_journal(directory / "invalid-admission.db");
    trainvm::Controller invalid_controller(*invalid_admission.plan, invalid_journal,
                                           "invalid-admission-run");
    invalid_controller.create_queued();
    bool rejected_before_lease = false;
    try {
      (void)invalid_controller.begin_acquisition(test_time(500));
    } catch (const std::exception&) {
      rejected_before_lease = true;
    }
    const auto invalid_projection = invalid_journal.projection("invalid-admission-run");
    check(rejected_before_lease && invalid_projection &&
              invalid_projection->desired_state == "queued" &&
              invalid_projection->observed_state == "queued" &&
              invalid_journal.event_count() == 1U &&
              !invalid_journal.active_lease(
                  invalid_admission.plan->experiment.spec.workspace.concurrency_key,
                  test_time(500)),
          "unsupported builtin admission is rejected before acquiring its lease");
  }

  auto conditional_admission_document = load_fixture();
  conditional_admission_document["spec"]["workflow"]["nodes"]["acquire_gpu"]
                                ["transitions"] = nlohmann::json::array(
      {{{"on", "resource.acquired"},
        {"where", {{"field", "payload.fencing_token"},
                   {"operator", "exists"},
                   {"value", nullptr}}},
        {"target", "$failed"}},
       {{"on", "resource.acquired"}, {"target", "train_to_boundary"}},
       {{"on", "operation.failed"}, {"target", "$failed"}}});
  const auto conditional_admission =
      trainvm::compile_document(conditional_admission_document);
  check(conditional_admission.valid(),
        "conditional admission with an unconditional fallback is compile-valid");
  if (conditional_admission.valid()) {
    const std::string conditional_run_id = "conditional-admission-run";
    const auto admission_state =
        trainvm::start_execution(*conditional_admission.plan, conditional_run_id);
    const auto real_payload_route = trainvm::advance_execution(
        *conditional_admission.plan, admission_state,
        event_for(admission_state, "conditional-real-payload",
                  "resource.acquired", {{"fencing_token", 1U}}));
    check(real_payload_route.transition_index == 0U &&
              real_payload_route.target == "$failed" &&
              real_payload_route.state.status == trainvm::ExecutionStatus::failed,
          "real fenced admission payload selects the conditional branch instead of fallback");
    trainvm::Journal conditional_journal(directory / "conditional-admission.db");
    trainvm::Controller conditional_controller(
        *conditional_admission.plan, conditional_journal, conditional_run_id);
    conditional_controller.create_queued();
    const auto projection_before = conditional_journal.projection(conditional_run_id);
    const auto events_before = conditional_journal.event_count();
    bool rejected_before_mutation = false;
    std::string rejection_message;
    try {
      (void)conditional_controller.begin_acquisition(test_time(600));
    } catch (const std::exception& exception) {
      rejected_before_mutation = true;
      rejection_message = exception.what();
    }
    const auto projection_after = conditional_journal.projection(conditional_run_id);
    const std::string& concurrency_key =
        conditional_admission.plan->experiment.spec.workspace.concurrency_key;
    check(rejected_before_mutation && !rejection_message.empty() &&
              conditional_journal.event_count() == events_before &&
              events_before == 1U && projection_after == projection_before &&
              projection_after && projection_after->desired_state == "queued" &&
              projection_after->observed_state == "queued" &&
              !conditional_journal.active_lease(concurrency_key, test_time(600)),
          "payload-dependent builtin admission is rejected before lease or "
          "lifecycle mutation");
  }

  auto external_admission_document = load_fixture();
  external_admission_document["spec"]["components"]["core"]["runtime"] =
      "python_worker";
  const auto external_admission =
      trainvm::compile_document(external_admission_document);
  check(external_admission.valid(),
        "externalized admission remains a compile-valid negative fixture");
  if (external_admission.valid()) {
    const std::string external_run_id = "external-admission-run";
    trainvm::Journal external_journal(directory / "external-admission.db");
    trainvm::Controller external_controller(
        *external_admission.plan, external_journal, external_run_id);
    external_controller.create_queued();
    const auto projection_before = external_journal.projection(external_run_id);
    bool rejected_before_mutation = false;
    try {
      (void)external_controller.begin_acquisition(test_time(700));
    } catch (const std::logic_error&) {
      rejected_before_mutation = true;
    }
    check(rejected_before_mutation && external_journal.event_count() == 1U &&
              external_journal.projection(external_run_id) == projection_before &&
              !external_journal.active_lease(
                  external_admission.plan->experiment.spec.workspace.concurrency_key,
                  test_time(700)),
          "non-builtin queued entrypoint is rejected before lease or lifecycle "
          "mutation");
  }
  std::filesystem::remove_all(directory);
}

void test_worker_launch_and_readiness_boundary() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by worker readiness compiles");
  if (!compiled.valid())
    return;

  const auto hello_for = [](const trainvm::WorkerLaunchTicket& launch,
                            std::vector<std::string> capabilities) {
    return trainvm::WorkerHelloEvidence{
        .run_id = launch.run_id,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .launch_nonce = launch.launch_nonce,
        .adapter = launch.adapter,
        .adapter_version = launch.adapter_version,
        .code_fingerprint = launch.code_fingerprint,
        .capabilities = std::move(capabilities),
        .last_acked_controller_sequence = 0,
        .concurrency_key = launch.concurrency_key,
        .lease_id = launch.lease_id,
        .fencing_token = launch.fencing_token,
    };
  };
  const trainvm::WorkerLaunchRequest launch_request{
      .code_fingerprint = "sha256:" + std::string(64U, '1'),
      .required_capabilities = {"worker.metrics", "worker.controls"},
  };

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-worker-readiness-test-" + std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  {
    const auto database_path = directory / "accepted.db";
    const std::string run_id = "worker-readiness-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued();
    const auto acquired = controller.begin_acquisition(test_time(1'000));
    const auto acquiring = journal.projection(run_id);
    check(acquired.status == trainvm::LeaseAcquireStatus::acquired &&
              controller.state().revision == 4U &&
              controller.state().current_node_id == "train_to_boundary" && acquiring &&
              acquiring->desired_state == "running" && acquiring->observed_state == "acquiring" &&
              acquiring->run_revision == 4U && acquiring->current_node_id.empty() &&
              acquiring->current_attempt_id.empty(),
          "queued acquisition advances builtin admission to an unassigned "
          "revision-four node");

    const auto launch = controller.prepare_worker_launch(launch_request, test_time(1'100));
    const auto launch_retry = controller.prepare_worker_launch(launch_request, test_time(1'100));
    check(launch == launch_retry && launch.run_id == run_id &&
              launch.node_id == "train_to_boundary" && launch.attempt_id == "train_to_boundary@1" &&
              launch.code_fingerprint == launch_request.code_fingerprint &&
              launch.required_capabilities ==
                  std::vector<std::string>({"worker.controls", "worker.metrics"}) &&
              !launch.launch_nonce.empty() && journal.event_count() == 7U,
          "worker launch preparation freezes fingerprint and capabilities "
          "idempotently");
    (void)bind_test_worker_launch(controller, launch, 1'150);

    auto hello = hello_for(launch, {"worker.metrics", "worker.events", "worker.controls"});
    auto wrong_nonce = hello;
    wrong_nonce.launch_nonce += "-wrong";
    bool wrong_nonce_rejected = false;
    try {
      (void)controller.accept_worker_hello(std::move(wrong_nonce), test_time(1'200));
    } catch (const std::invalid_argument&) {
      wrong_nonce_rejected = true;
    }
    auto missing_capability = hello;
    missing_capability.capabilities = {"worker.controls"};
    bool missing_capability_rejected = false;
    try {
      (void)controller.accept_worker_hello(std::move(missing_capability), test_time(1'200));
    } catch (const std::invalid_argument&) {
      missing_capability_rejected = true;
    }
    const auto still_acquiring = journal.projection(run_id);
    check(wrong_nonce_rejected && missing_capability_rejected && still_acquiring &&
              still_acquiring->observed_state == "acquiring" &&
              still_acquiring->current_node_id.empty() && journal.event_count() == 8U,
          "worker hello rejects wrong nonce and missing required capability "
          "without mutation");

    const auto ready = controller.accept_worker_hello(hello, test_time(1'200));
    const auto running = journal.projection(run_id);
    const auto readiness_events = journal.events_for_run(run_id);
    check(ready.disposition == trainvm::WorkerReadinessDisposition::accepted && running &&
              running->desired_state == "running" && running->observed_state == "running" &&
              running->run_revision == 5U && running->current_node_id == "train_to_boundary" &&
              running->current_attempt_id == "train_to_boundary@1" &&
              controller.state().revision == 5U && readiness_events.size() == 11U &&
              readiness_events[8].event_type == "worker.ready" &&
              readiness_events[9].event_type == "run.observed_state_changed" &&
              readiness_events[9].payload.value("state", std::string{}) == "running" &&
              readiness_events[10].event_type == "node.entered",
          "matching worker hello atomically publishes readiness, running, and "
          "node assignment");
    const auto ready_retry = controller.accept_worker_hello(hello, test_time(1'250));
    check(ready_retry.disposition == trainvm::WorkerReadinessDisposition::replayed &&
              ready_retry.launch == launch && journal.event_count() == 11U,
          "exact worker hello retry replays without duplicate readiness events");

    bool unfenced_dispatch_rejected = false;
    try {
      (void)controller.prepare_dispatch();
    } catch (const std::logic_error&) {
      unfenced_dispatch_rejected = true;
    }
    const auto dispatch = controller.prepare_dispatch(test_time(1'300));
    check(unfenced_dispatch_rejected && dispatch.run_id == run_id && dispatch.run_revision == 5U &&
              dispatch.node_id == "train_to_boundary" &&
              dispatch.attempt_id == "train_to_boundary@1" &&
              dispatch.status == trainvm::DispatchStatus::prepared,
          "verified worker readiness permits only fenced first-node dispatch preparation");
    const trainvm::WorkerSessionIdentity session{
        .run_id = launch.run_id,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .launch_nonce = launch.launch_nonce,
        .concurrency_key = launch.concurrency_key,
        .lease_id = launch.lease_id,
        .fencing_token = launch.fencing_token,
    };
    const trainvm::Event result_event{
        .event_id = dispatch.dispatch_id + ":worker-result",
        .run_id = run_id,
        .run_revision = 5,
        .plan_revision = 1,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .worker_sequence = 1,
        .event_type = "worker.completed",
        .event_version = 1,
        .wall_time_ns = 1'350,
        .monotonic_time_ns = 1,
        .optimizer_step = 5'500,
        .payload = {{"reason", "cache_span_complete"}},
    };
    bool unfenced_result_rejected = false;
    try {
      (void)controller.handle_event(result_event);
    } catch (const std::invalid_argument&) {
      unfenced_result_rejected = true;
    }
    check(unfenced_result_rejected,
          "worker-backed result requires its accepted session identity");
    trainvm::Controller restarted(*compiled.plan, journal, run_id);
    const auto& recovered = restarted.recover();
    check(recovered.revision == 5U && recovered.current_node_id == "train_to_boundary" &&
              recovered.current_attempt_id == "train_to_boundary@1",
          "fresh controller recovers the ready worker and assigned first node");
    const auto before_rebuild = journal.projection(run_id);
    journal.rebuild_projections();
    check(journal.projection(run_id) == before_rebuild,
          "projection rebuild reproduces worker readiness and node assignment");
    const auto before_takeover = journal.event_count();
    check(journal.release_lease(acquired.lease.concurrency_key,
                                acquired.lease.owner_run_id,
                                acquired.lease.lease_id,
                                acquired.lease.fencing_token, test_time(1'400)),
          "stale worker fixture releases the accepted fence");
    const auto successor = journal.acquire_lease(
        acquired.lease.concurrency_key, "successor-run", "successor-lease",
        test_time(1'401), 60'000'000'000LL);
    bool stale_result_rejected = false;
    try {
      (void)controller.handle_event(result_event, session, test_time(1'500));
    } catch (const std::runtime_error&) {
      stale_result_rejected = true;
    }
    check(successor.status == trainvm::LeaseAcquireStatus::acquired &&
              stale_result_rejected && journal.event_count() == before_takeover,
          "stale worker cannot complete a prepared dispatch after fence takeover");
  }

  {
    const auto database_path = directory / "without-launch.db";
    const std::string run_id = "worker-without-launch-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued();
    const auto acquired = controller.begin_acquisition(test_time(2'000));
    trainvm::WorkerHelloEvidence hello{
        .run_id = run_id,
        .node_id = controller.state().current_node_id,
        .attempt_id = controller.state().current_attempt_id,
        .launch_nonce = "unissued-nonce",
        .adapter = "rwkv-lab.mageflow",
        .adapter_version = "1.0.0",
        .code_fingerprint = launch_request.code_fingerprint,
        .capabilities = launch_request.required_capabilities,
        .last_acked_controller_sequence = 0,
        .concurrency_key = acquired.lease.concurrency_key,
        .lease_id = acquired.lease.lease_id,
        .fencing_token = acquired.lease.fencing_token,
    };
    bool rejected = false;
    try {
      (void)controller.accept_worker_hello(std::move(hello), test_time(2'100));
    } catch (const std::logic_error&) {
      rejected = true;
    }
    check(rejected && journal.event_count() == 6U,
          "worker hello without a durable launch request is rejected without "
          "mutation");
  }

  {
    const auto database_path = directory / "expired-lease.db";
    const std::string run_id = "worker-expired-lease-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued();
    const auto acquired = controller.begin_acquisition(test_time(3'000));
    const auto launch = controller.prepare_worker_launch(launch_request, test_time(3'100));
    (void)bind_test_worker_launch(controller, launch, 3'150);
    auto hello = hello_for(launch, launch_request.required_capabilities);
    bool rejected = false;
    try {
      (void)controller.accept_worker_hello(std::move(hello), test_time(acquired.lease.expires_boottime_ns + 1));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    const auto projection = journal.projection(run_id);
    check(rejected && projection && projection->observed_state == "acquiring" &&
              projection->current_node_id.empty() && journal.event_count() == 8U,
          "worker hello with an expired lease is rejected without readiness "
          "evidence");
  }

  {
    const auto database_path = directory / "released-lease.db";
    const std::string run_id = "worker-released-lease-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued();
    const auto acquired = controller.begin_acquisition(test_time(4'000));
    const auto launch = controller.prepare_worker_launch(launch_request, test_time(4'100));
    (void)bind_test_worker_launch(controller, launch, 4'150);
    check(journal.release_lease(acquired.lease.concurrency_key, acquired.lease.owner_run_id,
                                acquired.lease.lease_id, acquired.lease.fencing_token, test_time(4'200)),
          "released-lease readiness fixture releases its acquired fence");
    auto hello = hello_for(launch, launch_request.required_capabilities);
    bool rejected = false;
    try {
      (void)controller.accept_worker_hello(std::move(hello), test_time(4'300));
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    const auto projection = journal.projection(run_id);
    check(rejected && projection && projection->observed_state == "acquiring" &&
              projection->current_node_id.empty() && journal.event_count() == 8U,
          "worker hello with a released lease is rejected without readiness "
          "evidence");
  }

  {
    const auto database_path = directory / "fenced-result.db";
    const std::string run_id = "worker-fenced-result-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued();
    (void)controller.begin_acquisition(test_time(5'000));
    const auto launch = controller.prepare_worker_launch(launch_request, test_time(5'100));
    (void)bind_test_worker_launch(controller, launch, 5'150);
    (void)controller.accept_worker_hello(
        hello_for(launch, launch_request.required_capabilities), test_time(5'200));
    const auto dispatch = controller.prepare_dispatch(test_time(5'300));
    const trainvm::WorkerSessionIdentity session{
        .run_id = launch.run_id,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .launch_nonce = launch.launch_nonce,
        .concurrency_key = launch.concurrency_key,
        .lease_id = launch.lease_id,
        .fencing_token = launch.fencing_token,
    };
    const trainvm::Event result{
        .event_id = dispatch.dispatch_id + ":result",
        .run_id = run_id,
        .run_revision = 5,
        .plan_revision = 1,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .worker_sequence = 1,
        .event_type = "worker.completed",
        .event_version = 1,
        .wall_time_ns = 5'400,
        .monotonic_time_ns = 1,
        .optimizer_step = 5'500,
        .payload = {{"reason", "cache_span_complete"}},
    };
    const auto& advanced =
        controller.handle_event(result, session, test_time(5'400));
    const auto receipt = journal.dispatch(dispatch.dispatch_id);
    const auto reacquiring = journal.projection(run_id);
    bool unfenced_next_dispatch_rejected = false;
    bool unready_next_dispatch_rejected = false;
    try {
      (void)controller.prepare_dispatch();
    } catch (const std::logic_error&) {
      unfenced_next_dispatch_rejected = true;
    }
    try {
      (void)controller.prepare_dispatch(test_time(5'450));
    } catch (const std::logic_error&) {
      unready_next_dispatch_rejected = true;
    }
    check(advanced.revision == 7U && advanced.current_node_id == "prepare_cache" &&
              reacquiring && reacquiring->observed_state == "acquiring" &&
              reacquiring->run_revision == 7U &&
              reacquiring->current_node_id.empty() &&
              reacquiring->current_attempt_id.empty() &&
              receipt && receipt->status == trainvm::DispatchStatus::completed &&
              receipt->result_event_id == std::optional<std::string>{result.event_id} &&
              unfenced_next_dispatch_rejected && unready_next_dispatch_rejected,
          "active fenced result returns the next external node to an unassigned "
          "readiness boundary");
    const auto next_launch =
        controller.prepare_worker_launch(launch_request, test_time(5'500));
    (void)bind_test_worker_launch(controller, next_launch, 5'550);
    const auto next_ready = controller.accept_worker_hello(
        hello_for(next_launch, launch_request.required_capabilities),
        test_time(5'600));
    const auto next_running = journal.projection(run_id);
    const auto next_dispatch = controller.prepare_dispatch(test_time(5'700));
    check(next_launch.node_id == "prepare_cache" &&
              next_launch.attempt_id == "prepare_cache@1" &&
              next_launch.launch_nonce != launch.launch_nonce &&
              next_ready.disposition ==
                  trainvm::WorkerReadinessDisposition::accepted &&
              next_running && next_running->observed_state == "running" &&
              next_running->run_revision == 8U &&
              next_running->current_node_id == "prepare_cache" &&
              next_running->current_attempt_id == "prepare_cache@1" &&
              next_dispatch.node_id == "prepare_cache",
          "each external node requires a distinct launch and WorkerHello before "
          "fenced dispatch");

    const auto session_for = [](const trainvm::WorkerLaunchTicket& ticket) {
      return trainvm::WorkerSessionIdentity{
          .run_id = ticket.run_id,
          .node_id = ticket.node_id,
          .attempt_id = ticket.attempt_id,
          .launch_nonce = ticket.launch_nonce,
          .concurrency_key = ticket.concurrency_key,
          .lease_id = ticket.lease_id,
          .fencing_token = ticket.fencing_token,
      };
    };
    const trainvm::Event cache_prepared{
        .event_id = next_dispatch.dispatch_id + ":result",
        .run_id = run_id,
        .run_revision = 8,
        .plan_revision = 1,
        .node_id = next_launch.node_id,
        .attempt_id = next_launch.attempt_id,
        .worker_sequence = 1,
        .event_type = "operation.completed",
        .event_version = 1,
        .wall_time_ns = 5'800,
        .monotonic_time_ns = 2,
        .optimizer_step = 5'500,
        .payload = nlohmann::json::object(),
    };
    (void)controller.handle_event(cache_prepared, session_for(next_launch),
                                  test_time(5'800));
    const auto build_launch =
        controller.prepare_worker_launch(launch_request, test_time(5'900));
    (void)bind_test_worker_launch(controller, build_launch, 5'950);
    (void)controller.accept_worker_hello(
        hello_for(build_launch, launch_request.required_capabilities),
        test_time(6'000));
    const auto build_dispatch = controller.prepare_dispatch(test_time(6'100));
    const trainvm::Event cache_built{
        .event_id = build_dispatch.dispatch_id + ":result",
        .run_id = run_id,
        .run_revision = 11,
        .plan_revision = 1,
        .node_id = build_launch.node_id,
        .attempt_id = build_launch.attempt_id,
        .worker_sequence = 1,
        .event_type = "operation.completed",
        .event_version = 1,
        .wall_time_ns = 6'200,
        .monotonic_time_ns = 3,
        .optimizer_step = 5'500,
        .payload = nlohmann::json::object(),
    };
    const auto& builtin = controller.handle_event(
        cache_built, session_for(build_launch), test_time(6'200));
    check(builtin.revision == 12U &&
              builtin.current_node_id == "validate_cache" &&
              journal.projection(run_id)->observed_state == "running",
          "external result may enter a managed builtin node immediately");
    const auto before_wrong_builtin = journal.event_count();
    bool wrong_builtin_rejected = false;
    try {
      (void)controller.release_managed_resources(test_time(6'300));
    } catch (const std::logic_error&) {
      wrong_builtin_rejected = true;
    }
    bool simulation_dispatch_rejected = false;
    try {
      (void)controller.prepare_dispatch();
    } catch (const std::logic_error&) {
      simulation_dispatch_rejected = true;
    }
    const trainvm::Event simulated_validation{
        .event_id = run_id + ":simulated-validation",
        .run_id = run_id,
        .run_revision = builtin.revision,
        .plan_revision = 1,
        .node_id = builtin.current_node_id,
        .attempt_id = builtin.current_attempt_id,
        .worker_sequence = 1,
        .event_type = "artifact.validated",
        .event_version = 1,
        .wall_time_ns = 6'300,
        .monotonic_time_ns = 4,
        .optimizer_step = 5'500,
        .payload = nlohmann::json::object(),
    };
    bool simulation_result_rejected = false;
    try {
      (void)controller.handle_event(simulated_validation);
    } catch (const std::logic_error&) {
      simulation_result_rejected = true;
    }
    check(simulation_dispatch_rejected && simulation_result_rejected &&
              journal.event_count() == before_wrong_builtin,
          "managed artifact validation rejects generic simulation hooks without mutation");
    const auto& builtin_advanced = controller.complete_artifact_validation(
        trainvm::ArtifactValidationOutcome::valid, test_time(6'300));
    const auto builtin_reacquiring = journal.projection(run_id);
    const auto validation_events = journal.events_for_run(run_id);
    const auto validation_result = std::ranges::find_if(
        validation_events, [](const trainvm::Event& event) {
          return event.event_type == "artifact.validated";
        });
    trainvm::Controller builtin_restart(*compiled.plan, journal, run_id);
    const auto& builtin_recovered = builtin_restart.recover();
    check(wrong_builtin_rejected && simulation_dispatch_rejected &&
              simulation_result_rejected &&
              journal.event_count() == before_wrong_builtin + 5U &&
              validation_result != validation_events.end() &&
              validation_result->worker_sequence == 0U &&
              validation_result->payload.empty() &&
              builtin_advanced.revision == 14U &&
              builtin_advanced.current_node_id == "resume_training" &&
              builtin_reacquiring &&
              builtin_reacquiring->observed_state == "acquiring" &&
              builtin_reacquiring->current_node_id.empty() &&
              builtin_reacquiring->current_attempt_id.empty() &&
              builtin_recovered == builtin_advanced,
          "typed artifact validation authors a canonical result and durably "
          "returns to worker acquisition");
    const auto resume_launch =
        builtin_restart.prepare_worker_launch(launch_request, test_time(6'400));
    (void)bind_test_worker_launch(builtin_restart, resume_launch, 6'450);
    (void)builtin_restart.accept_worker_hello(
        hello_for(resume_launch, launch_request.required_capabilities),
        test_time(6'500));
    check(resume_launch.node_id == "resume_training" &&
              builtin_restart.state().revision == 15U &&
              journal.projection(run_id)->current_node_id == "resume_training",
          "builtin-to-external progress also requires a fresh WorkerHello");
  }

  std::filesystem::remove_all(directory);
}

void test_worker_control_service_boundary() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by WorkerControl service compiles");
  if (!compiled.valid()) return;
  const nlohmann::json submission_identity =
      fixture_adapter_locked_submission(*compiled.plan);

  const auto wire_hello = [](const trainvm::WorkerLaunchTicket& launch) {
    trainvm::v1::WorkerHello hello;
    hello.set_run_id(launch.run_id);
    hello.set_node_id(launch.node_id);
    hello.set_attempt_id(launch.attempt_id);
    hello.set_launch_nonce(launch.launch_nonce);
    hello.set_adapter(launch.adapter);
    hello.set_adapter_version(launch.adapter_version);
    hello.set_code_fingerprint(launch.code_fingerprint);
    for (const auto& capability : launch.required_capabilities) {
      hello.add_capabilities(capability);
    }
    hello.set_last_acked_controller_sequence(0);
    hello.set_concurrency_key(launch.concurrency_key);
    hello.set_lease_id(launch.lease_id);
    hello.set_fencing_token(launch.fencing_token);
    return hello;
  };
  const auto wire_result = [](const trainvm::TrainVMService::WorkerConnection& connection) {
    trainvm::v1::EventEnvelope event;
    event.set_event_id(connection.dispatch.dispatch_id + ":result");
    event.set_run_id(connection.identity.run_id);
    event.set_run_revision(connection.dispatch.run_revision);
    event.set_plan_revision(connection.dispatch.plan_revision);
    event.set_node_id(connection.identity.node_id);
    event.set_attempt_id(connection.identity.attempt_id);
    event.set_worker_sequence(1);
    event.set_event_type("worker.completed");
    event.set_event_version(1);
    event.mutable_wall_time()->set_seconds(1);
    event.set_monotonic_time_ns(1);
    event.set_optimizer_step(5'500);
    event.set_canonical_json_payload(R"({"reason":"cache_span_complete"})");
    return event;
  };
  const trainvm::WorkerLaunchRequest launch_request{
      .code_fingerprint = "sha256:" + std::string(64U, '2'),
      .required_capabilities = {"worker.controls", "worker.metrics"},
  };

  // WorkerToController is deliberately a closed oneof. Connect additionally
  // requires kHello first and currently rejects every subsequent case except
  // kEvent; the helper boundary below covers the accepted event path.
  trainvm::v1::WorkerToController first_message;
  check(first_message.message_case() ==
            trainvm::v1::WorkerToController::MESSAGE_NOT_SET,
        "WorkerControl stream has no implicit first-message variant");
  first_message.mutable_heartbeat()->set_worker_sequence(1);
  check(!first_message.has_hello() &&
            first_message.message_case() ==
                trainvm::v1::WorkerToController::kHeartbeat,
        "a heartbeat is distinguishable from the required first WorkerHello");
  first_message.mutable_metric()->set_name("loss");
  check(first_message.message_case() == trainvm::v1::WorkerToController::kMetric &&
            !first_message.has_heartbeat() && !first_message.has_event(),
        "unsupported WorkerControl variants cannot alias the result event case");

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-worker-control-service-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  {
    const auto database_path = directory / "accepted.db";
    const std::string run_id = "worker-control-service-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(database_path);
      trainvm::Controller controller(*compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      (void)controller.begin_acquisition(test_time(1'000));
      launch = controller.prepare_worker_launch(launch_request, test_time(1'100));
      (void)bind_test_worker_launch(controller, launch, 1'150);
      check(journal.event_count() == 8U,
            "WorkerControl fixture stops at a durable host launch binding");
    }

    std::int64_t authority_now_ns = 1'200;
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*compiled.plan, launch),
        fixture_test_host_identity(),
        [&authority_now_ns] { return test_time(authority_now_ns); });
    const auto hello = wire_hello(launch);
    trainvm::TrainVMService::WorkerConnection unbound_connection;
    const std::size_t before_unbound =
        trainvm::Journal(database_path).event_count();
    const grpc::Status unbound =
        service.open_worker_connection(hello, unbound_connection);
    check(unbound.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
              trainvm::Journal(database_path).event_count() == before_unbound,
          "WorkerControl refuses historical binding evidence until the current authority retains its exact bundle");
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(hello, connection);
    {
      trainvm::Journal observer(database_path);
      const auto projection = observer.projection(run_id);
      const auto dispatch = observer.dispatch(connection.dispatch.dispatch_id);
      check(open.ok() &&
                connection.welcome.disposition() ==
                    trainvm::v1::WorkerWelcome::DISPOSITION_ACCEPTED &&
                connection.welcome.run_revision() == 5U &&
                connection.welcome.dispatch_id() ==
                    connection.dispatch.dispatch_id &&
                projection && projection->observed_state == "running" &&
                projection->run_revision == 5U &&
                projection->current_node_id == launch.node_id && dispatch &&
                dispatch->status == trainvm::DispatchStatus::prepared &&
                observer.event_count() == 12U,
            "WorkerControl returns Welcome only after readiness and dispatch are durable");
    }
    if (!open.ok()) {
      std::filesystem::remove_all(directory);
      return;
    }

    auto canonical = wire_result(connection);
    const auto reject_without_mutation =
        [&](trainvm::v1::EventEnvelope candidate, grpc::StatusCode expected,
            std::string_view message) {
          trainvm::Journal before(database_path);
          const auto count = before.event_count();
          trainvm::v1::WorkerReceipt ignored;
          const grpc::Status status = service.complete_worker_connection(
              candidate, connection, ignored);
          trainvm::Journal after(database_path);
          check(status.error_code() == expected && after.event_count() == count,
                message);
        };

    auto malformed = canonical;
    malformed.set_canonical_json_payload("{not-json");
    reject_without_mutation(malformed, grpc::StatusCode::INVALID_ARGUMENT,
                            "WorkerControl rejects malformed JSON without mutation");
    auto noncanonical = canonical;
    noncanonical.set_canonical_json_payload(
        R"({"reason": "cache_span_complete"})");
    reject_without_mutation(
        noncanonical, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects noncanonical JSON without mutation");
    auto wrong_revision = canonical;
    wrong_revision.set_run_revision(canonical.run_revision() + 1U);
    reject_without_mutation(
        wrong_revision, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects a mismatched run revision without mutation");
    auto wrong_plan_revision = canonical;
    wrong_plan_revision.set_plan_revision(canonical.plan_revision() + 1U);
    reject_without_mutation(
        wrong_plan_revision, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects a mismatched plan revision without mutation");
    auto wrong_sequence = canonical;
    wrong_sequence.set_worker_sequence(2);
    reject_without_mutation(
        wrong_sequence, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects a noncanonical worker sequence without mutation");
    auto any_payload = canonical;
    any_payload.mutable_payload()->set_type_url("type.googleapis.com/test.Unsupported");
    any_payload.mutable_payload()->set_value("opaque");
    reject_without_mutation(
        any_payload, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects Any payload ingress without mutation");
    constexpr std::int64_t maximum_timestamp_seconds =
        std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL;
    constexpr std::int32_t maximum_timestamp_remainder =
        static_cast<std::int32_t>(
            std::numeric_limits<std::int64_t>::max() % 1'000'000'000LL);
    auto overflowing_timestamp = canonical;
    overflowing_timestamp.mutable_wall_time()->set_seconds(
        maximum_timestamp_seconds);
    overflowing_timestamp.mutable_wall_time()->set_nanos(
        maximum_timestamp_remainder + 1);
    reject_without_mutation(
        overflowing_timestamp, grpc::StatusCode::INVALID_ARGUMENT,
        "WorkerControl rejects a timestamp one nanosecond beyond int64 range");

    authority_now_ns = 1'400;
    canonical.mutable_wall_time()->set_seconds(maximum_timestamp_seconds);
    canonical.mutable_wall_time()->set_nanos(maximum_timestamp_remainder);
    trainvm::v1::WorkerReceipt receipt;
    const grpc::Status complete = service.complete_worker_connection(
        canonical, connection, receipt);
    const std::string first_receipt = receipt.SerializeAsString();
    {
      trainvm::Journal observer(database_path);
      const auto projection = observer.projection(run_id);
      const auto dispatch = observer.dispatch(connection.dispatch.dispatch_id);
      check(complete.ok() && receipt.event_id() == canonical.event_id() &&
                receipt.acknowledged_worker_sequence() == 1U &&
                receipt.committed_run_revision() == 7U &&
                receipt.observed_state() ==
                    trainvm::v1::OBSERVED_STATE_ACQUIRING &&
                projection && projection->run_revision == 7U &&
                projection->observed_state == "acquiring" && dispatch &&
                dispatch->status == trainvm::DispatchStatus::completed &&
                dispatch->result_event_id ==
                    std::optional<std::string>{canonical.event_id()} &&
                observer.event_count() == 16U &&
                service.resolved_launches_.empty(),
            "WorkerControl accepts the maximum int64 timestamp, commits the canonical result, "
            "returns its durable Receipt, and releases the retained launch bundle");
    }

    trainvm::v1::WorkerReceipt retry_receipt;
    const grpc::Status retry = service.complete_worker_connection(
        canonical, connection, retry_receipt);
    trainvm::TrainVMService::WorkerConnection reconnected;
    const grpc::Status reconnect =
        service.open_worker_connection(hello, reconnected);
    {
      trainvm::Journal observer(database_path);
      check(retry.ok() && retry_receipt.SerializeAsString() == first_receipt &&
                reconnect.ok() &&
                reconnected.welcome.disposition() ==
                    trainvm::v1::WorkerWelcome::DISPOSITION_ALREADY_COMPLETED &&
                reconnected.completed_receipt &&
                reconnected.completed_receipt->SerializeAsString() ==
                    first_receipt &&
                observer.event_count() == 16U,
            "lost WorkerControl receipts replay exactly without duplicate commits");
    }
  }

  {
    const auto database_path = directory / "expired-between-phases.db";
    const std::string run_id = "worker-control-expired-between-phases-run";
    trainvm::WorkerLaunchTicket launch;
    trainvm::ResourceLease lease;
    {
      trainvm::Journal journal(database_path);
      trainvm::Controller controller(*compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      lease = controller.begin_acquisition(test_time(3'000)).lease;
      launch = controller.prepare_worker_launch(launch_request, test_time(3'100));
      (void)bind_test_worker_launch(controller, launch, 3'150);
    }
    std::size_t clock_sample = 0;
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*compiled.plan, launch),
        fixture_test_host_identity(),
        [&] {
          ++clock_sample;
          return test_time(clock_sample == 1U
                               ? lease.expires_boottime_ns - 1
                               : lease.expires_boottime_ns);
        });
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status expired =
        service.open_worker_connection(wire_hello(launch), connection);
    trainvm::Journal observer(database_path);
    const auto projection = observer.projection(run_id);
    if (!(expired.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
          clock_sample == 2U && projection &&
          projection->observed_state == "running" &&
          projection->run_revision == 5U &&
          !observer.dispatch(run_id + ":dispatch:" + launch.node_id +
                             ":" + launch.attempt_id) &&
          observer.event_count() == 11U)) {
      std::cerr << "expired-between-phases: status=" << expired.error_code()
                << " message='" << expired.error_message()
                << "' samples=" << clock_sample
                << " observed="
                << (projection ? projection->observed_state : "<missing>")
                << " revision=" << (projection ? projection->run_revision : 0U)
                << " events=" << observer.event_count() << '\n';
    }
    check(expired.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
              clock_sample == 2U && projection &&
              projection->observed_state == "running" &&
              projection->run_revision == 5U &&
              !observer.dispatch(run_id + ":dispatch:" + launch.node_id +
                                 ":" + launch.attempt_id) &&
              observer.event_count() == 11U,
          "WorkerControl resamples time and refuses dispatch when the lease expires "
          "after hello readiness");
    service.prune_retained_launches(test_time(lease.expires_boottime_ns));
    check(service.resolved_launches_.empty(),
          "expired same-attempt leases release their retained launch bundles");
  }

  {
    const auto database_path = directory / "stale-fence.db";
    const std::string run_id = "worker-control-stale-fence-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(database_path);
      trainvm::Controller controller(*compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      (void)controller.begin_acquisition(test_time(2'000));
      launch = controller.prepare_worker_launch(launch_request, test_time(2'100));
      (void)bind_test_worker_launch(controller, launch, 2'150);
    }
    std::int64_t authority_now_ns = 2'200;
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*compiled.plan, launch),
        fixture_test_host_identity(),
        [&authority_now_ns] { return test_time(authority_now_ns); });
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(wire_hello(launch), connection);
    check(open.ok(), "stale-fence WorkerControl fixture opens a worker session");
    if (open.ok()) {
      trainvm::Journal authority_observer(database_path);
      const auto count = authority_observer.event_count();
      check(authority_observer.release_lease(
                launch.concurrency_key, launch.run_id, launch.lease_id,
                launch.fencing_token, test_time(2'300)),
            "stale-fence WorkerControl fixture releases the accepted lease");
      const auto successor = authority_observer.acquire_lease(
          launch.concurrency_key, "successor-run", "successor-lease", test_time(2'301),
          60'000'000'000LL);
      authority_now_ns = 2'400;
      trainvm::v1::WorkerReceipt receipt;
      const grpc::Status stale = service.complete_worker_connection(
          wire_result(connection), connection, receipt);
      check(successor.status == trainvm::LeaseAcquireStatus::acquired &&
                stale.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
                authority_observer.event_count() == count,
            "WorkerControl rejects a result after lease fence takeover without mutation");
      service.prune_retained_launches(test_time(authority_now_ns));
      check(service.resolved_launches_.empty(),
            "released and superseded same-attempt leases release their retained launch bundles");
    }
  }

  {
    const auto database_path = directory / "authority-corruption.db";
    const std::string run_id = "worker-control-authority-corruption-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(database_path);
      trainvm::Controller controller(*compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      (void)controller.begin_acquisition(test_time(4'000));
      launch = controller.prepare_worker_launch(launch_request, test_time(4'100));
      (void)bind_test_worker_launch(controller, launch, 4'150);
    }
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*compiled.plan, launch),
        fixture_test_host_identity(),
        [] { return test_time(4'200); });
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(wire_hello(launch), connection);
    check(open.ok(),
          "authority-corruption WorkerControl fixture opens a worker session");
    sqlite3* tamper = nullptr;
    check(sqlite3_open(database_path.c_str(), &tamper) == SQLITE_OK,
          "authority-corruption test opens the journal database");
    if (open.ok() && tamper != nullptr) {
      trainvm::Journal before(database_path);
      const auto count = before.event_count();
      check(sqlite3_exec(
                tamper,
                "UPDATE compiled_plans SET canonical_plan_json='{broken'",
                nullptr, nullptr, nullptr) == SQLITE_OK,
            "authority-corruption test damages persisted plan JSON");
      trainvm::v1::WorkerReceipt ignored;
      const grpc::Status corrupt = service.complete_worker_connection(
          wire_result(connection), connection, ignored);
      check(corrupt.error_code() == grpc::StatusCode::DATA_LOSS &&
                before.event_count() == count,
            "post-ingress authority JSON corruption maps to DATA_LOSS without mutation");
    }
    if (tamper != nullptr) sqlite3_close(tamper);
  }

  std::filesystem::remove_all(directory);
}

void test_worker_control_grpc_stream() {
  const auto compiled = trainvm::compile_document(load_fixture());
  auto eof_fixture = load_fixture();
  eof_fixture["metadata"]["name"] = "worker-control-clean-eof";
  eof_fixture["spec"]["workspace"]["concurrency_key"] =
      "local-gpu-training-clean-eof";
  const auto eof_compiled = trainvm::compile_document(eof_fixture);
  check(compiled.valid() && eof_compiled.valid(),
        "fixtures required by WorkerControl gRPC stream compile");
  if (!compiled.valid() || !eof_compiled.valid()) return;
  const nlohmann::json submission_identity =
      fixture_adapter_locked_submission(*compiled.plan);
  const nlohmann::json eof_submission_identity =
      fixture_adapter_locked_submission(*eof_compiled.plan);

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-worker-control-grpc-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "worker-control-grpc-run";
  const std::string eof_run_id = "worker-control-grpc-clean-eof-run";
  trainvm::WorkerLaunchTicket launch;
  trainvm::WorkerLaunchTicket eof_launch;
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(submission_identity);
    (void)controller.begin_acquisition(test_time(1'000));
    launch = controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '3'),
         .required_capabilities = {"worker.controls", "worker.metrics"}},
        test_time(1'100));
    (void)bind_test_worker_launch(controller, launch, 1'150);
    trainvm::Controller eof_controller(
        *eof_compiled.plan, journal, eof_run_id);
    eof_controller.create_queued(eof_submission_identity);
    (void)eof_controller.begin_acquisition(test_time(1'000));
    eof_launch = eof_controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '3'),
         .required_capabilities = {"worker.controls", "worker.metrics"}},
        test_time(1'100));
    (void)bind_test_worker_launch(eof_controller, eof_launch, 1'150);
  }

  const auto hello_for = [](const trainvm::WorkerLaunchTicket& ticket) {
    trainvm::v1::WorkerToController message;
    auto* hello = message.mutable_hello();
    hello->set_run_id(ticket.run_id);
    hello->set_node_id(ticket.node_id);
    hello->set_attempt_id(ticket.attempt_id);
    hello->set_launch_nonce(ticket.launch_nonce);
    hello->set_adapter(ticket.adapter);
    hello->set_adapter_version(ticket.adapter_version);
    hello->set_code_fingerprint(ticket.code_fingerprint);
    for (const auto& capability : ticket.required_capabilities) {
      hello->add_capabilities(capability);
    }
    hello->set_last_acked_controller_sequence(0);
    hello->set_concurrency_key(ticket.concurrency_key);
    hello->set_lease_id(ticket.lease_id);
    hello->set_fencing_token(ticket.fencing_token);
    return message;
  };
  const auto result_for = [](const trainvm::v1::WorkerWelcome& welcome) {
    trainvm::v1::WorkerToController message;
    auto* event = message.mutable_event();
    event->set_event_id(welcome.dispatch_id() + ":result");
    event->set_run_id(welcome.run_id());
    event->set_run_revision(welcome.run_revision());
    event->set_plan_revision(welcome.plan_revision());
    event->set_node_id(welcome.node_id());
    event->set_attempt_id(welcome.attempt_id());
    event->set_worker_sequence(1);
    event->set_event_type("worker.completed");
    event->set_event_version(1);
    event->mutable_wall_time()->set_seconds(1);
    event->set_monotonic_time_ns(1);
    event->set_optimizer_step(5'500);
    event->set_canonical_json_payload(
        R"({"reason":"cache_span_complete"})");
    return message;
  };

  trainvm::TrainVMService service(
      database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
      fixture_test_host_launch_registry(*compiled.plan, launch),
      fixture_test_host_identity(),
      [] { return test_time(1'200); });
  prime_test_service_launch(service, launch);
  prime_test_service_launch(service, eof_launch);
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(
      static_cast<trainvm::v1::WorkerControl::Service*>(&service));
  std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
  check(server != nullptr && port > 0,
        "WorkerControl integration fixture starts an in-process gRPC server");
  if (!server || port <= 0) {
    std::filesystem::remove_all(directory);
    return;
  }

  const auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(port),
      grpc::InsecureChannelCredentials());
  const auto stub = trainvm::v1::WorkerControl::NewStub(channel);

  {
    grpc::ClientContext context;
    auto stream = stub->Connect(&context);
    trainvm::v1::WorkerToController heartbeat;
    heartbeat.mutable_heartbeat()->set_worker_sequence(1);
    const bool wrote = stream->Write(heartbeat);
    stream->WritesDone();
    trainvm::v1::ControllerToWorker unexpected;
    const bool received = stream->Read(&unexpected);
    const grpc::Status status = stream->Finish();
    trainvm::Journal observer(database_path);
    check(wrote && !received &&
              status.error_code() == grpc::StatusCode::INVALID_ARGUMENT &&
              observer.events_for_run(run_id).size() == 8U &&
              observer.events_for_run(eof_run_id).size() == 8U,
          "WorkerControl gRPC rejects a non-Hello first message without mutation");
  }

  {
    grpc::ClientContext context;
    auto stream = stub->Connect(&context);
    const bool hello_written = stream->Write(hello_for(eof_launch));
    trainvm::v1::ControllerToWorker welcome;
    const bool welcome_received = stream->Read(&welcome);
    stream->WritesDone();
    trainvm::v1::ControllerToWorker unexpected_receipt;
    const bool receipt_received = stream->Read(&unexpected_receipt);
    const grpc::Status status = stream->Finish();
    trainvm::Journal observer(database_path);
    const auto dispatch = welcome.has_welcome()
                              ? observer.dispatch(welcome.welcome().dispatch_id())
                              : std::nullopt;
    const auto projection = observer.projection(eof_run_id);
    check(hello_written && welcome_received && welcome.has_welcome() &&
              !receipt_received &&
              status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION &&
              dispatch &&
              dispatch->status == trainvm::DispatchStatus::prepared &&
              !dispatch->result_event_id && projection &&
              projection->observed_state == "running" &&
              projection->run_revision == 5U &&
              observer.events_for_run(eof_run_id).size() == 12U,
          "WorkerControl gRPC treats clean EOF after Welcome as incomplete and "
          "leaves its dispatch prepared without a Receipt");
  }

  {
    grpc::ClientContext context;
    auto stream = stub->Connect(&context);
    const bool hello_written = stream->Write(hello_for(eof_launch));
    trainvm::v1::ControllerToWorker welcome;
    const bool welcome_received = stream->Read(&welcome);
    const bool result_written =
        welcome.has_welcome() &&
        stream->Write(result_for(welcome.welcome()));
    stream->WritesDone();
    trainvm::v1::ControllerToWorker receipt;
    const bool receipt_received = stream->Read(&receipt);
    trainvm::v1::ControllerToWorker trailing;
    const bool trailing_received = stream->Read(&trailing);
    const grpc::Status status = stream->Finish();
    trainvm::Journal observer(database_path);
    const auto dispatch = welcome.has_welcome()
                              ? observer.dispatch(welcome.welcome().dispatch_id())
                              : std::nullopt;
    const auto projection = observer.projection(eof_run_id);
    check(hello_written && welcome_received && welcome.has_welcome() &&
              welcome.welcome().disposition() ==
                  trainvm::v1::WorkerWelcome::DISPOSITION_REPLAYED &&
              result_written && receipt_received && receipt.has_receipt() &&
              !trailing_received && status.ok() && dispatch &&
              dispatch->status == trainvm::DispatchStatus::completed &&
              projection && projection->observed_state == "acquiring" &&
              projection->run_revision == 7U &&
              observer.events_for_run(eof_run_id).size() == 16U,
          "WorkerControl gRPC releases the EOF stream claim and completes the "
          "same durable dispatch after reconnect");
  }

  grpc::ClientContext primary_context;
  auto primary = stub->Connect(&primary_context);
  const auto hello = hello_for(launch);
  const bool hello_written = primary->Write(hello);
  trainvm::v1::ControllerToWorker welcome_message;
  const bool welcome_received = primary->Read(&welcome_message);
  check(hello_written && welcome_received && welcome_message.has_welcome() &&
            welcome_message.message_case() ==
                trainvm::v1::ControllerToWorker::kWelcome &&
            welcome_message.welcome().disposition() ==
                trainvm::v1::WorkerWelcome::DISPOSITION_ACCEPTED,
        "WorkerControl gRPC emits Welcome first for an accepted Hello");

  {
    grpc::ClientContext duplicate_context;
    auto duplicate = stub->Connect(&duplicate_context);
    const bool duplicate_written = duplicate->Write(hello);
    duplicate->WritesDone();
    trainvm::v1::ControllerToWorker unexpected;
    const bool duplicate_received = duplicate->Read(&unexpected);
    const grpc::Status duplicate_status = duplicate->Finish();
    check(duplicate_written && !duplicate_received &&
              duplicate_status.error_code() ==
                  grpc::StatusCode::ALREADY_EXISTS,
          "WorkerControl gRPC rejects a duplicate active attempt while its first stream is open");
  }

  bool result_written = false;
  trainvm::v1::ControllerToWorker receipt_message;
  bool receipt_received = false;
  grpc::Status primary_status;
  if (welcome_received && welcome_message.has_welcome()) {
    result_written = primary->Write(result_for(welcome_message.welcome()));
    primary->WritesDone();
    receipt_received = primary->Read(&receipt_message);
    trainvm::v1::ControllerToWorker trailing;
    check(!primary->Read(&trailing),
          "WorkerControl gRPC closes after its single result Receipt");
    primary_status = primary->Finish();
  } else {
    primary_context.TryCancel();
    primary_status = primary->Finish();
  }

  {
    trainvm::Journal observer(database_path);
    const auto dispatch = welcome_message.has_welcome()
                              ? observer.dispatch(
                                    welcome_message.welcome().dispatch_id())
                              : std::nullopt;
    const auto projection = observer.projection(run_id);
    check(result_written && receipt_received && receipt_message.has_receipt() &&
              receipt_message.message_case() ==
                  trainvm::v1::ControllerToWorker::kReceipt &&
              primary_status.ok() &&
              receipt_message.receipt().event_id() ==
                  welcome_message.welcome().dispatch_id() + ":result" &&
              receipt_message.receipt().acknowledged_worker_sequence() == 1U &&
              receipt_message.receipt().committed_run_revision() == 7U &&
              dispatch &&
              dispatch->status == trainvm::DispatchStatus::completed &&
              dispatch->result_event_id ==
                  std::optional<std::string>{
                      receipt_message.receipt().event_id()} &&
              projection && projection->observed_state == "acquiring" &&
              projection->run_revision == 7U &&
              observer.events_for_run(run_id).size() == 16U,
          "WorkerControl gRPC orders Hello, durable Welcome, Event, and durable Receipt");
  }

  server->Shutdown();
  server->Wait();
  std::filesystem::remove_all(directory);
}

void test_typed_managed_resource_release() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "typed resource release fixture compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-managed-release-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "managed-release-run";
  trainvm::Journal journal(database_path);
  trainvm::Controller controller(*compiled.plan, journal, run_id);
  controller.create_queued();
  const auto acquired = controller.begin_acquisition(test_time(1'000));
  const trainvm::WorkerLaunchRequest request{
      .code_fingerprint = "sha256:" + std::string(64U, '5'),
      .required_capabilities = {"worker.controls"},
  };
  const auto launch = controller.prepare_worker_launch(request, test_time(1'100));
  (void)bind_test_worker_launch(controller, launch, 1'150);
  (void)controller.accept_worker_hello(
      {.run_id = launch.run_id,
       .node_id = launch.node_id,
       .attempt_id = launch.attempt_id,
       .launch_nonce = launch.launch_nonce,
       .adapter = launch.adapter,
       .adapter_version = launch.adapter_version,
       .code_fingerprint = launch.code_fingerprint,
       .capabilities = launch.required_capabilities,
       .last_acked_controller_sequence = 0,
       .concurrency_key = launch.concurrency_key,
       .lease_id = launch.lease_id,
       .fencing_token = launch.fencing_token},
      test_time(1'200));
  const auto worker_dispatch = controller.prepare_dispatch(test_time(1'300));
  const trainvm::WorkerSessionIdentity session{
      .run_id = launch.run_id,
      .node_id = launch.node_id,
      .attempt_id = launch.attempt_id,
      .launch_nonce = launch.launch_nonce,
      .concurrency_key = launch.concurrency_key,
      .lease_id = launch.lease_id,
      .fencing_token = launch.fencing_token,
  };
  const trainvm::Event worker_result{
      .event_id = worker_dispatch.dispatch_id + ":result",
      .run_id = run_id,
      .run_revision = 5,
      .plan_revision = 1,
      .node_id = launch.node_id,
      .attempt_id = launch.attempt_id,
      .worker_sequence = 1,
      .event_type = "worker.completed",
      .event_version = 1,
      .wall_time_ns = 1'400,
      .monotonic_time_ns = 1,
      .optimizer_step = 5'500,
      .payload = {{"reason", "training_complete"}},
  };
  const auto& builtin =
      controller.handle_event(worker_result, session, test_time(1'400));
  check(builtin.revision == 6U && builtin.current_node_id == "release_gpu" &&
            journal.projection(run_id)->observed_state == "running",
        "managed worker result enters the resource release builtin");

  const auto before_wrong_builtin = journal.event_count();
  bool wrong_builtin_rejected = false;
  try {
    (void)controller.complete_artifact_validation(
        trainvm::ArtifactValidationOutcome::valid, test_time(1'500));
  } catch (const std::logic_error&) {
    wrong_builtin_rejected = true;
  }
  bool simulation_dispatch_rejected = false;
  try {
    (void)controller.prepare_dispatch();
  } catch (const std::logic_error&) {
    simulation_dispatch_rejected = true;
  }
  const trainvm::Event simulated_release{
      .event_id = run_id + ":simulated-release",
      .run_id = run_id,
      .run_revision = builtin.revision,
      .plan_revision = 1,
      .node_id = builtin.current_node_id,
      .attempt_id = builtin.current_attempt_id,
      .worker_sequence = 1,
      .event_type = "resource.released",
      .event_version = 1,
      .wall_time_ns = 1'500,
      .monotonic_time_ns = 2,
      .optimizer_step = 5'500,
      .payload = nlohmann::json::object(),
  };
  bool simulation_result_rejected = false;
  try {
    (void)controller.handle_event(simulated_release);
  } catch (const std::logic_error&) {
    simulation_result_rejected = true;
  }
  check(simulation_dispatch_rejected && simulation_result_rejected &&
            journal.event_count() == before_wrong_builtin,
        "managed resource release rejects generic simulation hooks without mutation");
  const auto& completed = controller.release_managed_resources(test_time(1'500));
  const auto projection = journal.projection(run_id);
  const auto release_events = journal.events_for_run(run_id);
  const auto release_result = std::ranges::find_if(
      release_events, [](const trainvm::Event& event) {
        return event.event_type == "resource.released";
      });
  const auto release_receipts = std::ranges::count_if(
      release_events, [](const trainvm::Event& event) {
        return event.event_type == "node.dispatch_completed" &&
               event.node_id == "release_gpu";
      });
  trainvm::Controller restarted(*compiled.plan, journal, run_id);
  const auto& recovered = restarted.recover();
  std::string chain_reason;
  check(wrong_builtin_rejected && simulation_dispatch_rejected &&
            simulation_result_rejected &&
            journal.event_count() == before_wrong_builtin + 5U &&
            completed.status == trainvm::ExecutionStatus::completed &&
            completed.revision == 7U && recovered == completed && projection &&
            projection->observed_state == "completed" &&
            projection->current_node_id.empty() &&
            projection->current_attempt_id.empty() && release_receipts == 1 &&
            release_result != release_events.end() &&
            release_result->worker_sequence == 0U &&
            release_result->payload ==
                nlohmann::json{{"concurrency_key", acquired.lease.concurrency_key},
                               {"lease_id", acquired.lease.lease_id},
                               {"fencing_token", acquired.lease.fencing_token}} &&
            !journal.active_lease(acquired.lease.concurrency_key, test_time(1'500)) &&
            journal.verify_chain(&chain_reason),
        "typed resource release atomically releases its fence and commits one terminal receipt");
  sqlite3* tamper = nullptr;
  check(sqlite3_open(database_path.c_str(), &tamper) == SQLITE_OK,
        "release receipt tamper test opens the journal database");
  if (tamper != nullptr) {
    check(sqlite3_exec(
              tamper,
              "UPDATE resource_leases SET released_wall_time_ns=NULL",
              nullptr, nullptr, nullptr) == SQLITE_OK,
          "release receipt tamper test resurrects the mutable lease row");
    check(!journal.active_lease(acquired.lease.concurrency_key, test_time(1'500)) &&
              !journal.renew_lease(
                  acquired.lease.concurrency_key, run_id,
                  acquired.lease.lease_id, acquired.lease.fencing_token,
                  test_time(1'500), 60'000'000'000LL) &&
              !journal.release_lease(
                  acquired.lease.concurrency_key, run_id,
                  acquired.lease.lease_id, acquired.lease.fencing_token,
                  test_time(1'501)),
          "immutable release receipt prevents a mutable lease resurrection");
    bool resurrected_release_recovered = false;
    try {
      trainvm::Controller durable_release(*compiled.plan, journal, run_id);
      resurrected_release_recovered = durable_release.recover() == completed;
    } catch (const std::exception&) {
      resurrected_release_recovered = false;
    }
    check(resurrected_release_recovered,
          "terminal recovery trusts the immutable release receipt over a stale mutable row");
    check(sqlite3_exec(
              tamper,
              "UPDATE resource_leases SET released_wall_time_ns=1500",
              nullptr, nullptr, nullptr) == SQLITE_OK,
          "release receipt tamper test restores the mutable release marker");
    check(sqlite3_exec(tamper, "DELETE FROM resource_lease_releases", nullptr,
                       nullptr, nullptr) == SQLITE_OK,
          "release receipt tamper test removes the unchained lease receipt");
    sqlite3_close(tamper);
  }
  bool missing_release_receipt_rejected = false;
  try {
    trainvm::Controller corrupted(*compiled.plan, journal, run_id);
    (void)corrupted.recover();
  } catch (const std::runtime_error&) {
    missing_release_receipt_rejected = true;
  }
  check(missing_release_receipt_rejected,
        "terminal recovery rejects a resource release without its durable lease receipt");
  std::filesystem::remove_all(directory);
}

void test_concurrent_worker_launch_and_readiness_replay() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by concurrent worker readiness compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-worker-readiness-race-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "worker-readiness-race-run";
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller creator(*compiled.plan, journal, run_id);
    creator.create_queued();
    check(creator.begin_acquisition(test_time(1'000)).status ==
              trainvm::LeaseAcquireStatus::acquired &&
              creator.state().revision == 4U &&
              creator.state().current_node_id == "train_to_boundary",
          "worker race fixture reaches the external-node acquiring boundary");
  }

  trainvm::Journal left_journal(database_path);
  trainvm::Journal right_journal(database_path);
  trainvm::Controller left(*compiled.plan, left_journal, run_id);
  trainvm::Controller right(*compiled.plan, right_journal, run_id);
  check(left.recover().revision == 4U && right.recover().revision == 4U,
        "both worker race controllers recover acquiring revision four");

  const trainvm::WorkerLaunchRequest request{
      .code_fingerprint = "sha256:" + std::string(64U, '6'),
      .required_capabilities = {"worker.metrics", "worker.controls"},
  };
  struct LaunchOutcome {
    std::optional<trainvm::WorkerLaunchTicket> ticket;
    std::string error;
  };
  const auto launch_one = [](trainvm::Controller& controller,
                             const trainvm::WorkerLaunchRequest& launch_request,
                             std::int64_t now_ns, std::shared_future<void> start) {
    start.wait();
    try {
      return LaunchOutcome{
          .ticket = controller.prepare_worker_launch(launch_request, test_time(now_ns)),
          .error = {}};
    } catch (const std::exception& exception) {
      return LaunchOutcome{.ticket = std::nullopt, .error = exception.what()};
    }
  };
  std::promise<void> launch_start;
  const std::shared_future<void> launch_gate = launch_start.get_future().share();
  auto left_launch_future = std::async(std::launch::async, launch_one,
                                       std::ref(left), std::cref(request),
                                       1'100, launch_gate);
  auto right_launch_future = std::async(std::launch::async, launch_one,
                                        std::ref(right), std::cref(request),
                                        1'150, launch_gate);
  launch_start.set_value();
  const LaunchOutcome left_launch = left_launch_future.get();
  const LaunchOutcome right_launch = right_launch_future.get();
  const auto after_launch_events = left_journal.events_for_run(run_id);
  const auto launch_event_count = static_cast<std::size_t>(std::count_if(
      after_launch_events.begin(), after_launch_events.end(),
      [](const trainvm::Event& event) {
        return event.event_type == "worker.launch_requested";
      }));
  if (!left_launch.ticket || !right_launch.ticket) {
    std::cerr << "worker launch race errors: left='" << left_launch.error
              << "' right='" << right_launch.error << "'\n";
  }
  check(left_launch.ticket && right_launch.ticket &&
            *left_launch.ticket == *right_launch.ticket &&
            launch_event_count == 1U && after_launch_events.size() == 7U,
        "concurrent exact worker launches converge on one durable ticket");
  if (!left_launch.ticket || !right_launch.ticket) {
    std::filesystem::remove_all(directory);
    return;
  }
  const trainvm::WorkerLaunchTicket ticket = *left_launch.ticket;
  (void)bind_test_worker_launch(left, ticket, 1'175);
  const auto hello_for = [](const trainvm::WorkerLaunchTicket& launch) {
    return trainvm::WorkerHelloEvidence{
        .run_id = launch.run_id,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .launch_nonce = launch.launch_nonce,
        .adapter = launch.adapter,
        .adapter_version = launch.adapter_version,
        .code_fingerprint = launch.code_fingerprint,
        .capabilities = {"worker.metrics", "worker.controls"},
        .last_acked_controller_sequence = 0,
        .concurrency_key = launch.concurrency_key,
        .lease_id = launch.lease_id,
        .fencing_token = launch.fencing_token,
    };
  };
  struct HelloOutcome {
    std::optional<trainvm::WorkerReadinessResult> result;
    std::string error;
  };
  const auto hello_one = [](trainvm::Controller& controller,
                            trainvm::WorkerHelloEvidence hello,
                            std::int64_t now_ns,
                            std::shared_future<void> start) {
    start.wait();
    try {
      return HelloOutcome{
          .result = controller.accept_worker_hello(std::move(hello), test_time(now_ns)),
          .error = {}};
    } catch (const std::exception& exception) {
      return HelloOutcome{.result = std::nullopt, .error = exception.what()};
    }
  };
  std::promise<void> hello_start;
  const std::shared_future<void> hello_gate = hello_start.get_future().share();
  auto left_hello_future = std::async(std::launch::async, hello_one,
                                      std::ref(left), hello_for(ticket), 1'200,
                                      hello_gate);
  auto right_hello_future = std::async(std::launch::async, hello_one,
                                       std::ref(right), hello_for(ticket), 1'250,
                                       hello_gate);
  hello_start.set_value();
  const HelloOutcome left_hello = left_hello_future.get();
  const HelloOutcome right_hello = right_hello_future.get();
  if (!left_hello.result || !right_hello.result) {
    std::cerr << "worker hello race errors: left='" << left_hello.error
              << "' right='" << right_hello.error << "'\n";
  }
  const bool accepted_and_replayed =
      left_hello.result && right_hello.result &&
      ((left_hello.result->disposition ==
            trainvm::WorkerReadinessDisposition::accepted &&
        right_hello.result->disposition ==
            trainvm::WorkerReadinessDisposition::replayed) ||
       (left_hello.result->disposition ==
            trainvm::WorkerReadinessDisposition::replayed &&
        right_hello.result->disposition ==
            trainvm::WorkerReadinessDisposition::accepted));
  trainvm::Journal observer(database_path);
  const auto readiness_events = observer.events_for_run(run_id);
  const auto count_type = [&](std::string_view event_type) {
    return std::count_if(readiness_events.begin(), readiness_events.end(),
                         [&](const trainvm::Event& event) {
                           return event.event_type == event_type;
                         });
  };
  const auto projection = observer.projection(run_id);
  check(accepted_and_replayed &&
            left_hello.result->launch == ticket &&
            right_hello.result->launch == ticket &&
            count_type("worker.ready") == 1 &&
            count_type("run.observed_state_changed") == 2 &&
            count_type("node.entered") == 1 &&
            readiness_events.size() == 11U && projection &&
            projection->desired_state == "running" &&
            projection->observed_state == "running" &&
            projection->run_revision == 5U &&
            projection->current_node_id == "train_to_boundary" &&
            projection->current_attempt_id == "train_to_boundary@1" &&
            left_hello.result->launch.fencing_token ==
                right_hello.result->launch.fencing_token &&
            left_hello.result->launch.lease_id ==
                right_hello.result->launch.lease_id,
        "concurrent exact worker hellos commit one readiness triplet on one fence");
  std::filesystem::remove_all(directory);
}

void test_concurrent_fenced_result_content_conflict() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by concurrent fenced result compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-fenced-result-race-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "fenced-result-race-run";
  trainvm::WorkerLaunchTicket launch;
  trainvm::Dispatch dispatch;
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller creator(*compiled.plan, journal, run_id);
    creator.create_queued();
    (void)creator.begin_acquisition(test_time(1'000));
    launch = creator.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '8'),
         .required_capabilities = {"worker.controls", "worker.metrics"}},
        test_time(1'100));
    (void)bind_test_worker_launch(creator, launch, 1'150);
    (void)creator.accept_worker_hello(
        {.run_id = launch.run_id,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .launch_nonce = launch.launch_nonce,
         .adapter = launch.adapter,
         .adapter_version = launch.adapter_version,
         .code_fingerprint = launch.code_fingerprint,
         .capabilities = launch.required_capabilities,
         .last_acked_controller_sequence = 0,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(1'200));
    dispatch = creator.prepare_dispatch(test_time(1'300));
    check(journal.event_count() == 12U &&
              dispatch.status == trainvm::DispatchStatus::prepared,
          "result race fixture prepares one fenced external dispatch");
  }

  const trainvm::WorkerSessionIdentity session{
      .run_id = launch.run_id,
      .node_id = launch.node_id,
      .attempt_id = launch.attempt_id,
      .launch_nonce = launch.launch_nonce,
      .concurrency_key = launch.concurrency_key,
      .lease_id = launch.lease_id,
      .fencing_token = launch.fencing_token,
  };
  const std::string result_id = dispatch.dispatch_id + ":adversarial-result";
  const auto result_for = [&](std::string candidate) {
    return trainvm::Event{
        .event_id = result_id,
        .run_id = run_id,
        .run_revision = 5,
        .plan_revision = 1,
        .node_id = launch.node_id,
        .attempt_id = launch.attempt_id,
        .worker_sequence = 1,
        .event_type = "worker.completed",
        .event_version = 1,
        .wall_time_ns = 1'400,
        .monotonic_time_ns = 1,
        .optimizer_step = 5'500,
        .payload = {{"reason", "cache_span_complete"},
                    {"candidate", std::move(candidate)}},
    };
  };
  const trainvm::Event left_event = result_for("left");
  const trainvm::Event right_event = result_for("right");

  trainvm::Journal left_journal(database_path);
  trainvm::Journal right_journal(database_path);
  trainvm::Controller left(*compiled.plan, left_journal, run_id);
  trainvm::Controller right(*compiled.plan, right_journal, run_id);
  check(left.recover().revision == 5U && right.recover().revision == 5U,
        "both result race controllers recover the same prepared attempt");
  struct Outcome {
    bool succeeded{};
    std::string error;
  };
  const auto complete = [](trainvm::Controller& controller,
                           trainvm::Event event,
                           const trainvm::WorkerSessionIdentity& worker_session,
                           std::shared_future<void> start) {
    start.wait();
    try {
      (void)controller.handle_event(event, worker_session, test_time(1'400));
      return Outcome{.succeeded = true, .error = {}};
    } catch (const std::exception& exception) {
      return Outcome{.succeeded = false, .error = exception.what()};
    }
  };
  std::promise<void> start;
  const std::shared_future<void> gate = start.get_future().share();
  auto left_future = std::async(std::launch::async, complete, std::ref(left),
                                left_event, std::cref(session), gate);
  auto right_future = std::async(std::launch::async, complete, std::ref(right),
                                 right_event, std::cref(session), gate);
  start.set_value();
  const Outcome left_outcome = left_future.get();
  const Outcome right_outcome = right_future.get();
  if (left_outcome.succeeded == right_outcome.succeeded) {
    std::cerr << "fenced result race outcomes: left_success="
              << left_outcome.succeeded << " left_error='" << left_outcome.error
              << "' right_success=" << right_outcome.succeeded
              << " right_error='" << right_outcome.error << "'\n";
  }

  trainvm::Journal observer(database_path);
  const auto durable_result = observer.event(result_id);
  const auto durable_transition = observer.event(result_id + ":transition");
  const auto durable_reacquiring = observer.event(result_id + ":acquiring");
  const auto durable_receipt = observer.event(dispatch.dispatch_id + ":completed");
  const auto receipt = observer.dispatch(dispatch.dispatch_id);
  const auto projection = observer.projection(run_id);
  const auto events = observer.events_for_run(run_id);
  const std::string winner = left_outcome.succeeded ? "left" : "right";
  const bool exactly_one_succeeded =
      left_outcome.succeeded != right_outcome.succeeded;
  const bool loser_rejected = left_outcome.succeeded
                                  ? !right_outcome.error.empty()
                                  : !left_outcome.error.empty();
  check(exactly_one_succeeded && loser_rejected && durable_result &&
            durable_result->payload.value("candidate", std::string{}) == winner &&
            durable_transition && durable_reacquiring && durable_receipt &&
            receipt && receipt->status == trainvm::DispatchStatus::completed &&
            receipt->result_event_id ==
                std::optional<std::string>{result_id} &&
            events.size() == 16U && projection &&
            projection->observed_state == "acquiring" &&
            projection->run_revision == 7U &&
            projection->current_node_id.empty() &&
            projection->current_attempt_id.empty(),
        "conflicting concurrent result retries commit only the winner's transition and receipt");
  std::string chain_reason;
  check(observer.verify_chain(&chain_reason),
        "conflicting result race retains one valid journal chain");
  std::filesystem::remove_all(directory);
}

void test_concurrent_queue_acquisition_replay() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by concurrent acquisition compiles");
  if (!compiled.valid()) return;
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-acquisition-race-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller creator(*compiled.plan, journal, "acquisition-race-run");
    creator.create_queued();
  }

  trainvm::Journal left_journal(database_path);
  trainvm::Journal right_journal(database_path);
  trainvm::Controller left(*compiled.plan, left_journal, "acquisition-race-run");
  trainvm::Controller right(*compiled.plan, right_journal, "acquisition-race-run");
  check(left.recover().revision == 1U && right.recover().revision == 1U,
        "both racing controllers recover the same queued revision");

  struct Outcome {
    bool succeeded{};
    trainvm::LeaseAcquireStatus status{};
    std::uint64_t fencing_token{};
    std::string error;
  };
  const auto acquire = [](trainvm::Controller& controller) {
    try {
      const auto result = controller.begin_acquisition(test_time(1'000));
      return Outcome{.succeeded = true,
                     .status = result.status,
                     .fencing_token = result.lease.fencing_token,
                     .error = {}};
    } catch (const std::exception& exception) {
      return Outcome{.error = exception.what()};
    }
  };
  auto left_future = std::async(std::launch::async, acquire, std::ref(left));
  auto right_future = std::async(std::launch::async, acquire, std::ref(right));
  const Outcome left_result = left_future.get();
  const Outcome right_result = right_future.get();
  const auto events = left_journal.events_for_run("acquisition-race-run");
  const auto projection = left_journal.projection("acquisition-race-run");
  if (!left_result.succeeded || !right_result.succeeded) {
    std::cerr << "acquisition race errors: left='" << left_result.error
              << "' right='" << right_result.error << "'\n";
  }
  check(left_result.succeeded && right_result.succeeded &&
            left_result.status != trainvm::LeaseAcquireStatus::busy &&
            right_result.status != trainvm::LeaseAcquireStatus::busy &&
            left_result.fencing_token == right_result.fencing_token && projection &&
            projection->desired_state == "running" &&
            projection->observed_state == "acquiring" &&
            projection->run_revision == 4U && events.size() == 6U,
        "concurrent exact acquisitions converge on one fence and admission transition");
  std::filesystem::remove_all(directory);
}

void test_acquiring_recovery_ignores_mutable_lease_lifecycle() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by lease lifecycle recovery compiles");
  if (!compiled.valid()) return;
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-acquisition-lease-lifecycle-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  trainvm::Journal journal(database_path);
  trainvm::Controller controller(*compiled.plan, journal, "acquisition-lifecycle-run");
  controller.create_queued();
  const auto acquired = controller.begin_acquisition(test_time(1'000));
  check(acquired.status == trainvm::LeaseAcquireStatus::acquired,
        "lease lifecycle recovery test acquires its initial fence");
  check(journal.renew_lease(acquired.lease.concurrency_key, acquired.lease.owner_run_id,
                            acquired.lease.lease_id, acquired.lease.fencing_token,
                            test_time(2'000), 60'000'000'000LL),
        "lease lifecycle recovery test renews the mutable lease row");
  bool renewed_recovered = false;
  try {
    trainvm::Controller renewed(*compiled.plan, journal, "acquisition-lifecycle-run");
    renewed_recovered = renewed.recover().revision == 4U;
  } catch (const std::exception&) {
  }
  check(renewed_recovered,
        "lease renewal does not invalidate immutable acquisition history");
  check(journal.release_lease(acquired.lease.concurrency_key, acquired.lease.owner_run_id,
                              acquired.lease.lease_id, acquired.lease.fencing_token, test_time(3'000)),
        "lease lifecycle recovery test releases the mutable lease row");
  bool released_recovered = false;
  try {
    trainvm::Controller released(*compiled.plan, journal, "acquisition-lifecycle-run");
    released_recovered = released.recover().revision == 4U;
  } catch (const std::exception&) {
  }
  check(released_recovered,
        "lease release does not invalidate immutable acquisition history");
  const auto reacquired = journal.acquire_lease(
      acquired.lease.concurrency_key, acquired.lease.owner_run_id,
      acquired.lease.lease_id, test_time(4'000), 60'000'000'000LL);
  check(reacquired.status == trainvm::LeaseAcquireStatus::acquired &&
            reacquired.lease.fencing_token > acquired.lease.fencing_token,
        "reacquiring a released textual identity advances its fence");
  bool replacement_rejected = false;
  try {
    trainvm::Controller replacement(*compiled.plan, journal,
                                    "acquisition-lifecycle-run");
    (void)replacement.begin_acquisition(test_time(5'000));
  } catch (const std::runtime_error&) {
    replacement_rejected = true;
  }
  check(replacement_rejected,
        "acquiring retry rejects a newer fence hidden behind the same textual identity");
  std::filesystem::remove_all(directory);
}

void test_acquiring_rejects_fabricated_running_transition() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by fabricated readiness test compiles");
  if (!compiled.valid()) return;
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-fabricated-readiness-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  trainvm::Journal journal(database_path);
  const std::string run_id = "fabricated-readiness-run";
  trainvm::Controller controller(*compiled.plan, journal, run_id);
  controller.create_queued();
  (void)controller.begin_acquisition(test_time(1'000));
  const trainvm::ExecutionState initial = trainvm::start_execution(*compiled.plan, run_id);
  const trainvm::Node& node =
      compiled.plan->experiment.spec.workflow.nodes.at(initial.current_node_id);
  trainvm::JournalTestAccess::append_batch(journal, {
      trainvm::Event{
          .event_id = run_id + ":fabricated-running",
          .run_id = run_id,
          .run_revision = 4,
          .plan_revision = 1,
          .node_id = "",
          .attempt_id = "",
          .worker_sequence = 0,
          .event_type = "run.observed_state_changed",
          .event_version = 1,
          .wall_time_ns = 2'000,
          .monotonic_time_ns = 0,
          .optimizer_step = std::nullopt,
          .payload = {{"state", "running"}},
      },
      trainvm::Event{
          .event_id = run_id + ":fabricated-node-entry",
          .run_id = run_id,
          .run_revision = 4,
          .plan_revision = 1,
          .node_id = initial.current_node_id,
          .attempt_id = initial.current_attempt_id,
          .worker_sequence = 0,
          .event_type = "node.entered",
          .event_version = 1,
          .wall_time_ns = 2'000,
          .monotonic_time_ns = 0,
          .optimizer_step = std::nullopt,
          .payload = {{"component", node.invoke.component},
                      {"operation", node.invoke.operation}},
      },
  });
  std::string chain_reason;
  check(journal.verify_chain(&chain_reason),
        "fabricated readiness fixture retains a valid journal chain");
  bool rejected = false;
  try {
    trainvm::Controller restarted(*compiled.plan, journal, run_id);
    (void)restarted.recover();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected,
        "acquiring recovery rejects running and node entry without readiness evidence");
  std::filesystem::remove_all(directory);
}

void test_host_launch_registry_contract() {
  check(trainvm::reflected_field_names<trainvm::HostLaunchProfile>() ==
            std::vector<std::string>({"key", "code_fingerprint",
                                      "executable_path",
                                      "executable_fingerprint", "code_path",
                                      "public_arguments",
                                      "working_directory"}) &&
            trainvm::reflected_field_names<
                trainvm::HostLaunchRegistryDocument>() ==
                std::vector<std::string>({"api_version", "trusted_roots",
                                          "profiles"}),
        "host launch registry types expose their complete reflected schema");

  const auto key = [](std::string adapter, std::string version,
                      trainvm::ComponentRuntime runtime,
                      std::string operation, std::string contract) {
    return trainvm::AdapterKey{
        .adapter = std::move(adapter),
        .version = std::move(version),
        .runtime = runtime,
        .operation = std::move(operation),
        .contract = std::move(contract),
    };
  };
  const std::string python_code = "sha256:" + std::string(64, 'a');
  const std::string python_executable = "sha256:" + std::string(64, 'b');
  const std::string native_code = "sha256:" + std::string(64, 'c');
  const trainvm::HostLaunchProfile python_profile{
      .key = key("rwkv-lab.mageflow", "1.0.0",
                 trainvm::ComponentRuntime::python_worker, "train",
                 "rwkv_lab.mageflow.v1.Train"),
      .code_fingerprint = python_code,
      .executable_path = "/opt/trainvm/python/bin/python3",
      .executable_fingerprint = python_executable,
      .code_path = "/opt/trainvm/adapters/mageflow/worker.py",
      .public_arguments = {"-I", "/opt/trainvm/adapters/mageflow/worker.py"},
      .working_directory = "/srv/trainvm/runs/run-1",
  };
  const trainvm::HostLaunchProfile native_profile{
      .key = key("example.native", "2.0.0",
                 trainvm::ComponentRuntime::native_worker, "execute",
                 "example.native.v1.Execute"),
      .code_fingerprint = native_code,
      .executable_path = "/usr/libexec/trainvm/native-worker",
      .executable_fingerprint = native_code,
      .code_path = std::nullopt,
      .public_arguments = {"--worker"},
      .working_directory = "/srv/trainvm/runs/run-1",
  };
  const trainvm::HostLaunchRegistryDocument document{
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {"/usr/libexec/trainvm", "/srv/trainvm",
                        "/opt/trainvm"},
      .profiles = {native_profile, python_profile},
  };

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-host-launch-registry-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto set_owner_only = [](const std::filesystem::path& path) {
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read |
            std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace);
  };
  const auto write_text = [&](const std::filesystem::path& path,
                              std::string_view text) {
    {
      std::ofstream output(path);
      output << text;
    }
    set_owner_only(path);
  };
  const auto write_document = [&](const std::filesystem::path& path,
                                  const nlohmann::json& value) {
    write_text(path, value.dump(2) + "\n");
  };

  const auto registry_path = directory / "host-launches.json";
  const nlohmann::json encoded = trainvm::encode_json(document);
  write_document(registry_path, encoded);
  const trainvm::HostLaunchRegistry loaded =
      trainvm::HostLaunchRegistry::load_file(registry_path);
  const auto& resolved_python =
      loaded.resolve(python_profile.key, python_code);
  const auto& resolved_native =
      loaded.resolve(native_profile.key, native_code);
  auto canonical_document = document;
  std::ranges::sort(canonical_document.trusted_roots);
  std::ranges::sort(
      canonical_document.profiles, {},
      [](const trainvm::HostLaunchProfile& profile) -> const trainvm::AdapterKey& {
        return profile.key;
      });
  const std::string expected_registry_digest =
      "sha256:" +
      trainvm::sha256_hex(trainvm::encode_json(canonical_document).dump());
  const std::string expected_python_digest =
      "sha256:" + trainvm::sha256_hex(
                       nlohmann::json{
                           {"api_version",
                            "trainvm.host-launch-profile/v1"},
                           {"profile", trainvm::encode_json(python_profile)},
                       }
                           .dump());
  check(resolved_python == python_profile &&
            resolved_native == native_profile &&
            loaded.trusted_roots() ==
                std::vector<std::string>({"/opt/trainvm", "/srv/trainvm",
                                          "/usr/libexec/trainvm"}) &&
            loaded.registry_digest() == expected_registry_digest &&
            loaded.profile_digest(python_profile.key, python_code) ==
                expected_python_digest,
        "host launch loader reflection-decodes both runtimes and canonicalizes trusted-root and profile order");

  auto reordered = document;
  std::ranges::reverse(reordered.trusted_roots);
  std::ranges::reverse(reordered.profiles);
  const auto reordered_path = directory / "host-launches-reordered.json";
  write_document(reordered_path, trainvm::encode_json(reordered));
  const trainvm::HostLaunchRegistry reordered_registry =
      trainvm::HostLaunchRegistry::load_file(reordered_path);
  check(reordered_registry.trusted_roots() == loaded.trusted_roots() &&
            reordered_registry.registry_digest() ==
                loaded.registry_digest() &&
            reordered_registry.profile_digest(python_profile.key,
                                                python_code) ==
                loaded.profile_digest(python_profile.key, python_code) &&
            reordered_registry.resolve(python_profile.key, python_code) ==
                resolved_python &&
            reordered_registry.resolve(native_profile.key, native_code) ==
                resolved_native,
        "host launch registry semantics are invariant to document collection order");
  auto changed_document = document;
  changed_document.profiles.at(1).public_arguments.push_back("--changed");
  const trainvm::HostLaunchRegistry changed_registry(
      std::move(changed_document));
  check(changed_registry.registry_digest() != loaded.registry_digest() &&
            changed_registry.profile_digest(python_profile.key, python_code) !=
                loaded.profile_digest(python_profile.key, python_code),
        "host launch registry and profile digests bind launch semantics");

  const auto disabled_path = directory / "host-launches-disabled.json";
  write_document(
      disabled_path,
      trainvm::encode_json(trainvm::HostLaunchRegistryDocument{
          .api_version = "trainvm.host-launches/v1",
          .trusted_roots = {},
          .profiles = {},
      }));
  const trainvm::HostLaunchRegistry disabled =
      trainvm::HostLaunchRegistry::load_file(disabled_path);
  bool disabled_resolve_rejected = false;
  try {
    (void)disabled.resolve(python_profile.key, python_code);
  } catch (const trainvm::HostLaunchResolutionError&) {
    disabled_resolve_rejected = true;
  }
  check(disabled.trusted_roots().empty() && disabled_resolve_rejected,
        "empty host launch collections form a valid launch-disabled registry");

  bool fingerprint_mismatch_rejected = false;
  try {
    (void)loaded.resolve(python_profile.key,
                         "sha256:" + std::string(64, 'd'));
  } catch (const trainvm::HostLaunchResolutionError&) {
    fingerprint_mismatch_rejected = true;
  }
  auto absent_key = python_profile.key;
  absent_key.contract = "rwkv_lab.mageflow.v2.Train";
  bool absent_key_rejected = false;
  try {
    (void)loaded.resolve(absent_key, python_code);
  } catch (const trainvm::HostLaunchResolutionError&) {
    absent_key_rejected = true;
  }
  check(fingerprint_mismatch_rejected && absent_key_rejected,
        "host launch resolution requires an exact adapter key and code fingerprint");

  std::size_t rejection_index = 0;
  const auto rejects = [&](nlohmann::json candidate) {
    const auto path = directory /
        ("rejected-" + std::to_string(rejection_index++) + ".json");
    write_document(path, candidate);
    try {
      (void)trainvm::HostLaunchRegistry::load_file(path);
      return false;
    } catch (const std::invalid_argument&) {
      return true;
    }
  };

  auto unknown = encoded;
  unknown["profiles"].at(0)["unknown"] = true;
  auto future = encoded;
  future["api_version"] = "trainvm.host-launches/v2";
  auto duplicate_profile = encoded;
  duplicate_profile["profiles"].push_back(duplicate_profile["profiles"].at(0));
  auto relative_root = encoded;
  relative_root["trusted_roots"].at(0) = "usr/libexec/trainvm";
  auto noncanonical_root = encoded;
  noncanonical_root["trusted_roots"].at(0) = "/usr/libexec/../libexec/trainvm";
  auto overlapping_roots = encoded;
  overlapping_roots["trusted_roots"].push_back("/opt/trainvm/python");
  auto executable_escape = encoded;
  executable_escape["profiles"].at(0)["executable_path"] = "/bin/worker";
  auto code_escape = encoded;
  code_escape["profiles"].at(1)["code_path"] = "/tmp/worker.py";
  auto working_directory_escape = encoded;
  working_directory_escape["profiles"].at(0)["working_directory"] =
      "/tmp/run-1";
  auto builtin_runtime = encoded;
  builtin_runtime["profiles"].at(0)["key"]["runtime"] = "builtin";
  auto external_runtime = encoded;
  external_runtime["profiles"].at(0)["key"]["runtime"] =
      "external_worker";
  auto python_without_code = encoded;
  python_without_code["profiles"].at(1).erase("code_path");
  auto native_with_code = encoded;
  native_with_code["profiles"].at(0)["code_path"] =
      "/usr/libexec/trainvm/native-worker";
  auto native_fingerprint_mismatch = encoded;
  native_fingerprint_mismatch["profiles"].at(0)["code_fingerprint"] =
      "sha256:" + std::string(64, 'd');
  auto invalid_fingerprint = encoded;
  invalid_fingerprint["profiles"].at(1)["executable_fingerprint"] =
      "sha256:" + std::string(64, 'G');
  auto too_many_arguments = encoded;
  too_many_arguments["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>(257U, "x");
  auto oversized_argument = encoded;
  oversized_argument["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>{std::string(4'097U, 'x')};
  auto embedded_nul = encoded;
  embedded_nul["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>{std::string("left\0right", 10U)};
  auto secret_argument = encoded;
  secret_argument["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>{"--token=secret://trainer-key"};
  auto dollar_template_argument = encoded;
  dollar_template_argument["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>{"${run_id}"};
  auto brace_template_argument = encoded;
  brace_template_argument["profiles"].at(0)["public_arguments"] =
      std::vector<std::string>{"{{run_id}}"};
  auto empty_roots = encoded;
  empty_roots["trusted_roots"] = nlohmann::json::array();
  check(rejects(unknown) && rejects(future) && rejects(duplicate_profile) &&
            rejects(relative_root) && rejects(noncanonical_root) &&
            rejects(overlapping_roots) && rejects(executable_escape) &&
            rejects(code_escape) && rejects(working_directory_escape) &&
            rejects(builtin_runtime) && rejects(external_runtime) &&
            rejects(python_without_code) && rejects(native_with_code) &&
            rejects(native_fingerprint_mismatch) &&
            rejects(invalid_fingerprint) && rejects(too_many_arguments) &&
            rejects(oversized_argument) && rejects(embedded_nul) &&
            rejects(secret_argument) && rejects(dollar_template_argument) &&
            rejects(brace_template_argument) &&
            rejects(empty_roots),
        "host launch registry rejects schema, duplicate, path-containment, runtime, fingerprint, and size violations");

  const auto duplicate_key_path = directory / "duplicate-key.json";
  write_text(duplicate_key_path,
             R"({"api_version":"trainvm.host-launches/v1","api_version":"trainvm.host-launches/v1","trusted_roots":["/opt/trainvm"],"profiles":[]})");
  bool duplicate_key_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(duplicate_key_path);
  } catch (const std::invalid_argument&) {
    duplicate_key_rejected = true;
  }
  const auto nested_duplicate_key_path =
      directory / "nested-duplicate-key.json";
  std::string nested_profile = encoded["profiles"].at(0).dump();
  nested_profile.insert(
      1U, "\"code_fingerprint\":\"" + native_code + "\",");
  write_text(nested_duplicate_key_path,
             "{\"api_version\":\"trainvm.host-launches/v1\","
             "\"trusted_roots\":" + encoded["trusted_roots"].dump() +
             ",\"profiles\":[" + nested_profile + "]}");
  bool nested_duplicate_key_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(
        nested_duplicate_key_path);
  } catch (const std::invalid_argument&) {
    nested_duplicate_key_rejected = true;
  }

  const auto malformed_path = directory / "malformed.json";
  write_text(malformed_path, "{\"api_version\":");
  bool malformed_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(malformed_path);
  } catch (const std::invalid_argument&) {
    malformed_rejected = true;
  }

  const auto symlink_path = directory / "host-launches-symlink.json";
  std::filesystem::create_symlink(registry_path, symlink_path);
  bool symlink_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(symlink_path);
  } catch (const std::invalid_argument&) {
    symlink_rejected = true;
  }

  const auto writable_path = directory / "group-writable.json";
  write_document(writable_path, encoded);
  std::filesystem::permissions(writable_path,
                               std::filesystem::perms::group_write,
                               std::filesystem::perm_options::add);
  bool writable_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(writable_path);
  } catch (const std::invalid_argument&) {
    writable_rejected = true;
  }

  const auto empty_path = directory / "empty.json";
  write_text(empty_path, "");
  bool empty_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(empty_path);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }
  const auto oversized_path = directory / "oversized.json";
  write_text(oversized_path, std::string((1U << 20U) + 1U, ' '));
  bool oversized_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(oversized_path);
  } catch (const std::invalid_argument&) {
    oversized_rejected = true;
  }
  bool directory_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file(directory);
  } catch (const std::invalid_argument&) {
    directory_rejected = true;
  }
  bool relative_file_rejected = false;
  try {
    (void)trainvm::HostLaunchRegistry::load_file("host-launches.json");
  } catch (const std::invalid_argument&) {
    relative_file_rejected = true;
  }
  check(duplicate_key_rejected && nested_duplicate_key_rejected &&
            malformed_rejected && symlink_rejected && writable_rejected &&
            empty_rejected && oversized_rejected && directory_rejected &&
            relative_file_rejected,
        "host launch file loading rejects duplicate keys at every depth, malformed JSON, symlinks, unsafe modes, non-regular or unbounded files, and relative authority paths");

  std::filesystem::remove_all(directory);
}

void test_host_launch_resolution_and_binding() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "host launch binding fixture compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-host-launch-resolution-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory / "work");
  std::filesystem::permissions(
      directory, std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  std::filesystem::permissions(
      directory / "work", std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  const auto executable = directory / "python";
  std::filesystem::copy_file("/proc/self/exe", executable);
  std::filesystem::permissions(
      executable,
      std::filesystem::perms::owner_read |
          std::filesystem::perms::owner_exec,
      std::filesystem::perm_options::replace);
  const auto code = directory / "worker.pyz";
  {
    std::ofstream output(code, std::ios::binary);
    output << "PK\003\004immutable-test-zipapp";
  }
  std::filesystem::permissions(
      code, std::filesystem::perms::owner_read,
      std::filesystem::perm_options::replace);
  const auto file_digest = [](const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    return "sha256:" + trainvm::sha256_hex(bytes);
  };
  const std::string executable_digest = file_digest(executable);
  const std::string code_digest = file_digest(code);
  const trainvm::AdapterKey key{
      .adapter = "rwkv-lab.mageflow",
      .version = "1.0.0",
      .runtime = trainvm::ComponentRuntime::python_worker,
      .operation = "train",
      .contract = "rwkv_lab.mageflow.v1.Train",
  };
  const trainvm::HostLaunchProfile profile{
      .key = key,
      .code_fingerprint = code_digest,
      .executable_path = executable.string(),
      .executable_fingerprint = executable_digest,
      .code_path = code.string(),
      .public_arguments = {"-I", "worker.pyz"},
      .working_directory = (directory / "work").string(),
  };
  const trainvm::HostLaunchRegistry registry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {directory.string()},
      .profiles = {profile},
  });
  const trainvm::HostIdentity host{
      .host_id = "sha256:" + std::string(64U, '1'),
      .boot_id = "11111111-1111-1111-1111-111111111111",
  };

  const auto database = directory / "journal.db";
  const std::string run_id = "host-launch-binding-run";
  trainvm::Journal journal(database);
  trainvm::Controller controller(*compiled.plan, journal, run_id);
  controller.create_queued();
  const auto acquisition = controller.begin_acquisition(test_time(1'000));
  const auto ticket = controller.prepare_worker_launch(
      {.code_fingerprint = code_digest,
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      test_time(1'100));
  trainvm::HostLaunchResolver resolver(registry, host);
  auto first = resolver.resolve(ticket, key);
  auto second = resolver.resolve(ticket, key);
  check(first.spec() == second.spec() &&
            first.spec().spec_digest.starts_with("sha256:") &&
            first.spec().identity.host_registry_digest ==
                registry.registry_digest() &&
            first.spec().identity.host_profile_digest ==
                registry.profile_digest(key, code_digest),
        "repeated host resolution produces one deterministic versioned binding");

  const int executable_fd = first.duplicate_executable_fd();
  const auto code_fd = first.duplicate_code_fd();
  const int work_fd = first.duplicate_working_directory_fd();
  const int executable_seals = ::fcntl(executable_fd, F_GET_SEALS);
  const int code_seals = code_fd ? ::fcntl(*code_fd, F_GET_SEALS) : -1;
  errno = 0;
  const bool write_rejected = ::write(executable_fd, "x", 1U) < 0 &&
                              errno == EPERM;
  errno = 0;
  const bool chmod_rejected = ::fchmod(executable_fd, 0400) < 0 &&
                              errno == EPERM;
  check(executable_seals >= 0 && code_seals >= 0 &&
            (executable_seals &
             (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_EXEC |
              F_SEAL_SEAL)) ==
                (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_EXEC |
                 F_SEAL_SEAL) &&
            (code_seals &
             (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL)) ==
                (F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) &&
            write_rejected && chmod_rejected,
        "resolved payloads are immutable sealed descriptors with executable mode sealed");
  (void)::close(executable_fd);
  if (code_fd) (void)::close(*code_fd);
  (void)::close(work_fd);

  const nlohmann::json public_manifest =
      trainvm::resolved_launch_spec_json(first.spec());
  const auto decoded = trainvm::resolved_launch_spec_from_json(public_manifest);
  const std::string manifest_text = public_manifest.dump();
  check(decoded == first.spec() &&
            manifest_text.find("authorization_token") == std::string::npos &&
            manifest_text.find("process_instance") == std::string::npos &&
            manifest_text.find("secret://") == std::string::npos,
        "resolved binding round-trips canonically without process credentials or secrets");
  auto forged = first.spec();
  forged.identity.api_version = "trainvm.resolved-launch/v2";
  forged.spec_digest =
      "sha256:" + trainvm::sha256_hex(
                       trainvm::resolved_launch_identity_json(
                           forged.identity)
                           .dump());
  bool forged_rejected = false;
  try {
    (void)trainvm::resolved_launch_spec_from_json(
        trainvm::resolved_launch_spec_json(forged));
  } catch (const std::invalid_argument&) {
    forged_rejected = true;
  }
  auto moved = std::move(second);
  auto move_assigned = resolver.resolve(ticket, key);
  move_assigned = std::move(moved);
  const int moved_fd = move_assigned.duplicate_executable_fd();
  check(forged_rejected && moved_fd >= 0,
        "self-hashed malformed bindings fail semantics and move-only FD ownership remains valid");
  if (moved_fd >= 0) (void)::close(moved_fd);

  sqlite3* raw_database = nullptr;
  check(sqlite3_open(database.c_str(), &raw_database) == SQLITE_OK,
        "legacy bind quarantine test opens the active journal");
  if (raw_database != nullptr) {
    check(sqlite3_exec(raw_database, R"sql(
      UPDATE resource_leases
      SET clock_domain='legacy-wall/v1', boot_id=NULL,
          acquired_boottime_ns=NULL, expires_boottime_ns=NULL
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "legacy bind quarantine test removes boot-scoped lease evidence");
    sqlite3_close(raw_database);
    raw_database = nullptr;
  }
  bool legacy_bind_rejected = false;
  try {
    (void)controller.bind_worker_launch(first, registry, host,
                                        test_time(1'150));
  } catch (const trainvm::OperationPreconditionError&) {
    legacy_bind_rejected = true;
  }
  check(legacy_bind_rejected,
        "legacy-wall lease rows cannot authorize a new host launch binding");
  check(sqlite3_open(database.c_str(), &raw_database) == SQLITE_OK,
        "legacy bind quarantine test reopens the active journal");
  if (raw_database != nullptr) {
    sqlite3_stmt* restore = nullptr;
    check(sqlite3_prepare_v2(raw_database, R"sql(
      UPDATE resource_leases
      SET clock_domain='boottime/v1', boot_id=?,
          acquired_boottime_ns=?, expires_boottime_ns=?
      WHERE concurrency_key=?
    )sql", -1, &restore, nullptr) == SQLITE_OK,
          "legacy bind quarantine test prepares typed lease restoration");
    if (restore != nullptr) {
      sqlite3_bind_text(restore, 1, acquisition.lease.boot_id.c_str(), -1,
                        SQLITE_TRANSIENT);
      sqlite3_bind_int64(restore, 2,
                         acquisition.lease.acquired_boottime_ns);
      sqlite3_bind_int64(restore, 3,
                         acquisition.lease.expires_boottime_ns);
      sqlite3_bind_text(restore, 4,
                        acquisition.lease.concurrency_key.c_str(), -1,
                        SQLITE_TRANSIENT);
      check(sqlite3_step(restore) == SQLITE_DONE,
            "legacy bind quarantine test restores boot-scoped lease evidence");
    }
    sqlite3_finalize(restore);
    sqlite3_close(raw_database);
  }

  const auto before_binding = journal.event_count();
  const auto bound =
      controller.bind_worker_launch(first, registry, host, test_time(1'200));
  const auto replayed =
      controller.bind_worker_launch(first, registry, host, test_time(1'250));
  const auto historical_replay = controller.bind_worker_launch(
      first, registry, host,
      test_time(acquisition.lease.expires_boottime_ns + 1));
  trainvm::Controller restarted(*compiled.plan, journal, run_id);
  const auto& recovered = restarted.recover();
  check(bound == first.spec() && replayed == bound &&
            historical_replay == bound &&
            journal.event_count() == before_binding + 1U &&
            journal.launch_binding(first.spec().identity.launch_event_id) ==
                std::optional<trainvm::ResolvedLaunchSpec>{bound} &&
            recovered.revision == controller.state().revision,
        "opaque host binding commits once, replays exactly, and survives controller recovery");

  auto wrong_host = host;
  wrong_host.boot_id = "22222222-2222-2222-2222-222222222222";
  const auto before_rejections = journal.event_count();
  bool wrong_host_rejected = false;
  try {
    (void)controller.bind_worker_launch(first, registry, wrong_host,
                                        test_time(1'300));
  } catch (const std::invalid_argument&) {
    wrong_host_rejected = true;
  }
  auto changed_profile = profile;
  changed_profile.public_arguments.push_back("--changed");
  const trainvm::HostLaunchRegistry changed_registry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {directory.string()},
      .profiles = {changed_profile},
  });
  bool changed_profile_rejected = false;
  try {
    (void)controller.bind_worker_launch(first, changed_registry, host,
                                        test_time(1'300));
  } catch (const std::invalid_argument&) {
    changed_profile_rejected = true;
  }
  const auto executable_link = directory / "python-link";
  std::filesystem::create_symlink(executable, executable_link);
  auto symlink_profile = profile;
  symlink_profile.executable_path = executable_link.string();
  const trainvm::HostLaunchRegistry symlink_registry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {directory.string()},
      .profiles = {symlink_profile},
  });
  bool symlink_rejected = false;
  try {
    trainvm::HostLaunchResolver symlink_resolver(symlink_registry, host);
    (void)symlink_resolver.resolve(ticket, key);
  } catch (const trainvm::HostLaunchResolutionError&) {
    symlink_rejected = true;
  }
  auto fingerprint_profile = profile;
  fingerprint_profile.executable_fingerprint =
      "sha256:" + std::string(64U, 'f');
  const trainvm::HostLaunchRegistry fingerprint_registry({
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {directory.string()},
      .profiles = {fingerprint_profile},
  });
  bool fingerprint_rejected = false;
  try {
    trainvm::HostLaunchResolver fingerprint_resolver(fingerprint_registry,
                                                     host);
    (void)fingerprint_resolver.resolve(ticket, key);
  } catch (const trainvm::HostLaunchResolutionError&) {
    fingerprint_rejected = true;
  }
  std::filesystem::permissions(
      code, std::filesystem::perms::group_write,
      std::filesystem::perm_options::add);
  bool writable_code_rejected = false;
  try {
    (void)resolver.resolve(ticket, key);
  } catch (const trainvm::HostLaunchResolutionError&) {
    writable_code_rejected = true;
  }
  check(wrong_host_rejected && changed_profile_rejected &&
            symlink_rejected && fingerprint_rejected &&
            writable_code_rejected &&
            journal.event_count() == before_rejections,
        "host, profile, and secure-path mismatches fail before durable mutation");

  std::filesystem::remove_all(directory);
}

void test_service_host_launch_binding() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "service host launch fixture compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-service-host-launch-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory / "work");
  std::filesystem::create_directories(directory / "unused-root");
  std::filesystem::permissions(
      directory, std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  std::filesystem::permissions(
      directory / "work", std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  std::filesystem::permissions(
      directory / "unused-root", std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  const auto executable = directory / "python";
  std::filesystem::copy_file("/proc/self/exe", executable);
  std::filesystem::permissions(
      executable,
      std::filesystem::perms::owner_read |
          std::filesystem::perms::owner_exec,
      std::filesystem::perm_options::replace);
  const auto code = directory / "worker.pyz";
  {
    std::ofstream output(code, std::ios::binary);
    output << "PK\003\004service-owned-immutable-zipapp";
  }
  std::filesystem::permissions(
      code, std::filesystem::perms::owner_read,
      std::filesystem::perm_options::replace);
  const auto file_digest = [](const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    const std::string bytes((std::istreambuf_iterator<char>(input)),
                            std::istreambuf_iterator<char>());
    return "sha256:" + trainvm::sha256_hex(bytes);
  };
  const std::string executable_digest = file_digest(executable);
  const std::string code_digest = file_digest(code);

  auto adapter_profiles = fixture_adapter_profiles();
  for (auto& profile : adapter_profiles) {
    if (profile.key.runtime != trainvm::ComponentRuntime::builtin) {
      profile.code_fingerprint = code_digest;
    }
  }
  std::vector<trainvm::HostLaunchProfile> launch_profiles;
  for (const auto& profile : adapter_profiles) {
    if (profile.key.runtime == trainvm::ComponentRuntime::builtin) continue;
    launch_profiles.push_back({
        .key = profile.key,
        .code_fingerprint = code_digest,
        .executable_path = executable.string(),
        .executable_fingerprint = executable_digest,
        .code_path = code.string(),
        .public_arguments = {"-I", "worker.pyz"},
        .working_directory = (directory / "work").string(),
    });
  }
  const trainvm::HostLaunchRegistryDocument host_document{
      .api_version = "trainvm.host-launches/v1",
      .trusted_roots = {directory.string()},
      .profiles = launch_profiles,
  };
  const trainvm::HostIdentity host{
      .host_id = "sha256:" + std::string(64U, '3'),
      .boot_id = "33333333-3333-3333-3333-333333333333",
  };
  const auto database = directory / "journal.db";
  const std::string run_id = "service-host-binding-run";
  {
    trainvm::AdapterRegistry registry(adapter_profiles);
    trainvm::Journal journal(database);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(adapter_locked_submission(*compiled.plan,
                                                       registry));
  }

  trainvm::WorkerLaunchTicket ticket;
  trainvm::ResolvedLaunchSpec binding;
  std::size_t bound_event_count = 0U;
  {
    std::int64_t now_ns = 1'000;
    trainvm::TrainVMService service(
        database, trainvm::AdapterRegistry(adapter_profiles),
        trainvm::HostLaunchRegistry(host_document), host,
        [&now_ns] { return test_time(now_ns); });
    const auto acquired = service.reconcile_once(run_id);
    now_ns = 1'100;
    const auto prepared = service.reconcile_once(run_id);
    check(acquired.disposition ==
              trainvm::ReconcileDisposition::lease_acquired &&
              prepared.launch.has_value(),
          "service prepares a portable ticket before host binding");
    if (!prepared.launch) {
      std::filesystem::remove_all(directory);
      return;
    }
    ticket = *prepared.launch;
    now_ns = 1'200;
    binding = service.bind_worker_launch(ticket);
    const auto hidden_executable = directory / "python.hidden";
    const auto hidden_code = directory / "worker.pyz.hidden";
    std::filesystem::rename(executable, hidden_executable);
    std::filesystem::rename(code, hidden_code);
    trainvm::ResolvedLaunchSpec replay;
    try {
      replay = service.bind_worker_launch(ticket);
    } catch (...) {
      std::filesystem::rename(hidden_executable, executable);
      std::filesystem::rename(hidden_code, code);
      throw;
    }
    std::filesystem::rename(hidden_executable, executable);
    std::filesystem::rename(hidden_code, code);
    trainvm::Journal observer(database);
    bound_event_count = observer.event_count();
    const int retained_fd =
        service.resolved_launches_.at(binding.identity.launch_event_id)
            .duplicate_executable_fd();
    check(binding == replay && retained_fd >= 0 &&
              service.resolved_launches_.size() == 1U &&
              observer.launch_binding(binding.identity.launch_event_id) ==
                  std::optional<trainvm::ResolvedLaunchSpec>{binding},
          "service binds once and exact replay retains one sealed authority bundle");
    if (retained_fd >= 0) (void)::close(retained_fd);

    for (std::size_t index = service.resolved_launches_.size();
         index < trainvm::TrainVMService::kMaximumRetainedLaunches; ++index) {
      service.resolved_launches_.emplace(
          "synthetic-count-" + std::to_string(index),
          trainvm::ResolvedLaunch(
              binding, -1,
              binding.identity.code ? std::optional<int>{-1} : std::nullopt,
              -1));
    }
    bool count_quota_rejected = false;
    try {
      service.require_retained_launch_capacity(binding);
    } catch (const trainvm::HostLaunchResolutionError&) {
      count_quota_rejected = true;
    }
    service.resolved_launches_.clear();
    auto large = binding;
    large.identity.executable.source_size = 256ULL << 20U;
    large.identity.code.reset();
    for (std::size_t index = 0; index < 8U; ++index) {
      service.resolved_launches_.emplace(
          "synthetic-bytes-" + std::to_string(index),
          trainvm::ResolvedLaunch(large, -1, std::nullopt, -1));
    }
    bool byte_quota_rejected = false;
    try {
      service.require_retained_launch_capacity(binding);
    } catch (const trainvm::HostLaunchResolutionError&) {
      byte_quota_rejected = true;
    }
    check(count_quota_rejected && byte_quota_rejected,
          "service fails closed at retained launch count and aggregate byte quotas");
  }

  {
    trainvm::TrainVMService restarted(
        database, trainvm::AdapterRegistry(adapter_profiles),
        trainvm::HostLaunchRegistry(host_document), host,
        [] { return test_time(1'250); });
    trainvm::v1::WorkerHello hello;
    hello.set_run_id(ticket.run_id);
    hello.set_node_id(ticket.node_id);
    hello.set_attempt_id(ticket.attempt_id);
    hello.set_launch_nonce(ticket.launch_nonce);
    hello.set_adapter(ticket.adapter);
    hello.set_adapter_version(ticket.adapter_version);
    hello.set_code_fingerprint(ticket.code_fingerprint);
    for (const auto& capability : ticket.required_capabilities) {
      hello.add_capabilities(capability);
    }
    hello.set_last_acked_controller_sequence(0);
    hello.set_concurrency_key(ticket.concurrency_key);
    hello.set_lease_id(ticket.lease_id);
    hello.set_fencing_token(ticket.fencing_token);
    trainvm::TrainVMService::WorkerConnection unbound_connection;
    const grpc::Status unbound =
        restarted.open_worker_connection(hello, unbound_connection);
    const auto replayed = restarted.bind_worker_launch(ticket);
    const int rehydrated_fd =
        restarted.resolved_launches_.at(binding.identity.launch_event_id)
            .duplicate_executable_fd();
    check(unbound.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
              replayed == binding && rehydrated_fd >= 0 &&
              restarted.resolved_launches_.size() == 1U &&
              trainvm::Journal(database).event_count() == bound_event_count,
          "service restart rejects readiness until exact durable binding re-resolution rehydrates one sealed bundle without mutation");
    if (rehydrated_fd >= 0) (void)::close(rehydrated_fd);
  }

  const auto rejects_without_mutation =
      [&](trainvm::HostLaunchRegistry registry,
          trainvm::HostIdentity authority_host,
          std::vector<trainvm::AdapterProfile> adapters) {
        bool rejected = false;
        try {
          trainvm::TrainVMService service(
              database, trainvm::AdapterRegistry(std::move(adapters)),
              std::move(registry), std::move(authority_host),
              [] { return test_time(1'300); });
          (void)service.bind_worker_launch(ticket);
        } catch (const std::exception&) {
          rejected = true;
        }
        trainvm::Journal observer(database);
        return rejected && observer.event_count() == bound_event_count;
      };

  auto wrong_host = host;
  wrong_host.boot_id = "44444444-4444-4444-4444-444444444444";
  const bool host_drift_rejected = rejects_without_mutation(
      trainvm::HostLaunchRegistry(host_document), wrong_host,
      adapter_profiles);
  auto changed_profiles = launch_profiles;
  for (auto& profile : changed_profiles) {
    profile.public_arguments.push_back("--profile-drift");
  }
  const bool profile_drift_rejected = rejects_without_mutation(
      trainvm::HostLaunchRegistry({
          .api_version = "trainvm.host-launches/v1",
          .trusted_roots = {directory.string()},
          .profiles = std::move(changed_profiles),
      }),
      host, adapter_profiles);
  const bool registry_drift_rejected = rejects_without_mutation(
      trainvm::HostLaunchRegistry({
          .api_version = "trainvm.host-launches/v1",
          .trusted_roots = {"/"},
          .profiles = launch_profiles,
      }),
      host, adapter_profiles);
  auto skewed_adapters = adapter_profiles;
  for (auto& profile : skewed_adapters) {
    if (profile.key.runtime != trainvm::ComponentRuntime::builtin) {
      profile.code_fingerprint = "sha256:" + std::string(64U, 'f');
    }
  }
  const bool adapter_skew_rejected = rejects_without_mutation(
      trainvm::HostLaunchRegistry(host_document), host,
      std::move(skewed_adapters));
  check(host_drift_rejected && profile_drift_rejected &&
            registry_drift_rejected && adapter_skew_rejected,
        "service restart rejects current host, profile, registry, and portable adapter drift before mutation");

  const auto disabled_database = directory / "disabled.db";
  const std::string disabled_run = "service-host-disabled-run";
  {
    trainvm::AdapterRegistry registry(adapter_profiles);
    trainvm::Journal journal(disabled_database);
    trainvm::Controller controller(*compiled.plan, journal, disabled_run);
    controller.create_queued(adapter_locked_submission(*compiled.plan,
                                                       registry));
  }
  bool disabled_rejected = false;
  {
    trainvm::TrainVMService service(
        disabled_database, trainvm::AdapterRegistry(adapter_profiles),
        fixture_disabled_host_launch_registry(), host,
        [] { return test_time(2'000); });
    (void)service.reconcile_once(disabled_run);
    const auto prepared = service.reconcile_once(disabled_run);
    const std::size_t before = trainvm::Journal(disabled_database).event_count();
    try {
      (void)service.bind_worker_launch(*prepared.launch);
    } catch (const trainvm::HostLaunchResolutionError&) {
      disabled_rejected = true;
    }
    check(disabled_rejected && service.resolved_launches_.empty() &&
              trainvm::Journal(disabled_database).event_count() == before,
          "explicit empty host registry disables binding without mutation");
  }

  std::filesystem::remove_all(directory);
}

void test_adapter_registry_file_contract() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-adapter-registry-file-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto set_owner_only = [](const std::filesystem::path& path) {
    std::filesystem::permissions(
        path,
        std::filesystem::perms::owner_read |
            std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace);
  };
  const auto write_json = [&](const std::filesystem::path& path,
                              const nlohmann::json& value) {
    {
      std::ofstream output(path);
      output << value.dump(2) << '\n';
    }
    set_owner_only(path);
  };

  const auto original_profiles = fixture_external_adapter_profiles();
  const nlohmann::json original_document = trainvm::encode_json(
      trainvm::AdapterRegistryDocument{
          .api_version = "trainvm.adapters/v1",
          .profiles = original_profiles,
      });
  const auto original_path = directory / "registry.json";
  write_json(original_path, original_document);
  const trainvm::AdapterRegistry loaded =
      trainvm::AdapterRegistry::load_file(original_path);

  auto reversed_profiles = original_profiles;
  std::ranges::reverse(reversed_profiles);
  for (auto& profile : reversed_profiles) {
    std::ranges::reverse(profile.required_capabilities);
  }
  const auto reordered_path = directory / "registry-reordered.json";
  write_json(reordered_path,
             trainvm::encode_json(trainvm::AdapterRegistryDocument{
                 .api_version = "trainvm.adapters/v1",
                 .profiles = std::move(reversed_profiles),
             }));
  const trainvm::AdapterRegistry reordered =
      trainvm::AdapterRegistry::load_file(reordered_path);
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "registry file fixture plan compiles");
  if (compiled.valid()) {
    const auto& mageflow =
        compiled.plan->experiment.spec.components.at("mageflow");
    const auto& resolved = loaded.resolve(mageflow, "train");
    check(resolved.code_fingerprint ==
              "sha256:" + std::string(64, 'a') &&
              resolved.required_capabilities ==
                  std::vector<std::string>({"worker.controls",
                                            "worker.metrics"}) &&
              loaded.registry_digest().starts_with("sha256:") &&
              loaded.registry_digest().size() == 71U &&
              loaded.registry_digest() == reordered.registry_digest() &&
              loaded.registry_digest() ==
                  trainvm::AdapterRegistry(fixture_adapter_profiles())
                      .registry_digest(),
          "reflection-decoded external sibling profiles may repeat schema field names and canonicalize order into a stable full-registry digest");
  }

  const auto symlink_path = directory / "registry-symlink.json";
  std::filesystem::create_symlink(original_path, symlink_path);
  bool symlink_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(symlink_path);
  } catch (const std::invalid_argument&) {
    symlink_rejected = true;
  }

  auto unknown_document = original_document;
  unknown_document["unknown_authority_field"] = true;
  const auto unknown_path = directory / "registry-unknown.json";
  write_json(unknown_path, unknown_document);
  bool unknown_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(unknown_path);
  } catch (const std::invalid_argument&) {
    unknown_rejected = true;
  }

  auto future_document = original_document;
  future_document["api_version"] = "trainvm.adapters/v2";
  const auto future_path = directory / "registry-future.json";
  write_json(future_path, future_document);
  bool version_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(future_path);
  } catch (const std::invalid_argument&) {
    version_rejected = true;
  }

  auto reserved_namespace_document = original_document;
  reserved_namespace_document["profiles"].at(0)["key"]["adapter"] =
      "trainvm.core";
  const auto reserved_namespace_path =
      directory / "registry-reserved-namespace.json";
  write_json(reserved_namespace_path, reserved_namespace_document);
  bool reserved_namespace_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(reserved_namespace_path);
  } catch (const std::invalid_argument&) {
    reserved_namespace_rejected = true;
  }

  const auto duplicate_key_path = directory / "registry-duplicate-key.json";
  {
    std::ofstream duplicate_key(duplicate_key_path);
    duplicate_key << R"({"api_version":"trainvm.adapters/v1","api_version":"trainvm.adapters/v1","profiles":[]})";
  }
  set_owner_only(duplicate_key_path);
  bool duplicate_key_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(duplicate_key_path);
  } catch (const std::invalid_argument&) {
    duplicate_key_rejected = true;
  }

  const auto empty_path = directory / "registry-empty.json";
  {
    std::ofstream empty(empty_path);
  }
  set_owner_only(empty_path);
  bool empty_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(empty_path);
  } catch (const std::invalid_argument&) {
    empty_rejected = true;
  }

  const auto oversized_path = directory / "registry-oversized.json";
  {
    std::ofstream oversized(oversized_path);
    oversized << std::string((1U << 20U) + 1U, ' ');
  }
  set_owner_only(oversized_path);
  bool oversized_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(oversized_path);
  } catch (const std::invalid_argument&) {
    oversized_rejected = true;
  }
  check(symlink_rejected && unknown_rejected && version_rejected &&
            reserved_namespace_rejected && duplicate_key_rejected &&
            empty_rejected && oversized_rejected,
        "registry file loading rejects symlinks, unknown fields, unsupported versions, reserved trainvm.core names, duplicate keys, and unbounded files");
  std::filesystem::remove_all(directory);
}

void test_service_registry_and_reconciliation() {
  const auto fixture = load_fixture();
  const auto compiled = trainvm::compile_document(fixture);
  check(compiled.valid(),
        "fixture required by service registry reconciliation compiles");
  if (!compiled.valid()) return;

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-service-reconciler-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto request_for = [&](std::string journal_id, bool create_run,
                               std::string key,
                               std::string adapter_lock_digest = {}) {
    trainvm::v1::SubmitExperimentRequest request;
    request.set_source_document(fixture.dump());
    request.set_source_format("json");
    request.set_create_run(create_run);
    request.set_idempotency_key(std::move(key));
    request.set_expected_journal_id(std::move(journal_id));
    request.set_author(create_run ? "scheduler" : "");
    request.set_reason(create_run ? "service reconciliation test" : "");
    if (create_run) {
      request.set_expected_plan_hash(compiled.plan->plan_hash);
      request.set_expected_adapter_lock_digest(
          std::move(adapter_lock_digest));
    }
    return request;
  };
  const auto has_adapter_error = [](const auto& response) {
    return std::ranges::any_of(response.diagnostics(),
                               [](const trainvm::v1::Diagnostic& diagnostic) {
      return diagnostic.severity() ==
                 trainvm::v1::Diagnostic::SEVERITY_ERROR &&
             diagnostic.code() == "adapter.registry" &&
             diagnostic.document_path() == "/spec/components";
    });
  };

  {
    const auto database_path = directory / "mismatch.db";
    std::string journal_id;
    {
      trainvm::Journal journal(database_path);
      journal_id = journal.journal_id();
    }
    auto incomplete_profiles = fixture_adapter_profiles();
    incomplete_profiles.erase(
        std::remove_if(incomplete_profiles.begin(), incomplete_profiles.end(),
                       [](const trainvm::AdapterProfile& profile) {
                         return profile.key.operation == "train";
                       }),
        incomplete_profiles.end());
    trainvm::TrainVMService service(
        database_path,
        trainvm::AdapterRegistry(std::move(incomplete_profiles)),
        fixture_disabled_host_launch_registry());

    auto preview_request = request_for(journal_id, false, "");
    trainvm::v1::SubmitExperimentResponse preview;
    const grpc::Status preview_status =
        service.SubmitExperiment(nullptr, &preview_request, &preview);
    auto create_request = request_for(
        journal_id, true, "registry-mismatch",
        "sha256:" + std::string(64, '0'));
    trainvm::v1::SubmitExperimentResponse create;
    const grpc::Status create_status =
        service.SubmitExperiment(nullptr, &create_request, &create);
    trainvm::Journal observer(database_path);
    check(preview_status.ok() && create_status.ok() &&
              has_adapter_error(preview) && has_adapter_error(create) &&
              preview.canonical_document() == compiled.plan->canonical_plan.dump() &&
              preview.canonical_plan() == preview.canonical_document() &&
              preview.plan_hash() == compiled.plan->plan_hash &&
              create.canonical_document() == preview.canonical_document() &&
              create.plan_hash() == compiled.plan->plan_hash &&
              !preview.has_run() && !create.has_run() &&
              observer.event_count() == 0U &&
              !observer.compiled_plan(compiled.plan->plan_hash) &&
              !observer.active_lease(
                  compiled.plan->experiment.spec.workspace.concurrency_key,
                  test_time(1'000)),
          "preview and creation retain canonical identity but reject registry mismatch without mutation");
  }

  {
    const auto database_path = directory / "reconcile.db";
    std::string journal_id;
    {
      trainvm::Journal journal(database_path);
      journal_id = journal.journal_id();
    }
    std::int64_t authority_now_ns = 5'000;
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_disabled_host_launch_registry(),
        [&authority_now_ns] { return test_time(authority_now_ns); });
    auto preview_request = request_for(journal_id, false, "");
    trainvm::v1::SubmitExperimentResponse preview;
    const grpc::Status preview_status =
        service.SubmitExperiment(nullptr, &preview_request, &preview);
    auto create_request = request_for(
        journal_id, true, "service-reconcile",
        preview.adapter_lock_digest());
    trainvm::v1::SubmitExperimentResponse created;
    const grpc::Status create_status =
        service.SubmitExperiment(nullptr, &create_request, &created);
    const std::string run_id = created.has_run() ? created.run().run_id() : "";
    {
      trainvm::Journal observer(database_path);
      const auto queued = observer.projection(run_id);
      const auto created_event = observer.event(run_id + ":created");
      check(preview_status.ok() &&
                preview.adapter_lock_digest().starts_with("sha256:") &&
                preview.adapter_lock_digest().size() == 71U &&
                !preview.canonical_adapter_lock().empty() &&
                create_status.ok() && !has_adapter_error(created) &&
                created.adapter_lock_digest() ==
                    preview.adapter_lock_digest() &&
                created.canonical_adapter_lock() ==
                    preview.canonical_adapter_lock() &&
                created.has_run() && queued && created_event &&
                created_event->payload.at("submission")
                        .at("adapter_lock_digest") ==
                    preview.adapter_lock_digest() &&
                created_event->payload.at("submission").at("adapter_lock") ==
                    nlohmann::json::parse(
                        preview.canonical_adapter_lock()) &&
                queued->desired_state == "queued" &&
                queued->observed_state == "queued" &&
                observer.events_for_run(run_id).size() == 1U,
            "valid service submission remains queued until explicit reconciliation");
    }

    const auto acquired = service.reconcile_once(run_id);
    authority_now_ns = 5'100;
    const auto launched = service.reconcile_once(run_id);
    authority_now_ns = 5'200;
    const auto replayed = service.reconcile_once(run_id);
    trainvm::Journal observer(database_path);
    const auto projection = observer.projection(run_id);
    const auto events = observer.events_for_run(run_id);
    const auto launch_event = std::ranges::find_if(
        events, [](const trainvm::Event& event) {
          return event.event_type == "worker.launch_requested";
        });
    check(acquired.disposition ==
              trainvm::ReconcileDisposition::lease_acquired &&
              launched.disposition ==
                  trainvm::ReconcileDisposition::launch_prepared &&
              replayed.disposition ==
                  trainvm::ReconcileDisposition::launch_replayed &&
              launched.launch && replayed.launch &&
              *launched.launch == *replayed.launch &&
              launched.launch->code_fingerprint ==
                  "sha256:" + std::string(64, 'a') &&
              launched.launch->required_capabilities ==
                  std::vector<std::string>({"worker.controls",
                                            "worker.metrics"}) &&
              projection && projection->observed_state == "acquiring" &&
              projection->run_revision == 4U && events.size() == 7U &&
              launch_event != events.end() &&
              launch_event->wall_time_ns == 5'100,
          "service-owned reconcile path uses its registry, mutation gate, and authority clock for acquisition then launch");
  }

  std::filesystem::remove_all(directory);
}

void test_adapter_registry_and_reconciler() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "fixture required by adapter reconciliation compiles");
  if (!compiled.valid()) return;

  const std::string expected_fingerprint = "sha256:" + std::string(64, 'a');
  const trainvm::AdapterRegistry registry(fixture_adapter_profiles());
  const nlohmann::json submission_identity =
      adapter_locked_submission(*compiled.plan, registry);
  bool plan_validated = true;
  try {
    registry.validate_plan(*compiled.plan);
  } catch (const std::exception& exception) {
    plan_validated = false;
    std::cerr << "adapter registry validation error: " << exception.what()
              << '\n';
  }
  const auto& mageflow =
      compiled.plan->experiment.spec.components.at("mageflow");
  const auto& train_profile = registry.resolve(mageflow, "train");
  check(plan_validated &&
            train_profile.key.adapter == "rwkv-lab.mageflow" &&
            train_profile.key.version == "1.0.0" &&
            train_profile.key.runtime ==
                trainvm::ComponentRuntime::python_worker &&
            train_profile.key.operation == "train" &&
            train_profile.key.contract ==
                "rwkv_lab.mageflow.v1.Train" &&
            train_profile.effect == trainvm::Effect::process &&
            train_profile.idempotency ==
                trainvm::Idempotency::receipt_required &&
            train_profile.code_fingerprint == expected_fingerprint &&
            train_profile.required_capabilities ==
                std::vector<std::string>({"worker.controls",
                                          "worker.metrics"}),
        "adapter registry resolves the exact authority-owned operation profile");

  auto duplicate_profiles = fixture_adapter_profiles();
  duplicate_profiles.push_back(duplicate_profiles.back());
  bool duplicate_rejected = false;
  try {
    trainvm::AdapterRegistry duplicate(std::move(duplicate_profiles));
  } catch (const std::invalid_argument&) {
    duplicate_rejected = true;
  }
  auto semantic_mismatch_profiles = fixture_adapter_profiles();
  const auto semantic_profile = std::ranges::find_if(
      semantic_mismatch_profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (semantic_profile != semantic_mismatch_profiles.end()) {
    semantic_profile->effect = trainvm::Effect::read_only;
  }
  bool semantic_mismatch_rejected = false;
  try {
    const trainvm::AdapterRegistry mismatch(
        std::move(semantic_mismatch_profiles));
    mismatch.validate_plan(*compiled.plan);
  } catch (const trainvm::AdapterResolutionError&) {
    semantic_mismatch_rejected = true;
  }
  check(duplicate_rejected && semantic_mismatch_rejected,
        "adapter registry rejects duplicate keys and under-declared operation semantics");

  const auto launch_bytes = [](const trainvm::WorkerLaunchTicket& launch) {
    return nlohmann::json{
        {"run_id", launch.run_id},
        {"node_id", launch.node_id},
        {"attempt_id", launch.attempt_id},
        {"launch_nonce", launch.launch_nonce},
        {"adapter", launch.adapter},
        {"adapter_version", launch.adapter_version},
        {"code_fingerprint", launch.code_fingerprint},
        {"required_capabilities", launch.required_capabilities},
        {"concurrency_key", launch.concurrency_key},
        {"lease_id", launch.lease_id},
        {"fencing_token", launch.fencing_token},
    }.dump();
  };

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-reconciler-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);

  {
    const auto database_path = directory / "registry-mismatch.db";
    const std::string run_id = "registry-mismatch-run";
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(submission_identity);
    const auto before_projection = journal.projection(run_id);
    const auto before_count = journal.event_count();
    auto missing_profiles = fixture_adapter_profiles();
    missing_profiles.erase(std::remove_if(
        missing_profiles.begin(), missing_profiles.end(),
        [](const trainvm::AdapterProfile& profile) {
          return profile.key.operation == "train";
        }), missing_profiles.end());
    const trainvm::AdapterRegistry missing_registry(
        std::move(missing_profiles));
    std::mutex authority_mutex;
    trainvm::Reconciler reconciler(journal, missing_registry,
                                   authority_mutex, [] { return test_time(1'000); });
    bool mismatch_rejected = false;
    try {
      (void)reconciler.step(run_id);
    } catch (const trainvm::AdapterResolutionError&) {
      mismatch_rejected = true;
    }
    check(mismatch_rejected && journal.event_count() == before_count &&
              journal.projection(run_id) == before_projection &&
              !journal.active_lease(
                  compiled.plan->experiment.spec.workspace.concurrency_key,
                  test_time(1'000)),
          "registry mismatch rejects reconciliation before any journal or lease mutation");
  }

  {
    const auto database_path = directory / "partial-admission.db";
    const std::string run_id = "reconciler-partial-admission-run";
    const std::string& concurrency_key =
        compiled.plan->experiment.spec.workspace.concurrency_key;
    const std::string lease_id =
        "lease-" + trainvm::sha256_hex(
            nlohmann::json{{"run_id", run_id},
                           {"plan_hash", compiled.plan->plan_hash},
                           {"concurrency_key", concurrency_key}}
                .dump());
    {
      trainvm::Journal journal(database_path);
      trainvm::Controller controller(*compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      const auto acquisition_event =
          [&](std::string event_id, std::uint64_t revision,
              std::string event_type, nlohmann::json payload) {
            return trainvm::Event{
                .event_id = std::move(event_id),
                .run_id = run_id,
                .run_revision = revision,
                .plan_revision = 1,
                .node_id = {},
                .attempt_id = {},
                .worker_sequence = 0,
                .event_type = std::move(event_type),
                .event_version = 1,
                .wall_time_ns = 4'000,
                .monotonic_time_ns = 0,
                .optimizer_step = std::nullopt,
                .payload = std::move(payload),
            };
          };
      const std::string acquired_id = run_id + ":lease-acquired";
      const auto acquired = journal.acquire_lease_with_events(
          concurrency_key, run_id, lease_id, test_time(4'000),
          30'000'000'000LL,
          {acquisition_event(
               run_id + ":lease-desired", 2,
               "run.desired_state_changed",
               {{"state", "running"},
                {"cause", "scheduler.lease_acquisition"},
                {"lease_id", lease_id},
                {"plan_hash", compiled.plan->plan_hash}}),
           acquisition_event(
               acquired_id, 2, "resource.lease_acquired",
               {{"concurrency_key", concurrency_key},
                {"owner_run_id", run_id},
                {"lease_id", lease_id}}),
           acquisition_event(
               run_id + ":acquiring", 3,
               "run.observed_state_changed",
               {{"state", "acquiring"},
                {"cause_event_id", acquired_id},
                {"concurrency_key", concurrency_key},
                {"lease_id", lease_id}})});
      const auto partial = journal.projection(run_id);
      check(acquired.status == trainvm::LeaseAcquireStatus::acquired &&
                partial && partial->desired_state == "running" &&
                partial->observed_state == "acquiring" &&
                partial->run_revision == 3U &&
                journal.events_for_run(run_id).size() == 4U,
            "partial admission fixture stops after the durable lease prefix");
    }
    trainvm::Journal restarted_journal(database_path);
    std::mutex authority_mutex;
    trainvm::Reconciler restarted(restarted_journal, registry,
                                  authority_mutex, [] { return test_time(4'100); });
    const auto resumed = restarted.step(run_id);
    const auto completed = restarted_journal.projection(run_id);
    trainvm::Controller recovered(*compiled.plan, restarted_journal, run_id);
    const auto& execution = recovered.recover();
    check(resumed.disposition ==
              trainvm::ReconcileDisposition::lease_acquired &&
              !resumed.launch && completed &&
              completed->observed_state == "acquiring" &&
              completed->run_revision == 4U &&
              completed->current_node_id.empty() &&
              execution.current_node_id == "train_to_boundary" &&
              restarted_journal.events_for_run(run_id).size() == 6U &&
              !restarted_journal.event(
                  run_id + ":worker-launch:train_to_boundary:"
                           "train_to_boundary@1"),
          "reconciler restart completes a partial builtin admission before worker launch");
  }

  const auto database_path = directory / "restart-replay.db";
  const std::string run_id = "reconciler-restart-run";
  {
    trainvm::Journal journal(database_path);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(submission_identity);
    std::mutex authority_mutex;
    trainvm::Reconciler reconciler(journal, registry, authority_mutex,
                                   [] { return test_time(2'000); });
    const auto acquired = reconciler.step(run_id);
    const auto projection = journal.projection(run_id);
    check(acquired.disposition ==
              trainvm::ReconcileDisposition::lease_acquired &&
              !acquired.launch && projection &&
              projection->desired_state == "running" &&
              projection->observed_state == "acquiring" &&
              projection->run_revision == 4U &&
              projection->current_node_id.empty() &&
              projection->current_attempt_id.empty() &&
              journal.events_for_run(run_id).size() == 6U,
          "first reconciler step atomically acquires the queued run without launching");
  }

  struct ReconcileOutcome {
    std::optional<trainvm::ReconcileResult> result;
    std::string error;
  };
  trainvm::WorkerLaunchTicket prepared_launch;
  {
    trainvm::Journal left_journal(database_path);
    trainvm::Journal right_journal(database_path);
    std::mutex authority_mutex;
    trainvm::Reconciler left(left_journal, registry, authority_mutex,
                             [] { return test_time(2'100); });
    trainvm::Reconciler right(right_journal, registry, authority_mutex,
                              [] { return test_time(2'100); });
    std::promise<void> start;
    const auto gate = start.get_future().share();
    const auto reconcile = [&](trainvm::Reconciler& candidate) {
      gate.wait();
      try {
        return ReconcileOutcome{
            .result = candidate.step(run_id), .error = {}};
      } catch (const std::exception& exception) {
        return ReconcileOutcome{
            .result = std::nullopt, .error = exception.what()};
      }
    };
    auto left_future =
        std::async(std::launch::async, reconcile, std::ref(left));
    auto right_future =
        std::async(std::launch::async, reconcile, std::ref(right));
    start.set_value();
    const auto left_result = left_future.get();
    const auto right_result = right_future.get();
    if (!left_result.result || !right_result.result) {
      std::cerr << "reconciler launch race errors: left='"
                << left_result.error << "' right='" << right_result.error
                << "'\n";
    }
    const bool prepared_and_replayed =
        left_result.result && right_result.result &&
        ((left_result.result->disposition ==
              trainvm::ReconcileDisposition::launch_prepared &&
          right_result.result->disposition ==
              trainvm::ReconcileDisposition::launch_replayed) ||
         (left_result.result->disposition ==
              trainvm::ReconcileDisposition::launch_replayed &&
          right_result.result->disposition ==
              trainvm::ReconcileDisposition::launch_prepared));
    if (left_result.result && left_result.result->launch) {
      prepared_launch = *left_result.result->launch;
    } else if (right_result.result && right_result.result->launch) {
      prepared_launch = *right_result.result->launch;
    }
    const auto events = left_journal.events_for_run(run_id);
    const auto launch_count = std::ranges::count_if(
        events, [](const trainvm::Event& event) {
          return event.event_type == "worker.launch_requested";
        });
    check(prepared_and_replayed && left_result.result->launch &&
              right_result.result->launch &&
              launch_bytes(*left_result.result->launch) ==
                  launch_bytes(*right_result.result->launch) &&
              prepared_launch.code_fingerprint == expected_fingerprint &&
              prepared_launch.required_capabilities ==
                  std::vector<std::string>({"worker.controls",
                                            "worker.metrics"}) &&
              launch_count == 1 && events.size() == 7U,
          "concurrent acquired-run steps prepare one byte-identical registry-authorized launch");
  }

  std::string durable_launch_payload;
  {
    trainvm::Journal restarted_journal(database_path);
    std::mutex restarted_mutex;
    trainvm::Reconciler restarted(restarted_journal, registry,
                                  restarted_mutex, [] { return test_time(2'200); });
    const auto first = restarted.step(run_id);
    const auto second = restarted.step(run_id);
    const std::string launch_id =
        run_id + ":worker-launch:" + prepared_launch.node_id + ":" +
        prepared_launch.attempt_id;
    const auto durable_launch = restarted_journal.event(launch_id);
    if (durable_launch) durable_launch_payload = durable_launch->payload.dump();
    check(first.disposition ==
              trainvm::ReconcileDisposition::launch_replayed &&
              second.disposition ==
                  trainvm::ReconcileDisposition::launch_replayed &&
              first.launch && second.launch &&
              launch_bytes(*first.launch) == launch_bytes(prepared_launch) &&
              launch_bytes(*second.launch) == launch_bytes(prepared_launch) &&
              durable_launch &&
              durable_launch->event_type == "worker.launch_requested" &&
              restarted_journal.events_for_run(run_id).size() == 7U,
          "restart and repeated reconciliation replay the byte-identical launch without events");
  }

  const auto drift_rejected_without_mutation =
      [&](std::vector<trainvm::AdapterProfile> profiles,
          std::string_view message) {
        const trainvm::AdapterRegistry drifted(std::move(profiles));
        trainvm::Journal journal(database_path);
        const auto before_projection = journal.projection(run_id);
        const auto before_count = journal.event_count();
        const std::string launch_id =
            run_id + ":worker-launch:" + prepared_launch.node_id + ":" +
            prepared_launch.attempt_id;
        const auto before_launch = journal.event(launch_id);
        std::mutex authority_mutex;
        trainvm::Reconciler reconciler(journal, drifted, authority_mutex,
                                       [] { return test_time(2'400); });
        bool rejected = false;
        try {
          (void)reconciler.step(run_id);
        } catch (const trainvm::AdapterResolutionError&) {
          rejected = true;
        }
        const auto after_launch = journal.event(launch_id);
        check(rejected && journal.event_count() == before_count &&
                  journal.projection(run_id) == before_projection &&
                  before_launch && after_launch &&
                  before_launch->payload.dump() == durable_launch_payload &&
                  after_launch->payload.dump() == durable_launch_payload,
              message);
      };
  drift_rejected_without_mutation(
      fixture_adapter_profiles('b'),
      "code fingerprint drift after launch intent fails closed without mutation");
  auto capability_drift = fixture_adapter_profiles();
  for (auto& profile : capability_drift) {
    if (profile.key.runtime != trainvm::ComponentRuntime::builtin) {
      profile.required_capabilities.push_back("worker.artifacts");
    }
  }
  drift_rejected_without_mutation(
      std::move(capability_drift),
      "adapter capability drift after launch intent fails closed without mutation");

  {
    const auto busy_path = directory / "busy.db";
    const std::string owner_run = "reconciler-lease-owner";
    const std::string waiting_run = "reconciler-lease-waiter";
    trainvm::Journal journal(busy_path);
    trainvm::Controller owner(*compiled.plan, journal, owner_run);
    trainvm::Controller waiter(*compiled.plan, journal, waiting_run);
    owner.create_queued(submission_identity);
    waiter.create_queued(submission_identity);
    std::mutex authority_mutex;
    std::int64_t authority_now_ns = 3'000;
    trainvm::Reconciler reconciler(
        journal, registry, authority_mutex,
        [&authority_now_ns] { return test_time(authority_now_ns); });
    const auto acquired = reconciler.step(owner_run);
    const auto before_waiter = journal.projection(waiting_run);
    authority_now_ns = 3'100;
    const auto busy = reconciler.step(waiting_run);
    const auto active = journal.active_lease(
        compiled.plan->experiment.spec.workspace.concurrency_key, test_time(3'100));
    check(acquired.disposition ==
              trainvm::ReconcileDisposition::lease_acquired &&
              busy.disposition == trainvm::ReconcileDisposition::lease_busy &&
              !busy.launch && journal.projection(waiting_run) == before_waiter &&
              before_waiter && before_waiter->desired_state == "queued" &&
              before_waiter->observed_state == "queued" &&
              journal.events_for_run(waiting_run).size() == 1U && active &&
              active->owner_run_id == owner_run &&
              !journal.event(waiting_run + ":worker-launch:acquire_gpu:"
                                           "acquire_gpu@1"),
          "busy reconciliation leaves the second run queued and launch-free");
  }

  std::filesystem::remove_all(directory);
}

void test_legacy_journal_migration_policy() {
  const std::filesystem::path directory = std::filesystem::temp_directory_path() /
      ("trainvm-legacy-journal-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "legacy.db";

  sqlite3* database = nullptr;
  const auto scalar = [](sqlite3* connection, const char* sql) {
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(connection, sql, -1, &statement, nullptr) != SQLITE_OK) {
      throw std::runtime_error("could not prepare migration fixture query");
    }
    std::string value;
    if (sqlite3_step(statement) == SQLITE_ROW &&
        sqlite3_column_type(statement, 0) != SQLITE_NULL) {
      value = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
    }
    sqlite3_finalize(statement);
    return value;
  };
  check(sqlite3_open(database_path.c_str(), &database) == SQLITE_OK,
        "legacy migration test opens its fixture database");
  if (database != nullptr) {
    const char* fixture = R"sql(
      CREATE TABLE journal_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
      INSERT INTO journal_meta(key, value) VALUES('schema_version', '3');
      CREATE TABLE events(marker INTEGER NOT NULL);
      CREATE TABLE resource_leases(marker INTEGER NOT NULL);
      INSERT INTO resource_leases(marker) VALUES(1);
    )sql";
    check(sqlite3_exec(database, fixture, nullptr, nullptr, nullptr) == SQLITE_OK,
          "legacy migration test creates non-event pre-v4 durable state");
    sqlite3_close(database);
    database = nullptr;
  }

  bool refused = false;
  try {
    trainvm::Journal journal(database_path);
  } catch (const std::runtime_error& exception) {
    refused = std::string(exception.what()).find("pre-v4 journal") != std::string::npos;
  }
  check(refused, "nonempty pre-v4 journals are preserved rather than silently blessed as v4");

  check(sqlite3_open(database_path.c_str(), &database) == SQLITE_OK,
        "refused legacy journal remains readable");
  if (database != nullptr) {
    sqlite3_stmt* statement = nullptr;
    std::string version;
    if (sqlite3_prepare_v2(database,
                           "SELECT value FROM journal_meta WHERE key='schema_version'", -1,
                           &statement, nullptr) == SQLITE_OK &&
        sqlite3_step(statement) == SQLITE_ROW) {
      version = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
    }
    sqlite3_finalize(statement);
    sqlite3_close(database);
    check(version == "3", "refused legacy journal schema version is not mutated");
  }

  const auto empty_legacy_path = directory / "empty-legacy.db";
  check(sqlite3_open(empty_legacy_path.c_str(), &database) == SQLITE_OK,
        "empty legacy migration test opens its fixture database");
  if (database != nullptr) {
    const char* fixture = R"sql(
      CREATE TABLE journal_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
      INSERT INTO journal_meta(key, value) VALUES('schema_version', '3');
      CREATE TABLE events(marker INTEGER NOT NULL);
      CREATE TABLE resource_leases(marker INTEGER NOT NULL);
    )sql";
    check(sqlite3_exec(database, fixture, nullptr, nullptr, nullptr) == SQLITE_OK,
          "empty legacy migration test creates old table definitions");
    sqlite3_close(database);
    database = nullptr;
  }
  bool empty_refused = false;
  try {
    trainvm::Journal journal(empty_legacy_path);
  } catch (const std::runtime_error& exception) {
    empty_refused =
        std::string(exception.what()).find("pre-v4 journal") != std::string::npos;
  }
  check(empty_refused,
        "empty pre-v4 journals are refused because old table definitions are unsafe");
  check(sqlite3_open(empty_legacy_path.c_str(), &database) == SQLITE_OK,
        "refused empty legacy journal remains readable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") == "3",
          "refused empty legacy journal schema version is not mutated");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto unversioned_path = directory / "unversioned-nonempty.db";
  check(sqlite3_open(unversioned_path.c_str(), &database) == SQLITE_OK,
        "unversioned journal test opens its fixture database");
  std::string unversioned_schema_before;
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
      CREATE TABLE resource_leases(marker TEXT NOT NULL);
      INSERT INTO resource_leases VALUES('preserve-me');
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "unversioned journal test creates a nonempty application schema");
    unversioned_schema_before = scalar(
        database,
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='resource_leases'");
    sqlite3_close(database);
    database = nullptr;
  }
  bool unversioned_refused = false;
  try {
    trainvm::Journal journal(unversioned_path);
  } catch (const std::runtime_error& exception) {
    unversioned_refused = std::string(exception.what()).find(
                              "unversioned nonempty journal") !=
                          std::string::npos;
  }
  check(unversioned_refused,
        "unversioned nonempty databases are never adopted as fresh journals");
  check(sqlite3_open(unversioned_path.c_str(), &database) == SQLITE_OK,
        "refused unversioned journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database, R"sql(
            SELECT EXISTS(
              SELECT 1 FROM sqlite_master
              WHERE type='table' AND name='journal_meta'
            )
          )sql") == "0" &&
              scalar(database,
                     "SELECT sql FROM sqlite_master WHERE type='table' "
                     "AND name='resource_leases'") ==
                  unversioned_schema_before &&
              scalar(database,
                     "SELECT marker FROM resource_leases LIMIT 1") ==
                  "preserve-me",
          "unversioned refusal preserves schema and rows without metadata writes");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto foreign_header_path = directory / "foreign-empty-sqlite.db";
  check(sqlite3_open(foreign_header_path.c_str(), &database) == SQLITE_OK,
        "foreign SQLite header test opens its fixture database");
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
      PRAGMA application_id=1414677846;
      PRAGMA user_version=77;
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "foreign SQLite header test claims its empty database");
    sqlite3_close(database);
    database = nullptr;
  }
  bool foreign_header_refused = false;
  try {
    trainvm::Journal journal(foreign_header_path);
  } catch (const std::runtime_error& exception) {
    foreign_header_refused =
        std::string(exception.what()).find("claimed by another application") !=
        std::string::npos;
  }
  check(foreign_header_refused,
        "metadata-free SQLite databases with foreign header ownership are refused");
  check(sqlite3_open(foreign_header_path.c_str(), &database) == SQLITE_OK,
        "refused foreign SQLite database remains inspectable");
  if (database != nullptr) {
    check(scalar(database, "PRAGMA application_id") == "1414677846" &&
              scalar(database, "PRAGMA user_version") == "77" &&
              scalar(database, "SELECT COUNT(*) FROM sqlite_master") == "0" &&
              scalar(database, "PRAGMA journal_mode") == "delete",
          "foreign SQLite refusal preserves header ownership and journal mode without adding schema");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto future_path = directory / "future.db";
  check(sqlite3_open(future_path.c_str(), &database) == SQLITE_OK,
        "future schema test opens its fixture database");
  if (database != nullptr) {
    const char* fixture = R"sql(
      CREATE TABLE journal_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
      INSERT INTO journal_meta(key, value) VALUES('schema_version', '999');
    )sql";
    check(sqlite3_exec(database, fixture, nullptr, nullptr, nullptr) == SQLITE_OK,
          "future schema test creates its metadata");
    sqlite3_close(database);
    database = nullptr;
  }
  bool future_refused = false;
  try {
    trainvm::Journal journal(future_path);
  } catch (const std::runtime_error& exception) {
    future_refused = std::string(exception.what()).find("unsupported journal schema version") !=
                     std::string::npos;
  }
  check(future_refused, "future journal schemas are rejected before initialization writes");
  check(sqlite3_open(future_path.c_str(), &database) == SQLITE_OK,
        "refused future journal remains readable");
  if (database != nullptr) {
    sqlite3_stmt* statement = nullptr;
    std::string version;
    if (sqlite3_prepare_v2(database,
                           "SELECT value FROM journal_meta WHERE key='schema_version'", -1,
                           &statement, nullptr) == SQLITE_OK &&
        sqlite3_step(statement) == SQLITE_ROW) {
      version = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
    }
    sqlite3_finalize(statement);
    sqlite3_close(database);
    check(version == "999", "future journal schema version is not mutated");
  }

  for (const auto& [name, identity] :
       std::array<std::pair<std::string_view, std::optional<std::string_view>>, 2>{
           std::pair<std::string_view, std::optional<std::string_view>>{"missing", std::nullopt},
           std::pair<std::string_view, std::optional<std::string_view>>{
               "malformed", "not-a-valid-journal-identity!!"}}) {
    const auto identity_path = directory / (std::string(name) + "-identity.db");
    {
      trainvm::Journal fresh(identity_path);
    }
    check(sqlite3_open(identity_path.c_str(), &database) == SQLITE_OK,
          "v4 identity test opens its fixture database");
    if (database != nullptr) {
      check(sqlite3_exec(database, R"sql(
        BEGIN IMMEDIATE;
        ALTER TABLE resource_leases RENAME TO resource_leases_v5;
        CREATE TABLE resource_leases (
          concurrency_key TEXT PRIMARY KEY,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          acquired_at_ns INTEGER NOT NULL,
          expires_at_ns INTEGER NOT NULL,
          released_at_ns INTEGER
        ) WITHOUT ROWID;
        DROP TABLE resource_leases_v5;
        ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
        CREATE TABLE resource_lease_releases (
          concurrency_key TEXT NOT NULL,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          released_at_ns INTEGER NOT NULL,
          PRIMARY KEY(concurrency_key, lease_id, fencing_token)
        ) WITHOUT ROWID;
        DROP TABLE resource_lease_releases_v5;
        UPDATE journal_meta SET value='4' WHERE key='schema_version';
        COMMIT;
      )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
            "v4 identity test creates an exact established v4 schema");
      if (identity) {
        sqlite3_stmt* statement = nullptr;
        check(sqlite3_prepare_v2(database,
                                 "UPDATE journal_meta SET value=? WHERE key='journal_id'",
                                 -1, &statement, nullptr) == SQLITE_OK,
              "v4 identity test prepares malformed identity");
        if (statement != nullptr) {
          sqlite3_bind_text(statement, 1, identity->data(),
                            static_cast<int>(identity->size()), SQLITE_TRANSIENT);
          check(sqlite3_step(statement) == SQLITE_DONE,
                "v4 identity test stores malformed identity");
        }
        sqlite3_finalize(statement);
      } else {
        check(sqlite3_exec(database,
                           "DELETE FROM journal_meta WHERE key='journal_id'", nullptr,
                           nullptr, nullptr) == SQLITE_OK,
              "v4 identity test removes its established identity");
      }
      sqlite3_close(database);
      database = nullptr;
    }
    bool identity_refused = false;
    try {
      trainvm::Journal journal(identity_path);
    } catch (const std::runtime_error& exception) {
      identity_refused =
          std::string(exception.what()).find("authority metadata") !=
          std::string::npos;
    }
    check(identity_refused, "established v4 journal rejects " + std::string(name) +
                                " identity without repairing it");
    check(sqlite3_open(identity_path.c_str(), &database) == SQLITE_OK,
          "refused v4 identity fixture remains inspectable");
    if (database != nullptr) {
      const std::string stored_identity = scalar(
          database,
          "SELECT value FROM journal_meta WHERE key='journal_id'");
      check(scalar(database,
                   "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                    "4" &&
                stored_identity == (identity ? std::string(*identity) : std::string{}) &&
                scalar(database, R"sql(
                  SELECT group_concat(name, ',')
                  FROM pragma_table_info('resource_leases')
                )sql") ==
                    "concurrency_key,owner_run_id,lease_id,fencing_token,acquired_at_ns,expires_at_ns,released_at_ns",
            "v4 identity refusal preserves metadata and lease schema without migration");
      sqlite3_close(database);
      database = nullptr;
    }
  }

  const auto corrupt_v4 = [&](std::string_view name,
                              std::string_view object_name,
                              std::string_view corruption) {
    const auto path = directory / ("malformed-v4-" + std::string(name) + ".db");
    {
      trainvm::Journal fresh(path);
    }
    check(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "malformed v4 " + std::string(name) + " test opens its fixture");
    std::string object_before;
    if (database != nullptr) {
      const char* downgrade = R"sql(
        BEGIN IMMEDIATE;
        ALTER TABLE resource_leases RENAME TO resource_leases_v5;
        CREATE TABLE resource_leases (
          concurrency_key TEXT PRIMARY KEY,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          acquired_at_ns INTEGER NOT NULL,
          expires_at_ns INTEGER NOT NULL,
          released_at_ns INTEGER
        ) WITHOUT ROWID;
        DROP TABLE resource_leases_v5;
        ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
        CREATE TABLE resource_lease_releases (
          concurrency_key TEXT NOT NULL,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          released_at_ns INTEGER NOT NULL,
          PRIMARY KEY(concurrency_key, lease_id, fencing_token)
        ) WITHOUT ROWID;
        DROP TABLE resource_lease_releases_v5;
        UPDATE journal_meta SET value='4' WHERE key='schema_version';
        COMMIT;
      )sql";
      check(sqlite3_exec(database, downgrade, nullptr, nullptr, nullptr) == SQLITE_OK,
            "malformed v4 fixture starts from the exact established lease schema");
      const std::string owned_corruption(corruption);
      check(sqlite3_exec(database, owned_corruption.c_str(), nullptr, nullptr,
                         nullptr) == SQLITE_OK,
            "malformed v4 fixture corrupts its " + std::string(name));
      sqlite3_stmt* object = nullptr;
      check(sqlite3_prepare_v2(
                database,
                "SELECT sql FROM sqlite_master WHERE name=? AND sql IS NOT NULL",
                -1, &object, nullptr) == SQLITE_OK,
            "malformed v4 fixture prepares object inspection");
      if (object != nullptr) {
        sqlite3_bind_text(object, 1, object_name.data(),
                          static_cast<int>(object_name.size()), SQLITE_TRANSIENT);
        if (sqlite3_step(object) == SQLITE_ROW) {
          object_before = reinterpret_cast<const char*>(sqlite3_column_text(object, 0));
        }
      }
      sqlite3_finalize(object);
      sqlite3_close(database);
      database = nullptr;
    }

    bool rejected = false;
    try {
      trainvm::Journal journal(path);
    } catch (const std::runtime_error& exception) {
      rejected = std::string(exception.what()).find(
                     "authoritative schema") != std::string::npos;
    }
    check(rejected, "malformed v4 " + std::string(name) +
                        " is rejected before migration");
    check(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "rejected malformed v4 " + std::string(name) + " remains inspectable");
    if (database != nullptr) {
      sqlite3_stmt* object = nullptr;
      std::string object_after;
      if (sqlite3_prepare_v2(
              database,
              "SELECT sql FROM sqlite_master WHERE name=? AND sql IS NOT NULL",
              -1, &object, nullptr) == SQLITE_OK) {
        sqlite3_bind_text(object, 1, object_name.data(),
                          static_cast<int>(object_name.size()), SQLITE_TRANSIENT);
        if (sqlite3_step(object) == SQLITE_ROW) {
          object_after = reinterpret_cast<const char*>(sqlite3_column_text(object, 0));
        }
      }
      sqlite3_finalize(object);
      check(scalar(database,
                   "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                    "4" &&
                scalar(database, R"sql(
                  SELECT group_concat(name, ',')
                  FROM pragma_table_info('resource_leases')
                )sql") ==
                    "concurrency_key,owner_run_id,lease_id,fencing_token,acquired_at_ns,expires_at_ns,released_at_ns" &&
                object_after == object_before,
            "malformed v4 " + std::string(name) +
                " refusal preserves version, lease schema, and corrupt object byte-for-byte");
      sqlite3_close(database);
      database = nullptr;
    }
  };

  corrupt_v4(
      "table", "events", R"sql(
        DROP TABLE events;
        CREATE TABLE events(marker INTEGER NOT NULL);
      )sql");
  corrupt_v4(
      "index", "idx_events_run_sequence", R"sql(
        DROP INDEX idx_events_run_sequence;
        CREATE INDEX idx_events_run_sequence ON events(event_id);
      )sql");
  corrupt_v4(
      "constraint", "node_dispatches", R"sql(
        DROP TABLE node_dispatches;
        CREATE TABLE node_dispatches (
          dispatch_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          run_revision INTEGER NOT NULL,
          plan_revision INTEGER NOT NULL,
          node_id TEXT NOT NULL,
          attempt_id TEXT NOT NULL,
          component TEXT NOT NULL,
          operation TEXT NOT NULL,
          status TEXT NOT NULL,
          result_event_id TEXT,
          UNIQUE(run_id, node_id, attempt_id)
        ) WITHOUT ROWID;
      )sql");
  corrupt_v4(
      "quoted-literal", "node_dispatches", R"sql(
        DROP TABLE node_dispatches;
        CREATE TABLE node_dispatches (
          dispatch_id TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          run_revision INTEGER NOT NULL,
          plan_revision INTEGER NOT NULL,
          node_id TEXT NOT NULL,
          attempt_id TEXT NOT NULL,
          component TEXT NOT NULL,
          operation TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('prepared',' completed')),
          result_event_id TEXT,
          UNIQUE(run_id, node_id, attempt_id)
        ) WITHOUT ROWID;
      )sql");

  for (const bool with_event : {false, true}) {
    const std::string fixture_name = with_event ? "event" : "empty";
    const auto path = directory / ("v4-missing-head-" + fixture_name + ".db");
    {
      trainvm::Journal fresh(path);
      if (with_event) {
        check(trainvm::JournalTestAccess::append(
                  fresh, created_event(std::string(64U, 'b'))) == 1U,
              "missing-head v4 fixture creates a valid event chain");
      }
    }
    check(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "missing-head v4 fixture opens for exact downgrade");
    std::string identity_before;
    if (database != nullptr) {
      const char* downgrade = R"sql(
        PRAGMA journal_mode=DELETE;
        BEGIN IMMEDIATE;
        ALTER TABLE resource_leases RENAME TO resource_leases_v5;
        CREATE TABLE resource_leases (
          concurrency_key TEXT PRIMARY KEY,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          acquired_at_ns INTEGER NOT NULL,
          expires_at_ns INTEGER NOT NULL,
          released_at_ns INTEGER
        ) WITHOUT ROWID;
        DROP TABLE resource_leases_v5;
        ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
        CREATE TABLE resource_lease_releases (
          concurrency_key TEXT NOT NULL,
          owner_run_id TEXT NOT NULL,
          lease_id TEXT NOT NULL,
          fencing_token INTEGER NOT NULL,
          released_at_ns INTEGER NOT NULL,
          PRIMARY KEY(concurrency_key, lease_id, fencing_token)
        ) WITHOUT ROWID;
        DROP TABLE resource_lease_releases_v5;
        UPDATE journal_meta SET value='4' WHERE key='schema_version';
        DELETE FROM journal_meta WHERE key='chain_head';
        COMMIT;
      )sql";
      check(sqlite3_exec(database, downgrade, nullptr, nullptr, nullptr) == SQLITE_OK,
            "missing-head fixture constructs an exact v4 schema without its head");
      identity_before = scalar(
          database, "SELECT value FROM journal_meta WHERE key='journal_id'");
      sqlite3_close(database);
      database = nullptr;
    }

    bool rejected = false;
    try {
      trainvm::Journal journal(path);
    } catch (const std::runtime_error& exception) {
      rejected = std::string(exception.what()).find("authority metadata") !=
                 std::string::npos;
    }
    check(rejected, "v4 " + fixture_name +
                        " journal with a missing chain head is refused");
    check(sqlite3_open(path.c_str(), &database) == SQLITE_OK,
          "refused missing-head v4 journal remains inspectable");
    if (database != nullptr) {
      check(scalar(database,
                   "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                    "4" &&
                scalar(database,
                       "SELECT COUNT(*) FROM journal_meta WHERE key='chain_head'") ==
                    "0" &&
                scalar(database, "SELECT COUNT(*) FROM events") ==
                    (with_event ? "1" : "0") &&
                scalar(database, R"sql(
                  SELECT group_concat(name, ',')
                  FROM pragma_table_info('resource_leases')
                )sql") ==
                    "concurrency_key,owner_run_id,lease_id,fencing_token,acquired_at_ns,expires_at_ns,released_at_ns" &&
                scalar(database,
                       "SELECT value FROM journal_meta WHERE key='journal_id'") ==
                    identity_before &&
                scalar(database, "PRAGMA journal_mode") == "delete",
            "missing-head v4 refusal preserves version, schema, events, identity, and journal mode without repair");
      sqlite3_close(database);
      database = nullptr;
    }
  }

  const auto v4_invalid_chain_path = directory / "v4-invalid-chain.db";
  {
    trainvm::Journal fresh(v4_invalid_chain_path);
    check(trainvm::JournalTestAccess::append(
              fresh, created_event(std::string(64U, 'd'))) == 1U,
          "invalid-chain v4 fixture starts with a valid event");
  }
  check(sqlite3_open(v4_invalid_chain_path.c_str(), &database) == SQLITE_OK,
        "invalid-chain v4 fixture opens for exact downgrade");
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
      BEGIN IMMEDIATE;
      ALTER TABLE resource_leases RENAME TO resource_leases_v5;
      CREATE TABLE resource_leases (
        concurrency_key TEXT PRIMARY KEY,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        acquired_at_ns INTEGER NOT NULL,
        expires_at_ns INTEGER NOT NULL,
        released_at_ns INTEGER
      ) WITHOUT ROWID;
      DROP TABLE resource_leases_v5;
      ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
      CREATE TABLE resource_lease_releases (
        concurrency_key TEXT NOT NULL,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        released_at_ns INTEGER NOT NULL,
        PRIMARY KEY(concurrency_key, lease_id, fencing_token)
      ) WITHOUT ROWID;
      DROP TABLE resource_lease_releases_v5;
      UPDATE journal_meta SET value='4' WHERE key='schema_version';
      UPDATE events SET payload_json='{}' WHERE journal_sequence=1;
      COMMIT;
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "invalid-chain fixture corrupts history behind an exact v4 schema");
    sqlite3_close(database);
    database = nullptr;
  }
  bool invalid_chain_refused = false;
  try {
    trainvm::Journal journal(v4_invalid_chain_path);
  } catch (const std::runtime_error& exception) {
    invalid_chain_refused =
        std::string(exception.what()).find("invalid event chain") !=
        std::string::npos;
  }
  check(invalid_chain_refused,
        "v4 migration refuses a corrupt event chain before changing schema version");
  check(sqlite3_open(v4_invalid_chain_path.c_str(), &database) == SQLITE_OK,
        "refused invalid-chain v4 journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                  "4" &&
              scalar(database,
                     "SELECT payload_json FROM events WHERE journal_sequence=1") ==
                  "{}" &&
              scalar(database, R"sql(
                SELECT group_concat(name, ',')
                FROM pragma_table_info('resource_leases')
              )sql") ==
                  "concurrency_key,owner_run_id,lease_id,fencing_token,acquired_at_ns,expires_at_ns,released_at_ns",
          "invalid-chain refusal preserves v4 schema and corrupted evidence for inspection");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto v5_missing_head_path = directory / "v5-missing-head.db";
  {
    trainvm::Journal fresh(v5_missing_head_path);
    check(trainvm::JournalTestAccess::append(
              fresh, created_event(std::string(64U, 'c'))) == 1U,
          "missing-head v5 fixture creates a valid event chain");
  }
  check(sqlite3_open(v5_missing_head_path.c_str(), &database) == SQLITE_OK,
        "missing-head v5 fixture opens for corruption");
  if (database != nullptr) {
    check(sqlite3_exec(database,
                       "DELETE FROM journal_meta WHERE key='chain_head'", nullptr,
                       nullptr, nullptr) == SQLITE_OK,
          "missing-head v5 fixture removes required authority metadata");
    sqlite3_close(database);
    database = nullptr;
  }
  bool v5_missing_head_refused = false;
  try {
    trainvm::Journal journal(v5_missing_head_path);
  } catch (const std::runtime_error& exception) {
    v5_missing_head_refused =
        std::string(exception.what()).find("authority metadata") !=
        std::string::npos;
  }
  check(v5_missing_head_refused,
        "established v5 journal never synthesizes a missing chain head");
  check(sqlite3_open(v5_missing_head_path.c_str(), &database) == SQLITE_OK,
        "refused missing-head v5 journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                  "5" &&
              scalar(database,
                     "SELECT COUNT(*) FROM journal_meta WHERE key='chain_head'") ==
                  "0" &&
              scalar(database, "SELECT COUNT(*) FROM events") == "1",
          "missing-head v5 refusal preserves history without metadata repair");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto v4_path = directory / "v4-quarantine.db";
  {
    trainvm::Journal fresh(v4_path);
    check(trainvm::JournalTestAccess::append(
              fresh, created_event(std::string(64U, 'a'))) == 1U,
          "v4 quarantine fixture starts with a valid event chain");
  }
  check(sqlite3_open(v4_path.c_str(), &database) == SQLITE_OK,
        "v4 quarantine test opens a fresh v5 fixture");
  std::string event_before;
  std::string head_before;
  if (database != nullptr) {
    const char* downgrade = R"sql(
      BEGIN IMMEDIATE;
      ALTER TABLE resource_leases RENAME TO resource_leases_v5;
      CREATE TABLE resource_leases (
        concurrency_key TEXT PRIMARY KEY,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        acquired_at_ns INTEGER NOT NULL,
        expires_at_ns INTEGER NOT NULL,
        released_at_ns INTEGER
      ) WITHOUT ROWID;
      INSERT INTO resource_leases VALUES
        ('legacy-released', 'old-run', 'old-released', 7, 10, 20, 30),
        ('legacy-future', 'old-run', 'old-live', 9, 100,
         9223372036854775807, NULL);
      DROP TABLE resource_leases_v5;

      ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
      CREATE TABLE resource_lease_releases (
        concurrency_key TEXT NOT NULL,
        owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        fencing_token INTEGER NOT NULL,
        released_at_ns INTEGER NOT NULL,
        PRIMARY KEY(concurrency_key, lease_id, fencing_token)
      ) WITHOUT ROWID;
      INSERT INTO resource_lease_releases VALUES
        ('legacy-released', 'old-run', 'old-released', 7, 30);
      DROP TABLE resource_lease_releases_v5;

      UPDATE journal_meta SET value='4' WHERE key='schema_version';
      COMMIT;
    )sql";
    check(sqlite3_exec(database, downgrade, nullptr, nullptr, nullptr) == SQLITE_OK,
          "v4 quarantine test constructs a complete established v4 journal");
    event_before = scalar(database, R"sql(
      SELECT quote(journal_sequence)||'|'||quote(event_id)||'|'||quote(run_id)||'|'||
             quote(run_revision)||'|'||quote(plan_revision)||'|'||quote(node_id)||'|'||
             quote(attempt_id)||'|'||quote(worker_sequence)||'|'||quote(event_type)||'|'||
             quote(event_version)||'|'||quote(wall_time_ns)||'|'||
             quote(monotonic_time_ns)||'|'||quote(optimizer_step)||'|'||
             quote(payload_json)||'|'||quote(previous_hash)||'|'||
             quote(content_hash)||'|'||quote(chain_hash)
      FROM events WHERE journal_sequence=1
    )sql");
    head_before = scalar(
        database, "SELECT value FROM journal_meta WHERE key='chain_head'");
    sqlite3_close(database);
    database = nullptr;
  }

  {
    trainvm::Journal migrated(v4_path);
    check(!migrated.active_lease("legacy-future", test_time(1, 999'999)) &&
              !migrated.renew_lease("legacy-future", "old-run", "old-live", 9,
                                    test_time(2, 1'000'000), 100) &&
              !migrated.release_lease("legacy-future", "old-run", "old-live", 9,
                                      test_time(3, 1'000'001)),
          "unreleased v4 leases are quarantined and never active, renewable, or releasable");
    const auto replacement = migrated.acquire_lease(
        "legacy-future", "new-run", "new-lease", test_time(50, 4'000), 25);
    check(replacement.status == trainvm::LeaseAcquireStatus::acquired &&
              replacement.lease.fencing_token == 10U &&
              replacement.lease.clock_domain == trainvm::ResourceLease::kBootTimeDomain &&
              replacement.lease.boot_id == kTestBootId &&
              replacement.lease.acquired_boottime_ns == 50 &&
              replacement.lease.expires_boottime_ns == 75 &&
              replacement.lease.acquired_wall_time_ns == 4'000 &&
              replacement.lease.expires_wall_time_ns == 4'025,
          "a boot-scoped acquisition supersedes quarantined future-wall authority");
  }
  check(sqlite3_open(v4_path.c_str(), &database) == SQLITE_OK,
        "migrated v5 journal remains directly inspectable");
  if (database != nullptr) {
    const std::string version = scalar(
        database, "SELECT value FROM journal_meta WHERE key='schema_version'");
    const std::string released_history = scalar(database, R"sql(
      SELECT clock_domain||'|'||quote(boot_id)||'|'||
             quote(acquired_boottime_ns)||'|'||quote(expires_boottime_ns)||'|'||
             acquired_wall_time_ns||'|'||expires_wall_time_ns||'|'||
             released_wall_time_ns
      FROM resource_leases WHERE concurrency_key='legacy-released'
    )sql");
    const std::string release_receipt = scalar(database, R"sql(
      SELECT clock_domain||'|'||quote(boot_id)||'|'||released_wall_time_ns
      FROM resource_lease_releases WHERE concurrency_key='legacy-released'
    )sql");
    const std::string event_after = scalar(database, R"sql(
      SELECT quote(journal_sequence)||'|'||quote(event_id)||'|'||quote(run_id)||'|'||
             quote(run_revision)||'|'||quote(plan_revision)||'|'||quote(node_id)||'|'||
             quote(attempt_id)||'|'||quote(worker_sequence)||'|'||quote(event_type)||'|'||
             quote(event_version)||'|'||quote(wall_time_ns)||'|'||
             quote(monotonic_time_ns)||'|'||quote(optimizer_step)||'|'||
             quote(payload_json)||'|'||quote(previous_hash)||'|'||
             quote(content_hash)||'|'||quote(chain_hash)
      FROM events WHERE journal_sequence=1
    )sql");
    const std::string head_after = scalar(
        database, "SELECT value FROM journal_meta WHERE key='chain_head'");
    check(version == "5" && released_history == "legacy-wall/v1|NULL|NULL|NULL|10|20|30" &&
              release_receipt == "legacy-wall/v1|NULL|30",
          "v4 lease and release history migrates into explicit legacy-wall quarantine");
    check(event_after == event_before && head_after == head_before,
          "v4 migration preserves event records and chain head byte-for-byte");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto empty_v4_path = directory / "empty-v4.db";
  {
    trainvm::Journal fresh(empty_v4_path);
  }
  check(sqlite3_open(empty_v4_path.c_str(), &database) == SQLITE_OK,
        "empty v4 migration test opens its fixture");
  if (database != nullptr) {
    const char* downgrade = R"sql(
      BEGIN IMMEDIATE;
      ALTER TABLE resource_leases RENAME TO resource_leases_v5;
      CREATE TABLE resource_leases (
        concurrency_key TEXT PRIMARY KEY, owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL, fencing_token INTEGER NOT NULL,
        acquired_at_ns INTEGER NOT NULL, expires_at_ns INTEGER NOT NULL,
        released_at_ns INTEGER
      ) WITHOUT ROWID;
      DROP TABLE resource_leases_v5;
      ALTER TABLE resource_lease_releases RENAME TO resource_lease_releases_v5;
      CREATE TABLE resource_lease_releases (
        concurrency_key TEXT NOT NULL, owner_run_id TEXT NOT NULL,
        lease_id TEXT NOT NULL, fencing_token INTEGER NOT NULL,
        released_at_ns INTEGER NOT NULL,
        PRIMARY KEY(concurrency_key, lease_id, fencing_token)
      ) WITHOUT ROWID;
      DROP TABLE resource_lease_releases_v5;
      UPDATE journal_meta SET value='4' WHERE key='schema_version';
      COMMIT;
    )sql";
    check(sqlite3_exec(database, downgrade, nullptr, nullptr, nullptr) == SQLITE_OK,
          "empty v4 fixture uses the exact established lease schema");
    sqlite3_close(database);
    database = nullptr;
  }
  {
    trainvm::Journal migrated(empty_v4_path);
    const auto acquired = migrated.acquire_lease(
        "empty-v4-gpu", "new-run", "new-lease", test_time(10, 500), 20);
    check(acquired.status == trainvm::LeaseAcquireStatus::acquired &&
              acquired.lease.clock_domain == trainvm::ResourceLease::kBootTimeDomain,
          "empty established v4 journal safely migrates and accepts typed leases");
  }

  const auto partial_v5_path = directory / "partial-v5.db";
  check(sqlite3_open(partial_v5_path.c_str(), &database) == SQLITE_OK,
        "partial v5 test opens its fixture");
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
      CREATE TABLE journal_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
      INSERT INTO journal_meta VALUES('schema_version', '5');
      INSERT INTO journal_meta VALUES('journal_id', '0123456789abcdef0123456789abcdef');
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "partial v5 test creates trusted metadata without its required tables");
    sqlite3_close(database);
    database = nullptr;
  }
  bool partial_refused = false;
  try {
    trainvm::Journal journal(partial_v5_path);
  } catch (const std::runtime_error& exception) {
    partial_refused = std::string(exception.what()).find("v5 is partial") !=
                      std::string::npos;
  }
  check(partial_refused, "partial v5 journals fail closed before schema repair");

  const auto malformed_v5_path = directory / "malformed-v5.db";
  {
    trainvm::Journal fresh(malformed_v5_path);
  }
  check(sqlite3_open(malformed_v5_path.c_str(), &database) == SQLITE_OK,
        "malformed v5 test opens its fixture");
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
      PRAGMA ignore_check_constraints=ON;
      INSERT INTO resource_leases VALUES(
        'bad-gpu', 'bad-run', 'bad-lease', 1, 'boottime/v1', 'not-a-boot-id',
        10, 20, 100, 110, NULL
      );
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "malformed v5 test injects invalid persisted authority data");
    sqlite3_close(database);
    database = nullptr;
  }
  bool malformed_refused = false;
  try {
    trainvm::Journal journal(malformed_v5_path);
  } catch (const std::runtime_error& exception) {
    malformed_refused = std::string(exception.what()).find(
                            "malformed lease authority data") != std::string::npos;
  }
  check(malformed_refused, "malformed v5 lease authority fails closed on open");
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  try {
    test_reflection_and_compiler();
    test_wire_contract();
    test_fsm();
    test_control_validation();
    test_journal();
    test_controller_and_fake_worker();
    test_compiled_plan_persistence();
    test_resource_leases();
    test_authority_clock_integration();
    test_authority_lock_file_identity();
    test_control_command_journal();
    test_command_service();
    test_submission_and_queue_boundary();
    test_atomic_queue_acquisition_boundary();
    test_worker_launch_and_readiness_boundary();
    test_worker_control_service_boundary();
    test_worker_control_grpc_stream();
    test_typed_managed_resource_release();
    test_concurrent_worker_launch_and_readiness_replay();
    test_concurrent_fenced_result_content_conflict();
    test_concurrent_queue_acquisition_replay();
    test_acquiring_recovery_ignores_mutable_lease_lifecycle();
    test_acquiring_rejects_fabricated_running_transition();
    test_host_launch_registry_contract();
    test_host_launch_resolution_and_binding();
    test_service_host_launch_binding();
    test_adapter_registry_file_contract();
    test_service_registry_and_reconciliation();
    test_adapter_registry_and_reconciler();
    test_legacy_journal_migration_policy();
  } catch (const std::exception& exception) {
    std::cerr << "UNCAUGHT: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all TrainVM tests passed\n";
  return 0;
}
