#include "trainvm/adapter_registry.hpp"
#include "trainvm/controller.hpp"
#include "trainvm/control.hpp"
#include "trainvm/document.hpp"
#include "trainvm/eval_examples_contract.hpp"
#include "trainvm/fake_worker.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/final_evaluation.hpp"
#include "trainvm/host_launch.hpp"
#include "trainvm/host_launch_registry.hpp"
#include "trainvm/hostd_mutation_claim_provider.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/lease_renewal.hpp"
#include "trainvm/model.hpp"
#include "trainvm/profiler_launch_profiles.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/reconciler.hpp"
#include "trainvm/service.hpp"
#include "trainvm/training_component_registry.hpp"
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
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "trainvm/json.hpp"

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

// Name of the case currently running, so that both an assertion failure and
// an uncaught exception can say which one it was.
std::string_view current_case = "<none>";

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
    std::cerr << "FAIL [" << current_case << "]: " << message << '\n';
    ++failures;
  }
}

trainvm::OperationPortDescriptor operation_port(
    trainvm::OperationPortType type, bool required,
    std::optional<trainvm::ArtifactType> artifact_type = std::nullopt,
    std::optional<std::string> artifact_schema = std::nullopt) {
  return {
      .type = type,
      .required = required,
      .artifact_type = artifact_type,
      .artifact_schema = std::move(artifact_schema),
      .description = std::nullopt,
  };
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

nlohmann::json cache_qualification_fixture() {
  nlohmann::json document = load_fixture();
  document["spec"]["components"]["core"]["operations"]["qualify_cache"] = {
      {"contract", "trainvm.v1.QualifyCache"}};
  document["spec"]["workflow"]["nodes"]["qualify_cache_artifact"] = {
      {"description", "Gate the published cache artifact before release."},
      {"invoke", {{"component", "core"}, {"operation", "qualify_cache"},
                  {"inputs", nlohmann::json::object()}}},
      {"idempotency", "replay_safe"},
      {"effect", "read_only"},
      {"transitions",
       nlohmann::json::array(
           {{{"on", "cache.qualified"}, {"target", "release_gpu"}},
            {{"on", "cache.rejected"}, {"target", "$failed"}}})}};
  document["spec"]["workflow"]["nodes"]["train_to_boundary"]["transitions"][1]
          ["target"] = "qualify_cache_artifact";
  return document;
}

// A self-comparison: the two arms ran the same implementation, so the measured
// deviation is exactly zero and `equivalent` still means bit-identity.
trainvm::TrajectoryParityEvidence bit_identical_trajectory() {
  return {
      .verdict = trainvm::TrajectoryParityVerdict::equivalent,
      .criterion = trainvm::TrajectoryEquivalenceCriterion::bit_identical,
      .effect_class = trainvm::TrajectoryEffectClass::optimizer_update,
      .candidate_divergence = {{.step = 1U, .relative_deviation = 0.0},
                               {.step = 2U, .relative_deviation = 0.0},
                               {.step = 5U, .relative_deviation = 0.0}},
      .reference_divergence = {},
      .checkpoint_quality = {},
      .analysis_seed = 0U,
  };
}

// Evidence that passes every implemented gate. Individual tests degrade one
// field at a time so a rejection reason is always attributable.
trainvm::CacheQualificationEvidence passing_cache_evidence() {
  const auto digest = [](char value) {
    return "sha256:" + std::string(64U, value);
  };
  return {
      .api_version = "trainvm.cache-qualification-evidence/v1",
      .authority_receipt_digest = digest('1'),
      .namespace_digest = digest('2'),
      .artifact_tree_digest = digest('3'),
      .workload_class = trainvm::CacheWorkloadClass::training,
      .baseline_run_digest = digest('4'),
      .candidate_run_digest = digest('5'),
      .shape_coverage_digest = digest('6'),
      .transition_coverage = true,
      .baseline_instrumented = false,
      .candidate_instrumented = false,
      .output_parity = true,
      .gradient_parity = true,
      .optimizer_update_parity = true,
      .state_parity = true,
      .optimizer_state_device_policy =
          trainvm::OptimizerStateDevicePolicy::normalized_on_load,
      .resumed_trajectory_parity = bit_identical_trajectory(),
      .determinism_parity = true,
      .content_parity = true,
      .ordering_parity = true,
      .manifest_parity = true,
      .model_quality_pass = true,
      .baseline_throughput = 100.0,
      .candidate_throughput = 125.0,
      .baseline_peak_memory_bytes = 1'000'000U,
      .candidate_peak_memory_bytes = 1'020'000U,
      .minimum_throughput_gain_ratio = 0.10,
      .maximum_memory_regression_ratio = 0.05,
  };
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
       .required_capabilities = {},
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {{"concurrency_key",
                       operation_port(trainvm::OperationPortType::string,
                                      true)}},
           .outputs = {}}},
      {.key = key("trainvm.core", "1.0.0",
                  trainvm::ComponentRuntime::builtin, "validate_artifact",
                  "trainvm.v1.ValidateArtifact"),
       .effect = trainvm::Effect::read_only,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {},
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {
               {"artifact",
                operation_port(trainvm::OperationPortType::artifact, true)},
               {"required_schema",
                operation_port(trainvm::OperationPortType::string, true)},
           },
           .outputs = {}}},
      {.key = key("trainvm.core", "1.0.0",
                  trainvm::ComponentRuntime::builtin, "release_resources",
                  "trainvm.v1.ReleaseResources"),
       .effect = trainvm::Effect::resource,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = {},
       .required_capabilities = {},
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {{"concurrency_key",
                       operation_port(trainvm::OperationPortType::string,
                                      true)}},
           .outputs = {}}},
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker, "train",
                  "rwkv_lab.mageflow.v1.Train"),
       .effect = trainvm::Effect::process,
       .idempotency = trainvm::Idempotency::receipt_required,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"},
       .lifecycle = {
           .stateful = true,
           .graceful_stop = true,
           .checkpoint_now = true,
           .pause_keep_resources = true,
           .pause_release_resources = true,
           .compile = true,
           .warmup = true,
           .qualify = true,
           .profile = true,
           .resume_grade = trainvm::ResumeGrade::exact,
       },
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {
               {"checkpoint",
                operation_port(trainvm::OperationPortType::artifact, false,
                               trainvm::ArtifactType::checkpoint,
                               "rwkv-lab.mageflow-checkpoint.v1")},
               {"config",
                operation_port(trainvm::OperationPortType::string, true)},
               {"encoder_cache",
                operation_port(trainvm::OperationPortType::artifact, false,
                               trainvm::ArtifactType::dataset,
                               "rwkv-lab.encoder-cache.v1")},
               {"run_directory",
                operation_port(trainvm::OperationPortType::string, true)},
               {"stop_at_step",
                operation_port(trainvm::OperationPortType::integer, false)},
           },
           .outputs = {
               {"checkpoint",
                operation_port(trainvm::OperationPortType::artifact, false,
                               trainvm::ArtifactType::checkpoint,
                               "rwkv-lab.mageflow-checkpoint.v1")},
               {"eval_gallery",
                operation_port(trainvm::OperationPortType::artifact, false,
                               trainvm::ArtifactType::image_gallery,
                               "rwkv-lab.eval-gallery.v2")},
               {"log",
               operation_port(trainvm::OperationPortType::artifact, false,
                               trainvm::ArtifactType::path)},
           },
       }},
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker,
                  "prepare_cache_span",
                  "rwkv_lab.mageflow.v1.PrepareCacheSpan"),
       .effect = trainvm::Effect::workspace_write,
       .idempotency = trainvm::Idempotency::replay_safe,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"},
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {
               {"checkpoint",
                operation_port(trainvm::OperationPortType::artifact, true,
                               trainvm::ArtifactType::checkpoint,
                               "rwkv-lab.mageflow-checkpoint.v1")},
               {"config",
                operation_port(trainvm::OperationPortType::string, true)},
               {"final_step",
                operation_port(trainvm::OperationPortType::integer, true)},
               {"output_directory",
                operation_port(trainvm::OperationPortType::string, true)},
           },
           .outputs = {{"plan",
                        operation_port(trainvm::OperationPortType::artifact,
                                       false,
                                       trainvm::ArtifactType::report,
                                       "rwkv-lab.mageflow-cache-plan.v1")}}},
       },
      {.key = key("rwkv-lab.mageflow", "1.0.0",
                  trainvm::ComponentRuntime::python_worker, "cache_encoders",
                  "rwkv_lab.mageflow.v1.CacheEncoders"),
       .effect = trainvm::Effect::process,
       .idempotency = trainvm::Idempotency::receipt_required,
       .code_fingerprint = worker_fingerprint,
       .required_capabilities = {"worker.metrics", "worker.controls"},
       .lifecycle = {
           .stateful = false,
           .graceful_stop = true,
           .checkpoint_now = false,
           .pause_keep_resources = false,
           .pause_release_resources = false,
           .compile = false,
           .warmup = false,
           .qualify = false,
           .profile = false,
           .resume_grade = trainvm::ResumeGrade::none,
       },
       .authoring = trainvm::OperationAuthoringDeclaration{
           .inputs = {
               {"output_directory",
                operation_port(trainvm::OperationPortType::string, true)},
               {"plan",
                operation_port(trainvm::OperationPortType::artifact, true,
                               trainvm::ArtifactType::report,
                               "rwkv-lab.mageflow-cache-plan.v1")},
           },
           .outputs = {{"cache",
                        operation_port(trainvm::OperationPortType::artifact,
                                       false,
                                       trainvm::ArtifactType::dataset,
                                       "rwkv-lab.encoder-cache.v1")}}},
       },
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
      .api_version = "trainvm.host-launches/v4",
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

trainvm::TrainingComponentRegistry fixture_training_component_registry() {
  return trainvm::TrainingComponentRegistry({{
      .key = {.category = trainvm::TrainingComponentCategory::activation,
              .name = "silu",
              .version = "1.0.0"},
      .backend = trainvm::TrainingComponentBackend::runtime_builtin,
      .implementation = "runtime.activation.silu",
      .model_families = {"mageflow"},
      .required_capabilities = {"activation.silu"},
      .configuration = {},
      .state = {},
      .step_domain = std::nullopt,
      .state_grade = trainvm::TrainingStateGrade::stateless,
      .reference_implementation = true,
  }});
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
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {"/test"},
      .profiles = {{
          .key = key,
          .code_fingerprint = launch.code_fingerprint,
          .bootstrap_runtime_closure_fingerprint =
              "sha256:" + std::string(64U, 'd'),
          .provided_capabilities = launch.required_capabilities,
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
  if (!binding) {
    throw std::runtime_error("test service launch binding is missing");
  }
  if (binding->identity.host != service.authority_host_) {
    throw std::runtime_error(
        "test service host authority disagrees with durable binding");
  }
  if (binding->identity.host_registry_digest !=
      service.host_launch_registry_.registry_digest()) {
    throw std::runtime_error(
        "test service host registry disagrees with durable binding: " +
        binding->identity.host_registry_digest + " != " +
        service.host_launch_registry_.registry_digest());
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
    const trainvm::WorkerLaunchTicket& launch, std::int64_t now_ns,
    std::optional<trainvm::HostIdentity> selected_host = std::nullopt) {
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
      .bootstrap_runtime_closure_fingerprint =
          "sha256:" + std::string(64U, 'd'),
      .provided_capabilities = launch.required_capabilities,
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
  const trainvm::HostIdentity host =
      selected_host.value_or(fixture_test_host_identity());
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
      .api_version = "trainvm.resolved-launch/v4",
      .launch_event_id = launch.run_id + ":worker-launch:" + launch.node_id +
                         ":" + launch.attempt_id,
      .run_id = launch.run_id,
      .node_id = launch.node_id,
      .attempt_id = launch.attempt_id,
      .launch_nonce = launch.launch_nonce,
      .adapter_key = key,
      .code_fingerprint = launch.code_fingerprint,
      .bootstrap_runtime_closure_fingerprint =
          profile.bootstrap_runtime_closure_fingerprint,
      .required_capabilities = launch.required_capabilities,
      .provided_capabilities = launch.required_capabilities,
      .host_registry_digest = registry.registry_digest(),
      .host_profile_digest =
          registry.profile_digest(key, launch.code_fingerprint),
      .concurrency_key = launch.concurrency_key,
      .lease_id = launch.lease_id,
      .fencing_token = launch.fencing_token,
      .host_grant = launch.host_grant,
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
                    "record_shapes", "profile_memory", "with_stack"}) &&
            trainvm::reflected_field_names<
                trainvm::OperationLifecycleCapabilities>() ==
                std::vector<std::string>({
                    "stateful", "graceful_stop", "checkpoint_now",
                    "pause_keep_resources", "pause_release_resources",
                    "compile", "warmup", "qualify", "profile",
                    "resume_grade"}) &&
            trainvm::reflected_field_names<
                trainvm::TrainingCompositionContract>() ==
                std::vector<std::string>(
                    {"model_family", "slots", "allowed_components"}) &&
            trainvm::enum_from_string<trainvm::ResumeGrade>("exact") ==
                trainvm::ResumeGrade::exact &&
            trainvm::enum_to_string(
                trainvm::ResumeGrade::terminal_checkpoint) ==
                "terminal_checkpoint",
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
  check(result.plan->plan_hash == "ac64a8117668f19c4f4f1d131b07388122207043894d85e60ead25270f28085e",
        "MageFlow canonical plan matches its golden SHA-256 identity");
  check(result.plan->canonical_plan["spec"]["controls"]["catalog"]["learning_rate"].contains("default"),
        "canonical plan uses schema field aliases");
  check(!result.plan->canonical_plan["spec"]["controls"]["catalog"]["learning_rate"].contains("default_value"),
        "canonical plan does not leak C++ keyword workarounds");

  check(result.plan->experiment.spec.workflow.nodes.at("resume_training")
                .loop_guard.has_value(),
        "resource lifecycle validation accepts branched and cyclic paths when "
        "every completion route passes through the exact release node");

  auto direct_external_entry = fixture;
  direct_external_entry["spec"]["workflow"]["entrypoint"] =
      "train_to_boundary";
  const auto direct_external_entry_result =
      trainvm::compile_document(direct_external_entry);
  check(!direct_external_entry_result.valid() &&
            has_diagnostic(direct_external_entry_result,
                           "workflow.resource_admission"),
        "semantic compiler rejects a direct external worker entrypoint");

  auto wrong_admission_component = fixture;
  wrong_admission_component["spec"]["components"]["alternate_core"] =
      wrong_admission_component["spec"]["components"]["core"];
  wrong_admission_component["spec"]["components"]["alternate_core"]
                           ["adapter"] = "rwkv-lab.alternate-core";
  wrong_admission_component["spec"]["workflow"]["nodes"]["acquire_gpu"]
                           ["invoke"]["component"] = "alternate_core";
  const auto wrong_admission_component_result =
      trainvm::compile_document(wrong_admission_component);
  check(!wrong_admission_component_result.valid() &&
            has_diagnostic(wrong_admission_component_result,
                           "workflow.resource_admission"),
        "semantic compiler rejects an admission operation from another component");

  auto wrong_admission_version = fixture;
  wrong_admission_version["spec"]["components"]["core"]["version"] =
      "1.0.1";
  const auto wrong_admission_version_result =
      trainvm::compile_document(wrong_admission_version);
  check(!wrong_admission_version_result.valid() &&
            has_diagnostic(wrong_admission_version_result,
                           "workflow.resource_admission"),
        "semantic compiler pins resource admission to trainvm.core 1.0.0");

  auto wrong_admission_operation = fixture;
  wrong_admission_operation["spec"]["workflow"]["nodes"]["acquire_gpu"]
                           ["invoke"]["operation"] = "validate_artifact";
  const auto wrong_admission_operation_result =
      trainvm::compile_document(wrong_admission_operation);
  check(!wrong_admission_operation_result.valid() &&
            has_diagnostic(wrong_admission_operation_result,
                           "workflow.resource_admission"),
        "semantic compiler requires the exact acquire_resources operation");

  auto wrong_admission_contract = fixture;
  wrong_admission_contract["spec"]["components"]["core"]["operations"]
                          ["acquire_resources"]["contract"] =
      "trainvm.v1.NotAcquireResources";
  const auto wrong_admission_contract_result =
      trainvm::compile_document(wrong_admission_contract);
  check(!wrong_admission_contract_result.valid() &&
            has_diagnostic(wrong_admission_contract_result,
                           "workflow.resource_admission"),
        "semantic compiler pins the typed resource acquisition contract");

  auto wrong_admission_effect = fixture;
  wrong_admission_effect["spec"]["workflow"]["nodes"]["acquire_gpu"]
                        ["effect"] = "process";
  const auto wrong_admission_effect_result =
      trainvm::compile_document(wrong_admission_effect);
  check(!wrong_admission_effect_result.valid() &&
            has_diagnostic(wrong_admission_effect_result,
                           "workflow.resource_admission"),
        "semantic compiler requires resource effect for admission");

  auto wrong_admission_idempotency = fixture;
  wrong_admission_idempotency["spec"]["workflow"]["nodes"]["acquire_gpu"]
                             ["idempotency"] = "replay_safe";
  const auto wrong_admission_idempotency_result =
      trainvm::compile_document(wrong_admission_idempotency);
  check(!wrong_admission_idempotency_result.valid() &&
            has_diagnostic(wrong_admission_idempotency_result,
                           "workflow.resource_admission"),
        "semantic compiler requires receipt-backed resource admission");

  auto missing_acquired_transition = fixture;
  missing_acquired_transition["spec"]["workflow"]["nodes"]["acquire_gpu"]
                             ["transitions"][0]["on"] =
      "resource.unexpected";
  const auto missing_acquired_transition_result =
      trainvm::compile_document(missing_acquired_transition);
  check(!missing_acquired_transition_result.valid() &&
            has_diagnostic(missing_acquired_transition_result,
                           "workflow.resource_admission_transition"),
        "semantic compiler requires the resource.acquired admission event");

  auto multiple_acquired_transitions = fixture;
  multiple_acquired_transitions["spec"]["workflow"]["nodes"]["acquire_gpu"]
                               ["transitions"].push_back(
      {{"on", "resource.acquired"}, {"target", "train_to_boundary"}});
  const auto multiple_acquired_transitions_result =
      trainvm::compile_document(multiple_acquired_transitions);
  check(!multiple_acquired_transitions_result.valid() &&
            has_diagnostic(multiple_acquired_transitions_result,
                           "workflow.resource_admission_transition"),
        "semantic compiler rejects multiple resource.acquired routes");

  auto conditional_acquired_transition = fixture;
  conditional_acquired_transition["spec"]["workflow"]["nodes"]["acquire_gpu"]
                                 ["transitions"][0]["where"] =
      {{"field", "payload.ready"}, {"operator", "eq"}, {"value", true}};
  const auto conditional_acquired_transition_result =
      trainvm::compile_document(conditional_acquired_transition);
  check(!conditional_acquired_transition_result.valid() &&
            has_diagnostic(conditional_acquired_transition_result,
                           "workflow.resource_admission_transition"),
        "semantic compiler rejects conditional resource admission routing");

  auto terminal_acquired_transition = fixture;
  terminal_acquired_transition["spec"]["workflow"]["nodes"]["acquire_gpu"]
                              ["transitions"][0]["target"] = "$failed";
  const auto terminal_acquired_transition_result =
      trainvm::compile_document(terminal_acquired_transition);
  check(!terminal_acquired_transition_result.valid() &&
            has_diagnostic(terminal_acquired_transition_result,
                           "workflow.resource_admission_target"),
        "semantic compiler rejects an alternate terminal admission route");

  auto builtin_acquired_target = fixture;
  builtin_acquired_target["spec"]["workflow"]["nodes"]["acquire_gpu"]
                         ["transitions"][0]["target"] = "validate_cache";
  const auto builtin_acquired_target_result =
      trainvm::compile_document(builtin_acquired_target);
  check(!builtin_acquired_target_result.valid() &&
            has_diagnostic(builtin_acquired_target_result,
                           "workflow.resource_admission_target"),
        "semantic compiler requires admission to enter a non-builtin worker node");

  auto completion_without_release = fixture;
  completion_without_release["spec"]["workflow"]["nodes"]
                            ["train_to_boundary"]["transitions"][1]["target"] =
      "$completed";
  const auto completion_without_release_result =
      trainvm::compile_document(completion_without_release);
  check(!completion_without_release_result.valid() &&
            has_diagnostic(completion_without_release_result,
                           "workflow.resource_release"),
        "semantic compiler rejects any completion branch that bypasses release");

  auto malformed_release = fixture;
  malformed_release["spec"]["workflow"]["nodes"]["release_gpu"]["effect"] =
      "read_only";
  const auto malformed_release_result =
      trainvm::compile_document(malformed_release);
  check(!malformed_release_result.valid() &&
            has_diagnostic(malformed_release_result,
                           "workflow.resource_release"),
        "semantic compiler counts only the exact builtin release operation");

  // Workspace authority drift. spec.workspace names the fence this run holds
  // and the one tree its worker may write; the workflow restates both. A
  // locked document that answers the same question twice, differently, was
  // accepted, and the disagreement only surfaced at launch.
  const auto drifted_concurrency = [&](const char* node, nlohmann::json value) {
    auto drifted = fixture;
    drifted["spec"]["workflow"]["nodes"][node]["invoke"]["inputs"]
           ["concurrency_key"] = std::move(value);
    return trainvm::compile_document(drifted);
  };
  const auto rejects_concurrency_drift = [&](const trainvm::CompileResult&
                                                 compiled) {
    return !compiled.valid() &&
           has_diagnostic(compiled, "workspace.concurrency_authority");
  };

  check(rejects_concurrency_drift(drifted_concurrency(
            "acquire_gpu", {{"literal", "some-other-fence"}})),
        "compiler rejects an acquire literal that no longer names the "
        "workspace concurrency key");
  check(rejects_concurrency_drift(drifted_concurrency(
            "release_gpu", {{"literal", "some-other-fence"}})),
        "compiler rejects a release literal that no longer names the "
        "workspace concurrency key");
  check(rejects_concurrency_drift(
            drifted_concurrency("acquire_gpu", {{"literal", 7}})),
        "compiler rejects a resource literal that is not even a string");
  check(rejects_concurrency_drift(drifted_concurrency(
            "acquire_gpu", {{"parameter", "source_config"}})),
        "compiler rejects a resource fence deferred to a parameter binding");

  auto renamed_workspace_key = fixture;
  renamed_workspace_key["spec"]["workspace"]["concurrency_key"] =
      "renamed-fence";
  const auto renamed_workspace_key_result =
      trainvm::compile_document(renamed_workspace_key);
  check(!renamed_workspace_key_result.valid() &&
            std::count_if(renamed_workspace_key_result.diagnostics.begin(),
                          renamed_workspace_key_result.diagnostics.end(),
                          [](const trainvm::Diagnostic& diagnostic) {
                            return diagnostic.code ==
                                   "workspace.concurrency_authority";
                          }) == 2,
        "renaming only the workspace fence is refused at both resource nodes");

  auto renamed_everywhere = renamed_workspace_key;
  for (const char* resource_node : {"acquire_gpu", "release_gpu"}) {
    renamed_everywhere["spec"]["workflow"]["nodes"][resource_node]["invoke"]
                      ["inputs"]["concurrency_key"]["literal"] =
        "renamed-fence";
  }
  const auto renamed_everywhere_result =
      trainvm::compile_document(renamed_everywhere);
  check(renamed_everywhere_result.valid(),
        "a fence renamed in every place it is stated still compiles");

  // The locked trainer config restates the run directory. The Python worker
  // refuses a mismatch when it launches, which is after the plan has been
  // hashed and submitted; the compiler refuses it while it is still a document.
  const std::string fixture_run_directory =
      fixture["spec"]["workspace"]["run_directory"].get<std::string>();
  const auto with_locked_config = [&](const char* field, nlohmann::json value) {
    auto locked = fixture;
    locked["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
          ["inputs"]["config"] = {
              {"literal", {{field, std::move(value)}, {"steps", 10}}}};
    return trainvm::compile_document(locked);
  };
  const auto rejects_run_directory_drift =
      [&](const trainvm::CompileResult& compiled) {
        return !compiled.valid() &&
               has_diagnostic(compiled, "workspace.run_directory_authority");
      };

  check(with_locked_config("output_dir", fixture_run_directory).valid(),
        "a locked trainer config naming the declared run directory compiles");
  check(rejects_run_directory_drift(with_locked_config(
            "output_dir", fixture_run_directory + "-v2")),
        "compiler rejects a locked output_dir that drifted from the workspace "
        "run directory");
  check(rejects_run_directory_drift(with_locked_config(
            "output_dir", fixture_run_directory + "/nested")),
        "run directory authority is exact, so even a child of it is refused");
  check(rejects_run_directory_drift(
            with_locked_config("output_dir", nlohmann::json::object())),
        "compiler rejects a locked output_dir that is not a path string");
  check(rejects_run_directory_drift(
            with_locked_config("run_dir", fixture_run_directory + "-v2")),
        "the same authority holds for a config that spells the field run_dir");
  check(with_locked_config("run_dir", fixture_run_directory).valid(),
        "a locked run_dir naming the declared run directory compiles");

  auto moved_run_directory = fixture;
  moved_run_directory["spec"]["workflow"]["nodes"]["train_to_boundary"]
                     ["invoke"]["inputs"]["config"] = {
                         {"literal", {{"output_dir", fixture_run_directory}}}};
  moved_run_directory["spec"]["workspace"]["run_directory"] =
      fixture_run_directory + "-moved";
  moved_run_directory["spec"]["workspace"]["allowed_write_roots"][0] =
      fixture_run_directory + "-moved";
  const auto moved_run_directory_result =
      trainvm::compile_document(moved_run_directory);
  check(rejects_run_directory_drift(moved_run_directory_result),
        "moving the workspace run directory under a locked config is refused "
        "from the other direction too");

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

  auto wrong_trace_schema = fixture;
  wrong_trace_schema["spec"]["artifacts"]["gpu_trace"]["schema"] =
      "user.trace.v1";
  const auto wrong_trace_schema_result =
      trainvm::compile_document(wrong_trace_schema);
  check(!wrong_trace_schema_result.valid() &&
            has_diagnostic(wrong_trace_schema_result,
                           "execution.trace_artifact"),
        "GPU trace output requires the closed trace manifest contract");

  auto wrong_trace_fingerprint = fixture;
  wrong_trace_fingerprint["spec"]["artifacts"]["gpu_trace"]
                         ["fingerprint"] = "sha256";
  const auto wrong_trace_fingerprint_result =
      trainvm::compile_document(wrong_trace_fingerprint);
  check(!wrong_trace_fingerprint_result.valid() &&
            has_diagnostic(wrong_trace_fingerprint_result,
                           "execution.trace_artifact"),
        "GPU trace output requires adapter-owned tree fingerprinting");

  auto wrong_eval_gallery_type = fixture;
  wrong_eval_gallery_type["spec"]["artifacts"]["eval_gallery"]["type"] =
      "dataset";
  const auto wrong_eval_gallery_type_result =
      trainvm::compile_document(wrong_eval_gallery_type);
  check(!wrong_eval_gallery_type_result.valid() &&
            has_diagnostic(wrong_eval_gallery_type_result,
                           "observability.eval_gallery_artifact"),
        "eval gallery observability requires an image-gallery artifact");

  auto wrong_eval_gallery_schema = fixture;
  wrong_eval_gallery_schema["spec"]["artifacts"]["eval_gallery"]["schema"] =
      "coverage.custom-eval-gallery.v1";
  const auto wrong_eval_gallery_schema_result =
      trainvm::compile_document(wrong_eval_gallery_schema);
  check(!wrong_eval_gallery_schema_result.valid() &&
            has_diagnostic(wrong_eval_gallery_schema_result,
                           "observability.eval_gallery_artifact"),
        "eval gallery observability requires the common v2 gallery schema");

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
  auto valid_nsys = valid_ncu;
  valid_nsys["spec"]["execution"]["gpu_trace"]["backend"] = "nsys";
  const auto valid_nsys_result = trainvm::compile_document(valid_nsys);
  check(valid_nsys_result.valid(),
        "nsys is a typed backend with an honest accelerator-only summary surface");

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
  check(::chmod(directory.c_str(), 0700) == 0,
        "terminal release fixture protects its directory");
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

  trainvm::Event metric{
      .event_id = "metric-loss-1",
      .run_id = "run-1",
      .run_revision = 6,
      .plan_revision = 1,
      .node_id = "train",
      .attempt_id = "train@1",
      .worker_sequence = 1,
      .event_type = "metric.sampled",
      .event_version = 1,
      .wall_time_ns = 500,
      .monotonic_time_ns = 50,
      .optimizer_step = 1,
      .payload = {{"name", "train.loss"},
                  {"value", 2.0},
                  {"unit", "loss"},
                  {"step_domain", "optimizer_step"},
                  {"step", 1},
                  {"sample_weight", 1},
                  {"labels", {{"route", "animation"}}}},
  };
  auto sparse_metric = metric;
  sparse_metric.event_id = "metric-eval-1";
  sparse_metric.worker_sequence = 2;
  sparse_metric.wall_time_ns = 510;
  sparse_metric.monotonic_time_ns = 51;
  sparse_metric.payload["name"] = "eval.quality";
  auto latest_metric = metric;
  latest_metric.event_id = "metric-loss-2";
  latest_metric.worker_sequence = 3;
  latest_metric.wall_time_ns = 520;
  latest_metric.monotonic_time_ns = 52;
  latest_metric.optimizer_step = 2;
  latest_metric.payload["value"] = 1.5;
  latest_metric.payload["step"] = 2;
  const auto metric_sequences = trainvm::JournalTestAccess::append_batch(
      journal, {metric, sparse_metric, latest_metric});
  const auto latest_series = journal.sequenced_events({
      .after_journal_sequence = 0U,
      .through_journal_sequence = metric_sequences.back(),
      .run_ids = {"run-1"},
      .event_types = {"metric.sampled"},
      .limit = 8U,
      .newest_per_metric_series = true,
  });
  check(latest_series.size() == 2U &&
            latest_series[0].event.payload.at("name") == "eval.quality" &&
            latest_series[1].event.payload.at("name") == "train.loss" &&
            latest_series[1].journal_sequence == metric_sequences.back(),
        "latest-per-metric-series scan preserves sparse series and selects each newest sample");
  bool incoherent_metric_scan_rejected = false;
  try {
    (void)journal.sequenced_events({
        .after_journal_sequence = 0U,
        .through_journal_sequence = metric_sequences.back(),
        .run_ids = {"run-1"},
        .event_types = {"metric.sampled"},
        .limit = 8U,
        .newest_first = true,
        .newest_per_metric_series = true,
    });
  } catch (const std::invalid_argument&) {
    incoherent_metric_scan_rejected = true;
  }
  check(incoherent_metric_scan_rejected,
        "latest-per-metric-series scan rejects incompatible newest-first semantics");

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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, "persisted-plan-run");
    controller.create();
    const auto stored = journal.compiled_plan(compiled.plan->plan_hash);
    check(stored && stored->plan_hash == compiled.plan->plan_hash &&
              stored->canonical_plan == compiled.plan->canonical_plan,
          "run creation atomically stores its content-addressed canonical plan");
  }
  {
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

void test_lease_renewal_authority() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-lease-renewal-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto scalar = [](sqlite3* connection, const char* sql) {
    sqlite3_stmt* statement = nullptr;
    if (sqlite3_prepare_v2(connection, sql, -1, &statement, nullptr) !=
        SQLITE_OK) {
      throw std::runtime_error("could not prepare renewal fixture query");
    }
    std::string value;
    if (sqlite3_step(statement) == SQLITE_ROW &&
        sqlite3_column_type(statement, 0) != SQLITE_NULL) {
      value = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
    }
    sqlite3_finalize(statement);
    return value;
  };

  const auto exact_path = directory / "exact.db";
  {
    trainvm::Journal journal(exact_path);
    const auto acquired = journal.acquire_lease(
        "exact-gpu", "exact-run", "exact-lease", test_time(100, 1'000),
        100);
    const auto renewed = journal.renew_lease_exact(
        acquired.lease, test_time(180, 1'080), 100);
    check(renewed.status == trainvm::LeaseRenewalStatus::renewed &&
              renewed.receipt &&
              renewed.receipt->concurrency_key == "exact-gpu" &&
              renewed.receipt->owner_run_id == "exact-run" &&
              renewed.receipt->lease_id == "exact-lease" &&
              renewed.receipt->fencing_token == 1U &&
              renewed.receipt->clock_domain ==
                  trainvm::ResourceLease::kBootTimeDomain &&
              renewed.receipt->boot_id == kTestBootId &&
              renewed.receipt->acquired_boottime_ns == 100 &&
              renewed.receipt->acquired_wall_time_ns == 1'000 &&
              renewed.receipt->prior_expires_boottime_ns == 200 &&
              renewed.receipt->new_expires_boottime_ns == 280 &&
              renewed.receipt->prior_expires_wall_time_ns == 1'100 &&
              renewed.receipt->new_expires_wall_time_ns == 1'180 &&
              renewed.receipt->renewed_boottime_ns == 180 &&
              renewed.receipt->renewed_wall_time_ns == 1'080,
          "exact renewal atomically returns its complete durable authority receipt");

    const auto replayed = journal.renew_lease_exact(
        acquired.lease, test_time(180, 1'080), 100);
    check(replayed.status == trainvm::LeaseRenewalStatus::replayed &&
              replayed.receipt == renewed.receipt,
          "an exact same-sample and same-timeout renewal retry replays one receipt");

    auto wrong_acquisition = acquired.lease;
    --wrong_acquisition.acquired_boottime_ns;
    bool acquisition_identity_conflicted = false;
    try {
      (void)journal.renew_lease_exact(
          wrong_acquisition, test_time(180, 1'080), 100);
    } catch (const std::runtime_error& exception) {
      acquisition_identity_conflicted =
          std::string(exception.what()).find("replay conflicts") !=
          std::string::npos;
    }
    check(acquisition_identity_conflicted,
          "exact renewal replay binds the acquisition timestamps as well as expiry and fence");

    bool later_sample_conflicted = false;
    try {
      (void)journal.renew_lease_exact(acquired.lease,
                                      test_time(181, 1'081), 100);
    } catch (const std::runtime_error& exception) {
      later_sample_conflicted =
          std::string(exception.what()).find("replay conflicts") !=
          std::string::npos;
    }
    const auto current = journal.active_lease("exact-gpu", test_time(182));
    check(later_sample_conflicted && current &&
              current->expires_boottime_ns == 280 &&
              current->expires_wall_time_ns == 1'180,
          "a later clock sample against stale expected expiry conflicts without mutation");

    sqlite3* database = nullptr;
    check(sqlite3_open(exact_path.c_str(), &database) == SQLITE_OK,
          "renewal receipt fixture is directly inspectable");
    if (database != nullptr) {
      check(scalar(database,
                   "SELECT COUNT(*) FROM resource_lease_renewals") == "1",
            "exact replay never duplicates a renewal receipt");
      check(sqlite3_exec(database, R"sql(
              UPDATE resource_lease_renewals
              SET renewed_wall_time_ns=renewed_wall_time_ns+1
            )sql", nullptr, nullptr, nullptr) != SQLITE_OK &&
                sqlite3_exec(database,
                             "DELETE FROM resource_lease_renewals", nullptr,
                             nullptr, nullptr) != SQLITE_OK &&
                sqlite3_exec(database, R"sql(
                  INSERT OR REPLACE INTO resource_lease_renewals
                  SELECT concurrency_key, owner_run_id, lease_id,
                         fencing_token, clock_domain, boot_id,
                         acquired_boottime_ns, acquired_wall_time_ns,
                         prior_expires_boottime_ns,
                         new_expires_boottime_ns,
                         prior_expires_wall_time_ns,
                         new_expires_wall_time_ns,
                         renewed_boottime_ns, renewed_wall_time_ns+1
                  FROM resource_lease_renewals
                )sql", nullptr, nullptr, nullptr) != SQLITE_OK,
            "renewal receipts reject update, delete, and INSERT OR REPLACE mutations");
      check(sqlite3_exec(database, R"sql(
              INSERT INTO resource_lease_renewals VALUES(
                'exact-gpu', 'exact-run', 'exact-lease', 1,
                'boottime/v1',
                'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
                100, 1000, 280, 400, 1180, 1300, 260, 1200
              )
            )sql", nullptr, nullptr, nullptr) != SQLITE_OK,
            "renewal receipt schema rejects unequal boot and wall timeout deltas");
      sqlite3_close(database);
    }

    std::size_t restart_clock_calls = 0U;
    auto restart_clock = std::make_shared<trainvm::AuthorityClock>([&] {
      ++restart_clock_calls;
      return test_time(183, 1'083);
    });
    trainvm::LeaseRenewalCoordinator restarted(journal, restart_clock);
    restarted.track({.lease = *current,
                     .timeout_ns = 100,
                     .renewal_margin_ns = 25});
    const auto restart_tick = restarted.tick();
    check(restart_tick.size() == 1U &&
              restart_tick.front().status ==
                  trainvm::LeaseRenewalTickStatus::not_due &&
              restart_clock_calls == 1U && restarted.tracked_count() == 1U,
          "a restarted coordinator tracks the journal's current renewed lease, not stale expected state");

    bool reboot_refused = false;
    try {
      (void)journal.renew_lease_exact(
          *current,
          test_time_on_boot(
              5, "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", 2'000),
          100);
    } catch (const std::invalid_argument&) {
      reboot_refused = true;
    }
    check(reboot_refused,
          "exact renewal never adopts a lease from a different boot identity");

    const auto before_second =
        journal.active_lease("exact-gpu", test_time(260, 1'160));
    const auto second = journal.renew_lease_exact(
        *before_second, test_time(260, 1'160), 100);
    check(second.status == trainvm::LeaseRenewalStatus::renewed &&
              second.receipt &&
              second.receipt->prior_expires_boottime_ns == 280 &&
              second.receipt->prior_expires_wall_time_ns == 1'180 &&
              second.receipt->new_expires_boottime_ns == 360 &&
              second.receipt->new_expires_wall_time_ns == 1'260,
          "successive renewal receipts preserve boot and wall expiry continuity");
  }
  {
    trainvm::Journal reopened(exact_path);
    const auto current = reopened.active_lease("exact-gpu", test_time(261));
    check(current && current->expires_boottime_ns == 360 &&
              current->expires_wall_time_ns == 1'260,
          "a multi-receipt renewal chain passes full reopen attestation");
  }

  const auto identity_mismatch_path = directory / "identity-mismatch.db";
  {
    trainvm::Journal journal(identity_mismatch_path);
    const auto acquired = journal.acquire_lease(
        "identity-gpu", "identity-run", "identity-lease", test_time(10),
        100);
    (void)journal.renew_lease_exact(acquired.lease, test_time(90), 100);
  }
  sqlite3* mismatch_database = nullptr;
  check(sqlite3_open(identity_mismatch_path.c_str(), &mismatch_database) ==
            SQLITE_OK,
        "current identity mismatch fixture opens for mutation");
  if (mismatch_database != nullptr) {
    check(sqlite3_exec(mismatch_database,
                       "UPDATE resource_leases SET owner_run_id='other-run'",
                       nullptr, nullptr, nullptr) == SQLITE_OK,
          "current identity mismatch fixture changes only the lease owner");
    sqlite3_close(mismatch_database);
  }
  bool current_identity_rejected = false;
  try {
    trainvm::Journal rejected(identity_mismatch_path);
  } catch (const std::runtime_error& exception) {
    current_identity_rejected =
        std::string(exception.what()).find("disagrees with renewal receipts") !=
        std::string::npos;
  }
  check(current_identity_rejected,
        "reopen attestation rejects a same-fence current lease identity that diverges from its receipts");

  const auto wall_chain_path = directory / "wall-chain-mismatch.db";
  {
    trainvm::Journal journal(wall_chain_path);
    const auto acquired = journal.acquire_lease(
        "wall-gpu", "wall-run", "wall-lease", test_time(10, 1'000), 100);
    const auto first =
        journal.renew_lease_exact(acquired.lease, test_time(90, 1'080), 100);
    const auto current = journal.active_lease("wall-gpu", test_time(91));
    (void)journal.renew_lease_exact(*current, test_time(170, 1'160), 100);
    check(first.status == trainvm::LeaseRenewalStatus::renewed,
          "wall continuity fixture creates its receipt chain");
  }
  check(sqlite3_open(wall_chain_path.c_str(), &mismatch_database) == SQLITE_OK,
        "wall continuity fixture opens for mutation");
  if (mismatch_database != nullptr) {
    check(sqlite3_exec(mismatch_database, R"sql(
            DROP TRIGGER resource_lease_renewals_no_conflicting_insert;
            INSERT OR REPLACE INTO resource_lease_renewals
            SELECT concurrency_key, owner_run_id, lease_id, fencing_token,
                   clock_domain, boot_id, acquired_boottime_ns,
                   acquired_wall_time_ns, prior_expires_boottime_ns,
                   new_expires_boottime_ns, prior_expires_wall_time_ns+1,
                   new_expires_wall_time_ns, renewed_boottime_ns,
                   renewed_wall_time_ns
            FROM resource_lease_renewals
            WHERE prior_expires_boottime_ns=190;
            CREATE TRIGGER resource_lease_renewals_no_conflicting_insert
            BEFORE INSERT ON resource_lease_renewals
            WHEN EXISTS(
              SELECT 1 FROM resource_lease_renewals
              WHERE (concurrency_key=NEW.concurrency_key AND
                     lease_id=NEW.lease_id AND
                     fencing_token=NEW.fencing_token AND
                     prior_expires_boottime_ns=
                         NEW.prior_expires_boottime_ns)
                 OR (concurrency_key=NEW.concurrency_key AND
                     lease_id=NEW.lease_id AND
                     fencing_token=NEW.fencing_token AND
                     new_expires_boottime_ns=NEW.new_expires_boottime_ns)
            )
            BEGIN
              SELECT RAISE(ABORT,
                           'resource lease renewal receipt already exists');
            END;
          )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "wall continuity fixture preserves exact schema while breaking the second receipt link");
    sqlite3_close(mismatch_database);
  }
  bool wall_chain_rejected = false;
  try {
    trainvm::Journal rejected(wall_chain_path);
  } catch (const std::runtime_error& exception) {
    wall_chain_rejected =
        std::string(exception.what()).find("discontinuous") !=
        std::string::npos;
  }
  check(wall_chain_rejected,
        "reopen attestation rejects discontinuous wall-expiry receipt state");

  {
    trainvm::Journal journal(directory / "coordinator.db");
    const auto acquired = journal.acquire_lease(
        "coordinator-gpu", "coordinator-run", "coordinator-lease",
        test_time(100, 10'000), 100);
    std::vector<trainvm::AuthorityTimeSample> samples{
        test_time(170, 10'070), test_time(180, 10'080),
        test_time(180, 10'080)};
    std::size_t sample_index = 0U;
    auto clock = std::make_shared<trainvm::AuthorityClock>([&] {
      return samples.at(sample_index++);
    });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = acquired.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    const auto early = coordinator.tick();
    const auto due = coordinator.tick();
    const auto duplicate = coordinator.tick();
    const auto status = coordinator.snapshot();
    check(early.size() == 1U &&
              early.front().status ==
                  trainvm::LeaseRenewalTickStatus::not_due &&
              due.size() == 1U &&
              due.front().status ==
                  trainvm::LeaseRenewalTickStatus::renewed &&
              due.front().receipt &&
              due.front().receipt->prior_expires_boottime_ns == 200 &&
              due.front().receipt->new_expires_boottime_ns == 280 &&
              duplicate.size() == 1U &&
              duplicate.front().status ==
                  trainvm::LeaseRenewalTickStatus::not_due &&
              coordinator.tracked_count() == 1U &&
              status.tracked_count == 1U && !status.poisoned &&
              status.poison_reason.empty(),
          "manual coordinator renews only inside its margin and duplicate ticks do not extend twice");
  }

  {
    trainvm::Journal journal(directory / "stale.db");
    const auto old = journal.acquire_lease(
        "stale-gpu", "old-run", "old-lease", test_time(0), 100);
    const auto successor = journal.acquire_lease(
        "stale-gpu", "new-run", "new-lease", test_time(100), 100);
    std::size_t clock_calls = 0U;
    auto clock = std::make_shared<trainvm::AuthorityClock>([&] {
      ++clock_calls;
      return test_time(101);
    });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = old.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    const auto tick = coordinator.tick();
    check(successor.status == trainvm::LeaseAcquireStatus::acquired &&
              successor.lease.fencing_token ==
                  old.lease.fencing_token + 1U &&
              tick.size() == 1U &&
              tick.front().status == trainvm::LeaseRenewalTickStatus::lost &&
              coordinator.tracked_count() == 0U && !coordinator.poisoned() &&
              clock_calls == 1U,
          "a stale fencing token is dropped without renewing or poisoning authority");
  }

  {
    trainvm::Journal journal(directory / "fresh-samples.db");
    const auto left = journal.acquire_lease(
        "fresh-a", "fresh-run-a", "fresh-lease-a", test_time(100), 100);
    const auto right = journal.acquire_lease(
        "fresh-b", "fresh-run-b", "fresh-lease-b", test_time(100), 100);
    std::vector<trainvm::AuthorityTimeSample> samples{
        test_time(180), test_time(200)};
    std::size_t sample_index = 0U;
    auto clock = std::make_shared<trainvm::AuthorityClock>([&] {
      return samples.at(sample_index++);
    });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = left.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    coordinator.track({.lease = right.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    const auto tick = coordinator.tick();
    check(tick.size() == 2U && sample_index == 2U &&
              tick[0].status == trainvm::LeaseRenewalTickStatus::renewed &&
              tick[1].status == trainvm::LeaseRenewalTickStatus::lost &&
              coordinator.tracked_count() == 1U,
          "each tracked target receives a fresh authority sample before renewal");
  }

  {
    trainvm::Journal journal(directory / "target-cap.db");
    const auto acquired = journal.acquire_lease(
        "cap-source", "cap-run", "cap-lease", test_time(10), 100);
    auto clock = std::make_shared<trainvm::AuthorityClock>(
        [] { return test_time(20); });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    for (std::size_t index = 0U;
         index < trainvm::LeaseRenewalCoordinator::kMaximumTrackedTargets;
         ++index) {
      auto lease = acquired.lease;
      lease.concurrency_key = "cap-" + std::to_string(index);
      lease.owner_run_id = "cap-run-" + std::to_string(index);
      lease.lease_id = "cap-lease-" + std::to_string(index);
      coordinator.track({.lease = std::move(lease),
                         .timeout_ns = 100,
                         .renewal_margin_ns = 25});
    }
    auto overflow = acquired.lease;
    overflow.concurrency_key = "cap-overflow";
    bool cap_rejected = false;
    try {
      coordinator.track({.lease = std::move(overflow),
                         .timeout_ns = 100,
                         .renewal_margin_ns = 25});
    } catch (const std::invalid_argument&) {
      cap_rejected = true;
    }
    check(cap_rejected &&
              coordinator.tracked_count() ==
                  trainvm::LeaseRenewalCoordinator::kMaximumTrackedTargets,
          "lease renewal coordinator enforces a hard tracked-target bound");
  }

  {
    trainvm::Journal journal(directory / "expired.db");
    const auto acquired = journal.acquire_lease(
        "expired-gpu", "expired-run", "expired-lease", test_time(0), 100);
    auto clock = std::make_shared<trainvm::AuthorityClock>(
        [] { return test_time(100); });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = acquired.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    const auto tick = coordinator.tick();
    check(tick.size() == 1U &&
              tick.front().status == trainvm::LeaseRenewalTickStatus::lost &&
              coordinator.tracked_count() == 0U,
          "a lease at exact expiry is lost and never renewed");
  }

  const auto tuple_path = directory / "tuple-chain.db";
  {
    trainvm::Journal journal(tuple_path);
    const auto left = journal.acquire_lease(
        "tuple-a", "tuple-run-a", "tuple-b\nc", test_time(10), 100);
    const auto right = journal.acquire_lease(
        "tuple-a\ntuple-b", "tuple-run-b", "c", test_time(10), 100);
    const auto left_renewed =
        journal.renew_lease_exact(left.lease, test_time(90), 100);
    const auto right_renewed =
        journal.renew_lease_exact(right.lease, test_time(90), 100);
    check(left_renewed.status == trainvm::LeaseRenewalStatus::renewed &&
              right_renewed.status == trainvm::LeaseRenewalStatus::renewed,
          "distinct renewal identities containing delimiters remain independent");
  }
  {
    trainvm::Journal reopened(tuple_path);
    check(reopened.active_lease("tuple-a", test_time(91)).has_value() &&
              reopened.active_lease("tuple-a\ntuple-b", test_time(91))
                  .has_value(),
          "tuple-keyed renewal chains reopen without delimiter collisions");
  }

  {
    trainvm::Journal journal(directory / "clock-regression.db");
    const auto acquired = journal.acquire_lease(
        "clock-gpu", "clock-run", "clock-lease", test_time(0), 100);
    std::size_t source_calls = 0U;
    auto clock = std::make_shared<trainvm::AuthorityClock>([&] {
      ++source_calls;
      return source_calls == 1U ? test_time(10) : test_time(9);
    });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = acquired.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    const auto first = coordinator.tick();
    bool regression_poisoned = false;
    bool poison_sticky = false;
    try {
      (void)coordinator.tick();
    } catch (const trainvm::LeaseRenewalCoordinatorError&) {
      regression_poisoned = true;
    }
    try {
      (void)coordinator.tick();
    } catch (const trainvm::LeaseRenewalCoordinatorError&) {
      poison_sticky = true;
    }
    const auto status = coordinator.snapshot();
    check(first.size() == 1U &&
              first.front().status ==
                  trainvm::LeaseRenewalTickStatus::not_due &&
              regression_poisoned && poison_sticky && coordinator.poisoned() &&
              coordinator.tracked_count() == 0U && source_calls == 2U &&
              status.poisoned && status.tracked_count == 0U &&
              !status.poison_reason.empty(),
          "clock regression permanently poisons and stops the manual coordinator");
  }

  const auto failure_path = directory / "receipt-failure.db";
  {
    trainvm::Journal journal(failure_path);
    const auto acquired = journal.acquire_lease(
        "failure-gpu", "failure-run", "failure-lease",
        test_time(100, 1'000), 100);
    sqlite3* database = nullptr;
    check(sqlite3_open(failure_path.c_str(), &database) == SQLITE_OK,
          "receipt failure fixture opens a fault-injection connection");
    if (database != nullptr) {
      check(sqlite3_exec(database, R"sql(
              CREATE TRIGGER reject_test_renewal_receipt
              BEFORE INSERT ON resource_lease_renewals
              BEGIN
                SELECT RAISE(ABORT, 'injected renewal receipt failure');
              END
            )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
            "receipt failure fixture installs an insert fault");
    }

    std::size_t source_calls = 0U;
    auto clock = std::make_shared<trainvm::AuthorityClock>([&] {
      ++source_calls;
      return test_time(180, 1'080);
    });
    trainvm::LeaseRenewalCoordinator coordinator(journal, clock);
    coordinator.track({.lease = acquired.lease,
                       .timeout_ns = 100,
                       .renewal_margin_ns = 25});
    bool receipt_failure_poisoned = false;
    bool poison_sticky = false;
    try {
      (void)coordinator.tick();
    } catch (const trainvm::LeaseRenewalCoordinatorError&) {
      receipt_failure_poisoned = true;
    }
    try {
      (void)coordinator.tick();
    } catch (const trainvm::LeaseRenewalCoordinatorError&) {
      poison_sticky = true;
    }
    const auto current = journal.active_lease("failure-gpu", test_time(181));
    check(receipt_failure_poisoned && poison_sticky && coordinator.poisoned() &&
              coordinator.tracked_count() == 0U && source_calls == 1U &&
              current && current->expires_boottime_ns == 200 &&
              database != nullptr &&
              scalar(database,
                     "SELECT COUNT(*) FROM resource_lease_renewals") == "0",
          "receipt insertion failure rolls back mutable expiry and permanently stops renewal");
    if (database != nullptr) {
      check(sqlite3_exec(database,
                         "DROP TRIGGER reject_test_renewal_receipt", nullptr,
                         nullptr, nullptr) == SQLITE_OK,
            "receipt failure fixture removes its fault trigger");
      sqlite3_close(database);
    }
  }

  const auto migration_path = directory / "migration-v5.db";
  {
    trainvm::Journal fresh(migration_path);
    const auto acquired = fresh.acquire_lease(
        "migration-gpu", "migration-run", "migration-lease",
        test_time(10, 1'000), 100);
    check(acquired.status == trainvm::LeaseAcquireStatus::acquired,
          "v5 migration fixture starts with an active lease");
  }
  sqlite3* database = nullptr;
  check(sqlite3_open(migration_path.c_str(), &database) == SQLITE_OK,
        "v5 migration fixture opens for exact downgrade");
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
            BEGIN IMMEDIATE;
            DROP TRIGGER host_resource_requests_no_update;
            DROP TRIGGER host_resource_requests_no_delete;
            DROP TRIGGER host_resource_grants_no_update;
            DROP TRIGGER host_resource_grants_no_delete;
            DROP TRIGGER host_resource_release_intents_no_update;
            DROP TRIGGER host_resource_release_intents_no_delete;
            DROP TRIGGER host_resource_release_receipts_no_update;
            DROP TRIGGER host_resource_release_receipts_no_delete;
            DROP TABLE host_resource_release_receipts;
            DROP TABLE host_resource_release_intents;
            DROP TABLE host_resource_grants;
            DROP TABLE host_resource_requests;
            DROP TRIGGER resource_lease_renewals_no_update;
            DROP TRIGGER resource_lease_renewals_no_delete;
            DROP TABLE resource_lease_renewals;
            UPDATE journal_meta SET value='5' WHERE key='schema_version';
            COMMIT;
          )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "v5 migration fixture exactly removes only v6 objects");
    sqlite3_close(database);
    database = nullptr;
  }
  {
    trainvm::Journal migrated(migration_path);
    const auto active = migrated.active_lease("migration-gpu", test_time(20));
    check(active && active->lease_id == "migration-lease" &&
              active->expires_boottime_ns == 110,
          "transactional v5-to-v6 migration preserves mutable lease authority");
  }
  check(sqlite3_open(migration_path.c_str(), &database) == SQLITE_OK,
        "migrated v6 journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                  "7" &&
              scalar(database,
                     "SELECT COUNT(*) FROM resource_lease_renewals") == "0" &&
              scalar(database, R"sql(
                SELECT COUNT(*) FROM sqlite_master
                WHERE type='trigger' AND name IN(
                  'resource_lease_renewals_no_conflicting_insert',
                  'resource_lease_renewals_no_update',
                  'resource_lease_renewals_no_delete')
              )sql") == "3",
          "v5-to-v6 migration atomically installs the empty receipt journal and immutable triggers");
    sqlite3_close(database);
    database = nullptr;
  }

  const auto adversarial_path = directory / "adversarial-v5.db";
  {
    trainvm::Journal fresh(adversarial_path);
  }
  check(sqlite3_open(adversarial_path.c_str(), &database) == SQLITE_OK,
        "adversarial v5 fixture opens for downgrade");
  std::string adversarial_sql;
  if (database != nullptr) {
    check(sqlite3_exec(database, R"sql(
            BEGIN IMMEDIATE;
            DROP TRIGGER host_resource_requests_no_update;
            DROP TRIGGER host_resource_requests_no_delete;
            DROP TRIGGER host_resource_grants_no_update;
            DROP TRIGGER host_resource_grants_no_delete;
            DROP TRIGGER host_resource_release_intents_no_update;
            DROP TRIGGER host_resource_release_intents_no_delete;
            DROP TRIGGER host_resource_release_receipts_no_update;
            DROP TRIGGER host_resource_release_receipts_no_delete;
            DROP TABLE host_resource_release_receipts;
            DROP TABLE host_resource_release_intents;
            DROP TABLE host_resource_grants;
            DROP TABLE host_resource_requests;
            DROP TRIGGER resource_lease_renewals_no_update;
            DROP TRIGGER resource_lease_renewals_no_delete;
            DROP TABLE resource_lease_renewals;
            CREATE TABLE resource_lease_renewals(marker TEXT NOT NULL);
            UPDATE journal_meta SET value='5' WHERE key='schema_version';
            COMMIT;
          )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "adversarial v5 fixture installs a conflicting future receipt table");
    adversarial_sql = scalar(
        database,
        "SELECT sql FROM sqlite_master WHERE name='resource_lease_renewals'");
    sqlite3_close(database);
    database = nullptr;
  }
  bool adversarial_refused = false;
  try {
    trainvm::Journal journal(adversarial_path);
  } catch (const std::runtime_error& exception) {
    adversarial_refused =
        std::string(exception.what()).find("authoritative schema") !=
        std::string::npos;
  }
  check(adversarial_refused,
        "v5 migration refuses conflicting receipt objects before mutation");
  check(sqlite3_open(adversarial_path.c_str(), &database) == SQLITE_OK,
        "refused adversarial v5 journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                  "5" &&
              scalar(database,
                     "SELECT sql FROM sqlite_master WHERE name='resource_lease_renewals'") ==
                  adversarial_sql,
          "failed v5 migration preserves version and conflicting evidence byte-for-byte");
    sqlite3_close(database);
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

void test_read_only_journal_observer() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-journal-read-only-observer-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  std::filesystem::permissions(
      directory, std::filesystem::perms::owner_all,
      std::filesystem::perm_options::replace);
  const auto database = directory / "journal.db";
  {
    trainvm::AuthorityLock authority(database);
    trainvm::Journal writer(authority.journal_path(),
                            authority.journal_identity());
    const std::string identity = writer.journal_id();

    trainvm::Journal observer(
        authority.journal_path(), authority.journal_identity(),
        trainvm::HostGrantEnforcement::required,
        trainvm::HostIdentity{
            .host_id = "sha256:" + std::string(64U, 'a'),
            .boot_id = kTestBootId},
        {}, false, trainvm::JournalAccessMode::read_only);
    const auto snapshot = observer.journal_authority_snapshot();
    check(observer.journal_id() == identity && observer.event_count() == 0U &&
              snapshot.journal_id == identity,
          "read-only journal observer attests an established live writer");

    bool mutation_refused = false;
    try {
      (void)observer.rebuild_projections();
    } catch (const std::runtime_error&) {
      mutation_refused = true;
    }
    check(mutation_refused && observer.event_count() == 0U,
          "read-only journal observer cannot mutate projections");
  }
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
    trainvm::Journal journal(
        directory / "journal.db", std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
              journal.control_commands("control-run", 2U).size() == 2U &&
              journal.control_commands("control-run", 2U).front().control_revision ==
                  4U &&
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, "service-run");
    controller.create();
    (void)controller.prepare_dispatch();
    journal_id = journal.journal_id();
  }

  trainvm::TrainVMService service(
      database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
      fixture_disabled_host_launch_registry());
  trainvm::v1::GetHostAuthorityStatusRequest host_status_request;
  trainvm::v1::GetHostAuthorityStatusResponse host_status_response;
  const grpc::Status host_status = service.GetHostAuthorityStatus(
      nullptr, &host_status_request, &host_status_response);
  check(host_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
            host_status_response.ByteSizeLong() == 0U,
        "service fails closed instead of synthesizing host health when hostd is not configured");
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

  trainvm::v1::GetControlViewRequest view_request;
  view_request.set_run_id("service-run");
  trainvm::v1::GetControlViewResponse view_response;
  const auto view_status =
      service.GetControlView(nullptr, &view_request, &view_response);
  const auto learning_rate = view_response.catalog().find("learning_rate");
  const auto effective_learning_rate = std::ranges::find_if(
      view_response.effective_values(), [](const auto& assignment) {
        return assignment.key() == "learning_rate";
      });
  check(view_status.ok() && learning_rate != view_response.catalog().end() &&
            learning_rate->second.type() == trainvm::v1::CONTROL_TYPE_NUMBER &&
            learning_rate->second.apply_point() ==
                trainvm::v1::APPLY_POINT_NEXT_OPTIMIZER_STEP &&
            effective_learning_rate != view_response.effective_values().end() &&
            effective_learning_rate->value().has_number_value() &&
            view_response.latest_requested_revision() == 1U &&
            view_response.latest_effective_revision() == 0U &&
            view_response.commands_size() == 1 &&
            view_response.commands(0).status() ==
                trainvm::v1::ControlCommandResult::STATUS_REQUESTED &&
            view_response.commands(0).assignments_size() == 1,
        "native control view returns typed catalog, effective defaults, and newest-first command history");

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
    trainvm::Journal worker_journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    const auto has_adapter_registry_error = [](const auto& response) {
      return std::ranges::any_of(
          response.diagnostics(),
          [](const trainvm::v1::Diagnostic& diagnostic) {
            return diagnostic.severity() ==
                       trainvm::v1::Diagnostic::SEVERITY_ERROR &&
                   diagnostic.code() == "adapter.registry" &&
                   diagnostic.document_path() == "/spec/components";
          });
    };
    trainvm::Journal journal(database_path);
    check(unresolved_registry_replay_status.ok() &&
              !unresolved_registry_replay.has_run() &&
              has_adapter_registry_error(unresolved_registry_replay) &&
              unresolved_registry_conflict_status.ok() &&
              !unresolved_registry_conflict_response.has_run() &&
              has_adapter_registry_error(
                  unresolved_registry_conflict_response) &&
              journal.event_count() == 1U,
          "idempotent submission replay fails closed when current authority no longer resolves its adapter registry");
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

  auto fork_fixture = fixture;
  fork_fixture["metadata"]["description"] = "explicit revision fork";
  const auto fork_compiled = trainvm::compile_document(fork_fixture);
  check(fork_compiled.valid(), "revision fork fixture compiles");
  auto fork_request = request(true, "submission-fork", fork_fixture.dump());
  fork_request.set_expected_plan_hash(fork_compiled.plan->plan_hash);
  fork_request.set_forked_from_run_id(run_id);
  fork_request.set_expected_parent_run_revision(created.run().revision());
  fork_request.set_expected_parent_plan_hash(created.run().plan_hash());
  trainvm::v1::SubmitExperimentResponse forked;
  const auto fork_status =
      service->SubmitExperiment(nullptr, &fork_request, &forked);
  trainvm::v1::SubmitExperimentResponse replayed_fork;
  const auto replayed_fork_status =
      service->SubmitExperiment(nullptr, &fork_request, &replayed_fork);
  auto stale_fork = fork_request;
  stale_fork.set_idempotency_key("submission-stale-fork");
  stale_fork.set_expected_parent_run_revision(created.run().revision() + 1U);
  trainvm::v1::SubmitExperimentResponse stale_fork_response;
  const auto stale_fork_status =
      service->SubmitExperiment(nullptr, &stale_fork, &stale_fork_response);
  {
    trainvm::Journal journal(database_path);
    const auto fork_event = forked.has_run()
        ? journal.event(forked.run().run_id() + ":created")
        : std::nullopt;
    check(fork_status.ok() && forked.has_run() &&
              forked.run().plan_hash() == fork_compiled.plan->plan_hash &&
              replayed_fork_status.ok() && replayed_fork.has_run() &&
              replayed_fork.run().run_id() == forked.run().run_id() &&
              fork_event &&
              fork_event->payload.at("submission").at("forked_from") ==
                  nlohmann::json({{"run_id", run_id},
                                  {"run_revision", created.run().revision()},
                                  {"plan_hash", created.run().plan_hash()}}) &&
              stale_fork_status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION,
          "revision forks persist exact parent lineage, replay idempotently, and reject stale parents");
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
  check(!invalid_admission.valid() &&
            has_diagnostic(invalid_admission,
                           "workflow.resource_admission_transition"),
        "unsupported builtin admission is rejected during compilation");

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
  check(!conditional_admission.valid() &&
            has_diagnostic(conditional_admission,
                           "workflow.resource_admission_transition"),
        "payload-dependent builtin admission is rejected during compilation");

  auto external_admission_document = load_fixture();
  external_admission_document["spec"]["components"]["core"]["runtime"] =
      "python_worker";
  const auto external_admission =
      trainvm::compile_document(external_admission_document);
  check(!external_admission.valid() &&
            has_diagnostic(external_admission,
                           "workflow.resource_admission"),
        "non-builtin queued entrypoint is rejected during compilation");
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

  // WorkerToController is deliberately a closed oneof. Connect requires
  // kHello first and durably dispatches every subsequent variant.
  trainvm::v1::WorkerToController first_message;
  check(first_message.message_case() ==
            trainvm::v1::WorkerToController::MESSAGE_NOT_SET,
        "WorkerControl stream has no implicit first-message variant");
  first_message.mutable_heartbeat()->set_worker_sequence(1);
  check(!first_message.has_hello() &&
            first_message.message_case() ==
                trainvm::v1::WorkerToController::kHeartbeat,
        "a heartbeat is distinguishable from the required first WorkerHello");
  first_message.mutable_metric()->set_name("train.loss");
  check(first_message.message_case() == trainvm::v1::WorkerToController::kMetric &&
            !first_message.has_heartbeat() && !first_message.has_event(),
        "WorkerControl telemetry variants cannot alias the result event case");
  first_message.mutable_phase_receipt()->set_worker_sequence(1U);
  check(first_message.message_case() ==
            trainvm::v1::WorkerToController::kPhaseReceipt &&
            !first_message.has_metric() && !first_message.has_event(),
        "execution-phase receipts are a distinct closed worker message variant");

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
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
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
        [&authority_now_ns] { return test_time(authority_now_ns); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
                !connection.welcome.canonical_invocation_json().empty() &&
                connection.welcome.invocation_digest().starts_with(
                    "sha256:") &&
                connection.attempt_baseline_optimizer_step == 0U &&
                connection.welcome.attempt_baseline_optimizer_step() == 0U &&
                observer.event_count() == 13U,
            "WorkerControl returns Welcome with a fresh-attempt baseline only "
            "after readiness and dispatch are durable");
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

    const nlohmann::json ordinary_publishes = connection.publishes;
    connection.publishes = {
        {"eval_examples",
         {{"logical_name", "eval_examples"},
          {"declaration",
           {{"required", true},
            {"type", "eval_examples"},
            {"schema", trainvm::kEvalExamplesSchema}}}}},
    };
    auto baseline_completion = canonical;
    baseline_completion.set_optimizer_step(
        connection.attempt_baseline_optimizer_step);
    reject_without_mutation(
        baseline_completion, grpc::StatusCode::FAILED_PRECONDITION,
        "WorkerControl blocks successful completion at the attempt baseline "
        "until its evaluation evidence is durable");
    connection.publishes = ordinary_publishes;

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
        wrong_sequence, grpc::StatusCode::FAILED_PRECONDITION,
        "WorkerControl rejects a worker sequence gap without mutation");
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
                observer.event_count() == 17U &&
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
                observer.event_count() == 17U,
            "lost WorkerControl receipts replay exactly without duplicate commits");
    }
  }

  {
    nlohmann::json required_output_fixture = load_fixture();
    required_output_fixture["spec"]["artifacts"]["checkpoint"]["required"] =
        true;
    const auto required_output_compiled =
        trainvm::compile_document(required_output_fixture);
    check(required_output_compiled.valid(),
          "required-output WorkerControl fixture compiles");
    if (!required_output_compiled.valid()) {
      std::filesystem::remove_all(directory);
      return;
    }
    const nlohmann::json required_output_submission_identity =
        fixture_adapter_locked_submission(*required_output_compiled.plan);
    const auto database_path = directory / "telemetry.db";
    const std::string run_id = "worker-control-telemetry-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
      trainvm::Controller controller(*required_output_compiled.plan, journal,
                                     run_id);
      controller.create_queued(required_output_submission_identity);
      (void)controller.begin_acquisition(test_time(2'000));
      launch = controller.prepare_worker_launch(launch_request, test_time(2'100));
      (void)bind_test_worker_launch(controller, launch, 2'150);
    }
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*required_output_compiled.plan,
                                          launch),
        fixture_test_host_identity(), [] { return test_time(2'200); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(wire_hello(launch), connection);

    trainvm::v1::WorkerHeartbeat heartbeat;
    heartbeat.set_worker_sequence(1);
    heartbeat.set_optimizer_step(10);
    heartbeat.set_phase("train");
    heartbeat.mutable_observed_at()->set_seconds(1);
    std::uint64_t acknowledged = 0;
    const grpc::Status heartbeat_status =
        service.record_worker_heartbeat(heartbeat, connection, acknowledged);
    const std::size_t after_heartbeat =
        trainvm::Journal(database_path).event_count();
    std::uint64_t replayed_acknowledgement = 0;
    const grpc::Status heartbeat_replay = service.record_worker_heartbeat(
        heartbeat, connection, replayed_acknowledgement);

    trainvm::v1::WorkerExecutionPhaseReceipt phase_receipt;
    const auto compile_request = std::ranges::find_if(
        connection.welcome.execution_phase_requests(),
        [](const trainvm::v1::WorkerExecutionPhaseRequest& request) {
          return request.phase() ==
                 trainvm::v1::WorkerExecutionPhaseRequest::PHASE_COMPILE;
        });
    if (compile_request !=
        connection.welcome.execution_phase_requests().end()) {
      phase_receipt.set_phase(compile_request->phase());
      phase_receipt.set_disposition(
          trainvm::v1::WorkerExecutionPhaseReceipt::DISPOSITION_COMPLETED);
      phase_receipt.set_request_digest(compile_request->request_digest());
    }
    phase_receipt.set_steps_executed(0U);
    phase_receipt.set_state_fingerprint_before(
        "sha256:" + std::string(64U, 'd'));
    phase_receipt.set_state_fingerprint_after(
        "sha256:" + std::string(64U, 'd'));
    phase_receipt.set_concurrency_key(connection.identity.concurrency_key);
    phase_receipt.set_lease_id(connection.identity.lease_id);
    phase_receipt.set_fencing_token(connection.identity.fencing_token);
    phase_receipt.set_worker_sequence(2U);
    phase_receipt.mutable_started_at()->set_seconds(2);
    phase_receipt.mutable_completed_at()->set_seconds(2);
    auto unrestored_phase_receipt = phase_receipt;
    unrestored_phase_receipt.set_state_fingerprint_after(
        "sha256:" + std::string(64U, 'e'));
    std::uint64_t rejected_phase_acknowledgement = 0;
    const grpc::Status unrestored_phase_status =
        service.record_worker_execution_phase_receipt(
            unrestored_phase_receipt, connection,
            rejected_phase_acknowledgement);
    std::uint64_t phase_acknowledgement = 0;
    const grpc::Status phase_status =
        service.record_worker_execution_phase_receipt(
            phase_receipt, connection, phase_acknowledgement);

    trainvm::v1::MetricSample metric;
    metric.set_worker_sequence(3);
    metric.set_name("train.loss");
    metric.mutable_value()->set_number_value(1.25);
    metric.set_unit("dimensionless");
    metric.set_step_domain("optimizer_step");
    metric.set_step(11);
    metric.set_sample_weight(1.0);
    (*metric.mutable_labels())["route"] = "animation";
    metric.mutable_observed_at()->set_seconds(2);
    auto undeclared_metric = metric;
    undeclared_metric.set_name("rogue.loss");
    std::uint64_t rejected_metric_acknowledgement = 0;
    const grpc::Status undeclared_metric_status = service.record_worker_metric(
        undeclared_metric, connection, rejected_metric_acknowledgement);
    auto mismatched_metric = metric;
    mismatched_metric.set_step_domain("token");
    const grpc::Status mismatched_metric_status = service.record_worker_metric(
        mismatched_metric, connection, rejected_metric_acknowledgement);
    std::uint64_t metric_acknowledgement = 0;
    const grpc::Status metric_status = service.record_worker_metric(
        metric, connection, metric_acknowledgement);

    auto missing_required_output = wire_result(connection);
    missing_required_output.set_worker_sequence(4U);
    trainvm::v1::WorkerReceipt missing_required_output_receipt;
    const std::size_t before_missing_required_output =
        trainvm::Journal(database_path).event_count();
    const grpc::Status missing_required_output_status =
        service.complete_worker_connection(
            missing_required_output, connection,
            missing_required_output_receipt);
    const std::size_t after_missing_required_output =
        trainvm::Journal(database_path).event_count();

    trainvm::v1::ArtifactManifest artifact;
    artifact.set_worker_sequence(4);
    artifact.set_artifact_id("checkpoint-step-11");
    artifact.set_logical_name("checkpoint");
    artifact.set_kind(trainvm::v1::ARTIFACT_KIND_CHECKPOINT);
    artifact.set_schema("rwkv-lab.mageflow-checkpoint.v1");
    artifact.set_uri("file:///sealed/checkpoint-step-11");
    artifact.set_size_bytes(4096);
    artifact.set_fingerprint_algorithm("manifest_sha256");
    artifact.set_fingerprint(std::string(64U, 'a'));
    artifact.set_complete(true);
    artifact.set_producer_node_id(connection.identity.node_id);
    artifact.set_producer_attempt_id(connection.identity.attempt_id);
    artifact.set_optimizer_step(11U);
    artifact.mutable_published_at()->set_seconds(3);
    auto undeclared_artifact = artifact;
    undeclared_artifact.set_logical_name("rogue-output");
    std::uint64_t rejected_artifact_acknowledgement = 0;
    const grpc::Status undeclared_artifact_status =
        service.record_worker_artifact(undeclared_artifact, connection,
                                       rejected_artifact_acknowledgement);
    auto mismatched_artifact = artifact;
    mismatched_artifact.set_kind(trainvm::v1::ARTIFACT_KIND_IMAGE_GALLERY);
    const grpc::Status mismatched_artifact_status =
        service.record_worker_artifact(mismatched_artifact, connection,
                                       rejected_artifact_acknowledgement);
    auto unstepped_checkpoint = artifact;
    unstepped_checkpoint.clear_optimizer_step();
    const grpc::Status unstepped_checkpoint_status =
        service.record_worker_artifact(unstepped_checkpoint, connection,
                                       rejected_artifact_acknowledgement);
    auto forged_parent_artifact = artifact;
    forged_parent_artifact.add_parent_artifact_ids(
        "missing-parent-artifact");
    const grpc::Status forged_parent_artifact_status =
        service.record_worker_artifact(forged_parent_artifact, connection,
                                       rejected_artifact_acknowledgement);
    std::uint64_t artifact_acknowledgement = 0;
    const grpc::Status artifact_status = service.record_worker_artifact(
        artifact, connection, artifact_acknowledgement);

    trainvm::Controller control_controller(*required_output_compiled.plan,
                                           service.journal_, run_id);
    control_controller.recover();
    const auto control_request = control_controller.request_controls(
        "telemetry-control", 5, 0, {{"caption_dropout", 0.2}}, "operator",
        "exercise streamed acknowledgement");
    trainvm::v1::ControlPatchAcknowledgement control_ack;
    if (control_request.command) {
      control_ack.set_command_id(control_request.command->command_id);
      control_ack.set_control_revision(control_request.command->control_revision);
    }
    control_ack.set_disposition(
        trainvm::v1::ControlPatchAcknowledgement::DISPOSITION_APPLIED);
    control_ack.set_apply_point(trainvm::v1::APPLY_POINT_NEXT_MICROBATCH);
    control_ack.set_effective_step(12);
    auto* effective = control_ack.add_effective_values();
    effective->set_key("caption_dropout");
    effective->mutable_value()->set_number_value(0.2);
    control_ack.set_concurrency_key(connection.identity.concurrency_key);
    control_ack.set_lease_id(connection.identity.lease_id);
    control_ack.set_fencing_token(connection.identity.fencing_token);
    control_ack.set_worker_sequence(5);
    control_ack.mutable_acknowledged_at()->set_seconds(4);
    std::uint64_t control_acknowledgement = 0;
    const grpc::Status control_status = service.acknowledge_worker_control(
        control_ack, connection, control_acknowledgement);
    const std::size_t after_control =
        trainvm::Journal(database_path).event_count();
    std::uint64_t replayed_control_acknowledgement = 0;
    const grpc::Status control_replay = service.acknowledge_worker_control(
        control_ack, connection, replayed_control_acknowledgement);

    auto gap = heartbeat;
    gap.set_worker_sequence(7);
    std::uint64_t ignored = 0;
    const grpc::Status gap_status =
        service.record_worker_heartbeat(gap, connection, ignored);

    trainvm::TrainVMService::WorkerConnection reconnected;
    const grpc::Status reconnect =
        service.open_worker_connection(wire_hello(launch), reconnected);
    auto result = wire_result(reconnected);
    result.set_worker_sequence(6);
    trainvm::v1::WorkerReceipt receipt;
    const grpc::Status complete = service.complete_worker_connection(
        result, reconnected, receipt);
    {
      trainvm::Journal observer(database_path);
      const auto projection = observer.projection(run_id);
      const auto events = observer.events_for_run(run_id);
      const auto durable_heartbeat = observer.event(
          connection.dispatch.dispatch_id + ":heartbeat:1");
      const auto durable_checkpoint = observer.event(
          connection.dispatch.dispatch_id + ":artifact:" +
          trainvm::sha256_hex("checkpoint-step-11"));
      const auto effective_controls = observer.effective_controls(run_id);
      check(open.ok() && heartbeat_status.ok() && acknowledged == 1U &&
                heartbeat_replay.ok() && replayed_acknowledgement == 1U &&
                after_heartbeat == 14U && observer.event_count() == 23U &&
                connection.welcome.execution_phase_requests_size() == 2 &&
                compile_request !=
                    connection.welcome.execution_phase_requests().end() &&
                unrestored_phase_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                rejected_phase_acknowledgement == 0U && phase_status.ok() &&
                phase_acknowledgement == 2U &&
                undeclared_metric_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                mismatched_metric_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                rejected_metric_acknowledgement == 0U &&
                metric_status.ok() && metric_acknowledgement == 3U &&
                missing_required_output_status.error_code() ==
                    grpc::StatusCode::FAILED_PRECONDITION &&
                before_missing_required_output ==
                    after_missing_required_output &&
                undeclared_artifact_status.error_code() ==
                    grpc::StatusCode::PERMISSION_DENIED &&
                mismatched_artifact_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                unstepped_checkpoint_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                forged_parent_artifact_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                artifact_status.ok() && artifact_acknowledgement == 4U &&
                control_request.command && control_status.ok() &&
                control_acknowledgement == 5U && control_replay.ok() &&
                replayed_control_acknowledgement == 5U &&
                after_control == 19U &&
                effective_controls.revision == 1U &&
                effective_controls.values ==
                    nlohmann::json{{"caption_dropout", 0.2}} &&
                gap_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
                reconnect.ok() &&
                reconnected.welcome.disposition() ==
                    trainvm::v1::WorkerWelcome::DISPOSITION_REPLAYED &&
                reconnected.welcome.canonical_invocation_json() ==
                    connection.welcome.canonical_invocation_json() &&
                reconnected.welcome.invocation_digest() ==
                    connection.welcome.invocation_digest() &&
                reconnected.welcome.acknowledged_worker_sequence() == 5U &&
                complete.ok() &&
                receipt.acknowledged_worker_sequence() == 6U && projection &&
                projection->optimizer_step == 5'500U &&
                projection->last_heartbeat_ns == 2'200 &&
                durable_heartbeat &&
                durable_heartbeat->wall_time_ns == 2'200 &&
                durable_checkpoint &&
                durable_checkpoint->optimizer_step == 11U &&
                durable_heartbeat->payload.value("observed_at_ns",
                                                  std::int64_t{}) ==
                    1'000'000'000LL &&
                std::ranges::count_if(events, [](const trainvm::Event& event) {
                  return event.event_type == "worker.heartbeat";
                }) == 1 &&
                std::ranges::count_if(events, [](const trainvm::Event& event) {
                  return event.event_type == "metric.sampled";
                }) == 1 &&
                std::ranges::count_if(events, [](const trainvm::Event& event) {
                  return event.event_type == "artifact.published";
                }) == 1 &&
                std::ranges::count_if(events, [](const trainvm::Event& event) {
                  return event.event_type ==
                         "worker.execution_phase_receipted";
                }) == 1,
            "WorkerControl durably acknowledges replay-safe heartbeat, metric, artifact, and execution-phase observations without advancing the FSM");
    }
  }

  {
    // Cancellation and the typed heartbeat phase. Compile is disabled here and
    // warmup is enabled, so one connection exercises both halves of the
    // capability question: a phase the authority requested may be heartbeat
    // against and cancelled, and one it did not may be neither.
    nlohmann::json cancellation_fixture = load_fixture();
    cancellation_fixture["spec"]["execution"]["compile"]["enabled"] = false;
    const auto cancellation_compiled =
        trainvm::compile_document(cancellation_fixture);
    check(cancellation_compiled.valid(),
          "phase-cancellation WorkerControl fixture compiles");
    if (!cancellation_compiled.valid()) {
      std::filesystem::remove_all(directory);
      return;
    }
    const nlohmann::json cancellation_submission_identity =
        fixture_adapter_locked_submission(*cancellation_compiled.plan);
    const auto database_path = directory / "phase-cancellation.db";
    const std::string run_id = "worker-control-phase-cancellation-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
      trainvm::Controller controller(*cancellation_compiled.plan, journal,
                                     run_id);
      controller.create_queued(cancellation_submission_identity);
      (void)controller.begin_acquisition(test_time(2'600));
      launch =
          controller.prepare_worker_launch(launch_request, test_time(2'700));
      (void)bind_test_worker_launch(controller, launch, 2'750);
    }
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*cancellation_compiled.plan, launch),
        fixture_test_host_identity(), [] { return test_time(2'800); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(wire_hello(launch), connection);

    const auto request_for =
        [&](trainvm::v1::WorkerExecutionPhaseRequest::Phase phase) {
          return std::ranges::find_if(
              connection.welcome.execution_phase_requests(),
              [&](const trainvm::v1::WorkerExecutionPhaseRequest& request) {
                return request.phase() == phase;
              });
        };
    const auto warmup_request =
        request_for(trainvm::v1::WorkerExecutionPhaseRequest::PHASE_WARMUP);
    const auto compile_request =
        request_for(trainvm::v1::WorkerExecutionPhaseRequest::PHASE_COMPILE);
    const bool requests_present =
        warmup_request != connection.welcome.execution_phase_requests().end() &&
        compile_request != connection.welcome.execution_phase_requests().end() &&
        warmup_request->enabled() && !compile_request->enabled() &&
        warmup_request->steps() == 8U;

    trainvm::v1::WorkerHeartbeat heartbeat;
    heartbeat.set_worker_sequence(1);
    heartbeat.set_optimizer_step(0);
    heartbeat.set_phase("warmup");
    heartbeat.set_execution_phase(
        trainvm::v1::WorkerExecutionPhaseRequest::PHASE_WARMUP);
    heartbeat.mutable_observed_at()->set_seconds(1);
    std::uint64_t heartbeat_acknowledgement = 0;
    const grpc::Status typed_heartbeat_status = service.record_worker_heartbeat(
        heartbeat, connection, heartbeat_acknowledgement);

    std::uint64_t rejected_heartbeat_acknowledgement = 0;
    // A label alone cannot claim a phase.
    auto impersonating_heartbeat = heartbeat;
    impersonating_heartbeat.set_worker_sequence(2);
    impersonating_heartbeat.clear_execution_phase();
    const grpc::Status impersonating_heartbeat_status =
        service.record_worker_heartbeat(impersonating_heartbeat, connection,
                                        rejected_heartbeat_acknowledgement);
    // Nor can a typed field name a phase this attempt never requested: compile
    // is declared but disabled, so the worker was never asked to compile.
    auto unrequested_heartbeat = heartbeat;
    unrequested_heartbeat.set_worker_sequence(2);
    unrequested_heartbeat.set_phase("compile");
    unrequested_heartbeat.set_execution_phase(
        trainvm::v1::WorkerExecutionPhaseRequest::PHASE_COMPILE);
    const grpc::Status unrequested_heartbeat_status =
        service.record_worker_heartbeat(unrequested_heartbeat, connection,
                                        rejected_heartbeat_acknowledgement);

    trainvm::v1::WorkerExecutionPhaseReceipt cancelled;
    cancelled.set_phase(trainvm::v1::WorkerExecutionPhaseRequest::PHASE_WARMUP);
    cancelled.set_disposition(
        trainvm::v1::WorkerExecutionPhaseReceipt::DISPOSITION_CANCELLED);
    if (requests_present) {
      cancelled.set_request_digest(warmup_request->request_digest());
    }
    cancelled.set_steps_executed(3U);
    cancelled.set_state_fingerprint_before("sha256:" + std::string(64U, 'c'));
    cancelled.set_state_fingerprint_after("sha256:" + std::string(64U, 'c'));
    auto* diagnostic = cancelled.add_diagnostics();
    diagnostic->set_severity(trainvm::v1::Diagnostic::SEVERITY_WARNING);
    diagnostic->set_code("execution.phase_cancelled");
    diagnostic->set_document_path("/spec/execution/warmup");
    diagnostic->set_message("the controller cancelled this attempt");
    cancelled.set_concurrency_key(connection.identity.concurrency_key);
    cancelled.set_lease_id(connection.identity.lease_id);
    cancelled.set_fencing_token(connection.identity.fencing_token);
    cancelled.set_worker_sequence(2U);
    cancelled.mutable_started_at()->set_seconds(2);
    cancelled.mutable_completed_at()->set_seconds(3);

    std::uint64_t rejected_phase_acknowledgement = 0;
    // A cancellation is still bounded by the declared step count.
    auto overrun = cancelled;
    overrun.set_steps_executed(9U);
    const grpc::Status overrun_status =
        service.record_worker_execution_phase_receipt(
            overrun, connection, rejected_phase_acknowledgement);
    // It has to say what stopped it.
    auto silent = cancelled;
    silent.clear_diagnostics();
    const grpc::Status silent_status =
        service.record_worker_execution_phase_receipt(
            silent, connection, rejected_phase_acknowledgement);
    // And it is as disposable as a completion: the trajectory must be intact.
    auto moved = cancelled;
    moved.set_state_fingerprint_after("sha256:" + std::string(64U, 'f'));
    const grpc::Status moved_status =
        service.record_worker_execution_phase_receipt(
            moved, connection, rejected_phase_acknowledgement);
    // A phase that was never enabled was never running, so nothing about it
    // can be cancelled — only skipped.
    auto disabled = cancelled;
    if (requests_present) {
      disabled.set_phase(
          trainvm::v1::WorkerExecutionPhaseRequest::PHASE_COMPILE);
      disabled.set_request_digest(compile_request->request_digest());
    }
    disabled.set_steps_executed(0U);
    const grpc::Status disabled_status =
        service.record_worker_execution_phase_receipt(
            disabled, connection, rejected_phase_acknowledgement);

    std::uint64_t cancelled_acknowledgement = 0;
    const grpc::Status cancelled_status =
        service.record_worker_execution_phase_receipt(
            cancelled, connection, cancelled_acknowledgement);
    // Replay is idempotent, not a second commit.
    std::uint64_t replayed_acknowledgement = 0;
    const grpc::Status cancelled_replay =
        service.record_worker_execution_phase_receipt(
            cancelled, connection, replayed_acknowledgement);

    {
      trainvm::Journal observer(database_path);
      const auto durable_heartbeat =
          observer.event(connection.dispatch.dispatch_id + ":heartbeat:1");
      const auto durable_cancellation = observer.event(
          connection.dispatch.dispatch_id + ":phase:warmup:2");
      // Recovery re-validates every journaled observation, so a cancelled
      // receipt and a typed heartbeat both have to survive replay.
      trainvm::Controller recovered(*cancellation_compiled.plan,
                                    service.journal_, run_id);
      bool recovers = true;
      try {
        recovered.recover();
      } catch (const std::exception&) {
        recovers = false;
      }
      check(open.ok() && requests_present && typed_heartbeat_status.ok() &&
                heartbeat_acknowledgement == 1U &&
                impersonating_heartbeat_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                unrequested_heartbeat_status.error_code() ==
                    grpc::StatusCode::PERMISSION_DENIED &&
                rejected_heartbeat_acknowledgement == 0U &&
                overrun_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                silent_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                moved_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                disabled_status.error_code() ==
                    grpc::StatusCode::INVALID_ARGUMENT &&
                rejected_phase_acknowledgement == 0U &&
                cancelled_status.ok() && cancelled_acknowledgement == 2U &&
                cancelled_replay.ok() && replayed_acknowledgement == 2U &&
                durable_heartbeat &&
                durable_heartbeat->payload.value("execution_phase",
                                                 std::string{}) == "warmup" &&
                durable_cancellation &&
                durable_cancellation->payload.value(
                    "disposition", std::string{}) == "cancelled" &&
                durable_cancellation->payload.value(
                    "steps_executed", std::uint64_t{}) == 3U &&
                durable_cancellation->payload.value(
                    "requested_steps", std::uint64_t{}) == 8U &&
                recovers,
            "a worker execution phase is cancellable mid-flight, bounded by its request, and replays from the journal");
    }
  }

  {
    auto domain_fixture = load_fixture();
    for (auto& metric : domain_fixture["spec"]["observability"]["metrics"]) {
      if (metric.value("name", std::string{}) ==
          "train.images_per_second") {
        metric["step_domain"] = "wall_time";
      }
    }
    const auto domain_compiled = trainvm::compile_document(domain_fixture);
    const auto database_path = directory / "non-optimizer-metric.db";
    const std::string run_id = "worker-control-non-optimizer-metric-run";
    trainvm::WorkerLaunchTicket launch;
    if (domain_compiled.valid()) {
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
      trainvm::Controller controller(*domain_compiled.plan, journal, run_id);
      controller.create_queued(submission_identity);
      (void)controller.begin_acquisition(test_time(2'300));
      launch = controller.prepare_worker_launch(launch_request,
                                                test_time(2'400));
      (void)bind_test_worker_launch(controller, launch, 2'450);
    }
    trainvm::TrainVMService service(
        database_path, trainvm::AdapterRegistry(fixture_adapter_profiles()),
        fixture_test_host_launch_registry(*domain_compiled.plan, launch),
        fixture_test_host_identity(), [] { return test_time(2'500); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    prime_test_service_launch(service, launch);
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status open =
        service.open_worker_connection(wire_hello(launch), connection);
    trainvm::v1::MetricSample metric;
    metric.set_worker_sequence(1);
    metric.set_name("train.images_per_second");
    metric.mutable_value()->set_number_value(4.5);
    metric.set_unit("image/second");
    metric.set_step_domain("wall_time");
    metric.set_step(1'700'000'000U);
    metric.set_sample_weight(1.0);
    metric.mutable_observed_at()->set_seconds(2);
    std::uint64_t acknowledgement = 0;
    const grpc::Status status =
        service.record_worker_metric(metric, connection, acknowledgement);
    const auto stored = trainvm::Journal(database_path).event(
        connection.dispatch.dispatch_id + ":metric:1");
    const auto projection =
        trainvm::Journal(database_path).projection(run_id);
    check(domain_compiled.valid() && open.ok() && status.ok() &&
              acknowledgement == 1U && stored && !stored->optimizer_step &&
              stored->payload.value("step", std::uint64_t{}) ==
                  1'700'000'000U &&
              projection && projection->optimizer_step == 0U,
          "non-optimizer metric domains preserve their own coordinate without corrupting run optimizer progress");
  }

  {
    const auto database_path = directory / "expired-between-phases.db";
    const std::string run_id = "worker-control-expired-between-phases-run";
    trainvm::WorkerLaunchTicket launch;
    trainvm::ResourceLease lease;
    {
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
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
        },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
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
        [&authority_now_ns] { return test_time(authority_now_ns); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
      trainvm::Journal journal(
          database_path, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
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
        [] { return test_time(4'200); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

void test_service_blocks_674_failed_final_members() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-final-evaluation-service-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  const std::filesystem::path run_directory = directory / "run";
  std::filesystem::create_directories(run_directory);

  auto source = load_fixture();
  source["metadata"]["name"] = "final-evaluation-674-errors";
  source["spec"]["workspace"]["root"] = directory.string();
  source["spec"]["workspace"]["run_directory"] = run_directory.string();
  source["spec"]["workspace"]["allowed_read_roots"] = {directory.string()};
  source["spec"]["workspace"]["allowed_write_roots"] =
      {run_directory.string()};
  source["spec"]["artifacts"]["checkpoint"]["required"] = true;
  source["spec"]["artifacts"]["eval_gallery"]["required"] = true;
  source["spec"]["artifacts"]["test_eval"] = {
      {"type", "report"},
      {"schema", "rwkv-lab.hf-test-caption-evidence-bundle.v1"},
      {"immutability", "immutable"},
      {"fingerprint", "manifest_sha256"},
      {"required", true}};
  source["spec"]["artifacts"]["final_evaluation"] = {
      {"type", "report"},
      {"schema", "rwkv-lab.final-evaluation.v1"},
      {"immutability", "immutable"},
      {"fingerprint", "manifest_sha256"},
      {"required", true}};
  auto& publishes = source["spec"]["workflow"]["nodes"]
                             ["train_to_boundary"]["publishes"];
  publishes.erase("log");
  publishes["test_eval"] = "test_eval";
  publishes["final_evaluation"] = "final_evaluation";
  auto& resume_publishes = source["spec"]["workflow"]["nodes"]
                                  ["resume_training"]["publishes"];
  resume_publishes["test_eval"] = "test_eval";
  resume_publishes["final_evaluation"] = "final_evaluation";
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid(),
        "674-error finalization service fixture compiles");
  if (!compiled.plan) {
    std::filesystem::remove_all(directory);
    return;
  }

  auto profiles = fixture_adapter_profiles();
  const auto train_profile = std::ranges::find_if(
      profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  train_profile->authoring->outputs.at("checkpoint").required = true;
  train_profile->authoring->outputs.at("eval_gallery").required = true;
  train_profile->authoring->outputs.emplace(
      "test_eval",
      operation_port(trainvm::OperationPortType::artifact, true,
                     trainvm::ArtifactType::report,
                     "rwkv-lab.hf-test-caption-evidence-bundle.v1"));
  train_profile->authoring->outputs.emplace(
      "final_evaluation",
      operation_port(trainvm::OperationPortType::artifact, true,
                     trainvm::ArtifactType::report,
                     "rwkv-lab.final-evaluation.v1"));
  const trainvm::FinalizationPolicyRegistry policy_registry({*train_profile});
  const auto& policy = policy_registry.resolve(train_profile->key);

  const auto database = directory / "journal.db";
  const std::string run_id = "service-final-evaluation-674-errors";
  trainvm::WorkerLaunchTicket launch;
  {
    trainvm::Journal journal(
        database, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(
        adapter_locked_submission(*compiled.plan,
                                  trainvm::AdapterRegistry(profiles)));
    (void)controller.begin_acquisition(test_time(1'000));
    launch = controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
         .required_capabilities = {"worker.controls", "worker.metrics"}},
        test_time(1'100));
    (void)bind_test_worker_launch(controller, launch, 1'150);
  }
  trainvm::TrainVMService service(
      database, trainvm::AdapterRegistry(profiles),
      fixture_test_host_launch_registry(*compiled.plan, launch),
      fixture_test_host_identity(), [] { return test_time(1'200); },
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  prime_test_service_launch(service, launch);
  trainvm::v1::WorkerHello hello;
  hello.set_run_id(launch.run_id);
  hello.set_node_id(launch.node_id);
  hello.set_attempt_id(launch.attempt_id);
  hello.set_launch_nonce(launch.launch_nonce);
  hello.set_adapter(launch.adapter);
  hello.set_adapter_version(launch.adapter_version);
  hello.set_code_fingerprint(launch.code_fingerprint);
  for (const auto& capability : launch.required_capabilities)
    hello.add_capabilities(capability);
  hello.set_concurrency_key(launch.concurrency_key);
  hello.set_lease_id(launch.lease_id);
  hello.set_fencing_token(launch.fencing_token);
  trainvm::TrainVMService::WorkerConnection connection;
  const grpc::Status opened =
      service.open_worker_connection(hello, connection);
  check(opened.ok(), "674-error finalization worker session opens");
  if (!opened.ok()) {
    std::filesystem::remove_all(directory);
    return;
  }

  std::vector<std::string> members;
  std::vector<trainvm::FinalMemberContext> contexts;
  std::vector<trainvm::FinalMemberRecord> records;
  for (std::size_t index = 0U; index < 674U; ++index) {
    const std::string member =
        "member-" + std::string(3U - std::to_string(index).size(), '0') +
        std::to_string(index);
    members.push_back(member);
    contexts.push_back(
        {.member_id = member,
         .context_digest = "sha256:" + std::string(64U, 'c')});
    records.push_back(
        {.member_id = member,
         .context_digest = "sha256:" + std::string(64U, 'c'),
         .attempt = 1U,
         .disposition = trainvm::FinalMemberDisposition::error,
         .result_digest = std::nullopt,
         .error_code = "generation_failed"});
  }
  const std::string checkpoint_id = "final-checkpoint";
  const std::string checkpoint_fingerprint =
      "sha256:" + std::string(64U, 'a');
  trainvm::FinalEvaluationExpectation expectation{
      .output_name = "test_eval",
      .required_output_names = {"checkpoint", "eval_gallery", "test_eval"},
      .policy_digest = trainvm::finalization_policy_digest(policy),
      .optimizer_step = 5'500U,
      .checkpoint_artifact_id = checkpoint_id,
      .checkpoint_fingerprint = checkpoint_fingerprint,
      .required_members = members,
      .member_contexts = contexts,
      .membership_digest = trainvm::final_membership_digest(members),
      .membership_count = members.size(),
      .required_scalars = {},
      .terminal_optimizer_fingerprint = std::nullopt,
  };
  trainvm::Controller authority(*compiled.plan, service.journal_, run_id);
  (void)authority.freeze_final_evaluation_expectation(expectation,
                                                       test_time(1'250));

  const std::string gallery_id = "final-gallery";
  const std::string gallery_fingerprint =
      "sha256:" + std::string(64U, 'b');
  const std::string test_id = "final-test-report";
  const std::string test_fingerprint =
      "sha256:" + std::string(64U, 'd');
  const auto publish = [&](std::uint64_t sequence, const std::string& output,
                           const std::string& artifact_id,
                           trainvm::v1::ArtifactKind kind,
                           const std::string& fingerprint,
                           const std::vector<std::string>& parents,
                           const std::string& uri,
                           std::uint64_t size_bytes) {
    const auto& publication = connection.publishes.at(output);
    trainvm::v1::ArtifactManifest artifact;
    artifact.set_worker_sequence(sequence);
    artifact.set_artifact_id(artifact_id);
    artifact.set_logical_name(
        publication.at("logical_name").get<std::string>());
    artifact.set_kind(kind);
    artifact.set_schema(
        publication.at("declaration").at("schema").get<std::string>());
    artifact.set_uri(uri);
    artifact.set_size_bytes(size_bytes);
    artifact.set_fingerprint_algorithm("manifest_sha256");
    artifact.set_fingerprint(fingerprint);
    artifact.set_complete(true);
    artifact.set_optimizer_step(5'500U);
    artifact.set_producer_node_id(connection.identity.node_id);
    artifact.set_producer_attempt_id(connection.identity.attempt_id);
    for (const std::string& parent : parents)
      artifact.add_parent_artifact_ids(parent);
    artifact.mutable_published_at()->set_seconds(2);
    std::uint64_t acknowledged = 0U;
    return service.record_worker_artifact(artifact, connection,
                                          acknowledged);
  };
  const grpc::Status checkpoint_status = publish(
      1U, "checkpoint", checkpoint_id,
      trainvm::v1::ARTIFACT_KIND_CHECKPOINT, checkpoint_fingerprint, {},
      "file:///sealed/final-checkpoint", 1U);
  const grpc::Status gallery_status = publish(
      2U, "eval_gallery", gallery_id,
      trainvm::v1::ARTIFACT_KIND_IMAGE_GALLERY, gallery_fingerprint,
      {checkpoint_id}, "file:///sealed/final-gallery", 1U);
  const grpc::Status test_status = publish(
      3U, "test_eval", test_id, trainvm::v1::ARTIFACT_KIND_REPORT,
      test_fingerprint, {checkpoint_id}, "file:///sealed/final-test", 1U);

  trainvm::FinalEvaluationManifest manifest{
      .api_version = "rwkv-lab.final-evaluation/v1",
      .policy_digest = expectation.policy_digest,
      .optimizer_step = 5'500U,
      .checkpoint_artifact_id = checkpoint_id,
      .checkpoint_fingerprint = checkpoint_fingerprint,
      .membership_digest = expectation.membership_digest,
      .membership_count = members.size(),
      .resolved_member_count = 0U,
      .failed_member_count = members.size(),
      .required_members = members,
      .output_receipts =
          {{.output_name = "checkpoint",
            .artifact_id = checkpoint_id,
            .artifact_fingerprint = checkpoint_fingerprint},
           {.output_name = "eval_gallery",
            .artifact_id = gallery_id,
            .artifact_fingerprint = gallery_fingerprint},
           {.output_name = "test_eval",
            .artifact_id = test_id,
            .artifact_fingerprint = test_fingerprint}},
      .required_scalars = {},
      .records = records,
      .recovery = std::nullopt,
  };
  const std::string closure_bytes =
      trainvm::final_evaluation_manifest_json(manifest).dump();
  const std::string closure_id = "final-evaluation-674-errors";
  const auto closure_directory = run_directory / "trainvm_artifacts" /
                                 "final_evaluation" / closure_id;
  std::filesystem::create_directories(closure_directory);
  const auto closure_path = closure_directory / "manifest.json";
  {
    std::ofstream output(closure_path, std::ios::binary);
    output.write(closure_bytes.data(),
                 static_cast<std::streamsize>(closure_bytes.size()));
  }
  const grpc::Status closure_status = publish(
      4U, "final_evaluation", closure_id,
      trainvm::v1::ARTIFACT_KIND_REPORT,
      "sha256:" + trainvm::sha256_hex(closure_bytes),
      {checkpoint_id, gallery_id, test_id},
      "file://" + closure_path.string(), closure_bytes.size());

  const trainvm::WorkerInvocationSpec invocation =
      trainvm::worker_invocation_from_canonical_json(
          connection.welcome.canonical_invocation_json());
  const auto durable_evidence = service.journal_.events_for_run(run_id);
  const auto resolved = trainvm::resolve_final_evaluation_receipts(
      invocation, expectation, durable_evidence);
  check(resolved.size() == 1U &&
            !std::ranges::binary_search(expectation.required_output_names,
                                        std::string("final_evaluation")) &&
            !expectation.terminal_optimizer_fingerprint,
        "controller resolution excludes the closure from its parent set and "
        "does not forge optimizer-state identity");
  const auto resolver_rejects = [&](std::vector<trainvm::Event> evidence,
                                    trainvm::FinalEvaluationExpectation
                                        candidate) {
    try {
      (void)trainvm::resolve_final_evaluation_receipts(
          invocation, candidate, evidence);
      return false;
    } catch (const std::invalid_argument&) {
      return true;
    }
  };
  const auto closure_event = std::ranges::find_if(
      durable_evidence, [&](const trainvm::Event& event) {
        return event.event_type == "artifact.published" &&
               event.payload.value("artifact_id", std::string{}) == closure_id;
      });
  const auto checkpoint_event = std::ranges::find_if(
      durable_evidence, [&](const trainvm::Event& event) {
        return event.event_type == "artifact.published" &&
               event.payload.value("artifact_id", std::string{}) ==
                   checkpoint_id;
      });
  check(closure_event != durable_evidence.end() &&
            checkpoint_event != durable_evidence.end(),
        "adversarial resolver fixture contains exact durable events");
  const auto closure_alias = closure_directory / "manifest-alias.json";
  std::filesystem::create_hard_link(closure_path, closure_alias);
  check(resolver_rejects(durable_evidence, expectation),
        "closure resolution rejects a hardlink alias of immutable bytes");
  std::filesystem::remove(closure_alias);
  if (closure_event != durable_evidence.end() &&
      checkpoint_event != durable_evidence.end()) {
    const std::size_t closure_index = static_cast<std::size_t>(
        std::distance(durable_evidence.begin(), closure_event));
    const auto mutate_closure = [&](std::string_view field,
                                    nlohmann::json value) {
      auto evidence = durable_evidence;
      evidence.at(closure_index).payload[std::string(field)] =
          std::move(value);
      return evidence;
    };
    auto resumed_invocation = invocation;
    auto resumed_evidence = durable_evidence;
    const std::size_t checkpoint_index = static_cast<std::size_t>(
        std::distance(durable_evidence.begin(), checkpoint_event));
    resumed_evidence.at(checkpoint_index).attempt_id = "train_to_boundary@0";
    resumed_evidence.at(checkpoint_index).worker_sequence = 99U;
    resumed_evidence.at(checkpoint_index)
        .payload["producer_attempt_id"] = "train_to_boundary@0";
    resumed_invocation.resume =
        {{"api_version", "trainvm.resume-checkpoint/v1"},
         {"checkpoint", resumed_evidence.at(checkpoint_index).payload},
         {"optimizer_step", 5'500U},
         {"pause_command_id", "pause-final"},
         {"resume_command_id", "resume-final"}};
    check(trainvm::resolve_final_evaluation_receipts(
              resumed_invocation, expectation, resumed_evidence)
                  .size() == 1U,
          "replacement finalization resolves its exact selected resume "
          "checkpoint across attempt-local worker sequences");
    check(resolver_rejects(mutate_closure("complete", false), expectation) &&
              resolver_rejects(mutate_closure("kind", "checkpoint"),
                               expectation) &&
              resolver_rejects(
                  mutate_closure("logical_name", "test_eval"), expectation) &&
              resolver_rejects(
                  mutate_closure("fingerprint_algorithm", "sha256"),
                  expectation),
          "closure completeness, kind, logical identity, and fingerprint "
          "algorithm are controller validated");

    auto duplicate_parent = durable_evidence;
    trainvm::Event duplicate = *checkpoint_event;
    duplicate.event_id += ":duplicate";
    duplicate_parent.push_back(std::move(duplicate));
    auto late_parent = durable_evidence;
    late_parent.at(checkpoint_index).worker_sequence =
        closure_event->worker_sequence;
    auto self_parent = durable_evidence;
    self_parent.at(closure_index)
        .payload["parent_artifact_ids"]
        .push_back(closure_id);
    check(resolver_rejects(std::move(duplicate_parent), expectation) &&
              resolver_rejects(std::move(late_parent), expectation) &&
              resolver_rejects(std::move(self_parent), expectation),
          "closure parents must be unique exact earlier events and cannot "
          "include the closure itself");

    auto scalar_expectation = expectation;
    scalar_expectation.required_scalars = {
        {.metric_name = "eval.loss", .step_domain = "optimizer_step"}};
    auto late_scalar = durable_evidence;
    late_scalar.push_back(
        {.event_id = "late-terminal-eval-loss",
         .run_id = run_id,
         .run_revision = connection.dispatch.run_revision,
         .plan_revision = connection.dispatch.plan_revision,
         .node_id = connection.identity.node_id,
         .attempt_id = connection.identity.attempt_id,
         .worker_sequence = closure_event->worker_sequence + 1U,
         .event_type = "metric.sampled",
         .event_version = 1U,
         .wall_time_ns = 2U,
         .monotonic_time_ns = 2U,
         .optimizer_step = 5'500U,
         .payload = {{"name", "eval.loss"},
                     {"step_domain", "optimizer_step"},
                     {"value", 1.0}}});
    check(resolver_rejects(std::move(late_scalar),
                           std::move(scalar_expectation)),
          "terminal scalar evidence must predate the closure");
  }

  trainvm::v1::EventEnvelope completion;
  completion.set_event_id(connection.dispatch.dispatch_id + ":result");
  completion.set_run_id(connection.identity.run_id);
  completion.set_run_revision(connection.dispatch.run_revision);
  completion.set_plan_revision(connection.dispatch.plan_revision);
  completion.set_node_id(connection.identity.node_id);
  completion.set_attempt_id(connection.identity.attempt_id);
  completion.set_worker_sequence(5U);
  completion.set_event_type("worker.completed");
  completion.set_event_version(1U);
  completion.mutable_wall_time()->set_seconds(3);
  completion.set_monotonic_time_ns(3U);
  completion.set_optimizer_step(5'500U);
  completion.set_canonical_json_payload(
      R"({"reason":"cache_span_complete"})");
  const std::size_t before_completion = service.journal_.event_count();
  trainvm::v1::WorkerReceipt ignored;
  const grpc::Status blocked = service.complete_worker_connection(
      completion, connection, ignored);
  const auto dispatch_after =
      service.journal_.dispatch(connection.dispatch.dispatch_id);
  const auto finalization_events =
      service.journal_.events_for_run(run_id);
  const auto verdict_event = std::ranges::find_if(
      finalization_events, [](const trainvm::Event& event) {
        return event.event_type == "finalization.verdict_recorded";
      });
  check(checkpoint_status.ok() && gallery_status.ok() && test_status.ok() &&
            closure_status.ok() &&
            blocked.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
            blocked.error_message().find("finalization_pending") !=
                std::string::npos &&
            blocked.error_message().find("no successful members") !=
                std::string::npos &&
            service.journal_.event_count() == before_completion + 1U &&
            verdict_event != finalization_events.end() &&
            verdict_event->payload.value("cause", std::string{}) ==
                "final evaluation has no successful members" &&
            dispatch_after &&
            dispatch_after->status == trainvm::DispatchStatus::prepared,
        "service completion rejects the production-class 674/674 error closure");
  std::filesystem::remove_all(directory);
}

void test_worker_control_grpc_stream() {
  const auto compiled = trainvm::compile_document(load_fixture());
  auto eof_fixture = load_fixture();
  eof_fixture["metadata"]["name"] = "worker-control-clean-eof";
  eof_fixture["spec"]["workspace"]["concurrency_key"] =
      "local-gpu-training-clean-eof";
  // A renamed fence has to be renamed everywhere it is restated, or the
  // document declares two different fences.
  for (const char* resource_node : {"acquire_gpu", "release_gpu"}) {
    eof_fixture["spec"]["workflow"]["nodes"][resource_node]["invoke"]["inputs"]
               ["concurrency_key"]["literal"] = "local-gpu-training-clean-eof";
  }
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
      [] { return test_time(1'200); },
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  prime_test_service_launch(service, launch);
  prime_test_service_launch(service, eof_launch);
  grpc::ServerBuilder builder;
  int port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(
      static_cast<trainvm::v1::WorkerControl::Service*>(&service));
  builder.RegisterService(
      static_cast<trainvm::v1::TrainVM::Service*>(&service));
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
  const auto read_stub = trainvm::v1::TrainVM::NewStub(channel);

  {
    grpc::ClientContext context;
    trainvm::v1::WatchEventsRequest request;
    request.add_run_ids(run_id);
    request.add_event_types("worker.launch_requested");
    request.set_replay_limit(1U);
    auto stream = read_stub->WatchEvents(&context, request);
    trainvm::v1::EventEnvelope envelope;
    const bool received = stream->Read(&envelope);
    trainvm::v1::EventEnvelope unexpected;
    const bool received_more = stream->Read(&unexpected);
    const grpc::Status status = stream->Finish();
    check(received && envelope.journal_sequence() > 0U &&
              envelope.run_id() == run_id &&
              envelope.event_type() == "worker.launch_requested" &&
              !envelope.canonical_json_payload().empty() &&
              nlohmann::json::parse(envelope.canonical_json_payload())
                  .is_object() &&
              !received_more && status.ok(),
          "TrainVM gRPC event stream performs a bounded filtered replay with a resumable cursor");
  }

  {
    const auto projection = trainvm::Journal(database_path).projection(run_id);
    grpc::ClientContext context;
    trainvm::v1::WatchEventsRequest request;
    request.add_run_ids(run_id);
    request.set_replay_limit(3U);
    request.set_through_journal_sequence(
        projection ? projection->last_event_sequence : 0U);
    request.set_newest_first(true);
    auto stream = read_stub->WatchEvents(&context, request);
    std::vector<std::uint64_t> sequences;
    trainvm::v1::EventEnvelope envelope;
    while (stream->Read(&envelope)) {
      sequences.push_back(envelope.journal_sequence());
    }
    const grpc::Status status = stream->Finish();
    check(status.ok() && sequences.size() == 3U &&
              sequences[0] > sequences[1] && sequences[1] > sequences[2] &&
              projection && sequences[0] <= projection->last_event_sequence,
          "TrainVM gRPC event stream exposes a bounded upper-fenced newest-first tail");
  }

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
              !welcome.welcome().canonical_invocation_json().empty() &&
              welcome.welcome().invocation_digest().starts_with("sha256:") &&
              observer.events_for_run(eof_run_id).size() == 13U,
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
              observer.events_for_run(eof_run_id).size() == 17U,
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
                trainvm::v1::WorkerWelcome::DISPOSITION_ACCEPTED &&
            !welcome_message.welcome().canonical_invocation_json().empty() &&
            welcome_message.welcome().invocation_digest().starts_with(
                "sha256:"),
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

  trainvm::v1::RunCommandRequest live_control_request;
  live_control_request.set_run_id(launch.run_id);
  live_control_request.set_expected_run_revision(5);
  live_control_request.set_idempotency_key("grpc-live-control");
  live_control_request.set_author("operator");
  live_control_request.set_reason("exercise live worker command delivery");
  live_control_request.set_expected_journal_id(service.journal_.journal_id());
  live_control_request.set_expected_plan_hash(compiled.plan->plan_hash);
  live_control_request.mutable_controls()->set_expected_control_revision(0);
  auto* live_assignment =
      live_control_request.mutable_controls()->add_assignments();
  live_assignment->set_key("caption_dropout");
  live_assignment->mutable_value()->set_number_value(0.25);
  trainvm::v1::RunCommandResponse live_control_response;
  const grpc::Status live_control_status = service.CommandRun(
      nullptr, &live_control_request, &live_control_response);

  bool result_written = false;
  bool telemetry_acknowledged = false;
  trainvm::v1::ControllerToWorker receipt_message;
  bool receipt_received = false;
  grpc::Status primary_status;
  // Read by message TYPE, not by position on the stream.
  //
  // The controller pushes commands (issued by CommandRun) onto the same stream
  // that carries telemetry acknowledgements, and a command enqueued before a
  // metric is written may legitimately arrive either side of that metric's ack.
  // The test previously assumed a fixed order -- ack, then command -- so a
  // perfectly valid interleaving consumed the command into the ack variable and
  // failed. It held on an idle host and flipped under load, where it failed
  // roughly one run in three.
  //
  // This waits on the CONDITION (a message of the wanted shape arrived) rather
  // than on elapsed time, and keeps anything that arrives early instead of
  // discarding it, so ordering between independent messages no longer matters
  // while ordering WITHIN a kind still does: each queue stays FIFO.
  std::vector<trainvm::v1::ControllerToWorker> stream_buffer;
  const auto read_matching =
      [&](const std::function<bool(const trainvm::v1::ControllerToWorker&)>&
              wanted,
          trainvm::v1::ControllerToWorker& out) {
        for (auto position = stream_buffer.begin();
             position != stream_buffer.end(); ++position) {
          if (!wanted(*position)) continue;
          out = *position;
          stream_buffer.erase(position);
          return true;
        }
        trainvm::v1::ControllerToWorker message;
        while (primary->Read(&message)) {
          if (wanted(message)) {
            out = message;
            return true;
          }
          stream_buffer.push_back(message);
        }
        return false;
      };
  const auto is_acknowledgement =
      [](const trainvm::v1::ControllerToWorker& message) {
        return message.has_acknowledge_worker_sequence();
      };
  const auto is_command = [](const trainvm::v1::ControllerToWorker& message) {
    return message.has_command();
  };

  if (welcome_received && welcome_message.has_welcome()) {
    trainvm::v1::WorkerToController heartbeat_message;
    auto* heartbeat = heartbeat_message.mutable_heartbeat();
    heartbeat->set_worker_sequence(1);
    heartbeat->set_optimizer_step(20);
    heartbeat->set_phase("train");
    heartbeat->mutable_observed_at()->set_seconds(1);
    trainvm::v1::ControllerToWorker heartbeat_ack;
    const bool heartbeat_written = primary->Write(heartbeat_message);
    const bool heartbeat_received = read_matching(is_acknowledgement, heartbeat_ack);

    trainvm::v1::ControllerToWorker worker_command_message;
    const bool worker_command_received = read_matching(is_command, worker_command_message);
    const bool worker_command_valid =
        live_control_status.ok() && live_control_response.has_control() &&
        worker_command_received && worker_command_message.has_command() &&
        worker_command_message.command().has_controls() &&
        worker_command_message.command().command_id() ==
            live_control_response.control().command_id() &&
        worker_command_message.command().controller_sequence() ==
            service.journal_.control_command_sequence(
                live_control_response.control().command_id()) &&
        worker_command_message.command().controller_sequence() ==
            live_control_response.command_sequence() &&
        worker_command_message.command().controls().control_revision() == 1U &&
        worker_command_message.command().controls().apply_point() ==
            trainvm::v1::APPLY_POINT_NEXT_MICROBATCH;

    trainvm::v1::WorkerToController control_ack_message;
    auto* control_ack = control_ack_message.mutable_control_ack();
    control_ack->set_command_id(live_control_response.control().command_id());
    control_ack->set_control_revision(1);
    control_ack->set_disposition(
        trainvm::v1::ControlPatchAcknowledgement::DISPOSITION_APPLIED);
    control_ack->set_apply_point(trainvm::v1::APPLY_POINT_NEXT_MICROBATCH);
    control_ack->set_effective_step(20);
    auto* applied = control_ack->add_effective_values();
    applied->set_key("caption_dropout");
    applied->mutable_value()->set_number_value(0.25);
    control_ack->set_concurrency_key(launch.concurrency_key);
    control_ack->set_lease_id(launch.lease_id);
    control_ack->set_fencing_token(launch.fencing_token);
    control_ack->set_worker_sequence(2);
    control_ack->mutable_acknowledged_at()->set_seconds(2);
    trainvm::v1::ControllerToWorker control_receipt;
    const bool control_ack_written = primary->Write(control_ack_message);
    const bool control_ack_received = read_matching(is_acknowledgement, control_receipt);

    trainvm::v1::RunCommandRequest checkpoint_request;
    checkpoint_request.set_run_id(launch.run_id);
    checkpoint_request.set_expected_run_revision(5);
    checkpoint_request.set_idempotency_key("grpc-checkpoint-now");
    checkpoint_request.set_author("operator");
    checkpoint_request.set_reason("exercise immutable checkpoint authority");
    checkpoint_request.set_expected_journal_id(service.journal_.journal_id());
    checkpoint_request.set_expected_plan_hash(compiled.plan->plan_hash);
    checkpoint_request.mutable_checkpoint()->set_reason("operator snapshot");
    trainvm::v1::RunCommandResponse checkpoint_response;
    const grpc::Status checkpoint_status = service.CommandRun(
        nullptr, &checkpoint_request, &checkpoint_response);

    trainvm::v1::WorkerToController metric_message;
    auto* metric = metric_message.mutable_metric();
    metric->set_worker_sequence(3);
    metric->set_name("train.loss");
    metric->mutable_value()->set_number_value(0.75);
    metric->set_unit("dimensionless");
    metric->set_step_domain("optimizer_step");
    metric->set_step(21);
    metric->set_sample_weight(1.0);
    metric->mutable_observed_at()->set_seconds(3);
    trainvm::v1::ControllerToWorker metric_ack;
    const bool metric_written = primary->Write(metric_message);
    const bool metric_received = read_matching(is_acknowledgement, metric_ack);

    trainvm::v1::ControllerToWorker checkpoint_command_message;
    const bool checkpoint_command_received =
        read_matching(is_command, checkpoint_command_message);
    const bool checkpoint_command_valid =
        checkpoint_status.ok() && checkpoint_response.has_checkpoint() &&
        checkpoint_response.disposition() ==
            trainvm::v1::RunCommandResponse::DISPOSITION_ACCEPTED &&
        checkpoint_command_received && checkpoint_command_message.has_command() &&
        checkpoint_command_message.command().has_checkpoint() &&
        checkpoint_command_message.command().command_id() ==
            checkpoint_response.checkpoint().command_id() &&
        checkpoint_command_message.command().controller_sequence() ==
            checkpoint_response.checkpoint().controller_sequence() &&
        checkpoint_command_message.command().checkpoint().reason() ==
            "operator snapshot";

    trainvm::v1::WorkerToController artifact_message;
    auto* artifact = artifact_message.mutable_artifact();
    artifact->set_worker_sequence(4);
    artifact->set_artifact_id("grpc-checkpoint");
    artifact->set_logical_name("checkpoint");
    artifact->set_kind(trainvm::v1::ARTIFACT_KIND_CHECKPOINT);
    artifact->set_schema("rwkv-lab.mageflow-checkpoint.v1");
    artifact->set_uri("file:///sealed/grpc-checkpoint");
    artifact->set_size_bytes(8192);
    artifact->set_fingerprint_algorithm("manifest_sha256");
    artifact->set_fingerprint(std::string(64U, 'b'));
    artifact->set_complete(true);
    artifact->set_producer_node_id(welcome_message.welcome().node_id());
    artifact->set_producer_attempt_id(welcome_message.welcome().attempt_id());
    artifact->set_optimizer_step(21U);
    artifact->mutable_published_at()->set_seconds(4);
    trainvm::v1::ControllerToWorker artifact_ack;
    const bool artifact_written = primary->Write(artifact_message);
    const bool artifact_received = read_matching(is_acknowledgement, artifact_ack);

    trainvm::v1::WorkerToController checkpoint_ack_message;
    auto* checkpoint_ack = checkpoint_ack_message.mutable_checkpoint_ack();
    checkpoint_ack->set_command_id(
        checkpoint_response.checkpoint().command_id());
    checkpoint_ack->set_disposition(
        trainvm::v1::CheckpointAcknowledgement::DISPOSITION_APPLIED);
    checkpoint_ack->set_optimizer_step(21);
    checkpoint_ack->set_artifact_id("grpc-checkpoint");
    checkpoint_ack->set_concurrency_key(launch.concurrency_key);
    checkpoint_ack->set_lease_id(launch.lease_id);
    checkpoint_ack->set_fencing_token(launch.fencing_token);
    checkpoint_ack->set_worker_sequence(5);
    checkpoint_ack->mutable_acknowledged_at()->set_seconds(5);
    trainvm::v1::ControllerToWorker checkpoint_receipt;
    const bool checkpoint_ack_written = primary->Write(checkpoint_ack_message);
    const bool checkpoint_ack_received = read_matching(is_acknowledgement, checkpoint_receipt);

    trainvm::v1::RunCommandRequest pause_request;
    pause_request.set_run_id(launch.run_id);
    pause_request.set_expected_run_revision(5);
    pause_request.set_idempotency_key("grpc-pause");
    pause_request.set_author("operator");
    pause_request.set_reason("exercise retained-resource pause");
    pause_request.set_expected_journal_id(service.journal_.journal_id());
    pause_request.set_expected_plan_hash(compiled.plan->plan_hash);
    pause_request.mutable_pause()->set_checkpoint_first(false);
    pause_request.mutable_pause()->set_release_resources(false);
    trainvm::v1::RunCommandResponse pause_response;
    const grpc::Status pause_status =
        service.CommandRun(nullptr, &pause_request, &pause_response);

    trainvm::v1::WorkerToController pause_trigger_message;
    auto* pause_trigger = pause_trigger_message.mutable_heartbeat();
    pause_trigger->set_worker_sequence(6);
    pause_trigger->set_optimizer_step(21);
    pause_trigger->set_phase("training");
    pause_trigger->mutable_observed_at()->set_seconds(6);
    trainvm::v1::ControllerToWorker pause_trigger_receipt;
    const bool pause_trigger_written = primary->Write(pause_trigger_message);
    const bool pause_trigger_received = read_matching(is_acknowledgement, pause_trigger_receipt);
    trainvm::v1::ControllerToWorker pause_command_message;
    const bool pause_command_received = read_matching(is_command, pause_command_message);
    const bool pause_command_valid =
        pause_status.ok() && pause_response.has_lifecycle() &&
        pause_response.lifecycle().kind() ==
            trainvm::v1::LifecycleCommandResult::KIND_PAUSE &&
        pause_command_received && pause_command_message.has_command() &&
        pause_command_message.command().has_pause() &&
        pause_command_message.command().controller_sequence() ==
            pause_response.command_sequence();

    trainvm::v1::WorkerToController pause_ack_message;
    auto* pause_ack = pause_ack_message.mutable_lifecycle_ack();
    pause_ack->set_command_id(pause_response.lifecycle().command_id());
    pause_ack->set_kind(
        trainvm::v1::LifecycleAcknowledgement::KIND_PAUSE);
    pause_ack->set_disposition(
        trainvm::v1::LifecycleAcknowledgement::DISPOSITION_APPLIED);
    pause_ack->set_concurrency_key(launch.concurrency_key);
    pause_ack->set_lease_id(launch.lease_id);
    pause_ack->set_fencing_token(launch.fencing_token);
    pause_ack->set_worker_sequence(7);
    pause_ack->mutable_acknowledged_at()->set_seconds(7);
    trainvm::v1::ControllerToWorker pause_ack_receipt;
    const bool pause_ack_written = primary->Write(pause_ack_message);
    const bool pause_ack_received = read_matching(is_acknowledgement, pause_ack_receipt);
    const auto paused_projection = service.journal_.projection(launch.run_id);

    trainvm::v1::RunCommandRequest resume_request;
    resume_request.set_run_id(launch.run_id);
    resume_request.set_expected_run_revision(8);
    resume_request.set_idempotency_key("grpc-resume");
    resume_request.set_author("operator");
    resume_request.set_reason("exercise retained-resource resume");
    resume_request.set_expected_journal_id(service.journal_.journal_id());
    resume_request.set_expected_plan_hash(compiled.plan->plan_hash);
    resume_request.mutable_resume();
    trainvm::v1::RunCommandResponse resume_response;
    const grpc::Status resume_status =
        service.CommandRun(nullptr, &resume_request, &resume_response);

    trainvm::v1::WorkerToController resume_trigger_message;
    auto* resume_trigger = resume_trigger_message.mutable_heartbeat();
    resume_trigger->set_worker_sequence(8);
    resume_trigger->set_optimizer_step(21);
    resume_trigger->set_phase("paused");
    resume_trigger->mutable_observed_at()->set_seconds(8);
    trainvm::v1::ControllerToWorker resume_trigger_receipt;
    const bool resume_trigger_written = primary->Write(resume_trigger_message);
    const bool resume_trigger_received = read_matching(is_acknowledgement, resume_trigger_receipt);
    trainvm::v1::ControllerToWorker resume_command_message;
    const bool resume_command_received = read_matching(is_command, resume_command_message);
    const bool resume_command_valid =
        resume_status.ok() && resume_response.has_lifecycle() &&
        resume_response.lifecycle().kind() ==
            trainvm::v1::LifecycleCommandResult::KIND_RESUME &&
        resume_command_received && resume_command_message.has_command() &&
        resume_command_message.command().has_resume() &&
        resume_command_message.command().controller_sequence() ==
            resume_response.command_sequence();

    trainvm::v1::WorkerToController resume_ack_message;
    auto* resume_ack = resume_ack_message.mutable_lifecycle_ack();
    resume_ack->set_command_id(resume_response.lifecycle().command_id());
    resume_ack->set_kind(
        trainvm::v1::LifecycleAcknowledgement::KIND_RESUME);
    resume_ack->set_disposition(
        trainvm::v1::LifecycleAcknowledgement::DISPOSITION_APPLIED);
    resume_ack->set_concurrency_key(launch.concurrency_key);
    resume_ack->set_lease_id(launch.lease_id);
    resume_ack->set_fencing_token(launch.fencing_token);
    resume_ack->set_worker_sequence(9);
    resume_ack->mutable_acknowledged_at()->set_seconds(9);
    trainvm::v1::ControllerToWorker resume_ack_receipt;
    const bool resume_ack_written = primary->Write(resume_ack_message);
    const bool resume_ack_received = read_matching(is_acknowledgement, resume_ack_receipt);

    trainvm::v1::WorkerToController resumed_metric_message;
    auto* resumed_metric = resumed_metric_message.mutable_metric();
    resumed_metric->set_worker_sequence(10);
    resumed_metric->set_name("train.loss");
    resumed_metric->mutable_value()->set_number_value(0.5);
    resumed_metric->set_unit("dimensionless");
    resumed_metric->set_step_domain("optimizer_step");
    resumed_metric->set_step(22);
    resumed_metric->set_sample_weight(1.0);
    resumed_metric->mutable_observed_at()->set_seconds(10);
    trainvm::v1::ControllerToWorker resumed_metric_receipt;
    const bool resumed_metric_written = primary->Write(resumed_metric_message);
    const bool resumed_metric_received = read_matching(is_acknowledgement, resumed_metric_receipt);
    // Same decomposition as the final check, and for the same reason: this
    // aggregated thirty terms into one bool, so every one of them reported as
    // the single opaque "durable telemetry" failure downstream. Each stage now
    // says which acknowledgement went missing and what sequence it saw.
    const auto stage = [&](bool written, bool received, bool has_sequence,
                           std::uint64_t expected, std::uint64_t actual,
                           std::string_view what) {
      check(written, std::string("WorkerControl writes the ") +
                         std::string(what) + " message");
      check(received, std::string("WorkerControl reads a receipt for the ") +
                          std::string(what) + " message");
      check(has_sequence && actual == expected,
            std::string("WorkerControl acknowledges ") + std::string(what) +
                " at worker sequence " + std::to_string(expected) + ", saw " +
                (has_sequence ? std::to_string(actual) : "no sequence"));
      return written && received && has_sequence && actual == expected;
    };
    telemetry_acknowledged =
        stage(heartbeat_written, heartbeat_received,
              heartbeat_ack.has_acknowledge_worker_sequence(), 1U,
              heartbeat_ack.acknowledge_worker_sequence(), "heartbeat") &
        stage(control_ack_written, control_ack_received,
              control_receipt.has_acknowledge_worker_sequence(), 2U,
              control_receipt.acknowledge_worker_sequence(),
              "control acknowledgement") &
        stage(metric_written, metric_received,
              metric_ack.has_acknowledge_worker_sequence(), 3U,
              metric_ack.acknowledge_worker_sequence(), "metric") &
        stage(artifact_written, artifact_received,
              artifact_ack.has_acknowledge_worker_sequence(), 4U,
              artifact_ack.acknowledge_worker_sequence(), "artifact") &
        stage(checkpoint_ack_written, checkpoint_ack_received,
              checkpoint_receipt.has_acknowledge_worker_sequence(), 5U,
              checkpoint_receipt.acknowledge_worker_sequence(),
              "checkpoint acknowledgement") &
        stage(pause_trigger_written, pause_trigger_received, true, 6U,
              pause_trigger_receipt.acknowledge_worker_sequence(),
              "pause trigger") &
        stage(pause_ack_written, pause_ack_received, true, 7U,
              pause_ack_receipt.acknowledge_worker_sequence(),
              "pause acknowledgement") &
        stage(resume_trigger_written, resume_trigger_received, true, 8U,
              resume_trigger_receipt.acknowledge_worker_sequence(),
              "resume trigger") &
        stage(resume_ack_written, resume_ack_received, true, 9U,
              resume_ack_receipt.acknowledge_worker_sequence(),
              "resume acknowledgement") &
        stage(resumed_metric_written, resumed_metric_received, true, 10U,
              resumed_metric_receipt.acknowledge_worker_sequence(),
              "resumed metric");
    check(worker_command_valid, "WorkerControl issues a valid worker command");
    check(checkpoint_command_valid,
          "WorkerControl issues a valid checkpoint command");
    check(pause_command_valid, "WorkerControl issues a valid pause command");
    check(resume_command_valid, "WorkerControl issues a valid resume command");
    check(paused_projection && paused_projection->desired_state == "paused" &&
              paused_projection->observed_state == "paused",
          "WorkerControl projects the run as paused");
    check(paused_projection && paused_projection->run_revision == 8U,
          "WorkerControl leaves the paused run at revision 8, saw " +
              (paused_projection
                   ? std::to_string(paused_projection->run_revision)
                   : std::string("no projection")));

    auto result = result_for(welcome_message.welcome());
    result.mutable_event()->set_worker_sequence(11);
    result_written = primary->Write(result);
    primary->WritesDone();
    // Same reason: a late command must not be mistaken for the result
    // Receipt just because it arrived first.
    receipt_received = read_matching(
        [](const trainvm::v1::ControllerToWorker& message) {
          return message.has_receipt();
        },
        receipt_message);
    trainvm::v1::ControllerToWorker trailing;
    check(!primary->Read(&trailing),
          "WorkerControl gRPC closes after its single result Receipt");
    primary_status = primary->Finish();
  } else {
    primary_context.TryCancel();
    primary_status = primary->Finish();
  }

  {
    const auto projection = trainvm::Journal(database_path).projection(run_id);
    grpc::ClientContext context;
    trainvm::v1::WatchEventsRequest request;
    request.add_run_ids(run_id);
    request.add_event_types("metric.sampled");
    request.set_replay_limit(8U);
    request.set_through_journal_sequence(
        projection ? projection->last_event_sequence : 0U);
    request.set_newest_per_metric_series(true);
    auto stream = read_stub->WatchEvents(&context, request);
    trainvm::v1::EventEnvelope envelope;
    std::vector<trainvm::v1::EventEnvelope> metrics;
    while (stream->Read(&envelope)) metrics.push_back(envelope);
    const grpc::Status status = stream->Finish();
    const auto payload = metrics.size() == 1U
                             ? nlohmann::json::parse(
                                   metrics.front().canonical_json_payload())
                             : nlohmann::json{};
    check(status.ok() && metrics.size() == 1U &&
              metrics.front().event_type() == "metric.sampled" &&
              payload.value("name", std::string{}) == "train.loss" &&
              payload.value("value", 0.0) == 0.5 &&
              payload.value("step", std::uint64_t{}) == 22U,
          "TrainVM gRPC exposes the bounded latest-per-metric-series projection");
  }

  {
    trainvm::Journal observer(database_path);
    const auto dispatch = welcome_message.has_welcome()
                              ? observer.dispatch(
                                    welcome_message.welcome().dispatch_id())
                              : std::nullopt;
    const auto projection = observer.projection(run_id);
    // Split rather than one ANDed expression. As a single check this reported
    // only "FAIL: WorkerControl gRPC orders ..." for any of eighteen distinct
    // reasons, so a failure said nothing about what broke -- which cost real
    // time when this went red under host load. Each term now names itself and
    // reports what it actually saw.
    const auto observed_events = observer.events_for_run(run_id).size();
    check(telemetry_acknowledged,
          "WorkerControl gRPC acknowledges the durable telemetry sequence");
    check(result_written && receipt_received,
          "WorkerControl gRPC writes the result and reads a receipt");
    check(receipt_message.has_receipt() &&
              receipt_message.message_case() ==
                  trainvm::v1::ControllerToWorker::kReceipt,
          "WorkerControl gRPC returns a Receipt message");
    check(primary_status.ok(), "WorkerControl gRPC stream finishes ok");
    check(receipt_message.receipt().event_id() ==
              welcome_message.welcome().dispatch_id() + ":result",
          "WorkerControl gRPC receipt names the dispatch result event");
    check(receipt_message.receipt().acknowledged_worker_sequence() == 11U,
          "WorkerControl gRPC receipt acknowledges worker sequence 11, saw " +
              std::to_string(
                  receipt_message.receipt().acknowledged_worker_sequence()));
    check(receipt_message.receipt().committed_run_revision() == 13U,
          "WorkerControl gRPC receipt commits run revision 13, saw " +
              std::to_string(
                  receipt_message.receipt().committed_run_revision()));
    check(dispatch && dispatch->status == trainvm::DispatchStatus::completed,
          "WorkerControl gRPC marks the dispatch completed");
    check(dispatch && dispatch->result_event_id ==
                          std::optional<std::string>{
                              receipt_message.receipt().event_id()},
          "WorkerControl gRPC records the dispatch result event id");
    check(projection && projection->observed_state == "acquiring",
          "WorkerControl gRPC leaves the run observed as acquiring, saw " +
              (projection ? projection->observed_state : "no projection"));
    check(projection && projection->run_revision == 13U,
          "WorkerControl gRPC leaves the run at revision 13, saw " +
              (projection ? std::to_string(projection->run_revision)
                          : std::string("no projection")));
    check(observed_events == 37U,
          "WorkerControl gRPC journals exactly 37 events for the run, saw " +
              std::to_string(observed_events));
  }

  server->Shutdown();
  server->Wait();
  std::filesystem::remove_all(directory);
}

void test_graceful_cancel_lifecycle() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "graceful cancellation fixture compiles");
  if (!compiled.valid()) return;
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-cancel-lifecycle-test-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "cancel-lifecycle-run";
  {
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(fixture_adapter_locked_submission(*compiled.plan));
    (void)controller.begin_acquisition(test_time(1'000));
    const auto launch = controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '8'),
         .required_capabilities = {"worker.controls"}},
        test_time(1'100));
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
         .last_acked_controller_sequence = 0U,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(1'175));
    const auto before = journal.projection(run_id);
    check(before && before->desired_state == "running" &&
              before->observed_state == "running",
          "graceful cancellation starts from an active worker");
    if (!before) return;

    const auto submitted = controller.request_cancel(
        "cancel-once", before->run_revision, "operator requested stop",
        5'000'000'000LL, "operator", "test graceful cancellation");
    const auto pending = journal.pending_lifecycle_commands(run_id, 0U);
    check(submitted.inserted &&
              submitted.command.kind ==
                  trainvm::LifecycleCommandKind::cancel &&
              submitted.command.cancel_reason == "operator requested stop" &&
              submitted.command.graceful_timeout_ns == 5'000'000'000LL &&
              pending.size() == 1U &&
              pending.front().command_id == submitted.command.command_id,
          "cancel command is durable and preserves its graceful-stop policy");

    (void)controller.acknowledge_lifecycle(
        submitted.command.command_id,
        {.concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .worker_sequence = 1U},
        trainvm::LifecycleCommandStatus::applied, std::nullopt, "",
        nlohmann::json::array(), test_time(1'200));
    controller.recover();
    const auto cancelling = journal.projection(run_id);
    check(cancelling && cancelling->desired_state == "cancelled" &&
              cancelling->observed_state == "cancelling" &&
              cancelling->run_revision == before->run_revision + 2U &&
              controller.state().status == trainvm::ExecutionStatus::running,
          "cancel acknowledgement records a nonterminal cleanup barrier");

    (void)controller.complete_cancellation(
        submitted.command.command_id, test_time(1'250));
    const auto cancelled = journal.projection(run_id);
    check(cancelled && cancelled->desired_state == "cancelled" &&
              cancelled->observed_state == "cancelled" &&
              cancelled->current_node_id.empty() &&
              cancelled->current_attempt_id.empty() &&
              cancelled->run_revision == before->run_revision + 3U &&
              controller.state().status == trainvm::ExecutionStatus::cancelled,
          "cancellation becomes terminal only at the explicit completion barrier");
    trainvm::Controller replay(*compiled.plan, journal, run_id);
    replay.recover();
    check(replay.state() == controller.state(),
          "graceful cancellation replays deterministically after restart");
  }
  std::filesystem::remove_all(directory);
}

// Adversarial control admission at a live safe point: a duplicate patch must
// replay rather than apply twice, the same key with different content must be
// refused outright, and every attempt must leave a state a reconnecting
// controller reconstructs exactly from the journal.
// A research topology is selected in the document and refused at compile time
// when it is invalid, so an operator sees a diagnostic instead of a launch
// failure on a GPU.
// The checked-in example must compile through the real authority, not merely
// satisfy the JSON schema. A schema-valid document that the compiler rejects
// would be a worse lie than no example at all.
void test_checked_in_topology_example_compiles() {
  const std::filesystem::path path =
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/rwkv-scratch-topologies.json";
  std::ifstream input(path);
  check(input.good(), "the topology example document is readable");
  if (!input) return;
  nlohmann::json document;
  input >> document;

  const auto compiled = trainvm::compile_document(document);
  if (!compiled.valid()) {
    for (const auto& value : compiled.diagnostics)
      std::cerr << "example diagnostic " << value.code << " @" << value.path
                << ": " << value.message << '\n';
  }
  check(compiled.valid(), "the checked-in topology example compiles");
  if (!compiled.valid()) return;

  const auto& node =
      compiled.plan->experiment.spec.workflow.nodes.at("train_to_boundary");
  check(node.invoke.training && node.invoke.training->topologies &&
            node.invoke.training->topologies->size() == 2U,
        "the example carries both selected topologies into the plan");
}

void test_post_training_arm_is_gated_at_compile_time() {
  const auto with_arm = [](nlohmann::json arm) {
    nlohmann::json document = load_fixture();
    auto& node = document["spec"]["workflow"]["nodes"]["train_to_boundary"];
    node["invoke"]["training"] = {
        {"model_family", "rwkv"},
        {"components",
         {{"optimizer",
           {{"key", {{"category", "optimizer"},
                     {"name", "optimizer.torch_adamw"},
                     {"version", "1"}}},
            {"configuration", nlohmann::json::object()}}}}},
        {"post_training", std::move(arm)},
    };
    return document;
  };
  const auto refused_with = [&](nlohmann::json arm, std::string_view code) {
    const auto compiled = trainvm::compile_document(with_arm(std::move(arm)));
    return !compiled.valid() &&
           std::ranges::any_of(compiled.diagnostics,
                               [code](const trainvm::Diagnostic& value) {
                                 return value.code == code;
                               });
  };

  const nlohmann::json honest = {
      {"arm_id", "arm.finetune-a"},
      {"kind", "supervised_finetune"},
      {"bounds", nlohmann::json::array({
                     {{"kind", "optimizer_steps"}, {"magnitude", 10000}}})},
      {"reproducibility_claim", "exact"},
      {"seed", 7},
  };
  check(trainvm::compile_document(with_arm(honest)).valid(),
        "a step-bounded seeded fine-tune arm compiles");

  // The card's first clause, now refused while the author is still writing
  // rather than at launch. A seed does not rescue a wall-clock endpoint.
  nlohmann::json timed = honest;
  timed["bounds"] = nlohmann::json::array(
      {{{"kind", "wall_clock_seconds"}, {"magnitude", 3600}}});
  check(refused_with(timed, "post-training-claim-unsupported"),
        "a wall-clock arm claiming exact reproducibility is refused");
  timed["reproducibility_claim"] = "none";
  check(trainvm::compile_document(with_arm(timed)).valid(),
        "the same wall-clock arm compiles once it is labelled honestly");

  // The card's second clause.
  nlohmann::json unauthorized = honest;
  unauthorized["kind"] = "direct_rlvr";
  unauthorized["verifier_identity"] = "verifier.unit-tests@sha256-abc";
  unauthorized["reproducibility_claim"] = "seeded";
  unauthorized["external_mutations"] = nlohmann::json::array(
      {{{"target", "verifier.unit-tests"},
        {"effect", "record_judgement"},
        {"authorized", false}}});
  check(refused_with(unauthorized, "post-training-mutation-unauthorized"),
        "an unauthorized external mutation is refused at compile time");

  nlohmann::json anonymous_verifier = honest;
  anonymous_verifier["kind"] = "direct_rlvr";
  anonymous_verifier["reproducibility_claim"] = "seeded";
  check(refused_with(anonymous_verifier, "post-training-verifier-unbound"),
        "an RLVR arm with no verifier identity is refused");

  // An unknown enum name must name itself rather than decode to a default and
  // then be validated as something the author never wrote.
  nlohmann::json unknown_kind = honest;
  unknown_kind["kind"] = "supervised_finetuning";
  check(refused_with(unknown_kind, "training.post_training.kind"),
        "an unknown arm kind is reported by name");
  nlohmann::json unknown_bound = honest;
  unknown_bound["bounds"] =
      nlohmann::json::array({{{"kind", "epochs"}, {"magnitude", 3}}});
  check(refused_with(unknown_bound, "training.post_training.bound"),
        "an unknown run bound kind is reported by name");
}

void test_topology_selection_compiles_and_refuses_invalid_combinations() {
  const auto with_topologies = [](nlohmann::json topologies) {
    nlohmann::json document = load_fixture();
    auto& node = document["spec"]["workflow"]["nodes"]["train_to_boundary"];
    node["invoke"]["training"] = {
        {"model_family", "rwkv"},
        {"components",
         {{"optimizer",
           {{"key", {{"category", "optimizer"},
                     {"name", "optimizer.torch_adamw"},
                     {"version", "1"}}},
            {"configuration", nlohmann::json::object()}}}}},
        {"topologies", std::move(topologies)},
    };
    return document;
  };

  const auto valid = trainvm::compile_document(with_topologies(
      nlohmann::json::array({
          {{"topology", "engram"},
           {"parameters", {{"enabled", 1}, {"rows", 8192}}}},
          {{"topology", "loop"}, {"parameters", {{"count", 4}}}},
      })));
  check(valid.valid(), "a document selecting engram and loop compiles");

  const auto incompatible = trainvm::compile_document(with_topologies(
      nlohmann::json::array({
          {{"topology", "loop"}, {"parameters", nlohmann::json::object()}},
          {{"topology", "seed_chain"},
           {"parameters", nlohmann::json::object()}},
      })));
  const bool refused_pair = !incompatible.valid() &&
      std::ranges::any_of(incompatible.diagnostics,
                          [](const trainvm::Diagnostic& value) {
                            return value.code == "training.topologies.invalid";
                          });
  check(refused_pair,
        "seed_chain combined with loop is refused at compile time");

  const auto unknown_switch = trainvm::compile_document(with_topologies(
      nlohmann::json::array({
          {{"topology", "loop"}, {"parameters", {{"engram_rows", 4096}}}},
      })));
  check(!unknown_switch.valid(),
        "a switch belonging to another topology is refused at compile time");

  const auto out_of_bound = trainvm::compile_document(with_topologies(
      nlohmann::json::array({
          {{"topology", "loop"}, {"parameters", {{"count", 99}}}},
      })));
  check(!out_of_bound.valid(),
        "a value outside its declared bound is refused at compile time");

  const auto unknown_topology = trainvm::compile_document(with_topologies(
      nlohmann::json::array({
          {{"topology", "warp_drive"},
           {"parameters", nlohmann::json::object()}},
      })));
  const bool refused_unknown = !unknown_topology.valid() &&
      std::ranges::any_of(unknown_topology.diagnostics,
                          [](const trainvm::Diagnostic& value) {
                            return value.code == "training.topologies.unknown";
                          });
  check(refused_unknown, "an unregistered topology is refused at compile time");

  // Topologies are closed per model family.
  nlohmann::json wrong_family = with_topologies(nlohmann::json::array({
      {{"topology", "loop"}, {"parameters", nlohmann::json::object()}}}));
  wrong_family["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
              ["training"]["model_family"] = "transformer";
  const auto refused_family = trainvm::compile_document(wrong_family);
  check(!refused_family.valid(),
        "only the rwkv family may declare research topologies");
}

void test_adversarial_control_idempotency_and_replay() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "adversarial control fixture compiles");
  if (!compiled.valid()) return;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-adversarial-control-test-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "adversarial-control-run";
  {
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(fixture_adapter_locked_submission(*compiled.plan));
    (void)controller.begin_acquisition(test_time(1'000));
    const auto launch = controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '7'),
         .required_capabilities = {"worker.controls"}},
        test_time(1'100));
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
         .last_acked_controller_sequence = 0U,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(1'175));

    const auto reconnects_exactly = [&](std::string_view what) {
      trainvm::Controller reconnected(*compiled.plan, journal, run_id);
      reconnected.recover();
      check(reconnected.state() == controller.state(),
            std::string("a reconnecting controller replays exactly after ") +
                std::string(what));
    };

    const auto before = journal.projection(run_id);
    check(before && before->observed_state == "running",
          "adversarial control fixture starts from an active worker");
    if (!before) return;
    reconnects_exactly("worker readiness");

    const nlohmann::json assignments{{"learning_rate", 0.00001},
                                     {"eval_every", 250}};
    const std::uint64_t control_revision =
        journal.latest_control_revision(run_id);
    const auto first = controller.request_controls(
        "patch-once", before->run_revision, control_revision, assignments,
        "operator", "adversarial duplicate patch");
    check(first.valid() && first.command && !first.replayed,
          "the first control patch is accepted once");
    if (!first.command) return;
    const auto duplicate = controller.request_controls(
        "patch-once", before->run_revision, control_revision, assignments,
        "operator", "adversarial duplicate patch");
    check(duplicate.replayed && duplicate.command &&
              duplicate.command->command_id == first.command->command_id &&
              journal.pending_control_commands(run_id, 0U).size() == 1U,
          "an exact duplicate patch replays one durable command");
    reconnects_exactly("a duplicate control patch");

    bool changed_content_refused = false;
    try {
      nlohmann::json different = assignments;
      different["__adversarial"] = true;
      (void)controller.request_controls(
          "patch-once", before->run_revision, control_revision, different,
          "operator", "adversarial duplicate patch");
    } catch (const std::invalid_argument&) {
      changed_content_refused = true;
    }
    bool changed_revision_refused = false;
    try {
      (void)controller.request_controls(
          "patch-once", before->run_revision + 41U, control_revision,
          assignments, "operator", "adversarial duplicate patch");
    } catch (const std::invalid_argument&) {
      changed_revision_refused = true;
    }
    bool changed_author_refused = false;
    try {
      (void)controller.request_controls(
          "patch-once", before->run_revision, control_revision, assignments,
          "someone-else", "adversarial duplicate patch");
    } catch (const std::invalid_argument&) {
      changed_author_refused = true;
    }
    check(changed_content_refused && changed_revision_refused &&
              changed_author_refused &&
              journal.pending_control_commands(run_id, 0U).size() == 1U,
          "a reused idempotency key with any changed field is refused and "
          "forks no durable command");
    reconnects_exactly("refused control patch retries");

    const auto checkpoint = controller.request_checkpoint(
        "checkpoint-once", before->run_revision, "operator checkpoint",
        "operator", "adversarial duplicate checkpoint");
    const auto checkpoint_replay = controller.request_checkpoint(
        "checkpoint-once", before->run_revision, "operator checkpoint",
        "operator", "adversarial duplicate checkpoint");
    check(checkpoint.inserted && !checkpoint_replay.inserted &&
              checkpoint_replay.command.command_id ==
                  checkpoint.command.command_id &&
              journal.pending_checkpoint_commands(run_id, 0U).size() == 1U,
          "a duplicate checkpoint-now request replays one durable command");
    reconnects_exactly("a duplicate checkpoint request");

    const auto cancel = controller.request_cancel(
        "cancel-once", before->run_revision, "operator requested stop",
        5'000'000'000LL, "operator", "adversarial duplicate cancel");
    const auto cancel_replay = controller.request_cancel(
        "cancel-once", before->run_revision, "operator requested stop",
        5'000'000'000LL, "operator", "adversarial duplicate cancel");
    check(cancel.inserted && !cancel_replay.inserted &&
              cancel_replay.command.command_id == cancel.command.command_id &&
              journal.pending_lifecycle_commands(run_id, 0U).size() == 1U,
          "a duplicate cancel request replays one durable command");

    bool changed_cancel_refused = false;
    try {
      (void)controller.request_cancel(
          "cancel-once", before->run_revision, "a different stop reason",
          5'000'000'000LL, "operator", "adversarial duplicate cancel");
    } catch (const std::invalid_argument&) {
      changed_cancel_refused = true;
    }
    bool changed_timeout_refused = false;
    try {
      (void)controller.request_cancel(
          "cancel-once", before->run_revision, "operator requested stop",
          9'000'000'000LL, "operator", "adversarial duplicate cancel");
    } catch (const std::invalid_argument&) {
      changed_timeout_refused = true;
    }
    check(changed_cancel_refused && changed_timeout_refused &&
              journal.pending_lifecycle_commands(run_id, 0U).size() == 1U,
          "a reused cancel key with a changed reason or timeout is refused");
    reconnects_exactly("refused cancel retries");
  }
  std::filesystem::remove_all(directory);
}

void test_resource_releasing_pause_lifecycle() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "resource-releasing pause fixture compiles");
  if (!compiled.valid()) return;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-release-pause-test-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database_path = directory / "journal.db";
  const std::string run_id = "resource-releasing-pause-run";
  {
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(fixture_adapter_locked_submission(*compiled.plan));
    (void)controller.begin_acquisition(test_time(1'000));
    const auto launch = controller.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '9'),
         .required_capabilities = {"worker.controls"}},
        test_time(1'100));
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
         .last_acked_controller_sequence = 0U,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(1'175));
    const auto dispatch = controller.prepare_dispatch(test_time(1'175));
    const std::string artifact_id = "release-pause-checkpoint";
    (void)controller.record_worker_observation(
        {.event_id = dispatch.dispatch_id + ":artifact:" +
                     trainvm::sha256_hex(artifact_id),
         .run_id = run_id,
         .run_revision = dispatch.run_revision,
         .plan_revision = dispatch.plan_revision,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .worker_sequence = 1U,
         .event_type = "artifact.published",
         .event_version = 1U,
         .wall_time_ns = 1'180,
         .monotonic_time_ns = 1'180U,
         .optimizer_step = std::uint64_t{9},
         .payload = {{"artifact_id", artifact_id},
                     {"logical_name", "checkpoint"},
                     {"kind", "checkpoint"},
                     {"schema", "trainvm.checkpoint.v1"},
                     {"uri", "file:///sealed/release-pause"},
                     {"size_bytes", std::uint64_t{4096}},
                     {"fingerprint_algorithm", "manifest_sha256"},
                     {"fingerprint", std::string(64U, 'a')},
                     {"complete", true},
                     {"producer_node_id", launch.node_id},
                     {"producer_attempt_id", launch.attempt_id},
                     {"parent_artifact_ids", nlohmann::json::array()},
                     {"published_at_ns", std::int64_t{1'180}}}},
        {.run_id = launch.run_id,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .launch_nonce = launch.launch_nonce,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(1'180));
    const auto before = journal.projection(run_id);
    const auto submitted = controller.request_lifecycle(
        trainvm::LifecycleCommandKind::pause, "release-pause-once",
        before->run_revision, true, true, "operator",
        "release accelerator while paused");
    (void)controller.acknowledge_lifecycle(
        submitted.command.command_id,
        {.concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .worker_sequence = 2U},
        trainvm::LifecycleCommandStatus::applied, std::uint64_t{9},
        artifact_id, nlohmann::json::array(), test_time(1'200));
    controller.recover();
    const auto pausing = journal.projection(run_id);
    check(pausing && pausing->desired_state == "paused" &&
              pausing->observed_state == "pausing" &&
              pausing->run_revision == before->run_revision + 2U,
          "resource-releasing pause remains nonfinal until cleanup");
    check(journal.release_lease(
              launch.concurrency_key, run_id, launch.lease_id,
              launch.fencing_token, test_time(1'240)),
          "resource-releasing pause drops its original logical fence before completion");
    (void)controller.complete_resource_releasing_pause(
        submitted.command.command_id, test_time(1'250));
    const auto paused = journal.projection(run_id);
    check(paused && paused->desired_state == "paused" &&
              paused->observed_state == "paused" &&
              paused->run_revision == before->run_revision + 3U &&
              paused->current_node_id == launch.node_id &&
              paused->current_attempt_id == launch.attempt_id,
          "resource-releasing pause becomes resumable only at cleanup completion");
    trainvm::Controller replay(*compiled.plan, journal, run_id);
    replay.recover();
    check(replay.state() == controller.state(),
          "resource-releasing pause replays deterministically after restart");

    const auto resume = replay.request_lifecycle(
        trainvm::LifecycleCommandKind::resume, "released-resume-once",
        paused->run_revision, false, false, "operator",
        "reacquire accelerator from immutable checkpoint");
    const std::string resume_lease_id =
        "lease-" + trainvm::sha256_hex(
                       nlohmann::json(
                           {{"run_id", run_id},
                            {"plan_hash", compiled.plan->plan_hash},
                            {"concurrency_key", launch.concurrency_key},
                            {"resume_command_id", resume.command.command_id}})
                           .dump());
    const auto interrupted_acquisition = journal.acquire_lease(
        launch.concurrency_key, run_id, resume_lease_id,
        test_time(1'290), 30'000'000'000LL);
    const auto resumed = replay.begin_released_resource_resume(
        resume.command.command_id, test_time(1'300));
    const auto acquiring = journal.projection(run_id);
    const auto durable_resume =
        journal.lifecycle_command(resume.command.command_id);
    const auto old_acquisition =
        journal.lease_acquisition_event(run_id, launch.lease_id);
    const auto new_acquisition =
        journal.lease_acquisition_event(run_id, resumed.lease.lease_id);
    check(interrupted_acquisition.status ==
                trainvm::LeaseAcquireStatus::acquired &&
              resumed.status == trainvm::LeaseAcquireStatus::already_owned &&
              resumed.lease.lease_id != launch.lease_id &&
              resumed.lease.fencing_token > launch.fencing_token &&
              acquiring && acquiring->desired_state == "running" &&
              acquiring->observed_state == "acquiring" &&
              acquiring->run_revision == paused->run_revision + 3U &&
              acquiring->current_node_id.empty() &&
              acquiring->current_attempt_id.empty() &&
              replay.state().current_node_id == launch.node_id &&
              replay.state().current_attempt_id == launch.node_id + "@2" &&
              durable_resume &&
              durable_resume->status ==
                  trainvm::LifecycleCommandStatus::applied &&
              durable_resume->optimizer_step == std::optional<std::uint64_t>{9} &&
              durable_resume->artifact_id == artifact_id &&
              durable_resume->acknowledgement &&
              durable_resume->acknowledgement->worker_sequence == 0U &&
              old_acquisition && new_acquisition &&
              old_acquisition->event_id != new_acquisition->event_id,
          "released-resource resume recovers an interrupted lease acquisition and creates a fenced @2 attempt bound to the pause checkpoint");

    const auto replayed_resume = journal.apply_released_resource_resume(
        run_id, resume.command.command_id, submitted.command.command_id,
        resumed.lease, replay.state().current_attempt_id, test_time(1'305));
    bool changed_resume_retry_rejected = false;
    try {
      (void)journal.apply_released_resource_resume(
          run_id, resume.command.command_id, submitted.command.command_id,
          resumed.lease, launch.node_id + "@3", test_time(1'306));
    } catch (const std::invalid_argument&) {
      changed_resume_retry_rejected = true;
    }
    check(replayed_resume == *durable_resume && changed_resume_retry_rejected,
          "released-resource resume is exactly idempotent and rejects a changed replacement attempt");

    const auto replacement = replay.prepare_worker_launch(
        {.code_fingerprint = "sha256:" + std::string(64U, '9'),
         .required_capabilities = {"worker.controls"}},
        test_time(1'310));
    check(replacement.attempt_id == launch.node_id + "@2" &&
              replacement.lease_id == resumed.lease.lease_id &&
              replacement.fencing_token == resumed.lease.fencing_token &&
              replacement.launch_nonce != launch.launch_nonce &&
              replacement.lease_id != launch.lease_id,
          "replacement worker launch cannot reuse the released attempt or fence");
    trainvm::Controller resumed_replay(*compiled.plan, journal, run_id);
    check(resumed_replay.recover() == replay.state(),
          "released-resource acquisition and replacement launch replay after restart");
  }
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
  trainvm::Journal journal(
      database_path, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
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

// Drives the run to the qualification gate and returns its live controller
// state so each scenario can commit a different verdict.
struct CacheQualificationRun {
  std::filesystem::path directory;
  std::unique_ptr<trainvm::Journal> journal;
  std::unique_ptr<trainvm::Controller> controller;
  trainvm::ResourceLease lease;
};

CacheQualificationRun start_cache_qualification_run(
    const trainvm::CompiledPlan& plan, std::string_view suffix) {
  CacheQualificationRun state;
  state.directory = std::filesystem::temp_directory_path() /
      ("trainvm-cache-qualification-" + std::string(suffix) + "-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(state.directory);
  std::filesystem::create_directories(state.directory);
  const std::string run_id = "cache-qualification-run";
  state.journal = std::make_unique<trainvm::Journal>(
      state.directory / "journal.db", std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  state.controller = std::make_unique<trainvm::Controller>(
      plan, *state.journal, run_id);
  state.controller->create_queued();
  state.lease = state.controller->begin_acquisition(test_time(1'000)).lease;
  const trainvm::WorkerLaunchRequest request{
      .code_fingerprint = "sha256:" + std::string(64U, '5'),
      .required_capabilities = {"worker.controls"},
  };
  const auto launch =
      state.controller->prepare_worker_launch(request, test_time(1'100));
  (void)bind_test_worker_launch(*state.controller, launch, 1'150);
  (void)state.controller->accept_worker_hello(
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
  const auto dispatch = state.controller->prepare_dispatch(test_time(1'300));
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
      .event_id = dispatch.dispatch_id + ":result",
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
  (void)state.controller->handle_event(worker_result, session,
                                       test_time(1'400));
  return state;
}

void test_typed_cache_qualification_executor() {
  const auto compiled = trainvm::compile_document(cache_qualification_fixture());
  check(compiled.valid(),
        "cache qualification fixture compiles");
  if (!compiled.valid()) {
    for (const auto& diagnostic : compiled.diagnostics) {
      std::cerr << "  " << diagnostic.code << " " << diagnostic.path << " "
                << diagnostic.message << '\n';
    }
    return;
  }

  const auto rejects_with = [](nlohmann::json document,
                               std::string_view code) {
    const auto result = trainvm::compile_document(std::move(document));
    return !result.valid() &&
           std::ranges::any_of(result.diagnostics,
                               [&](const trainvm::Diagnostic& diagnostic) {
                                 return diagnostic.code == code;
                               });
  };
  auto missing_rejection = cache_qualification_fixture();
  missing_rejection["spec"]["workflow"]["nodes"]["qualify_cache_artifact"]
                   ["transitions"].erase(1);
  auto wrong_effect = cache_qualification_fixture();
  wrong_effect["spec"]["workflow"]["nodes"]["qualify_cache_artifact"]
              ["effect"] = "process";
  auto undeclared_phase = cache_qualification_fixture();
  undeclared_phase["spec"]["execution"].erase("qualify");
  check(rejects_with(std::move(missing_rejection),
                     "workflow.cache_qualification_transition") &&
            rejects_with(std::move(wrong_effect),
                         "workflow.cache_qualification") &&
            rejects_with(std::move(undeclared_phase),
                         "workflow.cache_qualification_phase"),
        "cache qualification topology requires both verdicts, an exact typed operation, and a declared qualify phase");

  auto qualified_run = start_cache_qualification_run(*compiled.plan, "pass");
  check(qualified_run.controller->state().current_node_id ==
            "qualify_cache_artifact",
        "worker completion enters the typed cache qualification gate");

  const trainvm::CacheQualificationReceipt passing =
      trainvm::qualify_cache_artifact(passing_cache_evidence());
  auto forged = passing;
  forged.qualified = false;
  forged.rejection_reasons = {"output_parity_failed"};
  bool forged_verdict_rejected = false;
  try {
    (void)qualified_run.controller->complete_cache_qualification(
        forged, test_time(1'500));
  } catch (const std::exception&) {
    forged_verdict_rejected = true;
  }
  bool lease_release_refused = false;
  try {
    (void)qualified_run.controller->complete_managed_builtin(
        "qualify_cache", "cache.qualified", true, test_time(1'500));
  } catch (const std::logic_error&) {
    lease_release_refused = true;
  }
  bool wrong_event_refused = false;
  try {
    (void)qualified_run.controller->complete_managed_builtin(
        "qualify_cache", "resource.released", false, test_time(1'500));
  } catch (const std::logic_error&) {
    wrong_event_refused = true;
  }
  check(forged_verdict_rejected && lease_release_refused && wrong_event_refused,
        "the gate rejects a forged verdict, a lease-releasing gate, and an off-contract receipt event");

  const auto& advanced =
      qualified_run.controller->complete_cache_qualification(passing,
                                                             test_time(1'500));
  const auto events =
      qualified_run.journal->events_for_run("cache-qualification-run");
  const auto verdict = std::ranges::find_if(
      events, [](const trainvm::Event& event) {
        return event.event_type == "cache.qualified";
      });
  const auto receipts = std::ranges::count_if(
      events, [](const trainvm::Event& event) {
        return event.event_type == "node.dispatch_completed" &&
               event.node_id == "qualify_cache_artifact";
      });
  trainvm::Controller restarted(*compiled.plan, *qualified_run.journal,
                                "cache-qualification-run");
  const auto& recovered = restarted.recover();
  std::string chain_reason;
  check(advanced.current_node_id == "release_gpu" && receipts == 1 &&
            verdict != events.end() && verdict->worker_sequence == 0U &&
            verdict->payload.value("receipt_digest", std::string{}) ==
                passing.receipt_digest &&
            verdict->payload.value("qualified", false) &&
            recovered == advanced &&
            qualified_run.journal->verify_chain(&chain_reason),
        "a qualified verdict commits one receipt bound to its gate digest and replays identically");
  check(qualified_run.journal->active_lease(
            qualified_run.lease.concurrency_key, test_time(1'500)).has_value(),
        "the qualification gate keeps the fence it ran under");

  auto rejected_run = start_cache_qualification_run(*compiled.plan, "reject");
  auto failing_evidence = passing_cache_evidence();
  failing_evidence.candidate_throughput = 101.0;  // below the declared gain gate
  // A bit-identity claim the samples do not support: the arms measurably
  // separated. The declared verdict matches the derived one, so the receipt
  // rejects on the trajectory itself rather than on a mislabelled document.
  failing_evidence.resumed_trajectory_parity = {
      .verdict = trainvm::TrajectoryParityVerdict::diverged,
      .criterion = trainvm::TrajectoryEquivalenceCriterion::bit_identical,
      .effect_class = trainvm::TrajectoryEffectClass::optimizer_update,
      .candidate_divergence = {{.step = 1U, .relative_deviation = 1.6e-07},
                               {.step = 2U, .relative_deviation = 1.7e-06},
                               {.step = 5U, .relative_deviation = 1.3e-03}},
      .reference_divergence = {},
      .checkpoint_quality = {},
      .analysis_seed = 0U,
  };
  const trainvm::CacheQualificationReceipt rejection =
      trainvm::qualify_cache_artifact(failing_evidence);
  const auto& failed =
      rejected_run.controller->complete_cache_qualification(rejection,
                                                           test_time(1'500));
  const auto rejected_events =
      rejected_run.journal->events_for_run("cache-qualification-run");
  const auto rejection_event = std::ranges::find_if(
      rejected_events, [](const trainvm::Event& event) {
        return event.event_type == "cache.rejected";
      });
  check(!rejection.qualified &&
            rejection.rejection_reasons ==
                std::vector<std::string>({"resumed_trajectory_parity_failed",
                                          "throughput_gate_failed"}) &&
            failed.status == trainvm::ExecutionStatus::failed &&
            rejection_event != rejected_events.end() &&
            rejection_event->payload.at("rejection_reasons") ==
                nlohmann::json(rejection.rejection_reasons),
        "a rejected candidate routes to the declared failure path with attributable reasons");

  // The case the boolean could not express. A fused kernel agrees to float32
  // epsilon on one step and then amplifies; it can never be bit-identical, and
  // it diverges more slowly than the reference does against its own seed.
  auto tolerated = passing_cache_evidence();
  tolerated.resumed_trajectory_parity = {
      .verdict = trainvm::TrajectoryParityVerdict::diverged_within_tolerance,
      .criterion = trainvm::TrajectoryEquivalenceCriterion::divergence_rate,
      .effect_class = trainvm::TrajectoryEffectClass::optimizer_update,
      .candidate_divergence = {{.step = 1U, .relative_deviation = 1.6e-07},
                               {.step = 2U, .relative_deviation = 1.7e-06},
                               {.step = 5U, .relative_deviation = 1.3e-03}},
      .reference_divergence = {{.step = 1U, .relative_deviation = 2.0e-07},
                               {.step = 2U, .relative_deviation = 2.0e-06},
                               {.step = 5U, .relative_deviation = 2.0e-03}},
      .checkpoint_quality = {},
      .analysis_seed = 7U,
  };
  const trainvm::CacheQualificationReceipt tolerated_receipt =
      trainvm::qualify_cache_artifact(tolerated);
  check(tolerated_receipt.qualified &&
            tolerated_receipt.trajectory_assessment.has_value() &&
            tolerated_receipt.trajectory_assessment->verdict ==
                trainvm::TrajectoryParityVerdict::diverged_within_tolerance &&
            tolerated_receipt.trajectory_assessment->divergence_rate_ratio <
                1.0 &&
            tolerated_receipt.trajectory_assessment->tolerance
                    .maximum_divergence_rate_ratio == 1.25,
        "a candidate that cannot be bit-identical qualifies on a bounded "
        "divergence rate, and the receipt carries the statistics that admitted it");

  // The same measurements labelled as bit-identity. The label does not decide.
  auto mislabelled = tolerated;
  mislabelled.resumed_trajectory_parity.verdict =
      trainvm::TrajectoryParityVerdict::equivalent;
  check(trainvm::qualify_cache_artifact(mislabelled).rejection_reasons ==
            std::vector<std::string>({"resumed_trajectory_verdict_unsupported"}),
        "a verdict its own statistics do not support is rejected as such");

  // The optimizer-state device question, decided rather than left open.
  auto device_bound = tolerated;
  device_bound.optimizer_state_device_policy =
      trainvm::OptimizerStateDevicePolicy::device_bound;
  check(trainvm::qualify_cache_artifact(device_bound).rejection_reasons ==
            std::vector<std::string>({"optimizer_state_device_policy_failed"}),
        "optimizer state that is not normalized on load makes a resume depend "
        "on which implementation wrote it, and is rejected");

  std::filesystem::remove_all(qualified_run.directory);
  std::filesystem::remove_all(rejected_run.directory);
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

  trainvm::Journal left_journal(
      database_path, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  trainvm::Journal right_journal(
      database_path, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

  trainvm::Journal left_journal(
      database_path, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  trainvm::Journal right_journal(
      database_path, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
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
                                      "bootstrap_runtime_closure_fingerprint",
                                      "provided_capabilities", "executable_path",
                                      "executable_fingerprint", "code_path",
                                      "code_argument_index",
                                      "public_arguments",
                                      "working_directory"}) &&
            trainvm::reflected_field_names<
                trainvm::HostLaunchRegistryDocument>() ==
                std::vector<std::string>({"api_version", "trusted_roots",
                                          "profiles",
                                          "profiler_executables"}) &&
            trainvm::reflected_field_names<
                trainvm::HostProfilerExecutableProfile>() ==
                std::vector<std::string>({
                    "backend", "version", "executable_path",
                    "executable_fingerprint"}),
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
      .bootstrap_runtime_closure_fingerprint =
          "sha256:" + std::string(64U, 'd'),
      .provided_capabilities = {"worker.controls", "worker.metrics"},
      .executable_path = "/opt/trainvm/python/bin/python3",
      .executable_fingerprint = python_executable,
      .code_path = "/opt/trainvm/adapters/mageflow/worker.py",
      .code_argument_index = 1U,
      .public_arguments = {"-I", "/opt/trainvm/adapters/mageflow/worker.py"},
      .working_directory = "/srv/trainvm/runs/run-1",
  };
  const trainvm::HostLaunchProfile native_profile{
      .key = key("example.native", "2.0.0",
                 trainvm::ComponentRuntime::native_worker, "execute",
                 "example.native.v1.Execute"),
      .code_fingerprint = native_code,
      .bootstrap_runtime_closure_fingerprint =
          "sha256:" + std::string(64U, 'e'),
      .provided_capabilities = {"worker.controls"},
      .executable_path = "/usr/libexec/trainvm/native-worker",
      .executable_fingerprint = native_code,
      .code_path = std::nullopt,
      .public_arguments = {"--worker"},
      .working_directory = "/srv/trainvm/runs/run-1",
  };
  const trainvm::HostProfilerExecutableProfile nsys_profile{
      .backend = trainvm::ProfilerBackend::nsys,
      .version = trainvm::profiler_launch_profile(
                     trainvm::ProfilerBackend::nsys)
                     .version,
      .executable_path = "/opt/trainvm/profilers/nsys",
      .executable_fingerprint = "sha256:" + std::string(64U, 'f'),
  };
  const trainvm::HostLaunchRegistryDocument document{
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {"/usr/libexec/trainvm", "/srv/trainvm",
                        "/opt/trainvm"},
      .profiles = {native_profile, python_profile},
      .profiler_executables =
          std::vector<trainvm::HostProfilerExecutableProfile>{nsys_profile},
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
  const auto& resolved_nsys =
      loaded.resolve_profiler(trainvm::ProfilerBackend::nsys);
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
                            "trainvm.host-launch-profile/v4"},
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
                expected_python_digest &&
            resolved_nsys == nsys_profile &&
            !loaded.profiler_profile_digest(trainvm::ProfilerBackend::nsys)
                 .empty(),
        "host launch loader reflection-decodes both runtimes and canonicalizes trusted-root and profile order");

  auto reordered = document;
  std::ranges::reverse(reordered.trusted_roots);
  std::ranges::reverse(reordered.profiles);
  if (reordered.profiler_executables)
    std::ranges::reverse(*reordered.profiler_executables);
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
                resolved_native &&
            reordered_registry.resolve_profiler(
                trainvm::ProfilerBackend::nsys) == resolved_nsys,
        "host launch registry semantics are invariant to document collection order");
  auto changed_document = document;
  changed_document.profiles.at(1).public_arguments.push_back("--changed");
  const trainvm::HostLaunchRegistry changed_registry(
      std::move(changed_document));
  auto changed_closure_document = document;
  changed_closure_document.profiles.at(1)
      .bootstrap_runtime_closure_fingerprint =
      "sha256:" + std::string(64U, 'f');
  const trainvm::HostLaunchRegistry changed_closure_registry(
      std::move(changed_closure_document));
  check(changed_registry.registry_digest() != loaded.registry_digest() &&
            changed_registry.profile_digest(python_profile.key, python_code) !=
                loaded.profile_digest(python_profile.key, python_code) &&
            changed_closure_registry.registry_digest() !=
                loaded.registry_digest() &&
            changed_closure_registry.profile_digest(python_profile.key,
                                                     python_code) !=
                loaded.profile_digest(python_profile.key, python_code),
        "host launch registry and profile digests bind launch semantics and runtime closure identity");

  const auto disabled_path = directory / "host-launches-disabled.json";
  write_document(
      disabled_path,
      trainvm::encode_json(trainvm::HostLaunchRegistryDocument{
          .api_version = "trainvm.host-launches/v4",
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
  future["api_version"] = "trainvm.host-launches/v5";
  auto legacy = encoded;
  legacy["api_version"] = "trainvm.host-launches/v1";
  auto duplicate_profile = encoded;
  duplicate_profile["profiles"].push_back(duplicate_profile["profiles"].at(0));
  auto duplicate_profiler = encoded;
  duplicate_profiler["profiler_executables"].push_back(
      duplicate_profiler["profiler_executables"].at(0));
  auto torch_profiler = encoded;
  torch_profiler["profiler_executables"].at(0)["backend"] = "torch";
  auto profiler_version_skew = encoded;
  profiler_version_skew["profiler_executables"].at(0)["version"] = "2.0.0";
  auto profiler_escape = encoded;
  profiler_escape["profiler_executables"].at(0)["executable_path"] =
      "/usr/bin/nsys";
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
  auto python_code_slot_out_of_range = encoded;
  python_code_slot_out_of_range["profiles"].at(1)["code_argument_index"] = 2;
  auto native_with_code = encoded;
  native_with_code["profiles"].at(0)["code_path"] =
      "/usr/libexec/trainvm/native-worker";
  auto native_with_code_slot = encoded;
  native_with_code_slot["profiles"].at(0)["code_argument_index"] = 1;
  auto native_fingerprint_mismatch = encoded;
  native_fingerprint_mismatch["profiles"].at(0)["code_fingerprint"] =
      "sha256:" + std::string(64, 'd');
  auto invalid_fingerprint = encoded;
  invalid_fingerprint["profiles"].at(1)["executable_fingerprint"] =
      "sha256:" + std::string(64, 'G');
  auto invalid_runtime_closure = encoded;
  invalid_runtime_closure["profiles"].at(1)
      ["bootstrap_runtime_closure_fingerprint"] = "sha256:not-a-digest";
  auto duplicate_capability = encoded;
  duplicate_capability["profiles"].at(1)["provided_capabilities"] =
      std::vector<std::string>{"worker.controls", "worker.controls"};
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
  check(rejects(unknown) && rejects(future) && rejects(legacy) &&
            rejects(duplicate_profile) && rejects(duplicate_profiler) &&
            rejects(torch_profiler) && rejects(profiler_version_skew) &&
            rejects(profiler_escape) &&
            rejects(relative_root) && rejects(noncanonical_root) &&
            rejects(overlapping_roots) && rejects(executable_escape) &&
            rejects(code_escape) && rejects(working_directory_escape) &&
            rejects(builtin_runtime) && rejects(external_runtime) &&
            rejects(python_without_code) &&
            rejects(python_code_slot_out_of_range) &&
            rejects(native_with_code) && rejects(native_with_code_slot) &&
            rejects(native_fingerprint_mismatch) &&
            rejects(invalid_fingerprint) && rejects(invalid_runtime_closure) &&
            rejects(duplicate_capability) &&
            rejects(too_many_arguments) &&
            rejects(oversized_argument) && rejects(embedded_nul) &&
            rejects(secret_argument) && rejects(dollar_template_argument) &&
            rejects(brace_template_argument) &&
            rejects(empty_roots),
        "host launch registry rejects schema, duplicate, path-containment, runtime, fingerprint, and size violations");

  const auto duplicate_key_path = directory / "duplicate-key.json";
  write_text(duplicate_key_path,
             R"({"api_version":"trainvm.host-launches/v4","api_version":"trainvm.host-launches/v4","trusted_roots":["/opt/trainvm"],"profiles":[]})");
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
             "{\"api_version\":\"trainvm.host-launches/v4\","
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
  std::filesystem::copy_file("/usr/bin/true", executable);
  std::filesystem::permissions(
      executable,
      std::filesystem::perms::owner_read |
          std::filesystem::perms::owner_exec,
      std::filesystem::perm_options::replace);
  const auto profiler_executable = directory / "nsys";
  std::filesystem::copy_file("/usr/bin/true", profiler_executable);
  std::filesystem::permissions(
      profiler_executable,
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
  const std::string profiler_executable_digest =
      file_digest(profiler_executable);
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
      .bootstrap_runtime_closure_fingerprint =
          "sha256:" + std::string(64U, 'd'),
      .provided_capabilities = {"worker.controls", "worker.metrics"},
      .executable_path = executable.string(),
      .executable_fingerprint = executable_digest,
      .code_path = code.string(),
      .code_argument_index = 1U,
      .public_arguments = {"-I", "worker.pyz"},
      .working_directory = (directory / "work").string(),
  };
  const trainvm::HostLaunchRegistry registry({
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {directory.string()},
      .profiles = {profile},
      .profiler_executables =
          std::vector<trainvm::HostProfilerExecutableProfile>{
              {
                  .backend = trainvm::ProfilerBackend::nsys,
                  .version = trainvm::profiler_launch_profile(
                                 trainvm::ProfilerBackend::nsys)
                                 .version,
                  .executable_path = profiler_executable.string(),
                  .executable_fingerprint = profiler_executable_digest,
              },
              {
                  .backend = trainvm::ProfilerBackend::ncu,
                  .version = trainvm::profiler_launch_profile(
                                 trainvm::ProfilerBackend::ncu)
                                 .version,
                  .executable_path = profiler_executable.string(),
                  .executable_fingerprint = profiler_executable_digest,
              }},
  });
  const trainvm::HostIdentity host{
      .host_id = "sha256:" + std::string(64U, '1'),
      .boot_id = "11111111-1111-1111-1111-111111111111",
  };

  const auto database = directory / "journal.db";
  const std::string run_id = "host-launch-binding-run";
  trainvm::Journal journal(
      database, std::nullopt,
      trainvm::HostGrantEnforcement::legacy_process_free_test);
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
  trainvm::GpuTraceCapture external_capture{
      .enabled = true,
      .backend = trainvm::ProfilerBackend::nsys,
      .warmup_steps = 2,
      .skip_steps = 3,
      .capture_steps = 4,
      .output_artifact = "gpu_trace",
      .activities = std::vector<trainvm::ProfilerActivity>{
          trainvm::ProfilerActivity::cpu,
          trainvm::ProfilerActivity::accelerator},
      .record_shapes = false,
      .profile_memory = false,
      .with_stack = false,
  };
  const std::string external_output =
      (directory / "work" / "trainvm_artifacts" / "gpu_traces" /
       ".external" / trainvm::sha256_hex(ticket.launch_nonce))
          .string();
  auto external =
      resolver.resolve(ticket, key, external_capture, external_output);
  auto expected_profiler_arguments =
      trainvm::profiler_capture_argv(external_capture, external_output);
  expected_profiler_arguments.erase(expected_profiler_arguments.begin());
  auto ncu_capture = external_capture;
  ncu_capture.backend = trainvm::ProfilerBackend::ncu;
  auto ncu_external = resolver.resolve(
      ticket, key, ncu_capture, external_output + "-ncu");
  auto unsupported_ticket = ticket;
  unsupported_ticket.required_capabilities.push_back("worker.unimplemented");
  std::ranges::sort(unsupported_ticket.required_capabilities);
  bool unsupported_capability_rejected = false;
  try {
    (void)resolver.resolve(unsupported_ticket, key);
  } catch (const trainvm::HostLaunchResolutionError&) {
    unsupported_capability_rejected = true;
  }
  check(first.spec() == second.spec() &&
            first.spec().spec_digest.starts_with("sha256:") &&
            first.spec().identity.host_registry_digest ==
                registry.registry_digest() &&
            first.spec().identity.host_profile_digest ==
                registry.profile_digest(key, code_digest) &&
            first.spec().identity.provided_capabilities ==
                std::vector<std::string>({"worker.controls",
                                          "worker.metrics"}) &&
            first.spec().identity.code_argument_index == 1U &&
            first.spec().identity.public_arguments ==
                std::vector<std::string>({"-I", "worker.pyz"}) &&
            external.spec().identity.profiler &&
            external.spec().identity.profiler->backend ==
                trainvm::ProfilerBackend::nsys &&
            external.spec().identity.profiler->execute_from_source &&
            external.spec().identity.profiler->capture == external_capture &&
            external.spec().identity.profiler->raw_output_path ==
                external_output &&
            external.spec().identity.profiler->executable.sealed_sha256 ==
                profiler_executable_digest &&
            external.spec().identity.profiler->public_arguments ==
                expected_profiler_arguments &&
            ncu_external.spec().identity.profiler &&
            !ncu_external.spec().identity.profiler->execute_from_source &&
            unsupported_capability_rejected,
        "repeated host resolution produces one deterministic versioned binding and rejects requirements absent from sealed worker capability authority");

  const auto external_profiler_fd =
      external.duplicate_profiler_executable_fd();
  const auto external_authority_fd =
      external.duplicate_profiler_authority_fd();
  const auto ncu_profiler_fd =
      ncu_external.duplicate_profiler_executable_fd();
  struct stat pinned_profiler_metadata {};
  const bool pinned_profiler_matches =
      external_profiler_fd &&
      ::fstat(*external_profiler_fd, &pinned_profiler_metadata) == 0 &&
      static_cast<std::uint64_t>(pinned_profiler_metadata.st_dev) ==
          external.spec().identity.profiler->executable.source_device &&
      static_cast<std::uint64_t>(pinned_profiler_metadata.st_ino) ==
          external.spec().identity.profiler->executable.source_inode;
  constexpr int required_profiler_seals =
      F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
  check(external_profiler_fd && external_authority_fd &&
            pinned_profiler_matches && ncu_profiler_fd &&
            (::fcntl(*ncu_profiler_fd, F_GET_SEALS) &
             required_profiler_seals) == required_profiler_seals &&
            trainvm::external_profiler_authority_from_sealed_fd(
                *external_authority_fd,
                external.spec().identity.profiler->authority.authority_digest) ==
                external.spec().identity.profiler->authority,
        "external resolution retains backend-compatible profiler bytes and exact worker authority");
  if (ncu_profiler_fd) (void)::close(*ncu_profiler_fd);
  if (external_profiler_fd && external_authority_fd) {
    const int external_worker_fd = external.duplicate_executable_fd();
    const auto external_code_fd = external.duplicate_code_fd();
    const int external_work_fd = external.duplicate_working_directory_fd();
    auto adopted_external = trainvm::ResolvedLaunch::adopt_delegated(
        external.spec(), external_worker_fd, external_code_fd,
        external_work_fd, external_profiler_fd, external_authority_fd);
    check(adopted_external.spec() == external.spec(),
          "delegated external profiler authority is reattested exactly");
    (void)::close(external_worker_fd);
    if (external_code_fd) (void)::close(*external_code_fd);
    (void)::close(external_work_fd);
    (void)::close(*external_profiler_fd);
    (void)::close(*external_authority_fd);
  }

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
  auto delegated = trainvm::ResolvedLaunch::adopt_delegated(
      first.spec(), executable_fd, code_fd, work_fd);
  const int delegated_copy = delegated.duplicate_executable_fd();
  auto mismatched_delegation = first.spec();
  mismatched_delegation.identity.executable.sealed_sha256 =
      "sha256:" + std::string(64U, 'f');
  mismatched_delegation.spec_digest =
      "sha256:" + trainvm::sha256_hex(
                       trainvm::resolved_launch_identity_json(
                           mismatched_delegation.identity)
                           .dump());
  bool mismatched_delegation_rejected = false;
  try {
    (void)trainvm::ResolvedLaunch::adopt_delegated(
        mismatched_delegation, executable_fd, code_fd, work_fd);
  } catch (const trainvm::HostLaunchResolutionError&) {
    mismatched_delegation_rejected = true;
  }
  check(delegated_copy >= 0 && mismatched_delegation_rejected,
        "delegated launch descriptors are independently retained and reattested against exact sealed bytes");
  if (delegated_copy >= 0) (void)::close(delegated_copy);
  (void)::close(executable_fd);
  if (code_fd) (void)::close(*code_fd);
  (void)::close(work_fd);

  const nlohmann::json public_manifest =
      trainvm::resolved_launch_spec_json(first.spec());
  const auto decoded = trainvm::resolved_launch_spec_from_json(public_manifest);
  const auto decoded_external = trainvm::resolved_launch_spec_from_json(
      trainvm::resolved_launch_spec_json(external.spec()));
  const std::string manifest_text = public_manifest.dump();
  check(decoded == first.spec() && decoded_external == external.spec() &&
            manifest_text.find("authorization_token") == std::string::npos &&
            manifest_text.find("process_instance") == std::string::npos &&
            manifest_text.find("secret://") == std::string::npos,
        "resolved binding round-trips canonically without process credentials or secrets");
  auto forged = first.spec();
  forged.identity.api_version = "trainvm.resolved-launch/v1";
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
  auto forged_external = external.spec();
  forged_external.identity.profiler->execute_from_source = false;
  forged_external.spec_digest =
      "sha256:" + trainvm::sha256_hex(
                       trainvm::resolved_launch_identity_json(
                           forged_external.identity)
                           .dump());
  bool forged_external_rejected = false;
  try {
    (void)trainvm::resolved_launch_spec_from_json(
        trainvm::resolved_launch_spec_json(forged_external));
  } catch (const std::invalid_argument&) {
    forged_external_rejected = true;
  }
  auto moved = std::move(second);
  auto move_assigned = resolver.resolve(ticket, key);
  move_assigned = std::move(moved);
  const int moved_fd = move_assigned.duplicate_executable_fd();
  check(forged_rejected && forged_external_rejected && moved_fd >= 0,
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
      .api_version = "trainvm.host-launches/v4",
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
      .api_version = "trainvm.host-launches/v4",
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
      .api_version = "trainvm.host-launches/v4",
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
  std::filesystem::copy_file("/usr/bin/true", executable);
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
        .bootstrap_runtime_closure_fingerprint =
            "sha256:" + std::string(64U, 'd'),
        .provided_capabilities = profile.required_capabilities,
        .executable_path = executable.string(),
        .executable_fingerprint = executable_digest,
        .code_path = code.string(),
        .code_argument_index = 1U,
        .public_arguments = {"-I", "worker.pyz"},
        .working_directory = (directory / "work").string(),
    });
  }
  const trainvm::HostLaunchRegistryDocument host_document{
      .api_version = "trainvm.host-launches/v4",
      .trusted_roots = {directory.string()},
      .profiles = launch_profiles,
  };
  const trainvm::HostIdentity host{
      .host_id = "sha256:" + std::string(64U, '3'),
      .boot_id = kTestBootId,
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
        [&now_ns] { return test_time(now_ns); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

    trainvm::JournalHostdMutationClaimProvider process_claims(
        service.journal_,
        {.api_version = std::string(
             trainvm::kHostdMutationClaimProviderApiVersion),
         .broker_epoch = "broker-process-claim",
         .authority_clock = [&now_ns] { return test_time(now_ns); },
         .controller_id_source = [] {
           return std::string("process-claim-controller-001");
         }});
    const auto process_open = process_claims.open_for_process(
        binding.identity.launch_event_id);
    bool missing_process_claim_rejected = false;
    try {
      (void)process_claims.open_for_process("missing-launch");
    } catch (const trainvm::HostdMutationClaimProviderError&) {
      missing_process_claim_rejected = true;
    }
    check(missing_process_claim_rejected &&
              process_open.claim.controller.run_id == binding.identity.run_id &&
              process_open.claim.controller.concurrency_key ==
                  binding.identity.concurrency_key &&
              process_open.claim.controller.logical_lease_id ==
                  binding.identity.lease_id &&
              process_open.claim.controller.logical_fencing_token ==
                  binding.identity.fencing_token,
          "process mutation claims derive only from a durable launch binding");

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
        [] { return test_time(1'250); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
              [] { return test_time(1'300); },
              trainvm::HostGrantEnforcement::legacy_process_free_test);
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
          .api_version = "trainvm.host-launches/v4",
          .trusted_roots = {directory.string()},
          .profiles = std::move(changed_profiles),
      }),
      host, adapter_profiles);
  const bool registry_drift_rejected = rejects_without_mutation(
      trainvm::HostLaunchRegistry({
          .api_version = "trainvm.host-launches/v4",
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
        [] { return test_time(2'000); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

class ServiceBusyGrantClient final : public trainvm::IHostGrantClient {
 public:
  trainvm::BundleRequestResult request_bundle(
      const trainvm::ResourceBundleRequest& request) override {
    ++calls;
    last_request = request;
    return {.status = trainvm::BundleRequestStatus::busy,
            .grant = std::nullopt,
            .outcome_digest = "sha256:" + std::string(64U, 'b'),
            .replayed = false};
  }

  trainvm::BundleReleaseResult release_bundle(
      const trainvm::ResourceReleaseRequest&) override {
    throw std::runtime_error("busy service fixture cannot release");
  }

  std::size_t calls{};
  std::optional<trainvm::ResourceBundleRequest> last_request;
};

class ServiceNeverProcessClient final : public trainvm::IHostProcessClient {
 public:
  trainvm::HostdProcessPreparedResult prepare_process(
      const trainvm::HostdProcessPrepareRequest&,
      const trainvm::ResolvedLaunch&,
      const trainvm::SealedWorkerBootstrap&) override {
    throw std::runtime_error("busy service fixture cannot prepare a process");
  }

  trainvm::HostdProcessCommittedResult commit_process(
      const trainvm::HostdProcessCommitRequest&) override {
    throw std::runtime_error("busy service fixture cannot commit a process");
  }

  trainvm::HostProcessExitResult finalize_process(
      const trainvm::HostdProcessExitCommand&) override {
    throw std::runtime_error("busy service fixture cannot finalize a process");
  }
};

void test_service_host_grant_reconciliation() {
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "service host grant fixture compiles");
  if (!compiled.plan) return;
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-service-grant-test-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const auto database = directory / "journal.db";
  const std::string run_id = "service-grant-run";
  const auto adapters = fixture_adapter_profiles();
  {
    trainvm::AdapterRegistry registry(adapters);
    trainvm::Journal journal(database);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(
        adapter_locked_submission(*compiled.plan, registry));
  }
  auto grant = std::make_shared<ServiceBusyGrantClient>();
  auto process = std::make_shared<ServiceNeverProcessClient>();
  const trainvm::HostIdentity host{
      .host_id = "sha256:" + std::string(64U, '7'),
      .boot_id = kTestBootId,
  };
  std::int64_t now = 1'000;
  trainvm::TrainVMService service(
      database, trainvm::AdapterRegistry(adapters),
      fixture_disabled_host_launch_registry(), host,
      [&now] { return test_time(now); },
      trainvm::HostGrantEnforcement::required,
      trainvm::TrainingComponentRegistry({}), grant, process,
      "unix:/tmp/trainvm-service-grant.sock");
  const auto acquired = service.reconcile_once(run_id);
  now = 1'100;
  const auto busy = service.reconcile_once(run_id);
  now = 1'200;
  const auto busy_replay = service.reconcile_once(run_id);
  check(acquired.disposition ==
            trainvm::ReconcileDisposition::lease_acquired &&
            busy.disposition ==
                trainvm::ReconcileDisposition::host_grant_busy &&
            busy_replay.disposition ==
                trainvm::ReconcileDisposition::host_grant_busy &&
            grant->calls == 1U && grant->last_request &&
            grant->last_request->journal_id == service.journal_.journal_id() &&
            grant->last_request->run_id == run_id &&
            grant->last_request->count == 1U &&
            grant->last_request->selector.vendor ==
                trainvm::HostAcceleratorVendor::nvidia &&
            service.journal_
                .host_grant_saga(grant->last_request->request_id)
                ->busy_outcome_digest ==
                std::optional<std::string>{"sha256:" +
                                           std::string(64U, 'b')},
        "service lowers one exact resource request and replays durable busy without contacting hostd twice");
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
          .api_version = "trainvm.adapters/v2",
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
                 .api_version = "trainvm.adapters/v2",
                 .profiles = std::move(reversed_profiles),
             }));
  const trainvm::AdapterRegistry reordered =
      trainvm::AdapterRegistry::load_file(reordered_path);
  auto composition_document = original_document;
  composition_document["profiles"].at(0)["training_composition"] = {
      {"model_family", "mageflow"},
      {"slots", {{"optimizer", "optimizer"},
                 {"backbone_activation", "activation"}}},
  };
  const auto composition_path = directory / "registry-composition.json";
  write_json(composition_path, composition_document);
  const trainvm::AdapterRegistry composition_registry =
      trainvm::AdapterRegistry::load_file(composition_path);
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "registry file fixture plan compiles");
  if (compiled.valid()) {
    const auto& mageflow =
        compiled.plan->experiment.spec.components.at("mageflow");
    const auto& resolved = loaded.resolve(mageflow, "train");
    const auto& composition_resolved =
        composition_registry.resolve(mageflow, "train");
    check(resolved.code_fingerprint ==
              "sha256:" + std::string(64, 'a') &&
              resolved.required_capabilities ==
                  std::vector<std::string>({"worker.controls",
                                            "worker.metrics"}) &&
              resolved.lifecycle.stateful &&
              resolved.lifecycle.resume_grade ==
                  trainvm::ResumeGrade::exact &&
              original_document["profiles"].at(0).contains("lifecycle") &&
              original_document["profiles"].at(0)["lifecycle"]
                               ["resume_grade"] == "exact" &&
              loaded.registry_digest().starts_with("sha256:") &&
              loaded.registry_digest().size() == 71U &&
              loaded.registry_digest() == reordered.registry_digest() &&
              loaded.registry_digest() ==
                  trainvm::AdapterRegistry(fixture_adapter_profiles())
                      .registry_digest() &&
              composition_resolved.training_composition &&
              composition_resolved.training_composition->model_family ==
                  "mageflow" &&
              composition_resolved.training_composition->slots ==
                  std::map<std::string,
                           trainvm::TrainingComponentCategory>{
                      {"backbone_activation",
                       trainvm::TrainingComponentCategory::activation},
                      {"optimizer",
                       trainvm::TrainingComponentCategory::optimizer}},
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
  future_document["api_version"] = "trainvm.adapters/v3";
  const auto future_path = directory / "registry-future.json";
  write_json(future_path, future_document);
  bool version_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(future_path);
  } catch (const std::invalid_argument&) {
    version_rejected = true;
  }

  auto legacy_document = original_document;
  legacy_document["api_version"] = "trainvm.adapters/v1";
  const auto legacy_path = directory / "registry-legacy.json";
  write_json(legacy_path, legacy_document);
  bool legacy_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(legacy_path);
  } catch (const std::invalid_argument&) {
    legacy_rejected = true;
  }

  auto missing_lifecycle_document = original_document;
  missing_lifecycle_document["profiles"].at(0).erase("lifecycle");
  const auto missing_lifecycle_path =
      directory / "registry-missing-lifecycle.json";
  write_json(missing_lifecycle_path, missing_lifecycle_document);
  bool missing_lifecycle_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(missing_lifecycle_path);
  } catch (const std::invalid_argument&) {
    missing_lifecycle_rejected = true;
  }

  auto missing_lifecycle_field_document = original_document;
  missing_lifecycle_field_document["profiles"].at(0)["lifecycle"].erase(
      "graceful_stop");
  const auto missing_lifecycle_field_path =
      directory / "registry-missing-lifecycle-field.json";
  write_json(missing_lifecycle_field_path, missing_lifecycle_field_document);
  bool missing_lifecycle_field_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(missing_lifecycle_field_path);
  } catch (const std::invalid_argument&) {
    missing_lifecycle_field_rejected = true;
  }

  auto unknown_lifecycle_field_document = original_document;
  unknown_lifecycle_field_document["profiles"].at(0)["lifecycle"]
                                  ["checkpoint_magic"] = true;
  const auto unknown_lifecycle_field_path =
      directory / "registry-unknown-lifecycle-field.json";
  write_json(unknown_lifecycle_field_path, unknown_lifecycle_field_document);
  bool unknown_lifecycle_field_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(unknown_lifecycle_field_path);
  } catch (const std::invalid_argument&) {
    unknown_lifecycle_field_rejected = true;
  }

  auto unknown_resume_grade_document = original_document;
  unknown_resume_grade_document["profiles"].at(0)["lifecycle"]
                               ["resume_grade"] = "best_effort";
  const auto unknown_resume_grade_path =
      directory / "registry-unknown-resume-grade.json";
  write_json(unknown_resume_grade_path, unknown_resume_grade_document);
  bool unknown_resume_grade_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(unknown_resume_grade_path);
  } catch (const std::invalid_argument&) {
    unknown_resume_grade_rejected = true;
  }

  auto malformed_composition_document = original_document;
  malformed_composition_document["profiles"].at(0)
                                ["training_composition"] = {
      {"model_family", "mageflow"}, {"slots", nlohmann::json::object()}};
  const auto malformed_composition_path =
      directory / "registry-malformed-composition.json";
  write_json(malformed_composition_path, malformed_composition_document);
  bool malformed_composition_rejected = false;
  try {
    (void)trainvm::AdapterRegistry::load_file(malformed_composition_path);
  } catch (const std::invalid_argument&) {
    malformed_composition_rejected = true;
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
    duplicate_key << R"({"api_version":"trainvm.adapters/v2","api_version":"trainvm.adapters/v2","profiles":[]})";
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
            legacy_rejected && missing_lifecycle_rejected &&
            missing_lifecycle_field_rejected &&
            unknown_lifecycle_field_rejected &&
            unknown_resume_grade_rejected &&
            malformed_composition_rejected &&
            reserved_namespace_rejected && duplicate_key_rejected &&
            empty_rejected && oversized_rejected,
        "registry file loading rejects symlinks, missing or unknown lifecycle fields, unknown grades, malformed training contracts, incompatible versions, reserved trainvm.core names, duplicate keys, and unbounded files");
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
        fixture_test_host_identity(),
        [&authority_now_ns] { return test_time(authority_now_ns); },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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

    auto second_request = request_for(
        journal_id, true, "service-reconcile-second",
        preview.adapter_lock_digest());
    trainvm::v1::SubmitExperimentResponse second_created;
    const grpc::Status second_status = service.SubmitExperiment(
        nullptr, &second_request, &second_created);
    const std::string second_run_id =
        second_created.has_run() ? second_created.run().run_id() : "";
    trainvm::v1::GetRunRequest get_request;
    get_request.set_run_id(run_id);
    trainvm::v1::RunSummary run_summary;
    const grpc::Status get_status =
        service.GetRun(nullptr, &get_request, &run_summary);
    trainvm::v1::GetCompiledPlanRequest plan_request;
    plan_request.set_run_id(run_id);
    trainvm::v1::GetCompiledPlanResponse plan_response;
    const grpc::Status plan_status =
        service.GetCompiledPlan(nullptr, &plan_request, &plan_response);
    trainvm::v1::GetCompiledPlanRequest missing_plan_request;
    missing_plan_request.set_run_id("missing-run");
    trainvm::v1::GetCompiledPlanResponse missing_plan_response;
    const grpc::Status missing_plan_status = service.GetCompiledPlan(
        nullptr, &missing_plan_request, &missing_plan_response);
    trainvm::v1::ListRunsRequest list_request;
    list_request.add_observed_states(
        trainvm::v1::OBSERVED_STATE_QUEUED);
    (*list_request.mutable_labels())["family"] = "mageflow";
    list_request.set_limit(1U);
    trainvm::v1::ListRunsResponse first_page;
    const grpc::Status first_page_status =
        service.ListRuns(nullptr, &list_request, &first_page);
    trainvm::v1::ListRunsRequest next_request = list_request;
    next_request.set_page_token(first_page.next_page_token());
    trainvm::v1::ListRunsResponse second_page;
    const grpc::Status second_page_status =
        service.ListRuns(nullptr, &next_request, &second_page);
    trainvm::v1::ListRunsRequest mismatched_page = next_request;
    (*mismatched_page.mutable_labels())["family"] = "rwkv";
    trainvm::v1::ListRunsResponse mismatched_response;
    const grpc::Status mismatched_status = service.ListRuns(
        nullptr, &mismatched_page, &mismatched_response);
    trainvm::v1::ListRunsRequest malformed_page = list_request;
    malformed_page.set_page_token("{}");
    trainvm::v1::ListRunsResponse malformed_response;
    const grpc::Status malformed_status = service.ListRuns(
        nullptr, &malformed_page, &malformed_response);
    trainvm::v1::PlanDiffRequest same_diff;
    same_diff.set_run_id(run_id);
    same_diff.set_expected_revision(created.run().revision());
    same_diff.set_proposed_source_document(fixture.dump());
    same_diff.set_source_format("json");
    same_diff.set_expected_journal_id(journal_id);
    same_diff.set_expected_current_plan_hash(created.plan_hash());
    same_diff.set_expected_proposed_plan_hash(compiled.plan->plan_hash);
    trainvm::v1::PlanDiffResponse same_diff_response;
    const grpc::Status same_diff_status =
        service.DiffPlan(nullptr, &same_diff, &same_diff_response);
    auto changed_fixture = fixture;
    changed_fixture["metadata"]["description"] =
        "read-plane semantic diff fixture";
    const auto changed_compiled =
        trainvm::compile_document(changed_fixture);
    trainvm::v1::PlanDiffRequest changed_diff = same_diff;
    changed_diff.set_proposed_source_document(changed_fixture.dump());
    changed_diff.set_expected_proposed_plan_hash(
        changed_compiled.plan->plan_hash);
    trainvm::v1::PlanDiffResponse changed_diff_response;
    const grpc::Status changed_diff_status =
        service.DiffPlan(nullptr, &changed_diff, &changed_diff_response);
    trainvm::v1::PlanDiffRequest stale_diff = same_diff;
    stale_diff.set_expected_revision(created.run().revision() + 1U);
    trainvm::v1::PlanDiffResponse stale_diff_response;
    const grpc::Status stale_diff_status =
        service.DiffPlan(nullptr, &stale_diff, &stale_diff_response);
    trainvm::v1::PlanDiffRequest wrong_diff_journal = same_diff;
    wrong_diff_journal.set_expected_journal_id(
        "00000000000000000000000000000000");
    trainvm::v1::PlanDiffResponse wrong_diff_journal_response;
    const grpc::Status wrong_diff_journal_status = service.DiffPlan(
        nullptr, &wrong_diff_journal, &wrong_diff_journal_response);
    trainvm::v1::PlanDiffRequest wrong_current_plan = same_diff;
    wrong_current_plan.set_expected_current_plan_hash(std::string(64, '0'));
    trainvm::v1::PlanDiffResponse wrong_current_plan_response;
    const grpc::Status wrong_current_plan_status = service.DiffPlan(
        nullptr, &wrong_current_plan, &wrong_current_plan_response);
    trainvm::v1::PlanDiffRequest wrong_proposed_plan = same_diff;
    wrong_proposed_plan.set_expected_proposed_plan_hash(
        std::string(64, '0'));
    trainvm::v1::PlanDiffResponse wrong_proposed_plan_response;
    const grpc::Status wrong_proposed_plan_status = service.DiffPlan(
        nullptr, &wrong_proposed_plan, &wrong_proposed_plan_response);
    const auto first_page_id =
        first_page.runs_size() == 1
            ? first_page.runs(0).identity().run_id()
            : std::string{};
    const auto second_page_id =
        second_page.runs_size() == 1
            ? second_page.runs(0).identity().run_id()
            : std::string{};
    check(second_status.ok() && second_created.has_run() &&
              !second_run_id.empty() && get_status.ok() &&
              run_summary.identity().run_id() == run_id && plan_status.ok() &&
              plan_response.journal_id() == journal_id &&
              plan_response.run().run_id() == run_id &&
              plan_response.run().revision() == created.run().revision() &&
              plan_response.run().plan_hash() == created.plan_hash() &&
              nlohmann::json::parse(plan_response.canonical_plan_json()) ==
                  nlohmann::json::parse(created.canonical_plan()) &&
              missing_plan_status.error_code() == grpc::StatusCode::NOT_FOUND &&
              run_summary.observed_state() ==
                  trainvm::v1::OBSERVED_STATE_QUEUED &&
              run_summary.last_event_sequence() > 0U &&
              run_summary.has_created_at() && run_summary.has_updated_at() &&
              first_page_status.ok() && first_page.runs_size() == 1 &&
              first_page.journal_id() == journal_id &&
              !first_page.next_page_token().empty() &&
              second_page_status.ok() && second_page.runs_size() == 1 &&
              second_page.next_page_token().empty() &&
              first_page_id != second_page_id &&
              std::set<std::string>({first_page_id, second_page_id}) ==
                  std::set<std::string>({run_id, second_run_id}) &&
              mismatched_status.error_code() ==
                  grpc::StatusCode::INVALID_ARGUMENT &&
              malformed_status.error_code() ==
                  grpc::StatusCode::INVALID_ARGUMENT &&
              same_diff_status.ok() &&
              same_diff_response.adoptable_in_place() &&
              same_diff_response.semantic_diff() == "[]" &&
              same_diff_response.proposed_plan_hash() ==
                  compiled.plan->plan_hash &&
              changed_diff_status.ok() &&
              !changed_diff_response.adoptable_in_place() &&
              changed_diff_response.semantic_diff() != "[]" &&
              changed_diff_response.diagnostics_size() == 1 &&
              stale_diff_status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION &&
              wrong_diff_journal_status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION &&
              wrong_current_plan_status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION &&
              wrong_proposed_plan_status.error_code() ==
                  grpc::StatusCode::FAILED_PRECONDITION,
          "native read APIs expose bounded summaries, query-bound pagination, and honest semantic plan diffs");

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
    const auto sequenced_launches = observer.sequenced_events({
        .after_journal_sequence = 0U,
        .run_ids = {run_id},
        .event_types = {"worker.launch_requested"},
        .limit = 8U,
    });
    const auto no_more_launches = observer.sequenced_events({
        .after_journal_sequence =
            sequenced_launches.empty()
                ? 0U
                : sequenced_launches.back().journal_sequence,
        .run_ids = {run_id},
        .event_types = {"worker.launch_requested"},
        .limit = 8U,
    });
    const auto newest_events = observer.sequenced_events({
        .after_journal_sequence = 0U,
        .through_journal_sequence = projection
                                        ? projection->last_event_sequence
                                        : 0U,
        .run_ids = {run_id},
        .event_types = {},
        .limit = 3U,
        .newest_first = true,
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
              newest_events.size() == 3U &&
              newest_events[0].journal_sequence >
                  newest_events[1].journal_sequence &&
              newest_events[1].journal_sequence >
                  newest_events[2].journal_sequence &&
              projection->run_revision == 4U && events.size() == 7U &&
              launch_event != events.end() &&
              launch_event->wall_time_ns == 5'100 &&
              sequenced_launches.size() == 1U &&
              sequenced_launches.front().journal_sequence > 0U &&
              sequenced_launches.front().event.event_id ==
                  launch_event->event_id &&
              sequenced_launches.front().event.event_type ==
                  launch_event->event_type &&
              sequenced_launches.front().event.payload ==
                  launch_event->payload &&
              no_more_launches.empty(),
          "service-owned reconcile path uses its registry, mutation gate, and authority clock for acquisition then launch");
  }

  {
    nlohmann::json training_fixture = fixture;
    training_fixture["spec"]["workflow"]["nodes"]["train_to_boundary"]
                    ["invoke"]["training"] = {
        {"model_family", "mageflow"},
        {"components",
         {{"backbone_activation",
           {{"key",
             {{"category", "activation"},
              {"name", "silu"},
              {"version", "1.0.0"}}},
            {"configuration", nlohmann::json::object()}}}}},
    };
    training_fixture["spec"]["workflow"]["nodes"]["resume_training"]
                    ["invoke"]["training"] =
        training_fixture["spec"]["workflow"]["nodes"]
                        ["train_to_boundary"]["invoke"]["training"];
    const auto training_plan = trainvm::compile_document(training_fixture);
    check(training_plan.valid(),
          "training-component service fixture compiles");
    if (training_plan.valid()) {
      auto training_profiles = fixture_adapter_profiles();
      const auto training_profile = std::ranges::find_if(
          training_profiles, [](const trainvm::AdapterProfile& profile) {
            return profile.key.operation == "train";
          });
      if (training_profile != training_profiles.end()) {
        training_profile->training_composition =
            trainvm::TrainingCompositionContract{
                .model_family = "mageflow",
                .slots = {{"backbone_activation",
                           trainvm::TrainingComponentCategory::activation}},
            };
      }
      const trainvm::AdapterRegistry expected_operation_registry(
          training_profiles);
      const auto database_path = directory / "training-components.db";
      std::string journal_id;
      {
        trainvm::Journal journal(database_path);
        journal_id = journal.journal_id();
      }
      std::int64_t authority_now_ns = 8'000;
      trainvm::TrainVMService service(
          database_path,
          trainvm::AdapterRegistry(std::move(training_profiles)),
          fixture_disabled_host_launch_registry(),
          fixture_test_host_identity(),
          [&authority_now_ns] { return test_time(authority_now_ns); },
          trainvm::HostGrantEnforcement::legacy_process_free_test,
          fixture_training_component_registry());
      trainvm::v1::SubmitExperimentRequest preview_request;
      preview_request.set_source_document(training_fixture.dump());
      preview_request.set_source_format("json");
      preview_request.set_expected_journal_id(journal_id);
      trainvm::v1::SubmitExperimentResponse preview;
      const grpc::Status preview_status = service.SubmitExperiment(
          nullptr, &preview_request, &preview);

      trainvm::v1::DescriptorRequest descriptor_request;
      descriptor_request.set_adapter("trainvm.training-components");
      descriptor_request.set_version("1.0.0");
      trainvm::v1::DescriptorResponse descriptor_response;
      const grpc::Status descriptor_status = service.GetDescriptor(
          nullptr, &descriptor_request, &descriptor_response);
      trainvm::v1::DescriptorRequest operations_request;
      operations_request.set_adapter("trainvm.operations");
      operations_request.set_version("1.0.0");
      trainvm::v1::DescriptorResponse operations_response;
      const grpc::Status operations_status = service.GetDescriptor(
          nullptr, &operations_request, &operations_response);
      trainvm::v1::DescriptorRequest unknown_descriptor;
      unknown_descriptor.set_adapter("trainvm.operations");
      unknown_descriptor.set_version("2.0.0");
      trainvm::v1::DescriptorResponse unknown_descriptor_response;
      const grpc::Status unknown_descriptor_status = service.GetDescriptor(
          nullptr, &unknown_descriptor, &unknown_descriptor_response);

      trainvm::v1::SubmitExperimentRequest stale = preview_request;
      stale.set_create_run(true);
      stale.set_idempotency_key("training-components");
      stale.set_author("scheduler");
      stale.set_reason("lock the reflected training composition");
      stale.set_expected_plan_hash(training_plan.plan->plan_hash);
      stale.set_expected_adapter_lock_digest(
          preview.adapter_lock_digest());
      stale.set_expected_training_component_lock_digest(
          "sha256:" + std::string(64U, '0'));
      trainvm::v1::SubmitExperimentResponse stale_response;
      const grpc::Status stale_status = service.SubmitExperiment(
          nullptr, &stale, &stale_response);

      auto create_request = stale;
      create_request.set_expected_training_component_lock_digest(
          preview.training_component_lock_digest());
      trainvm::v1::SubmitExperimentResponse created;
      const grpc::Status create_status = service.SubmitExperiment(
          nullptr, &create_request, &created);
      trainvm::v1::SubmitExperimentResponse replayed;
      const grpc::Status replay_status = service.SubmitExperiment(
          nullptr, &create_request, &replayed);
      trainvm::Journal observer(database_path);
      const auto event = created.has_run()
                             ? observer.event(created.run().run_id() +
                                              ":created")
                             : std::nullopt;
      check(preview_status.ok() &&
                descriptor_status.ok() &&
                descriptor_response.schema_hash() ==
                    fixture_training_component_registry().registry_digest() &&
                nlohmann::json::parse(descriptor_response.schema_json()) ==
                    fixture_training_component_registry().document_json() &&
                operations_status.ok() &&
                operations_response.schema_hash() ==
                    expected_operation_registry
                        .operation_descriptors_digest() &&
                nlohmann::json::parse(operations_response.schema_json()) ==
                    expected_operation_registry
                        .operation_descriptors_json() &&
                nlohmann::json::parse(operations_response.schema_json())
                        .at("operations")
                        .size() == 6U &&
                unknown_descriptor_status.error_code() ==
                    grpc::StatusCode::NOT_FOUND &&
                preview.training_component_lock_digest().starts_with(
                    "sha256:") &&
                preview.training_component_lock_digest().size() == 71U &&
                !preview.canonical_training_component_lock().empty() &&
                stale_status.error_code() ==
                    grpc::StatusCode::FAILED_PRECONDITION &&
                observer.event_count() == 1U && create_status.ok() &&
                replay_status.ok() && created.has_run() &&
                replayed.has_run() &&
                created.run().run_id() == replayed.run().run_id() && event &&
                event->payload.at("submission")
                        .at("training_component_lock_digest") ==
                    preview.training_component_lock_digest() &&
                event->payload.at("submission")
                        .at("training_component_lock") ==
                    nlohmann::json::parse(
                        preview.canonical_training_component_lock()),
            "service previews, fences, persists, and exactly replays the reflected training-component lock");

      const auto acquired = service.reconcile_once(created.run().run_id());
      authority_now_ns = 8'100;
      const auto launched = service.reconcile_once(created.run().run_id());
      check(acquired.disposition ==
                trainvm::ReconcileDisposition::lease_acquired &&
                launched.disposition ==
                    trainvm::ReconcileDisposition::launch_prepared &&
                launched.launch &&
                launched.launch->required_capabilities ==
                    std::vector<std::string>{"activation.silu",
                                             "worker.controls",
                                             "worker.metrics"},
            "service launch authority unions adapter and resolved training-component capabilities");
    }
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
  const nlohmann::json adapter_lock =
      nlohmann::json::parse(registry.plan_lock_manifest(*compiled.plan));
  auto legacy_lock_submission = submission_identity;
  legacy_lock_submission["adapter_lock"]["api_version"] =
      "trainvm.adapter-lock/v1";
  bool legacy_lock_rejected = false;
  try {
    registry.validate_submission_lock(*compiled.plan, legacy_lock_submission);
  } catch (const trainvm::AdapterResolutionError&) {
    legacy_lock_rejected = true;
  }
  check(adapter_lock.at("api_version") == "trainvm.adapter-lock/v2" &&
            legacy_lock_rejected,
        "adapter lock v2 binds lifecycle schema and rejects legacy v1 manifests");
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
                                          "worker.metrics"}) &&
            train_profile.lifecycle.stateful &&
            train_profile.lifecycle.checkpoint_now &&
            train_profile.lifecycle.pause_release_resources &&
            train_profile.lifecycle.resume_grade ==
                trainvm::ResumeGrade::exact,
        "adapter registry resolves exact authority-owned operation and lifecycle profiles while allowing a stateless process node in an exact-recovery plan");

  const auto rejects_authoring_plan = [&](const trainvm::CompiledPlan& plan,
                                          std::vector<trainvm::AdapterProfile>
                                              profiles =
                                                  fixture_adapter_profiles()) {
    try {
      trainvm::AdapterRegistry(std::move(profiles)).validate_plan(plan);
    } catch (const trainvm::AdapterResolutionError&) {
      return true;
    }
    return false;
  };
  auto missing_input_plan = *compiled.plan;
  missing_input_plan.experiment.spec.workflow.nodes.at("train_to_boundary")
      .invoke.inputs.erase("config");
  auto unknown_input_plan = *compiled.plan;
  unknown_input_plan.experiment.spec.workflow.nodes.at("train_to_boundary")
      .invoke.inputs.emplace("invented", trainvm::Binding{
                                        .literal = "value",
                                        .parameter = std::nullopt,
                                        .artifact = std::nullopt,
                                        .control = std::nullopt,
                                        .context = std::nullopt,
                                        .node_output = std::nullopt,
                                    });
  auto wrong_value_type_plan = *compiled.plan;
  auto& wrong_value_binding = wrong_value_type_plan.experiment.spec.workflow
                                  .nodes.at("train_to_boundary")
                                  .invoke.inputs.at("stop_at_step");
  wrong_value_binding.parameter.reset();
  wrong_value_binding.literal = "not-an-integer";
  auto wrong_artifact_type_plan = *compiled.plan;
  wrong_artifact_type_plan.experiment.spec.workflow.nodes.at("resume_training")
      .invoke.inputs.at("checkpoint").artifact = "eval_gallery";
  auto wrong_artifact_schema_plan = *compiled.plan;
  wrong_artifact_schema_plan.experiment.spec.artifacts.at("encoder_cache")
      .schema = "rwkv-lab.wrong-cache.v1";
  auto undeclared_publish_plan = *compiled.plan;
  undeclared_publish_plan.experiment.spec.workflow.nodes.at("train_to_boundary")
      .publishes->emplace("invented", "checkpoint");
  auto required_output_profiles = fixture_adapter_profiles();
  const auto required_output_profile = std::ranges::find_if(
      required_output_profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (required_output_profile != required_output_profiles.end()) {
    required_output_profile->authoring->outputs.at("checkpoint").required =
        true;
  }
  auto missing_output_plan = *compiled.plan;
  missing_output_plan.experiment.spec.workflow.nodes.at("train_to_boundary")
      .publishes->erase("checkpoint");
  check(rejects_authoring_plan(missing_input_plan) &&
            rejects_authoring_plan(unknown_input_plan) &&
            rejects_authoring_plan(wrong_value_type_plan) &&
            rejects_authoring_plan(wrong_artifact_type_plan) &&
            rejects_authoring_plan(wrong_artifact_schema_plan) &&
            rejects_authoring_plan(undeclared_publish_plan) &&
            rejects_authoring_plan(missing_output_plan,
                                   std::move(required_output_profiles)),
        "operation authoring authority rejects missing and unknown inputs, value and artifact contract mismatches, undeclared publishes, and omitted required outputs");

  auto compatible_train_profiles = fixture_adapter_profiles();
  const auto compatible_train = std::ranges::find_if(
      compatible_train_profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (compatible_train != compatible_train_profiles.end()) {
    compatible_train->lifecycle.resume_grade =
        trainvm::ResumeGrade::compatible;
  }
  bool stateful_downgrade_rejected = false;
  try {
    const trainvm::AdapterRegistry compatible_registry(
        compatible_train_profiles);
    compatible_registry.validate_plan(*compiled.plan);
  } catch (const trainvm::AdapterResolutionError&) {
    stateful_downgrade_rejected = true;
  }
  auto restart_plan_source = load_fixture();
  restart_plan_source["spec"]["recovery"]["exact_resume"] = false;
  const auto restart_plan = trainvm::compile_document(restart_plan_source);
  bool compatible_nonexact_accepted = restart_plan.valid();
  if (restart_plan.valid()) {
    try {
      trainvm::AdapterRegistry(std::move(compatible_train_profiles))
          .validate_plan(*restart_plan.plan);
    } catch (const std::exception&) {
      compatible_nonexact_accepted = false;
    }
  }
  check(stateful_downgrade_rejected && compatible_nonexact_accepted,
        "exact recovery rejects a compatible-only stateful trainer while the same mixed stateful/stateless graph remains valid without an exact request");

  nlohmann::json composed_source = load_fixture();
  const nlohmann::json activation_composition = {
      {"model_family", "mageflow"},
      {"components",
       {{"backbone_activation",
         {{"key",
           {{"category", "activation"},
            {"name", "silu"},
            {"version", "1.0.0"}}},
          {"configuration", nlohmann::json::object()}}}}},
  };
  composed_source["spec"]["workflow"]["nodes"]["train_to_boundary"]
                 ["invoke"]["training"] = activation_composition;
  composed_source["spec"]["workflow"]["nodes"]["resume_training"]
                 ["invoke"]["training"] = activation_composition;
  const auto composed = trainvm::compile_document(composed_source);
  auto composed_profiles = fixture_adapter_profiles();
  const auto composed_profile = std::ranges::find_if(
      composed_profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (composed_profile != composed_profiles.end()) {
    composed_profile->training_composition =
        trainvm::TrainingCompositionContract{
            .model_family = "mageflow",
            .slots = {{"backbone_activation",
                       trainvm::TrainingComponentCategory::activation}},
            .allowed_components = std::map<
                std::string, std::vector<trainvm::TrainingComponentKey>>{
                {"backbone_activation",
                 {{trainvm::TrainingComponentCategory::activation,
                   "silu", "1.0.0"}}}},
        };
  }
  const auto rejects_composition_plan = [](
      std::vector<trainvm::AdapterProfile> profiles,
      const trainvm::CompiledPlan& plan) {
    try {
      trainvm::AdapterRegistry(std::move(profiles)).validate_plan(plan);
    } catch (const trainvm::AdapterResolutionError&) {
      return true;
    }
    return false;
  };
  bool exact_composition_accepted = composed.valid();
  if (composed.valid()) {
    try {
      trainvm::AdapterRegistry(composed_profiles).validate_plan(*composed.plan);
    } catch (const std::exception&) {
      exact_composition_accepted = false;
    }
  }
  const bool missing_composition_rejected =
      rejects_composition_plan(composed_profiles, *compiled.plan);
  const bool undeclared_composition_rejected =
      composed.valid() &&
      rejects_composition_plan(fixture_adapter_profiles(), *composed.plan);
  bool wrong_family_rejected = false;
  bool wrong_slot_rejected = false;
  bool wrong_category_rejected = false;
  bool disallowed_component_rejected = false;
  if (composed.valid()) {
    auto wrong_family = *composed.plan;
    wrong_family.experiment.spec.workflow.nodes.at("train_to_boundary")
        .invoke.training->model_family = "rwkv";
    wrong_family_rejected =
        rejects_composition_plan(composed_profiles, wrong_family);
    auto wrong_slot = *composed.plan;
    auto& slots = wrong_slot.experiment.spec.workflow.nodes
                      .at("train_to_boundary")
                      .invoke.training->components;
    slots.emplace("invented_slot", slots.begin()->second);
    wrong_slot_rejected =
        rejects_composition_plan(composed_profiles, wrong_slot);
    auto wrong_category = *composed.plan;
    wrong_category.experiment.spec.workflow.nodes.at("train_to_boundary")
        .invoke.training->components.at("backbone_activation")
        .key.category = trainvm::TrainingComponentCategory::optimizer;
    wrong_category_rejected =
        rejects_composition_plan(composed_profiles, wrong_category);
    auto disallowed_component = *composed.plan;
    disallowed_component.experiment.spec.workflow.nodes
        .at("train_to_boundary")
        .invoke.training->components.at("backbone_activation")
        .key.name = "gelu";
    disallowed_component_rejected =
        rejects_composition_plan(composed_profiles, disallowed_component);
  }
  check(exact_composition_accepted && missing_composition_rejected &&
            undeclared_composition_rejected && wrong_family_rejected &&
            wrong_slot_rejected && wrong_category_rejected &&
            disallowed_component_rejected,
        "adapter profiles close the exact training model family, slot names, categories, and component allowlists before worker launch");

  const auto rejects_lifecycle = [](trainvm::OperationLifecycleCapabilities lifecycle) {
    auto profiles = fixture_adapter_profiles();
    const auto train = std::ranges::find_if(
        profiles, [](const trainvm::AdapterProfile& profile) {
          return profile.key.operation == "train";
        });
    if (train != profiles.end()) train->lifecycle = lifecycle;
    try {
      (void)trainvm::AdapterRegistry(std::move(profiles));
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto stateless_checkpoint = train_profile.lifecycle;
  stateless_checkpoint.stateful = false;
  auto exact_without_checkpoint = train_profile.lifecycle;
  exact_without_checkpoint.checkpoint_now = false;
  exact_without_checkpoint.pause_release_resources = false;
  auto release_without_resume = train_profile.lifecycle;
  release_without_resume.resume_grade = trainvm::ResumeGrade::restart_only;
  auto terminal_checkpoint_now = train_profile.lifecycle;
  terminal_checkpoint_now.resume_grade =
      trainvm::ResumeGrade::terminal_checkpoint;
  auto terminal_release_pause = train_profile.lifecycle;
  terminal_release_pause.resume_grade =
      trainvm::ResumeGrade::terminal_checkpoint;
  terminal_release_pause.checkpoint_now = false;
  check(rejects_lifecycle(stateless_checkpoint) &&
            rejects_lifecycle(exact_without_checkpoint) &&
            rejects_lifecycle(release_without_resume) &&
            rejects_lifecycle(terminal_checkpoint_now) &&
            rejects_lifecycle(terminal_release_pause),
        "adapter lifecycle validation rejects stateless checkpoint state, exact resume without checkpoint-now, terminal checkpoint-now, and resource-releasing pause without compatible continuation state");

  const auto rejects_plan_profiles = [&](std::vector<trainvm::AdapterProfile> profiles,
                                         const trainvm::CompiledPlan& plan) {
    try {
      trainvm::AdapterRegistry(std::move(profiles)).validate_plan(plan);
    } catch (const trainvm::AdapterResolutionError&) {
      return true;
    }
    return false;
  };
  auto no_graceful_stop = fixture_adapter_profiles();
  auto no_release_pause = fixture_adapter_profiles();
  for (auto& profile : no_graceful_stop) {
    if (profile.key.operation == "train") {
      profile.lifecycle.graceful_stop = false;
    }
  }
  for (auto& profile : no_release_pause) {
    if (profile.key.operation == "train") {
      profile.lifecycle.pause_release_resources = false;
    }
  }
  auto retained_pause_source = load_fixture();
  retained_pause_source["spec"]["recovery"]
                       ["release_accelerators_when_paused"] = false;
  const auto retained_pause_plan =
      trainvm::compile_document(retained_pause_source);
  auto no_retained_pause = fixture_adapter_profiles();
  for (auto& profile : no_retained_pause) {
    if (profile.key.operation == "train") {
      profile.lifecycle.pause_keep_resources = false;
    }
  }
  auto unspecified_pause_source = load_fixture();
  unspecified_pause_source["spec"]["recovery"].erase(
      "release_accelerators_when_paused");
  const auto unspecified_pause_plan =
      trainvm::compile_document(unspecified_pause_source);
  auto no_pause_protocol = fixture_adapter_profiles();
  for (auto& profile : no_pause_protocol) {
    if (profile.key.operation == "train") {
      profile.lifecycle.pause_keep_resources = false;
      profile.lifecycle.pause_release_resources = false;
    }
  }
  check(rejects_plan_profiles(std::move(no_graceful_stop), *compiled.plan) &&
            rejects_plan_profiles(std::move(no_release_pause), *compiled.plan) &&
            retained_pause_plan.valid() &&
            rejects_plan_profiles(std::move(no_retained_pause),
                                  *retained_pause_plan.plan) &&
            unspecified_pause_plan.valid() &&
            rejects_plan_profiles(std::move(no_pause_protocol),
                                  *unspecified_pause_plan.plan),
        "reachable stateful operations must support graceful stop and the plan's release, retain, or pause-required control policy");

  auto stateful_at_most_once_source = load_fixture();
  stateful_at_most_once_source["spec"]["workflow"]["nodes"]
                               ["train_to_boundary"]["idempotency"] =
      "at_most_once";
  stateful_at_most_once_source["spec"]["workflow"]["nodes"]
                               ["resume_training"]["idempotency"] =
      "at_most_once";
  const auto stateful_at_most_once_plan =
      trainvm::compile_document(stateful_at_most_once_source);
  auto stateful_at_most_once_profiles = fixture_adapter_profiles();
  for (auto& profile : stateful_at_most_once_profiles) {
    if (profile.key.operation == "train") {
      profile.idempotency = trainvm::Idempotency::at_most_once;
    }
  }
  auto stateless_at_most_once_source = load_fixture();
  stateless_at_most_once_source["spec"]["workflow"]["nodes"]
                               ["build_cache"]["idempotency"] =
      "at_most_once";
  const auto stateless_at_most_once_plan =
      trainvm::compile_document(stateless_at_most_once_source);
  auto stateless_at_most_once_profiles = fixture_adapter_profiles();
  for (auto& profile : stateless_at_most_once_profiles) {
    if (profile.key.operation == "cache_encoders") {
      profile.idempotency = trainvm::Idempotency::at_most_once;
    }
  }
  check(stateful_at_most_once_plan.valid() &&
            rejects_plan_profiles(std::move(stateful_at_most_once_profiles),
                                  *stateful_at_most_once_plan.plan) &&
            stateless_at_most_once_plan.valid() &&
            rejects_plan_profiles(std::move(stateless_at_most_once_profiles),
                                  *stateless_at_most_once_plan.plan),
        "exact recovery rejects reachable stateful and stateless at-most-once process operations");

  auto mismatched_execution_source = load_fixture();
  mismatched_execution_source["spec"]["components"]["mageflow"]
                             ["operations"]["unused_phase"] = {
      {"contract", "rwkv_lab.mageflow.v1.UnusedPhase"}};
  mismatched_execution_source["spec"]["execution"]["operation"] =
      "unused_phase";
  const auto mismatched_execution =
      trainvm::compile_document(mismatched_execution_source);
  auto forged_execution_plan = *compiled.plan;
  forged_execution_plan.experiment.spec.components.at("mageflow")
      .operations.emplace(
          "unused_phase",
          trainvm::Operation{.contract =
                                 "rwkv_lab.mageflow.v1.UnusedPhase",
                             .description = std::nullopt});
  forged_execution_plan.experiment.spec.execution->operation = "unused_phase";
  auto forged_execution_profiles = fixture_adapter_profiles();
  auto unused_profile = train_profile;
  unused_profile.key.operation = "unused_phase";
  unused_profile.key.contract = "rwkv_lab.mageflow.v1.UnusedPhase";
  forged_execution_profiles.push_back(std::move(unused_profile));
  check(!mismatched_execution.valid() &&
            rejects_plan_profiles(std::move(forged_execution_profiles),
                                  forged_execution_plan),
        "document and registry validation reject execution phases targeting an operation absent from reachable workflow nodes");

  auto no_profile_support = fixture_adapter_profiles();
  const auto unprofiled_train = std::ranges::find_if(
      no_profile_support, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (unprofiled_train != no_profile_support.end()) {
    unprofiled_train->lifecycle.profile = false;
  }
  bool unsupported_profile_rejected = false;
  try {
    trainvm::AdapterRegistry(std::move(no_profile_support))
        .validate_plan(*compiled.plan);
  } catch (const trainvm::AdapterResolutionError&) {
    unsupported_profile_rejected = true;
  }
  check(unsupported_profile_rejected,
        "typed execution phases are rejected when the exact operation lacks the authority-owned lifecycle capability");

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
  } catch (const std::exception&) {
    semantic_mismatch_rejected = true;
  }
  auto builtin_lifecycle_profiles = fixture_adapter_profiles();
  for (auto& profile : builtin_lifecycle_profiles) {
    if (profile.key.runtime == trainvm::ComponentRuntime::builtin) {
      profile.lifecycle.profile = true;
      break;
    }
  }
  bool builtin_lifecycle_rejected = false;
  try {
    trainvm::AdapterRegistry invalid_builtin(
        std::move(builtin_lifecycle_profiles));
  } catch (const std::invalid_argument&) {
    builtin_lifecycle_rejected = true;
  }
  check(duplicate_rejected && semantic_mismatch_rejected &&
            builtin_lifecycle_rejected,
        "adapter registry rejects duplicate keys, stateful non-process effects, and noncanonical builtin lifecycle authority");

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
    trainvm::Journal left_journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Journal right_journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
    trainvm::Journal restarted_journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
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
  auto lifecycle_drift = fixture_adapter_profiles();
  for (auto& profile : lifecycle_drift) {
    if (profile.key.operation == "train") {
      profile.lifecycle.pause_keep_resources = false;
    }
  }
  drift_rejected_without_mutation(
      std::move(lifecycle_drift),
      "adapter lifecycle authority drift after launch intent fails closed without mutation");

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
        DROP TRIGGER host_resource_requests_no_update;
        DROP TRIGGER host_resource_requests_no_delete;
        DROP TRIGGER host_resource_grants_no_update;
        DROP TRIGGER host_resource_grants_no_delete;
        DROP TRIGGER host_resource_release_intents_no_update;
        DROP TRIGGER host_resource_release_intents_no_delete;
        DROP TRIGGER host_resource_release_receipts_no_update;
        DROP TRIGGER host_resource_release_receipts_no_delete;
        DROP TABLE host_resource_release_receipts;
        DROP TABLE host_resource_release_intents;
        DROP TABLE host_resource_grants;
        DROP TABLE host_resource_requests;
        DROP TRIGGER resource_lease_renewals_no_update;
        DROP TRIGGER resource_lease_renewals_no_delete;
        DROP TABLE resource_lease_renewals;
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
        DROP TRIGGER host_resource_requests_no_update;
        DROP TRIGGER host_resource_requests_no_delete;
        DROP TRIGGER host_resource_grants_no_update;
        DROP TRIGGER host_resource_grants_no_delete;
        DROP TRIGGER host_resource_release_intents_no_update;
        DROP TRIGGER host_resource_release_intents_no_delete;
        DROP TRIGGER host_resource_release_receipts_no_update;
        DROP TRIGGER host_resource_release_receipts_no_delete;
        DROP TABLE host_resource_release_receipts;
        DROP TABLE host_resource_release_intents;
        DROP TABLE host_resource_grants;
        DROP TABLE host_resource_requests;
        DROP TRIGGER resource_lease_renewals_no_update;
        DROP TRIGGER resource_lease_renewals_no_delete;
        DROP TABLE resource_lease_renewals;
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
        DROP TRIGGER host_resource_requests_no_update;
        DROP TRIGGER host_resource_requests_no_delete;
        DROP TRIGGER host_resource_grants_no_update;
        DROP TRIGGER host_resource_grants_no_delete;
        DROP TRIGGER host_resource_release_intents_no_update;
        DROP TRIGGER host_resource_release_intents_no_delete;
        DROP TRIGGER host_resource_release_receipts_no_update;
        DROP TRIGGER host_resource_release_receipts_no_delete;
        DROP TABLE host_resource_release_receipts;
        DROP TABLE host_resource_release_intents;
        DROP TABLE host_resource_grants;
        DROP TABLE host_resource_requests;
        DROP TRIGGER resource_lease_renewals_no_update;
        DROP TRIGGER resource_lease_renewals_no_delete;
        DROP TABLE resource_lease_renewals;
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
      DROP TRIGGER host_resource_requests_no_update;
      DROP TRIGGER host_resource_requests_no_delete;
      DROP TRIGGER host_resource_grants_no_update;
      DROP TRIGGER host_resource_grants_no_delete;
      DROP TRIGGER host_resource_release_intents_no_update;
      DROP TRIGGER host_resource_release_intents_no_delete;
      DROP TRIGGER host_resource_release_receipts_no_update;
      DROP TRIGGER host_resource_release_receipts_no_delete;
      DROP TABLE host_resource_release_receipts;
      DROP TABLE host_resource_release_intents;
      DROP TABLE host_resource_grants;
      DROP TABLE host_resource_requests;
      DROP TRIGGER resource_lease_renewals_no_update;
      DROP TRIGGER resource_lease_renewals_no_delete;
      DROP TABLE resource_lease_renewals;
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

  const auto v6_missing_head_path = directory / "v6-missing-head.db";
  {
    trainvm::Journal fresh(v6_missing_head_path);
    check(trainvm::JournalTestAccess::append(
              fresh, created_event(std::string(64U, 'c'))) == 1U,
          "missing-head v6 fixture creates a valid event chain");
  }
  check(sqlite3_open(v6_missing_head_path.c_str(), &database) == SQLITE_OK,
        "missing-head v6 fixture opens for corruption");
  if (database != nullptr) {
    check(sqlite3_exec(database,
                       "DELETE FROM journal_meta WHERE key='chain_head'", nullptr,
                       nullptr, nullptr) == SQLITE_OK,
          "missing-head v6 fixture removes required authority metadata");
    sqlite3_close(database);
    database = nullptr;
  }
  bool v6_missing_head_refused = false;
  try {
    trainvm::Journal journal(v6_missing_head_path);
  } catch (const std::runtime_error& exception) {
    v6_missing_head_refused =
        std::string(exception.what()).find("authority metadata") !=
        std::string::npos;
  }
  check(v6_missing_head_refused,
        "established v6 journal never synthesizes a missing chain head");
  check(sqlite3_open(v6_missing_head_path.c_str(), &database) == SQLITE_OK,
        "refused missing-head v6 journal remains inspectable");
  if (database != nullptr) {
    check(scalar(database,
                 "SELECT value FROM journal_meta WHERE key='schema_version'") ==
                  "7" &&
              scalar(database,
                     "SELECT COUNT(*) FROM journal_meta WHERE key='chain_head'") ==
                  "0" &&
              scalar(database, "SELECT COUNT(*) FROM events") == "1",
          "missing-head v6 refusal preserves history without metadata repair");
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
      DROP TRIGGER host_resource_requests_no_update;
      DROP TRIGGER host_resource_requests_no_delete;
      DROP TRIGGER host_resource_grants_no_update;
      DROP TRIGGER host_resource_grants_no_delete;
      DROP TRIGGER host_resource_release_intents_no_update;
      DROP TRIGGER host_resource_release_intents_no_delete;
      DROP TRIGGER host_resource_release_receipts_no_update;
      DROP TRIGGER host_resource_release_receipts_no_delete;
      DROP TABLE host_resource_release_receipts;
      DROP TABLE host_resource_release_intents;
      DROP TABLE host_resource_grants;
      DROP TABLE host_resource_requests;
      DROP TRIGGER resource_lease_renewals_no_update;
      DROP TRIGGER resource_lease_renewals_no_delete;
      DROP TABLE resource_lease_renewals;
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
    check(version == "7" && released_history == "legacy-wall/v1|NULL|NULL|NULL|10|20|30" &&
              release_receipt == "legacy-wall/v1|NULL|30",
          "v4 lease and release history migrates into explicit legacy-wall quarantine");
    check(event_after == event_before && head_after != head_before &&
              head_after.size() == 64U,
          "v4 migration preserves event bytes while a later authenticated lease acquisition advances the authority chain");
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
      DROP TRIGGER host_resource_requests_no_update;
      DROP TRIGGER host_resource_requests_no_delete;
      DROP TRIGGER host_resource_grants_no_update;
      DROP TRIGGER host_resource_grants_no_delete;
      DROP TRIGGER host_resource_release_intents_no_update;
      DROP TRIGGER host_resource_release_intents_no_delete;
      DROP TRIGGER host_resource_release_receipts_no_update;
      DROP TRIGGER host_resource_release_receipts_no_delete;
      DROP TABLE host_resource_release_receipts;
      DROP TABLE host_resource_release_intents;
      DROP TABLE host_resource_grants;
      DROP TABLE host_resource_requests;
      DROP TRIGGER resource_lease_renewals_no_update;
      DROP TRIGGER resource_lease_renewals_no_delete;
      DROP TABLE resource_lease_renewals;
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

class SagaHostClient final : public trainvm::IHostGrantClient {
 public:
  explicit SagaHostClient(trainvm::SQLiteHostLedger& ledger) : ledger_(ledger) {}

  trainvm::BundleRequestResult request_bundle(
      const trainvm::ResourceBundleRequest& request) override {
    ++request_calls;
    auto result = ledger_.request_bundle(request, {100, 1'000});
    if (result.replayed) ++request_replays;
    return result;
  }

  trainvm::BundleReleaseResult release_bundle(
      const trainvm::ResourceReleaseRequest& request) override {
    ++release_calls;
    auto result = ledger_.release_bundle(request, {200, 2'000});
    if (result.replayed) ++release_replays;
    return result;
  }

  std::size_t request_calls{};
  std::size_t request_replays{};
  std::size_t release_calls{};
  std::size_t release_replays{};

 private:
  trainvm::SQLiteHostLedger& ledger_;
};

class SagaOneShotFault final : public trainvm::IHostGrantSagaFaultInjector {
 public:
  explicit SagaOneShotFault(trainvm::HostGrantSagaFaultPoint point)
      : point_(point) {}

  void hit(trainvm::HostGrantSagaFaultPoint point) override {
    if (armed_ && point == point_) {
      armed_ = false;
      throw std::runtime_error("injected host saga boundary fault");
    }
  }

 private:
  trainvm::HostGrantSagaFaultPoint point_;
  bool armed_{true};
};

class SagaStaticResultClient final : public trainvm::IHostGrantClient {
 public:
  explicit SagaStaticResultClient(trainvm::BundleRequestResult result)
      : result_(std::move(result)) {}

  trainvm::BundleRequestResult request_bundle(
      const trainvm::ResourceBundleRequest&) override {
    ++request_calls;
    return result_;
  }

  trainvm::BundleReleaseResult release_bundle(
      const trainvm::ResourceReleaseRequest&) override {
    throw std::runtime_error("static request client cannot release");
  }

  std::size_t request_calls{};

 private:
  trainvm::BundleRequestResult result_;
};

class SagaProcessClient final : public trainvm::IHostProcessClient {
 public:
  explicit SagaProcessClient(trainvm::SQLiteHostLedger& ledger)
      : ledger_(ledger) {}

  trainvm::HostdProcessPreparedResult prepare_process(
      const trainvm::HostdProcessPrepareRequest& request,
      const trainvm::ResolvedLaunch&,
      const trainvm::SealedWorkerBootstrap& bootstrap) override {
    ++prepare_calls;
    const int bootstrap_fd = bootstrap.duplicate_fd();
    trainvm::WorkerBootstrapSpec inspected;
    try {
      inspected = trainvm::worker_bootstrap_from_sealed_fd(
          bootstrap_fd, request.worker_bootstrap_digest);
    } catch (...) {
      (void)::close(bootstrap_fd);
      throw;
    }
    (void)::close(bootstrap_fd);
    if (inspected.run_id != request.launch.identity.run_id ||
        inspected.node_id != request.launch.identity.node_id ||
        inspected.controller_target != "unix:/tmp/trainvm-process-saga.sock" ||
        inspected.capabilities !=
            request.launch.identity.provided_capabilities) {
      throw std::runtime_error("process client received the wrong bootstrap");
    }
    const auto process_identity =
        process_identities_
            .try_emplace(
                request.launch.identity.launch_event_id,
                std::pair<std::int64_t, std::uint64_t>{
                    12'345 + static_cast<std::int64_t>(
                                 process_identities_.size()),
                    67'890U + process_identities_.size()})
            .first;
    const auto launch = trainvm::seal_host_process_launch_request({
        .api_version =
            std::string(trainvm::kHostProcessLaunchRequestApiVersion),
        .launch_id = request.launch.identity.launch_event_id,
        .allocation_id = request.grant.allocation_id,
        .grant_digest = request.grant.receipt_digest,
        .journal_id = request.grant.journal_id,
        .run_id = request.grant.run_id,
        .logical_lease_id = request.grant.logical_lease_id,
        .logical_fencing_token = request.grant.logical_fencing_token,
        .resolved_launch_digest = trainvm::hostd_bound_process_launch_digest(
            request.launch, request.worker_bootstrap_digest,
            request.process_policy),
        .executable_path = request.launch.identity.executable.source_path,
        .executable_digest =
            request.launch.identity.executable.sealed_sha256,
        .cgroup_path = "/trainvm/process-saga",
        .cgroup_device = 41U,
        .cgroup_inode = 42U,
        .worker_credentials = std::nullopt,
        .device_policy = std::nullopt,
        .process_policy = std::nullopt,
        .canonical_request_digest = {},
    });
    const auto intended = ledger_.commit_process_launch_intent(
        launch, {300, 3'000});
    const auto spawn_request = trainvm::seal_host_process_spawn_request({
        .api_version =
            std::string(trainvm::kHostProcessSpawnRequestApiVersion),
        .launch_id = launch.launch_id,
        .launch_intent_digest = intended.intent.receipt_digest,
        .host_pid = process_identity->second.first,
        .process_starttime_ticks = process_identity->second.second,
        .boot_id = request.grant.boot_id,
        .cgroup_path = launch.cgroup_path,
        .cgroup_device = launch.cgroup_device,
        .cgroup_inode = launch.cgroup_inode,
        .executable_digest = launch.executable_digest,
        .worker_credentials = std::nullopt,
        .device_policy = std::nullopt,
        .process_policy = std::nullopt,
        .canonical_request_digest = {},
    });
    const auto spawned = ledger_.commit_process_spawn(
        spawn_request, {301, 3'001});
    if (intended.replayed || spawned.replayed) ++prepare_replays;
    spawn_ = spawned.receipt;
    return {.api_version =
                std::string(trainvm::kHostdProcessPreparedApiVersion),
            .intent = intended.intent,
            .spawn = spawned.receipt,
            .replayed = intended.replayed || spawned.replayed};
  }

  trainvm::HostdProcessCommittedResult commit_process(
      const trainvm::HostdProcessCommitRequest& request) override {
    ++commit_calls;
    const bool replayed = commit_calls > 1U;
    if (replayed) ++commit_replays;
    return {.api_version =
                std::string(trainvm::kHostdProcessCommittedApiVersion),
            .launch_id = request.launch_id,
            .spawn_receipt_digest = request.spawn_receipt_digest,
            .released_to_exec = true,
            .replayed = replayed};
  }

  trainvm::HostProcessExitResult finalize_process(
      const trainvm::HostdProcessExitCommand& command) override {
    if (!spawn_) throw std::runtime_error("process was never prepared");
    ++exit_calls;
    const auto& request = spawn_->request;
    if (command.launch_id != request.launch_id ||
        command.spawn_receipt_digest != spawn_->receipt_digest) {
      throw std::runtime_error("process exit command changed its spawn identity");
    }
    auto result = ledger_.commit_process_exit(
        trainvm::seal_host_process_exit_request({
            .api_version =
                std::string(trainvm::kHostProcessExitRequestApiVersion),
            .exit_request_id = command.exit_request_id,
            .launch_id = request.launch_id,
            .spawn_receipt_digest = spawn_->receipt_digest,
            .host_pid = request.host_pid,
            .process_starttime_ticks = request.process_starttime_ticks,
            .wait_code = 1,
            .wait_status = 0,
            .cgroup_path = request.cgroup_path,
            .cgroup_device = request.cgroup_device,
            .cgroup_inode = request.cgroup_inode,
            .cgroup_empty = true,
            .accelerator_contexts_empty = true,
            .context_audit_digest =
                "sha256:" + std::string(64U, 'c'),
            .canonical_request_digest = {},
        }),
        {400, 4'000});
    if (result.replayed) ++exit_replays;
    return result;
  }

  std::size_t prepare_calls{};
  std::size_t prepare_replays{};
  std::size_t commit_calls{};
  std::size_t commit_replays{};
  std::size_t exit_calls{};
  std::size_t exit_replays{};

 private:
  trainvm::SQLiteHostLedger& ledger_;
  std::map<std::string, std::pair<std::int64_t, std::uint64_t>, std::less<>>
      process_identities_;
  std::optional<trainvm::HostProcessSpawnReceipt> spawn_;
};

class ProcessSagaOneShotFault final
    : public trainvm::IHostProcessSagaFaultInjector {
 public:
  explicit ProcessSagaOneShotFault(trainvm::HostProcessSagaFaultPoint point)
      : point_(point) {}

  void hit(trainvm::HostProcessSagaFaultPoint point) override {
    if (armed_ && point == point_) {
      armed_ = false;
      throw std::runtime_error("injected host process saga fault");
    }
  }

 private:
  trainvm::HostProcessSagaFaultPoint point_;
  bool armed_{true};
};

trainvm::ObservedHostResource saga_mutex_resource() {
  return {
      .id = {.kind = trainvm::HostResourceKind::host_mutex,
             .vendor = std::nullopt,
             .stable_id = "host-mutex:saga",
             .parent_id = std::nullopt},
      .disposition =
          trainvm::ResourceObservationDisposition::audited_eligible,
      .compute_contexts = trainvm::ResourceContextDisposition::absent,
      .graphics_contexts = trainvm::ResourceContextDisposition::absent,
      .pci_bdf = std::nullopt,
      .device_major = std::nullopt,
      .device_minor = std::nullopt,
      .device_nodes = {},
      .numa_node = std::nullopt,
      .pcie_root_id = std::nullopt,
      .fabric_clique_id = std::nullopt,
      .total_memory_bytes = 0,
      .labels = {{"scope", "saga-test"}},
  };
}

void test_host_grant_saga() {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-host-saga-test-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  check(::chmod(directory.c_str(), 0700) == 0,
        "host saga fixture protects its directory");

  trainvm::HostKernelSnapshot kernel_snapshot{
      .api_version = std::string(trainvm::kHostInventoryApiVersion),
      .host_id = "sha256:" + std::string(64U, 'd'),
      .boot_id = kTestBootId,
      .broker_epoch = "broker-saga",
      .begin_revision = "revision-saga",
      .end_revision = "revision-saga",
      .probes = {},
      .resources = {saga_mutex_resource()},
  };
  trainvm::FakeHostKernel kernel(
      {{.snapshot = std::move(kernel_snapshot), .failure = std::nullopt}});
  const auto inventory = trainvm::capture_host_inventory(kernel);
  auto authority =
      std::make_shared<trainvm::HostLedgerFilesystemAuthority>(
          trainvm::HostLedgerFilesystemAuthority::acquire({
              .api_version =
                  std::string(trainvm::kHostLedgerAuthorityApiVersion),
              .ledger_path = directory / "host-resource.db",
              .expected_owner_uid = ::geteuid(),
              .expected_owner_gid = ::getegid(),
              .enforcement_grade =
                  trainvm::HostLedgerEnforcementGrade::cooperative_test,
          }));
  trainvm::SQLiteHostLedger host_ledger(authority, inventory);
  SagaHostClient host(host_ledger);
  trainvm::AuthorityLock journal_authority(directory / "journal.db");
  trainvm::Journal journal(
      journal_authority.journal_path(), journal_authority.journal_identity(),
      trainvm::HostGrantEnforcement::required,
      trainvm::HostIdentity{.host_id = inventory.host_id,
                            .boot_id = inventory.boot_id});
  const auto compiled = trainvm::compile_document(load_fixture());
  check(compiled.valid(), "host saga fixture compiles its external worker plan");
  if (!compiled.plan) {
    std::filesystem::remove_all(directory);
    return;
  }
  {
    trainvm::Journal unanchored(directory / "unanchored-journal.db");
    const auto unanchored_request = trainvm::seal_resource_request({
        .api_version = std::string(trainvm::kHostResourceRequestApiVersion),
        .request_id = "unanchored-request",
        .journal_id = unanchored.journal_id(),
        .run_id = "unanchored-run",
        .logical_lease_id = "unanchored-lease",
        .logical_fencing_token = 1,
        .count = 1,
        .access_mode = trainvm::ResourceAccessMode::mutex_exclusive,
        .topology = trainvm::TopologyPolicy::any,
        .selector = {},
        .canonical_request_digest = {},
    });
    const std::size_t host_calls_before_unanchored = host.request_calls;
    const std::uint64_t events_before_unanchored = unanchored.event_count();
    bool unanchored_rejected = false;
    try {
      trainvm::HostGrantSagaReconciler unsafe(unanchored, host);
      (void)unsafe.reconcile_request(unanchored_request, test_time(1));
    } catch (const trainvm::OperationPreconditionError&) {
      unanchored_rejected = true;
    }
    check(unanchored_rejected &&
              host.request_calls == host_calls_before_unanchored &&
              unanchored.event_count() == events_before_unanchored &&
              !unanchored.host_grant_saga(unanchored_request.request_id),
          "host saga refuses an unanchored journal before host contact or journal mutation");
  }
  const std::string saga_key =
      compiled.plan->experiment.spec.workspace.concurrency_key;
  trainvm::Controller controller(*compiled.plan, journal, "run-1");
  controller.create_queued();
  const auto lease = controller.begin_acquisition(test_time(10));
  check(lease.status == trainvm::LeaseAcquireStatus::acquired,
        "host saga fixture has a live logical lease");
  const auto request = trainvm::seal_resource_request({
      .api_version = std::string(trainvm::kHostResourceRequestApiVersion),
      .request_id = "saga-request",
      .journal_id = journal.journal_id(),
      .run_id = "run-1",
      .logical_lease_id = lease.lease.lease_id,
      .logical_fencing_token = lease.lease.fencing_token,
      .count = 1,
      .access_mode = trainvm::ResourceAccessMode::mutex_exclusive,
      .topology = trainvm::TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
  const auto recover_matches = [&](const trainvm::ExecutionState& expected,
                                   std::string_view boundary) {
    trainvm::Controller restarted(*compiled.plan, journal, "run-1");
    check(restarted.recover() == expected, boundary);
  };
  const trainvm::ExecutionState acquiring_state = controller.state();

  bool zero_request_launch_rejected = false;
  try {
    (void)journal.host_launch_grant_claim(
        request.run_id, saga_key, request.logical_lease_id,
        request.logical_fencing_token, test_time(11));
  } catch (const trainvm::OperationPreconditionError&) {
    zero_request_launch_rejected = true;
  }
  check(zero_request_launch_rejected,
        "default-strict launch authority rejects a live logical lease with no host request or grant");

  SagaOneShotFault grant_gap(
      trainvm::HostGrantSagaFaultPoint::host_before_journal);
  trainvm::HostGrantSagaReconciler interrupted(journal, host, &grant_gap);
  bool grant_interrupted = false;
  try {
    (void)interrupted.reconcile_request(request, test_time(20));
  } catch (const std::runtime_error&) {
    grant_interrupted = true;
  }
  const auto pending = journal.host_grant_saga(request.request_id);
  check(grant_interrupted && pending && !pending->grant,
        "host-before-journal fault leaves only the chained request intent");

  const auto make_claim_provider = [&](std::string controller_id) {
    return std::make_unique<trainvm::JournalHostdMutationClaimProvider>(
        journal,
        trainvm::HostdMutationClaimProviderConfig{
            .api_version = std::string(
                trainvm::kHostdMutationClaimProviderApiVersion),
            .broker_epoch = "broker-claim-provider",
            .authority_clock = [boot_id = inventory.boot_id] {
              return test_time_on_boot(20, boot_id);
            },
            .controller_id_source =
                [value = std::move(controller_id)] { return value; }});
  };
  auto first_claim_provider = make_claim_provider("claim-controller-001");
  const std::uint64_t before_missing_claim = journal.event_count();
  bool missing_claim_rejected = false;
  try {
    (void)first_claim_provider->open_for_resource("request-missing");
  } catch (const trainvm::HostdMutationClaimProviderError&) {
    missing_claim_rejected = true;
  }
  const auto first_open =
      first_claim_provider->open_for_resource(request.request_id);
  const std::uint64_t after_first_claim = journal.event_count();
  const auto replayed_open =
      first_claim_provider->open_for_resource(request.request_id);
  check(missing_claim_rejected &&
            before_missing_claim == after_first_claim &&
            journal.event_count() == after_first_claim,
        "mutation claim provisioning changes only the reserved authority event stream");
  check(first_open == replayed_open,
        "same-process mutation claim replay is byte-exact");
  check(first_open.claim.journal.journal_id == request.journal_id &&
            first_open.claim.controller.run_id == request.run_id &&
            first_open.claim.controller.concurrency_key == saga_key &&
            first_open.claim.controller.logical_lease_id ==
                request.logical_lease_id &&
            first_open.claim.controller.logical_fencing_token ==
                request.logical_fencing_token &&
            first_open.claim.controller.controller_generation == 1U &&
            first_open.claim.controller.controller_id ==
                "claim-controller-001",
        "mutation claims derive exact scope from the durable saga");
  check(trainvm::hostd_mutation_open_from_canonical_json(
            trainvm::hostd_mutation_open_canonical_json(first_open)) ==
            first_open,
        "mutation claims satisfy the canonical wire contract");
  auto restarted_claim_provider =
      make_claim_provider("claim-controller-002");
  const auto restarted_open =
      restarted_claim_provider->open_for_resource(request.request_id);
  bool superseded_claim_rejected = false;
  try {
    (void)first_claim_provider->open_for_resource(request.request_id);
  } catch (const trainvm::OperationPreconditionError&) {
    superseded_claim_rejected = true;
  }
  const auto current_controller =
      journal.current_hostd_controller_fence(saga_key);
  check(restarted_open.claim.controller.controller_generation == 2U &&
            restarted_open.claim.controller.controller_id ==
                "claim-controller-002" &&
            superseded_claim_rejected && current_controller &&
            current_controller->controller_generation == 2U &&
            current_controller->controller_id == "claim-controller-002",
        "service restart advances durable authority and fences the old provider");
  recover_matches(acquiring_state,
                  "controller restart ignores durable host request intent without changing FSM state");
  bool premature_launch = false;
  try {
    journal.require_host_launch_eligible(
        {.request_id = request.request_id,
         .grant_digest = std::string(71U, '0'),
         .fences = {}},
        test_time(21));
  } catch (const trainvm::OperationPreconditionError&) {
    premature_launch = true;
  }
  check(premature_launch, "launch is blocked until both authorities are durable");

  const auto host_replay = host_ledger.request_bundle(request, {101, 1'001});
  SagaStaticResultClient unknown_status({
      .status = static_cast<trainvm::BundleRequestStatus>(99),
      .grant = std::nullopt,
      .outcome_digest = std::string(71U, '0'),
      .replayed = false,
  });
  SagaStaticResultClient mismatched_outcome({
      .status = trainvm::BundleRequestStatus::granted,
      .grant = host_replay.grant,
      .outcome_digest = "sha256:" + std::string(64U, '0'),
      .replayed = false,
  });
  bool unknown_status_rejected = false;
  bool outcome_mismatch_rejected = false;
  try {
    trainvm::HostGrantSagaReconciler invalid(journal, unknown_status);
    (void)invalid.reconcile_request(request, test_time(21));
  } catch (const std::runtime_error&) {
    unknown_status_rejected = true;
  }
  try {
    trainvm::HostGrantSagaReconciler invalid(journal, mismatched_outcome);
    (void)invalid.reconcile_request(request, test_time(21));
  } catch (const std::runtime_error&) {
    outcome_mismatch_rejected = true;
  }
  check(host_replay.grant && unknown_status_rejected &&
            outcome_mismatch_rejected && unknown_status.request_calls == 1U &&
            mismatched_outcome.request_calls == 1U &&
            !journal.host_grant_saga(request.request_id)->grant,
        "reconciler rejects unknown host status and outcome/grant digest disagreement without journal mutation");

  trainvm::HostGrantSagaReconciler reconciler(journal, host);
  const auto granted = reconciler.reconcile_request(request, test_time(22));
  check(granted.grant && host.request_calls == 2U &&
            host.request_replays == 1U && granted.grant->fences.size() == 1U &&
            granted.grant->fences.front().generation == 1U,
        "request retry converges on one host-issued physical generation");
  recover_matches(acquiring_state,
                  "controller restart ignores durable host grant receipt without changing FSM state");
  const auto grant_copy_rejected = [&](trainvm::ResourceBundleGrant candidate) {
    try {
      (void)journal.record_host_grant_receipt(candidate);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  auto wrong_host = *granted.grant;
  wrong_host.host_id = "different-host";
  auto wrong_boot = *granted.grant;
  wrong_boot.boot_id = "different-boot";
  auto empty_fences = *granted.grant;
  empty_fences.fences.clear();
  auto too_many_fences = *granted.grant;
  too_many_fences.fences.resize(
      trainvm::HostResourceBounds::maximum_bundle_count + 1U,
      granted.grant->fences.front());
  auto oversized_id = *granted.grant;
  oversized_id.fences.front().resource.stable_id =
      std::string(trainvm::HostResourceBounds::maximum_identifier_bytes + 1U,
                  'x');
  auto noncanonical_kind_vendor = *granted.grant;
  noncanonical_kind_vendor.fences.front().resource.vendor =
      trainvm::HostAcceleratorVendor::nvidia;
  check(grant_copy_rejected(std::move(wrong_host)) &&
            grant_copy_rejected(std::move(wrong_boot)) &&
            grant_copy_rejected(std::move(empty_fences)) &&
            grant_copy_rejected(std::move(too_many_fences)) &&
            grant_copy_rejected(std::move(oversized_id)) &&
            grant_copy_rejected(std::move(noncanonical_kind_vendor)),
        "grant ingress rejects wrong host epochs and empty, oversized, or noncanonical nested fence identities before hashing");
  journal.require_host_launch_eligible(
      {.request_id = request.request_id,
       .grant_digest = granted.grant->receipt_digest,
       .fences = granted.grant->fences},
      test_time(23));
  const auto selected_claim = journal.host_launch_grant_claim(
      request.run_id, saga_key, request.logical_lease_id,
      request.logical_fencing_token, test_time(23));
  check(selected_claim &&
            selected_claim->request_id == request.request_id &&
            selected_claim->grant_digest == granted.grant->receipt_digest &&
            selected_claim->fences == granted.grant->fences,
        "strict launch selection binds the exact durable grant digest and physical fences");
  const auto launch = controller.prepare_worker_launch(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"worker.controls", "worker.metrics"}},
      test_time(23));
  const auto binding = bind_test_worker_launch(
      controller, launch, 23,
      trainvm::HostIdentity{.host_id = inventory.host_id,
                            .boot_id = inventory.boot_id});
  const auto durable_launch = journal.event(
      launch.run_id + ":worker-launch:" + launch.node_id + ":" +
      launch.attempt_id);
  check(launch.host_grant == selected_claim &&
            binding.identity.host_grant == selected_claim && durable_launch &&
            durable_launch->payload.contains("host_grant") &&
            journal.launch_binding(durable_launch->event_id)
                    ->identity.host_grant == selected_claim,
        "worker ticket, launch intent, and resolved binding seal the same exact host grant claim");

  trainvm::ResolvedLaunch process_launch(
      binding, -1,
      binding.identity.code ? std::optional<int>{-1} : std::nullopt, -1);
  const auto process_policy =
      trainvm::compile_linux_process_policy(std::nullopt);
  SagaProcessClient process_host(host_ledger);
  ProcessSagaOneShotFault lost_prepare_reply(
      trainvm::HostProcessSagaFaultPoint::after_prepare_host);
  bool prepare_reply_lost = false;
  try {
    trainvm::HostProcessSagaReconciler process_reconciler(
        journal, process_host, &lost_prepare_reply);
    (void)process_reconciler.reconcile(
        process_launch, *granted.grant, process_policy,
        "unix:/tmp/trainvm-process-saga.sock", test_time(23));
  } catch (const std::runtime_error&) {
    prepare_reply_lost = true;
  }
  check(prepare_reply_lost &&
            !journal.host_process_saga(binding.identity.launch_event_id),
        "lost prepare reply leaves no fabricated journal process receipt");

  ProcessSagaOneShotFault stopped_after_journal(
      trainvm::HostProcessSagaFaultPoint::after_prepare_journal);
  bool stopped_before_commit = false;
  try {
    trainvm::HostProcessSagaReconciler process_reconciler(
        journal, process_host, &stopped_after_journal);
    (void)process_reconciler.reconcile(
        process_launch, *granted.grant, process_policy,
        "unix:/tmp/trainvm-process-saga.sock", test_time(23));
  } catch (const std::runtime_error&) {
    stopped_before_commit = true;
  }
  const auto durably_stopped =
      journal.host_process_saga(binding.identity.launch_event_id);
  check(stopped_before_commit && durably_stopped &&
            !durably_stopped->committed && process_host.prepare_calls == 2U &&
            process_host.prepare_replays == 1U,
        "prepare retry converges on one stopped child and journals it before exec");

  ProcessSagaOneShotFault lost_commit_reply(
      trainvm::HostProcessSagaFaultPoint::after_commit_host);
  bool commit_reply_lost = false;
  try {
    trainvm::HostProcessSagaReconciler process_reconciler(
        journal, process_host, &lost_commit_reply);
    (void)process_reconciler.reconcile(
        process_launch, *granted.grant, process_policy,
        "unix:/tmp/trainvm-process-saga.sock", test_time(23));
  } catch (const std::runtime_error&) {
    commit_reply_lost = true;
  }
  check(commit_reply_lost &&
            !journal.host_process_saga(binding.identity.launch_event_id)
                 ->committed,
        "lost exec-commit reply preserves the durable stopped-child receipt");

  trainvm::HostProcessSagaReconciler process_reconciler(journal,
                                                         process_host);
  const auto process_complete = process_reconciler.reconcile(
      process_launch, *granted.grant, process_policy,
      "unix:/tmp/trainvm-process-saga.sock", test_time(23));
  const std::uint64_t process_event_count = journal.event_count();
  const auto process_replay = process_reconciler.reconcile(
      process_launch, *granted.grant, process_policy,
      "unix:/tmp/trainvm-process-saga.sock", test_time(23));
  check(process_complete.committed && process_replay == process_complete &&
            !process_complete.prepared.replayed &&
            !process_complete.committed->replayed &&
            process_host.commit_calls == 2U &&
            process_host.commit_replays == 1U &&
            journal.event_count() == process_event_count,
        "lost commit reply replays exactly and delivery flags never alter durable process identity");

  bool process_namespace_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append(
        journal,
        {.event_id = "forged-host-process-event",
         .run_id = launch.run_id,
         .run_revision = controller.state().revision,
         .plan_revision = 1,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .worker_sequence = 0,
         .event_type = "host.process_prepared",
         .event_version = 1,
         .wall_time_ns = 23,
         .monotonic_time_ns = 23,
         .optimizer_step = std::nullopt,
         .payload = {}});
  } catch (const std::invalid_argument&) {
    process_namespace_rejected = true;
  }
  check(process_namespace_rejected,
        "untyped callers cannot forge host process authority events");

  const trainvm::WorkerHelloEvidence hello{
      .run_id = launch.run_id,
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
      .fencing_token = launch.fencing_token,
  };
  check(controller.accept_worker_hello(hello, test_time(23)).disposition ==
            trainvm::WorkerReadinessDisposition::accepted,
        "worker readiness accepts while its exact physical grant remains live");
  const auto dispatch = controller.prepare_dispatch(test_time(23));
  const trainvm::ExecutionState running_state = controller.state();
  const auto pending_control = controller.request_controls(
      "saga-control-before-release", running_state.revision, 0,
      {{"learning_rate", 0.00001}}, "operator",
      "verify physical grant revokes control acknowledgement");
  check(pending_control.valid() && pending_control.command,
        "host saga fixture records a pending control for its live worker");
  if (!pending_control.command) {
    std::filesystem::remove_all(directory);
    return;
  }
  auto malicious_document = load_fixture();
  malicious_document["spec"]["workflow"]["nodes"].begin()
      .value()["transitions"][0]["on"] =
      "host.resource_grant_recorded";
  const auto malicious_compile = trainvm::compile_document(malicious_document);
  const std::uint64_t before_namespace_attack = journal.event_count();
  bool worker_namespace_rejected = false;
  bool journal_namespace_rejected = false;
  try {
    (void)trainvm::JournalTestAccess::append(
        journal,
        {.event_id = "forged-host-saga-event",
         .run_id = launch.run_id,
         .run_revision = running_state.revision,
         .plan_revision = 1,
         .node_id = {},
         .attempt_id = {},
         .worker_sequence = 0,
         .event_type = "host.resource_grant_recorded",
         .event_version = 1,
         .wall_time_ns = 23,
         .monotonic_time_ns = 23,
         .optimizer_step = std::nullopt,
         .payload = {}});
  } catch (const std::invalid_argument&) {
    journal_namespace_rejected = true;
  }
  try {
    (void)controller.handle_event(
        {.event_id = dispatch.dispatch_id + ":host-namespace-attack",
         .run_id = dispatch.run_id,
         .run_revision = dispatch.run_revision,
         .plan_revision = dispatch.plan_revision,
         .node_id = dispatch.node_id,
         .attempt_id = dispatch.attempt_id,
         .worker_sequence = 1,
         .event_type = "host.resource_grant_recorded",
         .event_version = 1,
         .wall_time_ns = 23,
         .monotonic_time_ns = 23,
         .optimizer_step = std::nullopt,
         .payload = {}},
        {.run_id = launch.run_id,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .launch_nonce = launch.launch_nonce,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(23));
  } catch (const std::invalid_argument&) {
    worker_namespace_rejected = true;
  }
  check(!malicious_compile.valid() && worker_namespace_rejected &&
            journal_namespace_rejected &&
            journal.event_count() == before_namespace_attack &&
            journal.verify_chain(),
        "compiler and worker ingress reserve host.resource_* without journal mutation");
  (void)journal.rebuild_projections();
  check(journal.verify_chain(),
        "reserved namespace attack leaves projection rebuild healthy");
  bool mismatch_rejected = false;
  try {
    journal.require_host_launch_eligible(
        {.request_id = request.request_id,
         .grant_digest = granted.grant->receipt_digest,
         .fences = {}},
        test_time(23));
  } catch (const trainvm::OperationPreconditionError&) {
    mismatch_rejected = true;
  }
  check(mismatch_rejected, "launch rejects a mismatched physical fence set");

  const auto busy_request = trainvm::seal_resource_request({
      .api_version = std::string(trainvm::kHostResourceRequestApiVersion),
      .request_id = "saga-busy-request",
      .journal_id = request.journal_id,
      .run_id = request.run_id,
      .logical_lease_id = request.logical_lease_id,
      .logical_fencing_token = request.logical_fencing_token,
      .count = 1,
      .access_mode = trainvm::ResourceAccessMode::mutex_exclusive,
      .topology = trainvm::TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
  const std::size_t calls_before_busy = host.request_calls;
  SagaOneShotFault before_host_fault(
      trainvm::HostGrantSagaFaultPoint::journal_before_host);
  trainvm::HostGrantSagaReconciler before_host(journal, host,
                                               &before_host_fault);
  bool before_host_interrupted = false;
  try {
    (void)before_host.reconcile_request(busy_request, test_time(24));
  } catch (const std::runtime_error&) {
    before_host_interrupted = true;
  }
  check(before_host_interrupted && host.request_calls == calls_before_busy &&
            journal.host_grant_saga(busy_request.request_id) &&
            !journal.host_grant_saga(busy_request.request_id)
                 ->busy_outcome_digest,
        "journal-before-host fault durably records intent without contacting host authority");
  recover_matches(running_state,
                  "controller restart ignores a second host request intent while running");

  SagaOneShotFault busy_copy_fault(
      trainvm::HostGrantSagaFaultPoint::host_before_journal);
  trainvm::HostGrantSagaReconciler busy_copy_interrupted(
      journal, host, &busy_copy_fault);
  bool busy_copy_gap = false;
  try {
    (void)busy_copy_interrupted.reconcile_request(busy_request,
                                                  test_time(25));
  } catch (const std::runtime_error&) {
    busy_copy_gap = true;
  }
  check(busy_copy_gap && host.request_calls == calls_before_busy + 1U &&
            !journal.host_grant_saga(busy_request.request_id)
                 ->busy_outcome_digest,
        "host-before-journal fault leaves a terminal busy outcome replayable but not falsely durable");

  const auto durable_busy = reconciler.reconcile_request(busy_request,
                                                          test_time(26));
  check(durable_busy.busy_outcome_digest && !durable_busy.grant &&
            host.request_calls == calls_before_busy + 2U &&
            host.request_replays == 2U,
        "busy retry copies the exact terminal host outcome digest into the journal chain");
  recover_matches(running_state,
                  "controller restart ignores durable host busy evidence while running");
  const std::size_t calls_after_busy = host.request_calls;
  SagaOneShotFault busy_replay_fault(
      trainvm::HostGrantSagaFaultPoint::replay_boundary);
  trainvm::HostGrantSagaReconciler busy_replay_interrupted(
      journal, host, &busy_replay_fault);
  bool busy_replay_gap = false;
  try {
    (void)busy_replay_interrupted.reconcile_request(busy_request,
                                                    test_time(27));
  } catch (const std::runtime_error&) {
    busy_replay_gap = true;
  }
  const auto busy_replayed = reconciler.reconcile_request(busy_request,
                                                           test_time(28));
  check(busy_replay_gap && busy_replayed == durable_busy &&
            host.request_calls == calls_after_busy,
        "replay-boundary retry returns the durable busy decision without another host call");

  sqlite3* raw = nullptr;
  check(sqlite3_open((directory / "journal.db").c_str(), &raw) == SQLITE_OK,
        "host saga projection tamper connection opens");
  if (raw != nullptr) {
    check(sqlite3_exec(raw, R"sql(
      DROP TRIGGER host_resource_grants_no_update;
      DROP TRIGGER host_resource_grants_no_delete;
      DELETE FROM host_resource_grants;
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "host saga projection can be removed only after raw trigger tamper");
    sqlite3_close(raw);
  }
  std::string reason;
  check(!journal.verify_chain(&reason),
        "event chain detects a missing host grant projection");
  (void)journal.rebuild_projections();
  check(journal.verify_chain() &&
            journal.host_grant_saga(request.request_id)->grant == granted.grant,
        "projection rebuild restores exact host grant state from chained events");

  check(sqlite3_open((directory / "journal.db").c_str(), &raw) == SQLITE_OK,
        "host saga orphan projection connection opens");
  if (raw != nullptr) {
    check(sqlite3_exec(raw, R"sql(
      PRAGMA foreign_keys=OFF;
      INSERT INTO host_resource_grants(
        request_id, allocation_id, request_digest, grant_digest,
        canonical_grant_json
      ) VALUES(
        'orphan-request', 'orphan-allocation', 'orphan-request-digest',
        'orphan-grant-digest', '{}'
      );
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "raw connection injects an unreachable grant child with FK checks disabled");
    sqlite3_close(raw);
  }
  check(!journal.verify_chain(&reason),
        "host saga verification rejects unreachable child projection rows");
  (void)journal.rebuild_projections();
  check(journal.verify_chain(),
        "projection rebuild removes unreachable host saga child rows");

  ProcessSagaOneShotFault lost_exit_reply(
      trainvm::HostProcessSagaFaultPoint::after_exit_host);
  bool exit_reply_lost = false;
  try {
    trainvm::HostProcessSagaReconciler interrupted_exit(
        journal, process_host, &lost_exit_reply);
    (void)interrupted_exit.reconcile_exit(
        binding.identity.launch_event_id, true, test_time(29));
  } catch (const std::runtime_error&) {
    exit_reply_lost = true;
  }
  check(exit_reply_lost &&
            !journal.host_process_saga(binding.identity.launch_event_id)
                 ->exited,
        "lost process-exit reply does not fabricate terminal journal evidence");
  const auto process_exited = process_reconciler.reconcile_exit(
      binding.identity.launch_event_id, true, test_time(29));
  const auto process_exit_replay = process_reconciler.reconcile_exit(
      binding.identity.launch_event_id, true, test_time(29));
  check(process_exited.exited && process_exit_replay == process_exited &&
            !process_exited.exited->replayed && process_host.exit_calls == 2U &&
            process_host.exit_replays == 1U,
        "process exit converges on one host receipt before grant release");
  recover_matches(running_state,
                  "controller restart accepts ordered terminal process evidence");

  const auto release = trainvm::seal_resource_release_request({
      .api_version = std::string(trainvm::kHostLedgerReleaseRequestApiVersion),
      .release_request_id = "saga-release",
      .allocation_id = granted.grant->allocation_id,
      .grant_digest = granted.grant->receipt_digest,
      .journal_id = request.journal_id,
      .run_id = request.run_id,
      .logical_lease_id = request.logical_lease_id,
      .logical_fencing_token = request.logical_fencing_token,
      .canonical_request_digest = {},
  });
  SagaOneShotFault release_gap(
      trainvm::HostGrantSagaFaultPoint::host_before_journal);
  trainvm::HostGrantSagaReconciler release_interrupted(journal, host,
                                                       &release_gap);
  bool release_interrupted_once = false;
  try {
    (void)release_interrupted.reconcile_release(request.request_id, release,
                                                test_time(30));
  } catch (const std::runtime_error&) {
    release_interrupted_once = true;
  }
  check(release_interrupted_once &&
            journal.host_grant_saga(request.request_id)->release_intent &&
            !journal.host_grant_saga(request.request_id)->release_receipt,
        "host release gap remains durably blocked and replayable");
  recover_matches(running_state,
                  "controller restart ignores durable host release intent without changing FSM state");
  const auto released = reconciler.reconcile_release(
      request.request_id, release, test_time(31));
  check(released.release_receipt && host.release_calls == 2U &&
            host.release_replays == 1U,
        "release retry converges without a second physical release");
  recover_matches(running_state,
                  "controller restart ignores durable host release receipt without changing FSM state");
  bool released_launch = false;
  try {
    journal.require_host_launch_eligible(
        {.request_id = request.request_id,
         .grant_digest = granted.grant->receipt_digest,
         .fences = granted.grant->fences},
        test_time(32));
  } catch (const trainvm::OperationPreconditionError&) {
    released_launch = true;
  }
  check(released_launch, "released host grants are never launch eligible");

  const auto replacement_request = trainvm::seal_resource_request({
      .api_version = std::string(trainvm::kHostResourceRequestApiVersion),
      .request_id = "saga-replacement-request",
      .journal_id = request.journal_id,
      .run_id = request.run_id,
      .logical_lease_id = request.logical_lease_id,
      .logical_fencing_token = request.logical_fencing_token,
      .count = 1,
      .access_mode = trainvm::ResourceAccessMode::mutex_exclusive,
      .topology = trainvm::TopologyPolicy::any,
      .selector = {},
      .canonical_request_digest = {},
  });
  const auto replacement = reconciler.reconcile_request(
      replacement_request, test_time(33));
  const auto replacement_claim = journal.host_launch_grant_claim(
      request.run_id, saga_key, request.logical_lease_id,
      request.logical_fencing_token, test_time(34));
  bool old_claim_rejected_after_regrant = false;
  try {
    journal.require_host_launch_eligible(*selected_claim, test_time(34));
  } catch (const trainvm::OperationPreconditionError&) {
    old_claim_rejected_after_regrant = true;
  }
  check(replacement.grant && replacement_claim &&
            replacement_claim->request_id == replacement_request.request_id &&
            replacement_claim->grant_digest ==
                replacement.grant->receipt_digest &&
            replacement.grant->fences.front().generation == 2U &&
            old_claim_rejected_after_regrant,
        "release and regrant select generation two without substituting it for the released launch claim");
  recover_matches(running_state,
                  "controller restart ignores replacement host saga evidence without changing FSM state");
  const std::uint64_t before_revocation_checks = journal.event_count();
  bool stale_hello_rejected = false;
  bool stale_dispatch_rejected = false;
  bool stale_result_rejected = false;
  bool stale_control_ack_rejected = false;
  try {
    (void)controller.accept_worker_hello(hello, test_time(35));
  } catch (const trainvm::OperationPreconditionError&) {
    stale_hello_rejected = true;
  }
  try {
    (void)controller.prepare_dispatch(test_time(35));
  } catch (const trainvm::OperationPreconditionError&) {
    stale_dispatch_rejected = true;
  }
  try {
    (void)controller.handle_event(
        {.event_id = dispatch.dispatch_id + ":stale-result",
         .run_id = dispatch.run_id,
         .run_revision = dispatch.run_revision,
         .plan_revision = dispatch.plan_revision,
         .node_id = dispatch.node_id,
         .attempt_id = dispatch.attempt_id,
         .worker_sequence = 1,
         .event_type = "worker.completed",
         .event_version = 1,
         .wall_time_ns = 35,
         .monotonic_time_ns = 35,
         .optimizer_step = std::uint64_t{1},
         .payload = {{"reason", "cache_span_complete"}}},
        {.run_id = launch.run_id,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .launch_nonce = launch.launch_nonce,
         .concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token},
        test_time(35));
  } catch (const trainvm::OperationPreconditionError&) {
    stale_result_rejected = true;
  }
  try {
    (void)controller.acknowledge_controls(
        pending_control.command->command_id,
        {.concurrency_key = launch.concurrency_key,
         .lease_id = launch.lease_id,
         .fencing_token = launch.fencing_token,
         .node_id = launch.node_id,
         .attempt_id = launch.attempt_id,
         .worker_sequence = 1},
        trainvm::ControlCommandStatus::applied, std::uint64_t{1},
        pending_control.command->assignments, nlohmann::json::array(),
        test_time(35));
  } catch (const trainvm::OperationPreconditionError&) {
    stale_control_ack_rejected = true;
  }
  const auto still_pending_control =
      journal.control_command(pending_control.command->command_id);
  check(stale_hello_rejected && stale_dispatch_rejected &&
            stale_result_rejected && stale_control_ack_rejected &&
            still_pending_control &&
            still_pending_control->status ==
                trainvm::ControlCommandStatus::requested &&
            journal.event_count() == before_revocation_checks,
        "release and regrant revoke old hello, dispatch, result, and control-ack authority without journal mutation");

  check(sqlite3_open((directory / "journal.db").c_str(), &raw) == SQLITE_OK,
        "host saga event tamper connection opens");
  if (raw != nullptr) {
    check(sqlite3_exec(raw, R"sql(
      UPDATE events SET payload_json='{}'
      WHERE event_type='host.resource_grant_recorded';
    )sql", nullptr, nullptr, nullptr) == SQLITE_OK,
          "host saga chained event tamper is injected");
    sqlite3_close(raw);
  }
  bool forged_rebuild_blocked = false;
  try {
    (void)journal.rebuild_projections();
  } catch (const std::runtime_error&) {
    forged_rebuild_blocked = true;
  }
  check(forged_rebuild_blocked,
        "tampered chained host receipt blocks projection rebuild");
  std::filesystem::remove_all(directory);
}

void test_service_reconciliation_supervisor() {
  auto source = load_fixture();
  source["spec"]["resources"]["lease_timeout_seconds"] = 5;
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid(), "supervisor restart fixture compiles");
  if (!compiled.plan) return;

  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-service-supervisor-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  check(::chmod(directory.c_str(), 0700) == 0,
        "supervisor fixture protects its authority directory");
  const auto database = directory / "journal.db";
  const std::string run_id = "service-supervisor-restart-run";
  const auto adapters = fixture_adapter_profiles();
  const trainvm::HostIdentity authority_host{
      .host_id = "sha256:" + std::string(64U, '7'),
      .boot_id = kTestBootId,
  };
  {
    trainvm::AdapterRegistry registry(adapters);
    trainvm::Journal journal(database);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(
        adapter_locked_submission(*compiled.plan, registry));
    const auto first = journal.reconcilable_projections({}, 1U);
    const auto exhausted =
        journal.reconcilable_projections(run_id, 1U);
    bool zero_limit_rejected = false;
    try {
      (void)journal.reconcilable_projections({}, 0U);
    } catch (const std::invalid_argument&) {
      zero_limit_rejected = true;
    }
    check(first.size() == 1U && first.front().run_id == run_id &&
              exhausted.empty() && zero_limit_rejected,
          "journal exposes bounded stable reconciliation pagination");
  }

  const auto wait_until = [](const std::function<bool()>& predicate) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
  };
  std::atomic<std::int64_t> authority_now_ns{1'000'000'000LL};
  std::size_t stable_event_count = 0U;
  {
    trainvm::TrainVMService service(
        database, trainvm::AdapterRegistry(adapters),
        fixture_disabled_host_launch_registry(), authority_host,
        [&authority_now_ns] {
          return test_time(authority_now_ns.load(std::memory_order_relaxed));
        },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Journal observer(database);
    service.start_reconciliation_supervisor();
    const bool launched = wait_until([&] {
      return observer
                 .event(run_id +
                        ":worker-launch:train_to_boundary:train_to_boundary@1")
                 .has_value() &&
             service.lease_renewals_.tracked_count() == 1U;
    });
    const auto initial_lease = observer.active_lease(
        compiled.plan->experiment.spec.workspace.concurrency_key,
        test_time(authority_now_ns.load(std::memory_order_relaxed)));
    check(launched && initial_lease &&
              service.lease_renewals_.tracked_count() == 1U,
          "supervisor restart scan advances a queued run and tracks its exact lease");
    if (!initial_lease) {
      service.stop_reconciliation_supervisor();
      std::filesystem::remove_all(directory);
      return;
    }
    authority_now_ns.store(initial_lease->expires_boottime_ns -
                               1'000'000'000LL,
                           std::memory_order_relaxed);
    const bool renewed = wait_until([&] {
      const auto active = observer.active_lease(
          initial_lease->concurrency_key,
          test_time(authority_now_ns.load(std::memory_order_relaxed)));
      return active && active->expires_boottime_ns >
                           initial_lease->expires_boottime_ns;
    });
    service.stop_reconciliation_supervisor();
    const auto renewed_lease = observer.active_lease(
        initial_lease->concurrency_key,
        test_time(authority_now_ns.load(std::memory_order_relaxed)));
    std::optional<trainvm::JournalLogicalFenceSnapshot> renewal_snapshot;
    if (renewed_lease) {
      renewal_snapshot = service.journal_.journal_logical_fence_snapshot(
          renewed_lease->concurrency_key, renewed_lease->owner_run_id,
          renewed_lease->lease_id, renewed_lease->fencing_token,
          test_time(authority_now_ns.load(std::memory_order_relaxed)));
    }
    const bool renewal_evidence =
        renewal_snapshot && renewal_snapshot->authority_revision >= 1U &&
        observer.verify_chain();
    const auto run_failure = service.reconciliation_failure(run_id);
    const auto scan_failure = service.reconciliation_failure("__scan__");
    const auto renewal_failure =
        service.reconciliation_failure("__lease_renewal__");
    if (!renewed || !renewal_evidence || run_failure || scan_failure ||
        renewal_failure) {
      std::cerr << "supervisor diagnostics: run="
                << run_failure.value_or("none")
                << " scan=" << scan_failure.value_or("none")
                << " renewal=" << renewal_failure.value_or("none")
                << " renewed=" << renewed
                << " evidence=" << renewal_evidence
                << " authority_revision="
                << (renewal_snapshot ? renewal_snapshot->authority_revision
                                     : 0U)
                << " chain=" << observer.verify_chain() << '\n';
    }
    check(renewed && renewal_evidence && !run_failure && !scan_failure &&
              !renewal_failure,
          "supervisor renews the live lease before expiry without poisoning authority");
    stable_event_count = observer.event_count();
  }

  {
    trainvm::TrainVMService restarted(
        database, trainvm::AdapterRegistry(adapters),
        fixture_disabled_host_launch_registry(), authority_host,
        [&authority_now_ns] {
          return test_time(authority_now_ns.load(std::memory_order_relaxed));
        },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    restarted.start_reconciliation_supervisor();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    restarted.stop_reconciliation_supervisor();
    trainvm::Journal observer(database);
    check(observer.event_count() == stable_event_count &&
              !restarted.reconciliation_failure(run_id),
          "supervisor restart replays an awaiting launch without journal mutation");
  }
  std::filesystem::remove_all(directory);
}

// Regression for the controller that sustained ~85% of a core for eight hours
// while its worker was defunct.
//
// The shape of the bug: a run whose worker will never report stays in the
// reconcilable projection set forever, and the supervisor re-reconciled every
// such run on every 250 ms tick. Each reconcile builds a Controller and calls
// recover(), which re-verifies the whole journal hash chain and replays the
// run's entire event history. On an eight-hour training journal that is a very
// large amount of work, repeated four times a second, producing nothing: the
// journal does not move, so every pass is provably a no-op.
//
// That is what makes it a busy poll rather than reconciliation that has not
// finished yet. This test pins the distinction directly: it asserts the journal
// does not move (no progress is even being attempted) *and* that the supervisor
// stops doing work anyway.
//
// Determinism comes from the injected authority clock. While it is frozen, a
// parked run can never reach its backoff deadline, so the post-fix step delta
// is exactly zero rather than "small". Before the fix the delta is one pass per
// 250 ms tick, because nothing consulted a deadline at all.
void test_service_supervisor_settles_after_terminal_worker() {
  auto source = load_fixture();
  source["spec"]["resources"]["lease_timeout_seconds"] = 5;
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid(), "terminal-worker spin fixture compiles");
  if (!compiled.plan) return;

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-supervisor-spin-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  check(::chmod(directory.c_str(), 0700) == 0,
        "spin fixture protects its authority directory");
  const auto database = directory / "journal.db";
  const std::string run_id = "service-supervisor-spin-run";
  const auto adapters = fixture_adapter_profiles();
  const trainvm::HostIdentity authority_host{
      .host_id = "sha256:" + std::string(64U, '7'),
      .boot_id = kTestBootId,
  };
  {
    trainvm::AdapterRegistry registry(adapters);
    trainvm::Journal journal(database);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(
        adapter_locked_submission(*compiled.plan, registry));
  }

  const auto wait_until = [](const std::function<bool()>& predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
  };

  std::atomic<std::int64_t> authority_now_ns{1'000'000'000LL};
  trainvm::TrainVMService service(
      database, trainvm::AdapterRegistry(adapters),
      fixture_disabled_host_launch_registry(), authority_host,
      [&authority_now_ns] {
        return test_time(authority_now_ns.load(std::memory_order_relaxed));
      },
      trainvm::HostGrantEnforcement::legacy_process_free_test);
  trainvm::Journal observer(database);
  service.start_reconciliation_supervisor();

  // The worker launch is prepared and then nothing ever reports: exactly the
  // state a defunct child or a dropped worker socket leaves behind.
  const bool parked = wait_until([&] {
    return observer
               .event(run_id +
                      ":worker-launch:train_to_boundary:train_to_boundary@1")
               .has_value() &&
           service.reconciliation_metrics().skipped_idle_runs > 0U;
  });
  const auto settled = service.reconciliation_metrics();
  const std::uint64_t settled_events = observer.event_count();

  // Six cadences of wall time. Every one of them used to be a full controller
  // recovery over the whole journal.
  std::this_thread::sleep_for(std::chrono::milliseconds(1'500));

  const auto idled = service.reconciliation_metrics();
  const std::uint64_t idled_events = observer.event_count();
  const auto waits = service.reconciliation_waits();
  const auto parked_wait = std::ranges::find_if(
      waits, [&](const trainvm::ReconciliationRunWait& wait) {
        return wait.run_id == run_id;
      });

  const bool journal_frozen = idled_events == settled_events;
  const bool stopped_working =
      idled.reconcile_steps == settled.reconcile_steps &&
      idled.reconcile_passes == settled.reconcile_passes;
  const bool kept_ticking = idled.wakes > settled.wakes + 2U;
  const bool skipped =
      idled.skipped_idle_runs > settled.skipped_idle_runs + 2U;
  const bool reported =
      parked_wait != waits.end() &&
      (parked_wait->wait_reason == "waiting for worker evidence" ||
       parked_wait->wait_reason ==
           "waiting for the launched worker to report") &&
      parked_wait->backoff_ns >=
          trainvm::TrainVMService::kSupervisorCadenceNs &&
      parked_wait->idle_passes > 0U;
  if (!parked || !journal_frozen || !stopped_working || !kept_ticking ||
      !skipped || !reported) {
    std::cerr << "supervisor spin diagnostics:"
              << " parked=" << parked
              << " events " << settled_events << "->" << idled_events
              << " steps " << settled.reconcile_steps << "->"
              << idled.reconcile_steps
              << " passes " << settled.reconcile_passes << "->"
              << idled.reconcile_passes
              << " wakes " << settled.wakes << "->" << idled.wakes
              << " skipped " << settled.skipped_idle_runs << "->"
              << idled.skipped_idle_runs
              << " reason="
              << (parked_wait != waits.end() ? parked_wait->wait_reason
                                             : std::string("<none>"))
              << " backoff="
              << (parked_wait != waits.end() ? parked_wait->backoff_ns : -1)
              << '\n';
  }
  check(parked && journal_frozen && kept_ticking && skipped,
        "terminal worker leaves a frozen journal the supervisor keeps scanning");
  check(stopped_working,
        "supervisor stops re-reconciling a run whose journal cannot move");
  check(reported,
        "supervisor telemetry names the parked run, its wait reason, and its backoff");

  // The same telemetry has to be reachable over the wire, and specifically
  // without a configured hostd: an authority whose host daemon is missing or
  // poisoned is exactly when this needs to be readable.
  trainvm::v1::GetReconciliationStatusRequest status_request;
  trainvm::v1::GetReconciliationStatusResponse status_response;
  const grpc::Status status =
      service.GetReconciliationStatus(nullptr, &status_request,
                                      &status_response);
  const auto wire_wait = std::ranges::find_if(
      status_response.waits(),
      [&](const trainvm::v1::ReconciliationRunWait& wait) {
        return wait.run_id() == run_id;
      });
  check(status.ok() && status_response.supervisor_running() &&
            status_response.reconcile_steps() == idled.reconcile_steps &&
            status_response.skipped_idle_runs() > 0U &&
            status_response.tracked_runs() > 0U &&
            wire_wait != status_response.waits().end() &&
            !wire_wait->wait_reason().empty() &&
            wire_wait->backoff_ns() >=
                trainvm::TrainVMService::kSupervisorCadenceNs,
        "GetReconciliationStatus reports the spin counters without a hostd");

  // The park is a backoff, not a wedge: an explicit wake and a journal write
  // both put the run back to work, and so does the clock once the backoff
  // ceiling elapses. Losing any of those would trade a spin for a stall.
  const auto before_wake = service.reconciliation_metrics();
  service.notify_reconciliation(run_id);
  const bool woke = wait_until([&] {
    return service.reconciliation_metrics().reconcile_steps >
           before_wake.reconcile_steps;
  });
  check(woke, "an explicit wake bypasses supervisor backoff");

  // Let the woken pass finish and re-park before measuring, so the deadline
  // being tested is the one this snapshot describes.
  const auto after_wake = service.reconciliation_metrics();
  (void)wait_until([&] {
    return service.reconciliation_metrics().skipped_idle_runs >
           after_wake.skipped_idle_runs + 1U;
  });
  // Advanced in steps smaller than the lease renewal margin: the point is that
  // the deadline is consulted at all, and expiring the lease under the
  // supervisor would change what is being measured.
  const auto before_clock = service.reconciliation_metrics();
  const auto clock_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  bool rechecked = false;
  while (!rechecked && std::chrono::steady_clock::now() < clock_deadline) {
    authority_now_ns.fetch_add(
        trainvm::TrainVMService::kSupervisorCadenceNs * 2LL,
        std::memory_order_relaxed);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    rechecked = service.reconciliation_metrics().reconcile_steps >
                before_clock.reconcile_steps;
  }
  check(rechecked,
        "a parked run is still re-checked once its backoff deadline elapses");

  service.stop_reconciliation_supervisor();
  std::filesystem::remove_all(directory);
}

// Soak over the four ways a run stops producing evidence. All of them must
// leave the supervisor idle rather than spinning, and all of them must name
// themselves in telemetry.
//
//   - defunct child / dropped worker socket: a committed launch with no worker
//     evidence. Indistinguishable to the authority, and covered together.
//   - poisoned authority dependency: every reconcile pass throws. Stands in for
//     a poisoned hostd, which fails the same way from the supervisor's side.
//   - pending lifecycle command: a durable command nobody can acknowledge.
void test_service_supervisor_idle_soak() {
  auto source = load_fixture();
  source["spec"]["resources"]["lease_timeout_seconds"] = 5;
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid(), "supervisor soak fixture compiles");
  if (!compiled.plan) return;

  const auto directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-supervisor-soak-" +
       std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  check(::chmod(directory.c_str(), 0700) == 0,
        "soak fixture protects its authority directory");
  const auto database = directory / "journal.db";
  const std::string run_id = "service-supervisor-soak-run";
  const auto adapters = fixture_adapter_profiles();
  const trainvm::HostIdentity authority_host{
      .host_id = "sha256:" + std::string(64U, '7'),
      .boot_id = kTestBootId,
  };
  {
    trainvm::AdapterRegistry registry(adapters);
    trainvm::Journal journal(database);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    controller.create_queued(
        adapter_locked_submission(*compiled.plan, registry));
  }

  const auto wait_until = [](const std::function<bool()>& predicate) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < deadline) {
      if (predicate()) return true;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
  };
  std::atomic<std::int64_t> authority_now_ns{1'000'000'000LL};
  const auto clock = [&authority_now_ns] {
    return test_time(authority_now_ns.load(std::memory_order_relaxed));
  };
  // Waits for the scenario's own transitions to finish, then measures four
  // cadences of genuinely idle supervisor. Without the settle phase this would
  // measure the run still working, not the idle path under test.
  const auto soak_delta = [](trainvm::TrainVMService& service) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::chrono::steady_clock::now() < deadline) {
      const auto before = service.reconciliation_metrics();
      std::this_thread::sleep_for(std::chrono::milliseconds(600));
      const auto after = service.reconciliation_metrics();
      if (after.reconcile_steps == before.reconcile_steps &&
          after.wakes > before.wakes) {
        break;
      }
    }
    const auto before = service.reconciliation_metrics();
    std::this_thread::sleep_for(std::chrono::milliseconds(1'000));
    const auto after = service.reconciliation_metrics();
    return std::pair<std::uint64_t, std::uint64_t>{
        after.reconcile_steps - before.reconcile_steps,
        after.wakes - before.wakes};
  };

  std::uint64_t launched_events = 0U;
  {
    trainvm::TrainVMService service(
        database, trainvm::AdapterRegistry(adapters),
        fixture_disabled_host_launch_registry(), authority_host, clock,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Journal observer(database);
    service.start_reconciliation_supervisor();
    const bool parked = wait_until([&] {
      return service.reconciliation_metrics().skipped_idle_runs > 0U;
    });
    const auto [steps, wakes] = soak_delta(service);
    launched_events = observer.event_count();
    check(parked && steps == 0U && wakes > 2U,
          "a defunct child that never says hello settles to an idle supervisor");
    service.stop_reconciliation_supervisor();
  }

  // Dropped worker socket, then a lifecycle command nobody can acknowledge.
  // A separate journal: this scenario needs a run that reached running/running,
  // which is the state a worker reaches by connecting and then vanishing.
  {
    const auto disconnected_database = directory / "disconnected.db";
    const std::string disconnected_run = "service-supervisor-disconnect-run";
    trainvm::WorkerLaunchTicket launch;
    {
      trainvm::Journal journal(
          disconnected_database, std::nullopt,
          trainvm::HostGrantEnforcement::legacy_process_free_test);
      trainvm::AdapterRegistry registry(adapters);
      trainvm::Controller controller(*compiled.plan, journal,
                                     disconnected_run);
      controller.create_queued(
          adapter_locked_submission(*compiled.plan, registry));
      (void)controller.begin_acquisition(test_time(2'000));
      const auto& node = compiled.plan->experiment.spec.workflow.nodes.at(
          controller.state().current_node_id);
      const auto& component =
          compiled.plan->experiment.spec.components.at(node.invoke.component);
      launch = controller.prepare_worker_launch(
          registry.worker_launch_request(component, node.invoke.operation),
          test_time(2'100));
      (void)bind_test_worker_launch(controller, launch, 2'150);
    }
    std::atomic<std::int64_t> disconnected_now_ns{2'200};
    trainvm::TrainVMService service(
        disconnected_database, trainvm::AdapterRegistry(adapters),
        fixture_test_host_launch_registry(*compiled.plan, launch),
        fixture_test_host_identity(),
        [&disconnected_now_ns] {
          return test_time(
              disconnected_now_ns.load(std::memory_order_relaxed));
        },
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    prime_test_service_launch(service, launch);
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
    trainvm::TrainVMService::WorkerConnection connection;
    const grpc::Status opened =
        service.open_worker_connection(hello, connection);
    // The worker is now gone: the stream is simply never read or written
    // again, which is what a dropped socket looks like to the authority.
    service.start_reconciliation_supervisor();
    const bool parked = wait_until([&] {
      return service.reconciliation_metrics().skipped_idle_runs > 0U;
    });
    const auto [steps, wakes] = soak_delta(service);
    check(opened.ok() && parked && steps == 0U && wakes > 2U,
          "a dropped worker socket settles to an idle supervisor");

    // A durable lifecycle command nobody can acknowledge must still be picked
    // up promptly. It appends lifecycle.requested, and a journal write is the
    // event-driven path out of backoff -- no explicit wake involved.
    const auto before = service.reconciliation_metrics();
    const auto projection = [&] {
      std::scoped_lock lock(service.command_mutex_);
      return service.journal_.projection(disconnected_run);
    }();
    bool submitted = false;
    if (projection && projection->desired_state == "running" &&
        projection->observed_state == "running") {
      trainvm::LifecycleCommand command;
      command.command_id = disconnected_run + ":soak-pause";
      command.run_id = disconnected_run;
      command.idempotency_key = "soak-pause";
      command.expected_run_revision = projection->run_revision;
      command.kind = trainvm::LifecycleCommandKind::pause;
      command.author = "soak";
      command.reason = "supervisor idle soak";
      std::scoped_lock lock(service.command_mutex_);
      (void)service.journal_.submit_lifecycle_command(std::move(command));
      submitted = true;
    }
    const bool noticed =
        submitted && wait_until([&] {
          return service.reconciliation_metrics().reconcile_steps >
                 before.reconcile_steps;
        });
    const auto [after_steps, after_wakes] = soak_delta(service);
    if (!opened.ok() || !submitted || !noticed || after_steps != 0U) {
      std::cerr << "supervisor disconnect diagnostics: open="
                << opened.error_message() << " state="
                << (projection ? projection->desired_state + "/" +
                                     projection->observed_state
                               : std::string("<none>"))
                << " submitted=" << submitted << " noticed=" << noticed
                << " after_steps=" << after_steps << '\n';
    }
    check(submitted && noticed && after_steps == 0U && after_wakes > 2U,
          "a pending lifecycle command wakes the supervisor once and then settles");
    service.stop_reconciliation_supervisor();
  }

  // Poisoned authority dependency: the restarted service is locked to a
  // different adapter fingerprint, so every reconcile pass throws. The failure
  // must be retried under backoff, not four times a second, and it must not be
  // silently swallowed either.
  {
    trainvm::TrainVMService poisoned(
        database, trainvm::AdapterRegistry(fixture_adapter_profiles('b')),
        fixture_disabled_host_launch_registry(), authority_host, clock,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Journal observer(database);
    poisoned.start_reconciliation_supervisor();
    const bool failed = wait_until([&] {
      return poisoned.reconciliation_failure(run_id).has_value();
    });
    const auto before = poisoned.reconciliation_metrics();
    std::this_thread::sleep_for(std::chrono::milliseconds(1'000));
    const auto after = poisoned.reconciliation_metrics();
    const auto waits = poisoned.reconciliation_waits();
    const auto poisoned_wait = std::ranges::find_if(
        waits, [&](const trainvm::ReconciliationRunWait& wait) {
          return wait.run_id == run_id;
        });
    const bool retried_slowly =
        after.reconcile_steps - before.reconcile_steps <= 2U;
    const bool retry_reported =
        poisoned_wait != waits.end() && poisoned_wait->retries > 0U &&
        poisoned_wait->wait_reason == "retrying after a reconciliation failure";
    if (!failed || !retried_slowly || !retry_reported) {
      std::cerr << "supervisor poison diagnostics: failed=" << failed
                << " steps " << before.reconcile_steps << "->"
                << after.reconcile_steps
                << " failures=" << after.failures << " reason="
                << (poisoned_wait != waits.end() ? poisoned_wait->wait_reason
                                                 : std::string("<none>"))
                << '\n';
    }
    check(failed && retried_slowly && retry_reported,
          "a poisoned authority dependency retries under backoff and says so");
    check(observer.event_count() == launched_events && observer.verify_chain(),
          "a poisoned authority retries without mutating the journal");
    poisoned.stop_reconciliation_supervisor();
  }
  std::filesystem::remove_all(directory);
}

void test_service_orders_physical_before_logical_release() {
  auto source = load_fixture();
  source["spec"]["artifacts"]["checkpoint"]["required"] = true;
  source["spec"]["artifacts"]["eval_gallery"]["required"] = true;
  source["spec"]["artifacts"]["test_eval"] = {
      {"type", "report"},
      {"schema", "rwkv-lab.hf-test-caption-evidence-bundle.v1"},
      {"immutability", "immutable"},
      {"fingerprint", "manifest_sha256"},
      {"required", true}};
  auto& nodes = source["spec"]["workflow"]["nodes"];
  const auto acquire = nodes.at("acquire_gpu");
  const auto train = nodes.at("train_to_boundary");
  const auto release = nodes.at("release_gpu");
  nodes = {{"acquire_gpu", acquire},
           {"train_to_boundary", train},
           {"release_gpu", release}};
  nodes["train_to_boundary"]["transitions"] = nlohmann::json::array(
      {train.at("transitions").at(1), train.at("transitions").at(2)});
  nodes["train_to_boundary"]["publishes"]["test_eval"] = "test_eval";
  const auto compiled = trainvm::compile_document(source);
  check(compiled.valid(),
        "terminal release fixture compiles a builtin-only lifecycle");
  if (!compiled.plan) return;

  const auto directory = std::filesystem::temp_directory_path() /
                         ("trainvm-service-terminal-release-" +
                          std::to_string(static_cast<long long>(getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  check(::chmod(directory.c_str(), 0700) == 0,
        "terminal release fixture protects its directory");
  trainvm::ObservedHostResource gpu{
      .id = {.kind = trainvm::HostResourceKind::accelerator,
             .vendor = trainvm::HostAcceleratorVendor::nvidia,
             .stable_id = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
             .parent_id = std::nullopt},
      .disposition =
          trainvm::ResourceObservationDisposition::audited_eligible,
      .compute_contexts = trainvm::ResourceContextDisposition::absent,
      .graphics_contexts = trainvm::ResourceContextDisposition::absent,
      .pci_bdf = "0000:01:00.0",
      .device_major = 195U,
      .device_minor = 0U,
      .device_nodes = {},
      .numa_node = 0,
      .pcie_root_id = "0000:00",
      .fabric_clique_id = "fabric-0",
      .total_memory_bytes = 80ULL << 30U,
      .labels = {},
  };
  trainvm::HostKernelSnapshot kernel_snapshot{
      .api_version = std::string(trainvm::kHostInventoryApiVersion),
      .host_id = "sha256:" + std::string(64U, 'e'),
      .boot_id = kTestBootId,
      .broker_epoch = "broker-service-release",
      .begin_revision = "revision-service-release",
      .end_revision = "revision-service-release",
      .probes = {{.vendor = trainvm::HostAcceleratorVendor::nvidia,
                  .disposition = trainvm::ProbeDisposition::complete,
                  .context_details_complete = true,
                  .detail = "deterministic service-release fixture"}},
      .resources = {gpu},
  };
  trainvm::FakeHostKernel kernel(
      std::vector<trainvm::FakeHostKernelStep>{{
          .snapshot = std::move(kernel_snapshot),
          .failure = std::nullopt,
      }});
  const auto inventory = trainvm::capture_host_inventory(kernel);
  auto ledger_authority =
      std::make_shared<trainvm::HostLedgerFilesystemAuthority>(
          trainvm::HostLedgerFilesystemAuthority::acquire({
              .api_version =
                  std::string(trainvm::kHostLedgerAuthorityApiVersion),
              .ledger_path = directory / "host-resource.db",
              .expected_owner_uid = ::geteuid(),
              .expected_owner_gid = ::getegid(),
              .enforcement_grade =
                  trainvm::HostLedgerEnforcementGrade::cooperative_test,
          }));
  trainvm::SQLiteHostLedger ledger(ledger_authority, inventory);
  auto host = std::make_shared<SagaHostClient>(ledger);
  auto process = std::make_shared<SagaProcessClient>(ledger);
  const trainvm::HostIdentity host_identity{.host_id = inventory.host_id,
                                             .boot_id = inventory.boot_id};
  auto profiles = fixture_adapter_profiles();
  const auto train_profile = std::ranges::find_if(
      profiles, [](const trainvm::AdapterProfile& profile) {
        return profile.key.operation == "train";
      });
  if (train_profile != profiles.end()) {
    train_profile->authoring->outputs.at("checkpoint").required = true;
    train_profile->authoring->outputs.at("eval_gallery").required = true;
    train_profile->authoring->outputs.emplace(
        "test_eval",
        operation_port(
            trainvm::OperationPortType::artifact, true,
            trainvm::ArtifactType::report,
            "rwkv-lab.hf-test-caption-evidence-bundle.v1"));
  }
  std::int64_t service_now_ns = 10;
  trainvm::WorkerLaunchTicket registry_ticket;
  registry_ticket.node_id = "train_to_boundary";
  registry_ticket.code_fingerprint =
      "sha256:" + std::string(64U, 'a');
  registry_ticket.required_capabilities = {"worker.controls",
                                           "worker.metrics"};
  trainvm::TrainVMService service(
      directory / "journal.db", trainvm::AdapterRegistry(profiles),
      fixture_test_host_launch_registry(*compiled.plan, registry_ticket),
      host_identity,
      [&service_now_ns] { return test_time(service_now_ns); },
      trainvm::HostGrantEnforcement::required,
      trainvm::TrainingComponentRegistry({}), host, process,
      "unix:/tmp/trainvm-process-saga.sock");
  const std::string run_id = "service-terminal-release-run";
  trainvm::Controller controller(*compiled.plan, service.journal_, run_id);
  controller.create_queued(
      adapter_locked_submission(*compiled.plan, service.adapter_registry_));
  const auto acquired = service.reconcile_once(run_id);
  const auto lease = service.journal_.active_lease(
      compiled.plan->experiment.spec.workspace.concurrency_key,
      test_time(10));
  check(acquired.disposition == trainvm::ReconcileDisposition::lease_acquired &&
            lease,
        "service terminal fixture retains its logical lease at release node");
  if (!lease) {
    std::filesystem::remove_all(directory);
    return;
  }
  const auto request = trainvm::build_resource_bundle_request({
      .journal_id = service.journal_.journal_id(),
      .plan_hash = compiled.plan->plan_hash,
      .run_id = run_id,
      .resources = compiled.plan->experiment.spec.resources,
      .lease = *lease,
  });
  const auto granted = service.host_grant_saga_->reconcile_request(
      request, test_time(10));
  trainvm::Controller worker(*compiled.plan, service.journal_, run_id);
  worker.recover();
  const auto launch = worker.prepare_worker_launch(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      test_time(10));
  (void)bind_test_worker_launch(worker, launch, 10, host_identity);
  (void)worker.accept_worker_hello(
      {.run_id = launch.run_id,
       .node_id = launch.node_id,
       .attempt_id = launch.attempt_id,
       .launch_nonce = launch.launch_nonce,
       .adapter = launch.adapter,
       .adapter_version = launch.adapter_version,
       .code_fingerprint = launch.code_fingerprint,
       .capabilities = launch.required_capabilities,
       .last_acked_controller_sequence = 0U,
       .concurrency_key = launch.concurrency_key,
       .lease_id = launch.lease_id,
       .fencing_token = launch.fencing_token},
      test_time(10));
  const auto dispatch = worker.prepare_dispatch(test_time(10));
  (void)worker.handle_event(
      {.event_id = dispatch.dispatch_id + ":result",
       .run_id = run_id,
       .run_revision = dispatch.run_revision,
       .plan_revision = dispatch.plan_revision,
       .node_id = launch.node_id,
       .attempt_id = launch.attempt_id,
       .worker_sequence = 1U,
       .event_type = "worker.completed",
       .event_version = 1U,
       .wall_time_ns = 10,
       .monotonic_time_ns = 10U,
       .optimizer_step = 1U,
       .payload = {{"reason", "training_complete"}}},
      {.run_id = launch.run_id,
       .node_id = launch.node_id,
       .attempt_id = launch.attempt_id,
       .launch_nonce = launch.launch_nonce,
       .concurrency_key = launch.concurrency_key,
       .lease_id = launch.lease_id,
       .fencing_token = launch.fencing_token},
      test_time(10));
  const auto physical = service.reconcile_once(run_id);
  const auto still_live = service.journal_.active_lease(
      lease->concurrency_key, test_time(10));
  const auto physical_saga =
      service.journal_.host_grant_saga(request.request_id);
  check(granted.grant &&
            physical.disposition ==
                trainvm::ReconcileDisposition::host_grant_released &&
            physical_saga && physical_saga->release_receipt && still_live &&
            host->release_calls == 1U,
        "physical release receipt is durable while logical lease remains live");
  const auto logical = service.reconcile_once(run_id);
  const auto projection = service.journal_.projection(run_id);
  check(logical.disposition == trainvm::ReconcileDisposition::builtin_completed &&
            !service.journal_.active_lease(lease->concurrency_key,
                                           test_time(10)) &&
            projection && projection->observed_state == "completed" &&
            host->release_calls == 1U,
        "logical release occurs only after physical release and does not replay host mutation");

  const std::string cancel_run_id = "service-cancel-release-run";
  trainvm::Controller cancel_controller(
      *compiled.plan, service.journal_, cancel_run_id);
  cancel_controller.create_queued(
      adapter_locked_submission(*compiled.plan, service.adapter_registry_));
  const auto cancel_acquired = service.reconcile_once(cancel_run_id);
  const auto cancel_lease = service.journal_.active_lease(
      compiled.plan->experiment.spec.workspace.concurrency_key,
      test_time(10));
  check(cancel_acquired.disposition ==
                trainvm::ReconcileDisposition::lease_acquired &&
            cancel_lease && cancel_lease->owner_run_id == cancel_run_id,
        "graceful cancellation fixture acquires a fresh fenced lease");
  if (!cancel_lease) {
    std::filesystem::remove_all(directory);
    return;
  }
  const auto cancel_request = trainvm::build_resource_bundle_request({
      .journal_id = service.journal_.journal_id(),
      .plan_hash = compiled.plan->plan_hash,
      .run_id = cancel_run_id,
      .resources = compiled.plan->experiment.spec.resources,
      .lease = *cancel_lease,
  });
  const auto cancel_granted = service.host_grant_saga_->reconcile_request(
      cancel_request, test_time(10));
  trainvm::Controller cancel_worker(
      *compiled.plan, service.journal_, cancel_run_id);
  cancel_worker.recover();
  const auto cancel_launch = cancel_worker.prepare_worker_launch(
      {.code_fingerprint = "sha256:" + std::string(64U, 'c'),
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      test_time(10));
  const auto cancel_binding = bind_test_worker_launch(
      cancel_worker, cancel_launch, 10, host_identity);
  (void)service.host_process_saga_->reconcile(
      trainvm::ResolvedLaunch(
          cancel_binding, -1,
          cancel_binding.identity.code ? std::optional<int>{-1}
                                       : std::nullopt,
          -1),
      *cancel_granted.grant,
      trainvm::compile_linux_process_policy(std::nullopt),
      "unix:/tmp/trainvm-process-saga.sock", test_time(10));
  (void)cancel_worker.accept_worker_hello(
      {.run_id = cancel_launch.run_id,
       .node_id = cancel_launch.node_id,
       .attempt_id = cancel_launch.attempt_id,
       .launch_nonce = cancel_launch.launch_nonce,
       .adapter = cancel_launch.adapter,
       .adapter_version = cancel_launch.adapter_version,
       .code_fingerprint = cancel_launch.code_fingerprint,
       .capabilities = cancel_launch.required_capabilities,
       .last_acked_controller_sequence = 0U,
       .concurrency_key = cancel_launch.concurrency_key,
       .lease_id = cancel_launch.lease_id,
       .fencing_token = cancel_launch.fencing_token},
      test_time(10));
  const auto cancel_projection = service.journal_.projection(cancel_run_id);
  const auto cancel_command = cancel_worker.request_cancel(
      "cancel-release-once", cancel_projection->run_revision,
      "operator requested stop", 0, "operator",
      "verify physical cleanup ordering");
  (void)cancel_worker.acknowledge_lifecycle(
      cancel_command.command.command_id,
      {.concurrency_key = cancel_launch.concurrency_key,
       .lease_id = cancel_launch.lease_id,
       .fencing_token = cancel_launch.fencing_token,
       .node_id = cancel_launch.node_id,
       .attempt_id = cancel_launch.attempt_id,
       .worker_sequence = 1U},
      trainvm::LifecycleCommandStatus::applied, std::nullopt, "",
      nlohmann::json::array(), test_time(10));
  const std::size_t releases_before_cancel = host->release_calls;
  const auto cancel_exit = service.reconcile_once(cancel_run_id);
  const auto after_exit = service.journal_.host_grant_saga(
      cancel_request.request_id);
  check(cancel_exit.disposition ==
                trainvm::ReconcileDisposition::host_process_exited &&
            after_exit && !after_exit->release_receipt &&
            service.journal_.active_lease(cancel_lease->concurrency_key,
                                           test_time(10)),
        "cancel waits for durable terminal process evidence before release");
  const auto cancel_release = service.reconcile_once(cancel_run_id);
  check(cancel_release.disposition ==
                trainvm::ReconcileDisposition::host_grant_released &&
            host->release_calls == releases_before_cancel + 1U &&
            service.journal_.active_lease(cancel_lease->concurrency_key,
                                           test_time(10)),
        "cancel releases the physical grant before its logical lease");
  const auto cancel_terminal = service.reconcile_once(cancel_run_id);
  const auto cancelled_projection =
      service.journal_.projection(cancel_run_id);
  check(cancel_terminal.disposition ==
                trainvm::ReconcileDisposition::builtin_completed &&
            !service.journal_.active_lease(cancel_lease->concurrency_key,
                                            test_time(10)) &&
            cancelled_projection &&
            cancelled_projection->desired_state == "cancelled" &&
            cancelled_projection->observed_state == "cancelled" &&
            process->exit_calls == 1U,
        "cancel becomes terminal only after process, grant, and lease cleanup");

  const auto wire_hello = [](const trainvm::WorkerLaunchTicket& ticket) {
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
    hello.set_last_acked_controller_sequence(0U);
    hello.set_concurrency_key(ticket.concurrency_key);
    hello.set_lease_id(ticket.lease_id);
    hello.set_fencing_token(ticket.fencing_token);
    return hello;
  };
  const std::string pause_run_id = "service-resource-pause-resume-run";
  trainvm::Controller pause_controller(
      *compiled.plan, service.journal_, pause_run_id);
  pause_controller.create_queued(
      adapter_locked_submission(*compiled.plan, service.adapter_registry_));
  const auto pause_acquired = service.reconcile_once(pause_run_id);
  const auto pause_lease = service.journal_.active_lease(
      compiled.plan->experiment.spec.workspace.concurrency_key,
      test_time(10));
  check(pause_acquired.disposition ==
                trainvm::ReconcileDisposition::lease_acquired &&
            pause_lease && pause_lease->owner_run_id == pause_run_id,
        "resource-releasing pause fixture acquires its initial logical fence");
  if (!pause_lease) {
    std::filesystem::remove_all(directory);
    return;
  }
  const auto pause_request = trainvm::build_resource_bundle_request({
      .journal_id = service.journal_.journal_id(),
      .plan_hash = compiled.plan->plan_hash,
      .run_id = pause_run_id,
      .resources = compiled.plan->experiment.spec.resources,
      .lease = *pause_lease,
  });
  const auto pause_granted = service.host_grant_saga_->reconcile_request(
      pause_request, test_time(10));
  trainvm::Controller pause_worker(
      *compiled.plan, service.journal_, pause_run_id);
  pause_worker.recover();
  const auto pause_launch = pause_worker.prepare_worker_launch(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      test_time(10));
  const auto pause_binding = bind_test_worker_launch(
      pause_worker, pause_launch, 10, host_identity);
  (void)service.host_process_saga_->reconcile(
      trainvm::ResolvedLaunch(
          pause_binding, -1,
          pause_binding.identity.code ? std::optional<int>{-1}
                                      : std::nullopt,
          -1),
      *pause_granted.grant,
      trainvm::compile_linux_process_policy(std::nullopt),
      "unix:/tmp/trainvm-process-saga.sock", test_time(10));
  prime_test_service_launch(service, pause_launch);
  trainvm::TrainVMService::WorkerConnection pause_connection;
  const auto pause_open = service.open_worker_connection(
      wire_hello(pause_launch), pause_connection);
  const std::string checkpoint_id = "service-pause-checkpoint";
  (void)pause_worker.record_worker_observation(
      {.event_id = pause_connection.dispatch.dispatch_id + ":artifact:" +
                   trainvm::sha256_hex(checkpoint_id),
       .run_id = pause_run_id,
       .run_revision = pause_connection.dispatch.run_revision,
       .plan_revision = pause_connection.dispatch.plan_revision,
       .node_id = pause_launch.node_id,
       .attempt_id = pause_launch.attempt_id,
       .worker_sequence = 1U,
       .event_type = "artifact.published",
       .event_version = 1U,
       .wall_time_ns = 10,
       .monotonic_time_ns = 10U,
       .optimizer_step = std::uint64_t{9},
       .payload = {{"artifact_id", checkpoint_id},
                   {"logical_name", "checkpoint"},
                   {"kind", "checkpoint"},
                   {"schema", "rwkv-lab.mageflow-checkpoint.v1"},
                   {"uri", "file:///sealed/service-pause/manifest.json"},
                   {"size_bytes", std::uint64_t{4096}},
                   {"fingerprint_algorithm", "manifest_sha256"},
                   {"fingerprint",
                    "sha256:" + std::string(64U, 'd')},
                   {"complete", true},
                   {"producer_node_id", pause_launch.node_id},
                   {"producer_attempt_id", pause_launch.attempt_id},
                   {"parent_artifact_ids", nlohmann::json::array()},
                   {"published_at_ns", std::int64_t{10}}}},
      pause_connection.identity, test_time(10));
  const auto before_pause = service.journal_.projection(pause_run_id);
  const auto pause_command = pause_worker.request_lifecycle(
      trainvm::LifecycleCommandKind::pause, "service-release-pause-once",
      before_pause->run_revision, true, true, "operator",
      "release the physical accelerator");
  (void)pause_worker.acknowledge_lifecycle(
      pause_command.command.command_id,
      {.concurrency_key = pause_launch.concurrency_key,
       .lease_id = pause_launch.lease_id,
       .fencing_token = pause_launch.fencing_token,
       .node_id = pause_launch.node_id,
       .attempt_id = pause_launch.attempt_id,
       .worker_sequence = 2U},
      trainvm::LifecycleCommandStatus::applied, std::uint64_t{9},
      checkpoint_id, nlohmann::json::array(), test_time(10));
  check(service.journal_.renew_lease(
            pause_lease->concurrency_key, pause_run_id,
            pause_lease->lease_id, pause_lease->fencing_token,
            test_time(10), 600'000'000'000LL),
        "resource pause fixture keeps its fence live through graceful shutdown");
  service_now_ns = 300'000'000'010LL;
  const std::size_t releases_before_pause = host->release_calls;
  const auto pause_exit = service.reconcile_once(pause_run_id);
  const auto pause_release = service.reconcile_once(pause_run_id);
  const auto pause_complete = service.reconcile_once(pause_run_id);
  const auto paused_projection = service.journal_.projection(pause_run_id);
  check(pause_open.ok() &&
            pause_exit.disposition ==
                trainvm::ReconcileDisposition::host_process_exited &&
            pause_release.disposition ==
                trainvm::ReconcileDisposition::host_grant_released &&
            pause_complete.disposition ==
                trainvm::ReconcileDisposition::builtin_completed &&
            host->release_calls == releases_before_pause + 1U &&
            !service.journal_.active_lease(pause_lease->concurrency_key,
                                           test_time(10)) &&
            paused_projection &&
            paused_projection->desired_state == "paused" &&
            paused_projection->observed_state == "paused",
        "resource pause orders process exit, physical release, logical release, and paused state");

  trainvm::Controller resume_controller(
      *compiled.plan, service.journal_, pause_run_id);
  resume_controller.recover();
  const auto resume_command = resume_controller.request_lifecycle(
      trainvm::LifecycleCommandKind::resume, "service-resume-once",
      paused_projection->run_revision, false, false, "operator",
      "reacquire from the exact immutable checkpoint");
  const auto resume_acquired = service.reconcile_once(pause_run_id);
  const auto replacement_lease = service.journal_.active_lease(
      pause_lease->concurrency_key, test_time(service_now_ns));
  const auto replacement_granted = service.reconcile_once(pause_run_id);
  check(resume_acquired.disposition ==
                trainvm::ReconcileDisposition::lease_acquired &&
            replacement_granted.disposition ==
                trainvm::ReconcileDisposition::host_grant_acquired &&
            replacement_lease &&
            replacement_lease->lease_id != pause_lease->lease_id &&
            replacement_lease->fencing_token > pause_lease->fencing_token,
        "released pause resume reacquires a fresh logical and physical fence");
  if (!replacement_lease) {
    std::filesystem::remove_all(directory);
    return;
  }
  const auto replacement_request = trainvm::build_resource_bundle_request({
      .journal_id = service.journal_.journal_id(),
      .plan_hash = compiled.plan->plan_hash,
      .run_id = pause_run_id,
      .resources = compiled.plan->experiment.spec.resources,
      .lease = *replacement_lease,
  });
  const auto replacement_saga =
      service.journal_.host_grant_saga(replacement_request.request_id);
  trainvm::Controller replacement_worker(
      *compiled.plan, service.journal_, pause_run_id);
  replacement_worker.recover();
  const auto replacement_launch = replacement_worker.prepare_worker_launch(
      {.code_fingerprint = "sha256:" + std::string(64U, 'a'),
       .required_capabilities = {"worker.metrics", "worker.controls"}},
      test_time(service_now_ns));
  const auto replacement_binding = bind_test_worker_launch(
      replacement_worker, replacement_launch, service_now_ns, host_identity);
  (void)service.host_process_saga_->reconcile(
      trainvm::ResolvedLaunch(
          replacement_binding, -1,
          replacement_binding.identity.code ? std::optional<int>{-1}
                                            : std::nullopt,
          -1),
      *replacement_saga->grant,
      trainvm::compile_linux_process_policy(std::nullopt),
      "unix:/tmp/trainvm-process-saga.sock", test_time(service_now_ns));
  prime_test_service_launch(service, replacement_launch);
  trainvm::TrainVMService::WorkerConnection replacement_connection;
  const auto replacement_open = service.open_worker_connection(
      wire_hello(replacement_launch), replacement_connection);
  const auto invocation = nlohmann::json::parse(
      replacement_connection.welcome.canonical_invocation_json());
  check(replacement_open.ok() &&
            replacement_launch.attempt_id == "train_to_boundary@2" &&
            invocation.at("api_version") ==
                trainvm::kWorkerInvocationApiVersion &&
            invocation.at("attempt_id") == replacement_launch.attempt_id &&
            invocation.at("resume").at("checkpoint").at("artifact_id") ==
                checkpoint_id &&
            invocation.at("resume").at("checkpoint")
                    .at("producer_attempt_id") == pause_launch.attempt_id &&
            invocation.at("resume").at("optimizer_step") == 9U &&
            replacement_connection.attempt_baseline_optimizer_step == 9U &&
            replacement_connection.welcome
                    .attempt_baseline_optimizer_step() == 9U &&
            invocation.at("resume").at("pause_command_id") ==
                pause_command.command.command_id &&
            invocation.at("resume").at("resume_command_id") ==
                resume_command.command.command_id,
        "replacement worker welcome is bound to the exact pause checkpoint and command lineage");

  const nlohmann::json replacement_publishes =
      replacement_connection.publishes;
  replacement_connection.publishes["eval_examples"] = {
      {"logical_name", "eval_examples"},
      {"declaration",
       {{"required", true},
        {"type", "eval_examples"},
        {"schema", trainvm::kEvalExamplesSchema}}},
  };
  const auto baseline_heartbeat = [&](std::uint64_t optimizer_step) {
    trainvm::v1::WorkerHeartbeat heartbeat;
    heartbeat.set_worker_sequence(1U);
    heartbeat.set_optimizer_step(optimizer_step);
    heartbeat.set_phase("train");
    heartbeat.mutable_observed_at()->set_seconds(11);
    std::uint64_t acknowledged = 0U;
    const std::size_t before = service.journal_.event_count();
    const grpc::Status status = service.record_worker_heartbeat(
        heartbeat, replacement_connection, acknowledged);
    return std::tuple{status, acknowledged, before,
                      service.journal_.event_count()};
  };
  const auto [backward_status, backward_ack, before_backward,
              after_backward] = baseline_heartbeat(8U);
  const auto [premature_status, premature_ack, before_premature,
              after_premature] = baseline_heartbeat(10U);
  trainvm::v1::MetricSample baseline_metric;
  baseline_metric.set_worker_sequence(1U);
  baseline_metric.set_name("train.loss");
  baseline_metric.mutable_value()->set_number_value(1.0);
  baseline_metric.set_unit("dimensionless");
  baseline_metric.set_step_domain("optimizer_step");
  baseline_metric.set_step(9U);
  baseline_metric.set_sample_weight(1.0);
  baseline_metric.mutable_observed_at()->set_seconds(11);
  std::uint64_t baseline_ack = 0U;
  const grpc::Status baseline_status = service.record_worker_metric(
      baseline_metric, replacement_connection, baseline_ack);
  trainvm::v1::WorkerHeartbeat still_premature;
  still_premature.set_worker_sequence(2U);
  still_premature.set_optimizer_step(10U);
  still_premature.set_phase("train");
  still_premature.mutable_observed_at()->set_seconds(11);
  std::uint64_t still_premature_ack = 0U;
  const std::size_t before_still_premature = service.journal_.event_count();
  const grpc::Status still_premature_status = service.record_worker_heartbeat(
      still_premature, replacement_connection, still_premature_ack);
  const std::size_t after_still_premature = service.journal_.event_count();
  check(backward_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
            backward_ack == 0U && before_backward == after_backward &&
            premature_status.error_code() ==
                grpc::StatusCode::FAILED_PRECONDITION &&
            premature_ack == 0U && before_premature == after_premature &&
            baseline_status.ok() && baseline_ack == 1U &&
            still_premature_status.error_code() ==
                grpc::StatusCode::FAILED_PRECONDITION &&
            still_premature_ack == 0U &&
            before_still_premature == after_still_premature,
        "replacement workers may publish at the immutable baseline but cannot "
        "regress below it or advance beyond it before gate closure");
  replacement_connection.publishes = replacement_publishes;

  const auto publish_resume_child =
      [&](std::uint64_t worker_sequence, std::string artifact_id,
          std::string logical_name, trainvm::v1::ArtifactKind kind,
          std::string schema) {
        trainvm::v1::ArtifactManifest artifact;
        artifact.set_worker_sequence(worker_sequence);
        artifact.set_artifact_id(std::move(artifact_id));
        artifact.set_logical_name(std::move(logical_name));
        artifact.set_kind(kind);
        artifact.set_schema(std::move(schema));
        artifact.set_uri("file:///sealed/resume-finalization/" +
                         artifact.artifact_id());
        artifact.set_size_bytes(4096U);
        artifact.set_fingerprint_algorithm("manifest_sha256");
        artifact.set_fingerprint(std::string(64U, 'e'));
        artifact.set_complete(true);
        artifact.set_optimizer_step(9U);
        artifact.set_producer_node_id(replacement_connection.identity.node_id);
        artifact.set_producer_attempt_id(
            replacement_connection.identity.attempt_id);
        artifact.add_parent_artifact_ids(checkpoint_id);
        artifact.mutable_published_at()->set_seconds(11);
        std::uint64_t acknowledged = 0U;
        return service.record_worker_artifact(
            artifact, replacement_connection, acknowledged);
      };
  const grpc::Status gallery_status = publish_resume_child(
      2U, "service-resume-gallery", "eval_gallery",
      trainvm::v1::ARTIFACT_KIND_IMAGE_GALLERY,
      "rwkv-lab.eval-gallery.v2");

  trainvm::v1::EventEnvelope resumed_completion;
  resumed_completion.set_event_id(
      replacement_connection.dispatch.dispatch_id + ":result");
  resumed_completion.set_run_id(replacement_connection.identity.run_id);
  resumed_completion.set_run_revision(
      replacement_connection.dispatch.run_revision);
  resumed_completion.set_plan_revision(
      replacement_connection.dispatch.plan_revision);
  resumed_completion.set_node_id(replacement_connection.identity.node_id);
  resumed_completion.set_attempt_id(
      replacement_connection.identity.attempt_id);
  resumed_completion.set_worker_sequence(3U);
  resumed_completion.set_event_type("worker.completed");
  resumed_completion.set_event_version(1U);
  resumed_completion.mutable_wall_time()->set_seconds(11);
  resumed_completion.set_monotonic_time_ns(11U);
  resumed_completion.set_optimizer_step(9U);
  resumed_completion.set_canonical_json_payload(
      R"({"reason":"training_complete"})");
  const std::size_t before_missing_report =
      service.journal_.event_count();
  trainvm::v1::WorkerReceipt rejected_receipt;
  const grpc::Status missing_report = service.complete_worker_connection(
      resumed_completion, replacement_connection, rejected_receipt);
  const std::size_t after_missing_report = service.journal_.event_count();

  const grpc::Status report_status = publish_resume_child(
      3U, "service-resume-test-eval", "test_eval",
      trainvm::v1::ARTIFACT_KIND_REPORT,
      "rwkv-lab.hf-test-caption-evidence-bundle.v1");
  resumed_completion.set_worker_sequence(4U);
  trainvm::v1::WorkerReceipt resumed_receipt;
  const std::size_t before_undeclared_terminal =
      service.journal_.event_count();
  const grpc::Status resumed_complete = service.complete_worker_connection(
      resumed_completion, replacement_connection, resumed_receipt);
  const std::size_t after_undeclared_terminal =
      service.journal_.event_count();
  const auto resumed_events = service.journal_.events_for_run(pause_run_id);
  const bool replacement_republished_checkpoint = std::ranges::any_of(
      resumed_events, [&](const trainvm::Event& event) {
        return event.event_type == "artifact.published" &&
               event.attempt_id == replacement_launch.attempt_id &&
               event.payload.value("kind", std::string{}) == "checkpoint";
      });
  check(gallery_status.ok() &&
            missing_report.error_code() ==
                grpc::StatusCode::FAILED_PRECONDITION &&
            before_missing_report == after_missing_report &&
            report_status.ok() &&
            resumed_complete.error_code() ==
                grpc::StatusCode::FAILED_PRECONDITION &&
            before_undeclared_terminal == after_undeclared_terminal &&
            !replacement_republished_checkpoint,
        "resume checkpoint reuse cannot bypass an undeclared terminal training step");
  std::filesystem::remove_all(directory);
}

// The universal pre-mutation gate, proved where it is actually enforced: the
// controller. An adapter-side test can only show that evidence was produced; it
// cannot show that the authority refuses to record a mutation without it, and
// "the artifact exists" passes just as happily when the mutation came first.
// The sentinel here is the journal: a blocked optimizer step must leave zero
// durable events behind, because a step the journal never saw is a step the run
// can never afterwards claim to have taken.
void test_universal_step_zero_gate_orders_controller_mutation() {
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("trainvm-step-zero-gate-test-" +
       std::to_string(static_cast<long long>(::getpid())));
  std::filesystem::remove_all(directory);
  std::filesystem::create_directories(directory);
  const std::filesystem::path run_directory = directory / "run";
  std::filesystem::create_directories(run_directory);

  // The publication declaration is lifted verbatim from the checked-in
  // hf_multimodal_sft recipe rather than restated here, so weakening that
  // recipe disarms this test instead of quietly disarming production.
  nlohmann::json recipes;
  {
    std::ifstream input(std::filesystem::path(TRAINVM_SOURCE_ROOT) /
                        "docs/experiment-vm/examples/"
                        "hf-multimodal-sft.recipe-profiles.v1.json");
    input >> recipes;
  }
  const nlohmann::json declared_eval_examples = recipes.at("recipes")
                                                    .at(0)
                                                    .at("template_document")
                                                    .at("spec")
                                                    .at("artifacts")
                                                    .at("eval_examples");

  nlohmann::json fixture = load_fixture();
  fixture["spec"]["workspace"]["root"] = directory.string();
  fixture["spec"]["workspace"]["run_directory"] = run_directory.string();
  fixture["spec"]["workspace"]["allowed_read_roots"] =
      nlohmann::json::array({directory.string()});
  fixture["spec"]["workspace"]["allowed_write_roots"] =
      nlohmann::json::array({run_directory.string()});
  fixture["spec"]["artifacts"]["checkpoint"]["required"] = true;
  fixture["spec"]["artifacts"]["eval_examples"] = declared_eval_examples;
  fixture["spec"]["workflow"]["nodes"]["train_to_boundary"]["publishes"]
         ["eval_examples"] = "eval_examples";
  fixture["spec"]["workflow"]["nodes"]["resume_training"]["publishes"]
         ["eval_examples"] = "eval_examples";
  // The training composition is the checked-in HF one, resolved against the
  // real component registry: the controller reads the evaluator's metrics and
  // descriptor digest out of it when it validates an eval-examples publication,
  // so a synthetic stand-in would prove nothing about this family.
  nlohmann::json hf_training = recipes.at("recipes")
                                   .at(0)
                                   .at("template_document")
                                   .at("spec")
                                   .at("workflow")
                                   .at("nodes")
                                   .at("train")
                                   .at("invoke")
                                   .at("training");
  // The recipe names one operator's absolute host paths, which do not exist on
  // a CI runner, and path-typed component configuration is resolved for real.
  // Repoint the three of them at this source tree, exactly as
  // recipe_profile_tests.cpp does; nothing here reads their contents.
  const std::filesystem::path source_root =
      std::filesystem::canonical(std::filesystem::path(TRAINVM_SOURCE_ROOT));
  hf_training["components"]["model_loader"]["configuration"]["model_path"] =
      source_root.string();
  hf_training["components"]["data"]["configuration"]["dataset_root"] =
      source_root.string();
  hf_training["components"]["trainability"]["configuration"]
             ["target_manifest_path"] = (source_root / "README.md").string();
  fixture["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
         ["training"] = hf_training;
  fixture["spec"]["workflow"]["nodes"]["resume_training"]["invoke"]
         ["training"] = hf_training;
  // The evaluator's own declared metric identities, so the scalar the
  // controller demands at the baseline is the one the recipe actually names.
  const nlohmann::json evaluator_metrics =
      hf_training.at("components").at("evaluator").at("configuration").at(
          "metrics");
  for (const auto& metric : evaluator_metrics) {
    fixture["spec"]["observability"]["metrics"].push_back(
        {{"name", metric},
         {"type", "gauge"},
         {"unit", "dimensionless"},
         {"step_domain", "optimizer_step"},
         {"aggregation", "last"}});
  }
  const std::string baseline_metric_name =
      evaluator_metrics.at(0).get<std::string>();
  const auto compiled = trainvm::compile_document(fixture);
  check(compiled.valid(), "universal step-zero gate fixture compiles");
  if (!compiled.valid()) {
    std::cerr << trainvm::diagnostics_json(compiled.diagnostics).dump(2) << '\n';
    std::filesystem::remove_all(directory);
    return;
  }

  // The slot map is derived from the recipe rather than transcribed, so a
  // component added to the family cannot silently fall outside the contract.
  std::map<std::string, trainvm::TrainingComponentCategory> hf_slots;
  for (const auto& [slot, component] :
       hf_training.at("components").items()) {
    trainvm::TrainingComponentKey key;
    std::vector<trainvm::Diagnostic> diagnostics;
    if (!trainvm::decode_json(component.at("key"), key, "/", diagnostics) ||
        !diagnostics.empty()) {
      check(false, "HF recipe component key is decodable");
      std::filesystem::remove_all(directory);
      return;
    }
    hf_slots.emplace(slot, key.category);
  }

  auto profiles = fixture_adapter_profiles();
  for (auto& profile : profiles) {
    if (profile.key.operation != "train") continue;
    profile.training_composition = trainvm::TrainingCompositionContract{
        .model_family = "transformer",
        .slots = hf_slots,
        .allowed_components = std::nullopt,
    };
    profile.authoring->outputs.emplace(
        "eval_examples",
        operation_port(trainvm::OperationPortType::artifact, false,
                       trainvm::ArtifactType::eval_examples,
                       "rwkv-lab.eval-examples.v1"));
  }
  const auto components = trainvm::TrainingComponentRegistry::load_file(
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/training-components.v1.json");

  const std::filesystem::path database_path = directory / "gate.db";
  const std::string run_id = "step-zero-gate-run";
  const trainvm::WorkerLaunchRequest launch_request{
      .code_fingerprint = "sha256:" + std::string(64U, '2'),
      .required_capabilities = {"worker.controls", "worker.metrics"},
  };
  trainvm::WorkerLaunchTicket launch;
  {
    trainvm::Journal journal(
        database_path, std::nullopt,
        trainvm::HostGrantEnforcement::legacy_process_free_test);
    trainvm::Controller controller(*compiled.plan, journal, run_id);
    nlohmann::json submission = adapter_locked_submission(
        *compiled.plan, trainvm::AdapterRegistry(profiles));
    const std::string component_manifest =
        components.plan_lock_manifest(*compiled.plan);
    submission["training_component_lock_digest"] =
        "sha256:" + trainvm::sha256_hex(component_manifest);
    submission["training_component_lock"] =
        nlohmann::json::parse(component_manifest);
    controller.create_queued(submission);
    (void)controller.begin_acquisition(test_time(2'000));
    launch = controller.prepare_worker_launch(launch_request, test_time(2'100));
    (void)bind_test_worker_launch(controller, launch, 2'150);
  }
  trainvm::TrainVMService service(
      database_path, trainvm::AdapterRegistry(std::move(profiles)),
      fixture_test_host_launch_registry(*compiled.plan, launch),
      fixture_test_host_identity(), [] { return test_time(2'200); },
      trainvm::HostGrantEnforcement::legacy_process_free_test, components);
  prime_test_service_launch(service, launch);
  trainvm::v1::WorkerHello hello;
  hello.set_run_id(launch.run_id);
  hello.set_node_id(launch.node_id);
  hello.set_attempt_id(launch.attempt_id);
  hello.set_launch_nonce(launch.launch_nonce);
  hello.set_adapter(launch.adapter);
  hello.set_adapter_version(launch.adapter_version);
  hello.set_code_fingerprint(launch.code_fingerprint);
  for (const auto& capability : launch.required_capabilities)
    hello.add_capabilities(capability);
  hello.set_last_acked_controller_sequence(0);
  hello.set_concurrency_key(launch.concurrency_key);
  hello.set_lease_id(launch.lease_id);
  hello.set_fencing_token(launch.fencing_token);
  trainvm::TrainVMService::WorkerConnection connection;
  const grpc::Status open = service.open_worker_connection(hello, connection);
  if (!open.ok())
    std::cerr << "open: " << open.error_message() << '\n';
  check(open.ok() && connection.welcome.step_zero_eval_gate_required() &&
            !connection.welcome.step_zero_eval_gate_satisfied() &&
            connection.attempt_baseline_optimizer_step == 0U,
        "a required eval-examples publication arms the controller gate at the "
        "fresh attempt baseline");
  if (!open.ok()) {
    std::filesystem::remove_all(directory);
    return;
  }

  // The mutation sentinel. Every attempt to record work past the baseline runs
  // through the same commit path a real optimizer step would, and is measured
  // by whether the journal grew.
  const auto mutation_attempt = [&](std::uint64_t worker_sequence,
                                    std::uint64_t step) {
    trainvm::v1::MetricSample metric;
    metric.set_worker_sequence(worker_sequence);
    metric.set_name(baseline_metric_name);
    metric.mutable_value()->set_number_value(0.5);
    metric.set_unit("dimensionless");
    metric.set_step_domain("optimizer_step");
    metric.set_step(step);
    metric.set_sample_weight(1.0);
    metric.mutable_observed_at()->set_seconds(3);
    std::uint64_t acknowledged = 0U;
    const std::size_t before = service.journal_.event_count();
    const grpc::Status status =
        service.record_worker_metric(metric, connection, acknowledged);
    return std::tuple{status, service.journal_.event_count() - before};
  };

  const auto [ungated_status, ungated_growth] = mutation_attempt(1U, 1U);
  check(ungated_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
            ungated_growth == 0U,
        "the controller refuses a step past the baseline before any "
        "attempt-baseline evidence, without mutation");

  const std::string checkpoint_id = "gate-checkpoint-0";
  const std::string checkpoint_digest = "sha256:" + std::string(64U, 'a');
  trainvm::v1::ArtifactManifest checkpoint;
  checkpoint.set_worker_sequence(1U);
  checkpoint.set_artifact_id(checkpoint_id);
  checkpoint.set_logical_name("checkpoint");
  checkpoint.set_kind(trainvm::v1::ARTIFACT_KIND_CHECKPOINT);
  checkpoint.set_schema("rwkv-lab.mageflow-checkpoint.v1");
  checkpoint.set_uri("file:///sealed/gate-checkpoint-0");
  checkpoint.set_size_bytes(4096U);
  checkpoint.set_fingerprint_algorithm("manifest_sha256");
  checkpoint.set_fingerprint(checkpoint_digest);
  checkpoint.set_complete(true);
  checkpoint.set_producer_node_id(connection.identity.node_id);
  checkpoint.set_producer_attempt_id(connection.identity.attempt_id);
  checkpoint.set_optimizer_step(0U);
  checkpoint.mutable_published_at()->set_seconds(3);
  std::uint64_t checkpoint_ack = 0U;
  const grpc::Status checkpoint_status =
      service.record_worker_artifact(checkpoint, connection, checkpoint_ack);
  trainvm::v1::MetricSample baseline_metric;
  baseline_metric.set_worker_sequence(2U);
  baseline_metric.set_name(baseline_metric_name);
  baseline_metric.mutable_value()->set_number_value(9.0);
  baseline_metric.set_unit("dimensionless");
  baseline_metric.set_step_domain("optimizer_step");
  baseline_metric.set_step(0U);
  baseline_metric.set_sample_weight(1.0);
  baseline_metric.mutable_observed_at()->set_seconds(3);
  std::uint64_t baseline_ack = 0U;
  const grpc::Status baseline_status =
      service.record_worker_metric(baseline_metric, connection, baseline_ack);
  const auto [half_status, half_growth] = mutation_attempt(3U, 1U);
  check(checkpoint_status.ok() && baseline_status.ok() &&
            half_status.error_code() == grpc::StatusCode::FAILED_PRECONDITION &&
            half_growth == 0U,
        "a durable baseline checkpoint and scalar are not enough on their own: "
        "the typed examples are still missing and the step stays refused");

  const std::string artifact_id = "gate-eval-examples-0";
  const std::string digest = "sha256:" + std::string(64U, 'b');
  nlohmann::json manifest_body{
      {"api_version", trainvm::kEvalExamplesSchema},
      {"run_id", run_id},
      {"node_id", connection.identity.node_id},
      {"attempt_id", connection.identity.attempt_id},
      {"optimizer_step", 0U},
      {"step_domain", "optimizer_step"},
      {"series_id", "qualitative"},
      {"heldout",
       {{"identity_field", "sample_id"},
        {"identities_digest", digest},
        {"selector_digest", digest}}},
      {"evaluator",
       {{"component_digest",
         nlohmann::json::parse(
             connection.welcome.canonical_invocation_json())
             .at("training")
             .at("components")
             .at("evaluator")
             .at("descriptor_digest")},
        {"metric_names", evaluator_metrics}}},
      {"checkpoint",
       {{"artifact_id", checkpoint_id},
        {"manifest_digest", checkpoint_digest}}},
      {"policy_digest", digest},
      {"examples",
       {{{"example_id", "sample-1"},
         {"heldout_item_id", "row-1"},
         {"heldout_item_digest", digest},
         {"input", {{{"kind", "text"}, {"text", "prompt"}}}},
         {"target", {{{"kind", "text"}, {"text", "target caption"}}}},
         {"prediction",
          {{{"kind", "text"}, {"text", "predicted caption"}}}}}}}};
  manifest_body["canonical_manifest_digest"] =
      "sha256:" + trainvm::sha256_hex(manifest_body.dump());
  const std::string manifest_bytes = manifest_body.dump();
  const std::filesystem::path revision = run_directory / "trainvm_artifacts" /
                                         "eval_examples" / "revisions" /
                                         artifact_id;
  std::filesystem::create_directories(revision);
  {
    std::ofstream output(revision / "manifest.json",
                         std::ios::binary | std::ios::trunc);
    output.write(manifest_bytes.data(),
                 static_cast<std::streamsize>(manifest_bytes.size()));
  }
  trainvm::v1::ArtifactManifest examples;
  // A refused observation is not durable, so the next accepted worker
  // sequence is still 3: the rejections above left no trace to skip past.
  examples.set_worker_sequence(3U);
  examples.set_artifact_id(artifact_id);
  examples.set_logical_name("eval_examples");
  examples.set_kind(trainvm::v1::ARTIFACT_KIND_EVAL_EXAMPLES);
  examples.set_schema(trainvm::kEvalExamplesSchema);
  examples.set_uri("file://" + (revision / "manifest.json").string());
  examples.set_size_bytes(manifest_bytes.size());
  examples.set_fingerprint_algorithm("manifest_sha256");
  examples.set_fingerprint("sha256:" + trainvm::sha256_hex(manifest_bytes));
  examples.set_complete(true);
  examples.set_producer_node_id(connection.identity.node_id);
  examples.set_producer_attempt_id(connection.identity.attempt_id);
  examples.set_optimizer_step(0U);
  examples.add_parent_artifact_ids(checkpoint_id);
  examples.mutable_published_at()->set_seconds(3);
  examples.set_canonical_manifest_json(manifest_bytes);
  std::uint64_t examples_ack = 0U;
  const grpc::Status examples_status =
      service.record_worker_artifact(examples, connection, examples_ack);
  if (!examples_status.ok())
    std::cerr << "examples: " << examples_status.error_message() << '\n';
  const auto [gated_status, gated_growth] = mutation_attempt(4U, 1U);
  check(examples_status.ok() && gated_status.ok() && gated_growth == 1U,
        "the same step is admitted, and only then, once the typed examples are "
        "durable and bound to the same-attempt baseline checkpoint");

  // A resumed attempt is gated at its own baseline. The evidence above is
  // durable in this run, so if the gate were keyed to the run rather than to
  // the attempt, a replacement worker would start already satisfied.
  const std::vector<trainvm::Event> durable =
      service.journal_.events_for_run(run_id);
  check(trainvm::durable_attempt_baseline_eval_gate_satisfied(
            durable, run_id, connection.identity.node_id,
            connection.identity.attempt_id, 0U) &&
            !trainvm::durable_attempt_baseline_eval_gate_satisfied(
                durable, run_id, connection.identity.node_id,
                connection.identity.attempt_id + "-replacement", 0U) &&
            !trainvm::durable_attempt_baseline_eval_gate_satisfied(
                durable, run_id, connection.identity.node_id,
                connection.identity.attempt_id, 1U),
        "durable evidence satisfies only its own attempt at its own baseline");
  std::filesystem::remove_all(directory);
}

}  // namespace

int main() {
  // Each case runs inside its own try so that a throw is attributable. The
  // aggregate previously wrapped all 46 calls in one try, so an uncaught
  // exception printed its message with no indication of which case was in
  // flight -- see the CI run on PR #125, where the only output was
  // "UNCAUGHT: training component configuration path is unavailable".
  struct NamedCase {
    std::string_view name;
    void (*run)();
  };
  static constexpr NamedCase kCases[] = {
      {"reflection_and_compiler", test_reflection_and_compiler},
      {"wire_contract", test_wire_contract},
      {"fsm", test_fsm},
      {"control_validation", test_control_validation},
      {"journal", test_journal},
      {"controller_and_fake_worker", test_controller_and_fake_worker},
      {"compiled_plan_persistence", test_compiled_plan_persistence},
      {"resource_leases", test_resource_leases},
      {"lease_renewal_authority", test_lease_renewal_authority},
      {"authority_clock_integration", test_authority_clock_integration},
      {"authority_lock_file_identity", test_authority_lock_file_identity},
      {"read_only_journal_observer", test_read_only_journal_observer},
      {"control_command_journal", test_control_command_journal},
      {"command_service", test_command_service},
      {"submission_and_queue_boundary", test_submission_and_queue_boundary},
      {"atomic_queue_acquisition_boundary", test_atomic_queue_acquisition_boundary},
      {"worker_launch_and_readiness_boundary", test_worker_launch_and_readiness_boundary},
      {"worker_control_service_boundary", test_worker_control_service_boundary},
      {"service_blocks_674_failed_final_members", test_service_blocks_674_failed_final_members},
      {"worker_control_grpc_stream", test_worker_control_grpc_stream},
      {"graceful_cancel_lifecycle", test_graceful_cancel_lifecycle},
      {"resource_releasing_pause_lifecycle", test_resource_releasing_pause_lifecycle},
      {"adversarial_control_idempotency_and_replay", test_adversarial_control_idempotency_and_replay},
      {"topology_selection_compiles_and_refuses_invalid_combinations", test_topology_selection_compiles_and_refuses_invalid_combinations},
      {"post_training_arm_is_gated_at_compile_time", test_post_training_arm_is_gated_at_compile_time},
      {"checked_in_topology_example_compiles", test_checked_in_topology_example_compiles},
      {"typed_managed_resource_release", test_typed_managed_resource_release},
      {"typed_cache_qualification_executor", test_typed_cache_qualification_executor},
      {"concurrent_worker_launch_and_readiness_replay", test_concurrent_worker_launch_and_readiness_replay},
      {"concurrent_fenced_result_content_conflict", test_concurrent_fenced_result_content_conflict},
      {"concurrent_queue_acquisition_replay", test_concurrent_queue_acquisition_replay},
      {"acquiring_recovery_ignores_mutable_lease_lifecycle", test_acquiring_recovery_ignores_mutable_lease_lifecycle},
      {"acquiring_rejects_fabricated_running_transition", test_acquiring_rejects_fabricated_running_transition},
      {"host_launch_registry_contract", test_host_launch_registry_contract},
      {"host_launch_resolution_and_binding", test_host_launch_resolution_and_binding},
      {"service_host_launch_binding", test_service_host_launch_binding},
      {"service_host_grant_reconciliation", test_service_host_grant_reconciliation},
      {"adapter_registry_file_contract", test_adapter_registry_file_contract},
      {"service_registry_and_reconciliation", test_service_registry_and_reconciliation},
      {"adapter_registry_and_reconciler", test_adapter_registry_and_reconciler},
      {"host_grant_saga", test_host_grant_saga},
      {"service_reconciliation_supervisor", test_service_reconciliation_supervisor},
      {"service_supervisor_settles_after_terminal_worker", test_service_supervisor_settles_after_terminal_worker},
      {"service_supervisor_idle_soak", test_service_supervisor_idle_soak},
      {"service_orders_physical_before_logical_release", test_service_orders_physical_before_logical_release},
      {"legacy_journal_migration_policy", test_legacy_journal_migration_policy},
      {"universal_step_zero_gate_orders_controller_mutation",
       test_universal_step_zero_gate_orders_controller_mutation},
  };

  for (const NamedCase& test_case : kCases) {
    current_case = test_case.name;
    try {
      test_case.run();
    } catch (const std::exception& exception) {
      std::cerr << "UNCAUGHT std::exception in case '" << test_case.name
                << "': " << exception.what() << '\n';
      return 1;
    } catch (...) {
      std::cerr << "UNCAUGHT non-std exception in case '" << test_case.name
                << "'\n";
      return 1;
    }
  }
  current_case = "<none>";

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "all TrainVM tests passed\n";
  return 0;
}
