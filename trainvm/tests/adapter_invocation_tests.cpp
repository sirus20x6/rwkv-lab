#include "trainvm/adapter_invocation.hpp"
#include "trainvm/document.hpp"
#include "trainvm/training_component_registry.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/json.hpp"

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

nlohmann::json load_fixture() {
  const std::filesystem::path path =
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/mageflow-cache-resume.json";
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open fixture");
  return nlohmann::json::parse(input);
}

trainvm::CompiledPlan compiled_fixture() {
  const trainvm::CompileResult compiled =
      trainvm::compile_document(load_fixture());
  if (!compiled.valid() || !compiled.plan)
    throw std::runtime_error("invocation fixture did not compile");
  return *compiled.plan;
}

nlohmann::json artifact_manifest(std::string logical_name, std::string kind,
                                 std::string schema,
                                 std::string fingerprint_algorithm) {
  return {{"artifact_id", "artifact-1"},
          {"logical_name", std::move(logical_name)},
          {"kind", std::move(kind)},
          {"schema", std::move(schema)},
          {"uri", "file:///run/checkpoint"},
          {"size_bytes", 4096U},
          {"fingerprint_algorithm", std::move(fingerprint_algorithm)},
          {"fingerprint", "sha256:" + std::string(64U, 'a')},
          {"complete", true},
          {"producer_node_id", "train_to_boundary"},
          {"producer_attempt_id", "attempt-1"},
          {"parent_artifact_ids", nlohmann::json::array()},
          {"published_at_ns", 1234}};
}

void round_trip_and_resolve_public_inputs() {
  const trainvm::CompiledPlan plan = compiled_fixture();
  trainvm::WorkerInvocationContext context{
      .run_id = "run-1",
      .node_id = "train_to_boundary",
      .attempt_id = "attempt-1",
      .dispatch_id = "run-1:dispatch:train_to_boundary:attempt-1",
      .plan_revision = 1U,
      .host_id = "sha256:" + std::string(64U, 'b'),
      .artifacts = {},
      .effective_controls = {{"learning_rate", 0.000002},
                             {"eval_every", 250}},
      .effective_control_revision = 2U,
      .resolved_training = nullptr,
  };
  const trainvm::WorkerInvocationSpec invocation =
      trainvm::build_worker_invocation(plan, context);
  const std::string encoded =
      trainvm::worker_invocation_canonical_json(invocation);
  const trainvm::WorkerInvocationSpec decoded =
      trainvm::worker_invocation_from_canonical_json(encoded);
  check(decoded == invocation, "invocation round trip is exact");
  check(decoded.host_id == "sha256:" + std::string(64U, 'b'),
        "invocation is bound to the authority host identity");
  // The expected values are the fixture's own declared parameter and workspace,
  // which is what "the binding resolves" means. Copying them out as literals
  // said the same thing only as long as nobody edited the fixture, and made the
  // suite read as though it asserted the deployment host's directory layout.
  const nlohmann::json fixture = load_fixture();
  check(decoded.inputs.at("config") ==
            fixture.at("spec").at("parameters").at("source_config").at("value"),
        "parameter binding resolves into the invocation");
  check(decoded.inputs.at("run_directory") ==
            fixture.at("spec").at("workspace").at("run_directory"),
        "context binding resolves into the invocation");
  check(decoded.controls.at("learning_rate") == 0.000002 &&
            decoded.controls.at("eval_every") == 250 &&
            decoded.effective_control_revision == 2U,
        "applied controls overlay immutable defaults");
  check(decoded.publishes.at("checkpoint").at("logical_name") ==
            "checkpoint",
        "declared publications are carried to the worker");
  check(decoded.execution.is_object(),
        "matching execution policy is carried to the worker");

  nlohmann::json tampered = nlohmann::json::parse(encoded);
  tampered["inputs"]["stop_at_step"] = 9999;
  bool rejected = false;
  try {
    (void)trainvm::worker_invocation_from_canonical_json(tampered.dump());
  } catch (const trainvm::AdapterResolutionError&) {
    rejected = true;
  }
  check(rejected, "tampered invocation is rejected by its digest");
}

void reflected_content_roots_are_frozen_into_invocation() {
  nlohmann::json fixture = load_fixture();
  // A content root has to sit inside the fixture's declared read roots, so it
  // is built from one rather than repeating the deployment host's directories.
  const std::string content_root =
      fixture["spec"]["workspace"]["allowed_read_roots"][0].get<std::string>() +
      "/frozen-pack";
  fixture["spec"]["workspace"]["input_content_roots"] = {
      {{"api_version", "trainvm.input-content-root/v1"},
       {"path", content_root},
       {"kind", "directory"},
       {"file_count", 12U},
       {"total_bytes", 4096U},
       {"tree_sha256", "sha256:" + std::string(64U, '4')}},
  };
  const trainvm::CompileResult compiled = trainvm::compile_document(fixture);
  check(compiled.valid() && compiled.plan,
        "reflected input content root compiles into the immutable plan");
  if (!compiled.valid() || !compiled.plan) return;
  trainvm::WorkerInvocationContext context{
      .run_id = "run-content",
      .node_id = "train_to_boundary",
      .attempt_id = "attempt-content",
      .dispatch_id =
          "run-content:dispatch:train_to_boundary:attempt-content",
      .plan_revision = 1U,
      .host_id = "sha256:" + std::string(64U, 'b'),
      .artifacts = {},
      .effective_controls = nlohmann::json::object(),
      .effective_control_revision = 0U,
      .resolved_training = nullptr,
  };
  const auto invocation =
      trainvm::build_worker_invocation(*compiled.plan, context);
  check(invocation.workspace.at("input_content_roots") ==
            fixture["spec"]["workspace"]["input_content_roots"],
        "worker invocation freezes exact recursive content identities");

  // A doubled leading separator names the same directory by a different string.
  fixture["spec"]["workspace"]["input_content_roots"][0]["path"] =
      "/" + content_root;
  check(!trainvm::compile_document(fixture).valid(),
        "content-root aliases fail canonical plan validation");
  fixture["spec"]["workspace"]["input_content_roots"][0]["path"] =
      "/outside/not-authorized";
  check(!trainvm::compile_document(fixture).valid(),
        "content roots outside allowed reads fail plan compilation");
}

void artifact_and_control_validation_fail_closed() {
  const trainvm::CompiledPlan plan = compiled_fixture();
  trainvm::WorkerInvocationContext context{
      .run_id = "run-1",
      .node_id = "prepare_cache",
      .attempt_id = "attempt-2",
      .dispatch_id = "run-1:dispatch:prepare_cache:attempt-2",
      .plan_revision = 1U,
      .host_id = "sha256:" + std::string(64U, 'c'),
      .artifacts = {{"checkpoint",
                     artifact_manifest(
                         "checkpoint", "checkpoint",
                         "rwkv-lab.mageflow-checkpoint.v1",
                         "manifest_sha256")}},
      .effective_controls = nlohmann::json::object(),
      .effective_control_revision = 0U,
      .resolved_training = nullptr,
  };
  const auto invocation = trainvm::build_worker_invocation(plan, context);
  check(invocation.inputs.at("checkpoint").at("logical_name") ==
            "checkpoint",
        "declared artifact manifest resolves into the invocation");
  check(invocation.controls.empty(),
        "operation invocation excludes unrelated trainer controls");

  context.artifacts.at("checkpoint")["schema"] = "wrong.schema";
  bool rejected_artifact = false;
  try {
    (void)trainvm::build_worker_invocation(plan, context);
  } catch (const trainvm::AdapterResolutionError&) {
    rejected_artifact = true;
  }
  check(rejected_artifact,
        "artifact metadata that disagrees with its declaration is rejected");

  context.node_id = "train_to_boundary";
  context.dispatch_id =
      "run-1:dispatch:train_to_boundary:attempt-2";
  context.artifacts.clear();
  context.effective_controls = {{"learning_rate", 2.0}};
  bool rejected_control = false;
  try {
    (void)trainvm::build_worker_invocation(plan, context);
  } catch (const trainvm::AdapterResolutionError&) {
    rejected_control = true;
  }
  check(rejected_control,
        "out-of-range effective control is rejected before dispatch");
}

void resolved_training_composition_is_frozen_into_invocation() {
  nlohmann::json source = load_fixture();
  source["spec"]["workflow"]["nodes"]["train_to_boundary"]["invoke"]
        ["training"] = {
      {"model_family", "mageflow"},
      {"components",
       {{"activation",
         {{"key",
           {{"category", "activation"},
            {"name", "silu"},
            {"version", "1.0.0"}}},
          {"configuration", nlohmann::json::object()}}}}},
  };
  const auto compiled = trainvm::compile_document(source);
  if (!compiled.valid() || !compiled.plan)
    throw std::runtime_error("training invocation fixture did not compile");
  const trainvm::TrainingComponentRegistry registry({{
      .key = {.category = trainvm::TrainingComponentCategory::activation,
              .name = "silu",
              .version = "1.0.0"},
      .backend = trainvm::TrainingComponentBackend::runtime_builtin,
      .implementation = "runtime.activation.silu",
      .model_families = {"mageflow"},
      .required_capabilities = {},
      .configuration = {},
      .state = {},
      .step_domain = std::nullopt,
      .state_grade = trainvm::TrainingStateGrade::stateless,
      .reference_implementation = true,
  }});
  const auto& declared = *compiled.plan->experiment.spec.workflow.nodes
                              .at("train_to_boundary")
                              .invoke.training;
  trainvm::WorkerInvocationContext context{
      .run_id = "run-training",
      .node_id = "train_to_boundary",
      .attempt_id = "attempt-training",
      .dispatch_id =
          "run-training:dispatch:train_to_boundary:attempt-training",
      .plan_revision = 1U,
      .host_id = "sha256:" + std::string(64U, 'd'),
      .artifacts = {},
      .effective_controls = nlohmann::json::object(),
      .effective_control_revision = 0U,
      .resolved_training = trainvm::resolved_training_composition_json(
          registry.resolve_composition(declared)),
  };
  const auto invocation =
      trainvm::build_worker_invocation(*compiled.plan, context);
  check(invocation.training.at("composition_digest") ==
            context.resolved_training.at("composition_digest") &&
            trainvm::worker_invocation_from_canonical_json(
                trainvm::worker_invocation_canonical_json(invocation)) ==
                invocation,
        "resolved training composition is content-addressed in the worker invocation");
  context.resolved_training = nullptr;
  bool rejected = false;
  try {
    (void)trainvm::build_worker_invocation(*compiled.plan, context);
  } catch (const trainvm::AdapterResolutionError&) {
    rejected = true;
  }
  check(rejected,
        "declared training composition cannot dispatch without authority resolution");
}

void replacement_invocation_carries_exact_resume_authority() {
  const trainvm::CompiledPlan plan = compiled_fixture();
  nlohmann::json checkpoint = artifact_manifest(
      "checkpoint", "checkpoint", "rwkv-lab.mageflow-checkpoint.v1",
      "manifest_sha256");
  trainvm::WorkerInvocationContext context{
      .run_id = "run-resume",
      .node_id = "train_to_boundary",
      .attempt_id = "attempt-2",
      .dispatch_id = "run-resume:dispatch:train_to_boundary:attempt-2",
      .plan_revision = 1U,
      .host_id = "sha256:" + std::string(64U, 'e'),
      .artifacts = {},
      .effective_controls = nlohmann::json::object(),
      .effective_control_revision = 0U,
      .resolved_training = nullptr,
      .resume = {{"api_version", "trainvm.resume-checkpoint/v1"},
                 {"checkpoint", checkpoint},
                 {"optimizer_step", 12U},
                 {"pause_command_id", "pause-1"},
                 {"resume_command_id", "resume-1"}},
  };
  const auto invocation = trainvm::build_worker_invocation(plan, context);
  const auto decoded = trainvm::worker_invocation_from_canonical_json(
      trainvm::worker_invocation_canonical_json(invocation));
  check(decoded.resume == context.resume,
        "replacement invocation carries controller-selected checkpoint authority");

  context.resume["checkpoint"]["producer_attempt_id"] = "attempt-2";
  bool rejected = false;
  try {
    (void)trainvm::build_worker_invocation(plan, context);
  } catch (const trainvm::AdapterResolutionError&) {
    rejected = true;
  }
  check(rejected,
        "replacement invocation rejects a checkpoint from its own attempt");
}

void python_worker_invocation_has_cross_runtime_golden_digest() {
  trainvm::WorkerInvocationSpec invocation;
  invocation.api_version = trainvm::kWorkerInvocationApiVersion;
  invocation.run_id = "run-1";
  invocation.host_id = "sha256:" + std::string(64U, 'b');
  invocation.plan_hash = std::string(64U, 'c');
  invocation.plan_revision = 3U;
  invocation.node_id = "train";
  invocation.attempt_id = "attempt-1";
  invocation.dispatch_id = "dispatch-1";
  invocation.adapter.adapter = "rwkv-lab.mageflow";
  invocation.adapter.version = "1.0.0";
  invocation.adapter.runtime = trainvm::ComponentRuntime::python_worker;
  invocation.adapter.operation = "train";
  invocation.adapter.contract = "rwkv-lab.mageflow.train/v1";
  invocation.workspace = nlohmann::json::object();
  invocation.resources = nlohmann::json::object();
  invocation.inputs = {{"caption", "雪"}};
  invocation.controls = {{"learning_rate", 0.000002}};
  invocation.effective_control_revision = 2U;
  invocation.publishes = nlohmann::json::object();
  invocation.observability = nlohmann::json::object();
  invocation.execution = nullptr;
  invocation.training = nullptr;
  invocation.resume = nullptr;
  invocation.invocation_digest =
      "sha256:c5871f676ccc1e24335e8b8445b195e361c4ba9360b4b5ab8d84feefcce89e89";
  check(trainvm::worker_invocation_from_canonical_json(
            trainvm::worker_invocation_canonical_json(invocation)) == invocation,
        "C++ and Python invocation canonicalization share one golden digest");
}

}  // namespace

int main() {
  try {
    round_trip_and_resolve_public_inputs();
    reflected_content_roots_are_frozen_into_invocation();
    artifact_and_control_validation_fail_closed();
    resolved_training_composition_is_frozen_into_invocation();
    replacement_invocation_carries_exact_resume_authority();
    python_worker_invocation_has_cross_runtime_golden_digest();
  } catch (const std::exception& exception) {
    std::cerr << "UNCAUGHT: " << exception.what() << '\n';
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " adapter invocation test(s) failed\n";
    return 1;
  }
  std::cout << "adapter invocation tests passed\n";
  return 0;
}
