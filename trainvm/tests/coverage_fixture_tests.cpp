#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/adapter_registry.hpp"
#include "trainvm/document.hpp"

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

const std::map<std::string, std::string> expected_fixtures{
    {"external-revision-pinned-trainer.json", "external-trainer"},
    {"rwkv-posttraining.json", "rwkv-posttraining"},
    {"rwkv-rlvr.json", "rwkv-rlvr"},
    {"scratch-rwkv-pretrain.json", "rwkv"},
    {"transformer-mla-continuation.json", "transformer-mla"},
    {"vision-distillation-training.json", "vision-distillation"},
};

bool has_forbidden_authority(const nlohmann::json& value,
                             const std::string& path,
                             std::string& reason) {
  static const std::set<std::string> forbidden_keys{
      "arguments",            "argv",               "code_fingerprint",
      "code_path",            "command",            "environment",
      "env",                  "executable",         "executable_fingerprint",
      "executable_path",      "host",               "host_launch",
      "host_launches",        "host_launch_profile", "host_profile",
      "host_profile_digest",  "host_registry_digest", "interpreter",
      "module",               "public_arguments",   "required_capabilities",
      "source_roots",         "trusted_roots",      "working_directory",
  };
  if (value.is_object()) {
    for (auto iterator = value.begin(); iterator != value.end(); ++iterator) {
      const std::string child_path = path + "/" + iterator.key();
      if (forbidden_keys.contains(iterator.key())) {
        reason = child_path + " is a host/executable authority field";
        return true;
      }
      if (has_forbidden_authority(iterator.value(), child_path, reason)) {
        return true;
      }
    }
  } else if (value.is_array()) {
    for (std::size_t index = 0; index < value.size(); ++index) {
      if (has_forbidden_authority(value.at(index),
                                  path + "/" + std::to_string(index),
                                  reason)) {
        return true;
      }
    }
  } else if (value.is_string()) {
    const std::string text = value.get<std::string>();
    static const std::vector<std::string_view> executable_prefixes{
        "#!", "/bin/", "/sbin/", "/usr/bin/", "/usr/local/bin/",
        "python -m ", "python3 -m ", "sh -c ", "bash -c ",
    };
    for (const std::string_view prefix : executable_prefixes) {
      if (text.starts_with(prefix)) {
        reason = path + " embeds executable authority";
        return true;
      }
    }
  }
  return false;
}

bool has_exact_transition(const trainvm::Node& node, std::string_view event,
                          std::string_view target) {
  std::size_t matches = 0;
  for (const trainvm::Transition& transition : node.transitions) {
    if (transition.on == event && transition.target == target &&
        !transition.where.has_value()) {
      ++matches;
    }
  }
  return matches == 1U;
}

bool has_literal_key(const trainvm::Node& node, const std::string& key) {
  const auto input = node.invoke.inputs.find("concurrency_key");
  return input != node.invoke.inputs.end() && input->second.literal.has_value() &&
         *input->second.literal == nlohmann::json(key) &&
         !input->second.parameter.has_value() &&
         !input->second.artifact.has_value() &&
         !input->second.control.has_value() &&
         !input->second.context.has_value() &&
         !input->second.node_output.has_value();
}

void check_core_topology(const trainvm::Experiment& experiment,
                         const std::string& filename) {
  const trainvm::Spec& spec = experiment.spec;
  check(spec.components.size() == 2U,
        filename + ": expected exactly one core and one family component");
  const auto core_iterator = spec.components.find("core");
  check(core_iterator != spec.components.end(),
        filename + ": missing core component");
  if (core_iterator == spec.components.end()) return;

  const trainvm::Component& core = core_iterator->second;
  check(core.adapter == "trainvm.core" && core.version == "1.0.0" &&
            core.runtime == trainvm::ComponentRuntime::builtin &&
            core.operations.size() == 2U &&
            core.operations.contains("acquire_resources") &&
            core.operations.at("acquire_resources").contract ==
                "trainvm.v1.AcquireResources" &&
            core.operations.contains("release_resources") &&
            core.operations.at("release_resources").contract ==
                "trainvm.v1.ReleaseResources",
        filename + ": core component must expose only exact acquire/release contracts");

  std::size_t non_core_components = 0;
  for (const auto& [name, component] : spec.components) {
    if (name == "core") continue;
    ++non_core_components;
    check(component.runtime != trainvm::ComponentRuntime::builtin &&
              component.adapter.starts_with("coverage."),
          filename + ": non-core component must use an unregistered coverage.* adapter");
  }
  check(non_core_components == 1U,
        filename + ": expected exactly one non-core component");

  const trainvm::Workflow& workflow = spec.workflow;
  check(workflow.entrypoint == "acquire_resources",
        filename + ": acquire_resources must be the entrypoint");
  const auto acquire_iterator = workflow.nodes.find("acquire_resources");
  const auto release_iterator = workflow.nodes.find("release_resources");
  check(acquire_iterator != workflow.nodes.end() &&
            release_iterator != workflow.nodes.end(),
        filename + ": missing exact core acquire/release nodes");
  if (acquire_iterator == workflow.nodes.end() ||
      release_iterator == workflow.nodes.end()) {
    return;
  }

  const trainvm::Node& acquire = acquire_iterator->second;
  const trainvm::Node& release = release_iterator->second;
  check(acquire.invoke.component == "core" &&
            acquire.invoke.operation == "acquire_resources" &&
            acquire.effect == trainvm::Effect::resource &&
            acquire.idempotency == trainvm::Idempotency::receipt_required &&
            has_literal_key(acquire, spec.workspace.concurrency_key) &&
            has_exact_transition(acquire, "operation.failed", "$failed"),
        filename + ": acquire node lacks exact core authority topology");
  check(release.invoke.component == "core" &&
            release.invoke.operation == "release_resources" &&
            release.effect == trainvm::Effect::resource &&
            release.idempotency == trainvm::Idempotency::replay_safe &&
            has_literal_key(release, spec.workspace.concurrency_key) &&
            has_exact_transition(release, "resource.released", "$completed") &&
            has_exact_transition(release, "operation.failed", "$failed") &&
            release.transitions.size() == 2U,
        filename + ": release node lacks exact core authority topology");

  std::size_t acquired_routes = 0;
  for (const trainvm::Transition& transition : acquire.transitions) {
    if (transition.on != "resource.acquired") continue;
    ++acquired_routes;
    const auto target = workflow.nodes.find(transition.target);
    check(!transition.where.has_value() && target != workflow.nodes.end() &&
              target->second.invoke.component != "core",
          filename + ": acquired route must enter the family component unconditionally");
  }
  check(acquired_routes == 1U && acquire.transitions.size() == 2U,
        filename + ": acquire must have one family route and one failure route");

  std::size_t completion_routes = 0;
  for (const auto& [node_name, node] : workflow.nodes) {
    for (const trainvm::Transition& transition : node.transitions) {
      if (transition.target != "$completed") continue;
      ++completion_routes;
      check(node_name == "release_resources" &&
                transition.on == "resource.released" &&
                !transition.where.has_value(),
            filename + ": only successful core release may complete the workflow");
    }
  }
  check(completion_routes == 1U,
        filename + ": expected exactly one completion route");
  check(!spec.recovery.exact_resume &&
            !spec.recovery.release_accelerators_when_paused.has_value(),
        filename + ": fixture must remain restart/compatible coverage without a pause authority claim");
}

trainvm::AdapterProfile external_profile(std::string adapter) {
  return trainvm::AdapterProfile{
      .key = trainvm::AdapterKey{
          .adapter = std::move(adapter),
          .version = "1.0.0",
          .runtime = trainvm::ComponentRuntime::python_worker,
          .operation = "train",
          .contract = "test.v1.Train",
      },
      .effect = trainvm::Effect::read_only,
      .idempotency = trainvm::Idempotency::replay_safe,
      .code_fingerprint = "sha256:" + std::string(64U, 'a'),
      .required_capabilities = {},
      .lifecycle = {},
  };
}

void test_reserved_registry_namespace() {
  bool exact_prefix_rejected = false;
  try {
    (void)trainvm::AdapterRegistry(
        {external_profile("coverage.adversarial")});
  } catch (const std::invalid_argument& error) {
    exact_prefix_rejected =
        std::string_view(error.what()).find("compile-only") !=
        std::string_view::npos;
  }

  bool nested_prefix_rejected = false;
  try {
    (void)trainvm::AdapterRegistry(
        {external_profile("coverage.deep.adversarial")});
  } catch (const std::invalid_argument&) {
    nested_prefix_rejected = true;
  }

  bool near_misses_accepted = true;
  try {
    (void)trainvm::AdapterRegistry(
        {external_profile("coveragex.adversarial"),
         external_profile("example.coverage.adversarial")});
  } catch (const std::exception&) {
    near_misses_accepted = false;
  }
  check(exact_prefix_rejected && nested_prefix_rejected &&
            near_misses_accepted,
        "registry reserves exactly the coverage.* namespace for compile-only fixtures");
}

void test_fixture_suite() {
  const std::filesystem::path directory =
      std::filesystem::path(TRAINVM_SOURCE_ROOT) /
      "docs/experiment-vm/examples/coverage";
  std::set<std::string> discovered;
  for (const std::filesystem::directory_entry& entry :
       std::filesystem::directory_iterator(directory)) {
    if (entry.path().extension() != ".json") continue;
    check(!entry.is_symlink() && entry.is_regular_file(),
          entry.path().filename().string() +
              ": coverage fixture must be a regular non-symlink file");
    discovered.insert(entry.path().filename().string());
  }

  std::set<std::string> expected_names;
  for (const auto& [filename, family] : expected_fixtures) {
    (void)family;
    expected_names.insert(filename);
  }
  check(discovered == expected_names,
        "coverage directory must contain exactly the six reviewed JSON fixtures");

  for (const auto& [filename, family] : expected_fixtures) {
    const std::filesystem::path path = directory / filename;
    if (!std::filesystem::is_regular_file(path)) {
      check(false, filename + ": reviewed fixture is missing");
      continue;
    }
    nlohmann::json source;
    try {
      std::ifstream input(path);
      input >> source;
    } catch (const std::exception& error) {
      check(false, filename + ": JSON read failed: " + error.what());
      continue;
    }
    std::string authority_reason;
    check(!has_forbidden_authority(source, "", authority_reason),
          filename + ": " + authority_reason);

    const trainvm::CompileResult result = trainvm::compile_document_file(path);
    check(result.valid() && result.diagnostics.empty(),
          filename + ": fixture must compile with zero diagnostics, including warnings");
    if (!result.plan.has_value()) continue;

    const trainvm::Experiment& experiment = result.plan->experiment;
    check(experiment.metadata.labels.has_value() &&
              experiment.metadata.labels->contains("family") &&
              experiment.metadata.labels->at("family") == family &&
              experiment.metadata.labels->contains("fixture") &&
              experiment.metadata.labels->at("fixture") ==
                  "compiler-coverage",
          filename +
              ": family and fixture labels must match the reviewed manifest");
    check_core_topology(experiment, filename);
  }
}

}  // namespace

int main() {
  try {
    test_reserved_registry_namespace();
    test_fixture_suite();
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "FAIL: uncaught exception: " << error.what() << '\n';
  }
  if (failures != 0) {
    std::cerr << failures << " coverage fixture test(s) failed\n";
    return 1;
  }
  std::cout << "coverage fixture tests passed\n";
  return 0;
}
