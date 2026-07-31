#include "trainvm/document.hpp"

#include <openssl/evp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <queue>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::string_view kApiVersion = "trainvm.rwkv-lab/v1alpha1";
constexpr std::string_view kKind = "Experiment";
const std::set<std::string> kTerminals{"$completed", "$failed", "$cancelled"};
const std::regex kIdentifier("^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$");
const std::regex kEventField("^[a-zA-Z][a-zA-Z0-9_.]*$");

void error(std::vector<Diagnostic>& diagnostics, std::string code, std::string path,
           std::string message) {
  diagnostics.push_back(
      {Diagnostic::Severity::error, std::move(code), std::move(path), std::move(message)});
}

void warning(std::vector<Diagnostic>& diagnostics, std::string code, std::string path,
             std::string message) {
  diagnostics.push_back(
      {Diagnostic::Severity::warning, std::move(code), std::move(path), std::move(message)});
}

bool is_identifier(const std::string& value) {
  return value.size() <= 128U && std::regex_match(value, kIdentifier);
}

void validate_identifier(const std::string& value, const std::string& path,
                         std::vector<Diagnostic>& diagnostics) {
  if (!is_identifier(value)) {
    error(diagnostics, "identifier.invalid", path,
          "must start with a lowercase letter and contain only lowercase letters, digits, '.', '_', or '-'");
  }
}

template <typename Map>
void validate_map_identifiers(const Map& values, const std::string& path,
                              std::vector<Diagnostic>& diagnostics) {
  for (const auto& [name, unused] : values) {
    (void)unused;
    validate_identifier(name, child_path(path, name), diagnostics);
  }
}

bool value_matches_parameter(ParameterType type, const Json& value) {
  switch (type) {
    case ParameterType::string:
    case ParameterType::path:
    case ParameterType::duration:
      return value.is_string();
    case ParameterType::integer:
      return value.is_number_integer();
    case ParameterType::number:
      return value.is_number();
    case ParameterType::boolean:
      return value.is_boolean();
  }
  return false;
}

bool value_matches_control(ControlType type, const Json& value) {
  switch (type) {
    case ControlType::number:
      return value.is_number();
    case ControlType::integer:
      return value.is_number_integer();
    case ControlType::boolean:
      return value.is_boolean();
    case ControlType::string:
    case ControlType::enumeration:
      return value.is_string();
  }
  return false;
}

bool path_within(const std::filesystem::path& child, const std::filesystem::path& root) {
  const auto normalized_child = child.lexically_normal();
  const auto normalized_root = root.lexically_normal();
  auto child_iterator = normalized_child.begin();
  auto root_iterator = normalized_root.begin();
  for (; root_iterator != normalized_root.end(); ++root_iterator, ++child_iterator) {
    if (child_iterator == normalized_child.end() || *child_iterator != *root_iterator) {
      return false;
    }
  }
  return true;
}

void validate_predicate(const Json& predicate, const std::string& path,
                        std::vector<Diagnostic>& diagnostics) {
  if (!predicate.is_object()) {
    error(diagnostics, "predicate.type", path, "predicate must be an object");
    return;
  }
  const bool comparison = predicate.contains("field") || predicate.contains("operator") ||
                          predicate.contains("value");
  const std::size_t compound_count = static_cast<std::size_t>(predicate.contains("all")) +
                                     static_cast<std::size_t>(predicate.contains("any")) +
                                     static_cast<std::size_t>(predicate.contains("not"));
  if (static_cast<std::size_t>(comparison) + compound_count != 1U) {
    error(diagnostics, "predicate.shape", path,
          "predicate must contain exactly one comparison, all, any, or not form");
    return;
  }
  if (comparison) {
    static const std::set<std::string> allowed{"field", "operator", "value"};
    for (auto iterator = predicate.begin(); iterator != predicate.end(); ++iterator) {
      if (!allowed.contains(iterator.key())) {
        error(diagnostics, "field.unknown", child_path(path, iterator.key()), "unknown field");
      }
    }
    if (!predicate.contains("field") || !predicate["field"].is_string() ||
        !std::regex_match(predicate.value("field", ""), kEventField)) {
      error(diagnostics, "predicate.field", child_path(path, "field"),
            "must be a dotted event field name");
    }
    static const std::set<std::string> operators{"eq", "ne", "lt", "le", "gt", "ge",
                                                  "in", "not_in", "exists"};
    if (!predicate.contains("operator") || !predicate["operator"].is_string() ||
        !operators.contains(predicate.value("operator", ""))) {
      error(diagnostics, "predicate.operator", child_path(path, "operator"),
            "unknown predicate operator");
    }
    if (!predicate.contains("value")) {
      error(diagnostics, "field.required", child_path(path, "value"),
            "comparison value is required");
    }
    return;
  }

  const std::string key = predicate.contains("all") ? "all" : predicate.contains("any") ? "any" : "not";
  for (auto iterator = predicate.begin(); iterator != predicate.end(); ++iterator) {
    if (iterator.key() != key) {
      error(diagnostics, "field.unknown", child_path(path, iterator.key()), "unknown field");
    }
  }
  if (key == "not") {
    if (!predicate["not"].is_object()) {
      error(diagnostics, "predicate.not", child_path(path, "not"), "not must contain one predicate");
    } else {
      validate_predicate(predicate["not"], child_path(path, "not"), diagnostics);
    }
    return;
  }
  const Json& children = predicate[key];
  if (!children.is_array() || children.empty()) {
    error(diagnostics, "predicate.children", child_path(path, key),
          key + " must contain at least one predicate");
    return;
  }
  for (std::size_t index = 0; index < children.size(); ++index) {
    validate_predicate(children[index], child_path(child_path(path, key), std::to_string(index)), diagnostics);
  }
}

std::size_t binding_count(const Binding& binding) {
  return static_cast<std::size_t>(binding.literal.has_value()) +
         static_cast<std::size_t>(binding.parameter.has_value()) +
         static_cast<std::size_t>(binding.artifact.has_value()) +
         static_cast<std::size_t>(binding.control.has_value()) +
         static_cast<std::size_t>(binding.context.has_value()) +
         static_cast<std::size_t>(binding.node_output.has_value());
}

void validate_binding(const Binding& binding, const std::string& path, const Spec& spec,
                      std::vector<Diagnostic>& diagnostics) {
  if (binding_count(binding) != 1U) {
    error(diagnostics, "binding.one_of", path, "binding must select exactly one source");
    return;
  }
  if (binding.parameter && !spec.parameters.contains(*binding.parameter)) {
    error(diagnostics, "reference.parameter", child_path(path, "parameter"),
          "unknown parameter: " + *binding.parameter);
  }
  if (binding.artifact && !spec.artifacts.contains(*binding.artifact)) {
    error(diagnostics, "reference.artifact", child_path(path, "artifact"),
          "unknown artifact: " + *binding.artifact);
  }
  if (binding.control && !spec.controls.catalog.contains(*binding.control)) {
    error(diagnostics, "reference.control", child_path(path, "control"),
          "unknown control: " + *binding.control);
  }
  if (binding.context) {
    static const std::set<std::string> contexts{"run_id", "run_directory", "plan_revision",
                                                 "attempt_id", "host_id"};
    if (!contexts.contains(*binding.context)) {
      error(diagnostics, "reference.context", child_path(path, "context"),
            "unknown run context value: " + *binding.context);
    }
  }
  if (binding.node_output) {
    const auto producer = spec.workflow.nodes.find(binding.node_output->node);
    if (producer == spec.workflow.nodes.end()) {
      error(diagnostics, "reference.node", child_path(path, "node_output/node"),
            "unknown producer node: " + binding.node_output->node);
    } else if (!producer->second.publishes ||
               !producer->second.publishes->contains(binding.node_output->name)) {
      error(diagnostics, "reference.node_output", child_path(path, "node_output/name"),
            "producer does not publish output: " + binding.node_output->name);
    }
  }
}

void validate_cycles(const Workflow& workflow, const std::map<std::string, std::vector<std::string>>& graph,
                     std::vector<Diagnostic>& diagnostics) {
  std::map<std::string, int> indices;
  std::map<std::string, int> low_links;
  std::vector<std::string> stack;
  std::set<std::string> on_stack;
  int next_index = 0;

  std::function<void(const std::string&)> visit = [&](const std::string& node_name) {
    indices[node_name] = next_index;
    low_links[node_name] = next_index;
    ++next_index;
    stack.push_back(node_name);
    on_stack.insert(node_name);

    const auto adjacency = graph.find(node_name);
    if (adjacency != graph.end()) {
      for (const auto& target : adjacency->second) {
        if (!indices.contains(target)) {
          visit(target);
          low_links[node_name] = std::min(low_links[node_name], low_links[target]);
        } else if (on_stack.contains(target)) {
          low_links[node_name] = std::min(low_links[node_name], indices[target]);
        }
      }
    }

    if (low_links[node_name] != indices[node_name]) {
      return;
    }
    std::vector<std::string> component;
    while (!stack.empty()) {
      const std::string current = stack.back();
      stack.pop_back();
      on_stack.erase(current);
      component.push_back(current);
      if (current == node_name) {
        break;
      }
    }
    bool cyclic = component.size() > 1U;
    if (!cyclic) {
      const auto edges = graph.find(component.front());
      cyclic = edges != graph.end() &&
               std::find(edges->second.begin(), edges->second.end(), component.front()) != edges->second.end();
    }
    if (!cyclic) {
      return;
    }
    std::vector<std::string> unguarded;
    for (const auto& name : component) {
      if (!workflow.nodes.at(name).loop_guard.has_value()) {
        unguarded.push_back(name);
      }
    }
    if (!unguarded.empty()) {
      std::sort(unguarded.begin(), unguarded.end());
      std::ostringstream names;
      for (std::size_t index = 0; index < unguarded.size(); ++index) {
        names << (index == 0U ? "" : ", ") << unguarded[index];
      }
      error(diagnostics, "workflow.unbounded_cycle", "/spec/workflow/nodes",
            "every node participating in a cycle requires a visit/progress guard; missing: " + names.str());
    }
  };

  for (const auto& [name, unused] : workflow.nodes) {
    (void)unused;
    if (!indices.contains(name)) {
      visit(name);
    }
  }
}

void validate_experiment(const Experiment& experiment, std::vector<Diagnostic>& diagnostics) {
  if (experiment.api_version != kApiVersion) {
    error(diagnostics, "api_version.unsupported", "/api_version",
          "expected " + std::string(kApiVersion));
  }
  if (experiment.kind != kKind) {
    error(diagnostics, "kind.unsupported", "/kind", "expected Experiment");
  }
  validate_identifier(experiment.metadata.name, "/metadata/name", diagnostics);

  const Spec& spec = experiment.spec;
  validate_identifier(spec.workspace.concurrency_key, "/spec/workspace/concurrency_key", diagnostics);
  validate_map_identifiers(spec.parameters, "/spec/parameters", diagnostics);
  validate_map_identifiers(spec.artifacts, "/spec/artifacts", diagnostics);
  validate_map_identifiers(spec.components, "/spec/components", diagnostics);
  validate_map_identifiers(spec.workflow.nodes, "/spec/workflow/nodes", diagnostics);
  validate_map_identifiers(spec.controls.catalog, "/spec/controls/catalog", diagnostics);

  if (spec.workspace.root.empty() || spec.workspace.run_directory.empty()) {
    error(diagnostics, "workspace.path", "/spec/workspace", "root and run_directory must not be empty");
  }
  if (spec.workspace.allowed_write_roots &&
      std::none_of(spec.workspace.allowed_write_roots->begin(), spec.workspace.allowed_write_roots->end(),
                   [&](const std::string& root) { return path_within(spec.workspace.run_directory, root); })) {
    error(diagnostics, "workspace.write_policy", "/spec/workspace/run_directory",
          "run_directory is outside every allowed_write_root");
  }

  const auto& accelerators = spec.resources.accelerators;
  if (accelerators.count < 0 || accelerators.count > 256) {
    error(diagnostics, "resources.accelerator_count", "/spec/resources/accelerators/count",
          "accelerator count must be between 0 and 256");
  }
  if (accelerators.vendor == AcceleratorVendor::none && accelerators.count != 0) {
    error(diagnostics, "resources.vendor_count", "/spec/resources/accelerators",
          "vendor 'none' requires count 0");
  }
  if (accelerators.vendor != AcceleratorVendor::none && accelerators.count == 0) {
    warning(diagnostics, "resources.vendor_count", "/spec/resources/accelerators",
            "a concrete accelerator vendor with count 0 is unusual");
  }
  if (accelerators.minimum_memory_gib && *accelerators.minimum_memory_gib < 0.0) {
    error(diagnostics, "number.minimum", "/spec/resources/accelerators/minimum_memory_gib",
          "must be nonnegative");
  }
  if (spec.resources.cpu_threads && *spec.resources.cpu_threads < 1) {
    error(diagnostics, "number.minimum", "/spec/resources/cpu_threads", "must be at least 1");
  }
  if (spec.resources.lease_timeout_seconds && *spec.resources.lease_timeout_seconds < 5) {
    error(diagnostics, "number.minimum", "/spec/resources/lease_timeout_seconds",
          "must be at least 5 seconds");
  }

  for (const auto& [name, parameter] : spec.parameters) {
    if (!value_matches_parameter(parameter.type, parameter.value)) {
      error(diagnostics, "parameter.value_type", "/spec/parameters/" + name + "/value",
            "value does not match declared parameter type");
    }
  }

  if (spec.components.empty()) {
    error(diagnostics, "components.empty", "/spec/components", "at least one component is required");
  }
  for (const auto& [component_name, component] : spec.components) {
    const std::string component_path = "/spec/components/" + component_name;
    if (component.adapter.empty() || component.version.empty()) {
      error(diagnostics, "component.identity", component_path,
            "adapter and version must not be empty");
    }
    if (component.operations.empty()) {
      error(diagnostics, "component.operations", child_path(component_path, "operations"),
            "at least one operation is required");
    }
    validate_map_identifiers(component.operations, child_path(component_path, "operations"), diagnostics);
    for (const auto& [operation_name, operation] : component.operations) {
      if (operation.contract.empty()) {
        error(diagnostics, "operation.contract",
              child_path(child_path(component_path, "operations"), operation_name),
              "operation contract must not be empty");
      }
    }
  }

  const Workflow& workflow = spec.workflow;
  if (!workflow.nodes.contains(workflow.entrypoint)) {
    error(diagnostics, "workflow.entrypoint", "/spec/workflow/entrypoint",
          "entrypoint does not name a workflow node");
  }
  std::map<std::string, std::vector<std::string>> graph;
  for (const auto& [node_name, node] : workflow.nodes) {
    const std::string node_path = "/spec/workflow/nodes/" + node_name;
    const auto component = spec.components.find(node.invoke.component);
    if (component == spec.components.end()) {
      error(diagnostics, "reference.component", child_path(node_path, "invoke/component"),
            "unknown component: " + node.invoke.component);
    } else if (!component->second.operations.contains(node.invoke.operation)) {
      error(diagnostics, "reference.operation", child_path(node_path, "invoke/operation"),
            "component does not provide operation: " + node.invoke.operation);
    }
    for (const auto& [input_name, binding] : node.invoke.inputs) {
      validate_identifier(input_name, child_path(child_path(node_path, "invoke/inputs"), input_name), diagnostics);
      validate_binding(binding, child_path(child_path(node_path, "invoke/inputs"), input_name), spec, diagnostics);
    }
    if (node.publishes) {
      for (const auto& [output_name, artifact_name] : *node.publishes) {
        validate_identifier(output_name, child_path(child_path(node_path, "publishes"), output_name), diagnostics);
        if (!spec.artifacts.contains(artifact_name)) {
          error(diagnostics, "reference.artifact", child_path(child_path(node_path, "publishes"), output_name),
                "unknown artifact contract: " + artifact_name);
        }
      }
    }
    if (node.transitions.empty()) {
      error(diagnostics, "workflow.transitions", child_path(node_path, "transitions"),
            "node must have at least one transition");
    }
    std::set<std::string> unconditional_events;
    for (std::size_t index = 0; index < node.transitions.size(); ++index) {
      const Transition& transition = node.transitions[index];
      const std::string transition_path = child_path(child_path(node_path, "transitions"), std::to_string(index));
      if (transition.on.empty()) {
        error(diagnostics, "transition.event", child_path(transition_path, "on"),
              "event type must not be empty");
      }
      if (transition.where) {
        validate_predicate(*transition.where, child_path(transition_path, "where"), diagnostics);
        if (unconditional_events.contains(transition.on)) {
          error(diagnostics, "transition.unreachable", transition_path,
                "conditional transition follows an unconditional transition for the same event");
        }
      } else if (!unconditional_events.insert(transition.on).second) {
        error(diagnostics, "transition.ambiguous", transition_path,
              "multiple unconditional transitions handle the same event");
      }
      if (transition.target.starts_with('$')) {
        if (!kTerminals.contains(transition.target)) {
          error(diagnostics, "transition.terminal", child_path(transition_path, "target"),
                "unknown terminal target: " + transition.target);
        }
      } else if (!workflow.nodes.contains(transition.target)) {
        error(diagnostics, "transition.target", child_path(transition_path, "target"),
              "unknown workflow node: " + transition.target);
      } else {
        graph[node_name].push_back(transition.target);
      }
    }
    if (node.retry) {
      if (node.retry->maximum_attempts < 1 || node.retry->maximum_attempts > 1000) {
        error(diagnostics, "retry.attempts", child_path(node_path, "retry/maximum_attempts"),
              "maximum_attempts must be between 1 and 1000");
      }
      if (node.retry->initial_delay_seconds && *node.retry->initial_delay_seconds < 0.0) {
        error(diagnostics, "number.minimum", child_path(node_path, "retry/initial_delay_seconds"),
              "delay must be nonnegative");
      }
    }
    if (node.loop_guard) {
      if (node.loop_guard->max_visits < 2 || node.loop_guard->max_visits > 1'000'000) {
        error(diagnostics, "loop_guard.visits", child_path(node_path, "loop_guard/max_visits"),
              "max_visits must be between 2 and 1000000");
      }
      if (!std::regex_match(node.loop_guard->progress_field, kEventField)) {
        error(diagnostics, "loop_guard.progress_field", child_path(node_path, "loop_guard/progress_field"),
              "must be a dotted event field name");
      }
    }
  }

  if (workflow.nodes.contains(workflow.entrypoint)) {
    std::set<std::string> reachable;
    std::queue<std::string> pending;
    pending.push(workflow.entrypoint);
    while (!pending.empty()) {
      std::string current = pending.front();
      pending.pop();
      if (!reachable.insert(current).second) {
        continue;
      }
      for (const auto& target : graph[current]) {
        pending.push(target);
      }
    }
    for (const auto& [name, unused] : workflow.nodes) {
      (void)unused;
      if (!reachable.contains(name)) {
        error(diagnostics, "workflow.unreachable", "/spec/workflow/nodes/" + name,
              "node is unreachable from the entrypoint");
      }
    }
  }
  validate_cycles(workflow, graph, diagnostics);

  std::map<std::string, std::vector<std::string>> predecessors;
  for (const auto& [source, targets] : graph) {
    for (const auto& target : targets) {
      predecessors[target].push_back(source);
    }
  }
  std::set<std::string> all_publications;
  std::map<std::string, std::set<std::string>> produced_by_node;
  for (const auto& [node_name, node] : workflow.nodes) {
    if (!node.publishes) {
      continue;
    }
    for (const auto& [output_name, artifact_name] : *node.publishes) {
      const std::string artifact_fact = "artifact:" + artifact_name;
      const std::string output_fact = "output:" + node_name + ":" + output_name;
      all_publications.insert(artifact_fact);
      all_publications.insert(output_fact);
      produced_by_node[node_name].insert(artifact_fact);
      produced_by_node[node_name].insert(output_fact);
    }
  }
  // Greatest fixed-point must-analysis: artifacts are durable once published,
  // including across a back-edge. The entrypoint starts with no publications;
  // every other node is narrowed until it contains only facts present on every
  // predecessor path.
  std::map<std::string, std::set<std::string>> available_at_entry;
  for (const auto& [node_name, unused] : workflow.nodes) {
    (void)unused;
    available_at_entry[node_name] = all_publications;
  }
  if (workflow.nodes.contains(workflow.entrypoint)) {
    available_at_entry[workflow.entrypoint].clear();
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& [node_name, unused] : workflow.nodes) {
      (void)unused;
      if (node_name == workflow.entrypoint) {
        continue;
      }
      const auto incoming = predecessors.find(node_name);
      std::set<std::string> next;
      bool first = true;
      if (incoming != predecessors.end()) {
        for (const auto& predecessor : incoming->second) {
          std::set<std::string> predecessor_output = available_at_entry[predecessor];
          predecessor_output.insert(produced_by_node[predecessor].begin(),
                                    produced_by_node[predecessor].end());
          if (first) {
            next = std::move(predecessor_output);
            first = false;
          } else {
            std::set<std::string> intersection;
            std::set_intersection(next.begin(), next.end(), predecessor_output.begin(),
                                  predecessor_output.end(),
                                  std::inserter(intersection, intersection.end()));
            next = std::move(intersection);
          }
        }
      }
      if (first) {
        next.clear();
      }
      if (next != available_at_entry[node_name]) {
        available_at_entry[node_name] = std::move(next);
        changed = true;
      }
    }
  }
  for (const auto& [node_name, node] : workflow.nodes) {
    for (const auto& [input_name, binding] : node.invoke.inputs) {
      const std::string input_path =
          "/spec/workflow/nodes/" + node_name + "/invoke/inputs/" + input_name;
      if (binding.artifact && spec.artifacts.contains(*binding.artifact)) {
        if (!available_at_entry[node_name].contains("artifact:" + *binding.artifact)) {
          error(diagnostics, "artifact.not_available", child_path(input_path, "artifact"),
                "artifact is not published on every path to this node: " + *binding.artifact);
        }
      }
      if (binding.node_output && workflow.nodes.contains(binding.node_output->node)) {
        const std::string output_fact = "output:" + binding.node_output->node + ":" +
                                        binding.node_output->name;
        if (!available_at_entry[node_name].contains(output_fact)) {
          error(diagnostics, "node_output.not_available", child_path(input_path, "node_output"),
                "producer node does not precede this input on every path");
        }
      }
    }
  }

  for (const auto& [name, control] : spec.controls.catalog) {
    const std::string control_path = "/spec/controls/catalog/" + name;
    if (!value_matches_control(control.type, control.default_value)) {
      error(diagnostics, "control.default_type", child_path(control_path, "default"),
            "default does not match declared control type");
    }
    if (control.minimum && control.maximum && *control.minimum > *control.maximum) {
      error(diagnostics, "control.range", control_path, "minimum exceeds maximum");
    }
    if (control.default_value.is_number()) {
      const double value = control.default_value.get<double>();
      if ((control.minimum && value < *control.minimum) || (control.maximum && value > *control.maximum)) {
        error(diagnostics, "control.default_range", child_path(control_path, "default"),
              "default is outside the declared range");
      }
    }
    if (control.type == ControlType::enumeration) {
      if (!control.values || control.values->empty()) {
        error(diagnostics, "control.enum_values", child_path(control_path, "values"),
              "enum control requires at least one value");
      } else if (std::find(control.values->begin(), control.values->end(), control.default_value) ==
                 control.values->end()) {
        error(diagnostics, "control.enum_default", child_path(control_path, "default"),
              "default is not present in values");
      }
    }
    if (control.targets.empty()) {
      error(diagnostics, "control.targets", child_path(control_path, "targets"),
            "control must have at least one target");
    }
    if (!control.mutable_after_start && control.apply != ApplyPoint::restart) {
      warning(diagnostics, "control.immutable", control_path,
              "immutable control is listed in the live-control catalog");
    }
  }

  if (spec.observability.heartbeat_seconds < 1 || spec.observability.heartbeat_seconds > 300) {
    error(diagnostics, "observability.heartbeat", "/spec/observability/heartbeat_seconds",
          "heartbeat must be between 1 and 300 seconds");
  }
  if (spec.observability.retain_raw_metrics_days < 1) {
    error(diagnostics, "observability.retention", "/spec/observability/retain_raw_metrics_days",
          "raw metric retention must be at least one day");
  }
  std::set<std::string> metric_names;
  for (std::size_t index = 0; index < spec.observability.metrics.size(); ++index) {
    const auto& metric = spec.observability.metrics[index];
    const std::string path = "/spec/observability/metrics/" + std::to_string(index);
    if (metric.name.empty() || metric.unit.empty()) {
      error(diagnostics, "metric.identity", path, "metric name and unit must not be empty");
    }
    if (!metric_names.insert(metric.name).second) {
      error(diagnostics, "metric.duplicate", child_path(path, "name"),
            "metric name is declared more than once");
    }
  }
  for (const auto& [path, artifact_name] :
       std::vector<std::pair<std::string, std::optional<std::string>>>{
           {"/spec/observability/eval_gallery_artifact", spec.observability.eval_gallery_artifact},
           {"/spec/observability/log_artifact", spec.observability.log_artifact},
           {"/spec/recovery/checkpoint_artifact", spec.recovery.checkpoint_artifact}}) {
    if (artifact_name && !spec.artifacts.contains(*artifact_name)) {
      error(diagnostics, "reference.artifact", path, "unknown artifact: " + *artifact_name);
    }
  }
  if (spec.recovery.exact_resume && !spec.recovery.checkpoint_artifact) {
    error(diagnostics, "recovery.checkpoint", "/spec/recovery/checkpoint_artifact",
          "exact_resume requires a checkpoint artifact");
  }
  if (spec.recovery.graceful_stop_seconds < 1 || spec.recovery.graceful_stop_seconds > 86'400) {
    error(diagnostics, "recovery.graceful_stop", "/spec/recovery/graceful_stop_seconds",
          "must be between 1 and 86400 seconds");
  }
}

bool has_errors(const std::vector<Diagnostic>& diagnostics) {
  return std::any_of(diagnostics.begin(), diagnostics.end(), [](const Diagnostic& diagnostic) {
    return diagnostic.severity == Diagnostic::Severity::error;
  });
}

nlohmann::json yaml_to_json(const YAML::Node& node) {
  if (!node || node.IsNull()) {
    return nullptr;
  }
  if (node.IsSequence()) {
    nlohmann::json output = nlohmann::json::array();
    for (const auto& child : node) {
      output.push_back(yaml_to_json(child));
    }
    return output;
  }
  if (node.IsMap()) {
    nlohmann::json output = nlohmann::json::object();
    for (const auto& entry : node) {
      if (!entry.first.IsScalar()) {
        throw std::invalid_argument("YAML mapping keys must be scalar strings");
      }
      output[entry.first.Scalar()] = yaml_to_json(entry.second);
    }
    return output;
  }
  if (!node.IsScalar()) {
    throw std::invalid_argument("unsupported YAML node kind");
  }
  const std::string value = node.Scalar();
  if (node.Tag() == "!") {
    return value;
  }
  if (value == "null" || value == "Null" || value == "NULL" || value == "~") {
    return nullptr;
  }
  if (value == "true" || value == "True" || value == "TRUE") {
    return true;
  }
  if (value == "false" || value == "False" || value == "FALSE") {
    return false;
  }
  std::int64_t integer = 0;
  const auto integer_result = std::from_chars(value.data(), value.data() + value.size(), integer);
  if (integer_result.ec == std::errc{} && integer_result.ptr == value.data() + value.size()) {
    return integer;
  }
  char* end = nullptr;
  errno = 0;
  const double number = std::strtod(value.c_str(), &end);
  if (!value.empty() && errno == 0 && end != value.c_str() &&
      end == value.c_str() + value.size() && std::isfinite(number)) {
    return number;
  }
  return value;
}

}  // namespace

bool CompileResult::valid() const {
  return plan.has_value() && !has_errors(diagnostics);
}

std::string sha256_hex(std::string_view value) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), value.data(), value.size()) != 1) {
    throw std::runtime_error("OpenSSL SHA-256 initialization failed");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
    throw std::runtime_error("OpenSSL SHA-256 finalization failed");
  }
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (unsigned int index = 0; index < digest_size; ++index) {
    output << std::setw(2) << static_cast<unsigned int>(digest[index]);
  }
  return output.str();
}

CompileResult compile_document(const nlohmann::json& source) {
  CompileResult result;
  Experiment experiment;
  decode_json(source, experiment, "", result.diagnostics);
  if (has_errors(result.diagnostics)) {
    return result;
  }
  validate_experiment(experiment, result.diagnostics);
  if (has_errors(result.diagnostics)) {
    return result;
  }
  nlohmann::json canonical = encode_json(experiment);
  const std::string canonical_text = canonical.dump();
  result.plan = CompiledPlan{std::move(experiment), std::move(canonical), sha256_hex(canonical_text)};
  return result;
}

CompileResult compile_document_file(const std::filesystem::path& path) {
  CompileResult result;
  std::ifstream input(path);
  if (!input) {
    error(result.diagnostics, "document.open", "", "could not open " + path.string());
    return result;
  }
  try {
    const std::string extension = path.extension().string();
    if (extension == ".yaml" || extension == ".yml") {
      return compile_document(yaml_to_json(YAML::LoadFile(path.string())));
    }
    nlohmann::json source;
    input >> source;
    return compile_document(source);
  } catch (const nlohmann::json::parse_error& exception) {
    error(result.diagnostics, "document.json", "", exception.what());
    return result;
  } catch (const YAML::Exception& exception) {
    error(result.diagnostics, "document.yaml", "", exception.what());
    return result;
  } catch (const std::invalid_argument& exception) {
    error(result.diagnostics, "document.yaml", "", exception.what());
    return result;
  }
}

std::string severity_name(Diagnostic::Severity severity) {
  switch (severity) {
    case Diagnostic::Severity::info:
      return "info";
    case Diagnostic::Severity::warning:
      return "warning";
    case Diagnostic::Severity::error:
      return "error";
  }
  return "error";
}

nlohmann::json diagnostics_json(const std::vector<Diagnostic>& diagnostics) {
  nlohmann::json output = nlohmann::json::array();
  for (const auto& diagnostic : diagnostics) {
    output.push_back({{"severity", severity_name(diagnostic.severity)},
                      {"code", diagnostic.code},
                      {"path", diagnostic.path},
                      {"message", diagnostic.message}});
  }
  return output;
}

nlohmann::json plan_summary(const CompiledPlan& plan) {
  nlohmann::json nodes = nlohmann::json::array();
  for (const auto& [name, node] : plan.experiment.spec.workflow.nodes) {
    nlohmann::json transitions = nlohmann::json::array();
    for (const auto& transition : node.transitions) {
      transitions.push_back({{"on", transition.on}, {"target", transition.target},
                             {"conditional", transition.where.has_value()}});
    }
    nodes.push_back({{"name", name},
                     {"component", node.invoke.component},
                     {"operation", node.invoke.operation},
                     {"effect", enum_to_string(node.effect)},
                     {"idempotency", enum_to_string(node.idempotency)},
                     {"transitions", std::move(transitions)}});
  }
  return {{"api_version", plan.experiment.api_version},
          {"experiment", plan.experiment.metadata.name},
          {"plan_hash", plan.plan_hash},
          {"entrypoint", plan.experiment.spec.workflow.entrypoint},
          {"node_count", plan.experiment.spec.workflow.nodes.size()},
          {"artifact_count", plan.experiment.spec.artifacts.size()},
          {"control_count", plan.experiment.spec.controls.catalog.size()},
          {"nodes", std::move(nodes)}};
}

}  // namespace trainvm
