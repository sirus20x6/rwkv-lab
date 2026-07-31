#include "trainvm/fsm.hpp"

#include "trainvm/reflection_json.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

const nlohmann::json* nested_field(const nlohmann::json& root, std::string_view field) {
  const nlohmann::json* current = &root;
  std::size_t start = 0;
  while (start <= field.size()) {
    const std::size_t separator = field.find('.', start);
    const std::string key(field.substr(start, separator == std::string_view::npos
                                                 ? field.size() - start
                                                 : separator - start));
    if (key.empty() || !current->is_object()) {
      return nullptr;
    }
    const auto iterator = current->find(key);
    if (iterator == current->end()) {
      return nullptr;
    }
    current = &*iterator;
    if (separator == std::string_view::npos) {
      return current;
    }
    start = separator + 1;
  }
  return nullptr;
}

bool ordered_compare(const nlohmann::json& left, const nlohmann::json& right,
                     std::string_view operation) {
  if (left.is_number() && right.is_number()) {
    const double left_number = left.get<double>();
    const double right_number = right.get<double>();
    if (operation == "lt") {
      return left_number < right_number;
    }
    if (operation == "le") {
      return left_number <= right_number;
    }
    if (operation == "gt") {
      return left_number > right_number;
    }
    return left_number >= right_number;
  }
  if (left.is_string() && right.is_string()) {
    const auto& left_string = left.get_ref<const std::string&>();
    const auto& right_string = right.get_ref<const std::string&>();
    if (operation == "lt") {
      return left_string < right_string;
    }
    if (operation == "le") {
      return left_string <= right_string;
    }
    if (operation == "gt") {
      return left_string > right_string;
    }
    return left_string >= right_string;
  }
  throw std::invalid_argument("ordered predicate operands must both be numbers or both be strings");
}

double progress_value(const Event& event, std::string_view field) {
  const auto value = event_field(event, field);
  if (!value || !value->is_number()) {
    throw std::invalid_argument("loop progress field is absent or nonnumeric: " + std::string(field));
  }
  const double number = value->get<double>();
  if (!std::isfinite(number)) {
    throw std::invalid_argument("loop progress field must be finite");
  }
  return number;
}

std::string attempt_id(const std::string& node_id, std::uint64_t visit) {
  return node_id + "@" + std::to_string(visit);
}

}  // namespace

std::optional<nlohmann::json> event_field(const Event& event, std::string_view field) {
  if (field == "step" || field == "optimizer_step") {
    if (!event.optimizer_step) {
      return std::nullopt;
    }
    return *event.optimizer_step;
  }
  if (field == "event_id") {
    return event.event_id;
  }
  if (field == "run_id") {
    return event.run_id;
  }
  if (field == "run_revision") {
    return event.run_revision;
  }
  if (field == "plan_revision") {
    return event.plan_revision;
  }
  if (field == "node_id") {
    return event.node_id;
  }
  if (field == "attempt_id") {
    return event.attempt_id;
  }
  if (field == "worker_sequence") {
    return event.worker_sequence;
  }
  if (field == "event_type") {
    return event.event_type;
  }
  if (field == "event_version") {
    return event.event_version;
  }
  if (field == "wall_time_ns") {
    return event.wall_time_ns;
  }
  if (field == "monotonic_time_ns") {
    return event.monotonic_time_ns;
  }
  if (field.starts_with("payload.")) {
    field.remove_prefix(std::string_view("payload.").size());
  }
  const auto* value = nested_field(event.payload, field);
  if (!value) {
    return std::nullopt;
  }
  return std::optional<nlohmann::json>{*value};
}

bool predicate_matches(const nlohmann::json& predicate, const Event& event) {
  if (predicate.contains("all")) {
    return std::all_of(predicate["all"].begin(), predicate["all"].end(),
                       [&](const nlohmann::json& child) { return predicate_matches(child, event); });
  }
  if (predicate.contains("any")) {
    return std::any_of(predicate["any"].begin(), predicate["any"].end(),
                       [&](const nlohmann::json& child) { return predicate_matches(child, event); });
  }
  if (predicate.contains("not")) {
    return !predicate_matches(predicate["not"], event);
  }

  const std::string field = predicate.at("field").get<std::string>();
  const std::string operation = predicate.at("operator").get<std::string>();
  const auto actual = event_field(event, field);
  if (operation == "exists") {
    return actual.has_value();
  }
  if (!actual) {
    return false;
  }
  const nlohmann::json& expected = predicate.at("value");
  if (operation == "eq") {
    return *actual == expected;
  }
  if (operation == "ne") {
    return *actual != expected;
  }
  if (operation == "lt" || operation == "le" || operation == "gt" || operation == "ge") {
    return ordered_compare(*actual, expected, operation);
  }
  if (operation == "in" || operation == "not_in") {
    if (!expected.is_array()) {
      throw std::invalid_argument("in/not_in predicate value must be an array");
    }
    const bool contained = std::find(expected.begin(), expected.end(), *actual) != expected.end();
    return operation == "in" ? contained : !contained;
  }
  throw std::invalid_argument("unsupported predicate operator: " + operation);
}

ExecutionState start_execution(const CompiledPlan& plan, std::string run_id) {
  if (run_id.empty()) {
    throw std::invalid_argument("run_id must not be empty");
  }
  const std::string& entrypoint = plan.experiment.spec.workflow.entrypoint;
  if (!plan.experiment.spec.workflow.nodes.contains(entrypoint)) {
    throw std::invalid_argument("compiled plan has no entrypoint node");
  }
  ExecutionState state{
      .run_id = std::move(run_id),
      .current_node_id = entrypoint,
      .current_attempt_id = attempt_id(entrypoint, 1),
      .status = ExecutionStatus::running,
      .revision = 1,
      .transition_count = 0,
      .visits = {},
      .loop_progress = {},
  };
  state.visits[entrypoint] = 1;
  return state;
}

TransitionResult advance_execution(const CompiledPlan& plan, const ExecutionState& state,
                                   const Event& event) {
  if (state.status != ExecutionStatus::running) {
    throw std::logic_error("cannot advance a terminal execution");
  }
  if (event.run_id != state.run_id) {
    throw std::invalid_argument("event belongs to a different run");
  }
  if (event.node_id != state.current_node_id) {
    throw std::invalid_argument("event does not belong to the active node");
  }
  if (event.attempt_id != state.current_attempt_id) {
    throw std::invalid_argument("event does not belong to the active node attempt");
  }
  const Node& node = plan.experiment.spec.workflow.nodes.at(state.current_node_id);
  std::vector<std::size_t> conditional_matches;
  std::optional<std::size_t> fallback;
  for (std::size_t index = 0; index < node.transitions.size(); ++index) {
    const Transition& transition = node.transitions[index];
    if (transition.on != event.event_type) {
      continue;
    }
    if (!transition.where) {
      fallback = index;
    } else if (predicate_matches(*transition.where, event)) {
      conditional_matches.push_back(index);
    }
  }
  if (conditional_matches.size() > 1U) {
    throw std::logic_error("event matches more than one conditional transition");
  }
  std::optional<std::size_t> selected;
  if (!conditional_matches.empty()) {
    selected = conditional_matches.front();
  } else {
    selected = fallback;
  }
  if (!selected) {
    throw std::logic_error("active node has no transition for event " + event.event_type);
  }

  const Transition& transition = node.transitions[*selected];
  ExecutionState next = state;
  ++next.revision;
  ++next.transition_count;
  const std::string source = state.current_node_id;
  if (transition.target == "$completed") {
    next.status = ExecutionStatus::completed;
    next.current_node_id.clear();
    next.current_attempt_id.clear();
  } else if (transition.target == "$failed") {
    next.status = ExecutionStatus::failed;
    next.current_node_id.clear();
    next.current_attempt_id.clear();
  } else if (transition.target == "$cancelled") {
    next.status = ExecutionStatus::cancelled;
    next.current_node_id.clear();
    next.current_attempt_id.clear();
  } else {
    const Node& target_node = plan.experiment.spec.workflow.nodes.at(transition.target);
    std::uint64_t& visits = next.visits[transition.target];
    ++visits;
    if (target_node.loop_guard && visits > 1U) {
      if (visits > static_cast<std::uint64_t>(target_node.loop_guard->max_visits)) {
        throw std::logic_error("loop visit limit exceeded for node " + transition.target);
      }
      const double progress = progress_value(event, target_node.loop_guard->progress_field);
      const auto previous = next.loop_progress.find(transition.target);
      if (previous != next.loop_progress.end()) {
        const bool moves = target_node.loop_guard->direction == ProgressDirection::increasing
                               ? progress > previous->second
                               : progress < previous->second;
        if (!moves) {
          throw std::logic_error("loop progress did not move monotonically for node " + transition.target);
        }
      }
      next.loop_progress[transition.target] = progress;
    }
    next.current_node_id = transition.target;
    next.current_attempt_id = attempt_id(transition.target, visits);
  }
  return TransitionResult{.state = std::move(next),
                          .source_node_id = source,
                          .target = transition.target,
                          .transition_index = *selected};
}

ExecutionState replay_execution(const CompiledPlan& plan, std::string run_id,
                                const std::vector<Event>& events) {
  ExecutionState state = start_execution(plan, std::move(run_id));
  for (const auto& event : events) {
    state = advance_execution(plan, state, event).state;
  }
  return state;
}

nlohmann::json execution_state_json(const ExecutionState& state) {
  nlohmann::json progress = nlohmann::json::object();
  for (const auto& [node, value] : state.loop_progress) {
    progress[node] = value;
  }
  return {{"run_id", state.run_id},
          {"current_node_id", state.current_node_id},
          {"current_attempt_id", state.current_attempt_id},
          {"status", enum_to_string(state.status)},
          {"revision", state.revision},
          {"transition_count", state.transition_count},
          {"visits", state.visits},
          {"loop_progress", std::move(progress)}};
}

}  // namespace trainvm
