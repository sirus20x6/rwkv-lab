#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

enum class ExecutionStatus { running, completed, failed, cancelled };

struct ExecutionState {
  std::string run_id;
  std::string current_node_id;
  std::string current_attempt_id;
  ExecutionStatus status{ExecutionStatus::running};
  std::uint64_t revision{1};
  std::uint64_t transition_count{};
  std::map<std::string, std::uint64_t> visits;
  std::map<std::string, double> loop_progress;

  bool operator==(const ExecutionState&) const = default;
};

struct TransitionResult {
  ExecutionState state;
  std::string source_node_id;
  std::string target;
  std::size_t transition_index{};
};

ExecutionState start_execution(const CompiledPlan& plan, std::string run_id);
ExecutionState restart_execution_attempt(const ExecutionState& state);
TransitionResult advance_execution(const CompiledPlan& plan, const ExecutionState& state,
                                   const Event& event);
ExecutionState replay_execution(const CompiledPlan& plan, std::string run_id,
                                const std::vector<Event>& events);
bool predicate_matches(const nlohmann::json& predicate, const Event& event);
std::optional<nlohmann::json> event_field(const Event& event, std::string_view field);
nlohmann::json execution_state_json(const ExecutionState& state);

}  // namespace trainvm
