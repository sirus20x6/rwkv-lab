#include "trainvm/controller.hpp"

#include "trainvm/reflection_json.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::uint64_t kInitialPlanRevision = 1;

bool same_worker_event(const Event& stored, const Event& input) {
  return stored.event_id == input.event_id && stored.run_id == input.run_id &&
         stored.node_id == input.node_id && stored.attempt_id == input.attempt_id &&
         stored.worker_sequence == input.worker_sequence && stored.event_type == input.event_type &&
         stored.event_version == input.event_version && stored.wall_time_ns == input.wall_time_ns &&
         stored.monotonic_time_ns == input.monotonic_time_ns &&
         stored.optimizer_step == input.optimizer_step && stored.payload == input.payload;
}

bool is_controller_event(std::string_view event_type) {
  return event_type == "run.created" || event_type == "node.entered" ||
         event_type == "fsm.transitioned" || event_type == "run.observed_state_changed";
}

Event created_event(const CompiledPlan& plan, const ExecutionState& state) {
  return Event{
      .event_id = state.run_id + ":created",
      .run_id = state.run_id,
      .run_revision = state.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "run.created",
      .event_version = 1,
      .wall_time_ns = 0,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"experiment_name", plan.experiment.metadata.name},
                  {"plan_hash", plan.plan_hash},
                  {"desired_state", "running"},
                  {"observed_state", "running"}},
  };
}

Event entered_event(const CompiledPlan& plan, const ExecutionState& state,
                    const std::string& event_id, const Event* cause = nullptr) {
  const Node& node = plan.experiment.spec.workflow.nodes.at(state.current_node_id);
  return Event{
      .event_id = event_id,
      .run_id = state.run_id,
      .run_revision = state.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state.current_node_id,
      .attempt_id = state.current_attempt_id,
      .worker_sequence = 0,
      .event_type = "node.entered",
      .event_version = 1,
      .wall_time_ns = cause ? cause->wall_time_ns : 0,
      .monotonic_time_ns = cause ? cause->monotonic_time_ns : 0,
      .optimizer_step = cause ? cause->optimizer_step : std::nullopt,
      .payload = {{"component", node.invoke.component}, {"operation", node.invoke.operation}},
  };
}

Event transitioned_event(const Event& cause, const TransitionResult& result) {
  return Event{
      .event_id = cause.event_id + ":transition",
      .run_id = cause.run_id,
      .run_revision = result.state.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = result.source_node_id,
      .attempt_id = cause.attempt_id,
      .worker_sequence = 0,
      .event_type = "fsm.transitioned",
      .event_version = 1,
      .wall_time_ns = cause.wall_time_ns,
      .monotonic_time_ns = cause.monotonic_time_ns,
      .optimizer_step = cause.optimizer_step,
      .payload = {{"cause_event_id", cause.event_id},
                  {"source", result.source_node_id},
                  {"target", result.target},
                  {"transition_index", result.transition_index},
                  {"execution_state", execution_state_json(result.state)}},
  };
}

Event terminal_event(const Event& cause, const ExecutionState& state) {
  return Event{
      .event_id = cause.event_id + ":terminal",
      .run_id = cause.run_id,
      .run_revision = state.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = "run.observed_state_changed",
      .event_version = 1,
      .wall_time_ns = cause.wall_time_ns,
      .monotonic_time_ns = cause.monotonic_time_ns,
      .optimizer_step = cause.optimizer_step,
      .payload = {{"state", enum_to_string(state.status)}},
  };
}

void require_payload_string(const Event& event, std::string_view field, std::string_view expected) {
  const auto found = event.payload.find(field);
  if (found == event.payload.end() || !found->is_string() || found->get<std::string>() != expected) {
    throw std::runtime_error("journal recovery found invalid " + event.event_type + " payload field " +
                             std::string(field));
  }
}

}  // namespace

Controller::Controller(const CompiledPlan& plan, Journal& journal, std::string run_id)
    : plan_(plan), journal_(journal), run_id_(std::move(run_id)) {
  if (run_id_.empty()) {
    throw std::invalid_argument("controller run_id must not be empty");
  }
}

const ExecutionState& Controller::create() {
  if (journal_.projection(run_id_)) {
    return recover();
  }
  ExecutionState initial = start_execution(plan_, run_id_);
  journal_.append_batch({created_event(plan_, initial),
                         entered_event(plan_, initial, run_id_ + ":initial-node")});
  state_ = std::move(initial);
  initialized_ = true;
  return state_;
}

const ExecutionState& Controller::recover() {
  std::string reason;
  if (!journal_.verify_chain(&reason)) {
    throw std::runtime_error("refusing controller recovery: " + reason);
  }
  const std::vector<Event> events = journal_.events_for_run(run_id_);
  if (events.empty()) {
    throw std::runtime_error("cannot recover a run with no journal events");
  }
  if (events.front().event_type != "run.created") {
    throw std::runtime_error("run journal does not begin with run.created");
  }
  require_payload_string(events.front(), "plan_hash", plan_.plan_hash);
  if (events.front().run_revision != 1 || events.front().plan_revision != kInitialPlanRevision) {
    throw std::runtime_error("run.created carries an invalid initial revision");
  }

  ExecutionState recovered = start_execution(plan_, run_id_);
  enum class ReplayPhase { expecting_entry, ready, awaiting_transition, expecting_terminal, terminal };
  ReplayPhase phase = ReplayPhase::expecting_entry;
  std::optional<Event> pending_cause;
  for (const Event& event : events) {
    if (event.plan_revision != kInitialPlanRevision) {
      throw std::runtime_error("journal recovery encountered an unsupported plan revision");
    }
    if (event.event_type == "run.created") {
      if (event.event_id != events.front().event_id) {
        throw std::runtime_error("journal recovery found more than one run.created event");
      }
      continue;
    }
    if (event.event_type == "node.entered") {
      if (phase != ReplayPhase::expecting_entry || recovered.status != ExecutionStatus::running ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id || event.run_revision != recovered.revision) {
        throw std::runtime_error("journal node.entered disagrees with deterministic FSM state");
      }
      const Node& node = plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      require_payload_string(event, "component", node.invoke.component);
      require_payload_string(event, "operation", node.invoke.operation);
      phase = ReplayPhase::ready;
      continue;
    }
    if (event.event_type == "run.observed_state_changed") {
      if (phase != ReplayPhase::expecting_terminal || event.run_revision != recovered.revision) {
        throw std::runtime_error("journal contains an unexpected terminal state observation");
      }
      require_payload_string(event, "state", enum_to_string(recovered.status));
      phase = ReplayPhase::terminal;
      continue;
    }
    if (event.event_type == "fsm.transitioned") {
      if (phase != ReplayPhase::awaiting_transition || !pending_cause) {
        throw std::runtime_error("journal contains an unexpected FSM transition");
      }
      const auto cause_id = event.payload.find("cause_event_id");
      if (cause_id == event.payload.end() || !cause_id->is_string()) {
        throw std::runtime_error("fsm.transitioned is missing cause_event_id");
      }
      if (cause_id->get<std::string>() != pending_cause->event_id) {
        throw std::runtime_error("fsm.transitioned has no preceding causing event");
      }
      const TransitionResult result = advance_execution(plan_, recovered, *pending_cause);
      require_payload_string(event, "source", result.source_node_id);
      require_payload_string(event, "target", result.target);
      const std::size_t transition_index =
          event.payload.value("transition_index", std::numeric_limits<std::size_t>::max());
      if (transition_index != result.transition_index || !event.payload.contains("execution_state") ||
          event.payload.at("execution_state") != execution_state_json(result.state) ||
          event.run_revision != result.state.revision) {
        throw std::runtime_error("persisted FSM transition disagrees with deterministic replay");
      }
      recovered = result.state;
      pending_cause.reset();
      phase = recovered.status == ExecutionStatus::running ? ReplayPhase::expecting_entry
                                                           : ReplayPhase::expecting_terminal;
      continue;
    }
    if (is_controller_event(event.event_type)) {
      throw std::runtime_error("journal recovery encountered an unsupported controller event");
    }
    if (phase != ReplayPhase::ready || recovered.status != ExecutionStatus::running) {
      throw std::runtime_error("journal contains a causing event outside an active node");
    }
    if (event.run_revision != recovered.revision || event.worker_sequence == 0) {
      throw std::runtime_error("journal contains a causing event with an invalid revision or sequence");
    }
    pending_cause = event;
    phase = ReplayPhase::awaiting_transition;
  }
  if (phase != ReplayPhase::ready && phase != ReplayPhase::terminal) {
    throw std::runtime_error("run journal ends in an incomplete controller transaction");
  }
  state_ = std::move(recovered);
  initialized_ = true;
  return state_;
}

const ExecutionState& Controller::handle_event(const Event& input) {
  if (const auto stored = journal_.event(input.event_id)) {
    if (!same_worker_event(*stored, input)) {
      throw std::invalid_argument("event_id already exists with different worker content");
    }
    return recover();
  }
  if (!initialized_) {
    throw std::logic_error("controller must create or recover the run before handling events");
  }
  if (input.run_id != run_id_) {
    throw std::invalid_argument("worker event belongs to a different controller run");
  }
  if (is_controller_event(input.event_type)) {
    throw std::invalid_argument("worker event uses a controller-reserved event type");
  }
  if (input.run_revision != state_.revision || input.plan_revision != kInitialPlanRevision) {
    throw std::invalid_argument("worker event carries a stale run or plan revision");
  }
  if (input.worker_sequence == 0) {
    throw std::invalid_argument("worker event sequence must be nonzero");
  }
  Event cause = input;
  cause.run_revision = state_.revision;
  cause.plan_revision = kInitialPlanRevision;
  const TransitionResult result = advance_execution(plan_, state_, cause);
  std::vector<Event> batch{cause, transitioned_event(cause, result)};
  if (result.state.status == ExecutionStatus::running) {
    batch.push_back(entered_event(plan_, result.state, cause.event_id + ":node-entered", &cause));
  } else {
    batch.push_back(terminal_event(cause, result.state));
  }
  journal_.append_batch(batch);
  state_ = result.state;
  return state_;
}

const ExecutionState& Controller::state() const {
  if (!initialized_) {
    throw std::logic_error("controller has no initialized execution state");
  }
  return state_;
}

const CompiledPlan& Controller::plan() const { return plan_; }

bool Controller::initialized() const { return initialized_; }

}  // namespace trainvm
