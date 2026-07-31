#include "trainvm/controller.hpp"

#include "trainvm/reflection_json.hpp"

#include <cstddef>
#include <limits>
#include <map>
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
         event_type == "fsm.transitioned" || event_type == "run.desired_state_changed" ||
         event_type == "run.observed_state_changed" ||
         event_type == "node.dispatch_prepared" || event_type == "node.dispatch_completed" ||
         event_type.starts_with("control.");
}

std::string dispatch_id_for(const ExecutionState& state) {
  return state.run_id + ":dispatch:" + state.current_node_id + ":" + state.current_attempt_id;
}

Event dispatch_prepared_event(const Dispatch& dispatch) {
  return Event{
      .event_id = dispatch.dispatch_id + ":prepared",
      .run_id = dispatch.run_id,
      .run_revision = dispatch.run_revision,
      .plan_revision = dispatch.plan_revision,
      .node_id = dispatch.node_id,
      .attempt_id = dispatch.attempt_id,
      .worker_sequence = 0,
      .event_type = "node.dispatch_prepared",
      .event_version = 1,
      .wall_time_ns = 0,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"dispatch_id", dispatch.dispatch_id},
                  {"component", dispatch.component},
                  {"operation", dispatch.operation}},
  };
}

Event dispatch_completed_event(const Dispatch& dispatch, const Event& result,
                               std::uint64_t run_revision) {
  return Event{
      .event_id = dispatch.dispatch_id + ":completed",
      .run_id = dispatch.run_id,
      .run_revision = run_revision,
      .plan_revision = dispatch.plan_revision,
      .node_id = dispatch.node_id,
      .attempt_id = dispatch.attempt_id,
      .worker_sequence = 0,
      .event_type = "node.dispatch_completed",
      .event_version = 1,
      .wall_time_ns = result.wall_time_ns,
      .monotonic_time_ns = result.monotonic_time_ns,
      .optimizer_step = result.optimizer_step,
      .payload = {{"dispatch_id", dispatch.dispatch_id}, {"result_event_id", result.event_id}},
  };
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

std::string_view control_status_name(ControlCommandStatus status) {
  switch (status) {
    case ControlCommandStatus::requested:
      return "requested";
    case ControlCommandStatus::applied:
      return "applied";
    case ControlCommandStatus::rejected:
      return "rejected";
    case ControlCommandStatus::restart_required:
      return "restart_required";
  }
  throw std::runtime_error("invalid control command status");
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
  journal_.create_run(plan_, {created_event(plan_, initial),
                              entered_event(plan_, initial, run_id_ + ":initial-node")});
  state_ = std::move(initial);
  initialized_ = true;
  paused_ = false;
  return state_;
}

const ExecutionState& Controller::recover() {
  const auto persisted_plan = journal_.compiled_plan(plan_.plan_hash);
  if (!persisted_plan) {
    throw std::runtime_error("cannot recover a run whose compiled plan is not persisted");
  }
  if (persisted_plan->canonical_plan != plan_.canonical_plan ||
      persisted_plan->experiment.metadata.name != plan_.experiment.metadata.name) {
    throw std::runtime_error("persisted compiled plan disagrees with controller plan");
  }
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
  std::optional<std::pair<std::string, std::string>> expected_completion;
  std::map<std::string, std::string> replayed_control_status;
  std::string runtime_desired_state = "running";
  std::string runtime_observed_state = "running";
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
    if (event.event_type == "node.dispatch_prepared") {
      if (phase != ReplayPhase::ready || expected_completion ||
          event.run_revision != recovered.revision) {
        throw std::runtime_error("journal contains dispatch preparation outside an active node");
      }
      const std::string expected_id = dispatch_id_for(recovered);
      require_payload_string(event, "dispatch_id", expected_id);
      const auto stored = journal_.dispatch(expected_id);
      const Node& node = plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      if (!stored || stored->run_id != run_id_ || stored->node_id != recovered.current_node_id ||
          stored->attempt_id != recovered.current_attempt_id ||
          stored->component != node.invoke.component || stored->operation != node.invoke.operation) {
        throw std::runtime_error("persisted dispatch disagrees with its preparation event");
      }
      require_payload_string(event, "component", node.invoke.component);
      require_payload_string(event, "operation", node.invoke.operation);
      continue;
    }
    if (event.event_type == "run.desired_state_changed") {
      const auto desired = event.payload.find("state");
      if (desired == event.payload.end() || !desired->is_string()) {
        throw std::runtime_error("run desire is missing its state");
      }
      const std::string desired_value = desired->get<std::string>();
      const bool valid_desire =
          (desired_value == "paused" && runtime_desired_state == "running" &&
           runtime_observed_state == "running") ||
          (desired_value == "running" && runtime_desired_state == "paused" &&
           runtime_observed_state == "paused");
      if (!valid_desire || phase != ReplayPhase::ready || expected_completion ||
          recovered.status != ExecutionStatus::running ||
          event.run_revision != recovered.revision + 1U) {
        throw std::runtime_error("journal contains an invalid pause/resume desire");
      }
      runtime_desired_state = desired_value;
      recovered.revision = event.run_revision;
      continue;
    }
    if (event.event_type == "run.observed_state_changed") {
      const auto observed = event.payload.find("state");
      if (observed == event.payload.end() || !observed->is_string()) {
        throw std::runtime_error("run observation is missing its state");
      }
      const std::string observed_value = observed->get<std::string>();
      if (observed_value == "pausing" || observed_value == "paused" ||
          observed_value == "resuming" || observed_value == "running") {
        const bool valid_observation =
            (observed_value == "pausing" && runtime_desired_state == "paused" &&
             runtime_observed_state == "running") ||
            (observed_value == "paused" && runtime_desired_state == "paused" &&
             runtime_observed_state == "pausing") ||
            (observed_value == "resuming" && runtime_desired_state == "running" &&
             runtime_observed_state == "paused") ||
            (observed_value == "running" && runtime_desired_state == "running" &&
             runtime_observed_state == "resuming");
        if (!valid_observation || phase != ReplayPhase::ready || expected_completion ||
            recovered.status != ExecutionStatus::running ||
            event.run_revision != recovered.revision + 1U ||
            (!event.node_id.empty() && event.node_id != recovered.current_node_id) ||
            (!event.attempt_id.empty() && event.attempt_id != recovered.current_attempt_id)) {
          throw std::runtime_error("journal contains an invalid pause/resume observation");
        }
        runtime_observed_state = observed_value;
        recovered.revision = event.run_revision;
        continue;
      }
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
      expected_completion = std::pair{dispatch_id_for(recovered), pending_cause->event_id};
      recovered = result.state;
      pending_cause.reset();
      phase = recovered.status == ExecutionStatus::running ? ReplayPhase::expecting_entry
                                                           : ReplayPhase::expecting_terminal;
      continue;
    }
    if (event.event_type == "node.dispatch_completed") {
      if ((phase != ReplayPhase::ready && phase != ReplayPhase::terminal) ||
          event.run_revision != recovered.revision || !expected_completion) {
        throw std::runtime_error("journal contains dispatch completion at an invalid boundary");
      }
      const auto dispatch_id = event.payload.find("dispatch_id");
      const auto result_id = event.payload.find("result_event_id");
      if (dispatch_id == event.payload.end() || !dispatch_id->is_string() ||
          result_id == event.payload.end() || !result_id->is_string()) {
        throw std::runtime_error("dispatch completion is missing its receipt identity");
      }
      if (dispatch_id->get<std::string>() != expected_completion->first ||
          result_id->get<std::string>() != expected_completion->second) {
        throw std::runtime_error("dispatch completion does not match the preceding FSM transition");
      }
      const auto stored = journal_.dispatch(dispatch_id->get<std::string>());
      if (!stored || stored->status != DispatchStatus::completed ||
          stored->result_event_id != std::optional<std::string>{result_id->get<std::string>()}) {
        throw std::runtime_error("dispatch completion event disagrees with its durable receipt");
      }
      expected_completion.reset();
      continue;
    }
    if (event.event_type.starts_with("control.")) {
      const auto command_id = event.payload.find("command_id");
      if (command_id == event.payload.end() || !command_id->is_string()) {
        throw std::runtime_error("control journal event is missing command_id");
      }
      const auto command = journal_.control_command(command_id->get<std::string>());
      if (!command || command->run_id != run_id_) {
        throw std::runtime_error("control journal event has no durable command record");
      }
      const auto revision = event.payload.find("control_revision");
      if (revision == event.payload.end() || !revision->is_number_unsigned() ||
          revision->get<std::uint64_t>() != command->control_revision) {
        throw std::runtime_error("control journal event has the wrong control revision");
      }
      const std::string suffix = event.event_type.substr(std::string("control.").size());
      auto previous = replayed_control_status.find(command->command_id);
      if (suffix == "requested") {
        if (previous != replayed_control_status.end()) {
          throw std::runtime_error("control command has more than one request event");
        }
        const nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"idempotency_key", command->idempotency_key},
            {"expected_run_revision", command->expected_run_revision},
            {"expected_control_revision", command->expected_control_revision},
            {"control_revision", command->control_revision},
            {"plan_revision", command->plan_revision},
            {"apply_point", enum_to_string(command->apply_point)},
            {"requires_pause", command->requires_pause},
            {"assignments", command->assignments},
            {"author", command->author},
            {"reason", command->reason},
        };
        if (event.payload != expected_payload ||
            event.run_revision != command->expected_run_revision) {
          throw std::runtime_error("control request event disagrees with its durable command");
        }
      } else if (previous == replayed_control_status.end() || previous->second != "requested") {
        throw std::runtime_error("control acknowledgement has no preceding request event");
      } else {
        if (!command->acknowledgement) {
          throw std::runtime_error("terminal control command has no worker acknowledgement identity");
        }
        const auto& identity = *command->acknowledgement;
        const nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"control_revision", command->control_revision},
            {"apply_point", enum_to_string(command->apply_point)},
            {"status", std::string(control_status_name(command->status))},
            {"concurrency_key", identity.concurrency_key},
            {"lease_id", identity.lease_id},
            {"fencing_token", identity.fencing_token},
            {"effective_values", command->effective_values},
            {"diagnostics", command->diagnostics},
        };
        if (suffix != control_status_name(command->status) || event.payload != expected_payload ||
            event.optimizer_step != command->effective_step || event.node_id != identity.node_id ||
            event.attempt_id != identity.attempt_id ||
            event.worker_sequence != identity.worker_sequence ||
            !command->acknowledged_at_ns ||
            event.wall_time_ns != *command->acknowledged_at_ns) {
          throw std::runtime_error(
              "control acknowledgement event disagrees with its durable command");
        }
      }
      replayed_control_status[command->command_id] = suffix;
      continue;
    }
    if (is_controller_event(event.event_type)) {
      throw std::runtime_error("journal recovery encountered an unsupported controller event");
    }
    if (phase != ReplayPhase::ready || expected_completion ||
        recovered.status != ExecutionStatus::running || runtime_observed_state != "running") {
      throw std::runtime_error("journal contains a causing event outside an active node");
    }
    if (event.run_revision != recovered.revision || event.worker_sequence == 0) {
      throw std::runtime_error("journal contains a causing event with an invalid revision or sequence");
    }
    const auto receipt = journal_.dispatch(dispatch_id_for(recovered));
    if (!receipt || receipt->status != DispatchStatus::completed ||
        receipt->result_event_id != std::optional<std::string>{event.event_id}) {
      throw std::runtime_error("causing event has no matching completed dispatch receipt");
    }
    pending_cause = event;
    phase = ReplayPhase::awaiting_transition;
  }
  if ((phase != ReplayPhase::ready && phase != ReplayPhase::terminal) || expected_completion) {
    throw std::runtime_error("run journal ends in an incomplete controller transaction");
  }
  for (const auto& [command_id, status] : replayed_control_status) {
    const auto command = journal_.control_command(command_id);
    if (!command || status != control_status_name(command->status)) {
      throw std::runtime_error("control command projection disagrees with journal replay");
    }
  }
  const auto projection = journal_.projection(run_id_);
  const std::string expected_observed =
      recovered.status == ExecutionStatus::running
          ? runtime_observed_state
          : std::string(enum_to_string(recovered.status));
  const std::string expected_node =
      recovered.status == ExecutionStatus::running ? recovered.current_node_id : std::string{};
  const std::string expected_attempt =
      recovered.status == ExecutionStatus::running ? recovered.current_attempt_id : std::string{};
  if (!projection || projection->plan_hash != plan_.plan_hash ||
      projection->experiment_name != plan_.experiment.metadata.name ||
      projection->run_revision != recovered.revision ||
      projection->desired_state != runtime_desired_state ||
      projection->observed_state != expected_observed ||
      projection->current_node_id != expected_node ||
      projection->current_attempt_id != expected_attempt) {
    throw std::runtime_error("run projection disagrees with deterministic journal replay");
  }
  state_ = std::move(recovered);
  initialized_ = true;
  paused_ = runtime_observed_state != "running";
  return state_;
}

Dispatch Controller::prepare_dispatch() {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover the run before dispatching work");
  }
  if (state_.status != ExecutionStatus::running || paused_) {
    throw std::logic_error("controller cannot dispatch work for a terminal run");
  }
  const Node& node = plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  Dispatch dispatch{
      .dispatch_id = dispatch_id_for(state_),
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .component = node.invoke.component,
      .operation = node.invoke.operation,
      .status = DispatchStatus::prepared,
      .result_event_id = std::nullopt,
  };
  return journal_.prepare_dispatch(dispatch, dispatch_prepared_event(dispatch));
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
  if (paused_) {
    throw std::logic_error("controller cannot accept worker events while paused");
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
  const auto dispatch = journal_.dispatch(dispatch_id_for(state_));
  if (!dispatch || dispatch->status != DispatchStatus::prepared) {
    throw std::logic_error("worker event has no prepared dispatch for the active attempt");
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
  batch.push_back(dispatch_completed_event(*dispatch, cause, result.state.revision));
  journal_.complete_dispatch(dispatch->dispatch_id, cause.event_id, batch);
  state_ = result.state;
  return state_;
}

ControlPatchValidation Controller::request_controls(
    const std::string& idempotency_key, std::uint64_t expected_run_revision,
    std::uint64_t expected_control_revision, const nlohmann::json& assignments,
    const std::string& author, const std::string& reason) {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover the run before requesting controls");
  }
  if (idempotency_key.empty() || author.empty() || reason.empty()) {
    throw std::invalid_argument("control request idempotency key, author, and reason are required");
  }
  const std::string command_id =
      "control-" + sha256_hex(nlohmann::json({{"run_id", run_id_},
                                               {"idempotency_key", idempotency_key}})
                                  .dump());
  if (const auto existing = journal_.control_command(command_id)) {
    if (existing->run_id != run_id_ || existing->idempotency_key != idempotency_key ||
        existing->expected_run_revision != expected_run_revision ||
        existing->expected_control_revision != expected_control_revision ||
        existing->assignments != assignments || existing->author != author ||
        existing->reason != reason) {
      throw std::invalid_argument("control command idempotency identity has different content");
    }
    return ControlPatchValidation{
        .assignments = existing->assignments,
        .apply_point = existing->apply_point,
        .requires_pause = existing->requires_pause,
        .replayed = true,
        .diagnostics = {},
        .command = *existing,
    };
  }
  const auto projection = journal_.projection(run_id_);
  if (!projection) {
    throw std::runtime_error("initialized controller has no durable run projection");
  }
  const bool run_paused = projection->observed_state == "paused";
  ControlPatchValidation validation = validate_control_patch(plan_, assignments, true, run_paused);
  if (!validation.valid()) {
    return validation;
  }
  ControlCommand command{
      .command_id = command_id,
      .run_id = run_id_,
      .idempotency_key = idempotency_key,
      .expected_run_revision = expected_run_revision,
      .expected_control_revision = expected_control_revision,
      .control_revision = 0,
      .plan_revision = kInitialPlanRevision,
      .apply_point = validation.apply_point,
      .requires_pause = validation.requires_pause,
      .assignments = validation.assignments,
      .author = author,
      .reason = reason,
      .status = ControlCommandStatus::requested,
      .effective_step = std::nullopt,
      .effective_values = nlohmann::json::object(),
      .diagnostics = nlohmann::json::array(),
      .acknowledgement = std::nullopt,
      .acknowledged_at_ns = std::nullopt,
  };
  ControlSubmission submission = journal_.submit_control_command(std::move(command));
  validation.replayed = !submission.inserted;
  validation.command = std::move(submission.command);
  return validation;
}

ControlCommand Controller::acknowledge_controls(
    const std::string& command_id, const ControlAcknowledgementIdentity& identity,
    ControlCommandStatus status,
    std::optional<std::uint64_t> effective_step, nlohmann::json effective_values,
    nlohmann::json diagnostics) {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover before acknowledging controls");
  }
  if (identity.concurrency_key != plan_.experiment.spec.workspace.concurrency_key) {
    throw std::invalid_argument(
        "control acknowledgement lease key differs from the compiled workspace");
  }
  return journal_.acknowledge_control_command(run_id_, command_id, identity, status, effective_step,
                                              std::move(effective_values),
                                              std::move(diagnostics));
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
