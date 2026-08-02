#include "trainvm/controller.hpp"

#include "trainvm/reflection_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
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
         stored.run_revision == input.run_revision &&
         stored.plan_revision == input.plan_revision &&
         stored.node_id == input.node_id && stored.attempt_id == input.attempt_id &&
         stored.worker_sequence == input.worker_sequence && stored.event_type == input.event_type &&
         stored.event_version == input.event_version && stored.wall_time_ns == input.wall_time_ns &&
         stored.monotonic_time_ns == input.monotonic_time_ns &&
         stored.optimizer_step == input.optimizer_step && stored.payload == input.payload;
}

bool is_controller_event(std::string_view event_type) {
  return event_type == "run.created" || event_type == "node.entered" ||
         event_type == "node.attempt_restarted" ||
         event_type == "fsm.transitioned" || event_type == "run.desired_state_changed" ||
         event_type == "run.observed_state_changed" ||
         event_type == "resource.lease_acquired" ||
         event_type == "worker.launch_requested" ||
         event_type == "worker.launch_bound" || event_type == "worker.ready" ||
         event_type == "node.dispatch_prepared" ||
         event_type == "worker.invocation_bound" ||
         event_type == "node.dispatch_completed" ||
         event_type.starts_with("host.resource_") ||
         event_type.starts_with("host.process_") ||
         event_type.starts_with("control.") ||
         event_type.starts_with("checkpoint.") ||
         event_type.starts_with("lifecycle.");
}

bool is_worker_observation(std::string_view event_type) {
  return event_type == "worker.heartbeat" ||
         event_type == "metric.sampled" ||
         event_type == "artifact.published";
}

void require_worker_observation_shape(const Event& event) {
  if (!is_worker_observation(event.event_type) || event.event_version != 1U ||
      event.worker_sequence == 0U || !event.payload.is_object()) {
    throw std::invalid_argument("worker observation is not canonical");
  }
  if (event.event_type == "worker.heartbeat") {
    if (!event.optimizer_step || event.payload.size() != 2U ||
        !event.payload.contains("phase") ||
        !event.payload.at("phase").is_string() ||
        event.payload.at("phase").get_ref<const std::string&>().empty() ||
        !event.payload.contains("observed_at_ns") ||
        !event.payload.at("observed_at_ns").is_number_integer()) {
      throw std::invalid_argument("worker heartbeat is not canonical");
    }
    return;
  }
  if (event.event_type == "metric.sampled") {
    constexpr std::array<std::string_view, 7> fields{
        "name", "value", "unit", "step_domain", "step", "sample_weight",
        "labels"};
    const auto& value = event.payload.contains("value")
                            ? event.payload.at("value")
                            : nlohmann::json{};
    const auto& labels = event.payload.contains("labels")
                             ? event.payload.at("labels")
                             : nlohmann::json{};
    if (!event.optimizer_step || event.payload.size() != fields.size() ||
        !std::ranges::all_of(fields, [&](std::string_view field) {
          return event.payload.contains(std::string(field));
        }) ||
        !event.payload.at("name").is_string() ||
        event.payload.at("name").get_ref<const std::string&>().empty() ||
        !event.payload.at("unit").is_string() ||
        !event.payload.at("step_domain").is_string() ||
        event.payload.at("step_domain").get_ref<const std::string&>().empty() ||
        !event.payload.at("step").is_number_unsigned() ||
        event.payload.at("step").get<std::uint64_t>() != *event.optimizer_step ||
        !event.payload.at("sample_weight").is_number() ||
        !std::isfinite(event.payload.at("sample_weight").get<double>()) ||
        event.payload.at("sample_weight").get<double>() <= 0.0 ||
        !labels.is_object() ||
        !std::ranges::all_of(labels.items(), [](const auto& item) {
          return !item.key().empty() && item.value().is_string();
        }) ||
        !(value.is_number() || value.is_boolean() || value.is_string()) ||
        (value.is_number_float() && !std::isfinite(value.get<double>()))) {
      throw std::invalid_argument("worker metric sample is not canonical");
    }
    return;
  }
  constexpr std::array<std::string_view, 13> fields{
      "artifact_id", "logical_name", "kind", "schema", "uri",
      "size_bytes", "fingerprint_algorithm", "fingerprint", "complete",
      "producer_node_id", "producer_attempt_id", "parent_artifact_ids",
      "published_at_ns"};
  const auto& parents = event.payload.contains("parent_artifact_ids")
                            ? event.payload.at("parent_artifact_ids")
                            : nlohmann::json{};
  const std::string artifact_id =
      event.payload.value("artifact_id", std::string{});
  std::set<std::string> parent_ids;
  const bool canonical_parents = parents.is_array() &&
      std::ranges::all_of(parents, [&](const nlohmann::json& parent) {
        return parent.is_string() && !parent.get_ref<const std::string&>().empty() &&
               parent.get_ref<const std::string&>() != artifact_id &&
               parent_ids.insert(parent.get_ref<const std::string&>()).second;
      });
  const std::string kind = event.payload.value("kind", std::string{});
  if (event.optimizer_step || event.payload.size() != fields.size() ||
      !std::ranges::all_of(fields, [&](std::string_view field) {
        return event.payload.contains(std::string(field));
      }) ||
      !event.payload.at("artifact_id").is_string() ||
      event.payload.at("artifact_id").get_ref<const std::string&>().empty() ||
      !event.payload.at("logical_name").is_string() ||
      !event.payload.at("kind").is_string() ||
      !event.payload.at("schema").is_string() ||
      !event.payload.at("uri").is_string() ||
      event.payload.at("uri").get_ref<const std::string&>().empty() ||
      !event.payload.at("size_bytes").is_number_unsigned() ||
      !event.payload.at("fingerprint_algorithm").is_string() ||
      event.payload.at("fingerprint_algorithm")
          .get_ref<const std::string&>().empty() ||
      !event.payload.at("fingerprint").is_string() ||
      event.payload.at("fingerprint").get_ref<const std::string&>().empty() ||
      !event.payload.at("complete").is_boolean() ||
      !event.payload.at("complete").get<bool>() ||
      !event.payload.at("producer_node_id").is_string() ||
      event.payload.at("producer_node_id").get<std::string>() != event.node_id ||
      !event.payload.at("producer_attempt_id").is_string() ||
      event.payload.at("producer_attempt_id").get<std::string>() !=
          event.attempt_id ||
      !canonical_parents ||
      (kind != "path" && kind != "checkpoint" && kind != "dataset" &&
       kind != "image_gallery" && kind != "metrics" && kind != "report" &&
       kind != "opaque") ||
      !event.payload.at("published_at_ns").is_number_integer() ||
      event.payload.at("published_at_ns").get<std::int64_t>() !=
          event.wall_time_ns) {
    throw std::invalid_argument("worker artifact manifest is not canonical");
  }
}

std::string worker_observation_event_id(const Event& event,
                                        const std::string& dispatch_id) {
  if (event.event_type == "worker.heartbeat") {
    return dispatch_id + ":heartbeat:" +
           std::to_string(event.worker_sequence);
  }
  if (event.event_type == "metric.sampled") {
    return dispatch_id + ":metric:" + std::to_string(event.worker_sequence);
  }
  return dispatch_id + ":artifact:" + sha256_hex(
      event.payload.value("artifact_id", std::string{}));
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

Event created_event(const CompiledPlan& plan, const ExecutionState& state,
                    std::string_view desired_state = "running",
                    std::string_view observed_state = "running",
                    nlohmann::json submission = nlohmann::json::object()) {
  Event event{
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
                  {"desired_state", desired_state},
                  {"observed_state", observed_state}},
  };
  if (!submission.empty()) {
    if (!submission.is_object()) {
      throw std::invalid_argument("run submission identity must be an object");
    }
    event.payload["submission"] = std::move(submission);
  }
  return event;
}

Event acquisition_event(const ExecutionState& state, std::string event_id,
                        std::uint64_t revision, std::string event_type,
                        nlohmann::json payload, std::int64_t now_ns) {
  return Event{
      .event_id = std::move(event_id),
      .run_id = state.run_id,
      .run_revision = revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .worker_sequence = 0,
      .event_type = std::move(event_type),
      .event_version = 1,
      .wall_time_ns = now_ns,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = std::move(payload),
  };
}

std::vector<std::string> canonical_capabilities(
    std::vector<std::string> capabilities) {
  if (capabilities.size() > 256U ||
      std::ranges::any_of(capabilities, [](const std::string& capability) {
        return capability.empty() || capability.size() > 256U;
      })) {
    throw std::invalid_argument("worker capabilities contain an invalid entry");
  }
  std::ranges::sort(capabilities);
  if (std::ranges::adjacent_find(capabilities) != capabilities.end()) {
    throw std::invalid_argument("worker capabilities must be unique");
  }
  return capabilities;
}

std::string worker_launch_event_id(const ExecutionState& state) {
  return state.run_id + ":worker-launch:" + state.current_node_id + ":" +
         state.current_attempt_id;
}

std::string worker_launch_nonce(const CompiledPlan& plan,
                                const WorkerLaunchTicket& launch) {
  nlohmann::json identity{{"run_id", launch.run_id},
                          {"plan_hash", plan.plan_hash},
                          {"node_id", launch.node_id},
                          {"attempt_id", launch.attempt_id},
                          {"adapter", launch.adapter},
                          {"adapter_version", launch.adapter_version},
                          {"code_fingerprint", launch.code_fingerprint},
                          {"required_capabilities", launch.required_capabilities},
                          {"concurrency_key", launch.concurrency_key},
                          {"lease_id", launch.lease_id},
                          {"fencing_token", launch.fencing_token}};
  if (launch.host_grant) {
    identity["host_grant"] = encode_json(*launch.host_grant);
  }
  return sha256_hex(identity.dump());
}

nlohmann::json worker_launch_payload(const WorkerLaunchTicket& launch) {
  nlohmann::json payload{{"launch_nonce", launch.launch_nonce},
                         {"adapter", launch.adapter},
                         {"adapter_version", launch.adapter_version},
                         {"code_fingerprint", launch.code_fingerprint},
                         {"required_capabilities", launch.required_capabilities},
                         {"concurrency_key", launch.concurrency_key},
                         {"lease_id", launch.lease_id},
                         {"fencing_token", launch.fencing_token}};
  if (launch.host_grant) {
    payload["host_grant"] = encode_json(*launch.host_grant);
  }
  return payload;
}

WorkerLaunchTicket launch_from_event(const Event& event) {
  try {
    WorkerLaunchTicket launch{
        .run_id = event.run_id,
        .node_id = event.node_id,
        .attempt_id = event.attempt_id,
        .launch_nonce = event.payload.at("launch_nonce").get<std::string>(),
        .adapter = event.payload.at("adapter").get<std::string>(),
        .adapter_version = event.payload.at("adapter_version").get<std::string>(),
        .code_fingerprint = event.payload.at("code_fingerprint").get<std::string>(),
        .required_capabilities = canonical_capabilities(
            event.payload.at("required_capabilities")
                .get<std::vector<std::string>>()),
        .concurrency_key =
            event.payload.at("concurrency_key").get<std::string>(),
        .lease_id = event.payload.at("lease_id").get<std::string>(),
        .fencing_token = event.payload.at("fencing_token").get<std::uint64_t>(),
        .host_grant = std::nullopt,
    };
    if (event.payload.contains("host_grant")) {
      HostLaunchGrantClaim claim;
      std::vector<Diagnostic> diagnostics;
      if (!decode_json(event.payload.at("host_grant"), claim, "host_grant",
                       diagnostics)) {
        throw std::runtime_error("worker launch host grant is malformed");
      }
      launch.host_grant = std::move(claim);
    }
    return launch;
  } catch (const nlohmann::json::exception& exception) {
    throw std::runtime_error(std::string("worker launch event is malformed: ") +
                             exception.what());
  }
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
      .payload = {{"component", node.invoke.component},
                  {"operation", node.invoke.operation}},
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

std::string_view checkpoint_status_name(CheckpointCommandStatus status) {
  switch (status) {
    case CheckpointCommandStatus::requested:
      return "requested";
    case CheckpointCommandStatus::applied:
      return "applied";
    case CheckpointCommandStatus::rejected:
      return "rejected";
  }
  throw std::invalid_argument("invalid checkpoint command status");
}

std::string_view lifecycle_status_name(LifecycleCommandStatus status) {
  switch (status) {
    case LifecycleCommandStatus::requested:
      return "requested";
    case LifecycleCommandStatus::applied:
      return "applied";
    case LifecycleCommandStatus::rejected:
      return "rejected";
  }
  throw std::invalid_argument("invalid lifecycle command status");
}

std::string_view lifecycle_kind_name(LifecycleCommandKind kind) {
  switch (kind) {
    case LifecycleCommandKind::pause:
      return "pause";
    case LifecycleCommandKind::resume:
      return "resume";
    case LifecycleCommandKind::cancel:
      return "cancel";
  }
  throw std::invalid_argument("invalid lifecycle command kind");
}

}  // namespace

Controller::Controller(const CompiledPlan& plan, Journal& journal, std::string run_id)
    : plan_(plan), journal_(journal), run_id_(std::move(run_id)) {
  if (run_id_.empty()) {
    throw std::invalid_argument("controller run_id must not be empty");
  }
}

const ExecutionState& Controller::create() {
  ExecutionState initial = start_execution(plan_, run_id_);
  const RunCreationResult creation =
      journal_.create_run(plan_, {created_event(plan_, initial),
                                  entered_event(plan_, initial, run_id_ + ":initial-node")});
  if (creation.disposition == RunCreationDisposition::replayed) {
    return recover();
  }
  state_ = std::move(initial);
  initialized_ = true;
  paused_ = false;
  return state_;
}

const ExecutionState& Controller::create_queued(nlohmann::json submission) {
  ExecutionState initial = start_execution(plan_, run_id_);
  const RunCreationResult creation = journal_.create_run(
      plan_, {created_event(plan_, initial, "queued", "queued", std::move(submission))});
  if (creation.disposition == RunCreationDisposition::replayed) {
    return recover();
  }
  state_ = std::move(initial);
  initialized_ = true;
  paused_ = true;
  return state_;
}

const ExecutionState& Controller::recover() {
  [[maybe_unused]] auto snapshot = journal_.read_snapshot();
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
  const bool managed_run = std::ranges::any_of(events, [](const Event& event) {
    return event.event_type == "resource.lease_acquired";
  });
  require_payload_string(events.front(), "plan_hash", plan_.plan_hash);
  if (events.front().run_revision != 1 || events.front().plan_revision != kInitialPlanRevision) {
    throw std::runtime_error("run.created carries an invalid initial revision");
  }

  ExecutionState recovered = start_execution(plan_, run_id_);
  enum class ReplayPhase {
    queued,
    acquisition_requested,
    lease_acquired,
    acquiring,
    launch_requested,
    launch_bound,
    worker_ready,
    expecting_entry,
    ready,
    awaiting_transition,
    expecting_terminal,
    terminal
  };
  const auto initial_desired = events.front().payload.find("desired_state");
  const auto initial_observed = events.front().payload.find("observed_state");
  if (initial_desired == events.front().payload.end() || !initial_desired->is_string() ||
      initial_observed == events.front().payload.end() || !initial_observed->is_string()) {
    throw std::runtime_error("run.created is missing its lifecycle state");
  }
  std::string runtime_desired_state = initial_desired->get<std::string>();
  std::string runtime_observed_state = initial_observed->get<std::string>();
  const bool initially_queued =
      runtime_desired_state == "queued" && runtime_observed_state == "queued";
  if (!initially_queued &&
      (runtime_desired_state != "running" || runtime_observed_state != "running")) {
    throw std::runtime_error("run.created carries an unsupported lifecycle state");
  }
  ReplayPhase phase = initially_queued ? ReplayPhase::queued : ReplayPhase::expecting_entry;
  std::optional<Event> pending_cause;
  bool pending_builtin_admission = false;
  std::optional<std::string> expected_worker_entry_id;
  std::optional<std::pair<std::string, std::string>> expected_completion;
  std::optional<std::string> expected_reacquisition_cause_id;
  std::map<std::string, std::string> replayed_control_status;
  std::map<std::string, std::string> replayed_checkpoint_status;
  std::map<std::string, std::string> replayed_lifecycle_status;
  std::optional<std::string> released_resume_command_id;
  std::optional<std::string> released_resume_pause_command_id;
  std::set<std::string> replayed_process_prepares;
  std::set<std::string> replayed_process_commits;
  std::set<std::string> replayed_process_exits;
  bool current_dispatch_prepared = false;
  std::optional<Event> current_lease_acquisition;
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
    if (event.event_type.starts_with("host.resource_")) {
      if ((event.event_type != "host.resource_request_recorded" &&
           event.event_type != "host.resource_grant_recorded" &&
           event.event_type != "host.resource_busy_recorded" &&
           event.event_type != "host.resource_release_intent_recorded" &&
           event.event_type != "host.resource_release_receipt_recorded") ||
          event.event_version != 1U || event.worker_sequence != 0U ||
          !event.node_id.empty() || !event.attempt_id.empty() ||
          event.run_revision != recovered.revision) {
        throw std::runtime_error(
            "journal recovery found malformed host saga authority evidence");
      }
      continue;
    }
    if (event.event_type.starts_with("host.process_")) {
      const bool prepared = event.event_type == "host.process_prepared";
      const bool committed = event.event_type == "host.process_committed";
      const bool exited = event.event_type == "host.process_exited";
      std::string launch_id;
      if (exited && event.payload.contains("exit_command")) {
        launch_id = hostd_process_exit_from_canonical_json(
                        event.payload.at("exit_command").dump())
                        .launch_id;
      } else {
        launch_id = run_id_ + ":worker-launch:" +
                    recovered.current_node_id + ":" +
                    recovered.current_attempt_id;
      }
      const auto saga = journal_.host_process_saga(launch_id);
      if ((!prepared && !committed && !exited) || !saga ||
          ((!exited && phase != ReplayPhase::launch_bound)) ||
          event.event_version != 1U ||
          event.worker_sequence != 0U ||
          event.run_revision != recovered.revision ||
          saga->prepare.launch.identity.launch_event_id != launch_id ||
          saga->prepare.launch.identity.run_id != run_id_ ||
          event.node_id != saga->prepare.launch.identity.node_id ||
          event.attempt_id != saga->prepare.launch.identity.attempt_id ||
          (!exited &&
           (event.node_id != recovered.current_node_id ||
            event.attempt_id != recovered.current_attempt_id))) {
        throw std::runtime_error(
            "journal recovery found malformed host process authority evidence");
      }
      if (prepared) {
        if (event.event_id != launch_id + ":host-process-prepared" ||
            !replayed_process_prepares.insert(launch_id).second) {
          throw std::runtime_error(
              "journal recovery found duplicate host process prepare evidence");
        }
      } else if (committed) {
        if (!saga->committed ||
            !replayed_process_prepares.contains(launch_id) ||
            event.event_id != launch_id + ":host-process-committed" ||
            !replayed_process_commits.insert(launch_id).second) {
          throw std::runtime_error(
              "journal recovery found unordered host process commit evidence");
        }
      } else {
        if (!saga->exited ||
            !replayed_process_commits.contains(launch_id) ||
            event.event_id != launch_id + ":host-process-exited" ||
            !replayed_process_exits.insert(launch_id).second) {
          throw std::runtime_error(
              "journal recovery found unordered host process exit evidence");
        }
      }
      continue;
    }
    if (event.event_type == "node.entered") {
      if (phase != ReplayPhase::expecting_entry || expected_reacquisition_cause_id ||
          recovered.status != ExecutionStatus::running ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.run_revision != recovered.revision) {
        throw std::runtime_error("journal node.entered disagrees with deterministic FSM state");
      }
      const Node& node = plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      require_payload_string(event, "component", node.invoke.component);
      require_payload_string(event, "operation", node.invoke.operation);
      if ((expected_worker_entry_id && event.event_id != *expected_worker_entry_id) ||
          event.payload != nlohmann::json{{"component", node.invoke.component},
                                         {"operation", node.invoke.operation}}) {
        throw std::runtime_error("journal node entry is not the canonical readiness receipt");
      }
      expected_worker_entry_id.reset();
      phase = ReplayPhase::ready;
      current_dispatch_prepared = false;
      continue;
    }
    if (event.event_type == "node.attempt_restarted") {
      if (phase != ReplayPhase::ready || expected_completion ||
          !released_resume_command_id ||
          !released_resume_pause_command_id ||
          runtime_desired_state != "running" ||
          runtime_observed_state != "paused" ||
          recovered.status != ExecutionStatus::running ||
          event.run_revision != recovered.revision + 1U ||
          event.node_id != recovered.current_node_id ||
          event.worker_sequence != 0U) {
        throw std::runtime_error(
            "journal attempt restart is outside released-resource resume");
      }
      const auto resume =
          journal_.lifecycle_command(*released_resume_command_id);
      if (!resume || !resume->optimizer_step || resume->artifact_id.empty()) {
        throw std::runtime_error(
            "journal attempt restart has no applied resume checkpoint");
      }
      const ExecutionState restarted = restart_execution_attempt(recovered);
      const nlohmann::json expected_payload{
          {"previous_attempt_id", recovered.current_attempt_id},
          {"next_attempt_id", restarted.current_attempt_id},
          {"cause_command_id", *released_resume_command_id},
          {"pause_command_id", *released_resume_pause_command_id},
          {"checkpoint_artifact_id", resume->artifact_id},
      };
      if (event.event_id != *released_resume_command_id +
                                ":attempt-restarted" ||
          event.attempt_id != restarted.current_attempt_id ||
          event.optimizer_step != resume->optimizer_step ||
          event.payload != expected_payload) {
        throw std::runtime_error(
            "journal attempt restart is not canonical");
      }
      recovered = restarted;
      phase = ReplayPhase::acquisition_requested;
      current_dispatch_prepared = false;
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
          stored->component != node.invoke.component ||
          stored->operation != node.invoke.operation) {
        throw std::runtime_error("persisted dispatch disagrees with its preparation event");
      }
      require_payload_string(event, "component", node.invoke.component);
      require_payload_string(event, "operation", node.invoke.operation);
      current_dispatch_prepared = true;
      continue;
    }
    if (event.event_type == "worker.invocation_bound") {
      const auto invocation = journal_.worker_invocation(
          dispatch_id_for(recovered));
      if (phase != ReplayPhase::ready || expected_completion ||
          !current_dispatch_prepared || !invocation ||
          event.event_id != dispatch_id_for(recovered) + ":invocation" ||
          event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          invocation->run_id != run_id_ ||
          invocation->plan_hash != plan_.plan_hash ||
          invocation->plan_revision != event.plan_revision ||
          invocation->node_id != recovered.current_node_id ||
          invocation->attempt_id != recovered.current_attempt_id ||
          invocation->dispatch_id != dispatch_id_for(recovered)) {
        throw std::runtime_error(
            "journal worker invocation disagrees with its active dispatch");
      }
      continue;
    }
    if (event.event_type == "run.desired_state_changed") {
      const auto desired = event.payload.find("state");
      if (desired == event.payload.end() || !desired->is_string()) {
        throw std::runtime_error("run desire is missing its state");
      }
      const std::string desired_value = desired->get<std::string>();
      const bool starting = desired_value == "running" && runtime_desired_state == "queued" &&
                            runtime_observed_state == "queued" && phase == ReplayPhase::queued;
      const bool released_resume =
          desired_value == "running" &&
          runtime_desired_state == "paused" &&
          runtime_observed_state == "paused" &&
          phase == ReplayPhase::ready &&
          released_resume_command_id &&
          event.payload.value("cause_command_id", std::string{}) ==
              *released_resume_command_id;
      const bool cancelling =
          desired_value == "cancelled" &&
          ((runtime_desired_state == "running" &&
            runtime_observed_state == "running") ||
           (runtime_desired_state == "paused" &&
            runtime_observed_state == "paused"));
      const bool valid_desire = starting || cancelling ||
                                (desired_value == "paused" && runtime_desired_state == "running" &&
                                 runtime_observed_state == "running") ||
                                (desired_value == "running" && runtime_desired_state == "paused" &&
                                 runtime_observed_state == "paused");
      if (!valid_desire || (!starting && phase != ReplayPhase::ready) || expected_completion ||
          recovered.status != ExecutionStatus::running ||
          event.run_revision != recovered.revision + 1U ||
          (released_resume &&
           event.payload != nlohmann::json{{"state", "running"},
                                           {"cause_command_id",
                                            *released_resume_command_id}})) {
        throw std::runtime_error("journal contains an invalid lifecycle desire");
      }
      runtime_desired_state = desired_value;
      recovered.revision = event.run_revision;
      if (starting) {
        phase = ReplayPhase::acquisition_requested;
      }
      continue;
    }
    if (event.event_type == "resource.lease_acquired") {
      const bool resumed_acquisition =
          released_resume_command_id &&
          runtime_observed_state == "paused";
      if (phase != ReplayPhase::acquisition_requested ||
          runtime_desired_state != "running" ||
          (runtime_observed_state != "queued" && !resumed_acquisition) ||
          event.run_revision != recovered.revision || !event.node_id.empty() ||
          !event.attempt_id.empty()) {
        throw std::runtime_error(
            "journal contains lease acquisition outside a start or resume");
      }
      require_payload_string(event, "concurrency_key",
                             plan_.experiment.spec.workspace.concurrency_key);
      require_payload_string(event, "owner_run_id", run_id_);
      const auto lease_id = event.payload.find("lease_id");
      const auto fencing_token = event.payload.find("fencing_token");
      const auto clock_domain = event.payload.find("clock_domain");
      const auto boot_id = event.payload.find("boot_id");
      const auto acquired_at = event.payload.find("acquired_boottime_ns");
      const auto expires_at = event.payload.find("expires_boottime_ns");
      if (lease_id == event.payload.end() || !lease_id->is_string() ||
          lease_id->get_ref<const std::string&>().empty() ||
          fencing_token == event.payload.end() || !fencing_token->is_number_unsigned() ||
          clock_domain == event.payload.end() || !clock_domain->is_string() ||
          clock_domain->get_ref<const std::string&>() != ResourceLease::kBootTimeDomain ||
          boot_id == event.payload.end() || !boot_id->is_string() ||
          boot_id->get_ref<const std::string&>().empty() ||
          acquired_at == event.payload.end() || !acquired_at->is_number_integer() ||
          expires_at == event.payload.end() || !expires_at->is_number_integer() ||
          acquired_at->get<std::int64_t>() < 0 ||
          expires_at->get<std::int64_t>() <= acquired_at->get<std::int64_t>()) {
        throw std::runtime_error("lease acquisition event has an invalid fenced identity");
      }
      current_lease_acquisition = event;
      phase = ReplayPhase::lease_acquired;
      continue;
    }
    if (event.event_type == "run.observed_state_changed") {
      const auto observed = event.payload.find("state");
      if (observed == event.payload.end() || !observed->is_string()) {
        throw std::runtime_error("run observation is missing its state");
      }
      const std::string observed_value = observed->get<std::string>();
      if (observed_value == "acquiring" || observed_value == "pausing" ||
          observed_value == "paused" || observed_value == "resuming" ||
          observed_value == "running" || observed_value == "cancelling" ||
          observed_value == "cancelled") {
        const bool acquiring =
            observed_value == "acquiring" &&
            runtime_desired_state == "running" &&
            (runtime_observed_state == "queued" ||
             (runtime_observed_state == "paused" &&
              released_resume_command_id.has_value())) &&
            phase == ReplayPhase::lease_acquired;
        const bool worker_reacquiring =
            observed_value == "acquiring" &&
            runtime_desired_state == "running" &&
            runtime_observed_state == "running" &&
            phase == ReplayPhase::expecting_entry && !expected_completion &&
            expected_reacquisition_cause_id.has_value();
        const bool worker_started =
            observed_value == "running" && runtime_desired_state == "running" &&
            runtime_observed_state == "acquiring" &&
            phase == ReplayPhase::worker_ready;
        const bool valid_observation =
            acquiring || worker_reacquiring || worker_started ||
            (observed_value == "cancelling" &&
             runtime_desired_state == "cancelled" &&
             (runtime_observed_state == "running" ||
              runtime_observed_state == "paused")) ||
            (observed_value == "cancelled" &&
             runtime_desired_state == "cancelled" &&
             runtime_observed_state == "cancelling") ||
            (observed_value == "pausing" && runtime_desired_state == "paused" &&
             runtime_observed_state == "running") ||
            (observed_value == "paused" && runtime_desired_state == "paused" &&
             runtime_observed_state == "pausing") ||
            (observed_value == "resuming" && runtime_desired_state == "running" &&
             runtime_observed_state == "paused") ||
            (observed_value == "running" && runtime_desired_state == "running" &&
             runtime_observed_state == "resuming");
        if (!valid_observation ||
            (!acquiring && !worker_reacquiring && !worker_started &&
             phase != ReplayPhase::ready) ||
            (!worker_reacquiring && expected_completion) ||
            recovered.status != ExecutionStatus::running ||
            event.run_revision != recovered.revision + 1U ||
            (!event.node_id.empty() &&
             event.node_id != recovered.current_node_id) ||
            (!event.attempt_id.empty() &&
             event.attempt_id != recovered.current_attempt_id)) {
          throw std::runtime_error(
              "journal contains an invalid lifecycle observation");
        }
        runtime_observed_state = observed_value;
        recovered.revision = event.run_revision;
        if (observed_value == "cancelled") {
          recovered.status = ExecutionStatus::cancelled;
        }
        if (acquiring) {
          require_payload_string(event, "concurrency_key",
                                 plan_.experiment.spec.workspace.concurrency_key);
          const auto cause_id = event.payload.find("cause_event_id");
          const auto lease_id = event.payload.find("lease_id");
          const auto fencing_token = event.payload.find("fencing_token");
          if (cause_id == event.payload.end() || !cause_id->is_string() ||
              lease_id == event.payload.end() || !lease_id->is_string() ||
              fencing_token == event.payload.end() || !fencing_token->is_number_unsigned()) {
            throw std::runtime_error("acquiring observation lacks its lease cause identity");
          }
          const auto cause = journal_.event(cause_id->get<std::string>());
          if (!cause || cause->event_type != "resource.lease_acquired" ||
              cause->run_id != run_id_ ||
              cause->payload.value("lease_id", std::string{}) != lease_id->get<std::string>() ||
              cause->payload.value("fencing_token", std::uint64_t{}) !=
                  fencing_token->get<std::uint64_t>()) {
            throw std::runtime_error("acquiring observation disagrees with its lease cause");
          }
          phase = ReplayPhase::acquiring;
        } else if (worker_reacquiring) {
          const Node& node =
              plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
          const Component& component =
              plan_.experiment.spec.components.at(node.invoke.component);
          const std::string cause_id = *expected_reacquisition_cause_id;
          if (component.runtime == ComponentRuntime::builtin ||
              event.event_id != cause_id + ":acquiring" ||
              event.node_id != recovered.current_node_id ||
              event.attempt_id != recovered.current_attempt_id ||
              event.payload != nlohmann::json{{"state", "acquiring"},
                                              {"cause_event_id", cause_id}}) {
            throw std::runtime_error(
                "worker reacquisition observation is not canonical");
          }
          expected_reacquisition_cause_id.reset();
          phase = ReplayPhase::acquiring;
        } else if (worker_started) {
          const auto cause_id = event.payload.find("cause_event_id");
          if (cause_id == event.payload.end() || !cause_id->is_string()) {
            throw std::runtime_error("running observation lacks worker readiness cause");
          }
          const auto cause = journal_.event(cause_id->get<std::string>());
          if (!cause || cause->event_type != "worker.ready" ||
              cause->run_id != run_id_ || cause->node_id != recovered.current_node_id ||
              cause->attempt_id != recovered.current_attempt_id ||
              event.node_id != recovered.current_node_id ||
              event.attempt_id != recovered.current_attempt_id ||
              event.payload != nlohmann::json{{"state", "running"},
                                              {"cause_event_id", cause->event_id},
                                              {"launch_nonce", cause->payload.value(
                                                   "launch_nonce", std::string{})}}) {
            throw std::runtime_error("running observation disagrees with worker readiness");
          }
          const std::string launch_event_id =
              cause->payload.value("cause_event_id", std::string{});
          if (launch_event_id.empty()) {
            throw std::runtime_error("worker readiness has no launch identity");
          }
          expected_worker_entry_id = launch_event_id + ":entered";
          phase = ReplayPhase::expecting_entry;
        }
        continue;
      }
      if (phase != ReplayPhase::expecting_terminal ||
          event.run_revision != recovered.revision) {
        throw std::runtime_error(
            "journal contains an unexpected terminal state observation");
      }
      require_payload_string(event, "state", enum_to_string(recovered.status));
      phase = ReplayPhase::terminal;
      continue;
    }
    if (event.event_type == "worker.launch_requested") {
      if (phase != ReplayPhase::acquiring ||
          runtime_observed_state != "acquiring" ||
          event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.worker_sequence != 0) {
        throw std::runtime_error("journal contains a launch request outside acquisition");
      }
      const Node& node = plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      const Component& component =
          plan_.experiment.spec.components.at(node.invoke.component);
      if (component.runtime == ComponentRuntime::builtin) {
        throw std::runtime_error("builtin nodes cannot request an external worker launch");
      }
      const WorkerLaunchTicket launch = launch_from_event(event);
      const nlohmann::json expected_payload = worker_launch_payload(launch);
      const auto& acquisition = current_lease_acquisition;
      if (launch.run_id != run_id_ || launch.adapter != component.adapter ||
          launch.adapter_version != component.version ||
          launch.code_fingerprint.empty() || launch.launch_nonce.empty() ||
          launch.launch_nonce != worker_launch_nonce(plan_, launch) ||
          event.payload != expected_payload || !acquisition ||
          launch.concurrency_key !=
              plan_.experiment.spec.workspace.concurrency_key ||
          launch.lease_id !=
              acquisition->payload.value("lease_id", std::string{}) ||
          launch.fencing_token !=
              acquisition->payload.value("fencing_token", std::uint64_t{})) {
        throw std::runtime_error("worker launch request disagrees with plan or lease evidence");
      }
      phase = ReplayPhase::launch_requested;
      continue;
    }
    if (event.event_type == "worker.launch_bound") {
      if (phase != ReplayPhase::launch_requested ||
          event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.worker_sequence != 0) {
        throw std::runtime_error(
            "journal contains a host launch binding without a launch request");
      }
      const std::string launch_id = event.payload.value(
          "cause_event_id", std::string{});
      const auto launch_event = journal_.event(launch_id);
      if (!launch_event ||
          launch_event->event_type != "worker.launch_requested" ||
          launch_event->run_id != run_id_ ||
          launch_event->node_id != recovered.current_node_id ||
          launch_event->attempt_id != recovered.current_attempt_id ||
          event.event_id != launch_id + ":bound" ||
          event.event_version != 1U ||
          !event.payload.contains("spec")) {
        throw std::runtime_error(
            "host launch binding has no matching durable launch request");
      }
      const WorkerLaunchTicket launch = launch_from_event(*launch_event);
      const Node& active_node =
          plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      const Component& active_component =
          plan_.experiment.spec.components.at(active_node.invoke.component);
      const Operation& active_operation = active_component.operations.at(
          active_node.invoke.operation);
      ResolvedLaunchSpec binding;
      try {
        binding = resolved_launch_spec_from_json(event.payload.at("spec"));
      } catch (const std::exception& exception) {
        throw std::runtime_error(
            std::string("host launch binding is malformed: ") +
            exception.what());
      }
      const ResolvedLaunchIdentity& identity = binding.identity;
      if (identity.launch_event_id != launch_event->event_id ||
          identity.run_id != launch.run_id ||
          identity.node_id != launch.node_id ||
          identity.attempt_id != launch.attempt_id ||
          identity.launch_nonce != launch.launch_nonce ||
          identity.adapter_key.adapter != launch.adapter ||
          identity.adapter_key.version != launch.adapter_version ||
          identity.adapter_key.runtime != active_component.runtime ||
          identity.adapter_key.operation != active_node.invoke.operation ||
          identity.adapter_key.contract != active_operation.contract ||
          identity.code_fingerprint != launch.code_fingerprint ||
          identity.required_capabilities != launch.required_capabilities ||
          identity.concurrency_key != launch.concurrency_key ||
          identity.lease_id != launch.lease_id ||
          identity.fencing_token != launch.fencing_token ||
          identity.host_grant != launch.host_grant ||
          event.payload != nlohmann::json{
                               {"cause_event_id", launch_event->event_id},
                               {"spec", resolved_launch_spec_json(binding)}}) {
        throw std::runtime_error(
            "host launch binding disagrees with its launch ticket");
      }
      phase = ReplayPhase::launch_bound;
      continue;
    }
    if (event.event_type == "worker.ready") {
      if (phase != ReplayPhase::launch_bound ||
          event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.worker_sequence != 0) {
        throw std::runtime_error("journal contains readiness without a launch request");
      }
      const auto cause_id = event.payload.find("cause_event_id");
      if (cause_id == event.payload.end() || !cause_id->is_string()) {
        throw std::runtime_error("worker readiness lacks its launch cause");
      }
      const auto launch_event = journal_.event(cause_id->get<std::string>());
      if (!launch_event || launch_event->event_type != "worker.launch_requested" ||
          launch_event->run_id != run_id_ ||
          launch_event->node_id != recovered.current_node_id ||
          launch_event->attempt_id != recovered.current_attempt_id) {
        throw std::runtime_error("worker readiness disagrees with its launch request");
      }
      const WorkerLaunchTicket launch = launch_from_event(*launch_event);
      const auto capabilities = event.payload.find("capabilities");
      if (capabilities == event.payload.end() || !capabilities->is_array()) {
        throw std::runtime_error("worker readiness lacks capabilities");
      }
      std::vector<std::string> ready_capabilities;
      try {
        ready_capabilities = canonical_capabilities(
            capabilities->get<std::vector<std::string>>());
      } catch (const std::exception& exception) {
        throw std::runtime_error(std::string("worker readiness is malformed: ") +
                                 exception.what());
      }
      if (!std::ranges::includes(ready_capabilities,
                                 launch.required_capabilities) ||
          event.payload.value("launch_nonce", std::string{}) != launch.launch_nonce ||
          event.payload.value("adapter", std::string{}) != launch.adapter ||
          event.payload.value("adapter_version", std::string{}) !=
              launch.adapter_version ||
          event.payload.value("code_fingerprint", std::string{}) !=
              launch.code_fingerprint ||
          event.payload.value("concurrency_key", std::string{}) !=
              launch.concurrency_key ||
          event.payload.value("lease_id", std::string{}) != launch.lease_id ||
          event.payload.value("fencing_token", std::uint64_t{}) !=
              launch.fencing_token ||
          event.payload.value("last_acked_controller_sequence", std::uint64_t{}) != 0U) {
        throw std::runtime_error("worker readiness disagrees with its launch ticket");
      }
      const nlohmann::json expected_payload{
          {"cause_event_id", launch_event->event_id},
          {"launch_nonce", launch.launch_nonce},
          {"adapter", launch.adapter},
          {"adapter_version", launch.adapter_version},
          {"code_fingerprint", launch.code_fingerprint},
          {"capabilities", ready_capabilities},
          {"last_acked_controller_sequence", std::uint64_t{0}},
          {"concurrency_key", launch.concurrency_key},
          {"lease_id", launch.lease_id},
          {"fencing_token", launch.fencing_token},
      };
      if (event.payload != expected_payload) {
        throw std::runtime_error("worker readiness payload is not canonical");
      }
      phase = ReplayPhase::worker_ready;
      continue;
    }
    if (event.event_type == "resource.acquired" && phase == ReplayPhase::acquiring) {
      const Node& node = plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      const Component& component =
          plan_.experiment.spec.components.at(node.invoke.component);
      if (component.runtime != ComponentRuntime::builtin ||
          component.adapter != "trainvm.core" ||
          node.invoke.operation != "acquire_resources" ||
          event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.worker_sequence != 0 || expected_completion || pending_cause) {
        throw std::runtime_error("journal contains an invalid builtin admission result");
      }
      require_payload_string(event, "concurrency_key",
                             plan_.experiment.spec.workspace.concurrency_key);
      const auto& acquisition = current_lease_acquisition;
      if (!acquisition ||
          event.payload.value("lease_id", std::string{}) !=
              acquisition->payload.value("lease_id", std::string{}) ||
          event.payload.value("fencing_token", std::uint64_t{}) !=
              acquisition->payload.value("fencing_token", std::uint64_t{}) ||
          event.payload !=
              nlohmann::json{{"concurrency_key",
                              plan_.experiment.spec.workspace.concurrency_key},
                             {"lease_id", acquisition->payload.value(
                                              "lease_id", std::string{})},
                             {"fencing_token", acquisition->payload.value(
                                                   "fencing_token", std::uint64_t{})}}) {
        throw std::runtime_error("builtin admission result disagrees with lease evidence");
      }
      pending_cause = event;
      pending_builtin_admission = true;
      phase = ReplayPhase::awaiting_transition;
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
      if (!pending_builtin_admission) {
        expected_completion =
            std::pair{dispatch_id_for(recovered), pending_cause->event_id};
      }
      recovered = result.state;
      pending_cause.reset();
      if (pending_builtin_admission) {
        pending_builtin_admission = false;
        phase = ReplayPhase::acquiring;
      } else {
        if (recovered.status == ExecutionStatus::running) {
          const Node& node =
              plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
          const Component& component =
              plan_.experiment.spec.components.at(node.invoke.component);
          if (managed_run && component.runtime != ComponentRuntime::builtin) {
            expected_reacquisition_cause_id = expected_completion->second;
          }
          phase = ReplayPhase::expecting_entry;
        } else {
          phase = ReplayPhase::expecting_terminal;
        }
      }
      continue;
    }
    if (event.event_type == "node.dispatch_completed") {
      if ((phase != ReplayPhase::ready && phase != ReplayPhase::expecting_entry &&
           phase != ReplayPhase::terminal) ||
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
          throw std::runtime_error(
              "terminal control command has no worker acknowledgement identity");
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
            event.worker_sequence != identity.worker_sequence || !command->acknowledged_at_ns ||
            event.wall_time_ns != *command->acknowledged_at_ns) {
          throw std::runtime_error(
              "control acknowledgement event disagrees with its durable command");
        }
      }
      replayed_control_status[command->command_id] = suffix;
      continue;
    }
    if (event.event_type.starts_with("checkpoint.")) {
      const std::string command_id =
          event.payload.value("command_id", std::string{});
      const auto command = journal_.checkpoint_command(command_id);
      if (command_id.empty() || !command || command->run_id != run_id_ ||
          event.run_revision != recovered.revision ||
          event.plan_revision != kInitialPlanRevision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          phase != ReplayPhase::ready || expected_completion ||
          runtime_observed_state != "running") {
        throw std::runtime_error(
            "checkpoint journal event is outside its active worker attempt");
      }
      auto previous = replayed_checkpoint_status.find(command_id);
      if (event.event_type == "checkpoint.requested") {
        const nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"idempotency_key", command->idempotency_key},
            {"expected_run_revision", command->expected_run_revision},
            {"plan_revision", command->plan_revision},
            {"reason", command->reason},
            {"author", command->author},
            {"audit_reason", command->audit_reason},
        };
        if (previous != replayed_checkpoint_status.end() ||
            event.worker_sequence != 0U || event.payload != expected_payload ||
            event.run_revision != command->expected_run_revision) {
          throw std::runtime_error(
              "checkpoint request event disagrees with its durable command");
        }
        replayed_checkpoint_status[command_id] = "requested";
      } else {
        if (event.event_type != "checkpoint.acknowledged" ||
            previous == replayed_checkpoint_status.end() ||
            previous->second != "requested" || !command->acknowledgement ||
            !command->acknowledged_at_ns) {
          throw std::runtime_error(
              "checkpoint acknowledgement has no preceding request event");
        }
        const auto& identity = *command->acknowledgement;
        const std::string status(checkpoint_status_name(command->status));
        const nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"status", status},
            {"artifact_id", command->artifact_id},
            {"diagnostics", command->diagnostics},
            {"concurrency_key", identity.concurrency_key},
            {"lease_id", identity.lease_id},
            {"fencing_token", identity.fencing_token},
        };
        if (event.payload != expected_payload ||
            event.optimizer_step != command->optimizer_step ||
            event.worker_sequence != identity.worker_sequence ||
            event.wall_time_ns != *command->acknowledged_at_ns) {
          throw std::runtime_error(
              "checkpoint acknowledgement event disagrees with its durable command");
        }
        replayed_checkpoint_status[command_id] = status;
      }
      continue;
    }
    if (event.event_type.starts_with("lifecycle.")) {
      const std::string command_id =
          event.payload.value("command_id", std::string{});
      const auto command = journal_.lifecycle_command(command_id);
      if (command_id.empty() || !command || command->run_id != run_id_ ||
          event.run_revision != recovered.revision ||
          event.plan_revision != kInitialPlanRevision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          phase != ReplayPhase::ready || expected_completion) {
        throw std::runtime_error(
            "lifecycle journal event is outside its active worker attempt");
      }
      const bool resume = command->kind == LifecycleCommandKind::resume;
      const bool cancel = command->kind == LifecycleCommandKind::cancel;
      const bool valid_state =
          resume
              ? runtime_desired_state == "paused" &&
                    runtime_observed_state == "paused"
              : cancel
                    ? (runtime_desired_state == "running" &&
                       runtime_observed_state == "running") ||
                          (runtime_desired_state == "paused" &&
                           runtime_observed_state == "paused")
                    : runtime_desired_state == "running" &&
                          runtime_observed_state == "running";
      if (!valid_state) {
        throw std::runtime_error(
            "lifecycle journal event is stale for the replayed run state");
      }
      auto previous = replayed_lifecycle_status.find(command_id);
      if (event.event_type == "lifecycle.requested") {
        nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"idempotency_key", command->idempotency_key},
            {"expected_run_revision", command->expected_run_revision},
            {"kind", lifecycle_kind_name(command->kind)},
            {"checkpoint_first", command->checkpoint_first},
            {"release_resources", command->release_resources},
            {"author", command->author},
            {"reason", command->reason},
        };
        if (command->kind == LifecycleCommandKind::cancel) {
          expected_payload["cancel_reason"] = command->cancel_reason;
          expected_payload["graceful_timeout_ns"] =
              command->graceful_timeout_ns;
        }
        if (previous != replayed_lifecycle_status.end() ||
            event.worker_sequence != 0U || event.payload != expected_payload ||
            event.run_revision != command->expected_run_revision) {
          throw std::runtime_error(
              "lifecycle request event disagrees with its durable command");
        }
        replayed_lifecycle_status[command_id] = "requested";
      } else {
        if (event.event_type != "lifecycle.acknowledged" ||
            previous == replayed_lifecycle_status.end() ||
            previous->second != "requested" || !command->acknowledgement ||
            !command->acknowledged_at_ns) {
          throw std::runtime_error(
              "lifecycle acknowledgement has no preceding request event");
        }
        const auto& identity = *command->acknowledgement;
        const std::string status(lifecycle_status_name(command->status));
        nlohmann::json expected_payload{
            {"command_id", command->command_id},
            {"kind", lifecycle_kind_name(command->kind)},
            {"status", status},
            {"artifact_id", command->artifact_id},
            {"diagnostics", command->diagnostics},
            {"concurrency_key", identity.concurrency_key},
            {"lease_id", identity.lease_id},
            {"fencing_token", identity.fencing_token},
        };
        const bool controller_resume =
            command->kind == LifecycleCommandKind::resume &&
            identity.worker_sequence == 0U;
        if (controller_resume) {
          const std::string pause_command_id =
              event.payload.value("pause_command_id", std::string{});
          const auto pause =
              journal_.lifecycle_command(pause_command_id);
          if (pause_command_id.empty() || !pause ||
              pause->kind != LifecycleCommandKind::pause ||
              !pause->checkpoint_first || !pause->release_resources ||
              pause->status != LifecycleCommandStatus::applied ||
              pause->node_id != command->node_id ||
              pause->attempt_id != command->attempt_id ||
              pause->optimizer_step != command->optimizer_step ||
              pause->artifact_id != command->artifact_id) {
            throw std::runtime_error(
                "controller resume acknowledgement has no pause checkpoint lineage");
          }
          expected_payload["authority"] = "controller";
          expected_payload["pause_command_id"] = pause_command_id;
          released_resume_command_id = command_id;
          released_resume_pause_command_id = pause_command_id;
        }
        if (event.payload != expected_payload ||
            event.optimizer_step != command->optimizer_step ||
            event.worker_sequence != identity.worker_sequence ||
            event.wall_time_ns != *command->acknowledged_at_ns) {
          throw std::runtime_error(
              "lifecycle acknowledgement event disagrees with its durable command");
        }
        replayed_lifecycle_status[command_id] = status;
      }
      continue;
    }
    if (is_worker_observation(event.event_type)) {
      try {
        require_worker_observation_shape(event);
      } catch (const std::invalid_argument& exception) {
        throw std::runtime_error(exception.what());
      }
      const bool paused_heartbeat =
          runtime_observed_state == "paused" &&
          event.event_type == "worker.heartbeat" &&
          event.payload.value("phase", std::string{}) == "paused";
      if (phase != ReplayPhase::ready || expected_completion ||
          recovered.status != ExecutionStatus::running ||
          (runtime_observed_state != "running" && !paused_heartbeat) ||
          event.run_revision != recovered.revision ||
          event.plan_revision != kInitialPlanRevision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.event_id != worker_observation_event_id(
                                event, dispatch_id_for(recovered))) {
        throw std::runtime_error(
            "journal contains a worker observation outside its active attempt");
      }
      continue;
    }
    if (phase == ReplayPhase::ready && managed_run &&
        recovered.status == ExecutionStatus::running &&
        runtime_observed_state == "running" && event.worker_sequence == 0) {
      const Node& node =
          plan_.experiment.spec.workflow.nodes.at(recovered.current_node_id);
      const Component& component =
          plan_.experiment.spec.components.at(node.invoke.component);
      const bool validation = node.invoke.operation == "validate_artifact";
      const bool releasing = node.invoke.operation == "release_resources";
      const auto dispatch = journal_.dispatch(dispatch_id_for(recovered));
      const auto& acquisition = current_lease_acquisition;
      nlohmann::json expected_payload = nlohmann::json::object();
      if (releasing && acquisition) {
        expected_payload = {
            {"concurrency_key",
             plan_.experiment.spec.workspace.concurrency_key},
            {"lease_id",
             acquisition->payload.value("lease_id", std::string{})},
            {"fencing_token", acquisition->payload.value(
                                  "fencing_token", std::uint64_t{})},
        };
      }
      if (component.runtime != ComponentRuntime::builtin ||
          component.adapter != "trainvm.core" || (!validation && !releasing) ||
          !dispatch || dispatch->status != DispatchStatus::completed ||
          dispatch->result_event_id !=
              std::optional<std::string>{event.event_id} ||
          event.event_id != dispatch->dispatch_id + ":builtin-result" ||
          event.run_id != run_id_ || event.run_revision != recovered.revision ||
          event.node_id != recovered.current_node_id ||
          event.attempt_id != recovered.current_attempt_id ||
          event.monotonic_time_ns != 0 ||
          (validation && event.event_type != "artifact.validated" &&
           event.event_type != "artifact.invalid") ||
          (releasing && event.event_type != "resource.released") ||
          event.payload != expected_payload || !acquisition) {
        throw std::runtime_error(
            "journal contains a noncanonical managed builtin result");
      }
      if (releasing &&
          !journal_.has_lease_release_receipt(
              plan_.experiment.spec.workspace.concurrency_key, run_id_,
              acquisition->payload.value("lease_id", std::string{}),
              acquisition->payload.value("fencing_token", std::uint64_t{}),
              acquisition->payload.value("clock_domain", std::string{}),
              acquisition->payload.value("boot_id", std::string{}),
              event.wall_time_ns)) {
        throw std::runtime_error(
            "managed resource release has no durable lease receipt");
      }
      pending_cause = event;
      phase = ReplayPhase::awaiting_transition;
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
      throw std::runtime_error(
          "journal contains a causing event with an invalid revision or sequence");
    }
    const auto receipt = journal_.dispatch(dispatch_id_for(recovered));
    if (!receipt || receipt->status != DispatchStatus::completed ||
        receipt->result_event_id != std::optional<std::string>{event.event_id}) {
      throw std::runtime_error(
          "causing event has no matching completed dispatch receipt");
    }
    pending_cause = event;
    phase = ReplayPhase::awaiting_transition;
  }
  if ((phase != ReplayPhase::queued && phase != ReplayPhase::acquiring &&
       phase != ReplayPhase::launch_requested &&
       phase != ReplayPhase::launch_bound &&
       phase != ReplayPhase::ready && phase != ReplayPhase::terminal) ||
      expected_completion || expected_reacquisition_cause_id) {
    throw std::runtime_error(
        "run journal ends in an incomplete controller transaction");
  }
  for (const auto &[command_id, status] : replayed_control_status) {
    const auto command = journal_.control_command(command_id);
    if (!command || status != control_status_name(command->status)) {
      throw std::runtime_error(
          "control command projection disagrees with journal replay");
    }
  }
  for (const auto& [command_id, status] : replayed_checkpoint_status) {
    const auto command = journal_.checkpoint_command(command_id);
    if (!command || status != checkpoint_status_name(command->status)) {
      throw std::runtime_error(
          "checkpoint command projection disagrees with journal replay");
    }
  }
  for (const auto& [command_id, status] : replayed_lifecycle_status) {
    const auto command = journal_.lifecycle_command(command_id);
    if (!command || status != lifecycle_status_name(command->status)) {
      throw std::runtime_error(
          "lifecycle command projection disagrees with journal replay");
    }
  }
  const auto projection = journal_.projection(run_id_);
  const std::string expected_observed =
      recovered.status == ExecutionStatus::running
          ? runtime_observed_state
          : std::string(enum_to_string(recovered.status));
  const bool active_node = recovered.status == ExecutionStatus::running &&
                           runtime_observed_state != "queued" &&
                           runtime_observed_state != "acquiring";
  const std::string expected_node =
      active_node ? recovered.current_node_id : std::string{};
  const std::string expected_attempt =
      active_node ? recovered.current_attempt_id : std::string{};
  if (!projection || projection->plan_hash != plan_.plan_hash ||
      projection->experiment_name != plan_.experiment.metadata.name ||
      projection->run_revision != recovered.revision ||
      projection->desired_state != runtime_desired_state ||
      projection->observed_state != expected_observed ||
      projection->current_node_id != expected_node ||
      projection->current_attempt_id != expected_attempt) {
    throw std::runtime_error(
        "run projection disagrees with deterministic journal replay");
  }
  state_ = std::move(recovered);
  initialized_ = true;
  paused_ = runtime_observed_state != "running";
  return state_;
}

LeaseAcquireResult Controller::begin_acquisition(
    const AuthorityTimeSample& now) {
  const ExecutionState admission_state = start_execution(plan_, run_id_);
  const Node& admission_node = plan_.experiment.spec.workflow.nodes.at(
      admission_state.current_node_id);
  const Component& admission_component =
      plan_.experiment.spec.components.at(admission_node.invoke.component);
  const auto admission_operation =
      admission_component.operations.find(admission_node.invoke.operation);
  if (admission_component.runtime != ComponentRuntime::builtin ||
      admission_component.adapter != "trainvm.core" ||
      admission_component.version != "1.0.0" ||
      admission_node.invoke.operation != "acquire_resources" ||
      admission_operation == admission_component.operations.end() ||
      admission_operation->second.contract !=
          "trainvm.v1.AcquireResources" ||
      admission_node.effect != Effect::resource ||
      admission_node.idempotency != Idempotency::receipt_required) {
    throw std::logic_error(
        "queued admission supports only the exact builtin trainvm.core 1.0.0 "
        "acquire_resources contract");
  }
  const Transition* admission_transition = nullptr;
  for (const Transition& transition : admission_node.transitions) {
    if (transition.on != "resource.acquired") {
      continue;
    }
    if (admission_transition != nullptr || transition.where) {
      throw std::logic_error(
          "resource admission requires one unconditional resource.acquired transition");
    }
    admission_transition = &transition;
  }
  if (admission_transition == nullptr ||
      admission_transition->target.starts_with('$')) {
    throw std::logic_error(
        "resource admission must target one external worker node");
  }
  const Node& target_node = plan_.experiment.spec.workflow.nodes.at(
      admission_transition->target);
  const Component& target_component =
      plan_.experiment.spec.components.at(target_node.invoke.component);
  if (target_component.runtime == ComponentRuntime::builtin) {
    throw std::logic_error("resource admission target must be an external worker node");
  }
  recover();
  const auto projection = journal_.projection(run_id_);
  if (!projection) {
    throw std::runtime_error("initialized controller has no durable run projection");
  }
  const std::string& concurrency_key = plan_.experiment.spec.workspace.concurrency_key;
  const std::string lease_id =
      "lease-" + sha256_hex(nlohmann::json({{"run_id", run_id_},
                                            {"plan_hash", plan_.plan_hash},
                                            {"concurrency_key", concurrency_key}})
                               .dump());
  const std::int64_t timeout_seconds =
      plan_.experiment.spec.resources.lease_timeout_seconds.value_or(30);
  if (timeout_seconds <= 0 ||
      timeout_seconds > std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL) {
    throw std::runtime_error("compiled plan has an invalid resource lease timeout");
  }
  const std::int64_t timeout_ns = timeout_seconds * 1'000'000'000LL;
  if (projection->desired_state == "running" && projection->observed_state == "acquiring") {
    const auto acquired = journal_.event(run_id_ + ":lease-acquired");
    if (!acquired || acquired->event_type != "resource.lease_acquired" ||
        acquired->payload.value("concurrency_key", std::string{}) != concurrency_key ||
        acquired->payload.value("owner_run_id", std::string{}) != run_id_ ||
        acquired->payload.value("lease_id", std::string{}) != lease_id) {
      throw std::runtime_error("acquiring run has no durable lease event");
    }
    const auto event_fence = acquired->payload.find("fencing_token");
    const auto event_clock_domain = acquired->payload.find("clock_domain");
    const auto event_boot_id = acquired->payload.find("boot_id");
    const auto event_acquired_at = acquired->payload.find("acquired_boottime_ns");
    if (event_fence == acquired->payload.end() || !event_fence->is_number_unsigned() ||
        event_clock_domain == acquired->payload.end() ||
        !event_clock_domain->is_string() ||
        event_clock_domain->get_ref<const std::string&>() !=
            ResourceLease::kBootTimeDomain ||
        event_boot_id == acquired->payload.end() || !event_boot_id->is_string() ||
        event_acquired_at == acquired->payload.end() || !event_acquired_at->is_number_integer()) {
      throw std::runtime_error("acquiring run has malformed durable lease evidence");
    }
    const auto lease = journal_.active_lease(concurrency_key, now);
    if (!lease || lease->owner_run_id != run_id_ || lease->lease_id != lease_id ||
        lease->clock_domain != event_clock_domain->get<std::string>() ||
        lease->boot_id != event_boot_id->get<std::string>() ||
        lease->fencing_token != event_fence->get<std::uint64_t>() ||
        lease->acquired_boottime_ns != event_acquired_at->get<std::int64_t>()) {
      throw std::runtime_error("acquiring run no longer owns its fenced lease identity");
    }
    if (lease->expires_boottime_ns <= now.boot.nanoseconds) {
      throw std::runtime_error(
          "acquiring run lease has expired and requires a fenced reconciliation decision");
    }
    LeaseAcquireResult result{.status = LeaseAcquireStatus::already_owned,
                              .lease = *lease};
    recover();
    complete_builtin_admission(result.lease, now);
    recover();
    return result;
  }
  if (projection->desired_state != "queued" || projection->observed_state != "queued" ||
      projection->run_revision != state_.revision || !projection->current_node_id.empty() ||
      !projection->current_attempt_id.empty()) {
    throw std::logic_error("controller can acquire resources only for a queued run");
  }
  const std::uint64_t desired_revision = state_.revision + 1U;
  const std::uint64_t acquiring_revision = desired_revision + 1U;
  const std::string acquired_event_id = run_id_ + ":lease-acquired";
  const std::vector<Event> events{
      acquisition_event(state_, run_id_ + ":lease-desired", desired_revision,
                        "run.desired_state_changed",
                        {{"state", "running"},
                         {"cause", "scheduler.lease_acquisition"},
                         {"lease_id", lease_id},
                         {"plan_hash", plan_.plan_hash}},
                        now.wall.nanoseconds),
      acquisition_event(state_, acquired_event_id, desired_revision,
                        "resource.lease_acquired",
                        {{"concurrency_key", concurrency_key},
                         {"owner_run_id", run_id_},
                         {"lease_id", lease_id}},
                        now.wall.nanoseconds),
      acquisition_event(state_, run_id_ + ":acquiring", acquiring_revision,
                        "run.observed_state_changed",
                        {{"state", "acquiring"},
                         {"cause_event_id", acquired_event_id},
                         {"concurrency_key", concurrency_key},
                         {"lease_id", lease_id}},
                        now.wall.nanoseconds),
  };
  LeaseAcquireResult result = journal_.acquire_lease_with_events(
      concurrency_key, run_id_, lease_id, now, timeout_ns, events);
  if (result.status != LeaseAcquireStatus::busy) {
    recover();
    complete_builtin_admission(result.lease, now);
    recover();
  }
  return result;
}

void Controller::complete_builtin_admission(const ResourceLease& lease,
                                            const AuthorityTimeSample& now) {
  const Node& node =
      plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  const Component& component =
      plan_.experiment.spec.components.at(node.invoke.component);
  if (component.runtime != ComponentRuntime::builtin ||
      component.adapter != "trainvm.core" ||
      node.invoke.operation != "acquire_resources") {
    return;
  }
  Event acquired{
      .event_id = run_id_ + ":admission:resource-acquired",
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0,
      .event_type = "resource.acquired",
      .event_version = 1,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"concurrency_key", lease.concurrency_key},
                  {"lease_id", lease.lease_id},
                  {"fencing_token", lease.fencing_token}},
  };
  const TransitionResult transition = advance_execution(plan_, state_, acquired);
  if (transition.state.status != ExecutionStatus::running) {
    throw std::runtime_error("builtin resource admission must advance to a running node");
  }
  journal_.complete_builtin_admission(
      lease, now, {acquired, transitioned_event(acquired, transition)});
}

WorkerLaunchTicket Controller::prepare_worker_launch(WorkerLaunchRequest request,
                                                       const AuthorityTimeSample& now) {
  if (request.code_fingerprint.empty() ||
      request.code_fingerprint.size() > 512U) {
    throw std::invalid_argument("worker launch requires a bounded code fingerprint and clock");
  }
  request.required_capabilities =
      canonical_capabilities(std::move(request.required_capabilities));
  recover();
  const auto projection = journal_.projection(run_id_);
  if (!projection || projection->observed_state != "acquiring" ||
      projection->desired_state != "running" ||
      projection->run_revision != state_.revision ||
      !projection->current_node_id.empty() ||
      !projection->current_attempt_id.empty()) {
    throw std::logic_error("worker launch requires an unassigned acquiring run");
  }
  const Node& node = plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  const Component& component =
      plan_.experiment.spec.components.at(node.invoke.component);
  if (component.runtime == ComponentRuntime::builtin) {
    throw std::logic_error("builtin node must complete before external worker launch");
  }
  const std::string& concurrency_key =
      plan_.experiment.spec.workspace.concurrency_key;
  const auto active = journal_.active_lease(concurrency_key, now);
  const auto acquisition =
      active ? journal_.lease_acquisition_event(run_id_, active->lease_id)
             : std::nullopt;
  if (!active || !acquisition || active->owner_run_id != run_id_ ||
      active->lease_id !=
          acquisition->payload.value("lease_id", std::string{}) ||
      active->fencing_token !=
          acquisition->payload.value("fencing_token", std::uint64_t{}) ||
      active->clock_domain !=
          acquisition->payload.value("clock_domain", std::string{}) ||
      active->boot_id != acquisition->payload.value("boot_id", std::string{}) ||
      active->acquired_boottime_ns !=
          acquisition->payload.value("acquired_boottime_ns", std::int64_t{})) {
    throw OperationPreconditionError(
        "worker launch no longer owns its acquired lease fence");
  }
  WorkerLaunchTicket launch{
      .run_id = run_id_,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .launch_nonce = "",
      .adapter = component.adapter,
      .adapter_version = component.version,
      .code_fingerprint = std::move(request.code_fingerprint),
      .required_capabilities = std::move(request.required_capabilities),
      .concurrency_key = concurrency_key,
      .lease_id = active->lease_id,
      .fencing_token = active->fencing_token,
      .host_grant = std::nullopt,
  };
  launch.host_grant = journal_.host_launch_grant_claim(
      launch.run_id, launch.concurrency_key, launch.lease_id,
      launch.fencing_token, now);
  launch.launch_nonce = worker_launch_nonce(plan_, launch);
  const Event event{
      .event_id = worker_launch_event_id(state_),
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0,
      .event_type = "worker.launch_requested",
      .event_version = 1,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = worker_launch_payload(launch),
  };
  journal_.prepare_worker_launch(launch, now, event);
  recover();
  return launch;
}

ResolvedLaunchSpec Controller::bind_worker_launch(
    const ResolvedLaunch& resolved,
    const HostLaunchRegistry& host_registry,
    const HostIdentity& authority_host, const AuthorityTimeSample& now) {
  const ResolvedLaunchSpec& binding = resolved.spec();
  recover();
  const std::string launch_id = worker_launch_event_id(state_);
  const Node& active_node =
      plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  const Component& active_component =
      plan_.experiment.spec.components.at(active_node.invoke.component);
  const Operation& active_operation =
      active_component.operations.at(active_node.invoke.operation);
  const AdapterKey expected_key{
      .adapter = active_component.adapter,
      .version = active_component.version,
      .runtime = active_component.runtime,
      .operation = active_node.invoke.operation,
      .contract = active_operation.contract,
  };
  const HostLaunchProfile& host_profile = host_registry.resolve(
      expected_key, binding.identity.code_fingerprint);
  if (binding.identity.launch_event_id != launch_id ||
      binding.identity.adapter_key != expected_key ||
      binding.identity.host != authority_host ||
      binding.identity.host_registry_digest !=
          host_registry.registry_digest() ||
      binding.identity.host_profile_digest != host_registry.profile_digest(
          expected_key, binding.identity.code_fingerprint) ||
      binding.identity.executable.source_path !=
          host_profile.executable_path ||
      binding.identity.executable.sealed_sha256 !=
          host_profile.executable_fingerprint ||
      binding.identity.public_arguments != host_profile.public_arguments ||
      binding.identity.working_directory.source_path !=
          host_profile.working_directory ||
      binding.identity.code.has_value() != host_profile.code_path.has_value() ||
      (binding.identity.code &&
       (binding.identity.code->source_path != *host_profile.code_path ||
        binding.identity.code->sealed_sha256 !=
            host_profile.code_fingerprint))) {
    throw std::invalid_argument(
        "host launch binding disagrees with the active operation");
  }
  const Event event{
      .event_id = launch_id + ":bound",
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0,
      .event_type = "worker.launch_bound",
      .event_version = 1,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"cause_event_id", launch_id},
                  {"spec", resolved_launch_spec_json(binding)}},
  };
  (void)journal_.bind_worker_launch(binding, now, event);
  recover();
  const auto durable = journal_.launch_binding(launch_id);
  if (!durable) {
    throw std::runtime_error(
        "durable host launch binding disappeared after commit");
  }
  return *durable;
}

WorkerReadinessResult Controller::accept_worker_hello(WorkerHelloEvidence hello,
                                                       const AuthorityTimeSample& now) {
  hello.capabilities = canonical_capabilities(std::move(hello.capabilities));
  recover();
  const auto launch_event = journal_.event(worker_launch_event_id(state_));
  if (!launch_event || launch_event->event_type != "worker.launch_requested") {
    throw std::logic_error("worker hello has no durable launch request");
  }
  if (!journal_.launch_binding(launch_event->event_id)) {
    throw std::logic_error(
        "worker hello has no durable host launch binding");
  }
  const WorkerLaunchTicket launch = launch_from_event(*launch_event);
  if (hello.run_id != launch.run_id || hello.node_id != launch.node_id ||
      hello.attempt_id != launch.attempt_id ||
      hello.launch_nonce != launch.launch_nonce || hello.adapter != launch.adapter ||
      hello.adapter_version != launch.adapter_version ||
      hello.code_fingerprint != launch.code_fingerprint ||
      hello.concurrency_key != launch.concurrency_key ||
      hello.lease_id != launch.lease_id ||
      hello.fencing_token != launch.fencing_token ||
      hello.last_acked_controller_sequence != 0U ||
      !std::ranges::includes(hello.capabilities,
                             launch.required_capabilities)) {
    throw std::invalid_argument("worker hello disagrees with its launch ticket");
  }
  const std::uint64_t readiness_revision = launch_event->run_revision;
  const std::uint64_t running_revision = readiness_revision + 1U;
  const std::string ready_id = launch_event->event_id + ":ready";
  Event ready{
      .event_id = ready_id,
      .run_id = run_id_,
      .run_revision = readiness_revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0,
      .event_type = "worker.ready",
      .event_version = 1,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0,
      .optimizer_step = std::nullopt,
      .payload = {{"cause_event_id", launch_event->event_id},
                  {"launch_nonce", hello.launch_nonce},
                  {"adapter", hello.adapter},
                  {"adapter_version", hello.adapter_version},
                  {"code_fingerprint", hello.code_fingerprint},
                  {"capabilities", hello.capabilities},
                  {"last_acked_controller_sequence",
                   hello.last_acked_controller_sequence},
                  {"concurrency_key", hello.concurrency_key},
                  {"lease_id", hello.lease_id},
                  {"fencing_token", hello.fencing_token}},
  };
  Event running = acquisition_event(
      state_, launch_event->event_id + ":running", running_revision,
      "run.observed_state_changed",
      {{"state", "running"}, {"cause_event_id", ready_id},
       {"launch_nonce", launch.launch_nonce}},
      now.wall.nanoseconds);
  running.node_id = state_.current_node_id;
  running.attempt_id = state_.current_attempt_id;
  ExecutionState running_state = state_;
  running_state.revision = running_revision;
  const Event entered = entered_event(
      plan_, running_state, launch_event->event_id + ":entered", &running);
  const WorkerReadinessDisposition disposition = journal_.accept_worker_ready(
      launch, hello, now, {ready, running, entered});
  recover();
  return {.disposition = disposition, .launch = launch};
}

Dispatch Controller::prepare_dispatch() {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover the run before dispatching work");
  }
  if (state_.status != ExecutionStatus::running || paused_) {
    throw std::logic_error("controller cannot dispatch work for a terminal run");
  }
  const Node& node =
      plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  const bool managed_run = journal_.event(run_id_ + ":lease-acquired").has_value();
  if (managed_run) {
    throw std::logic_error(
        "managed dispatch requires a typed builtin or fenced worker operation");
  }
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

Dispatch Controller::prepare_dispatch(const AuthorityTimeSample& now) {
  recover();
  if (state_.status != ExecutionStatus::running || paused_) {
    throw std::logic_error("controller cannot dispatch work for an inactive run");
  }
  const auto launch_event = journal_.event(worker_launch_event_id(state_));
  if (!launch_event ||
      !journal_.event(launch_event->event_id + ":ready")) {
    throw std::logic_error("fenced dispatch requires verified worker readiness");
  }
  const WorkerLaunchTicket launch = launch_from_event(*launch_event);
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
  return journal_.prepare_fenced_dispatch(
      dispatch, dispatch_prepared_event(dispatch), launch, now);
}

WorkerInvocationSpec Controller::bind_worker_invocation(
    const WorkerInvocationSpec& invocation,
    const WorkerSessionIdentity& identity, const AuthorityTimeSample& now) {
  recover();
  const std::string dispatch_id = dispatch_id_for(state_);
  const auto dispatch = journal_.dispatch(dispatch_id);
  if (state_.status != ExecutionStatus::running || paused_ || !dispatch ||
      dispatch->status != DispatchStatus::prepared ||
      invocation.run_id != run_id_ || invocation.plan_hash != plan_.plan_hash ||
      invocation.plan_revision != kInitialPlanRevision ||
      invocation.node_id != state_.current_node_id ||
      invocation.attempt_id != state_.current_attempt_id ||
      invocation.dispatch_id != dispatch_id) {
    throw std::logic_error(
        "worker invocation does not describe the active prepared dispatch");
  }
  const Event event{
      .event_id = dispatch_id + ":invocation",
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0U,
      .event_type = "worker.invocation_bound",
      .event_version = 1U,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0U,
      .optimizer_step = std::nullopt,
      .payload = {
          {"canonical_invocation_json",
           worker_invocation_canonical_json(invocation)},
          {"dispatch_id", dispatch_id},
          {"invocation_digest", invocation.invocation_digest},
      },
  };
  (void)journal_.bind_worker_invocation(invocation, identity, now, event);
  recover();
  const auto durable = journal_.worker_invocation(dispatch_id);
  if (!durable)
    throw std::runtime_error(
        "durable worker invocation disappeared after commit");
  return *durable;
}

const ExecutionState& Controller::handle_event(const Event& input) {
  return handle_event_impl(input, std::nullopt, std::nullopt);
}

const ExecutionState& Controller::handle_event(
    const Event& input, const WorkerSessionIdentity& identity,
    const AuthorityTimeSample& now) {
  return handle_event_impl(input, identity, now);
}

const ExecutionState& Controller::record_worker_observation(
    const Event& input, const WorkerSessionIdentity& identity,
    const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before recording worker observations");
  }
  recover();
  require_worker_observation_shape(input);
  const auto stored = journal_.event(input.event_id);
  if (stored && !same_worker_event(*stored, input)) {
    throw std::invalid_argument(
        "worker observation event_id already has different content");
  }
  const bool paused_heartbeat =
      paused_ && input.event_type == "worker.heartbeat" &&
      input.payload.value("phase", std::string{}) == "paused";
  if (input.run_id != run_id_ || input.run_revision != state_.revision ||
      input.plan_revision != kInitialPlanRevision ||
      input.node_id != state_.current_node_id ||
      input.attempt_id != state_.current_attempt_id ||
      (paused_ && !paused_heartbeat) ||
      state_.status != ExecutionStatus::running ||
      identity.run_id != input.run_id || identity.node_id != input.node_id ||
      identity.attempt_id != input.attempt_id ||
      input.event_id != worker_observation_event_id(
                            input, dispatch_id_for(state_))) {
    throw OperationPreconditionError(
        "worker observation is stale for the active controller attempt");
  }
  const auto dispatch = journal_.dispatch(dispatch_id_for(state_));
  if (!dispatch || dispatch->status != DispatchStatus::prepared) {
    throw OperationPreconditionError(
        "worker observation has no active prepared dispatch");
  }
  journal_.append_fenced_worker_observation(input, identity, now);
  return recover();
}

const ExecutionState& Controller::complete_artifact_validation(
    ArtifactValidationOutcome outcome, const AuthorityTimeSample& now) {
  return complete_managed_builtin(
      "validate_artifact",
      outcome == ArtifactValidationOutcome::valid ? "artifact.validated"
                                                   : "artifact.invalid",
      false, now);
}

const ExecutionState& Controller::release_managed_resources(
    const AuthorityTimeSample& now) {
  return complete_managed_builtin("release_resources", "resource.released",
                                  true, now);
}

const ExecutionState& Controller::complete_managed_builtin(
    std::string_view expected_operation, std::string event_type,
    bool release_lease, const AuthorityTimeSample& now) {
  recover();
  if (state_.status != ExecutionStatus::running || paused_) {
    throw std::logic_error("managed builtin requires an active running node");
  }
  const Node& node =
      plan_.experiment.spec.workflow.nodes.at(state_.current_node_id);
  const Component& component =
      plan_.experiment.spec.components.at(node.invoke.component);
  if (component.runtime != ComponentRuntime::builtin ||
      component.adapter != "trainvm.core" ||
      node.invoke.operation != expected_operation ||
      (expected_operation == "validate_artifact" &&
       event_type != "artifact.validated" && event_type != "artifact.invalid") ||
      (expected_operation == "release_resources" &&
       (event_type != "resource.released" || !release_lease)) ||
      (expected_operation != "validate_artifact" &&
       expected_operation != "release_resources")) {
    throw std::logic_error(
        "managed builtin call does not match the active trainvm.core operation");
  }
  const std::string& concurrency_key =
      plan_.experiment.spec.workspace.concurrency_key;
  const auto active = journal_.active_lease(concurrency_key, now);
  const auto acquisition =
      active ? journal_.lease_acquisition_event(run_id_, active->lease_id)
             : std::nullopt;
  if (!active || !acquisition || active->owner_run_id != run_id_ ||
      active->lease_id !=
          acquisition->payload.value("lease_id", std::string{}) ||
      active->fencing_token !=
          acquisition->payload.value("fencing_token", std::uint64_t{}) ||
      active->clock_domain !=
          acquisition->payload.value("clock_domain", std::string{}) ||
      active->boot_id != acquisition->payload.value("boot_id", std::string{}) ||
      active->acquired_boottime_ns !=
          acquisition->payload.value("acquired_boottime_ns", std::int64_t{})) {
    throw OperationPreconditionError(
        "managed builtin no longer owns its acquired lease fence");
  }

  const Dispatch dispatch{
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
  const Dispatch prepared =
      journal_.prepare_dispatch(dispatch, dispatch_prepared_event(dispatch));
  const auto projection = journal_.projection(run_id_);
  if (!projection || projection->observed_state != "running" ||
      projection->current_node_id != state_.current_node_id ||
      projection->current_attempt_id != state_.current_attempt_id) {
    throw std::runtime_error(
        "managed builtin dispatch has no matching active projection");
  }
  nlohmann::json payload = nlohmann::json::object();
  if (release_lease) {
    payload = {{"concurrency_key", active->concurrency_key},
               {"lease_id", active->lease_id},
               {"fencing_token", active->fencing_token}};
  }
  const Event cause{
      .event_id = prepared.dispatch_id + ":builtin-result",
      .run_id = run_id_,
      .run_revision = state_.revision,
      .plan_revision = kInitialPlanRevision,
      .node_id = state_.current_node_id,
      .attempt_id = state_.current_attempt_id,
      .worker_sequence = 0,
      .event_type = std::move(event_type),
      .event_version = 1,
      .wall_time_ns = now.wall.nanoseconds,
      .monotonic_time_ns = 0,
      .optimizer_step = projection->optimizer_step == 0U
                            ? std::nullopt
                            : std::optional<std::uint64_t>{
                                  projection->optimizer_step},
      .payload = std::move(payload),
  };
  const TransitionResult result = advance_execution(plan_, state_, cause);
  std::vector<Event> batch{cause, transitioned_event(cause, result)};
  if (result.state.status == ExecutionStatus::running) {
    const Node& target_node =
        plan_.experiment.spec.workflow.nodes.at(result.state.current_node_id);
    const Component& target_component =
        plan_.experiment.spec.components.at(target_node.invoke.component);
    if (target_component.runtime != ComponentRuntime::builtin) {
      batch.push_back(
          dispatch_completed_event(dispatch, cause, result.state.revision));
      batch.push_back(Event{
          .event_id = cause.event_id + ":acquiring",
          .run_id = run_id_,
          .run_revision = result.state.revision + 1U,
          .plan_revision = kInitialPlanRevision,
          .node_id = result.state.current_node_id,
          .attempt_id = result.state.current_attempt_id,
          .worker_sequence = 0,
          .event_type = "run.observed_state_changed",
          .event_version = 1,
          .wall_time_ns = now.wall.nanoseconds,
          .monotonic_time_ns = 0,
          .optimizer_step = cause.optimizer_step,
          .payload = {{"state", "acquiring"},
                      {"cause_event_id", cause.event_id}},
      });
    } else {
      batch.push_back(
          entered_event(plan_, result.state, cause.event_id + ":node-entered", &cause));
      batch.push_back(
          dispatch_completed_event(dispatch, cause, result.state.revision));
    }
  } else {
    batch.push_back(terminal_event(cause, result.state));
    batch.push_back(
        dispatch_completed_event(dispatch, cause, result.state.revision));
  }
  journal_.complete_managed_builtin_dispatch(prepared, *active, now,
                                             release_lease, batch);
  return recover();
}

const ExecutionState& Controller::handle_event_impl(
    const Event& input,
    const std::optional<WorkerSessionIdentity>& identity,
    std::optional<AuthorityTimeSample> now) {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover the run before handling events");
  }
  if (identity) {
    recover();
  }
  const auto stored = journal_.event(input.event_id);
  if (stored && !same_worker_event(*stored, input)) {
    throw std::invalid_argument(
        "event_id already exists with different worker content");
  }
  const std::string authority_node_id =
      stored ? input.node_id : state_.current_node_id;
  const auto authority_node =
      plan_.experiment.spec.workflow.nodes.find(authority_node_id);
  if (authority_node == plan_.experiment.spec.workflow.nodes.end()) {
    throw std::invalid_argument("worker event names an unknown active node");
  }
  const Component& active_component =
      plan_.experiment.spec.components.at(authority_node->second.invoke.component);
  const bool managed_run =
      journal_.event(run_id_ + ":lease-acquired").has_value();
  const bool worker_authority =
      managed_run && active_component.runtime != ComponentRuntime::builtin;
  if (managed_run && active_component.runtime == ComponentRuntime::builtin &&
      !identity) {
    throw std::logic_error(
        "managed builtin results require an operation-specific controller API");
  }
  const std::string launch_id = input.run_id + ":worker-launch:" + input.node_id +
                                ":" + input.attempt_id;
  const auto ready = journal_.event(launch_id + ":ready");
  if (worker_authority) {
    if (!ready) {
      throw std::invalid_argument(
          "worker event has no durable readiness for the active external node");
    }
    if (!identity || !now || identity->run_id != input.run_id ||
        identity->node_id != input.node_id ||
        identity->attempt_id != input.attempt_id ||
        identity->launch_nonce !=
            ready->payload.value("launch_nonce", std::string{}) ||
        identity->concurrency_key !=
            ready->payload.value("concurrency_key", std::string{}) ||
        identity->lease_id != ready->payload.value("lease_id", std::string{}) ||
        identity->fencing_token !=
            ready->payload.value("fencing_token", std::uint64_t{})) {
      throw std::invalid_argument("worker event lacks its accepted session identity");
    }
    const auto active = journal_.active_lease(identity->concurrency_key, *now);
    if (!active || active->owner_run_id != identity->run_id ||
        active->lease_id != identity->lease_id ||
        active->fencing_token != identity->fencing_token) {
      throw OperationPreconditionError(
          "worker event session no longer owns its active lease");
    }
  } else if (identity) {
    throw std::invalid_argument(
        "worker session identity is invalid outside managed external execution");
  }
  if (stored) {
    return recover();
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
    if (worker_authority) {
      throw OperationPreconditionError(
          "worker event carries a stale run or plan revision");
    }
    throw std::invalid_argument(
        "worker event carries a stale run or plan revision");
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
    const Node& target_node =
        plan_.experiment.spec.workflow.nodes.at(result.state.current_node_id);
    const Component& target_component =
        plan_.experiment.spec.components.at(target_node.invoke.component);
    const bool target_requires_worker =
        managed_run && target_component.runtime != ComponentRuntime::builtin;
    if (target_requires_worker) {
      batch.push_back(
          dispatch_completed_event(*dispatch, cause, result.state.revision));
      batch.push_back(Event{
          .event_id = cause.event_id + ":acquiring",
          .run_id = run_id_,
          .run_revision = result.state.revision + 1U,
          .plan_revision = kInitialPlanRevision,
          .node_id = result.state.current_node_id,
          .attempt_id = result.state.current_attempt_id,
          .worker_sequence = 0,
          .event_type = "run.observed_state_changed",
          .event_version = 1,
          .wall_time_ns = cause.wall_time_ns,
          .monotonic_time_ns = cause.monotonic_time_ns,
          .optimizer_step = cause.optimizer_step,
          .payload = {{"state", "acquiring"},
                      {"cause_event_id", cause.event_id}},
      });
    } else {
      batch.push_back(
          entered_event(plan_, result.state, cause.event_id + ":node-entered", &cause));
      batch.push_back(
          dispatch_completed_event(*dispatch, cause, result.state.revision));
    }
  } else {
    batch.push_back(terminal_event(cause, result.state));
    batch.push_back(
        dispatch_completed_event(*dispatch, cause, result.state.revision));
  }
  if (identity && now) {
    journal_.complete_fenced_dispatch(dispatch->dispatch_id, cause.event_id,
                                      batch, *identity, *now);
  } else {
    journal_.complete_dispatch(dispatch->dispatch_id, cause.event_id, batch);
  }
  return recover();
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
    nlohmann::json diagnostics, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error("controller must create or recover before acknowledging controls");
  }
  if (identity.concurrency_key != plan_.experiment.spec.workspace.concurrency_key) {
    throw std::invalid_argument(
        "control acknowledgement lease key differs from the compiled workspace");
  }
  return journal_.acknowledge_control_command(run_id_, command_id, identity, status, effective_step,
                                              std::move(effective_values),
                                              std::move(diagnostics), now);
}

CheckpointSubmission Controller::request_checkpoint(
    const std::string& idempotency_key,
    std::uint64_t expected_run_revision, const std::string& reason,
    const std::string& author, const std::string& audit_reason) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before requesting a checkpoint");
  }
  if (idempotency_key.empty() || reason.empty() || author.empty() ||
      audit_reason.empty()) {
    throw std::invalid_argument(
        "checkpoint request idempotency key, reason, author, and audit reason are required");
  }
  const std::string command_id =
      "checkpoint-" +
      sha256_hex(nlohmann::json({{"run_id", run_id_},
                                 {"idempotency_key", idempotency_key}})
                     .dump());
  return journal_.submit_checkpoint_command(CheckpointCommand{
      .command_id = command_id,
      .run_id = run_id_,
      .idempotency_key = idempotency_key,
      .expected_run_revision = expected_run_revision,
      .controller_sequence = 0,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .reason = reason,
      .author = author,
      .audit_reason = audit_reason,
      .status = CheckpointCommandStatus::requested,
      .optimizer_step = std::nullopt,
      .artifact_id = "",
      .diagnostics = nlohmann::json::array(),
      .acknowledgement = std::nullopt,
      .acknowledged_at_ns = std::nullopt,
  });
}

CheckpointCommand Controller::acknowledge_checkpoint(
    const std::string& command_id,
    const ControlAcknowledgementIdentity& identity,
    CheckpointCommandStatus status,
    std::optional<std::uint64_t> optimizer_step, std::string artifact_id,
    nlohmann::json diagnostics, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before acknowledging a checkpoint");
  }
  if (identity.concurrency_key !=
      plan_.experiment.spec.workspace.concurrency_key) {
    throw std::invalid_argument(
        "checkpoint acknowledgement lease key differs from the compiled workspace");
  }
  return journal_.acknowledge_checkpoint_command(
      run_id_, command_id, identity, status, optimizer_step,
      std::move(artifact_id), std::move(diagnostics), now);
}

LifecycleSubmission Controller::request_lifecycle(
    LifecycleCommandKind kind, const std::string& idempotency_key,
    std::uint64_t expected_run_revision, bool checkpoint_first,
    bool release_resources, const std::string& author,
    const std::string& reason) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before requesting lifecycle change");
  }
  if (idempotency_key.empty() || author.empty() || reason.empty()) {
    throw std::invalid_argument(
        "lifecycle request idempotency key, author, and reason are required");
  }
  if (kind == LifecycleCommandKind::cancel) {
    throw std::invalid_argument("cancel requests require request_cancel");
  }
  const std::string kind_name =
      kind == LifecycleCommandKind::pause ? "pause" : "resume";
  const std::string command_id =
      kind_name + "-" +
      sha256_hex(nlohmann::json({{"run_id", run_id_},
                                 {"idempotency_key", idempotency_key}})
                     .dump());
  return journal_.submit_lifecycle_command(LifecycleCommand{
      .command_id = command_id,
      .run_id = run_id_,
      .idempotency_key = idempotency_key,
      .expected_run_revision = expected_run_revision,
      .controller_sequence = 0,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .kind = kind,
      .checkpoint_first = checkpoint_first,
      .release_resources = release_resources,
      .cancel_reason = "",
      .graceful_timeout_ns = 0,
      .author = author,
      .reason = reason,
      .status = LifecycleCommandStatus::requested,
      .optimizer_step = std::nullopt,
      .artifact_id = "",
      .diagnostics = nlohmann::json::array(),
      .acknowledgement = std::nullopt,
      .acknowledged_at_ns = std::nullopt,
  });
}

LifecycleSubmission Controller::request_cancel(
    const std::string& idempotency_key,
    std::uint64_t expected_run_revision, std::string cancel_reason,
    std::int64_t graceful_timeout_ns, const std::string& author,
    const std::string& reason) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before requesting cancellation");
  }
  if (idempotency_key.empty() || author.empty() || reason.empty() ||
      cancel_reason.empty() || graceful_timeout_ns < 0) {
    throw std::invalid_argument(
        "cancel idempotency key, author, reasons, and nonnegative timeout are required");
  }
  const std::string command_id =
      "cancel-" +
      sha256_hex(nlohmann::json({{"run_id", run_id_},
                                 {"idempotency_key", idempotency_key}})
                     .dump());
  return journal_.submit_lifecycle_command(LifecycleCommand{
      .command_id = command_id,
      .run_id = run_id_,
      .idempotency_key = idempotency_key,
      .expected_run_revision = expected_run_revision,
      .controller_sequence = 0,
      .plan_revision = kInitialPlanRevision,
      .node_id = "",
      .attempt_id = "",
      .kind = LifecycleCommandKind::cancel,
      .checkpoint_first = false,
      .release_resources = false,
      .cancel_reason = std::move(cancel_reason),
      .graceful_timeout_ns = graceful_timeout_ns,
      .author = author,
      .reason = reason,
      .status = LifecycleCommandStatus::requested,
      .optimizer_step = std::nullopt,
      .artifact_id = "",
      .diagnostics = nlohmann::json::array(),
      .acknowledgement = std::nullopt,
      .acknowledged_at_ns = std::nullopt,
  });
}

LifecycleCommand Controller::acknowledge_lifecycle(
    const std::string& command_id,
    const ControlAcknowledgementIdentity& identity,
    LifecycleCommandStatus status,
    std::optional<std::uint64_t> optimizer_step, std::string artifact_id,
    nlohmann::json diagnostics, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before acknowledging lifecycle change");
  }
  if (identity.concurrency_key !=
      plan_.experiment.spec.workspace.concurrency_key) {
    throw std::invalid_argument(
        "lifecycle acknowledgement lease key differs from the compiled workspace");
  }
  return journal_.acknowledge_lifecycle_command(
      run_id_, command_id, identity, status, optimizer_step,
      std::move(artifact_id), std::move(diagnostics), now);
}

const ExecutionState& Controller::complete_cancellation(
    const std::string& command_id, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before completing cancellation");
  }
  journal_.complete_cancellation(run_id_, command_id, now);
  return recover();
}

const ExecutionState& Controller::complete_resource_releasing_pause(
    const std::string& command_id, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before completing resource pause");
  }
  journal_.complete_resource_releasing_pause(run_id_, command_id, now);
  return recover();
}

LeaseAcquireResult Controller::begin_released_resource_resume(
    const std::string& command_id, const AuthorityTimeSample& now) {
  if (!initialized_) {
    throw std::logic_error(
        "controller must create or recover before resuming released resources");
  }
  recover();
  const auto projection = journal_.projection(run_id_);
  const auto command = journal_.lifecycle_command(command_id);
  if (!projection || !command || command->run_id != run_id_ ||
      command->kind != LifecycleCommandKind::resume ||
      command->status != LifecycleCommandStatus::requested ||
      projection->desired_state != "paused" ||
      projection->observed_state != "paused" ||
      projection->run_revision != command->expected_run_revision ||
      projection->current_node_id != command->node_id ||
      projection->current_attempt_id != command->attempt_id ||
      state_.status != ExecutionStatus::running || !paused_) {
    throw OperationPreconditionError(
        "released-resource resume is stale for the paused run");
  }

  std::string pause_command_id;
  const std::vector<Event> history = journal_.events_for_run(run_id_);
  for (auto event = history.rbegin(); event != history.rend(); ++event) {
    if (event->event_type != "run.observed_state_changed" ||
        event->run_revision != projection->run_revision ||
        event->payload.value("state", std::string{}) != "paused") {
      continue;
    }
    const std::string candidate =
        event->payload.value("cause_command_id", std::string{});
    const auto pause = journal_.lifecycle_command(candidate);
    if (pause && pause->kind == LifecycleCommandKind::pause &&
        pause->checkpoint_first && pause->release_resources &&
        pause->status == LifecycleCommandStatus::applied &&
        pause->node_id == command->node_id &&
        pause->attempt_id == command->attempt_id &&
        pause->optimizer_step && !pause->artifact_id.empty()) {
      pause_command_id = candidate;
      break;
    }
  }
  if (pause_command_id.empty()) {
    throw OperationPreconditionError(
        "released-resource resume has no completed checkpoint-first pause");
  }

  const std::string& concurrency_key =
      plan_.experiment.spec.workspace.concurrency_key;
  const std::string lease_id =
      "lease-" +
      sha256_hex(nlohmann::json({{"run_id", run_id_},
                                 {"plan_hash", plan_.plan_hash},
                                 {"concurrency_key", concurrency_key},
                                 {"resume_command_id", command_id}})
                     .dump());
  const std::int64_t timeout_seconds =
      plan_.experiment.spec.resources.lease_timeout_seconds.value_or(30);
  if (timeout_seconds <= 0 ||
      timeout_seconds >
          std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL) {
    throw std::runtime_error(
        "compiled plan has an invalid resource lease timeout");
  }
  LeaseAcquireResult result = journal_.acquire_lease(
      concurrency_key, run_id_, lease_id, now,
      timeout_seconds * 1'000'000'000LL);
  if (result.status == LeaseAcquireStatus::busy) return result;

  ExecutionState restart_base = state_;
  ++restart_base.revision;  // desired running precedes the attempt restart.
  const ExecutionState restarted = restart_execution_attempt(restart_base);
  (void)journal_.apply_released_resource_resume(
      run_id_, command_id, pause_command_id, result.lease,
      restarted.current_attempt_id, now);
  recover();
  return result;
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
