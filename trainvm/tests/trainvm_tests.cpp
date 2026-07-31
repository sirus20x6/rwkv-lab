#include "trainvm/document.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/model.hpp"
#include "trainvm/reflection_json.hpp"
#include "trainvm/v1/trainvm.pb.h"

#include <sqlite3.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

int failures = 0;

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

  auto fixture = load_fixture();
  auto result = trainvm::compile_document(fixture);
  check(result.valid(), "MageFlow fixture compiles");
  if (!result.valid()) {
    std::cerr << trainvm::diagnostics_json(result.diagnostics).dump(2) << '\n';
    return;
  }
  check(result.plan->experiment.spec.workflow.nodes.size() == 7U, "compiled plan has seven nodes");
  check(result.plan->experiment.spec.controls.catalog.size() == 4U, "compiled plan has four controls");
  check(result.plan->plan_hash == "783d2860b51374138e7352d39607cb07254c3b774f9d776946a6f2b5e6ad468c",
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

  auto bad_enum = fixture;
  bad_enum["spec"]["resources"]["accelerators"]["vendor"] = "cuda-ish";
  auto bad_enum_result = trainvm::compile_document(bad_enum);
  check(!bad_enum_result.valid() && has_diagnostic(bad_enum_result, "enum.unknown"),
        "reflected enum decoder rejects unknown values");

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
  std::string wire;
  check(summary.SerializeToString(&wire), "Protobuf RunSummary serializes");
  trainvm::v1::RunSummary decoded;
  check(decoded.ParseFromString(wire), "Protobuf RunSummary parses");
  check(decoded.identity().run_id() == "run-1" && decoded.identity().revision() == 4U,
        "generated C++ protocol types preserve the run identity");
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
  check(journal.append(created) == 1U, "first event receives journal sequence one");
  check(journal.append(created) == 1U, "identical event append is idempotent");
  check(journal.event_count() == 1U, "idempotent append does not duplicate the event");

  auto conflicting = created;
  conflicting.payload["observed_state"] = "running";
  bool conflict_rejected = false;
  try {
    (void)journal.append(conflicting);
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
  journal.append(entered);
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
  journal.append(heartbeat);
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
  journal.append(desired);

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

  auto regressed = heartbeat;
  regressed.event_id = "event-regressed";
  regressed.worker_sequence = 1;
  bool sequence_rejected = false;
  try {
    (void)journal.append(regressed);
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
    (void)journal.append(stale_revision);
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

}  // namespace

int main() {
  try {
    test_reflection_and_compiler();
    test_wire_contract();
    test_fsm();
    test_journal();
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
