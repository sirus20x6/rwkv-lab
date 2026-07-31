#include "trainvm/fake_worker.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace trainvm {

FakeWorker::FakeWorker(std::vector<FakeOutcome> outcomes) : outcomes_(std::move(outcomes)) {}

Event FakeWorker::next(const CompiledPlan& plan, const ExecutionState& state) {
  if (state.status != ExecutionStatus::running) {
    throw std::logic_error("fake worker cannot run a terminal execution");
  }
  if (cursor_ >= outcomes_.size()) {
    throw std::out_of_range("fake worker script is exhausted");
  }
  const FakeOutcome& outcome = outcomes_[cursor_];
  if (outcome.expected_node_id != state.current_node_id) {
    throw std::logic_error("fake worker expected node " + outcome.expected_node_id +
                           " but controller requested " + state.current_node_id);
  }
  const Node& node = plan.experiment.spec.workflow.nodes.at(state.current_node_id);
  if (outcome.expected_operation != node.invoke.operation) {
    throw std::logic_error("fake worker expected operation " + outcome.expected_operation +
                           " but plan requested " + node.invoke.operation);
  }
  ++cursor_;
  ++event_number_;
  return Event{
      .event_id = state.run_id + ":fake:" + std::to_string(event_number_),
      .run_id = state.run_id,
      .run_revision = state.revision,
      .plan_revision = 1,
      .node_id = state.current_node_id,
      .attempt_id = state.current_attempt_id,
      .worker_sequence = 1,
      .event_type = outcome.event_type,
      .event_version = 1,
      .wall_time_ns = static_cast<std::int64_t>(event_number_),
      .monotonic_time_ns = event_number_,
      .optimizer_step = outcome.optimizer_step,
      .payload = outcome.payload,
  };
}

std::size_t FakeWorker::remaining() const { return outcomes_.size() - cursor_; }

}  // namespace trainvm
