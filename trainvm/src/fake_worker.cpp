#include "trainvm/fake_worker.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace trainvm {

FakeWorker::FakeWorker(std::vector<FakeOutcome> outcomes) : outcomes_(std::move(outcomes)) {}

Event FakeWorker::execute(const CompiledPlan& plan, const ExecutionState& state,
                          const Dispatch& dispatch) {
  if (state.status != ExecutionStatus::running) {
    throw std::logic_error("fake worker cannot run a terminal execution");
  }
  if (dispatch.run_id != state.run_id || dispatch.run_revision != state.revision ||
      dispatch.node_id != state.current_node_id || dispatch.attempt_id != state.current_attempt_id ||
      dispatch.status != DispatchStatus::prepared) {
    throw std::logic_error("fake worker received a dispatch that disagrees with controller state");
  }
  if (const auto receipt = receipts_.find(dispatch.dispatch_id); receipt != receipts_.end()) {
    return receipt->second;
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
  if (dispatch.component != node.invoke.component || dispatch.operation != node.invoke.operation) {
    throw std::logic_error("fake worker dispatch operation disagrees with the compiled plan");
  }
  ++cursor_;
  Event event{
      .event_id = dispatch.dispatch_id + ":result",
      .run_id = state.run_id,
      .run_revision = state.revision,
      .plan_revision = 1,
      .node_id = state.current_node_id,
      .attempt_id = state.current_attempt_id,
      .worker_sequence = 1,
      .event_type = outcome.event_type,
      .event_version = 1,
      .wall_time_ns = static_cast<std::int64_t>(cursor_),
      .monotonic_time_ns = static_cast<std::uint64_t>(cursor_),
      .optimizer_step = outcome.optimizer_step,
      .payload = outcome.payload,
  };
  receipts_.emplace(dispatch.dispatch_id, event);
  return event;
}

std::size_t FakeWorker::remaining() const { return outcomes_.size() - cursor_; }

}  // namespace trainvm
