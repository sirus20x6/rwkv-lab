#include "trainvm/reconciler.hpp"

#include <stdexcept>
#include <utility>

#include "trainvm/controller.hpp"

namespace trainvm {

Reconciler::Reconciler(Journal& journal, const AdapterRegistry& registry,
                       std::mutex& authority_mutex,
                       std::function<std::int64_t()> authority_clock)
    : journal_(journal), registry_(registry),
      authority_mutex_(authority_mutex),
      authority_clock_(std::move(authority_clock)) {
  if (!authority_clock_) {
    throw std::invalid_argument(
        "reconciler requires an authority-owned clock");
  }
}

ReconcileResult Reconciler::step(const std::string& run_id) {
  if (run_id.empty()) {
    throw std::invalid_argument("reconciliation requires a run identity");
  }
  std::scoped_lock lock(authority_mutex_);
  const std::int64_t now_ns = authority_clock_();
  if (now_ns < 0) {
    throw std::runtime_error(
        "reconciler authority clock returned a negative timestamp");
  }
  const auto projection = journal_.projection(run_id);
  if (!projection) {
    throw std::invalid_argument("cannot reconcile an unknown run");
  }
  const auto plan = journal_.compiled_plan(projection->plan_hash);
  if (!plan) {
    throw std::runtime_error(
        "cannot reconcile a run without its persisted compiled plan");
  }
  // Registry authority is checked before any state or lease mutation.
  registry_.validate_plan(*plan);
  const auto created = journal_.event(run_id + ":created");
  if (!created || created->event_type != "run.created" ||
      !created->payload.contains("submission")) {
    throw std::runtime_error("run has no durable adapter lock identity");
  }
  registry_.validate_submission_lock(
      *plan, created->payload.at("submission"));

  Controller controller(*plan, journal_, run_id);
  controller.recover();
  ReconcileResult result{
      .disposition = ReconcileDisposition::no_action,
      .run_id = run_id,
      .launch = std::nullopt,
  };
  auto current = journal_.projection(run_id);
  if (!current) {
    throw std::runtime_error("run projection disappeared during reconciliation");
  }

  if (current->desired_state == "queued" &&
      current->observed_state == "queued") {
    const LeaseAcquireResult acquisition = controller.begin_acquisition(now_ns);
    result.disposition =
        acquisition.status == LeaseAcquireStatus::busy
            ? ReconcileDisposition::lease_busy
            : ReconcileDisposition::lease_acquired;
    return result;
  }

  if (controller.state().status != ExecutionStatus::running) {
    return result;
  }
  if (current->desired_state != "running") {
    return result;
  }
  const Node& node = plan->experiment.spec.workflow.nodes.at(
      controller.state().current_node_id);
  const Component& component =
      plan->experiment.spec.components.at(node.invoke.component);

  if (current->observed_state == "acquiring") {
    if (component.runtime == ComponentRuntime::builtin) {
      if (component.adapter == "trainvm.core" &&
          node.invoke.operation == "acquire_resources") {
        const LeaseAcquireResult acquisition =
            controller.begin_acquisition(now_ns);
        if (acquisition.status == LeaseAcquireStatus::busy) {
          throw std::runtime_error(
              "durable acquiring run unexpectedly lost its admission lease");
        }
        result.disposition = ReconcileDisposition::lease_acquired;
        return result;
      }
      throw std::runtime_error(
          "an unassigned acquiring run cannot target a builtin component");
    }
    const std::string launch_id =
        run_id + ":worker-launch:" + controller.state().current_node_id + ":" +
        controller.state().current_attempt_id;
    const bool replay = journal_.event(launch_id).has_value();
    const WorkerLaunchRequest request =
        registry_.worker_launch_request(component, node.invoke.operation);
    result.launch = controller.prepare_worker_launch(request, now_ns);
    result.disposition = replay ? ReconcileDisposition::launch_replayed
                                : ReconcileDisposition::launch_prepared;
    return result;
  }

  if (current->observed_state != "running") {
    return result;
  }
  if (component.runtime != ComponentRuntime::builtin) {
    result.disposition = ReconcileDisposition::awaiting_worker;
    return result;
  }
  if (component.adapter != "trainvm.core") {
    throw AdapterResolutionError(
        "authority refuses an unrecognized builtin adapter");
  }
  if (node.invoke.operation == "validate_artifact") {
    result.disposition = ReconcileDisposition::input_required;
    return result;
  }
  if (node.invoke.operation == "release_resources") {
    (void)controller.release_managed_resources(now_ns);
    result.disposition = ReconcileDisposition::builtin_completed;
    return result;
  }
  throw AdapterResolutionError(
      "authority has no typed executor for the active builtin operation");
}

}  // namespace trainvm
