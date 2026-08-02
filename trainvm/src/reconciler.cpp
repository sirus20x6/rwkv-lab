#include "trainvm/reconciler.hpp"

#include <stdexcept>
#include <utility>

#include "trainvm/controller.hpp"

namespace trainvm {
namespace {

const TrainingComponentRegistry& empty_training_component_registry() {
  static const TrainingComponentRegistry registry({});
  return registry;
}

}  // namespace

HostGrantSagaReconciler::HostGrantSagaReconciler(
    Journal& journal, IHostGrantClient& host,
    IHostGrantSagaFaultInjector* faults)
    : journal_(journal), host_(host), faults_(faults) {}

void HostGrantSagaReconciler::fault(HostGrantSagaFaultPoint point) const {
  if (faults_ != nullptr) faults_->hit(point);
}

HostGrantSagaSnapshot HostGrantSagaReconciler::reconcile_request(
    const ResourceBundleRequest& request, const AuthorityTimeSample& now) {
  HostGrantSagaSnapshot saga =
      journal_.record_host_resource_request(request, now);
  if (saga.grant || saga.busy_outcome_digest) {
    fault(HostGrantSagaFaultPoint::replay_boundary);
    return saga;
  }
  fault(HostGrantSagaFaultPoint::journal_before_host);
  const BundleRequestResult result = host_.request_bundle(request);
  if (result.status == BundleRequestStatus::busy) {
    if (result.grant) {
      throw std::runtime_error("busy host bundle result contains a grant");
    }
    fault(HostGrantSagaFaultPoint::host_before_journal);
    saga = journal_.record_host_busy_outcome(request.request_id,
                                             result.outcome_digest, now);
    fault(HostGrantSagaFaultPoint::replay_boundary);
    return saga;
  }
  if (result.status != BundleRequestStatus::granted) {
    throw std::runtime_error("host returned an unknown bundle request status");
  }
  if (!result.grant) {
    throw std::runtime_error("granted host bundle result has no receipt");
  }
  if (result.outcome_digest != result.grant->receipt_digest) {
    throw std::runtime_error(
        "host bundle outcome digest disagrees with its grant receipt");
  }
  fault(HostGrantSagaFaultPoint::host_before_journal);
  saga = journal_.record_host_grant_receipt(*result.grant);
  fault(HostGrantSagaFaultPoint::replay_boundary);
  return saga;
}

HostGrantSagaSnapshot HostGrantSagaReconciler::reconcile_release(
    const std::string& request_id, const ResourceReleaseRequest& release,
    const AuthorityTimeSample& now) {
  HostGrantSagaSnapshot saga =
      journal_.record_host_release_intent(request_id, release, now);
  if (saga.release_receipt) {
    fault(HostGrantSagaFaultPoint::replay_boundary);
    return saga;
  }
  fault(HostGrantSagaFaultPoint::journal_before_host);
  const BundleReleaseResult result = host_.release_bundle(release);
  fault(HostGrantSagaFaultPoint::host_before_journal);
  saga = journal_.record_host_release_receipt(request_id, result.receipt);
  fault(HostGrantSagaFaultPoint::replay_boundary);
  return saga;
}

Reconciler::Reconciler(Journal& journal, const AdapterRegistry& registry,
                       std::mutex& authority_mutex,
                       std::function<AuthorityTimeSample()> authority_clock)
    : Reconciler(journal, registry, empty_training_component_registry(),
                 authority_mutex, std::move(authority_clock)) {}

Reconciler::Reconciler(Journal& journal, const AdapterRegistry& registry,
                       const TrainingComponentRegistry& training_components,
                       std::mutex& authority_mutex,
                       std::function<AuthorityTimeSample()> authority_clock)
    : journal_(journal), registry_(registry),
      training_components_(training_components),
      authority_mutex_(authority_mutex),
      authority_clock_(
          std::make_shared<AuthorityClock>(std::move(authority_clock))) {
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
  const AuthorityTimeSample now = authority_clock_->sample();
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
  training_components_.validate_plan(*plan);
  const auto created = journal_.event(run_id + ":created");
  if (!created || created->event_type != "run.created" ||
      !created->payload.contains("submission")) {
    throw std::runtime_error("run has no durable adapter lock identity");
  }
  registry_.validate_submission_lock(
      *plan, created->payload.at("submission"));
  training_components_.validate_submission_lock(
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
    const LeaseAcquireResult acquisition = controller.begin_acquisition(now);
    result.disposition =
        acquisition.status == LeaseAcquireStatus::busy
            ? ReconcileDisposition::lease_busy
            : ReconcileDisposition::lease_acquired;
    return result;
  }

  if (current->desired_state == "paused" &&
      current->observed_state == "paused") {
    std::optional<LifecycleCommand> resume;
    for (const LifecycleCommand& command :
         journal_.pending_lifecycle_commands(run_id, 0U)) {
      if (command.kind != LifecycleCommandKind::resume) continue;
      if (resume) {
        throw OperationPreconditionError(
            "paused run has more than one pending resume command");
      }
      resume = command;
    }
    if (resume) {
      const LeaseAcquireResult acquisition =
          controller.begin_released_resource_resume(
              resume->command_id, now);
      result.disposition =
          acquisition.status == LeaseAcquireStatus::busy
              ? ReconcileDisposition::lease_busy
              : ReconcileDisposition::lease_acquired;
      return result;
    }
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
            controller.begin_acquisition(now);
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
        training_components_.augment_worker_launch_request(
            registry_.worker_launch_request(component,
                                            node.invoke.operation),
            node.invoke.training);
    result.launch = controller.prepare_worker_launch(request, now);
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
    (void)controller.release_managed_resources(now);
    result.disposition = ReconcileDisposition::builtin_completed;
    return result;
  }
  throw AdapterResolutionError(
      "authority has no typed executor for the active builtin operation");
}

}  // namespace trainvm
