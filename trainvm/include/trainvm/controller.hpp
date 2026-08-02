#pragma once

#include <string>
#include <string_view>

#include "trainvm/document.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/control.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"
#include "trainvm/worker.hpp"

namespace trainvm {

enum class ArtifactValidationOutcome { valid, invalid };

class Controller {
 public:
  Controller(const CompiledPlan& plan, Journal& journal, std::string run_id);

  const ExecutionState& create_queued(nlohmann::json submission = nlohmann::json::object());
  const ExecutionState& recover();
  LeaseAcquireResult begin_acquisition(const AuthorityTimeSample& now);
  WorkerLaunchTicket prepare_worker_launch(WorkerLaunchRequest request,
                                           const AuthorityTimeSample& now);
  ResolvedLaunchSpec bind_worker_launch(
      const ResolvedLaunch& resolved,
      const HostLaunchRegistry& host_registry,
      const HostIdentity& authority_host, const AuthorityTimeSample& now);
  WorkerReadinessResult accept_worker_hello(WorkerHelloEvidence hello,
                                             const AuthorityTimeSample& now);
  Dispatch prepare_dispatch(const AuthorityTimeSample& now);
  WorkerInvocationSpec bind_worker_invocation(
      const WorkerInvocationSpec& invocation,
      const WorkerSessionIdentity& identity, const AuthorityTimeSample& now);
  const ExecutionState& handle_event(const Event& event,
                                     const WorkerSessionIdentity& identity,
                                     const AuthorityTimeSample& now);
  const ExecutionState& record_worker_observation(
      const Event& event, const WorkerSessionIdentity& identity,
      const AuthorityTimeSample& now);
  const ExecutionState& complete_artifact_validation(
      ArtifactValidationOutcome outcome, const AuthorityTimeSample& now);
  const ExecutionState& release_managed_resources(
      const AuthorityTimeSample& now);
  ControlPatchValidation request_controls(const std::string& idempotency_key,
                                          std::uint64_t expected_run_revision,
                                          std::uint64_t expected_control_revision,
                                          const nlohmann::json& assignments,
                                          const std::string& author,
                                          const std::string& reason);
  ControlCommand acknowledge_controls(const std::string& command_id,
                                      const ControlAcknowledgementIdentity& identity,
                                      ControlCommandStatus status,
                                      std::optional<std::uint64_t> effective_step,
                                      nlohmann::json effective_values,
                                      nlohmann::json diagnostics,
                                      const AuthorityTimeSample& now);
  CheckpointSubmission request_checkpoint(
      const std::string& idempotency_key,
      std::uint64_t expected_run_revision, const std::string& reason,
      const std::string& author, const std::string& audit_reason);
  CheckpointCommand acknowledge_checkpoint(
      const std::string& command_id,
      const ControlAcknowledgementIdentity& identity,
      CheckpointCommandStatus status,
      std::optional<std::uint64_t> optimizer_step,
      std::string artifact_id, nlohmann::json diagnostics,
      const AuthorityTimeSample& now);
  LifecycleSubmission request_lifecycle(
      LifecycleCommandKind kind, const std::string& idempotency_key,
      std::uint64_t expected_run_revision, bool checkpoint_first,
      bool release_resources, const std::string& author,
      const std::string& reason);
  LifecycleSubmission request_cancel(
      const std::string& idempotency_key,
      std::uint64_t expected_run_revision, std::string cancel_reason,
      std::int64_t graceful_timeout_ns, const std::string& author,
      const std::string& reason);
  LifecycleCommand acknowledge_lifecycle(
      const std::string& command_id,
      const ControlAcknowledgementIdentity& identity,
      LifecycleCommandStatus status,
      std::optional<std::uint64_t> optimizer_step,
      std::string artifact_id, nlohmann::json diagnostics,
      const AuthorityTimeSample& now);
  const ExecutionState& complete_cancellation(
      const std::string& command_id, const AuthorityTimeSample& now);
  const ExecutionState& complete_resource_releasing_pause(
      const std::string& command_id, const AuthorityTimeSample& now);
  LeaseAcquireResult begin_released_resource_resume(
      const std::string& command_id, const AuthorityTimeSample& now);

  [[nodiscard]] const ExecutionState& state() const;
  [[nodiscard]] const CompiledPlan& plan() const;
  [[nodiscard]] bool initialized() const;

 private:
  const CompiledPlan& plan_;
  Journal& journal_;
  std::string run_id_;
  ExecutionState state_;
  bool initialized_{};
  bool paused_{};

  // In-process deterministic simulation hooks. The production authority starts
  // queued runs and uses the fenced overloads above.
  const ExecutionState& create();
  Dispatch prepare_dispatch();
  const ExecutionState& handle_event(const Event& event);
  void complete_builtin_admission(const ResourceLease& lease,
                                  const AuthorityTimeSample& now);
  const ExecutionState& handle_event_impl(
      const Event& event, const std::optional<WorkerSessionIdentity>& identity,
      std::optional<AuthorityTimeSample> now);
  const ExecutionState& complete_managed_builtin(
      std::string_view expected_operation, std::string event_type,
      bool release_lease, const AuthorityTimeSample& now);
};

}  // namespace trainvm
