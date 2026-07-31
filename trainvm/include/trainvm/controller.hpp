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
  LeaseAcquireResult begin_acquisition(std::int64_t now_ns);
  WorkerLaunchTicket prepare_worker_launch(WorkerLaunchRequest request,
                                           std::int64_t now_ns);
  WorkerReadinessResult accept_worker_hello(WorkerHelloEvidence hello,
                                             std::int64_t now_ns);
  Dispatch prepare_dispatch(std::int64_t now_ns);
  const ExecutionState& handle_event(const Event& event,
                                     const WorkerSessionIdentity& identity,
                                     std::int64_t now_ns);
  const ExecutionState& complete_artifact_validation(
      ArtifactValidationOutcome outcome, std::int64_t now_ns);
  const ExecutionState& release_managed_resources(std::int64_t now_ns);
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
                                      nlohmann::json diagnostics);

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
                                  std::int64_t now_ns);
  const ExecutionState& handle_event_impl(
      const Event& event, const std::optional<WorkerSessionIdentity>& identity,
      std::optional<std::int64_t> now_ns);
  const ExecutionState& complete_managed_builtin(
      std::string_view expected_operation, std::string event_type,
      bool release_lease, std::int64_t now_ns);
};

}  // namespace trainvm
