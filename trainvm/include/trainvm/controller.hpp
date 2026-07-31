#pragma once

#include <string>

#include "trainvm/document.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/control.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

class Controller {
 public:
  Controller(const CompiledPlan& plan, Journal& journal, std::string run_id);

  const ExecutionState& create();
  const ExecutionState& create_queued(nlohmann::json submission = nlohmann::json::object());
  const ExecutionState& recover();
  LeaseAcquireResult begin_acquisition(std::int64_t now_ns);
  Dispatch prepare_dispatch();
  const ExecutionState& handle_event(const Event& event);
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
};

}  // namespace trainvm
