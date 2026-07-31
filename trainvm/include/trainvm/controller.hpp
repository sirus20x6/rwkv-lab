#pragma once

#include <string>

#include "trainvm/document.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

class Controller {
 public:
  Controller(const CompiledPlan& plan, Journal& journal, std::string run_id);

  const ExecutionState& create();
  const ExecutionState& recover();
  Dispatch prepare_dispatch();
  const ExecutionState& handle_event(const Event& event);

  [[nodiscard]] const ExecutionState& state() const;
  [[nodiscard]] const CompiledPlan& plan() const;
  [[nodiscard]] bool initialized() const;

 private:
  const CompiledPlan& plan_;
  Journal& journal_;
  std::string run_id_;
  ExecutionState state_;
  bool initialized_{};
};

}  // namespace trainvm
