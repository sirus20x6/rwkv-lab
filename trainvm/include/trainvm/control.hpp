#pragma once

#include <optional>
#include <string>
#include <vector>

#include "trainvm/json.hpp"

#include "trainvm/document.hpp"
#include "trainvm/command.hpp"

namespace trainvm {

struct ControlPatchValidation {
  nlohmann::json assignments = nlohmann::json::object();
  ApplyPoint apply_point{ApplyPoint::immediate};
  bool requires_pause{};
  bool replayed{};
  std::vector<Diagnostic> diagnostics;
  std::optional<ControlCommand> command;

  [[nodiscard]] bool valid() const;
};

ControlPatchValidation validate_control_patch(const CompiledPlan& plan,
                                              const nlohmann::json& assignments,
                                              bool run_started, bool run_paused);

}  // namespace trainvm
