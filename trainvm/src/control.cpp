#include "trainvm/control.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace trainvm {
namespace {

void error(ControlPatchValidation& result, std::string code, std::string path,
           std::string message) {
  result.diagnostics.push_back(
      {Diagnostic::Severity::error, std::move(code), std::move(path), std::move(message)});
}

std::size_t apply_rank(ApplyPoint point) {
  switch (point) {
    case ApplyPoint::immediate:
      return 0;
    case ApplyPoint::next_microbatch:
      return 1;
    case ApplyPoint::next_optimizer_step:
      return 2;
    case ApplyPoint::next_eval:
      return 3;
    case ApplyPoint::next_checkpoint:
      return 4;
    case ApplyPoint::restart:
      return 5;
  }
  return 0;
}

bool value_has_type(ControlType type, const nlohmann::json& value) {
  switch (type) {
    case ControlType::number:
      return value.is_number() && std::isfinite(value.get<double>());
    case ControlType::integer:
      return value.is_number_integer() || value.is_number_unsigned();
    case ControlType::boolean:
      return value.is_boolean();
    case ControlType::string:
    case ControlType::enumeration:
      return value.is_string();
  }
  return false;
}

}  // namespace

bool ControlPatchValidation::valid() const {
  return std::none_of(diagnostics.begin(), diagnostics.end(), [](const Diagnostic& diagnostic) {
    return diagnostic.severity == Diagnostic::Severity::error;
  });
}

ControlPatchValidation validate_control_patch(const CompiledPlan& plan,
                                              const nlohmann::json& assignments,
                                              bool run_started, bool run_paused) {
  ControlPatchValidation result;
  if (!assignments.is_object()) {
    error(result, "control.patch_type", "/assignments", "control assignments must be an object");
    return result;
  }
  if (assignments.empty()) {
    error(result, "control.patch_empty", "/assignments", "control patch must contain an assignment");
    return result;
  }
  bool contains_next_eval = false;
  bool contains_next_checkpoint = false;
  for (auto iterator = assignments.begin(); iterator != assignments.end(); ++iterator) {
    const std::string path = "/assignments/" + iterator.key();
    const auto declared = plan.experiment.spec.controls.catalog.find(iterator.key());
    if (declared == plan.experiment.spec.controls.catalog.end()) {
      error(result, "control.unknown", path, "control is not declared by the compiled plan");
      continue;
    }
    const Control& control = declared->second;
    if (run_started && !control.mutable_after_start) {
      error(result, "control.immutable", path, "control is immutable after the run starts");
      continue;
    }
    if (!value_has_type(control.type, iterator.value())) {
      error(result, "control.value_type", path, "assigned value has the wrong declared type");
      continue;
    }
    if (iterator.value().is_number()) {
      const double value = iterator.value().get<double>();
      if (control.minimum && value < *control.minimum) {
        error(result, "control.minimum", path, "assigned value is below the declared minimum");
        continue;
      }
      if (control.maximum && value > *control.maximum) {
        error(result, "control.maximum", path, "assigned value is above the declared maximum");
        continue;
      }
    }
    if (control.values &&
        std::find(control.values->begin(), control.values->end(), iterator.value()) ==
            control.values->end()) {
      error(result, "control.values", path, "assigned value is not in the declared value set");
      continue;
    }
    const bool needs_pause = control.requires_pause.value_or(false);
    result.requires_pause = result.requires_pause || needs_pause;
    if (needs_pause && run_started && !run_paused) {
      error(result, "control.requires_pause", path,
            "control may only be changed while the run is paused");
      continue;
    }
    if (apply_rank(control.apply) > apply_rank(result.apply_point)) {
      result.apply_point = control.apply;
    }
    contains_next_eval = contains_next_eval || control.apply == ApplyPoint::next_eval;
    contains_next_checkpoint =
        contains_next_checkpoint || control.apply == ApplyPoint::next_checkpoint;
    result.assignments[iterator.key()] = iterator.value();
  }
  if (contains_next_eval && contains_next_checkpoint &&
      result.apply_point != ApplyPoint::restart) {
    error(result, "control.apply_incompatible", "/assignments",
          "next-eval and next-checkpoint controls have no declared common application barrier");
  }
  if (!result.valid()) {
    result.assignments = nlohmann::json::object();
  }
  return result;
}

}  // namespace trainvm
