#include "trainvm/lifecycle_admission.hpp"

namespace trainvm {
namespace {

LifecycleAdmissionRefusal refusal(std::string code, std::string message) {
  return {.code = std::move(code), .message = std::move(message)};
}

}  // namespace

bool resume_preserves_trajectory(ResumeGrade grade) {
  return grade == ResumeGrade::exact;
}

bool resume_from_checkpoint_supported(ResumeGrade grade) {
  return grade == ResumeGrade::compatible || grade == ResumeGrade::exact;
}

std::string_view lifecycle_control_verb_name(LifecycleControlVerb verb) {
  switch (verb) {
    case LifecycleControlVerb::cancel:
      return "cancel";
    case LifecycleControlVerb::pause_keep_resources:
      return "pause_keep_resources";
    case LifecycleControlVerb::pause_release_resources:
      return "pause_release_resources";
    case LifecycleControlVerb::resume:
      return "resume";
    case LifecycleControlVerb::checkpoint_now:
      return "checkpoint_now";
  }
  return "unknown";
}

std::optional<LifecycleAdmissionRefusal> admit_lifecycle_control(
    const OperationLifecycleCapabilities& lifecycle, LifecycleControlVerb verb,
    bool checkpoint_first) {
  switch (verb) {
    case LifecycleControlVerb::cancel:
      if (!lifecycle.graceful_stop)
        return refusal(
            "cancel.unsupported_by_operation",
            "the active adapter operation does not declare graceful stop");
      return std::nullopt;
    case LifecycleControlVerb::checkpoint_now:
      if (!lifecycle.checkpoint_now)
        return refusal(
            "checkpoint.unsupported_by_operation",
            "the active adapter operation does not declare checkpoint-now");
      return std::nullopt;
    case LifecycleControlVerb::pause_keep_resources:
    case LifecycleControlVerb::pause_release_resources:
    case LifecycleControlVerb::resume:
      break;
  }

  const bool release = verb == LifecycleControlVerb::pause_release_resources;
  const bool pause = verb != LifecycleControlVerb::resume;
  // Resume is admitted by the same pause protocol that produced the paused
  // state; a run cannot be resumed into a protocol its operation never had.
  const bool supported = release ? lifecycle.pause_release_resources
                                 : lifecycle.pause_keep_resources;
  if (!supported ||
      (pause && checkpoint_first && !lifecycle.checkpoint_now)) {
    return refusal("lifecycle.unsupported_by_operation",
                   "the active adapter operation does not declare the "
                   "requested lifecycle protocol");
  }
  return std::nullopt;
}

}  // namespace trainvm
