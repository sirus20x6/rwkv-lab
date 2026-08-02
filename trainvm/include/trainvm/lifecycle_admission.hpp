#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "trainvm/adapter_registry.hpp"

namespace trainvm {

// The operator-visible control verbs. Every one of them is admitted only by an
// exact capability the adapter operation declares; none is inferred from
// signals, files, worker output, or the absence of a refusal.
enum class LifecycleControlVerb {
  cancel,
  pause_keep_resources,
  pause_release_resources,
  resume,
  checkpoint_now,
};

// A refusal carries the same stable code the dashboard already renders, so the
// gate can be tested exhaustively without a transport.
struct LifecycleAdmissionRefusal final {
  std::string code;
  std::string message;

  bool operator==(const LifecycleAdmissionRefusal&) const = default;
};

// Returns a refusal when the operation does not declare the requested
// protocol, and nothing when it does. `checkpoint_first` is only meaningful
// for pause; a resource-releasing pause always requires it, because the
// replacement worker can only resume from a durable checkpoint.
[[nodiscard]] std::optional<LifecycleAdmissionRefusal> admit_lifecycle_control(
    const OperationLifecycleCapabilities& lifecycle, LifecycleControlVerb verb,
    bool checkpoint_first);

// True when a paused run of this operation can resume from its checkpoint at
// all. terminal_checkpoint and restart_only operations cannot: they may only
// be restarted, and no caller may present them as trajectory-preserving.
[[nodiscard]] bool resume_preserves_trajectory(ResumeGrade grade);
[[nodiscard]] bool resume_from_checkpoint_supported(ResumeGrade grade);

[[nodiscard]] std::string_view lifecycle_control_verb_name(
    LifecycleControlVerb verb);

}  // namespace trainvm
