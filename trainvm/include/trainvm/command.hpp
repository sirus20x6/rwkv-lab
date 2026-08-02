#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

enum class ControlCommandStatus { requested, applied, rejected, restart_required };

enum class CheckpointCommandStatus { requested, applied, rejected };

struct ControlAcknowledgementIdentity {
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string node_id;
  std::string attempt_id;
  std::uint64_t worker_sequence{};

  bool operator==(const ControlAcknowledgementIdentity&) const = default;
};

struct ControlCommand {
  std::string command_id;
  std::string run_id;
  std::string idempotency_key;
  std::uint64_t expected_run_revision{};
  std::uint64_t expected_control_revision{};
  std::uint64_t control_revision{};
  std::uint64_t plan_revision{1};
  ApplyPoint apply_point{ApplyPoint::immediate};
  bool requires_pause{};
  nlohmann::json assignments = nlohmann::json::object();
  std::string author;
  std::string reason;
  ControlCommandStatus status{ControlCommandStatus::requested};
  std::optional<std::uint64_t> effective_step;
  nlohmann::json effective_values = nlohmann::json::object();
  nlohmann::json diagnostics = nlohmann::json::array();
  std::optional<ControlAcknowledgementIdentity> acknowledgement;
  std::optional<std::int64_t> acknowledged_at_ns;

  bool operator==(const ControlCommand&) const = default;
};

struct ControlSubmission {
  ControlCommand command;
  bool inserted{};
};

// Checkpoint-now commands are journal events instead of mutable scheduler
// state.  controller_sequence is the request event's global journal sequence,
// which gives lifecycle and control commands one ordered worker stream without
// conflating that ordering with the independently versioned control document.
struct CheckpointCommand {
  std::string command_id;
  std::string run_id;
  std::string idempotency_key;
  std::uint64_t expected_run_revision{};
  std::uint64_t controller_sequence{};
  std::uint64_t plan_revision{1};
  std::string node_id;
  std::string attempt_id;
  std::string reason;
  std::string author;
  std::string audit_reason;
  CheckpointCommandStatus status{CheckpointCommandStatus::requested};
  std::optional<std::uint64_t> optimizer_step;
  std::string artifact_id;
  nlohmann::json diagnostics = nlohmann::json::array();
  std::optional<ControlAcknowledgementIdentity> acknowledgement;
  std::optional<std::int64_t> acknowledged_at_ns;

  bool operator==(const CheckpointCommand&) const = default;
};

struct CheckpointSubmission {
  CheckpointCommand command;
  bool inserted{};
};

enum class LifecycleCommandKind { pause, resume };
enum class LifecycleCommandStatus { requested, applied, rejected };

struct LifecycleCommand {
  std::string command_id;
  std::string run_id;
  std::string idempotency_key;
  std::uint64_t expected_run_revision{};
  std::uint64_t controller_sequence{};
  std::uint64_t plan_revision{1};
  std::string node_id;
  std::string attempt_id;
  LifecycleCommandKind kind{LifecycleCommandKind::pause};
  bool checkpoint_first{};
  bool release_resources{};
  std::string author;
  std::string reason;
  LifecycleCommandStatus status{LifecycleCommandStatus::requested};
  std::optional<std::uint64_t> optimizer_step;
  std::string artifact_id;
  nlohmann::json diagnostics = nlohmann::json::array();
  std::optional<ControlAcknowledgementIdentity> acknowledgement;
  std::optional<std::int64_t> acknowledged_at_ns;

  bool operator==(const LifecycleCommand&) const = default;
};

struct LifecycleSubmission {
  LifecycleCommand command;
  bool inserted{};
};

}  // namespace trainvm
