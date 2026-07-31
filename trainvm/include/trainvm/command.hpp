#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

enum class ControlCommandStatus { requested, applied, rejected, restart_required };

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

}  // namespace trainvm
