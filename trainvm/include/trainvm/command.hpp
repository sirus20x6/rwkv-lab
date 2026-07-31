#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "trainvm/model.hpp"

namespace trainvm {

enum class ControlCommandStatus { requested, applied, rejected, restart_required };

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

  bool operator==(const ControlCommand&) const = default;
};

}  // namespace trainvm
