#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace trainvm {

enum class DispatchStatus { prepared, completed };

struct Dispatch {
  std::string dispatch_id;
  std::string run_id;
  std::uint64_t run_revision{};
  std::uint64_t plan_revision{};
  std::string node_id;
  std::string attempt_id;
  std::string component;
  std::string operation;
  DispatchStatus status{DispatchStatus::prepared};
  std::optional<std::string> result_event_id;

  bool operator==(const Dispatch&) const = default;
};

}  // namespace trainvm
