#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"
#include "trainvm/dispatch.hpp"
#include "trainvm/fsm.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

struct FakeOutcome {
  std::string expected_node_id;
  std::string expected_operation;
  std::string event_type;
  nlohmann::json payload = nlohmann::json::object();
  std::optional<std::uint64_t> optimizer_step;
};

class FakeWorker {
 public:
  explicit FakeWorker(std::vector<FakeOutcome> outcomes);

  Event execute(const CompiledPlan& plan, const ExecutionState& state, const Dispatch& dispatch);
  [[nodiscard]] std::size_t remaining() const;

 private:
  std::vector<FakeOutcome> outcomes_;
  std::size_t cursor_{};
  std::map<std::string, Event> receipts_;
};

}  // namespace trainvm
