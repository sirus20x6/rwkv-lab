#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"
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

  Event next(const CompiledPlan& plan, const ExecutionState& state);
  [[nodiscard]] std::size_t remaining() const;

 private:
  std::vector<FakeOutcome> outcomes_;
  std::size_t cursor_{};
  std::uint64_t event_number_{};
};

}  // namespace trainvm
