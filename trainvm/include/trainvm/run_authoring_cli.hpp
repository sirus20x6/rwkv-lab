#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "trainvm/json.hpp"

#include "trainvm/v1/trainvm.pb.h"

namespace trainvm {

struct AuthorRunStreamSummary final {
  bool failed{false};
  std::string plan_hash;
};

class AuthorRunStreamValidator final {
public:
  explicit AuthorRunStreamValidator(
      bool dry_run, std::optional<std::string> expected_plan_hash = std::nullopt);

  void observe(const v1::AuthorRunUpdate &update);
  [[nodiscard]] AuthorRunStreamSummary finish() const;

private:
  bool dry_run_;
  std::optional<std::string> expected_plan_hash_;
  std::string observed_plan_hash_;
  v1::AuthorRunStage last_stage_{v1::AUTHOR_RUN_STAGE_UNSPECIFIED};
  bool saw_terminal_{false};
  bool failed_{false};
};

[[nodiscard]] nlohmann::json author_run_update_json(
    const v1::AuthorRunUpdate &update, std::string_view dashboard_base_url);

} // namespace trainvm
