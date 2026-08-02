#pragma once

#include <cstdint>
#include <stdexcept>

namespace trainvm {

class TrainingScheduleError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct LinearWarmupCosineSchedule {
  std::int64_t warmup_steps{};
  std::int64_t max_steps{};
  double minimum_ratio{0.1};

  bool operator==(const LinearWarmupCosineSchedule&) const = default;
};

struct PowerCoolSchedule {
  std::int64_t warmup_steps{};
  std::int64_t max_steps{};
  double minimum_ratio{0.0};
  double cooldown_fraction{0.2};
  double power{2.0};

  bool operator==(const PowerCoolSchedule&) const = default;
};

void validate_linear_warmup_cosine_schedule(
    const LinearWarmupCosineSchedule& schedule);
void validate_powercool_schedule(const PowerCoolSchedule& schedule);

[[nodiscard]] double linear_warmup_cosine_multiplier(
    std::int64_t step, const LinearWarmupCosineSchedule& schedule);
[[nodiscard]] double powercool_multiplier(
    std::int64_t step, const PowerCoolSchedule& schedule);

}  // namespace trainvm
