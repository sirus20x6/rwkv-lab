#include "trainvm/training_schedules.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace trainvm {
namespace {

double powercool_lr(std::int64_t step, double peak_lr, double min_lr,
                    std::int64_t total_steps, std::int64_t warmup_steps,
                    double cooldown_fraction, double power) {
  if (total_steps <= 0) {
    throw TrainingScheduleError("total_steps must be positive");
  }
  if (step < 0) {
    throw TrainingScheduleError("step must be non-negative");
  }
  if (peak_lr < 0.0 || min_lr < 0.0 || min_lr > peak_lr) {
    throw TrainingScheduleError("require 0 <= min_lr <= peak_lr");
  }
  if (warmup_steps < 0 || warmup_steps > total_steps) {
    throw TrainingScheduleError("warmup_steps must be in [0, total_steps]");
  }
  if (!(0.0 < cooldown_fraction && cooldown_fraction <= 1.0)) {
    throw TrainingScheduleError("cooldown_fraction must be in (0, 1]");
  }
  if (power <= 0.0 || !std::isfinite(power)) {
    throw TrainingScheduleError("power must be finite and positive");
  }

  double warmup = 1.0;
  if (warmup_steps != 0) {
    const std::int64_t warmup_step =
        step < warmup_steps ? step + 1 : warmup_steps;
    warmup = static_cast<double>(warmup_step) /
             static_cast<double>(warmup_steps);
  }

  const double cooldown_start =
      static_cast<double>(total_steps) * (1.0 - cooldown_fraction);
  if (static_cast<double>(step) < cooldown_start) {
    return peak_lr * warmup;
  }

  const double cooldown_span = std::max(
      static_cast<double>(total_steps) * cooldown_fraction, 1.0);
  const double progress = std::clamp(
      (static_cast<double>(step) - cooldown_start) / cooldown_span, 0.0,
      1.0);
  const double cooled =
      min_lr + (peak_lr - min_lr) * std::pow(1.0 - progress, power);
  return cooled * warmup;
}

}  // namespace

void validate_linear_warmup_cosine_schedule(
    const LinearWarmupCosineSchedule& schedule) {
  if (schedule.warmup_steps < 0) {
    throw TrainingScheduleError("warmup_steps must be a nonnegative integer");
  }
  if (schedule.max_steps < 1) {
    throw TrainingScheduleError("max_steps must be a positive integer");
  }
  if (!std::isfinite(schedule.minimum_ratio) ||
      schedule.minimum_ratio < 0.0 || schedule.minimum_ratio > 1.0) {
    throw TrainingScheduleError("minimum_ratio must be finite and in [0, 1]");
  }
}

void validate_powercool_schedule(const PowerCoolSchedule& schedule) {
  if (schedule.warmup_steps < 0) {
    throw TrainingScheduleError("warmup_steps must be a nonnegative integer");
  }
  if (schedule.max_steps < 1) {
    throw TrainingScheduleError("max_steps must be a positive integer");
  }
  if (schedule.warmup_steps > schedule.max_steps) {
    throw TrainingScheduleError("warmup_steps cannot exceed max_steps");
  }
  if (!std::isfinite(schedule.minimum_ratio)) {
    throw TrainingScheduleError("minimum_ratio must be finite");
  }
  if (!std::isfinite(schedule.cooldown_fraction)) {
    throw TrainingScheduleError("cooldown_fraction must be finite");
  }
  if (!std::isfinite(schedule.power)) {
    throw TrainingScheduleError("power must be finite");
  }
  if (schedule.minimum_ratio < 0.0 || schedule.minimum_ratio > 1.0) {
    throw TrainingScheduleError("minimum_ratio must be in [0, 1]");
  }
  if (!(0.0 < schedule.cooldown_fraction &&
        schedule.cooldown_fraction <= 1.0)) {
    throw TrainingScheduleError("cooldown_fraction must be in (0, 1]");
  }
  if (schedule.power <= 0.0) {
    throw TrainingScheduleError("power must be positive");
  }
}

double linear_warmup_cosine_multiplier(
    std::int64_t step, const LinearWarmupCosineSchedule& schedule) {
  if (step < 0) {
    throw TrainingScheduleError(
        "schedule step must be a nonnegative integer");
  }
  validate_linear_warmup_cosine_schedule(schedule);

  if (schedule.warmup_steps != 0 && step < schedule.warmup_steps) {
    return std::max(
        1.0e-8, static_cast<double>(step) /
                    static_cast<double>(schedule.warmup_steps));
  }
  const std::int64_t span =
      std::max(std::int64_t{1}, schedule.max_steps - schedule.warmup_steps);
  const double progress = std::clamp(
      static_cast<double>(step - schedule.warmup_steps) /
          static_cast<double>(span),
      0.0, 1.0);
  return schedule.minimum_ratio +
         (1.0 - schedule.minimum_ratio) * 0.5 *
             (1.0 + std::cos(std::numbers::pi * progress));
}

double powercool_multiplier(std::int64_t step,
                            const PowerCoolSchedule& schedule) {
  if (step < 0) {
    throw TrainingScheduleError("step must be non-negative");
  }
  validate_powercool_schedule(schedule);
  return powercool_lr(step, 1.0, schedule.minimum_ratio, schedule.max_steps,
                      schedule.warmup_steps, schedule.cooldown_fraction,
                      schedule.power);
}

}  // namespace trainvm
