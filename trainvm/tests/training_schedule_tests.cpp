#include "trainvm/training_schedules.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance,
          const std::string& message) {
  require(std::abs(actual - expected) <= tolerance,
          message + ": expected " + std::to_string(expected) + ", got " +
              std::to_string(actual));
}

template <typename Callable>
void rejected(Callable&& callable, std::string_view expected,
              const std::string& message) {
  try {
    callable();
  } catch (const trainvm::TrainingScheduleError& error) {
    require(std::string_view(error.what()).find(expected) !=
                std::string_view::npos,
            message + ": wrong error: " + error.what());
    return;
  }
  throw std::runtime_error(message + ": input was accepted");
}

void cosine_warmup_boundaries_and_clamping_match_oracle() {
  const trainvm::LinearWarmupCosineSchedule schedule{
      .warmup_steps = 4, .max_steps = 12, .minimum_ratio = 0.1};
  near(trainvm::linear_warmup_cosine_multiplier(0, schedule), 1.0e-8,
       0.0, "cosine step zero uses the warmup floor");
  near(trainvm::linear_warmup_cosine_multiplier(1, schedule), 0.25, 0.0,
       "cosine warmup uses the zero-based step");
  near(trainvm::linear_warmup_cosine_multiplier(3, schedule), 0.75, 0.0,
       "cosine last warmup step remains below the peak");
  near(trainvm::linear_warmup_cosine_multiplier(4, schedule), 1.0, 0.0,
       "cosine warmup boundary reaches the peak");
  near(trainvm::linear_warmup_cosine_multiplier(12, schedule), 0.1,
       1.0e-15, "cosine max step reaches the minimum ratio");
  near(trainvm::linear_warmup_cosine_multiplier(100, schedule), 0.1,
       1.0e-15, "cosine steps beyond max clamp to the minimum ratio");

  const trainvm::LinearWarmupCosineSchedule zero_minimum{
      .warmup_steps = 0, .max_steps = 2, .minimum_ratio = 0.0};
  near(trainvm::linear_warmup_cosine_multiplier(2, zero_minimum), 0.0,
       0.0, "cosine permits a zero minimum ratio");
  const trainvm::LinearWarmupCosineSchedule unit_minimum{
      .warmup_steps = 0, .max_steps = 2, .minimum_ratio = 1.0};
  near(trainvm::linear_warmup_cosine_multiplier(1, unit_minimum), 1.0,
       0.0, "cosine unit minimum remains constant");

  const trainvm::LinearWarmupCosineSchedule warmup_beyond_max{
      .warmup_steps = 4, .max_steps = 2, .minimum_ratio = 0.2};
  near(trainvm::linear_warmup_cosine_multiplier(3, warmup_beyond_max),
       0.75, 0.0, "cosine permits warmup_steps beyond max_steps");
  near(trainvm::linear_warmup_cosine_multiplier(5, warmup_beyond_max),
       0.2, 1.0e-15,
       "cosine uses its one-step denominator when warmup exceeds max");
}

void powercool_warmup_cooldown_and_powers_match_oracle() {
  const trainvm::PowerCoolSchedule warmup_schedule{
      .warmup_steps = 4,
      .max_steps = 10,
      .minimum_ratio = 0.0,
      .cooldown_fraction = 0.2,
      .power = 2.0,
  };
  near(trainvm::powercool_multiplier(0, warmup_schedule), 0.25, 0.0,
       "PowerCool first update uses step plus one");
  near(trainvm::powercool_multiplier(3, warmup_schedule), 1.0, 0.0,
       "PowerCool last warmup update reaches the peak");
  require(trainvm::powercool_multiplier(0, warmup_schedule) != 1.0e-8,
          "PowerCool and cosine retain distinct step-zero conventions");

  const trainvm::PowerCoolSchedule full_cooldown{
      .warmup_steps = 0,
      .max_steps = 4,
      .minimum_ratio = 0.0,
      .cooldown_fraction = 1.0,
      .power = 1.0,
  };
  near(trainvm::powercool_multiplier(0, full_cooldown), 1.0, 0.0,
       "full cooldown starts at the peak");
  near(trainvm::powercool_multiplier(2, full_cooldown), 0.5, 0.0,
       "power one produces a linear cooldown");
  near(trainvm::powercool_multiplier(4, full_cooldown), 0.0, 0.0,
       "full cooldown reaches a zero minimum");
  near(trainvm::powercool_multiplier(40, full_cooldown), 0.0, 0.0,
       "PowerCool steps beyond max clamp to the minimum");

  auto fractional_power = full_cooldown;
  fractional_power.power = 0.5;
  near(trainvm::powercool_multiplier(3, fractional_power), 0.5, 1.0e-15,
       "power below one retains more rate late in cooldown");
  auto cubic_power = full_cooldown;
  cubic_power.power = 3.0;
  near(trainvm::powercool_multiplier(2, cubic_power), 0.125, 0.0,
       "power above one cools more aggressively");

  auto unit_minimum = full_cooldown;
  unit_minimum.minimum_ratio = 1.0;
  unit_minimum.power = 2.0;
  near(trainvm::powercool_multiplier(4, unit_minimum), 1.0, 0.0,
       "PowerCool permits a unit minimum ratio");
  near(trainvm::powercool_multiplier(100, unit_minimum), 1.0, 0.0,
       "unit minimum remains clamped beyond max_steps");
}

void cosine_rejects_every_invalid_domain() {
  const trainvm::LinearWarmupCosineSchedule valid{
      .warmup_steps = 1, .max_steps = 2, .minimum_ratio = 0.1};
  rejected(
      [&] { (void)trainvm::linear_warmup_cosine_multiplier(-1, valid); },
      "schedule step", "cosine rejects negative steps");
  rejected(
      [&] {
        trainvm::validate_linear_warmup_cosine_schedule(
            {.warmup_steps = -1, .max_steps = 2, .minimum_ratio = 0.1});
      },
      "warmup_steps", "cosine rejects negative warmup_steps");
  rejected(
      [&] {
        trainvm::validate_linear_warmup_cosine_schedule(
            {.warmup_steps = 0, .max_steps = 0, .minimum_ratio = 0.1});
      },
      "max_steps", "cosine rejects nonpositive max_steps");
  for (double value :
       {std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity(), -0.1, 1.1}) {
    rejected(
        [&] {
          trainvm::validate_linear_warmup_cosine_schedule(
              {.warmup_steps = 0, .max_steps = 1, .minimum_ratio = value});
        },
        "minimum_ratio", "cosine rejects an invalid minimum ratio");
  }
}

void powercool_rejects_every_invalid_domain() {
  const trainvm::PowerCoolSchedule valid{
      .warmup_steps = 1,
      .max_steps = 2,
      .minimum_ratio = 0.1,
      .cooldown_fraction = 0.2,
      .power = 2.0,
  };
  rejected([&] { (void)trainvm::powercool_multiplier(-1, valid); },
           "step must be non-negative", "PowerCool rejects negative steps");
  rejected(
      [&] {
        auto schedule = valid;
        schedule.warmup_steps = -1;
        trainvm::validate_powercool_schedule(schedule);
      },
      "warmup_steps", "PowerCool rejects negative warmup_steps");
  rejected(
      [&] {
        auto schedule = valid;
        schedule.max_steps = 0;
        trainvm::validate_powercool_schedule(schedule);
      },
      "max_steps", "PowerCool rejects nonpositive max_steps");
  rejected(
      [&] {
        auto schedule = valid;
        schedule.warmup_steps = 3;
        trainvm::validate_powercool_schedule(schedule);
      },
      "cannot exceed", "PowerCool alone rejects warmup beyond max_steps");

  for (double value : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
    rejected(
        [&] {
          auto schedule = valid;
          schedule.minimum_ratio = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "minimum_ratio must be finite",
        "PowerCool rejects non-finite minimum_ratio");
    rejected(
        [&] {
          auto schedule = valid;
          schedule.cooldown_fraction = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "cooldown_fraction must be finite",
        "PowerCool rejects non-finite cooldown_fraction");
    rejected(
        [&] {
          auto schedule = valid;
          schedule.power = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "power must be finite", "PowerCool rejects non-finite power");
  }
  for (double value : {-0.1, 1.1}) {
    rejected(
        [&] {
          auto schedule = valid;
          schedule.minimum_ratio = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "minimum_ratio must be in [0, 1]",
        "PowerCool rejects minimum_ratio outside the unit interval");
  }
  for (double value : {0.0, -0.1, 1.1}) {
    rejected(
        [&] {
          auto schedule = valid;
          schedule.cooldown_fraction = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "cooldown_fraction must be in (0, 1]",
        "PowerCool rejects cooldown_fraction outside its domain");
  }
  for (double value : {0.0, -0.1}) {
    rejected(
        [&] {
          auto schedule = valid;
          schedule.power = value;
          trainvm::validate_powercool_schedule(schedule);
        },
        "power must be positive", "PowerCool rejects nonpositive power");
  }
}

}  // namespace

int main() {
  try {
    const std::vector<std::pair<std::string, void (*)()>> tests{
        {"cosine boundaries",
         cosine_warmup_boundaries_and_clamping_match_oracle},
        {"PowerCool values", powercool_warmup_cooldown_and_powers_match_oracle},
        {"cosine rejection", cosine_rejects_every_invalid_domain},
        {"PowerCool rejection", powercool_rejects_every_invalid_domain},
    };
    for (const auto& [name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    std::cout << "All training schedule tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
