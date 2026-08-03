#include "trainvm/authority_time.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(!std::is_same_v<trainvm::WallTime, trainvm::BootTime>);
static_assert(!std::is_convertible_v<trainvm::WallTime, trainvm::BootTime>);
static_assert(!std::is_convertible_v<trainvm::BootTime, trainvm::WallTime>);

constexpr const char* kBootOne = "11111111-1111-1111-1111-111111111111";
constexpr const char* kBootTwo = "22222222-2222-2222-2222-222222222222";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, const std::string& message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(message);
}

trainvm::AuthorityClock scripted_clock(
    std::vector<trainvm::AuthorityTimeSample> samples) {
  return trainvm::AuthorityClock(
      [samples = std::move(samples), index = std::size_t{0}]() mutable {
        if (index >= samples.size()) {
          throw std::runtime_error("scripted authority clock exhausted");
        }
        return samples[index++];
      });
}

}  // namespace

int main() {
  try {
    auto clock = scripted_clock({
        {.wall = {100}, .boot = {10}, .boot_id = kBootOne},
        {.wall = {50}, .boot = {11}, .boot_id = kBootOne},
        {.wall = {50}, .boot = {11}, .boot_id = kBootOne},
    });
    require(clock.sample().boot.nanoseconds == 10 &&
                clock.sample().boot.nanoseconds == 11 &&
                clock.sample().boot.nanoseconds == 11,
            "wall regression must not affect nondecreasing BOOTTIME liveness");

    auto regressed = scripted_clock({
        {.wall = {100}, .boot = {10}, .boot_id = kBootOne},
        {.wall = {101}, .boot = {9}, .boot_id = kBootOne},
        {.wall = {102}, .boot = {12}, .boot_id = kBootOne},
    });
    (void)regressed.sample();
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)regressed.sample(); },
        "BOOTTIME regression must fail closed");
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)regressed.sample(); },
        "BOOTTIME regression must poison the authority until restart");

    auto rebooted = scripted_clock({
        {.wall = {100}, .boot = {10}, .boot_id = kBootOne},
        {.wall = {101}, .boot = {11}, .boot_id = kBootTwo},
        {.wall = {102}, .boot = {12}, .boot_id = kBootOne},
    });
    (void)rebooted.sample();
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)rebooted.sample(); },
        "boot identity changes must fail closed");
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)rebooted.sample(); },
        "boot identity changes must poison the authority until restart");

    for (const auto& malformed : std::vector<trainvm::AuthorityTimeSample>{
             {.wall = {-1}, .boot = {0}, .boot_id = kBootOne},
             {.wall = {0}, .boot = {-1}, .boot_id = kBootOne},
             {.wall = {0}, .boot = {0}, .boot_id = "NOT-A-BOOT-ID"},
         }) {
      auto invalid = scripted_clock({malformed});
      require_throws<trainvm::AuthorityClockError>(
          [&] { (void)invalid.sample(); },
          "malformed authority samples must fail closed");
    }

    require_throws<std::invalid_argument>(
        [] {
          trainvm::AuthorityClock invalid(trainvm::AuthorityClock::Source{});
        },
        "empty authority clock source must be rejected");

    std::size_t nonstandard_source_calls = 0;
    trainvm::AuthorityClock nonstandard_exception_clock(
        [&nonstandard_source_calls]() -> trainvm::AuthorityTimeSample {
          ++nonstandard_source_calls;
          throw 7;
        });
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)nonstandard_exception_clock.sample(); },
        "non-standard source exceptions must fail closed");
    require_throws<trainvm::AuthorityClockError>(
        [&] { (void)nonstandard_exception_clock.sample(); },
        "non-standard source exceptions must permanently poison the authority");
    require(nonstandard_source_calls == 1U,
            "a poisoned authority clock must never invoke its source again");

    trainvm::AuthorityClock local;
    const auto local_sample = local.sample();
    require(local_sample.wall.nanoseconds > 0 &&
                local_sample.boot.nanoseconds > 0 &&
                local_sample.boot_id.size() == 36U,
            "local authority clock must produce typed Linux clock evidence");
  } catch (const std::exception& exception) {
    std::cerr << "authority_time_tests: " << exception.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
