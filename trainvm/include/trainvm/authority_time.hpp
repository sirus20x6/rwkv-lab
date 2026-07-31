#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace trainvm {

struct WallTime final {
  std::int64_t nanoseconds{};

  bool operator==(const WallTime&) const = default;
};

struct BootTime final {
  std::int64_t nanoseconds{};

  bool operator==(const BootTime&) const = default;
};

struct AuthorityTimeSample final {
  WallTime wall;
  BootTime boot;
  std::string boot_id;

  bool operator==(const AuthorityTimeSample&) const = default;
};

class AuthorityClockError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Samples wall time for audit/display and Linux CLOCK_BOOTTIME for lease
// liveness. The two strong types prevent accidental interchange at new API
// boundaries. An authority clock permanently latches its first boot identity
// and fails closed if BOOTTIME regresses.
class AuthorityClock final {
 public:
  using Source = std::function<AuthorityTimeSample()>;

  AuthorityClock();
  explicit AuthorityClock(Source source);

  AuthorityClock(const AuthorityClock&) = delete;
  AuthorityClock& operator=(const AuthorityClock&) = delete;

  [[nodiscard]] AuthorityTimeSample sample();

 private:
  Source source_;
  std::mutex mutex_;
  std::optional<std::string> boot_id_;
  std::optional<BootTime> last_boot_time_;
  std::optional<std::string> poison_reason_;
};

}  // namespace trainvm
