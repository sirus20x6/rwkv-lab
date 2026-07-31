#include "trainvm/authority_time.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string_view>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <utility>

namespace trainvm {
namespace {

bool canonical_uuid(std::string_view value) {
  if (value.size() != 36U || value[8U] != '-' || value[13U] != '-' ||
      value[18U] != '-' || value[23U] != '-') {
    return false;
  }
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (index == 8U || index == 13U || index == 18U || index == 23U) continue;
    const char character = value[index];
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

std::string local_boot_id() {
  constexpr const char* path = "/proc/sys/kernel/random/boot_id";
  const int descriptor = ::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw AuthorityClockError("could not open Linux boot identity: " +
                              std::string(std::strerror(errno)));
  }
  struct CloseGuard final {
    int descriptor;
    ~CloseGuard() { (void)::close(descriptor); }
  } guard{descriptor};
  struct stat before {};
  if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_uid != 0U || (before.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    throw AuthorityClockError("Linux boot identity has unsafe file metadata");
  }
  std::array<char, 65U> bytes{};
  const ssize_t count = ::read(descriptor, bytes.data(), bytes.size());
  if (count < 0) {
    throw AuthorityClockError("could not read Linux boot identity: " +
                              std::string(std::strerror(errno)));
  }
  if (count == static_cast<ssize_t>(bytes.size())) {
    throw AuthorityClockError("Linux boot identity exceeds its bounded format");
  }
  struct stat after {};
  if (::fstat(descriptor, &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_mode != after.st_mode ||
      before.st_uid != after.st_uid || before.st_gid != after.st_gid) {
    throw AuthorityClockError("Linux boot identity changed while reading");
  }
  std::string value(bytes.data(), static_cast<std::size_t>(count));
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  if (!canonical_uuid(value)) {
    throw AuthorityClockError("Linux boot identity is not canonical");
  }
  return value;
}

std::int64_t timespec_nanoseconds(const timespec& value,
                                  std::string_view clock_name) {
  if (value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1'000'000'000L ||
      value.tv_sec > std::numeric_limits<std::int64_t>::max() / 1'000'000'000LL) {
    throw AuthorityClockError(std::string(clock_name) +
                              " returned an out-of-range timestamp");
  }
  const std::int64_t seconds = static_cast<std::int64_t>(value.tv_sec);
  const std::int64_t seconds_ns = seconds * 1'000'000'000LL;
  const std::int64_t remainder = static_cast<std::int64_t>(value.tv_nsec);
  if (remainder > std::numeric_limits<std::int64_t>::max() - seconds_ns) {
    throw AuthorityClockError(std::string(clock_name) +
                              " returned an out-of-range timestamp");
  }
  return seconds_ns + remainder;
}

AuthorityTimeSample local_sample() {
  timespec wall{};
  timespec boot{};
  if (::clock_gettime(CLOCK_REALTIME, &wall) != 0) {
    throw AuthorityClockError("CLOCK_REALTIME failed: " +
                              std::string(std::strerror(errno)));
  }
  if (::clock_gettime(CLOCK_BOOTTIME, &boot) != 0) {
    throw AuthorityClockError("CLOCK_BOOTTIME failed: " +
                              std::string(std::strerror(errno)));
  }
  return {
      .wall = {.nanoseconds = timespec_nanoseconds(wall, "CLOCK_REALTIME")},
      .boot = {.nanoseconds = timespec_nanoseconds(boot, "CLOCK_BOOTTIME")},
      .boot_id = local_boot_id(),
  };
}

void validate_sample(const AuthorityTimeSample& sample) {
  if (sample.wall.nanoseconds < 0 || sample.boot.nanoseconds < 0 ||
      !canonical_uuid(sample.boot_id)) {
    throw AuthorityClockError("authority time source returned a malformed sample");
  }
}

}  // namespace

AuthorityClock::AuthorityClock() : AuthorityClock(local_sample) {}

AuthorityClock::AuthorityClock(Source source) : source_(std::move(source)) {
  if (!source_) {
    throw std::invalid_argument("authority clock requires a time source");
  }
}

AuthorityClock::~AuthorityClock() = default;

AuthorityTimeSample AuthorityClock::sample() {
  std::scoped_lock lock(mutex_);
  if (poison_reason_) {
    throw AuthorityClockError(*poison_reason_);
  }
  try {
    AuthorityTimeSample current = source_();
    validate_sample(current);
    if (boot_id_ && *boot_id_ != current.boot_id) {
      throw AuthorityClockError("authority boot identity changed");
    }
    if (last_boot_time_ &&
        current.boot.nanoseconds < last_boot_time_->nanoseconds) {
      throw AuthorityClockError("CLOCK_BOOTTIME regressed within one boot");
    }
    boot_id_ = current.boot_id;
    last_boot_time_ = current.boot;
    return current;
  } catch (const std::exception& exception) {
    poison_reason_ = "authority clock is poisoned until restart: " +
                     std::string(exception.what());
    throw AuthorityClockError(*poison_reason_);
  } catch (...) {
    poison_reason_ =
        "authority clock is poisoned until restart: time source threw a "
        "non-standard exception";
    throw AuthorityClockError(*poison_reason_);
  }
}

}  // namespace trainvm
