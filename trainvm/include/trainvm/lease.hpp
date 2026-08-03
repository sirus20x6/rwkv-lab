#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "trainvm/authority_time.hpp"

namespace trainvm {

struct ResourceLease {
  static constexpr const char* kBootTimeDomain = "boottime/v1";
  static constexpr const char* kLegacyWallDomain = "legacy-wall/v1";

  std::string concurrency_key;
  std::string owner_run_id;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string clock_domain;
  std::string boot_id;
  std::int64_t acquired_boottime_ns{};
  std::int64_t expires_boottime_ns{};
  std::int64_t acquired_wall_time_ns{};
  std::int64_t expires_wall_time_ns{};

  bool operator==(const ResourceLease&) const = default;
};

enum class LeaseAcquireStatus { acquired, already_owned, busy };

struct LeaseAcquireResult {
  LeaseAcquireStatus status{};
  ResourceLease lease;
};

struct LeaseRenewalReceipt final {
  std::string concurrency_key;
  std::string owner_run_id;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string clock_domain;
  std::string boot_id;
  std::int64_t acquired_boottime_ns{};
  std::int64_t acquired_wall_time_ns{};
  std::int64_t prior_expires_boottime_ns{};
  std::int64_t new_expires_boottime_ns{};
  std::int64_t prior_expires_wall_time_ns{};
  std::int64_t new_expires_wall_time_ns{};
  std::int64_t renewed_boottime_ns{};
  std::int64_t renewed_wall_time_ns{};

  bool operator==(const LeaseRenewalReceipt&) const = default;
};

enum class LeaseRenewalStatus { renewed, replayed, not_owned };

struct LeaseRenewalResult final {
  LeaseRenewalStatus status{};
  std::optional<LeaseRenewalReceipt> receipt;
};

}  // namespace trainvm
