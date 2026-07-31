#pragma once

#include <cstdint>
#include <string>

namespace trainvm {

struct ResourceLease {
  std::string concurrency_key;
  std::string owner_run_id;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::int64_t acquired_at_ns{};
  std::int64_t expires_at_ns{};

  bool operator==(const ResourceLease&) const = default;
};

enum class LeaseAcquireStatus { acquired, already_owned, busy };

struct LeaseAcquireResult {
  LeaseAcquireStatus status{};
  ResourceLease lease;
};

}  // namespace trainvm
