#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "trainvm/authority_time.hpp"
#include "trainvm/lease.hpp"

namespace trainvm {

class Journal;

class LeaseRenewalCoordinatorError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct LeaseRenewalTarget final {
  ResourceLease lease;
  std::int64_t timeout_ns{};
  std::int64_t renewal_margin_ns{};

  bool operator==(const LeaseRenewalTarget&) const = default;
};

enum class LeaseRenewalTickStatus { not_due, renewed, replayed, lost };

struct LeaseRenewalTickResult final {
  std::string concurrency_key;
  LeaseRenewalTickStatus status{};
  std::optional<LeaseRenewalReceipt> receipt;

  bool operator==(const LeaseRenewalTickResult&) const = default;
};

struct LeaseRenewalCoordinatorSnapshot final {
  std::size_t tracked_count{};
  bool poisoned{};
  std::string poison_reason;

  bool operator==(const LeaseRenewalCoordinatorSnapshot&) const = default;
};

// A production-shaped but deliberately manual lease-renewal state machine.
// Service integration may call tick from a future supervisor, but this class
// owns no thread, timer, process, or worker-launch capability. A restarted
// coordinator must track the journal's current active lease; it must not
// reconstruct a target from a stale pre-renewal expiry.
class LeaseRenewalCoordinator final {
 public:
  static constexpr std::int64_t kMaximumTimeoutNs =
      24LL * 60LL * 60LL * 1'000'000'000LL;
  static constexpr std::int64_t kMaximumMarginNs =
      5LL * 60LL * 1'000'000'000LL;
  static constexpr std::size_t kMaximumTrackedTargets = 256U;

  LeaseRenewalCoordinator(Journal& journal,
                          std::shared_ptr<AuthorityClock> authority_clock);

  void track(LeaseRenewalTarget target);
  bool untrack(const std::string& concurrency_key);
  [[nodiscard]] std::vector<LeaseRenewalTickResult> tick();
  [[nodiscard]] bool poisoned() const;
  [[nodiscard]] std::size_t tracked_count() const;
  [[nodiscard]] LeaseRenewalCoordinatorSnapshot snapshot() const;

 private:
  [[noreturn]] void poison(std::string reason);

  Journal& journal_;
  std::shared_ptr<AuthorityClock> authority_clock_;
  mutable std::mutex mutex_;
  std::map<std::string, LeaseRenewalTarget> targets_;
  std::optional<std::string> poison_reason_;
};

}  // namespace trainvm
