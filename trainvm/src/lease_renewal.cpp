#include "trainvm/lease_renewal.hpp"

#include "trainvm/journal.hpp"

#include <algorithm>
#include <exception>
#include <utility>

namespace trainvm {
namespace {

bool canonical_boot_id(std::string_view value) {
  if (value.size() != 36U) return false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    const char character = value[index];
    const bool separator = index == 8U || index == 13U || index == 18U ||
                           index == 23U;
    if (separator ? character != '-'
                  : !((character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

void validate_target(const LeaseRenewalTarget& target) {
  const ResourceLease& lease = target.lease;
  if (lease.concurrency_key.empty() || lease.owner_run_id.empty() ||
      lease.lease_id.empty() || lease.fencing_token == 0U ||
      lease.clock_domain != ResourceLease::kBootTimeDomain ||
      !canonical_boot_id(lease.boot_id) || lease.acquired_boottime_ns < 0 ||
      lease.expires_boottime_ns <= lease.acquired_boottime_ns ||
      lease.acquired_wall_time_ns < 0 || lease.expires_wall_time_ns < 0) {
    throw std::invalid_argument(
        "lease renewal target requires an exact boot-scoped lease identity");
  }
  if (target.timeout_ns <= 0 ||
      target.timeout_ns > LeaseRenewalCoordinator::kMaximumTimeoutNs ||
      target.renewal_margin_ns <= 0 ||
      target.renewal_margin_ns > LeaseRenewalCoordinator::kMaximumMarginNs ||
      target.renewal_margin_ns >= target.timeout_ns) {
    throw std::invalid_argument(
        "lease renewal timeout and margin are outside bounded policy");
  }
}

}  // namespace

LeaseRenewalCoordinator::LeaseRenewalCoordinator(
    Journal& journal, std::shared_ptr<AuthorityClock> authority_clock)
    : journal_(journal), authority_clock_(std::move(authority_clock)) {
  if (!authority_clock_) {
    throw std::invalid_argument("lease renewal coordinator requires an authority clock");
  }
}

void LeaseRenewalCoordinator::track(LeaseRenewalTarget target) {
  validate_target(target);
  std::scoped_lock lock(mutex_);
  if (poison_reason_) throw LeaseRenewalCoordinatorError(*poison_reason_);
  const auto found = targets_.find(target.lease.concurrency_key);
  if (found != targets_.end()) {
    if (found->second == target) return;
    throw std::invalid_argument(
        "lease renewal coordinator already tracks a different exact lease");
  }
  if (targets_.size() >= kMaximumTrackedTargets) {
    throw std::invalid_argument(
        "lease renewal coordinator target limit is exhausted");
  }
  targets_.emplace(target.lease.concurrency_key, std::move(target));
}

bool LeaseRenewalCoordinator::untrack(const std::string& concurrency_key) {
  if (concurrency_key.empty()) {
    throw std::invalid_argument("lease renewal concurrency key must not be empty");
  }
  std::scoped_lock lock(mutex_);
  if (poison_reason_) throw LeaseRenewalCoordinatorError(*poison_reason_);
  return targets_.erase(concurrency_key) == 1U;
}

std::vector<LeaseRenewalTickResult> LeaseRenewalCoordinator::tick() {
  std::scoped_lock lock(mutex_);
  if (poison_reason_) throw LeaseRenewalCoordinatorError(*poison_reason_);

  std::vector<LeaseRenewalTickResult> results;
  results.reserve(targets_.size());
  try {
    for (auto target = targets_.begin(); target != targets_.end();) {
      const std::string key = target->first;
      // Each target receives a fresh sample immediately before its authority
      // checks. A long bounded tick cannot renew later targets using the stale
      // time captured for an earlier database operation.
      const AuthorityTimeSample now = authority_clock_->sample();
      const auto active = journal_.active_lease(key, now);
      if (!active || *active != target->second.lease) {
        results.push_back({.concurrency_key = key,
                           .status = LeaseRenewalTickStatus::lost,
                           .receipt = std::nullopt});
        target = targets_.erase(target);
        continue;
      }
      const std::int64_t remaining =
          active->expires_boottime_ns - now.boot.nanoseconds;
      if (remaining > target->second.renewal_margin_ns) {
        results.push_back({.concurrency_key = key,
                           .status = LeaseRenewalTickStatus::not_due,
                           .receipt = std::nullopt});
        ++target;
        continue;
      }

      const LeaseRenewalResult renewed = journal_.renew_lease_exact(
          target->second.lease, now, target->second.timeout_ns);
      if (renewed.status == LeaseRenewalStatus::not_owned) {
        results.push_back({.concurrency_key = key,
                           .status = LeaseRenewalTickStatus::lost,
                           .receipt = std::nullopt});
        target = targets_.erase(target);
        continue;
      }
      if (!renewed.receipt) {
        throw std::runtime_error(
            "successful lease renewal has no durable receipt");
      }
      target->second.lease.expires_boottime_ns =
          renewed.receipt->new_expires_boottime_ns;
      target->second.lease.expires_wall_time_ns =
          renewed.receipt->new_expires_wall_time_ns;
      results.push_back(
          {.concurrency_key = key,
           .status = renewed.status == LeaseRenewalStatus::renewed
                         ? LeaseRenewalTickStatus::renewed
                         : LeaseRenewalTickStatus::replayed,
           .receipt = renewed.receipt});
      ++target;
    }
  } catch (const std::exception& exception) {
    poison("lease renewal clock/journal/receipt failure: " +
           std::string(exception.what()));
  } catch (...) {
    poison(
        "lease renewal clock/journal/receipt failed with a non-standard exception");
  }
  return results;
}

bool LeaseRenewalCoordinator::poisoned() const {
  std::scoped_lock lock(mutex_);
  return poison_reason_.has_value();
}

std::size_t LeaseRenewalCoordinator::tracked_count() const {
  std::scoped_lock lock(mutex_);
  return targets_.size();
}

[[noreturn]] void LeaseRenewalCoordinator::poison(std::string reason) {
  poison_reason_ = "lease renewal coordinator is poisoned until restart: " +
                   std::move(reason);
  targets_.clear();
  throw LeaseRenewalCoordinatorError(*poison_reason_);
}

}  // namespace trainvm
