#include "trainvm/journal_cache_lease_authority.hpp"

#include <stdexcept>
#include <utility>

namespace trainvm {

JournalCacheLeaseAuthority::JournalCacheLeaseAuthority(Journal& journal,
                                                       TimeSource time_source)
    : journal_(journal), time_source_(std::move(time_source)) {
  if (!time_source_) {
    throw std::invalid_argument(
        "journal cache lease authority requires an authority time source");
  }
}

void JournalCacheLeaseAuthority::require_current(const ResourceLease& lease) {
  const AuthorityTimeSample now = time_source_();
  const std::optional<ResourceLease> active =
      journal_.active_lease(lease.concurrency_key, now);
  if (!active || *active != lease ||
      active->clock_domain != ResourceLease::kBootTimeDomain ||
      active->boot_id != now.boot_id ||
      active->expires_boottime_ns <= now.boot.nanoseconds) {
    throw CacheArtifactAuthorityError(
        "cache publication or adoption no longer owns its exact live lease");
  }
}

} // namespace trainvm
