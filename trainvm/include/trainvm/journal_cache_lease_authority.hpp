#pragma once

#include <functional>

#include "trainvm/cache_artifact_authority.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

// Production cache publication/adoption fence backed by the already-open,
// integrity-checked journal authority. The injected clock is the same
// boot-scoped authority clock used by controller mutations.
class JournalCacheLeaseAuthority final : public ICacheLeaseAuthority {
 public:
  using TimeSource = std::function<AuthorityTimeSample()>;

  JournalCacheLeaseAuthority(Journal& journal, TimeSource time_source);
  void require_current(const ResourceLease& lease) override;

 private:
  Journal& journal_;
  TimeSource time_source_;
};

} // namespace trainvm
