#pragma once

#include <cstddef>

#include "trainvm/hostd_linux_process_authority.hpp"
#include "trainvm/hostd_terminal_release_recovery.hpp"

namespace trainvm {

enum class HostdExactRecoveredProcessPolicy {
  leave_and_block,
  terminate_and_reconcile,
};

struct HostdRestartProcessRecoveryConfig final {
  HostdExactRecoveredProcessPolicy exact_live_policy{
      HostdExactRecoveredProcessPolicy::leave_and_block};
  bool reconcile_observed_nonlive{};

  bool operator==(const HostdRestartProcessRecoveryConfig&) const = default;
};

struct HostdRestartProcessRecoverySummary final {
  HostdTerminalReleaseRecoverySummary cleanup_before;
  LinuxProcessRecoverySummary observed;
  std::size_t newly_adopted{};
  std::size_t nonlive_finalized{};
  HostdRecoveredProcessProgress progress;
  HostdTerminalReleaseRecoverySummary cleanup_after;
  std::size_t remaining_unclosed_records{};
  std::size_t remaining_terminal_release_records{};

  bool operator==(const HostdRestartProcessRecoverySummary&) const = default;
};

class IHostdRestartProcessRecovery {
 public:
  virtual ~IHostdRestartProcessRecovery() = default;
  [[nodiscard]] virtual HostdRestartProcessRecoverySummary step() = 0;
};

// One bounded reconciliation step performed before the final startup audit.
// Callers may repeat it while exact recovered SIGKILL delivery is pending; all
// ledger/cgroup actions are exact-replay or already-absent safe.
class HostdRestartProcessRecovery final
    : public IHostdRestartProcessRecovery {
 public:
  HostdRestartProcessRecovery(
      IHostdTerminalReleaseAuthority& records,
      HostdTerminalReleaseRecovery& cleanup,
      IHostdRecoveredProcessSupervisor& supervisor,
      HostdRestartProcessRecoveryConfig config);

  [[nodiscard]] HostdRestartProcessRecoverySummary step() override;

 private:
  IHostdTerminalReleaseAuthority& records_;
  HostdTerminalReleaseRecovery& cleanup_;
  IHostdRecoveredProcessSupervisor& supervisor_;
  HostdRestartProcessRecoveryConfig config_;
};

}  // namespace trainvm
