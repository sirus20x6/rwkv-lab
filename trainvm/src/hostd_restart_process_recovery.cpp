#include "trainvm/hostd_restart_process_recovery.hpp"

#include <utility>
#include <vector>

namespace trainvm {

HostdRestartProcessRecovery::HostdRestartProcessRecovery(
    IHostdTerminalReleaseAuthority& records,
    HostdTerminalReleaseRecovery& cleanup,
    IHostdRecoveredProcessSupervisor& supervisor,
    HostdRestartProcessRecoveryConfig config)
    : records_(records), cleanup_(cleanup), supervisor_(supervisor),
      config_(config) {}

HostdRestartProcessRecoverySummary HostdRestartProcessRecovery::step() {
  HostdRestartProcessRecoverySummary summary;
  summary.cleanup_before = cleanup_.recover();

  std::vector<HostProcessRecoveryRecord> records =
      records_.unclosed_records();
  LinuxProcessRecoveryProbe probe;
  LinuxProcessRecoverySet recovery;
  recovery.recover(std::move(records), probe);
  summary.observed = recovery.summary();
  if (config_.exact_live_policy ==
      HostdExactRecoveredProcessPolicy::terminate_and_reconcile) {
    summary.newly_adopted =
        supervisor_.adopt_exact_recovered_processes(recovery);
    summary.progress = supervisor_.progress_recovered_terminations();
  }

  summary.cleanup_after = cleanup_.recover();
  summary.remaining_unclosed_records = records_.unclosed_records().size();
  summary.remaining_terminal_release_records =
      records_.terminal_records().size();
  return summary;
}

}  // namespace trainvm
