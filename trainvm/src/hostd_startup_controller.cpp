#include "trainvm/hostd_startup_controller.hpp"

#include <utility>

namespace trainvm {

AuthorityClockStartupAuditCommitTime::AuthorityClockStartupAuditCommitTime(
    AuthorityClock& clock) noexcept
    : clock_(clock) {}

HostLedgerTime AuthorityClockStartupAuditCommitTime::commit_time() {
  const AuthorityTimeSample sample = clock_.sample();
  return {.boottime_ns = sample.boot.nanoseconds,
          .wall_time_ns = sample.wall.nanoseconds};
}

HostdCoordinatorStartupAdmission::HostdCoordinatorStartupAdmission(
    HostGrantCoordinator& coordinator)
    : coordinator_(coordinator) {}

HostStartupAuditReceipt HostdCoordinatorStartupAdmission::admit(
    IConfiguredHostStartupAuditorV2& auditor, AuthorityClock& clock) {
  AuthorityClockStartupAuditCommitTime commit_time(clock);
  return coordinator_.run_startup_audit(auditor, commit_time);
}

HostdStartupController::HostdStartupController(
    IHostdRestartProcessRecovery& recovery,
    IHostdStartupAdmissionAuthority& admission,
    IConfiguredHostStartupAuditorV2& auditor, AuthorityClock& clock,
    HostdStartupControllerConfig config)
    : recovery_(recovery), admission_(admission), auditor_(auditor),
      clock_(clock), config_(std::move(config)) {
  if (config_.api_version != kHostdStartupControllerApiVersion ||
      config_.maximum_recovery_steps == 0U ||
      config_.maximum_recovery_steps > 1'000'000U) {
    throw HostdStartupControllerError(
        "hostd startup controller config is invalid");
  }
}

HostdStartupControllerStatus HostdStartupController::advance() {
  if (status_.phase == HostdStartupPhase::admitting) return status_;
  if (status_.phase == HostdStartupPhase::exhausted)
    throw HostdStartupControllerError(
        "hostd startup recovery step bound is exhausted");
  if (status_.phase == HostdStartupPhase::failed)
    throw HostdStartupControllerError("hostd startup controller has failed");
  if (status_.phase != HostdStartupPhase::reconciling)
    throw HostdStartupControllerError("hostd startup phase is invalid");
  if (status_.recovery_steps >= config_.maximum_recovery_steps) {
    status_.phase = HostdStartupPhase::exhausted;
    throw HostdStartupControllerError(
        "hostd startup recovery did not converge within its bound");
  }

  try {
    status_.last_recovery = recovery_.step();
    ++status_.recovery_steps;
  } catch (...) {
    status_.phase = HostdStartupPhase::failed;
    throw;
  }
  if (status_.last_recovery->remaining_unclosed_records != 0U ||
      status_.last_recovery->remaining_terminal_release_records != 0U) {
    if (status_.recovery_steps >= config_.maximum_recovery_steps) {
      status_.phase = HostdStartupPhase::exhausted;
      throw HostdStartupControllerError(
          "hostd startup recovery did not converge within its bound");
    }
    return status_;
  }

  status_.phase = HostdStartupPhase::auditing;
  try {
    // The clock is handed to the admission authority rather than sampled here:
    // the audit observation runs inside admit(), and its commit time must not
    // predate it.
    status_.admission_receipt = admission_.admit(auditor_, clock_);
    status_.phase = HostdStartupPhase::admitting;
    return status_;
  } catch (...) {
    status_.phase = HostdStartupPhase::failed;
    throw;
  }
}

const HostdStartupControllerStatus& HostdStartupController::status()
    const noexcept {
  return status_;
}

}  // namespace trainvm
