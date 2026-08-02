#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

#include "trainvm/authority_time.hpp"
#include "trainvm/hostd.hpp"
#include "trainvm/hostd_restart_process_recovery.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdStartupControllerApiVersion =
    "trainvm.hostd-startup-controller/v1";

class HostdStartupControllerError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class HostdStartupPhase {
  reconciling,
  auditing,
  admitting,
  exhausted,
  failed,
};

struct HostdStartupControllerConfig final {
  std::string api_version{std::string(kHostdStartupControllerApiVersion)};
  std::size_t maximum_recovery_steps{64U};

  bool operator==(const HostdStartupControllerConfig&) const = default;
};

struct HostdStartupControllerStatus final {
  HostdStartupPhase phase{HostdStartupPhase::reconciling};
  std::size_t recovery_steps{};
  std::optional<HostdRestartProcessRecoverySummary> last_recovery;
  std::optional<HostStartupAuditReceipt> admission_receipt;

  bool operator==(const HostdStartupControllerStatus&) const = default;
};

// Samples the authority clock on demand, so the coordinator can take the
// startup-audit commit time after the audit observation has finished.
class AuthorityClockStartupAuditCommitTime final
    : public IHostStartupAuditCommitTimeSource {
 public:
  explicit AuthorityClockStartupAuditCommitTime(AuthorityClock& clock) noexcept;
  [[nodiscard]] HostLedgerTime commit_time() override;

 private:
  AuthorityClock& clock_;
};

// The admission authority owns the whole audit-and-commit boundary, including
// when the commit time is sampled. Passing a pre-sampled time in is what made
// the production auditor uncommittable, so the clock is handed over instead.
class IHostdStartupAdmissionAuthority {
 public:
  virtual ~IHostdStartupAdmissionAuthority() = default;
  [[nodiscard]] virtual HostStartupAuditReceipt admit(
      IConfiguredHostStartupAuditorV2& auditor, AuthorityClock& clock) = 0;
};

class HostdCoordinatorStartupAdmission final
    : public IHostdStartupAdmissionAuthority {
 public:
  explicit HostdCoordinatorStartupAdmission(HostGrantCoordinator& coordinator);
  [[nodiscard]] HostStartupAuditReceipt admit(
      IConfiguredHostStartupAuditorV2& auditor,
      AuthorityClock& clock) override;

 private:
  HostGrantCoordinator& coordinator_;
};

// A wake-driven startup FSM. Each advance performs at most one bounded restart
// recovery step. It never commits the one-shot admission audit while any
// durable process or terminal-release record remains. The caller may wait on
// pidfds/timers between advances; there is no hidden sleep or busy loop here.
class HostdStartupController final {
 public:
  HostdStartupController(
      IHostdRestartProcessRecovery& recovery,
      IHostdStartupAdmissionAuthority& admission,
      IConfiguredHostStartupAuditorV2& auditor, AuthorityClock& clock,
      HostdStartupControllerConfig config = {});

  [[nodiscard]] HostdStartupControllerStatus advance();
  [[nodiscard]] const HostdStartupControllerStatus& status() const noexcept;

 private:
  IHostdRestartProcessRecovery& recovery_;
  IHostdStartupAdmissionAuthority& admission_;
  IConfiguredHostStartupAuditorV2& auditor_;
  AuthorityClock& clock_;
  HostdStartupControllerConfig config_;
  HostdStartupControllerStatus status_;
};

}  // namespace trainvm
