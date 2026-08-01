#include "trainvm/hostd_startup_controller.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace trainvm;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class FakeRecovery final : public IHostdRestartProcessRecovery {
 public:
  explicit FakeRecovery(
      std::vector<HostdRestartProcessRecoverySummary> values = {},
      bool should_fail = false)
      : script(std::move(values)), fail(should_fail) {}

  std::vector<HostdRestartProcessRecoverySummary> script;
  bool fail{};
  std::size_t calls{};

  HostdRestartProcessRecoverySummary step() override {
    ++calls;
    if (fail) throw std::runtime_error("recovery failed");
    if (script.empty()) throw std::runtime_error("recovery script exhausted");
    auto result = script.front();
    script.erase(script.begin());
    return result;
  }
};

class FakeAuditor final : public IConfiguredHostStartupAuditorV2 {
 public:
  HostStartupAuditReport audit() override { return {}; }
};

class FakeAdmission final : public IHostdStartupAdmissionAuthority {
 public:
  bool fail{};
  std::size_t calls{};
  HostLedgerTime observed_time{};

  HostStartupAuditReceipt admit(IConfiguredHostStartupAuditorV2&,
                                const HostLedgerTime& now) override {
    ++calls;
    observed_time = now;
    if (fail) throw std::runtime_error("admission failed");
    HostStartupAuditReceipt receipt;
    receipt.audit_id = "admitted";
    return receipt;
  }
};

AuthorityClock clock() {
  return AuthorityClock([] {
    return AuthorityTimeSample{.wall = {.nanoseconds = 2000},
                               .boot = {.nanoseconds = 1000},
                               .boot_id =
                                   "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"};
  });
}

HostdRestartProcessRecoverySummary remaining(std::size_t unclosed,
                                             std::size_t terminal) {
  HostdRestartProcessRecoverySummary result;
  result.remaining_unclosed_records = unclosed;
  result.remaining_terminal_release_records = terminal;
  return result;
}

void audit_runs_once_only_after_recovery_converges() {
  FakeRecovery recovery({remaining(1U, 0U), remaining(0U, 1U),
                         remaining(0U, 0U)});
  FakeAdmission admission;
  FakeAuditor auditor;
  AuthorityClock authority_clock = clock();
  HostdStartupController controller(recovery, admission, auditor,
                                    authority_clock);

  const auto first = controller.advance();
  require(first.phase == HostdStartupPhase::reconciling &&
              first.recovery_steps == 1U && admission.calls == 0U,
          "an unclosed process prevents the one-shot audit");
  const auto second = controller.advance();
  require(second.phase == HostdStartupPhase::reconciling &&
              second.recovery_steps == 2U && admission.calls == 0U,
          "terminal release residue also prevents admission");
  const auto third = controller.advance();
  require(third.phase == HostdStartupPhase::admitting &&
              third.recovery_steps == 3U && third.admission_receipt &&
              third.admission_receipt->audit_id == "admitted" &&
              admission.calls == 1U &&
              admission.observed_time ==
                  HostLedgerTime{.boottime_ns = 1000, .wall_time_ns = 2000},
          "convergence triggers one host-clock-bound admission audit");
  const auto replay = controller.advance();
  require(replay == third && recovery.calls == 3U && admission.calls == 1U,
          "an admitted controller is an exact in-memory no-op");
}

void recovery_bound_exhaustion_never_audits() {
  FakeRecovery recovery(
      {remaining(1U, 0U), remaining(1U, 0U), remaining(0U, 0U)});
  FakeAdmission admission;
  FakeAuditor auditor;
  AuthorityClock authority_clock = clock();
  HostdStartupController controller(
      recovery, admission, auditor, authority_clock,
      {.api_version = std::string(kHostdStartupControllerApiVersion),
       .maximum_recovery_steps = 2U});
  (void)controller.advance();
  require_throws<HostdStartupControllerError>(
      [&] { (void)controller.advance(); },
      "the configured recovery step bound must be enforced");
  require(controller.status().phase == HostdStartupPhase::exhausted &&
              recovery.calls == 2U && admission.calls == 0U,
          "exhaustion latches before any audit or extra recovery mutation");
}

void recovery_and_admission_failures_latch() {
  FakeRecovery recovery({remaining(0U, 0U)}, true);
  FakeAdmission admission;
  FakeAuditor auditor;
  AuthorityClock authority_clock = clock();
  HostdStartupController controller(recovery, admission, auditor,
                                    authority_clock);
  require_throws<std::runtime_error>([&] { (void)controller.advance(); },
                                     "recovery failure propagates");
  require(controller.status().phase == HostdStartupPhase::failed &&
              admission.calls == 0U,
          "recovery failure permanently prevents admission");

  FakeRecovery converged({remaining(0U, 0U)});
  FakeAdmission rejected;
  rejected.fail = true;
  AuthorityClock second_clock = clock();
  HostdStartupController second(converged, rejected, auditor, second_clock);
  require_throws<std::runtime_error>([&] { (void)second.advance(); },
                                     "admission failure propagates");
  require(second.status().phase == HostdStartupPhase::failed &&
              converged.calls == 1U && rejected.calls == 1U,
          "failed one-shot admission cannot be retried in place");
  require_throws<HostdStartupControllerError>(
      [&] { (void)second.advance(); },
      "latched admission failure rejects a second audit attempt");
}

void invalid_bounds_are_rejected() {
  FakeRecovery recovery;
  FakeAdmission admission;
  FakeAuditor auditor;
  AuthorityClock authority_clock = clock();
  require_throws<HostdStartupControllerError>(
      [&] {
        HostdStartupController invalid(
            recovery, admission, auditor, authority_clock,
            {.api_version = std::string(kHostdStartupControllerApiVersion),
             .maximum_recovery_steps = 0U});
      },
      "zero recovery steps cannot create a startup controller");
}

}  // namespace

int main() {
  try {
    audit_runs_once_only_after_recovery_converges();
    recovery_bound_exhaustion_never_audits();
    recovery_and_admission_failures_latch();
    invalid_bounds_are_rejected();
    std::cout << "hostd startup controller tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd startup controller test failure: " << error.what()
              << '\n';
    return 1;
  }
}
