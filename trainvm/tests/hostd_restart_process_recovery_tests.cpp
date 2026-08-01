#include "trainvm/hostd_restart_process_recovery.hpp"

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

HostProcessRecoveryRecord intent_only() {
  HostProcessRecoveryRecord result;
  result.grant.allocation_id = "allocation-intent";
  result.grant.receipt_digest = "sha256:" + std::string(64U, 'a');
  result.intent.request.allocation_id = result.grant.allocation_id;
  result.intent.request.grant_digest = result.grant.receipt_digest;
  result.intent.request.launch_id = "launch-intent";
  return result;
}

class FakeAuthority final : public IHostdTerminalReleaseAuthority {
 public:
  std::vector<HostProcessRecoveryRecord> unclosed{intent_only()};
  std::size_t releases{};

  std::vector<HostProcessTerminalReleaseRecord> terminal_records() override {
    return {};
  }
  std::vector<HostProcessRecoveryRecord> unclosed_records() override {
    return unclosed;
  }
  BundleReleaseResult release(const ResourceBundleGrant& grant) override {
    require(grant.allocation_id == "allocation-intent",
            "recovery releases exact intent grant");
    ++releases;
    unclosed.clear();
    return {.receipt = {}, .replayed = false};
  }
};

class FakeCleaner final : public IHostdTerminalCgroupCleaner {
 public:
  std::size_t intents{};

  LinuxTerminalCgroupCleanupDisposition cleanup(
      const HostProcessTerminalReleaseRecord&) override {
    throw std::runtime_error("unexpected terminal cleanup");
  }
  LinuxTerminalCgroupCleanupDisposition cleanup_intent(
      const HostProcessRecoveryRecord&) override {
    ++intents;
    return LinuxTerminalCgroupCleanupDisposition::already_absent;
  }
};

class FakeSupervisor final : public IHostdRecoveredProcessSupervisor {
 public:
  std::size_t adopt_calls{};
  std::size_t progress_calls{};

  std::size_t adopt_exact_recovered_processes(
      LinuxProcessRecoverySet&) override {
    ++adopt_calls;
    return 0U;
  }
  HostdRecoveredProcessProgress progress_recovered_terminations() override {
    ++progress_calls;
    return {};
  }
};

void restart_step_closes_intent_then_becomes_noop() {
  FakeAuthority authority;
  FakeCleaner cleaner;
  FakeSupervisor supervisor;
  HostdTerminalReleaseRecovery cleanup(authority, cleaner);
  HostdRestartProcessRecovery recovery(
      authority, cleanup, supervisor,
      {.exact_live_policy =
           HostdExactRecoveredProcessPolicy::terminate_and_reconcile});
  const auto first = recovery.step();
  require(first.cleanup_before.intent_only_records == 1U &&
              first.cleanup_before.intent_cgroups_already_absent == 1U &&
              first.cleanup_before.allocations_released == 1U &&
              first.observed.records == 0U &&
              first.remaining_unclosed_records == 0U &&
              first.remaining_terminal_release_records == 0U &&
              authority.releases == 1U && cleaner.intents == 1U &&
              supervisor.adopt_calls == 1U && supervisor.progress_calls == 1U,
          "restart step cleans abandoned intent before final observation");
  const auto second = recovery.step();
  require(second.cleanup_before.terminal_records == 0U &&
              second.cleanup_before.intent_only_records == 0U &&
              second.remaining_unclosed_records == 0U &&
              authority.releases == 1U && cleaner.intents == 1U,
          "repeated restart reconciliation is an exact no-op");
}

void leave_policy_never_invokes_mutating_supervisor() {
  FakeAuthority authority;
  authority.unclosed.clear();
  FakeCleaner cleaner;
  FakeSupervisor supervisor;
  HostdTerminalReleaseRecovery cleanup(authority, cleaner);
  HostdRestartProcessRecovery recovery(
      authority, cleanup, supervisor,
      {.exact_live_policy = HostdExactRecoveredProcessPolicy::leave_and_block});
  const auto summary = recovery.step();
  require(summary.remaining_unclosed_records == 0U &&
              supervisor.adopt_calls == 0U && supervisor.progress_calls == 0U,
          "leave policy has no recovered-process mutation path");
}

}  // namespace

int main() {
  try {
    restart_step_closes_intent_then_becomes_noop();
    leave_policy_never_invokes_mutating_supervisor();
    std::cout << "hostd restart process recovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd restart process recovery test failure: "
              << error.what() << '\n';
    return 1;
  }
}
