#include "trainvm/hostd_terminal_release_recovery.hpp"

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

HostProcessTerminalReleaseRecord terminal_record(
    std::string allocation_id, std::string launch_id, bool recovered) {
  HostProcessTerminalReleaseRecord result;
  result.grant.allocation_id = std::move(allocation_id);
  result.grant.receipt_digest = "sha256:" + std::string(64U, 'a');
  result.intent.request.allocation_id = result.grant.allocation_id;
  result.intent.request.launch_id = std::move(launch_id);
  result.spawn.request.launch_id = result.intent.request.launch_id;
  if (recovered) {
    result.recovery_exit = HostProcessRecoveryExitReceipt{};
  } else {
    result.child_exit = HostProcessExitReceipt{};
  }
  return result;
}

HostProcessRecoveryRecord unclosed_record(std::string allocation_id,
                                          std::string launch_id,
                                          bool spawned) {
  HostProcessRecoveryRecord result;
  result.grant.allocation_id = std::move(allocation_id);
  result.grant.receipt_digest = "sha256:" + std::string(64U, 'a');
  result.intent.request.allocation_id = result.grant.allocation_id;
  result.intent.request.grant_digest = result.grant.receipt_digest;
  result.intent.request.launch_id = std::move(launch_id);
  if (spawned) result.spawn = HostProcessSpawnReceipt{};
  return result;
}

class FakeAuthority final : public IHostdTerminalReleaseAuthority {
 public:
  std::vector<HostProcessTerminalReleaseRecord> terminal;
  std::vector<HostProcessRecoveryRecord> unclosed;
  std::vector<std::string> released;

  std::vector<HostProcessTerminalReleaseRecord> terminal_records() override {
    return terminal;
  }
  std::vector<HostProcessRecoveryRecord> unclosed_records() override {
    return unclosed;
  }
  BundleReleaseResult release(const ResourceBundleGrant& grant) override {
    released.push_back(grant.allocation_id);
    return {.receipt = {}, .replayed = false};
  }
};

class FakeCleaner final : public IHostdTerminalCgroupCleaner {
 public:
  std::vector<LinuxTerminalCgroupCleanupDisposition> dispositions;
  std::vector<LinuxTerminalCgroupCleanupDisposition> intent_dispositions;
  std::vector<std::string> cleaned;
  std::vector<std::string> cleaned_intents;

  LinuxTerminalCgroupCleanupDisposition cleanup(
      const HostProcessTerminalReleaseRecord& record) override {
    cleaned.push_back(record.intent.request.launch_id);
    if (cleaned.size() > dispositions.size()) {
      throw std::runtime_error("missing fake cleanup disposition");
    }
    return dispositions[cleaned.size() - 1U];
  }
  LinuxTerminalCgroupCleanupDisposition cleanup_intent(
      const HostProcessRecoveryRecord& record) override {
    cleaned_intents.push_back(record.intent.request.launch_id);
    if (cleaned_intents.size() > intent_dispositions.size()) {
      throw std::runtime_error("missing fake intent cleanup disposition");
    }
    return intent_dispositions[cleaned_intents.size() - 1U];
  }
};

void resumes_cleanup_and_releases_only_closed_allocations() {
  FakeAuthority authority;
  authority.terminal.push_back(terminal_record("allocation-a", "launch-a1", false));
  authority.terminal.push_back(terminal_record("allocation-a", "launch-a2", true));
  authority.terminal.push_back(terminal_record("allocation-b", "launch-b1", false));
  authority.terminal.push_back(terminal_record("allocation-d", "launch-d1", false));
  authority.unclosed.push_back(
      unclosed_record("allocation-b", "launch-b2", true));
  authority.unclosed.push_back(
      unclosed_record("allocation-c", "launch-c1", false));
  authority.unclosed.push_back(
      unclosed_record("allocation-d", "launch-d2", false));
  FakeCleaner cleaner;
  cleaner.dispositions = {
      LinuxTerminalCgroupCleanupDisposition::removed,
      LinuxTerminalCgroupCleanupDisposition::already_absent,
      LinuxTerminalCgroupCleanupDisposition::removed,
      LinuxTerminalCgroupCleanupDisposition::removed,
  };
  cleaner.intent_dispositions = {
      LinuxTerminalCgroupCleanupDisposition::already_absent,
      LinuxTerminalCgroupCleanupDisposition::removed,
  };
  HostdTerminalReleaseRecovery recovery(authority, cleaner);
  const auto summary = recovery.recover();
  require(summary == HostdTerminalReleaseRecoverySummary{
                         .terminal_records = 4U,
                         .cgroups_removed = 3U,
                         .cgroups_already_absent = 1U,
                         .intent_only_records = 2U,
                         .intent_cgroups_removed = 1U,
                         .intent_cgroups_already_absent = 1U,
                         .allocations_released = 3U,
                         .release_replays = 0U,
                         .allocations_blocked_by_unclosed_process = 1U,
                     } &&
              authority.released ==
                  std::vector<std::string>{"allocation-a", "allocation-c",
                                           "allocation-d"} &&
              cleaner.cleaned.size() == 4U &&
              cleaner.cleaned_intents.size() == 2U,
          "terminal cleanup resumes idempotently without releasing live sibling");
}

void inconsistent_terminal_shape_fails_before_cleanup() {
  FakeAuthority authority;
  authority.terminal.push_back(
      terminal_record("allocation-valid", "launch-valid", false));
  auto malformed = terminal_record("allocation-a", "launch-a", false);
  malformed.recovery_exit = HostProcessRecoveryExitReceipt{};
  authority.terminal.push_back(std::move(malformed));
  FakeCleaner cleaner;
  cleaner.dispositions = {
      LinuxTerminalCgroupCleanupDisposition::removed,
      LinuxTerminalCgroupCleanupDisposition::removed,
  };
  HostdTerminalReleaseRecovery recovery(authority, cleaner);
  bool rejected = false;
  try {
    (void)recovery.recover();
  } catch (const HostLedgerError&) {
    rejected = true;
  }
  require(rejected && cleaner.cleaned.empty() && authority.released.empty(),
          "ambiguous terminal evidence never reaches cleanup or release");
}

}  // namespace

int main() {
  try {
    resumes_cleanup_and_releases_only_closed_allocations();
    inconsistent_terminal_shape_fails_before_cleanup();
    std::cout << "hostd terminal release recovery tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd terminal release recovery test failure: "
              << error.what() << '\n';
    return 1;
  }
}
