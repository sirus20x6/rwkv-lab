#include "trainvm/hostd_terminal_release_recovery.hpp"

#include <map>
#include <set>
#include <utility>

namespace trainvm {
namespace {

[[noreturn]] void reject(std::string message) {
  throw HostLedgerError(std::move(message));
}

HostLedgerTime ledger_time(const AuthorityTimeSample& sample,
                           std::string_view expected_boot_id) {
  if (sample.boot_id != expected_boot_id) {
    reject("terminal release recovery crossed a boot identity boundary");
  }
  return {.boottime_ns = sample.boot.nanoseconds,
          .wall_time_ns = sample.wall.nanoseconds};
}

ResourceReleaseRequest release_request(const ResourceBundleGrant& grant) {
  return seal_resource_release_request({
      .api_version = std::string(kHostLedgerReleaseRequestApiVersion),
      .release_request_id =
          "hostd-terminal-release:" + grant.allocation_id,
      .allocation_id = grant.allocation_id,
      .grant_digest = grant.receipt_digest,
      .journal_id = grant.journal_id,
      .run_id = grant.run_id,
      .logical_lease_id = grant.logical_lease_id,
      .logical_fencing_token = grant.logical_fencing_token,
      .canonical_request_digest = {},
  });
}

}  // namespace

LinuxHostdTerminalCgroupCleaner::LinuxHostdTerminalCgroupCleaner(
    LinuxCgroupAuthority& cgroups)
    : cgroups_(cgroups) {}

LinuxTerminalCgroupCleanupDisposition
LinuxHostdTerminalCgroupCleaner::cleanup(
    const HostProcessTerminalReleaseRecord& record) {
  const auto& spawn = record.spawn.request;
  return cgroups_.cleanup_terminal_or_confirm_absent(
      record.grant.allocation_id, record.intent.request.launch_id,
      {.unified_path = spawn.cgroup_path,
       .device = spawn.cgroup_device,
       .inode = spawn.cgroup_inode});
}

LinuxTerminalCgroupCleanupDisposition
LinuxHostdTerminalCgroupCleaner::cleanup_intent(
    const HostProcessRecoveryRecord& record) {
  const auto& request = record.intent.request;
  return cgroups_.terminate_intent_or_confirm_absent(
      record.grant.allocation_id, request.launch_id,
      {.unified_path = request.cgroup_path,
       .device = request.cgroup_device,
       .inode = request.cgroup_inode});
}

SQLiteHostdTerminalReleaseAuthority::SQLiteHostdTerminalReleaseAuthority(
    SQLiteHostLedger& ledger, AuthorityClock& clock)
    : ledger_(ledger), clock_(clock) {}

std::vector<HostProcessTerminalReleaseRecord>
SQLiteHostdTerminalReleaseAuthority::terminal_records() {
  return ledger_.active_terminal_process_release_records();
}

std::vector<HostProcessRecoveryRecord>
SQLiteHostdTerminalReleaseAuthority::unclosed_records() {
  return ledger_.active_process_recovery_records();
}

BundleReleaseResult SQLiteHostdTerminalReleaseAuthority::release(
    const ResourceBundleGrant& grant) {
  return ledger_.release_bundle(
      release_request(grant), ledger_time(clock_.sample(), grant.boot_id));
}

HostdTerminalReleaseRecovery::HostdTerminalReleaseRecovery(
    IHostdTerminalReleaseAuthority& authority,
    IHostdTerminalCgroupCleaner& cleaner)
    : authority_(authority), cleaner_(cleaner) {}

HostdTerminalReleaseRecoverySummary
HostdTerminalReleaseRecovery::recover() {
  const std::vector<HostProcessTerminalReleaseRecord> terminal =
      authority_.terminal_records();
  const std::vector<HostProcessRecoveryRecord> unclosed =
      authority_.unclosed_records();
  std::set<std::string, std::less<>> blocked_allocations;
  std::vector<const HostProcessRecoveryRecord*> intent_only;
  for (const HostProcessRecoveryRecord& record : unclosed) {
    if (record.spawn) {
      blocked_allocations.insert(record.grant.allocation_id);
    } else {
      intent_only.push_back(&record);
    }
  }
  std::map<std::string, ResourceBundleGrant, std::less<>> grants;
  HostdTerminalReleaseRecoverySummary summary{
      .terminal_records = terminal.size(),
  };
  for (const HostProcessTerminalReleaseRecord& record : terminal) {
    if (record.intent.request.allocation_id != record.grant.allocation_id ||
        record.spawn.request.launch_id != record.intent.request.launch_id ||
        record.child_exit.has_value() == record.recovery_exit.has_value()) {
      reject("terminal release recovery record is internally inconsistent");
    }
    const auto [found, inserted] =
        grants.emplace(record.grant.allocation_id, record.grant);
    if (!inserted && found->second != record.grant) {
      reject("terminal siblings disagree on their resource grant");
    }
  }
  for (const HostProcessRecoveryRecord* record : intent_only) {
    if (record->intent.request.allocation_id != record->grant.allocation_id ||
        record->intent.request.grant_digest != record->grant.receipt_digest) {
      reject("intent-only recovery record is internally inconsistent");
    }
    const auto [found, inserted] =
        grants.emplace(record->grant.allocation_id, record->grant);
    if (!inserted && found->second != record->grant) {
      reject("process siblings disagree on their resource grant");
    }
  }
  for (const HostProcessTerminalReleaseRecord& record : terminal) {
    switch (cleaner_.cleanup(record)) {
      case LinuxTerminalCgroupCleanupDisposition::removed:
        ++summary.cgroups_removed;
        break;
      case LinuxTerminalCgroupCleanupDisposition::already_absent:
        ++summary.cgroups_already_absent;
        break;
      case LinuxTerminalCgroupCleanupDisposition::termination_pending:
        reject("terminal cgroup unexpectedly requires process termination");
    }
  }
  summary.intent_only_records = intent_only.size();
  for (const HostProcessRecoveryRecord* record : intent_only) {
    switch (cleaner_.cleanup_intent(*record)) {
      case LinuxTerminalCgroupCleanupDisposition::removed:
        ++summary.intent_cgroups_removed;
        break;
      case LinuxTerminalCgroupCleanupDisposition::already_absent:
        ++summary.intent_cgroups_already_absent;
        break;
      case LinuxTerminalCgroupCleanupDisposition::termination_pending:
        ++summary.intent_terminations_pending;
        blocked_allocations.insert(record->grant.allocation_id);
        break;
    }
  }
  for (const auto& [allocation_id, grant] : grants) {
    if (blocked_allocations.contains(allocation_id)) {
      ++summary.allocations_blocked_by_unclosed_process;
      continue;
    }
    const BundleReleaseResult released = authority_.release(grant);
    ++summary.allocations_released;
    if (released.replayed) ++summary.release_replays;
  }
  return summary;
}

}  // namespace trainvm
