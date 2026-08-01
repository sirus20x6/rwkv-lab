#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_linux_cgroup_authority.hpp"

namespace trainvm {

class IHostdTerminalCgroupCleaner {
 public:
  virtual ~IHostdTerminalCgroupCleaner() = default;
  [[nodiscard]] virtual LinuxTerminalCgroupCleanupDisposition cleanup(
      const HostProcessTerminalReleaseRecord& record) = 0;
};

class LinuxHostdTerminalCgroupCleaner final
    : public IHostdTerminalCgroupCleaner {
 public:
  explicit LinuxHostdTerminalCgroupCleaner(LinuxCgroupAuthority& cgroups);
  [[nodiscard]] LinuxTerminalCgroupCleanupDisposition cleanup(
      const HostProcessTerminalReleaseRecord& record) override;

 private:
  LinuxCgroupAuthority& cgroups_;
};

class IHostdTerminalReleaseAuthority {
 public:
  virtual ~IHostdTerminalReleaseAuthority() = default;
  [[nodiscard]] virtual std::vector<HostProcessTerminalReleaseRecord>
  terminal_records() = 0;
  [[nodiscard]] virtual std::vector<HostProcessRecoveryRecord>
  unclosed_records() = 0;
  [[nodiscard]] virtual BundleReleaseResult release(
      const ResourceBundleGrant& grant) = 0;
};

class SQLiteHostdTerminalReleaseAuthority final
    : public IHostdTerminalReleaseAuthority {
 public:
  SQLiteHostdTerminalReleaseAuthority(SQLiteHostLedger& ledger,
                                      AuthorityClock& clock);
  [[nodiscard]] std::vector<HostProcessTerminalReleaseRecord>
  terminal_records() override;
  [[nodiscard]] std::vector<HostProcessRecoveryRecord> unclosed_records()
      override;
  [[nodiscard]] BundleReleaseResult release(
      const ResourceBundleGrant& grant) override;

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
};

struct HostdTerminalReleaseRecoverySummary final {
  std::size_t terminal_records{};
  std::size_t cgroups_removed{};
  std::size_t cgroups_already_absent{};
  std::size_t allocations_released{};
  std::size_t release_replays{};
  std::size_t allocations_blocked_by_unclosed_process{};

  bool operator==(const HostdTerminalReleaseRecoverySummary&) const = default;
};

// Resumes the commit-terminal -> remove-cgroup -> release-bundle saga. It may
// clean terminal sibling cgroups while another sibling remains live, but it
// releases an allocation only when no durable unclosed process record remains.
class HostdTerminalReleaseRecovery final {
 public:
  HostdTerminalReleaseRecovery(IHostdTerminalReleaseAuthority& authority,
                               IHostdTerminalCgroupCleaner& cleaner);
  [[nodiscard]] HostdTerminalReleaseRecoverySummary recover();

 private:
  IHostdTerminalReleaseAuthority& authority_;
  IHostdTerminalCgroupCleaner& cleaner_;
};

}  // namespace trainvm
