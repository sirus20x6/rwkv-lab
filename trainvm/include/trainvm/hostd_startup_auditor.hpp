#pragma once

#include <string>
#include <vector>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_ledger.hpp"
#include "trainvm/hostd_linux_process_recovery.hpp"
#include "trainvm/host_startup_audit.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdConfiguredStartupAuditorApiVersion =
    "trainvm.hostd-configured-startup-auditor/v1";

struct HostdConfiguredStartupAuditorConfig final {
  std::string api_version{
      std::string(kHostdConfiguredStartupAuditorApiVersion)};
  std::string broker_instance_id;
  HostStartupAuditPolicy policy;

  bool operator==(const HostdConfiguredStartupAuditorConfig&) const = default;
};

// Production startup evidence assembled from the already-open, pinned host
// ledger and the authority clock. This class has no process mutation methods.
// It classifies durable process records once and pins exact live identities for
// explicit transfer to the recovery supervisor. Until terminal reconciliation
// exists, any retained active fence remains a blocking finding.
class HostdConfiguredStartupAuditor final
    : public IConfiguredHostStartupAuditorV2 {
 public:
  HostdConfiguredStartupAuditor(SQLiteHostLedger& ledger,
                                AuthorityClock& clock,
                                HostdConfiguredStartupAuditorConfig config);

  [[nodiscard]] HostStartupAuditReport audit() override;
  [[nodiscard]] const LinuxProcessRecoverySet& process_recovery() const
      noexcept;
  [[nodiscard]] LinuxProcessRecoverySet& process_recovery() noexcept;
  [[nodiscard]] const std::vector<HostProcessTerminalReleaseRecord>&
  terminal_process_releases() const noexcept;

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
  HostdConfiguredStartupAuditorConfig config_;
  LinuxProcessRecoverySet process_recovery_;
  std::vector<HostProcessTerminalReleaseRecord> terminal_process_releases_;
};

}  // namespace trainvm
