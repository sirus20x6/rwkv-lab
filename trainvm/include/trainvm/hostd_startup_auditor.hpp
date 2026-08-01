#pragma once

#include <string>

#include "trainvm/authority_time.hpp"
#include "trainvm/host_ledger.hpp"
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
// Until durable process adoption exists, any retained active fence is a
// blocking finding rather than something the daemon guesses how to recover.
class HostdConfiguredStartupAuditor final
    : public IConfiguredHostStartupAuditorV2 {
 public:
  HostdConfiguredStartupAuditor(SQLiteHostLedger& ledger,
                                AuthorityClock& clock,
                                HostdConfiguredStartupAuditorConfig config);

  [[nodiscard]] HostStartupAuditReport audit() override;

 private:
  SQLiteHostLedger& ledger_;
  AuthorityClock& clock_;
  HostdConfiguredStartupAuditorConfig config_;
};

}  // namespace trainvm
