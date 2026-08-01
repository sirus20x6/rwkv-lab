#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "trainvm/host_resources.hpp"

namespace trainvm {

inline constexpr std::string_view kHostStartupAuditReportApiVersion =
    "trainvm.host-startup-audit-report/v2";
inline constexpr std::string_view kHostStartupAuditReceiptApiVersion =
    "trainvm.host-startup-audit-receipt/v2";
inline constexpr std::string_view kHostStartupAuditPolicyApiVersion =
    "trainvm.host-startup-audit-policy/v2";

struct HostStartupAuditBounds final {
  static constexpr std::size_t maximum_findings = 128U;
  static constexpr std::size_t maximum_identifier_bytes = 128U;
  static constexpr std::size_t maximum_finding_code_bytes = 96U;
  static constexpr std::size_t maximum_finding_subject_bytes = 256U;
  static constexpr std::size_t maximum_finding_detail_bytes = 512U;
  static constexpr std::size_t maximum_json_bytes = 2U * 1024U * 1024U;
  static constexpr std::size_t maximum_json_depth = 64U;
  static constexpr std::size_t maximum_json_nodes = 65'536U;
  static constexpr std::size_t maximum_json_container_width = 1'024U;
};

enum class HostStartupAuditFindingSeverity {
  informational,
  warning,
  blocking,
};

enum class HostStartupAuditDisposition {
  passed,
  failed,
};

struct HostLedgerChainHead final {
  std::uint64_t ledger_sequence{};
  std::string chain_hash;

  bool operator==(const HostLedgerChainHead&) const = default;
};

struct HostStartupAuditPolicy final {
  std::string api_version;
  bool require_stable_occupancy{true};
  bool fail_on_blocking_findings{true};
  std::uint32_t maximum_findings{
      static_cast<std::uint32_t>(HostStartupAuditBounds::maximum_findings)};
  std::string policy_digest;

  bool operator==(const HostStartupAuditPolicy&) const = default;
};

struct HostStartupAuditFinding final {
  HostStartupAuditFindingSeverity severity{};
  std::string code;
  std::string subject;
  std::string detail;
  std::string evidence_digest;

  bool operator==(const HostStartupAuditFinding&) const = default;
};

// A report is data-only evidence. The auditor API below deliberately receives
// no process handles, PIDs, signals, adoption hooks, or cleanup capabilities.
// It may classify observations; it cannot signal, kill, or adopt anything.
struct HostStartupAuditReport final {
  std::string api_version;
  std::string audit_id;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string broker_instance_id;
  HostInventoryReceipt inventory;
  ResourceOccupancySnapshot pre_audit_occupancy;
  ResourceOccupancySnapshot post_audit_occupancy;
  HostLedgerChainHead ledger_head_before;
  HostLedgerChainHead ledger_head_after_observation;
  HostStartupAuditPolicy policy;
  std::vector<HostStartupAuditFinding> findings;
  HostStartupAuditDisposition disposition{};
  std::int64_t observed_begin_boottime_ns{};
  std::int64_t observed_end_boottime_ns{};
  std::string findings_digest;
  std::string report_digest;

  bool operator==(const HostStartupAuditReport&) const = default;
};

struct HostStartupAuditReceipt final {
  std::string api_version;
  std::string audit_id;
  std::string report_digest;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::string broker_instance_id;
  std::string inventory_digest;
  std::string topology_digest;
  std::string pre_occupancy_digest;
  std::string post_occupancy_digest;
  std::string policy_digest;
  std::string findings_digest;
  HostStartupAuditDisposition disposition{};
  HostLedgerChainHead ledger_head_before;
  HostLedgerChainHead committed_ledger_head;
  std::string commit_record_digest;
  std::int64_t committed_boottime_ns{};
  std::int64_t committed_wall_time_ns{};
  std::string receipt_digest;

  bool operator==(const HostStartupAuditReceipt&) const = default;
};

// This is inspection data describing an in-process commit result. It is not an
// admission capability, even when returned directly by SQLiteHostLedger.
struct HostStartupAuditLedgerCommitResult final {
  HostStartupAuditReceipt receipt;
  bool replayed{};

  bool operator==(const HostStartupAuditLedgerCommitResult&) const = default;
};

class HostStartupAuditError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// The host application must construct this implementation from trusted local
// configuration. Reports obtained from an untrusted decoder do not establish
// that a configured auditor produced them.
class IConfiguredHostStartupAuditorV2 {
 public:
  virtual ~IConfiguredHostStartupAuditorV2() = default;
  [[nodiscard]] virtual HostStartupAuditReport audit() = 0;
};

// Canonicalization computes unkeyed integrity digests. It does not prove
// provenance, ledger inclusion, or admission authority.
[[nodiscard]] HostStartupAuditPolicy canonicalize_host_startup_audit_policy(
    HostStartupAuditPolicy policy);
[[nodiscard]] HostStartupAuditFinding canonicalize_host_startup_audit_finding(
    HostStartupAuditFinding finding);
[[nodiscard]] HostStartupAuditReport canonicalize_host_startup_audit_report(
    HostStartupAuditReport report);
[[nodiscard]] HostStartupAuditReceipt canonicalize_host_startup_audit_receipt(
    HostStartupAuditReceipt receipt, const HostStartupAuditReport& report);

void validate_host_startup_audit_report(const HostStartupAuditReport& report);
void validate_host_startup_audit_receipt(
    const HostStartupAuditReceipt& receipt,
    const HostStartupAuditReport& report);

[[nodiscard]] nlohmann::json host_startup_audit_report_json(
    const HostStartupAuditReport& report);
// Decoded values are untrusted inspection data. They may be validated and
// replay-checked, but must never be treated as proof of ledger inclusion.
[[nodiscard]] HostStartupAuditReport decode_untrusted_host_startup_audit_report(
    const nlohmann::json& source);
[[nodiscard]] nlohmann::json host_startup_audit_receipt_json(
    const HostStartupAuditReceipt& receipt,
    const HostStartupAuditReport& report);
[[nodiscard]] HostStartupAuditReceipt decode_untrusted_host_startup_audit_receipt(
    const nlohmann::json& source, const HostStartupAuditReport& report);

}  // namespace trainvm
