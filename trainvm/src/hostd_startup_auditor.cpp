#include "trainvm/hostd_startup_auditor.hpp"

#include <openssl/rand.h>

#include <array>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

bool valid_identifier(std::string_view value) {
  if (value.empty() || value.size() >
                           HostStartupAuditBounds::maximum_identifier_bytes)
    return false;
  return std::ranges::all_of(value, [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '.' ||
           character == '_' || character == '-' || character == ':' ||
           character == '/' || character == '@';
  });
}

std::string audit_id() {
  std::array<unsigned char, 16U> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1)
    throw HostStartupAuditError("could not generate startup audit identity");
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result = "hostd-audit-";
  result.reserve(result.size() + bytes.size() * 2U);
  for (unsigned char byte : bytes) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

HostStartupAuditFinding active_fence_finding(std::size_t count,
                                             std::size_t exact_live,
                                             std::size_t gone,
                                             std::size_t mismatch,
                                             std::size_t failed,
                                             std::size_t intent_only) {
  return canonicalize_host_startup_audit_finding({
      .severity = HostStartupAuditFindingSeverity::blocking,
      .code = "process-adoption-required",
      .subject = "host-ledger",
      .detail =
          "startup found " + std::to_string(count) +
          " active resource fences; process recovery classified exact_live=" +
          std::to_string(exact_live) + ", gone=" + std::to_string(gone) +
          ", mismatch=" + std::to_string(mismatch) +
          ", observation_failed=" + std::to_string(failed) +
          ", intent_only=" + std::to_string(intent_only) +
          "; durable process adoption is required",
      .evidence_digest = {},
  });
}

}  // namespace

HostdConfiguredStartupAuditor::HostdConfiguredStartupAuditor(
    SQLiteHostLedger& ledger, AuthorityClock& clock,
    HostdConfiguredStartupAuditorConfig config)
    : ledger_(ledger), clock_(clock), config_(std::move(config)) {
  if (config_.api_version != kHostdConfiguredStartupAuditorApiVersion ||
      !valid_identifier(config_.broker_instance_id))
    throw HostStartupAuditError(
        "configured startup auditor identity is invalid");
  const HostStartupAuditPolicy canonical =
      canonicalize_host_startup_audit_policy(config_.policy);
  if (canonical != config_.policy)
    throw HostStartupAuditError(
        "configured startup auditor policy is not canonical");
}

HostStartupAuditReport HostdConfiguredStartupAuditor::audit() {
  const AuthorityTimeSample begin = clock_.sample();
  const HostInventoryReceipt inventory = ledger_.inventory();
  const HostLedgerChainHead head_before = ledger_.chain_head();
  const ResourceOccupancySnapshot occupancy_before = ledger_.occupancy();
  std::vector<HostProcessRecoveryRecord> recovery =
      ledger_.active_process_recovery_records();

  std::vector<HostStartupAuditFinding> findings;
  if (!occupancy_before.active_fences.empty()) {
    LinuxProcessRecoveryProbe probe;
    process_recovery_.recover(std::move(recovery), probe);
    const LinuxProcessRecoverySummary& summary = process_recovery_.summary();
    findings.push_back(active_fence_finding(
        occupancy_before.active_fences.size(), summary.exact_live,
        summary.already_gone, summary.identity_mismatch,
        summary.observation_failed, summary.intent_only));
  } else {
    LinuxProcessRecoveryProbe probe;
    process_recovery_.recover({}, probe);
  }

  const HostLedgerChainHead head_after = ledger_.chain_head();
  const ResourceOccupancySnapshot occupancy_after = ledger_.occupancy();
  const AuthorityTimeSample end = clock_.sample();
  if (begin.boot_id != inventory.boot_id || end.boot_id != inventory.boot_id)
    throw HostStartupAuditError(
        "startup audit clock and inventory boot identity disagree");

  const bool blocking = std::ranges::any_of(
      findings, [](const HostStartupAuditFinding& finding) {
        return finding.severity == HostStartupAuditFindingSeverity::blocking;
      });
  const bool unstable = head_before != head_after ||
                        occupancy_before != occupancy_after;
  const bool failed =
      (config_.policy.fail_on_blocking_findings && blocking) ||
      (config_.policy.require_stable_occupancy && unstable);
  return canonicalize_host_startup_audit_report({
      .api_version = std::string(kHostStartupAuditReportApiVersion),
      .audit_id = audit_id(),
      .host_id = inventory.host_id,
      .boot_id = inventory.boot_id,
      .broker_epoch = inventory.broker_epoch,
      .broker_instance_id = config_.broker_instance_id,
      .inventory = inventory,
      .pre_audit_occupancy = occupancy_before,
      .post_audit_occupancy = occupancy_after,
      .ledger_head_before = head_before,
      .ledger_head_after_observation = head_after,
      .policy = config_.policy,
      .findings = std::move(findings),
      .disposition = failed ? HostStartupAuditDisposition::failed
                            : HostStartupAuditDisposition::passed,
      .observed_begin_boottime_ns = begin.boot.nanoseconds,
      .observed_end_boottime_ns = end.boot.nanoseconds,
      .findings_digest = {},
      .report_digest = {},
  });
}

const LinuxProcessRecoverySet&
HostdConfiguredStartupAuditor::process_recovery() const noexcept {
  return process_recovery_;
}

LinuxProcessRecoverySet& HostdConfiguredStartupAuditor::process_recovery()
    noexcept {
  return process_recovery_;
}

}  // namespace trainvm
