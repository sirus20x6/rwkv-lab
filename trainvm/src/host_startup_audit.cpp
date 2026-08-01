#include "trainvm/host_startup_audit.hpp"

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <ranges>
#include <tuple>

#include "trainvm/reflection_json.hpp"

namespace trainvm {
namespace {

struct EvpDeleter final {
  void operator()(EVP_MD_CTX* value) const { EVP_MD_CTX_free(value); }
};

std::string sha256(std::string_view domain, std::string_view value) {
  std::unique_ptr<EVP_MD_CTX, EvpDeleter> context(EVP_MD_CTX_new());
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1 ||
      EVP_DigestUpdate(context.get(), domain.data(), domain.size()) != 1) {
    throw HostStartupAuditError("could not initialize startup-audit digest");
  }
  const char separator = '\0';
  if (EVP_DigestUpdate(context.get(), &separator, 1U) != 1 ||
      EVP_DigestUpdate(context.get(), value.data(), value.size()) != 1) {
    throw HostStartupAuditError("could not update startup-audit digest");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> bytes{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), bytes.data(), &length) != 1) {
    throw HostStartupAuditError("could not finalize startup-audit digest");
  }
  static constexpr char digits[] = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(7U + static_cast<std::size_t>(length) * 2U);
  for (unsigned int index = 0; index < length; ++index) {
    result.push_back(digits[bytes[index] >> 4U]);
    result.push_back(digits[bytes[index] & 0x0fU]);
  }
  return result;
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool printable_ascii(std::string_view value) {
  return std::ranges::all_of(value, [](unsigned char character) {
    return character >= 0x20U && character <= 0x7eU;
  });
}

void require_identifier(std::string_view value, std::string_view field) {
  if (value.empty() ||
      value.size() > HostStartupAuditBounds::maximum_identifier_bytes ||
      !printable_ascii(value)) {
    throw HostStartupAuditError(std::string(field) + " is not a bounded identifier");
  }
}

void require_head(const HostLedgerChainHead& head, std::string_view field) {
  if (head.ledger_sequence >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
      !valid_digest(head.chain_hash)) {
    throw HostStartupAuditError(std::string(field) + " has an invalid chain hash");
  }
}

void require_json_structure(const nlohmann::json& source, std::size_t depth,
                            std::size_t& nodes, std::size_t& scalar_bytes,
                            std::string_view description) {
  if (depth > HostStartupAuditBounds::maximum_json_depth) {
    throw HostStartupAuditError(std::string(description) +
                                " exceeds its JSON structural-depth bound");
  }
  if (++nodes > HostStartupAuditBounds::maximum_json_nodes) {
    throw HostStartupAuditError(std::string(description) +
                                " exceeds its JSON node bound");
  }
  if ((source.is_array() || source.is_object()) &&
      source.size() > HostStartupAuditBounds::maximum_json_container_width) {
    throw HostStartupAuditError(std::string(description) +
                                " exceeds its JSON container-width bound");
  }
  if (source.is_string()) {
    const std::size_t bytes = source.get_ref<const std::string&>().size();
    if (bytes > HostStartupAuditBounds::maximum_json_bytes -
                    std::min(scalar_bytes,
                             HostStartupAuditBounds::maximum_json_bytes)) {
      throw HostStartupAuditError(std::string(description) +
                                  " exceeds its JSON scalar-byte bound");
    }
    scalar_bytes += bytes;
  }
  if (source.is_object()) {
    for (const auto& [key, child] : source.items()) {
      if (key.size() > HostStartupAuditBounds::maximum_json_bytes -
                           std::min(scalar_bytes,
                                    HostStartupAuditBounds::maximum_json_bytes)) {
        throw HostStartupAuditError(std::string(description) +
                                    " exceeds its JSON scalar-byte bound");
      }
      scalar_bytes += key.size();
      require_json_structure(child, depth + 1U, nodes, scalar_bytes,
                             description);
    }
  } else if (source.is_array()) {
    for (const auto& child : source) {
      require_json_structure(child, depth + 1U, nodes, scalar_bytes,
                             description);
    }
  }
}

// Every startup-audit encoder, digest calculation, and decoder passes through
// this one canonical byte boundary. In particular, an outer report is checked
// after its complete nested inventory and occupancy evidence is encoded.
std::string bounded_canonical_json_bytes(const nlohmann::json& source,
                                         std::string_view description) {
  std::size_t nodes = 0U;
  std::size_t scalar_bytes = 0U;
  require_json_structure(source, 1U, nodes, scalar_bytes, description);
  const std::string bytes = source.dump();
  if (bytes.size() > HostStartupAuditBounds::maximum_json_bytes) {
    throw HostStartupAuditError(std::string(description) +
                                " exceeds its canonical JSON byte bound");
  }
  return bytes;
}

template <typename T>
T strict_decode(const nlohmann::json& source, std::string_view description) {
  (void)bounded_canonical_json_bytes(source, description);
  T result;
  std::vector<Diagnostic> diagnostics;
  if (!decode_json(source, result, "", diagnostics)) {
    throw HostStartupAuditError(std::string(description) + " decoding failed");
  }
  return result;
}

nlohmann::json policy_digest_json(const HostStartupAuditPolicy& policy) {
  return {{"api_version", policy.api_version},
          {"require_stable_occupancy", policy.require_stable_occupancy},
          {"fail_on_blocking_findings", policy.fail_on_blocking_findings},
          {"maximum_findings", policy.maximum_findings}};
}

nlohmann::json finding_digest_json(const HostStartupAuditFinding& finding) {
  return {{"severity", enum_to_string(finding.severity)},
          {"code", finding.code},
          {"subject", finding.subject},
          {"detail", finding.detail}};
}

nlohmann::json report_digest_json(const HostStartupAuditReport& report) {
  nlohmann::json value = encode_json(report);
  value.erase("report_digest");
  return value;
}

nlohmann::json receipt_digest_json(const HostStartupAuditReceipt& receipt) {
  nlohmann::json value = encode_json(receipt);
  value.erase("receipt_digest");
  return value;
}

auto finding_order(const HostStartupAuditFinding& finding) {
  return std::tuple{static_cast<unsigned int>(finding.severity), finding.code,
                    finding.subject, finding.detail, finding.evidence_digest};
}

std::string findings_digest(
    const std::vector<HostStartupAuditFinding>& findings) {
  return sha256("trainvm.host-startup-audit-findings/v2",
                bounded_canonical_json_bytes(encode_json(findings),
                                             "startup-audit findings"));
}

void validate_policy(const HostStartupAuditPolicy& policy) {
  if (policy.api_version != kHostStartupAuditPolicyApiVersion ||
      policy.maximum_findings == 0U ||
      static_cast<std::size_t>(policy.maximum_findings) >
          HostStartupAuditBounds::maximum_findings ||
      policy.policy_digest !=
          sha256("trainvm.host-startup-audit-policy/v2",
                 bounded_canonical_json_bytes(policy_digest_json(policy),
                                              "startup-audit policy"))) {
    throw HostStartupAuditError("startup-audit policy is invalid");
  }
}

void validate_finding(const HostStartupAuditFinding& finding) {
  const bool valid_severity =
      finding.severity == HostStartupAuditFindingSeverity::informational ||
      finding.severity == HostStartupAuditFindingSeverity::warning ||
      finding.severity == HostStartupAuditFindingSeverity::blocking;
  if (!valid_severity || finding.code.empty() ||
      finding.code.size() > HostStartupAuditBounds::maximum_finding_code_bytes ||
      finding.subject.size() >
          HostStartupAuditBounds::maximum_finding_subject_bytes ||
      finding.detail.size() >
          HostStartupAuditBounds::maximum_finding_detail_bytes ||
      !printable_ascii(finding.code) || !printable_ascii(finding.subject) ||
      !printable_ascii(finding.detail) ||
      finding.evidence_digest !=
          sha256("trainvm.host-startup-audit-finding/v2",
                 bounded_canonical_json_bytes(finding_digest_json(finding),
                                              "startup-audit finding"))) {
    throw HostStartupAuditError("startup-audit finding is invalid");
  }
}

}  // namespace

HostStartupAuditPolicy canonicalize_host_startup_audit_policy(
    HostStartupAuditPolicy policy) {
  policy.policy_digest = sha256("trainvm.host-startup-audit-policy/v2",
                                bounded_canonical_json_bytes(
                                    policy_digest_json(policy),
                                    "startup-audit policy"));
  validate_policy(policy);
  return policy;
}

HostStartupAuditFinding canonicalize_host_startup_audit_finding(
    HostStartupAuditFinding finding) {
  finding.evidence_digest = sha256("trainvm.host-startup-audit-finding/v2",
                                   bounded_canonical_json_bytes(
                                       finding_digest_json(finding),
                                       "startup-audit finding"));
  validate_finding(finding);
  return finding;
}

HostStartupAuditReport canonicalize_host_startup_audit_report(
    HostStartupAuditReport report) {
  std::ranges::sort(report.findings, {}, finding_order);
  report.findings_digest = findings_digest(report.findings);
  report.report_digest = sha256("trainvm.host-startup-audit-report/v2",
                                bounded_canonical_json_bytes(
                                    report_digest_json(report),
                                    "startup-audit report"));
  validate_host_startup_audit_report(report);
  (void)bounded_canonical_json_bytes(encode_json(report),
                                     "startup-audit report");
  return report;
}

HostStartupAuditReceipt canonicalize_host_startup_audit_receipt(
    HostStartupAuditReceipt receipt, const HostStartupAuditReport& report) {
  receipt.receipt_digest = sha256("trainvm.host-startup-audit-receipt/v2",
                                  bounded_canonical_json_bytes(
                                      receipt_digest_json(receipt),
                                      "startup-audit receipt"));
  validate_host_startup_audit_receipt(receipt, report);
  (void)bounded_canonical_json_bytes(encode_json(receipt),
                                     "startup-audit receipt");
  return receipt;
}

void validate_host_startup_audit_report(const HostStartupAuditReport& report) {
  if (report.api_version != kHostStartupAuditReportApiVersion) {
    throw HostStartupAuditError("unsupported startup-audit report api_version");
  }
  require_identifier(report.audit_id, "audit_id");
  require_identifier(report.host_id, "host_id");
  require_identifier(report.boot_id, "boot_id");
  require_identifier(report.broker_epoch, "broker_epoch");
  require_identifier(report.broker_instance_id, "broker_instance_id");
  validate_host_inventory(report.inventory);
  validate_resource_occupancy(report.inventory, report.pre_audit_occupancy);
  validate_resource_occupancy(report.inventory, report.post_audit_occupancy);
  require_head(report.ledger_head_before, "ledger_head_before");
  require_head(report.ledger_head_after_observation,
               "ledger_head_after_observation");
  validate_policy(report.policy);
  if (report.host_id != report.inventory.host_id ||
      report.boot_id != report.inventory.boot_id ||
      report.broker_epoch != report.inventory.broker_epoch) {
    throw HostStartupAuditError("startup-audit identity diverges from inventory");
  }
  if (report.ledger_head_before != report.ledger_head_after_observation) {
    throw HostStartupAuditError(
        "startup-audit changed the host ledger while observing it");
  }
  if (report.pre_audit_occupancy.ledger_sequence !=
          report.ledger_head_before.ledger_sequence ||
      report.post_audit_occupancy.ledger_sequence !=
          report.ledger_head_after_observation.ledger_sequence) {
    throw HostStartupAuditError(
        "startup-audit occupancy is not bound to its ledger heads");
  }
  if (report.findings.size() >
          static_cast<std::size_t>(report.policy.maximum_findings) ||
      !std::ranges::is_sorted(report.findings, {}, finding_order) ||
      std::ranges::adjacent_find(report.findings, {}, finding_order) !=
          report.findings.end()) {
    throw HostStartupAuditError("startup-audit findings are not bounded canonical evidence");
  }
  for (const auto& finding : report.findings) validate_finding(finding);
  if (report.observed_begin_boottime_ns < 0 ||
      report.observed_end_boottime_ns < report.observed_begin_boottime_ns) {
    throw HostStartupAuditError("startup-audit observation time is invalid");
  }
  const bool occupancy_changed =
      report.pre_audit_occupancy != report.post_audit_occupancy;
  const bool blocking = std::ranges::any_of(report.findings, [](const auto& finding) {
    return finding.severity == HostStartupAuditFindingSeverity::blocking;
  });
  const bool must_fail =
      (report.policy.require_stable_occupancy && occupancy_changed) ||
      (report.policy.fail_on_blocking_findings && blocking);
  const bool valid_disposition =
      report.disposition == HostStartupAuditDisposition::passed ||
      report.disposition == HostStartupAuditDisposition::failed;
  if (!valid_disposition ||
      (report.disposition == HostStartupAuditDisposition::failed) != must_fail ||
      report.findings_digest != findings_digest(report.findings) ||
      report.report_digest !=
          sha256("trainvm.host-startup-audit-report/v2",
                 bounded_canonical_json_bytes(report_digest_json(report),
                                              "startup-audit report"))) {
    throw HostStartupAuditError("startup-audit report decision or digest is invalid");
  }
}

void validate_host_startup_audit_receipt(
    const HostStartupAuditReceipt& receipt,
    const HostStartupAuditReport& report) {
  validate_host_startup_audit_report(report);
  require_head(receipt.ledger_head_before, "receipt ledger_head_before");
  require_head(receipt.committed_ledger_head, "committed_ledger_head");
  if (receipt.api_version != kHostStartupAuditReceiptApiVersion ||
      receipt.audit_id != report.audit_id ||
      receipt.report_digest != report.report_digest ||
      receipt.host_id != report.host_id || receipt.boot_id != report.boot_id ||
      receipt.broker_epoch != report.broker_epoch ||
      receipt.broker_instance_id != report.broker_instance_id ||
      receipt.inventory_digest != report.inventory.inventory_digest ||
      receipt.topology_digest != report.inventory.topology_digest ||
      receipt.pre_occupancy_digest !=
          report.pre_audit_occupancy.occupancy_digest ||
      receipt.post_occupancy_digest !=
          report.post_audit_occupancy.occupancy_digest ||
      receipt.policy_digest != report.policy.policy_digest ||
      receipt.findings_digest != report.findings_digest ||
      receipt.disposition != report.disposition ||
      receipt.ledger_head_before != report.ledger_head_before ||
      receipt.committed_ledger_head.ledger_sequence !=
          receipt.ledger_head_before.ledger_sequence + 1U ||
      !valid_digest(receipt.commit_record_digest) ||
      receipt.committed_boottime_ns < report.observed_end_boottime_ns ||
      receipt.committed_wall_time_ns < 0 ||
      receipt.receipt_digest !=
          sha256("trainvm.host-startup-audit-receipt/v2",
                 bounded_canonical_json_bytes(receipt_digest_json(receipt),
                                              "startup-audit receipt"))) {
    throw HostStartupAuditError("startup-audit receipt is invalid");
  }
}

nlohmann::json host_startup_audit_report_json(
    const HostStartupAuditReport& report) {
  validate_host_startup_audit_report(report);
  nlohmann::json result = encode_json(report);
  (void)bounded_canonical_json_bytes(result, "startup-audit report");
  return result;
}

HostStartupAuditReport decode_untrusted_host_startup_audit_report(
    const nlohmann::json& source) {
  auto report = strict_decode<HostStartupAuditReport>(source,
                                                       "startup-audit report");
  validate_host_startup_audit_report(report);
  return report;
}

nlohmann::json host_startup_audit_receipt_json(
    const HostStartupAuditReceipt& receipt,
    const HostStartupAuditReport& report) {
  validate_host_startup_audit_receipt(receipt, report);
  nlohmann::json result = encode_json(receipt);
  (void)bounded_canonical_json_bytes(result, "startup-audit receipt");
  return result;
}

HostStartupAuditReceipt decode_untrusted_host_startup_audit_receipt(
    const nlohmann::json& source, const HostStartupAuditReport& report) {
  auto receipt = strict_decode<HostStartupAuditReceipt>(
      source, "startup-audit receipt");
  validate_host_startup_audit_receipt(receipt, report);
  return receipt;
}

}  // namespace trainvm
