#include "trainvm/hostd_journal_logical_fence.hpp"

#include <ranges>
#include <stdexcept>
#include <utility>

#include "trainvm/json.hpp"

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

[[noreturn]] void reject(std::string message) {
  throw HostdUnauthorized(std::move(message));
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

}  // namespace

HostdJournalLogicalFenceBoundary::HostdJournalLogicalFenceBoundary(
    Journal& journal)
    : journal_(journal) {}

JournalLogicalFenceSnapshot HostdJournalLogicalFenceBoundary::snapshot(
    const HostdSessionAttribution& attribution,
    const AuthorityTimeSample& now) {
  return journal_.journal_logical_fence_snapshot(
      attribution.concurrency_key, attribution.run_id,
      attribution.logical_lease_id, attribution.logical_fencing_token, now);
}

JournalHostdLogicalFenceEvidenceSource::
    JournalHostdLogicalFenceEvidenceSource(
        std::shared_ptr<IHostdJournalLogicalFenceBoundary> journal,
        AuthorityClock& clock)
    : journal_(std::move(journal)), clock_(clock) {
  if (!journal_)
    throw std::invalid_argument(
        "journal logical-fence evidence requires a retained boundary");
}

JournalHostdLogicalFenceEvidenceSource::
    JournalHostdLogicalFenceEvidenceSource(Journal& journal,
                                            AuthorityClock& clock)
    : JournalHostdLogicalFenceEvidenceSource(
          std::make_shared<HostdJournalLogicalFenceBoundary>(journal), clock) {}

HostdLogicalFenceEvidence JournalHostdLogicalFenceEvidenceSource::attest(
    const HostdSessionAttribution& attribution) {
  const AuthorityTimeSample observed = clock_.sample();
  const JournalLogicalFenceSnapshot durable =
      journal_->snapshot(attribution, observed);
  const ResourceLease& lease = durable.lease;
  if (durable.authority.journal_id != attribution.journal_id ||
      durable.authority.host.boot_id != observed.boot_id ||
      lease.concurrency_key != attribution.concurrency_key ||
      lease.owner_run_id != attribution.run_id ||
      lease.lease_id != attribution.logical_lease_id ||
      lease.fencing_token != attribution.logical_fencing_token ||
      lease.clock_domain != ResourceLease::kBootTimeDomain ||
      lease.boot_id != observed.boot_id ||
      lease.expires_boottime_ns <= observed.boot.nanoseconds ||
      durable.authority_event_sequence == 0U ||
      !valid_digest(durable.authority.host.host_id) ||
      !valid_digest(durable.authority_event_hash)) {
    reject("journal logical-fence evidence is stale or inexact");
  }

  const nlohmann::json material{
      {"api_version", kHostdLogicalFenceEvidenceApiVersion},
      {"journal_id", attribution.journal_id},
      {"run_id", attribution.run_id},
      {"concurrency_key", attribution.concurrency_key},
      {"logical_lease_id", attribution.logical_lease_id},
      {"logical_fencing_token", attribution.logical_fencing_token},
      {"boot_id", observed.boot_id},
      {"host_id", durable.authority.host.host_id},
      {"journal_directory_device", durable.authority.file.directory_device},
      {"journal_directory_inode", durable.authority.file.directory_inode},
      {"journal_device", durable.authority.file.device},
      {"journal_inode", durable.authority.file.inode},
      {"journal_authority_device", durable.authority.file.authority_device},
      {"journal_authority_inode", durable.authority.file.authority_inode},
      {"observed_boottime_ns", observed.boot.nanoseconds},
      {"expires_boottime_ns", lease.expires_boottime_ns},
      {"authority_revision", durable.authority_revision},
      {"authority_event_sequence", durable.authority_event_sequence},
      {"authority_event_hash", durable.authority_event_hash},
      {"live", true},
      {"cleanup_authorized", false},
  };
  return {.api_version = std::string(kHostdLogicalFenceEvidenceApiVersion),
          .attribution = attribution,
          .live = true,
          .cleanup_authorized = false,
          .cleanup_allocation_id = {},
          .cleanup_grant_digest = {},
          .cleanup_release_request_digest = {},
          .evidence_digest = "sha256:" + sha256_hex(material.dump())};
}

}  // namespace trainvm
