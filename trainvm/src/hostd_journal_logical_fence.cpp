#include "trainvm/hostd_journal_logical_fence.hpp"

#include <ranges>
#include <stdexcept>
#include <string>
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
  // Enumerated rather than one disjunction. "Stale or inexact" named twelve
  // possible causes and distinguished none of them, and it is the last word a
  // rejected grant produces: the controller retries until its lease expires,
  // after which every later attempt reports only the expired lease and this
  // reason is gone.
  if (durable.authority.journal_id != attribution.journal_id)
    reject("journal names authority " + durable.authority.journal_id +
           "; the session claims " + attribution.journal_id);
  if (durable.authority.host.boot_id != observed.boot_id)
    reject("journal authority was written under boot " +
           durable.authority.host.boot_id + "; this host is now booted as " +
           observed.boot_id);
  if (lease.concurrency_key != attribution.concurrency_key)
    reject("durable lease holds concurrency key " + lease.concurrency_key +
           "; the session claims " + attribution.concurrency_key);
  if (lease.owner_run_id != attribution.run_id)
    reject("durable lease is owned by " + lease.owner_run_id +
           "; the session claims " + attribution.run_id);
  if (lease.lease_id != attribution.logical_lease_id)
    reject("durable lease is " + lease.lease_id + "; the session claims " +
           attribution.logical_lease_id);
  if (lease.fencing_token != attribution.logical_fencing_token)
    reject("durable lease carries fencing token " +
           std::to_string(lease.fencing_token) + "; the session claims " +
           std::to_string(attribution.logical_fencing_token));
  if (lease.clock_domain != ResourceLease::kBootTimeDomain)
    reject("durable lease uses clock domain " + lease.clock_domain +
           "; only " + std::string(ResourceLease::kBootTimeDomain) +
           " may fence a host mutation");
  if (lease.boot_id != observed.boot_id)
    reject("durable lease was taken under boot " + lease.boot_id +
           "; this host is now booted as " + observed.boot_id);
  if (lease.expires_boottime_ns <= observed.boot.nanoseconds)
    reject("durable lease expired " +
           std::to_string((observed.boot.nanoseconds -
                           lease.expires_boottime_ns) / 1'000'000LL) +
           "ms ago");
  if (durable.authority_event_sequence == 0U)
    reject("journal reports no authority event sequence for this lease");
  if (!valid_digest(durable.authority.host.host_id))
    reject("journal authority host id is not a sha256 digest: " +
           durable.authority.host.host_id);
  // The journal writes this as bare hex and enforces that on its own chain
  // head; only content digests such as host_id carry the "sha256:" prefix.
  // Validating it as a namespaced digest rejected every well-formed hash.
  if (!journal_event_hash_valid(durable.authority_event_hash))
    reject("journal authority event hash is not 64 hex characters: " +
           durable.authority_event_hash);

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
