#include "trainvm/hostd.hpp"

#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumIdentifierBytes = 192U;
constexpr std::size_t kMaximumPoisonReasonBytes = 512U;
constexpr std::size_t kMaximumConfiguredSessions = 65536U;
constexpr std::size_t kMaximumConfiguredScopes = 262144U;
constexpr std::string_view kSessionIdPrefix = "hostd-session-";
constexpr std::size_t kSessionIdRandomBytes = 24U;
constexpr std::size_t kSessionIdBytes =
    kSessionIdPrefix.size() + kSessionIdRandomBytes * 2U;

bool valid_identifier(std::string_view value) {
  if (value.empty() || value.size() > kMaximumIdentifierBytes)
    return false;
  return std::ranges::all_of(value, [](const char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '.' || character == '_' ||
           character == '-' || character == ':' || character == '/' ||
           character == '@';
  });
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_attribution(const HostdSessionAttribution &attribution) {
  return valid_identifier(attribution.journal_id) &&
         valid_identifier(attribution.run_id) &&
         valid_identifier(attribution.concurrency_key) &&
         valid_identifier(attribution.logical_lease_id) &&
         attribution.logical_fencing_token > 0U;
}

bool valid_session_id(std::string_view value) {
  return value.size() == kSessionIdBytes && value.starts_with(kSessionIdPrefix) &&
         std::ranges::all_of(value.substr(kSessionIdPrefix.size()),
                             [](const char character) {
                               return (character >= '0' && character <= '9') ||
                                      (character >= 'a' && character <= 'f');
                             });
}

bool valid_peer_grade(HostdPeerEnforcementGrade grade) {
  return grade == HostdPeerEnforcementGrade::observed_only ||
         grade == HostdPeerEnforcementGrade::service_identity_enforced;
}

bool valid_access(HostdSessionAccess access) {
  return access == HostdSessionAccess::denied ||
         access == HostdSessionAccess::read_only ||
         access == HostdSessionAccess::release_only ||
         access == HostdSessionAccess::grant_release;
}

std::string logical_scope_key(const HostdSessionAttribution &attribution) {
  return attribution.journal_id + "\n" + attribution.run_id + "\n" +
         attribution.concurrency_key + "\n" + attribution.logical_lease_id;
}

std::string random_session_id() {
  std::array<unsigned char, kSessionIdRandomBytes> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw HostdError("could not generate hostd session identity");
  }
  static constexpr char digits[] = "0123456789abcdef";
  std::string result(kSessionIdPrefix);
  result.reserve(result.size() + bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    result.push_back(digits[byte >> 4U]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

class SQLiteHostdLedgerBoundary final : public IHostdLedgerBoundary {
public:
  explicit SQLiteHostdLedgerBoundary(std::shared_ptr<SQLiteHostLedger> ledger)
      : ledger_(std::move(ledger)) {}

  [[nodiscard]] bool verify() const override { return ledger_->verify(); }
  [[nodiscard]] HostInventoryReceipt inventory() const override {
    return ledger_->inventory();
  }
  [[nodiscard]] HostLedgerChainHead chain_head() const override {
    return ledger_->chain_head();
  }
  [[nodiscard]] ResourceOccupancySnapshot occupancy() const override {
    return ledger_->occupancy();
  }
  [[nodiscard]] HostStartupAuditLedgerCommitResult
  commit_startup_audit(const HostStartupAuditReport &report,
                       const HostLedgerTime &now) override {
    return ledger_->commit_startup_audit(report, now);
  }
  [[nodiscard]] HostLedgerAdmissionFinalizeResult
  finalize_startup_admission(const HostStartupAuditReport &report,
                             const HostStartupAuditReceipt &receipt,
                             const HostLedgerTime &now) override {
    return ledger_->finalize_startup_admission(report, receipt, now);
  }
  [[nodiscard]] BundleRequestResult
  request_bundle(const ResourceBundleRequest &request,
                 const HostLedgerTime &now,
                 const HostLedgerAdmissionEpoch &admission_epoch) override {
    return ledger_->request_bundle(request, now, admission_epoch);
  }
  [[nodiscard]] std::optional<BundleRequestResult>
  reconcile_bundle_outcome(
      const ResourceBundleRequest &request) const override {
    return ledger_->reconcile_bundle_outcome(request);
  }
  [[nodiscard]] BundleReleaseResult
  release_bundle(const ResourceReleaseRequest &request,
                 const HostLedgerTime &now) override {
    return ledger_->release_bundle(request, now);
  }

private:
  std::shared_ptr<SQLiteHostLedger> ledger_;
};

std::shared_ptr<IHostdLedgerBoundary>
wrap_sqlite_ledger(std::shared_ptr<SQLiteHostLedger> ledger) {
  if (!ledger)
    return nullptr;
  return std::make_shared<SQLiteHostdLedgerBoundary>(std::move(ledger));
}

} // namespace

struct HostGrantCoordinator::Implementation final {
  enum class MutationKind { none, grant, release };

  struct SessionRecord final {
    HostdConnectedSession session;
    std::shared_ptr<IHostdPeerEvidenceSource> peer_source;
    std::uint64_t generation{};
    std::size_t active_mutations{};
  };

  struct SessionSnapshot final {
    HostdConnectedSession session;
    std::shared_ptr<IHostdPeerEvidenceSource> peer_source;
    std::uint64_t generation{};
  };

  struct RuntimeIdentityEvidence final {
    bool verified{};
    std::optional<HostInventoryReceipt> inventory;
    std::string failure;
  };

  HostdCoordinatorConfig config;
  std::shared_ptr<IHostdLedgerBoundary> ledger;
  std::shared_ptr<IHostdLogicalFenceEvidenceSource> logical_fence_evidence;
  HostInventoryReceipt startup_inventory;
  HostdLifecycle lifecycle{HostdLifecycle::sealed};
  std::optional<HostStartupAuditReceipt> startup_audit;
  std::optional<HostLedgerAdmissionEpoch> admission_epoch;
  std::string poison_reason;
  std::map<std::string, SessionRecord> sessions;
  std::map<std::string, std::uint64_t> highest_logical_fences;
  std::map<std::string, std::size_t> active_scope_grants;
  std::uint64_t next_session_generation{1U};
  std::string reap_cursor;
  mutable std::mutex mutex;

  void poison(std::string reason) {
    // Poison is a one-way latch. In particular, an audit completion racing an
    // identity failure must neither clear the lifecycle nor replace the first
    // causal reason operators need to diagnose.
    if (lifecycle == HostdLifecycle::poisoned)
      return;
    lifecycle = HostdLifecycle::poisoned;
    if (reason.size() > kMaximumPoisonReasonBytes) {
      reason.resize(kMaximumPoisonReasonBytes);
    }
    poison_reason = std::move(reason);
  }

  [[noreturn]] void poison_and_throw(std::string reason) {
    poison(reason);
    throw HostdStateError(poison_reason);
  }

  [[nodiscard]] RuntimeIdentityEvidence observe_runtime_identity() const {
    RuntimeIdentityEvidence result;
    try {
      if (!ledger->verify()) {
        result.failure = "host ledger failed integrity verification";
        return result;
      }
      result.inventory = ledger->inventory();
      result.verified = true;
    } catch (const std::exception &error) {
      result.failure = std::string("host ledger identity is unavailable: ") +
                       error.what();
    } catch (...) {
      result.failure = "host ledger identity observation failed";
    }
    return result;
  }

  void apply_runtime_identity(const RuntimeIdentityEvidence &evidence,
                              bool throw_on_failure) {
    if (lifecycle == HostdLifecycle::poisoned) {
      if (throw_on_failure)
        throw HostdStateError(poison_reason);
      return;
    }
    if (!evidence.verified || !evidence.inventory) {
      poison(evidence.failure.empty() ? "host ledger identity is unavailable"
                                      : evidence.failure);
    } else if (*evidence.inventory != startup_inventory ||
               evidence.inventory->host_id != config.host_id ||
               evidence.inventory->boot_id != config.boot_id ||
               evidence.inventory->broker_epoch != config.broker_epoch) {
      poison("host ledger identity changed outside the admitted startup epoch");
    }
    if (throw_on_failure && lifecycle == HostdLifecycle::poisoned)
      throw HostdStateError(poison_reason);
  }

  static bool peer_shape_valid(const HostdPeerEvidence &peer) {
    return peer.api_version == kHostdPeerEvidenceApiVersion &&
           peer.peer_pid > 0 && valid_identifier(peer.service_identity) &&
           valid_peer_grade(peer.enforcement_grade) &&
           valid_access(peer.access) && valid_digest(peer.evidence_digest);
  }

  static HostdPeerEvidence
  observe_peer(const std::shared_ptr<IHostdPeerEvidenceSource> &source) {
    try {
      return source->observe();
    } catch (const HostdUnauthorized &) {
      throw;
    } catch (const std::exception &) {
      throw HostdUnauthorized("hostd session peer is no longer observable");
    } catch (...) {
      throw HostdUnauthorized("hostd session peer observation failed");
    }
  }

  HostdLogicalFenceEvidence observe_logical_fence(
      const HostdSessionAttribution &attribution) const {
    if (!logical_fence_evidence) {
      throw HostdStateError(
          "hostd has no durable logical-fence evidence source");
    }
    try {
      const HostdLogicalFenceEvidence evidence =
          logical_fence_evidence->attest(attribution);
      const bool cleanup_shape =
          evidence.cleanup_authorized
              ? valid_identifier(evidence.cleanup_allocation_id) &&
                    valid_digest(evidence.cleanup_grant_digest) &&
                    valid_digest(evidence.cleanup_release_request_digest)
              : evidence.cleanup_allocation_id.empty() &&
                    evidence.cleanup_grant_digest.empty() &&
                    evidence.cleanup_release_request_digest.empty();
      if (evidence.api_version != kHostdLogicalFenceEvidenceApiVersion ||
          evidence.attribution != attribution ||
          !cleanup_shape || !valid_digest(evidence.evidence_digest)) {
        throw HostdUnauthorized("logical lease evidence is malformed or inexact");
      }
      return evidence;
    } catch (const HostdStateError &) {
      throw;
    } catch (const HostdUnauthorized &) {
      throw;
    } catch (const std::exception &) {
      throw HostdUnauthorized("logical lease evidence is unavailable");
    } catch (...) {
      throw HostdUnauthorized("logical lease evidence failed");
    }
  }

  SessionSnapshot snapshot_session(std::string_view session_id,
                                   MutationKind mutation) const {
    if (!valid_session_id(session_id)) {
      throw HostdUnauthorized("hostd session identity is malformed");
    }
    const auto found = sessions.find(std::string(session_id));
    if (found == sessions.end()) {
      throw HostdUnauthorized("hostd session is unknown or disconnected");
    }
    const SessionRecord &record = found->second;
    const HostdConnectedSession &session = record.session;
    if (mutation == MutationKind::none)
      return {.session = session,
              .peer_source = record.peer_source,
              .generation = record.generation};
    if ((session.effective_access != HostdSessionAccess::grant_release &&
         session.effective_access != HostdSessionAccess::release_only) ||
        session.peer.enforcement_grade !=
            HostdPeerEnforcementGrade::service_identity_enforced ||
        !session.attribution) {
      throw HostdUnauthorized("hostd session is read-only");
    }
    const bool release_cleanup_while_blocked =
        mutation == MutationKind::release &&
        session.effective_access == HostdSessionAccess::release_only &&
        lifecycle == HostdLifecycle::startup_blocked;
    if (lifecycle != HostdLifecycle::admitting &&
        !release_cleanup_while_blocked) {
      throw HostdStateError("hostd is not admitting resource mutations");
    }
    if (mutation == MutationKind::grant) {
      if (session.effective_access != HostdSessionAccess::grant_release) {
        throw HostdUnauthorized("hostd cleanup session cannot request grants");
      }
      const auto fence =
          highest_logical_fences.find(logical_scope_key(*session.attribution));
      if (fence == highest_logical_fences.end() ||
          fence->second != session.attribution->logical_fencing_token) {
        throw HostdUnauthorized("hostd session has a stale logical fence");
      }
    }
    // A stale exact session may release its own ledger allocation. Applying the
    // current grant fence here would strand token-N resources as soon as token
    // N+1 becomes live. The ledger's exact allocation/grant CAS is the cleanup
    // authority; stale sessions can never request another grant.
    return {.session = session,
            .peer_source = record.peer_source,
            .generation = record.generation};
  }

  SessionRecord &exact_session(const SessionSnapshot &snapshot) {
    const auto found = sessions.find(snapshot.session.session_id);
    if (found == sessions.end() ||
        found->second.generation != snapshot.generation ||
        found->second.peer_source != snapshot.peer_source ||
        found->second.session != snapshot.session) {
      throw HostdUnauthorized("hostd session changed during authorization");
    }
    return found->second;
  }

  SessionRecord &revalidate_session(const SessionSnapshot &snapshot,
                                    MutationKind mutation) {
    SessionRecord &record = exact_session(snapshot);
    // Re-run all local predicates: a higher fence or poison may have landed
    // while callbacks executed without the mutex.
    (void)snapshot_session(snapshot.session.session_id, mutation);
    return record;
  }

  void require_exact_peer(const SessionSnapshot &snapshot,
                          const HostdPeerEvidence &observed) const {
    if (!peer_shape_valid(observed) || observed != snapshot.session.peer) {
      throw HostdUnauthorized(
          "hostd session peer identity is no longer live and exact");
    }
  }

  void finish_mutation(SessionRecord &record, bool reserved_grant) {
    if (record.active_mutations == 0U)
      poison_and_throw("hostd mutation reservation accounting underflow");
    --record.active_mutations;
    if (!reserved_grant)
      return;
    const std::string scope = logical_scope_key(*record.session.attribution);
    const auto active = active_scope_grants.find(scope);
    if (active == active_scope_grants.end() || active->second == 0U)
      poison_and_throw("hostd scope reservation accounting underflow");
    if (--active->second == 0U)
      active_scope_grants.erase(active);
  }

  void reclaim_scope_if_unused(std::string_view scope) {
    const bool retained = std::ranges::any_of(
        sessions, [&](const auto &entry) {
          return entry.second.session.effective_access ==
                     HostdSessionAccess::grant_release &&
                 entry.second.session.attribution &&
                 logical_scope_key(*entry.second.session.attribution) == scope;
        });
    if (!retained && !active_scope_grants.contains(std::string(scope)))
      highest_logical_fences.erase(std::string(scope));
  }

  static bool request_is_exactly_sealed(const ResourceBundleRequest &request) {
    try {
      return seal_resource_request(request) == request;
    } catch (...) {
      return false;
    }
  }

  static bool release_is_exactly_sealed(const ResourceReleaseRequest &request) {
    try {
      return seal_resource_release_request(request) == request;
    } catch (...) {
      return false;
    }
  }

  bool valid_grant_result(const BundleRequestResult &result,
                          const ResourceBundleRequest &request,
                          const HostdSessionAttribution &attribution) const {
    if (!valid_digest(result.outcome_digest))
      return false;
    if (result.status == BundleRequestStatus::busy)
      return !result.grant;
    if (result.status != BundleRequestStatus::granted || !result.grant)
      return false;
    const ResourceBundleGrant &grant = *result.grant;
    try {
      if (resource_bundle_grant_from_json(resource_bundle_grant_json(grant)) !=
          grant)
        return false;
    } catch (...) {
      return false;
    }
    return grant.api_version == kHostLedgerGrantApiVersion &&
           valid_identifier(grant.allocation_id) &&
           grant.request_id == request.request_id &&
           grant.request_digest == request.canonical_request_digest &&
           grant.journal_id == attribution.journal_id &&
           grant.run_id == attribution.run_id &&
           grant.logical_lease_id == attribution.logical_lease_id &&
           grant.logical_fencing_token == attribution.logical_fencing_token &&
           grant.host_id == config.host_id && grant.boot_id == config.boot_id &&
           grant.broker_epoch == config.broker_epoch &&
           grant.receipt_digest == result.outcome_digest;
  }

  bool valid_release_result(const BundleReleaseResult &result,
                            const ResourceReleaseRequest &request) const {
    const ResourceReleaseReceipt &receipt = result.receipt;
    try {
      if (resource_release_receipt_from_json(
              resource_release_receipt_json(receipt)) != receipt)
        return false;
    } catch (...) {
      return false;
    }
    return receipt.api_version == kHostLedgerReleaseApiVersion &&
           receipt.release_request_id == request.release_request_id &&
           receipt.release_request_digest == request.canonical_request_digest &&
           receipt.allocation_id == request.allocation_id &&
           receipt.grant_digest == request.grant_digest &&
           receipt.host_id == config.host_id && receipt.boot_id == config.boot_id &&
           receipt.broker_epoch == config.broker_epoch;
  }

  HostdCoordinatorStatus status_value() const {
    std::size_t admission_sessions = 0U;
    std::size_t stale_admission_sessions = 0U;
    std::size_t release_only_sessions = 0U;
    for (const auto &[id, record] : sessions) {
      (void)id;
      const auto &session = record.session;
      if (session.effective_access == HostdSessionAccess::release_only) {
        ++release_only_sessions;
        continue;
      }
      if (session.effective_access != HostdSessionAccess::grant_release ||
          !session.attribution)
        continue;
      const auto fence =
          highest_logical_fences.find(logical_scope_key(*session.attribution));
      const bool current = lifecycle == HostdLifecycle::admitting &&
                           fence != highest_logical_fences.end() &&
                           fence->second ==
                               session.attribution->logical_fencing_token;
      current ? ++admission_sessions : ++stale_admission_sessions;
    }
    return {.api_version = std::string(kHostdCoordinatorApiVersion),
            .lifecycle = lifecycle,
            .host_id = config.host_id,
            .boot_id = config.boot_id,
            .broker_epoch = config.broker_epoch,
            .inventory_digest = startup_inventory.inventory_digest,
            .live_sessions = sessions.size(),
            .admission_sessions = admission_sessions,
            .stale_admission_sessions = stale_admission_sessions,
            .release_only_sessions = release_only_sessions,
            .admission_counts_are_cached_evidence = true,
            .startup_audit = startup_audit,
            .poison_reason = poison_reason};
  }
};

HostGrantCoordinator::HostGrantCoordinator(
    HostdCoordinatorConfig config, std::shared_ptr<SQLiteHostLedger> ledger,
    std::shared_ptr<IHostdLogicalFenceEvidenceSource> logical_fence_evidence)
    : HostGrantCoordinator(std::move(config), wrap_sqlite_ledger(std::move(ledger)),
                           std::move(logical_fence_evidence)) {}

HostGrantCoordinator::HostGrantCoordinator(
    HostdCoordinatorConfig config,
    std::shared_ptr<IHostdLedgerBoundary> ledger,
    std::shared_ptr<IHostdLogicalFenceEvidenceSource> logical_fence_evidence)
    : implementation_(std::make_unique<Implementation>()) {
  if (config.api_version != kHostdCoordinatorApiVersion ||
      !valid_identifier(config.host_id) || !valid_identifier(config.boot_id) ||
      !valid_identifier(config.broker_epoch) ||
      config.maximum_live_sessions == 0U ||
      config.maximum_live_sessions > kMaximumConfiguredSessions ||
      config.maximum_logical_scopes == 0U ||
      config.maximum_logical_scopes > kMaximumConfiguredScopes) {
    throw HostdError("hostd coordinator configuration is invalid");
  }
  if (!ledger)
    throw HostdError("hostd coordinator requires a host ledger");
  try {
    if (!ledger->verify())
      throw HostdError("hostd coordinator requires a verified host ledger");
    implementation_->startup_inventory = ledger->inventory();
  } catch (const HostdError &) {
    throw;
  } catch (const std::exception &error) {
    throw HostdError(std::string("hostd coordinator ledger is unavailable: ") +
                     error.what());
  } catch (...) {
    throw HostdError("hostd coordinator ledger observation failed");
  }
  implementation_->config = std::move(config);
  implementation_->ledger = std::move(ledger);
  implementation_->logical_fence_evidence =
      std::move(logical_fence_evidence);
}

HostGrantCoordinator::~HostGrantCoordinator() = default;

namespace {

// Preserves the historical fixed-time contract for callers that already know a
// time later than the audit observation they are about to trigger.
class FixedStartupAuditCommitTime final
    : public IHostStartupAuditCommitTimeSource {
public:
  explicit FixedStartupAuditCommitTime(HostLedgerTime value) noexcept
      : value_(value) {}
  [[nodiscard]] HostLedgerTime commit_time() override { return value_; }

private:
  HostLedgerTime value_;
};

} // namespace

HostStartupAuditReceipt HostGrantCoordinator::run_startup_audit(
    IConfiguredHostStartupAuditorV2 &auditor, const HostLedgerTime &now) {
  FixedStartupAuditCommitTime fixed(now);
  return run_startup_audit_sampled(auditor, fixed);
}

HostStartupAuditReceipt HostGrantCoordinator::run_startup_audit(
    IConfiguredHostStartupAuditorV2 &auditor,
    IHostStartupAuditCommitTimeSource &commit_time) {
  return run_startup_audit_sampled(auditor, commit_time);
}

HostStartupAuditReceipt HostGrantCoordinator::run_startup_audit_sampled(
    IConfiguredHostStartupAuditorV2 &auditor,
    IHostStartupAuditCommitTimeSource &commit_time_source) {
  {
    std::scoped_lock lock(implementation_->mutex);
    if (implementation_->lifecycle != HostdLifecycle::sealed) {
      throw HostdStateError("startup audit may run exactly once from sealed");
    }
    implementation_->lifecycle = HostdLifecycle::startup_auditing;
  }

  HostStartupAuditReport report;
  try {
    // Auditor implementations are configured authority and may perform slow
    // observations. They never execute under the coordinator mutex.
    report = auditor.audit();
  } catch (const std::exception &error) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(std::string("startup audit failed: ") +
                            error.what());
    throw HostdStateError(implementation_->poison_reason);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison("startup audit failed with an unknown error");
    throw HostdStateError(implementation_->poison_reason);
  }

  // The commit time is sampled only now. The ledger refuses a startup-audit
  // commit that claims to predate the report's own end of observation, so a
  // time sampled before auditor.audit() ran can never be committed.
  HostLedgerTime now;
  try {
    now = commit_time_source.commit_time();
  } catch (const std::exception &error) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(
        std::string("startup audit commit time is unavailable: ") +
        error.what());
    throw HostdStateError(implementation_->poison_reason);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison("startup audit commit time is unavailable");
    throw HostdStateError(implementation_->poison_reason);
  }

  Implementation::RuntimeIdentityEvidence runtime;
  HostLedgerChainHead observed_head;
  ResourceOccupancySnapshot observed_occupancy;
  bool report_shape_valid = false;
  try {
    validate_host_startup_audit_report(report);
    report_shape_valid =
        decode_untrusted_host_startup_audit_report(
            host_startup_audit_report_json(report)) == report;
    runtime = implementation_->observe_runtime_identity();
    observed_head = implementation_->ledger->chain_head();
    observed_occupancy = implementation_->ledger->occupancy();
    if (!valid_digest(observed_head.chain_hash))
      throw HostdError("startup ledger returned a malformed chain head");
    validate_resource_occupancy(implementation_->startup_inventory,
                                observed_occupancy);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison("startup auditor returned malformed evidence");
    throw HostdStateError(implementation_->poison_reason);
  }

  const bool exact_identity =
      report_shape_valid && runtime.verified && runtime.inventory &&
      report.host_id == implementation_->config.host_id &&
      report.boot_id == implementation_->config.boot_id &&
      report.broker_epoch == implementation_->config.broker_epoch &&
      report.inventory == implementation_->startup_inventory &&
      report.inventory == *runtime.inventory;
  const bool exact_current_observation =
      exact_identity && report.ledger_head_before == observed_head &&
      report.ledger_head_after_observation == observed_head &&
      report.pre_audit_occupancy == observed_occupancy &&
      report.post_audit_occupancy == observed_occupancy;
  if (!exact_identity) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(
        "startup audit identity/inventory evidence is stale or inexact");
    throw HostdStateError(implementation_->poison_reason);
  }
  {
    std::scoped_lock lock(implementation_->mutex);
    if (implementation_->lifecycle != HostdLifecycle::startup_auditing) {
      if (implementation_->lifecycle != HostdLifecycle::poisoned)
        implementation_->poison(
            "startup audit lifecycle changed asynchronously");
      throw HostdStateError(implementation_->poison_reason);
    }
  }

  HostStartupAuditLedgerCommitResult committed;
  try {
    // A head/occupancy mismatch is allowed to reach the ledger only so an
    // exact lost-reply retry can replay its already committed audit ID. A new
    // commit still requires exact_current_observation below and in the ledger.
    committed = implementation_->ledger->commit_startup_audit(report, now);
  } catch (const std::exception &error) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(std::string("startup audit commit failed: ") +
                            error.what());
    throw HostdStateError(implementation_->poison_reason);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison("startup audit commit failed");
    throw HostdStateError(implementation_->poison_reason);
  }

  bool receipt_exact = false;
  try {
    validate_host_startup_audit_receipt(committed.receipt, report);
    receipt_exact =
        decode_untrusted_host_startup_audit_receipt(
            host_startup_audit_receipt_json(committed.receipt, report),
            report) == committed.receipt;
  } catch (...) {
    receipt_exact = false;
  }
  const bool blocking = std::ranges::any_of(
      report.findings, [](const HostStartupAuditFinding &finding) {
        return finding.severity == HostStartupAuditFindingSeverity::blocking;
      });
  if (!receipt_exact || (!committed.replayed && !exact_current_observation)) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(
        "startup ledger returned inexact audit commit inspection data");
    throw HostdStateError(implementation_->poison_reason);
  }
  if (report.disposition != HostStartupAuditDisposition::passed || blocking) {
    std::scoped_lock lock(implementation_->mutex);
    if (implementation_->lifecycle != HostdLifecycle::startup_auditing) {
      if (implementation_->lifecycle != HostdLifecycle::poisoned)
        implementation_->poison(
            "startup audit lifecycle changed after commit");
      throw HostdStateError(implementation_->poison_reason);
    }
    implementation_->startup_audit = committed.receipt;
    implementation_->lifecycle = HostdLifecycle::startup_blocked;
    implementation_->poison_reason =
        "startup audit committed blocking or failed evidence";
    throw HostdStateError(implementation_->poison_reason);
  }

  std::optional<HostLedgerAdmissionFinalizeResult> finalized;
  try {
    // The ledger owns the atomic audit-head/occupancy CAS. No inspection read
    // occurs after this call: its opaque epoch is the grant authority even if
    // a request lands before hostd acquires its local latch.
    finalized = implementation_->ledger->finalize_startup_admission(
        report, committed.receipt, now);
  } catch (const std::exception &error) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison(std::string("startup admission finalize failed: ") +
                            error.what());
    throw HostdStateError(implementation_->poison_reason);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    implementation_->poison("startup admission finalize failed");
    throw HostdStateError(implementation_->poison_reason);
  }

  std::scoped_lock lock(implementation_->mutex);
  if (implementation_->lifecycle != HostdLifecycle::startup_auditing) {
    if (implementation_->lifecycle != HostdLifecycle::poisoned)
      implementation_->poison(
          "startup audit lifecycle changed after admission finalize");
    throw HostdStateError(implementation_->poison_reason);
  }
  implementation_->startup_audit = committed.receipt;
  implementation_->admission_epoch = finalized->epoch;
  implementation_->lifecycle = HostdLifecycle::admitting;
  return committed.receipt;
}

HostdConnectedSession
HostGrantCoordinator::connect(HostdConnectRequest request,
                              std::shared_ptr<IHostdPeerEvidenceSource>
                                  peer_source) {
  if (!peer_source)
    throw HostdUnauthorized("hostd peer evidence source is missing");
  if (request.attribution && !valid_attribution(*request.attribution))
    throw HostdUnauthorized("hostd session attribution is invalid");

  bool near_capacity = false;
  {
    std::scoped_lock lock(implementation_->mutex);
    near_capacity = implementation_->sessions.size() + 1U >=
                    implementation_->config.maximum_live_sessions;
    if (request.attribution) {
      const std::string scope = logical_scope_key(*request.attribution);
      near_capacity =
          near_capacity ||
          (!implementation_->highest_logical_fences.contains(scope) &&
           implementation_->highest_logical_fences.size() >=
               implementation_->config.maximum_logical_scopes);
    }
  }
  if (near_capacity)
    (void)reap_dead_sessions(64U);

  const HostdPeerEvidence peer = Implementation::observe_peer(peer_source);
  if (!Implementation::peer_shape_valid(peer) ||
      peer.access == HostdSessionAccess::denied)
    throw HostdUnauthorized("peer evidence does not authorize a hostd session");
  if ((peer.access == HostdSessionAccess::grant_release ||
       peer.access == HostdSessionAccess::release_only) &&
      (peer.enforcement_grade !=
           HostdPeerEnforcementGrade::service_identity_enforced ||
       !request.attribution)) {
    throw HostdUnauthorized(
        "grant/release access requires enforced service identity and scope");
  }

  const bool mutation_requested =
      peer.access == HostdSessionAccess::grant_release ||
      peer.access == HostdSessionAccess::release_only;
  std::optional<HostdLogicalFenceEvidence> logical;
  std::optional<Implementation::RuntimeIdentityEvidence> runtime;
  if (mutation_requested) {
    logical = implementation_->observe_logical_fence(*request.attribution);
    runtime = implementation_->observe_runtime_identity();
  }

  std::scoped_lock lock(implementation_->mutex);
  if (implementation_->sessions.size() >=
      implementation_->config.maximum_live_sessions) {
    throw HostdStateError("hostd live-session capacity is exhausted");
  }
  if (!mutation_requested &&
      implementation_->sessions.size() >=
          implementation_->config.maximum_live_sessions - 1U) {
    throw HostdStateError(
        "hostd observer-session capacity preserves one mutation slot");
  }
  HostdSessionAccess effective_access = peer.access;
  if (mutation_requested) {
    const bool cleanup_while_blocked =
        peer.access == HostdSessionAccess::release_only &&
        implementation_->lifecycle == HostdLifecycle::startup_blocked;
    if (implementation_->lifecycle != HostdLifecycle::admitting &&
        !cleanup_while_blocked) {
      throw HostdStateError(
          "grant/release sessions require a passed startup audit");
    }
    implementation_->apply_runtime_identity(*runtime, true);
    if (peer.access == HostdSessionAccess::release_only) {
      if (!logical->cleanup_authorized) {
        throw HostdUnauthorized(
            "release-only session lacks durable cleanup authorization");
      }
    } else if (!logical->live) {
      if (!logical->cleanup_authorized) {
        throw HostdUnauthorized(
            "logical lease evidence is stale and has no cleanup intent");
      }
      effective_access = HostdSessionAccess::release_only;
    }
    const std::string scope = logical_scope_key(*request.attribution);
    const auto found = implementation_->highest_logical_fences.find(scope);
    if (effective_access == HostdSessionAccess::grant_release &&
        found != implementation_->highest_logical_fences.end() &&
        request.attribution->logical_fencing_token < found->second) {
      throw HostdUnauthorized("cannot connect with a stale logical fence");
    }
    if (effective_access == HostdSessionAccess::grant_release &&
        found != implementation_->highest_logical_fences.end() &&
        request.attribution->logical_fencing_token > found->second &&
        implementation_->active_scope_grants.contains(scope)) {
      throw HostdStateError(
          "cannot advance a logical fence while an older grant is in flight");
    }
    if (effective_access == HostdSessionAccess::grant_release &&
        found == implementation_->highest_logical_fences.end() &&
        implementation_->highest_logical_fences.size() >=
            implementation_->config.maximum_logical_scopes) {
      throw HostdStateError("hostd logical-scope capacity is exhausted");
    }
  }

  std::string session_id;
  for (std::size_t attempt = 0U; attempt < 8U; ++attempt) {
    session_id = random_session_id();
    if (!implementation_->sessions.contains(session_id))
      break;
    session_id.clear();
  }
  if (session_id.empty()) {
    throw HostdError("could not allocate a unique hostd session identity");
  }
  HostdConnectedSession session{.session_id = session_id,
                                .peer = peer,
                                .effective_access = effective_access,
                                .attribution = std::move(request.attribution)};
  if (implementation_->next_session_generation == 0U)
    implementation_->poison_and_throw("hostd session generation exhausted");
  const auto inserted = implementation_->sessions.emplace(
      session_id, Implementation::SessionRecord{
                      .session = session,
                      .peer_source = std::move(peer_source),
                      .generation = implementation_->next_session_generation++,
                      .active_mutations = 0U});
  if (!inserted.second) {
    throw HostdError("could not atomically reserve a hostd session");
  }
  if (session.effective_access == HostdSessionAccess::grant_release) {
    const std::string scope = logical_scope_key(*session.attribution);
    try {
      const auto [fence, created] =
          implementation_->highest_logical_fences.emplace(
              scope, session.attribution->logical_fencing_token);
      if (!created && session.attribution->logical_fencing_token > fence->second)
        fence->second = session.attribution->logical_fencing_token;
    } catch (...) {
      implementation_->sessions.erase(inserted.first);
      throw;
    }
  }
  return session;
}

void HostGrantCoordinator::disconnect(std::string_view session_id) {
  std::scoped_lock lock(implementation_->mutex);
  if (!valid_session_id(session_id)) {
    throw HostdUnauthorized("hostd session identity is malformed");
  }
  const auto found = implementation_->sessions.find(std::string(session_id));
  if (found == implementation_->sessions.end()) {
    throw HostdUnauthorized("hostd session is unknown or already disconnected");
  }
  if (found->second.active_mutations != 0U)
    throw HostdStateError("hostd session has an active resource mutation");
  std::optional<std::string> scope;
  if (found->second.session.effective_access ==
          HostdSessionAccess::grant_release &&
      found->second.session.attribution) {
    scope = logical_scope_key(*found->second.session.attribution);
  }
  implementation_->sessions.erase(found);
  if (scope)
    implementation_->reclaim_scope_if_unused(*scope);
}

std::size_t
HostGrantCoordinator::reap_dead_sessions(std::size_t max_observations) {
  if (max_observations == 0U)
    return 0U;
  max_observations = std::min(max_observations,
                              implementation_->config.maximum_live_sessions);
  std::vector<Implementation::SessionSnapshot> candidates;
  {
    std::scoped_lock lock(implementation_->mutex);
    if (implementation_->sessions.empty())
      return 0U;
    auto cursor =
        implementation_->sessions.upper_bound(implementation_->reap_cursor);
    for (std::size_t visited = 0U;
         visited < implementation_->sessions.size() &&
         candidates.size() < max_observations;
         ++visited) {
      if (cursor == implementation_->sessions.end())
        cursor = implementation_->sessions.begin();
      const auto current = cursor++;
      implementation_->reap_cursor = current->first;
      if (current->second.active_mutations == 0U) {
        candidates.push_back({.session = current->second.session,
                              .peer_source = current->second.peer_source,
                              .generation = current->second.generation});
      }
    }
  }

  std::vector<Implementation::SessionSnapshot> dead;
  for (const auto &candidate : candidates) {
    try {
      const HostdPeerEvidence observed =
          Implementation::observe_peer(candidate.peer_source);
      if (!Implementation::peer_shape_valid(observed) ||
          observed != candidate.session.peer)
        dead.push_back(candidate);
    } catch (...) {
      dead.push_back(candidate);
    }
  }

  std::size_t reclaimed = 0U;
  std::scoped_lock lock(implementation_->mutex);
  for (const auto &candidate : dead) {
    const auto found =
        implementation_->sessions.find(candidate.session.session_id);
    if (found == implementation_->sessions.end() ||
        found->second.generation != candidate.generation ||
        found->second.peer_source != candidate.peer_source ||
        found->second.session != candidate.session ||
        found->second.active_mutations != 0U)
      continue;
    std::optional<std::string> scope;
    if (found->second.session.effective_access ==
            HostdSessionAccess::grant_release &&
        found->second.session.attribution)
      scope = logical_scope_key(*found->second.session.attribution);
    implementation_->sessions.erase(found);
    if (scope)
      implementation_->reclaim_scope_if_unused(*scope);
    ++reclaimed;
  }
  return reclaimed;
}

BundleRequestResult
HostGrantCoordinator::request_bundle(std::string_view session_id,
                                     const ResourceBundleRequest &request,
                                     const HostLedgerTime &now) {
  if (!Implementation::request_is_exactly_sealed(request))
    throw HostdUnauthorized("bundle request is not exactly sealed");
  Implementation::SessionSnapshot snapshot;
  std::optional<HostLedgerAdmissionEpoch> admission_epoch;
  {
    std::scoped_lock lock(implementation_->mutex);
    snapshot = implementation_->snapshot_session(
        session_id, Implementation::MutationKind::grant);
    if (!implementation_->admission_epoch)
      implementation_->poison_and_throw(
          "admitting hostd has no ledger admission epoch");
    admission_epoch = implementation_->admission_epoch;
  }
  const HostdSessionAttribution attribution = *snapshot.session.attribution;
  if (request.journal_id != attribution.journal_id ||
      request.run_id != attribution.run_id ||
      request.logical_lease_id != attribution.logical_lease_id ||
      request.logical_fencing_token != attribution.logical_fencing_token) {
    throw HostdUnauthorized(
        "bundle request does not match the connected session attribution");
  }
  const HostdPeerEvidence peer =
      Implementation::observe_peer(snapshot.peer_source);
  const HostdLogicalFenceEvidence logical =
      implementation_->observe_logical_fence(attribution);
  const Implementation::RuntimeIdentityEvidence runtime =
      implementation_->observe_runtime_identity();
  {
    std::scoped_lock lock(implementation_->mutex);
    Implementation::SessionRecord &record = implementation_->revalidate_session(
        snapshot, Implementation::MutationKind::grant);
    implementation_->require_exact_peer(snapshot, peer);
    if (!logical.live)
      throw HostdUnauthorized("logical lease evidence is no longer live");
    implementation_->apply_runtime_identity(runtime, true);
    ++record.active_mutations;
    ++implementation_->active_scope_grants[logical_scope_key(attribution)];
  }

  BundleRequestResult result;
  try {
    result = implementation_->ledger->request_bundle(request, now,
                                                       *admission_epoch);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    auto &record = implementation_->exact_session(snapshot);
    implementation_->finish_mutation(record, true);
    throw;
  }
  std::scoped_lock lock(implementation_->mutex);
  Implementation::SessionRecord &record =
      implementation_->exact_session(snapshot);
  implementation_->finish_mutation(record, true);
  if (!implementation_->valid_grant_result(result, request, attribution))
    implementation_->poison_and_throw(
        "host ledger returned an unsealed or inexact grant outcome");
  if (implementation_->lifecycle != HostdLifecycle::admitting)
    throw HostdStateError(implementation_->poison_reason);
  return result;
}

std::optional<BundleRequestResult>
HostGrantCoordinator::reconcile_bundle_outcome(
    std::string_view session_id, const ResourceBundleRequest &request) {
  if (!Implementation::request_is_exactly_sealed(request))
    throw HostdUnauthorized("bundle request is not exactly sealed");
  Implementation::SessionSnapshot snapshot;
  {
    std::scoped_lock lock(implementation_->mutex);
    snapshot = implementation_->snapshot_session(
        session_id, Implementation::MutationKind::none);
    if (snapshot.session.effective_access == HostdSessionAccess::denied ||
        snapshot.session.peer.access == HostdSessionAccess::denied ||
        snapshot.session.peer.enforcement_grade !=
            HostdPeerEnforcementGrade::service_identity_enforced ||
        !snapshot.session.attribution) {
      throw HostdUnauthorized(
          "request outcome recovery requires an attributed service session");
    }
  }
  const HostdSessionAttribution attribution = *snapshot.session.attribution;
  if (request.journal_id != attribution.journal_id ||
      request.run_id != attribution.run_id ||
      request.logical_lease_id != attribution.logical_lease_id ||
      request.logical_fencing_token != attribution.logical_fencing_token) {
    throw HostdUnauthorized(
        "bundle request does not match the recovery session attribution");
  }
  const HostdPeerEvidence peer =
      Implementation::observe_peer(snapshot.peer_source);
  // Exact stale attribution is deliberately recoverable: a lost grant reply
  // may be the evidence needed to construct cleanup after its logical lease
  // has expired or been superseded. This observation still validates durable
  // evidence shape and attribution, but liveness is not grant authority here.
  (void)implementation_->observe_logical_fence(attribution);
  const Implementation::RuntimeIdentityEvidence runtime =
      implementation_->observe_runtime_identity();
  {
    std::scoped_lock lock(implementation_->mutex);
    Implementation::SessionRecord &record =
        implementation_->revalidate_session(
            snapshot, Implementation::MutationKind::none);
    implementation_->require_exact_peer(snapshot, peer);
    implementation_->apply_runtime_identity(runtime, true);
    ++record.active_mutations;
  }

  std::optional<BundleRequestResult> result;
  try {
    result = implementation_->ledger->reconcile_bundle_outcome(request);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    auto &record = implementation_->exact_session(snapshot);
    implementation_->finish_mutation(record, false);
    throw;
  }
  std::scoped_lock lock(implementation_->mutex);
  Implementation::SessionRecord &record =
      implementation_->exact_session(snapshot);
  implementation_->finish_mutation(record, false);
  if (result &&
      (!result->replayed ||
       !implementation_->valid_grant_result(*result, request, attribution))) {
    implementation_->poison_and_throw(
        "host ledger returned an unsealed or inexact recovery outcome");
  }
  if (implementation_->lifecycle == HostdLifecycle::poisoned)
    throw HostdStateError(implementation_->poison_reason);
  return result;
}

BundleReleaseResult
HostGrantCoordinator::release_bundle(std::string_view session_id,
                                     const ResourceReleaseRequest &request,
                                     const HostLedgerTime &now) {
  if (!Implementation::release_is_exactly_sealed(request))
    throw HostdUnauthorized("bundle release request is not exactly sealed");
  Implementation::SessionSnapshot snapshot;
  {
    std::scoped_lock lock(implementation_->mutex);
    snapshot = implementation_->snapshot_session(
        session_id, Implementation::MutationKind::release);
  }
  const HostdSessionAttribution attribution = *snapshot.session.attribution;
  if (request.journal_id != attribution.journal_id ||
      request.run_id != attribution.run_id ||
      request.logical_lease_id != attribution.logical_lease_id ||
      request.logical_fencing_token != attribution.logical_fencing_token) {
    throw HostdUnauthorized(
        "bundle release does not match the connected session attribution");
  }
  const HostdPeerEvidence peer =
      Implementation::observe_peer(snapshot.peer_source);
  std::optional<HostdLogicalFenceEvidence> cleanup;
  if (snapshot.session.effective_access == HostdSessionAccess::release_only)
    cleanup = implementation_->observe_logical_fence(attribution);
  const Implementation::RuntimeIdentityEvidence runtime =
      implementation_->observe_runtime_identity();
  {
    std::scoped_lock lock(implementation_->mutex);
    Implementation::SessionRecord &record = implementation_->revalidate_session(
        snapshot, Implementation::MutationKind::release);
    implementation_->require_exact_peer(snapshot, peer);
    if (cleanup &&
        (!cleanup->cleanup_authorized ||
         cleanup->cleanup_allocation_id != request.allocation_id ||
         cleanup->cleanup_grant_digest != request.grant_digest ||
         cleanup->cleanup_release_request_digest !=
             request.canonical_request_digest)) {
      throw HostdUnauthorized(
          "release-only session does not match exact durable cleanup evidence");
    }
    implementation_->apply_runtime_identity(runtime, true);
    ++record.active_mutations;
  }

  BundleReleaseResult result;
  try {
    result = implementation_->ledger->release_bundle(request, now);
  } catch (...) {
    std::scoped_lock lock(implementation_->mutex);
    auto &record = implementation_->exact_session(snapshot);
    implementation_->finish_mutation(record, false);
    throw;
  }
  std::scoped_lock lock(implementation_->mutex);
  Implementation::SessionRecord &record =
      implementation_->exact_session(snapshot);
  implementation_->finish_mutation(record, false);
  if (!implementation_->valid_release_result(result, request))
    implementation_->poison_and_throw(
        "host ledger returned an unsealed or inexact release receipt");
  const bool cleanup_while_blocked =
      snapshot.session.effective_access == HostdSessionAccess::release_only &&
      implementation_->lifecycle == HostdLifecycle::startup_blocked;
  if (implementation_->lifecycle != HostdLifecycle::admitting &&
      !cleanup_while_blocked)
    throw HostdStateError(implementation_->poison_reason);
  return result;
}

HostdCoordinatorStatus HostGrantCoordinator::status() const {
  const Implementation::RuntimeIdentityEvidence runtime =
      implementation_->observe_runtime_identity();
  std::scoped_lock lock(implementation_->mutex);
  implementation_->apply_runtime_identity(runtime, false);
  return implementation_->status_value();
}

HostdCoordinatorStatus
HostGrantCoordinator::status(std::string_view session_id) const {
  Implementation::SessionSnapshot snapshot;
  {
    std::scoped_lock lock(implementation_->mutex);
    snapshot = implementation_->snapshot_session(
        session_id, Implementation::MutationKind::none);
  }
  const HostdPeerEvidence peer =
      Implementation::observe_peer(snapshot.peer_source);
  const Implementation::RuntimeIdentityEvidence runtime =
      implementation_->observe_runtime_identity();
  std::scoped_lock lock(implementation_->mutex);
  (void)implementation_->revalidate_session(
      snapshot, Implementation::MutationKind::none);
  implementation_->require_exact_peer(snapshot, peer);
  implementation_->apply_runtime_identity(runtime, false);
  return implementation_->status_value();
}

} // namespace trainvm
