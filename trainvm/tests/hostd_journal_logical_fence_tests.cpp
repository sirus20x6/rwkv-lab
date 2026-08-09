#include "trainvm/hostd_journal_logical_fence.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace trainvm;

constexpr std::string_view kBoot =
    "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa";

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable&& callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

HostdSessionAttribution attribution() {
  return {.journal_id = "journal-logical",
          .run_id = "run-logical",
          .concurrency_key = "gpu:logical",
          .logical_lease_id = "lease-logical",
          .logical_fencing_token = 7U};
}

JournalLogicalFenceSnapshot durable_snapshot() {
  const auto scope = attribution();
  JournalLogicalFenceSnapshot result;
  result.authority.journal_id = scope.journal_id;
  result.authority.host = {.host_id = "sha256:" + std::string(64U, 'a'),
                           .boot_id = std::string(kBoot)};
  result.lease = {.concurrency_key = scope.concurrency_key,
                  .owner_run_id = scope.run_id,
                  .lease_id = scope.logical_lease_id,
                  .fencing_token = scope.logical_fencing_token,
                  .clock_domain = ResourceLease::kBootTimeDomain,
                  .boot_id = std::string(kBoot),
                  .acquired_boottime_ns = 100U,
                  .expires_boottime_ns = 10'000U,
                  .acquired_wall_time_ns = 200U,
                  .expires_wall_time_ns = 20'000U};
  result.authority_revision = 0U;
  result.authority_event_sequence = 9U;
  // Bare hex, no algorithm prefix — the form the journal writes and
  // enforces on its own chain head via journal_event_hash_valid. This
  // fixture previously carried the namespaced "sha256:..." form, agreeing
  // with a verifier that demanded it and with nothing that produced it, so
  // both sides passed while every real journal hash was rejected.
  result.authority_event_hash = std::string(64U, 'b');
  return result;
}

class FakeBoundary final : public IHostdJournalLogicalFenceBoundary {
 public:
  explicit FakeBoundary(JournalLogicalFenceSnapshot value)
      : value_(std::move(value)) {}

  JournalLogicalFenceSnapshot snapshot(
      const HostdSessionAttribution& observed,
      const AuthorityTimeSample& now) override {
    ++calls;
    last_attribution = observed;
    last_time = now;
    if (fail) throw std::runtime_error("journal unavailable");
    return value_;
  }

  bool fail{};
  std::size_t calls{};
  HostdSessionAttribution last_attribution;
  AuthorityTimeSample last_time;

 private:
  JournalLogicalFenceSnapshot value_;
};

AuthorityClock clock() {
  return AuthorityClock([] {
    return AuthorityTimeSample{.wall = {.nanoseconds = 2000},
                               .boot = {.nanoseconds = 1000},
                               .boot_id = std::string(kBoot)};
  });
}

void exact_live_snapshot_produces_bounded_evidence() {
  auto boundary = std::make_shared<FakeBoundary>(durable_snapshot());
  AuthorityClock authority_clock = clock();
  JournalHostdLogicalFenceEvidenceSource source(boundary, authority_clock);
  const auto scope = attribution();
  const auto evidence = source.attest(scope);
  require(evidence.api_version == kHostdLogicalFenceEvidenceApiVersion &&
              evidence.attribution == scope && evidence.live &&
              !evidence.cleanup_authorized &&
              evidence.cleanup_allocation_id.empty() &&
              evidence.cleanup_grant_digest.empty() &&
              evidence.cleanup_release_request_digest.empty() &&
              evidence.evidence_digest.starts_with("sha256:") &&
              evidence.evidence_digest.size() == 71U &&
              boundary->calls == 1U &&
              boundary->last_attribution == scope &&
              boundary->last_time.boot.nanoseconds == 1000,
          "live journal authority seals exact grant-time fence evidence");
}

void journal_or_lease_identity_drift_is_rejected() {
  auto drifted = durable_snapshot();
  drifted.authority.journal_id = "other-journal";
  auto boundary = std::make_shared<FakeBoundary>(std::move(drifted));
  AuthorityClock authority_clock = clock();
  JournalHostdLogicalFenceEvidenceSource source(boundary, authority_clock);
  require_throws<HostdUnauthorized>([&] { (void)source.attest(attribution()); },
                                    "journal drift must fail closed");

  auto expired = durable_snapshot();
  expired.lease.expires_boottime_ns = 1000;
  auto expired_boundary =
      std::make_shared<FakeBoundary>(std::move(expired));
  AuthorityClock second_clock = clock();
  JournalHostdLogicalFenceEvidenceSource expired_source(expired_boundary,
                                                        second_clock);
  require_throws<HostdUnauthorized>(
      [&] { (void)expired_source.attest(attribution()); },
      "an expired logical fence can never authorize a new host grant");
}

void boundary_failure_propagates_without_fabricated_evidence() {
  auto boundary = std::make_shared<FakeBoundary>(durable_snapshot());
  boundary->fail = true;
  AuthorityClock authority_clock = clock();
  JournalHostdLogicalFenceEvidenceSource source(boundary, authority_clock);
  require_throws<std::runtime_error>(
      [&] { (void)source.attest(attribution()); },
      "journal read failure cannot become a live fence");
  require_throws<std::invalid_argument>(
      [&] {
        JournalHostdLogicalFenceEvidenceSource invalid(
            std::shared_ptr<IHostdJournalLogicalFenceBoundary>{},
            authority_clock);
      },
      "a missing retained journal boundary is rejected at construction");
}

}  // namespace

// The fixture must carry a hash the journal would actually write. Asserting it
// against the journal's own predicate is what keeps this test honest: the
// previous fixture and the verifier agreed on a namespaced digest that the
// journal never produces, so both passed while every real hash was rejected.
void fixture_hash_is_what_the_journal_writes() {
  const auto snapshot = durable_snapshot();
  if (!journal_event_hash_valid(snapshot.authority_event_hash))
    throw std::runtime_error(
        "fixture authority_event_hash is not in the form the journal writes: " +
        snapshot.authority_event_hash);
}

int main() {
  try {
    fixture_hash_is_what_the_journal_writes();
    exact_live_snapshot_produces_bounded_evidence();
    journal_or_lease_identity_drift_is_rejected();
    boundary_failure_propagates_without_fabricated_evidence();
    std::cout << "hostd journal logical-fence tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "hostd journal logical-fence test failure: " << error.what()
              << '\n';
    return 1;
  }
}
