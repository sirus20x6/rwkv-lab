#include "trainvm/hostd_session_challenge.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

using namespace trainvm;

class CallbackConcurrencyProbe final {
public:
  class Guard final {
  public:
    explicit Guard(CallbackConcurrencyProbe &probe) : probe_(probe) {
      const int active = probe_.active_.fetch_add(1) + 1;
      int observed = probe_.maximum_.load();
      while (active > observed &&
             !probe_.maximum_.compare_exchange_weak(observed, active)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ~Guard() { (void)probe_.active_.fetch_sub(1); }

    Guard(const Guard &) = delete;
    Guard &operator=(const Guard &) = delete;

  private:
    CallbackConcurrencyProbe &probe_;
  };

  [[nodiscard]] int maximum() const { return maximum_.load(); }

private:
  std::atomic<int> active_{};
  std::atomic<int> maximum_{};
};

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <typename Exception, typename Callable>
void require_throws(Callable &&callable, std::string_view message) {
  try {
    std::forward<Callable>(callable)();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(std::string(message));
}

class DeterministicNonceSource final
    : public IHostdSessionChallengeNonceSource {
public:
  std::string next_hex_256(std::string_view purpose) override {
    std::optional<CallbackConcurrencyProbe::Guard> guard;
    if (probe != nullptr)
      guard.emplace(*probe);
    purposes.emplace_back(purpose);
    if (throw_next) {
      throw_next = false;
      throw std::runtime_error("injected nonce failure");
    }
    if (malformed_next) {
      malformed_next = false;
      return "not-entropy";
    }
    constexpr std::string_view digits = "0123456789abcdef";
    ++counter;
    return std::string(64U, digits.at(counter % digits.size()));
  }

  std::size_t counter{};
  bool throw_next{};
  bool malformed_next{};
  CallbackConcurrencyProbe *probe{};
  std::vector<std::string> purposes;
};

class FakeTimeSource final : public IHostdSessionChallengeTimeSource {
public:
  HostdSessionChallengeTime now() override {
    std::optional<CallbackConcurrencyProbe::Guard> guard;
    if (probe != nullptr)
      guard.emplace(*probe);
    if (fail)
      throw std::runtime_error("injected time failure");
    return value;
  }

  HostdSessionChallengeTime value{.boot_id = "boot-001",
                                  .boottime_ns = 1'000'000'000LL};
  bool fail{};
  CallbackConcurrencyProbe *probe{};
};

HostdSocketPeerInstance peer() {
  return {.uid = static_cast<uid_t>(1001U),
          .gid = static_cast<gid_t>(1002U),
          .pid = static_cast<pid_t>(4242),
          .process_starttime_ticks = 9001U};
}

HostdSessionChallengeClaim claim() {
  return {.journal = {.directory_path = "/var/lib/trainvm/journals",
                      .journal_name = "experiment.db",
                      .authority_name = "experiment.lock",
                      .journal_id = "journal-001",
                      .directory_device = 11U,
                      .directory_inode = 12U,
                      .journal_device = 11U,
                      .journal_inode = 13U,
                      .authority_device = 11U,
                      .authority_inode = 14U,
                      .owner_uid = 1001U},
          .controller = {.run_id = "run-001",
                         .controller_id = "controller-001",
                         .controller_generation = 7U,
                         .logical_lease_id = "lease-001",
                         .logical_fencing_token = 19U}};
}

class FakeJournalAttestor final : public IHostdJournalFenceAttestor {
public:
  HostdJournalFenceEvidence
  attest(const HostdJournalFenceQuery &query) override {
    std::optional<CallbackConcurrencyProbe::Guard> guard;
    if (probe != nullptr)
      guard.emplace(*probe);
    ++calls;
    last_query = query;
    if (fail)
      throw std::runtime_error("injected journal failure");
    HostdJournalFenceEvidence evidence{
        .api_version = std::string(kHostdJournalFenceEvidenceApiVersion),
        .challenge_id = query.challenge_id,
        .session_nonce = query.session_nonce,
        .host_id = host_id_override.value_or(query.host_id),
        .boot_id = query.boot_id,
        .broker_epoch = broker_epoch_override.value_or(query.broker_epoch),
        .observed_boottime_ns = observed_boottime_ns,
        .journal = authoritative_claim.journal,
        .controller = authoritative_claim.controller,
        .live = live,
        .evidence_digest = {}};
    evidence = hostd_seal_journal_fence_evidence(std::move(evidence));
    if (tamper_digest)
      evidence.evidence_digest.back() =
          evidence.evidence_digest.back() == '0' ? '1' : '0';
    if (time_source != nullptr && return_boottime_ns)
      time_source->value.boottime_ns = *return_boottime_ns;
    return evidence;
  }

  HostdSessionChallengeClaim authoritative_claim{claim()};
  std::int64_t observed_boottime_ns{1'100'000'000LL};
  bool live{true};
  bool fail{};
  bool tamper_digest{};
  CallbackConcurrencyProbe *probe{};
  FakeTimeSource *time_source{};
  std::optional<std::int64_t> return_boottime_ns;
  std::optional<std::string> host_id_override;
  std::optional<std::string> broker_epoch_override;
  std::size_t calls{};
  HostdJournalFenceQuery last_query;
};

class ReentrantNonceSource final : public IHostdSessionChallengeNonceSource {
public:
  std::string next_hex_256(std::string_view) override {
    attempted = true;
    if (verifier == nullptr)
      throw std::runtime_error("reentrant verifier was not installed");
    (void)verifier->issue(peer(), claim());
    return std::string(64U, 'a');
  }

  HostdSessionChallengeVerifier *verifier{};
  bool attempted{};
};

class ReentrantJournalAttestor final : public IHostdJournalFenceAttestor {
public:
  HostdJournalFenceEvidence
  attest(const HostdJournalFenceQuery &query) override {
    attempted = true;
    if (verifier == nullptr)
      throw std::runtime_error("reentrant verifier was not installed");
    (void)verifier->issue(peer(), query.claim);
    throw std::runtime_error("reentrant issue unexpectedly returned");
  }

  HostdSessionChallengeVerifier *verifier{};
  bool attempted{};
};

struct Fixture final {
  Fixture()
      : nonce(std::make_shared<DeterministicNonceSource>()),
        time(std::make_shared<FakeTimeSource>()),
        journal(std::make_shared<FakeJournalAttestor>()),
        verifier(std::make_unique<HostdSessionChallengeVerifier>(
            HostdSessionChallengeVerifierConfig{
                .api_version = std::string(kHostdSessionChallengeApiVersion),
                .host_id = "host-001",
                .boot_id = "boot-001",
                .broker_epoch = "broker-001",
                .challenge_ttl_ns = 2'000'000'000LL,
                .maximum_outstanding_challenges = 8U},
            nonce, time, journal)) {}

  HostdSessionChallenge issue() { return verifier->issue(peer(), claim()); }

  std::shared_ptr<DeterministicNonceSource> nonce;
  std::shared_ptr<FakeTimeSource> time;
  std::shared_ptr<FakeJournalAttestor> journal;
  std::unique_ptr<HostdSessionChallengeVerifier> verifier;
};

void canonical_success_binds_every_authority_field() {
  Fixture fixture;
  const HostdSessionChallenge challenge = fixture.issue();
  require(challenge.peer == peer() && challenge.claim == claim() &&
              challenge.host_id == "host-001" &&
              challenge.boot_id == "boot-001" &&
              challenge.broker_epoch == "broker-001" &&
              challenge.issued_boottime_ns == 1'000'000'000LL &&
              challenge.expires_boottime_ns == 3'000'000'000LL,
          "issued challenge binds peer, epoch, journal, fence, and lifetime");
  require(fixture.nonce->purposes ==
              std::vector<std::string>{"challenge_id", "session_nonce"},
          "nonce domains are explicitly separated");

  const std::string challenge_bytes =
      hostd_session_challenge_canonical_json(challenge);
  require(hostd_session_challenge_from_canonical_json(challenge_bytes) ==
              challenge,
          "challenge canonical codec round trips exactly");

  const HostdSessionChallengeResponse response =
      hostd_session_challenge_response(challenge);
  const std::string response_bytes =
      hostd_session_challenge_response_canonical_json(response);
  require(hostd_session_challenge_response_from_canonical_json(
              response_bytes) == response,
          "response canonical codec round trips exactly");
  require(kHostdSessionChallengeDigestDomain !=
                  kHostdSessionChallengeResponseDigestDomain &&
              kHostdSessionChallengeDigestDomain !=
                  kHostdJournalFenceEvidenceDigestDomain &&
              kHostdSessionChallengeDigestDomain !=
                  kHostdSessionChallengeEvidenceDigestDomain &&
              kHostdSessionChallengeResponseDigestDomain !=
                  kHostdJournalFenceEvidenceDigestDomain &&
              kHostdSessionChallengeResponseDigestDomain !=
                  kHostdSessionChallengeEvidenceDigestDomain &&
              kHostdJournalFenceEvidenceDigestDomain !=
                  kHostdSessionChallengeEvidenceDigestDomain,
          "every sealed structure has a distinct public digest domain");

  fixture.time->value.boottime_ns = 1'200'000'000LL;
  fixture.journal->observed_boottime_ns = 1'150'000'000LL;
  const HostdSessionChallengeEvidence evidence =
      fixture.verifier->verify(response, peer());
  require(evidence.peer == peer() && evidence.claim == claim() &&
              evidence.challenge_digest == challenge.challenge_digest &&
              evidence.response_digest == response.response_digest &&
              evidence.verified_boottime_ns == 1'200'000'000LL &&
              fixture.journal->last_query.claim == claim() &&
              fixture.journal->last_query.challenge_id ==
                  challenge.challenge_id,
          "verification reattests the exact journal and controller fence");
  const std::string evidence_bytes =
      hostd_session_challenge_evidence_canonical_json(evidence);
  require(hostd_session_challenge_evidence_from_canonical_json(
              evidence_bytes) == evidence,
          "observational evidence canonical codec round trips exactly");
  require(fixture.verifier->outstanding_challenges() == 0U,
          "successful challenge is consumed");
}

void replay_and_tampering_are_single_use() {
  Fixture fixture;
  const auto challenge = fixture.issue();
  const auto response = hostd_session_challenge_response(challenge);
  fixture.time->value.boottime_ns = 1'200'000'000LL;
  fixture.journal->observed_boottime_ns = 1'100'000'000LL;
  (void)fixture.verifier->verify(response, peer());
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)fixture.verifier->verify(response, peer()); },
      "successful response cannot be replayed");

  const auto tamper_challenge = fixture.issue();
  auto tampered = hostd_session_challenge_response(tamper_challenge);
  tampered.broker_epoch = "broker-tampered";
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)fixture.verifier->verify(tampered, peer()); },
      "tampered response fails closed");
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)fixture.verifier->verify(
            hostd_session_challenge_response(tamper_challenge), peer());
      },
      "tampered matching-ID attempt consumes the challenge");

  nlohmann::json altered = nlohmann::json::parse(
      hostd_session_challenge_canonical_json(fixture.issue()));
  altered["peer"]["process_starttime_ticks"] = 1U;
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)hostd_session_challenge_from_canonical_json(altered.dump());
      },
      "canonical field tampering invalidates the digest");
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)hostd_session_challenge_from_canonical_json(" " + altered.dump());
      },
      "noncanonical JSON is rejected");
  nlohmann::json wrong_type =
      nlohmann::json::parse(hostd_session_challenge_response_canonical_json(
          hostd_session_challenge_response(fixture.issue())));
  wrong_type["peer"] = "not-an-object";
  require_throws<HostdSessionChallengeRejected>(
      [&] {
        (void)hostd_session_challenge_response_from_canonical_json(
            wrong_type.dump());
      },
      "JSON type exceptions are normalized to challenge rejection");
}

void expiry_boot_change_and_pid_reuse_fail_closed() {
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.time->value.boottime_ns = challenge.expires_boottime_ns;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "expiry is exclusive and fails closed");
    require(fixture.journal->calls == 0U &&
                fixture.verifier->outstanding_challenges() == 0U,
            "expired challenge is consumed before journal access");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.time->value.boot_id = "boot-002";
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "boot change invalidates an outstanding challenge");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    HostdSocketPeerInstance reused = peer();
    ++reused.process_starttime_ticks;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), reused);
        },
        "same PID with a new starttime is rejected as PID reuse");
    require(fixture.journal->calls == 0U,
            "PID reuse is rejected before journal access");
  }
  for (std::size_t field = 0U; field < 3U; ++field) {
    Fixture fixture;
    const auto challenge = fixture.issue();
    HostdSocketPeerInstance changed = peer();
    if (field == 0U)
      ++changed.uid;
    else if (field == 1U)
      ++changed.gid;
    else
      ++changed.pid;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), changed);
        },
        "UID, GID, and PID are each exact socket-peer bindings");
  }
}

void attestation_callback_time_window_is_exact() {
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.time->value.boottime_ns = 1'200'000'000LL;
    fixture.journal->observed_boottime_ns = 2'900'000'000LL;
    fixture.journal->time_source = fixture.time.get();
    fixture.journal->return_boottime_ns = challenge.expires_boottime_ns;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "attestation callback crossing expiry fails closed");
    require(fixture.journal->calls == 1U &&
                fixture.verifier->outstanding_challenges() == 0U,
            "slow attestation consumes the expired challenge exactly once");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.time->value.boottime_ns = 1'200'000'000LL;
    fixture.journal->observed_boottime_ns = 1'300'000'000LL;
    fixture.journal->time_source = fixture.time.get();
    fixture.journal->return_boottime_ns = 1'400'000'000LL;
    const HostdSessionChallengeEvidence evidence = fixture.verifier->verify(
        hostd_session_challenge_response(challenge), peer());
    require(evidence.verified_boottime_ns == 1'400'000'000LL,
            "post-attestation boottime is the verified evidence time");
  }
}

void journal_fence_and_identity_must_remain_exact() {
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    ++fixture.journal->authoritative_claim.controller.controller_generation;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "controller generation change invalidates challenge");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    ++fixture.journal->authoritative_claim.controller.logical_fencing_token;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "journal fencing-token change invalidates challenge");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    ++fixture.journal->authoritative_claim.journal.journal_inode;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "wrong journal inode fails closed");
  }
  {
    Fixture fixture;
    HostdSessionChallengeClaim wrong = claim();
    wrong.journal.journal_id = "journal-other";
    const auto challenge = fixture.verifier->issue(peer(), wrong);
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "challenge naming a different journal fails independent attestation");
  }
  for (std::size_t field = 0U; field < 2U; ++field) {
    Fixture fixture;
    const auto challenge = fixture.issue();
    if (field == 0U)
      fixture.journal->host_id_override = "host-other";
    else
      fixture.journal->broker_epoch_override = "broker-other";
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "journal evidence must echo the exact host and broker epoch");
  }
}

void stale_failed_and_tampered_attestation_fail_closed() {
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.journal->live = false;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "non-live journal fence fails closed");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.journal->observed_boottime_ns = challenge.issued_boottime_ns - 1LL;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "journal evidence predating issuance is rejected");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.journal->tamper_digest = true;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "tampered journal evidence digest is rejected");
  }
  {
    Fixture fixture;
    const auto challenge = fixture.issue();
    fixture.journal->fail = true;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "journal attestor failure is normalized and fails closed");
    fixture.journal->fail = false;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)fixture.verifier->verify(
              hostd_session_challenge_response(challenge), peer());
        },
        "attestor failure still consumes the challenge");
  }
}

void bounds_and_injected_authorities_fail_closed() {
  {
    Fixture fixture;
    HostdSessionChallengeClaim malformed = claim();
    malformed.controller.controller_generation = 0U;
    require_throws<HostdSessionChallengeRejected>(
        [&] { (void)fixture.verifier->issue(peer(), malformed); },
        "zero controller generation is rejected");
  }
  {
    Fixture fixture;
    fixture.nonce->malformed_next = true;
    require_throws<HostdSessionChallengeRejected>(
        [&] { (void)fixture.issue(); },
        "malformed nonce source output is rejected");
  }
  {
    Fixture fixture;
    fixture.time->fail = true;
    require_throws<HostdSessionChallengeRejected>(
        [&] { (void)fixture.issue(); }, "time-source failure is normalized");
  }
  {
    auto nonce = std::make_shared<DeterministicNonceSource>();
    auto time = std::make_shared<FakeTimeSource>();
    auto journal = std::make_shared<FakeJournalAttestor>();
    HostdSessionChallengeVerifier bounded(
        {.api_version = std::string(kHostdSessionChallengeApiVersion),
         .host_id = "host-001",
         .boot_id = "boot-001",
         .broker_epoch = "broker-001",
         .challenge_ttl_ns = 2'000'000'000LL,
         .maximum_outstanding_challenges = 1U,
         .maximum_outstanding_challenges_per_peer = 1U},
        nonce, time, journal);
    (void)bounded.issue(peer(), claim());
    const std::size_t entropy_calls = nonce->counter;
    require_throws<HostdSessionChallengeRejected>(
        [&] { (void)bounded.issue(peer(), claim()); },
        "outstanding challenge capacity is bounded");
    require(nonce->counter == entropy_calls,
            "capacity rejection occurs before nonce generation");
    time->value.boottime_ns = 3'000'000'000LL;
    (void)bounded.issue(peer(), claim());
    require(bounded.outstanding_challenges() == 1U,
            "expired challenges are pruned before capacity admission");
  }
  require_throws<HostdSessionChallengeError>(
      [&] {
        auto nonce = std::make_shared<DeterministicNonceSource>();
        auto time = std::make_shared<FakeTimeSource>();
        auto journal = std::make_shared<FakeJournalAttestor>();
        HostdSessionChallengeVerifier invalid(
            {.api_version = std::string(kHostdSessionChallengeApiVersion),
             .host_id = "host-001",
             .boot_id = "boot-001",
             .broker_epoch = "broker-001",
             .challenge_ttl_ns = std::numeric_limits<std::int64_t>::max(),
             .maximum_outstanding_challenges = 1U,
             .maximum_outstanding_challenges_per_peer = 1U},
            nonce, time, journal);
        (void)invalid;
      },
      "unbounded challenge lifetime is rejected");
}

void per_peer_quota_and_boottime_high_water_fail_closed() {
  auto nonce = std::make_shared<DeterministicNonceSource>();
  auto time = std::make_shared<FakeTimeSource>();
  auto journal = std::make_shared<FakeJournalAttestor>();
  HostdSessionChallengeVerifier verifier(
      {.api_version = std::string(kHostdSessionChallengeApiVersion),
       .host_id = "host-001",
       .boot_id = "boot-001",
       .broker_epoch = "broker-001",
       .challenge_ttl_ns = 2'000'000'000LL,
       .maximum_outstanding_challenges = 4U,
       .maximum_outstanding_challenges_per_peer = 2U},
      nonce, time, journal);
  (void)verifier.issue(peer(), claim());
  (void)verifier.issue(peer(), claim());
  const std::size_t before_quota_rejection = nonce->counter;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)verifier.issue(peer(), claim()); },
      "one socket-peer process instance cannot consume global capacity");
  require(nonce->counter == before_quota_rejection,
          "per-peer quota is checked before entropy work");

  HostdSocketPeerInstance other = peer();
  ++other.pid;
  ++other.process_starttime_ticks;
  (void)verifier.issue(other, claim());
  const std::size_t before_regression = nonce->counter;
  time->value.boottime_ns = 999'999'999LL;
  require_throws<HostdSessionChallengeRejected>(
      [&] { (void)verifier.issue(other, claim()); },
      "boottime regression is rejected against a synchronized high-water mark");
  require(nonce->counter == before_regression,
          "boottime regression fails before entropy work");
}

void collaborator_reentry_is_rejected_without_deadlock() {
  {
    auto nonce = std::make_shared<ReentrantNonceSource>();
    auto time = std::make_shared<FakeTimeSource>();
    auto journal = std::make_shared<FakeJournalAttestor>();
    HostdSessionChallengeVerifier verifier(
        {.api_version = std::string(kHostdSessionChallengeApiVersion),
         .host_id = "host-001",
         .boot_id = "boot-001",
         .broker_epoch = "broker-001",
         .challenge_ttl_ns = 2'000'000'000LL,
         .maximum_outstanding_challenges = 4U,
         .maximum_outstanding_challenges_per_peer = 2U},
        nonce, time, journal);
    nonce->verifier = &verifier;
    require_throws<HostdSessionChallengeRejected>(
        [&] { (void)verifier.issue(peer(), claim()); },
        "nonce collaborator re-entry is rejected without deadlock");
    require(nonce->attempted && verifier.outstanding_challenges() == 0U,
            "reentrant nonce callback publishes no challenge");
  }
  {
    auto nonce = std::make_shared<DeterministicNonceSource>();
    auto time = std::make_shared<FakeTimeSource>();
    auto journal = std::make_shared<ReentrantJournalAttestor>();
    HostdSessionChallengeVerifier verifier(
        {.api_version = std::string(kHostdSessionChallengeApiVersion),
         .host_id = "host-001",
         .boot_id = "boot-001",
         .broker_epoch = "broker-001",
         .challenge_ttl_ns = 2'000'000'000LL,
         .maximum_outstanding_challenges = 4U,
         .maximum_outstanding_challenges_per_peer = 2U},
        nonce, time, journal);
    journal->verifier = &verifier;
    const auto challenge = verifier.issue(peer(), claim());
    time->value.boottime_ns = 1'200'000'000LL;
    require_throws<HostdSessionChallengeRejected>(
        [&] {
          (void)verifier.verify(hostd_session_challenge_response(challenge),
                                peer());
        },
        "journal collaborator re-entry is rejected without deadlock");
    require(journal->attempted && verifier.outstanding_challenges() == 0U,
            "reentrant attestation consumes but cannot validate a challenge");
  }
}

void concurrent_issue_and_verify_serialize_collaborators() {
  constexpr std::size_t count = 8U;
  CallbackConcurrencyProbe probe;
  auto nonce = std::make_shared<DeterministicNonceSource>();
  auto time = std::make_shared<FakeTimeSource>();
  auto journal = std::make_shared<FakeJournalAttestor>();
  nonce->probe = &probe;
  time->probe = &probe;
  journal->probe = &probe;
  HostdSessionChallengeVerifier verifier(
      {.api_version = std::string(kHostdSessionChallengeApiVersion),
       .host_id = "host-001",
       .boot_id = "boot-001",
       .broker_epoch = "broker-001",
       .challenge_ttl_ns = 2'000'000'000LL,
       .maximum_outstanding_challenges = 16U,
       .maximum_outstanding_challenges_per_peer = 2U},
      nonce, time, journal);

  std::vector<HostdSocketPeerInstance> peers(count, peer());
  std::vector<HostdSessionChallenge> challenges(count);
  std::vector<std::exception_ptr> errors(count);
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    peers[index].pid =
        static_cast<pid_t>(peers[index].pid + static_cast<pid_t>(index));
    peers[index].process_starttime_ticks += index;
    threads.emplace_back([&, index] {
      try {
        challenges[index] = verifier.issue(peers[index], claim());
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  threads.clear();
  require(
      std::ranges::none_of(
          errors, [](const auto &error) { return static_cast<bool>(error); }) &&
          verifier.outstanding_challenges() == count,
      "concurrent issue creates each bounded challenge exactly once");

  time->value.boottime_ns = 1'200'000'000LL;
  journal->observed_boottime_ns = 1'100'000'000LL;
  std::vector<HostdSessionChallengeEvidence> evidence(count);
  for (std::size_t index = 0U; index < count; ++index) {
    threads.emplace_back([&, index] {
      try {
        evidence[index] = verifier.verify(
            hostd_session_challenge_response(challenges[index]), peers[index]);
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  require(
      std::ranges::none_of(
          errors, [](const auto &error) { return static_cast<bool>(error); }) &&
          verifier.outstanding_challenges() == 0U &&
          std::ranges::all_of(
              evidence,
              [](const auto &value) { return !value.evidence_digest.empty(); }),
      "concurrent verify consumes and validates every challenge once");
  require(probe.maximum() == 1,
          "nonce, time, and journal collaborators are globally serialized");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"canonical-success", canonical_success_binds_every_authority_field},
      {"replay-tamper", replay_and_tampering_are_single_use},
      {"time-peer", expiry_boot_change_and_pid_reuse_fail_closed},
      {"attestation-time-window", attestation_callback_time_window_is_exact},
      {"journal-fence", journal_fence_and_identity_must_remain_exact},
      {"attestation", stale_failed_and_tampered_attestation_fail_closed},
      {"bounds", bounds_and_injected_authorities_fail_closed},
      {"quota-time", per_peer_quota_and_boottime_high_water_fail_closed},
      {"reentry", collaborator_reentry_is_rejected_without_deadlock},
      {"concurrency", concurrent_issue_and_verify_serialize_collaborators},
  };
  try {
    for (const auto &[name, test] : tests) {
      test();
      std::cout << "PASS " << name << '\n';
    }
    int status = 0;
    errno = 0;
    require(::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD,
            "challenge slice creates no child process");
    std::cout << "hostd session challenge tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "hostd session challenge test failure: " << error.what()
              << '\n';
    return 1;
  }
}
