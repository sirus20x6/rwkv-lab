#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include <sys/types.h>

namespace trainvm {

inline constexpr std::string_view kHostdSessionChallengeApiVersion =
    "trainvm.hostd-session-challenge/v1";
inline constexpr std::string_view kHostdJournalFenceQueryApiVersion =
    "trainvm.hostd-journal-fence-query/v1";
inline constexpr std::string_view kHostdJournalFenceEvidenceApiVersion =
    "trainvm.hostd-journal-fence-evidence/v1";
inline constexpr std::string_view kHostdSessionChallengeEvidenceApiVersion =
    "trainvm.hostd-session-challenge-evidence/v1";
inline constexpr std::string_view kHostdSessionChallengeDigestDomain =
    "trainvm.hostd-session-challenge-digest/v1";
inline constexpr std::string_view kHostdSessionChallengeResponseDigestDomain =
    "trainvm.hostd-session-challenge-response-digest/v1";
inline constexpr std::string_view kHostdJournalFenceEvidenceDigestDomain =
    "trainvm.hostd-journal-fence-evidence-digest/v1";
inline constexpr std::string_view kHostdSessionChallengeEvidenceDigestDomain =
    "trainvm.hostd-session-challenge-evidence-digest/v1";

class HostdSessionChallengeError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class HostdSessionChallengeRejected final : public HostdSessionChallengeError {
public:
  using HostdSessionChallengeError::HostdSessionChallengeError;
};

struct HostdSocketPeerInstance final {
  uid_t uid{};
  gid_t gid{};
  pid_t pid{};
  std::uint64_t process_starttime_ticks{};

  bool operator==(const HostdSocketPeerInstance &) const = default;
};

// Exact identity of the already-authority-locked journal namespace. A future
// Linux attestor obtains this from pinned descriptors; no pathname alone is
// authority.
struct HostdJournalAuthorityIdentity final {
  std::string directory_path;
  std::string journal_name;
  std::string authority_name;
  std::string journal_id;
  std::uint64_t directory_device{};
  std::uint64_t directory_inode{};
  std::uint64_t journal_device{};
  std::uint64_t journal_inode{};
  std::uint64_t authority_device{};
  std::uint64_t authority_inode{};
  std::uint64_t owner_uid{};

  bool operator==(const HostdJournalAuthorityIdentity &) const = default;
};

// Durable logical identity observed from the journal controller projection.
// Both generation and fencing token are monotonic, nonzero fences.
struct HostdJournalControllerFence final {
  std::string run_id;
  std::string concurrency_key;
  std::string controller_id;
  std::uint64_t controller_generation{};
  std::string logical_lease_id;
  std::uint64_t logical_fencing_token{};

  bool operator==(const HostdJournalControllerFence &) const = default;
};

struct HostdSessionChallengeClaim final {
  HostdJournalAuthorityIdentity journal;
  HostdJournalControllerFence controller;

  bool operator==(const HostdSessionChallengeClaim &) const = default;
};

struct HostdSessionChallengeTime final {
  std::string boot_id;
  std::int64_t boottime_ns{};

  bool operator==(const HostdSessionChallengeTime &) const = default;
};

struct HostdSessionChallenge final {
  std::string api_version;
  std::string challenge_id;
  std::string session_nonce;
  HostdSocketPeerInstance peer;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  HostdSessionChallengeClaim claim;
  std::int64_t issued_boottime_ns{};
  std::int64_t expires_boottime_ns{};
  std::string challenge_digest;

  bool operator==(const HostdSessionChallenge &) const = default;
};

// The response deliberately echoes every authority-bearing field. The socket
// peer is re-observed independently and the journal is re-attested; the nonce
// is freshness/correlation evidence, not a bearer capability.
struct HostdSessionChallengeResponse final {
  std::string api_version;
  std::string challenge_id;
  std::string session_nonce;
  std::string challenge_digest;
  HostdSocketPeerInstance peer;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  HostdSessionChallengeClaim claim;
  std::int64_t issued_boottime_ns{};
  std::int64_t expires_boottime_ns{};
  std::string response_digest;

  bool operator==(const HostdSessionChallengeResponse &) const = default;
};

struct HostdJournalFenceQuery final {
  std::string api_version;
  std::string challenge_id;
  std::string session_nonce;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  HostdSessionChallengeClaim claim;
  std::int64_t issued_boottime_ns{};
  std::int64_t expires_boottime_ns{};

  bool operator==(const HostdJournalFenceQuery &) const = default;
};

struct HostdJournalFenceEvidence final {
  std::string api_version;
  std::string challenge_id;
  std::string session_nonce;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t observed_boottime_ns{};
  HostdJournalAuthorityIdentity journal;
  HostdJournalControllerFence controller;
  bool live{};
  std::string evidence_digest;

  bool operator==(const HostdJournalFenceEvidence &) const = default;
};

// Observational proof only. This type contains no access level, grant method,
// mutation token, or launch authority. Only the object returned directly by an
// in-process verifier invocation may be considered by a future authority
// adapter. A decoded/deserialized instance is untrusted data: its unkeyed
// digest detects accidental/tampered bytes but does not authenticate its
// origin. It is never itself a grant/release/launch capability.
struct HostdSessionChallengeEvidence final {
  std::string api_version;
  std::string challenge_id;
  std::string session_nonce;
  std::string challenge_digest;
  std::string response_digest;
  std::string journal_evidence_digest;
  HostdSocketPeerInstance peer;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  HostdSessionChallengeClaim claim;
  std::int64_t issued_boottime_ns{};
  std::int64_t expires_boottime_ns{};
  std::int64_t verified_boottime_ns{};
  std::string evidence_digest;

  bool operator==(const HostdSessionChallengeEvidence &) const = default;
};

class IHostdSessionChallengeNonceSource {
public:
  virtual ~IHostdSessionChallengeNonceSource() = default;
  // Returns exactly 64 lowercase hexadecimal characters. A production source
  // must provide 256 CSPRNG bits and purpose separation. This slice
  // deliberately provides no production implementation and must not be enabled
  // with a test or deterministic source. Collaborators must not re-enter
  // issue()/verify().
  [[nodiscard]] virtual std::string next_hex_256(std::string_view purpose) = 0;
};

class IHostdSessionChallengeTimeSource {
public:
  virtual ~IHostdSessionChallengeTimeSource() = default;
  // Production must read CLOCK_BOOTTIME and the current boot identity. This
  // slice deliberately provides no production implementation. Collaborators
  // must not re-enter issue()/verify().
  [[nodiscard]] virtual HostdSessionChallengeTime now() = 0;
};

class IHostdJournalFenceAttestor {
public:
  virtual ~IHostdJournalFenceAttestor() = default;
  // Production must pin/reinspect the journal authority and read a consistent,
  // read-only controller projection. This slice deliberately provides no such
  // production implementation. It must not advance a fence, mutate the journal,
  // or re-enter issue()/verify() while answering this query.
  [[nodiscard]] virtual HostdJournalFenceEvidence
  attest(const HostdJournalFenceQuery &query) = 0;
};

struct HostdSessionChallengeVerifierConfig final {
  std::string api_version;
  std::string host_id;
  std::string boot_id;
  std::string broker_epoch;
  std::int64_t challenge_ttl_ns{5'000'000'000LL};
  std::size_t maximum_outstanding_challenges{256U};
  std::size_t maximum_outstanding_challenges_per_peer{8U};

  bool operator==(const HostdSessionChallengeVerifierConfig &) const = default;
};

class HostdSessionChallengeVerifier final {
public:
  HostdSessionChallengeVerifier(
      HostdSessionChallengeVerifierConfig config,
      std::shared_ptr<IHostdSessionChallengeNonceSource> nonce_source,
      std::shared_ptr<IHostdSessionChallengeTimeSource> time_source,
      std::shared_ptr<IHostdJournalFenceAttestor> journal_attestor);
  ~HostdSessionChallengeVerifier();

  HostdSessionChallengeVerifier(const HostdSessionChallengeVerifier &) = delete;
  HostdSessionChallengeVerifier &
  operator=(const HostdSessionChallengeVerifier &) = delete;
  HostdSessionChallengeVerifier(HostdSessionChallengeVerifier &&) = delete;
  HostdSessionChallengeVerifier &
  operator=(HostdSessionChallengeVerifier &&) = delete;

  [[nodiscard]] HostdSessionChallenge
  issue(const HostdSocketPeerInstance &peer,
        const HostdSessionChallengeClaim &claim);

  // Consumes a known challenge before time/journal callbacks. Consequently a
  // matching-ID failed attempt, expiry, callback failure, or success is always
  // single-use. Unknown IDs do not consume unrelated challenges. The caller
  // must independently obtain SO_PEERCRED and a pinned /proc PID starttime;
  // this slice intentionally supplies neither production observation.
  [[nodiscard]] HostdSessionChallengeEvidence
  verify(const HostdSessionChallengeResponse &response,
         const HostdSocketPeerInstance &observed_peer);

  [[nodiscard]] std::size_t outstanding_challenges() const;

private:
  struct Implementation;
  std::unique_ptr<Implementation> implementation_;
};

[[nodiscard]] HostdSessionChallengeResponse
hostd_session_challenge_response(const HostdSessionChallenge &challenge);
[[nodiscard]] HostdJournalFenceEvidence
hostd_seal_journal_fence_evidence(HostdJournalFenceEvidence evidence);

[[nodiscard]] std::string
hostd_session_challenge_canonical_json(const HostdSessionChallenge &value);
[[nodiscard]] HostdSessionChallenge
hostd_session_challenge_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_session_challenge_response_canonical_json(
    const HostdSessionChallengeResponse &value);
[[nodiscard]] HostdSessionChallengeResponse
hostd_session_challenge_response_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_journal_fence_evidence_canonical_json(
    const HostdJournalFenceEvidence &value);
[[nodiscard]] HostdJournalFenceEvidence
hostd_journal_fence_evidence_from_canonical_json(std::string_view value);
[[nodiscard]] std::string hostd_session_challenge_evidence_canonical_json(
    const HostdSessionChallengeEvidence &value);
[[nodiscard]] HostdSessionChallengeEvidence
hostd_session_challenge_evidence_from_canonical_json(std::string_view value);

// Canonical evidence decoding is for storage, diagnostics, and transport only.
// A deserialized HostdSessionChallengeEvidence has no authority; only the
// direct return of HostdSessionChallengeVerifier::verify() may cross a future
// in-process authority-admission boundary.

} // namespace trainvm
