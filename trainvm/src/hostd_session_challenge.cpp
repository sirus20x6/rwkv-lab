#include "trainvm/hostd_session_challenge.hpp"

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <ranges>
#include <utility>

#include <nlohmann/json.hpp>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumIdentifierBytes = 192U;
constexpr std::size_t kMaximumPathBytes = 1024U;
constexpr std::size_t kMaximumBasenameBytes = 192U;
constexpr std::size_t kMaximumCanonicalBytes = 64U * 1024U;
constexpr std::size_t kMaximumOutstandingChallenges = 4096U;
constexpr std::size_t kMaximumOutstandingChallengesPerPeer = 64U;
constexpr std::int64_t kMinimumChallengeTtlNs = 1'000'000LL;
constexpr std::int64_t kMaximumChallengeTtlNs = 60'000'000'000LL;
constexpr std::size_t kNonceAttempts = 8U;
constexpr std::string_view kChallengeIdPrefix = "hostd-challenge-";
constexpr std::string_view kSessionNoncePrefix = "hostd-session-nonce-";

using Json = nlohmann::json;

[[noreturn]] void reject(std::string_view message) {
  throw HostdSessionChallengeRejected(std::string(message));
}

bool valid_identifier(std::string_view value) {
  if (value.empty() || value.size() > kMaximumIdentifierBytes)
    return false;
  return std::ranges::all_of(value, [](unsigned char character) {
    const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    return alphanumeric || character == '.' || character == '_' ||
           character == '-' || character == ':' || character == '/' ||
           character == '@';
  });
}

bool valid_basename(std::string_view value) {
  return !value.empty() && value.size() <= kMaximumBasenameBytes &&
         value != "." && value != ".." &&
         std::ranges::all_of(value, [](unsigned char character) {
           const bool alphanumeric = (character >= 'a' && character <= 'z') ||
                                     (character >= 'A' && character <= 'Z') ||
                                     (character >= '0' && character <= '9');
           return alphanumeric || character == '.' || character == '_' ||
                  character == '-';
         });
}

bool valid_absolute_path(std::string_view value) {
  if (value.empty() || value.size() > kMaximumPathBytes ||
      value.front() != '/' || value.find('\0') != std::string_view::npos ||
      (value.size() > 1U && value.starts_with("//")) ||
      !std::ranges::all_of(value, [](unsigned char character) {
        return character >= 0x20U && character <= 0x7eU;
      }))
    return false;
  try {
    const std::filesystem::path path{std::string(value)};
    return path.is_absolute() && path.lexically_normal() == path;
  } catch (...) {
    return false;
  }
}

bool valid_hex_token(std::string_view value) {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_prefixed_token(std::string_view value, std::string_view prefix) {
  return value.starts_with(prefix) &&
         valid_hex_token(value.substr(prefix.size()));
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         valid_hex_token(value.substr(7U));
}

std::string sha256(std::string_view domain, std::string_view value) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> bytes{};
  std::string framed;
  framed.reserve(domain.size() + 1U + value.size());
  framed.append(domain);
  framed.push_back('\0');
  framed.append(value);
  if (::SHA256(reinterpret_cast<const unsigned char *>(framed.data()),
               framed.size(), bytes.data()) == nullptr) {
    reject("challenge digest computation failed");
  }
  constexpr std::array<char, 16U> hex{'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string result = "sha256:";
  result.reserve(7U + bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    result.push_back(hex[(byte >> 4U) & 0x0fU]);
    result.push_back(hex[byte & 0x0fU]);
  }
  return result;
}

std::string digest_json(std::string_view domain, const Json &value) {
  return sha256(domain, value.dump());
}

void require_fields(const Json &value,
                    std::initializer_list<std::string_view> fields) {
  if (!value.is_object() || value.size() != fields.size())
    reject("challenge JSON fields are inexact");
  for (const auto field : fields)
    if (!value.contains(std::string(field)))
      reject("challenge JSON field is missing");
}

Json parse_canonical(std::string_view value) {
  if (value.empty() || value.size() > kMaximumCanonicalBytes)
    reject("challenge canonical JSON size is invalid");
  try {
    Json parsed = Json::parse(value);
    if (parsed.dump() != value)
      reject("challenge JSON is not canonical");
    return parsed;
  } catch (const HostdSessionChallengeError &) {
    throw;
  } catch (const Json::exception &) {
    reject("challenge JSON is malformed");
  }
}

Json peer_json(const HostdSocketPeerInstance &peer) {
  return {{"gid", static_cast<std::uint64_t>(peer.gid)},
          {"pid", static_cast<std::int64_t>(peer.pid)},
          {"process_starttime_ticks", peer.process_starttime_ticks},
          {"uid", static_cast<std::uint64_t>(peer.uid)}};
}

HostdSocketPeerInstance parse_peer(const Json &value) {
  require_fields(value, {"gid", "pid", "process_starttime_ticks", "uid"});
  const auto uid = value.at("uid").get<std::uint64_t>();
  const auto gid = value.at("gid").get<std::uint64_t>();
  const auto pid = value.at("pid").get<std::int64_t>();
  if (uid > std::numeric_limits<uid_t>::max() ||
      gid > std::numeric_limits<gid_t>::max() || pid <= 0 ||
      static_cast<std::uint64_t>(pid) >
          static_cast<std::uint64_t>(std::numeric_limits<pid_t>::max()))
    reject("challenge peer numeric identity is invalid");
  return {.uid = static_cast<uid_t>(uid),
          .gid = static_cast<gid_t>(gid),
          .pid = static_cast<pid_t>(pid),
          .process_starttime_ticks =
              value.at("process_starttime_ticks").get<std::uint64_t>()};
}

bool valid_peer(const HostdSocketPeerInstance &peer) {
  return peer.pid > 0 && peer.process_starttime_ticks > 0U;
}

Json journal_json(const HostdJournalAuthorityIdentity &journal) {
  return {{"authority_device", journal.authority_device},
          {"authority_inode", journal.authority_inode},
          {"authority_name", journal.authority_name},
          {"directory_device", journal.directory_device},
          {"directory_inode", journal.directory_inode},
          {"directory_path", journal.directory_path},
          {"journal_device", journal.journal_device},
          {"journal_id", journal.journal_id},
          {"journal_inode", journal.journal_inode},
          {"journal_name", journal.journal_name},
          {"owner_uid", journal.owner_uid}};
}

HostdJournalAuthorityIdentity parse_journal(const Json &value) {
  require_fields(value,
                 {"authority_device", "authority_inode", "authority_name",
                  "directory_device", "directory_inode", "directory_path",
                  "journal_device", "journal_id", "journal_inode",
                  "journal_name", "owner_uid"});
  return {.directory_path = value.at("directory_path").get<std::string>(),
          .journal_name = value.at("journal_name").get<std::string>(),
          .authority_name = value.at("authority_name").get<std::string>(),
          .journal_id = value.at("journal_id").get<std::string>(),
          .directory_device = value.at("directory_device").get<std::uint64_t>(),
          .directory_inode = value.at("directory_inode").get<std::uint64_t>(),
          .journal_device = value.at("journal_device").get<std::uint64_t>(),
          .journal_inode = value.at("journal_inode").get<std::uint64_t>(),
          .authority_device = value.at("authority_device").get<std::uint64_t>(),
          .authority_inode = value.at("authority_inode").get<std::uint64_t>(),
          .owner_uid = value.at("owner_uid").get<std::uint64_t>()};
}

bool valid_journal(const HostdJournalAuthorityIdentity &journal) {
  return valid_absolute_path(journal.directory_path) &&
         valid_basename(journal.journal_name) &&
         valid_basename(journal.authority_name) &&
         valid_identifier(journal.journal_id) && journal.directory_inode > 0U &&
         journal.journal_inode > 0U && journal.authority_inode > 0U &&
         journal.owner_uid <= std::numeric_limits<uid_t>::max();
}

Json controller_json(const HostdJournalControllerFence &controller) {
  return {{"concurrency_key", controller.concurrency_key},
          {"controller_generation", controller.controller_generation},
          {"controller_id", controller.controller_id},
          {"logical_fencing_token", controller.logical_fencing_token},
          {"logical_lease_id", controller.logical_lease_id},
          {"run_id", controller.run_id}};
}

HostdJournalControllerFence parse_controller(const Json &value) {
  require_fields(value,
                 {"concurrency_key", "controller_generation", "controller_id",
                  "logical_fencing_token", "logical_lease_id", "run_id"});
  return {.run_id = value.at("run_id").get<std::string>(),
          .concurrency_key = value.at("concurrency_key").get<std::string>(),
          .controller_id = value.at("controller_id").get<std::string>(),
          .controller_generation =
              value.at("controller_generation").get<std::uint64_t>(),
          .logical_lease_id = value.at("logical_lease_id").get<std::string>(),
          .logical_fencing_token =
              value.at("logical_fencing_token").get<std::uint64_t>()};
}

bool valid_controller(const HostdJournalControllerFence &controller) {
  return valid_identifier(controller.run_id) &&
         valid_identifier(controller.concurrency_key) &&
         valid_identifier(controller.controller_id) &&
         valid_identifier(controller.logical_lease_id) &&
         controller.controller_generation > 0U &&
         controller.logical_fencing_token > 0U;
}

Json claim_json(const HostdSessionChallengeClaim &claim) {
  return {{"controller", controller_json(claim.controller)},
          {"journal", journal_json(claim.journal)}};
}

HostdSessionChallengeClaim parse_claim(const Json &value) {
  require_fields(value, {"controller", "journal"});
  return {.journal = parse_journal(value.at("journal")),
          .controller = parse_controller(value.at("controller"))};
}

bool valid_claim(const HostdSessionChallengeClaim &claim) {
  return valid_journal(claim.journal) && valid_controller(claim.controller);
}

Json challenge_json(const HostdSessionChallenge &value, bool include_digest) {
  Json result{{"api_version", value.api_version},
              {"boot_id", value.boot_id},
              {"broker_epoch", value.broker_epoch},
              {"challenge_id", value.challenge_id},
              {"claim", claim_json(value.claim)},
              {"expires_boottime_ns", value.expires_boottime_ns},
              {"host_id", value.host_id},
              {"issued_boottime_ns", value.issued_boottime_ns},
              {"peer", peer_json(value.peer)},
              {"session_nonce", value.session_nonce}};
  if (include_digest)
    result["challenge_digest"] = value.challenge_digest;
  return result;
}

bool valid_challenge(const HostdSessionChallenge &value) {
  return value.api_version == kHostdSessionChallengeApiVersion &&
         valid_prefixed_token(value.challenge_id, kChallengeIdPrefix) &&
         valid_prefixed_token(value.session_nonce, kSessionNoncePrefix) &&
         value.challenge_id.substr(kChallengeIdPrefix.size()) !=
             value.session_nonce.substr(kSessionNoncePrefix.size()) &&
         valid_peer(value.peer) && valid_identifier(value.host_id) &&
         valid_identifier(value.boot_id) &&
         valid_identifier(value.broker_epoch) && valid_claim(value.claim) &&
         value.issued_boottime_ns >= 0 &&
         value.expires_boottime_ns > value.issued_boottime_ns &&
         value.expires_boottime_ns - value.issued_boottime_ns >=
             kMinimumChallengeTtlNs &&
         value.expires_boottime_ns - value.issued_boottime_ns <=
             kMaximumChallengeTtlNs &&
         valid_digest(value.challenge_digest) &&
         value.challenge_digest ==
             digest_json(kHostdSessionChallengeDigestDomain,
                         challenge_json(value, false));
}

HostdSessionChallenge parse_challenge(const Json &value) {
  require_fields(value,
                 {"api_version", "boot_id", "broker_epoch", "challenge_digest",
                  "challenge_id", "claim", "expires_boottime_ns", "host_id",
                  "issued_boottime_ns", "peer", "session_nonce"});
  return {.api_version = value.at("api_version").get<std::string>(),
          .challenge_id = value.at("challenge_id").get<std::string>(),
          .session_nonce = value.at("session_nonce").get<std::string>(),
          .peer = parse_peer(value.at("peer")),
          .host_id = value.at("host_id").get<std::string>(),
          .boot_id = value.at("boot_id").get<std::string>(),
          .broker_epoch = value.at("broker_epoch").get<std::string>(),
          .claim = parse_claim(value.at("claim")),
          .issued_boottime_ns =
              value.at("issued_boottime_ns").get<std::int64_t>(),
          .expires_boottime_ns =
              value.at("expires_boottime_ns").get<std::int64_t>(),
          .challenge_digest = value.at("challenge_digest").get<std::string>()};
}

Json response_json(const HostdSessionChallengeResponse &value,
                   bool include_digest) {
  Json result{{"api_version", value.api_version},
              {"boot_id", value.boot_id},
              {"broker_epoch", value.broker_epoch},
              {"challenge_digest", value.challenge_digest},
              {"challenge_id", value.challenge_id},
              {"claim", claim_json(value.claim)},
              {"expires_boottime_ns", value.expires_boottime_ns},
              {"host_id", value.host_id},
              {"issued_boottime_ns", value.issued_boottime_ns},
              {"peer", peer_json(value.peer)},
              {"session_nonce", value.session_nonce}};
  if (include_digest)
    result["response_digest"] = value.response_digest;
  return result;
}

bool valid_response(const HostdSessionChallengeResponse &value) {
  return value.api_version == kHostdSessionChallengeApiVersion &&
         valid_prefixed_token(value.challenge_id, kChallengeIdPrefix) &&
         valid_prefixed_token(value.session_nonce, kSessionNoncePrefix) &&
         value.challenge_id.substr(kChallengeIdPrefix.size()) !=
             value.session_nonce.substr(kSessionNoncePrefix.size()) &&
         valid_digest(value.challenge_digest) && valid_peer(value.peer) &&
         valid_identifier(value.host_id) && valid_identifier(value.boot_id) &&
         valid_identifier(value.broker_epoch) && valid_claim(value.claim) &&
         value.issued_boottime_ns >= 0 &&
         value.expires_boottime_ns > value.issued_boottime_ns &&
         value.expires_boottime_ns - value.issued_boottime_ns >=
             kMinimumChallengeTtlNs &&
         value.expires_boottime_ns - value.issued_boottime_ns <=
             kMaximumChallengeTtlNs &&
         valid_digest(value.response_digest) &&
         value.response_digest ==
             digest_json(kHostdSessionChallengeResponseDigestDomain,
                         response_json(value, false));
}

HostdSessionChallengeResponse parse_response(const Json &value) {
  require_fields(value, {"api_version", "boot_id", "broker_epoch",
                         "challenge_digest", "challenge_id", "claim",
                         "expires_boottime_ns", "host_id", "issued_boottime_ns",
                         "peer", "response_digest", "session_nonce"});
  return {.api_version = value.at("api_version").get<std::string>(),
          .challenge_id = value.at("challenge_id").get<std::string>(),
          .session_nonce = value.at("session_nonce").get<std::string>(),
          .challenge_digest = value.at("challenge_digest").get<std::string>(),
          .peer = parse_peer(value.at("peer")),
          .host_id = value.at("host_id").get<std::string>(),
          .boot_id = value.at("boot_id").get<std::string>(),
          .broker_epoch = value.at("broker_epoch").get<std::string>(),
          .claim = parse_claim(value.at("claim")),
          .issued_boottime_ns =
              value.at("issued_boottime_ns").get<std::int64_t>(),
          .expires_boottime_ns =
              value.at("expires_boottime_ns").get<std::int64_t>(),
          .response_digest = value.at("response_digest").get<std::string>()};
}

Json journal_evidence_json(const HostdJournalFenceEvidence &value,
                           bool include_digest) {
  Json result{{"api_version", value.api_version},
              {"boot_id", value.boot_id},
              {"broker_epoch", value.broker_epoch},
              {"challenge_id", value.challenge_id},
              {"controller", controller_json(value.controller)},
              {"host_id", value.host_id},
              {"journal", journal_json(value.journal)},
              {"live", value.live},
              {"observed_boottime_ns", value.observed_boottime_ns},
              {"session_nonce", value.session_nonce}};
  if (include_digest)
    result["evidence_digest"] = value.evidence_digest;
  return result;
}

bool valid_journal_evidence(const HostdJournalFenceEvidence &value) {
  return value.api_version == kHostdJournalFenceEvidenceApiVersion &&
         valid_prefixed_token(value.challenge_id, kChallengeIdPrefix) &&
         valid_prefixed_token(value.session_nonce, kSessionNoncePrefix) &&
         value.challenge_id.substr(kChallengeIdPrefix.size()) !=
             value.session_nonce.substr(kSessionNoncePrefix.size()) &&
         valid_identifier(value.host_id) && valid_identifier(value.boot_id) &&
         valid_identifier(value.broker_epoch) &&
         value.observed_boottime_ns >= 0 && valid_journal(value.journal) &&
         valid_controller(value.controller) &&
         valid_digest(value.evidence_digest) &&
         value.evidence_digest ==
             digest_json(kHostdJournalFenceEvidenceDigestDomain,
                         journal_evidence_json(value, false));
}

HostdJournalFenceEvidence parse_journal_evidence(const Json &value) {
  require_fields(value,
                 {"api_version", "boot_id", "broker_epoch", "challenge_id",
                  "controller", "evidence_digest", "host_id", "journal", "live",
                  "observed_boottime_ns", "session_nonce"});
  return {.api_version = value.at("api_version").get<std::string>(),
          .challenge_id = value.at("challenge_id").get<std::string>(),
          .session_nonce = value.at("session_nonce").get<std::string>(),
          .host_id = value.at("host_id").get<std::string>(),
          .boot_id = value.at("boot_id").get<std::string>(),
          .broker_epoch = value.at("broker_epoch").get<std::string>(),
          .observed_boottime_ns =
              value.at("observed_boottime_ns").get<std::int64_t>(),
          .journal = parse_journal(value.at("journal")),
          .controller = parse_controller(value.at("controller")),
          .live = value.at("live").get<bool>(),
          .evidence_digest = value.at("evidence_digest").get<std::string>()};
}

Json verified_evidence_json(const HostdSessionChallengeEvidence &value,
                            bool include_digest) {
  Json result{{"api_version", value.api_version},
              {"boot_id", value.boot_id},
              {"broker_epoch", value.broker_epoch},
              {"challenge_digest", value.challenge_digest},
              {"challenge_id", value.challenge_id},
              {"claim", claim_json(value.claim)},
              {"expires_boottime_ns", value.expires_boottime_ns},
              {"host_id", value.host_id},
              {"issued_boottime_ns", value.issued_boottime_ns},
              {"journal_evidence_digest", value.journal_evidence_digest},
              {"peer", peer_json(value.peer)},
              {"response_digest", value.response_digest},
              {"session_nonce", value.session_nonce},
              {"verified_boottime_ns", value.verified_boottime_ns}};
  if (include_digest)
    result["evidence_digest"] = value.evidence_digest;
  return result;
}

bool valid_verified_evidence(const HostdSessionChallengeEvidence &value) {
  return value.api_version == kHostdSessionChallengeEvidenceApiVersion &&
         valid_prefixed_token(value.challenge_id, kChallengeIdPrefix) &&
         valid_prefixed_token(value.session_nonce, kSessionNoncePrefix) &&
         value.challenge_id.substr(kChallengeIdPrefix.size()) !=
             value.session_nonce.substr(kSessionNoncePrefix.size()) &&
         valid_digest(value.challenge_digest) &&
         valid_digest(value.response_digest) &&
         valid_digest(value.journal_evidence_digest) &&
         valid_peer(value.peer) && valid_identifier(value.host_id) &&
         valid_identifier(value.boot_id) &&
         valid_identifier(value.broker_epoch) && valid_claim(value.claim) &&
         value.issued_boottime_ns >= 0 &&
         value.verified_boottime_ns >= value.issued_boottime_ns &&
         value.expires_boottime_ns > value.verified_boottime_ns &&
         value.expires_boottime_ns - value.issued_boottime_ns >=
             kMinimumChallengeTtlNs &&
         value.expires_boottime_ns - value.issued_boottime_ns <=
             kMaximumChallengeTtlNs &&
         valid_digest(value.evidence_digest) &&
         value.evidence_digest ==
             digest_json(kHostdSessionChallengeEvidenceDigestDomain,
                         verified_evidence_json(value, false));
}

HostdSessionChallengeEvidence parse_verified_evidence(const Json &value) {
  require_fields(value,
                 {"api_version", "boot_id", "broker_epoch", "challenge_digest",
                  "challenge_id", "claim", "evidence_digest",
                  "expires_boottime_ns", "host_id", "issued_boottime_ns",
                  "journal_evidence_digest", "peer", "response_digest",
                  "session_nonce", "verified_boottime_ns"});
  return {.api_version = value.at("api_version").get<std::string>(),
          .challenge_id = value.at("challenge_id").get<std::string>(),
          .session_nonce = value.at("session_nonce").get<std::string>(),
          .challenge_digest = value.at("challenge_digest").get<std::string>(),
          .response_digest = value.at("response_digest").get<std::string>(),
          .journal_evidence_digest =
              value.at("journal_evidence_digest").get<std::string>(),
          .peer = parse_peer(value.at("peer")),
          .host_id = value.at("host_id").get<std::string>(),
          .boot_id = value.at("boot_id").get<std::string>(),
          .broker_epoch = value.at("broker_epoch").get<std::string>(),
          .claim = parse_claim(value.at("claim")),
          .issued_boottime_ns =
              value.at("issued_boottime_ns").get<std::int64_t>(),
          .expires_boottime_ns =
              value.at("expires_boottime_ns").get<std::int64_t>(),
          .verified_boottime_ns =
              value.at("verified_boottime_ns").get<std::int64_t>(),
          .evidence_digest = value.at("evidence_digest").get<std::string>()};
}

HostdJournalFenceQuery query_for(const HostdSessionChallenge &challenge) {
  return {.api_version = std::string(kHostdJournalFenceQueryApiVersion),
          .challenge_id = challenge.challenge_id,
          .session_nonce = challenge.session_nonce,
          .host_id = challenge.host_id,
          .boot_id = challenge.boot_id,
          .broker_epoch = challenge.broker_epoch,
          .claim = challenge.claim,
          .issued_boottime_ns = challenge.issued_boottime_ns,
          .expires_boottime_ns = challenge.expires_boottime_ns};
}

bool response_matches(const HostdSessionChallengeResponse &response,
                      const HostdSessionChallenge &challenge) {
  return response.challenge_id == challenge.challenge_id &&
         response.session_nonce == challenge.session_nonce &&
         response.challenge_digest == challenge.challenge_digest &&
         response.peer == challenge.peer &&
         response.host_id == challenge.host_id &&
         response.boot_id == challenge.boot_id &&
         response.broker_epoch == challenge.broker_epoch &&
         response.claim == challenge.claim &&
         response.issued_boottime_ns == challenge.issued_boottime_ns &&
         response.expires_boottime_ns == challenge.expires_boottime_ns;
}

} // namespace

struct HostdSessionChallengeVerifier::Implementation final {
  HostdSessionChallengeVerifierConfig config;
  std::shared_ptr<IHostdSessionChallengeNonceSource> nonce_source;
  std::shared_ptr<IHostdSessionChallengeTimeSource> time_source;
  std::shared_ptr<IHostdJournalFenceAttestor> journal_attestor;
  // The complete issue/verify operation, including collaborator callbacks, is
  // serialized. The recursive form permits us to detect same-thread callback
  // re-entry and reject it instead of deadlocking.
  mutable std::recursive_mutex mutex;
  bool collaborator_callback_active{};
  std::optional<std::int64_t> boottime_high_water_ns;
  std::map<std::string, HostdSessionChallenge> outstanding;

  void reject_callback_reentry() const {
    if (collaborator_callback_active)
      reject("challenge collaborator re-entry is forbidden");
  }

  HostdSessionChallengeTime observe_time() {
    if (collaborator_callback_active)
      reject("challenge time callback re-entry is forbidden");
    collaborator_callback_active = true;
    HostdSessionChallengeTime value;
    try {
      value = time_source->now();
    } catch (...) {
      collaborator_callback_active = false;
      reject("challenge time authority is unavailable");
    }
    collaborator_callback_active = false;
    if (value.boot_id != config.boot_id || value.boottime_ns < 0)
      reject("challenge time authority changed or is invalid");
    if (boottime_high_water_ns && value.boottime_ns < *boottime_high_water_ns)
      reject("challenge boottime authority regressed");
    boottime_high_water_ns = value.boottime_ns;
    return value;
  }

  std::string nonce(std::string_view purpose) {
    if (collaborator_callback_active)
      reject("challenge nonce callback re-entry is forbidden");
    collaborator_callback_active = true;
    std::string value;
    try {
      value = nonce_source->next_hex_256(purpose);
    } catch (...) {
      collaborator_callback_active = false;
      reject("challenge nonce source is unavailable");
    }
    collaborator_callback_active = false;
    if (!valid_hex_token(value))
      reject("challenge nonce source returned malformed entropy");
    return value;
  }

  HostdJournalFenceEvidence attest(const HostdJournalFenceQuery &query) {
    if (collaborator_callback_active)
      reject("journal attestor callback re-entry is forbidden");
    collaborator_callback_active = true;
    HostdJournalFenceEvidence value;
    try {
      value = journal_attestor->attest(query);
    } catch (...) {
      collaborator_callback_active = false;
      reject("journal fence attestation is unavailable");
    }
    collaborator_callback_active = false;
    return value;
  }

  void prune_expired(std::int64_t now_boottime_ns) {
    std::erase_if(outstanding, [&](const auto &entry) {
      return entry.second.expires_boottime_ns <= now_boottime_ns;
    });
  }

  std::size_t outstanding_for_peer(const HostdSocketPeerInstance &peer) const {
    return static_cast<std::size_t>(
        std::ranges::count_if(outstanding, [&](const auto &entry) {
          return entry.second.peer == peer;
        }));
  }
};

HostdSessionChallengeVerifier::HostdSessionChallengeVerifier(
    HostdSessionChallengeVerifierConfig config,
    std::shared_ptr<IHostdSessionChallengeNonceSource> nonce_source,
    std::shared_ptr<IHostdSessionChallengeTimeSource> time_source,
    std::shared_ptr<IHostdJournalFenceAttestor> journal_attestor)
    : implementation_(std::make_unique<Implementation>()) {
  if (config.api_version != kHostdSessionChallengeApiVersion ||
      !valid_identifier(config.host_id) || !valid_identifier(config.boot_id) ||
      !valid_identifier(config.broker_epoch) ||
      config.challenge_ttl_ns < kMinimumChallengeTtlNs ||
      config.challenge_ttl_ns > kMaximumChallengeTtlNs ||
      config.maximum_outstanding_challenges == 0U ||
      config.maximum_outstanding_challenges > kMaximumOutstandingChallenges ||
      config.maximum_outstanding_challenges_per_peer == 0U ||
      config.maximum_outstanding_challenges_per_peer >
          kMaximumOutstandingChallengesPerPeer ||
      config.maximum_outstanding_challenges_per_peer >
          config.maximum_outstanding_challenges ||
      !nonce_source || !time_source || !journal_attestor)
    throw HostdSessionChallengeError(
        "hostd session challenge verifier configuration is invalid");
  implementation_->config = std::move(config);
  implementation_->nonce_source = std::move(nonce_source);
  implementation_->time_source = std::move(time_source);
  implementation_->journal_attestor = std::move(journal_attestor);
}

HostdSessionChallengeVerifier::~HostdSessionChallengeVerifier() = default;

HostdSessionChallenge
HostdSessionChallengeVerifier::issue(const HostdSocketPeerInstance &peer,
                                     const HostdSessionChallengeClaim &claim) {
  std::scoped_lock lock(implementation_->mutex);
  implementation_->reject_callback_reentry();
  if (!valid_peer(peer) || !valid_claim(claim))
    reject("challenge issue request is malformed");
  const HostdSessionChallengeTime now = implementation_->observe_time();
  implementation_->prune_expired(now.boottime_ns);
  if (implementation_->outstanding.size() >=
      implementation_->config.maximum_outstanding_challenges)
    reject("hostd session challenge capacity is exhausted");
  if (implementation_->outstanding_for_peer(peer) >=
      implementation_->config.maximum_outstanding_challenges_per_peer)
    reject("hostd session challenge per-peer capacity is exhausted");
  if (implementation_->config.challenge_ttl_ns >
      std::numeric_limits<std::int64_t>::max() - now.boottime_ns)
    reject("challenge expiry overflows boottime");

  for (std::size_t attempt = 0U; attempt < kNonceAttempts; ++attempt) {
    const std::string id_token = implementation_->nonce("challenge_id");
    const std::string nonce_token = implementation_->nonce("session_nonce");
    if (id_token == nonce_token)
      continue;
    HostdSessionChallenge challenge{
        .api_version = std::string(kHostdSessionChallengeApiVersion),
        .challenge_id = std::string(kChallengeIdPrefix) + id_token,
        .session_nonce = std::string(kSessionNoncePrefix) + nonce_token,
        .peer = peer,
        .host_id = implementation_->config.host_id,
        .boot_id = implementation_->config.boot_id,
        .broker_epoch = implementation_->config.broker_epoch,
        .claim = claim,
        .issued_boottime_ns = now.boottime_ns,
        .expires_boottime_ns =
            now.boottime_ns + implementation_->config.challenge_ttl_ns,
        .challenge_digest = {}};
    challenge.challenge_digest = digest_json(kHostdSessionChallengeDigestDomain,
                                             challenge_json(challenge, false));
    if (!valid_challenge(challenge))
      reject("issued challenge failed canonical validation");

    const bool nonce_reused = std::ranges::any_of(
        implementation_->outstanding, [&](const auto &entry) {
          return entry.second.session_nonce == challenge.session_nonce;
        });
    if (nonce_reused ||
        implementation_->outstanding.contains(challenge.challenge_id))
      continue;
    implementation_->outstanding.emplace(challenge.challenge_id, challenge);
    return challenge;
  }
  throw HostdSessionChallengeError(
      "hostd session challenge entropy repeatedly collided");
}

HostdSessionChallengeEvidence HostdSessionChallengeVerifier::verify(
    const HostdSessionChallengeResponse &response,
    const HostdSocketPeerInstance &observed_peer) {
  std::scoped_lock lock(implementation_->mutex);
  implementation_->reject_callback_reentry();
  HostdSessionChallenge challenge;
  const auto found = implementation_->outstanding.find(response.challenge_id);
  if (found == implementation_->outstanding.end())
    reject("challenge is unknown, expired, or already consumed");
  challenge = found->second;
  implementation_->outstanding.erase(found);

  try {
    if (!valid_response(response) || !valid_challenge(challenge) ||
        !response_matches(response, challenge))
      reject("challenge response is malformed, tampered, or inexact");
    if (!valid_peer(observed_peer) || observed_peer != challenge.peer)
      reject("socket peer process instance changed during challenge");

    const HostdSessionChallengeTime before_attestation =
        implementation_->observe_time();
    if (before_attestation.boottime_ns < challenge.issued_boottime_ns ||
        before_attestation.boottime_ns >= challenge.expires_boottime_ns)
      reject("challenge is not live in the current boottime interval");

    const HostdJournalFenceEvidence journal =
        implementation_->attest(query_for(challenge));
    const HostdSessionChallengeTime after_attestation =
        implementation_->observe_time();
    if (!valid_journal_evidence(journal) || !journal.live ||
        journal.challenge_id != challenge.challenge_id ||
        journal.session_nonce != challenge.session_nonce ||
        journal.host_id != challenge.host_id ||
        journal.boot_id != challenge.boot_id ||
        journal.broker_epoch != challenge.broker_epoch ||
        journal.journal != challenge.claim.journal ||
        journal.controller != challenge.claim.controller ||
        journal.observed_boottime_ns < challenge.issued_boottime_ns ||
        journal.observed_boottime_ns > after_attestation.boottime_ns ||
        after_attestation.boottime_ns >= challenge.expires_boottime_ns)
      reject("journal fence evidence is stale, tampered, or inexact");

    HostdSessionChallengeEvidence evidence{
        .api_version = std::string(kHostdSessionChallengeEvidenceApiVersion),
        .challenge_id = challenge.challenge_id,
        .session_nonce = challenge.session_nonce,
        .challenge_digest = challenge.challenge_digest,
        .response_digest = response.response_digest,
        .journal_evidence_digest = journal.evidence_digest,
        .peer = challenge.peer,
        .host_id = challenge.host_id,
        .boot_id = challenge.boot_id,
        .broker_epoch = challenge.broker_epoch,
        .claim = challenge.claim,
        .issued_boottime_ns = challenge.issued_boottime_ns,
        .expires_boottime_ns = challenge.expires_boottime_ns,
        .verified_boottime_ns = after_attestation.boottime_ns,
        .evidence_digest = {}};
    evidence.evidence_digest =
        digest_json(kHostdSessionChallengeEvidenceDigestDomain,
                    verified_evidence_json(evidence, false));
    if (!valid_verified_evidence(evidence))
      reject("verified challenge evidence failed canonical validation");
    return evidence;
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("challenge verification evidence was malformed");
  }
}

std::size_t HostdSessionChallengeVerifier::outstanding_challenges() const {
  std::scoped_lock lock(implementation_->mutex);
  return implementation_->outstanding.size();
}

HostdSessionChallengeResponse
hostd_session_challenge_response(const HostdSessionChallenge &challenge) {
  if (!valid_challenge(challenge))
    reject("cannot answer a malformed challenge");
  HostdSessionChallengeResponse response{
      .api_version = std::string(kHostdSessionChallengeApiVersion),
      .challenge_id = challenge.challenge_id,
      .session_nonce = challenge.session_nonce,
      .challenge_digest = challenge.challenge_digest,
      .peer = challenge.peer,
      .host_id = challenge.host_id,
      .boot_id = challenge.boot_id,
      .broker_epoch = challenge.broker_epoch,
      .claim = challenge.claim,
      .issued_boottime_ns = challenge.issued_boottime_ns,
      .expires_boottime_ns = challenge.expires_boottime_ns,
      .response_digest = {}};
  response.response_digest =
      digest_json(kHostdSessionChallengeResponseDigestDomain,
                  response_json(response, false));
  return response;
}

HostdJournalFenceEvidence
hostd_seal_journal_fence_evidence(HostdJournalFenceEvidence evidence) {
  try {
    evidence.api_version = std::string(kHostdJournalFenceEvidenceApiVersion);
    evidence.evidence_digest =
        digest_json(kHostdJournalFenceEvidenceDigestDomain,
                    journal_evidence_json(evidence, false));
    if (!valid_journal_evidence(evidence))
      reject("cannot seal malformed journal fence evidence");
    return evidence;
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("journal fence evidence sealing failed closed");
  }
}

std::string
hostd_session_challenge_canonical_json(const HostdSessionChallenge &value) {
  if (!valid_challenge(value))
    reject("challenge is not exactly sealed");
  return challenge_json(value, true).dump();
}

HostdSessionChallenge
hostd_session_challenge_from_canonical_json(std::string_view value) {
  try {
    HostdSessionChallenge parsed = parse_challenge(parse_canonical(value));
    if (!valid_challenge(parsed))
      reject("challenge canonical semantics are invalid");
    return parsed;
  } catch (const HostdSessionChallengeError &) {
    throw;
  } catch (...) {
    reject("challenge canonical decoding failed");
  }
}

std::string hostd_session_challenge_response_canonical_json(
    const HostdSessionChallengeResponse &value) {
  if (!valid_response(value))
    reject("challenge response is not exactly sealed");
  return response_json(value, true).dump();
}

HostdSessionChallengeResponse
hostd_session_challenge_response_from_canonical_json(std::string_view value) {
  try {
    HostdSessionChallengeResponse parsed =
        parse_response(parse_canonical(value));
    if (!valid_response(parsed))
      reject("challenge response canonical semantics are invalid");
    return parsed;
  } catch (const HostdSessionChallengeError &) {
    throw;
  } catch (...) {
    reject("challenge response canonical decoding failed");
  }
}

std::string hostd_journal_fence_evidence_canonical_json(
    const HostdJournalFenceEvidence &value) {
  try {
    if (!valid_journal_evidence(value))
      reject("journal fence evidence is not exactly sealed");
    return journal_evidence_json(value, true).dump();
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("journal fence evidence serialization failed closed");
  }
}

HostdJournalFenceEvidence
hostd_journal_fence_evidence_from_canonical_json(std::string_view value) {
  try {
    HostdJournalFenceEvidence parsed =
        parse_journal_evidence(parse_canonical(value));
    if (!valid_journal_evidence(parsed))
      reject("journal fence evidence canonical semantics are invalid");
    return parsed;
  } catch (const HostdSessionChallengeError &) {
    throw;
  } catch (...) {
    reject("journal fence evidence canonical decoding failed");
  }
}

std::string hostd_session_challenge_evidence_canonical_json(
    const HostdSessionChallengeEvidence &value) {
  try {
    if (!valid_verified_evidence(value))
      reject("challenge verification evidence is not exactly sealed");
    return verified_evidence_json(value, true).dump();
  } catch (const HostdSessionChallengeRejected &) {
    throw;
  } catch (...) {
    reject("challenge verification evidence serialization failed closed");
  }
}

HostdSessionChallengeEvidence
hostd_session_challenge_evidence_from_canonical_json(std::string_view value) {
  try {
    HostdSessionChallengeEvidence parsed =
        parse_verified_evidence(parse_canonical(value));
    if (!valid_verified_evidence(parsed))
      reject("challenge evidence canonical semantics are invalid");
    return parsed;
  } catch (const HostdSessionChallengeError &) {
    throw;
  } catch (...) {
    reject("challenge evidence canonical decoding failed");
  }
}

} // namespace trainvm
