#include "trainvm/hostd_journal_fence_attestor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string_view>

#include <sys/types.h>

namespace trainvm {
namespace {

constexpr std::size_t kMaximumIdentifierBytes = 192U;
constexpr std::int64_t kMaximumChallengeTtlNs = 60'000'000'000LL;
constexpr std::string_view kChallengeIdPrefix = "hostd-challenge-";
constexpr std::string_view kSessionNoncePrefix = "hostd-session-nonce-";

bool valid_identifier(std::string_view value) {
  return !value.empty() && value.size() <= kMaximumIdentifierBytes &&
         std::ranges::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return std::isalnum(byte) != 0 || character == '.' ||
                  character == '_' || character == ':' || character == '/' ||
                  character == '-' || character == '@';
         });
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

bool valid_controller(const HostdJournalControllerFence& value) {
  return valid_identifier(value.run_id) &&
         valid_identifier(value.concurrency_key) &&
         valid_identifier(value.controller_id) &&
         valid_identifier(value.logical_lease_id) &&
         value.controller_generation > 0U &&
         value.logical_fencing_token > 0U &&
         value.logical_fencing_token <=
             static_cast<std::uint64_t>(
                 std::numeric_limits<std::int64_t>::max());
}

HostdJournalAuthorityIdentity
challenge_identity(const JournalAuthoritySnapshot& authority) {
  return {.directory_path = authority.file.directory_path,
          .journal_name = authority.file.journal_name,
          .authority_name = authority.file.authority_name,
          .journal_id = authority.journal_id,
          .directory_device = authority.file.directory_device,
          .directory_inode = authority.file.directory_inode,
          .journal_device = authority.file.device,
          .journal_inode = authority.file.inode,
          .authority_device = authority.file.authority_device,
          .authority_inode = authority.file.authority_inode,
          .owner_uid = authority.file.owner_uid};
}

bool valid_query_shape(const HostdJournalFenceQuery& query) {
  return query.api_version == kHostdJournalFenceQueryApiVersion &&
         valid_prefixed_token(query.challenge_id, kChallengeIdPrefix) &&
         valid_prefixed_token(query.session_nonce, kSessionNoncePrefix) &&
         query.challenge_id.substr(kChallengeIdPrefix.size()) !=
             query.session_nonce.substr(kSessionNoncePrefix.size()) &&
         valid_identifier(query.host_id) && valid_identifier(query.boot_id) &&
         valid_identifier(query.broker_epoch) &&
         query.issued_boottime_ns >= 0 &&
         query.expires_boottime_ns > query.issued_boottime_ns &&
         query.expires_boottime_ns - query.issued_boottime_ns <=
             kMaximumChallengeTtlNs &&
         valid_controller(query.claim.controller);
}

[[noreturn]] void fail() {
  throw HostdJournalFenceAttestorError(
      "journal fence attestation failed closed");
}

}  // namespace

HostdJournalFenceAttestor::HostdJournalFenceAttestor(
    Journal& journal, HostdJournalFenceAttestorConfig config,
    std::shared_ptr<IHostdSessionChallengeTimeSource> time_source)
    : journal_(journal),
      config_(std::move(config)),
      time_source_(std::move(time_source)) {
  if (config_.api_version != kHostdJournalFenceAttestorApiVersion ||
      !valid_identifier(config_.broker_epoch) ||
      !valid_controller(config_.controller) || !time_source_) {
    throw HostdJournalFenceAttestorError(
        "journal fence attestor configuration is invalid");
  }
  try {
    const HostdSessionChallengeTime registered_at = time_source_->now();
    if (!valid_identifier(registered_at.boot_id) ||
        registered_at.boottime_ns < 0)
      fail();
    const JournalControllerFence durable =
        journal_.register_hostd_controller_fence(
            {.broker_epoch = config_.broker_epoch,
             .run_id = config_.controller.run_id,
             .concurrency_key = config_.controller.concurrency_key,
             .controller_id = config_.controller.controller_id,
             .controller_generation =
                 config_.controller.controller_generation,
             .logical_lease_id = config_.controller.logical_lease_id,
             .logical_fencing_token =
                 config_.controller.logical_fencing_token},
            {.wall = {.nanoseconds = 0},
             .boot = {.nanoseconds = registered_at.boottime_ns},
             .boot_id = registered_at.boot_id});
    if (durable.broker_epoch != config_.broker_epoch ||
        durable.run_id != config_.controller.run_id ||
        durable.concurrency_key != config_.controller.concurrency_key ||
        durable.controller_id != config_.controller.controller_id ||
        durable.controller_generation !=
            config_.controller.controller_generation ||
        durable.logical_lease_id !=
            config_.controller.logical_lease_id ||
        durable.logical_fencing_token !=
            config_.controller.logical_fencing_token)
      fail();
    const JournalAuthoritySnapshot authority =
        journal_.journal_authority_snapshot();
    claim_ = {.journal = challenge_identity(authority),
              .controller = config_.controller};
    if (authority.host.host_id.empty() || authority.host.boot_id.empty() ||
        authority.file.owner_uid >
            static_cast<std::uint64_t>(std::numeric_limits<uid_t>::max())) {
      fail();
    }
  } catch (const HostdJournalFenceAttestorError&) {
    throw;
  } catch (...) {
    fail();
  }
}

const HostdSessionChallengeClaim&
HostdJournalFenceAttestor::claim() const noexcept {
  return claim_;
}

HostdJournalFenceEvidence HostdJournalFenceAttestor::attest(
    const HostdJournalFenceQuery& query) {
  try {
    if (!valid_query_shape(query) || query.broker_epoch != config_.broker_epoch ||
        query.claim != claim_) {
      fail();
    }
    const JournalControllerFence controller{
        .broker_epoch = config_.broker_epoch,
        .run_id = config_.controller.run_id,
        .concurrency_key = config_.controller.concurrency_key,
        .controller_id = config_.controller.controller_id,
        .controller_generation = config_.controller.controller_generation,
        .logical_lease_id = config_.controller.logical_lease_id,
        .logical_fencing_token = config_.controller.logical_fencing_token};
    journal_.require_current_hostd_controller_fence(controller);
    const HostdSessionChallengeTime before_snapshot = time_source_->now();
    if (before_snapshot.boot_id != query.boot_id ||
        before_snapshot.boottime_ns < query.issued_boottime_ns ||
        before_snapshot.boottime_ns >= query.expires_boottime_ns) {
      fail();
    }
    const JournalLogicalFenceSnapshot snapshot =
        journal_.journal_logical_fence_snapshot(
            config_.controller.concurrency_key, config_.controller.run_id,
            config_.controller.logical_lease_id,
            config_.controller.logical_fencing_token,
            {.wall = {.nanoseconds = 0},
             .boot = {.nanoseconds = before_snapshot.boottime_ns},
             .boot_id = before_snapshot.boot_id});
    const HostdSessionChallengeTime after_snapshot = time_source_->now();
    if (after_snapshot.boot_id != query.boot_id ||
        after_snapshot.boottime_ns < before_snapshot.boottime_ns ||
        after_snapshot.boottime_ns >= query.expires_boottime_ns ||
        after_snapshot.boottime_ns >= snapshot.lease.expires_boottime_ns) {
      fail();
    }
    const JournalLogicalFenceSnapshot confirmed =
        journal_.journal_logical_fence_snapshot(
            config_.controller.concurrency_key, config_.controller.run_id,
            config_.controller.logical_lease_id,
            config_.controller.logical_fencing_token,
            {.wall = {.nanoseconds = 0},
             .boot = {.nanoseconds = after_snapshot.boottime_ns},
             .boot_id = after_snapshot.boot_id});
    journal_.require_current_hostd_controller_fence(controller);
    if (confirmed != snapshot)
      fail();
    if (snapshot.authority.host.host_id != query.host_id ||
        snapshot.authority.host.boot_id != query.boot_id ||
        challenge_identity(snapshot.authority) != query.claim.journal ||
        snapshot.lease.concurrency_key !=
            config_.controller.concurrency_key ||
        snapshot.lease.owner_run_id != config_.controller.run_id ||
        snapshot.lease.lease_id != config_.controller.logical_lease_id ||
        snapshot.lease.fencing_token !=
            config_.controller.logical_fencing_token) {
      fail();
    }
    return hostd_seal_journal_fence_evidence(
        {.api_version = std::string(kHostdJournalFenceEvidenceApiVersion),
         .challenge_id = query.challenge_id,
         .session_nonce = query.session_nonce,
         .host_id = query.host_id,
         .boot_id = query.boot_id,
         .broker_epoch = query.broker_epoch,
         .observed_boottime_ns = after_snapshot.boottime_ns,
         .journal = query.claim.journal,
         .controller = query.claim.controller,
         .live = true,
         .evidence_digest = {}});
  } catch (const HostdJournalFenceAttestorError&) {
    throw;
  } catch (...) {
    fail();
  }
}

}  // namespace trainvm
