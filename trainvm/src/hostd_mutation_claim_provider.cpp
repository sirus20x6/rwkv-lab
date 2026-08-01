#include "trainvm/hostd_mutation_claim_provider.hpp"

#include <openssl/rand.h>

#include <array>
#include <cctype>
#include <cstdint>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace trainvm {
namespace {

bool valid_identifier(std::string_view value) {
  return !value.empty() && value.size() <= 192U &&
         std::ranges::all_of(value, [](char character) {
           const auto byte = static_cast<unsigned char>(character);
           return std::isalnum(byte) != 0 || character == '.' ||
                  character == '_' || character == ':' || character == '/' ||
                  character == '-' || character == '@';
         });
}

std::string random_controller_id() {
  std::array<unsigned char, 32> bytes{};
  if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
    throw HostdMutationClaimProviderError(
        "could not create a cryptographic controller identity");
  }
  constexpr char digits[] = "0123456789abcdef";
  std::string result = "trainvm-controller-";
  result.reserve(result.size() + bytes.size() * 2U);
  for (const unsigned char byte : bytes) {
    result.push_back(digits[(byte >> 4U) & 0x0fU]);
    result.push_back(digits[byte & 0x0fU]);
  }
  return result;
}

HostdJournalAuthorityIdentity journal_identity(
    const JournalAuthoritySnapshot& authority) {
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

}  // namespace

JournalHostdMutationClaimProvider::JournalHostdMutationClaimProvider(
    Journal& journal, HostdMutationClaimProviderConfig config)
    : journal_(journal),
      config_(std::move(config)),
      clock_(config_.authority_clock) {
  if (config_.api_version != kHostdMutationClaimProviderApiVersion ||
      !valid_identifier(config_.broker_epoch) ||
      !config_.authority_clock) {
    throw HostdMutationClaimProviderError(
        "hostd mutation claim provider configuration is invalid");
  }
}

HostdMutationOpen JournalHostdMutationClaimProvider::open_for_resource(
    std::string_view request_or_release_id) {
  const auto identity = journal_.host_resource_mutation_identity(
      std::string(request_or_release_id));
  if (!identity) {
    throw HostdMutationClaimProviderError(
        "host resource mutation is not durably journaled");
  }
  return open_for_scope({.run_id = identity->run_id,
                         .concurrency_key = identity->concurrency_key,
                         .logical_lease_id = identity->logical_lease_id,
                         .logical_fencing_token =
                             identity->logical_fencing_token});
}

HostdMutationOpen JournalHostdMutationClaimProvider::open_for_process(
    std::string_view launch_id) {
  const auto binding = journal_.launch_binding(std::string(launch_id));
  if (!binding) {
    throw HostdMutationClaimProviderError(
        "host process launch is not durably bound");
  }
  const auto& identity = binding->identity;
  return open_for_scope({.run_id = identity.run_id,
                         .concurrency_key = identity.concurrency_key,
                         .logical_lease_id = identity.lease_id,
                         .logical_fencing_token = identity.fencing_token});
}

HostdMutationOpen JournalHostdMutationClaimProvider::open_for_scope(
    const Scope& scope) {
  if (!valid_identifier(scope.run_id) ||
      !valid_identifier(scope.concurrency_key) ||
      !valid_identifier(scope.logical_lease_id) ||
      scope.logical_fencing_token == 0U) {
    throw HostdMutationClaimProviderError(
        "journal mutation scope is noncanonical");
  }
  std::lock_guard lock(mutex_);
  const auto owned = owned_fences_.find(scope.concurrency_key);
  if (owned != owned_fences_.end()) {
    // This is the critical split-brain check: an old process may never advance
    // a generation after a replacement process has superseded it.
    journal_.require_current_hostd_controller_fence(owned->second);
    if (owned->second.run_id == scope.run_id &&
        owned->second.concurrency_key == scope.concurrency_key &&
        owned->second.logical_lease_id == scope.logical_lease_id &&
        owned->second.logical_fencing_token ==
            scope.logical_fencing_token) {
      const JournalAuthoritySnapshot authority =
          journal_.journal_authority_snapshot();
      return {.api_version = std::string(kHostdMutationProtocolApiVersion),
              .claim = {
                  .journal = journal_identity(authority),
                  .controller = {
                      .run_id = owned->second.run_id,
                      .concurrency_key = owned->second.concurrency_key,
                      .controller_id = owned->second.controller_id,
                      .controller_generation =
                          owned->second.controller_generation,
                      .logical_lease_id = owned->second.logical_lease_id,
                      .logical_fencing_token =
                          owned->second.logical_fencing_token}}};
    }
  }

  const auto current = journal_.current_hostd_controller_fence(
      scope.concurrency_key);
  if (current &&
      current->controller_generation ==
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
    throw HostdMutationClaimProviderError(
        "hostd controller generation is exhausted");
  }
  const std::uint64_t generation =
      current ? current->controller_generation + 1U : 1U;
  JournalControllerFence requested{
      .broker_epoch = config_.broker_epoch,
      .run_id = scope.run_id,
      .concurrency_key = scope.concurrency_key,
      .controller_id = next_controller_id(),
      .controller_generation = generation,
      .logical_lease_id = scope.logical_lease_id,
      .logical_fencing_token = scope.logical_fencing_token,
  };
  JournalControllerFence durable =
      journal_.register_hostd_controller_fence(requested, clock_.sample());
  if (durable != requested) {
    throw HostdMutationClaimProviderError(
        "journal returned a different hostd controller fence");
  }
  owned_fences_.insert_or_assign(scope.concurrency_key, durable);
  const JournalAuthoritySnapshot authority =
      journal_.journal_authority_snapshot();
  return {.api_version = std::string(kHostdMutationProtocolApiVersion),
          .claim = {
              .journal = journal_identity(authority),
              .controller = {
                  .run_id = durable.run_id,
                  .concurrency_key = durable.concurrency_key,
                  .controller_id = durable.controller_id,
                  .controller_generation = durable.controller_generation,
                  .logical_lease_id = durable.logical_lease_id,
                  .logical_fencing_token =
                      durable.logical_fencing_token}}};
}

std::string JournalHostdMutationClaimProvider::next_controller_id() {
  std::string result = config_.controller_id_source
                           ? config_.controller_id_source()
                           : random_controller_id();
  if (!valid_identifier(result)) {
    throw HostdMutationClaimProviderError(
        "controller identity source returned a noncanonical identity");
  }
  return result;
}

}  // namespace trainvm
