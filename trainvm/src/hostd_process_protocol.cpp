#include "trainvm/hostd_process_protocol.hpp"

#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace trainvm {
namespace {

using Json = nlohmann::json;
constexpr std::size_t kMaximumCanonicalBytes = 64U * 1024U;

[[noreturn]] void reject(std::string_view message) {
  throw std::invalid_argument(std::string(message));
}

Json parse_canonical(std::string_view value) {
  if (value.empty() || value.size() > kMaximumCanonicalBytes)
    reject("hostd process canonical JSON size is invalid");
  try {
    Json parsed = Json::parse(value);
    if (parsed.dump() != value)
      reject("hostd process JSON is not canonical");
    return parsed;
  } catch (const std::invalid_argument&) {
    throw;
  } catch (const Json::exception&) {
    reject("hostd process JSON is malformed");
  }
}

void exact_fields(const Json& value,
                  std::initializer_list<std::string_view> fields) {
  if (!value.is_object() || value.size() != fields.size())
    reject("hostd process JSON fields are inexact");
  for (const std::string_view field : fields)
    if (!value.contains(std::string(field)))
      reject("hostd process JSON field is missing");
}

bool valid_digest(std::string_view value) {
  return value.starts_with("sha256:") && value.size() == 71U &&
         std::ranges::all_of(value.substr(7U), [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_attribution(std::string_view launch_id,
                         std::string_view allocation_id,
                         std::string_view grant_digest,
                         std::string_view journal_id,
                         std::string_view run_id,
                         std::string_view logical_lease_id,
                         std::uint64_t logical_fencing_token,
                         std::string_view spawn_receipt_digest) {
  if (launch_id.empty() || allocation_id.empty() ||
      !valid_digest(grant_digest) || journal_id.empty() || run_id.empty() ||
      logical_lease_id.empty() || logical_fencing_token == 0U ||
      !valid_digest(spawn_receipt_digest)) {
    reject("hostd process attribution is invalid");
  }
}

Json prepare_json(const HostdProcessPrepareRequest& value) {
  if (value.api_version != kHostdProcessPrepareApiVersion)
    reject("hostd process prepare API is invalid");
  const auto& identity = value.launch.identity;
  if (!identity.host_grant ||
      identity.launch_event_id.empty() ||
      identity.host_grant->request_id != value.grant.request_id ||
      identity.host_grant->grant_digest != value.grant.receipt_digest ||
      identity.host_grant->fences != value.grant.fences ||
      identity.run_id != value.grant.run_id ||
      identity.lease_id != value.grant.logical_lease_id ||
      identity.fencing_token != value.grant.logical_fencing_token ||
      identity.host.host_id != value.grant.host_id ||
      identity.host.boot_id != value.grant.boot_id) {
    reject("hostd process prepare launch and grant are inexact");
  }
  const std::vector<std::string> expected_roles =
      identity.code ? std::vector<std::string>{"executable", "code",
                                                "working_directory"}
                    : std::vector<std::string>{"executable",
                                                "working_directory"};
  if (value.descriptor_roles != expected_roles)
    reject("hostd process prepare descriptor roles are inexact");
  return {{"api_version", value.api_version},
          {"descriptor_roles", value.descriptor_roles},
          {"grant", resource_bundle_grant_json(value.grant)},
          {"launch", resolved_launch_spec_json(value.launch)}};
}

Json commit_json(const HostdProcessCommitRequest& value) {
  if (value.api_version != kHostdProcessCommitApiVersion)
    reject("hostd process commit API is invalid");
  require_attribution(value.launch_id, value.allocation_id,
                      value.grant_digest, value.journal_id, value.run_id,
                      value.logical_lease_id, value.logical_fencing_token,
                      value.spawn_receipt_digest);
  return {{"allocation_id", value.allocation_id},
          {"api_version", value.api_version},
          {"grant_digest", value.grant_digest},
          {"journal_id", value.journal_id},
          {"launch_id", value.launch_id},
          {"logical_fencing_token", value.logical_fencing_token},
          {"logical_lease_id", value.logical_lease_id},
          {"run_id", value.run_id},
          {"spawn_receipt_digest", value.spawn_receipt_digest}};
}

Json exit_json(const HostdProcessExitCommand& value) {
  if (value.api_version != kHostdProcessExitApiVersion ||
      value.exit_request_id.empty())
    reject("hostd process exit API or request identity is invalid");
  require_attribution(value.launch_id, value.allocation_id,
                      value.grant_digest, value.journal_id, value.run_id,
                      value.logical_lease_id, value.logical_fencing_token,
                      value.spawn_receipt_digest);
  return {{"allocation_id", value.allocation_id},
          {"api_version", value.api_version},
          {"exit_request_id", value.exit_request_id},
          {"grant_digest", value.grant_digest},
          {"journal_id", value.journal_id},
          {"launch_id", value.launch_id},
          {"logical_fencing_token", value.logical_fencing_token},
          {"logical_lease_id", value.logical_lease_id},
          {"request_termination", value.request_termination},
          {"run_id", value.run_id},
          {"spawn_receipt_digest", value.spawn_receipt_digest}};
}

Json prepared_json(const HostdProcessPreparedResult& value) {
  if (value.api_version != kHostdProcessPreparedApiVersion ||
      value.intent.request.launch_id != value.spawn.request.launch_id ||
      value.intent.receipt_digest !=
          value.spawn.request.launch_intent_digest) {
    reject("hostd prepared process result is inconsistent");
  }
  return {{"api_version", value.api_version},
          {"intent", host_process_launch_intent_json(value.intent)},
          {"replayed", value.replayed},
          {"spawn", host_process_spawn_receipt_json(value.spawn)}};
}

Json committed_json(const HostdProcessCommittedResult& value) {
  if (value.api_version != kHostdProcessCommittedApiVersion ||
      value.launch_id.empty() || !valid_digest(value.spawn_receipt_digest) ||
      !value.released_to_exec) {
    reject("hostd committed process result is invalid");
  }
  return {{"api_version", value.api_version},
          {"launch_id", value.launch_id},
          {"released_to_exec", value.released_to_exec},
          {"replayed", value.replayed},
          {"spawn_receipt_digest", value.spawn_receipt_digest}};
}

}  // namespace

std::string hostd_process_prepare_canonical_json(
    const HostdProcessPrepareRequest& value) {
  return prepare_json(value).dump();
}

HostdProcessPrepareRequest hostd_process_prepare_from_canonical_json(
    std::string_view value) {
  const Json parsed = parse_canonical(value);
  exact_fields(parsed, {"api_version", "descriptor_roles", "grant", "launch"});
  HostdProcessPrepareRequest result{
      .api_version = parsed.at("api_version").get<std::string>(),
      .launch = resolved_launch_spec_from_json(parsed.at("launch")),
      .grant = resource_bundle_grant_from_json(parsed.at("grant")),
      .descriptor_roles =
          parsed.at("descriptor_roles").get<std::vector<std::string>>(),
  };
  if (prepare_json(result) != parsed)
    reject("hostd process prepare is not canonical");
  return result;
}

std::string hostd_process_commit_canonical_json(
    const HostdProcessCommitRequest& value) {
  return commit_json(value).dump();
}

HostdProcessCommitRequest hostd_process_commit_from_canonical_json(
    std::string_view value) {
  const Json parsed = parse_canonical(value);
  exact_fields(parsed, {"allocation_id", "api_version", "grant_digest",
                        "journal_id", "launch_id", "logical_fencing_token",
                        "logical_lease_id", "run_id",
                        "spawn_receipt_digest"});
  HostdProcessCommitRequest result{
      .api_version = parsed.at("api_version").get<std::string>(),
      .launch_id = parsed.at("launch_id").get<std::string>(),
      .allocation_id = parsed.at("allocation_id").get<std::string>(),
      .grant_digest = parsed.at("grant_digest").get<std::string>(),
      .journal_id = parsed.at("journal_id").get<std::string>(),
      .run_id = parsed.at("run_id").get<std::string>(),
      .logical_lease_id = parsed.at("logical_lease_id").get<std::string>(),
      .logical_fencing_token =
          parsed.at("logical_fencing_token").get<std::uint64_t>(),
      .spawn_receipt_digest =
          parsed.at("spawn_receipt_digest").get<std::string>(),
  };
  if (commit_json(result) != parsed)
    reject("hostd process commit is not canonical");
  return result;
}

std::string hostd_process_exit_canonical_json(
    const HostdProcessExitCommand& value) {
  return exit_json(value).dump();
}

HostdProcessExitCommand hostd_process_exit_from_canonical_json(
    std::string_view value) {
  const Json parsed = parse_canonical(value);
  exact_fields(parsed, {"allocation_id", "api_version", "exit_request_id",
                        "grant_digest", "journal_id", "launch_id",
                        "logical_fencing_token", "logical_lease_id",
                        "request_termination", "run_id",
                        "spawn_receipt_digest"});
  HostdProcessExitCommand result{
      .api_version = parsed.at("api_version").get<std::string>(),
      .exit_request_id = parsed.at("exit_request_id").get<std::string>(),
      .launch_id = parsed.at("launch_id").get<std::string>(),
      .allocation_id = parsed.at("allocation_id").get<std::string>(),
      .grant_digest = parsed.at("grant_digest").get<std::string>(),
      .journal_id = parsed.at("journal_id").get<std::string>(),
      .run_id = parsed.at("run_id").get<std::string>(),
      .logical_lease_id = parsed.at("logical_lease_id").get<std::string>(),
      .logical_fencing_token =
          parsed.at("logical_fencing_token").get<std::uint64_t>(),
      .spawn_receipt_digest =
          parsed.at("spawn_receipt_digest").get<std::string>(),
      .request_termination =
          parsed.at("request_termination").get<bool>(),
  };
  if (exit_json(result) != parsed)
    reject("hostd process exit is not canonical");
  return result;
}

std::string hostd_process_prepared_canonical_json(
    const HostdProcessPreparedResult& value) {
  return prepared_json(value).dump();
}

HostdProcessPreparedResult hostd_process_prepared_from_canonical_json(
    std::string_view value) {
  const Json parsed = parse_canonical(value);
  exact_fields(parsed, {"api_version", "intent", "replayed", "spawn"});
  HostdProcessPreparedResult result{
      .api_version = parsed.at("api_version").get<std::string>(),
      .intent = host_process_launch_intent_from_json(parsed.at("intent")),
      .spawn = host_process_spawn_receipt_from_json(parsed.at("spawn")),
      .replayed = parsed.at("replayed").get<bool>(),
  };
  if (prepared_json(result) != parsed)
    reject("hostd prepared process result is not canonical");
  return result;
}

std::string hostd_process_committed_canonical_json(
    const HostdProcessCommittedResult& value) {
  return committed_json(value).dump();
}

HostdProcessCommittedResult hostd_process_committed_from_canonical_json(
    std::string_view value) {
  const Json parsed = parse_canonical(value);
  exact_fields(parsed, {"api_version", "launch_id", "released_to_exec",
                        "replayed", "spawn_receipt_digest"});
  HostdProcessCommittedResult result{
      .api_version = parsed.at("api_version").get<std::string>(),
      .launch_id = parsed.at("launch_id").get<std::string>(),
      .spawn_receipt_digest =
          parsed.at("spawn_receipt_digest").get<std::string>(),
      .released_to_exec = parsed.at("released_to_exec").get<bool>(),
      .replayed = parsed.at("replayed").get<bool>(),
  };
  if (committed_json(result) != parsed)
    reject("hostd committed process result is not canonical");
  return result;
}

}  // namespace trainvm
