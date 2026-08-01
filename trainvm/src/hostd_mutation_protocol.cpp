#include "trainvm/hostd_mutation_protocol.hpp"

#include <openssl/sha.h>

#include <array>
#include <ranges>
#include <utility>

#include <nlohmann/json.hpp>

namespace trainvm {
namespace {

using Json = nlohmann::json;
constexpr std::size_t kMaximumCanonicalBytes = 64U * 1024U;

[[noreturn]] void reject(std::string_view message) {
  throw HostdMutationProtocolError(std::string(message));
}

bool valid_hex(std::string_view value) {
  return value.size() == 64U &&
         std::ranges::all_of(value, [](unsigned char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool valid_digest(std::string_view value) {
  return value.starts_with("sha256:") && valid_hex(value.substr(7U));
}

bool valid_challenge_id(std::string_view value) {
  constexpr std::string_view prefix = "hostd-challenge-";
  return value.starts_with(prefix) && valid_hex(value.substr(prefix.size()));
}

std::string digest(std::string_view domain, std::string_view canonical) {
  std::string framed;
  framed.reserve(domain.size() + 1U + canonical.size());
  framed.append(domain);
  framed.push_back('\0');
  framed.append(canonical);
  std::array<unsigned char, SHA256_DIGEST_LENGTH> bytes{};
  if (::SHA256(reinterpret_cast<const unsigned char*>(framed.data()),
               framed.size(), bytes.data()) == nullptr) {
    reject("hostd mutation digest computation failed");
  }
  constexpr std::string_view hex = "0123456789abcdef";
  std::string result = "sha256:";
  result.reserve(71U);
  for (const unsigned char byte : bytes) {
    result.push_back(hex[byte >> 4U]);
    result.push_back(hex[byte & 0x0fU]);
  }
  return result;
}

Json parse_canonical(std::string_view value) {
  if (value.empty() || value.size() > kMaximumCanonicalBytes)
    reject("hostd mutation canonical JSON size is invalid");
  try {
    Json parsed = Json::parse(value);
    if (parsed.dump() != value)
      reject("hostd mutation JSON is not canonical");
    return parsed;
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (const Json::exception&) {
    reject("hostd mutation JSON is malformed");
  }
}

void exact_fields(const Json& value,
                  std::initializer_list<std::string_view> fields) {
  if (!value.is_object() || value.size() != fields.size())
    reject("hostd mutation JSON fields are inexact");
  for (const auto field : fields)
    if (!value.contains(std::string(field)))
      reject("hostd mutation JSON field is missing");
}

Json nested_json(std::string canonical) {
  return Json::parse(std::move(canonical));
}

std::string mutation_name(HostdMutationKind kind) {
  switch (kind) {
    case HostdMutationKind::request_bundle:
      return "request_bundle";
    case HostdMutationKind::reconcile_bundle_outcome:
      return "reconcile_bundle_outcome";
    case HostdMutationKind::release_bundle:
      return "release_bundle";
    case HostdMutationKind::prepare_process:
      return "prepare_process";
    case HostdMutationKind::commit_process:
      return "commit_process";
    case HostdMutationKind::finalize_process:
      return "finalize_process";
  }
  reject("hostd mutation kind is invalid");
}

HostdMutationKind parse_mutation(const Json& value) {
  if (!value.is_string()) reject("hostd mutation kind must be text");
  const auto name = value.get<std::string>();
  if (name == "request_bundle") return HostdMutationKind::request_bundle;
  if (name == "reconcile_bundle_outcome")
    return HostdMutationKind::reconcile_bundle_outcome;
  if (name == "release_bundle") return HostdMutationKind::release_bundle;
  if (name == "prepare_process") return HostdMutationKind::prepare_process;
  if (name == "commit_process") return HostdMutationKind::commit_process;
  if (name == "finalize_process") return HostdMutationKind::finalize_process;
  reject("hostd mutation kind is unknown");
}

std::string reply_name(HostdMutationReplyKind kind) {
  switch (kind) {
    case HostdMutationReplyKind::bundle_outcome:
      return "bundle_outcome";
    case HostdMutationReplyKind::reconciliation_missing:
      return "reconciliation_missing";
    case HostdMutationReplyKind::release_outcome:
      return "release_outcome";
    case HostdMutationReplyKind::process_prepared:
      return "process_prepared";
    case HostdMutationReplyKind::process_committed:
      return "process_committed";
    case HostdMutationReplyKind::process_exited:
      return "process_exited";
  }
  reject("hostd mutation reply kind is invalid");
}

HostdMutationReplyKind parse_reply_kind(const Json& value) {
  if (!value.is_string()) reject("hostd mutation reply kind must be text");
  const auto name = value.get<std::string>();
  if (name == "bundle_outcome") return HostdMutationReplyKind::bundle_outcome;
  if (name == "reconciliation_missing")
    return HostdMutationReplyKind::reconciliation_missing;
  if (name == "release_outcome") return HostdMutationReplyKind::release_outcome;
  if (name == "process_prepared")
    return HostdMutationReplyKind::process_prepared;
  if (name == "process_committed")
    return HostdMutationReplyKind::process_committed;
  if (name == "process_exited") return HostdMutationReplyKind::process_exited;
  reject("hostd mutation reply kind is unknown");
}

void require_attribution(const HostdMutationCommand& command) {
  const auto& controller = command.challenge_response.claim.controller;
  const auto& journal = command.challenge_response.claim.journal;
  if (command.bundle_request) {
    const auto& request = *command.bundle_request;
    if (request.journal_id != journal.journal_id ||
        request.run_id != controller.run_id ||
        request.logical_lease_id != controller.logical_lease_id ||
        request.logical_fencing_token != controller.logical_fencing_token)
      reject("hostd bundle command attribution disagrees with challenge");
  }
  if (command.release_request) {
    const auto& request = *command.release_request;
    if (request.journal_id != journal.journal_id ||
        request.run_id != controller.run_id ||
        request.logical_lease_id != controller.logical_lease_id ||
        request.logical_fencing_token != controller.logical_fencing_token)
      reject("hostd release command attribution disagrees with challenge");
  }
  const auto require_process = [&](const std::string& journal_id,
                                   const std::string& run_id,
                                   const std::string& lease_id,
                                   std::uint64_t fencing_token) {
    if (journal_id != journal.journal_id || run_id != controller.run_id ||
        lease_id != controller.logical_lease_id ||
        fencing_token != controller.logical_fencing_token) {
      reject("hostd process command attribution disagrees with challenge");
    }
  };
  if (command.process_prepare) {
    const auto& grant = command.process_prepare->grant;
    require_process(grant.journal_id, grant.run_id, grant.logical_lease_id,
                    grant.logical_fencing_token);
  }
  if (command.process_commit) {
    const auto& request = *command.process_commit;
    require_process(request.journal_id, request.run_id,
                    request.logical_lease_id,
                    request.logical_fencing_token);
  }
  if (command.process_exit) {
    const auto& request = *command.process_exit;
    require_process(request.journal_id, request.run_id,
                    request.logical_lease_id,
                    request.logical_fencing_token);
  }
}

Json command_json(const HostdMutationCommand& value) {
  if (value.api_version != kHostdMutationProtocolApiVersion)
    reject("hostd mutation command API version is invalid");
  (void)hostd_session_challenge_response_canonical_json(
      value.challenge_response);
  const bool bundle_kind =
      value.mutation == HostdMutationKind::request_bundle ||
      value.mutation == HostdMutationKind::reconcile_bundle_outcome;
  const bool release_kind =
      value.mutation == HostdMutationKind::release_bundle;
  const bool prepare_kind =
      value.mutation == HostdMutationKind::prepare_process;
  const bool commit_kind =
      value.mutation == HostdMutationKind::commit_process;
  const bool exit_kind =
      value.mutation == HostdMutationKind::finalize_process;
  const std::size_t payload_count =
      static_cast<std::size_t>(value.bundle_request.has_value()) +
      static_cast<std::size_t>(value.release_request.has_value()) +
      static_cast<std::size_t>(value.process_prepare.has_value()) +
      static_cast<std::size_t>(value.process_commit.has_value()) +
      static_cast<std::size_t>(value.process_exit.has_value());
  if (bundle_kind != value.bundle_request.has_value() ||
      release_kind != value.release_request.has_value() ||
      prepare_kind != value.process_prepare.has_value() ||
      commit_kind != value.process_commit.has_value() ||
      exit_kind != value.process_exit.has_value() || payload_count != 1U)
    reject("hostd mutation command payload contradicts its kind");
  require_attribution(value);
  Json bundle = nullptr;
  Json release = nullptr;
  Json process_prepare = nullptr;
  Json process_commit = nullptr;
  Json process_exit = nullptr;
  if (value.bundle_request)
    bundle = resource_request_json(*value.bundle_request);
  if (value.release_request)
    release = resource_release_request_json(*value.release_request);
  if (value.process_prepare)
    process_prepare = Json::parse(hostd_process_prepare_canonical_json(
        *value.process_prepare));
  if (value.process_commit)
    process_commit = Json::parse(hostd_process_commit_canonical_json(
        *value.process_commit));
  if (value.process_exit)
    process_exit = Json::parse(hostd_process_exit_canonical_json(
        *value.process_exit));
  Json result{{"api_version", value.api_version},
              {"bundle_request", std::move(bundle)},
              {"challenge_response",
               nested_json(hostd_session_challenge_response_canonical_json(
                   value.challenge_response))},
              {"mutation", mutation_name(value.mutation)},
              {"process_commit", std::move(process_commit)},
              {"process_exit", std::move(process_exit)},
              {"process_prepare", std::move(process_prepare)},
              {"release_request", std::move(release)}};
  return result;
}

void validate_reply(const HostdMutationReply& value) {
  if (value.api_version != kHostdMutationProtocolApiVersion ||
      !valid_challenge_id(value.challenge_id) ||
      !valid_digest(value.command_digest))
    reject("hostd mutation reply identity is invalid");
  const bool bundle = value.kind == HostdMutationReplyKind::bundle_outcome;
  const bool release = value.kind == HostdMutationReplyKind::release_outcome;
  const bool prepared = value.kind == HostdMutationReplyKind::process_prepared;
  const bool committed =
      value.kind == HostdMutationReplyKind::process_committed;
  const bool exited = value.kind == HostdMutationReplyKind::process_exited;
  const std::size_t payload_count =
      static_cast<std::size_t>(value.bundle_result.has_value()) +
      static_cast<std::size_t>(value.release_result.has_value()) +
      static_cast<std::size_t>(value.process_prepared.has_value()) +
      static_cast<std::size_t>(value.process_committed.has_value()) +
      static_cast<std::size_t>(value.process_exit.has_value());
  if (bundle != value.bundle_result.has_value() ||
      release != value.release_result.has_value() ||
      prepared != value.process_prepared.has_value() ||
      committed != value.process_committed.has_value() ||
      exited != value.process_exit.has_value() ||
      (value.kind == HostdMutationReplyKind::reconciliation_missing
           ? payload_count != 0U
           : payload_count != 1U))
    reject("hostd mutation reply payload contradicts its kind");
  if (value.bundle_result) (void)bundle_request_result_json(*value.bundle_result);
  if (value.release_result) (void)bundle_release_result_json(*value.release_result);
  if (value.process_prepared)
    (void)hostd_process_prepared_canonical_json(*value.process_prepared);
  if (value.process_committed)
    (void)hostd_process_committed_canonical_json(*value.process_committed);
  if (value.process_exit)
    (void)host_process_exit_receipt_json(value.process_exit->receipt);
}

}  // namespace

HostdMutationCommand seal_hostd_mutation_command(HostdMutationCommand command) {
  try {
    command.command_digest.clear();
    const auto canonical = command_json(command).dump();
    command.command_digest =
        digest(kHostdMutationCommandDigestDomain, canonical);
    return command;
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation command sealing failed closed");
  }
}

std::string hostd_mutation_open_canonical_json(const HostdMutationOpen& value) {
  try {
    if (value.api_version != kHostdMutationProtocolApiVersion)
      reject("hostd mutation open API version is invalid");
    return Json{{"api_version", value.api_version},
                {"claim", nested_json(
                              hostd_session_challenge_claim_canonical_json(
                                  value.claim))}}
        .dump();
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation open serialization failed closed");
  }
}

HostdMutationOpen hostd_mutation_open_from_canonical_json(
    std::string_view value) {
  try {
    const Json parsed = parse_canonical(value);
    exact_fields(parsed, {"api_version", "claim"});
    HostdMutationOpen result{
        .api_version = parsed.at("api_version").get<std::string>(),
        .claim = hostd_session_challenge_claim_from_canonical_json(
            parsed.at("claim").dump()),
    };
    (void)hostd_mutation_open_canonical_json(result);
    return result;
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation open decoding failed");
  }
}

std::string hostd_mutation_command_canonical_json(
    const HostdMutationCommand& value) {
  try {
    const Json canonical = command_json(value);
    if (!valid_digest(value.command_digest) ||
        value.command_digest !=
            digest(kHostdMutationCommandDigestDomain, canonical.dump()))
      reject("hostd mutation command is not exactly sealed");
    Json result = canonical;
    result["command_digest"] = value.command_digest;
    return result.dump();
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation command serialization failed closed");
  }
}

HostdMutationCommand hostd_mutation_command_from_canonical_json(
    std::string_view value) {
  try {
    const Json parsed = parse_canonical(value);
    exact_fields(parsed, {"api_version", "bundle_request", "challenge_response",
                          "command_digest", "mutation", "process_commit",
                          "process_exit", "process_prepare",
                          "release_request"});
    HostdMutationCommand result{
        .api_version = parsed.at("api_version").get<std::string>(),
        .challenge_response =
            hostd_session_challenge_response_from_canonical_json(
                parsed.at("challenge_response").dump()),
        .mutation = parse_mutation(parsed.at("mutation")),
        .bundle_request = std::nullopt,
        .release_request = std::nullopt,
        .process_prepare = std::nullopt,
        .process_commit = std::nullopt,
        .process_exit = std::nullopt,
        .command_digest = parsed.at("command_digest").get<std::string>(),
    };
    if (!parsed.at("bundle_request").is_null())
      result.bundle_request =
          resource_request_from_json(parsed.at("bundle_request"));
    if (!parsed.at("release_request").is_null())
      result.release_request =
          resource_release_request_from_json(parsed.at("release_request"));
    if (!parsed.at("process_prepare").is_null())
      result.process_prepare = hostd_process_prepare_from_canonical_json(
          parsed.at("process_prepare").dump());
    if (!parsed.at("process_commit").is_null())
      result.process_commit = hostd_process_commit_from_canonical_json(
          parsed.at("process_commit").dump());
    if (!parsed.at("process_exit").is_null())
      result.process_exit = hostd_process_exit_from_canonical_json(
          parsed.at("process_exit").dump());
    (void)hostd_mutation_command_canonical_json(result);
    return result;
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation command decoding failed");
  }
}

std::string hostd_mutation_reply_canonical_json(
    const HostdMutationReply& value) {
  try {
    validate_reply(value);
    Json bundle = nullptr;
    Json release = nullptr;
    Json process_prepared = nullptr;
    Json process_committed = nullptr;
    Json process_exit = nullptr;
    if (value.bundle_result)
      bundle = bundle_request_result_json(*value.bundle_result);
    if (value.release_result)
      release = bundle_release_result_json(*value.release_result);
    if (value.process_prepared)
      process_prepared = Json::parse(hostd_process_prepared_canonical_json(
          *value.process_prepared));
    if (value.process_committed)
      process_committed = Json::parse(hostd_process_committed_canonical_json(
          *value.process_committed));
    if (value.process_exit)
      process_exit = {
          {"receipt", host_process_exit_receipt_json(
                          value.process_exit->receipt)},
          {"replayed", value.process_exit->replayed}};
    return Json{{"api_version", value.api_version},
                {"bundle_result", std::move(bundle)},
                {"challenge_id", value.challenge_id},
                {"command_digest", value.command_digest},
                {"kind", reply_name(value.kind)},
                {"process_committed", std::move(process_committed)},
                {"process_exit", std::move(process_exit)},
                {"process_prepared", std::move(process_prepared)},
                {"release_result", std::move(release)}}
        .dump();
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation reply serialization failed closed");
  }
}

HostdMutationReply hostd_mutation_reply_from_canonical_json(
    std::string_view value) {
  try {
    const Json parsed = parse_canonical(value);
    exact_fields(parsed, {"api_version", "bundle_result", "challenge_id",
                          "command_digest", "kind", "process_committed",
                          "process_exit", "process_prepared",
                          "release_result"});
    HostdMutationReply result{
        .api_version = parsed.at("api_version").get<std::string>(),
        .kind = parse_reply_kind(parsed.at("kind")),
        .challenge_id = parsed.at("challenge_id").get<std::string>(),
        .command_digest = parsed.at("command_digest").get<std::string>(),
        .bundle_result = std::nullopt,
        .release_result = std::nullopt,
        .process_prepared = std::nullopt,
        .process_committed = std::nullopt,
        .process_exit = std::nullopt,
    };
    if (!parsed.at("bundle_result").is_null())
      result.bundle_result =
          bundle_request_result_from_json(parsed.at("bundle_result"));
    if (!parsed.at("release_result").is_null())
      result.release_result =
          bundle_release_result_from_json(parsed.at("release_result"));
    if (!parsed.at("process_prepared").is_null())
      result.process_prepared = hostd_process_prepared_from_canonical_json(
          parsed.at("process_prepared").dump());
    if (!parsed.at("process_committed").is_null())
      result.process_committed = hostd_process_committed_from_canonical_json(
          parsed.at("process_committed").dump());
    if (!parsed.at("process_exit").is_null()) {
      const Json& exited = parsed.at("process_exit");
      exact_fields(exited, {"receipt", "replayed"});
      result.process_exit = HostProcessExitResult{
          .receipt = host_process_exit_receipt_from_json(exited.at("receipt")),
          .replayed = exited.at("replayed").get<bool>(),
      };
    }
    validate_reply(result);
    return result;
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation reply decoding failed");
  }
}

void validate_hostd_mutation_exchange(const HostdMutationOpen& open,
                                      const HostdSessionChallenge& challenge,
                                      const HostdMutationCommand& command) {
  try {
    (void)hostd_mutation_open_canonical_json(open);
    (void)hostd_session_challenge_canonical_json(challenge);
    (void)hostd_mutation_command_canonical_json(command);
    if (open.claim != challenge.claim ||
        command.challenge_response !=
            hostd_session_challenge_response(challenge))
      reject("hostd mutation exchange challenge binding is inexact");
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation exchange validation failed closed");
  }
}

void validate_hostd_mutation_reply(const HostdMutationCommand& command,
                                   const HostdMutationReply& reply) {
  try {
    (void)hostd_mutation_command_canonical_json(command);
    (void)hostd_mutation_reply_canonical_json(reply);
    if (reply.challenge_id != command.challenge_response.challenge_id ||
        reply.command_digest != command.command_digest)
      reject("hostd mutation reply does not bind the exact command");
    switch (command.mutation) {
      case HostdMutationKind::request_bundle:
        if (reply.kind != HostdMutationReplyKind::bundle_outcome)
          reject("bundle request received the wrong reply kind");
        return;
      case HostdMutationKind::reconcile_bundle_outcome:
        if (reply.kind != HostdMutationReplyKind::bundle_outcome &&
            reply.kind != HostdMutationReplyKind::reconciliation_missing)
          reject("bundle reconciliation received the wrong reply kind");
        if (reply.bundle_result && !reply.bundle_result->replayed)
          reject("bundle reconciliation returned a non-replayed outcome");
        return;
      case HostdMutationKind::release_bundle:
        if (reply.kind != HostdMutationReplyKind::release_outcome)
          reject("bundle release received the wrong reply kind");
        return;
      case HostdMutationKind::prepare_process: {
        if (reply.kind != HostdMutationReplyKind::process_prepared ||
            !reply.process_prepared || !command.process_prepare)
          reject("process prepare received the wrong reply kind");
        const auto& request = *command.process_prepare;
        const auto& result = *reply.process_prepared;
        if (result.intent.request.launch_id !=
                request.launch.identity.launch_event_id ||
            result.intent.request.allocation_id != request.grant.allocation_id ||
            result.intent.request.grant_digest != request.grant.receipt_digest ||
            result.intent.request.resolved_launch_digest !=
                request.launch.spec_digest ||
            result.spawn.request.launch_id !=
                request.launch.identity.launch_event_id) {
          reject("process prepare reply disagrees with its exact request");
        }
        return;
      }
      case HostdMutationKind::commit_process:
        if (reply.kind != HostdMutationReplyKind::process_committed ||
            !reply.process_committed || !command.process_commit ||
            reply.process_committed->launch_id !=
                command.process_commit->launch_id ||
            reply.process_committed->spawn_receipt_digest !=
                command.process_commit->spawn_receipt_digest)
          reject("process commit reply disagrees with its exact request");
        return;
      case HostdMutationKind::finalize_process:
        if (reply.kind != HostdMutationReplyKind::process_exited ||
            !reply.process_exit || !command.process_exit ||
            reply.process_exit->receipt.request.exit_request_id !=
                command.process_exit->exit_request_id ||
            reply.process_exit->receipt.request.launch_id !=
                command.process_exit->launch_id ||
            reply.process_exit->receipt.request.spawn_receipt_digest !=
                command.process_exit->spawn_receipt_digest)
          reject("process exit reply disagrees with its exact request");
        return;
    }
    reject("hostd mutation command kind is invalid");
  } catch (const HostdMutationProtocolError&) {
    throw;
  } catch (...) {
    reject("hostd mutation reply validation failed closed");
  }
}

}  // namespace trainvm
