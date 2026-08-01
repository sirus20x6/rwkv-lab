#include "trainvm/worker_bootstrap.hpp"

#include <algorithm>
#include <ranges>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "trainvm/document.hpp"

namespace trainvm {
namespace {

using Json = nlohmann::json;

[[noreturn]] void reject(std::string message) {
  throw std::invalid_argument(std::move(message));
}

bool bounded_text(std::string_view value, std::size_t maximum) {
  return !value.empty() && value.size() <= maximum &&
         !value.contains('\0') && !value.contains('\n') &&
         !value.contains('\r');
}

bool valid_digest(std::string_view value) {
  return value.size() == 71U && value.starts_with("sha256:") &&
         std::ranges::all_of(value.substr(7U), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void exact_fields(const Json& value,
                  std::initializer_list<std::string_view> fields) {
  if (!value.is_object() || value.size() != fields.size())
    reject("worker bootstrap fields are inexact");
  for (const std::string_view field : fields)
    if (!value.contains(std::string(field)))
      reject("worker bootstrap field is missing");
}

Json bootstrap_body(const WorkerBootstrapSpec& value) {
  if (value.api_version != kWorkerBootstrapApiVersion ||
      !bounded_text(value.controller_target, 4096U) ||
      !value.controller_target.starts_with("unix:/") ||
      !bounded_text(value.run_id, 1024U) ||
      !bounded_text(value.node_id, 1024U) ||
      !bounded_text(value.attempt_id, 1024U) ||
      !bounded_text(value.launch_nonce, 1024U) ||
      !bounded_text(value.adapter, 256U) ||
      !bounded_text(value.adapter_version, 256U) ||
      !valid_digest(value.code_fingerprint) ||
      !bounded_text(value.concurrency_key, 1024U) ||
      !bounded_text(value.lease_id, 1024U) || value.fencing_token == 0U ||
      value.capabilities.size() > 256U ||
      !std::ranges::is_sorted(value.capabilities) ||
      std::adjacent_find(value.capabilities.begin(), value.capabilities.end()) !=
          value.capabilities.end() ||
      !std::ranges::all_of(value.capabilities, [](const std::string& capability) {
        return bounded_text(capability, 256U);
      })) {
    reject("worker bootstrap semantics are invalid");
  }
  return {
      {"adapter", value.adapter},
      {"adapter_version", value.adapter_version},
      {"api_version", value.api_version},
      {"attempt_id", value.attempt_id},
      {"capabilities", value.capabilities},
      {"code_fingerprint", value.code_fingerprint},
      {"concurrency_key", value.concurrency_key},
      {"controller_target", value.controller_target},
      {"fencing_token", value.fencing_token},
      {"last_acked_controller_sequence",
       value.last_acked_controller_sequence},
      {"launch_nonce", value.launch_nonce},
      {"lease_id", value.lease_id},
      {"node_id", value.node_id},
      {"run_id", value.run_id},
  };
}

}  // namespace

WorkerBootstrapSpec seal_worker_bootstrap(WorkerBootstrapSpec value) {
  value.bootstrap_digest.clear();
  const std::string body = bootstrap_body(value).dump();
  if (body.size() > kMaximumWorkerBootstrapBytes)
    reject("worker bootstrap exceeds its canonical size bound");
  value.bootstrap_digest = "sha256:" + sha256_hex(body);
  return value;
}

std::string worker_bootstrap_canonical_json(
    const WorkerBootstrapSpec& value) {
  const WorkerBootstrapSpec canonical = seal_worker_bootstrap(value);
  if (canonical.bootstrap_digest != value.bootstrap_digest)
    reject("worker bootstrap digest is not canonical");
  Json output = bootstrap_body(value);
  output["bootstrap_digest"] = value.bootstrap_digest;
  const std::string encoded = output.dump();
  if (encoded.size() > kMaximumWorkerBootstrapBytes)
    reject("worker bootstrap exceeds its wire size bound");
  return encoded;
}

WorkerBootstrapSpec worker_bootstrap_from_canonical_json(
    std::string_view value) {
  if (value.empty() || value.size() > kMaximumWorkerBootstrapBytes)
    reject("worker bootstrap canonical JSON size is invalid");
  try {
    const Json parsed = Json::parse(value);
    if (parsed.dump() != value)
      reject("worker bootstrap JSON is not canonical");
    exact_fields(parsed,
                 {"adapter", "adapter_version", "api_version", "attempt_id",
                  "bootstrap_digest", "capabilities", "code_fingerprint",
                  "concurrency_key", "controller_target", "fencing_token",
                  "last_acked_controller_sequence", "launch_nonce", "lease_id",
                  "node_id", "run_id"});
    WorkerBootstrapSpec result{
        .api_version = parsed.at("api_version").get<std::string>(),
        .controller_target =
            parsed.at("controller_target").get<std::string>(),
        .run_id = parsed.at("run_id").get<std::string>(),
        .node_id = parsed.at("node_id").get<std::string>(),
        .attempt_id = parsed.at("attempt_id").get<std::string>(),
        .launch_nonce = parsed.at("launch_nonce").get<std::string>(),
        .adapter = parsed.at("adapter").get<std::string>(),
        .adapter_version = parsed.at("adapter_version").get<std::string>(),
        .code_fingerprint =
            parsed.at("code_fingerprint").get<std::string>(),
        .capabilities =
            parsed.at("capabilities").get<std::vector<std::string>>(),
        .last_acked_controller_sequence =
            parsed.at("last_acked_controller_sequence").get<std::uint64_t>(),
        .concurrency_key = parsed.at("concurrency_key").get<std::string>(),
        .lease_id = parsed.at("lease_id").get<std::string>(),
        .fencing_token = parsed.at("fencing_token").get<std::uint64_t>(),
        .bootstrap_digest =
            parsed.at("bootstrap_digest").get<std::string>(),
    };
    if (!valid_digest(result.bootstrap_digest) ||
        worker_bootstrap_canonical_json(result) != value)
      reject("worker bootstrap is not canonical or content-addressed");
    return result;
  } catch (const std::invalid_argument&) {
    throw;
  } catch (...) {
    reject("worker bootstrap decoding failed closed");
  }
}

}  // namespace trainvm
