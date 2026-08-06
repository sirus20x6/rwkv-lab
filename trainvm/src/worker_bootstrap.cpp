#include "trainvm/worker_bootstrap.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/memfd.h>
#include <ranges>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

#include "trainvm/json.hpp"

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

void write_all(int descriptor, std::string_view value) {
  std::size_t offset = 0U;
  while (offset < value.size()) {
    const ssize_t count =
        ::write(descriptor, value.data() + offset, value.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) reject("could not write sealed worker bootstrap");
    offset += static_cast<std::size_t>(count);
  }
}

constexpr int kRequiredSeals =
    F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;

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

SealedWorkerBootstrap::SealedWorkerBootstrap(WorkerBootstrapSpec spec,
                                             int descriptor) noexcept
    : spec_(std::move(spec)), descriptor_(descriptor) {}

SealedWorkerBootstrap::SealedWorkerBootstrap(
    SealedWorkerBootstrap&& other) noexcept
    : spec_(std::move(other.spec_)),
      descriptor_(std::exchange(other.descriptor_, -1)) {}

SealedWorkerBootstrap& SealedWorkerBootstrap::operator=(
    SealedWorkerBootstrap&& other) noexcept {
  if (this != &other) {
    if (descriptor_ >= 0) (void)::close(descriptor_);
    spec_ = std::move(other.spec_);
    descriptor_ = std::exchange(other.descriptor_, -1);
  }
  return *this;
}

SealedWorkerBootstrap::~SealedWorkerBootstrap() {
  if (descriptor_ >= 0) (void)::close(descriptor_);
}

const WorkerBootstrapSpec& SealedWorkerBootstrap::spec() const { return spec_; }

int SealedWorkerBootstrap::duplicate_fd() const {
  const int duplicate = ::fcntl(descriptor_, F_DUPFD_CLOEXEC, 5);
  if (duplicate < 0)
    reject("could not duplicate sealed worker bootstrap descriptor");
  return duplicate;
}

SealedWorkerBootstrap create_sealed_worker_bootstrap(
    WorkerBootstrapSpec value) {
  value = seal_worker_bootstrap(std::move(value));
  const std::string encoded = worker_bootstrap_canonical_json(value);
  const int descriptor = static_cast<int>(::syscall(
      SYS_memfd_create, "trainvm-worker-bootstrap", MFD_CLOEXEC | MFD_ALLOW_SEALING));
  if (descriptor < 0)
    reject("could not create sealed worker bootstrap descriptor");
  try {
    write_all(descriptor, encoded);
    if (::fcntl(descriptor, F_ADD_SEALS, kRequiredSeals) != 0)
      reject("could not seal worker bootstrap descriptor");
    return SealedWorkerBootstrap(std::move(value), descriptor);
  } catch (...) {
    (void)::close(descriptor);
    throw;
  }
}

WorkerBootstrapSpec worker_bootstrap_from_sealed_fd(
    int descriptor, std::string_view expected_digest) {
  struct stat metadata {};
  const int seals = ::fcntl(descriptor, F_GET_SEALS);
  if (descriptor < 0 || ::fstat(descriptor, &metadata) != 0 ||
      !S_ISREG(metadata.st_mode) || metadata.st_size <= 0 ||
      static_cast<std::uintmax_t>(metadata.st_size) >
          kMaximumWorkerBootstrapBytes ||
      seals < 0 || (seals & kRequiredSeals) != kRequiredSeals) {
    reject("worker bootstrap descriptor is not sealed authority");
  }
  std::string encoded(static_cast<std::size_t>(metadata.st_size), '\0');
  std::size_t offset = 0U;
  while (offset < encoded.size()) {
    const ssize_t count = ::pread(descriptor, encoded.data() + offset,
                                  encoded.size() - offset,
                                  static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) reject("could not read sealed worker bootstrap descriptor");
    offset += static_cast<std::size_t>(count);
  }
  const WorkerBootstrapSpec result =
      worker_bootstrap_from_canonical_json(encoded);
  if (!expected_digest.empty() && result.bootstrap_digest != expected_digest)
    reject("worker bootstrap descriptor digest disagrees with request");
  return result;
}

}  // namespace trainvm
