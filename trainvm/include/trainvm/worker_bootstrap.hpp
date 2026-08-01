#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace trainvm {

inline constexpr std::string_view kWorkerBootstrapApiVersion =
    "trainvm.worker-bootstrap/v1";
inline constexpr std::size_t kMaximumWorkerBootstrapBytes = 16U * 1024U;

// Minimal, immutable information needed for a freshly exec'd worker to open
// WorkerControl. Experiment inputs and training configuration are deliberately
// absent: they arrive only in the content-addressed WorkerWelcome invocation.
struct WorkerBootstrapSpec final {
  std::string api_version;
  std::string controller_target;
  std::string run_id;
  std::string node_id;
  std::string attempt_id;
  std::string launch_nonce;
  std::string adapter;
  std::string adapter_version;
  std::string code_fingerprint;
  std::vector<std::string> capabilities;
  std::uint64_t last_acked_controller_sequence{};
  std::string concurrency_key;
  std::string lease_id;
  std::uint64_t fencing_token{};
  std::string bootstrap_digest;

  bool operator==(const WorkerBootstrapSpec&) const = default;
};

[[nodiscard]] WorkerBootstrapSpec seal_worker_bootstrap(
    WorkerBootstrapSpec value);
[[nodiscard]] std::string worker_bootstrap_canonical_json(
    const WorkerBootstrapSpec& value);
[[nodiscard]] WorkerBootstrapSpec worker_bootstrap_from_canonical_json(
    std::string_view value);

class SealedWorkerBootstrap final {
 public:
  SealedWorkerBootstrap(SealedWorkerBootstrap&& other) noexcept;
  SealedWorkerBootstrap& operator=(SealedWorkerBootstrap&& other) noexcept;
  ~SealedWorkerBootstrap();

  SealedWorkerBootstrap(const SealedWorkerBootstrap&) = delete;
  SealedWorkerBootstrap& operator=(const SealedWorkerBootstrap&) = delete;

  [[nodiscard]] const WorkerBootstrapSpec& spec() const;
  [[nodiscard]] int duplicate_fd() const;

 private:
  friend SealedWorkerBootstrap create_sealed_worker_bootstrap(
      WorkerBootstrapSpec value);
  SealedWorkerBootstrap(WorkerBootstrapSpec spec, int descriptor) noexcept;
  WorkerBootstrapSpec spec_;
  int descriptor_{-1};
};

[[nodiscard]] SealedWorkerBootstrap create_sealed_worker_bootstrap(
    WorkerBootstrapSpec value);
[[nodiscard]] WorkerBootstrapSpec worker_bootstrap_from_sealed_fd(
    int descriptor, std::string_view expected_digest = {});

}  // namespace trainvm
