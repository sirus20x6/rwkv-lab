#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "trainvm/authority_time.hpp"
#include "trainvm/hostd_mutation_protocol.hpp"
#include "trainvm/journal.hpp"

namespace trainvm {

inline constexpr std::string_view kHostdMutationClaimProviderApiVersion =
    "trainvm.hostd-mutation-claim-provider/v1";

class HostdMutationClaimProviderError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct HostdMutationClaimProviderConfig final {
  std::string api_version{
      std::string(kHostdMutationClaimProviderApiVersion)};
  std::string broker_epoch;
  AuthorityClock::Source authority_clock;
  // Production leaves this empty and receives a RAND_bytes-backed identity.
  // Tests may inject a deterministic sequence; each call must be fresh.
  std::function<std::string()> controller_id_source;
};

// TrainVM-side authority bridge for mutation clients. Every claim is derived
// from immutable journal records and a durable per-concurrency controller
// generation. A new provider instance advances the generation, fencing a
// controller from the previous service process. Once this instance observes
// that one of its generations was superseded, it fails closed.
class JournalHostdMutationClaimProvider final {
 public:
  JournalHostdMutationClaimProvider(
      Journal& journal, HostdMutationClaimProviderConfig config);

  [[nodiscard]] HostdMutationOpen open_for_resource(
      std::string_view request_or_release_id);
  [[nodiscard]] HostdMutationOpen open_for_process(
      std::string_view launch_id);

 private:
  struct Scope final {
    std::string run_id;
    std::string concurrency_key;
    std::string logical_lease_id;
    std::uint64_t logical_fencing_token{};
  };

  [[nodiscard]] HostdMutationOpen open_for_scope(const Scope& scope);
  [[nodiscard]] std::string next_controller_id();

  Journal& journal_;
  HostdMutationClaimProviderConfig config_;
  AuthorityClock clock_;
  std::mutex mutex_;
  std::unordered_map<std::string, JournalControllerFence> owned_fences_;
};

}  // namespace trainvm
